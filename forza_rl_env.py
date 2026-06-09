"""
forza_rl_env.py - Horizon FSD, Phase 5

The real-time Forza Horizon 6 RL environment for DreamerV3 (NM512 dreamerv3-torch).

Follows NM512's old-gym env contract (see envs/dmc.py):
  * observation_space : gym.spaces.Dict({'image': (H,W,1) uint8, 'speed': (1,) f32})
  * action_space      : gym.spaces.Box([-1,-1,-1], [1,1,1]) (steer, throttle, brake)
  * reset()           -> obs dict (with 'is_first', 'is_terminal')
  * step(action)      -> (obs, reward, done, info)

Dreamer's RSSM does the temporal modelling, so we feed a SINGLE 64x64 frame (no
stack). action_repeat holds each decision for N base ticks (10 Hz at repeat=2).
On a detected crash the episode ends (done) and the next reset() runs the recovery
ladder (rewind -> reset-to-road) to put the car back on the road.
"""
from __future__ import annotations

import time
import os

import gym
import numpy as np

from action_utils import exclusive_pedals, model_to_physical_action
from capture import ScreenCapture
from config import load_config
from gamepad import ForzaGamepad
from racing_line import LineReading, RacingLineReader
from recovery import CrashDetector, DetectorConfig, ForzaResetter, ResetConfig
from recovery_demo import RecoveryDemoConfig, RecoveryDemoRecorder
from reward import DriveReward, DriveRewardConfig
from telemetry_receiver import TelemetryReceiver

from dataclasses import fields as _dc_fields


def _cfg_kwargs(dc_cls, d: dict) -> dict:
    """Keep only keys that are real fields of `dc_cls`, so a stale or renamed config.yaml key
    can't crash env construction (e.g. a ResetConfig field removed in code but left in YAML).
    Warns about dropped keys so the drift is visible."""
    valid = {f.name for f in _dc_fields(dc_cls)}
    extra = [k for k in d if k not in valid]
    if extra:
        print(f"[config] ignoring unknown {dc_cls.__name__} keys: {sorted(extra)}")
    return {k: v for k, v in d.items() if k in valid}


class ForzaDriveEnv:
    metadata = {}

    def __init__(self, task: str = "drive", action_repeat: int = 2,
                 size=(64, 64), mode: str = "train",
                 forza_config: str | None = None, base_hz: float = 20.0) -> None:
        cfg = load_config(forza_config)
        self.size = (int(size[0]), int(size[1]))            # (H, W)
        self.action_repeat = int(action_repeat)
        self.tick_dt = 1.0 / base_hz
        self.reward_range = [-np.inf, np.inf]

        cap, tel = cfg["capture"], cfg["telemetry"]
        self.capture = ScreenCapture(
            monitor_index=cap.get("monitor_index", 1),
            window_name=cap.get("window_name"),
            region=cap.get("region"),
            img_size=self.size,
            grayscale=True,
        )
        self.rx = TelemetryReceiver(host=tel.get("host", "0.0.0.0"), port=int(tel.get("port", 9999)))
        self.rx.start()
        self.gamepad = ForzaGamepad()
        det_cfg = DetectorConfig(**_cfg_kwargs(DetectorConfig, cfg.get("detector", {})))
        self.detector = CrashDetector(det_cfg)
        self.reward_fn = DriveReward(DriveRewardConfig(**_cfg_kwargs(DriveRewardConfig, cfg.get("rl_reward", {}))))
        self.line_reader = RacingLineReader()     # reads FH's driving line from the colour frame
        demo_cfg_dict = dict(cfg.get("recovery_demos", {}))
        if os.environ.get("HORIZON_FSD_LOGDIR"):
            demo_cfg_dict["out_dir"] = os.path.join(os.environ["HORIZON_FSD_LOGDIR"], "train_eps")
        demo_cfg = RecoveryDemoConfig(**_cfg_kwargs(RecoveryDemoConfig, demo_cfg_dict))
        # Recovery demos must use the SAME reward branch as the warm-start episodes (speed fallback,
        # since recordings carry no position) so the OFFLINE replay holds ONE consistent reward scale.
        # Force the speed fallback with centerline_path="" - otherwise recovery demos (which DO carry
        # position) would be centerline-scaled while warm-start is speed-scaled: two scales in one buffer.
        demo_reward_kwargs = _cfg_kwargs(DriveRewardConfig, cfg.get("rl_reward", {}))
        demo_reward_kwargs["centerline_path"] = ""
        demo_reward = DriveReward(DriveRewardConfig(**demo_reward_kwargs))
        self.recovery_demo = RecoveryDemoRecorder(
            self.capture,
            self.line_reader,
            demo_reward,
            demo_cfg,
        ) if demo_cfg.enabled else None
        self.resetter = ForzaResetter(
            self.gamepad,
            self.rx,
            ResetConfig(**_cfg_kwargs(ResetConfig, cfg.get("reset", {}))),
            demo_recorder=self.recovery_demo,
            detector_cfg=det_cfg,                      # recovery uses the SAME flip thresholds (no drift)
        )
        safety = cfg.get("rl_safety", {})
        self.steer_limit = float(safety.get("steer_limit", 0.55))
        self.high_speed_steer_limit = float(safety.get("high_speed_steer_limit", 0.35))
        self.high_speed_threshold = float(safety.get("high_speed_threshold", 15.0))
        self.grace_s = float(safety.get("recovery_grace_s", 1.5))  # skip detection just after a reset
        self.stale_after = float(tel.get("stale_after_s", 3.0 * self.tick_dt))  # telemetry freshness bound
        self.resume_timeout = float(tel.get("resume_timeout_s", 30.0))

        self._prev_t = None
        self._prev_applied_action = np.zeros(3, dtype=np.float32)
        self._prev_line = LineReading(0.0, 0.0, 0.0)   # the line the agent saw last step
        self._pending_reason: str | None = None
        self._tick_deadline = time.perf_counter()
        self._recovered_at = 0.0                       # perf_counter of the last reset (post-recovery grace)
        self._overruns = 0

    # ---- spaces -----------------------------------------------------------
    @property
    def observation_space(self):
        return gym.spaces.Dict({
            "image": gym.spaces.Box(0, 255, self.size + (1,), dtype=np.uint8),
            "speed": gym.spaces.Box(0.0, np.inf, (1,), dtype=np.float32),
            # racing line: [cue (-1 brake .. +1 accelerate), lateral offset, confidence]
            "line": gym.spaces.Box(-1.0, 1.0, (3,), dtype=np.float32),
        })

    @property
    def action_space(self):
        # Dreamer-facing coordinates. The vendored NormalizeActions wrapper becomes
        # an identity wrapper for this env, and step() maps pedals to gamepad [0, 1].
        return gym.spaces.Box(
            np.array([-1.0, -1.0, -1.0], dtype=np.float32),
            np.array([1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

    # ---- helpers ----------------------------------------------------------
    def _capture(self) -> tuple[np.ndarray, LineReading]:
        """Grab ONE colour frame; return (grayscale obs image (H,W,1), racing-line reading).
        Reading the line from the same grab keeps the obs image and the line in sync and
        avoids a second capture."""
        raw = self.capture.grab()                 # full-res BGR
        line = self.line_reader.read(raw)
        img = self.capture.preprocess(raw)        # (H, W) uint8 grayscale
        if img.ndim == 2:
            img = img[:, :, None]                 # (H, W, 1)
        return np.ascontiguousarray(img, dtype=np.uint8), line

    def _obs(self, t, image: np.ndarray, line: LineReading,
             is_first: bool, is_terminal: bool) -> dict:
        return {
            "image": image,
            "speed": np.array([t.speed if t is not None else 0.0], dtype=np.float32),
            "line": np.array([line.cue, line.offset, line.confidence], dtype=np.float32),
            "is_first": is_first,
            "is_terminal": is_terminal,
        }

    def _sleep_to_tick(self) -> None:
        self._tick_deadline += self.tick_dt
        dt = self._tick_deadline - time.perf_counter()
        if dt > 0:
            time.sleep(dt)
        else:                                      # we overran the budget; resync + count it
            self._tick_deadline = time.perf_counter()
            self._overruns += 1
            if self._overruns % 200 == 0:          # heartbeat: real control Hz is below target
                print(f"[forza-env] {self._overruns} tick overruns - control loop running slower than "
                      f"{1.0 / self.tick_dt:.0f}Hz (GPU contention?); progress/impact calibration is approximate")

    def _limit_steer(self, steer: float) -> float:
        t = self.rx.latest()
        speed = max(0.0, float(t.speed)) if t is not None else 0.0
        if self.high_speed_threshold <= 0.0:
            limit = self.steer_limit
        else:
            a = min(1.0, speed / self.high_speed_threshold)
            limit = (1.0 - a) * self.steer_limit + a * self.high_speed_steer_limit
        limit = max(0.05, min(1.0, limit))
        return float(np.clip(steer, -limit, limit))

    # ---- gym API ----------------------------------------------------------
    def step(self, action):
        model_action = np.asarray(action, dtype=np.float32)
        applied_action = model_to_physical_action(model_action)
        steer = self._limit_steer(float(applied_action[0]))
        throttle = float(applied_action[1])
        brake = float(applied_action[2])
        # Never press throttle AND brake at once: an untrained agent outputs both ~0.5,
        # and in-game the brake wins -> the car only reverses/idles and can NEVER explore
        # going forward (so it never learns to drive). Collapse to a single net pedal.
        throttle, brake = exclusive_pedals(throttle, brake)
        applied_action = np.array([steer, throttle, brake], dtype=np.float32)
        self.gamepad.apply([steer, throttle, brake])

        reward = 0.0
        reason = None
        got_telemetry = False
        # post-recovery grace: a freshly-recovered car is handed back stopped; don't let that dead-stop
        # immediately re-trip stuck/noprogress while the agent is still taking the wheel.
        in_grace = (time.perf_counter() - self._recovered_at) < self.grace_s
        for _ in range(self.action_repeat):
            self._sleep_to_tick()
            t = self.rx.latest(max_age=self.stale_after)   # None if the stream is stale (alt-tab/hitch)
            if t is not None:
                got_telemetry = True
                reward += self.reward_fn(t, self._prev_t, applied_action, self._prev_applied_action)
                if reason is None and not in_grace:
                    reason = self.detector.update(t, time.perf_counter(),
                                                  throttle_cmd=throttle, brake_cmd=brake)
                self._prev_t = t
        self._prev_applied_action = applied_action
        if not got_telemetry:                          # frozen stream -> phantom data, not a crash to teleport from
            reason = "telemetry_lost"
            self._prev_t = None                        # a resumed stream must not read a false impact delta
            self.detector._prev_speed = None

        # racing-line shaping: reward agreeing with the line the agent SAW this step
        # (accelerate on blue cue>0, brake on red cue<0), gated by detection confidence.
        pl = self._prev_line
        reward += self.reward_fn.cfg.line_follow_w * pl.confidence * pl.cue * (throttle - brake)

        done = reason is not None
        if done:
            self._pending_reason = reason
            # route_complete (reached the end of the route) and telemetry_lost (a stream hitch) are NOT
            # crashes - don't punish the agent for finishing the route or for the game alt-tabbing.
            if reason not in ("route_complete", "telemetry_lost"):
                reward -= self.reward_fn.cfg.crash_penalty
            latest = self.rx.latest()
            speed_kmh = latest.speed_kmh if latest is not None else 0.0
            print(f"[forza-env] episode done: {reason} "
                  f"speed={speed_kmh:.1f}km/h steer={steer:+.2f} "
                  f"throttle={throttle:.2f} brake={brake:.2f}")
        image, line = self._capture()
        self._prev_line = line
        obs = self._obs(self.rx.latest(), image, line, is_first=False, is_terminal=done)
        info = {
            "crash_reason": reason,
            "speed_kmh": float(obs["speed"][0] * 3.6),
            "applied_steer": steer,
            "applied_throttle": throttle,
            "applied_brake": brake,
        }
        return obs, float(reward), done, info

    def reset(self):
        if self._pending_reason is not None:       # recover from the event that ended the last episode
            self.gamepad.reset()
            if self._pending_reason == "telemetry_lost":
                # a stream hitch, not a stuck car: wait for telemetry to come back, don't teleport.
                if not self.rx.wait_for_packet(self.resume_timeout):
                    raise RuntimeError("Forza telemetry did not resume - is Data Out ON ('Car Dash') "
                                       "and the game running/focused?")
            else:
                recovered_by = self.resetter.recover(self._pending_reason)
                if recovered_by == "FAILED":
                    raise RuntimeError(
                        f"Forza recovery failed after {self._pending_reason}; "
                        "manual reposition is required before training can continue."
                    )
                print(f"[forza-env] recovered by {recovered_by}")
            self._pending_reason = None
        self.gamepad.reset()
        self.detector.reset()
        self._prev_t = None
        self._prev_applied_action = np.zeros(3, dtype=np.float32)
        if not self.rx.wait_for_packet(5.0):       # a run without live telemetry is worthless - fail loud
            raise RuntimeError("no live Forza telemetry at reset - is Data Out ON ('Car Dash'), the "
                               "port correct, and the game focused?")
        self._tick_deadline = time.perf_counter()
        self._recovered_at = time.perf_counter()   # start the post-recovery grace window
        image, line = self._capture()
        self._prev_line = line
        return self._obs(self.rx.latest(), image, line, is_first=True, is_terminal=False)

    def close(self):
        for fn in (self.gamepad.close, self.capture.close, self.rx.close):
            try:
                fn()
            except Exception:
                pass

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

import json
import time
import os
from collections import deque

import gym
import numpy as np

from action_utils import (apply_steer_limit, exclusive_pedals, model_to_physical_action,
                          physical_to_model_action)
from centerline import ROUTE_DIM, route_features
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
        # COLOR obs (capture.grayscale: false): grayscale physically erased the racing-line chevrons
        # (blue->luma 29, red->76, both inside the asphalt range), forcing day-only stopgaps. In color
        # the model sees the line itself, day AND night. Demos must be color too - make_warmstart
        # skips legacy gray sessions rather than shipping a gray/color demo-vs-live discriminator.
        self.grayscale = bool(cap.get("grayscale", True))
        self.channels = 1 if self.grayscale else 3
        self.capture = ScreenCapture(
            monitor_index=cap.get("monitor_index", 1),
            window_name=cap.get("window_name"),
            region=cap.get("region"),
            img_size=self.size,
            grayscale=self.grayscale,
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
            # recovery demos live in a SIBLING dir so train_eps cleanup/purges never sweep them;
            # the trainer's load_new_episodes scans both.
            demo_cfg_dict["out_dir"] = os.path.join(os.environ["HORIZON_FSD_LOGDIR"], "recovery_eps")
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
        self.runway_steps = int(safety.get("runway_steps", 20))    # decision-steps of runway per episode
        route_cfg = cfg.get("rl_route", {})
        self._route_cl = self.reward_fn._centerline                # same reference path as the reward
        self.route_max_dist = float(self.reward_fn.cfg.route_max_dist)
        self.route_spacing = float(route_cfg.get("lookahead_spacing_m", 8.0))
        self.route_min_speed = float(route_cfg.get("heading_min_speed", 2.0))
        # When automated recovery is exhausted: False (default) = PAUSE and wait for the car to become
        # drivable (don't kill an overnight run); True = raise and stop.
        self.crash_on_unrecoverable = bool(safety.get("crash_on_unrecoverable", False))
        self.stale_after = float(tel.get("stale_after_s", 3.0 * self.tick_dt))  # telemetry freshness bound
        self.resume_timeout = float(tel.get("resume_timeout_s", 30.0))
        # WGC only delivers a frame on content change, so a real freeze (load screen / alt-tab) stops
        # frames while a static-but-HUD scene still updates within ~1s; keep this generous so we end the
        # episode only on a genuine freeze, never on a quiet driving frame. (Tune live if it false-fires.)
        self.stale_frame_s = float(cfg.get("capture", {}).get("max_frame_age_s", 1.5))

        self._prev_t = None
        self._prev_applied_action = np.zeros(3, dtype=np.float32)
        self._prev_line = LineReading(0.0, 0.0, 0.0)   # the line the agent saw last step
        self._pending_reason: str | None = None
        self._tick_deadline = time.perf_counter()
        self._recovered_at = 0.0                       # perf_counter of the last reset (post-recovery grace)
        self._overruns = 0
        self._episode_steps = 0
        self._episode_return = 0.0
        self._episode_steer_sum = 0.0
        self._recent_mean_steers: deque = deque(maxlen=20)   # collapse alarm window
        logdir = os.environ.get("HORIZON_FSD_LOGDIR", "")
        self._reasons_path = os.path.join(logdir, "reasons.jsonl") if logdir else None

    # ---- spaces -----------------------------------------------------------
    @property
    def observation_space(self):
        return gym.spaces.Dict({
            "image": gym.spaces.Box(0, 255, self.size + (self.channels,), dtype=np.uint8),
            "speed": gym.spaces.Box(0.0, np.inf, (1,), dtype=np.float32),
            # racing line: [cue (-1 brake .. +1 accelerate), lateral offset, confidence]
            "line": gym.spaces.Box(-1.0, 1.0, (3,), dtype=np.float32),
            # route geometry (telemetry-based, light-invariant): signed cross-track, heading error,
            # prev applied action, and a lookahead preview of the upcoming road - see route_features.
            "route": gym.spaces.Box(-1.0, 1.0, (ROUTE_DIM,), dtype=np.float32),
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
        """Grab ONE colour frame; return (obs image (H,W,1) gray or (H,W,3) color per config,
        racing-line reading). Reading the line from the same grab keeps the obs image and the
        line in sync and avoids a second capture."""
        raw = self.capture.grab()                 # full-res BGR
        line = self.line_reader.read(raw)
        img = self.capture.preprocess(raw)        # (H, W) gray or (H, W, 3) BGR per config
        if img.ndim == 2:
            img = img[:, :, None]                 # (H, W, 1)
        return np.ascontiguousarray(img, dtype=np.uint8), line

    def _obs(self, t, image: np.ndarray, line: LineReading,
             is_first: bool, is_terminal: bool) -> dict:
        prev_a = physical_to_model_action(self._prev_applied_action)
        if t is not None:
            route = route_features(self._route_cl, t.position_x, t.position_z,
                                   getattr(t, "velocity_x", 0.0), getattr(t, "velocity_z", 0.0),
                                   max(0.0, t.speed), prev_a,
                                   max_dist=self.route_max_dist, spacing=self.route_spacing,
                                   min_speed=self.route_min_speed)
        else:
            route = route_features(None, float("nan"), float("nan"), 0.0, 0.0, 0.0, prev_a)
        return {
            "image": image,
            "speed": np.array([t.speed if t is not None else 0.0], dtype=np.float32),
            "line": np.array([line.cue, line.offset, line.confidence], dtype=np.float32),
            "route": route,
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
        return apply_steer_limit(steer, speed, self.steer_limit,
                                 self.high_speed_steer_limit, self.high_speed_threshold)

    # ---- gym API ----------------------------------------------------------
    # detector reasons that are just the post-recovery dead-stop settling (suppressed during grace);
    # flipped/offroad/impact stay ACTIVE - a freshly recovered car that's flipped/off-road means
    # recovery FAILED and should re-trigger, not be masked.
    _GRACE_SUPPRESSED = ("stuck", "noprogress", "offroute")

    def _log_episode(self, reason: str) -> None:
        """Append the auditable per-episode record (93% of episodes once ended 'crash-flavored' with
        no on-disk reason distribution) and run the COLLAPSE ALARM: |mean applied steer| persistently
        high across recent episodes = the constant-turn failure mode; alert immediately, not after an
        overnight run is wasted."""
        n = max(1, self._episode_steps)
        mean_steer = self._episode_steer_sum / n
        self._recent_mean_steers.append(mean_steer)
        if self._reasons_path:
            try:
                with open(self._reasons_path, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps({
                        "t": time.time(), "reason": reason, "length": self._episode_steps,
                        "return": round(self._episode_return, 2), "mean_steer": round(mean_steer, 3),
                        "overruns": self._overruns,
                    }) + "\n")
            except OSError:
                pass
        if len(self._recent_mean_steers) >= 10:
            m = float(np.mean(self._recent_mean_steers))
            if abs(m) > 0.4:
                print(f"[forza-env] !! COLLAPSE ALARM: mean applied steer {m:+.2f} over the last "
                      f"{len(self._recent_mean_steers)} episodes - the actor is locking the wheel. "
                      "Stop and re-pretrain rather than burning an overnight run.\a")
        self._episode_return = 0.0
        self._episode_steer_sum = 0.0

    def _benign_terminal(self, reason):
        """End the episode WITHOUT a crash penalty and without driving the agent's action - for the
        not-a-crash halts (paused menu, frozen render frame). reset() waits for live driving to
        resume for these reasons rather than running the teleport ladder."""
        self._log_episode(reason)
        self.gamepad.reset()
        self._pending_reason = reason
        self._prev_t = None
        self.detector._prev_speed = None
        image, line = self._capture()
        self._prev_line = line
        obs = self._obs(self.rx.latest(), image, line, is_first=False, is_terminal=True)
        print(f"[forza-env] episode halted (benign): {reason}")
        return obs, 0.0, True, {"crash_reason": reason, "speed_kmh": float(obs["speed"][0] * 3.6),
                                "applied_steer": 0.0, "applied_throttle": 0.0, "applied_brake": 0.0,
                                "applied_action_model": physical_to_model_action([0.0, 0.0, 0.0])}

    def step(self, action):
        # Not-a-crash halts, checked BEFORE applying the agent's action:
        t0 = self.rx.latest(max_age=self.stale_after)
        if t0 is not None and not t0.is_driving:
            # paused / menu / loading with telemetry still flowing: never drive the agent's action into
            # a menu (A=confirm would accept events/teleports). Neutralize and wait for racing to resume.
            return self._benign_terminal("paused")
        if self.capture.frame_age() > self.stale_frame_s:
            # WGC stopped delivering frames (real freeze) -> don't learn dynamics from a frozen image.
            return self._benign_terminal("frame_lost")

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
        self._episode_steps += 1
        # post-recovery grace + episode RUNWAY (tmrl pattern): a freshly-recovered car is handed back
        # stopped, and an agent needs a couple of seconds of consecutive on-road steps to act before a
        # detector can kill the episode - otherwise the buffer is wall-to-wall 8-15-step episode-starts
        # (median was 27 steps). flipped/offroad/impact stay ACTIVE; the wedge-grind window (~8 s) is
        # longer than the runway, so this cannot reopen the -4037 wedge hole.
        in_grace = ((time.perf_counter() - self._recovered_at) < self.grace_s
                    or self._episode_steps <= self.runway_steps)
        for _ in range(self.action_repeat):
            self._sleep_to_tick()
            t = self.rx.latest(max_age=self.stale_after)   # None if the stream is stale (alt-tab/hitch)
            if t is not None:
                got_telemetry = True
                reward += self.reward_fn(t, self._prev_t, applied_action, self._prev_applied_action)
                if reason is None:
                    r = self.detector.update(t, time.perf_counter(),
                                             throttle_cmd=throttle, brake_cmd=brake)
                    if r is not None and not (in_grace and r in self._GRACE_SUPPRESSED):
                        reason = r          # keep flipped/offroad/impact even in grace; drop settling ones
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
        self._episode_steer_sum += steer
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
        self._episode_return += float(reward)
        if done:
            self._log_episode(reason)
        image, line = self._capture()
        self._prev_line = line
        obs = self._obs(self.rx.latest(), image, line, is_first=False, is_terminal=done)
        info = {
            "crash_reason": reason,
            "speed_kmh": float(obs["speed"][0] * 3.6),
            "applied_steer": steer,
            "applied_throttle": throttle,
            "applied_brake": brake,
            # The action the car ACTUALLY executed (post steer-clamp + exclusive pedals), in model
            # coordinates. The trainer caches THIS into replay, not the raw policy sample: 73.5% of
            # raw samples saturated beyond the clamp and 36.7% co-pressed both pedals, so the world
            # model was learning dynamics for actions that never happened - which made saturated
            # steering free in imagination (a direct mechanism for the steering collapse).
            "applied_action_model": physical_to_model_action([steer, throttle, brake]),
        }
        return obs, float(reward), done, info

    def _wait_until_driving(self, timeout: float) -> bool:
        """Wait until the game is actually RACING again with fresh telemetry and a fresh frame - for the
        benign halts (telemetry_lost / paused / frame_lost) where the fix is to wait, not to teleport."""
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < timeout:
            t = self.rx.latest(max_age=self.stale_after)
            if t is not None and t.is_driving and self.capture.frame_age() < self.stale_frame_s:
                return True
            time.sleep(0.1)
        return False

    def _await_manual_recovery(self, reason: str) -> None:
        """Automated recovery is exhausted (car unroutable - most often NO AutoDrive waypoint is pinned,
        or the pause-reset bind doesn't work). Rather than KILL an overnight run, neutralize and WAIT:
        watch for the car to become drivable again (the human repositions it onto a road; the agent then
        resumes). The background learner keeps training on the replay buffer the whole time. Ctrl+C still
        stops the run cleanly."""
        self.gamepad.reset()
        bar = "=" * 74
        print(f"\n{bar}\n[forza-env] RECOVERY EXHAUSTED after '{reason}'. Training is PAUSED, not stopped."
              f"\n  -> Put the car back on a road; PIN A ROUTE WAYPOINT so AutoDrive works next time."
              f"\n  -> Resumes automatically once the car is driving on a road. Ctrl+C to stop.\n{bar}\a")
        on_road_rumble = self.resetter.cfg.on_road_rumble
        last_msg = time.perf_counter()
        while True:
            t = self.rx.latest(max_age=self.stale_after)
            if (t is not None and t.is_driving and self.capture.frame_age() < self.stale_frame_s
                    and t.mean_surface_rumble < on_road_rumble):
                print("[forza-env] car is drivable again - resuming training")
                return
            now = time.perf_counter()
            if now - last_msg > 30.0:
                last_msg = now
                print(f"[forza-env] still waiting for a drivable car after '{reason}' "
                      "(reposition it on a road; Ctrl+C to stop)")
            time.sleep(1.0)

    def reset(self):
        if self._pending_reason is not None:       # recover from the event that ended the last episode
            self.gamepad.reset()
            if self._pending_reason in ("telemetry_lost", "paused", "frame_lost"):
                # stream/menu/render hitch, not a stuck car: wait for live driving to resume, don't teleport.
                if not self._wait_until_driving(self.resume_timeout):
                    raise RuntimeError(f"Forza did not resume live driving after '{self._pending_reason}' "
                                       "- is Data Out ON ('Car Dash'), the game focused, and unpaused?")
            else:
                recovered_by = self.resetter.recover(self._pending_reason)
                if recovered_by == "FAILED":
                    if self.crash_on_unrecoverable:
                        raise RuntimeError(
                            f"Forza recovery failed after {self._pending_reason}; "
                            "manual reposition is required before training can continue."
                        )
                    self._await_manual_recovery(self._pending_reason)
                else:
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
        self._episode_steps = 0                    # start the episode runway
        image, line = self._capture()
        self._prev_line = line
        return self._obs(self.rx.latest(), image, line, is_first=True, is_terminal=False)

    def close(self):
        for fn in (self.gamepad.close, self.capture.close, self.rx.close):
            try:
                fn()
            except Exception:
                pass

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

import gym
import numpy as np

from action_utils import exclusive_pedals, model_to_physical_action
from capture import ScreenCapture
from config import load_config
from gamepad import ForzaGamepad
from racing_line import LineReading, RacingLineReader
from recovery import CrashDetector, ForzaResetter
from reward import DriveReward, DriveRewardConfig
from telemetry_receiver import TelemetryReceiver


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
        self.detector = CrashDetector()
        self.resetter = ForzaResetter(self.gamepad, self.rx)
        self.reward_fn = DriveReward(DriveRewardConfig(**cfg.get("rl_reward", {})))
        safety = cfg.get("rl_safety", {})
        self.steer_limit = float(safety.get("steer_limit", 0.55))
        self.high_speed_steer_limit = float(safety.get("high_speed_steer_limit", 0.35))
        self.high_speed_threshold = float(safety.get("high_speed_threshold", 15.0))
        self.line_reader = RacingLineReader()     # reads FH's driving line from the colour frame

        self._prev_t = None
        self._prev_applied_action = np.zeros(3, dtype=np.float32)
        self._prev_line = LineReading(0.0, 0.0, 0.0)   # the line the agent saw last step
        self._pending_reason: str | None = None
        self._tick_deadline = time.perf_counter()

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
        else:                                      # we overran the budget; resync
            self._tick_deadline = time.perf_counter()

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
        for _ in range(self.action_repeat):
            self._sleep_to_tick()
            t = self.rx.latest()
            if t is not None:
                reward += self.reward_fn(t, self._prev_t, applied_action, self._prev_applied_action)
                if reason is None:
                    reason = self.detector.update(t, time.perf_counter(),
                                                  throttle_cmd=throttle, brake_cmd=brake)
                self._prev_t = t
        self._prev_applied_action = applied_action

        # racing-line shaping: reward agreeing with the line the agent SAW this step
        # (accelerate on blue cue>0, brake on red cue<0), gated by detection confidence.
        pl = self._prev_line
        reward += self.reward_fn.cfg.line_follow_w * pl.confidence * pl.cue * (throttle - brake)

        done = reason is not None
        if done:
            self._pending_reason = reason
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
        if self._pending_reason is not None:       # recover from the crash that ended the last episode
            self.gamepad.reset()
            self.resetter.recover(self._pending_reason)
            self._pending_reason = None
        self.gamepad.reset()
        self.detector.reset()
        self._prev_t = None
        self._prev_applied_action = np.zeros(3, dtype=np.float32)
        self.rx.wait_for_packet(2.0)
        self._tick_deadline = time.perf_counter()
        image, line = self._capture()
        self._prev_line = line
        return self._obs(self.rx.latest(), image, line, is_first=True, is_terminal=False)

    def close(self):
        for fn in (self.gamepad.close, self.capture.close, self.rx.close):
            try:
                fn()
            except Exception:
                pass

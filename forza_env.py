"""
forza_env.py - Horizon FSD, Phase 1

The real-time FH6 driving environment, built on rtgym (the same real-time-gym
pattern tmrl uses for TrackMania). rtgym enforces a fixed control time step:
it sends the action, waits, then captures the observation, so policy inference
overlaps the next real step instead of pausing the (non-pausable) game.

We implement one `RealTimeGymInterface` that fuses three I/O channels:
  * observation : windows-capture screen frames (stacked) + speed scalar
  * reward/state: Forza "Data Out" UDP telemetry
  * action      : virtual Xbox gamepad  [steer, throttle, brake]

rtgym appends the last `act_buf_len` actions to the observation automatically
(act_in_obs=True), so get_observation_space() EXCLUDES them.

Build one with `make_forza_env()`.
"""
from __future__ import annotations

import logging
from collections import deque
from typing import Any, Optional

import gymnasium
import numpy as np
from gymnasium import spaces
from rtgym import DEFAULT_CONFIG_DICT, RealTimeGymInterface

import reward as reward_module
from capture import ScreenCapture
from forza_telemetry import ForzaTelemetry
from gamepad import ACTION_DIM, ForzaGamepad
from telemetry_receiver import TelemetryReceiver

logger = logging.getLogger(__name__)

RTGYM_ENV_ID = "real-time-gym-ts-v1"  # thread-safe backend (recommended)


class ForzaInterface(RealTimeGymInterface):
    """rtgym interface wiring capture + telemetry + gamepad for FH6.

    Args:
        cfg: the full config.yaml dict (telemetry/capture/reward sections).
    """

    # The -ts-v1 backend builds the interface inside its worker thread, so
    # env.unwrapped.interface isn't reachable. Stash the latest instance here so
    # tools (random_agent) can read stats / close it.
    _instance: "Optional[ForzaInterface]" = None

    def __init__(self, cfg: dict[str, Any]) -> None:
        self.cfg = cfg
        ForzaInterface._instance = self
        cap_cfg = cfg["capture"]
        self.img_size = (int(cap_cfg["img_height"]), int(cap_cfg["img_width"]))  # (H, W)
        self.frame_stack = int(cap_cfg["frame_stack"])
        self.grayscale = bool(cap_cfg.get("grayscale", True))

        self.capture = ScreenCapture(
            monitor_index=cap_cfg.get("monitor_index", 1),
            window_name=cap_cfg.get("window_name"),
            region=cap_cfg.get("region"),
            img_size=self.img_size,
            grayscale=self.grayscale,
            cursor_capture=cap_cfg.get("cursor_capture", False),
        )

        tel_cfg = cfg["telemetry"]
        self.rx = TelemetryReceiver(
            host=tel_cfg.get("host", "0.0.0.0"),
            port=int(tel_cfg.get("port", 9999)),
            recv_timeout=float(tel_cfg.get("recv_timeout_s", 1.0)),
        )
        self.rx.start()

        self.gamepad = ForzaGamepad()
        self.reward_fn = reward_module.get(cfg.get("reward", {}).get("type", "forward_speed"))

        self._frames: deque[np.ndarray] = deque(maxlen=self.frame_stack)
        self._prev_telemetry: Optional[ForzaTelemetry] = None

    # ---- helpers ----------------------------------------------------------
    def _grab_frame(self) -> np.ndarray:
        return self.capture.observation()

    def _stacked(self) -> np.ndarray:
        return np.stack(list(self._frames), axis=0).astype(np.uint8)

    def _observation(self) -> list[np.ndarray]:
        telemetry = self.rx.latest()
        speed = np.array([telemetry.speed if telemetry else 0.0], dtype=np.float32)
        return [self._stacked(), speed]

    # ---- rtgym interface --------------------------------------------------
    def get_observation_space(self) -> spaces.Tuple:
        h, w = self.img_size
        if self.grayscale:
            img_shape = (self.frame_stack, h, w)
        else:
            img_shape = (self.frame_stack, h, w, 3)
        img_space = spaces.Box(low=0, high=255, shape=img_shape, dtype=np.uint8)
        speed_space = spaces.Box(low=0.0, high=np.inf, shape=(1,), dtype=np.float32)
        return spaces.Tuple((img_space, speed_space))

    def get_action_space(self) -> spaces.Box:
        low = np.array([-1.0, 0.0, 0.0], dtype=np.float32)
        high = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        return spaces.Box(low=low, high=high, dtype=np.float32)

    def get_default_action(self) -> np.ndarray:
        return np.zeros(ACTION_DIM, dtype=np.float32)

    def send_control(self, control) -> None:
        if control is not None:
            self.gamepad.apply([float(control[0]), float(control[1]), float(control[2])])

    def reset(self, seed=None, options=None):
        self.gamepad.reset()
        self._prev_telemetry = None
        frame = self._grab_frame()
        self._frames.clear()
        for _ in range(self.frame_stack):
            self._frames.append(frame)
        telemetry = self.rx.latest()
        info = {"has_telemetry": telemetry is not None}
        return self._observation(), info

    def get_obs_rew_terminated_info(self):
        self._frames.append(self._grab_frame())
        telemetry = self.rx.latest()
        if telemetry is not None:
            rew = float(self.reward_fn(telemetry, self._prev_telemetry))
            self._prev_telemetry = telemetry
        else:
            rew = 0.0
        obs = self._observation()
        terminated = False  # open-world continuing task; rtgym truncates at ep_max_length
        info = {
            "has_telemetry": telemetry is not None,
            "speed_kmh": telemetry.speed_kmh if telemetry else 0.0,
            "gear": telemetry.gear if telemetry else None,
        }
        return obs, rew, terminated, info

    def wait(self) -> None:
        """Idle between episodes: release the controls."""
        self.gamepad.reset()

    def render(self) -> None:
        pass

    # ---- cleanup (not part of the rtgym abstract API) ---------------------
    def close(self) -> None:
        for fn in (self.gamepad.close, self.capture.close, self.rx.close):
            try:
                fn()
            except Exception:  # pragma: no cover
                pass


def get_interface() -> Optional[ForzaInterface]:
    """The most recently constructed ForzaInterface (the -ts-v1 backend hides it)."""
    return ForzaInterface._instance


def make_forza_env(cfg: Optional[dict[str, Any]] = None, config_path: Optional[str] = None):
    """Build the rtgym FH6 env from config.yaml."""
    if cfg is None:
        from config import load_config
        cfg = load_config(config_path)

    env_cfg = cfg.get("env", {})
    config = dict(DEFAULT_CONFIG_DICT)
    config["interface"] = ForzaInterface
    config["interface_kwargs"] = {"cfg": cfg}
    config["time_step_duration"] = float(env_cfg.get("time_step_duration", 0.05))
    config["start_obs_capture"] = float(env_cfg.get("start_obs_capture", 0.04))
    config["act_buf_len"] = int(env_cfg.get("act_buf_len", 4))
    config["ep_max_length"] = int(env_cfg.get("ep_max_length", 2000))
    config["act_in_obs"] = True
    config["reset_act_buf"] = True

    return gymnasium.make(RTGYM_ENV_ID, config=config, disable_env_checker=True)

"""
Record route-verified AutoDrive recoveries as Dreamer replay demonstrations.

Only non-teleport recoveries are useful for imitation: if the car jumps from an
off-road/stuck position to a road, the action sequence did not cause that state
transition. When the coordinates change smoothly, ANNA's actual telemetry inputs
are good labels for "how to get back to the route".
"""
from __future__ import annotations

import os
import time
import uuid
from dataclasses import dataclass
from typing import Optional

import numpy as np

from action_utils import physical_to_model_action
from centerline import ROUTE_DIM


@dataclass
class RecoveryDemoConfig:
    enabled: bool = False
    out_dir: str = r"C:\Horizon_FSD\dreamer_logs\forza\recovery_eps"   # sibling of train_eps: purges
    #                                                                    of train_eps never sweep demos
    min_len: int = 16
    max_len: int = 1200
    sample_hz: float = 10.0
    teleport_jump_m: float = 30.0


@dataclass
class _Snap:
    speed: float
    mean_surface_rumble: float
    mean_tire_slip_ratio: float
    angular_velocity_y: float
    position_x: float
    position_z: float


def _dist(a: Optional[tuple[float, float]], b: Optional[tuple[float, float]]) -> float:
    if a is None or b is None:
        return 0.0
    dx, dz = b[0] - a[0], b[1] - a[1]
    return float((dx * dx + dz * dz) ** 0.5)


class RecoveryDemoRecorder:
    def __init__(self, capture, line_reader, reward_fn,
                 cfg: RecoveryDemoConfig = RecoveryDemoConfig()) -> None:
        self.capture = capture
        self.line_reader = line_reader
        self.reward_fn = reward_fn
        self.cfg = cfg
        self.active = False
        self.teleported = False
        self._method = ""
        self._frames: list[np.ndarray] = []
        self._lines: list[np.ndarray] = []
        self._actions: list[np.ndarray] = []
        self._snaps: list[_Snap] = []
        self._last_pos: Optional[tuple[float, float]] = None
        self._last_sample_t = 0.0

    def begin(self, method: str, start_pos: Optional[tuple[float, float]] = None) -> None:
        if not self.cfg.enabled:
            return
        self.active = True
        self.teleported = False
        self._method = method
        self._frames.clear()
        self._lines.clear()
        self._actions.clear()
        self._snaps.clear()
        self._last_pos = start_pos
        self._last_sample_t = 0.0

    def sample(self, telemetry) -> None:
        if not self.active or telemetry is None or not telemetry.is_driving:
            return
        if self.cfg.max_len > 0 and len(self._frames) >= self.cfg.max_len:
            return
        now = time.perf_counter()
        interval = 1.0 / max(1e-6, float(self.cfg.sample_hz))
        if self._last_sample_t and now - self._last_sample_t < interval:
            return
        self._last_sample_t = now

        pos = (float(telemetry.position_x), float(telemetry.position_z))
        if _dist(self._last_pos, pos) > self.cfg.teleport_jump_m:
            self.teleported = True
        self._last_pos = pos

        raw = self.capture.grab()
        line = self.line_reader.read(raw) if self.line_reader is not None else None
        img = self.capture.preprocess(raw)
        if img.ndim == 2:
            img = img[:, :, None]

        self._frames.append(np.ascontiguousarray(img, dtype=np.uint8))
        if line is None:
            self._lines.append(np.zeros((3,), dtype=np.float32))
        else:
            self._lines.append(np.array([line.cue, line.offset, line.confidence], dtype=np.float32))
        self._actions.append(np.array([
            telemetry.steer_norm,
            telemetry.throttle,
            telemetry.brake,
        ], dtype=np.float32))
        self._snaps.append(_Snap(
            speed=float(telemetry.speed),
            mean_surface_rumble=float(telemetry.mean_surface_rumble),
            mean_tire_slip_ratio=float(telemetry.mean_tire_slip_ratio),
            angular_velocity_y=float(getattr(telemetry, "angular_velocity_y", 0.0)),
            position_x=float(telemetry.position_x),
            position_z=float(telemetry.position_z),
        ))

    def end(self, success: bool, teleported: bool = False) -> Optional[str]:
        if not self.active:
            return None
        self.active = False
        self.teleported = self.teleported or teleported
        if not success or self.teleported or len(self._frames) < self.cfg.min_len:
            return None

        ep = self._episode()
        os.makedirs(self.cfg.out_dir, exist_ok=True)
        length = len(ep["reward"])
        stamp = time.strftime("%Y%m%dT%H%M%S")
        name = f"recovery-{stamp}-{uuid.uuid4().hex[:8]}-{self._method}-{length}.npz"
        path = os.path.join(self.cfg.out_dir, name)
        with open(path, "wb") as fh:
            np.savez_compressed(fh, **ep)
        print(f"[recovery-demo] saved {length} non-teleport AutoDrive samples -> {path}")
        return path

    def _episode(self) -> dict:
        length = len(self._frames)
        image = np.stack(self._frames, axis=0)
        line = np.stack(self._lines, axis=0)
        speed = np.asarray([[s.speed] for s in self._snaps], dtype=np.float32)
        action = np.zeros((length, 3), dtype=np.float32)
        applied = np.zeros((length, 3), dtype=np.float32)
        reward = np.zeros((length,), dtype=np.float32)

        for i in range(1, length):
            applied[i] = self._actions[i - 1]
            action[i] = physical_to_model_action(applied[i])
            reward[i] = self.reward_fn(
                self._snaps[i],
                self._snaps[i - 1],
                applied[i],
                applied[i - 1],
            )

        is_first = np.zeros((length,), dtype=bool)
        is_first[0] = True
        return {
            "image": image,
            "speed": speed,
            "line": line,
            # route features unavailable here (the recorder keeps no centerline reference);
            # all-zeros = "route unknown", same convention as legacy demos.
            "route": np.zeros((length, ROUTE_DIM), dtype=np.float32),
            "action": action,
            "reward": reward,
            "is_first": is_first,
            "is_terminal": np.zeros((length,), dtype=bool),
            "discount": np.ones((length,), dtype=np.float32),
            "logprob": np.zeros((length,), dtype=np.float32),
        }

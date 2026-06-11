"""
dataset.py - Horizon FSD

Recording-session loaders + telemetry-based frame filters, used by make_warmstart.py
to turn recorded .npz shards into Dreamer replay episodes.

Filtering (telemetry-based):
  * crash/impact frames: speed drop > crash_speed_drop in one step, OR
    |acceleration| > crash_accel; a window of +/- crash_window frames is dropped.
  * idle frames: speed < min_speed.
  * extreme wheelspin/slide: tire_slip > max_tire_slip.
  * IsRaceOn == 0 (menus) - already skipped at record time, dropped again here.
  * autodrive-quality sessions additionally drop rough-surface frames.

Frames load lazily (LazyJpegFrames) for the JPEG-color recording format, so a
multi-GB session never has to fit decoded in RAM.
"""
from __future__ import annotations

import glob
import os
from typing import Any, Optional

import numpy as np


DEFAULT_DATASET = {
    "min_speed": 1.0,
    "crash_speed_drop": 3.0,
    "crash_accel": 60.0,
    "crash_window": 8,
    "max_tire_slip": 8.0,
    "straight_steer_thresh": 0.05,
    "max_straight_frac": 0.5,
    "val_frac": 0.1,
    "autodrive_surface_rumble": 0.15,
}

_SCALAR_KEYS = ("actions", "speed", "accel", "surface_rumble",
                "tire_slip", "is_race_on", "distance")
_OPTIONAL_KEYS = ("position", "brightness")


class LazyJpegFrames:
    """Sequence view over JPEG-encoded source frames (the new recording format): decodes one frame
    at a time on access, so a multi-GB color session never has to fit in RAM decoded."""

    def __init__(self, encoded: list) -> None:
        self._enc = encoded

    def __len__(self) -> int:
        return len(self._enc)

    def __getitem__(self, i):
        import cv2
        frame = cv2.imdecode(np.frombuffer(self._enc[i], dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError(f"corrupt JPEG frame at index {i}")
        return frame                                   # BGR (H, W, 3)


def load_session(session_dir: str) -> Optional[dict[str, Any]]:
    shards = sorted(glob.glob(os.path.join(session_dir, "shard_*.npz")))
    if not shards:
        return None
    cat: dict[str, list] = {k: [] for k in _SCALAR_KEYS}
    opt: dict[str, list] = {k: [] for k in _OPTIONAL_KEYS}
    frames_legacy: list = []
    frames_jpeg: list = []
    quality = "manual"
    for sh in shards:
        z = np.load(sh, allow_pickle=True)             # pickle only for the jpeg object array
        for k in _SCALAR_KEYS:
            cat[k].append(z[k])
        for k in _OPTIONAL_KEYS:
            if k in z.files:
                opt[k].append(z[k])
        if "frames_jpeg" in z.files:
            frames_jpeg.extend(list(z["frames_jpeg"]))
        elif "frames" in z.files:
            frames_legacy.append(z["frames"])
        quality = str(z["quality"])
    out: dict[str, Any] = {k: np.concatenate(cat[k], axis=0) for k in _SCALAR_KEYS}
    for k, v in opt.items():
        if v:
            out[k] = np.concatenate(v, axis=0)
    out["frames"] = (LazyJpegFrames(frames_jpeg) if frames_jpeg
                     else np.concatenate(frames_legacy, axis=0))
    out["quality"] = quality
    return out


def _impact_mask(speed: np.ndarray, accel: np.ndarray, ds: dict) -> tuple[np.ndarray, int]:
    n = len(speed)
    dspeed = np.diff(speed, prepend=speed[:1])
    amag = np.linalg.norm(accel, axis=1)
    impacts = (dspeed < -ds["crash_speed_drop"]) | (amag > ds["crash_accel"])
    drop = np.zeros(n, dtype=bool)
    w = int(ds["crash_window"])
    idxs = np.where(impacts)[0]
    for i in idxs:
        drop[max(0, i - w):min(n, i + w + 1)] = True
    return drop, len(idxs)


def _valid_mask(s: dict, ds: dict) -> tuple[np.ndarray, int]:
    aggressive = s["quality"] == "autodrive"
    m = s["is_race_on"].astype(bool)
    m &= s["speed"] >= ds["min_speed"]
    m &= s["tire_slip"] <= ds["max_tire_slip"]
    crash, n_impacts = _impact_mask(s["speed"], s["accel"], ds)
    m &= ~crash
    if aggressive:
        m &= s["surface_rumble"] <= ds["autodrive_surface_rumble"]
    return m, n_impacts

"""
dataset.py - Horizon FSD, Phase 2

Turn recorded .npz shards into a filtered, balanced behavioral-cloning dataset.

Filtering (telemetry-based, tuned to the recorded data):
  * crash/impact frames: speed drop > crash_speed_drop in one step, OR
    |acceleration| > crash_accel; a window of +/- crash_window frames is dropped.
  * idle frames: speed < min_speed.
  * extreme wheelspin/slide: tire_slip > max_tire_slip (kept high so intentional
    dirt driving / drifting survives).
  * IsRaceOn == 0 (menus) - already skipped at record time, dropped again here.
  * autodrive-quality sessions additionally drop rough-surface frames.

Balancing: straights (|steer| < straight_steer_thresh) are downsampled so they are
at most max_straight_frac of the kept frames (raw data is ~61% straight).

Split: the last val_frac of each session (contiguous, by time) becomes validation,
so train/val frames are never temporal neighbours (no leakage).

Frame stacks are built lazily from per-session frames (no 4x memory blow-up).

CLI:
    python dataset.py                 # report filtering/balancing stats
    python dataset.py --save proc.npz # also materialize+save the stacked dataset
"""
from __future__ import annotations

import argparse
import glob
import os
from typing import Any, Optional

import numpy as np

from config import load_config

DEFAULT_DATASET = {
    "frame_stack": 4,
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

_SCALAR_KEYS = ("frames", "actions", "speed", "accel", "surface_rumble",
                "tire_slip", "is_race_on", "distance")


def load_session(session_dir: str) -> Optional[dict[str, Any]]:
    shards = sorted(glob.glob(os.path.join(session_dir, "shard_*.npz")))
    if not shards:
        return None
    cat: dict[str, list] = {k: [] for k in _SCALAR_KEYS}
    quality = "manual"
    for sh in shards:
        z = np.load(sh)
        for k in _SCALAR_KEYS:
            cat[k].append(z[k])
        quality = str(z["quality"])
    out: dict[str, Any] = {k: np.concatenate(cat[k], axis=0) for k in _SCALAR_KEYS}
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


class BCDataset:
    """Lazily builds (stack, speed, action) samples from per-session frames."""

    def __init__(self, sessions: list[dict], samples: list[tuple[int, int]], frame_stack: int):
        self.sessions = sessions
        self.samples = samples
        self.K = frame_stack

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, j: int):
        sess, i = self.samples[j]
        frames = self.sessions[sess]["frames"]
        lo = i - self.K + 1
        stack = np.stack([frames[max(0, t)] for t in range(lo, i + 1)], axis=0).astype(np.uint8)
        speed = np.float32(self.sessions[sess]["speed"][i])
        action = self.sessions[sess]["actions"][i].astype(np.float32)
        return stack, speed, action

    def steer_fractions(self, thresh: float) -> tuple[float, float]:
        if not self.samples:
            return 0.0, 0.0
        steer = np.array([self.sessions[s]["actions"][i, 0] for s, i in self.samples])
        straight = float(np.mean(np.abs(steer) < thresh))
        return straight, 1.0 - straight


def build_dataset(cfg: dict, recordings_dir: Optional[str] = None, seed: int = 0):
    ds = {**DEFAULT_DATASET, **cfg.get("dataset", {})}
    base = recordings_dir or cfg.get("paths", {}).get("recordings_dir", "recordings")
    session_dirs = sorted(d for d in glob.glob(os.path.join(base, "*")) if os.path.isdir(d))
    rng = np.random.default_rng(seed)

    sessions: list[dict] = []
    train_samples: list[tuple[int, int]] = []
    val_samples: list[tuple[int, int]] = []
    stats: list[dict] = []

    for sd in session_dirs:
        s = load_session(sd)
        if s is None:
            continue
        sess_id = len(sessions)
        sessions.append(s)
        n = len(s["speed"])

        m, n_impacts = _valid_mask(s, ds)
        valid_idx = np.where(m)[0]

        # balance: downsample straights among valid frames
        steer = s["actions"][valid_idx, 0]
        straight = np.abs(steer) < ds["straight_steer_thresh"]
        n_straight, n_corner = int(straight.sum()), int((~straight).sum())
        f = ds["max_straight_frac"]
        max_straight = int(round(n_corner * f / (1.0 - f))) if f < 1.0 else n_straight
        keep = np.ones(len(valid_idx), dtype=bool)
        if n_straight > max_straight:
            straight_pos = np.where(straight)[0]
            drop_pos = rng.choice(straight_pos, size=n_straight - max_straight, replace=False)
            keep[drop_pos] = False
        kept_idx = valid_idx[keep]

        cut = int(n * (1.0 - ds["val_frac"]))  # last val_frac (by time) -> val
        for i in kept_idx:
            (val_samples if i >= cut else train_samples).append((sess_id, int(i)))

        stats.append({
            "session": os.path.basename(sd), "quality": s["quality"], "n": n,
            "valid": int(m.sum()), "kept": int(len(kept_idx)),
            "impacts": n_impacts, "straight": n_straight, "corner": n_corner,
        })

    K = ds["frame_stack"]
    return BCDataset(sessions, train_samples, K), BCDataset(sessions, val_samples, K), stats


def _materialize(d: BCDataset):
    frames = np.empty((len(d), d.K, *d.sessions[0]["frames"].shape[1:]), dtype=np.uint8)
    speeds = np.empty((len(d), 1), dtype=np.float32)
    actions = np.empty((len(d), 3), dtype=np.float32)
    for j in range(len(d)):
        stk, sp, act = d[j]
        frames[j] = stk
        speeds[j, 0] = sp
        actions[j] = act
    return frames, speeds, actions


def main() -> int:
    p = argparse.ArgumentParser(description="Build/inspect the BC dataset from recordings.")
    p.add_argument("--config", default=None)
    p.add_argument("--recordings-dir", default=None)
    p.add_argument("--save", default=None, help="Materialize and save the stacked dataset to a .npz.")
    args = p.parse_args()

    cfg = load_config(args.config)
    train, val, stats = build_dataset(cfg, args.recordings_dir)

    print("=" * 78)
    print(" BC dataset build")
    print("=" * 78)
    for st in stats:
        print(f"  {st['session']:24s} [{st['quality']:8s}]  n={st['n']:6d}  valid={st['valid']:6d}  "
              f"kept={st['kept']:6d}  impacts={st['impacts']:3d}  "
              f"(straight={st['straight']}, corner={st['corner']})")
    thr = {**DEFAULT_DATASET, **cfg.get("dataset", {})}["straight_steer_thresh"]
    tr_s, tr_c = train.steer_fractions(thr)
    va_s, va_c = val.steer_fractions(thr)
    print("-" * 78)
    print(f"  TRAIN samples: {len(train):6d}   straight/corner = {tr_s:.2f}/{tr_c:.2f}")
    print(f"  VAL   samples: {len(val):6d}   straight/corner = {va_s:.2f}/{va_c:.2f}")
    print("=" * 78)

    if args.save:
        print(f" materializing + saving -> {args.save} ...")
        tf, ts, ta = _materialize(train)
        vf, vs, va = _materialize(val)
        np.savez_compressed(args.save, train_frames=tf, train_speed=ts, train_actions=ta,
                            val_frames=vf, val_speed=vs, val_actions=va)
        print(f" saved: train {tf.shape} / val {vf.shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

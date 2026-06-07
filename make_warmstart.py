"""
make_warmstart.py - Horizon FSD, Phase 5

Convert the recorded driving shards into DreamerV3 (NM512) replay episodes, so the
world model pre-learns real FH6 dynamics + driving before any live interaction.

Each episode is a maximal CONTIGUOUS run of clean (filtered) frames within a session
(the world model needs temporally-contiguous frames). Frames are resized 84->64 and
single (Dreamer's RSSM does the temporal modelling - no stacking). Rewards are
computed with the SAME DriveReward the live env uses, so warm-start and online
rewards are consistent.

Episode format mirrors what tools.simulate writes (image, speed, action, reward,
discount, is_first, is_terminal); action[t]/reward[t] are aligned to the transition
INTO obs[t] (action[0]=reward[0]=0, is_first[0]=True), and the file is named
'<id>-<length>.npz' as NM512's count_steps expects.
Actions are saved in Dreamer's symmetric coordinates; recorded telemetry pedals
(`0..1`) are converted to `-1..1` before writing replay.

Run:
    .\\.venv\\Scripts\\python.exe make_warmstart.py --logdir C:/Horizon_FSD/dreamer_logs/forza
"""
from __future__ import annotations

import argparse
import glob
import os

import cv2
import numpy as np

from action_utils import physical_to_model_action
import dataset as ds_mod
from config import load_config
from reward import DriveReward, DriveRewardConfig


class _Snap:
    """Minimal telemetry shim exposing only what DriveReward reads."""
    __slots__ = ("speed", "mean_surface_rumble", "mean_tire_slip_ratio")

    def __init__(self, speed, sr, slip):
        self.speed = float(speed)
        self.mean_surface_rumble = float(sr)
        self.mean_tire_slip_ratio = float(slip)


def _episode_from_run(s: dict, run: np.ndarray, reward_fn, size) -> dict:
    L = len(run)
    H, W = size
    frames, physical_actions, speed = s["frames"], s["actions"], s["speed"]
    sr, slip = s["surface_rumble"], s["tire_slip"]

    image = np.zeros((L, H, W, 1), np.uint8)
    act = np.zeros((L, 3), np.float32)             # Dreamer coordinates [-1, 1]
    applied = np.zeros((L, 3), np.float32)         # gamepad/reward coordinates
    rew = np.zeros((L,), np.float32)
    spd = np.zeros((L, 1), np.float32)

    for t in range(L):
        i = run[t]
        f = frames[i]
        if f.shape[:2] != (H, W):
            f = cv2.resize(f, (W, H), interpolation=cv2.INTER_AREA)
        image[t, :, :, 0] = f
        spd[t, 0] = speed[i]
        if t >= 1:                                 # action that LED to obs[t]
            applied[t] = physical_actions[run[t - 1]]
            act[t] = physical_to_model_action(applied[t])
    for t in range(1, L):
        i, j = run[t], run[t - 1]
        rew[t] = reward_fn(_Snap(speed[i], sr[i], slip[i]),
                           _Snap(speed[j], sr[j], slip[j]), applied[t], applied[t - 1])

    is_first = np.zeros((L,), bool); is_first[0] = True
    return {
        "image": image, "speed": spd, "action": act, "reward": rew,
        "is_first": is_first, "is_terminal": np.zeros((L,), bool),
        "discount": np.ones((L,), np.float32),
        # recordings are grayscale (no colour) so the racing-line cue is unavailable here;
        # give a neutral, zero-confidence reading. The model learns the line from live
        # episodes (where it's detected) and to ignore it when confidence is 0.
        "line": np.zeros((L, 3), np.float32),
        # live episodes carry a 'logprob' from the policy; demos have none, so add zeros
        # to keep every episode's keys identical (the batch stacker requires it).
        "logprob": np.zeros((L,), np.float32),
    }


def main() -> int:
    p = argparse.ArgumentParser(description="Recordings -> DreamerV3 replay episodes.")
    p.add_argument("--config", default=None)
    p.add_argument("--recordings-dir", default=None)
    p.add_argument("--out-dir", default=None, help="Dreamer traindir (default: <logdir>/train_eps).")
    p.add_argument("--logdir", default="C:/Horizon_FSD/dreamer_logs/forza")
    p.add_argument("--size", type=int, default=64)
    p.add_argument("--min-len", type=int, default=16)
    args = p.parse_args()

    cfg = load_config(args.config)
    ds = {**ds_mod.DEFAULT_DATASET, **cfg.get("dataset", {})}
    reward_fn = DriveReward(DriveRewardConfig(**cfg.get("rl_reward", {})))
    base = args.recordings_dir or cfg.get("paths", {}).get("recordings_dir", "recordings")
    out = args.out_dir or os.path.join(args.logdir, "train_eps")
    os.makedirs(out, exist_ok=True)
    size = (args.size, args.size)

    sessions = sorted(d for d in glob.glob(os.path.join(base, "*")) if os.path.isdir(d))
    n_eps, n_steps = 0, 0
    for sd in sessions:
        s = ds_mod.load_session(sd)
        if s is None:
            continue
        mask, _ = ds_mod._valid_mask(s, ds)
        valid = np.where(mask)[0]
        if len(valid) < args.min_len:
            continue
        runs = np.split(valid, np.where(np.diff(valid) > 1)[0] + 1)
        sess = os.path.basename(sd)
        kept = 0
        for ri, run in enumerate(runs):
            if len(run) < args.min_len:
                continue
            ep = _episode_from_run(s, run, reward_fn, size)
            L = len(ep["reward"])
            with open(os.path.join(out, f"ws-{sess}-{ri:03d}-{L}.npz"), "wb") as fh:
                np.savez_compressed(fh, **ep)
            n_eps += 1
            n_steps += L - 1
            kept += 1
        print(f"  {sess}: {kept} episodes")
    print(f"\nDONE: {n_eps} episodes, ~{n_steps} steps -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

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

from action_utils import apply_steer_limit, exclusive_pedals, physical_to_model_action
from centerline import ROUTE_DIM, route_features
import dataset as ds_mod
from config import load_config
from racing_line import RacingLineReader
from reward import DriveReward, DriveRewardConfig


class _Snap:
    """Minimal telemetry shim exposing only what DriveReward reads."""
    __slots__ = ("speed", "mean_surface_rumble", "mean_tire_slip_ratio")

    def __init__(self, speed, sr, slip):
        self.speed = float(speed)
        self.mean_surface_rumble = float(sr)
        self.mean_tire_slip_ratio = float(slip)


def _clamped_applied(s: dict, i: int, steer_cfg: dict) -> np.ndarray:
    """The recorded human action, passed through the SAME actuator constraints the live env applies
    (speed-dependent steer clamp + exclusive pedals). Without this, demos teach steer targets the
    actuator silently halves (12-31% of manual steps exceeded the limit) - mismatched action labels
    that corrupt the world model's dynamics."""
    steer, thr, brk = (float(v) for v in s["actions"][i])
    steer = apply_steer_limit(steer, float(s["speed"][i]), **steer_cfg)
    thr, brk = exclusive_pedals(thr, brk)
    return np.array([steer, thr, brk], np.float32)


def _obs_frame(f: np.ndarray, H: int, W: int, channels: int) -> np.ndarray:
    """Source frame (gray (h,w) legacy or BGR (h,w,3) new) -> the training obs (H,W,channels).
    NO gray->color replication shim: a replicated-gray corpus next to color live frames is a
    perfect demo/live discriminator for the reward head (callers must skip those sessions)."""
    if channels == 1 and f.ndim == 3:
        f = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
    if f.shape[:2] != (H, W):
        f = cv2.resize(f, (W, H), interpolation=cv2.INTER_AREA)
    return f[:, :, None] if f.ndim == 2 else f


def _episode_from_run(s: dict, run: np.ndarray, reward_fn, size, stride: int, steer_cfg: dict,
                      channels: int = 1) -> dict:
    """Strided conversion: recordings are 20 Hz ticks, live decisions are 10 Hz (action_repeat=2).
    Keep every `stride`-th frame so demo and live transitions share ONE timescale (one RSSM cannot
    fit two time constants), and accumulate the per-tick rewards across each window exactly like the
    live env's action_repeat loop does."""
    frames, speed = s["frames"], s["speed"]
    sr, slip = s["surface_rumble"], s["tire_slip"]
    H, W = size

    # per-tick applied actions (actuator-constrained) + per-tick rewards over the FULL run
    applied_full = np.stack([_clamped_applied(s, i, steer_cfg) for i in run])
    r_tick = np.zeros((len(run),), np.float32)
    for t in range(1, len(run)):
        i, j = run[t], run[t - 1]
        r_tick[t] = reward_fn(_Snap(speed[i], sr[i], slip[i]),
                              _Snap(speed[j], sr[j], slip[j]), applied_full[t], applied_full[t - 1])

    idx = run[::stride]
    L = len(idx)
    image = np.zeros((L, H, W, channels), np.uint8)
    act = np.zeros((L, 3), np.float32)             # Dreamer coordinates [-1, 1]
    rew = np.zeros((L,), np.float32)
    spd = np.zeros((L, 1), np.float32)

    # route-geometry obs: REAL when the recording carries world position (recordings made after
    # record.py learned to store it), zeros = "route unknown" otherwise. Same route_features the
    # live env uses, so demo and live route channels are identical.
    pos = s.get("position")
    has_pos = (pos is not None and np.isfinite(pos[run]).all()
               and np.ptp(pos[run, 0]) + np.ptp(pos[run, 2]) > 1.0)
    cl = getattr(reward_fn, "_centerline", None)
    route = np.zeros((L, ROUTE_DIM), np.float32)
    # racing-line obs: BACKFILLED from the color source frames with the SAME day/night-adaptive
    # reader the live env runs - demos and live now carry an identical line channel in any light
    # (legacy gray sessions can't provide it, but those are skipped under the color target anyway).
    line = np.zeros((L, 3), np.float32)
    line_reader = RacingLineReader() if channels == 3 else None

    for t in range(L):
        src = frames[idx[t]]
        image[t] = _obs_frame(src, H, W, channels)
        spd[t, 0] = speed[idx[t]]
        if line_reader is not None and getattr(src, "ndim", 2) == 3:
            r = line_reader.read(src)
            line[t] = (r.cue, r.offset, r.confidence)
        if t >= 1:                                 # decision held INTO obs[t] = action at the window start
            act[t] = physical_to_model_action(applied_full[(t - 1) * stride])
            rew[t] = float(r_tick[(t - 1) * stride + 1: t * stride + 1].sum())
        if has_pos and cl is not None:
            i = idx[t]
            j = idx[t - 1] if t >= 1 else i
            dt_w = max(1e-3, (i - j) * 0.05)       # recordings tick at 20 Hz
            vx = (pos[i, 0] - pos[j, 0]) / dt_w if i != j else 0.0
            vz = (pos[i, 2] - pos[j, 2]) / dt_w if i != j else 0.0
            route[t] = route_features(cl, pos[i, 0], pos[i, 2], vx, vz,
                                      float(speed[i]), act[t],
                                      max_dist=reward_fn.cfg.route_max_dist)

    is_first = np.zeros((L,), bool); is_first[0] = True
    return {
        "image": image, "speed": spd, "action": act, "reward": rew,
        "is_first": is_first, "is_terminal": np.zeros((L,), bool),
        "discount": np.ones((L,), np.float32),
        "line": line,
        "route": route,
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
    p.add_argument("--stride", type=int, default=2,
                   help="Keep every Nth recorded tick (recordings are 20Hz, live decisions 10Hz -> 2).")
    p.add_argument("--bang-bang-max", type=float, default=0.05,
                   help="Sessions with more than this fraction of full-lock steer (keyboard play) are "
                        "DEMOTED to wsx-*: world-model food, but excluded from BC/demo-oversampling.")
    p.add_argument("--autodrive-as-demo", action="store_true",
                   help="Let ANNA AutoDrive sessions teach the POLICY too (ws-*). Default keeps them "
                        "wsx-* (world model + reward head only): ANNA's steering telemetry is real and "
                        "smooth, but it's ~90%% straight-line driving and contains zero recovery "
                        "content - useful dynamics, weak steering teacher.")
    args = p.parse_args()

    cfg = load_config(args.config)
    ds = {**ds_mod.DEFAULT_DATASET, **cfg.get("dataset", {})}
    reward_fn = DriveReward(DriveRewardConfig(**cfg.get("rl_reward", {})))
    safety = cfg.get("rl_safety", {})
    steer_cfg = dict(steer_limit=float(safety.get("steer_limit", 0.55)),
                     high_speed_steer_limit=float(safety.get("high_speed_steer_limit", 0.35)),
                     high_speed_threshold=float(safety.get("high_speed_threshold", 15.0)))
    base = args.recordings_dir or cfg.get("paths", {}).get("recordings_dir", "recordings")
    out = args.out_dir or os.path.join(args.logdir, "train_eps")
    os.makedirs(out, exist_ok=True)
    size = (args.size, args.size)
    stride = max(1, int(args.stride))
    # target colorspace follows the LIVE env's config switch, so demo and live obs always match
    channels = 1 if bool(cfg.get("capture", {}).get("grayscale", True)) else 3

    sessions = sorted(d for d in glob.glob(os.path.join(base, "*")) if os.path.isdir(d))
    n_eps, n_steps = 0, 0
    for sd in sessions:
        s = ds_mod.load_session(sd)
        if s is None:
            continue
        mask, _ = ds_mod._valid_mask(s, ds)
        valid = np.where(mask)[0]
        if len(valid) < args.min_len * stride:
            continue
        sess = os.path.basename(sd)
        if channels == 3 and np.asarray(s["frames"][valid[0]]).ndim == 2:
            # color target but a legacy GRAYSCALE session: no replication shim (a replicated-gray
            # demo corpus is a perfect demo/live discriminator for the reward head) - re-record.
            print(f"  {sess}: SKIPPED (grayscale session, color target - re-record in color)")
            continue
        # Demo QUALITY gate: only clean manual-analog driving may teach the policy (BC + demo
        # oversampling). ANNA autodrive sessions (near-zero steer, excluded as a teacher) and
        # bang-bang keyboard sessions (21% full-lock frames - they teach the exact saturation
        # collapse seen live) are demoted to wsx-*: still world-model food, never policy targets.
        steer_v = np.abs(np.asarray(s["actions"], np.float32)[valid, 0])
        bang_frac = float((steer_v > 0.99).mean())
        is_demo = ((args.autodrive_as_demo or not sess.startswith("autodrive"))
                   and bang_frac <= args.bang_bang_max)
        prefix = "ws" if is_demo else "wsx"
        runs = np.split(valid, np.where(np.diff(valid) > 1)[0] + 1)
        kept = 0
        for ri, run in enumerate(runs):
            if len(run) < args.min_len * stride:
                continue
            ep = _episode_from_run(s, run, reward_fn, size, stride, steer_cfg, channels)
            L = len(ep["reward"])
            with open(os.path.join(out, f"{prefix}-{sess}-{ri:03d}-{L}.npz"), "wb") as fh:
                np.savez_compressed(fh, **ep)
            n_eps += 1
            n_steps += L - 1
            kept += 1
        tag = "demo" if is_demo else f"wsx (bang-bang {bang_frac:.0%})" if bang_frac > args.bang_bang_max else "wsx (autodrive)"
        print(f"  {sess}: {kept} episodes [{tag}]")
    print(f"\nDONE: {n_eps} episodes, ~{n_steps} steps @{20 // stride}Hz -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

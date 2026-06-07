"""
build_centerline.py - Horizon FSD

Build a centerline polyline from a REFERENCE-LAP recording (made with the updated record.py,
which now logs world position). Resamples the driven path to ~equally-spaced points and saves
it for the RL progress reward (centerline.py / reward.py). Drive ONE clean lap along your
training route (stay roughly centered; the smoothing handles minor wobble).

Run:
    .\\.venv\\Scripts\\python.exe build_centerline.py --session recordings\\manual_<ts>
    .\\.venv\\Scripts\\python.exe build_centerline.py --session recordings\\manual_<ts> --out C:\\Horizon_FSD\\centerline.npy
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np


def main() -> int:
    p = argparse.ArgumentParser(description="Reference-lap recording -> centerline polyline.")
    p.add_argument("--session", required=True, help="recording session dir (or a single .npz)")
    p.add_argument("--out", default=r"C:\Horizon_FSD\centerline.npy")
    p.add_argument("--spacing", type=float, default=3.0, help="metres between centerline points")
    p.add_argument("--smooth", type=int, default=5, help="moving-average window (points); 0/1 = off")
    p.add_argument("--min-step", type=float, default=0.5, help="drop points closer than this (m)")
    args = p.parse_args()

    shards = (sorted(glob.glob(os.path.join(args.session, "*.npz")))
              if os.path.isdir(args.session) else [args.session])
    pos = []
    for s in shards:
        z = np.load(s)
        if "position" not in z.files:
            print(f"  [!] {os.path.basename(s)} has no 'position' (recorded before position logging) - skip")
            continue
        pos.append(z["position"])
    if not pos:
        print("no position data found - record a reference lap with the updated record.py first.")
        return 1
    pos = np.concatenate(pos, 0)
    xz = pos[:, [0, 2]].astype(np.float64)             # world X, Z = the ground plane (Y is up)

    # drop stationary / near-duplicate points
    keep = [0]
    for i in range(1, len(xz)):
        if np.hypot(*(xz[i] - xz[keep[-1]])) > args.min_step:
            keep.append(i)
    xz = xz[keep]
    if len(xz) < 2:
        print("reference lap too short / no movement captured.")
        return 1

    # resample to ~equal spacing along cumulative distance
    seg = np.hypot(*np.diff(xz, axis=0).T)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(cum[-1])
    n = max(2, int(total / args.spacing))
    s_new = np.linspace(0.0, total, n)
    out = np.stack([np.interp(s_new, cum, xz[:, 0]), np.interp(s_new, cum, xz[:, 1])], axis=1)

    k = int(args.smooth)
    if k > 1 and len(out) > k:
        if k % 2 == 0:
            k += 1                                     # odd window keeps length symmetric
        pad, ker = k // 2, np.ones(k) / k
        out = np.stack([                                # edge-pad (NOT zero-pad) so endpoints
            np.convolve(np.pad(out[:, 0], pad, mode="edge"), ker, mode="valid"),  # aren't dragged to origin
            np.convolve(np.pad(out[:, 1], pad, mode="edge"), ker, mode="valid"),
        ], axis=1)

    np.save(args.out, out.astype(np.float32))
    print(f"centerline: {len(out)} points spanning {total:.0f} m  ->  {args.out}")
    print("set reward to use it by leaving DriveRewardConfig.centerline_path at this path.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

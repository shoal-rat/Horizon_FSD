"""
capture_preview.py - Horizon FSD, Phase 1 [HUMAN-IN-THE-LOOP TEST]

Save one captured frame (full-res + the 84x84 observation) to PNGs so you can
confirm screen capture is actually grabbing FH6 (and the right monitor).

Run with FH6 visible:
    .\\.venv\\Scripts\\python.exe capture_preview.py
    .\\.venv\\Scripts\\python.exe capture_preview.py --window "Forza Horizon 6"
    .\\.venv\\Scripts\\python.exe capture_preview.py --monitor 2

Then open capture_raw.png. If it shows the game, capture is good. If it shows the
wrong screen, set capture.monitor_index or capture.window_name in config.yaml.
"""
from __future__ import annotations

import argparse
import os

import cv2
import numpy as np

from capture import ScreenCapture
from config import load_config


def main() -> int:
    p = argparse.ArgumentParser(description="Save a captured frame to confirm capture sees the game.")
    p.add_argument("--config", default=None)
    p.add_argument("--out-dir", default=".")
    p.add_argument("--monitor", type=int, default=None, help="Override capture.monitor_index.")
    p.add_argument("--window", default=None, help='Override capture.window_name, e.g. "Forza Horizon 6".')
    args = p.parse_args()

    cap_cfg = load_config(args.config)["capture"]
    sc = ScreenCapture(
        monitor_index=args.monitor if args.monitor is not None else cap_cfg.get("monitor_index", 1),
        window_name=args.window if args.window is not None else cap_cfg.get("window_name"),
        region=cap_cfg.get("region"),
        img_size=(cap_cfg["img_height"], cap_cfg["img_width"]),
        grayscale=cap_cfg.get("grayscale", True),
    )
    try:
        raw = sc.grab()
        obs = sc.observation()
    finally:
        sc.close()

    os.makedirs(args.out_dir, exist_ok=True)
    raw_path = os.path.join(args.out_dir, "capture_raw.png")
    obs_path = os.path.join(args.out_dir, "capture_obs.png")
    cv2.imwrite(raw_path, raw)
    cv2.imwrite(obs_path, cv2.resize(np.asarray(obs), (336, 336), interpolation=cv2.INTER_NEAREST))

    print(f" raw frame : {raw.shape}  ->  {os.path.abspath(raw_path)}")
    print(f" obs (84x84): {np.asarray(obs).shape}  ->  {os.path.abspath(obs_path)}  (upscaled x4)")
    print(f" obs mean pixel: {float(np.asarray(obs).mean()):.1f}  (near 0 => black/wrong source)")
    print(" Open capture_raw.png and confirm it shows Forza Horizon 6.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

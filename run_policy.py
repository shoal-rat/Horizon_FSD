"""
run_policy.py - Horizon FSD, Phase 3 [HUMAN-IN-THE-LOOP TEST]

Load a behavioral-cloning checkpoint and drive the FH6 env in real time. Watch it
drive and report how it does.

WARNING: the policy controls the car. Start parked in OFFLINE Free Roam on a road,
keep FH6 focused. Ctrl+C to stop.

Run:
    .\\.venv\\Scripts\\python.exe run_policy.py
    .\\.venv\\Scripts\\python.exe run_policy.py --checkpoint checkpoints\\bc_best.pt --duration 120
    .\\.venv\\Scripts\\python.exe run_policy.py --smooth 0.5    # EMA-smooth the actions
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from bc_model import load_policy, predict_action
from config import load_config
from forza_env import make_forza_env
from hitl import countdown


def main() -> int:
    p = argparse.ArgumentParser(description="Drive FH6 with a trained BC policy.")
    p.add_argument("--checkpoint", default=None, help="Default: <checkpoints_dir>/bc_best.pt")
    p.add_argument("--duration", type=float, default=120.0)
    p.add_argument("--countdown", type=float, default=5.0)
    p.add_argument("--smooth", type=float, default=0.0, help="EMA action smoothing in [0,1); 0 = off.")
    p.add_argument("--max-speed", type=float, default=0.0,
                   help="Cap cruise speed (km/h) so it doesn't floor into obstacles; 0 = off.")
    p.add_argument("--config", default=None)
    args = p.parse_args()

    cfg = load_config(args.config)
    ckpt = args.checkpoint or (cfg.get("paths", {}).get("checkpoints_dir", "checkpoints") + "/bc_best.pt")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, speed_norm = load_policy(ckpt, device=device)
    print(f"loaded {ckpt}  on {device}  (backbone={model.backbone_name})")

    env = make_forza_env(cfg)
    print(" action_space:", env.action_space)
    print(" Policy will DRIVE the car. Park on a road, keep FH6 focused. Ctrl+C to stop.")
    countdown(args.countdown)

    try:
        obs, info = env.reset()
        smoothed = np.zeros(3, dtype=np.float32)
        n = 0
        t0 = time.perf_counter()
        try:
            while time.perf_counter() - t0 < args.duration:
                frames = np.asarray(obs[0])        # (4,84,84) uint8
                speed = float(np.asarray(obs[1])[0])
                action = predict_action(model, frames, speed, speed_norm, device)
                if args.smooth > 0.0:
                    smoothed = args.smooth * smoothed + (1.0 - args.smooth) * action
                    action = smoothed.copy()
                if args.max_speed > 0.0:
                    # P-governor: cap throttle as we approach the target speed.
                    gov = float(np.clip((args.max_speed - speed * 3.6) / 10.0, 0.0, 1.0))
                    action[1] = min(float(action[1]), gov)
                obs, rew, terminated, truncated, info = env.step(action)
                n += 1
                if n % 20 == 0:
                    print(f"  step {n:5d}  speed={info.get('speed_kmh', 0):6.1f} km/h  "
                          f"steer={action[0]:+.2f} throttle={action[1]:.2f} brake={action[2]:.2f}")
                if terminated or truncated:
                    obs, info = env.reset()
        except KeyboardInterrupt:
            print("\n stopped.")
        print(f"\n ran {n} steps in {time.perf_counter()-t0:.1f}s "
              f"({n/max(1e-6, time.perf_counter()-t0):.1f} Hz)")
    finally:
        from forza_env import get_interface
        itf = get_interface()
        if itf is not None:
            itf.close()
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

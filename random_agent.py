"""
random_agent.py - Horizon FSD, Phase 1 [HUMAN-IN-THE-LOOP TEST]

Steps the env with RANDOM actions to confirm the whole real-time loop works
(capture -> observation, telemetry -> reward, action -> gamepad) and to measure
the real control-loop rate.

WARNING: this sends random steering/throttle/brake to the car. Park in OFFLINE
Free Roam somewhere open; the car WILL jerk around for the duration.

Run (FH6 focused, Data Out ON, parked):
    C:\\Horizon_FSD\\.venv\\Scripts\\python.exe random_agent.py --duration 30

It also runs WITHOUT the game (headless): capture grabs the desktop, telemetry is
absent (speed 0, reward 0), but the loop + timing are exercised.
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from config import load_config
from forza_env import get_interface, make_forza_env
from hitl import countdown


def describe(obs) -> str:
    parts = []
    for i, el in enumerate(obs):
        arr = np.asarray(el)
        parts.append(f"[{i}] {arr.shape} {arr.dtype}")
    return "  ".join(parts)


def main() -> int:
    p = argparse.ArgumentParser(description="Random-action smoke test for the FH6 env.")
    p.add_argument("--duration", type=float, default=30.0, help="Seconds to run.")
    p.add_argument("--countdown", type=float, default=5.0,
                   help="Seconds to alt-tab back into FH6 before stepping starts.")
    p.add_argument("--config", default=None, help="Path to config.yaml.")
    args = p.parse_args()

    cfg = load_config(args.config)
    print("=" * 72)
    print(" Horizon FSD - random-agent loop test")
    print("=" * 72)
    print(" Building env (starts screen capture, telemetry receiver, virtual gamepad)...")
    env = make_forza_env(cfg)
    print(" action_space     :", env.action_space)
    print(" observation_space:", env.observation_space)
    print("-" * 72)
    print(" Sending RANDOM actions - the car will move erratically. Ctrl+C to stop early.")
    countdown(args.countdown)

    try:
        obs, info = env.reset()
        interface = get_interface()
        print(" reset obs        :", describe(obs), "| info:", info)

        n = 0
        rew_sum = 0.0
        step_durations = []
        t0 = time.perf_counter()
        try:
            while time.perf_counter() - t0 < args.duration:
                action = env.action_space.sample()
                s = time.perf_counter()
                obs, rew, terminated, truncated, info = env.step(action)
                step_durations.append(time.perf_counter() - s)
                rew_sum += rew
                n += 1
                if n % 20 == 0:
                    print(f"  step {n:5d}  speed={info.get('speed_kmh', 0):6.1f} km/h  "
                          f"gear={info.get('gear')}  rew={rew:+.3f}  "
                          f"telemetry={'Y' if info.get('has_telemetry') else 'N'}")
                if terminated or truncated:
                    obs, info = env.reset()
        except KeyboardInterrupt:
            print("\n interrupted.")

        dt = time.perf_counter() - t0
        rx_packets = interface.rx.packet_count if interface else "?"
        cap_frames = interface.capture.frame_count if interface else "?"
        durs = np.array(step_durations) if step_durations else np.array([0.0])
        print("\n" + "=" * 72)
        print(" SUMMARY")
        print("-" * 72)
        print(f" steps               : {n}")
        print(f" wall time           : {dt:.1f}s")
        print(f" control rate        : {n/dt:.1f} steps/s  (target {1/cfg['env']['time_step_duration']:.0f} Hz)")
        print(f" step time mean/max  : {durs.mean()*1000:.1f} / {durs.max()*1000:.1f} ms")
        print(f" telemetry packets   : {rx_packets}  (0 => Data Out off or no game)")
        print(f" capture frames seen : {cap_frames}")
        print(f" total reward        : {rew_sum:+.2f}")
        print("=" * 72)
    finally:
        if interface is not None:
            interface.close()
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

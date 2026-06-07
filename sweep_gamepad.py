"""
sweep_gamepad.py - Horizon FSD, Phase 1 [HUMAN-IN-THE-LOOP TEST]

Confirms the virtual gamepad actually controls the car. It sweeps steering fully
left <-> right, then pulses throttle, then pulses brake, on a loop, printing each
action. Watch the game and confirm the car responds.

Requires `vgamepad` + the ViGEmBus driver (see README / gamepad.py).

Run (with FH6 in the foreground, parked in offline Free Roam):
    python sweep_gamepad.py
    python sweep_gamepad.py --duration 20 --hz 50

You have a few seconds after launch to click back into the game window.
"""
from __future__ import annotations

import argparse
import math
import time

from gamepad import ForzaGamepad
from hitl import countdown


def main() -> int:
    p = argparse.ArgumentParser(description="Sweep the virtual gamepad to verify in-game response.")
    p.add_argument("--duration", type=float, default=20.0, help="Total seconds to run.")
    p.add_argument("--hz", type=float, default=50.0, help="Update rate.")
    p.add_argument("--countdown", type=float, default=5.0,
                   help="Seconds to switch to the game window before sweeping starts.")
    args = p.parse_args()

    print("=" * 70)
    print(" Horizon FSD - gamepad sweep test")
    print("=" * 70)
    print(" Make sure:")
    print("   * ViGEmBus is installed and `pip install vgamepad` succeeded")
    print("   * FH6 is running, you are parked in OFFLINE Free Roam, in a car")
    print("   * the game window will have focus during the sweep")
    print("   * ideally no other controller is connected (so the virtual one wins)")
    print("-" * 70)

    pad = ForzaGamepad()
    try:
        countdown(args.countdown, "click into the game window now")

        dt = 1.0 / args.hz
        steps = int(args.duration * args.hz)
        print(" SWEEPING - watch the car's front wheels and throttle.\n")
        for i in range(steps):
            t = i * dt
            phase = t % 6.0
            if phase < 3.0:
                # Steering sine sweep, no throttle/brake.
                steer = math.sin(2.0 * math.pi * (phase / 3.0))
                throttle, brake = 0.0, 0.0
                mode = "STEER"
            elif phase < 4.5:
                # Throttle pulse, wheels centered.
                steer = 0.0
                throttle = 0.5 + 0.5 * math.sin(2.0 * math.pi * ((phase - 3.0) / 1.5))
                brake = 0.0
                mode = "THROTTLE"
            else:
                # Brake pulse.
                steer = 0.0
                throttle = 0.0
                brake = 0.5 + 0.5 * math.sin(2.0 * math.pi * ((phase - 4.5) / 1.5))
                mode = "BRAKE"

            pad.apply([steer, throttle, brake])
            if i % max(1, int(args.hz / 5)) == 0:  # ~5 prints/sec
                print(f"  [{mode:8}] steer={steer:+.2f}  throttle={throttle:.2f}  brake={brake:.2f}")
            time.sleep(dt)
    except KeyboardInterrupt:
        print("\n interrupted.")
    finally:
        pad.close()
        print("\n done - gamepad released. Did the car steer left/right, accelerate, and brake?")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

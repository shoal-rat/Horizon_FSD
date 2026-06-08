"""
reset_test.py - Horizon FSD, Phase 5 [HUMAN-IN-THE-LOOP TEST]

Validate the automated reset LADDER before trusting any unattended run. The script
drives the car itself (throttle + a weaving steer, so it crashes/leaves the road on
its own - the same single-input setup RL training uses), detects crash/stuck/flip
from telemetry, fires recovery (rewind -> reset-to-road), and logs whether the car
came back. Watch it and confirm it reliably recovers.

The research gate before an unattended night: ~30+ consecutive successful auto-resets
from varied states (including a flip) with zero un-recovered.

FH6 setup: offline Solo, damage None/Cosmetic (so REWIND isn't greyed out), a stable
car, an area with things to hit. Keep FH6 focused.

Run:
    .\\.venv\\Scripts\\python.exe reset_test.py --duration 600
To exercise the FLIPPED -> reset-to-road path, drive it onto a ramp/hillside.
"""
from __future__ import annotations

import argparse
import math
import time

from config import load_config
from gamepad import ForzaGamepad
from hitl import countdown
from recovery import CrashDetector, ForzaResetter
from telemetry_receiver import TelemetryReceiver


def main() -> int:
    p = argparse.ArgumentParser(description="Validate the automated crash-reset ladder.")
    p.add_argument("--duration", type=float, default=600.0)
    p.add_argument("--countdown", type=float, default=5.0)
    p.add_argument("--throttle", type=float, default=0.6, help="Throttle while cruising (induces crashes).")
    p.add_argument("--weave", type=float, default=0.55, help="Steering weave amplitude.")
    p.add_argument("--config", default=None)
    args = p.parse_args()

    cfg = load_config(args.config)
    tel = cfg["telemetry"]
    rx = TelemetryReceiver(host=tel.get("host", "0.0.0.0"), port=int(tel.get("port", 9999)))
    rx.start()
    pad = ForzaGamepad()
    detector = CrashDetector()
    resetter = ForzaResetter(pad, rx)

    print("=" * 74)
    print(" Horizon FSD - reset-ladder validation")
    print("=" * 74)
    print(" FH6: offline Solo, damage None/Cosmetic (so Rewind works), things to hit.")
    if not rx.wait_for_packet(5.0):
        print(" [!] No telemetry - is Data Out ON and are you driving?")
    print(" The script will DRIVE and crash on purpose, then auto-recover. Ctrl+C to stop.")
    countdown(args.countdown)

    stats = {"detections": 0, "rewind": 0, "reset_position": 0, "reset_to_road": 0,
             "autodrive": 0, "autodrive_persistent": 0, "FAILED": 0}
    by_reason: dict[str, int] = {}
    start = time.perf_counter()
    try:
        while time.perf_counter() - start < args.duration:
            loop_t = time.perf_counter()
            t = rx.latest()
            if t is None:
                time.sleep(0.05)
                continue

            reason = detector.update(t, loop_t, throttle_cmd=args.throttle)
            if reason:
                stats["detections"] += 1
                by_reason[reason] = by_reason.get(reason, 0) + 1
                print(f"\n[DETECT #{stats['detections']}] {reason}  "
                      f"(speed={t.speed_kmh:.0f}km/h roll={t.roll:+.2f} pitch={t.pitch:+.2f})")
                pad.reset()
                t0 = time.perf_counter()
                method = resetter.recover(reason)
                stats[method] = stats.get(method, 0) + 1
                ok = "OK" if method != "FAILED" else "*** FAILED ***"
                print(f"[RECOVER] {reason} -> {method} in {time.perf_counter()-t0:.1f}s  {ok}")
                detector.reset()
                continue

            # cruise: throttle + slow weave so it wanders into trouble
            steer = args.weave * math.sin(2.0 * math.pi * (loop_t - start) / 5.0)
            pad.apply([steer, args.throttle, 0.0])
            dt = 0.05 - (time.perf_counter() - loop_t)
            if dt > 0:
                time.sleep(dt)
    except KeyboardInterrupt:
        print("\n stopped.")
    finally:
        pad.close()
        rx.close()

    n = stats["detections"]
    ok = (stats["rewind"] + stats["reset_position"] + stats["reset_to_road"] +
          stats["autodrive"] + stats["autodrive_persistent"])
    print("\n" + "=" * 74)
    print(" RESET VALIDATION SUMMARY")
    print("-" * 74)
    print(f" crashes detected   : {n}   by reason: {by_reason}")
    print(f" recovered by rewind: {stats['rewind']}")
    print(f" recovered by reset : {stats['reset_position'] + stats['reset_to_road']}")
    print(f" recovered by AD    : {stats['autodrive'] + stats['autodrive_persistent']}")
    print(f" FAILED to recover  : {stats['FAILED']}")
    print(f" success rate       : {ok}/{n}" + (f"  ({100*ok/n:.0f}%)" if n else ""))
    print("=" * 74)
    print(" Gate before an unattended run: 30+ consecutive recoveries, 0 failures.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""
record.py - Horizon FSD, Phase 2 [HUMAN-IN-THE-LOOP: you drive]

Log synchronized (frame, telemetry, action) tuples at a fixed rate while you drive
FH6 manually, into compressed .npz shards for behavioral cloning.

The ACTION label is read from the telemetry's own steer/throttle/brake fields, so
each label is exactly what the car did (not what you think you pressed).

Frames are stored already preprocessed (downscaled grayscale, per config.yaml), to
keep shards small; dataset.py builds the frame stacks and filters bad frames.

Examples (FH6 in offline Free Roam, Data Out ON):
    .\\.venv\\Scripts\\python.exe record.py                       # record until Ctrl+C
    .\\.venv\\Scripts\\python.exe record.py --duration 300        # ~5 min
    .\\.venv\\Scripts\\python.exe record.py --autodrive           # tag as low-quality (ANNA AutoDrive)
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from capture import ScreenCapture
from config import load_config
from hitl import countdown
from telemetry_receiver import TelemetryReceiver


def main() -> int:
    p = argparse.ArgumentParser(description="Record manual driving for behavioral cloning.")
    p.add_argument("--duration", type=float, default=0.0, help="Seconds to record (0 = until Ctrl+C).")
    p.add_argument("--hz", type=float, default=20.0, help="Recording rate.")
    p.add_argument("--out-dir", default=None, help="Base dir (default: config paths.recordings_dir).")
    p.add_argument("--shard-size", type=int, default=1000, help="Samples per .npz shard.")
    p.add_argument("--autodrive", action="store_true",
                   help="Tag this session as low-quality ANNA AutoDrive data.")
    p.add_argument("--include-nondriving", action="store_true",
                   help="Also record frames where IsRaceOn=0 (menus/paused). Default: skip them.")
    p.add_argument("--countdown", type=float, default=5.0,
                   help="Seconds to alt-tab back into FH6 before recording starts.")
    p.add_argument("--config", default=None)
    args = p.parse_args()

    cfg = load_config(args.config)
    cap_cfg, tel_cfg = cfg["capture"], cfg["telemetry"]

    capture = ScreenCapture(
        monitor_index=cap_cfg.get("monitor_index", 1),
        window_name=cap_cfg.get("window_name"),
        region=cap_cfg.get("region"),
        img_size=(cap_cfg["img_height"], cap_cfg["img_width"]),
        grayscale=cap_cfg.get("grayscale", True),
    )
    rx = TelemetryReceiver(
        host=tel_cfg.get("host", "0.0.0.0"),
        port=int(tel_cfg.get("port", 9999)),
        recv_timeout=float(tel_cfg.get("recv_timeout_s", 1.0)),
    )
    rx.start()

    quality = "autodrive" if args.autodrive else "manual"
    out_base = args.out_dir or cfg.get("paths", {}).get("recordings_dir", "recordings")
    session = time.strftime("%Y%m%d_%H%M%S")
    session_dir = os.path.join(out_base, f"{quality}_{session}")
    os.makedirs(session_dir, exist_ok=True)

    print("=" * 72)
    print(f" Horizon FSD - recording ({quality}) -> {session_dir}")
    print("=" * 72)
    if not rx.wait_for_packet(5.0):
        print(" [!] No telemetry yet - is Data Out ON (127.0.0.1:%d) and are you driving?"
              % int(tel_cfg.get("port", 9999)))
    countdown(args.countdown, "switch to FH6 and start driving")

    # buffers
    keys = ("frames", "actions", "speed", "accel", "surface_rumble",
            "tire_slip", "is_race_on", "distance", "timestamp_ms", "position")
    buf: dict[str, list] = {k: [] for k in keys}
    shard_idx = 0
    total = 0
    skipped = 0

    def save_shard() -> None:
        nonlocal shard_idx
        if not buf["frames"]:
            return
        path = os.path.join(session_dir, f"shard_{shard_idx:04d}.npz")
        np.savez_compressed(
            path,
            frames=np.asarray(buf["frames"], dtype=np.uint8),
            actions=np.asarray(buf["actions"], dtype=np.float32),
            speed=np.asarray(buf["speed"], dtype=np.float32),
            accel=np.asarray(buf["accel"], dtype=np.float32),
            surface_rumble=np.asarray(buf["surface_rumble"], dtype=np.float32),
            tire_slip=np.asarray(buf["tire_slip"], dtype=np.float32),
            is_race_on=np.asarray(buf["is_race_on"], dtype=np.int8),
            distance=np.asarray(buf["distance"], dtype=np.float32),
            timestamp_ms=np.asarray(buf["timestamp_ms"], dtype=np.uint32),
            position=np.asarray(buf["position"], dtype=np.float32),   # (N,3) world x,y,z for centerline
            quality=quality,
        )
        print(f"  [shard] {path}  ({len(buf['frames'])} samples)")
        for k in keys:
            buf[k].clear()
        shard_idx += 1

    dt = 1.0 / args.hz
    t_start = time.perf_counter()
    t_end = (t_start + args.duration) if args.duration > 0 else float("inf")
    next_t = t_start
    print(" Recording. Drive! Ctrl+C to stop.\n")
    try:
        while time.perf_counter() < t_end:
            now = time.perf_counter()
            if now < next_t:
                time.sleep(next_t - now)
            next_t += dt

            t = rx.latest()
            if t is None:
                skipped += 1
                continue
            if not args.include_nondriving and not t.is_driving:
                skipped += 1
                continue

            buf["frames"].append(capture.observation())
            buf["actions"].append([t.steer_norm, t.throttle, t.brake])
            buf["speed"].append(t.speed)
            buf["accel"].append([t.acceleration_x, t.acceleration_y, t.acceleration_z])
            buf["surface_rumble"].append(t.mean_surface_rumble)
            buf["tire_slip"].append(t.mean_tire_slip_ratio)
            buf["is_race_on"].append(t.is_race_on)
            buf["distance"].append(t.distance_traveled)
            buf["timestamp_ms"].append(t.timestamp_ms)
            buf["position"].append([t.position_x, t.position_y, t.position_z])
            total += 1

            if total % int(args.hz) == 0:
                print(f"  {total:6d} samples  speed={t.speed_kmh:6.1f} km/h  "
                      f"steer={t.steer_norm:+.2f} throttle={t.throttle:.2f} brake={t.brake:.2f}")
            if len(buf["frames"]) >= args.shard_size:
                save_shard()
    except KeyboardInterrupt:
        print("\n stopping...")
    finally:
        save_shard()
        capture.close()
        rx.close()

    elapsed = time.perf_counter() - t_start
    meta = {
        "quality": quality,
        "session": session,
        "hz": args.hz,
        "img_size": [cap_cfg["img_height"], cap_cfg["img_width"]],
        "grayscale": bool(cap_cfg.get("grayscale", True)),
        "samples": total,
        "skipped": skipped,
        "shards": shard_idx,
        "seconds": round(elapsed, 1),
    }
    with open(os.path.join(session_dir, "meta.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)

    print("\n" + "=" * 72)
    print(f" DONE  {total} samples in {shard_idx} shard(s), {skipped} skipped, {elapsed:.1f}s")
    print(f" -> {session_dir}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

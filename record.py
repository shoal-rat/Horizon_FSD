"""
record.py - Horizon FSD, Phase 2 [HUMAN-IN-THE-LOOP: you drive]

Log synchronized (frame, telemetry, action) tuples at a fixed rate while you drive
FH6 manually, into compressed .npz shards.

The ACTION label is read from the telemetry's own steer/throttle/brake fields, so
each label is exactly what the car did (not what you think you pressed).

Frames are stored as 320x180 JPEG-encoded COLOR source frames (~21 MB/min): every
downstream observation (64x64 gray today, 64x64 color after the flip, any future
size) plus the racing-line cue and day/night classification are all REGENERABLE
offline from the same recording - the session never goes stale when the obs design
changes. Telemetry includes world position, so route-geometry obs backfill too.

Use an ANALOG gamepad. Keyboard play records bang-bang full-lock steering, which
teaches the policy the exact saturation collapse seen live - the recorder warns
and aborts if it detects it (override with --allow-bang-bang for a reason).

Examples (FH6 in offline Free Roam, chase cam, Data Out ON):
    .\\.venv\\Scripts\\python.exe record.py --tag "day fwd laps"
    .\\.venv\\Scripts\\python.exe record.py --duration 300 --tag "recovery cycles day"
    .\\.venv\\Scripts\\python.exe record.py --autodrive           # tag as low-quality (ANNA AutoDrive)
"""
from __future__ import annotations

import argparse
import json
import os
import time

import cv2
import numpy as np

from capture import ScreenCapture
from config import load_config
from hitl import countdown
from racing_line import RacingLineReader
from telemetry_receiver import TelemetryReceiver

SOURCE_W, SOURCE_H = 320, 180        # 16:9 color source frames; obs are derived offline
JPEG_QUALITY = 90
BANG_BANG_CHECK_AT = 600             # samples before the keyboard (full-lock) check
BANG_BANG_MAX_FRAC = 0.05


def main() -> int:
    p = argparse.ArgumentParser(description="Record manual driving for behavioral cloning.")
    p.add_argument("--duration", type=float, default=0.0, help="Seconds to record (0 = until Ctrl+C).")
    p.add_argument("--hz", type=float, default=20.0, help="Recording rate.")
    p.add_argument("--out-dir", default=None, help="Base dir (default: config paths.recordings_dir).")
    p.add_argument("--shard-size", type=int, default=1000, help="Samples per .npz shard.")
    p.add_argument("--autodrive", action="store_true",
                   help="Tag this session as low-quality ANNA AutoDrive data.")
    p.add_argument("--tag", default="",
                   help='Free-text session tag, e.g. "day fwd laps" / "night recovery" / "hairpins".')
    p.add_argument("--allow-bang-bang", action="store_true",
                   help="Skip the keyboard (full-lock steering) abort check.")
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
        img_size=(SOURCE_H, SOURCE_W),   # we store SOURCE frames; obs are derived offline
        grayscale=False,                 # color sources - gray obs are derivable, color isn't
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
    keys = ("frames_jpeg", "actions", "speed", "accel", "surface_rumble",
            "tire_slip", "is_race_on", "distance", "timestamp_ms", "position", "brightness")
    buf: dict[str, list] = {k: [] for k in keys}
    shard_idx = 0
    total = 0
    skipped = 0
    # AUTO day/night: lighting is identified PER FRAME (scene brightness in the line ROI), so one
    # session can sweep FH6's whole day/night cycle - no separate day/night recording or training.
    line_reader = RacingLineReader()
    lighting_counts = {"day": 0, "dusk": 0, "night": 0}

    def save_shard() -> None:
        nonlocal shard_idx
        if not buf["frames_jpeg"]:
            return
        path = os.path.join(session_dir, f"shard_{shard_idx:04d}.npz")
        jpeg = np.empty(len(buf["frames_jpeg"]), dtype=object)   # variable-length encoded frames
        jpeg[:] = buf["frames_jpeg"]
        np.savez_compressed(
            path,
            frames_jpeg=jpeg,
            actions=np.asarray(buf["actions"], dtype=np.float32),
            speed=np.asarray(buf["speed"], dtype=np.float32),
            accel=np.asarray(buf["accel"], dtype=np.float32),
            surface_rumble=np.asarray(buf["surface_rumble"], dtype=np.float32),
            tire_slip=np.asarray(buf["tire_slip"], dtype=np.float32),
            is_race_on=np.asarray(buf["is_race_on"], dtype=np.int8),
            distance=np.asarray(buf["distance"], dtype=np.float32),
            timestamp_ms=np.asarray(buf["timestamp_ms"], dtype=np.uint32),
            position=np.asarray(buf["position"], dtype=np.float32),   # (N,3) world x,y,z for centerline/route
            brightness=np.asarray(buf["brightness"], dtype=np.float32),  # per-frame scene lighting (auto day/night)
            quality=quality,
        )
        print(f"  [shard] {path}  ({len(buf['actions'])} samples)")
        for k in keys:
            buf[k].clear()
        shard_idx += 1

    dt = 1.0 / args.hz
    t_start = time.perf_counter()
    t_end = (t_start + args.duration) if args.duration > 0 else float("inf")
    next_t = t_start
    steer_hist: list[float] = []
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

            frame = capture.observation()              # (SOURCE_H, SOURCE_W, 3) BGR
            ok, enc = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            if not ok:
                skipped += 1
                continue
            buf["frames_jpeg"].append(enc.tobytes())
            buf["brightness"].append(line_reader.scene_brightness(frame))
            lighting_counts[line_reader.classify_lighting(frame)] += 1
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

            steer_hist.append(abs(t.steer_norm))
            if (not args.allow_bang_bang and not args.autodrive
                    and total == BANG_BANG_CHECK_AT):
                frac = sum(1 for v in steer_hist if v > 0.99) / max(1, len(steer_hist))
                if frac > BANG_BANG_MAX_FRAC:
                    print(f"\n [!] ABORT: {frac:.0%} of steering is at FULL LOCK - keyboard play? "
                          "Bang-bang demos teach the policy the exact saturation collapse seen live. "
                          "Use an ANALOG gamepad (or --allow-bang-bang to override).")
                    return 2

            if total % int(args.hz) == 0:
                light = line_reader.classify_lighting(frame)
                print(f"  {total:6d} samples  speed={t.speed_kmh:6.1f} km/h  "
                      f"steer={t.steer_norm:+.2f} throttle={t.throttle:.2f} brake={t.brake:.2f}  [{light}]")
            if len(buf["frames_jpeg"]) >= args.shard_size:
                save_shard()
    except KeyboardInterrupt:
        print("\n stopping...")
    finally:
        save_shard()
        capture.close()
        rx.close()

    elapsed = time.perf_counter() - t_start
    lit_total = max(1, sum(lighting_counts.values()))
    lighting_mix = {k: round(v / lit_total, 3) for k, v in lighting_counts.items()}
    meta = {
        "quality": quality,
        "tag": args.tag,
        "session": session,
        "hz": args.hz,
        "frame_format": f"jpeg_bgr_{SOURCE_W}x{SOURCE_H}_q{JPEG_QUALITY}",
        "cam": "chase",                  # racing_line.py's ROI is calibrated for the chase cam
        "lighting_mix": lighting_mix,    # AUTO-detected per frame; mixed sessions are fine
        "samples": total,
        "skipped": skipped,
        "shards": shard_idx,
        "seconds": round(elapsed, 1),
    }
    with open(os.path.join(session_dir, "meta.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)

    print("\n" + "=" * 72)
    print(f" DONE  {total} samples in {shard_idx} shard(s), {skipped} skipped, {elapsed:.1f}s")
    print(f" lighting (auto): day {lighting_mix['day']:.0%} / dusk {lighting_mix['dusk']:.0%} / "
          f"night {lighting_mix['night']:.0%}")
    print(f" -> {session_dir}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

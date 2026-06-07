#!/usr/bin/env python3
"""
telemetry_probe.py - Horizon FSD, Phase 0

Bind a UDP socket and inspect Forza Horizon 6 "Data Out" telemetry packets so we
can CONFIRM the real FH6 byte layout BEFORE writing a parser.

This is a STANDALONE diagnostic tool. It uses ONLY the Python standard library,
so you do NOT need to `pip install` anything to run it (any Python 3.8+ works,
including your system Python 3.13).

For every packet it:
  * prints the byte length and the sender address,
  * labels the length against known Forza formats
    (Sled=232, FM7 Dash=311, FH4/FH5 Dash=323/324, FM2023=331),
  * hex-dumps the raw bytes,
  * best-effort decodes the documented FH4/FH5 "Car Dash" fields.

  *** The decode is a HYPOTHESIS. ***  FH6 is new (May 2026). Community projects
  report the FH6 "Car Dash" payload is byte-for-byte identical to FH5 (324 bytes),
  but that has NOT been confirmed field-by-field against a live FH6 packet. The
  whole point of this probe is to verify it empirically.

On exit (Ctrl+C) it prints a summary; the most important line is the set of
DISTINCT PACKET LENGTHS observed - that is what confirms the format.

Usage:
  python telemetry_probe.py                      # listen on 0.0.0.0:9999
  python telemetry_probe.py --port 5606
  python telemetry_probe.py --save-raw first.bin # dump first packet for sharing
  python telemetry_probe.py --all                # print ALL decoded fields
"""
from __future__ import annotations

import argparse
import socket
import struct
import sys

# ---------------------------------------------------------------------------
# Forza "Data Out" V2 "Car Dash" field table  (Forza Horizon 4 / 5 layout).
#
# Layout notes (verified from richstokes/Forza-data-tools FH4_packetformat.dat,
# nikidziuba's parser, xxr0ss/fh5_telemetry, and the official FH5 forum spec;
# all little-endian, packed, NO padding):
#
#   * Sled block   : bytes   0..231  (IsRaceOn .. NumCylinders)
#   * Horizon block: bytes 232..243  (CarCategory + 2 unknown u32) - FH ONLY.
#                    This 12-byte block shifts PositionX to offset 244
#                    (in Forza Motorsport, PositionX is at 232 - do not confuse).
#   * Dash block   : bytes 244..323 (PositionX .. trailing byte)
#   * FH4/5 do NOT carry TireWear x4 / TrackOrdinal - those exist only in the
#     331-byte Forza Motorsport 2023 layout.
#   * Total = 324 bytes (some older FH4 builds send 323, omitting the last byte).
#
# struct codes: i=int32  I=uint32  f=float32  H=uint16  B=uint8  b=int8
# ---------------------------------------------------------------------------
FIELDS: list[tuple[str, str]] = [
    ("IsRaceOn", "i"),
    ("TimestampMS", "I"),
    ("EngineMaxRpm", "f"),
    ("EngineIdleRpm", "f"),
    ("CurrentEngineRpm", "f"),
    ("AccelerationX", "f"), ("AccelerationY", "f"), ("AccelerationZ", "f"),
    ("VelocityX", "f"), ("VelocityY", "f"), ("VelocityZ", "f"),
    ("AngularVelocityX", "f"), ("AngularVelocityY", "f"), ("AngularVelocityZ", "f"),
    ("Yaw", "f"), ("Pitch", "f"), ("Roll", "f"),
    ("NormSuspTravelFL", "f"), ("NormSuspTravelFR", "f"),
    ("NormSuspTravelRL", "f"), ("NormSuspTravelRR", "f"),
    ("TireSlipRatioFL", "f"), ("TireSlipRatioFR", "f"),
    ("TireSlipRatioRL", "f"), ("TireSlipRatioRR", "f"),
    ("WheelRotSpeedFL", "f"), ("WheelRotSpeedFR", "f"),
    ("WheelRotSpeedRL", "f"), ("WheelRotSpeedRR", "f"),
    ("WheelOnRumbleStripFL", "i"), ("WheelOnRumbleStripFR", "i"),
    ("WheelOnRumbleStripRL", "i"), ("WheelOnRumbleStripRR", "i"),
    ("WheelInPuddleDepthFL", "f"), ("WheelInPuddleDepthFR", "f"),
    ("WheelInPuddleDepthRL", "f"), ("WheelInPuddleDepthRR", "f"),
    ("SurfaceRumbleFL", "f"), ("SurfaceRumbleFR", "f"),
    ("SurfaceRumbleRL", "f"), ("SurfaceRumbleRR", "f"),
    ("TireSlipAngleFL", "f"), ("TireSlipAngleFR", "f"),
    ("TireSlipAngleRL", "f"), ("TireSlipAngleRR", "f"),
    ("TireCombinedSlipFL", "f"), ("TireCombinedSlipFR", "f"),
    ("TireCombinedSlipRL", "f"), ("TireCombinedSlipRR", "f"),
    ("SuspTravelMetersFL", "f"), ("SuspTravelMetersFR", "f"),
    ("SuspTravelMetersRL", "f"), ("SuspTravelMetersRR", "f"),
    ("CarOrdinal", "i"), ("CarClass", "i"), ("CarPerformanceIndex", "i"),
    ("DrivetrainType", "i"), ("NumCylinders", "i"),
    # ---- end of 232-byte Sled ----
    ("HorizonCarCategory", "i"), ("HorizonUnknown1", "I"), ("HorizonUnknown2", "I"),
    # ---- PositionX starts at offset 244 ----
    ("PositionX", "f"), ("PositionY", "f"), ("PositionZ", "f"),
    ("Speed", "f"), ("Power", "f"), ("Torque", "f"),
    ("TireTempFL", "f"), ("TireTempFR", "f"),
    ("TireTempRL", "f"), ("TireTempRR", "f"),
    ("Boost", "f"), ("Fuel", "f"), ("DistanceTraveled", "f"),
    ("BestLap", "f"), ("LastLap", "f"), ("CurrentLap", "f"), ("CurrentRaceTime", "f"),
    ("LapNumber", "H"),
    ("RacePosition", "B"),
    ("AccelInput", "B"), ("BrakeInput", "B"), ("ClutchInput", "B"),
    ("HandBrakeInput", "B"), ("Gear", "B"),
    ("Steer", "b"), ("NormalizedDrivingLine", "b"), ("NormalizedAIBrakeDifference", "b"),
    ("HorizonTrailingByte", "B"),
]

FH_FORMAT = "<" + "".join(code for _, code in FIELDS)
FH_SIZE = struct.calcsize(FH_FORMAT)
# Self-check: if this fails, the field table is wrong - fix it before trusting a decode.
assert FH_SIZE == 324, f"FH 'Car Dash' field table must total 324 bytes, got {FH_SIZE}"

# Pre-compute each field's byte offset (enables decoding short packets too).
_FIELD_OFFSETS: list[tuple[str, str, int]] = []
_off = 0
for _name, _code in FIELDS:
    _FIELD_OFFSETS.append((_name, _code, _off))
    _off += struct.calcsize("<" + _code)

KNOWN_LENGTHS = {
    232: "Forza 'Sled' (physics only - no position/inputs)",
    311: "Forza Motorsport 7 'Dash'",
    323: "Forza Horizon 4/5 'Dash' (no trailing byte)",
    324: "Forza Horizon 4/5/6 'Car Dash'  <-- expected for FH6",
    331: "Forza Motorsport 2023 'Dash'",
}

# Fields worth printing for a quick human sanity-check.
KEY_FIELDS = [
    "IsRaceOn", "TimestampMS", "CurrentEngineRpm", "Gear",
    "Speed", "VelocityZ", "PositionX", "PositionY", "PositionZ",
    "AccelInput", "BrakeInput", "Steer", "DistanceTraveled",
]


def decode_fields(data: bytes) -> dict[str, float | int]:
    """Best-effort decode using the FH4/5 'Car Dash' hypothesis.

    Decodes only fields that fit inside the actual packet length, so a short
    packet (e.g. a 232-byte Sled) still yields its physics fields.
    """
    out: dict[str, float | int] = {}
    for name, code, off in _FIELD_OFFSETS:
        size = struct.calcsize("<" + code)
        if off + size <= len(data):
            (out[name],) = struct.unpack_from("<" + code, data, off)
    return out


def hexdump(data: bytes, limit: int) -> str:
    lines: list[str] = []
    n = len(data) if limit <= 0 else min(limit, len(data))
    for i in range(0, n, 16):
        chunk = data[i:i + 16]
        hex_part = " ".join(f"{b:02x}" for b in chunk)
        ascii_part = "".join(chr(b) if 32 <= b < 127 else "." for b in chunk)
        lines.append(f"  {i:04d}  {hex_part:<47}  {ascii_part}")
    if 0 < n < len(data):
        lines.append(f"  ....  (+{len(data) - n} more bytes; use --hex-bytes 0 to see all)")
    return "\n".join(lines)


def format_key_fields(d: dict) -> str:
    parts: list[str] = []
    if "IsRaceOn" in d:
        parts.append(f"IsRaceOn={d['IsRaceOn']}")
    if "Speed" in d:
        ms = d["Speed"]
        parts.append(f"Speed={ms:7.2f} m/s  ({ms * 3.6:6.1f} km/h / {ms * 2.237:6.1f} mph)")
    if "CurrentEngineRpm" in d:
        parts.append(f"RPM={d['CurrentEngineRpm']:.0f}")
    if "Gear" in d:
        parts.append(f"Gear={d['Gear']}")
    if "AccelInput" in d:
        parts.append(f"Throttle={d['AccelInput']}/255")
    if "BrakeInput" in d:
        parts.append(f"Brake={d['BrakeInput']}/255")
    if "Steer" in d:
        parts.append(f"Steer={d['Steer']:+d}/127")
    if "PositionX" in d:
        parts.append(f"Pos=({d['PositionX']:.1f}, {d['PositionY']:.1f}, {d['PositionZ']:.1f})")
    if "DistanceTraveled" in d:
        parts.append(f"Dist={d['DistanceTraveled']:.1f} m")
    return "\n    ".join(parts)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Forza Horizon 6 'Data Out' UDP telemetry probe (Horizon FSD, Phase 0).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--host", default="0.0.0.0",
                   help="Interface to bind (default: all interfaces).")
    p.add_argument("--port", type=int, default=9999,
                   help="UDP port to listen on; MUST match the in-game Data Out port. "
                        "Avoid 5200-5300 (reserved by the game).")
    p.add_argument("--hex-bytes", type=int, default=64,
                   help="Bytes of hex dump to print per packet (0 = whole packet).")
    p.add_argument("--full-packets", type=int, default=3,
                   help="Show full hexdump+decode for the first N packets, then compact lines.")
    p.add_argument("--every", type=int, default=30,
                   help="After --full-packets, print one compact line every Nth packet.")
    p.add_argument("--all", action="store_true",
                   help="Decode and print ALL fields (not just key ones) for the full packets.")
    p.add_argument("--save-raw", metavar="PATH", default=None,
                   help="Write the bytes of the first received packet to PATH.")
    p.add_argument("--max-packets", type=int, default=0,
                   help="Stop after N packets (0 = run until Ctrl+C).")
    args = p.parse_args(argv)

    bar = "=" * 78
    print(bar)
    print(" Horizon FSD - Phase 0 telemetry probe")
    print(bar)
    print(f" Listening on  udp://{args.host}:{args.port}")
    print(" In-game:  Settings -> HUD and Gameplay -> Data Out = ON")
    print(f"           IP Address = 127.0.0.1 ,  Port = {args.port}")
    print(" Then DRIVE - telemetry is only sent while you are actively driving")
    print(" (no data in menus, pause, replay, or rewind).")
    print(" Press Ctrl+C to stop and print a summary.")
    print("-" * 78)
    print(f" NOTE: the field decode assumes the FH4/FH5 'Car Dash' layout ({FH_SIZE} bytes).")
    print("       FH6 is unconfirmed - we verify via the observed packet length(s)")
    print("       and whether the decoded values look sane (Speed, Gear, Throttle...).")
    print(bar)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind((args.host, args.port))
    except OSError as e:
        print(f"\n[ERROR] Could not bind {args.host}:{args.port} -> {e}", file=sys.stderr)
        print("        Is another app already listening on that UDP port?", file=sys.stderr)
        return 1
    sock.settimeout(1.0)

    count = 0
    lengths_seen: dict[int, int] = {}
    idle_secs = 0
    warned_idle = False
    first_saved = False

    try:
        while True:
            try:
                data, addr = sock.recvfrom(8192)
            except socket.timeout:
                if count == 0:
                    idle_secs += 1
                    if idle_secs >= 6 and not warned_idle:
                        print("\n[..] No packets yet. Check:")
                        print("     - Data Out is ON and you are actively DRIVING (not in a menu)")
                        print("     - IP = 127.0.0.1 and the in-game Port matches --port above")
                        print(f"     - Windows Firewall allows inbound UDP on port {args.port}")
                        print("     - The game window has focus and is not paused\n")
                        warned_idle = True
                continue

            count += 1
            lengths_seen[len(data)] = lengths_seen.get(len(data), 0) + 1

            if args.save_raw and not first_saved:
                with open(args.save_raw, "wb") as fh:
                    fh.write(data)
                print(f"[saved] first packet ({len(data)} bytes) -> {args.save_raw}")
                first_saved = True

            label = KNOWN_LENGTHS.get(len(data), "UNKNOWN length - please paste this hexdump!")
            decoded = decode_fields(data)

            if count <= args.full_packets:
                print(f"\n#{count}  len={len(data)} bytes from {addr[0]}:{addr[1]}  [{label}]")
                print(hexdump(data, args.hex_bytes))
                if args.all:
                    for name, _code, off in _FIELD_OFFSETS:
                        if name in decoded:
                            print(f"    @{off:<3} {name:<28} = {decoded[name]}")
                else:
                    print("    " + format_key_fields(decoded))
            elif args.every > 0 and count % args.every == 0:
                spd = decoded.get("Speed")
                spd_s = f"{spd * 3.6:6.1f} km/h" if isinstance(spd, float) else "  n/a    "
                print(f"#{count:<6} len={len(data)}  IsRaceOn={decoded.get('IsRaceOn', '?')}  "
                      f"Speed={spd_s}  Gear={decoded.get('Gear', '?')}  "
                      f"Throttle={decoded.get('AccelInput', '?')}  Steer={decoded.get('Steer', '?')}")

            if args.max_packets and count >= args.max_packets:
                break
    except KeyboardInterrupt:
        pass
    finally:
        sock.close()

    print("\n" + bar)
    print(" SUMMARY")
    print("-" * 78)
    print(f" total packets received : {count}")
    if lengths_seen:
        print(" distinct packet lengths (THIS is what confirms the format):")
        for ln in sorted(lengths_seen):
            lab = KNOWN_LENGTHS.get(ln, "UNKNOWN - paste a full hexdump of this length!")
            print(f"     {ln:>4} bytes  x{lengths_seen[ln]:<7} {lab}")
    else:
        print(" no packets received - see the troubleshooting hints above.")
    print(bar)
    print(" Next: paste this summary + one full-packet hexdump back here to start Phase 1.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# Forza "Data Out" telemetry format reference

This is the working reference for the UDP telemetry parser. It is a **hypothesis
for FH6**, assembled from FH4/FH5 sources, to be **confirmed empirically** by
`telemetry_probe.py` in Phase 0.

## Summary of what we know

- **FH6 officially supports "Data Out"** UDP telemetry (Forza Support doc
  `support.forza.net/.../51744149102611`). One-way UDP, sent at the game's frame
  rate, works to `127.0.0.1`.
- The FH6 **"Car Dash" payload is reported byte-for-byte identical to FH5** (a
  single fixed **324-byte** format) by community projects
  (`satyajiit/forza-horizon-6-moza-bridge`, `TheBanHammer/fh6-tel`). **Not yet
  confirmed field-by-field against a live FH6 packet** - that is Phase 0's job.
- **No game-mandated default port.** You pick the port in-game; your app binds the
  same one. Community apps use various ports (fh6-tel: 20440, moza-bridge: 4009).
  **Hard rule from the official doc: avoid ports 5200-5300** (the game binds its
  own outgoing socket there). Our default is 9999 (safe, outside that range).
- Data is sent **only while actively driving** - nothing in menus, pause, replay,
  rewind, or after a race finishes. Detect driving state via `IsRaceOn` / a packet
  timeout.

## Packet sizes (reconciled, internally consistent)

| Format                         | Bytes | Notes                                              |
|--------------------------------|------:|----------------------------------------------------|
| Sled (physics only)            |   232 | `IsRaceOn` .. `NumCylinders`                       |
| Forza Motorsport 7 "Dash"      |   311 | Sled + dash, **no** Horizon 12-byte block          |
| **Forza Horizon 4/5/6 "Dash"** | **324** | Sled + **12-byte Horizon block** + dash + trailing |
| Forza Horizon 4 (older builds) |   323 | same as above without the final trailing byte      |
| Forza Motorsport 2023 "Dash"   |   331 | FM7 + `TireWear` x4 + `TrackOrdinal` (**not** FH!)  |

**Critical FH-vs-FM difference:** Forza Horizon inserts a **12-byte block**
(`CarCategory` s32 @232 + 2 undocumented u32 @236/@240) immediately after the Sled,
**shifting `PositionX` to offset 244**. In Forza Motorsport `PositionX` is at 232.
Do not reuse a Motorsport offset table for Horizon.

**FH does NOT carry** `TireWear` or `TrackOrdinal`; those are Motorsport-2023-only.

## FH4/5/6 "Car Dash" field table (324 bytes, little-endian, packed)

`struct` codes: `i`=int32, `I`=uint32, `f`=float32, `H`=uint16, `B`=uint8, `b`=int8.
This table is the single source of truth in `telemetry_probe.py`, where
`struct.calcsize` asserts it totals exactly 324 bytes.

| Offset | Type | Field |
|------:|------|-------|
| 0   | s32 | IsRaceOn (1 = driving, 0 = menu/paused) |
| 4   | u32 | TimestampMS |
| 8   | f32 | EngineMaxRpm |
| 12  | f32 | EngineIdleRpm |
| 16  | f32 | CurrentEngineRpm |
| 20  | f32 | AccelerationX/Y/Z (local: X=right, Y=up, Z=fwd) |
| 32  | f32 | VelocityX/Y/Z |
| 44  | f32 | AngularVelocityX/Y/Z (pitch/yaw/roll) |
| 56  | f32 | Yaw, Pitch, Roll |
| 68  | f32 | NormSuspensionTravel FL/FR/RL/RR |
| 84  | f32 | TireSlipRatio FL/FR/RL/RR |
| 100 | f32 | WheelRotationSpeed FL/FR/RL/RR |
| 116 | s32 | WheelOnRumbleStrip FL/FR/RL/RR |
| 132 | f32 | WheelInPuddleDepth FL/FR/RL/RR |
| 148 | f32 | SurfaceRumble FL/FR/RL/RR |
| 164 | f32 | TireSlipAngle FL/FR/RL/RR |
| 180 | f32 | TireCombinedSlip FL/FR/RL/RR |
| 196 | f32 | SuspensionTravelMeters FL/FR/RL/RR |
| 212 | s32 | CarOrdinal |
| 216 | s32 | CarClass (0=D .. 7=X) |
| 220 | s32 | CarPerformanceIndex (100..999) |
| 224 | s32 | DrivetrainType (0=FWD, 1=RWD, 2=AWD) |
| 228 | s32 | NumCylinders  *(end of Sled @232)* |
| 232 | s32 | HorizonCarCategory  *(FH-only 12-byte block)* |
| 236 | u32 | HorizonUnknown1 (undocumented) |
| 240 | u32 | HorizonUnknown2 (undocumented) |
| 244 | f32 | PositionX/Y/Z (meters) |
| 256 | f32 | Speed (m/s) |
| 260 | f32 | Power (watts) |
| 264 | f32 | Torque (Nm) |
| 268 | f32 | TireTemp FL/FR/RL/RR |
| 284 | f32 | Boost |
| 288 | f32 | Fuel |
| 292 | f32 | DistanceTraveled |
| 296 | f32 | BestLap, LastLap, CurrentLap, CurrentRaceTime |
| 312 | u16 | LapNumber |
| 314 | u8  | RacePosition |
| 315 | u8  | AccelInput / throttle (0..255) |
| 316 | u8  | BrakeInput (0..255) |
| 317 | u8  | ClutchInput (0..255) |
| 318 | u8  | HandBrakeInput (0..255) |
| 319 | u8  | Gear (0 = reverse; neutral encoding varies by car - verify) |
| 320 | s8  | Steer (-127..127) |
| 321 | s8  | NormalizedDrivingLine |
| 322 | s8  | NormalizedAIBrakeDifference |
| 323 | u8  | trailing byte (present in 324-byte packets; absent in 323-byte builds) |

## Open questions to resolve in Phase 0

1. **Is the FH6 packet actually 324 bytes?** (323 and 331 would mean a different
   layout - paste a hexdump if so.)
2. **Does `PositionX` really sit at offset 244?** Confirm by checking the decoded
   `Speed` (offset 256) reads a sane m/s while driving, and `PositionX/Y/Z` change
   smoothly as you move.
3. **Gear / neutral encoding** for the cars we use.
4. The two **`HorizonUnknown` u32s** (@236/@240) - leave as raw unknowns unless a
   use emerges.

## Sources

- `richstokes/Forza-data-tools` (FH4_packetformat.dat) - verified offsets
- `nikidziuba/Forza_horizon_data_out_python` (12-byte "hzn" block)
- `xxr0ss/fh5_telemetry` (`DATA_SIZE = 324`)
- Official FH5 forum: "Data Out Telemetry Variables and Structure"
- `satyajiit/forza-horizon-6-moza-bridge`, `TheBanHammer/fh6-tel` (FH6 == FH5 claim)
- Official FH6 Data Out doc: `support.forza.net/hc/en-us/articles/51744149102611`

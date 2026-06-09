"""
forza_telemetry.py - Horizon FSD, Phase 1

Parser for Forza Horizon 6 "Data Out" UDP telemetry. Turns a 324-byte "Car Dash"
packet into a typed, immutable `ForzaTelemetry` dataclass plus driving-oriented
derived values (normalized inputs, speed in km/h, off-road proxies).

Pure standard library (struct + dataclasses) - importable and testable with no
third-party dependencies.

The byte layout (offsets, types) was CONFIRMED against live FH6 in Phase 0:
324-byte packets, ~60 Hz, with IsRaceOn @0, Speed @256, AccelInput @315,
Steer @320, Gear @319 all reading sane values while driving. See
docs/telemetry_format.md for the full table and sources.

`SPEC` here is the canonical layout; an import-time assert keeps it byte-identical
to telemetry_probe.FIELDS so the two never drift.
"""
from __future__ import annotations

import dataclasses
import math
import struct
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Canonical field layout: (snake_case_name, struct_code), in packet order.
# Codes: i=int32  I=uint32  f=float32  H=uint16  B=uint8  b=int8.  Little-endian.
# ---------------------------------------------------------------------------
SPEC: list[tuple[str, str]] = [
    ("is_race_on", "i"),
    ("timestamp_ms", "I"),
    ("engine_max_rpm", "f"),
    ("engine_idle_rpm", "f"),
    ("current_engine_rpm", "f"),
    ("acceleration_x", "f"), ("acceleration_y", "f"), ("acceleration_z", "f"),
    ("velocity_x", "f"), ("velocity_y", "f"), ("velocity_z", "f"),
    ("angular_velocity_x", "f"), ("angular_velocity_y", "f"), ("angular_velocity_z", "f"),
    ("yaw", "f"), ("pitch", "f"), ("roll", "f"),
    ("norm_susp_travel_fl", "f"), ("norm_susp_travel_fr", "f"),
    ("norm_susp_travel_rl", "f"), ("norm_susp_travel_rr", "f"),
    ("tire_slip_ratio_fl", "f"), ("tire_slip_ratio_fr", "f"),
    ("tire_slip_ratio_rl", "f"), ("tire_slip_ratio_rr", "f"),
    ("wheel_rot_speed_fl", "f"), ("wheel_rot_speed_fr", "f"),
    ("wheel_rot_speed_rl", "f"), ("wheel_rot_speed_rr", "f"),
    ("wheel_on_rumble_strip_fl", "i"), ("wheel_on_rumble_strip_fr", "i"),
    ("wheel_on_rumble_strip_rl", "i"), ("wheel_on_rumble_strip_rr", "i"),
    ("wheel_in_puddle_depth_fl", "f"), ("wheel_in_puddle_depth_fr", "f"),
    ("wheel_in_puddle_depth_rl", "f"), ("wheel_in_puddle_depth_rr", "f"),
    ("surface_rumble_fl", "f"), ("surface_rumble_fr", "f"),
    ("surface_rumble_rl", "f"), ("surface_rumble_rr", "f"),
    ("tire_slip_angle_fl", "f"), ("tire_slip_angle_fr", "f"),
    ("tire_slip_angle_rl", "f"), ("tire_slip_angle_rr", "f"),
    ("tire_combined_slip_fl", "f"), ("tire_combined_slip_fr", "f"),
    ("tire_combined_slip_rl", "f"), ("tire_combined_slip_rr", "f"),
    ("susp_travel_meters_fl", "f"), ("susp_travel_meters_fr", "f"),
    ("susp_travel_meters_rl", "f"), ("susp_travel_meters_rr", "f"),
    ("car_ordinal", "i"), ("car_class", "i"), ("car_performance_index", "i"),
    ("drivetrain_type", "i"), ("num_cylinders", "i"),
    # ---- end of 232-byte Sled ----
    ("horizon_car_category", "i"), ("horizon_unknown1", "I"), ("horizon_unknown2", "I"),
    # ---- PositionX starts at offset 244 ----
    ("position_x", "f"), ("position_y", "f"), ("position_z", "f"),
    ("speed", "f"), ("power", "f"), ("torque", "f"),
    ("tire_temp_fl", "f"), ("tire_temp_fr", "f"),
    ("tire_temp_rl", "f"), ("tire_temp_rr", "f"),
    ("boost", "f"), ("fuel", "f"), ("distance_traveled", "f"),
    ("best_lap", "f"), ("last_lap", "f"), ("current_lap", "f"), ("current_race_time", "f"),
    ("lap_number", "H"),
    ("race_position", "B"),
    ("accel_input", "B"), ("brake_input", "B"), ("clutch_input", "B"),
    ("handbrake_input", "B"), ("gear", "B"),
    ("steer", "b"), ("normalized_driving_line", "b"), ("normalized_ai_brake_difference", "b"),
    ("horizon_trailing_byte", "B"),
]

STRUCT_FORMAT = "<" + "".join(code for _, code in SPEC)
_STRUCT = struct.Struct(STRUCT_FORMAT)
PACKET_SIZE = _STRUCT.size
assert PACKET_SIZE == 324, f"FH 'Car Dash' must be 324 bytes, got {PACKET_SIZE}"

# Gear sentinel emitted transiently during shifts / clutch (observed live in Phase 0).
GEAR_SHIFTING = 11

_FLOAT_PCT = 1.0 / 255.0   # u8 input -> 0..1
_STEER_DIV = 1.0 / 127.0   # s8 steer -> -1..1


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v


@dataclass(slots=True, frozen=True, repr=False)
class ForzaTelemetry:
    """One decoded FH6 'Car Dash' telemetry packet.

    Raw fields mirror the packet exactly (units: Speed/Velocity m/s, Position m,
    Power W, Torque Nm, AccelInput/BrakeInput 0..255, Steer -127..127).
    Use the derived properties for driving-friendly values.
    """
    is_race_on: int
    timestamp_ms: int
    engine_max_rpm: float
    engine_idle_rpm: float
    current_engine_rpm: float
    acceleration_x: float
    acceleration_y: float
    acceleration_z: float
    velocity_x: float
    velocity_y: float
    velocity_z: float
    angular_velocity_x: float
    angular_velocity_y: float
    angular_velocity_z: float
    yaw: float
    pitch: float
    roll: float
    norm_susp_travel_fl: float
    norm_susp_travel_fr: float
    norm_susp_travel_rl: float
    norm_susp_travel_rr: float
    tire_slip_ratio_fl: float
    tire_slip_ratio_fr: float
    tire_slip_ratio_rl: float
    tire_slip_ratio_rr: float
    wheel_rot_speed_fl: float
    wheel_rot_speed_fr: float
    wheel_rot_speed_rl: float
    wheel_rot_speed_rr: float
    wheel_on_rumble_strip_fl: int
    wheel_on_rumble_strip_fr: int
    wheel_on_rumble_strip_rl: int
    wheel_on_rumble_strip_rr: int
    wheel_in_puddle_depth_fl: float
    wheel_in_puddle_depth_fr: float
    wheel_in_puddle_depth_rl: float
    wheel_in_puddle_depth_rr: float
    surface_rumble_fl: float
    surface_rumble_fr: float
    surface_rumble_rl: float
    surface_rumble_rr: float
    tire_slip_angle_fl: float
    tire_slip_angle_fr: float
    tire_slip_angle_rl: float
    tire_slip_angle_rr: float
    tire_combined_slip_fl: float
    tire_combined_slip_fr: float
    tire_combined_slip_rl: float
    tire_combined_slip_rr: float
    susp_travel_meters_fl: float
    susp_travel_meters_fr: float
    susp_travel_meters_rl: float
    susp_travel_meters_rr: float
    car_ordinal: int
    car_class: int
    car_performance_index: int
    drivetrain_type: int
    num_cylinders: int
    horizon_car_category: int
    horizon_unknown1: int
    horizon_unknown2: int
    position_x: float
    position_y: float
    position_z: float
    speed: float
    power: float
    torque: float
    tire_temp_fl: float
    tire_temp_fr: float
    tire_temp_rl: float
    tire_temp_rr: float
    boost: float
    fuel: float
    distance_traveled: float
    best_lap: float
    last_lap: float
    current_lap: float
    current_race_time: float
    lap_number: int
    race_position: int
    accel_input: int
    brake_input: int
    clutch_input: int
    handbrake_input: int
    gear: int
    steer: int
    normalized_driving_line: int
    normalized_ai_brake_difference: int
    horizon_trailing_byte: int

    # ---- constructors -----------------------------------------------------
    @classmethod
    def from_bytes(cls, payload: bytes) -> "ForzaTelemetry":
        """Parse a raw UDP payload. Accepts 324-byte packets (and 323-byte
        older-build packets, padded). Raises ValueError on any other length."""
        n = len(payload)
        if n == PACKET_SIZE:
            obj = cls(*_STRUCT.unpack(payload))
        elif n == PACKET_SIZE - 1:  # 323-byte build: missing the trailing byte
            obj = cls(*_STRUCT.unpack(payload + b"\x00"))
        else:
            raise ValueError(
                f"unexpected packet length {n}; expected {PACKET_SIZE} "
                f"(or {PACKET_SIZE - 1}) for the FH 'Car Dash' format"
            )
        # Reject a garbage frame at the boundary: one NaN/inf physics field would otherwise
        # silently poison reward/detector/centerline/obs (and the async learner's gradients).
        # The receiver catches this ValueError, drops the frame, and keeps the last good one.
        for v in (obj.speed, obj.velocity_z, obj.position_x, obj.position_z,
                  obj.roll, obj.pitch, obj.angular_velocity_y,
                  obj.surface_rumble_fl, obj.surface_rumble_fr,
                  obj.surface_rumble_rl, obj.surface_rumble_rr,
                  obj.tire_slip_ratio_fl, obj.tire_slip_ratio_fr,
                  obj.tire_slip_ratio_rl, obj.tire_slip_ratio_rr):
            if not math.isfinite(v):
                raise ValueError("non-finite physics field in Data Out packet")
        return obj

    # ---- driving-friendly derived values ----------------------------------
    @property
    def is_driving(self) -> bool:
        """True when the car is on track (telemetry physics are live)."""
        return self.is_race_on != 0

    @property
    def speed_kmh(self) -> float:
        return self.speed * 3.6

    @property
    def speed_mph(self) -> float:
        return self.speed * 2.2369362921

    @property
    def throttle(self) -> float:
        """Applied throttle, 0..1 (from the 0..255 AccelInput)."""
        return self.accel_input * _FLOAT_PCT

    @property
    def brake(self) -> float:
        """Applied brake, 0..1 (from the 0..255 BrakeInput)."""
        return self.brake_input * _FLOAT_PCT

    @property
    def clutch(self) -> float:
        return self.clutch_input * _FLOAT_PCT

    @property
    def handbrake(self) -> float:
        return self.handbrake_input * _FLOAT_PCT

    @property
    def steer_norm(self) -> float:
        """Applied steering, -1..1 (from the -127..127 Steer byte)."""
        return _clamp(self.steer * _STEER_DIV, -1.0, 1.0)

    @property
    def is_shifting(self) -> bool:
        return self.gear == GEAR_SHIFTING

    @property
    def position(self) -> tuple[float, float, float]:
        return (self.position_x, self.position_y, self.position_z)

    @property
    def velocity(self) -> tuple[float, float, float]:
        return (self.velocity_x, self.velocity_y, self.velocity_z)

    @property
    def acceleration(self) -> tuple[float, float, float]:
        return (self.acceleration_x, self.acceleration_y, self.acceleration_z)

    @property
    def forward_speed(self) -> float:
        """Ground speed (m/s), frame-independent. NOTE: Velocity X/Y/Z are WORLD-frame in Data
        Out, so velocity_z is NOT local forward speed (it only equals it when the car points
        along world +Z). We return the scalar `speed`; for a SIGNED forward component, project
        velocity onto heading via yaw - but verify the yaw sign against live FH6 first."""
        return self.speed

    @property
    def mean_tire_slip_ratio(self) -> float:
        """Mean |slip ratio| across 4 tires - a wheelspin / loss-of-grip proxy."""
        return (abs(self.tire_slip_ratio_fl) + abs(self.tire_slip_ratio_fr)
                + abs(self.tire_slip_ratio_rl) + abs(self.tire_slip_ratio_rr)) / 4.0

    @property
    def mean_tire_combined_slip(self) -> float:
        return (abs(self.tire_combined_slip_fl) + abs(self.tire_combined_slip_fr)
                + abs(self.tire_combined_slip_rl) + abs(self.tire_combined_slip_rr)) / 4.0

    @property
    def mean_surface_rumble(self) -> float:
        """Mean surface rumble across 4 tires - a rough-surface / off-road proxy."""
        return (self.surface_rumble_fl + self.surface_rumble_fr
                + self.surface_rumble_rl + self.surface_rumble_rr) / 4.0

    def __repr__(self) -> str:
        return (
            f"ForzaTelemetry(race={'on' if self.is_race_on else 'off'}, "
            f"speed={self.speed_kmh:6.1f}km/h, rpm={self.current_engine_rpm:5.0f}, "
            f"gear={self.gear}, throttle={self.throttle:.2f}, brake={self.brake:.2f}, "
            f"steer={self.steer_norm:+.2f}, "
            f"pos=({self.position_x:.1f},{self.position_y:.1f},{self.position_z:.1f}))"
        )


def parse(payload: bytes) -> ForzaTelemetry:
    """Module-level alias for ForzaTelemetry.from_bytes()."""
    return ForzaTelemetry.from_bytes(payload)


# --- import-time integrity checks: fail loudly if the layout ever drifts ----
_DC_FIELDS = [f.name for f in dataclasses.fields(ForzaTelemetry)]
assert _DC_FIELDS == [n for n, _ in SPEC], (
    "ForzaTelemetry field order drifted from SPEC - they must match exactly "
    "because from_bytes() unpacks positionally."
)

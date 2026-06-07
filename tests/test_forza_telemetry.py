"""
Unit tests for forza_telemetry.py (stdlib unittest - no pytest needed).

Run from the project root:
    python -m unittest tests.test_forza_telemetry -v
or:
    python tests/test_forza_telemetry.py

The synthetic-packet tests pack values at the LITERAL documented byte offsets
(independent of SPEC ordering), so they act as an external oracle that would
catch any mistake in the field table.
"""
import os
import struct
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import forza_telemetry as ft  # noqa: E402
import telemetry_probe as probe  # noqa: E402


def make_packet(**overrides) -> bytes:
    """Build a 324-byte FH 'Car Dash' packet, zero-filled, with fields set at
    their documented byte offsets."""
    buf = bytearray(324)
    offsets = {
        "is_race_on": (0, "<i"),
        "timestamp_ms": (4, "<I"),
        "current_engine_rpm": (16, "<f"),
        "velocity_z": (40, "<f"),
        "car_ordinal": (212, "<i"),
        "horizon_car_category": (232, "<i"),
        "position_x": (244, "<f"),
        "position_y": (248, "<f"),
        "position_z": (252, "<f"),
        "speed": (256, "<f"),
        "distance_traveled": (292, "<f"),
        "lap_number": (312, "<H"),
        "race_position": (314, "<B"),
        "accel_input": (315, "<B"),
        "brake_input": (316, "<B"),
        "gear": (319, "<B"),
        "steer": (320, "<b"),
    }
    for name, value in overrides.items():
        off, fmt = offsets[name]
        struct.pack_into(fmt, buf, off, value)
    return bytes(buf)


class TestLayout(unittest.TestCase):
    def test_struct_size_is_324(self):
        self.assertEqual(ft.PACKET_SIZE, 324)

    def test_spec_matches_probe_binary_layout(self):
        # Same struct codes in the same order => byte-identical layouts.
        self.assertEqual(
            [c for _, c in ft.SPEC],
            [c for _, c in probe.FIELDS],
            "forza_telemetry.SPEC and telemetry_probe.FIELDS describe different layouts",
        )

    def test_dataclass_field_order_matches_spec(self):
        import dataclasses
        names = [f.name for f in dataclasses.fields(ft.ForzaTelemetry)]
        self.assertEqual(names, [n for n, _ in ft.SPEC])


class TestParsing(unittest.TestCase):
    def test_driving_packet_decodes(self):
        pkt = make_packet(
            is_race_on=1, current_engine_rpm=4500.0,
            position_x=1000.5, position_y=12.25, position_z=-2000.0,
            speed=50.0, velocity_z=49.0, distance_traveled=1234.5,
            accel_input=200, brake_input=10, gear=3, steer=-64,
            race_position=1, lap_number=2, car_ordinal=2387,
        )
        t = ft.parse(pkt)
        self.assertTrue(t.is_driving)
        self.assertEqual(t.gear, 3)
        self.assertEqual(t.car_ordinal, 2387)
        self.assertAlmostEqual(t.speed, 50.0, places=4)
        self.assertAlmostEqual(t.speed_kmh, 180.0, places=3)
        self.assertAlmostEqual(t.position_x, 1000.5, places=3)
        self.assertAlmostEqual(t.position_z, -2000.0, places=3)
        self.assertAlmostEqual(t.forward_speed, 49.0, places=3)
        self.assertAlmostEqual(t.throttle, 200 / 255, places=6)
        self.assertAlmostEqual(t.brake, 10 / 255, places=6)
        self.assertAlmostEqual(t.steer_norm, -64 / 127, places=6)
        self.assertEqual(t.race_position, 1)

    def test_parked_packet(self):
        t = ft.parse(make_packet(is_race_on=0, timestamp_ms=12345))
        self.assertFalse(t.is_driving)
        self.assertEqual(t.timestamp_ms, 12345)
        self.assertEqual(t.speed, 0.0)

    def test_gear_shifting_sentinel(self):
        self.assertTrue(ft.parse(make_packet(gear=ft.GEAR_SHIFTING)).is_shifting)
        self.assertFalse(ft.parse(make_packet(gear=3)).is_shifting)

    def test_steer_is_clamped(self):
        # -128 (full s8) should clamp to -1.0, not exceed it.
        self.assertEqual(ft.parse(make_packet(steer=-128)).steer_norm, -1.0)

    def test_323_byte_packet_is_padded(self):
        pkt = make_packet(is_race_on=1, speed=10.0)[:323]
        t = ft.parse(pkt)
        self.assertTrue(t.is_driving)
        self.assertAlmostEqual(t.speed, 10.0, places=4)

    def test_wrong_length_raises(self):
        for bad in (0, 232, 311, 325, 331):
            with self.assertRaises(ValueError):
                ft.parse(bytes(bad))


class TestRealPacket(unittest.TestCase):
    """Decode the real packet saved by the Phase 0 probe, if present."""

    def test_real_first_packet(self):
        path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "first_packet.bin",
        )
        if not os.path.exists(path):
            self.skipTest("first_packet.bin not present")
        with open(path, "rb") as fh:
            data = fh.read()
        self.assertEqual(len(data), 324)
        t = ft.parse(data)  # must not raise
        # It was captured parked, so race should be off and speed zero.
        self.assertFalse(t.is_driving)
        self.assertEqual(t.speed, 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)

"""
test_stress.py - Horizon FSD

Adversarial / property tests that hammer the strategies (centerline, reward, detector, telemetry
parse) with strange inputs - NaN/inf, teleport jumps, the route seam/end, reverse, off-route,
slow-but-advancing, hard braking vs a crash, and malformed packets. These lock down the hardening
fixes so the strategy stops needing per-incident patches.
"""
import math
import os
import struct
import sys
import unittest
from dataclasses import dataclass

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import forza_telemetry as ft  # noqa: E402
from centerline import Centerline  # noqa: E402
from recovery import CrashDetector, DetectorConfig  # noqa: E402
from reward import DriveReward, DriveRewardConfig  # noqa: E402


def straight_centerline(n=11, step=10.0):
    """A straight reference path along +x, total length (n-1)*step (default 100 m)."""
    return Centerline(np.stack([np.arange(n) * step, np.zeros(n)], axis=1))


@dataclass
class Tel:
    speed: float = 0.0
    is_driving: bool = True
    mean_surface_rumble: float = 0.0
    mean_tire_slip_ratio: float = 0.0
    angular_velocity_y: float = 0.0
    roll: float = 0.0
    pitch: float = 0.0
    position_x: float = 0.0
    position_z: float = 0.0


class TestCenterlineRobust(unittest.TestCase):
    def setUp(self):
        self.cl = straight_centerline()  # length 100

    def test_arclength_monotonic_and_on_line(self):
        prev = -1.0
        for x in range(0, 101, 5):
            s, lat, _ = self.cl.project(float(x), 0.0)
            self.assertGreaterEqual(s + 1e-6, prev)
            prev = s
            self.assertLess(lat, 1e-6)

    def test_past_open_end_reports_perpendicular_not_overshoot(self):
        s, lat, at_end = self.cl.project(110.0, 3.0)  # 10 m past the end, 3 m to the side
        self.assertTrue(at_end)
        self.assertAlmostEqual(lat, 3.0, places=3)    # perpendicular distance, NOT ~10 m overshoot

    def test_nonfinite_position_fails_safe(self):
        for bad in (float("nan"), float("inf")):
            s, lat, at_end = self.cl.project(bad, 0.0)
            self.assertFalse(math.isfinite(s))
            self.assertEqual(lat, float("inf"))       # inf > offroute_dist -> treated off-route, not missed
            self.assertFalse(at_end)

    def test_point_on_vertex(self):
        s, lat, _ = self.cl.project(20.0, 0.0)
        self.assertAlmostEqual(s, 20.0, places=3)
        self.assertLess(lat, 1e-6)

    def test_single_segment(self):
        cl = Centerline(np.array([[0.0, 0.0], [10.0, 0.0]]))
        s, lat, _ = cl.project(5.0, 2.0)
        self.assertAlmostEqual(s, 5.0, places=3)
        self.assertAlmostEqual(lat, 2.0, places=3)


class TestRewardNeverNonFinite(unittest.TestCase):
    def test_finite_output_for_adversarial_inputs(self):
        r = DriveReward(DriveRewardConfig(centerline_path=""))
        speeds = [0.0, -5.0, 1e9, float("nan"), float("inf"), -float("inf")]
        rumbles = [0.0, 0.5, float("nan")]
        acts = [None,
                np.array([0.0, 1.0, 0.0], np.float32),
                np.array([float("nan"), 1.0, 0.0], np.float32),
                np.array([1e9, 1e9, 1e9], np.float32)]
        for sp in speeds:
            for rum in rumbles:
                for a in acts:
                    val = r(Tel(speed=sp, mean_surface_rumble=rum), Tel(), a, a)
                    self.assertTrue(math.isfinite(val), f"non-finite reward: speed={sp} rumble={rum}")

    def test_teleport_jump_gives_no_credit(self):
        r = DriveReward(DriveRewardConfig(centerline_path=""))
        r._centerline = straight_centerline()
        a = np.array([0.0, 1.0, 0.0], np.float32)
        val = r(Tel(speed=20.0, position_x=95.0), Tel(speed=20.0, position_x=5.0), a, a)  # 90 m jump
        self.assertLessEqual(val, 0.01)               # beyond progress_jump_guard -> no progress, no boot


class TestDetectorScenarios(unittest.TestCase):
    def mk(self, **kw):
        d = CrashDetector(DetectorConfig(centerline_path="", **kw))
        d._centerline = straight_centerline()         # length 100
        return d

    def test_offroute_far_from_route(self):
        d = self.mk(offroute_seconds=0.3)
        t = Tel(speed=20.0, position_x=50.0, position_z=50.0)
        self.assertIsNone(d.update(t, 0.0))
        self.assertEqual(d.update(t, 0.5), "offroute")

    def test_nan_position_fails_safe_to_offroute(self):
        d = self.mk(offroute_seconds=0.3)
        t = Tel(speed=20.0, position_x=float("nan"), position_z=float("nan"))
        self.assertIsNone(d.update(t, 0.0))
        self.assertEqual(d.update(t, 0.5), "offroute")   # not silently missed

    def test_route_end_is_route_complete_not_noprogress(self):
        d = self.mk(noprogress_seconds=0.5, noprogress_speed=1.0)
        self.assertIsNone(d.update(Tel(speed=20.0, position_x=98.0), 0.0))
        self.assertEqual(d.update(Tel(speed=20.0, position_x=99.0), 1.0), "route_complete")

    def test_slow_but_advancing_is_not_noprogress(self):
        d = self.mk(noprogress_seconds=1.0, noprogress_min_advance=3.0)
        for i, x in enumerate(range(0, 30, 2)):       # 2 m/step forward -> always advancing
            self.assertIsNone(d.update(Tel(speed=2.0, position_x=float(x)), i * 1.0))

    def test_circling_at_speed_is_noprogress(self):
        d = self.mk(noprogress_seconds=1.0, noprogress_speed=3.0)
        t = Tel(speed=15.0, position_x=50.0)          # moving fast but arc-length stuck
        self.assertIsNone(d.update(t, 0.0))
        self.assertEqual(d.update(t, 1.2), "noprogress")

    def test_braking_is_not_impact_but_a_crash_is(self):
        slow = CrashDetector(DetectorConfig(centerline_path=""))
        slow.update(Tel(speed=12.0), 0.0)
        self.assertIsNone(slow.update(Tel(speed=7.0), 0.4))   # -5 m/s over a 0.4 s tick = braking, not a crash
        crash = CrashDetector(DetectorConfig(centerline_path=""))
        crash.update(Tel(speed=12.0), 0.0)
        self.assertEqual(crash.update(Tel(speed=7.0), 0.05), "impact")  # -5 m/s in one 0.05 s tick = crash

    def test_teleport_jump_is_suppressed(self):
        d = self.mk(offroute_seconds=0.3, teleport_jump_m=30.0)
        self.assertIsNone(d.update(Tel(speed=20.0, position_x=10.0), 0.0))           # on-route
        # a 60 m one-tick jump (fast-travel/respawn) to off-route: NOT a crash this tick - re-anchor
        self.assertIsNone(d.update(Tel(speed=20.0, position_x=10.0, position_z=60.0), 0.05))
        self.assertIsNone(d._best_arc)                                               # re-anchored

    def test_gpu_stall_tick_does_not_fire_impact(self):
        d = CrashDetector(DetectorConfig(centerline_path=""))
        d.update(Tel(speed=20.0), 0.0)
        d.update(Tel(speed=20.0), 0.05)
        self.assertIsNone(d.update(Tel(speed=14.0), 0.55))   # 0.5 s stall, speed bled off = braking, not a crash

    def test_uphill_crawl_is_not_stuck(self):
        d = CrashDetector(DetectorConfig(centerline_path="", stuck_seconds=1.0, stuck_displacement_m=1.0))
        reason, x = None, 0.0
        for i in range(40):                                  # 2 s of 1.5 m/s crawl, +0.15 m/tick = 6 m covered
            reason = d.update(Tel(speed=1.5, position_x=x), i * 0.05, throttle_cmd=1.0, brake_cmd=0.0)
            x += 0.15
            if reason:
                break
        self.assertIsNone(reason)

    def test_wedged_car_is_stuck(self):
        d = CrashDetector(DetectorConfig(centerline_path="", stuck_seconds=1.0, stuck_displacement_m=1.0))
        reason = None
        for i in range(40):                                  # slow + throttle + NOT moving in the world
            reason = d.update(Tel(speed=0.0, position_x=5.0), i * 0.05, throttle_cmd=1.0, brake_cmd=0.0)
            if reason:
                break
        self.assertEqual(reason, "stuck")


class TestTelemetryPacket(unittest.TestCase):
    def test_valid_zero_packet_parses(self):
        ft.ForzaTelemetry.from_bytes(bytes(bytearray(324)))   # all-finite zeros

    def test_nonfinite_field_is_rejected(self):
        for offset in (244, 256):                     # position_x, speed - both in the finite-check set
            buf = bytearray(324)
            struct.pack_into("<f", buf, offset, float("nan"))
            with self.assertRaises(ValueError):
                ft.ForzaTelemetry.from_bytes(bytes(buf))

    def test_wrong_length_is_rejected(self):
        with self.assertRaises(ValueError):
            ft.ForzaTelemetry.from_bytes(b"\x00" * 100)


if __name__ == "__main__":
    unittest.main(verbosity=2)

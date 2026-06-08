import os
import sys
import unittest
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from recovery import CrashDetector, DetectorConfig, ForzaResetter, ResetConfig  # noqa: E402


@dataclass
class FakeTelemetry:
    speed: float = 0.0
    is_driving: bool = True
    mean_surface_rumble: float = 0.0
    roll: float = 0.0
    pitch: float = 0.0
    position_x: float = 0.0
    position_z: float = 0.0


class TestCrashDetector(unittest.TestCase):
    def test_on_road_idle_does_not_become_stuck_without_throttle(self):
        detector = CrashDetector(DetectorConfig(stuck_seconds=1.0, stuck_hard_seconds=2.0))
        t = FakeTelemetry(speed=0.0, mean_surface_rumble=0.0)
        self.assertIsNone(detector.update(t, 0.0, throttle_cmd=0.0, brake_cmd=0.0))
        self.assertIsNone(detector.update(t, 5.0, throttle_cmd=0.0, brake_cmd=0.0))

    def test_throttle_against_obstacle_becomes_stuck(self):
        detector = CrashDetector(DetectorConfig(stuck_seconds=1.0))
        t = FakeTelemetry(speed=0.0, mean_surface_rumble=0.0)
        self.assertIsNone(detector.update(t, 0.0, throttle_cmd=0.5, brake_cmd=0.0))
        self.assertEqual(detector.update(t, 1.2, throttle_cmd=0.5, brake_cmd=0.0), "stuck")

    def test_impact_speed_drop_still_detects(self):
        detector = CrashDetector(DetectorConfig(impact_speed_drop=4.0))
        self.assertIsNone(detector.update(FakeTelemetry(speed=12.0), 0.0))
        self.assertEqual(detector.update(FakeTelemetry(speed=7.0), 0.1), "impact")


class FakePad:
    def reset(self):
        pass

    def tap_button(self, *args, **kwargs):
        pass

    def apply(self, *args, **kwargs):
        pass


class FakeRx:
    def latest(self):
        return FakeTelemetry(speed=6.0, mean_surface_rumble=0.0)


class TestForzaResetter(unittest.TestCase):
    def test_autodrive_requires_actual_displacement(self):
        resetter = ForzaResetter(
            FakePad(),
            FakeRx(),
            ResetConfig(autodrive_min_displacement=10.0, press_gap_s=0.0),
        )
        resetter._position = lambda: (100.0, 200.0)
        resetter._confirm_autodrive = lambda: True
        resetter._wait_on_road = lambda: True
        self.assertFalse(resetter.autodrive_reset())

    def test_recovered_state_rejects_position_far_from_centerline(self):
        resetter = ForzaResetter(
            FakePad(),
            FakeRx(),
            ResetConfig(route_max_dist=5.0, require_route_if_available=True),
        )
        resetter._centerline = type("FakeCenterline", (), {
            "project": staticmethod(lambda x, z: (0.0, 12.0))
        })()
        t = FakeTelemetry(speed=6.0, mean_surface_rumble=0.0)
        self.assertFalse(resetter._is_recovered(t, require_speed=3.0))

    def test_recovered_state_accepts_position_near_centerline(self):
        resetter = ForzaResetter(
            FakePad(),
            FakeRx(),
            ResetConfig(route_max_dist=5.0, require_route_if_available=True),
        )
        resetter._centerline = type("FakeCenterline", (), {
            "project": staticmethod(lambda x, z: (0.0, 2.0))
        })()
        t = FakeTelemetry(speed=6.0, mean_surface_rumble=0.0)
        self.assertTrue(resetter._is_recovered(t, require_speed=3.0))


if __name__ == "__main__":
    unittest.main(verbosity=2)

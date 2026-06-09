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
    def __init__(self, on_tap=None):
        self.on_tap = on_tap
        self.taps = []
        self.actions = []

    def reset(self):
        pass

    def tap_button(self, button_name, *args, **kwargs):
        self.taps.append(button_name)
        if self.on_tap is not None:
            self.on_tap(button_name)

    def apply(self, action, *args, **kwargs):
        self.actions.append(action)


class FakeRx:
    def latest(self):
        return FakeTelemetry(speed=6.0, mean_surface_rumble=0.0)


class TestForzaResetter(unittest.TestCase):
    def test_autodrive_accepts_route_verified_recovery_without_large_displacement(self):
        resetter = ForzaResetter(
            FakePad(),
            FakeRx(),
            ResetConfig(autodrive_min_displacement=10.0, press_gap_s=0.0),
        )
        resetter._position = lambda: (100.0, 200.0)
        resetter._wait_autodrive_resolved = lambda start_pos: (True, False)
        self.assertTrue(resetter.autodrive_reset())

    def test_autodrive_accepts_optional_teleport_prompt_with_a(self):
        state = {"prompt": True}

        def on_tap(button):
            if button == "A":
                state["prompt"] = False

        class PromptRx:
            def latest(self):
                if state["prompt"]:
                    return FakeTelemetry(speed=0.0, is_driving=False)
                return FakeTelemetry(speed=0.0, is_driving=True,
                                     mean_surface_rumble=0.0,
                                     position_x=1.0, position_z=1.0)

        pad = FakePad(on_tap=on_tap)
        resetter = ForzaResetter(
            pad,
            PromptRx(),
            ResetConfig(
                press_gap_s=0.0,
                autodrive_timeout_s=0.5,
                autodrive_prompt_retry_s=0.0,
                autodrive_on_route_settle_s=0.0,
            ),
        )
        resetter._centerline = type("FakeCenterline", (), {
            "project": staticmethod(lambda x, z: (0.0, 0.0))
        })()

        self.assertTrue(resetter.autodrive_reset())
        self.assertIn("A", pad.taps)

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

    def test_recover_falls_through_to_pause_reset_when_autodrive_fails(self):
        # AutoDrive is the primary recovery (it teleports-far / drives-stuck itself); the pause
        # reset is the LAST rung. When rewind + AutoDrive fail, the ladder falls to the pause reset.
        resetter = ForzaResetter(
            FakePad(), FakeRx(),
            ResetConfig(autodrive_persistent_retry_s=0.0, heartbeat_every=1000),
        )
        resetter.rewind = lambda: False
        resetter.autodrive_reset = lambda: False
        resetter.reset_to_road = lambda: False
        resetter.reset_position = lambda: True          # only the pause reset works
        self.assertEqual(resetter.recover("stuck"), "reset_position")

    def test_recover_non_persistent_returns_failed_when_nothing_works(self):
        resetter = ForzaResetter(
            FakePad(), FakeRx(),
            ResetConfig(autodrive_persistent=False, autodrive_persistent_retry_s=0.0,
                        heartbeat_every=1000),
        )
        for name in ("rewind", "autodrive_reset", "reset_position", "reset_to_road"):
            setattr(resetter, name, lambda: False)
        self.assertEqual(resetter.recover("flipped", max_attempts=3), "FAILED")


if __name__ == "__main__":
    unittest.main(verbosity=2)

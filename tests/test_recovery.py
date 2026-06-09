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
        detector = CrashDetector(DetectorConfig(stuck_seconds=1.0, stuck_hard_seconds=2.0, centerline_path=""))
        t = FakeTelemetry(speed=0.0, mean_surface_rumble=0.0)
        self.assertIsNone(detector.update(t, 0.0, throttle_cmd=0.0, brake_cmd=0.0))
        self.assertIsNone(detector.update(t, 5.0, throttle_cmd=0.0, brake_cmd=0.0))

    def test_throttle_against_obstacle_becomes_stuck(self):
        detector = CrashDetector(DetectorConfig(stuck_seconds=1.0, centerline_path=""))
        t = FakeTelemetry(speed=0.0, mean_surface_rumble=0.0)
        self.assertIsNone(detector.update(t, 0.0, throttle_cmd=0.5, brake_cmd=0.0))
        self.assertEqual(detector.update(t, 1.2, throttle_cmd=0.5, brake_cmd=0.0), "stuck")

    def test_impact_speed_drop_still_detects(self):
        detector = CrashDetector(DetectorConfig(impact_speed_drop=4.0, centerline_path=""))
        self.assertIsNone(detector.update(FakeTelemetry(speed=12.0), 0.0))
        # one normal ~0.05s tick: a 5 m/s drop exceeds the dt-scaled threshold -> impact
        self.assertEqual(detector.update(FakeTelemetry(speed=7.0), 0.05), "impact")

    def test_impact_not_fired_when_slow_tick_explains_the_drop(self):
        # over a long (over-run) tick, a moderate speed loss is normal braking, not a crash
        detector = CrashDetector(DetectorConfig(impact_speed_drop=4.0, centerline_path=""))
        self.assertIsNone(detector.update(FakeTelemetry(speed=12.0), 0.0))
        self.assertIsNone(detector.update(FakeTelemetry(speed=7.0), 0.4))  # dt=0.4 -> threshold ~-8

    def test_fast_offroad_resets_regardless_of_speed(self):
        # the old speed<10 gate let a car driving FAST off-road run forever without a reset
        detector = CrashDetector(DetectorConfig(offroad_seconds=0.5, centerline_path=""))
        t = FakeTelemetry(speed=30.0, mean_surface_rumble=0.5)
        self.assertIsNone(detector.update(t, 0.0))
        self.assertEqual(detector.update(t, 1.0), "offroad")

    def test_offroute_terminates_when_far_from_centerline(self):
        detector = CrashDetector(DetectorConfig(offroute_dist=18.0, offroute_seconds=0.5, centerline_path=""))
        detector._centerline = type("FC", (), {"project": staticmethod(lambda x, z: (0.0, 50.0, False)),
                                                "length": 5000.0})()
        t = FakeTelemetry(speed=20.0, position_x=1.0, position_z=1.0)
        self.assertIsNone(detector.update(t, 0.0))
        self.assertEqual(detector.update(t, 1.0), "offroute")

    def test_noprogress_terminates_when_on_route_but_not_advancing(self):
        detector = CrashDetector(DetectorConfig(noprogress_seconds=1.0, noprogress_min_advance=3.0,
                                                offroute_dist=18.0, centerline_path=""))
        detector._centerline = type("FC", (), {"project": staticmethod(lambda x, z: (100.0, 2.0, False)),
                                                "length": 5000.0})()
        t = FakeTelemetry(speed=15.0, position_x=1.0, position_z=1.0)
        self.assertIsNone(detector.update(t, 0.0))
        self.assertEqual(detector.update(t, 1.2), "noprogress")


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
            ResetConfig(press_gap_s=0.0),
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

    def test_confirm_a_fires_after_coasting_then_frozen(self):
        # THE bug from the live "Fast Travel Warning" prompt: a car still COASTING when recovery starts
        # tripped the old cumulative-displacement `driving` flag, which BLOCKED the confirm-A that accepts
        # the teleport prompt. Now the A is gated on the car being positionally FROZEN, so it fires even
        # after a coast-down. (Calling _wait_autodrive_resolved directly, so the only A taps are the prompt
        # confirms - not _open_autodrive's menu-select A.)
        coast = [(0.0, 0.0), (5.0, 0.0), (10.0, 0.0)]   # rolls forward (sets `driving`), then holds at (10,0)

        class CoastRx:
            def __init__(self):
                self.i = 0

            def latest(self):
                p = coast[self.i] if self.i < len(coast) else (10.0, 0.0)
                self.i += 1
                return FakeTelemetry(speed=0.0, is_driving=False, position_x=p[0], position_z=p[1])

        pad = FakePad()
        resetter = ForzaResetter(
            pad, CoastRx(),
            ResetConfig(press_gap_s=0.0, autodrive_timeout_s=1.0, autodrive_prompt_retry_s=0.0,
                        autodrive_prompt_settle_s=0.2, autodrive_frozen_eps=0.5,
                        autodrive_teleport_jump_m=30.0),
        )
        resetter._wait_autodrive_resolved((0.0, 0.0))
        self.assertIn("A", pad.taps)                    # confirmed the prompt despite the initial coast

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
        # AutoDrive is the primary recovery (it teleports-far / drives-stuck itself); the pause reset
        # is the LAST rung. With the car still NOT drivable, a failed AutoDrive falls to the pause reset.
        class StuckRx:                                  # car off-road, not recovered -> no offroute-ok short-circuit
            def latest(self): return FakeTelemetry(speed=0.0, mean_surface_rumble=0.5)
        resetter = ForzaResetter(
            FakePad(), StuckRx(),
            ResetConfig(autodrive_persistent_retry_s=0.0, heartbeat_every=1000),
        )
        resetter.autodrive_reset = lambda: False
        resetter.reset_to_road = lambda: False
        resetter.reset_position = lambda: True          # only the pause reset works
        self.assertEqual(resetter.recover("stuck"), "reset_position")

    def test_recover_offroute_ok_when_autodrive_leaves_car_drivable(self):
        # AutoDrive left the car upright on a real road (just off our route) -> accept it, don't teleport.
        resetter = ForzaResetter(FakePad(), FakeRx(),  # FakeRx = speed 6, on-road, upright
                                 ResetConfig(autodrive_persistent_retry_s=0.0, heartbeat_every=1000))
        resetter.autodrive_reset = lambda: False
        self.assertEqual(resetter.recover("offroute"), "autodrive_offroute_ok")

    def test_recover_non_persistent_returns_failed_when_nothing_works(self):
        class StuckRx:
            def latest(self): return FakeTelemetry(speed=0.0, mean_surface_rumble=0.5)
        resetter = ForzaResetter(
            FakePad(), StuckRx(),
            ResetConfig(autodrive_persistent=False, autodrive_persistent_retry_s=0.0,
                        heartbeat_every=1000),
        )
        for name in ("autodrive_reset", "reset_position", "reset_to_road"):
            setattr(resetter, name, lambda: False)
        self.assertEqual(resetter.recover("flipped", max_attempts=3), "FAILED")


if __name__ == "__main__":
    unittest.main(verbosity=2)

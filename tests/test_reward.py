import os
import sys
import unittest
from dataclasses import dataclass

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reward import DriveReward, DriveRewardConfig  # noqa: E402


@dataclass
class FakeTelemetry:
    speed: float
    mean_surface_rumble: float = 0.0
    mean_tire_slip_ratio: float = 0.0
    angular_velocity_y: float = 0.0


class TestDriveReward(unittest.TestCase):
    def test_idle_full_left_brake_is_worse_than_launching_straight(self):
        reward = DriveReward(DriveRewardConfig(centerline_path=""))
        stopped = FakeTelemetry(speed=0.0)
        bad = reward(stopped, None, np.array([-1.0, 0.0, 1.0], dtype=np.float32), None)
        launch = reward(stopped, None, np.array([0.0, 1.0, 0.0], dtype=np.float32), None)
        self.assertLess(bad, launch)
        self.assertLess(bad, 0.0)

    def test_forward_motion_is_positive_on_road(self):
        reward = DriveReward(DriveRewardConfig(centerline_path=""))
        moving = FakeTelemetry(speed=20.0)
        value = reward(moving, None, np.array([0.1, 0.8, 0.0], dtype=np.float32), None)
        self.assertGreater(value, 0.0)

    def test_offline_generators_use_speed_branch(self):
        # WS4: warm-start + recovery demos must resolve to the SAME (speed) branch -> one replay scale
        self.assertIsNone(DriveReward(DriveRewardConfig(centerline_path=""))._centerline)

    def test_non_finite_input_yields_zero(self):
        # W1: a NaN must never reach the learner's gradients
        reward = DriveReward(DriveRewardConfig(centerline_path=""))
        r = reward(FakeTelemetry(speed=float("nan")), None, np.array([0.0, 1.0, 0.0], np.float32), None)
        self.assertEqual(r, 0.0)

    def test_centering_rewards_being_near_the_line(self):
        # the night-safe steering signal: on the line must beat being far off it (same forward progress)
        @dataclass
        class PosTel:
            speed: float
            position_x: float = 0.0
            position_z: float = 0.0
            mean_surface_rumble: float = 0.0
            mean_tire_slip_ratio: float = 0.0
            angular_velocity_y: float = 0.0
        reward = DriveReward(DriveRewardConfig(centerline_path=""))
        # signed_cte = position_z (off the line); tangent along +x; same ds (forward) either way
        reward._centerline = type("FC", (), {
            "project": staticmethod(lambda x, z: (float(x), abs(float(z)), False)),
            "project_frame": staticmethod(lambda x, z: (float(x), float(z), 1.0, 0.0, False)),
            "length": 1000.0})()
        a = np.array([0.0, 1.0, 0.0], np.float32)
        centered = reward(PosTel(speed=10.0, position_x=5.0, position_z=0.0),
                          PosTel(speed=10.0, position_x=0.0, position_z=0.0), a, None)
        off = reward(PosTel(speed=10.0, position_x=5.0, position_z=15.0),
                     PosTel(speed=10.0, position_x=0.0, position_z=15.0), a, None)
        self.assertGreater(centered, off)

    def test_alignment_rewards_driving_along_route(self):
        # the directional signal: moving ALONG the tangent beats moving the WRONG way (same pos/speed)
        @dataclass
        class VelTel:
            speed: float
            velocity_x: float = 0.0
            velocity_z: float = 0.0
            position_x: float = 0.0
            position_z: float = 0.0
            mean_surface_rumble: float = 0.0
            mean_tire_slip_ratio: float = 0.0
            angular_velocity_y: float = 0.0
        reward = DriveReward(DriveRewardConfig(centerline_path=""))
        reward._centerline = type("FC", (), {
            "project": staticmethod(lambda x, z: (float(x), 0.0, False)),
            "project_frame": staticmethod(lambda x, z: (float(x), 0.0, 1.0, 0.0, False)),  # tangent +x
            "length": 1000.0})()
        a = np.array([0.0, 1.0, 0.0], np.float32)
        prev = VelTel(speed=10.0, position_x=0.5)
        along = reward(VelTel(speed=10.0, velocity_x=10.0, position_x=1.0), prev, a, None)
        wrong = reward(VelTel(speed=10.0, velocity_x=-10.0, position_x=1.0), prev, a, None)
        self.assertGreater(along, wrong)

    def test_reverse_earns_no_speed_bonus(self):
        # W3: with a centerline, going backward (ds<0) must NOT pay the forward-speed bonus
        @dataclass
        class PosTel:
            speed: float
            position_x: float = 0.0
            position_z: float = 0.0
            mean_surface_rumble: float = 0.0
            mean_tire_slip_ratio: float = 0.0
            angular_velocity_y: float = 0.0
        reward = DriveReward(DriveRewardConfig(centerline_path=""))
        reward._centerline = type("FC", (), {
            "project": staticmethod(lambda x, z: (float(x), 1.0, False)),
            "project_frame": staticmethod(lambda x, z: (float(x), 1.0, 1.0, 0.0, False)),
            "length": 1000.0})()
        a = np.array([0.0, 1.0, 0.0], np.float32)
        fwd = reward(PosTel(speed=10.0, position_x=5.0), PosTel(speed=10.0, position_x=0.0), a, None)
        rev = reward(PosTel(speed=10.0, position_x=0.0), PosTel(speed=10.0, position_x=5.0), a, None)
        self.assertGreater(fwd, rev)


if __name__ == "__main__":
    unittest.main(verbosity=2)

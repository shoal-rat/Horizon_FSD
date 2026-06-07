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


if __name__ == "__main__":
    unittest.main(verbosity=2)

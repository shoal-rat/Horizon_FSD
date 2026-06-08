import os
import sys
import tempfile
import unittest
from dataclasses import dataclass

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from recovery_demo import RecoveryDemoConfig, RecoveryDemoRecorder  # noqa: E402


@dataclass
class FakeTelemetry:
    position_x: float
    position_z: float
    speed: float = 5.0
    is_driving: bool = True
    mean_surface_rumble: float = 0.0
    mean_tire_slip_ratio: float = 0.0
    angular_velocity_y: float = 0.0
    steer_norm: float = 0.25
    throttle: float = 0.6
    brake: float = 0.0


class FakeCapture:
    def grab(self):
        return np.zeros((8, 8, 3), dtype=np.uint8)

    def preprocess(self, raw):
        return np.zeros((4, 4), dtype=np.uint8)


@dataclass
class FakeLine:
    cue: float = 0.0
    offset: float = 0.0
    confidence: float = 0.0


class FakeLineReader:
    def read(self, raw):
        return FakeLine()


class FakeReward:
    def __call__(self, *args, **kwargs):
        return 0.5


class TestRecoveryDemoRecorder(unittest.TestCase):
    def test_saves_non_teleport_recovery_episode(self):
        with tempfile.TemporaryDirectory() as tmp:
            recorder = RecoveryDemoRecorder(
                FakeCapture(),
                FakeLineReader(),
                FakeReward(),
                RecoveryDemoConfig(
                    enabled=True,
                    out_dir=tmp,
                    min_len=3,
                    sample_hz=1e9,
                    teleport_jump_m=30.0,
                ),
            )
            recorder.begin("autodrive", start_pos=(0.0, 0.0))
            for x in (0.5, 1.0, 1.5):
                recorder.sample(FakeTelemetry(position_x=x, position_z=0.0))

            path = recorder.end(success=True, teleported=False)
            self.assertIsNotNone(path)
            with np.load(path) as ep:
                self.assertEqual(ep["image"].shape, (3, 4, 4, 1))
                self.assertEqual(ep["action"].shape, (3, 3))
                self.assertTrue(ep["is_first"][0])
                self.assertAlmostEqual(float(ep["reward"][1]), 0.5)

    def test_discards_coordinate_jump_teleport(self):
        with tempfile.TemporaryDirectory() as tmp:
            recorder = RecoveryDemoRecorder(
                FakeCapture(),
                FakeLineReader(),
                FakeReward(),
                RecoveryDemoConfig(
                    enabled=True,
                    out_dir=tmp,
                    min_len=2,
                    sample_hz=1e9,
                    teleport_jump_m=30.0,
                ),
            )
            recorder.begin("autodrive", start_pos=(0.0, 0.0))
            recorder.sample(FakeTelemetry(position_x=1.0, position_z=0.0))
            recorder.sample(FakeTelemetry(position_x=100.0, position_z=0.0))

            self.assertIsNone(recorder.end(success=True, teleported=False))
            self.assertEqual(os.listdir(tmp), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)

"""
test_color_pipeline.py - Horizon FSD

End-to-end checks of the COLOR data path (no game needed): JPEG source-frame recording
format -> lazy loading -> warm-start conversion with color obs, racing-line backfill,
stride-2 timescale, and actuator-clamped demo actions.
"""
import os
import sys
import tempfile
import unittest

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset import LazyJpegFrames, load_session  # noqa: E402
from make_warmstart import _episode_from_run  # noqa: E402
from reward import DriveReward, DriveRewardConfig  # noqa: E402

STEER_CFG = dict(steer_limit=0.55, high_speed_steer_limit=0.35, high_speed_threshold=15.0)


def _frame_with_blue_chevron() -> np.ndarray:
    """Bright 320x180 color frame with a vivid blue chevron patch inside the line-reader ROI."""
    f = np.full((180, 320, 3), 150, np.uint8)          # bright day scene
    f[70:95, 130:190] = (230, 60, 40)                  # vivid blue (BGR) in ROI
    return f


def _session(n=40) -> dict:
    frames = [_frame_with_blue_chevron() for _ in range(n)]
    return {
        "frames": frames,
        "actions": np.tile(np.array([[0.9, 1.0, 0.0]], np.float32), (n, 1)),  # steer beyond the clamp
        "speed": np.full((n,), 10.0, np.float32),
        "surface_rumble": np.zeros((n,), np.float32),
        "tire_slip": np.zeros((n,), np.float32),
        "quality": "manual",
    }


class TestColorPipeline(unittest.TestCase):
    def test_jpeg_roundtrip_via_load_session(self):
        with tempfile.TemporaryDirectory() as d:
            enc = []
            for _ in range(5):
                ok, e = cv2.imencode(".jpg", _frame_with_blue_chevron(),
                                     [cv2.IMWRITE_JPEG_QUALITY, 90])
                self.assertTrue(ok)
                enc.append(e.tobytes())
            jpeg = np.empty(5, dtype=object)
            jpeg[:] = enc
            np.savez_compressed(
                os.path.join(d, "shard_0000.npz"), frames_jpeg=jpeg,
                actions=np.zeros((5, 3), np.float32), speed=np.zeros(5, np.float32),
                accel=np.zeros((5, 3), np.float32), surface_rumble=np.zeros(5, np.float32),
                tire_slip=np.zeros(5, np.float32), is_race_on=np.ones(5, np.int8),
                distance=np.zeros(5, np.float32), timestamp_ms=np.zeros(5, np.uint32),
                position=np.zeros((5, 3), np.float32), quality="manual")
            s = load_session(d)
            self.assertIsInstance(s["frames"], LazyJpegFrames)
            self.assertEqual(len(s["frames"]), 5)
            self.assertEqual(s["frames"][0].shape, (180, 320, 3))   # decodes to BGR color

    def test_color_episode_with_line_backfill_stride_and_clamp(self):
        s = _session(40)
        reward_fn = DriveReward(DriveRewardConfig(centerline_path=""))
        ep = _episode_from_run(s, np.arange(40), reward_fn, (64, 64), stride=2,
                               steer_cfg=STEER_CFG, channels=3)
        L = len(ep["reward"])
        self.assertEqual(L, 20)                                     # stride-2: 40 ticks -> 20 decisions
        self.assertEqual(ep["image"].shape, (L, 64, 64, 3))         # COLOR obs
        self.assertGreater(float(ep["line"][:, 2].mean()), 0.1)     # line BACKFILLED (confidence > 0)
        self.assertGreater(float(ep["line"][:, 0].mean()), 0.5)     # ... and reads the blue as accelerate
        self.assertLessEqual(float(np.abs(ep["action"][1:, 0]).max()), 0.56)  # steer clamped to actuator
        self.assertEqual(ep["route"].shape, (L, 27))

    def test_gray_target_still_works_for_color_sources(self):
        s = _session(20)
        reward_fn = DriveReward(DriveRewardConfig(centerline_path=""))
        ep = _episode_from_run(s, np.arange(20), reward_fn, (64, 64), stride=2,
                               steer_cfg=STEER_CFG, channels=1)
        self.assertEqual(ep["image"].shape[-1], 1)                  # color source -> gray obs derivable


if __name__ == "__main__":
    unittest.main(verbosity=2)

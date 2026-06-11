"""
test_racing_line.py - Horizon FSD

Day/night-adaptive line reading. The chevrons stay visible at night, but the cold night scene
(moonlit snow/sky) reads as false BLUE - so at night the reader drops blue and trusts only the
snow-immune warm cues (amber=ease, red=brake). Worst-case night error is over-cautious braking,
never a false 'accelerate into a corner'. Day behaviour (all colours) is unchanged.
"""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from racing_line import RacingLineReader  # noqa: E402


def _frame(base, patches):
    f = np.full((720, 1280, 3), base, np.uint8)
    for (y0, y1, x0, x1, bgr) in patches:
        f[y0:y1, x0:x1] = bgr
    return f


BLUE = (230, 60, 40)     # vivid cyan/blue (accelerate)
AMBER = (20, 170, 220)   # vivid amber/yellow (ease)


class TestRacingLineDayNight(unittest.TestCase):
    def test_day_reads_blue_as_accelerate(self):
        f = _frame(150, [(260, 320, 500, 760, BLUE)])   # bright scene + blue chevrons
        rd = RacingLineReader().read(f)
        self.assertGreater(rd.cue, 0.5)                  # blue -> accelerate
        self.assertGreater(rd.confidence, 0.0)

    def test_night_ignores_false_blue(self):
        # dark scene with a big BLUE region (moonlit snow) + a small amber chevron: night mode must NOT
        # report 'accelerate' off the blue - it drops blue and reads the warm cue.
        f = _frame(28, [(255, 300, 400, 820, BLUE), (300, 322, 600, 670, AMBER)])
        rd = RacingLineReader().read(f)
        self.assertLessEqual(rd.cue, 0.2)                # never a false accelerate at night
        self.assertEqual(rd.counts[0], 0)               # blue dropped

    def test_pitch_black_is_off(self):
        f = _frame(5, [(260, 320, 500, 760, BLUE)])      # below min_night_brightness -> off
        rd = RacingLineReader().read(f)
        self.assertEqual(rd.confidence, 0.0)

    def test_degenerate_roi_no_crash(self):
        rd = RacingLineReader().read(np.full((4, 4, 3), 120, np.uint8))
        self.assertEqual(rd.confidence, 0.0)

    def test_auto_lighting_classification(self):
        # the AUTO day/night identifier: per-frame, so mixed sessions need no manual labels
        r = RacingLineReader()
        self.assertEqual(r.classify_lighting(np.full((180, 320, 3), 150, np.uint8)), "day")
        self.assertEqual(r.classify_lighting(np.full((180, 320, 3), 70, np.uint8)), "dusk")
        self.assertEqual(r.classify_lighting(np.full((180, 320, 3), 35, np.uint8)), "night")
        self.assertAlmostEqual(r.scene_brightness(np.full((180, 320, 3), 150, np.uint8)), 150.0, delta=1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)

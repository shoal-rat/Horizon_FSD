"""
racing_line.py - Horizon FSD

Read Forza Horizon's on-road driving line from a FULL-RESOLUTION COLOUR frame and
turn it into a compact, low-dimensional signal the RL agent can actually use:

  * line_cue    in [-1, +1]  speed instruction from the line colour:
                              +1 = blue  (accelerate / good speed)
                               0 = yellow (ease off / coast)
                              -1 = red    (brake - corner ahead)
  * line_offset in [-1, +1]  horizontal position of the line, left(-)..right(+)
                              of screen centre = a steering hint toward the line
  * confidence  in [0, 1]    how much of the road-ahead ROI is line-coloured
                              (~0 => line off / not in view => treat cue as 0)

The line is drawn as VIVID chevrons on MUTED grey asphalt, so we segment by HSV
saturation (the line is the only highly-saturated thing on the road) and classify
the surviving pixels by hue. The nearer (lower) chevrons weigh more, since they
are the more immediate instruction.

IMPORTANT: feed the full-res colour frame (capture.grab()), NOT the 64x64 gray obs
- at 64x64 the chevrons are a few pixels and the colour is gone. Colours are
camera-independent but the ROI is NOT: calibrate roi_* for the camera you train
with, using racing_line_preview.py on real frames.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple

import cv2
import numpy as np


@dataclass
class RacingLineConfig:
    # ROI = the road just ahead of the car, as fractions of (W, H). Defaults are a
    # starting point from chase-cam 1280x720 frames; RE-CALIBRATE for your camera.
    roi_x: Tuple[float, float] = (0.20, 0.80)   # exclude minimap/speedo at the edges
    roi_y: Tuple[float, float] = (0.32, 0.54)   # below sky/scenery, above the car body
    sat_min: int = 110          # line is vivid; asphalt/snow/sky/most scenery are washed out
    val_min: int = 100          # and bright (not in shadow)
    # OpenCV HSV hue bands (H in 0..180)
    blue_hue: Tuple[int, int] = (90, 135)
    yellow_hue: Tuple[int, int] = (16, 42)
    red_hue_lo: Tuple[int, int] = (0, 12)
    red_hue_hi: Tuple[int, int] = (168, 180)
    min_pixels: int = 40        # fewer line pixels than this => line not present
    near_weight: float = 2.0    # nearer (lower) rows count this much more for the cue
    conf_full_frac: float = 0.02  # ROI fraction of line pixels that maps to confidence 1.0
    # Day/night handling. The chevrons stay visible at night (headlight-lit), but the COLD night scene
    # (moonlit snow/sky) reads as false BLUE, so a naive read flips an amber 'ease' line to 'accelerate'.
    # Below day_brightness we switch to NIGHT mode: drop blue, trust only the snow-immune WARM cues
    # (amber=ease, red=brake) at reduced confidence (worst-case error = over-cautious braking, never a
    # false 'accelerate into a corner'). Steering at night leans on the telemetry centering reward.
    min_scene_brightness: int = 85  # ROI mean grey >= this = DAY (trust all colours); below = NIGHT mode
    night_conf_scale: float = 0.5   # scale night-mode confidence down (the warm cue is real but noisier)
    min_night_brightness: int = 20  # truly too dark even for warm cues -> report confidence 0 (off)


@dataclass
class LineReading:
    cue: float          # [-1,+1]  -1 brake .. 0 ease .. +1 accelerate
    offset: float       # [-1,+1]  line left(-)..right(+) of centre
    confidence: float   # [0,1]
    counts: Tuple[int, int, int] = field(default=(0, 0, 0))  # (blue, yellow, red) pixels


class RacingLineReader:
    """Stateless: call .read(frame_bgr) per step. Cheap (one HSV convert on a crop)."""

    def __init__(self, cfg: RacingLineConfig | None = None) -> None:
        self.cfg = cfg or RacingLineConfig()

    def scene_brightness(self, frame_bgr: np.ndarray) -> float:
        """Mean grey of the line ROI - the per-frame lighting scalar the day/night decision uses."""
        c = self.cfg
        if frame_bgr.ndim == 3 and frame_bgr.shape[2] == 4:
            frame_bgr = frame_bgr[:, :, :3]
        H, W = frame_bgr.shape[:2]
        roi = frame_bgr[int(c.roi_y[0] * H):int(c.roi_y[1] * H),
                        int(c.roi_x[0] * W):int(c.roi_x[1] * W)]
        if roi.size == 0:
            return 0.0
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.ndim == 3 else roi
        return float(gray.mean())

    def classify_lighting(self, frame_bgr: np.ndarray) -> str:
        """'day' / 'dusk' / 'night' for THIS frame - the auto day/night identification. Per-frame,
        so a single recording session can sweep FH6's whole day/night cycle and just be thrown into
        one corpus: nothing downstream needs day and night separated."""
        b = self.scene_brightness(frame_bgr)
        if b >= self.cfg.min_scene_brightness:
            return "day"
        return "dusk" if b >= (self.cfg.min_scene_brightness + self.cfg.min_night_brightness) / 2 else "night"

    def _masks(self, frame_bgr: np.ndarray):
        c = self.cfg
        if frame_bgr.ndim == 3 and frame_bgr.shape[2] == 4:   # BGRA (live grab) -> BGR
            frame_bgr = frame_bgr[:, :, :3]
        H, W = frame_bgr.shape[:2]
        x0, x1 = int(c.roi_x[0] * W), int(c.roi_x[1] * W)
        y0, y1 = int(c.roi_y[0] * H), int(c.roi_y[1] * H)
        roi = frame_bgr[y0:y1, x0:x1]
        if roi.shape[0] == 0 or roi.shape[1] == 0:      # degenerate ROI (tiny/transient frame): bail
            empty = np.zeros((0, 0), dtype=bool)        # read()'s roi.size==0 guard then returns neutral
            return (x0, y0, 0, 0), roi, empty, empty, empty
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        vivid = (s >= c.sat_min) & (v >= c.val_min)
        blue = vivid & (h >= c.blue_hue[0]) & (h <= c.blue_hue[1])
        yellow = vivid & (h >= c.yellow_hue[0]) & (h <= c.yellow_hue[1])
        red = vivid & (((h >= c.red_hue_lo[0]) & (h <= c.red_hue_lo[1])) |
                       ((h >= c.red_hue_hi[0]) & (h <= c.red_hue_hi[1])))
        return (x0, y0, roi.shape[1], roi.shape[0]), roi, blue, yellow, red

    def read(self, frame_bgr: np.ndarray) -> LineReading:
        c = self.cfg
        H, W = frame_bgr.shape[:2]
        (x0, y0, rw, rh), roi, blue, yellow, red = self._masks(frame_bgr)
        if roi.size == 0:
            return LineReading(0.0, 0.0, 0.0, (0, 0, 0))

        scene = float(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY).mean())
        if scene < c.min_night_brightness:          # truly too dark even for warm cues -> off (fail safe)
            return LineReading(0.0, 0.0, 0.0, (0, 0, 0))
        conf_scale = 1.0
        if scene < c.min_scene_brightness:          # NIGHT: the cold scene reads as false blue (moonlit
            blue = np.zeros_like(blue)              # snow/sky), so trust only the snow-immune WARM cues
            conf_scale = c.night_conf_scale         # (amber=ease, red=brake) at reduced confidence.

        nb, ny, nr = int(blue.sum()), int(yellow.sum()), int(red.sum())
        npix = nb + ny + nr
        if npix < c.min_pixels:
            return LineReading(0.0, 0.0, 0.0, (nb, ny, nr))

        # nearer (lower) rows weigh more: the immediate instruction is the closest chevron
        roww = (1.0 + c.near_weight * (np.arange(rh) / max(rh - 1, 1)))[:, None]
        wb = float((blue * roww).sum())
        wy = float((yellow * roww).sum())
        wr = float((red * roww).sum())
        tot = wb + wy + wr
        cue = (wb - wr) / tot if tot > 0 else 0.0      # +1 blue .. 0 yellow .. -1 red

        # offset = PIXEL-MASS centroid (near-row weighted, same as the cue), not mere column-presence:
        # otherwise one stray vivid HUD/glint column counts as much as the whole chevron stack.
        mask = blue | yellow | red
        colmass = (mask * roww).sum(axis=0)
        msum = float(colmass.sum())
        cx = (float((np.arange(rw) * colmass).sum() / msum) + x0) if msum > 0 else (W / 2.0)
        offset = (cx - W / 2.0) / (W / 2.0)

        conf = conf_scale * npix / (c.conf_full_frac * rw * rh)
        return LineReading(float(np.clip(cue, -1, 1)),
                           float(np.clip(offset, -1, 1)),
                           float(np.clip(conf, 0, 1)),
                           (nb, ny, nr))

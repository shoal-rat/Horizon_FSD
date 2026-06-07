"""
racing_line_preview.py - Horizon FSD

Validate / calibrate RacingLineReader on real Forza frames BEFORE wiring it into
training. For each image it:
  * prints  cue / offset / confidence  and the blue/yellow/red pixel counts
  * saves  <name>_overlay.png  with the ROI box drawn and the detected line
    pixels tinted (blue/yellow/red) so you can SEE what it locked onto.

Drop a few colour frames (the screenshots, or fresh captures) into a folder and:
    .\\.venv\\Scripts\\python.exe racing_line_preview.py C:\\Horizon_FSD\\line_frames
    .\\.venv\\Scripts\\python.exe racing_line_preview.py some\\dir\\*.png

Send the overlays back and we tune sat_min / hue bands / ROI in racing_line.py.
"""
from __future__ import annotations

import glob
import os
import sys

import cv2
import numpy as np

from racing_line import RacingLineConfig, RacingLineReader


def _gather(arg: str) -> list[str]:
    if os.path.isdir(arg):
        out = []
        for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp"):
            out += glob.glob(os.path.join(arg, ext))
        return sorted(p for p in out if "_overlay" not in p)
    return sorted(p for p in glob.glob(arg) if "_overlay" not in p)


def main() -> int:
    arg = sys.argv[1] if len(sys.argv) > 1 else r"C:\Horizon_FSD\line_frames"
    paths = _gather(arg)
    if not paths:
        print(f"no images found at: {arg}")
        print(r"  put the frames in C:\Horizon_FSD\line_frames\ (png/jpg) and re-run")
        return 1

    cfg = RacingLineConfig()
    reader = RacingLineReader(cfg)
    print(f"ROI x={cfg.roi_x} y={cfg.roi_y}  sat_min={cfg.sat_min} val_min={cfg.val_min}")
    print("-" * 78)
    for p in paths:
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        if img is None:
            print(f"  (could not read {p})")
            continue
        H, W = img.shape[:2]
        r = reader.read(img)
        verdict = ("BRAKE" if r.cue < -0.25 else "accelerate" if r.cue > 0.25 else "ease/coast")
        print(f"{os.path.basename(p):42s} cue={r.cue:+.2f} ({verdict:10s}) "
              f"offset={r.offset:+.2f} conf={r.confidence:.2f}  "
              f"px b/y/r={r.counts[0]}/{r.counts[1]}/{r.counts[2]}")

        # overlay: ROI box + tint detected pixels
        (x0, y0, rw, rh), _roi, blue, yellow, red = reader._masks(img)
        ov = img.copy()
        cv2.rectangle(ov, (x0, y0), (x0 + rw, y0 + rh), (255, 255, 255), 2)
        crop = ov[y0:y0 + rh, x0:x0 + rw]
        crop[blue] = (255, 0, 0)       # BGR: blue
        crop[yellow] = (0, 255, 255)   # yellow
        crop[red] = (0, 0, 255)        # red
        cv2.putText(ov, f"{verdict} cue={r.cue:+.2f} off={r.offset:+.2f} conf={r.confidence:.2f}",
                    (10, H - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        out = os.path.splitext(p)[0] + "_overlay.png"
        cv2.imwrite(out, ov)
    print("-" * 78)
    print("saved *_overlay.png next to each input. Eyeball: are the chevrons tinted")
    print("(blue/yellow/red) and is the verdict right? If scenery is tinted, raise sat_min.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

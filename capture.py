"""
capture.py - Horizon FSD, Phase 1

Screen capture for the vision observation, via `windows-capture` (the modern
Windows.Graphics.Capture API). Runs the capture loop on a background thread and
keeps the latest frame; `grab()` returns the most recent one.

Why not dxcam/bettercam? On this project's target machine (an NVIDIA Optimus
laptop, often with HDR on) DXGI Desktop Duplication returns all-black frames and
its comtypes teardown crashes on Python 3.13. Windows.Graphics.Capture handles
HDR + hybrid GPUs and has no comtypes dependency.

`windows_capture` and `cv2` are imported lazily so this module (and the env)
import fine without them.

Setup:
    <python> -m pip install windows-capture opencv-python
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)

_CAPTURE_HINT = (
    "windows-capture + opencv are required for screen capture. Install them with:\n"
    "    <python> -m pip install windows-capture opencv-python"
)


class ScreenCapture:
    """Background screen grabber (Windows.Graphics.Capture).

    Args:
        monitor_index: 1-based monitor (1 = primary). Ignored when window_name is set.
        window_name:   capture a specific window by title substring, or None for a monitor.
        region:        (left, top, right, bottom) crop in captured pixels, or None.
        img_size:      (height, width) to downscale the observation to.
        grayscale:     convert to a single channel.
        cursor_capture: include the mouse cursor in the capture.
    """

    def __init__(
        self,
        monitor_index: int = 1,
        window_name: Optional[str] = None,
        region: Optional[Sequence[int]] = None,
        img_size: Sequence[int] = (84, 84),
        grayscale: bool = True,
        cursor_capture: bool = False,
    ) -> None:
        try:
            from windows_capture import WindowsCapture
            import cv2
        except ImportError as exc:  # pragma: no cover - depends on the host
            raise RuntimeError(_CAPTURE_HINT) from exc

        self._cv2 = cv2
        self.region = tuple(int(v) for v in region) if region else None
        self.img_size = (int(img_size[0]), int(img_size[1]))  # (H, W)
        self.grayscale = grayscale

        self._latest: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self._frames = 0
        self._control = None

        self._cap = WindowsCapture(
            cursor_capture=cursor_capture,
            draw_border=False,
            monitor_index=monitor_index,
            window_name=window_name,
        )

        @self._cap.event
        def on_frame_arrived(frame, capture_control):  # runs on the capture thread
            buf = frame.frame_buffer  # (H, W, 4) BGRA, valid only inside the callback
            bgr = np.ascontiguousarray(buf[:, :, :3])  # copy out as BGR
            with self._lock:
                self._latest = bgr
                self._frames += 1

        @self._cap.event
        def on_closed():
            logger.info("windows-capture session closed.")

        self._control = self._cap.start_free_threaded()
        self._wait_first_frame()

    def _wait_first_frame(self, timeout: float = 5.0) -> None:
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < timeout:
            with self._lock:
                if self._latest is not None:
                    return
            time.sleep(0.01)
        self.close()
        raise RuntimeError(
            f"windows-capture produced no frame within {timeout}s "
            "(check monitor_index / window_name)."
        )

    # ---- capture ----------------------------------------------------------
    def grab(self) -> np.ndarray:
        """The most recent full-res BGR frame. The capture thread rebinds a fresh
        array each frame, so the returned reference is safe to read without copying."""
        with self._lock:
            frame = self._latest
        if frame is None:
            raise RuntimeError("no frame captured yet")
        return frame

    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Crop (optional) -> downscale -> optional grayscale, to img_size."""
        if self.region:
            left, top, right, bottom = self.region
            frame = frame[top:bottom, left:right]
        h, w = self.img_size
        img = frame
        if self.grayscale:
            img = self._cv2.cvtColor(img, self._cv2.COLOR_BGR2GRAY)
        img = self._cv2.resize(img, (w, h), interpolation=self._cv2.INTER_AREA)
        return img

    def observation(self) -> np.ndarray:
        """Latest frame, preprocessed -> (H, W) gray or (H, W, 3)."""
        return self.preprocess(self.grab())

    @property
    def frame_count(self) -> int:
        return self._frames

    # ---- lifecycle --------------------------------------------------------
    def close(self) -> None:
        try:
            if self._control is not None:
                self._control.stop()
        except Exception:  # pragma: no cover
            pass
        self._control = None

    def __enter__(self) -> "ScreenCapture":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

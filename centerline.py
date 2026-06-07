"""
centerline.py - Horizon FSD

A reference path (centerline) for the RL progress reward. Projecting the car's world
position onto this polyline yields a scalar ARC-LENGTH; rewarding the per-step INCREASE in
arc-length is progress ALONG THE ROUTE. Unlike rewarding raw speed, this cannot be farmed by
driving in circles (a closed loop nets ~0, reversing is negative) and it works in the dark
(telemetry position, not vision). This is the reward shape used by tmrl / GT Sophy / Linesight.
"""
from __future__ import annotations

import numpy as np


class Centerline:
    def __init__(self, points) -> None:
        pts = np.asarray(points, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) < 2:
            raise ValueError("centerline needs (N>=2, 2) ground-plane (x, z) points")
        self.pts = pts
        self._a = pts[:-1]                              # segment starts
        self._ab = pts[1:] - pts[:-1]                   # segment vectors
        self._ab2 = (self._ab ** 2).sum(1).clip(1e-9)   # |ab|^2
        self._seglen = np.sqrt((self._ab ** 2).sum(1))
        self.cum = np.concatenate([[0.0], np.cumsum(self._seglen)])  # arc-length at each vertex
        self.length = float(self.cum[-1])

    def project(self, x: float, z: float) -> tuple[float, float]:
        """Nearest point on the polyline -> (arc_length_s, lateral_distance)."""
        px, pz = float(x), float(z)
        t = (((np.array([px, pz]) - self._a) * self._ab).sum(1) / self._ab2).clip(0.0, 1.0)
        proj = self._a + t[:, None] * self._ab
        d = np.hypot(proj[:, 0] - px, proj[:, 1] - pz)
        i = int(d.argmin())
        s = self.cum[i] + t[i] * self._seglen[i]
        return float(s), float(d[i])

    @classmethod
    def load(cls, path: str) -> "Centerline":
        return cls(np.load(path))

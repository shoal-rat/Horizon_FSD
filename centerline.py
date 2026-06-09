"""
centerline.py - Horizon FSD

A reference path (centerline) for the RL progress reward. Projecting the car's world
position onto this polyline yields a scalar ARC-LENGTH; rewarding the per-step INCREASE in
arc-length is progress ALONG THE ROUTE. Unlike rewarding raw speed, this cannot be farmed by
driving in circles (a closed loop nets ~0, reversing is negative) and it works in the dark
(telemetry position, not vision). This is the reward shape used by tmrl / GT Sophy / Linesight.
"""
from __future__ import annotations

import math

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

    def project(self, x: float, z: float) -> tuple[float, float, bool]:
        """Nearest point on the polyline -> (arc_length_s, lateral_distance, at_end).

        Non-finite input fails SAFE: returns (nan, inf, False) so `inf > offroute_dist` reads as
        off-route (a debounced reset) rather than silently being missed. Past the OPEN end of the
        route, lateral_distance is the PERPENDICULAR distance to the last segment's line (not the
        endpoint distance, which would read along-track overshoot as huge lateral error) and
        at_end=True - so driving straight past the finish isn't mistaken for going off-route."""
        px, pz = float(x), float(z)
        if not (math.isfinite(px) and math.isfinite(pz)):
            return float("nan"), float("inf"), False
        rel = np.array([px, pz]) - self._a
        t_raw = (rel * self._ab).sum(1) / self._ab2
        t = t_raw.clip(0.0, 1.0)
        proj = self._a + t[:, None] * self._ab
        d = np.hypot(proj[:, 0] - px, proj[:, 1] - pz)
        i = int(d.argmin())
        s = self.cum[i] + t[i] * self._seglen[i]
        if i == len(self._ab) - 1 and t_raw[i] > 1.0:          # projected past the final vertex
            ab = self._ab[i]
            lat = abs(rel[i, 0] * ab[1] - rel[i, 1] * ab[0]) / max(self._seglen[i], 1e-9)
            return float(s), float(lat), True
        return float(s), float(d[i]), False

    @classmethod
    def load(cls, path: str) -> "Centerline":
        return cls(np.load(path))

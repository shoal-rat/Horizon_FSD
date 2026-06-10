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

    def project_frame(self, x: float, z: float) -> tuple[float, float, float, float, bool]:
        """Like project() but also returns the route's UNIT TANGENT and the SIGNED cross-track error
        (right of the path positive, left negative). Returns (s, signed_cte, tan_x, tan_z, at_end);
        |signed_cte| == project()'s lateral_distance. The tangent is in the SAME world (x, z) frame as
        the car's velocity, so dot(velocity, tangent) is a sign-safe heading-alignment signal (no yaw
        convention needed). Non-finite input fails safe: (nan, inf, 1, 0, False)."""
        px, pz = float(x), float(z)
        if not (math.isfinite(px) and math.isfinite(pz)):
            return float("nan"), float("inf"), 1.0, 0.0, False
        rel = np.array([px, pz]) - self._a
        t_raw = (rel * self._ab).sum(1) / self._ab2
        t = t_raw.clip(0.0, 1.0)
        proj = self._a + t[:, None] * self._ab
        d = np.hypot(proj[:, 0] - px, proj[:, 1] - pz)
        i = int(d.argmin())
        s = self.cum[i] + t[i] * self._seglen[i]
        seglen = max(self._seglen[i], 1e-9)
        tx, tz = self._ab[i, 0] / seglen, self._ab[i, 1] / seglen          # unit tangent
        at_end = i == len(self._ab) - 1 and t_raw[i] > 1.0
        rx, rz = (rel[i, 0], rel[i, 1]) if at_end else (px - proj[i, 0], pz - proj[i, 1])
        signed_cte = tx * rz - tz * rx                                     # cross(tangent, offset): + = right
        return float(s), float(signed_cte), float(tx), float(tz), at_end

    def lookahead(self, s: float, k: int, spacing: float) -> np.ndarray:
        """k points ON the route at arc lengths s+spacing .. s+k*spacing (clamped to the end),
        world (x, z) coords - the upcoming road geometry, light-invariant."""
        targets = np.clip(s + spacing * np.arange(1, k + 1), 0.0, self.length)
        return np.stack([np.interp(targets, self.cum, self.pts[:, 0]),
                         np.interp(targets, self.cum, self.pts[:, 1])], axis=1)

    @classmethod
    def load(cls, path: str) -> "Centerline":
        return cls(np.load(path))


# ---------------------------------------------------------------------------
# Route-geometry observation vector (shared by the live env and the demo converters,
# so both sides produce IDENTICAL features - the GT-Sophy/Linesight-style path preview).
# ---------------------------------------------------------------------------
ROUTE_LOOKAHEAD_K = 10
ROUTE_DIM = 7 + 2 * ROUTE_LOOKAHEAD_K   # [cte, sin_he, cos_he, valid, prev_action(3), k*(fwd, lat)]


def route_features(cl: "Centerline | None", x: float, z: float, vx: float, vz: float,
                   speed: float, prev_action_model, max_dist: float = 18.0,
                   spacing: float = 8.0, min_speed: float = 2.0) -> np.ndarray:
    """Light-invariant route-geometry obs (~ROUTE_DIM floats, all in [-1, 1]):
      [0]   signed cross-track / max_dist        (which side of the route line, how far)
      [1:3] sin/cos of heading error (velocity vs route tangent), speed-GATED - velocity
            direction is undefined near rest, so below min_speed they are 0 with valid=0
      [3]   validity flag (1 = heading valid)
      [4:7] previous APPLIED action (model coords) - what the car actually just did
      [7:]  K lookahead points on the route in the TANGENT frame at the projection
            (fwd, lat)/span - the upcoming curvature; defined even at rest (no car
            heading needed), so the agent can SEE the road bend before it arrives,
            day or night. An all-zero vector = "route unknown" (legacy demos).
    """
    out = np.zeros((ROUTE_DIM,), np.float32)
    pa = np.asarray(prev_action_model, np.float32).reshape(-1)[:3]
    out[4:4 + len(pa)] = np.clip(pa, -1.0, 1.0)
    if cl is None or not (math.isfinite(float(x)) and math.isfinite(float(z))):
        return out
    s, cte, tx, tz, _ = cl.project_frame(float(x), float(z))
    if not math.isfinite(cte):
        return out
    out[0] = float(np.clip(cte / max(1e-6, max_dist), -1.0, 1.0))
    vmag = math.hypot(float(vx), float(vz))
    if speed >= min_speed and math.isfinite(vmag) and vmag > 1e-6:
        dx, dz = float(vx) / vmag, float(vz) / vmag
        out[1] = float(np.clip(tx * dz - tz * dx, -1.0, 1.0))   # sin(heading err): + = drifting right
        out[2] = float(np.clip(dx * tx + dz * tz, -1.0, 1.0))   # cos(heading err): 1 = along the route
        out[3] = 1.0
    pts = cl.lookahead(s, ROUTE_LOOKAHEAD_K, spacing)
    rel = pts - np.array([float(x), float(z)])
    span = max(1e-6, ROUTE_LOOKAHEAD_K * spacing)
    out[7::2] = np.clip((rel[:, 0] * tx + rel[:, 1] * tz) / span, -1.0, 1.0)   # forward along tangent
    out[8::2] = np.clip((tx * rel[:, 1] - tz * rel[:, 0]) / span, -1.0, 1.0)   # lateral off tangent
    return out

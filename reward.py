"""
reward.py - Horizon FSD, Phase 1

Modular, swappable reward functions over `ForzaTelemetry`. Pick one by name from
config; the env calls it each step. Phase 4 (RL) will add route-progress rewards;
this file starts with speed-based shaping so the structure is in place.

A reward function has signature:
    fn(current: ForzaTelemetry, prev: ForzaTelemetry | None) -> float
"""
from __future__ import annotations

import logging
import math
import os
from typing import Callable, Optional

from forza_telemetry import ForzaTelemetry

logger = logging.getLogger(__name__)

RewardFn = Callable[[ForzaTelemetry, Optional[ForzaTelemetry]], float]

_REGISTRY: dict[str, RewardFn] = {}


def register(name: str) -> Callable[[RewardFn], RewardFn]:
    def deco(fn: RewardFn) -> RewardFn:
        _REGISTRY[name] = fn
        return fn
    return deco


def get(name: str) -> RewardFn:
    if name not in _REGISTRY:
        raise KeyError(f"unknown reward '{name}'; available: {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def available() -> list[str]:
    return sorted(_REGISTRY)


@register("forward_speed")
def forward_speed(current: ForzaTelemetry, prev: Optional[ForzaTelemetry] = None) -> float:
    """Reward forward motion (m/s). Simple, good for the first BC/RL sanity run."""
    return max(0.0, current.forward_speed)


@register("speed_minus_offroad")
def speed_minus_offroad(current: ForzaTelemetry, prev: Optional[ForzaTelemetry] = None) -> float:
    """Forward speed, penalized for off-road/rough surface and wheelspin/sliding."""
    r = max(0.0, current.forward_speed)
    r -= 2.0 * current.mean_surface_rumble      # rough-surface / off-road proxy
    r -= 1.0 * current.mean_tire_slip_ratio      # wheelspin / loss-of-grip proxy
    return r


@register("distance_progress")
def distance_progress(current: ForzaTelemetry, prev: Optional[ForzaTelemetry] = None) -> float:
    """Meters of DistanceTraveled gained since the previous step (route-agnostic
    progress). Falls back to forward speed when there's no previous frame."""
    if prev is None:
        return max(0.0, current.forward_speed) * 0.0
    delta = current.distance_traveled - prev.distance_traveled
    # Guard against resets / wraparound producing huge spurious jumps.
    if delta < 0.0 or delta > 1000.0:
        return 0.0
    return delta


# ---------------------------------------------------------------------------
# Phase 5: the RL driving reward (used by forza_rl_env AND the warm-start
# converter, so it uses only fields present in BOTH live telemetry and the
# logged shards: speed scalar, surface_rumble, tire_slip).
# ---------------------------------------------------------------------------
from dataclasses import dataclass


@dataclass
class DriveRewardConfig:
    speed_cap: float = 40.0       # m/s (~144 km/h): forward progress saturates here (fallback path)
    progress_scale: float = 1.0
    # --- centerline progress (preferred): reward arc-length ALONG a reference path, not raw
    #     speed -> circling nets ~0 (no anti-spin needed) and it works at night. Falls back to
    #     speed-based progress when the file is missing or position is unavailable (warm-start). ---
    centerline_path: str = r"C:\Horizon_FSD\centerline.npy"
    progress_max_step: float = 2.0    # m of arc-length per ~tick at speed_cap (40*0.05) -> normalises to ~[0,1]
    progress_jump_guard: float = 5.0  # ignore per-call arc-length deltas bigger than this (teleport/seam)
    route_max_dist: float = 30.0      # m: beyond this lateral distance from the path = off-route, no progress
    boot_w: float = 0.3           # BOOTSTRAP: small dense forward-speed bonus (on-road) so a fresh
    #                               actor gets a gradient toward driving forward BEFORE it can make
    #                               clean centerline progress (which is otherwise too sparse to learn
    #                               from cold). Kept below the spin penalty so circling stays negative.
    offroad_rumble: float = 0.15  # on-road gate + off-road penalty threshold
    offroad_w: float = 2.0
    slip_deadband: float = 1.0    # mean tire slip below this is normal cornering
    slip_w: float = 0.5
    jerk_w: float = 0.02          # GENTLE smoothness nudge. Was 1.0: dSteer^2 (up to 4.0)
    #                               dwarfed the +1.0 progress cap, so a still car (r=0) beat a
    #                               moving one (r<0) -> agent learned to die fast. Keep tiny so
    #                               forward progress always dominates and driving is net-positive.
    spin_w: float = 0.5           # ANTI-CIRCLING: a speed-only progress reward is hacked by
    spin_deadband: float = 0.6    # driving tight circles (keeps speed, dodges the forward crash).
    #                               Penalize sustained yaw-rate (rad/s) beyond a cornering deadband
    #                               so circling stops paying. Straight driving (yaw~0) is untouched.
    crash_penalty: float = 5.0    # one-off, applied by the env on a crash/terminate
    line_follow_w: float = 0.3    # reward agreeing with the driving line (accelerate on
    #                               blue, brake on red), scaled by detection confidence;
    #                               applied by the env (it has the line + the action)
    idle_speed: float = 1.0       # m/s: below this, "parked" should not be a free local optimum
    idle_w: float = 0.05
    steer_w: float = 0.01         # tiny absolute-steer cost; progress still dominates real corners
    low_speed_steer_w: float = 0.08
    low_speed_steer_speed: float = 5.0
    brake_w: float = 0.03         # discourages holding brake at/near rest
    launch_w: float = 0.04        # small on-road nudge to press throttle from rest


class DriveReward:
    """r = on-road forward-progress (0..1) - off-road - slip - steering-jerk.

    Callable as fn(current, prev=None, action=None, prev_action=None) -> float.
    action/prev_action are expected in applied gamepad coordinates:
    [steer -1..1, throttle 0..1, brake 0..1].
    """

    def __init__(self, cfg: "DriveRewardConfig | None" = None) -> None:
        self.cfg = cfg or DriveRewardConfig()
        self._centerline = None
        # Only fall back to the (circle-hackable) speed reward when NO centerline was requested.
        # If a path is set AND the file exists but fails to load, FAIL LOUD - silently downgrading
        # to the speed reward wastes a whole GPU run before anyone notices.
        if self.cfg.centerline_path and os.path.exists(self.cfg.centerline_path):
            try:
                from centerline import Centerline
                self._centerline = Centerline.load(self.cfg.centerline_path)
            except Exception as e:
                logger.exception("failed to load centerline %s", self.cfg.centerline_path)
                raise RuntimeError(
                    f"centerline {self.cfg.centerline_path} exists but won't load ({e}); refusing to "
                    "fall back to the circle-hackable speed reward. Rebuild it with build_centerline.py "
                    'or set centerline_path="" to use the speed reward deliberately.') from e

    def _progress(self, t, prev_t, on_road: bool) -> tuple[float, Optional[float]]:
        """Return (progress, ds) where ds = signed arc-length advance along the centerline (or None
        when no centerline/position, e.g. warm-start demos). Path progress is circle-proof; the
        fallback is capped speed. ds lets the caller gate the forward-speed bonus on REAL forward
        motion so reversing can't farm it."""
        c = self.cfg
        if not on_road:
            return 0.0, None
        if (self._centerline is not None and prev_t is not None
                and hasattr(t, "position_x") and hasattr(prev_t, "position_x")):
            if not (math.isfinite(t.position_x) and math.isfinite(t.position_z)):
                return 0.0, 0.0
            s_now, lat = self._centerline.project(t.position_x, t.position_z)[:2]
            s_prev, _ = self._centerline.project(prev_t.position_x, prev_t.position_z)[:2]
            ds = s_now - s_prev
            if lat <= c.route_max_dist and abs(ds) <= c.progress_jump_guard:
                return (max(0.0, ds) / c.progress_max_step) * c.progress_scale, ds
            return 0.0, 0.0                              # off-route or teleport/seam -> no credit, no forward
        fwd = max(0.0, t.speed)                          # fallback: capped speed (warm-start demos drive fwd)
        return (min(fwd, c.speed_cap) / c.speed_cap) * c.progress_scale, None

    def __call__(self, t, prev_t=None, action=None, prev_action=None) -> float:
        c = self.cfg
        # non-finite physics -> 0.0 (defense-in-depth behind the telemetry finite-check); never let
        # NaN/inf reach the async learner's gradients.
        if not (math.isfinite(t.mean_surface_rumble) and math.isfinite(t.speed)
                and math.isfinite(t.mean_tire_slip_ratio)):
            return 0.0
        on_road = t.mean_surface_rumble < c.offroad_rumble
        progress, ds = self._progress(t, prev_t, on_road)
        # the speed bonuses (boot/launch) only pay for REAL forward motion: advancing along the route
        # (ds>0), or - with no centerline - the warm-start demos, which drive forward. This stops a car
        # from farming the speed bonus by REVERSING (ds<0, the circle-proof reward leaking back in).
        fwd_ok = (ds is None) or (ds > 0.0)
        offroad = c.offroad_w * t.mean_surface_rumble
        slip = c.slip_w * max(0.0, t.mean_tire_slip_ratio - c.slip_deadband)
        jerk = 0.0
        if action is not None and prev_action is not None:
            jerk = c.jerk_w * float((float(action[0]) - float(prev_action[0])) ** 2)
        steer_pen = 0.0
        brake_pen = 0.0
        launch = 0.0
        if action is not None:
            steer = float(action[0])
            throttle = float(action[1]) if len(action) > 1 else 0.0
            brake = float(action[2]) if len(action) > 2 else 0.0
            low_speed = max(0.0, min(1.0, (c.low_speed_steer_speed - max(0.0, t.speed))
                                     / max(1e-6, c.low_speed_steer_speed)))
            steer_pen = (c.steer_w + c.low_speed_steer_w * low_speed) * steer * steer
            brake_pen = c.brake_w * brake * (0.5 + low_speed)
            launch = c.launch_w * low_speed * max(0.0, throttle - brake) if (on_road and fwd_ok) else 0.0
        # anti-circling: penalize sustained yaw beyond a cornering deadband, so the agent
        # can't farm the speed reward by spinning in place. Live telemetry only (the
        # warm-start shim has no yaw -> 0 penalty, correct: the demos drive straight).
        yaw_rate = abs(float(getattr(t, "angular_velocity_y", 0.0)))
        spin = c.spin_w * max(0.0, yaw_rate - c.spin_deadband)
        # dense bootstrap: small forward-speed bonus on-road (see boot_w). Circling has speed
        # too, but its spin penalty (>boot) keeps it net-negative, so this can't revive the hack.
        boot = c.boot_w * (min(max(0.0, t.speed), c.speed_cap) / c.speed_cap) if (on_road and fwd_ok) else 0.0
        idle = c.idle_w if on_road and max(0.0, t.speed) < c.idle_speed else 0.0
        r = float(progress + boot + launch - offroad - slip - jerk - spin
                  - steer_pen - brake_pen - idle)
        return r if math.isfinite(r) else 0.0

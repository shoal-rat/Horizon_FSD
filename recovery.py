"""
recovery.py - Horizon FSD, Phase 5

Crash/stuck/flip detection (from telemetry) + the automated reset LADDER that lets
RL training run unattended.

Detection (debounced):
  * impact  : speed collapses by > impact_speed_drop in ONE ~0.05s tick = a collision.
              Detected INSTANTLY so a rewind (which only goes back a few seconds) still
              catches the pre-crash state.
  * stuck   : speed < stuck_speed for stuck_seconds while throttle is commanded (wedged).
  * flipped : |roll|/|pitch| past upright for flip_seconds.
  * offroad : mean surface rumble high + slow for offroad_seconds.

Three real game mechanics (do NOT conflate them):
  * REWIND (Y)                   : rolls back a few seconds of your own path, upright. Fast
              fix for a FRESH on-road impact or a just-happened flip (short rewind window).
  * ANNA AutoDrive (Down,Left,A) : the PRIMARY recovery - ONE feature that branches on state:
              - car FAR off-road -> a transfer PROMPT appears; A snaps the car to road centre,
                then it drives.  (Telemetry signature: car FROZEN, then a >30 m position JUMP.)
              - car ON a road but stuck -> NO prompt; it just drives along the route line.
              So AutoDrive is ITSELF both the teleport (far) and the drive-back (stuck). It needs
              a waypoint pinned to the route. We press A ONLY while the car is frozen in a short
              post-open window (the prompt) and STOP the instant it teleports or starts driving -
              tapping A into a live AutoDrive cancels it.
  * Pause RESET CAR POSITION (Start,L3,A) : LAST-RESORT fallback only, for states AutoDrive
              cannot route (water, deep wedge, ANNA unavailable). NOT the universal teleport.

Every reset is TELEMETRY-GATED (poll Data Out until the car is live/on-road), not a fixed sleep.
Requires FH6 damage = None/Cosmetic so rewind isn't greyed out.
"""
from __future__ import annotations

import logging
import math
import os
import time
from dataclasses import dataclass
from typing import Optional

from forza_telemetry import ForzaTelemetry

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------
@dataclass
class DetectorConfig:
    stuck_speed: float = 2.0        # m/s: "not moving"
    stuck_seconds: float = 1.5      # accelerating-but-not-moving this long -> stuck
    stuck_hard_seconds: float = 30.0 # off-road frozen this long -> stuck (safety net)
    stuck_throttle_min: float = 0.2 # "accelerator applied"
    stuck_brake_max: float = 0.2    # ... and not braking/reversing
    stuck_throttle_grace: float = 0.5  # only "stuck" if throttle was applied within this long (else the
    #                                    agent deliberately stopped; a single stale tap shouldn't latch it)
    impact_speed_drop: float = 4.0  # m/s lost per ~0.05s tick = collision; scaled by real dt (loop overrun)
    flip_roll: float = 1.2          # rad (~69 deg)
    flip_pitch: float = 0.9         # rad (~52 deg)
    flip_seconds: float = 0.5
    offroad_rumble: float = 0.15    # mean surface rumble above this = off the paved road
    offroad_seconds: float = 1.0    # ... sustained this long = off-road (was 2.0 AND gated on speed<10,
    #                                 so a car driving FAST off-road was never caught - that gate is gone)
    # --- route-aware termination (needs centerline.npy): end the episode the MOMENT the car leaves the
    #     training route. This gives a prompt "off-route = bad" signal AND keeps the car NEAR the route so
    #     AutoDrive recovers it with a short drive back instead of teleporting to a far road. ---
    centerline_path: str = r"C:\Horizon_FSD\centerline.npy"
    offroute_dist: float = 18.0     # m of lateral distance from the route centreline = off the route
    offroute_seconds: float = 0.8
    noprogress_seconds: float = 5.0 # on-route + MOVING but no centreline advance this long = circling/wrong-way
    noprogress_min_advance: float = 3.0  # m of new arc-length that counts as "making progress"
    noprogress_speed: float = 3.0   # only flag noprogress while actually MOVING (a slow/stopped car is
    #                                 the stuck/idle detector's job; this catches circling at speed)
    nominal_dt: float = 0.05        # expected control tick; the rate/timer tests are calibrated to it
    dt_skip_factor: float = 3.0     # a tick longer than this*nominal_dt (GPU stall) is a discontinuity:
    #                                 skip the rate/timer checks rather than fabricate impact/stuck/noprogress
    teleport_jump_m: float = 30.0   # a one-tick world-position jump bigger than this = fast-travel/respawn/
    #                                 teleport, not driving -> suppress offroute/impact/noprogress, re-anchor
    stuck_displacement_m: float = 1.0  # "stuck" also requires near-zero WORLD displacement, so a slow uphill
    #                                    grind that is still crawling forward isn't mistaken for wedged


class CrashDetector:
    """Feed telemetry over time; update() returns a reason
    ('impact'/'stuck'/'flipped'/'offroad'/'offroute'/'noprogress') or None. Reasons are debounced."""

    def __init__(self, cfg: DetectorConfig = DetectorConfig()) -> None:
        self.cfg = cfg
        self._centerline = None
        try:                                            # route-aware checks need the same centerline
            if cfg.centerline_path and os.path.exists(cfg.centerline_path):
                from centerline import Centerline
                self._centerline = Centerline.load(cfg.centerline_path)
        except Exception:
            logger.exception("detector: failed to load centerline")
            self._centerline = None
        self.reset()

    @staticmethod
    def _xz(t) -> Optional[tuple]:
        """Finite (x, z) world position, or None (so all callers fail safe on a garbage/absent frame)."""
        if hasattr(t, "position_x") and hasattr(t, "position_z"):
            x, z = float(t.position_x), float(t.position_z)
            if math.isfinite(x) and math.isfinite(z):
                return (x, z)
        return None

    def reset(self) -> None:
        self._flip_since: Optional[float] = None
        self._slow_since: Optional[float] = None
        self._slow_pos: Optional[tuple] = None
        self._offroad_since: Optional[float] = None
        self._prev_speed: Optional[float] = None
        self._prev_t_wall: Optional[float] = None
        self._prev_pos: Optional[tuple] = None
        self._throttle_last: Optional[float] = None
        self._offroute_since: Optional[float] = None
        self._best_arc: Optional[float] = None
        self._arc_since: Optional[float] = None

    def update(self, t: ForzaTelemetry, now: float,
               throttle_cmd: float = 1.0, brake_cmd: float = 0.0) -> Optional[str]:
        if not t.is_driving:          # menu/pause/replay/rewind - not a crash
            self._prev_speed = None
            return None
        c = self.cfg

        dt = (now - self._prev_t_wall) if self._prev_t_wall is not None else c.nominal_dt
        self._prev_t_wall = now
        pos = self._xz(t)
        overrun = dt > c.dt_skip_factor * c.nominal_dt   # a GPU/learner stall: a much-longer-than-normal tick

        # Teleport guard: a one-tick world-position JUMP (fast-travel / respawn / AutoDrive transfer /
        # load screen) is not driving - every position-based test is meaningless this tick, so re-anchor
        # and skip so we never fabricate impact/offroute/noprogress from the discontinuity.
        teleport = (pos is not None and self._prev_pos is not None
                    and math.dist(pos, self._prev_pos) > c.teleport_jump_m)
        self._prev_pos = pos
        if teleport:
            self._prev_speed = t.speed
            self._slow_since = self._slow_pos = self._throttle_last = None
            self._offroute_since = self._arc_since = self._flip_since = self._offroad_since = None
            self._best_arc = None              # arc is discontinuous after a teleport -> re-anchor
            return None

        # impact: a sudden speed collapse = collision. SKIP on an over-running tick (a stall's long dt
        # would read normal braking over the gap as a crash, teaching the agent that braking is fatal);
        # otherwise scale the threshold by the real dt.
        if self._prev_speed is not None and not overrun:
            thresh = -c.impact_speed_drop * min(2.0, max(0.5, dt / c.nominal_dt))
            if (t.speed - self._prev_speed) < thresh:
                self._prev_speed = t.speed
                return "impact"
        self._prev_speed = t.speed

        if abs(t.roll) > c.flip_roll or abs(t.pitch) > c.flip_pitch:
            if self._flip_since is None:
                self._flip_since = now
            if now - self._flip_since > c.flip_seconds:
                return "flipped"
        else:
            self._flip_since = None

        # stuck = the agent is TRYING to accelerate but the car won't move, for a
        # sustained window. The window is SPEED-based (so a momentary throttle dip
        # doesn't reset it), and we just require that throttle was applied at some
        # point while slow - i.e. "accelerator pressed AND speed ~0".
        if t.speed < c.stuck_speed:
            if self._slow_since is None:
                self._slow_since = now
                self._slow_pos = pos
            # remember WHEN throttle was last applied (not just that it ever was): a single stale tap
            # shouldn't latch "stuck" forever once the agent legitimately brakes/holds.
            if throttle_cmd > c.stuck_throttle_min and brake_cmd < c.stuck_brake_max:
                self._throttle_last = now
            elapsed = now - self._slow_since
            trying = (self._throttle_last is not None
                      and now - self._throttle_last <= c.stuck_throttle_grace)
            # ...and the car is genuinely NOT moving through the world: a slow uphill crawl that still
            # covers ground isn't "stuck". Falls back to speed-only when position is unavailable.
            not_moving = (pos is None or self._slow_pos is None
                          or math.dist(pos, self._slow_pos) < c.stuck_displacement_m)
            hard_stuck = (
                c.stuck_hard_seconds > 0.0
                and t.mean_surface_rumble > c.offroad_rumble
                and elapsed > c.stuck_hard_seconds
            )
            if (trying and not_moving and elapsed > c.stuck_seconds) or hard_stuck:
                return "stuck"
        else:
            self._slow_since = None
            self._slow_pos = None
            self._throttle_last = None

        # off-road: on a rough surface (grass/snow), AT ANY SPEED, for a short window. The old
        # speed<10 gate let the agent drive fast off-road forever without ever being caught.
        if t.mean_surface_rumble > c.offroad_rumble:
            if self._offroad_since is None:
                self._offroad_since = now
            if now - self._offroad_since > c.offroad_seconds:
                return "offroad"
        else:
            self._offroad_since = None

        # route-aware termination (only when a centerline + live position are available)
        if self._centerline is not None and hasattr(t, "position_x") and hasattr(t, "position_z"):
            s, lat, at_end = self._centerline.project(t.position_x, t.position_z)
            if lat > c.offroute_dist:
                # left the route -> end NOW, while still near it (AutoDrive recovers with a short drive)
                if self._offroute_since is None:
                    self._offroute_since = now
                if now - self._offroute_since > c.offroute_seconds:
                    return "offroute"
            else:
                self._offroute_since = None
                # advance the best-arc marker (NOTE: `is None` check, not `or` - arc 0.0 is falsy and
                # the `or` form re-read s, firing noprogress every episode that starts at the origin)
                if self._best_arc is None or s > self._best_arc + c.noprogress_min_advance:
                    self._best_arc = s if self._best_arc is None else max(self._best_arc, s)
                    self._arc_since = now
                elif (now - self._arc_since > c.noprogress_seconds
                      and t.speed >= c.noprogress_speed):     # moving but not advancing = circling/wrong-way
                    if at_end or (self._centerline.length - s) <= max(c.noprogress_min_advance, 5.0):
                        return "route_complete"               # reached the end of an open route - benign
                    return "noprogress"

        return None


# ---------------------------------------------------------------------------
# Reset ladder
# ---------------------------------------------------------------------------
@dataclass
class ResetConfig:
    rewind_button: str = "Y"
    confirm_button: str = "A"
    autodrive_down_button: str = "DPAD_DOWN"   # ANNA satnav: Down,
    autodrive_left_button: str = "DPAD_LEFT"   #             then Left,
    #                                          then A -> AutoDrive (re-centres on the road)
    menu_button: str = "START"                 # (legacy reset-to-road, unreliable on this build)
    reset_position_button: str = "LEFT_THUMB"  # pause menu "Reset Car Position" on controller
    reset_select_button: str = "X"             # keyboard/controller fallback used by some builds
    rewind_presses: int = 1
    rewind_settle_s: float = 3.0      # wait for the rewind animation to FINISH before A
    press_gap_s: float = 0.5          # gap between AutoDrive menu presses (was 0.35: too fast, missed presses)
    tap_hold_s: float = 0.10          # hold the D-pad menu taps a touch longer than the 0.08 default
    confirm_hold_s: float = 0.15      # hold A longer so the AutoDrive confirm isn't dropped
    autodrive_prompt_retry_s: float = 1.0 # gap between confirm-A taps while the prompt is up
    autodrive_prompt_attempts: int = 3    # confirm-A presses for the teleport prompt. Sent ONLY while the
    #                                       car is FROZEN (a modal holds it still), so a couple of spares are
    #                                       safe and cover a missed first press / a late prompt.
    autodrive_prompt_settle_s: float = 0.6 # the car must be positionally FROZEN this long before the confirm-A,
    #                                        so we never tap during the post-crash COAST-DOWN nor into a live
    #                                        AutoDrive drive (both keep the car moving). NOT a window after open.
    autodrive_frozen_eps: float = 0.5     # per-tick world motion (m) at/below which the car counts as frozen
    autodrive_reissue_s: float = 10.0 # no progress this long -> re-open ANNA AutoDrive
    autodrive_engage_deadline_s: float = 8.0  # if AutoDrive never moves the car by now (no waypoint /
    #                                           unroutable), bail to the pause-reset instead of waiting out
    #                                           the full timeout every crash
    autodrive_progress_min_m: float = 2.0
    autodrive_on_route_settle_s: float = 0.7
    autodrive_persistent: bool = True # keep training alive by waiting/retrying AutoDrive
    autodrive_persistent_retry_s: float = 2.0
    autodrive_teleport_jump_m: float = 30.0 # per-sample coordinate jump => teleport, not teachable
    autodrive_break_s: float = 1.0    # after on-road, hold a BRAKE input this long to CANCEL the lingering
    #                                   AutoDrive (~1s to release) before handing back. Brake, not throttle:
    #                                   AutoDrive places the car at road centre with no guaranteed heading,
    #                                   so a throttle pulse would launch it (maybe into oncoming/a barrier)
    menu_open_s: float = 0.9
    confirm_dialog_s: float = 0.5
    settle_s: float = 1.5
    confirm_timeout_s: float = 12.0   # max wait for the car to be live after rewind
    min_recovered_speed: float = 3.0  # rewind success = moving again (m/s)
    autodrive_timeout_s: float = 45.0 # max wait for one AutoDrive attempt (was 90: too long; escalate sooner)
    on_road_speed: float = 5.0        # m/s: "back on the road and moving"
    on_road_rumble: float = 0.15      # mean surface rumble below this = on a road
    centerline_path: str = r"C:\Horizon_FSD\centerline.npy"
    route_max_dist: float = 22.0      # m: recovered state must be near our training route if centerline exists
    require_route_if_available: bool = True
    # The pause RESET CAR POSITION is the last rung of every ladder (not a queue-jumping
    # escalation), so a wedged/flipped car AutoDrive can't route reaches it each round.
    persistent_max_backoff_s: float = 15.0  # cap the persistent-retry backoff
    persistent_max_seconds: float = 300.0   # even when persistent, give up after this so a truly unroutable
    #                                         car returns FAILED (the env can then stop / hand off) - never an
    #                                         infinite silent hang with the learner training on zero transitions
    heartbeat_every: int = 5          # log a warning every N failed rounds (visibility, not a silent hang)


class ForzaResetter:
    def __init__(self, gamepad, telemetry, cfg: ResetConfig = ResetConfig(),
                 demo_recorder=None, detector_cfg: Optional[DetectorConfig] = None) -> None:
        self.pad = gamepad
        self.rx = telemetry
        self.cfg = cfg
        # Use the SAME flip thresholds the detector terminates on, so a tuned flip_roll/flip_pitch
        # in config.yaml is honored by recovery acceptance too (no config drift).
        self.det_cfg = detector_cfg or DetectorConfig()
        self.demo_recorder = demo_recorder
        self._centerline = None
        try:
            if cfg.centerline_path and os.path.exists(cfg.centerline_path):
                from centerline import Centerline
                self._centerline = Centerline.load(cfg.centerline_path)
        except Exception:
            logger.exception("failed to load recovery centerline")
            self._centerline = None

    # ---- waits -----------------------------------------------------------
    def _route_distance(self, t: ForzaTelemetry) -> Optional[float]:
        if self._centerline is None:
            return None
        if not hasattr(t, "position_x") or not hasattr(t, "position_z"):
            return None
        return self._centerline.project(t.position_x, t.position_z)[1]

    def _is_recovered(self, t: Optional[ForzaTelemetry],
                      require_speed: Optional[float] = None,
                      require_route: bool = True) -> bool:
        if t is None or not t.is_driving:
            return False
        if abs(t.roll) > self.det_cfg.flip_roll or abs(t.pitch) > self.det_cfg.flip_pitch:
            return False
        if require_speed is not None and t.speed < require_speed:
            return False
        if t.mean_surface_rumble >= self.cfg.on_road_rumble:
            return False
        # require_route=False for a TELEPORT escalation: its job is just to FREE a wedged/flipped
        # car onto a road (upright, live); AutoDrive or the agent's own driving brings it to route.
        lat = self._route_distance(t)
        if (require_route and self.cfg.require_route_if_available
                and lat is not None and lat > self.cfg.route_max_dist):
            return False
        return True

    def _wait_recovered(self, require_speed: Optional[float] = None,
                        timeout_s: Optional[float] = None,
                        require_route: bool = True) -> bool:
        t0 = time.perf_counter()
        timeout = self.cfg.confirm_timeout_s if timeout_s is None else timeout_s
        while time.perf_counter() - t0 < timeout:
            t = self.rx.latest()
            if self._is_recovered(t, require_speed=require_speed, require_route=require_route):
                return True
            time.sleep(0.05)
        return False

    def _wait_live(self, require_speed: Optional[float] = None) -> bool:
        return self._wait_recovered(require_speed=require_speed)

    def _wait_on_road(self) -> bool:
        """AutoDrive drives the car back; wait until it is route-verified and moving."""
        return self._wait_recovered(require_speed=self.cfg.on_road_speed,
                                    timeout_s=self.cfg.autodrive_timeout_s)

    def _open_autodrive(self) -> None:
        """Open ANNA AutoDrive for the already-pinned destination.

        The final A chooses AutoDrive. Some off-road states then show a second
        "teleport to nearby road" confirmation; _wait_autodrive_resolved handles
        that by pressing A only while telemetry still looks paused or stationary.
        """
        c = self.cfg
        self.pad.reset()
        self.pad.tap_button(c.autodrive_down_button, hold_s=c.tap_hold_s)
        time.sleep(c.press_gap_s)
        self.pad.tap_button(c.autodrive_left_button, hold_s=c.tap_hold_s)
        time.sleep(c.press_gap_s)
        self.pad.tap_button(c.confirm_button, hold_s=c.confirm_hold_s)

    def _wait_autodrive_resolved(self, start_pos: Optional[tuple[float, float]]) -> tuple[bool, bool]:
        """Wait for AutoDrive to recover the car, telling its two branches apart from telemetry
        alone (it can't read the prompt text):
          * teleport branch (far off-road): the car stays FROZEN, then a >teleport_jump_m position
            JUMP fires the instant A accepts the transfer prompt and snaps it to road centre.
          * drive branch (on-road-stuck): no prompt, the car accumulates forward displacement.
        A is pressed ONLY to accept the prompt - while the car is frozen, after a short settle, in a
        bounded window, capped. We STOP for good the instant it teleports OR starts driving, so we
        never tap A into a live AutoDrive drive (a stray A there cancels it). Returns
        (recovered, teleported)."""
        c = self.cfg
        t0 = time.perf_counter()
        last_a = t0 - c.autodrive_prompt_retry_s
        last_reissue = t0
        last_progress = t0
        last_move = t0                         # last tick the car's world position changed appreciably
        prev_pos = start_pos
        last_pos = start_pos
        teleported = False
        driving = False                        # car has actually moved => AutoDrive engaged (or teleport fired)
        route_seen_since: Optional[float] = None
        prompt_presses = 0

        while time.perf_counter() - t0 < c.autodrive_timeout_s:
            now = time.perf_counter()
            t = self.rx.latest()

            if self.demo_recorder is not None:
                self.demo_recorder.sample(t)

            if self._is_recovered(t, require_speed=None):
                route_seen_since = route_seen_since or now
                if now - route_seen_since >= c.autodrive_on_route_settle_s:
                    return True, teleported
            else:
                route_seen_since = None

            pos = self._position()
            step_move = self._displacement(prev_pos, pos)   # how far the car moved THIS tick
            if step_move > c.autodrive_teleport_jump_m:
                teleported = True              # a discrete snap to road centre = the prompt was accepted
            if step_move > c.autodrive_frozen_eps:
                last_move = now                # the car is moving (coast-down / AutoDrive driving / teleport)
            if pos is not None:
                prev_pos = pos
            if self._displacement(start_pos, pos) >= c.autodrive_progress_min_m:
                driving = True                 # net displacement from start => engaged/coasted at some point
            if self._displacement(last_pos, pos) >= c.autodrive_progress_min_m:
                last_pos = pos
                last_progress = now

            # Confirm the teleport PROMPT (this is a modal: it holds the car positionally FROZEN). Tap A
            # only after the car has been FROZEN for the settle - NOT gated on cumulative displacement,
            # because a car still COASTING from the crash trips `driving` and would wrongly block the very
            # A that accepts this prompt. Frozen != a live drive, so A here can't cancel AutoDrive. Stop
            # the instant it teleports.
            frozen_for = now - last_move
            if (not teleported and frozen_for >= c.autodrive_prompt_settle_s
                    and prompt_presses < c.autodrive_prompt_attempts
                    and now - last_a >= c.autodrive_prompt_retry_s):
                self.pad.tap_button(c.confirm_button, hold_s=c.confirm_hold_s)
                last_a = now
                prompt_presses += 1

            # AutoDrive never engaged (no waypoint / unreachable) -> re-open ANNA once...
            if (not driving and not teleported
                    and now - last_progress >= c.autodrive_reissue_s
                    and now - last_reissue >= c.autodrive_reissue_s):
                self._open_autodrive()
                last_reissue = now
                last_progress = now

            # ...and if it has neither teleported nor recovered by the engage deadline AFTER the confirm-A
            # presses were spent (the prompt didn't take, or there's no waypoint), bail to the pause-reset
            # rather than burning the whole timeout. NOT gated on `driving`: the post-crash coast-down sets
            # it and would wedge us here forever.
            if (not teleported and now - t0 > c.autodrive_engage_deadline_s
                    and prompt_presses >= c.autodrive_prompt_attempts):
                logger.warning("AutoDrive did not engage in %.0fs (prompt unconfirmed / no waypoint pinned?) "
                               "- falling through to the pause-reset fallback", now - t0)
                return False, teleported

            time.sleep(0.05)
        return False, teleported

    def _position(self) -> Optional[tuple[float, float]]:
        t = self.rx.latest()
        if t is None:
            return None
        return (float(t.position_x), float(t.position_z))

    @staticmethod
    def _displacement(a: Optional[tuple[float, float]],
                      b: Optional[tuple[float, float]]) -> float:
        if a is None or b is None:
            return 0.0
        dx, dz = b[0] - a[0], b[1] - a[1]
        return (dx * dx + dz * dz) ** 0.5

    # ---- macros ----------------------------------------------------------
    def rewind(self) -> bool:
        c = self.cfg
        self.pad.reset()
        for _ in range(c.rewind_presses):
            self.pad.tap_button(c.rewind_button)
            time.sleep(c.press_gap_s)
        time.sleep(c.rewind_settle_s)                # wait for the rewind to FINISH
        t = self.rx.latest()                         # only A-resume if still in the rewind/timeline UI;
        if t is None or not t.is_driving:            # if rewind already handed back, A would hit the live world
            self.pad.tap_button(c.confirm_button)
        return self._wait_recovered(require_speed=c.min_recovered_speed)

    def autodrive_reset(self) -> bool:
        """ANNA AutoDrive back to the route, accepting the optional teleport prompt
        if the game offers it, then explicitly cancel AutoDrive before handoff."""
        c = self.cfg
        start_pos = self._position()
        if self.demo_recorder is not None:
            self.demo_recorder.begin("autodrive", start_pos=start_pos)
        self._open_autodrive()
        success, teleported = self._wait_autodrive_resolved(start_pos)
        if self.demo_recorder is not None:
            self.demo_recorder.end(success, teleported=teleported)
        if not success:
            return False
        # Cancel the lingering AutoDrive (any input cancels it; it takes ~1s to release). Use a
        # BRAKE, not throttle: AutoDrive drops the car at road centre with no guaranteed heading,
        # so a throttle pulse would launch it (maybe into oncoming traffic / a barrier). Brake
        # cancels just as well and hands back a settled car, then neutral.
        self.pad.apply([0.0, 0.0, 0.4])
        time.sleep(c.autodrive_break_s)
        self.pad.reset()
        return True

    def reset_position(self) -> bool:
        """Pause-menu 'Reset Car Position'. This is a teleport-style recovery, unlike
        ANNA AutoDrive. It may be unavailable in some modes, so callers still keep
        rewind/autodrive fallbacks."""
        c = self.cfg
        self.pad.reset()
        self.pad.tap_button(c.menu_button)
        time.sleep(c.menu_open_s)
        self.pad.tap_button(c.reset_position_button)
        time.sleep(c.confirm_dialog_s)
        self.pad.tap_button(c.confirm_button)
        time.sleep(c.settle_s)
        return self._wait_recovered(require_route=False)   # teleport: just free the car onto a road

    def reset_to_road(self) -> bool:
        """Legacy alternate binding for reset-position prompts."""
        c = self.cfg
        self.pad.reset()
        self.pad.tap_button(c.menu_button)
        time.sleep(c.menu_open_s)
        self.pad.tap_button(c.reset_select_button)
        time.sleep(c.confirm_dialog_s)
        self.pad.tap_button(c.confirm_button)
        time.sleep(c.settle_s)
        return self._wait_recovered(require_route=False)   # teleport: just free the car onto a road

    # ---- ladder ----------------------------------------------------------
    def recover(self, reason: str, max_attempts: int = 3) -> str:
        """Run the reset ladder. Returns the method that worked, else 'FAILED'.

        AutoDrive is the primary "get back to the route" tool - it internally teleports (far off-road,
        via its transfer prompt) or drives back (on-road stuck), and its transfer branch also rights a
        flipped car. If AutoDrive instead leaves the car upright on a DIFFERENT real road (off-route
        but perfectly drivable), we accept that rather than pause-and-teleport it. The pause-menu RESET
        CAR POSITION is the last-resort rung. With autodrive_persistent=True this keeps retrying with
        backoff + a heartbeat, but it gives up after persistent_max_seconds and returns 'FAILED' so a
        truly unroutable car stops the run / triggers a handoff instead of hanging the learner forever.
        """
        c = self.cfg
        start = time.perf_counter()
        attempt = 0
        while True:
            attempt += 1
            try:
                if self.autodrive_reset():
                    return "autodrive"
            except Exception:  # pragma: no cover
                logger.exception("autodrive recovery error")
            # AutoDrive may have routed the car onto a real parallel road just past route_max_dist:
            # upright, on tarmac, moving - don't pause-and-teleport a perfectly drivable car. Cancel
            # the lingering AutoDrive (brake) and hand back.
            if self._is_recovered(self.rx.latest(), require_route=False):
                self.pad.apply([0.0, 0.0, 0.4])
                time.sleep(c.autodrive_break_s)
                self.pad.reset()
                return "autodrive_offroute_ok"
            for name, fn in (("reset_position", self.reset_position), ("reset_to_road", self.reset_to_road)):
                try:
                    if fn():
                        return name
                except Exception:  # pragma: no cover
                    logger.exception("%s recovery error", name)

            elapsed = time.perf_counter() - start
            if (not c.autodrive_persistent and attempt >= max_attempts) or elapsed > c.persistent_max_seconds:
                logger.error("recovery FAILED (reason=%s, attempts=%d, elapsed=%.0fs) - car unroutable",
                             reason, attempt, elapsed)
                return "FAILED"
            if attempt % max(1, c.heartbeat_every) == 0:
                logger.warning("recovery still failing: reason=%s attempts=%d elapsed=%.0fs - retrying "
                               "with backoff", reason, attempt, elapsed)
            time.sleep(min(c.autodrive_persistent_retry_s * (1 + attempt // max(1, c.heartbeat_every)),
                           c.persistent_max_backoff_s))

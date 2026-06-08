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

Reset ladder:
  * impact/stuck near a barrier : REWIND first. This is the only option that reliably
              backs out of an on-road guardrail hit.
  * off-route/stuck fallback    : ANNA AutoDrive to the pinned route. If the game asks
              to teleport to a nearby road, keep tapping A while telemetry looks paused
              or stationary; otherwise wait while AutoDrive drives back to the center.
  * legacy fallback             : pause-menu RESET CAR POSITION, kept for flipped cars
              and builds where ANNA is unavailable.

Every reset is TELEMETRY-GATED (poll Data Out until the car is live/on-road), not a fixed sleep.
Requires FH6 damage = None/Cosmetic so rewind isn't greyed out.
"""
from __future__ import annotations

import logging
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
    impact_speed_drop: float = 4.0  # m/s lost in one ~0.05s tick = collision (detect instantly)
    flip_roll: float = 1.2          # rad (~69 deg)
    flip_pitch: float = 0.9         # rad (~52 deg)
    flip_seconds: float = 0.5
    offroad_rumble: float = 0.15    # mean surface rumble
    offroad_speed: float = 10.0     # only "off-road wallowing" if also slow
    offroad_seconds: float = 2.0


class CrashDetector:
    """Feed telemetry over time; update() returns a reason
    ('impact'/'stuck'/'flipped'/'offroad') or None. Reasons are debounced."""

    def __init__(self, cfg: DetectorConfig = DetectorConfig()) -> None:
        self.cfg = cfg
        self.reset()

    def reset(self) -> None:
        self._flip_since: Optional[float] = None
        self._slow_since: Optional[float] = None
        self._offroad_since: Optional[float] = None
        self._prev_speed: Optional[float] = None
        self._throttle_seen = False

    def update(self, t: ForzaTelemetry, now: float,
               throttle_cmd: float = 1.0, brake_cmd: float = 0.0) -> Optional[str]:
        if not t.is_driving:          # menu/pause/replay/rewind - not a crash
            self._prev_speed = None
            return None
        c = self.cfg

        # impact: a sudden speed collapse in one tick = collision -> detect INSTANTLY,
        # so rewind catches the pre-crash state within its short window.
        if self._prev_speed is not None and (t.speed - self._prev_speed) < -c.impact_speed_drop:
            self._prev_speed = t.speed
            return "impact"
        self._prev_speed = t.speed

        if abs(t.roll) > c.flip_roll or abs(t.pitch) > c.flip_pitch:
            self._flip_since = self._flip_since or now
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
                self._throttle_seen = False
            # "accelerator pressed AND not braking/reversing" -> the agent is trying to
            # go forward but can't. A momentary dip doesn't reset it (the latch stays).
            if throttle_cmd > c.stuck_throttle_min and brake_cmd < c.stuck_brake_max:
                self._throttle_seen = True
            elapsed = now - self._slow_since
            hard_stuck = (
                c.stuck_hard_seconds > 0.0
                and t.mean_surface_rumble > c.offroad_rumble
                and elapsed > c.stuck_hard_seconds
            )
            if (self._throttle_seen and elapsed > c.stuck_seconds) or hard_stuck:
                return "stuck"
        else:
            self._slow_since = None
            self._throttle_seen = False

        if t.mean_surface_rumble > c.offroad_rumble and t.speed < c.offroad_speed:
            self._offroad_since = self._offroad_since or now
            if now - self._offroad_since > c.offroad_seconds:
                return "offroad"
        else:
            self._offroad_since = None

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
    confirm_attempts: int = 3         # re-press A up to this many times if AutoDrive didn't engage
    confirm_check_s: float = 2.0      # after A, watch this long for the car to start driving itself
    autodrive_engaged_speed: float = 2.0  # m/s: car moving on its own (we hold neutral) = AutoDrive took over
    autodrive_min_displacement: float = 12.0 # legacy metric; route verification is authoritative
    autodrive_prompt_retry_s: float = 1.0 # while stopped/menu-like, tap A for "teleport to nearby road"
    autodrive_prompt_attempts: int = 12
    autodrive_reissue_s: float = 10.0 # no progress this long -> re-open ANNA AutoDrive
    autodrive_progress_min_m: float = 2.0
    autodrive_on_route_settle_s: float = 0.7
    autodrive_persistent: bool = True # keep training alive by waiting/retrying AutoDrive
    autodrive_persistent_retry_s: float = 2.0
    autodrive_teleport_jump_m: float = 30.0 # per-sample coordinate jump => teleport, not teachable
    autodrive_break_s: float = 0.6    # after on-road, hold a control input this long to CANCEL the
    #                                   lingering AutoDrive (~1s) before handing back, so the agent's
    #                                   first real action isn't eaten by the leftover autopilot
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
    escalate_window_s: float = 8.0    # a new crash within this of a reset = a "repeat"
    max_consecutive_rewinds: int = 2  # after this many quick repeats, fall to AutoDrive reset
    # --- escalation: after the in-distribution methods (rewind/AutoDrive) keep failing, the car
    #     is likely wedged/flipped where AutoDrive can't drive it out -> force a TELEPORT, which
    #     recovers from ANY state. This is what stops the persistent loop hanging forever. ---
    escalate_after_failures: int = 2  # full failed rounds before a teleport leads the ladder
    persistent_max_backoff_s: float = 15.0  # cap the persistent-retry backoff
    heartbeat_every: int = 5          # log a warning every N failed rounds (visibility, not silent hang)


class ForzaResetter:
    def __init__(self, gamepad, telemetry, cfg: ResetConfig = ResetConfig(),
                 demo_recorder=None) -> None:
        self.pad = gamepad
        self.rx = telemetry
        self.cfg = cfg
        self.demo_recorder = demo_recorder
        self._last_recovery_time = 0.0
        self._consecutive_rewinds = 0
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
        _, lat = self._centerline.project(t.position_x, t.position_z)
        return lat

    def _is_recovered(self, t: Optional[ForzaTelemetry],
                      require_speed: Optional[float] = None,
                      require_route: bool = True) -> bool:
        if t is None or not t.is_driving:
            return False
        if abs(t.roll) > DetectorConfig.flip_roll or abs(t.pitch) > DetectorConfig.flip_pitch:
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
        """Wait for either AutoDrive to return to the route or the optional
        teleport prompt to be accepted.

        Telemetry cannot read the dialog text, so the signal is behavioral:
        if the car is not live, not moving, or still far from the route, occasional
        A taps are safe and useful. Once AutoDrive is moving, we stop tapping and
        just wait for the route/on-road gate.
        """
        c = self.cfg
        t0 = time.perf_counter()
        last_a = t0 - c.autodrive_prompt_retry_s
        last_reissue = t0
        last_progress = t0
        last_pos = start_pos
        teleport_pos = start_pos
        teleported = False
        route_seen_since: Optional[float] = None
        prompt_attempts_left = c.autodrive_prompt_attempts

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
            if self._displacement(teleport_pos, pos) > c.autodrive_teleport_jump_m:
                teleported = True
            if pos is not None:
                teleport_pos = pos
            if self._displacement(last_pos, pos) >= c.autodrive_progress_min_m:
                last_pos = pos
                last_progress = now

            moving = bool(t is not None and t.is_driving and t.speed >= c.autodrive_engaged_speed)
            prompt_like = t is None or not t.is_driving or not moving
            if prompt_like and prompt_attempts_left > 0 and now - last_a >= c.autodrive_prompt_retry_s:
                self.pad.tap_button(c.confirm_button, hold_s=c.confirm_hold_s)
                last_a = now
                prompt_attempts_left -= 1

            if now - last_progress >= c.autodrive_reissue_s and now - last_reissue >= c.autodrive_reissue_s:
                self._open_autodrive()
                last_reissue = now
                last_progress = now
                last_a = now
                prompt_attempts_left = c.autodrive_prompt_attempts

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

    def _confirm_autodrive(self) -> bool:
        """Tap A to confirm AutoDrive; the tap is sometimes dropped, so retry until the car
        starts driving ITSELF (speed rises while we hold neutral = AutoDrive took over).
        Catching a missed A in ~2s beats burning the full on-road timeout waiting for a car
        that was never handed to AutoDrive."""
        c = self.cfg
        for _ in range(c.confirm_attempts):
            self.pad.tap_button(c.confirm_button, hold_s=c.confirm_hold_s)
            t0 = time.perf_counter()
            while time.perf_counter() - t0 < c.confirm_check_s:
                t = self.rx.latest()
                if t is not None and t.is_driving and t.speed >= c.autodrive_engaged_speed:
                    return True
                time.sleep(0.05)
        return False

    # ---- macros ----------------------------------------------------------
    def rewind(self) -> bool:
        c = self.cfg
        self.pad.reset()
        for _ in range(c.rewind_presses):
            self.pad.tap_button(c.rewind_button)
            time.sleep(c.press_gap_s)
        time.sleep(c.rewind_settle_s)                # wait for the rewind to FINISH
        self.pad.tap_button(c.confirm_button)        # A to resume
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
        # Break the lingering AutoDrive: a control input cancels it, but it takes ~1s to
        # fully release. Hold light throttle so the handoff isn't "missed" (the agent's
        # first action ignored), then neutral -> the episode resumes in clean control.
        self.pad.apply([0.0, 0.3, 0.0])
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

        ESCALATING: try the in-distribution method for the failure type first (rewind /
        AutoDrive). If those keep failing, the car is likely wedged or flipped where AutoDrive
        cannot drive it out, so a TELEPORT (reset position) is forced to the front of the ladder
        - it recovers from ANY state. With autodrive_persistent=True (unattended training) this
        never returns FAILED and never hangs silently on AutoDrive alone: it keeps escalating
        with capped backoff and a periodic heartbeat log.
        """
        c = self.cfg
        rewind = ("rewind", self.rewind)
        autodrive = ("autodrive", self.autodrive_reset)
        teleport = [("reset_position", self.reset_position), ("reset_to_road", self.reset_to_road)]

        # in-distribution first choice per failure type (AutoDrive cannot un-flip a car, so a
        # flipped/unknown state leads with a teleport)
        if reason in ("impact", "stuck"):
            primary = [rewind, autodrive]
        elif reason == "offroad":
            primary = [autodrive]
        else:
            primary = teleport + [autodrive]

        start = time.perf_counter()
        attempt = 0
        while True:
            attempt += 1
            escalating = attempt > c.escalate_after_failures
            order = (teleport + primary) if escalating else primary  # teleport leads once escalating
            for name, fn in order:
                try:
                    if fn():
                        return name + ("+esc" if escalating else "")
                except Exception:  # pragma: no cover
                    logger.exception("%s recovery error", name)

            if not c.autodrive_persistent and attempt >= max_attempts:
                logger.error("recovery FAILED after %d attempts (reason=%s)", attempt, reason)
                return "FAILED"
            if attempt % max(1, c.heartbeat_every) == 0:
                logger.warning(
                    "recovery still failing: reason=%s attempts=%d elapsed=%.0fs - car may be "
                    "wedged/flipped; escalating to teleport and backing off", reason, attempt,
                    time.perf_counter() - start)
            time.sleep(min(c.autodrive_persistent_retry_s * (1 + attempt // max(1, c.heartbeat_every)),
                           c.persistent_max_backoff_s))

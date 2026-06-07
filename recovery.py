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
  * fallback                    : pause-menu RESET CAR POSITION.
  * last resort                 : ANNA AutoDrive, but only if it actually moves the car
              away; AutoDrive is navigation, not a teleport, so it can stay wedged
              against road furniture.

Every reset is TELEMETRY-GATED (poll Data Out until the car is live/on-road), not a fixed sleep.
Requires FH6 damage = None/Cosmetic so rewind isn't greyed out.
"""
from __future__ import annotations

import logging
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
    autodrive_min_displacement: float = 12.0 # m: AutoDrive must actually escape, not push a guardrail
    autodrive_break_s: float = 0.6    # after on-road, hold a control input this long to CANCEL the
    #                                   lingering AutoDrive (~1s) before handing back, so the agent's
    #                                   first real action isn't eaten by the leftover autopilot
    menu_open_s: float = 0.9
    confirm_dialog_s: float = 0.5
    settle_s: float = 1.5
    confirm_timeout_s: float = 12.0   # max wait for the car to be live after rewind
    min_recovered_speed: float = 3.0  # rewind success = moving again (m/s)
    autodrive_timeout_s: float = 20.0 # max wait for AutoDrive to get back on-road
    on_road_speed: float = 5.0        # m/s: "back on the road and moving"
    on_road_rumble: float = 0.15      # mean surface rumble below this = on a road
    escalate_window_s: float = 8.0    # a new crash within this of a reset = a "repeat"
    max_consecutive_rewinds: int = 2  # after this many quick repeats, fall to AutoDrive reset


class ForzaResetter:
    def __init__(self, gamepad, telemetry, cfg: ResetConfig = ResetConfig()) -> None:
        self.pad = gamepad
        self.rx = telemetry
        self.cfg = cfg
        self._last_recovery_time = 0.0
        self._consecutive_rewinds = 0

    # ---- waits -----------------------------------------------------------
    def _wait_live(self, require_speed: Optional[float] = None) -> bool:
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < self.cfg.confirm_timeout_s:
            t = self.rx.latest()
            if t is not None and t.is_driving:
                if require_speed is None or t.speed >= require_speed:
                    return True
            time.sleep(0.05)
        return False

    def _wait_on_road(self) -> bool:
        """AutoDrive drives the car back; wait until it's on a road AND moving."""
        c = self.cfg
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < c.autodrive_timeout_s:
            t = self.rx.latest()
            if (t is not None and t.is_driving and t.speed >= c.on_road_speed
                    and t.mean_surface_rumble < c.on_road_rumble):
                return True
            time.sleep(0.05)
        return False

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
        return self._wait_live(require_speed=c.min_recovered_speed)

    def autodrive_reset(self) -> bool:
        """ANNA AutoDrive back to the middle of the road, then explicitly cancel it so
        the agent has clean control when the episode resumes."""
        c = self.cfg
        start_pos = self._position()
        self.pad.reset()
        self.pad.tap_button(c.autodrive_down_button, hold_s=c.tap_hold_s)
        time.sleep(c.press_gap_s)
        self.pad.tap_button(c.autodrive_left_button, hold_s=c.tap_hold_s)
        time.sleep(c.press_gap_s)
        if not self._confirm_autodrive():            # tap A, retry if it didn't take over
            return False
        if not self._wait_on_road():                 # AutoDrive drives the car back to the road
            return False
        moved = self._displacement(start_pos, self._position())
        if moved < c.autodrive_min_displacement:
            # ANNA can accept the command while still pushing into a guardrail. Treat
            # that as a failed recovery; otherwise the next episode starts wedged.
            self.pad.reset()
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
        return self._wait_live()

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
        return self._wait_live()

    # ---- ladder ----------------------------------------------------------
    def recover(self, reason: str, max_attempts: int = 3) -> str:
        """Run the reset. Returns the recovery method if it worked, else 'FAILED'."""
        for _ in range(max_attempts):
            if reason in ("impact", "stuck"):
                try:
                    if self.rewind():
                        return "rewind"
                except Exception:  # pragma: no cover
                    logger.exception("rewind recovery error")
            for name, fn in (
                ("reset_position", self.reset_position),
                ("reset_to_road", self.reset_to_road),
                ("autodrive", self.autodrive_reset),
            ):
                try:
                    if fn():
                        return name
                except Exception:  # pragma: no cover
                    logger.exception("%s recovery error", name)
            time.sleep(0.5)
        return "FAILED"

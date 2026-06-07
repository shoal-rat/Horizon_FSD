"""
gamepad.py - Horizon FSD, Phase 1

Virtual Xbox 360 controller wrapper around `vgamepad` (backed by the ViGEmBus
kernel driver). Maps the agent's continuous action vector

    action = [steer, throttle, brake]
             steer    in [-1, 1]   (left = -1, right = +1)
             throttle in [ 0, 1]   (right trigger)
             brake    in [ 0, 1]   (left trigger)

onto a virtual controller the game reads as a normal gamepad.

`vgamepad` is imported lazily so this module (and the env) can be imported on a
machine without it installed; you only need it to actually drive the car.

Setup (Windows):
    pip install vgamepad          # runs the bundled ViGEmBus installer (accept UAC/EULA)
    # or install the driver manually from https://github.com/nefarius/ViGEmBus/releases
"""
from __future__ import annotations

import logging
from typing import Sequence

logger = logging.getLogger(__name__)

ACTION_DIM = 3  # [steer, throttle, brake]

_INSTALL_HINT = (
    "vgamepad is required to drive the car. Install it into THIS interpreter with:\n"
    "    <python> -m pip install vgamepad\n"
    "and make sure the ViGEmBus driver is installed "
    "(https://github.com/nefarius/ViGEmBus/releases)."
)

_DRIVER_HINT = (
    "vgamepad is installed, but it could not connect to the ViGEmBus driver "
    "(VIGEM_ERROR_BUS_NOT_FOUND).\n"
    "The driver is most likely not installed/loaded. Run the bundled installer "
    "(accept the UAC prompt), then reboot if asked:\n"
    "    <venv>\\Lib\\site-packages\\vgamepad\\win\\vigem\\install\\x64\\ViGEmBusSetup_x64.msi\n"
    "or download it from https://github.com/nefarius/ViGEmBus/releases"
)


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v


class ForzaGamepad:
    """A virtual Xbox 360 pad. Construct it, call `apply([steer, throttle, brake])`
    once per control step, and `reset()` to release all inputs."""

    def __init__(self) -> None:
        try:
            import vgamepad as vg
        except ImportError as exc:  # pragma: no cover - depends on the host
            raise RuntimeError(_INSTALL_HINT) from exc
        except Exception as exc:  # vgamepad connects to ViGEmBus at IMPORT time
            # e.g. VIGEM_ERROR_BUS_NOT_FOUND when the driver isn't loaded.
            raise RuntimeError(_DRIVER_HINT) from exc

        self._vg = vg
        try:
            self._pad = vg.VX360Gamepad()
        except Exception as exc:  # ViGEmBus missing / not running
            raise RuntimeError(_DRIVER_HINT) from exc
        self.reset()
        logger.info("Virtual Xbox 360 gamepad connected.")

    # ---- main control -----------------------------------------------------
    def apply(self, action: Sequence[float]) -> None:
        """Apply one [steer, throttle, brake] action and push it to the OS."""
        if len(action) != ACTION_DIM:
            raise ValueError(f"action must have {ACTION_DIM} elements, got {len(action)}")
        steer = _clamp(float(action[0]), -1.0, 1.0)
        throttle = _clamp(float(action[1]), 0.0, 1.0)
        brake = _clamp(float(action[2]), 0.0, 1.0)
        self.set(steer, throttle, brake)

    def set(self, steer: float, throttle: float, brake: float) -> None:
        """Set steering / throttle / brake explicitly (already in range)."""
        self._pad.left_joystick_float(x_value_float=steer, y_value_float=0.0)
        self._pad.right_trigger_float(value_float=throttle)
        self._pad.left_trigger_float(value_float=brake)
        self._pad.update()

    def reset(self) -> None:
        """Release everything (centered stick, triggers up)."""
        self._pad.reset()
        self._pad.update()

    # ---- buttons (e.g. to map an in-game 'reset car' bind) ----------------
    def tap_button(self, button_name: str, hold_s: float = 0.08) -> None:
        """Press and release an Xbox button by name, e.g. 'A', 'B', 'X', 'Y',
        'START', 'BACK', 'LEFT_SHOULDER'. Blocks for `hold_s` seconds."""
        import time

        button = getattr(self._vg.XUSB_BUTTON, f"XUSB_GAMEPAD_{button_name.upper()}")
        self._pad.press_button(button=button)
        self._pad.update()
        time.sleep(hold_s)
        self._pad.release_button(button=button)
        self._pad.update()

    # ---- lifecycle --------------------------------------------------------
    def close(self) -> None:
        try:
            self.reset()
        except Exception:  # pragma: no cover
            pass

    def __enter__(self) -> "ForzaGamepad":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

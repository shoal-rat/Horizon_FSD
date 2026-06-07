"""
hitl.py - Horizon FSD

Small human-in-the-loop helpers shared by the interactive scripts. Every program
that captures the screen or sends gamepad/keyboard input gives a startup countdown
so you can alt-tab back into the FH6 window before it starts acting.
"""
from __future__ import annotations

import time


def countdown(seconds: float = 5.0, message: str = "switch to the FH6 window now") -> None:
    """Print a 1 Hz countdown so the user can refocus the game before the program acts."""
    for s in range(int(round(seconds)), 0, -1):
        print(f"  starting in {s}s ... ({message})")
        time.sleep(1.0)

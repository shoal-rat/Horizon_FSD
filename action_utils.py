"""
Action coordinate helpers for the Dreamer-facing Forza RL environment.

Dreamer acts in a symmetric Box([-1, -1, -1], [1, 1, 1]):
    [steer, throttle, brake]

The virtual gamepad expects physical trigger values in [0, 1]. Keeping this
mapping explicit prevents warm-start replay from mixing coordinate systems.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np


def pedal_to_model(value: float) -> float:
    """Physical trigger [0, 1] -> model action [-1, 1]."""
    return float(np.clip(value, 0.0, 1.0) * 2.0 - 1.0)


def pedal_to_physical(value: float) -> float:
    """Model action [-1, 1] -> physical trigger [0, 1]."""
    return float((np.clip(value, -1.0, 1.0) + 1.0) * 0.5)


def physical_to_model_action(action: Sequence[float]) -> np.ndarray:
    """Convert [steer -1..1, throttle 0..1, brake 0..1] to Dreamer coordinates."""
    if len(action) != 3:
        raise ValueError(f"action must have 3 elements, got {len(action)}")
    return np.array([
        float(np.clip(action[0], -1.0, 1.0)),
        pedal_to_model(float(action[1])),
        pedal_to_model(float(action[2])),
    ], dtype=np.float32)


def model_to_physical_action(action: Sequence[float]) -> np.ndarray:
    """Convert Dreamer coordinates to [steer -1..1, throttle 0..1, brake 0..1]."""
    if len(action) != 3:
        raise ValueError(f"action must have 3 elements, got {len(action)}")
    return np.array([
        float(np.clip(action[0], -1.0, 1.0)),
        pedal_to_physical(float(action[1])),
        pedal_to_physical(float(action[2])),
    ], dtype=np.float32)


def exclusive_pedals(throttle: float, brake: float) -> tuple[float, float]:
    """Collapse simultaneous throttle/brake into one net pedal command."""
    net = float(np.clip(throttle, 0.0, 1.0) - np.clip(brake, 0.0, 1.0))
    return max(0.0, net), max(0.0, -net)

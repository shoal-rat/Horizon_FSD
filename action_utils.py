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


def steer_limit_for_speed(speed: float, steer_limit: float = 0.55,
                          high_speed_steer_limit: float = 0.35,
                          high_speed_threshold: float = 15.0) -> float:
    """The speed-dependent steering clamp. ONE definition shared by the live env, the demo
    converter, and any teacher - so the replayed action, the demo target, and what the actuator
    can actually do never drift apart (mismatched action labels corrupt the world model: it
    learns 'steer=+1 caused a 0.35 turn' and saturation becomes free in imagination)."""
    speed = max(0.0, float(speed))
    if high_speed_threshold <= 0.0:
        limit = steer_limit
    else:
        a = min(1.0, speed / high_speed_threshold)
        limit = (1.0 - a) * steer_limit + a * high_speed_steer_limit
    return max(0.05, min(1.0, limit))


def apply_steer_limit(steer: float, speed: float, steer_limit: float = 0.55,
                      high_speed_steer_limit: float = 0.35,
                      high_speed_threshold: float = 15.0) -> float:
    limit = steer_limit_for_speed(speed, steer_limit, high_speed_steer_limit, high_speed_threshold)
    return float(np.clip(steer, -limit, limit))

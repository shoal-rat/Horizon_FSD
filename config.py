"""
config.py - Horizon FSD

Load the central config.yaml. Kept tiny on purpose; returns a plain nested dict.
"""
from __future__ import annotations

import os
from typing import Any, Optional

import yaml

_DEFAULT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")


def load_config(path: Optional[str] = None) -> dict[str, Any]:
    with open(path or _DEFAULT_PATH, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)

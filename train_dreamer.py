"""
train_dreamer.py - Horizon FSD, Phase 5 [HUMAN-IN-THE-LOOP: supervised shakedown / training]

Launch DreamerV3 (NM512) training on the real-time FH6 env. Runs the vendored
dreamer.py with our 'forza' config, after a startup countdown so you can focus the
game. Checkpoints + resumes automatically from --logdir (re-run to continue).

Before running:
  * One-time: build the warm-start replay -> `python make_warmstart.py --logdir <logdir>`
  * FH6: offline Solo, damage None/Cosmetic (rewind), BUMPER cam, parked on a road, focused.
  * The agent WILL drive the car (badly at first) and auto-reset on crashes.

Run:
    # VRAM-tight path: close FH6, warm the checkpoint offline, then re-open FH6.
    .\\.venv\\Scripts\\python.exe offline_pretrain_dreamer.py --updates 200
    .\\.venv\\Scripts\\python.exe train_dreamer.py
    .\\.venv\\Scripts\\python.exe train_dreamer.py --logdir C:\\Horizon_FSD\\dreamer_logs\\forza
"""
from __future__ import annotations

import argparse
import ctypes
import os
import subprocess
import sys

from hitl import countdown

DREAMER_DIR = r"C:\Horizon_FSD\dreamerv3_torch"
HFSD_DIR = r"C:\Horizon_FSD"

# Keep Windows (and the display) awake for the whole run, so an overnight session
# doesn't get killed by sleep/screensaver. SetThreadExecutionState, ES_CONTINUOUS.
_ES_CONTINUOUS = 0x80000000
_ES_SYSTEM_REQUIRED = 0x00000001
_ES_DISPLAY_REQUIRED = 0x00000002


def _keep_awake(on: bool) -> None:
    try:
        flags = _ES_CONTINUOUS
        if on:
            flags |= _ES_SYSTEM_REQUIRED | _ES_DISPLAY_REQUIRED
        ctypes.windll.kernel32.SetThreadExecutionState(flags)
    except Exception:
        pass


def main() -> int:
    p = argparse.ArgumentParser(description="Train DreamerV3 on the FH6 env.")
    p.add_argument("--logdir", default=r"C:\Horizon_FSD\dreamer_logs\forza")
    p.add_argument("--countdown", type=float, default=8.0)
    p.add_argument("extra", nargs="*", help="Extra --key value overrides passed to dreamer.py.")
    args = p.parse_args()

    print("=" * 74)
    print(" Horizon FSD - DreamerV3 training")
    print("=" * 74)
    print(" FH6: offline Solo, damage None/Cosmetic, BUMPER cam, on a road, FOCUSED.")
    print(" The agent will drive (badly at first) and auto-reset on crashes. Ctrl+C to stop;")
    print(" re-run with the same --logdir to resume.")
    print(f" logdir: {args.logdir}")
    countdown(args.countdown, "switch to FH6 - the agent is about to take the wheel")

    cmd = [sys.executable, "dreamer.py", "--configs", "forza", "--logdir", args.logdir, *args.extra]
    env = dict(os.environ, HORIZON_FSD_DIR=HFSD_DIR, HORIZON_FSD_LOGDIR=args.logdir)
    _keep_awake(True)                       # no sleep/screensaver for the whole run
    try:
        return subprocess.run(cmd, cwd=DREAMER_DIR, env=env).returncode
    finally:
        _keep_awake(False)


if __name__ == "__main__":
    raise SystemExit(main())

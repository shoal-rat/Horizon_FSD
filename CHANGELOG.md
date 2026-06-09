# Release Notes

## v0.2.0 - Hardened AutoDrive And Full-Config Preview

This release documents and publishes the hardened pipeline after the large
strategy update.

### Highlights

- Added route-aware terminations: `offroute`, `noprogress`, and `route_complete`.
- Added strange-situation guards for stale telemetry, paused/menu states, frozen
  capture frames, teleport jumps, long GPU-stall ticks, and slow-but-advancing
  motion.
- Reworked AutoDrive recovery around the observed FH6 behavior:
  - far off-road can show a Fast Travel Warning / transfer prompt
  - on-road stuck states usually have no prompt and AutoDrive drives back
  - confirm `A` is only sent while the car is positionally frozen
- Added `forza_full` Dreamer config for larger world-model training when GPU
  headroom is available.
- Added `train_dreamer.py --config` so `forza` and `forza_full` are selectable.
- Added stress tests for NaN/inf handling, malformed packets, teleport jumps,
  route seams, braking-vs-impact, route ends, offroute, noprogress, and stuck
  edge cases.
- Updated the public README and SVG diagrams to match the current stack.

### Validation

```text
python -m unittest discover -s tests -v
python -m py_compile recovery.py recovery_demo.py forza_rl_env.py train_dreamer.py reset_test.py offline_pretrain_dreamer.py
```

At publication time, the suite contains 55 tests.

### Notes

- `forza_full` should only be used after reducing FH6 GPU pressure and verifying
  that VRAM does not spill into shared system memory.
- A waypoint must be pinned for AutoDrive recovery.
- Teleport recoveries keep training alive but are not saved as dynamics demos.

## v0.1.0 - Public Research Preview

Initial public-ready snapshot.

### Highlights

- Real-time Forza Horizon driving environment built from screen capture, Data Out
  telemetry, and virtual Xbox controller input.
- DreamerV3 integration through a reproducible vendor patch for
  `NM512/dreamerv3-torch`.
- Low-VRAM workflow with offline replay pretraining.
- Centerline-aware reward for route progress.
- Crash, stuck, flipped and off-road detection.
- AutoDrive recovery demonstrations for smooth non-teleport recoveries.
- Public documentation and SVG diagrams.

# Release Notes

## v0.1.0 - Public Research Preview

This is the first public-ready snapshot of Horizon FSD.

### Highlights

- Real-time Forza Horizon driving environment built from screen capture,
  Data Out telemetry, and virtual Xbox controller input.
- DreamerV3 integration through a reproducible vendor patch for
  `NM512/dreamerv3-torch`.
- Low-VRAM workflow:
  - close FH6
  - pretrain from replay with `offline_pretrain_dreamer.py`
  - reopen FH6 for live collection
- Centerline-aware reward for route progress.
- Steering and pedal safety guards for early live RL.
- Crash, stuck, flipped and off-road detection.
- Recovery ladder with rewind, ANNA AutoDrive, prompt acceptance, and escalation
  to teleport-style reset when the car is wedged or flipped.
- Smooth non-teleport AutoDrive recoveries are saved as Dreamer replay
  demonstrations.
- Teleport recoveries keep training alive but are discarded as dynamics data.

### Included

- Project code and tests.
- Public documentation.
- SVG diagrams for architecture and recovery flow.
- DreamerV3 patch file.
- Configuration templates.

### Excluded

- FH6 recordings.
- Dreamer replay buffers.
- Trained checkpoints.
- Local `centerline.npy`.
- TensorBoard logs.
- Python virtual environments.
- The vendored `dreamerv3_torch/` checkout.

### Validation

The public preview was validated with:

```text
python -m unittest discover -s tests -v
python -m py_compile recovery.py recovery_demo.py forza_rl_env.py train_dreamer.py reset_test.py offline_pretrain_dreamer.py
```

At publication time, the test suite contained 26 unit tests.

### Known Limitations

- The project is Windows-only.
- Live training must be supervised.
- Capture, route, steering limits and reward weights are hardware/game-state
  dependent.
- No license file is included yet.
- The learned policy is experimental and not expected to generalize broadly from
  one route without more data and curriculum work.

# DreamerV3 Integration

Horizon FSD trains the live Forza environment with the PyTorch
`NM512/dreamerv3-torch` implementation. The upstream Dreamer checkout is
vendored locally in `dreamerv3_torch/`, but that directory is ignored by git.
All Horizon-specific changes are captured in:

```text
patches/dreamerv3_torch_horizon.patch
```

## Restore The Vendor

After cloning this repository:

```powershell
cd C:\Horizon_FSD
git clone https://github.com/NM512/dreamerv3-torch dreamerv3_torch
.\.venv\Scripts\python.exe -m pip install --no-cache-dir gym==0.26.2 ruamel.yaml einops
git -C dreamerv3_torch apply ..\patches\dreamerv3_torch_horizon.patch
```

## What The Patch Adds

The patch makes these changes to upstream Dreamer:

1. Adds `envs/forza.py`, a thin bridge that imports `ForzaDriveEnv` from this
   repository. The path is controlled by `HORIZON_FSD_DIR`.
2. Adds a `forza` branch in `dreamer.py::make_env()`.
3. Adds a `forza:` config block in `configs.yaml`.
4. Updates the ruamel.yaml loader API.
5. Adds tolerant checkpoint loading, so matching tensors are kept and changed
   actor/value/reward-head tensors are reinitialized.
6. Adds optional async training, paced to `train_ratio`, so live control is not
   blocked by every gradient step.
7. Dynamically loads externally written `recovery-*.npz` episodes into the
   in-memory replay cache during live training.
8. Samples replay from a snapshot of the episode dict to avoid concurrent
   modification while recovery demos are added.

## Forza Dreamer Config

The patched `configs.yaml` contains:

```yaml
forza:
  task: forza_drive
  steps: 1e6
  envs: 1
  parallel: False
  precision: 16
  pretrain: 200
  action_repeat: 2
  time_limit: 1200
  grayscale: True
  prefill: 2500
  train_ratio: 4
  compile: False
  video_pred_log: False
  eval_episode_num: 0
  eval_every: 12000
  log_every: 200
  async_train: True
  batch_size: 4
  batch_length: 32
  dyn_hidden: 256
  dyn_deter: 256
  dyn_stoch: 24
  units: 256
  encoder: {mlp_keys: 'speed|line', cnn_keys: 'image', cnn_depth: 16}
  decoder: {mlp_keys: 'speed|line', cnn_keys: 'image', cnn_depth: 16}
```

The model is intentionally small because FH6 and training share one GPU.

## Environment Contract

`forza_rl_env.py` exposes an old-gym style environment compatible with NM512:

```text
reset() -> obs
step(action) -> obs, reward, done, info
```

Observation:

```text
image: uint8, shape (64, 64, 1)
speed: float32, shape (1,)
line:  float32, shape (3,)
```

`line` is `[cue, lateral_offset, confidence]` from the visual racing-line reader.

Action:

```text
Box([-1, -1, -1], [1, 1, 1])
[steer, throttle, brake]
```

Steering is already `[-1, 1]`. Pedals use Dreamer coordinates where `-1` means
released and `+1` means fully pressed. `forza_rl_env.py` maps those to gamepad
trigger values `[0, 1]` and collapses simultaneous throttle/brake into one net
pedal command.

Episode termination:

- impact, stuck, flipped or off-road conditions end the episode
- the next `reset()` runs the recovery ladder
- recovery must be telemetry-verified before the next episode starts

## Training Order

```powershell
# 1. Convert recordings to Dreamer warm-start replay.
.\.venv\Scripts\python.exe make_warmstart.py --logdir C:\Horizon_FSD\dreamer_logs\forza

# 2. VRAM-tight path: close FH6 and pretrain from replay.
.\.venv\Scripts\python.exe offline_pretrain_dreamer.py --updates 200 --logdir C:\Horizon_FSD\dreamer_logs\forza

# 3. Re-open FH6 and run live training.
.\.venv\Scripts\python.exe train_dreamer.py --logdir C:\Horizon_FSD\dreamer_logs\forza
```

`train_dreamer.py` sets:

```text
HORIZON_FSD_DIR=C:\Horizon_FSD
HORIZON_FSD_LOGDIR=<selected logdir>
```

`HORIZON_FSD_LOGDIR` ensures recovery demos are saved into the same
`<logdir>/train_eps` directory that Dreamer is sampling.

## Recovery And Demonstrations

Forza respawn/reset is not treated as authoritative, because it may place the car
on nearby flat ground rather than the route. The recovery ladder verifies:

- Data Out is live
- the car is upright
- surface rumble is low
- when `centerline.npy` exists, the car is close to the route

For off-road or guardrail states, AutoDrive is the main recovery path after any
rewind attempt. The recovery loop handles two cases:

1. FH6 shows a "teleport to nearby road" prompt. The loop taps `A` while telemetry
   is paused or stationary. The teleport is accepted for safety but discarded as
   training data.
2. No prompt appears. AutoDrive drives back toward the pinned route. If the
   coordinates change smoothly, the sequence is saved as `recovery-*.npz`.

If the in-distribution methods keep failing, the ladder escalates to
teleport-style reset methods. Those are accepted as safety recoveries but they
do not create recovery demonstrations.

The saved recovery episode contains:

```text
image, speed, line, action, reward, is_first, is_terminal, discount, logprob
```

Actions are read from Forza telemetry, so they are ANNA's actual
steer/throttle/brake inputs. These episodes let the world model and policy see
examples of recovering from grass, barriers and off-route states without learning
physically impossible teleport transitions.

## Checkpoint Notes

Use `reset_actor.py` when the policy collapses into a bad behavior but the world
model should be kept:

```powershell
.\.venv\Scripts\python.exe reset_actor.py --logdir C:\Horizon_FSD\dreamer_logs\forza
```

After reward changes, add:

```powershell
--reset-reward-head
```

The tolerant Dreamer loader keeps matching world-model tensors and reinitializes
only missing or shape-changed tensors.

## Tuning Knobs

- `dreamerv3_torch/configs.yaml::forza.batch_size`
- `dreamerv3_torch/configs.yaml::forza.train_ratio`
- `config.yaml::rl_safety`
- `config.yaml::rl_reward`
- `config.yaml::reset`
- `config.yaml::recovery_demos`

If FH6 and Dreamer fight for VRAM, prefer offline pretraining with FH6 closed and
lower `batch_size` during live training.

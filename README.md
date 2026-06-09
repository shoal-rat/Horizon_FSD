# Horizon FSD

<p align="center">
  <img src="docs/assets/horizon-fsd-hero.svg" alt="Horizon FSD hero banner" width="100%">
</p>

<p align="center">
  <a href="https://github.com/shoal-rat/Horizon_FSD/releases"><img alt="release" src="https://img.shields.io/github/v/release/shoal-rat/Horizon_FSD?include_prereleases&style=for-the-badge"></a>
  <img alt="platform" src="https://img.shields.io/badge/platform-Windows-2563eb?style=for-the-badge">
  <img alt="Python" src="https://img.shields.io/badge/python-3.10--3.13-0f766e?style=for-the-badge">
  <img alt="mode" src="https://img.shields.io/badge/scope-Offline%20Solo-f59e0b?style=for-the-badge">
  <img alt="tests" src="https://img.shields.io/badge/tests-55%20unit%20%2B%20stress-16a34a?style=for-the-badge">
</p>

Horizon FSD runs a local self-driving experiment in Forza Horizon 6. It uses
Windows screen capture for images, Forza Data Out UDP for telemetry, and a
virtual Xbox controller for steering, throttle, and brake.

The project does not read game memory, patch the game, or inject code. It is
meant for Offline Solo / Free Roam only.

The RL path uses DreamerV3. Manual driving, AutoDrive driving, and smooth
AutoDrive recoveries can be converted into replay episodes. Large local files
such as recordings, replay buffers, checkpoints, `centerline.npy`, TensorBoard
logs, `.venv`, and `dreamerv3_torch/` are not stored in this repo.

## Links

- [DreamerV3 integration](docs/dreamer_integration.md)
- [Recovery mechanics](docs/recovery_mechanics.md)
- [Performance and hardening guide](docs/performance_and_hardening.md)
- [Driving RL notes](docs/driving_rl_lessons.md)
- [Telemetry format](docs/telemetry_format.md)
- [Release notes](CHANGELOG.md)

## Scope

- Run in Offline Solo / Free Roam.
- Do not use automated input online, in races, in competitive modes, or for
  leaderboards.
- Supervise live training. A fresh policy will drive badly.
- Follow the game EULA and Code of Conduct.

## Current State

- Windows.Graphics.Capture screen input.
- 324-byte Forza Data Out parser with finite-value checks.
- Virtual Xbox 360 controller output through `vgamepad` / ViGEmBus.
- DreamerV3 environment with `image`, `speed`, and `line` observations.
- Centerline progress reward with action and safety penalties.
- Episode endings for impact, stuck, flipped, offroad, offroute, noprogress,
  route_complete, telemetry_lost, paused, and frame_lost.
- AutoDrive recovery that handles both FH6 cases:
  - far off road: FH6 may show a Fast Travel Warning / transfer prompt; the code
    presses `A` only while the car is positionally frozen
  - no prompt: AutoDrive is left alone so it can drive back toward the waypoint
- Smooth AutoDrive recoveries are saved as `recovery-*.npz` replay episodes.
- Teleport jumps keep the run alive but are not saved as dynamics data.
- `forza` and `forza_full` Dreamer configs.
- 55 unit and stress tests.

## Diagrams

<p align="center">
  <img src="docs/assets/architecture.svg" alt="Horizon FSD architecture diagram" width="100%">
</p>

<p align="center">
  <img src="docs/assets/hardening-stack.svg" alt="Horizon FSD hardening stack" width="100%">
</p>

## Architecture

```text
FH6 screen + Data Out UDP
        |
        v
capture.py + telemetry_receiver.py + forza_telemetry.py
        |
        v
forza_rl_env.py
  obs: image, speed, line
  action: steer, throttle, brake
  reward: centerline progress plus penalties
  done: detector reason or benign halt
        |
        v
DreamerV3 actor + async learner
        |
        v
gamepad.py -> virtual Xbox controller -> FH6
```

Recovery data enters replay only when it is smooth:

```text
detector -> recovery.py -> ANNA AutoDrive
                      |-> smooth drive-back -> recovery-*.npz
                      |-> teleport jump     -> safety only
```

## Repository Layout

```text
Horizon_FSD/
  action_utils.py                 Dreamer <-> gamepad action mapping
  build_centerline.py             Build route centerline from recorded position
  capture.py                      Screen capture wrapper and frame age
  centerline.py                   Route projection and arc-length utilities
  config.yaml                     Runtime configuration
  dataset.py                      Recording filters and BC dataset builder
  forza_rl_env.py                 DreamerV3 live FH6 environment
  forza_telemetry.py              Forza Data Out parser
  gamepad.py                      Virtual Xbox controller wrapper
  make_warmstart.py               Recordings -> Dreamer replay
  offline_pretrain_dreamer.py     Replay training without FH6 running
  racing_line.py                  Racing-line cue reader
  recovery.py                     Detector and AutoDrive recovery
  recovery_demo.py                Save smooth recoveries as replay
  reward.py                       Driving reward
  train_dreamer.py                Live Dreamer launcher
  docs/
  tests/
```

## Requirements

- Windows 10/11.
- Forza Horizon 6 for PC with Data Out enabled.
- Python 3.10-3.13. The local development machine used Python 3.13.
- NVIDIA GPU recommended for DreamerV3.
- ViGEmBus for virtual controller input.

Install the base environment:

```powershell
cd C:\Horizon_FSD
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install --no-cache-dir -r requirements.txt
```

Install CUDA PyTorch before packages that depend on `torch` / `torchvision`:

```powershell
.\.venv\Scripts\python.exe -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu126
.\.venv\Scripts\python.exe -m pip install timm==1.0.27 tensorboard==2.20.0
```

If `vgamepad` cannot connect, install ViGEmBus from the bundled package path or
from `https://github.com/nefarius/ViGEmBus/releases`.

## FH6 Setup

```text
Settings -> HUD and Gameplay -> Data Out
Data Out: On
IP Address: 127.0.0.1
Port: 9999
```

Use a stable camera view. Pin a waypoint before training if you want ANNA
AutoDrive recovery to work.

## Restore DreamerV3

`dreamerv3_torch/` is ignored. Recreate it after cloning:

```powershell
cd C:\Horizon_FSD
git clone https://github.com/NM512/dreamerv3-torch dreamerv3_torch
.\.venv\Scripts\python.exe -m pip install --no-cache-dir gym==0.26.2 ruamel.yaml einops
git -C dreamerv3_torch apply ..\patches\dreamerv3_torch_horizon.patch
```

The patch adds the Forza environment bridge, async training, tolerant checkpoint
loading, the `forza` and `forza_full` configs, recovery-demo loading, train-lock
checkpoint snapshots, and CUDA benchmark settings.

## Validation

Tests that do not require the game:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Live checks:

```powershell
.\.venv\Scripts\python.exe telemetry_probe.py
.\.venv\Scripts\python.exe sweep_gamepad.py
.\.venv\Scripts\python.exe capture_preview.py
.\.venv\Scripts\python.exe reset_test.py --duration 600
```

## Training

1. Record driving.

```powershell
.\.venv\Scripts\python.exe record.py --duration 600
.\.venv\Scripts\python.exe record.py --autodrive --duration 600
```

2. Build a centerline from a clean route recording.

```powershell
.\.venv\Scripts\python.exe build_centerline.py --session C:\Horizon_FSD\recordings\manual_YYYYMMDD_HHMMSS --out C:\Horizon_FSD\centerline.npy
```

3. Convert recordings to Dreamer replay.

```powershell
.\.venv\Scripts\python.exe make_warmstart.py --logdir C:\Horizon_FSD\dreamer_logs\forza
```

4. If VRAM is tight, close FH6 and pretrain from replay.

```powershell
.\.venv\Scripts\python.exe offline_pretrain_dreamer.py --updates 200 --logdir C:\Horizon_FSD\dreamer_logs\forza
```

5. Run live training.

```powershell
.\.venv\Scripts\python.exe train_dreamer.py --config forza --logdir C:\Horizon_FSD\dreamer_logs\forza
```

Use `forza_full` only after checking GPU memory:

```powershell
.\.venv\Scripts\python.exe train_dreamer.py --config forza_full --logdir C:\Horizon_FSD\dreamer_logs\forza
```

See [performance_and_hardening.md](docs/performance_and_hardening.md) before a
long `forza_full` run.

## Recovery

<p align="center">
  <img src="docs/assets/recovery-loop.svg" alt="AutoDrive recovery and learning loop" width="100%">
</p>

FH6 reset does not always put the car back on the route. This project treats
AutoDrive as the main recovery path.

- If FH6 shows the Fast Travel Warning / transfer prompt, recovery confirms it
  with `A` only while the car is frozen.
- If no prompt appears, recovery waits for AutoDrive to drive back.
- If AutoDrive leaves the car upright on a drivable parallel road, training can
  resume instead of forcing another reset.
- If all automated options fail, the pad is neutralized and the run waits for a
  drivable state. The learner can keep training from replay while the live env
  is paused.

## Config Blocks

- `telemetry`: Data Out freshness and resume timeouts.
- `capture`: capture source and frozen-frame threshold.
- `detector`: impact, offroad, offroute, noprogress, stuck, and teleport rules.
- `rl_reward`: centerline progress and penalty weights.
- `rl_safety`: steering clamps and post-recovery grace.
- `reset`: AutoDrive prompt timing, retry, backoff, and fallback behavior.
- `recovery_demos`: smooth recovery replay recording.

## Git-Ignored Local Files

- game recordings and replay episodes
- trained checkpoints
- `centerline.npy`
- TensorBoard logs
- `dreamerv3_torch/`
- Python virtual environments

No license file is included yet. Until one is added, standard GitHub default
copyright rules apply.

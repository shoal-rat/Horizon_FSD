# Horizon FSD

End-to-end, vision-based self-driving agent for **Forza Horizon 6** (PC, native
Windows). The agent learns to drive from screen pixels + UDP telemetry, outputs
virtual-gamepad inputs, and improves via imitation learning then reinforcement
learning.

> **Status: Phase 1 complete.** Telemetry format confirmed live (324-byte "Car Dash"
> @ ~60 Hz); telemetry parser, virtual gamepad, screen capture, and the rtgym
> real-time env are built and verified (20 Hz control loop). Next: Phase 2 data
> recording for behavioral cloning.

---

## How this project is built (human-in-the-loop)

Code is written and unit-tested without the game; **anything that needs the live
game is run by you**, the human. The build proceeds phase by phase, and **stops**
at every step that depends on FH6 actually running. You paste back results; the
build continues.

## ⚠️ Safety / Terms of Service

- **Offline Solo / Free Roam only.** Run the agent with Horizon Life / online
  disconnected. Community consensus is that automation or injected input while
  **online** triggers a permanent anti-cheat ban; offline solo play does not.
- Do **not** run the agent in online, competitive, or leaderboard modes.
- Even offline, automated input is technically against most game ToS and is done
  at your own risk. Read the FH6 EULA / Code of Conduct.

## Architecture (planned)

A `gymnasium` real-time environment (rtgym-style, mirroring **tmrl**'s TrackMania
pipeline) wraps three I/O channels:

- **Observation** - screen capture (`bettercam`): a stack of the last 4 grayscale
  84x84 frames + a scalar vector `[speed, last_action]`.
- **State / reward** - UDP "Data Out" telemetry (speed, position, inputs, ...).
- **Action** - virtual Xbox gamepad (`vgamepad` + ViGEmBus): continuous
  `[steer in -1..1, throttle in 0..1, brake in 0..1]`.
  DreamerV3 replay/policy uses symmetric pedal coordinates instead
  (`throttle/brake -1..1`, mapped to gamepad triggers inside `forza_rl_env.py`).

Training: behavioral cloning on your manual driving → SAC fine-tune
(`stable-baselines3`) warm-started from BC → (stretch) DreamerV3 world model.

## Repo layout

```
Horizon_FSD/
├── README.md              # this file
├── requirements.txt       # pinned deps for Phases 1-4 (Phase 0 needs none)
├── config.yaml            # central config (ports, capture, env, paths)
├── telemetry_probe.py     # Phase 0: confirm the live FH6 packet layout (stdlib)
├── forza_telemetry.py     # Phase 1: 324-byte "Car Dash" parser -> ForzaTelemetry
├── telemetry_receiver.py  # Phase 1: background UDP listener -> latest telemetry
├── capture.py             # Phase 1: screen capture (windows-capture / WGC)
├── gamepad.py             # Phase 1: virtual Xbox pad (vgamepad + ViGEmBus)
├── sweep_gamepad.py       # Phase 1: [human test] confirm the car responds
├── reward.py              # Phase 1: swappable reward functions
├── config.py              # Phase 1: config.yaml loader
├── forza_env.py           # Phase 1: rtgym real-time env (capture+telemetry+gamepad)
├── random_agent.py        # Phase 1: [human test] random-action loop + FPS
├── capture_preview.py     # Phase 1: [human test] save a frame to verify capture
├── hitl.py                # shared human-in-the-loop helpers (startup countdown)
├── record.py              # Phase 2: [you drive] log frames+telemetry+actions
├── dataset.py             # Phase 2: filter/balance recordings -> BC dataset
├── bc_model.py            # Phase 3: BCPolicy (timm backbone + classification steer head)
├── train_bc.py            # Phase 3: train behavioral cloning
├── run_policy.py          # Phase 3: [human test] drive with a BC checkpoint
├── recovery.py            # Phase 5: crash detector + reset ladder (rewind/reset-to-road)
├── reset_test.py          # Phase 5: [human test] validate the auto-reset
├── forza_rl_env.py        # Phase 5: real-time DreamerV3 env (dict obs, action_repeat)
├── make_warmstart.py      # Phase 5: recordings -> DreamerV3 warm-start episodes
├── train_dreamer.py       # Phase 5: [you supervise] launch DreamerV3 training
├── dreamerv3_torch/       # vendored NM512/dreamerv3-torch (gitignored; see docs)
├── tests/
│   └── test_forza_telemetry.py
└── docs/
    ├── telemetry_format.md   # the FH "Car Dash" byte layout (confirmed live)
    └── dreamer_integration.md# DreamerV3 setup + the vendored-repo edits
```

## Phase roadmap

| Phase | Deliverable | Status |
|------:|-------------|------------|
| **0** ✅ | repo scaffold + `telemetry_probe.py` | confirmed 324-byte format @ 60 Hz |
| **1** ✅ | telemetry parser + gamepad + capture + rtgym env | gamepad drives the car; 20 Hz loop |
| **2** ✅ | `record.py` + `dataset.py` | ~75 min recorded (manual + AutoDrive) |
| **3** ✅ | behavioral cloning | trained; hit the covariate-shift ceiling (BC is a warm start, not the driver) |
| ~~4~~ | ~~SAC~~ | skipped — went straight to the world model |
| **5** 🔧 | **DreamerV3 world-model RL** | env + reward + reset ladder + warm-start built; **supervised shakedown next** |

> AutoDrive's controls land in the telemetry, so it doubles as a scalable data
> engine. Recovery now prefers FH6 rewind + ANNA AutoDrive; smooth non-teleport
> AutoDrive recoveries are saved as replay demos. See `docs/dreamer_integration.md`.

---

## ▶ Phase 0 - run the telemetry probe

The probe is **pure Python standard library** - no `pip install` needed. Any
Python 3.8+ works, including your system Python 3.13.

### 1. Enable Data Out in FH6

In-game: **Settings → HUD and Gameplay → Data Out**

- **Data Out:** On
- **IP Address:** `127.0.0.1`
- **Port:** `9999`  *(any port outside 5200-5300; must match the probe)*

### 2. Run the probe

From `C:\Horizon_FSD`:

```powershell
python telemetry_probe.py
```

Optionally save the first raw packet so we have ground truth:

```powershell
python telemetry_probe.py --save-raw first_packet.bin
```

If you picked a different in-game port, pass it: `python telemetry_probe.py --port 5606`.

> If Windows Firewall prompts, **allow** Python on private networks. If you see no
> packets, the probe prints troubleshooting hints after ~6 seconds.

### 3. Drive

Get into a car and **drive manually for ~30 seconds** (steer left/right, accelerate,
brake, change gear). Telemetry is only sent while actively driving.

### 4. Stop and paste back

Press **Ctrl+C**. Copy back to me:

1. The whole **SUMMARY** block (especially the **distinct packet lengths**).
2. The full **hexdump + decoded fields** of the first packet or two.
3. Whether the decoded **Speed / Gear / Throttle / Steer** values looked correct
   for what you were doing while driving.

That tells us whether the FH5 hypothesis (324-byte "Car Dash", `PositionX` @244)
holds for FH6. **Then I build the Phase 1 parser against the confirmed layout.**

---

## Environment setup (already done on the dev machine)

Python 3.13 venv at `.venv`. PowerShell `Activate.ps1` may be blocked by execution
policy — just call the venv Python directly. Install with `--no-cache-dir` to avoid
a pip cache-permission error seen on this machine:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --no-cache-dir -r requirements.txt
# GPU torch instead of CPU (Phase 3+):
#   .\.venv\Scripts\python.exe -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu126
```

**ViGEmBus driver (for the virtual gamepad):** pip's *wheel* install of `vgamepad`
skips the bundled driver installer, so run it manually once (accept the UAC prompt):
`.\.venv\Scripts\...\vgamepad\win\vigem\install\x64\ViGEmBusSetup_x64.msi`, or get it
from <https://github.com/nefarius/ViGEmBus/releases>.

**Screen capture = `windows-capture` (WGC), not dxcam/bettercam.** On this NVIDIA
Optimus + HDR laptop, DXGI Desktop Duplication returns all-black frames and crashes
comtypes on Python 3.13. `windows-capture` handles HDR + hybrid GPUs.

## Phase 1 run commands

```powershell
# Telemetry parser unit tests (no game needed)
.\.venv\Scripts\python.exe tests\test_forza_telemetry.py

# [human] confirm the virtual gamepad drives the car (FH6 focused, parked)
.\.venv\Scripts\python.exe sweep_gamepad.py

# [human] random-action loop: drives the car, measures the 20 Hz control rate
.\.venv\Scripts\python.exe random_agent.py --duration 30
```

If capture grabs the wrong screen, set `capture.monitor_index` in `config.yaml`, or
`capture.window_name: "Forza Horizon 6"` to capture the game window directly. The
rtgym "time-step timed out" warnings during the first few steps are benign.

## Phase 2 run commands

```powershell
# [you drive] record manual driving (FH6 focused). Ctrl+C to stop, or --duration SECONDS.
.\.venv\Scripts\python.exe record.py --duration 600
# bulk low-quality ANNA AutoDrive data (filtered hard later); supplement only:
.\.venv\Scripts\python.exe record.py --autodrive --duration 600
```

Shards land in `recordings/<quality>_<timestamp>/`. Record variety (highway, town,
twisty mountain, dirt/off-road; different speeds, biomes, weather, cars) and DRIVE
WELL — behavioral cloning's ceiling is the quality of the demonstrations.

DreamerV3 (Phase 5) needs a **separate** venv (`sheeprl==0.5.7` requires gymnasium
0.29.* and Python ≤3.11), incompatible with the main stack.

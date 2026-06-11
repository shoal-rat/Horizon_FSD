# Horizon FSD

<p align="center">
  <img src="docs/assets/horizon-fsd-hero.svg" alt="Horizon FSD hero banner" width="100%">
</p>

<p align="center">
  <a href="https://github.com/shoal-rat/Horizon_FSD/releases"><img alt="release" src="https://img.shields.io/github/v/release/shoal-rat/Horizon_FSD?include_prereleases&style=for-the-badge"></a>
  <img alt="platform" src="https://img.shields.io/badge/platform-Windows-2563eb?style=for-the-badge">
  <img alt="Python" src="https://img.shields.io/badge/python-3.10--3.13-0f766e?style=for-the-badge">
  <img alt="mode" src="https://img.shields.io/badge/scope-Offline%20Solo-f59e0b?style=for-the-badge">
  <img alt="tests" src="https://img.shields.io/badge/tests-60%2B%20unit%20%2B%20stress-16a34a?style=for-the-badge">
</p>

Horizon FSD is a world-model RL agent (DreamerV3) that drives Forza Horizon 6 live at
~10 Hz on a single Windows laptop whose GPU is **shared with the running game** (8 GB
RTX 4070 Laptop). Inputs are Windows screen capture and Forza's Data Out UDP telemetry;
output is a virtual Xbox controller. No game memory reading, no patching, no injection.
Offline Solo / Free Roam only.

Design constraints that shape everything here: a ~100 ms decision budget, VRAM shared
with the game, a human in the loop for the game-side steps, and a single 2.76 km
reference route with day/night/snow conditions.

## Links

- [DreamerV3 integration](docs/dreamer_integration.md)
- [Recovery mechanics](docs/recovery_mechanics.md)
- [Performance and hardening guide](docs/performance_and_hardening.md)
- [Driving RL notes](docs/driving_rl_lessons.md)
- [Telemetry format](docs/telemetry_format.md)
- [Release notes](CHANGELOG.md)

## Scope

- Run in Offline Solo / Free Roam.
- Do not use automated input online, in races, in competitive modes, or for leaderboards.
- Supervise live training. A fresh policy will drive badly.
- Follow the game EULA and Code of Conduct.

## Architecture

```text
FH6 (chase cam) ----- screen ----> capture.py (WGC, color full-res)
   |                                  |-- racing_line.py --> line(3)  [day/night-adaptive HSV cue]
   |                                  '-- preprocess -----> image 64x64x3 COLOR (the chevrons keep
   |                                      their semantics in the obs - the model sees the line itself)
   '-- Data Out UDP -> telemetry_receiver.py / forza_telemetry.py (finite-checked, freshness-clocked)
                                      |-- speed(1)
                                      '-- centerline.py route_features --> route(27)
                                          [signed cross-track, heading error vs route tangent,
                                           prev APPLIED action, 10-point lookahead road preview -
                                           light-invariant: works identically at night]
            obs {image, speed, line, route}
                        |
                        v
            DreamerV3 RSSM world model + actor-critic trained in imagination
            (async background learner paced to train_ratio; BC loss on human demos)
                        |
                  action [steer, throttle, brake]
                        v
            safety layer: speed-dependent steer clamp + exclusive pedals
            (ONE shared definition: live actuator, replay storage, demo targets)
                        |
                        v
            gamepad.py (ViGEm virtual Xbox pad, atexit-neutralized) ----> FH6

side channels:
  recovery.py   CrashDetector (impact/stuck/flipped/offroad/offroute/noprogress/route_complete,
                sliding-window wedge detection, teleport/overrun guards, episode runway)
                + ANNA AutoDrive recovery (teleport-when-far / drive-when-stuck) with a
                pause-menu Reset Car Position fallback and a hard time budget
  replay        train_eps/ (live episodes store the APPLIED action) + ws-* human demos
                + wsx-* quality-demoted sessions + recovery_eps/ AutoDrive demos
  audit trail   reasons.jsonl (per-episode reason/length/return/mean-steer + collapse alarm),
                provenance.jsonl (which checkpoint actually drove)
```

## Why the reward looks the way it does

Every term exists because a logged failure forced it. The reward is
`progress + align + boot + launch - penalties`, all telemetry-based so it works in the dark:

| Term | What | The failure that motivated it |
|---|---|---|
| `progress` | Per-step arc-length advance along `centerline.npy` | Speed-based progress was farmed by driving in circles; arc-length is circle-proof. Validated by rare good episodes (+53.7, +131.6) earning large positive returns. |
| `align` | `velocity · route tangent` (signed, ±1) | The agent locked the wheel full-left (later full-right) at night: the earlier unsigned centering penalty told it *how far* off the line it was but not *which way to steer*. Both vectors are world-frame, so the sign needs no yaw convention. |
| `cross` | small penalty ∝ signed cross-track magnitude | Pulls toward the line; capped at the same scale as `align`. |
| `boot`/`launch` | small forward-speed/launch nudges, gated on real forward motion (`ds > 0`) | A cold actor needs a dense gradient toward "move"; the gate stops reverse-farming. |
| `offroad`/`slip` | rumble/slip penalties, **capped at `penalty_cap`** | Uncapped telemetry hit −32/step (worth six crash penalties), making "crash fast" cheaper than "keep driving" and stretching the reward scale until the reward head treated demos and live data as different worlds. |
| `spin`/`jerk`/`steer`/`brake`/`idle` | small shaping terms | Anti-circling, smoothness, anti-idle. Deliberately 1–2 orders below the main terms (a jerk_w=1.0 once taught the agent to crash immediately — a still car outscored a moving one). |
| `crash_penalty` | one-off terminal | The dominant negative, by design — no per-step penalty may exceed it. |

Reward-scale discipline: **one scale in one buffer.** Demos, recovery demos, and live
episodes are all labeled by the same `DriveReward`; changing the reward requires
`reset_actor.py --reset-reward-head` (purges all labeled episodes) + a rebuild.

## The training pipeline

```text
1. record.py        you drive (ANALOG gamepad, chase cam) -> 320x180 JPEG COLOR source
                    frames @20 Hz + full telemetry incl. world position. Sessions are
                    quality-gated (a bang-bang/full-lock steering check aborts keyboard play).
                    ANNA AutoDrive sessions (record.py --autodrive, AFK) are valid data too:
                    telemetry logs ANNA's real steering/throttle/brake. By default they feed
                    the world model + reward head only (wsx-*); --autodrive-as-demo opts them
                    into the policy teachers.
2. make_warmstart   recordings -> Dreamer replay @10 Hz (stride-2: demos and live MUST
                    share one timescale), actions actuator-clamped (the same steer limit
                    the live env applies), rewards accumulated per decision window exactly
                    like the live env. Clean manual sessions -> ws-* (policy teachers);
                    autodrive / bang-bang sessions -> wsx-* (world-model food only).
3. offline pretrain world model + BEHAVIORAL CLONING on the actor (demo latents -> human
                    action; per-dim weighting up-weights steer). First N updates are
                    BC-only on the actor (no imagination REINFORCE thrashing a random
                    actor's optimizer). Held-out bc_holdout is the convergence signal.
4. live training    train_dreamer.py: gated shakedown first (~20 episodes vs the previous
                    run's median return/length), then longer runs. Replay stores the action
                    the car ACTUALLY executed; demos are oversampled to ~50% of batches so
                    imagination doesn't start from crash states.
```

## What broke and what fixed it (the failure-mode log)

This project's history is the documentation. Each failure is real, logged, and has a fix
with a test:

1. **Steering collapse (constant full-lock turn, both directions across runs).** Causes,
   compounding: no directional steering signal at night; 73.5% of replayed live actions
   were the RAW policy sample, not what the clamped actuator did (so saturation looked
   free to the world model); bang-bang keyboard demos taught full-lock as normal; BC never
   reached the actor at all. Fixes: signed `align` reward, applied-action storage, demo
   quality gates, the BC pretrain, and a live **collapse alarm** (mean applied steer over
   the last 20 episodes) in `reasons.jsonl`.
2. **The −4,037-return wedge.** A car wedged at 87% throttle creeping 0.22 m/s evaded
   `stuck` (fixed anchor drifts), `hard_stuck` (was rumble-gated), and `noprogress`
   (exempted <3 m/s) for 600 steps — outweighing ~35 good episodes. Fixes: sliding-window
   displacement, rumble gate removed, a wedge-grind catch-all; plus per-step penalty caps
   so grinding can never be reward-cheaper than driving.
3. **1-second episodes.** Median episode was 27 steps; 29% were ≤15 — the buffer was
   wall-to-wall episode-starts. Fixes: per-episode runway (~20 decisions with
   offroute/noprogress/stuck suppressed; the wedge window is longer, so the hole stays
   closed), post-recovery grace, and route-end `route_complete` (benign, no crash penalty).
4. **AutoDrive dependence / "doesn't know the road".** The agent had no road-geometry
   input at all (image + speed only at night). Fix: the `route(27)` vector — signed
   cross-track, heading error, and a 10-point lookahead preview of the upcoming road,
   from telemetry + the centerline, identical day or night. Recovery itself requires a
   **pinned route waypoint** in FH6 or ANNA has nothing to drive toward.
5. **Phantom data.** NaN/garbage UDP frames, frozen capture (WGC only fires on change),
   stale telemetry after alt-tab, fast-travel teleports, GPU-stall long ticks — each got
   a guard (finite-check at parse, freshness clocks, teleport/overrun detector guards,
   benign `paused`/`frame_lost`/`telemetry_lost` halts that wait instead of teleporting).
6. **Run-provenance confusion.** A whole run's conclusions were once drawn from a
   checkpoint that finished pretraining *after* the run ended. Fix: `provenance.jsonl`
   records the checkpoint identity at every launch; `reasons.jsonl` makes the termination
   distribution auditable.

**Standing triggers** (decided once — do not relitigate without them firing):
- *Stanley scripted teacher + residual steering*: implement only if episodes still
  collapse to constant steer after the label/detector/reward fixes plus a live-tested BC
  checkpoint.
- *Model family*: reconsider DreamerV3 only if episodes still die early with well-fit
  world-model losses after the fixes above. (TD-MPC2 fails the 100 ms budget on a shared
  GPU; VLA models don't fit 8 GB; diffusion policies can't improve online.)
- *96×96 obs*: only in `forza_full` on a freed GPU, and only if decoder reconstructions
  show the road edge unresolved AND offroute deaths cluster at high-speed sweepers.

## Day/night — one strategy, in color

The obs is **64×64×3 color** (`capture.grayscale: false`). Grayscale physically erased
the chevron semantics (blue → luma 29, red → 76, both inside the asphalt's 38–144 range),
which is what forced the old day-only stopgaps. In color the model sees the racing line
*in the image itself*, day and night — the night chevrons are headlight-lit and clearly
visible in color captures.

Three redundant signals, all day/night-capable (redundancy is how GT Sophy-class systems
work):
- **image (color)** — the line, the road, the scene, as the model sees it;
- **line(3)** — the full-res HSV chevron read (higher-res than the 64×64 obs sees), with
  the night mode that drops snow-prone blue and trusts the warm cues;
- **route(27)** + the whole reward — pure telemetry, identical in the dark.

One consequence, by design: demos must be color. `make_warmstart` *skips* legacy gray
sessions rather than shipping a replicated-gray corpus (a perfect demo/live discriminator
for the reward head). The recorder stores 320×180 color source frames, and demo episodes
get their `line(3)` **backfilled with the same reader the live env runs** — demo and live
channels match exactly, in any light.

## Repository layout

```text
Horizon_FSD/
  action_utils.py                 Dreamer <-> gamepad action mapping + the ONE steer-limit definition
  build_centerline.py             Reference lap -> centerline.npy
  capture.py                      WGC capture, frame freshness, color/gray preprocess
  centerline.py                   Route projection, signed cross-track, lookahead, route_features
  config.yaml                     Runtime configuration (obs/reward/detector/recovery/safety)
  dataset.py                      Recording loaders (JPEG-color + legacy gray), quality filters
  forza_rl_env.py                 The live FH6 env: obs/reward/termination/runway/benign halts
  forza_telemetry.py              324-byte Data Out parser (finite-checked)
  gamepad.py                      ViGEm virtual pad (atexit-neutralized)
  make_warmstart.py               Recordings -> replay (stride-2, clamps, ws-/wsx- quality split)
  offline_pretrain_dreamer.py     World model + BC pretrain (warmup, held-out bc eval)
  racing_line.py                  Day/night-adaptive racing-line cue
  record.py                       Session recorder (320x180 JPEG color sources, bang-bang abort)
  recovery.py                     CrashDetector + ANNA AutoDrive recovery ladder
  recovery_demo.py                Saves smooth recoveries -> recovery_eps/
  reset_actor.py                  Reset policy/reward-head + purge stale-labeled episodes
  train_dreamer.py                Live launcher (provenance logging, keep-awake)
  patches/dreamerv3_torch_horizon.patch   All vendored-Dreamer edits
  docs/  tests/
```

## Requirements & install

- Windows 10/11, Forza Horizon 6 (PC) with Data Out enabled, Python 3.10–3.13,
  NVIDIA GPU, ViGEmBus, **an analog gamepad for recording** (keyboard demos are
  rejected by the recorder — they teach the steering collapse).

```powershell
cd C:\Horizon_FSD
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install --no-cache-dir -r requirements.txt
.\.venv\Scripts\python.exe -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu126
.\.venv\Scripts\python.exe -m pip install timm==1.0.27 tensorboard==2.20.0
```

Restore the vendored DreamerV3 (ignored by git):

```powershell
git clone https://github.com/NM512/dreamerv3-torch dreamerv3_torch
.\.venv\Scripts\python.exe -m pip install --no-cache-dir gym==0.26.2 ruamel.yaml einops
git -C dreamerv3_torch apply ..\patches\dreamerv3_torch_horizon.patch
```

FH6 settings: `HUD and Gameplay -> Data Out: On, 127.0.0.1:9999`. **Chase cam**
(the line reader's ROI is calibrated for it). Offline Solo, damage None/Cosmetic.
**Pin a route waypoint** — recovery is blind without it.

## Quickstart

```powershell
# sanity (no game needed)
.\.venv\Scripts\python.exe -m unittest discover -s tests

# live checks (game open)
.\.venv\Scripts\python.exe telemetry_probe.py        # telemetry flowing? pause -> does is_race_on flip?
.\.venv\Scripts\python.exe capture_preview.py        # capture sees the game?
.\.venv\Scripts\python.exe racing_line_preview.py    # line ROI calibrated for YOUR cam?

# pipeline
.\.venv\Scripts\python.exe record.py --tag "day fwd laps"      # you drive (analog pad, chase cam)
.\.venv\Scripts\python.exe build_centerline.py --session recordings\manual_... --out centerline.npy
.\.venv\Scripts\python.exe make_warmstart.py
.\.venv\Scripts\python.exe offline_pretrain_dreamer.py --updates 2000   # game closed; BC included
.\.venv\Scripts\python.exe train_dreamer.py                              # game open, waypoint pinned
```

After any reward change: `reset_actor.py --reset-reward-head` -> `make_warmstart.py` ->
re-pretrain. Watch `reasons.jsonl` for the termination mix and the collapse alarm.

## Performance

The lean `forza` config shares the 8 GB GPU with FH6. To grow (`forza_full`:
`train_ratio 64`, `batch 16x48`, `dyn_deter 1024`, `cnn_depth 32`): cap FH6 to 1080p
Low/60 fps, move the desktop/capture to the iGPU (NOT the game), verify
`nvidia-smi` < ~6.5 GB, then `train_dreamer.py --config forza_full`. Details and the
validation protocol: [performance_and_hardening.md](docs/performance_and_hardening.md).
"More frames" means `train_ratio`, not capture FPS — at 10 Hz control, faster capture
adds nothing.

# DreamerV3 integration (Phase 5)

We train a **DreamerV3 world-model RL agent** on the real-time FH6 env using
**NM512/dreamerv3-torch** (pure PyTorch, runs in our py3.13 + torch-2.11-cu126 venv).

## Why this implementation
- sheeprl pins gymnasium 0.29 / py≤3.11 (conflicts with our stack) → rejected.
- danijar/dreamerv3 is JAX (painful on Windows) → rejected.
- NM512 is pure PyTorch, not gymnasium-version-locked, uses old `gym` (0.26.2 installs
  fine on 3.13). Its core (`tools`/`networks`/`models`) imports and trains on our stack.

## Vendored repo (gitignored)
`dreamerv3_torch/` is `git clone https://github.com/NM512/dreamerv3-torch` — **not**
tracked by our git. After a fresh clone, re-apply the 3 edits below, plus:
```
.\.venv\Scripts\python.exe -m pip install --no-cache-dir gym==0.26.2 ruamel.yaml einops
git -C dreamerv3_torch apply ..\patches\dreamerv3_torch_horizon.patch
```

### Edit 1 - add the env bridge `dreamerv3_torch/envs/forza.py`
```python
import os, sys
_HFSD = os.environ.get("HORIZON_FSD_DIR", r"C:\Horizon_FSD")
if _HFSD not in sys.path:
    sys.path.insert(0, _HFSD)
from forza_rl_env import ForzaDriveEnv  # noqa
```

### Edit 2 - add a `forza` branch in `dreamerv3_torch/dreamer.py` `make_env()`
Immediately before `else: raise NotImplementedError(suite)`:
```python
    elif suite == "forza":
        import envs.forza as forza
        env = forza.ForzaDriveEnv(task, config.action_repeat, config.size, mode)
        env = wrappers.NormalizeActions(env)
```

### Edit 3 - append the `forza` config block to `dreamerv3_torch/configs.yaml`
(see the `forza:` block: task `forza_drive`, size `[64,64]`, `action_repeat 2`,
`train_ratio 32` (low for real-time), `encoder/decoder` with `cnn_keys: 'image'` +
`mlp_keys: 'speed'`, `video_pred_log False`, `eval_episode_num 0`.)

### Edit 4 - modern `ruamel.yaml` API in `dreamerv3_torch/dreamer.py`
NM512 calls `yaml.safe_load(...)`, removed in ruamel.yaml ≥0.18. In the `__main__`
block change:
```python
    configs = yaml.safe_load(
```
to:
```python
    configs = yaml.YAML(typ="safe").load(
```

### Edit 6 - ASYNC background learner in `dreamerv3_torch/dreamer.py`
Real-time control was stalled by inline grad steps (the car froze ~0.8s per training burst).
Added an `async_train` config flag (forza: True) and a background learner thread.
On the first training call, `Dreamer.__call__` runs `pretrain` synchronously before
handing control to the car; this matters after resetting actor/value or the reward
head. After that, `_ensure_learner()` starts a daemon thread (`_loop`/`_safe_train`)
that trains continuously but PACED to the train_ratio budget (`update_count <=
env_steps * train_ratio / batch_steps`), so it uses idle GPU time (resets,
per-tick sleeps) without hogging the GPU while driving. `_metrics` is now guarded
by `self._metrics_lock` (written by the learner, read/cleared in `__call__`). To disable, set
`async_train: False` (reverts to the original inline training). NOTE single-GPU caveat: control
and training still contend for the GPU during active grad steps; this decouples + paces them,
it doesn't give true simultaneous 10Hz. `train_ratio` lowered 8→4 (half the training load).

### Edit 5 - TOLERANT checkpoint load in `dreamerv3_torch/dreamer.py`
NM512 does `agent.load_state_dict(checkpoint["agent_state_dict"])` (strict). Replace the
`if (logdir / "latest.pt").exists():` block so it keeps every weight whose name+shape still
matches and re-inits the rest (and tolerates an optimiser-state mismatch). This lets us add
an observation key (e.g. the racing-line feature) WITHOUT throwing away the expensive image
CNN + RSSM dynamics. See the block in `dreamer.py` (kept/re-init print + try/except optim).

## Observation / action contract (our `ForzaDriveEnv`)
- obs = `Dict{'image': (64,64,1) uint8, 'speed': (1,) f32, 'line': (3,) f32}` — **single**
  frame (the RSSM does temporal modelling; no 4-stack), 64×64 (Dreamer's CNN size), grayscale.
  `line` = `[cue (-1 brake..+1 accelerate), lateral offset, confidence]` from `racing_line.py`,
  read off the full-res COLOUR grab; config `mlp_keys: 'speed|line'`.
- action = `Box([-1,-1,-1],[1,1,1])` (steer/throttle/brake), all in Dreamer
  coordinates. For pedals, `-1` means released and `+1` means fully pressed; the
  env maps them to gamepad trigger values `[0,1]`. `NormalizeActions` is therefore
  an identity wrapper for this env.
- `step → (obs, reward, done, info)`; obs carries `is_first`/`is_terminal` (old gym API).
- crash (CrashDetector) → `done`; next `reset()` runs the recovery ladder.

## Run order
```powershell
# 1. one-time: convert recordings -> warm-start replay episodes
.\.venv\Scripts\python.exe make_warmstart.py --logdir C:\Horizon_FSD\dreamer_logs\forza
# 2. if VRAM is tight: close FH6 and warm/rebuild actor/value/reward head offline.
.\.venv\Scripts\python.exe offline_pretrain_dreamer.py --updates 200 --logdir C:\Horizon_FSD\dreamer_logs\forza
# 3. open FH6 again, focus it on a road, then collect live experience.
.\.venv\Scripts\python.exe train_dreamer.py --logdir C:\Horizon_FSD\dreamer_logs\forza
```

## Known tuning knobs / risks
- **Real-time cadence**: Dreamer trains inline between env steps; `train_ratio` (start 32)
  controls gradient-steps-per-step. Too high → the car drifts during the training stall.
  If stalling, lower it or move the learner to a background thread.
- **Reward weights** (`reward.py` `DriveRewardConfig`): tune during the supervised shakedown
  (`jerk_w`, `offroad_w`, `speed_cap`, `crash_penalty`).
- Warm-start episodes live in `<logdir>/train_eps/`; they count toward `prefill`.
  Current Forza runs pretrain synchronously before live control, then trains in
  the background. After reward/action-contract changes, regenerate warm-start and
  quarantine old live episodes so stale rewards do not train the reward head.

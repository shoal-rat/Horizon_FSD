# Driving RL Lessons

This note records the engineering lessons that shaped Horizon FSD. It is not a
survey paper; it is a compact checklist for keeping the project pointed in the
right direction.

## Highest-Value Lessons

### Reward progress along a route, not raw speed

Speed-only rewards are easy to hack. A car can spin, drift, or loop while still
moving quickly. Racing and driving RL projects usually reward progress along a
reference path:

```text
reward_progress = s(position_t) - s(position_t_minus_1)
```

where `s(position)` is arc length after projecting the car position onto a
centerline polyline. Horizon FSD implements this through `centerline.npy` and
`reward.py`.

Current `record.py` stores world position, so a fresh clean reference session can
be converted with:

```powershell
.\.venv\Scripts\python.exe build_centerline.py --session <recording-session> --out C:\Horizon_FSD\centerline.npy
```

Old recordings made before position logging should not be used to build a
centerline.

### Keep penalties smaller than useful progress

Early policies flail. If steering, slip, off-road or jerk penalties dominate
route progress, the easiest local optimum is to stop or terminate quickly. The
current reward keeps the smoothness terms small and uses progress plus a small
launch/forward-speed bootstrap so the policy has a gradient toward moving.

### Real-time control must not block on training

The game keeps running while gradients compute. Dreamer therefore trains in a
background learner thread paced by `train_ratio`. Live control still shares the
GPU with training, so the model and batch are intentionally small.

If the car freezes during training bursts:

- reduce `batch_size`
- reduce `train_ratio`
- use `offline_pretrain_dreamer.py` with FH6 closed
- keep `video_pred_log: False`

### Resets must be telemetry-verified

Forza reset/respawn is not guaranteed to put the car on the target route. The
recovery ladder verifies live telemetry, upright attitude, surface rumble, and
route distance before handing control back to the policy.

AutoDrive is the main recovery path for off-road and guardrail states:

- if the game offers a teleport prompt, accept it for safety
- if AutoDrive drives back smoothly, save that recovery as replay
- do not train on teleport jumps

### Replay quality matters

Warm-start replay dominated by straight driving teaches a weak recovery policy.
Useful data includes:

- clean route-following
- cornering at different speeds
- low-speed recovery from grass and barriers
- off-route return-to-road sequences
- terminations and near-terminations

The `recovery_demo.py` path is meant to add exactly this missing recovery data
without requiring the learned policy to discover every escape behavior from
scratch.

## Known Risks

- Missing previous action in Dreamer observations can make latency harder to
  model. Adding `prev_action` is still a useful future improvement.
- Racing-line visual cues can be unreliable at night or under visual clutter.
  They should help, not replace telemetry route progress.
- A single fixed route can overfit. For a demo, that is acceptable; for a general
  driver, build multiple centerlines and vary start states.
- Reward curves can hide bad behavior. Watch live rollouts and inspect saved
  episodes.
- Online automation risks anti-cheat and ToS violations. Use Offline Solo only.

## Practical Next Steps

1. Keep building high-quality centerline-based replay.
2. Let AutoDrive collect smooth recovery demonstrations.
3. Periodically quarantine poor live episodes before long offline pretraining.
4. Tune `rl_safety` steering clamps down early, then loosen them as the policy
   improves.
5. Add an evaluation script that reports meters along centerline before
   termination, not just total reward.

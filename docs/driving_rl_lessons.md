# Lessons from comparable driving-RL projects

Research brief (2026-06-06) synthesizing tmrl/Trackmania, GT Sophy, Wayve "Learning to
Drive in a Day", DreamerV3/DayDreamer/CarDreamer/Think2Drive, reward-hacking literature,
real-time async-RL, BC→RL/DAgger, and in-game driving projects (GTA V, ETS2, OpenPilot).
Mapped to Horizon FSD (Forza Horizon 6 + DreamerV3, single shared 8GB GPU, ~10Hz).

> NOTE (correction to the brief): our recorded DEMOS do **not** store `position_x/z`
> (record.py logs speed/accel/rumble/slip/distance/etc., not position). Live telemetry
> HAS position. So the "build a centerline from the existing demos" step in R1 needs a
> reference lap re-recorded WITH position logged, OR a telemetry-only net-displacement
> reward that needs no centerline. See the decision note at the bottom.

## 1. Top lessons for issues we already hit

### 1A. Progress reward that resists hacking — the highest-value change
Every serious racing-RL project (tmrl, Linesight, PedroAI, GT Sophy, CarDreamer,
Think2Drive) defines progress as **net displacement along a fixed reference path**, never
raw speed. Our speed-based `progress` is direction-blind → that is exactly why the agent
spins in circles. The anti-spin yaw penalty is a band-aid; circling still pays.

Adopt: `r_progress = s(p_t) − s(p_{t-1})`, where `s(p)` = car position projected onto a
centerline polyline (arc-length). Potential-based (Ng 1999): loop-sum is provably zero, so
circling earns ~0 and reversing is negative — no anti-spin penalty needed. Works at night
(telemetry, not vision). Uncapped (GT Sophy/Linesight need no `+1` cap).
Cheaper drop-in: velocity projected on path tangent `r = v·t̂`. Telemetry-only
net-displacement-over-window is the no-centerline fallback.

### 1B. Suicidal-reward fix: uncap progress, speed-scale penalties, never let penalties win
- Wayve: NO shaping penalties; failure = termination (loss of future reward), not a spike.
- GT Sophy: penalties scale with kinetic energy (`wall = c·‖v‖²`); fixed penalties made it
  brake-and-sit. Progress unbounded.
- Raffin/aleju: tiny speed-bonus weight, large speed-scaled crash penalty, clip total reward.
Fix: remove `speed_cap`/`progress_scale`; scale `offroad`/`slip` by speed (near-free when
slow); ensure max per-step progress > worst-case per-step penalty; RAMP penalty weights up
over training rather than full weight on a flailing warm-started actor.

### 1C. Real-time throughput — async actor-learner done right (validates our direction)
DayDreamer (= DreamerV3 on hardware): learner thread trains continuously from replay; actor
thread acts in real time; **policy weights sync on a timer (~20s), not per gradient step.**
tmrl: collector never blocks on learner; throttle by one replay-to-env-step ratio (=
train_ratio). Concrete: learner = low-priority interruptible GPU job, inference = hard
real-time; lower train_ratio is correct (we're at 4); **train during the ~20s AutoDrive
reset window** (Wayve ran 250 grad steps during reset) but PAUSE the ratio clock during
reset so it isn't miscounted; keep action_repeat (10Hz is the proven sweet spot — SAC at
100Hz failed to converge in Bouteiller's elastic-timestep paper).

## 2. Pitfalls we haven't hit yet (ranked by likelihood × impact)
- **P1 [very likely × high] Missing in-flight action in the obs (RTMDP violation).** Our obs
  is {image, speed, line}, no prev_action. At 10Hz with GPU-contended latency the policy
  can't tell what command is already taking effect → can CAUSE oscillation/spin. Fix: add
  prev_action to the obs (a few scalars).
- **P2 [likely × high] Reconstruction loss dwarfs reward loss** in the world model
  (HarmonyDream, ~100x). Grayscale helps; watch reward-prediction quality, be ready to raise
  reward-head loss_scale. Keep DreamerV3's symexp/two-hot/percentile-return stabilizers.
- **P3 [very likely × med-high] Warm-start replay dominated by straight cruising.** AutoDrive
  is mostly center-throttle-no-steer; balance turns/recoveries, oversample crash/termination
  transitions (Think2Drive: 50% termination-priority). Reuse `dataset: max_straight_frac`.
- **P4 [likely × med] Single fixed reset pose → position-specific farming + primacy bias.**
  Vary reset position/heading; seed resets from demo positions (curriculum + cuts reset time).
- **P5 [likely × med] No curriculum.** Stage: straight/daytime → corners → night LAST. Scope
  to a fixed route for the thesis demo (Wayve/aleju used one ~250m section — field standard).
- **P6 [certain × med] No eval protocol.** Reward curves hide bad behavior. Hand-rank
  trajectories and confirm the reward orders them (full-lap > slow-stop > crash-at-speed);
  fixed metric = centerline-meters-before-crash; ALWAYS watch rendered rollouts.
- **P7 [low × catastrophic] Anti-cheat ban.** Run ONLY offline/Solo while automating input;
  ViGEm gamepad is lower-profile than memory injection. Assert offline before running.
- **P8 [low × med] BatchNorm on tiny correlated batches breaks** (aleju). DreamerV3 uses
  LayerNorm — never add BatchNorm to encoder layers.
- **P9 [med × low] Transient world-model collapse on visual-domain shift** (DayDreamer
  sunrise). Don't over-react to a temporary drop after a time-of-day change; keep training.

## 3. Recommended changes (ranked)
- **R1. Centerline arc-length progress in reward.py** (removes spin incentive at source,
  lets us uncap progress). Needs a reference path (see decision note). Medium.
- **R2. Uncap progress + speed-scale penalties + small time penalty + penalty ramp.** Low.
- **R3. Add prev_action to the DreamerV3 obs** (RTMDP; prevents latency oscillation). Low.
- **R4. Background learner + train during reset; sync weights on timer; pause ratio clock on
  reset.** (Partly done — async learner shipped.) Medium-high.
- **R5. State-checked reset + episode invalidation** (verify on-road before resuming; drop
  botched episodes from replay; add no-progress early termination). Medium.
- **R6. Balance warm-start replay; oversample turns/recoveries/terminations.** Low-med.
- **R7. Varied reset poses + start-state curriculum.** Medium, staged.
- **R8. Eval protocol + pre-training reward sanity check.** Low.

## 4. Disagreements / things to verify
- D1 penalties zero (Wayve) vs small-speed-scaled (GT Sophy/Raffin) — both work; we keep
  small speed-scaled, fall back to zero if hacking persists.
- D2 the `line` cue may help the critic more than the actor (GT Sophy asymmetric AC); don't
  make it load-bearing (it's night-unreliable); ablate with/without; rely on telemetry
  position for the progress reward, not vision.
- D3 arc-length deltas may be noisy at low speed/10Hz — check on demo data, tune point spacing.
- D4 BC→RL: cheapest high-confidence borrow = RLPD 50/50 demo replay (a sampler change);
  residual-on-frozen-BC caps the ceiling + leans on the night-unreliable line cue — lower priority.
- D5 train_ratio sweet spot under shared-GPU contention is unverified — profile forward-pass
  latency with FH6 running before committing.

## Decision note (position logging)
The centerline reward (R1) — the single highest-value fix — needs a reference path. Our
demos lack position. Two routes: (a) add position logging to record.py + drive a short
reference lap → build centerline → arc-length progress (proper, also sets up a fixed route);
(b) telemetry-only net-displacement-over-window reward (no centerline, works now, kills
circling, day+night). Both beat the current speed+anti-spin patch.

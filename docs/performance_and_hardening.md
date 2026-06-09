# Horizon FSD — performance + hardening guide

From the harden-and-accelerate reassessment (2026-06-09). Two goals: make the strategy **strong**
(no mis-fires in strange situations) and make training **faster + more accurate** on this laptop
(RTX 4070 Laptop 8 GB Optimus + AMD iGPU) **without shrinking the model**.

---

## 1. Performance — go faster + train on more frames, model NOT shrunk

### What's already implemented (in code)
- **`forza_full` config** (`dreamerv3_torch/configs.yaml`) — a *bigger* model for when the GPU has
  headroom: `train_ratio 4→64`, `batch_size 4→16`, `batch_length 32→48`, `dyn_deter 256→1024`,
  `dyn_stoch 24→32`, `cnn_depth 16→32`. Run with `train_dreamer.py --configs forza_full`.
- **`cudnn.benchmark = True`** (`dreamerv3_torch/dreamer.py`) — autotunes the conv kernels for the
  fixed 64×64 obs. Free throughput.
- **Async background learner** (already present) — control and training overlap on separate threads.
- **Consistent checkpoint snapshot under a train lock** (D2/D3) — lets the learner run hard without
  risking a torn save.

### What "more frames to improve accuracy" actually means here
At 10 Hz you collect ~10 env steps/s **no matter how fast capture runs** — feeding 60 fps just
dilutes the ratio. The real levers (all in `forza_full`, grow in this order, one at a time):
1. **`train_ratio` 4 → 64 (→128)** — replayed frames *learned from* per env frame. THE primary
   data-efficiency lever, and free on VRAM. Do first.
2. **`batch_length` 32 → 48 → 64** — more temporal context per sample (a sequence model loves this).
3. **`batch_size` 4 → 16** — lower-variance gradients.
4. **`dataset_size`** (replay lives in CPU RAM, not VRAM) — only raise if you have spare system RAM.

### USER HARDWARE STEPS (do these to unlock `forza_full`)
- **B0 — cap Forza, the biggest single win.** In FH6: 1080p (or 720p) Low/Medium, 60 fps frame
  limiter. FH6 VRAM often drops from ~7 GB to 3–5 GB. Keep FH6 **on the dGPU** (the 4070).
- **B1 — move the *desktop + Python/capture* to the iGPU (NOT the game).** Windows Settings →
  Display → Graphics: set Forza's `.exe` = **High performance** (dGPU), and the Python/launcher
  `.exe` = **Power saving** (iGPU). WGC capture is cross-adapter, so the iGPU can grab a window the
  dGPU rendered; CUDA still runs on the dGPU. Frees the desktop framebuffer + DWM composition off
  the dGPU. Keep Optimus/hybrid mode (a MUX "direct" mode disconnects the iGPU and defeats this).
- **Do NOT run FH6 itself on the iGPU** — it's below FH6's min spec (~4× slower than the 4070) and
  the iGPU has no dedicated VRAM, so it would spill into system RAM, not free the dGPU pool.

### Validation protocol BEFORE an overnight `forza_full` run
1. `nvidia-smi` → steady-state `memory.used` < ~6.5 GB with **no** shared-memory spill (a spill to
   system RAM is catastrophic latency, not a clean OOM). Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
2. Learner grad-steps/s must keep up with `train_ratio × 10`/s; if not, **lower `train_ratio`, not
   the model**.
3. Control thread still hits its 0.05 s budget (watch the `[forza-env] N tick overruns` heartbeat).
4. Bisect upward one axis at a time; back off on OOM in reverse: `cnn_depth → batch_length →
   batch_size → dyn_deter`, never below the lean values.

### Designed, implement-after-measuring (don't ship blind into vendored RL code)
These are real parallel-compute wins but should be **measured live** before committing (the
research itself flagged them as hardware-hint / maturity-dependent):
- **B2 — prefetched, pinned, non-blocking H2D copy.** A 1–2-slot background thread that samples the
  next batch and `pin_memory()`s it; `models.py` copies with `.to(device, non_blocking=True)` on a
  dedicated copy stream. Overlaps batch prep with GPU compute, zero VRAM cost.
- **B3 — CUDA stream priority.** High-priority stream for the actor (`_policy`), low-priority for the
  learner (`_train`), so a learner burst doesn't jitter the 10 Hz action. Lets you push `train_ratio`
  up further. (Priority is a hint, not a guarantee; no MPS on Windows, so it can't fix cross-process
  contention with FH6 — measure per-action latency.)
- **B5 — `torch.compile` on the LEARNER only** (`mode='reduce-overhead'`), warmed during pretrain,
  with an eager fallback. Not the actor (shape-change recompiles stall the car). Needs triton-windows.
- **B7 — gradient accumulation** for a bigger effective batch at flat peak VRAM.

---

## 2. Robustness — strange-situation guards

### Implemented + tested (this pass)
- **Paused/menu with packets still flowing** → `step()` neutralizes the pad and ends the episode
  benignly (`"paused"`); `reset()` waits for racing to resume instead of mashing buttons into a menu.
- **Frozen render frame** (load screen / alt-tab) via `capture.frame_age()` → benign `"frame_lost"`
  end, no teleport, wait for fresh frames. (Generous 1.5 s threshold — tune live; see below.)
- **ViGEm latch on crash** → `gamepad` registers an `atexit` neutralizer so a crash can't leave the
  car at full throttle.
- **Teleport / fast-travel position jump** (>30 m in one tick) → detector suppresses
  impact/offroute/noprogress and re-anchors, so a respawn isn't read as a crash.
- **GPU-stall long tick** → detector skips the impact rate-check (a long-`dt` braking gap isn't a crash).
- **Slow uphill crawl vs wedged** → `stuck` now requires near-zero **world displacement**, not just
  low speed, so a 1.5 m/s climb that still covers ground isn't "stuck".
- **Post-recovery grace** scoped: suppresses only stuck/noprogress/offroute (settling); keeps
  flipped/offroad/impact active (a recovered car that's flipped means recovery FAILED).
- **Reward/detector lateral consistency** — reward credits progress only within `offroute_dist`.
- (Plus the prior pass: NaN at parse boundary, stale telemetry, route-end, reverse-farming, etc.)

### Needs a LIVE-GAME check before finalizing
- **Does FH6 keep emitting Data Out packets while paused/in a menu?** Drive → pause → watch
  `telemetry_probe.py`. If the stream **stops**, the existing `telemetry_lost` path covers it; if it
  **continues with `is_race_on=0`**, the new `"paused"` path covers it. (Both are handled; this just
  confirms which fires.)
- **`capture.frame_age()` on a static-but-HUD driving scene** — confirm WGC keeps delivering frames
  (HUD/minimap animate) so `frame_lost` only fires on a real freeze. Bump `capture.max_frame_age_s`
  if it false-fires on quiet scenes.
- **Window-focus guard** (deferred) — neutralize + pause the agent when FH6 loses foreground (ViGEm
  writes are dropped unfocused while WGC vision keeps working). Needs the FH6 window handle.
- **Jumps / airtime** (deferred) — a launch/landing can read as phantom `impact`/`offroad`. Proper
  fix needs an airborne signal (`norm_susp_travel` near full extension, or `position_y` velocity);
  confirm those fields on FH6 first, then suppress impact/offroad while airborne.
- **Falling into water / off the map** (deferred) — `wheel_in_puddle_depth_*` and `position_y` are
  decoded but unused; needs live thresholds, then terminate `"submerged"` → straight to `reset_position`.
- **Recovery macro into the wrong UI** (deferred) — `_open_autodrive` mashes D-pad/A blind; add a
  cheap template-match that the AutoDrive/map UI is actually up. Needs a real frame to calibrate.
- **Post-FH6-update packet-layout sanity probe** (deferred) — a same-size field reorder passes the
  size assert + finite-check silently; add a plausibility probe at startup.

### Reward-design DECISIONS for you (not silently changed)
- **Drifting** — spin + slip both penalize a controlled drift with no progress credit. If you want
  drifting allowed on this route, raise `spin_deadband` / gate the spin penalty on `ds<=0`.
- **Continuous lateral penalty** — there is none, so "drive fast anywhere in the corridor" is a
  viable non-tracking policy. A small `-k·lateral` term would pull the car to the line. Add it?
- **Switchback/roundabout projection** — `Centerline.project` is a global nearest-segment search; if
  a future `centerline.npy` has legs passing within a few metres, make projection stateful (search
  near the previous arc index). Not live on the current open 2.76 km route.

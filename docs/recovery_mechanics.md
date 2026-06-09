# FH6 recovery mechanics (and how recovery.py models them)

Researched 2026-06-09 (Forza wiki/forums/shacknews/gamerant/player guides + the user's
first-hand FH6 play). The three in-game recovery mechanics are **distinct and not
interchangeable** — conflating them was the bug in the previous recovery code.

## The three mechanics

| Mechanic | Controller input | What it does | Right tool for |
|---|---|---|---|
| **Rewind** | `Y` (tap) | Rolls back a few seconds of your own path, upright. Needs damage = None/Cosmetic (greyed out under Simulation). | Fresh on-road impact / just-happened flip (short window) |
| **ANNA AutoDrive** | `D-pad Down → Down-Left → A` | **One feature, two state branches** (see below). Needs a waypoint pinned. | PRIMARY "get back to the route" |
| **Reset Car Position** | Pause(`Start`) → `L3` → `A` | Respawns upright on nearest flat road, speed 0. | LAST-RESORT fallback only |

## AutoDrive is itself both the teleport and the drive-back

This is the key correction. Opening AutoDrive branches on the car's state:

- **Far off-road** → a "transfer car?" **prompt** appears → `A` **teleports** the car to the
  centre of the road, then it drives.
- **On a road but stuck** → **no prompt**; AutoDrive just **drives** along the route line.

So there is **no need for a separate pause-menu teleport "escalation" for the far case** —
AutoDrive's own prompt is the teleport. (The earlier code wrongly treated Reset Car Position
as the universal teleport that leads the ladder once AutoDrive "fails".)

## Telling the two AutoDrive branches apart from telemetry

Telemetry can't read the prompt text, so `_wait_autodrive_resolved` uses **world-position
displacement** (not instantaneous speed) as a small state machine:

- **Frozen** (no net displacement) for a moment after opening, then a **> `autodrive_teleport_jump_m`
  (30 m) position JUMP** = the transfer prompt was accepted (teleport branch).
- **Net forward displacement accumulating, no jump** = AutoDrive is driving (on-road branch).
- We press `A` **only** while the car is **frozen**, after a short settle (`autodrive_prompt_settle_s`),
  within a bounded window (`autodrive_prompt_window_s`), capped at `autodrive_prompt_attempts` (2),
  and **stop for good** the instant it teleports OR starts driving. The old code tapped `A`
  whenever `speed < 2`, which spammed `A` into a live AutoDrive drive through every slow corner —
  and a stray `A` cancels AutoDrive.

## When the episode ends (CrashDetector) — and why it matters for recovery

The episode ends (and recovery runs) on: `impact`, `stuck`, `flipped`, `offroad`, **`offroute`**,
**`noprogress`**. Two route-aware terminations were added (they need `centerline.npy`):

- **offroad** now fires at **any speed** (the old `speed < 10` gate let a car drive *fast*
  off-road forever without resetting).
- **offroute**: lateral distance from the route centreline > `offroute_dist` (18 m). Ending the
  episode the moment the car leaves the route keeps it **near** the route, so AutoDrive recovers
  it with a short drive — instead of letting it wander far, where AutoDrive's teleport drops it on
  a distant road with a long drive back to the waypoint.
- **noprogress**: on-route but centreline arc-length not advancing for `noprogress_seconds` (5 s)
  = circling / wrong-way / stuck-at-speed.

## The recovery ladder (recover())

Every reason uses the same ladder: **`[autodrive, reset_position, reset_to_road]`**.

- **Rewind is removed** (unreliable on this build; AutoDrive only).
- AutoDrive is primary — it covers far-teleport *and* on-road-drive, and its transfer branch even
  rights a flipped car. Reset Car Position is the **last-resort rung**, reached each round only if
  AutoDrive can't route the car.
- `autodrive_persistent` (unattended): never returns FAILED; re-runs the ladder (ending in the
  pause reset) with capped backoff + a heartbeat log — never a silent hang.

## After AutoDrive succeeds

Cancel the lingering AutoDrive with a **brake** tap (`autodrive_break_s ≈ 1 s`), not throttle:
AutoDrive drops the car at road centre with no guaranteed heading, so a throttle pulse would
launch it (possibly into oncoming/a barrier).

## Preconditions / open items

- **A waypoint must be pinned** to the route or AutoDrive has nothing to drive toward and just
  stalls (→ the ladder then falls to the pause reset). Keep a route waypoint set during training.
- The far-off-road transfer-prompt branch is the user's first-hand observation (FH6 is new and
  no public source documents it verbatim). The A-tap logic is deliberately conservative so that
  if the prompt is absent on a build, the worst case is 1–2 harmless early A presses, not the old
  A-spam-into-a-drive.
- Bindings are for a **controller**; PC keyboard uses `C` for the second ANNA input.

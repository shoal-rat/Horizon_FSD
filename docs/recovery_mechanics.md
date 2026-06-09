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

## The recovery ladder (recover())

- `impact` / `stuck` / `flipped` → `[rewind, autodrive, reset_position, reset_to_road]`
- `offroad` / unknown → `[autodrive, reset_position, reset_to_road]`
- AutoDrive is primary (covers far-teleport + on-road-drive). Reset Car Position is the **last
  rung**, not a queue-jumping escalation — reached each round only if everything above failed.
- A flipped car: rewind (if recent) → AutoDrive's transfer branch rights it when off-road →
  reset position. (AutoDrive *can* recover a flipped car via its teleport — the old "AutoDrive
  cannot un-flip" claim was wrong.)
- After `max_consecutive_rewinds` quick repeats, rewind is dropped so we don't bounce off the
  same wall.
- `autodrive_persistent` (unattended): never returns FAILED; re-runs the full ladder (ending in
  the pause reset) with capped backoff + a heartbeat log — never a silent hang.

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

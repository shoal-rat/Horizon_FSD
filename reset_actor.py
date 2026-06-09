"""
reset_actor.py - Horizon FSD

Reset ONLY the actor + critic (the collapsed policy) in a DreamerV3 checkpoint, while
keeping the expensive world model (image CNN + RSSM dynamics + decoder/cont heads). Use
when the policy has collapsed into a degenerate behaviour (e.g. driving in circles) and
you want a clean behavioural restart under a corrected reward, without re-learning the
world model for hours.

It deletes the actor/value/slow-value/reward-EMA tensors from latest.pt and clears the
optimiser state. On the next training run, the TOLERANT checkpoint load (dreamer.py edit
#5) loads the kept world-model weights and RE-INITIALISES the missing actor/critic fresh.
If `--reset-reward-head` is passed, it also re-initialises the world-model reward head;
use this after changing `reward.py`.

A timestamped backup is written first. To undo: copy the .bak back over latest.pt.

Run:
    .\\.venv\\Scripts\\python.exe reset_actor.py
    .\\.venv\\Scripts\\python.exe reset_actor.py --logdir C:\\Horizon_FSD\\dreamer_logs\\forza
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import time

import torch

# actor + critic params (NOT _task_behavior._world_model, which IS the world model)
_RESET = re.compile(r"^(_task_behavior|_expl_behavior)\.(actor|value|_slow_value|ema_vals)")
_REWARD_HEAD = re.compile(
    r"^(_wm|_task_behavior\._world_model|_expl_behavior\._world_model)\.heads\.reward\."
)


def _backup_path(path: str) -> str:
    base = path + ".bak_before_actor_reset"
    if not os.path.exists(base):
        return base
    return path + ".bak_before_actor_reset_" + time.strftime("%Y%m%d_%H%M%S")


def main() -> int:
    p = argparse.ArgumentParser(description="Reset the actor/critic in a Dreamer checkpoint.")
    p.add_argument("--logdir", default=r"C:\Horizon_FSD\dreamer_logs\forza")
    p.add_argument("--reset-reward-head", action="store_true",
                   help="Also re-init the world-model reward head after changing reward.py.")
    p.add_argument("--dry-run", action="store_true", help="Report what would change, don't write.")
    args = p.parse_args()

    path = os.path.join(args.logdir, "latest.pt")
    if not os.path.exists(path):
        print(f"no checkpoint at {path}")
        return 1

    ckpt = torch.load(path, map_location="cpu")
    sd = ckpt["agent_state_dict"]
    drop = [
        k for k in sd
        if _RESET.match(k) or (args.reset_reward_head and _REWARD_HEAD.match(k))
    ]
    keep = len(sd) - len(drop)
    print(f"checkpoint: {len(sd)} tensors")
    print(f"  KEEP  {keep}  (world model dynamics/encoder/decoder and any non-reset heads)")
    extra = " + reward_head" if args.reset_reward_head else ""
    print(f"  RESET {len(drop)}  (actor/value/slow_value/ema{extra} -> re-init fresh on next run)")

    if args.dry_run:
        print("dry run - nothing written.")
        return 0

    bak = _backup_path(path)
    shutil.copy(path, bak)
    for k in drop:
        del sd[k]
    ckpt["optims_state_dict"] = {}          # fresh optimisers (world-model moments re-warm quickly)
    torch.save(ckpt, path)
    print(f"\ndone. backup: {bak}")

    # Resetting the reward HEAD doesn't fix STALE reward LABELS already frozen into replay episodes:
    # they were computed by the OLD reward.py, so the fresh head would just learn the old targets.
    # Purge them so they're regenerated under the current reward.
    if args.reset_reward_head:
        import glob
        eps_dir = os.path.join(args.logdir, "train_eps")
        stale = (glob.glob(os.path.join(eps_dir, "ws-*.npz"))
                 + glob.glob(os.path.join(eps_dir, "recovery-*.npz")))
        for f in stale:
            try:
                os.remove(f)
            except OSError:
                pass
        print(f"purged {len(stale)} stale-reward episodes (ws-*/recovery-*) from {eps_dir}")
        print("  -> REBUILD warm-start before training: make_warmstart.py --logdir <logdir>")
    print("re-run train_dreamer.py: the tolerant load will keep the world model and start a fresh policy.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

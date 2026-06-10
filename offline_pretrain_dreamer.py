"""
offline_pretrain_dreamer.py - Horizon FSD

Train DreamerV3 from replay WITHOUT creating the live Forza environment.

Use this when VRAM is tight:
  1. Close FH6 to free GPU memory.
  2. Run this script to warm/rebuild actor/value/reward-head from train_eps.
  3. Re-open FH6 and run train_dreamer.py for live collection.

The script preserves the same tolerant checkpoint loading behavior as the live
trainer: matching world-model tensors are kept, missing actor/value/reward-head
tensors are re-initialised and then trained from replay.

Run:
    .\\.venv\\Scripts\\python.exe offline_pretrain_dreamer.py --updates 200
"""
from __future__ import annotations

import argparse
import os
import pathlib
import sys

import gym
import numpy as np
import ruamel.yaml as yaml
import torch

DREAMER_DIR = pathlib.Path(r"C:\Horizon_FSD\dreamerv3_torch")
HFSD_DIR = pathlib.Path(r"C:\Horizon_FSD")
if str(DREAMER_DIR) not in sys.path:
    sys.path.insert(0, str(DREAMER_DIR))
if str(HFSD_DIR) not in sys.path:
    sys.path.insert(0, str(HFSD_DIR))

import tools  # noqa: E402
from dreamer import Dreamer, count_steps, make_dataset  # noqa: E402


def _recursive_update(base: dict, update: dict) -> None:
    for key, value in update.items():
        if isinstance(value, dict) and key in base:
            _recursive_update(base[key], value)
        else:
            base[key] = value


def _load_config(config_names: list[str], remaining: list[str], logdir: str | None):
    configs = yaml.YAML(typ="safe").load((DREAMER_DIR / "configs.yaml").read_text())
    defaults: dict = {}
    for name in ["defaults", *config_names]:
        _recursive_update(defaults, configs[name])

    parser = argparse.ArgumentParser(add_help=False)
    for key, value in sorted(defaults.items(), key=lambda x: x[0]):
        arg_type = tools.args_type(value)
        parser.add_argument(f"--{key}", type=arg_type, default=arg_type(value))
    config = parser.parse_args(remaining)
    if logdir is not None:
        config.logdir = logdir
    return config


def _spaces(config):
    h, w = int(config.size[0]), int(config.size[1])
    obs_space = gym.spaces.Dict({
        "image": gym.spaces.Box(0, 255, (h, w, 1), dtype=np.uint8),
        "speed": gym.spaces.Box(0.0, np.inf, (1,), dtype=np.float32),
        "line": gym.spaces.Box(-1.0, 1.0, (3,), dtype=np.float32),
    })
    act_space = gym.spaces.Box(
        np.array([-1.0, -1.0, -1.0], dtype=np.float32),
        np.array([1.0, 1.0, 1.0], dtype=np.float32),
        dtype=np.float32,
    )
    return obs_space, act_space


def _load_checkpoint(agent: Dreamer, path: pathlib.Path) -> None:
    if not path.exists():
        print(f"[checkpoint] no existing checkpoint at {path}; training fresh")
        return
    checkpoint = torch.load(path, map_location=agent._config.device)
    saved = checkpoint["agent_state_dict"]
    model_sd = agent.state_dict()
    kept = {k: v for k, v in saved.items() if k in model_sd and v.shape == model_sd[k].shape}
    agent.load_state_dict(kept, strict=False)
    reinit = [k for k in model_sd if k not in kept]
    if reinit:
        print(f"[checkpoint] kept {len(kept)}/{len(model_sd)} tensors; "
              f"re-init {len(reinit)}: {reinit[:6]}{'...' if len(reinit) > 6 else ''}")
        try:
            tools.recursively_load_optim_state_dict(agent, checkpoint["optims_state_dict"])
        except Exception as e:
            print(f"[checkpoint] optimiser state not restored ({type(e).__name__}); using fresh optim")
    else:
        tools.recursively_load_optim_state_dict(agent, checkpoint["optims_state_dict"])
        print(f"[checkpoint] loaded all {len(model_sd)} tensors")


def _bc_step(agent, demo_dataset) -> float:
    """One behavioral-cloning step: push the ACTOR toward the human demo action at the demo latents
    (DreamerFD recipe). The world model is FROZEN here (it learns demos via its own loss); only the
    actor moves. This is what makes the demos reach the POLICY instead of only the world model - the
    fix for an agent that can't discover steering by RL alone, and never imitates the human's wheel."""
    wm = agent._wm
    actor = agent._task_behavior.actor
    bc_obs = next(demo_dataset)
    with torch.no_grad():                                 # latents from the (frozen) world model
        data = wm.preprocess(bc_obs)
        embed = wm.encoder(data)
        post, _ = wm.dynamics.observe(embed, data["action"], data["is_first"])
        feat = wm.dynamics.get_feat(post)
    a_demo = torch.clamp(data["action"], -0.999, 0.999)   # fp16-safe at the tanh boundary
    with tools.RequiresGrad(actor):
        bc_loss = -actor(feat).log_prob(a_demo).mean()
        agent._task_behavior._actor_opt(bc_loss, actor.parameters())
    return float(bc_loss.detach().cpu())


def main() -> int:
    p = argparse.ArgumentParser(description="Offline Dreamer replay training; no FH6 env is created.")
    p.add_argument("--logdir", default=r"C:\Horizon_FSD\dreamer_logs\forza")
    p.add_argument("--configs", nargs="+", default=["forza"])
    p.add_argument("--updates", type=int, default=None,
                   help="Gradient updates to run; default uses config.pretrain.")
    p.add_argument("--bc", type=int, default=1,
                   help="Interleave a behavioral-cloning step (actor imitates the human demos) each "
                        "update. 1=on (default), 0=off. Needs ws-* demo episodes in train_eps.")
    p.add_argument("--save-every", type=int, default=50)
    args, remaining = p.parse_known_args()

    config = _load_config(args.configs, remaining, args.logdir)
    updates = int(args.updates if args.updates is not None else config.pretrain)

    tools.set_seed_everywhere(config.seed)
    logdir = pathlib.Path(config.logdir).expanduser()
    config.traindir = pathlib.Path(config.traindir or logdir / "train_eps")
    config.evaldir = pathlib.Path(config.evaldir or logdir / "eval_eps")
    logdir.mkdir(parents=True, exist_ok=True)
    config.traindir.mkdir(parents=True, exist_ok=True)

    train_eps = tools.load_episodes(config.traindir, limit=config.dataset_size)
    if not train_eps:
        raise RuntimeError(f"no replay episodes found in {config.traindir}")
    step = count_steps(config.traindir)
    logger = tools.Logger(logdir, config.action_repeat * step)
    dataset = make_dataset(train_eps, config)
    # demo-only stream (human recordings) for the behavioral-cloning step
    demo_eps = {k: v for k, v in train_eps.items() if "ws-" in str(k)}
    demo_dataset = make_dataset(demo_eps, config) if (args.bc and demo_eps) else None
    obs_space, act_space = _spaces(config)
    config.num_actions = act_space.shape[0]

    print("=" * 74)
    print(" Horizon FSD - offline Dreamer pretrain")
    print("=" * 74)
    print(f" logdir      : {logdir}")
    print(f" train_eps   : {config.traindir} ({len(train_eps)} episodes, ~{step} steps)")
    print(f" updates     : {updates}")
    print(f" device      : {config.device}")
    print(f" BC (imitate demos): {'on, ' + str(len(demo_eps)) + ' demo eps' if demo_dataset else 'off'}")
    print(" FH6/env     : not created; safe to keep the game closed")

    agent = Dreamer(obs_space, act_space, config, logger, dataset).to(config.device)
    agent.requires_grad_(requires_grad=False)
    ckpt_path = logdir / "latest.pt"
    _load_checkpoint(agent, ckpt_path)

    bc_loss = float("nan")
    for i in range(1, updates + 1):
        agent._safe_train()
        if demo_dataset is not None:
            bc_loss = _bc_step(agent, demo_dataset)
        if i == 1 or i == updates or (args.save_every and i % args.save_every == 0):
            with agent._metrics_lock:
                reward_loss = agent._metrics.get("reward_loss", [float("nan")])[-1]
                actor_loss = agent._metrics.get("actor_loss", [float("nan")])[-1]
            print(f"[offline] update {i:4d}/{updates}  "
                  f"reward_loss={float(reward_loss):+.4f} actor_loss={float(actor_loss):+.4f} "
                  f"bc_loss={bc_loss:+.4f}")
            torch.save({
                "agent_state_dict": agent.state_dict(),
                "optims_state_dict": tools.recursively_collect_optim_state_dict(agent),
            }, ckpt_path)

    logger.scalar("offline_pretrain_updates", updates)
    logger.write(step=logger.step)
    if updates > 0:
        print(f"saved: {ckpt_path}")
    else:
        print("updates=0; checkpoint left unchanged")
    print("Next: open FH6, then run train_dreamer.py with the same --logdir.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

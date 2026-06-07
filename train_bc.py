"""
train_bc.py - Horizon FSD, Phase 3

Train the behavioral-cloning policy on the recorded dataset (offline - no game).
A timm backbone + MLP head regresses [steer, throttle, brake] from stacked frames
+ speed, with MSE loss, per-dimension MAE metrics, TensorBoard logging, and
best/last checkpointing.

Run:
    .\\.venv\\Scripts\\python.exe train_bc.py
    .\\.venv\\Scripts\\python.exe train_bc.py --epochs 20 --backbone efficientnet_b0
    .\\.venv\\Scripts\\python.exe train_bc.py --freeze-backbone
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

from bc_model import BCPolicy, save_checkpoint
from config import load_config
from dataset import build_dataset


class TorchBC(Dataset):
    """Wrap a (lazy) BCDataset for torch: returns (img float[0,1], speed_norm, action).

    When augment=True (train only): horizontal flip (negate steer), horizontal shift
    with a proportional steer correction (synthesizes recovery), and brightness jitter.
    """

    def __init__(self, bcds, speed_norm: float, augment: bool = False, aug: dict | None = None):
        self.ds = bcds
        self.speed_norm = speed_norm
        self.augment = augment
        a = aug or {}
        self.flip = a.get("aug_flip", True)
        self.bright = float(a.get("aug_brightness", 0.0) or 0.0)
        self.shift_px = int(a.get("aug_shift_px", 0) or 0)
        self.shift_sigma = float(a.get("aug_shift_sigma", 4.0) or 4.0)
        self.steer_per_px = float(a.get("aug_steer_per_px", 0.0) or 0.0)
        self.speed_ref = float(a.get("aug_speed_ref", 20.0) or 20.0)
        self.speed_min = float(a.get("aug_speed_min", 5.0) or 5.0)
        self.rng = np.random.default_rng()

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, i: int):
        stack, speed, action = self.ds[i]                  # (C,H,W) uint8, float, (3,)
        img = stack.astype(np.float32) / 255.0
        action = action.astype(np.float32).copy()
        if self.augment:
            if self.flip and self.rng.random() < 0.5:       # mirror left<->right
                img = img[:, :, ::-1]
                action[0] = -action[0]
            if self.shift_px > 0:                            # speed-aware recovery shift
                dx = int(round(self.rng.normal(0.0, self.shift_sigma)))  # mass near 0
                dx = max(-self.shift_px, min(self.shift_px, dx))
                if dx != 0:
                    img = np.roll(img, dx, axis=2)
                    if dx > 0:
                        img[:, :, :dx] = img[:, :, dx:dx + 1]
                    else:
                        img[:, :, dx:] = img[:, :, dx - 1:dx]
                    # "return to center" correction, gentler at higher speed (PilotNet)
                    speed_scale = min(4.0, max(0.25, self.speed_ref / max(speed, self.speed_min)))
                    action[0] = float(np.clip(action[0] + self.steer_per_px * dx * speed_scale, -1.0, 1.0))
            if self.bright > 0:                              # brightness jitter
                img = np.clip(img * self.rng.uniform(1.0 - self.bright, 1.0 + self.bright), 0.0, 1.0)
        img_t = torch.from_numpy(np.ascontiguousarray(img)).float()
        sp = torch.tensor([speed / self.speed_norm], dtype=torch.float32)
        act = torch.from_numpy(action)
        return img_t, sp, act


def evaluate(model, loader, device, loss_fn) -> tuple[float, np.ndarray]:
    """loss_fn(img, sp, act) -> scalar; MAE measured on the decoded continuous action."""
    model.eval()
    tot_loss, n = 0.0, 0
    abs_err = np.zeros(3, dtype=np.float64)
    with torch.no_grad():
        for img, sp, act in loader:
            img, sp, act = img.to(device), sp.to(device), act.to(device)
            tot_loss += loss_fn(img, sp, act).item() * len(img)
            abs_err += (model.act(img, sp) - act).abs().sum(dim=0).cpu().numpy()
            n += len(img)
    return tot_loss / max(1, n), abs_err / max(1, n)


def main() -> int:
    p = argparse.ArgumentParser(description="Train behavioral cloning on recorded driving.")
    p.add_argument("--config", default=None)
    p.add_argument("--recordings-dir", default=None)
    p.add_argument("--backbone", default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--freeze-backbone", action="store_true")
    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument("--out-dir", default=None, help="Checkpoint dir (default config paths.checkpoints_dir).")
    args = p.parse_args()

    cfg = load_config(args.config)
    bc = cfg.get("bc", {})
    backbone = args.backbone or bc.get("backbone", "resnet18")
    epochs = args.epochs or bc.get("epochs", 15)
    batch_size = args.batch_size or bc.get("batch_size", 128)
    lr = args.lr or bc.get("lr", 3e-4)
    wd = bc.get("weight_decay", 1e-4)
    hidden = bc.get("hidden", 256)
    dropout = bc.get("dropout", 0.1)
    speed_norm = bc.get("speed_norm", 100.0)
    num_workers = bc.get("num_workers", 0)
    pretrained = bc.get("pretrained", True) and not args.no_pretrained
    freeze = args.freeze_backbone or bc.get("freeze_backbone", False)
    steer_bins = bc.get("steer_bins", 21)
    steer_bin_power = bc.get("steer_bin_power", 1.5)
    corner_k = bc.get("loss_corner_k", 0.0)
    w_steer = bc.get("w_steer", 1.0)
    w_tb = bc.get("w_throttle_brake", 5.0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}  ({torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'})")

    train_ds, val_ds, stats = build_dataset(cfg, args.recordings_dir)
    frame_stack = train_ds.K
    print(f"dataset: train={len(train_ds)}  val={len(val_ds)}  frame_stack={frame_stack}")
    train_loader = DataLoader(TorchBC(train_ds, speed_norm, augment=True, aug=bc), batch_size=batch_size,
                              shuffle=True, num_workers=num_workers, drop_last=True, pin_memory=(device == "cuda"))
    val_loader = DataLoader(TorchBC(val_ds, speed_norm, augment=False), batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=(device == "cuda"))

    model = BCPolicy(backbone=backbone, in_chans=frame_stack, hidden=hidden,
                     dropout=dropout, pretrained=pretrained,
                     steer_bins=steer_bins, steer_bin_power=steer_bin_power).to(device)
    if freeze:
        model.freeze_backbone()
        print("backbone frozen (training head only)")
    print(f"steer head: {'classification %d bins' % steer_bins if steer_bins > 0 else 'regression'}"
          f"  corner_k={corner_k}  w_steer/tb={w_steer}/{w_tb}")
    params = [q for q in model.parameters() if q.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    def compute_loss(img, sp, act):
        w = 1.0 + corner_k * act[:, 0].abs()                 # per-sample corner weight
        if model.steer_bins > 0:
            logits, tb = model(img, sp)
            idx = (act[:, 0:1] - model.bin_centers.unsqueeze(0)).abs().argmin(dim=1)
            per = w_steer * F.cross_entropy(logits, idx, reduction="none") \
                + w_tb * ((tb - act[:, 1:3]) ** 2).mean(dim=1)
        else:
            pred = model(img, sp)
            per = w_steer * (pred[:, 0] - act[:, 0]) ** 2 \
                + w_tb * ((pred[:, 1:3] - act[:, 1:3]) ** 2).mean(dim=1)
        return (w * per).mean()

    out_dir = args.out_dir or cfg.get("paths", {}).get("checkpoints_dir", "checkpoints")
    os.makedirs(out_dir, exist_ok=True)
    run_name = f"bc_{backbone}_{time.strftime('%Y%m%d_%H%M%S')}"
    writer = SummaryWriter(os.path.join(cfg.get("paths", {}).get("tensorboard_dir", "runs"), run_name))

    best_val = float("inf")
    dims = ["steer", "throttle", "brake"]
    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.perf_counter()
        run_loss, seen = 0.0, 0
        for img, sp, act in train_loader:
            img, sp, act = img.to(device, non_blocking=True), sp.to(device, non_blocking=True), act.to(device, non_blocking=True)
            opt.zero_grad()
            loss = compute_loss(img, sp, act)
            loss.backward()
            opt.step()
            run_loss += loss.item() * len(img)
            seen += len(img)
        sched.step()
        train_loss = run_loss / max(1, seen)
        val_loss, val_mae = evaluate(model, val_loader, device, compute_loss)

        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("loss/val", val_loss, epoch)
        for d, e in zip(dims, val_mae):
            writer.add_scalar(f"val_mae/{d}", e, epoch)
        print(f"epoch {epoch:2d}/{epochs}  train={train_loss:.4f}  val={val_loss:.4f}  "
              f"MAE[steer/thr/brk]={val_mae[0]:.3f}/{val_mae[1]:.3f}/{val_mae[2]:.3f}  "
              f"({time.perf_counter()-t0:.1f}s)")

        save_checkpoint(os.path.join(out_dir, "bc_last.pt"), model, speed_norm,
                        extra={"epoch": epoch, "val_loss": val_loss, "frame_stack": frame_stack})
        # Select on decoded-action MAE (driving-relevant), not the CE loss, which
        # overfits/rises for the classification head even as steering keeps improving.
        val_score = float(val_mae.sum())
        if val_score < best_val:
            best_val = val_score
            save_checkpoint(os.path.join(out_dir, "bc_best.pt"), model, speed_norm,
                            extra={"epoch": epoch, "val_loss": val_loss, "val_mae_sum": val_score,
                                   "frame_stack": frame_stack})
            print(f"   ^ new best (MAE sum={val_score:.3f}) -> {os.path.join(out_dir, 'bc_best.pt')}")

    writer.close()
    print(f"\nDONE. best val MAE-sum={best_val:.3f}. Checkpoints in {out_dir}/ (bc_best.pt, bc_last.pt)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""
bc_model.py - Horizon FSD, Phase 3 (v2: classification steering head)

The behavioral-cloning policy, shared by train_bc.py and run_policy.py.

A timm vision backbone consumes the 4-channel stacked frames; its features are
concatenated with the normalized speed scalar; an MLP head outputs the action.

Steering is predicted as a CLASSIFICATION over nonlinearly-spaced bins (denser
near 0) rather than MSE regression: MSE on a mostly-straight, symmetric steering
distribution mode-averages toward 0 (drives straight into corners), while a
softmax preserves multimodality. At inference, steer = the softmax's expected
value (smooth). Throttle/brake stay regression (sigmoid). Set steer_bins=0 to
fall back to the original tanh-regression head.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn

DEFAULT_SPEED_NORM = 100.0


def bin_centers(n: int, power: float) -> np.ndarray:
    """n bin centers spanning [-1, 1], denser near 0 when power > 1."""
    x = np.linspace(-1.0, 1.0, n)
    return np.sign(x) * (np.abs(x) ** power)


class BCPolicy(nn.Module):
    def __init__(
        self,
        backbone: str = "resnet18",
        in_chans: int = 4,
        scalar_dim: int = 1,
        hidden: int = 256,
        dropout: float = 0.1,
        pretrained: bool = True,
        steer_bins: int = 21,
        steer_bin_power: float = 1.5,
    ) -> None:
        super().__init__()
        import timm

        self.backbone_name = backbone
        self.in_chans = in_chans
        self.scalar_dim = scalar_dim
        self.hidden = hidden
        self.dropout = dropout
        self.steer_bins = steer_bins
        self.steer_bin_power = steer_bin_power

        self.backbone = timm.create_model(
            backbone, pretrained=pretrained, in_chans=in_chans, num_classes=0, global_pool="avg"
        )
        feat = self.backbone.num_features
        out_dim = (steer_bins + 2) if steer_bins > 0 else 3
        self.head = nn.Sequential(
            nn.Linear(feat + scalar_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )
        if steer_bins > 0:
            self.register_buffer(
                "bin_centers", torch.tensor(bin_centers(steer_bins, steer_bin_power), dtype=torch.float32)
            )

    def forward(self, img: torch.Tensor, speed: torch.Tensor):
        x = torch.cat([self.backbone(img), speed], dim=1)
        out = self.head(x)
        if self.steer_bins > 0:
            steer_logits = out[:, : self.steer_bins]
            tb = torch.sigmoid(out[:, self.steer_bins:])  # throttle, brake
            return steer_logits, tb
        steer = torch.tanh(out[:, 0:1])
        throttle = torch.sigmoid(out[:, 1:2])
        brake = torch.sigmoid(out[:, 2:3])
        return torch.cat([steer, throttle, brake], dim=1)

    def act(self, img: torch.Tensor, speed: torch.Tensor) -> torch.Tensor:
        """Continuous action [B,3] = [steer, throttle, brake]."""
        if self.steer_bins > 0:
            logits, tb = self.forward(img, speed)
            probs = torch.softmax(logits, dim=1)
            steer = (probs * self.bin_centers.unsqueeze(0)).sum(dim=1, keepdim=True)
            return torch.cat([steer, tb], dim=1)
        return self.forward(img, speed)

    def freeze_backbone(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = False

    def config(self) -> dict[str, Any]:
        return {
            "backbone": self.backbone_name, "in_chans": self.in_chans,
            "scalar_dim": self.scalar_dim, "hidden": self.hidden, "dropout": self.dropout,
            "steer_bins": self.steer_bins, "steer_bin_power": self.steer_bin_power,
        }


def save_checkpoint(path: str, model: BCPolicy, speed_norm: float, extra: dict | None = None) -> None:
    torch.save(
        {"model_config": model.config(), "state_dict": model.state_dict(),
         "speed_norm": speed_norm, "extra": extra or {}},
        path,
    )


def load_policy(path: str, device: str = "cpu") -> tuple[BCPolicy, float]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    mc = ckpt["model_config"]
    model = BCPolicy(
        backbone=mc["backbone"], in_chans=mc["in_chans"], scalar_dim=mc["scalar_dim"],
        hidden=mc["hidden"], dropout=mc["dropout"], pretrained=False,
        steer_bins=mc.get("steer_bins", 0), steer_bin_power=mc.get("steer_bin_power", 1.5),
    )
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model, float(ckpt.get("speed_norm", DEFAULT_SPEED_NORM))


@torch.no_grad()
def predict_action(model: BCPolicy, frames_uint8: np.ndarray, speed_ms: float,
                   speed_norm: float, device: str = "cpu") -> np.ndarray:
    img = torch.as_tensor(frames_uint8, dtype=torch.float32, device=device).unsqueeze(0) / 255.0
    speed = torch.tensor([[speed_ms / speed_norm]], dtype=torch.float32, device=device)
    return model.act(img, speed)[0].cpu().numpy().astype(np.float32)

"""UMapNet: noisy tile -> predicted IR-SDE uncertainty map (per-pixel std, 0-255 units).

Class from cell 4 of umap_B_train_test_generator_kaggle.ipynb. The network predicts the
normalised log-variance (log(u^2 + EPS) - T_MEAN) / T_STD; load() undoes that with the
T_MEAN, T_STD and EPS stored in the checkpoint.
"""
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from app.config import DEVICE

DEFAULT_WEIGHTS = Path(__file__).resolve().parents[2] / "hf_weights" / "umap_generator.pt.zip"


class Block(nn.Module):
    def __init__(self, i, o):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.GELU(), nn.Conv2d(o, o, 3, padding=1), nn.GELU())

    def forward(self, x):
        return self.net(x)


class UMapNet(nn.Module):
    """Small U-Net: noisy RGB crop -> one channel of normalised log-variance."""

    def __init__(self, b=32):
        super().__init__()
        self.e1, self.e2, self.e3, self.mid = Block(3, b), Block(b, 2 * b), Block(2 * b, 4 * b), Block(4 * b, 8 * b)
        self.d3, self.d2, self.d1 = Block(12 * b, 4 * b), Block(6 * b, 2 * b), Block(3 * b, b)
        self.out = nn.Conv2d(b, 1, 1)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x = x - 0.5
        e1 = self.e1(x)
        e2 = self.e2(self.pool(e1))
        e3 = self.e3(self.pool(e2))
        m = self.mid(self.pool(e3))
        up = lambda t: F.interpolate(t, scale_factor=2, mode="bilinear", align_corners=False)
        d3 = self.d3(torch.cat([up(m), e3], 1))
        d2 = self.d2(torch.cat([up(d3), e2], 1))
        d1 = self.d1(torch.cat([up(d2), e1], 1))
        return self.out(d1)


def load(weights_path=DEFAULT_WEIGHTS):
    """Returns model_fn: (N, 3, H, W) in [0, 1] -> (N, 1, H, W) u-map in 0-255 units.

    H and W must be multiples of 8 (three 2x poolings).
    The network runs on config.DEVICE; model_fn takes and returns CPU tensors.
    """
    ckpt = torch.load(weights_path, map_location="cpu", weights_only=True)
    net = UMapNet(ckpt["BASE"])
    net.load_state_dict({k.removeprefix("module."): v for k, v in ckpt["model"].items()}, strict=True)
    net = net.float().eval().to(DEVICE)
    t_mean, t_std, eps = ckpt["T_MEAN"], ckpt["T_STD"], ckpt["EPS"]

    def model_fn(x):
        if x.shape[-2] % 8 or x.shape[-1] % 8:
            raise ValueError(f"UMapNet needs H and W to be multiples of 8, got {tuple(x.shape[-2:])}")
        # Same as the notebook's denorm_u, which works in float64; float32 differs by far less than a grey level.
        return torch.sqrt(torch.clamp(torch.exp(net(x.to(DEVICE)) * t_std + t_mean) - eps, min=0.0)).cpu()

    return model_fn

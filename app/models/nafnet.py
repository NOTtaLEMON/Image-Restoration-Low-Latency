"""NAFNet (width 32) trained on SIDD: noisy tile -> denoised tile.

Settings from NAFNet_SIDD_colab.ipynb (= the repo's options/train/SIDD/NAFNet-width32.yml).
The notebook tests with plain NAFNet, not NAFNetLocal (TLC), so this does too.

Not cross-checked: the notebook saved no PSNR, so 44.02 dB (noisy 28.78 dB) on the 24 SIDD val
crops has no reference to compare against. The 240 test crops were not checked (no local GT).
"""
from pathlib import Path

import torch

from app.models._nafnet_arch.NAFNet_arch import NAFNet

DEFAULT_WEIGHTS = Path(__file__).resolve().parents[2] / "hf_weights" / "nafnet.pth.zip"
NET = {"width": 32, "enc_blk_nums": [2, 2, 4, 8], "middle_blk_num": 12, "dec_blk_nums": [2, 2, 2, 2]}


def load(weights_path=DEFAULT_WEIGHTS):
    """Returns model_fn: (N, 3, H, W) in [0, 1] -> (N, 3, H, W) in [0, 1]."""
    net = NAFNet(img_channel=3, **NET)
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    state = state.get("params", state)
    net.load_state_dict({k.removeprefix("module."): v for k, v in state.items()}, strict=True)
    net = net.float().eval()

    def model_fn(x):
        return net(x).clamp(0, 1)

    return model_fn

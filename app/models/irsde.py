"""IR-SDE (ConditionalNAFNet denoiser) trained on SIDD: noisy tile -> one restored sample.

Settings from IRSDE_SIDD_colab.ipynb. The noisy tile is the SDE mean mu; sampling starts from
mu plus max_sigma noise and runs the stochastic reverse SDE, as in the notebook's Cell 7.
"""
from pathlib import Path

import torch

from app.models._irsde_arch.DenoisingNAFNet_arch import ConditionalNAFNet
from app.models._irsde_arch.sde import IRSDE

DEFAULT_WEIGHTS = Path(__file__).resolve().parents[2] / "hf_weights" / "irsde.pth.zip"
SDE = {"max_sigma": 50, "T": 100, "schedule": "cosine", "eps": 0.005}
NET = {"width": 64, "enc_blk_nums": [1, 1, 1, 28], "middle_blk_num": 1, "dec_blk_nums": [1, 1, 1, 1]}


def load(weights_path=DEFAULT_WEIGHTS, steps=SDE["T"]):
    """Returns model_fn: (N, 3, H, W) noisy in [0, 1] -> (N, 3, H, W) restored in [0, 1].

    Output is stochastic (one reverse-SDE sample per call); seed torch for repeatable results.

    KNOWN ISSUE, steps < 100: the schedule is rebuilt for `steps` steps and the network is told
    the matching time on its 1..100 scale. This does not work: on SIDD val crop
    SIDD_0192_010_val_00 (noisy 33.41 dB), steps=100 gave 41.12 dB but steps=25 gave 30.70 dB,
    worse than the input. A real few-step sampler is needed before lowering steps.

    Checked at steps=100 (CPU, 16 threads): 74 s per 256x256 tile. One sample at 41.12 dB fits
    notebook A's 20-sample mean of 44.53 dB on the same crop plus its ~1.65 grey-level spread.
    """
    if not 1 <= steps <= SDE["T"]:
        raise ValueError(f"steps must be between 1 and {SDE['T']}, got {steps}")
    net = ConditionalNAFNet(img_channel=3, upscale=1, **NET)
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    state = state.get("params", state.get("state_dict", state))
    net.load_state_dict({k.removeprefix("module."): v for k, v in state.items()}, strict=True)
    net = net.float().eval()

    sde = IRSDE(max_sigma=SDE["max_sigma"], T=steps, schedule=SDE["schedule"], eps=SDE["eps"])
    if steps == SDE["T"]:
        sde.set_model(net)
    else:
        scale = SDE["T"] / steps
        sde.set_model(lambda x, mu, t: net(x, mu, t * scale))

    def model_fn(x):
        sde.set_mu(x)
        return sde.reverse_sde(sde.noise_state(x)).clamp(0, 1)

    return model_fn

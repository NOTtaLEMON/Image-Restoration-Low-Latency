"""Settings shared by the pipeline and the model wrappers."""
import os

import torch

# Where the models run. Uses the GPU when one is visible (e.g. on Colab), else the CPU.
# Override with the RESTORE_DEVICE environment variable, e.g. RESTORE_DEVICE=cpu.
# Only the networks run on this device; tiling, gating and blending stay on the CPU.
DEVICE = os.environ.get("RESTORE_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")

TILE_SIZE = 256  # what IR-SDE was sampled at in its notebook
OVERLAP = 64
BATCH_SIZE = 8  # tiles per NAFNet / UMapNet call
# IR-SDE is ~100 network calls per window: batching buys little on CPU but a lot on a GPU.
IRSDE_BATCH_SIZE = 8 if DEVICE == "cuda" else 1
# IR-SDE measured at ~74 s per tile on CPU: a large photo with many flagged tiles can take minutes to hours.

# Which 64x64 patches get IR-SDE (see app/pipeline.py). Starting guesses; tune on real photos.
GATE_CUTOFF = 0.85  # relative: only patches ranked in the top 15% of the image (gate.py's default is 0.70)
UNCERTAINTY_FLOOR = 0.02  # absolute: NAFNet must change the patch by >= 0.02 (about 5 grey levels) on average
IRSDE_WINDOW = 128  # IR-SDE restores each flagged patch from a window this size (its training crop size)

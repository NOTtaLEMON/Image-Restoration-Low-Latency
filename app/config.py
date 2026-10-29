"""Tiling settings shared by the pipeline."""

TILE_SIZE = 256  # what IR-SDE was sampled at in its notebook
OVERLAP = 64
BATCH_SIZE = 8  # tiles per NAFNet / UMapNet call
IRSDE_BATCH_SIZE = 1  # IR-SDE is ~100 network calls per tile, so batching buys little on CPU
# IR-SDE measured at ~74 s per tile on CPU: a large photo with many flagged tiles can take minutes to hours.

# Which 64x64 patches get IR-SDE (see app/pipeline.py). Starting guesses; tune on real photos.
GATE_CUTOFF = 0.85  # relative: only patches ranked in the top 15% of the image (gate.py's default is 0.70)
UNCERTAINTY_FLOOR = 0.02  # absolute: NAFNet must change the patch by >= 0.02 (about 5 grey levels) on average
IRSDE_WINDOW = 128  # IR-SDE restores each flagged patch from a window this size (its training crop size)

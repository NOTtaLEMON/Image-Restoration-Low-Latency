"""Tiling settings shared by the pipeline."""

TILE_SIZE = 256  # what IR-SDE was sampled at in its notebook
OVERLAP = 64
BATCH_SIZE = 8  # tiles per NAFNet / UMapNet call
IRSDE_BATCH_SIZE = 1  # IR-SDE is ~100 network calls per tile, so batching buys little on CPU
# IR-SDE measured at ~74 s per tile on CPU: a large photo with many flagged tiles can take minutes to hours.

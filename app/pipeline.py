"""The restoration recipe: image in, u-map and restored image out.

Knows nothing about HTTP. The public side is numpy (float32 in [0, 1]): restore() takes an
HxWx3 image and yields an HxW u-map, then an HxWx3 restored image. Inside, everything is
(N, C, H, W) torch tensors, the convention of tiling.py and gate.py.

What this file expects from the other modules:
  app.models.{nafnet,umapnet,irsde}.load() -> model_fn, as described in tiling.py
      nafnet, irsde: (N, 3, H, W) -> (N, 3, H, W)
      umapnet:       (N, 3, H, W) -> (N, 1, H, W)   (raw output, already de-normalised)
  app.config: TILE_SIZE, OVERLAP, BATCH_SIZE, IRSDE_BATCH_SIZE
The patch size, fusion weights and gate cutoff come from app.gate.
"""
import torch

from app.config import BATCH_SIZE, IRSDE_BATCH_SIZE, OVERLAP, TILE_SIZE
from app.gate import (
    PATCH_SIZE,
    blend,
    compute_classical_signal,
    compute_patch_weights,
    fuse_signals,
    get_flagged_patch_mask,
    normalize_per_image,
    upsample_patch_weights,
)
from app.models import irsde, nafnet, umapnet
from app.tiling import (
    pad_to_multiple,
    run_model_on_tiles,
    split_into_tiles,
    stitch_tiles,
    tiled_inference,
    unpad,
)

_denoise = None  # NAFNet: noisy tile -> D_det tile
_umap = None  # UMapNet: noisy tile -> uncertainty tile
_diffuse = None  # IR-SDE: noisy tile -> D_gen tile


def load():
    """Load all three models into memory. Slow; call once at startup."""
    global _denoise, _umap, _diffuse
    _denoise = nafnet.load()
    _umap = umapnet.load()
    _diffuse = irsde.load()


def warm_up():
    """Push one blank tile through every model so the first real request is not the slow one.

    Each model is called directly: going through restore() would skip IR-SDE whenever the
    gate gives a blank image zero weight.
    """
    _require_loaded()
    blank = torch.zeros((1, 3, TILE_SIZE, TILE_SIZE))
    with torch.no_grad():
        _denoise(blank)
        _umap(blank)
        _diffuse(blank)


def restore(image):
    """Generator: yields ("umap", HxW array in [0, 1]) as soon as it exists, then
    ("restored", HxWx3 array in [0, 1]). Both have the input's height and width.
    """
    _require_loaded()
    x = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).contiguous()
    # Pad once to a multiple of the gate's patch size and work on that padded image throughout,
    # so the patch grid lines up with the pixels; crop back at the end.
    padded, original_size = pad_to_multiple(x, PATCH_SIZE)

    d_det = _tiled(padded, _denoise, BATCH_SIZE)
    u_learned = _tiled(padded, _umap, BATCH_SIZE)
    u_classical = compute_classical_signal(padded, d_det)
    combined = fuse_signals(normalize_per_image(u_learned), u_classical)
    yield "umap", unpad(combined, original_size)[0, 0].numpy()

    patch_weights = compute_patch_weights(combined, patch_size=PATCH_SIZE)
    flagged = get_flagged_patch_mask(patch_weights)[0, 0]
    if not flagged.any():
        yield "restored", _to_numpy(unpad(d_det, original_size))
        return

    d_gen = _diffuse_flagged(padded, flagged)
    weight_map = upsample_patch_weights(patch_weights, target_size=padded.shape[-2:])
    yield "restored", _to_numpy(unpad(blend(d_det, d_gen, weight_map), original_size))


def _tiled(padded, model_fn, batch_size):
    return tiled_inference(
        padded, model_fn, tile_size=TILE_SIZE, overlap=OVERLAP, pad_multiple=PATCH_SIZE, batch_size=batch_size
    )


def _diffuse_flagged(padded, flagged):
    """Run IR-SDE only on tiles that touch a flagged patch; everywhere else d_gen stays zero.

    A pixel with non-zero gate weight lies in a flagged patch, and every tile covering it
    overlaps that patch, so it is always covered by a tile that was run. Rounding the patch
    range outwards means a tile is never wrongly skipped, whatever its alignment.
    """
    tiles, coords = split_into_tiles(padded, TILE_SIZE, OVERLAP)
    keep = [
        i
        for i, ((y, x), tile) in enumerate(zip(coords, tiles))
        if flagged[
            y // PATCH_SIZE : -(-(y + tile.shape[-2]) // PATCH_SIZE),
            x // PATCH_SIZE : -(-(x + tile.shape[-1]) // PATCH_SIZE),
        ].any()
    ]
    outputs = run_model_on_tiles(_diffuse, [tiles[i] for i in keep], batch_size=IRSDE_BATCH_SIZE)
    return stitch_tiles(outputs, [coords[i] for i in keep], padded.shape[-2:], channels=3)


def _to_numpy(t):
    return t.clamp(0, 1)[0].permute(1, 2, 0).contiguous().numpy()


def _require_loaded():
    if _denoise is None:
        raise RuntimeError("Models are not loaded: call pipeline.load() first.")

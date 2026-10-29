"""The restoration recipe: image in, u-map and restored image out.

Knows nothing about HTTP. The public side is numpy (float32 in [0, 1]): restore() takes an
HxWx3 image and yields an HxW u-map, then an HxWx3 restored image. Inside, everything is
(N, C, H, W) torch tensors, the convention of tiling.py and gate.py.

What this file expects from the other modules:
  app.models.{nafnet,umapnet,irsde}.load() -> model_fn, as described in tiling.py
      nafnet, irsde: (N, 3, H, W) -> (N, 3, H, W)
      umapnet:       (N, 3, H, W) -> (N, 1, H, W)   (raw output, already de-normalised)
  app.config: TILE_SIZE, OVERLAP, BATCH_SIZE, IRSDE_BATCH_SIZE, GATE_CUTOFF, UNCERTAINTY_FLOOR,
              IRSDE_WINDOW
The patch size and fusion weights come from app.gate.

Where IR-SDE runs: only on flagged 64x64 PATCHES, not on whole tiles. A patch is flagged when
  1. its fused score ranks above GATE_CUTOFF among the image's patches (relative), and
  2. NAFNet changed it by at least UNCERTAINTY_FLOOR on average (absolute), so a clean image
     does not get IR-SDE just because some of its patches rank highest.
Each flagged patch is restored from an IRSDE_WINDOW-sized window around it (context for the
model), and only the patch's own 64x64 centre is kept.
"""
import logging
import time

import torch
import torch.nn.functional as F

from app.config import (
    DEVICE,
    BATCH_SIZE,
    GATE_CUTOFF,
    IRSDE_BATCH_SIZE,
    IRSDE_WINDOW,
    OVERLAP,
    TILE_SIZE,
    UNCERTAINTY_FLOOR,
)
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
from app.tiling import pad_to_multiple, run_model_on_tiles, tiled_inference, unpad

# uvicorn's logger, so the per-stage timings appear in the server's console / log file.
log = logging.getLogger("uvicorn.error")

_denoise = None  # NAFNet: noisy tile -> D_det tile
_umap = None  # UMapNet: noisy tile -> uncertainty tile
_diffuse = None  # IR-SDE: noisy window -> D_gen window


def load():
    """Load all three models into memory. Slow; call once at startup."""
    global _denoise, _umap, _diffuse
    _denoise = nafnet.load()
    _umap = umapnet.load()
    _diffuse = irsde.load()


def warm_up():
    """Push one blank tile through NAFNet and UMapNet so the first real request is not the slow one.

    IR-SDE is left out on purpose: one call is a full 100-step sampling run (about a minute per
    tile on CPU), which would delay startup far more than PyTorch's one-time setup it would save.
    """
    _require_loaded()
    blank = torch.zeros((1, 3, TILE_SIZE, TILE_SIZE))
    with torch.no_grad():
        _denoise(blank)
        _umap(blank)


def restore(image):
    """Generator: yields ("umap", HxW array in [0, 1]) as soon as it exists, then
    ("restored", HxWx3 array in [0, 1]). Both have the input's height and width.
    """
    _require_loaded()
    x = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).contiguous()
    # Pad once to a multiple of the gate's patch size and work on that padded image throughout,
    # so the patch grid lines up with the pixels; crop back at the end.
    padded, original_size = pad_to_multiple(x, PATCH_SIZE)

    t0 = time.perf_counter()
    d_det = _tiled(padded, _denoise, BATCH_SIZE)
    t_nafnet = time.perf_counter()
    u_learned = _tiled(padded, _umap, BATCH_SIZE)
    t_umapnet = time.perf_counter()
    u_classical = compute_classical_signal(padded, d_det)
    combined = fuse_signals(normalize_per_image(u_learned), u_classical)
    log.info(
        "%dx%d on %s: NAFNet %.1f s, UMapNet %.1f s",
        *original_size, DEVICE, t_nafnet - t0, t_umapnet - t_nafnet,
    )
    yield "umap", unpad(combined, original_size)[0, 0].numpy()

    patch_weights = compute_patch_weights(combined, patch_size=PATCH_SIZE, percentile_cutoff=GATE_CUTOFF)
    # Absolute floor: compute_classical_signal is normalised per image, so recompute the raw
    # mean |input - D_det| per patch and zero the weight of patches NAFNet barely changed.
    raw_change = F.avg_pool2d((padded - d_det).abs().mean(dim=1, keepdim=True), PATCH_SIZE)
    patch_weights = patch_weights * (raw_change >= UNCERTAINTY_FLOOR)

    flagged = get_flagged_patch_mask(patch_weights)[0, 0]
    if not flagged.any():
        log.info("IR-SDE: no flagged patches, skipped")
        yield "restored", _to_numpy(unpad(d_det, original_size))
        return

    t_irsde = time.perf_counter()
    d_gen = _diffuse_patches(padded, flagged, original_size)
    log.info(
        "IR-SDE: %d of %d patches in %.1f s",
        int(flagged.sum()), flagged.numel(), time.perf_counter() - t_irsde,
    )
    weight_map = upsample_patch_weights(patch_weights, target_size=padded.shape[-2:])
    yield "restored", _to_numpy(unpad(blend(d_det, d_gen, weight_map), original_size))


def _tiled(padded, model_fn, batch_size):
    return tiled_inference(
        padded, model_fn, tile_size=TILE_SIZE, overlap=OVERLAP, pad_multiple=PATCH_SIZE, batch_size=batch_size
    )


def _diffuse_patches(padded, flagged, original_size):
    """Run IR-SDE once per flagged patch and return D_gen; unflagged patches stay zero.

    blend() multiplies D_gen by the patch weight, which is zero exactly where no patch was
    restored, so the zeros never reach the output.
    """
    _, _, ph, pw = padded.shape
    h, w = original_size
    win_h, win_w = min(IRSDE_WINDOW, ph), min(IRSDE_WINDOW, pw)
    margin = (IRSDE_WINDOW - PATCH_SIZE) // 2

    windows, placements = [], []
    for row, col in flagged.nonzero().tolist():
        py, px = row * PATCH_SIZE, col * PATCH_SIZE
        if py >= h or px >= w:
            continue  # patch lies entirely in the padding, never shown
        # Centre the window on the patch, shifted back inside the image at the edges.
        wy = min(max(py - margin, 0), ph - win_h)
        wx = min(max(px - margin, 0), pw - win_w)
        windows.append(padded[:, :, wy : wy + win_h, wx : wx + win_w])
        placements.append((py, px, py - wy, px - wx))

    d_gen = torch.zeros_like(padded)
    outputs = run_model_on_tiles(_diffuse, windows, batch_size=IRSDE_BATCH_SIZE)
    for out, (py, px, oy, ox) in zip(outputs, placements):
        d_gen[:, :, py : py + PATCH_SIZE, px : px + PATCH_SIZE] = out[:, :, oy : oy + PATCH_SIZE, ox : ox + PATCH_SIZE]
    return d_gen


def _to_numpy(t):
    return t.clamp(0, 1)[0].permute(1, 2, 0).contiguous().numpy()


def _require_loaded():
    if _denoise is None:
        raise RuntimeError("Models are not loaded: call pipeline.load() first.")

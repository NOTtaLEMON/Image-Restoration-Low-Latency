"""The restoration recipe: image in, u-map and restored image out.

Knows nothing about HTTP. Images are float32 HxWx3 numpy arrays in [0, 1].

What this file expects from the other modules (they follow this, not the other way round):
  app.models.{nafnet,umapnet,irsde}.load() -> fn(tile)
      nafnet, irsde: HxWx3 float32 tile -> HxWx3 float32 tile
      umapnet:       HxWx3 float32 tile -> HxW float32 map (already de-normalised)
  app.tiling.tile_apply(image, fn, tile, stride, pad_multiple, needed=None)
      `needed(y, x)` is optional. It gets a tile's top-left corner in the input image's
      coordinates (so the padding must go on the bottom/right only) and returns False to
      skip the tile; a skipped tile passes through unchanged and fn is not called on it.
  app.gate.{classical_signal, rank_fuse, patch_gate}: as in tests/
  app.config: TILE, STRIDE, PAD_MULTIPLE, PATCH, CUTOFF_PERCENTILE, W_LEARNED, W_CLASSICAL
"""
import numpy as np

from app.config import (
    CUTOFF_PERCENTILE,
    PAD_MULTIPLE,
    PATCH,
    STRIDE,
    TILE,
    W_CLASSICAL,
    W_LEARNED,
)
from app.gate import classical_signal, patch_gate, rank_fuse
from app.models import irsde, nafnet, umapnet
from app.tiling import tile_apply

_denoise = None  # NAFNet: noisy tile -> D_det tile
_umap = None  # UMapNet: noisy tile -> uncertainty map
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
    blank = np.zeros((TILE, TILE, 3), dtype=np.float32)
    _denoise(blank)
    _umap(blank)
    _diffuse(blank)


def restore(image):
    """Generator: yields ("umap", HxW array in [0, 1]) as soon as it exists, then
    ("restored", HxWx3 array in [0, 1]). Both have the input's height and width.
    """
    _require_loaded()
    h, w, _ = image.shape

    d_det = _tiled(image, _denoise)
    # tile_apply only takes 3-channel tiles, so the 1-channel map is repeated and one channel kept
    u_learned = _tiled(image, _umap_as_rgb)[..., 0]
    u_classical = classical_signal(image, d_det)
    u_map = rank_fuse(u_learned, u_classical, W_LEARNED, W_CLASSICAL)
    yield "umap", u_map

    # patch_gate needs a multiple of PATCH; the padded rows/columns are cropped off again below
    weights = patch_gate(_pad_to_multiple(u_map, PATCH), PATCH, CUTOFF_PERCENTILE)
    if not weights.any():
        yield "restored", np.clip(d_det, 0, 1)
        return

    # Tiles start on multiples of STRIDE (a multiple of PATCH) and span whole patches, so a pixel
    # with non-zero weight is only ever covered by tiles that are run, never by a skipped one.
    def needed(y, x):
        return weights[y // PATCH : -(-(y + TILE) // PATCH), x // PATCH : -(-(x + TILE) // PATCH)].any()

    d_gen = _tiled(image, _diffuse, needed=needed)
    w_pix = np.repeat(np.repeat(weights, PATCH, axis=0), PATCH, axis=1)[:h, :w, None]
    yield "restored", np.clip((1 - w_pix) * d_det + w_pix * d_gen, 0, 1).astype(np.float32)


def _tiled(image, fn, **kwargs):
    return tile_apply(image, fn, tile=TILE, stride=STRIDE, pad_multiple=PAD_MULTIPLE, **kwargs)


def _umap_as_rgb(tile):
    return np.repeat(_umap(tile)[..., None], 3, axis=-1)


def _pad_to_multiple(a, m):
    return np.pad(a, ((0, -a.shape[0] % m), (0, -a.shape[1] % m)), mode="reflect")


def _require_loaded():
    if _denoise is None:
        raise RuntimeError("Models are not loaded: call pipeline.load() first.")

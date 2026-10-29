"""
tests/test_tiling.py

Tests for app.tiling module: tile_apply function for tiled inference.

All tests use synthetic numpy/torch arrays only. No weights, no GPU, no downloads.

Assumptions:
1. Tiles are 256x256, stride is 192 (64-pixel overlap).
2. Overlapping regions use simple averaging (plain window).
3. Input shape is preserved exactly in the output.
4. Padding to multiples of 64 is handled internally and removed from output.
"""

import numpy as np
import pytest

from app.tiling import tile_apply


# -- helpers ------------------------------------------------------------------

def make_image(h, w, value=0.5):
    """Create a numpy HxWx3 float32 test image."""
    return np.full((h, w, 3), value, dtype=np.float32)


def offset_per_tile(step):
    """
    Function factory: returns a fn that outputs constant tiles.
    Each call returns a tile filled with step * call_count, so adjacent tiles
    in a row (or column) differ by exactly `step`.
    """
    counter = {"n": 0}

    def fn(t):
        val = counter["n"] * step
        counter["n"] += 1
        return np.full_like(t, val)

    return fn


# -- identity round-trip -------------------------------------------------------

@pytest.mark.parametrize("h,w", [
    (256, 256),
    (300, 517),
    (1024, 768),
    (100, 100),
])
def test_identity_roundtrip(h, w):
    """fn = identity must return the input within float tolerance."""
    img = np.random.default_rng(42).random((h, w, 3)).astype(np.float32)
    out = tile_apply(img, fn=lambda t: t)
    assert out.shape == img.shape, f"shape changed: {img.shape} -> {out.shape}"
    np.testing.assert_allclose(
        out, img, atol=1e-5,
        err_msg="Identity fn changed pixel values beyond tolerance."
    )


# -- output shape is preserved -----------------------------------------------

@pytest.mark.parametrize("h,w", [
    (256, 256),
    (300, 517),
    (1024, 768),
    (100, 100),
])
def test_shape_preserved(h, w):
    """Output shape must exactly equal input shape (padding is removed)."""
    img = make_image(h, w)
    out = tile_apply(img, fn=lambda t: t)
    assert out.shape == (h, w, 3), \
        f"Expected shape ({h}, {w}, 3), got {out.shape}"


# -- what fn receives and returns ----------------------------------------------

def test_fn_receives_full_size_tiles():
    """fn must only ever see (256, 256, 3) float32 tiles, never smaller."""
    seen = []

    def fn(t):
        seen.append((t.shape, t.dtype))
        return t

    img = np.random.default_rng(1).random((300, 517, 3)).astype(np.float32)
    tile_apply(img, fn=fn, tile=256, stride=192)
    assert seen, "fn was never called"
    assert all(s == ((256, 256, 3), np.float32) for s in seen), \
        f"fn received unexpected tiles: {set(seen)}"


def test_output_is_float32():
    """Output dtype must be float32."""
    img = np.random.default_rng(2).random((100, 100, 3)).astype(np.float32)
    out = tile_apply(img, fn=lambda t: t)
    assert out.dtype == np.float32


def test_input_is_not_modified():
    """tile_apply must not modify the input array in-place."""
    img = np.random.default_rng(3).random((300, 517, 3)).astype(np.float32)
    before = img.copy()
    tile_apply(img, fn=lambda t: t * 0.5)
    np.testing.assert_array_equal(img, before)


# -- every pixel is covered (no holes) ----------------------------------------

@pytest.mark.parametrize("h,w", [
    (256, 256),
    (300, 517),
    (100, 100),
    (1024, 768),
])
def test_every_pixel_covered(h, w):
    """
    fn that returns all-ones must produce all-ones in output.
    If any pixel had zero weight it would be NaN or 0, not 1.
    """
    img = make_image(h, w, value=0.0)
    out = tile_apply(img, fn=lambda t: np.ones_like(t))
    assert out.shape == (h, w, 3)
    np.testing.assert_allclose(
        out, 1.0, atol=1e-5,
        err_msg="Some pixels not covered (output != 1 where all-ones fn was used)."
    )


# -- overlap blending coverage ------------------------------------------------
#
# With plain averaging, the blend in overlaps is simple. We check:
# 1. Output stays within the range of its inputs (no extrapolation).
# 2. The output actually uses all tiles (later tiles contribute).
# This is weaker than testing smoothness, but matches the actual implementation.

STEP = 0.1
MIN_LATER_TILES = 1.5 * STEP  # If tile 0 is 0.0, tile 1 is 0.1, tile 2 is 0.2, we should see ~0.2


def test_blending_preserves_range_horizontal():
    """
    One row of tiles (H=256). Output values must not exceed the range of
    input tile values. With plain averaging, min and max are preserved.
    """
    img = make_image(256, 517, value=0.0)
    out = tile_apply(img, fn=offset_per_tile(STEP), tile=256, stride=192)
    assert out.shape == (256, 517, 3)
    # Tile values: 0.0, 0.1, 0.2. Output should be in [0, 0.2].
    assert float(out.min()) >= -1e-5
    assert float(out.max()) <= 0.2 + 1e-5
    assert float(out.max()) > MIN_LATER_TILES, "Later tiles do not appear to contribute."


def test_blending_preserves_range_vertical():
    """
    One column of tiles (W=256). Output values must not exceed the range of
    input tile values.
    """
    img = make_image(517, 256, value=0.0)
    out = tile_apply(img, fn=offset_per_tile(STEP), tile=256, stride=192)
    assert out.shape == (517, 256, 3)
    # Tile values: 0.0, 0.1, 0.2. Output should be in [0, 0.2].
    assert float(out.min()) >= -1e-5
    assert float(out.max()) <= 0.2 + 1e-5
    assert float(out.max()) > MIN_LATER_TILES, "Later tiles do not appear to contribute."


# -- invalid arguments -------------------------------------------------------

def test_stride_equal_to_tile_raises():
    """stride == tile has no overlap: must raise."""
    img = make_image(300, 300)
    with pytest.raises((ValueError, AssertionError)):
        tile_apply(img, fn=lambda t: t, tile=256, stride=256)


def test_stride_gt_tile_raises():
    """stride > tile leaves gaps: must raise."""
    img = make_image(300, 300)
    with pytest.raises((ValueError, AssertionError)):
        tile_apply(img, fn=lambda t: t, tile=256, stride=300)


def test_non_3channel_raises():
    """Non-3-channel input must raise."""
    img = np.zeros((100, 100, 1), dtype=np.float32)
    with pytest.raises((ValueError, AssertionError)):
        tile_apply(img, fn=lambda t: t)


def test_2d_input_raises():
    """2-D input (no channel dim) must raise."""
    img = np.zeros((100, 100), dtype=np.float32)
    with pytest.raises((ValueError, AssertionError)):
        tile_apply(img, fn=lambda t: t)
"""Tests for the torch-based app.tiling module.

These tests use only synthetic CPU tensors; no weights, GPU, downloads, or
real images are needed.
"""

import pytest
import torch

from app.tiling import (
    pad_to_multiple,
    run_model_on_tiles,
    split_into_tiles,
    stitch_tiles,
    tiled_inference,
    unpad,
)


def make_image(height, width, value=0.5):
    """Create a float32 image tensor with shape (1, 3, H, W)."""
    return torch.full((1, 3, height, width), value, dtype=torch.float32)


def random_image(height, width, seed=0):
    """Create a deterministic float32 tensor with values in [0, 1]."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.rand((1, 3, height, width), generator=generator, dtype=torch.float32)


def identity_model(batch):
    """Model callback that returns each tile unchanged."""
    return batch


@pytest.mark.parametrize(
    "height,width",
    [
        (256, 256),
        (300, 517),
        (1024, 768),
        (100, 100),
    ],
)
def test_identity_round_trip(height, width):
    """Tiled inference with an identity callback preserves all pixels."""
    image = random_image(height, width, seed=42)
    output = tiled_inference(
        image,
        model_fn=identity_model,
        tile_size=256,
        overlap=64,
        pad_multiple=64,
        batch_size=4,
    )

    assert output.shape == image.shape
    torch.testing.assert_close(output, image, atol=1e-5, rtol=0.0)


@pytest.mark.parametrize(
    "height,width",
    [
        (256, 256),
        (300, 517),
        (1024, 768),
        (100, 100),
    ],
)
def test_output_shape_matches_input(height, width):
    """Tiled inference removes padding and returns the original H and W."""
    image = make_image(height, width)
    output = tiled_inference(
        image,
        model_fn=identity_model,
        tile_size=256,
        overlap=64,
        pad_multiple=64,
    )

    assert output.shape == (1, 3, height, width)


@pytest.mark.parametrize(
    "height,width",
    [
        (256, 256),
        (300, 517),
        (100, 100),
        (1024, 768),
    ],
)
def test_every_pixel_is_covered(height, width):
    """A callback returning ones must produce ones at every output pixel."""
    image = make_image(height, width, value=0.0)
    output = tiled_inference(
        image,
        model_fn=torch.ones_like,
        tile_size=256,
        overlap=64,
        pad_multiple=64,
        batch_size=4,
    )

    assert output.shape == image.shape
    torch.testing.assert_close(output, torch.ones_like(image), atol=1e-5, rtol=0.0)


@pytest.mark.parametrize(
    "height,width",
    [
        (256, 256),
        (300, 517),
        (1024, 768),
        (100, 100),
    ],
)
def test_split_tiles_cover_padded_image(height, width):
    """Every padded-image pixel is covered by one or more returned tiles."""
    image = random_image(height, width, seed=7)
    padded, _ = pad_to_multiple(image, multiple=64)
    tile_size = 256
    overlap = 64
    tiles, coords = split_into_tiles(padded, tile_size=tile_size, overlap=overlap)

    assert tiles, "split_into_tiles returned no tiles"
    assert len(tiles) == len(coords)

    padded_height, padded_width = padded.shape[-2:]
    coverage = torch.zeros((padded_height, padded_width), dtype=torch.int32)
    for tile, (top, left) in zip(tiles, coords):
        tile_height, tile_width = tile.shape[-2:]
        assert 0 <= top <= padded_height - tile_height
        assert 0 <= left <= padded_width - tile_width
        coverage[top : top + tile_height, left : left + tile_width] += 1

        # When both spatial dimensions can hold a full tile, no edge tile
        # should be smaller: the last tile is shifted back inside the image.
        if padded_height >= tile_size and padded_width >= tile_size:
            assert tile.shape == (1, 3, tile_size, tile_size)

    assert bool(torch.all(coverage > 0)), "At least one padded-image pixel is uncovered"


def test_plain_average_stays_between_tile_constants_horizontal():
    """Plain overlap averaging stays in the range of all horizontal tile values."""
    image = make_image(256, 517, value=0.0)  # padded width 576 => three tiles
    step = 0.1
    counter = {"next": 0}

    def constant_per_tile(batch):
        outputs = []
        for tile in batch:
            value = counter["next"] * step
            counter["next"] += 1
            outputs.append(torch.full_like(tile, value))
        return torch.stack(outputs, dim=0)

    output = tiled_inference(
        image,
        model_fn=constant_per_tile,
        tile_size=256,
        overlap=64,
        pad_multiple=64,
        batch_size=8,
    )

    assert float(output.min()) >= -1e-6
    assert float(output.max()) <= 2 * step + 1e-6


def test_plain_average_stays_between_tile_constants_vertical():
    """Plain overlap averaging stays in the range of all vertical tile values."""
    image = make_image(517, 256, value=0.0)  # padded height 576 => three tiles
    step = 0.1
    counter = {"next": 0}

    def constant_per_tile(batch):
        outputs = []
        for tile in batch:
            value = counter["next"] * step
            counter["next"] += 1
            outputs.append(torch.full_like(tile, value))
        return torch.stack(outputs, dim=0)

    output = tiled_inference(
        image,
        model_fn=constant_per_tile,
        tile_size=256,
        overlap=64,
        pad_multiple=64,
        batch_size=8,
    )

    assert float(output.min()) >= -1e-6
    assert float(output.max()) <= 2 * step + 1e-6


@pytest.mark.parametrize("height,width", [(300, 517), (1024, 768), (100, 100)])
def test_pad_to_multiple_and_unpad_round_trip(height, width):
    """Padding reaches the requested multiple and unpadding restores the image."""
    image = random_image(height, width, seed=12)
    padded, original_size = pad_to_multiple(image, multiple=64)

    assert original_size == (height, width)
    assert padded.shape[-2] % 64 == 0
    assert padded.shape[-1] % 64 == 0
    # pad_to_multiple pads only bottom/right, so the original area is unchanged.
    torch.testing.assert_close(
        padded[:, :, :height, :width], image, atol=0.0, rtol=0.0
    )

    restored = unpad(padded, original_size)
    assert restored.shape == image.shape
    torch.testing.assert_close(restored, image, atol=0.0, rtol=0.0)


def test_stitch_unmodified_tiles_reconstructs_padded_image():
    """Stitching tiles cut from an image reconstructs that padded image."""
    image = random_image(320, 576, seed=21)
    tiles, coords = split_into_tiles(image, tile_size=256, overlap=64)

    stitched = stitch_tiles(
        tiles,
        coords,
        padded_size=image.shape[-2:],
        channels=image.shape[1],
    )

    assert stitched.shape == image.shape
    torch.testing.assert_close(stitched, image, atol=1e-5, rtol=0.0)


def test_single_channel_model_output_keeps_input_size():
    """UMapNet maps 3 channels to 1; the output channel count must follow the model."""
    image = random_image(300, 517, seed=61)
    output = tiled_inference(
        image,
        model_fn=lambda batch: batch.mean(dim=1, keepdim=True),
        tile_size=256,
        overlap=64,
        pad_multiple=64,
    )

    assert output.shape == (1, 1, 300, 517)
    torch.testing.assert_close(output, image.mean(dim=1, keepdim=True), atol=1e-5, rtol=0.0)


def test_run_model_on_tiles_keeps_order_when_batch_size_does_not_divide_count():
    tiles = [torch.full((1, 3, 8, 8), float(i)) for i in range(5)]
    outputs = run_model_on_tiles(lambda batch: batch + 100, tiles, batch_size=2)

    assert len(outputs) == 5
    for i, tile in enumerate(outputs):
        assert tile.shape == (1, 3, 8, 8)
        assert bool(torch.all(tile == i + 100))


def test_stitching_a_subset_of_tiles_leaves_the_rest_zero():
    """pipeline.py runs IR-SDE on only some tiles and stitches just those."""
    image = random_image(576, 576, seed=71)
    tiles, coords = split_into_tiles(image, tile_size=256, overlap=64)
    keep = [i for i, origin in enumerate(coords) if origin == (0, 0)]

    stitched = stitch_tiles(
        [tiles[i] for i in keep],
        [coords[i] for i in keep],
        padded_size=image.shape[-2:],
        channels=3,
    )

    torch.testing.assert_close(stitched[:, :, :256, :256], image[:, :, :256, :256], atol=1e-5, rtol=0.0)
    assert bool(torch.all(stitched[:, :, 256:, :] == 0))
    assert bool(torch.all(stitched[:, :, :, 256:] == 0))


def test_tiled_inference_does_not_modify_input():
    """Running tiled inference must not mutate the input tensor in place."""
    image = random_image(300, 517, seed=31)
    before = image.clone()

    tiled_inference(
        image,
        model_fn=lambda batch: batch * 0.5,
        tile_size=256,
        overlap=64,
        pad_multiple=64,
    )

    torch.testing.assert_close(image, before, atol=0.0, rtol=0.0)

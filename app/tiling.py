"""
tiling.py
=========
Generic, model-agnostic tiled inference utilities.

WHY THIS EXISTS
---------------
NAFNet / IRSDE / the uncertainty predictor were all trained on small crops
(e.g. 256x256), but the product requirement is: restore the FULL phone photo,
at its FULL original dimensions, with no cropping and no content loss.

These models are fully convolutional, so they can technically run on any
input size -- but a full-resolution phone photo (e.g. 4000x3000) may be too
slow/expensive to run through in one shot, especially on CPU. The fix is
TILING, not cropping: split the whole image into overlapping tiles at a
model-friendly size, run the model on every tile (so 100% of the image is
covered), then stitch the tiles back together with overlap-averaging to hide
seams.

This module does NOT know anything about NAFNet, IRSDE, or the uncertainty
predictor specifically. It just needs a plain Python callable with this
contract:

    model_fn(tensor) -> tensor
        tensor: torch.FloatTensor, shape (N, C, H, W), values in [0, 1]
        returns: torch.FloatTensor, shape (N, C, H*scale, W*scale)

Whoever writes nafnet.py / irsde.py / umapnet.py should expose a function or
a __call__ method that satisfies this contract. If a model's native forward()
doesn't match (different value range, extra args, etc.), wrap it in a small
lambda or helper function in pipeline.py before passing it in here -- don't
change this file to special-case a particular model.

USAGE (from pipeline.py):

    from tiling import tiled_inference

    restored = tiled_inference(
        image,                  # (1, 3, H, W) float tensor, values in [0, 1]
        model_fn=nafnet_model,  # callable matching the contract above
        tile_size=256,          # match whatever NAFNet/IRSDE/umapnet expect
        overlap=16,
        pad_multiple=16,        # CONFIRM this against the actual model code
        scale=1,                # 1 for denoising; >1 only if doing SR
        batch_size=8,           # how many tiles to run through the model at once
    )
    # restored.shape == (1, 3, H, W) -- exact same H, W as the input, always.
"""

import torch
import torch.nn.functional as F


def pad_to_multiple(image: torch.Tensor, multiple: int):
    """
    Reflection-pads an image so both H and W are exact multiples of `multiple`.

    This matters because models with N encoder downsampling stages reduce
    spatial size by 2^N internally; if the input isn't a multiple of 2^N,
    the decoder's skip connections can mismatch by a pixel when concatenated.

    Returns:
        padded_image: the padded tensor
        original_size: (H, W) of the input BEFORE padding, so the caller can
                        crop back to the exact original dimensions afterward.
    """
    _, _, h, w = image.shape
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    # Pad only on the bottom/right so top-left tile coordinates stay simple.
    padded = F.pad(image, (0, pad_w, 0, pad_h), mode="reflect")
    return padded, (h, w)


def unpad(image: torch.Tensor, original_size: tuple, scale: int = 1):
    """
    Crops a padded (and possibly upscaled) image back down to the exact
    original input dimensions, scaled if needed.

    `scale` must match whatever scale the model applied (1 for denoising,
    4 for a x4 super-resolution model, etc.) so the crop target is correct.
    """
    h, w = original_size
    return image[:, :, : h * scale, : w * scale]


def split_into_tiles(image: torch.Tensor, tile_size: int, overlap: int):
    """
    Splits a (padded) image into overlapping square tiles.

    Returns:
        tiles: list of tensors, each (1, C, tile_size, tile_size)
        coords: list of (y_start, x_start) giving each tile's top-left
                position in the PADDED image -- needed later for stitching.
    """
    _, _, h, w = image.shape
    stride = tile_size - overlap
    tiles, coords = [], []

    y = 0
    while True:
        x = 0
        while True:
            y_end = min(y + tile_size, h)
            x_end = min(x + tile_size, w)
            y_start = max(0, y_end - tile_size)
            x_start = max(0, x_end - tile_size)

            tile = image[:, :, y_start:y_end, x_start:x_end]
            tiles.append(tile)
            coords.append((y_start, x_start))

            if x_end >= w:
                break
            x += stride
        if y_end >= h:
            break
        y += stride

    return tiles, coords


def run_model_on_tiles(model_fn, tiles, batch_size: int = 8):
    """
    Runs `model_fn` over all tiles, batching them for throughput instead of
    looping one tile at a time (meaningfully faster on both CPU and GPU --
    see the latency discussion this was designed around).

    Assumes every tile is already the same spatial size (true for every tile
    except possibly a few at the image's bottom/right edge if tile_size
    doesn't evenly divide the padded dimensions -- pad_to_multiple should
    prevent that in practice, but this function will still work correctly
    even if some tiles differ in size, just at reduced batching efficiency
    for that batch).

    Returns a list of output tiles, in the same order as the input tiles.
    """
    outputs = []
    with torch.no_grad():
        for i in range(0, len(tiles), batch_size):
            batch = tiles[i : i + batch_size]
            # If tile sizes differ, fall back to one-by-one for this batch.
            same_shape = all(t.shape == batch[0].shape for t in batch)
            if same_shape:
                stacked = torch.cat(batch, dim=0)
                out = model_fn(stacked)
                outputs.extend(torch.split(out, 1, dim=0))
            else:
                for t in batch:
                    outputs.append(model_fn(t))
    return outputs


def stitch_tiles(tiles, coords, padded_size: tuple, channels: int, scale: int = 1):
    """
    Reassembles output tiles into a full image using overlap-averaging:
    every pixel is the average of all tile predictions that covered it.
    This is what hides the seams between tiles.

    `padded_size` is the (H, W) of the PADDED image the tiles were cut from
    (before unpad() crops it back to the original size).
    """
    h, w = padded_size
    out_h, out_w = h * scale, w * scale

    canvas = torch.zeros((1, channels, out_h, out_w))
    weight = torch.zeros((1, channels, out_h, out_w))

    for tile_out, (y, x) in zip(tiles, coords):
        th, tw = tile_out.shape[-2:]
        oy, ox = y * scale, x * scale
        canvas[:, :, oy : oy + th, ox : ox + tw] += tile_out
        weight[:, :, oy : oy + th, ox : ox + tw] += 1

    return canvas / weight.clamp(min=1e-8)


def tiled_inference(
    image: torch.Tensor,
    model_fn,
    tile_size: int,
    overlap: int,
    pad_multiple: int,
    scale: int = 1,
    batch_size: int = 8,
):
    """
    Main entry point. Full pipeline: pad -> split -> batched inference ->
    stitch -> crop back to the EXACT original input dimensions.

    This is what guarantees the "output must match input dimensions"
    requirement: pad_to_multiple + unpad bracket everything else, so no
    matter what tile_size/overlap/padding happens internally, the returned
    image is always (1, C, H, W) with H, W identical to the input.

    Args:
        image: (1, C, H, W) float tensor, values in [0, 1].
        model_fn: callable matching the contract documented at the top of
                  this file. Pass in a wrapped nafnet/irsde/umapnet call.
        tile_size: must match what the given model was trained/validated on.
        overlap: pixels of overlap between adjacent tiles (try 16-32).
        pad_multiple: REQUIRED, no default on purpose -- set this to match
                      the specific model's downsampling depth (e.g. 2^(number
                      of encoder stages)). Get this value from whoever wrote
                      nafnet.py / irsde.py / umapnet.py; don't guess it here.
        scale: 1 for denoising (NAFNet, IRSDE, uncertainty predictor here).
               Only set >1 if this is ever reused for a super-resolution model.
        batch_size: how many tiles to run through the model in one forward call.

    Returns:
        (1, C, H*scale, W*scale) tensor -- same H, W as input when scale=1.
    """
    padded, original_size = pad_to_multiple(image, pad_multiple)
    tiles, coords = split_into_tiles(padded, tile_size, overlap)
    output_tiles = run_model_on_tiles(model_fn, tiles, batch_size=batch_size)

    channels = output_tiles[0].shape[1]
    padded_h, padded_w = padded.shape[-2:]
    stitched = stitch_tiles(output_tiles, coords, (padded_h, padded_w), channels, scale)

    return unpad(stitched, original_size, scale)

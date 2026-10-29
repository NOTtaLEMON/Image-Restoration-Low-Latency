"""
gate.py
=======
The gated fusion logic: decides, per patch, how much to trust the
deterministic restoration (NAFNet) vs. the generative restoration (IRSDE),
using a FUSED uncertainty signal rather than the raw uncertainty predictor
alone.

WHY FUSION, NOT JUST THE RAW PREDICTOR OUTPUT
----------------------------------------------
The uncertainty predictor (umapnet) was diagnosed as having a real but
limited signal: decent LOCAL ranking within an image (it can often tell
which regions of ONE image are relatively harder), but an unreliable
GLOBAL level (it tends to predict roughly the same overall uncertainty for
every image, and a simple classical noise measure currently outperforms it
on that). To compensate, this module combines the learned signal with a
free classical signal -- how much NAFNet's own output differs from the
input -- which is a decent proxy for "how hard was this region" and costs
nothing extra to compute (NAFNet's output is already available).

WHY A CONTINUOUS (SOFT) GATE, NOT A HARD FLAG
-----------------------------------------------
Because the learned signal is sometimes WRONG (inverted ranking was
observed on the worst test case), a hard "this patch uses IRSDE, that one
doesn't" decision has no protection against a wrong call -- a misranked
patch would be fully replaced by a possibly-bad generative guess. A
continuous blend weight bounds the damage: even a wrong gate decision only
partially shows up in the final output, instead of fully committing to it.
The deterministic output is always the floor; this gate can only pull a
patch TOWARD the generative output, never fully away from the safe baseline
unless the fused signal is confidently high.

THIS FILE DOES NOT CALL ANY MODELS.
------------------------------------
Everything here is plain tensor math on already-computed outputs:
  - the input image
  - D_det       : NAFNet's output (same shape as input)
  - U_learned   : the uncertainty predictor's raw output map
  - D_gen       : IRSDE's output, computed ONLY on flagged patches
                  (pipeline.py decides which patches to actually run IRSDE
                  on, using get_flagged_patch_mask() from this file, BEFORE
                  calling blend()).

Expected tensor shapes (all batch size 1 in this pipeline, but written to
tolerate batch > 1):
    input, D_det, D_gen : (N, C, H, W), values in [0, 1]
    U_learned           : (N, 1, H, W)  -- single-channel uncertainty map

USAGE (from pipeline.py), rough shape of the call sequence:

    from gate import (
        compute_classical_signal, normalize_per_image, fuse_signals,
        compute_patch_weights, upsample_patch_weights,
        get_flagged_patch_mask, blend,
    )

    u_learned_norm  = normalize_per_image(u_learned_raw)
    u_classical     = compute_classical_signal(input_img, d_det)   # already normalized
    combined        = fuse_signals(u_learned_norm, u_classical)
    patch_weights   = compute_patch_weights(combined, patch_size=PATCH_SIZE)

    flagged_mask = get_flagged_patch_mask(patch_weights)
    # -> pipeline.py runs IRSDE ONLY on the flagged patches/tiles, producing
    #    d_gen for those regions (and can leave the rest as zeros/unset --
    #    the blend below multiplies unflagged regions by weight 0 anyway).

    gate_weight_map = upsample_patch_weights(patch_weights, target_size=input_img.shape[-2:])
    final_output = blend(d_det, d_gen, gate_weight_map)
"""

import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Tunable parameters. Starting defaults based on the fusion design -- revisit
# once the uncertainty predictor is retrained with broader scene diversity
# (these weights specifically compensate for ITS CURRENT, measured weakness,
# not a permanent property of the system).
# ---------------------------------------------------------------------------
FUSION_WEIGHT_LEARNED = 0.3     # weight on the learned uncertainty predictor
FUSION_WEIGHT_CLASSICAL = 0.7   # weight on the classical |input - D_det| signal
PATCH_SIZE = 64                 # patch granularity for gating decisions
GATE_PERCENTILE_CUTOFF = 0.70   # patches below this percentile get weight 0


def normalize_per_image(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Min-max normalizes each image in a batch independently to [0, 1].

    This is used instead of trusting the raw magnitude of either signal,
    because the uncertainty predictor's absolute/global level is known to
    be unreliable (near-constant across different images). Per-image
    normalization throws away the (broken) absolute level and keeps only
    the (partially trustworthy) relative shape within that image.
    """
    n = x.shape[0]
    flat = x.view(n, -1)
    min_val = flat.min(dim=1, keepdim=True).values
    max_val = flat.max(dim=1, keepdim=True).values
    norm_flat = (flat - min_val) / (max_val - min_val + eps)
    return norm_flat.view_as(x)


def compute_classical_signal(input_img: torch.Tensor, d_det: torch.Tensor) -> torch.Tensor:
    """
    The free classical uncertainty proxy: how much did the deterministic
    model change each pixel? Large correction => proxy for "this region was
    hard". Averaged across color channels to produce a single-channel map,
    then normalized per-image the same way as the learned signal so the two
    are on a comparable footing before fusion.
    """
    diff = (input_img - d_det).abs().mean(dim=1, keepdim=True)  # (N, 1, H, W)
    return normalize_per_image(diff)


def _per_image_percentile_rank(x: torch.Tensor) -> torch.Tensor:
    """
    Converts each pixel's value into its percentile rank (0 to 1) WITHIN
    its own image, computed independently per image in the batch.

    Rank-based fusion is used instead of just averaging the two normalized
    signals directly, because percentile rank is robust to differences in
    how "peaked" vs "flat" each signal's distribution is -- a straight
    average can be dominated by whichever signal happens to have a wider
    numeric spread, which isn't necessarily the more informative one.
    """
    n = x.shape[0]
    flat = x.view(n, -1)
    # argsort twice gives each element's rank position in the sorted order.
    order = flat.argsort(dim=1)
    ranks = order.argsort(dim=1).float()
    ranks = ranks / (flat.shape[1] - 1)  # normalize rank position to [0, 1]
    return ranks.view_as(x)


def fuse_signals(
    u_learned_norm: torch.Tensor,
    u_classical_norm: torch.Tensor,
    weight_learned: float = FUSION_WEIGHT_LEARNED,
    weight_classical: float = FUSION_WEIGHT_CLASSICAL,
) -> torch.Tensor:
    """
    Combines the two normalized uncertainty signals by rank, weighted
    toward the classical signal by default -- reflecting the CURRENT,
    measured state where the classical measure outranks the learned model
    on overall level. Re-tune these weights once the predictor improves.
    """
    rank_learned = _per_image_percentile_rank(u_learned_norm)
    rank_classical = _per_image_percentile_rank(u_classical_norm)
    return weight_learned * rank_learned + weight_classical * rank_classical


def compute_patch_weights(
    combined_score: torch.Tensor,
    patch_size: int = PATCH_SIZE,
    percentile_cutoff: float = GATE_PERCENTILE_CUTOFF,
) -> torch.Tensor:
    """
    Downsamples the pixel-level combined score to a coarse patch grid
    (average-pooling each patch_size x patch_size block to one value), then
    maps that to a CONTINUOUS gate weight in [0, 1] via a percentile cutoff.

    Patches scoring below `percentile_cutoff` (relative to other patches in
    the SAME image) get weight 0 -- no generative compute spent on them.
    Patches above it get a weight that scales smoothly from 0 to 1 as their
    score rises further above the cutoff -- this is the "soft gate" that
    bounds the damage from a wrong ranking, as explained at the top of this
    file.

    Returns a (N, 1, H/patch_size, W/patch_size) tensor of patch weights.
    """
    patch_scores = F.avg_pool2d(combined_score, kernel_size=patch_size, stride=patch_size)

    n = patch_scores.shape[0]
    flat = patch_scores.view(n, -1)
    patch_ranks = _per_image_percentile_rank(flat.view(n, 1, -1, 1)).view_as(patch_scores)

    gate_weight = (patch_ranks - percentile_cutoff) / (1.0 - percentile_cutoff)
    return gate_weight.clamp(0.0, 1.0)


def upsample_patch_weights(patch_weights: torch.Tensor, target_size: tuple) -> torch.Tensor:
    """
    Expands the coarse per-patch gate weight grid back up to full pixel
    resolution (nearest-neighbor, so every pixel in a patch shares that
    patch's exact weight -- avoids inventing a fake smooth gradient across
    patch boundaries that isn't actually backed by the underlying signal).
    """
    return F.interpolate(patch_weights, size=target_size, mode="nearest")


def get_flagged_patch_mask(patch_weights: torch.Tensor, threshold: float = 0.0) -> torch.Tensor:
    """
    Boolean mask of which patches have ANY non-zero gate weight -- this is
    what pipeline.py should use to decide which tiles/patches actually need
    an IRSDE pass at all. Skipping IRSDE entirely on zero-weight patches is
    the main latency saving in this design; don't run IRSDE everywhere and
    then blend -- only run it where this mask is True.
    """
    return patch_weights > threshold


def blend(d_det: torch.Tensor, d_gen: torch.Tensor, gate_weight_map: torch.Tensor) -> torch.Tensor:
    """
    Final soft blend between the deterministic and generative outputs.

    Safe to call even if `d_gen` is garbage/zero/unset in regions where
    gate_weight_map is exactly 0 -- those regions are multiplied by 0 and
    the deterministic output passes through unchanged regardless of what's
    in d_gen there. This means pipeline.py does NOT need to carefully mask
    d_gen before calling this -- just make sure gate_weight_map is 0 for
    every patch IRSDE wasn't actually run on.
    """
    return (1.0 - gate_weight_map) * d_det + gate_weight_map * d_gen

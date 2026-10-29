"""Tests for the torch-based app.gate module.

All tests use synthetic CPU tensors; no model weights, GPU, or downloads are
required. Tests assert the specified behaviour without fixing the gate ramp
formula above the cutoff.
"""

import pytest
import torch

from app.gate import (
    blend,
    compute_classical_signal,
    compute_patch_weights,
    fuse_signals,
    get_flagged_patch_mask,
    normalize_per_image,
    upsample_patch_weights,
)


def make_nchw(height, width, value=0.5, batch=1, channels=3):
    """Create a float32 tensor with shape (N, C, H, W)."""
    return torch.full((batch, channels, height, width), value, dtype=torch.float32)


def random_nchw(height, width, seed=0, batch=1, channels=1):
    """Create a deterministic float32 tensor with values in [0, 1]."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.rand((batch, channels, height, width), generator=generator, dtype=torch.float32)


def patch_grid_4x4():
    """Create a 256x256 map with 16 distinct constant 64x64 patch scores."""
    combined = torch.zeros((1, 1, 256, 256), dtype=torch.float32)
    scores = torch.empty((4, 4), dtype=torch.float32)
    for row in range(4):
        for col in range(4):
            value = (row * 4 + col) * 0.1
            combined[:, :, row * 64 : (row + 1) * 64, col * 64 : (col + 1) * 64] = value
            scores[row, col] = value
    return combined, scores


class TestNormalizePerImage:
    def test_output_in_zero_one_and_shape_preserved(self):
        x = random_nchw(128, 160, seed=1, batch=2, channels=1)
        output = normalize_per_image(x)

        assert output.shape == x.shape
        assert torch.isfinite(output).all()
        assert float(output.min()) >= -1e-6
        assert float(output.max()) <= 1.0 + 1e-6
        for item in output:
            assert float(item.min()) == pytest.approx(0.0, abs=1e-6)
            assert float(item.max()) == pytest.approx(1.0, abs=1e-6)

    def test_constant_image_does_not_produce_nan_or_inf(self):
        x = make_nchw(128, 128, value=0.4, channels=1)
        output = normalize_per_image(x)

        assert torch.isfinite(output).all()
        assert float(output.min()) >= 0.0 - 1e-6
        assert float(output.max()) <= 1.0 + 1e-6


class TestComputeClassicalSignal:
    def test_identical_inputs_give_all_zeros(self):
        image = random_nchw(128, 128, seed=2, channels=3)
        output = compute_classical_signal(image, image.clone())

        assert output.shape == (1, 1, 128, 128)
        torch.testing.assert_close(output, torch.zeros_like(output), atol=1e-6, rtol=0.0)

    def test_output_shape_is_single_channel(self):
        input_img = random_nchw(128, 160, seed=3, channels=3)
        denoised = random_nchw(128, 160, seed=4, channels=3)
        output = compute_classical_signal(input_img, denoised)

        assert output.shape == (1, 1, 128, 160)
        assert torch.isfinite(output).all()
        assert float(output.min()) >= -1e-6
        assert float(output.max()) <= 1.0 + 1e-6

    def test_signal_is_symmetric_in_arguments(self):
        first = random_nchw(128, 128, seed=5, channels=3)
        second = random_nchw(128, 128, seed=6, channels=3)

        forward = compute_classical_signal(first, second)
        reverse = compute_classical_signal(second, first)
        torch.testing.assert_close(forward, reverse, atol=1e-6, rtol=0.0)

    def test_constant_difference_gives_finite_constant_map(self):
        output = compute_classical_signal(make_nchw(64, 64, value=0.8), make_nchw(64, 64, value=0.3))

        assert torch.isfinite(output).all()
        assert float(output.max() - output.min()) < 1e-6

    def test_channels_are_averaged_not_maxed(self):
        """A 0.9 difference in one channel normalizes to 1/3, not 1."""
        input_img = torch.zeros((1, 3, 128, 128), dtype=torch.float32)
        denoised = torch.zeros_like(input_img)

        # Pixel A: only channel 0 differs by 0.9; channel-mean difference = 0.3.
        denoised[0, 0, 0, 0] = 0.9
        # Pixel B: all three channels differ by 0.9; channel-mean difference = 0.9.
        denoised[0, :, 0, 1] = 0.9

        output = compute_classical_signal(input_img, denoised)

        assert float(output[0, 0, 0, 0]) == pytest.approx(1.0 / 3.0, abs=1e-5)
        assert float(output[0, 0, 0, 1]) == pytest.approx(1.0, abs=1e-6)
        assert float(output[0, 0, 10, 10]) == pytest.approx(0.0, abs=1e-6)


class TestFuseSignals:
    def test_weights_one_zero_depend_only_on_learned_rank(self):
        learned = random_nchw(128, 128, seed=10, channels=1)
        classical_a = random_nchw(128, 128, seed=11, channels=1)
        classical_b = random_nchw(128, 128, seed=12, channels=1)

        out_a = fuse_signals(learned, classical_a, weight_learned=1.0, weight_classical=0.0)
        out_b = fuse_signals(learned, classical_b, weight_learned=1.0, weight_classical=0.0)

        torch.testing.assert_close(out_a, out_b, atol=0.0, rtol=0.0)
        order = torch.argsort(learned.flatten())
        differences = torch.diff(out_a.flatten()[order])
        assert bool(torch.all(differences >= -1e-7))

    def test_weights_zero_one_depend_only_on_classical_rank(self):
        learned_a = random_nchw(128, 128, seed=20, channels=1)
        learned_b = random_nchw(128, 128, seed=21, channels=1)
        classical = random_nchw(128, 128, seed=22, channels=1)

        out_a = fuse_signals(learned_a, classical, weight_learned=0.0, weight_classical=1.0)
        out_b = fuse_signals(learned_b, classical, weight_learned=0.0, weight_classical=1.0)

        torch.testing.assert_close(out_a, out_b, atol=0.0, rtol=0.0)
        order = torch.argsort(classical.flatten())
        differences = torch.diff(out_a.flatten()[order])
        assert bool(torch.all(differences >= -1e-7))

    def test_output_is_invariant_to_strictly_increasing_transform(self):
        learned = random_nchw(128, 128, seed=30, channels=1)
        classical = random_nchw(128, 128, seed=31, channels=1)

        original = fuse_signals(learned, classical)
        transformed = fuse_signals(0.5 * learned, classical)
        torch.testing.assert_close(original, transformed, atol=0.0, rtol=0.0)

    def test_default_weights_are_0_3_learned_and_0_7_classical(self):
        """Learned rank 1 / classical rank 0 scores 0.3; learned 0 / classical 1 scores 0.7."""
        ramp = torch.arange(100, dtype=torch.float32).view(1, 1, 1, 100)
        output = fuse_signals(ramp, ramp.flip(-1))

        assert float(output[0, 0, 0, 99]) == pytest.approx(0.3, abs=1e-6)
        assert float(output[0, 0, 0, 0]) == pytest.approx(0.7, abs=1e-6)

    def test_output_shape_matches_inputs(self):
        learned = random_nchw(128, 160, seed=32, channels=1)
        classical = random_nchw(128, 160, seed=33, channels=1)
        output = fuse_signals(learned, classical)
        assert output.shape == learned.shape
        assert torch.isfinite(output).all()


class TestComputePatchWeights:
    def test_shape_and_range(self):
        combined = random_nchw(256, 320, seed=40, channels=1)
        output = compute_patch_weights(combined, patch_size=64)

        assert output.shape == (1, 1, 4, 5)
        assert torch.isfinite(output).all()
        assert float(output.min()) >= 0.0 - 1e-6
        assert float(output.max()) <= 1.0 + 1e-6

    def test_scores_above_cutoff_get_positive_weights_and_lower_scores_get_zero(self):
        combined, scores = patch_grid_4x4()
        cutoff = 0.70
        output = compute_patch_weights(combined, patch_size=64, percentile_cutoff=cutoff)
        weights = output[0, 0]

        # With 16 distinct patch scores, rank positions are 0/15, ..., 15/15.
        ordered = torch.argsort(scores.flatten())
        ranks = torch.empty_like(ordered, dtype=torch.float32)
        ranks[ordered] = torch.arange(16, dtype=torch.float32) / 15.0
        ranks = ranks.view(4, 4)
        below_or_at = ranks <= cutoff
        above = ranks > cutoff

        assert bool(torch.all(weights[below_or_at] == 0)), (
            "Patches whose rank is at or below the cutoff must have zero weight"
        )
        assert int(above.sum()) == 5  # sanity check for this fixed 4x4 distinct-score grid
        assert bool(torch.all(weights[above] > 0)), (
            "Every patch ranked above the cutoff should get a positive weight"
        )
        assert int((weights > 0).sum()) == int(above.sum()), (
            "Exactly patches above the cutoff should have non-zero weights"
        )

    def test_highest_scoring_patch_has_highest_weight(self):
        combined, _ = patch_grid_4x4()
        output = compute_patch_weights(combined, patch_size=64, percentile_cutoff=0.70)[0, 0]

        highest = output[3, 3]
        assert float(highest) > 0.0
        assert float(highest) == pytest.approx(float(output.max()), abs=1e-6)

    def test_top_patch_reaches_full_weight(self):
        """The ramp runs from 0 at the cutoff up to 1 for the highest-scoring patch."""
        combined, _ = patch_grid_4x4()
        output = compute_patch_weights(combined, patch_size=64, percentile_cutoff=0.70)[0, 0]

        assert float(output[3, 3]) == pytest.approx(1.0, abs=1e-6)

    def test_defaults_are_64_px_patches_and_cutoff_0_70(self):
        combined = random_nchw(256, 256, seed=42, channels=1)

        default = compute_patch_weights(combined)
        explicit = compute_patch_weights(combined, patch_size=64, percentile_cutoff=0.70)

        assert default.shape == (1, 1, 4, 4)
        torch.testing.assert_close(default, explicit, atol=0.0, rtol=0.0)

    def test_patch_score_is_the_mean_not_the_max(self):
        """A single bright pixel must not outrank a patch that is brighter on average."""
        combined = torch.zeros((1, 1, 128, 128), dtype=torch.float32)
        combined[:, :, :64, :64] = 0.5  # A: highest mean
        combined[0, 0, 0, 64] = 1.0  # B: one spike, lowest mean (highest max)
        combined[:, :, 64:, :64] = 0.2  # C
        combined[:, :, 64:, 64:] = 0.1  # D
        weights = compute_patch_weights(combined, patch_size=64, percentile_cutoff=0.70)[0, 0]

        assert float(weights[0, 0]) > 0.0
        assert float(weights[0, 1]) == 0.0

    def test_larger_patch_score_never_has_smaller_weight(self):
        combined, scores = patch_grid_4x4()
        output = compute_patch_weights(combined, patch_size=64, percentile_cutoff=0.70)[0, 0]
        order = torch.argsort(scores.flatten())
        sorted_weights = output.flatten()[order]

        assert bool(torch.all(torch.diff(sorted_weights) >= -1e-6))

    def test_constant_input_does_not_crash(self):
        """The specified range/finite invariants hold for a constant map."""
        combined = make_nchw(128, 128, value=0.5, channels=1)
        output = compute_patch_weights(combined, patch_size=64)

        assert output.shape == (1, 1, 2, 2)
        assert torch.isfinite(output).all()
        assert float(output.min()) >= 0.0 - 1e-6
        assert float(output.max()) <= 1.0 + 1e-6

    def test_custom_patch_size(self):
        combined = random_nchw(128, 192, seed=41, channels=1)
        output = compute_patch_weights(combined, patch_size=32)
        assert output.shape == (1, 1, 4, 6)


class TestUpsamplePatchWeights:
    def test_each_weight_is_repeated_over_its_patch_block(self):
        patch_weights = torch.tensor(
            [[[[0.0, 0.25, 0.5], [0.75, 1.0, 0.125]]]], dtype=torch.float32
        )
        output = upsample_patch_weights(patch_weights, target_size=(128, 192))

        assert output.shape == (1, 1, 128, 192)
        for row in range(2):
            for col in range(3):
                block = output[
                    0,
                    0,
                    row * 64 : (row + 1) * 64,
                    col * 64 : (col + 1) * 64,
                ]
                expected = patch_weights[0, 0, row, col]
                torch.testing.assert_close(block, expected.expand_as(block), atol=0.0, rtol=0.0)


class TestGetFlaggedPatchMask:
    def test_true_exactly_where_weight_exceeds_threshold(self):
        weights = torch.tensor(
            [[[[0.0, 0.1, 0.25], [0.3, 0.5, 1.0]]]], dtype=torch.float32
        )
        expected = torch.tensor(
            [[[[False, False, False], [True, True, True]]]], dtype=torch.bool
        )
        output = get_flagged_patch_mask(weights, threshold=0.25)

        assert output.dtype == torch.bool
        assert output.shape == weights.shape
        assert torch.equal(output, expected)


class TestBlend:
    def test_zero_weight_returns_deterministic_output_exactly(self):
        d_det = random_nchw(128, 128, seed=50, channels=3)
        d_gen = random_nchw(128, 128, seed=51, channels=3)
        weights = torch.zeros((1, 1, 128, 128), dtype=torch.float32)

        output = blend(d_det, d_gen, weights)
        torch.testing.assert_close(output, d_det, atol=0.0, rtol=0.0)

    def test_one_weight_returns_generative_output_exactly(self):
        d_det = random_nchw(128, 128, seed=52, channels=3)
        d_gen = random_nchw(128, 128, seed=53, channels=3)
        weights = torch.ones((1, 1, 128, 128), dtype=torch.float32)

        output = blend(d_det, d_gen, weights)
        torch.testing.assert_close(output, d_gen, atol=0.0, rtol=0.0)

    def test_garbage_in_zero_weight_region_does_not_change_output(self):
        d_det = random_nchw(128, 128, seed=54, channels=3)
        d_gen = random_nchw(128, 128, seed=55, channels=3)
        gate = torch.zeros((1, 1, 128, 128), dtype=torch.float32)
        gate[:, :, :, 64:] = 0.5

        zero_weight_region = gate.expand_as(d_gen) == 0
        d_gen[zero_weight_region] = 10_000.0  # finite but deliberately invalid values

        output = blend(d_det, d_gen, gate)
        torch.testing.assert_close(
            output[:, :, :, :64], d_det[:, :, :, :64], atol=0.0, rtol=0.0
        )

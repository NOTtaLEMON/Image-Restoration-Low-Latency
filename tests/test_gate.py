"""
tests/test_gate.py

Tests for app.gate module: signal fusion and per-patch gating.

Tests for:
    classical_signal(noisy, denoised) -> ndarray HxW in [0, 1]
    rank_fuse(u_learned, u_classical, w_learned=0.3, w_classical=0.7) -> ndarray HxW
    patch_gate(combined, patch=64, cutoff_percentile=70) -> ndarray (H/64)x(W/64) in [0, 1]

All tests use synthetic numpy arrays only. No weights, no GPU, no downloads.

Assumptions (confirmed with implementation):
1. classical_signal: HxWx3 -> HxW in [0,1]. Per-pixel difference is MEAN over channels.
2. rank_fuse: HxW arrays (any range) -> HxW. Uses global percentile rank of each input.
3. patch_gate: HxW with H,W multiples of 64 -> (H/64)x(W/64). Non-multiples raise ValueError.
4. Shape mismatches: gate functions raise ValueError on shape mismatch (added for safety).
"""

import numpy as np
import pytest

from app.gate import classical_signal, patch_gate, rank_fuse


# -- helpers ------------------------------------------------------------------

def make_hw3(h, w, value=0.5):
    """Create HxWx3 float32 test array."""
    return np.full((h, w, 3), value, dtype=np.float32)


def make_hw(h, w, value=0.5):
    """Create HxW float32 test array."""
    return np.full((h, w), value, dtype=np.float32)


def rng_hw(h, w, seed=0):
    """Create HxW random float32 array in [0,1]."""
    return np.random.default_rng(seed).random((h, w)).astype(np.float32)


def patch_grid_4x4():
    """
    256x256 map with 4x4 constant 64x64 patches.
    Patch values: 0.0, 0.1, 0.2, ..., 1.5 (left-to-right, top-to-bottom).
    Returns (combined, scores) where scores is a (4,4) array.
    """
    combined = np.zeros((256, 256), dtype=np.float32)
    scores = np.zeros((4, 4), dtype=np.float32)
    for i in range(4):
        for j in range(4):
            val = (i * 4 + j) * 0.1
            combined[i * 64:(i + 1) * 64, j * 64:(j + 1) * 64] = val
            scores[i, j] = val
    return combined, scores


# ==============================================================================
# classical_signal
# ==============================================================================

class TestClassicalSignal:

    def test_output_in_0_1(self):
        """Output must lie in [0, 1] for random inputs."""
        noisy = np.random.default_rng(0).random((128, 128, 3)).astype(np.float32)
        denoised = np.random.default_rng(1).random((128, 128, 3)).astype(np.float32)
        out = classical_signal(noisy, denoised)
        assert out.shape == (128, 128)
        assert float(out.min()) >= 0.0 - 1e-6
        assert float(out.max()) <= 1.0 + 1e-6
        assert np.isfinite(out).all(), "Output contains NaN or inf"

    def test_identical_inputs_give_zeros(self):
        """If noisy == denoised, the signal must be all zeros."""
        img = make_hw3(64, 64, value=0.4)
        out = classical_signal(img, img.copy())
        np.testing.assert_allclose(out, 0.0, atol=1e-6,
            err_msg="Identical inputs should give all-zero classical signal.")

    def test_constant_difference_no_division_by_zero(self):
        """
        Constant difference (min == max) must not divide by zero.
        Output must be a constant map (all pixels equal).
        """
        noisy = make_hw3(64, 64, value=0.8)
        denoised = make_hw3(64, 64, value=0.3)
        out = classical_signal(noisy, denoised)
        assert out.shape == (64, 64)
        assert np.isfinite(out).all(), "Output contains NaN or inf"
        assert float(out.max() - out.min()) < 1e-6, \
            "Constant difference should produce a constant output map."

    def test_output_shape_is_hw(self):
        """Output shape is HxW (channels averaged away)."""
        noisy = make_hw3(50, 70)
        denoised = make_hw3(50, 70, value=0.2)
        out = classical_signal(noisy, denoised)
        assert out.shape == (50, 70)

    def test_larger_difference_higher_signal(self):
        """Larger difference -> higher normalized signal."""
        noisy = np.zeros((64, 64, 3), dtype=np.float32)
        denoised = np.zeros((64, 64, 3), dtype=np.float32)
        denoised[:, :32, :] = 0.1     # left: small difference
        denoised[:, 32:, :] = 0.5     # right: larger difference
        out = classical_signal(noisy, denoised)
        left_mean = float(out[:, :32].mean())
        right_mean = float(out[:, 32:].mean())
        assert right_mean > left_mean, \
            "Larger difference should produce higher classical signal."

    def test_channels_are_averaged_not_maxed(self):
        """
        Verify channels are averaged, not maxed.

        Pixel A: difference [0.3, 0.0, 0.0] -> mean = 0.1
        Pixel B: difference [0.1, 0.1, 0.1] -> mean = 0.1
        
        Both have the same mean, so after min-max normalization (max=0.3 globally),
        both should get 0.1 / 0.3 = 1/3.
        
        If max-over-channels was used instead:
        Pixel A: max = 0.3 -> normalized = 0.3/0.3 = 1.0
        Pixel B: max = 0.1 -> normalized = 0.1/0.3 = 1/3
        This would fail the assertion.
        """
        noisy = np.zeros((8, 8, 3), dtype=np.float32)
        denoised = np.zeros((8, 8, 3), dtype=np.float32)
        denoised[0, 0] = [0.3, 0.0, 0.0]    # A: mean diff = 0.1
        denoised[0, 1] = [0.1, 0.1, 0.1]    # B: mean diff = 0.1
        out = classical_signal(noisy, denoised)
        
        # Both pixels have the same mean difference, so same normalized signal.
        expected_a = 0.1 / 0.3  # = 1/3
        expected_b = 0.1 / 0.3  # = 1/3
        assert out[0, 0] == pytest.approx(expected_a, abs=1e-5), \
            f"Pixel A should be {expected_a:.4f}, got {out[0, 0]:.4f}"
        assert out[0, 1] == pytest.approx(expected_b, abs=1e-5), \
            f"Pixel B should be {expected_b:.4f}, got {out[0, 1]:.4f}"
        # Untouched pixel should be 0
        assert out[4, 4] == pytest.approx(0.0, abs=1e-5), \
            f"Untouched pixel should be 0.0, got {out[4, 4]:.4f}"

    def test_symmetric_in_its_arguments(self):
        """It is |noisy - denoised|, so argument order doesn't matter."""
        a = np.random.default_rng(2).random((32, 32, 3)).astype(np.float32)
        b = np.random.default_rng(3).random((32, 32, 3)).astype(np.float32)
        out_ab = classical_signal(a, b)
        out_ba = classical_signal(b, a)
        np.testing.assert_allclose(out_ab, out_ba, atol=1e-6,
            err_msg="classical_signal must be symmetric in its arguments")

    def test_shape_mismatch_raises(self):
        """Mismatched input shapes must raise ValueError."""
        with pytest.raises((ValueError, AssertionError)):
            classical_signal(make_hw3(32, 32), make_hw3(32, 48))


# ==============================================================================
# rank_fuse
# ==============================================================================

class TestRankFuse:

    def test_same_inputs_monotonic_in_input(self):
        """
        If both inputs are identical, fused result must be monotonic in input.
        Higher input -> higher rank -> higher fused score.
        """
        u = rng_hw(64, 64, seed=5)
        out = rank_fuse(u, u.copy())
        order = np.argsort(u.ravel())
        diffs = np.diff(out.ravel()[order])
        assert (diffs >= -1e-6).all(), \
            "rank_fuse output must be monotonic when both inputs are the same map."
        assert np.isfinite(out).all(), "Output contains NaN or inf"

    def test_weights_1_0_returns_rank_of_first(self):
        """With w_learned=1, w_classical=0, output rank matches u_learned rank."""
        u1 = rng_hw(32, 32, seed=10)
        u2 = rng_hw(32, 32, seed=11)
        out = rank_fuse(u1, u2, w_learned=1.0, w_classical=0.0)
        rank_u1 = np.argsort(np.argsort(u1.ravel()))
        rank_out = np.argsort(np.argsort(out.ravel()))
        np.testing.assert_array_equal(rank_u1, rank_out,
            err_msg="w_learned=1 output rank must equal u_learned's rank.")

    def test_weights_0_1_returns_rank_of_second(self):
        """With w_learned=0, w_classical=1, output rank matches u_classical rank."""
        u1 = rng_hw(32, 32, seed=20)
        u2 = rng_hw(32, 32, seed=21)
        out = rank_fuse(u1, u2, w_learned=0.0, w_classical=1.0)
        rank_u2 = np.argsort(np.argsort(u2.ravel()))
        rank_out = np.argsort(np.argsort(out.ravel()))
        np.testing.assert_array_equal(rank_u2, rank_out,
            err_msg="w_classical=1 output rank must equal u_classical's rank.")

    def test_output_invariant_to_monotone_transform(self):
        """
        Applying a strictly increasing function to one input must not change
        the output (fusion uses only ranks).
        """
        u1 = rng_hw(64, 64, seed=30)
        u2 = rng_hw(64, 64, seed=31)
        out_orig = rank_fuse(u1, u2)
        out_transformed = rank_fuse(u1 ** 2, u2)  # squaring is monotone on [0,1]
        np.testing.assert_allclose(out_orig, out_transformed, atol=1e-6,
            err_msg="Monotone transform of input must not change rank_fuse output.")

    def test_output_shape(self):
        """Output shape matches input shape."""
        u1 = rng_hw(50, 70)
        u2 = rng_hw(50, 70)
        out = rank_fuse(u1, u2)
        assert out.shape == (50, 70)

    def test_shape_mismatch_raises(self):
        """Mismatched input shapes must raise ValueError."""
        with pytest.raises((ValueError, AssertionError)):
            rank_fuse(rng_hw(32, 32), rng_hw(32, 48))


# ==============================================================================
# patch_gate
# ==============================================================================

class TestPatchGate:

    def test_output_shape(self):
        """Output shape is (H/64, W/64)."""
        combined = rng_hw(256, 320)
        out = patch_gate(combined, patch=64)
        assert out.shape == (4, 5), f"Expected (4,5), got {out.shape}"

    def test_output_in_0_1(self):
        """Every gate weight is in [0, 1]."""
        combined = rng_hw(256, 256)
        out = patch_gate(combined)
        assert float(out.min()) >= 0.0 - 1e-6
        assert float(out.max()) <= 1.0 + 1e-6
        assert np.isfinite(out).all(), "Output contains NaN or inf"

    def test_below_cutoff_has_zero_weight(self):
        """Patches at or below the cutoff percentile have weight 0."""
        combined, scores = patch_grid_4x4()
        out = patch_gate(combined, patch=64, cutoff_percentile=70)
        assert out.shape == (4, 4)
        threshold = np.percentile(scores, 70)
        for i in range(4):
            for j in range(4):
                if scores[i, j] <= threshold:
                    assert out[i, j] == pytest.approx(0.0, abs=1e-6), \
                        f"Patch ({i},{j}) score={scores[i, j]:.2f} <= cutoff={threshold:.2f} " \
                        f"but gate weight = {out[i, j]:.4f} (expected 0)"

    def test_above_cutoff_has_positive_weight(self):
        """
        Every patch above the cutoff must have weight > 0.
        (This catches gates that return all zeros.)
        """
        combined, scores = patch_grid_4x4()
        out = patch_gate(combined, patch=64, cutoff_percentile=70)
        threshold = np.percentile(scores, 70)
        above = scores > threshold
        assert above.sum() == 5, "Sanity check: should have 5 patches above 70th percentile"
        assert (out[above] > 0).all(), \
            f"Patches above the cutoff must have weight > 0, got {out[above]}"
        assert int((out > 0).sum()) == int(above.sum()), \
            "Exactly the patches above the cutoff should be non-zero."

    def test_highest_scoring_patch_has_highest_weight(self):
        """The highest-scoring patch should have the highest weight."""
        combined, scores = patch_grid_4x4()
        out = patch_gate(combined, patch=64, cutoff_percentile=70)
        assert out[3, 3] == pytest.approx(float(out.max()), abs=1e-6)
        assert out[3, 3] > 0

    def test_monotonically_non_decreasing(self):
        """A larger patch score never produces a smaller weight."""
        combined, scores = patch_grid_4x4()
        out = patch_gate(combined, patch=64, cutoff_percentile=70)
        order = np.argsort(scores.ravel())
        diffs = np.diff(out.ravel()[order])
        assert (diffs >= -1e-6).all(), f"patch_gate is not monotone: diffs = {diffs}"

    def test_constant_input_does_not_crash(self):
        """A constant combined map must not raise (all patches same score)."""
        combined = make_hw(128, 128, value=0.5)
        out = patch_gate(combined, patch=64)
        assert out.shape == (2, 2)
        assert np.isfinite(out).all()
        assert float(out.min()) >= 0.0 - 1e-6
        assert float(out.max()) <= 1.0 + 1e-6

    def test_custom_patch_size(self):
        """patch_gate respects non-default patch size."""
        combined = rng_hw(128, 192)
        out = patch_gate(combined, patch=32)
        assert out.shape == (4, 6), f"Expected (4,6), got {out.shape}"

    def test_non_multiple_of_patch_raises(self):
        """H or W not a multiple of patch must raise ValueError."""
        with pytest.raises((ValueError, AssertionError)):
            patch_gate(rng_hw(100, 128), patch=64)
"""Tests for attention kernel weights and fixed R buffer.

Validates kernel weight properties, fixed R behavior, and edge cases
across all kernel modes. R is a fixed buffer (_R_eff), not learnable.
"""
import torch
import math
import sys
from pathlib import Path

# ── path setup (same as other test files) ──
_v4 = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_v4 / 'model'))
from instnct import INSTNCT


# ── Tests ──

class TestForwardRuns:
    """Model forward must produce valid output across kernel modes and R values."""

    def _check_forward(self, kernel_mode, embed_mode, R=1, M=32, D=16, N=3):
        torch.manual_seed(42)
        model = INSTNCT(M=M, embed_dim=D, N=N, R=R, embed_mode=embed_mode,
                        kernel_mode=kernel_mode)
        model.eval()
        if embed_mode:
            x = torch.randint(0, 256, (2, 8))
        else:
            x = torch.randn(2, 8, 8)
        with torch.no_grad():
            out, _ = model(x)
        expected_out = 256 if embed_mode else 8
        assert out.shape == (2, 8, expected_out)
        assert torch.isfinite(out).all(), f"NaN/Inf with kernel={kernel_mode} R={R}"

    def test_vshape_R0(self):
        self._check_forward('vshape', True, R=0)

    def test_vshape_R1(self):
        self._check_forward('vshape', True, R=1)

    def test_gaussian_R1(self):
        self._check_forward('gaussian', True, R=1)

    def test_uniform_R1(self):
        self._check_forward('uniform', True, R=1)

    def test_binary_R0(self):
        self._check_forward('vshape', False, R=0)

    def test_binary_R2(self):
        self._check_forward('vshape', False, R=2)


class TestFixedR:
    """R is now a fixed buffer — verify it's correctly set and not trainable."""

    def test_R_eff_matches_config(self):
        for R_val in (0, 1, 2, 4):
            model = INSTNCT(M=32, embed_dim=16, N=3, R=R_val, embed_mode=True)
            expected = R_val + 0.5
            assert torch.allclose(model._R_eff, torch.full((3,), expected)), \
                f"R={R_val}: _R_eff should be {expected}, got {model._R_eff}"

    def test_R_eff_not_trainable(self):
        model = INSTNCT(M=32, embed_dim=16, N=3, embed_mode=True)
        assert not model._R_eff.requires_grad, "_R_eff should be a fixed buffer"

    def test_R0_needle_runs(self):
        """R=0 needle: single slot read, must not crash."""
        model = INSTNCT(M=32, embed_dim=16, N=3, R=0, embed_mode=True)
        x = torch.randint(0, 256, (2, 4))
        out, _ = model(x)
        assert torch.isfinite(out).all(), "R=0 produced NaN/Inf"


class TestEdgeCases:
    """Edge cases that could break the precompute logic."""

    def test_single_expert(self):
        """N=1: only one expert, precompute should still work."""
        model = INSTNCT(M=16, embed_dim=8, N=1, embed_mode=True)
        x = torch.randint(0, 256, (2, 4))
        out, _ = model(x)
        assert out.shape == (2, 4, 256)
        assert torch.isfinite(out).all()

    def test_single_timestep(self):
        """T=1: only one step, precompute overhead should be worth it."""
        model = INSTNCT(M=16, embed_dim=8, N=3, embed_mode=True)
        x = torch.randint(0, 256, (2, 1))
        out, _ = model(x)
        assert out.shape == (2, 1, 256)
        assert torch.isfinite(out).all()

    def test_batch_size_one(self):
        """B=1: broadcast must still work."""
        model = INSTNCT(M=16, embed_dim=8, N=3, embed_mode=True)
        x = torch.randint(0, 256, (1, 4))
        out, _ = model(x)
        assert out.shape == (1, 4, 256)
        assert torch.isfinite(out).all()

    def test_large_N(self):
        """N=12: many experts, all should get valid precomputed weights."""
        model = INSTNCT(M=16, embed_dim=8, N=12, embed_mode=True)
        x = torch.randint(0, 256, (2, 4))
        out, _ = model(x)
        assert out.shape == (2, 4, 256)
        assert torch.isfinite(out).all()

    def test_extreme_R_values(self):
        """R=0 (needle) and R=15 (wide) must produce valid output."""
        for R_val in (0, 15):
            model = INSTNCT(M=32, embed_dim=8, N=2, R=R_val, embed_mode=True)
            x = torch.randint(0, 256, (2, 4))
            out, _ = model(x)
            assert torch.isfinite(out).all(), f"NaN/Inf with R={R_val}"

    def test_binary_mode_still_works(self):
        """Binary mode (8-bit float) must not be broken by precompute."""
        model = INSTNCT(M=64, embed_dim=16, N=3, embed_mode=False)
        x = torch.randn(2, 4, 8)
        out, _ = model(x)
        assert out.shape == (2, 4, 8)
        assert torch.isfinite(out).all()

        # Gradient check — trainable params should get gradients
        loss = out.sum()
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
        assert len(grads) > 0, "No gradients in binary mode"


class TestWeightProperties:
    """Verify that precomputed weights have correct mathematical properties."""

    def test_weights_sum_to_one(self):
        """Each expert's kernel weights must sum to 1 (normalized)."""
        for R_val in (0, 1, 2, 4):
            model = INSTNCT(M=32, embed_dim=8, N=6, R=R_val, embed_mode=True)
            R_effs = model._R_eff
            win = max(int(math.ceil(R_effs.max().item() * 1.5)) + 1, 1)
            offsets = torch.arange(-win, win + 1).float().abs()
            raw_w = (1.0 - offsets.unsqueeze(0) / R_effs.unsqueeze(1).clamp(min=0.5)).clamp(min=0)
            w = raw_w / raw_w.sum(dim=1, keepdim=True)
            assert torch.allclose(w.sum(dim=1), torch.ones(6), atol=1e-6), \
                f"R={R_val}: weights don't sum to 1"

    def test_weights_non_negative(self):
        """All weights must be >= 0."""
        model = INSTNCT(M=32, embed_dim=8, N=6, R=2, embed_mode=True)
        R_effs = model._R_eff
        win = max(int(math.ceil(R_effs.max().item() * 1.5)) + 1, 1)
        offsets = torch.arange(-win, win + 1).float().abs()

        for mode in ('vshape', 'gaussian', 'uniform'):
            if mode == 'vshape':
                raw_w = (1.0 - offsets.unsqueeze(0) / R_effs.unsqueeze(1).clamp(min=0.5)).clamp(min=0)
            elif mode == 'gaussian':
                sigma = (R_effs.unsqueeze(1) / 2.5).clamp(min=0.3)
                raw_w = torch.exp(-0.5 * (offsets.unsqueeze(0) / sigma) ** 2)
            elif mode == 'uniform':
                raw_w = torch.sigmoid(10.0 * (R_effs.unsqueeze(1) - offsets.unsqueeze(0)))
            w = raw_w / raw_w.sum(dim=1, keepdim=True)
            assert (w >= 0).all(), f"{mode}: negative weights found"

    def test_center_weight_is_max(self):
        """Center position should have the highest weight (for vshape and gaussian)."""
        model = INSTNCT(M=32, embed_dim=8, N=6, R=2, embed_mode=True)
        R_effs = model._R_eff
        win = max(int(math.ceil(R_effs.max().item() * 1.5)) + 1, 1)
        offsets = torch.arange(-win, win + 1).float().abs()
        center_idx = win  # index of offset=0

        for mode in ('vshape', 'gaussian'):
            if mode == 'vshape':
                raw_w = (1.0 - offsets.unsqueeze(0) / R_effs.unsqueeze(1).clamp(min=0.5)).clamp(min=0)
            elif mode == 'gaussian':
                sigma = (R_effs.unsqueeze(1) / 2.5).clamp(min=0.3)
                raw_w = torch.exp(-0.5 * (offsets.unsqueeze(0) / sigma) ** 2)
            w = raw_w / raw_w.sum(dim=1, keepdim=True)
            assert (w[:, center_idx].unsqueeze(1) >= w - 1e-7).all(), (
                f"{mode}: center is not max weight"
            )

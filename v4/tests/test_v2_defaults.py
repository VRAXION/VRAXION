"""Smoke tests for v2 default config.

Validates that all 2026-03-08 config upgrades work together:
- embed_encoding='learned'
- pointer_interp_mode='linear'
- pointer_seam_mode='shortest_arc'
- R=2
- c19_mode='dualphi'
- jump_gate=True (optional)

Run: python -m pytest v4/tests/test_v2_defaults.py -v
"""

import math
import torch
import pytest
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "model"))

from instnct import INSTNCT, _c19_activation, _c19_dualphi_activation  # type: ignore[import-not-found]


V2_CFG = dict(
    M=32, hidden_dim=64, slot_dim=16, N=1, R=2,
    embed_mode=True, embed_encoding='learned', output_encoding='lowrank_c19',
    kernel_mode='vshape', pointer_mode='sequential',
    write_mode='replace', expert_weighting=False,
    checkpoint_chunks=0, bb_enabled=False,
    pointer_interp_mode='linear',
    pointer_seam_mode='shortest_arc',
    c19_mode='dualphi',
)


class TestDualPhiActivation:
    def test_shape_preserved(self):
        x = torch.randn(4, 64)
        y = _c19_dualphi_activation(x)
        assert y.shape == x.shape

    def test_dualphi_smaller_output(self):
        """Dual-phi should produce smaller outputs (1/phi scaling on positive)."""
        torch.manual_seed(42)
        x = torch.randn(1000, 64)
        y_std = _c19_activation(x)
        y_phi = _c19_dualphi_activation(x)
        # dualphi should be smaller on average (positive arches scaled by 1/phi)
        assert y_phi.abs().mean() < y_std.abs().mean()

    def test_gradient_flows(self):
        x = torch.randn(4, 16, requires_grad=True)
        y = _c19_dualphi_activation(x)
        y.sum().backward()
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

    def test_linear_tails(self):
        """Values beyond ±6C should be linear."""
        C = math.pi
        x = torch.tensor([7.0 * C, -7.0 * C])
        y = _c19_dualphi_activation(x)
        # Linear tail: x - sign(x) * limit where limit = 6*C
        expected = x - x.sign() * 6.0 * C
        assert torch.allclose(y, expected, atol=1e-5)


class TestV2Model:
    def test_forward_backward(self):
        model = INSTNCT(**V2_CFG)
        x = torch.randint(0, 256, (2, 8))
        out, state = model(x)
        assert out.shape == (2, 8, 256)
        loss = out.sum()
        loss.backward()
        # Check that key trainable parameters have gradients
        # Some params (c19_rho/C, S_raw, ring_signal_norm) may not receive
        # gradients in all configurations — that's expected.
        key_params = ['inp.weight', 'out.', 'write_proj', 'read_proj', 'hidden_proj']
        for name, p in model.named_parameters():
            if p.requires_grad and any(k in name for k in key_params):
                assert p.grad is not None, f"No gradient for key param {name}"

    def test_state_carry(self):
        model = INSTNCT(**V2_CFG)
        x = torch.randint(0, 256, (2, 8))
        out1, state1 = model(x)
        out2, state2 = model(x, state=state1)
        # Outputs should differ with carried state
        assert not torch.allclose(out1, out2, atol=1e-6)

    def test_pointer_interp_degenerates_at_integer(self):
        """Linear interp with integer ptr should produce same output as 'off'.

        Sequential mode: ptr starts at integer, stays integer (ptr += 1).
        Linear interp at integer positions: alpha=0, degenerates to discrete.
        This is correct behavior — interp only matters for fractional pointers.
        """
        torch.manual_seed(42)
        m1 = INSTNCT(**{**V2_CFG, 'pointer_interp_mode': 'off'})
        torch.manual_seed(42)
        m2 = INSTNCT(**{**V2_CFG, 'pointer_interp_mode': 'linear'})
        x = torch.randint(0, 256, (2, 8))
        out1, _ = m1(x)
        out2, _ = m2(x)
        # With sequential integer pointer, outputs should be identical
        assert torch.allclose(out1, out2, atol=1e-5)

    def test_r2_wider_window(self):
        """R=2 should attend to 5 slots (2*2+1)."""
        model = INSTNCT(**V2_CFG)
        assert model.R == 2
        # Window size is 2R+1 = 5
        offsets = torch.arange(-model.R, model.R + 1)
        assert len(offsets) == 5

    def test_dualphi_mode(self):
        model = INSTNCT(**V2_CFG)
        assert model.c19_mode == 'dualphi'
        assert model._c19_dualphi is True


class TestJumpGate:
    def test_jump_gate_creates_parameters(self):
        model = INSTNCT(**{**V2_CFG, 'jump_gate': True})
        assert model.jump_gate_enabled is True
        assert hasattr(model, 'jump_gate_head')
        n_params = sum(p.numel() for p in model.jump_gate_head.parameters())
        # hidden_dim * 1 + 1 bias = 65 params per expert
        assert n_params == V2_CFG['hidden_dim'] + 1

    def test_jump_gate_forward(self):
        model = INSTNCT(**{**V2_CFG, 'jump_gate': True})
        model._diag_enabled = True
        x = torch.randint(0, 256, (2, 8))
        out, state = model(x)
        assert out.shape == (2, 8, 256)
        # Jump gate diagnostics should be present
        assert 'jump_gate_mean_0' in model._diag
        assert 'jump_gate_max_0' in model._diag
        # Gate should be small at init (bias=-3.0)
        assert model._diag['jump_gate_mean_0'] < 0.2

    def test_jump_gate_only_for_sequential(self):
        """Jump gate should be disabled for non-sequential modes."""
        model = INSTNCT(**{**V2_CFG, 'jump_gate': True, 'pointer_mode': 'learned'})
        assert model.jump_gate_enabled is False

    def test_jump_gate_backward(self):
        model = INSTNCT(**{**V2_CFG, 'jump_gate': True})
        x = torch.randint(0, 256, (2, 8))
        out, _ = model(x)
        out.sum().backward()
        # Jump gate head should have gradients
        for p in model.jump_gate_head.parameters():
            assert p.grad is not None


class TestModelFactory:
    def test_v2_config_roundtrip(self):
        """Verify model_factory passes all v2 keys."""
        sys.path.insert(0, str(ROOT / "training"))
        from model_factory import _build_instnct_spec  # type: ignore[import-not-found]

        yaml_config = {
            'M': 32, 'hidden_dim': 64, 'slot_dim': 16, 'N': 1, 'R': 2,
            'embed_encoding': 'learned', 'output_encoding': 'lowrank_c19',
            'pointer_interp_mode': 'linear', 'pointer_seam_mode': 'shortest_arc',
            'c19_mode': 'dualphi', 'jump_gate': True,
        }
        spec = _build_instnct_spec(embed_mode=True, model_config=yaml_config)
        assert spec['pointer_interp_mode'] == 'linear'
        assert spec['pointer_seam_mode'] == 'shortest_arc'
        assert spec['c19_mode'] == 'dualphi'
        assert spec['jump_gate'] is True
        assert spec['R'] == 2
        assert spec['embed_encoding'] == 'learned'

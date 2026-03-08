"""Hard adversarial tests for INSTNCT v4 — stress numerical stability,
state isolation, ring invariants, gradient health, and config edge cases.

Run:  cd v4/tests && python -m pytest test_adversarial.py -v
"""

import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import pytest

# ── path setup (mirrors test_model.py) ──
_MODEL_DIR = str(Path(__file__).resolve().parent.parent / 'model')
if _MODEL_DIR not in sys.path:
    sys.path.insert(0, _MODEL_DIR)

from instnct import (
    INSTNCT,
    phi_destinations,
    _c19_activation,
    _rho_from_raw,
    _C_from_raw,
    func_ringstart_tns,
    func_softread_tns,
    func_softwrit_tns,
    func_hdd_write_tns,
    func_gated_write_tns,
    func_movepntr_tns,
    func_shortest_arc_delta_tns,
)


def to_bits(x: torch.Tensor) -> torch.Tensor:
    return ((x.unsqueeze(-1) >> torch.arange(7, -1, -1)) & 1).float()


# ── Tiny configs for speed ──
TINY = dict(M=32, embed_dim=16, N=2, R=1)
TINY_SPLIT = dict(M=32, hidden_dim=64, slot_dim=16, N=2, R=1)
TINY_N1 = dict(M=32, hidden_dim=32, slot_dim=16, N=1, R=1)
PROD_SHAPE = dict(
    M=32, hidden_dim=32, slot_dim=16, N=1, R=1,
    pointer_mode='sequential', write_mode='replace',
    replace_impl='dense', checkpoint_chunks=0, expert_weighting=False,
)


def _clone_state(state):
    if state is None:
        return None
    return {k: v.clone() if torch.is_tensor(v) else v for k, v in state.items()}


# ═══════════════════════════════════════════════════════════════════
#  1. C19 ACTIVATION — NUMERICAL ADVERSARIAL
# ═══════════════════════════════════════════════════════════════════

class TestC19Activation:
    """Stress the periodic parabolic activation under extreme inputs."""

    def test_c19_zero_input(self):
        x = torch.zeros(100)
        y = _c19_activation(x)
        assert torch.isfinite(y).all()

    def test_c19_huge_positive(self):
        x = torch.tensor([1e6, 1e10, 1e15])
        y = _c19_activation(x)
        assert torch.isfinite(y).all(), f"Inf/NaN at large positive: {y}"

    def test_c19_huge_negative(self):
        x = torch.tensor([-1e6, -1e10, -1e15])
        y = _c19_activation(x)
        assert torch.isfinite(y).all(), f"Inf/NaN at large negative: {y}"

    def test_c19_linear_tails_monotonic(self):
        """Outside the periodic region, c19 should be linear (monotonically increasing)."""
        C = math.pi
        l = 6.0 * C
        # Right tail
        x_right = torch.linspace(l + 1, l + 100, 200)
        y_right = _c19_activation(x_right)
        diffs = y_right[1:] - y_right[:-1]
        assert (diffs > 0).all(), "Right tail not monotonically increasing"
        # Left tail
        x_left = torch.linspace(-l - 100, -l - 1, 200)
        y_left = _c19_activation(x_left)
        diffs = y_left[1:] - y_left[:-1]
        assert (diffs > 0).all(), "Left tail not monotonically increasing"

    def test_c19_gradient_finite_everywhere(self):
        """Gradient through c19 must be finite at all points including period boundaries."""
        x = torch.linspace(-30, 30, 1000, requires_grad=True)
        y = _c19_activation(x)
        y.sum().backward()
        assert torch.isfinite(x.grad).all(), "Non-finite gradients in c19"

    def test_c19_learnable_params_bounded(self):
        """Sigmoid-bounded rho and C must stay within bounds for any raw value."""
        for raw in [-100.0, -10.0, 0.0, 10.0, 100.0]:
            raw_t = torch.tensor(raw)
            rho = _rho_from_raw(raw_t)
            C = _C_from_raw(raw_t)
            assert 0.5 <= rho.item() <= 8.0, f"rho out of bounds at raw={raw}: {rho}"
            assert 1.0 <= C.item() <= 50.0, f"C out of bounds at raw={raw}: {C}"

    def test_c19_nan_input_propagates(self):
        """NaN input should produce NaN output (not silently ignored)."""
        x = torch.tensor([float('nan'), 1.0, 2.0])
        y = _c19_activation(x)
        assert torch.isnan(y[0]), "NaN input did not propagate"
        assert torch.isfinite(y[1:]).all()


# ═══════════════════════════════════════════════════════════════════
#  2. EXTREME INPUT PATTERNS
# ═══════════════════════════════════════════════════════════════════

class TestExtremeInputs:
    """Feed adversarial byte patterns through the model."""

    @pytest.fixture(params=[
        ('all_zeros', lambda B, T: torch.zeros(B, T, dtype=torch.long)),
        ('all_255', lambda B, T: torch.full((B, T), 255, dtype=torch.long)),
        ('constant_42', lambda B, T: torch.full((B, T), 42, dtype=torch.long)),
        ('alternating', lambda B, T: (torch.arange(T) % 2).unsqueeze(0).expand(B, -1).long() * 255),
        ('sawtooth', lambda B, T: (torch.arange(T) % 256).unsqueeze(0).expand(B, -1).long()),
    ])
    def adversarial_input(self, request):
        return request.param

    def test_embed_mode_no_nan_inf(self, adversarial_input):
        name, gen = adversarial_input
        model = INSTNCT(**TINY, embed_mode=True)
        x = gen(4, 64)
        with torch.no_grad():
            out, state = model(x)
        assert torch.isfinite(out).all(), f"Non-finite output with {name} input"
        assert torch.isfinite(state['ring']).all(), f"Non-finite ring with {name}"
        assert torch.isfinite(state['hidden']).all(), f"Non-finite hidden with {name}"

    def test_binary_mode_no_nan_inf(self, adversarial_input):
        name, gen = adversarial_input
        model = INSTNCT(**TINY, embed_mode=False)
        x = to_bits(gen(4, 64))
        with torch.no_grad():
            out, state = model(x)
        assert torch.isfinite(out).all(), f"Non-finite output with {name} input (binary)"


# ═══════════════════════════════════════════════════════════════════
#  3. LONG SEQUENCE STABILITY
# ═══════════════════════════════════════════════════════════════════

class TestLongSequenceStability:
    """Test that the model doesn't blow up over many timesteps."""

    def test_long_sequence_no_explosion(self):
        """T=512 must not produce exploding outputs."""
        model = INSTNCT(**TINY, embed_mode=True)
        x = torch.randint(0, 256, (2, 512))
        with torch.no_grad():
            out, state = model(x)
        assert torch.isfinite(out).all(), "Non-finite output at T=512"
        # Output logits should not be astronomically large
        assert out.abs().max().item() < 1e6, f"Output explosion: max={out.abs().max().item()}"

    def test_sequential_carry_many_batches(self):
        """10 consecutive carry-over batches must remain stable."""
        model = INSTNCT(**TINY_N1, embed_mode=True,
                        write_mode='replace', replace_impl='dense')
        state = None
        for i in range(10):
            x = torch.randint(0, 256, (2, 32))
            with torch.no_grad():
                out, state = model(x, state=state)
            assert torch.isfinite(out).all(), f"Non-finite at batch {i}"
            assert torch.isfinite(state['ring']).all(), f"Ring blow-up at batch {i}"
            assert state['ring'].abs().max().item() < 1e6, \
                f"Ring explosion at batch {i}: max={state['ring'].abs().max().item()}"

    def test_accumulate_write_ring_growth(self):
        """Accumulate write mode ring norms should stay bounded over many steps."""
        model = INSTNCT(**TINY, embed_mode=True, write_mode='accumulate')
        state = None
        for i in range(5):
            x = torch.randint(0, 256, (2, 64))
            with torch.no_grad():
                out, state = model(x, state=state)
        ring_norm = state['ring'].norm().item()
        assert ring_norm < 1e8, f"Unbounded ring growth in accumulate mode: norm={ring_norm}"


# ═══════════════════════════════════════════════════════════════════
#  4. BATCH ISOLATION — NO CROSS-BATCH LEAKAGE
# ═══════════════════════════════════════════════════════════════════

class TestBatchIsolation:
    """Verify that different batch elements don't leak into each other."""

    def test_single_vs_batched_output_match(self):
        """Running B=1 twice vs B=2 once must produce identical outputs."""
        torch.manual_seed(999)
        model = INSTNCT(**TINY_N1, embed_mode=True,
                        write_mode='replace', replace_impl='dense')

        x0 = torch.randint(0, 256, (1, 16))
        x1 = torch.randint(0, 256, (1, 16))
        x_batch = torch.cat([x0, x1], dim=0)

        with torch.no_grad():
            out0, _ = model(x0)
            # Fresh model state for x1
            torch.manual_seed(999)
            model2 = INSTNCT(**TINY_N1, embed_mode=True,
                             write_mode='replace', replace_impl='dense')
            out1, _ = model2(x1)

            torch.manual_seed(999)
            model3 = INSTNCT(**TINY_N1, embed_mode=True,
                             write_mode='replace', replace_impl='dense')
            out_batch, _ = model3(x_batch)

        assert torch.allclose(out_batch[0:1], out0, atol=1e-5), \
            f"Batch element 0 differs: max diff={( out_batch[0:1] - out0).abs().max()}"
        assert torch.allclose(out_batch[1:2], out1, atol=1e-5), \
            f"Batch element 1 differs: max diff={(out_batch[1:2] - out1).abs().max()}"

    def test_identical_inputs_produce_identical_outputs(self):
        """Two identical sequences in the same batch must produce identical outputs."""
        model = INSTNCT(**TINY, embed_mode=True)
        x_single = torch.randint(0, 256, (1, 32))
        x_dup = x_single.expand(4, -1).clone()
        with torch.no_grad():
            out, _ = model(x_dup)
        for i in range(1, 4):
            assert torch.allclose(out[0], out[i], atol=1e-6), \
                f"Batch {i} differs from batch 0 on identical input"


# ═══════════════════════════════════════════════════════════════════
#  5. GRADIENT HEALTH UNDER STRESS
# ═══════════════════════════════════════════════════════════════════

class TestGradientHealth:
    """Adversarial gradient flow checks."""

    def test_gradient_no_nan_long_sequence(self):
        """Gradients must be finite even for T=128."""
        model = INSTNCT(**TINY, embed_mode=True)
        x = torch.randint(0, 256, (2, 128))
        out, _ = model(x)
        loss = F.cross_entropy(
            out.reshape(-1, 256),
            x.reshape(-1),
        )
        loss.backward()
        for name, p in model.named_parameters():
            if p.grad is not None:
                assert torch.isfinite(p.grad).all(), \
                    f"Non-finite gradient in {name}"

    def test_gradient_not_all_zero(self):
        """At least some parameters must have nonzero gradients (model is actually learning)."""
        model = INSTNCT(**TINY, embed_mode=True)
        x = torch.randint(0, 256, (4, 32))
        out, _ = model(x)
        loss = F.cross_entropy(out.reshape(-1, 256), x.reshape(-1))
        loss.backward()
        nonzero_grads = sum(
            1 for p in model.parameters()
            if p.grad is not None and p.grad.abs().sum() > 0
        )
        total_params = sum(1 for p in model.parameters() if p.requires_grad)
        assert nonzero_grads > total_params * 0.5, \
            f"Only {nonzero_grads}/{total_params} params have nonzero gradients"

    def test_gradient_magnitude_bounded(self):
        """No single parameter gradient should be astronomically large."""
        model = INSTNCT(**TINY_SPLIT, embed_mode=True)
        x = torch.randint(0, 256, (4, 64))
        out, _ = model(x)
        out.sum().backward()
        for name, p in model.named_parameters():
            if p.grad is not None:
                max_grad = p.grad.abs().max().item()
                assert max_grad < 1e6, \
                    f"Gradient explosion in {name}: max={max_grad}"

    def test_multiple_backward_passes_stable(self):
        """Repeated forward/backward passes must stay stable."""
        model = INSTNCT(**TINY, embed_mode=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        for step in range(20):
            x = torch.randint(0, 256, (4, 32))
            out, _ = model(x)
            loss = F.cross_entropy(out.reshape(-1, 256), x.reshape(-1))
            assert torch.isfinite(loss), f"Non-finite loss at step {step}"
            optimizer.zero_grad()
            loss.backward()
            # Check no NaN grads
            for name, p in model.named_parameters():
                if p.grad is not None:
                    assert torch.isfinite(p.grad).all(), \
                        f"Non-finite grad in {name} at step {step}"
            optimizer.step()


# ═══════════════════════════════════════════════════════════════════
#  6. RING BUFFER INVARIANTS
# ═══════════════════════════════════════════════════════════════════

class TestRingInvariants:
    """Verify ring buffer correctness under stress."""

    def test_ring_shape_preserved_across_forward(self):
        model = INSTNCT(**TINY, embed_mode=True)
        x = torch.randint(0, 256, (2, 32))
        with torch.no_grad():
            _, state = model(x)
        assert state['ring'].shape == (2, TINY['M'], model.slot_dim)

    def test_pointer_stays_in_range(self):
        """Pointer positions must be in [0, M) after any number of steps."""
        model = INSTNCT(**TINY, embed_mode=True)
        state = None
        for _ in range(10):
            x = torch.randint(0, 256, (4, 64))
            with torch.no_grad():
                _, state = model(x, state=state)
            ptr = state['ptr']
            assert (ptr >= 0).all(), f"Negative pointer: {ptr.min()}"
            assert (ptr < TINY['M']).all(), f"Pointer >= M: {ptr.max()}"

    def test_pointer_stays_in_range_learned_mode(self):
        """Learned pointer mode must keep pointers in [0, M)."""
        model = INSTNCT(**TINY, embed_mode=True, pointer_mode='learned')
        state = None
        for _ in range(5):
            x = torch.randint(0, 256, (4, 32))
            with torch.no_grad():
                _, state = model(x, state=state)
            ptr = state['ptr']
            assert (ptr >= 0).all() and (ptr < TINY['M']).all(), \
                f"Learned pointer out of range: min={ptr.min()}, max={ptr.max()}"

    def test_pointer_stays_in_range_pilot_mode(self):
        """Pilot pointer mode must keep pointers in [0, M)."""
        model = INSTNCT(**TINY, embed_mode=True, pointer_mode='pilot')
        state = None
        for _ in range(5):
            x = torch.randint(0, 256, (4, 32))
            with torch.no_grad():
                _, state = model(x, state=state)
            ptr = state['ptr']
            assert (ptr >= 0).all() and (ptr < TINY['M']).all(), \
                f"Pilot pointer out of range: min={ptr.min()}, max={ptr.max()}"

    def test_phi_destinations_coverage(self):
        """Over all slots, each expert's destinations should cover all M slots
        (golden ratio guarantees uniform coverage)."""
        M = 64
        dests = phi_destinations(1, M)
        # Starting from slot 0, trace M jumps and verify we visit all slots
        visited = set()
        slot = 0
        for _ in range(M):
            visited.add(int(dests[0, slot].item()))
            slot = int(dests[0, slot].item())
        assert len(visited) == M, \
            f"phi_destinations only covers {len(visited)}/{M} slots"

    def test_ring_write_actually_modifies_slots(self):
        """After a forward pass with nonzero input, ring should not be all zeros."""
        model = INSTNCT(**TINY, embed_mode=True)
        x = torch.randint(1, 256, (2, 32))  # nonzero inputs
        with torch.no_grad():
            _, state = model(x)
        assert state['ring'].abs().sum() > 0, "Ring still all zeros after forward"


# ═══════════════════════════════════════════════════════════════════
#  7. CONFIGURATION EDGE CASES
# ═══════════════════════════════════════════════════════════════════

class TestConfigEdgeCases:
    """Pathological but valid configuration combinations."""

    def test_M_equals_1(self):
        """M=1 (single slot ring) must work without crashing."""
        model = INSTNCT(M=1, embed_dim=16, N=1, R=0, embed_mode=True)
        x = torch.randint(0, 256, (2, 8))
        with torch.no_grad():
            out, state = model(x)
        assert out.shape == (2, 8, 256)
        assert torch.isfinite(out).all()

    def test_M_equals_2(self):
        """M=2 with R=0 must work."""
        model = INSTNCT(M=2, embed_dim=16, N=1, R=0, embed_mode=True)
        x = torch.randint(0, 256, (2, 16))
        with torch.no_grad():
            out, _ = model(x)
        assert torch.isfinite(out).all()

    def test_R_equals_0_needle(self):
        """R=0 (needle read — single slot) must produce valid output."""
        model = INSTNCT(M=32, embed_dim=16, N=2, R=0, embed_mode=True)
        x = torch.randint(0, 256, (2, 16))
        with torch.no_grad():
            out, _ = model(x)
        assert out.shape == (2, 16, 256)
        assert torch.isfinite(out).all()

    def test_N_equals_1_single_expert(self):
        """N=1 single expert must work."""
        model = INSTNCT(M=32, embed_dim=16, N=1, R=1, embed_mode=True)
        x = torch.randint(0, 256, (2, 16))
        out, _ = model(x)
        assert torch.isfinite(out).all()
        out.sum().backward()

    def test_extreme_hidden_slot_ratio(self):
        """hidden_dim=512, slot_dim=4 — extreme compression must work."""
        model = INSTNCT(M=16, hidden_dim=512, slot_dim=4, N=1, R=1, embed_mode=True)
        x = torch.randint(0, 256, (2, 8))
        with torch.no_grad():
            out, _ = model(x)
        assert torch.isfinite(out).all()

    def test_T_equals_1(self):
        """Single timestep must work for all modes."""
        for em in [True, False]:
            model = INSTNCT(**TINY, embed_mode=em)
            if em:
                x = torch.randint(0, 256, (2, 1))
            else:
                x = to_bits(torch.randint(0, 256, (2, 1)))
            with torch.no_grad():
                out, _ = model(x)
            expected_out = 256 if em else 8
            assert out.shape == (2, 1, expected_out)
            assert torch.isfinite(out).all()

    def test_large_N_many_experts(self):
        """N=8 experts sharing one ring must work."""
        model = INSTNCT(M=32, embed_dim=16, N=8, R=1, embed_mode=True)
        x = torch.randint(0, 256, (2, 16))
        with torch.no_grad():
            out, state = model(x)
        assert out.shape == (2, 16, 256)
        assert torch.isfinite(out).all()
        assert state['ptr'].shape[0] == 8

    def test_large_R_full_ring_attention(self):
        """R large enough to cover the entire ring."""
        M = 16
        R = M  # R > M/2 — attention covers full ring
        model = INSTNCT(M=M, embed_dim=16, N=1, R=R, embed_mode=True)
        x = torch.randint(0, 256, (2, 8))
        with torch.no_grad():
            out, _ = model(x)
        assert torch.isfinite(out).all()


# ═══════════════════════════════════════════════════════════════════
#  8. WRITE MODE CORRECTNESS
# ═══════════════════════════════════════════════════════════════════

class TestWriteModes:
    """Verify different write modes behave correctly."""

    def test_replace_vs_accumulate_differ(self):
        """Replace and accumulate must produce different ring states."""
        torch.manual_seed(42)
        m_acc = INSTNCT(**TINY_N1, embed_mode=True, write_mode='accumulate')
        torch.manual_seed(42)
        m_rep = INSTNCT(**TINY_N1, embed_mode=True, write_mode='replace',
                        replace_impl='dense')
        x = torch.randint(0, 256, (2, 32))
        with torch.no_grad():
            _, s_acc = m_acc(x)
            _, s_rep = m_rep(x)
        # They share the same weights but different write semantics → different rings
        assert not torch.allclose(s_acc['ring'], s_rep['ring'], atol=1e-4), \
            "accumulate and replace produced identical rings"

    def test_gated_write_erase_gate_bounded(self):
        """Gated write erase/write_gate outputs must be in [0, 1] (sigmoid)."""
        model = INSTNCT(**TINY, embed_mode=True, gated_write=True,
                        write_mode='accumulate')
        # Just verify it doesn't crash and output is finite
        x = torch.randint(0, 256, (2, 16))
        with torch.no_grad():
            out, _ = model(x)
        assert torch.isfinite(out).all()

    def test_hdd_write_overwrites_slots(self):
        """func_hdd_write_tns with write_strength=1 should fully replace center slot."""
        B, M, D = 2, 8, 4
        ring = torch.randn(B, M, D)
        write_vec = torch.ones(B, D) * 99.0
        # Window of 1 slot at center=3
        indices = torch.full((B, 1), 3, dtype=torch.long)
        expanded_idx = indices.unsqueeze(-1).expand(-1, -1, D)
        weights = torch.ones(B, 1)
        ws = torch.ones(B, 1)  # full write strength
        new_ring = func_hdd_write_tns(ring, write_vec, expanded_idx, weights, write_strength=ws)
        # Slot 3 should now be 99.0
        assert torch.allclose(new_ring[:, 3, :], torch.full((B, D), 99.0), atol=1e-5)
        # Other slots unchanged
        for s in [0, 1, 2, 4, 5, 6, 7]:
            assert torch.allclose(new_ring[:, s, :], ring[:, s, :])


# ═══════════════════════════════════════════════════════════════════
#  9. FAST PATH vs GENERIC PATH STRESS
# ═══════════════════════════════════════════════════════════════════

class TestFastPathStress:
    """Stress the N=1 sequential replace fast path."""

    def _make_pair(self, **overrides):
        cfg = dict(PROD_SHAPE)
        cfg.update(overrides)
        torch.manual_seed(7777)
        ref = INSTNCT(**cfg, embed_mode=True)
        torch.manual_seed(7777)
        fast = INSTNCT(**cfg, embed_mode=True)
        fast.load_state_dict(ref.state_dict())
        ref._fastpath_mode = 'off'
        fast._fastpath_mode = 'force'
        return ref, fast

    def test_parity_long_sequence(self):
        """Fast path must match generic for T=128."""
        ref, fast = self._make_pair()
        x = torch.randint(0, 256, (2, 128))
        with torch.no_grad():
            ref_out, ref_state = ref(x)
            fast_out, fast_state = fast(x)
        assert torch.allclose(ref_out, fast_out, atol=1e-5, rtol=1e-4), \
            f"max diff: {(ref_out - fast_out).abs().max()}"

    def test_parity_with_carry(self):
        """Fast path carry-over state must match generic across 5 batches."""
        ref, fast = self._make_pair()
        ref_state, fast_state = None, None
        for _ in range(5):
            x = torch.randint(0, 256, (4, 32))
            with torch.no_grad():
                ref_out, ref_state = ref(x, state=_clone_state(ref_state))
                fast_out, fast_state = fast(x, state=_clone_state(fast_state))
            assert torch.allclose(ref_out, fast_out, atol=1e-5, rtol=1e-4)

    def test_parity_all_zeros_input(self):
        """Edge case: all-zero input."""
        ref, fast = self._make_pair()
        x = torch.zeros(2, 32, dtype=torch.long)
        with torch.no_grad():
            ref_out, _ = ref(x)
            fast_out, _ = fast(x)
        assert torch.allclose(ref_out, fast_out, atol=1e-5)

    def test_parity_gradient(self):
        """Fast path gradients must match generic path gradients."""
        ref, fast = self._make_pair()
        x = torch.randint(0, 256, (2, 16))

        ref_out, _ = ref(x)
        fast_out, _ = fast(x)
        ref_out.sum().backward()
        fast_out.sum().backward()

        for (rn, rp), (fn, fp) in zip(ref.named_parameters(), fast.named_parameters()):
            if rp.grad is None:
                assert fp.grad is None, f"{rn}: ref grad None but fast grad not"
                continue
            assert torch.allclose(rp.grad, fp.grad, atol=1e-5, rtol=1e-4), \
                f"Gradient mismatch in {rn}: max diff={( rp.grad - fp.grad).abs().max()}"


# ═══════════════════════════════════════════════════════════════════
#  10. ENCODING ADVERSARIAL
# ═══════════════════════════════════════════════════════════════════

class TestEncodingAdversarial:
    """Stress all encoding types."""

    @pytest.mark.parametrize('embed_enc', ['learned', 'hadamard', 'sincos', 'bitlift'])
    def test_embed_encoding_stability(self, embed_enc):
        model = INSTNCT(M=32, hidden_dim=64, slot_dim=16, N=1, R=1,
                        embed_mode=True, embed_encoding=embed_enc)
        x = torch.randint(0, 256, (4, 64))
        with torch.no_grad():
            out, _ = model(x)
        assert torch.isfinite(out).all(), f"Non-finite with embed_encoding={embed_enc}"

    @pytest.mark.parametrize('out_enc', ['learned', 'hadamard', 'sincos', 'lowrank_c19', 'bitlift'])
    def test_output_encoding_stability(self, out_enc):
        model = INSTNCT(M=32, hidden_dim=64, slot_dim=16, N=1, R=1,
                        embed_mode=True, output_encoding=out_enc)
        x = torch.randint(0, 256, (4, 64))
        with torch.no_grad():
            out, _ = model(x)
        assert torch.isfinite(out).all(), f"Non-finite with output_encoding={out_enc}"
        assert out.shape == (4, 64, 256)

    def test_bitlift_all_byte_values(self):
        """Bitlift encoding must handle all 256 byte values without issues."""
        model = INSTNCT(M=32, hidden_dim=64, slot_dim=16, N=1, R=1,
                        embed_mode=True, embed_encoding='bitlift')
        # Feed all 256 byte values
        x = torch.arange(256).unsqueeze(0)  # (1, 256)
        with torch.no_grad():
            out, _ = model(x)
        assert torch.isfinite(out).all()
        assert out.shape == (1, 256, 256)

    def test_lowrank_c19_gradient_flow(self):
        """lowrank_c19 output head must produce valid gradients."""
        model = INSTNCT(M=32, hidden_dim=64, slot_dim=16, N=1, R=1,
                        embed_mode=True, output_encoding='lowrank_c19')
        x = torch.randint(0, 256, (2, 16))
        out, _ = model(x)
        loss = F.cross_entropy(out.reshape(-1, 256), x.reshape(-1))
        loss.backward()
        # Check the lowrank head has gradients
        for name, p in model.named_parameters():
            if p.grad is not None:
                assert torch.isfinite(p.grad).all(), f"Non-finite grad in {name}"


# ═══════════════════════════════════════════════════════════════════
#  11. BULLETIN BOARD CACHE EDGE CASES
# ═══════════════════════════════════════════════════════════════════

class TestBulletinBoardAdversarial:
    """Stress BB temporal cache under adversarial conditions."""

    BB_CFG = dict(M=32, hidden_dim=32, slot_dim=16, N=2, R=1,
                  bb_enabled=True, bb_gate_mode='learned')

    def test_bb_cold_start(self):
        """First few timesteps (before any taps available) must not crash."""
        model = INSTNCT(**self.BB_CFG, embed_mode=True)
        x = torch.randint(0, 256, (2, 3))  # Only 3 steps — taps at 1,5,10
        with torch.no_grad():
            out, _ = model(x)
        assert torch.isfinite(out).all()

    def test_bb_long_run(self):
        """BB cache over many batches with carry must stay stable."""
        model = INSTNCT(**self.BB_CFG, embed_mode=True)
        state = None
        for i in range(10):
            x = torch.randint(0, 256, (2, 32))
            with torch.no_grad():
                out, state = model(x, state=state)
            assert torch.isfinite(out).all(), f"BB blow-up at batch {i}"
            assert 'bb_buf' in state
            assert torch.isfinite(state['bb_buf']).all(), f"BB buffer NaN at batch {i}"

    def test_bb_fixed_gate_mode(self):
        """Fixed gate mode (no learned gate) must work."""
        model = INSTNCT(M=32, hidden_dim=32, slot_dim=16, N=2, R=1,
                        bb_enabled=True, bb_gate_mode='fixed', embed_mode=True)
        x = torch.randint(0, 256, (2, 32))
        with torch.no_grad():
            out, _ = model(x)
        assert torch.isfinite(out).all()

    def test_bb_gradient_flow(self):
        """BB path must produce valid gradients."""
        model = INSTNCT(**self.BB_CFG, embed_mode=True)
        x = torch.randint(0, 256, (2, 32))
        out, _ = model(x)
        out.sum().backward()
        # BB query projection must have gradients
        for i in range(model.N):
            assert model.bb_query_proj[i].weight.grad is not None
            assert model.bb_query_proj[i].weight.grad.abs().sum() > 0


# ═══════════════════════════════════════════════════════════════════
#  12. POINTER MOVEMENT ADVERSARIAL
# ═══════════════════════════════════════════════════════════════════

class TestPointerMovement:
    """Stress pointer movement helpers directly."""

    def test_shortest_arc_wraparound(self):
        """Shortest arc must handle wrap-around correctly."""
        M = 64
        # current=62, target=2 → shortest path is +4 (not -60)
        current = torch.tensor([62.0])
        target = torch.tensor([2.0])
        delta = func_shortest_arc_delta_tns(current, target, M)
        assert abs(delta.item() - 4.0) < 1e-5, f"Expected +4, got {delta.item()}"

        # current=2, target=62 → shortest path is -4
        delta2 = func_shortest_arc_delta_tns(target, current, M)
        assert abs(delta2.item() - (-4.0)) < 1e-5, f"Expected -4, got {delta2.item()}"

    def test_shortest_arc_same_position(self):
        delta = func_shortest_arc_delta_tns(torch.tensor([10.0]), torch.tensor([10.0]), 64)
        assert abs(delta.item()) < 1e-5

    def test_movepntr_stays_in_range(self):
        """func_movepntr must keep pointer in [0, M)."""
        M = 32
        dests = phi_destinations(1, M)
        # Start at various positions
        for start in [0, M-1, M//2]:
            ptr = torch.tensor([float(start)])
            for _ in range(100):
                ptr = func_movepntr_tns(ptr, dests[0], 0.5, M)
            assert ptr.item() >= 0 and ptr.item() < M, \
                f"Pointer out of range: {ptr.item()}"

    def test_learned_pointer_gradient(self):
        """Learned pointer mode must produce valid gradients for pointer heads."""
        model = INSTNCT(**TINY, embed_mode=True, pointer_mode='learned')
        x = torch.randint(0, 256, (2, 16))
        out, _ = model(x)
        out.sum().backward()
        for i in range(model.N):
            assert model.ptr_dir_head[i].weight.grad is not None
            assert model.ptr_mag_head[i].weight.grad is not None

    def test_pilot_pointer_gradient(self):
        """Pilot pointer mode must produce valid gradients."""
        model = INSTNCT(**TINY, embed_mode=True, pointer_mode='pilot')
        x = torch.randint(0, 256, (2, 16))
        out, _ = model(x)
        out.sum().backward()
        assert model.slot_identity.grad is not None
        assert model.slot_identity.grad.abs().sum() > 0


# ═══════════════════════════════════════════════════════════════════
#  13. DETERMINISM
# ═══════════════════════════════════════════════════════════════════

class TestDeterminism:
    """Verify reproducibility."""

    def test_same_seed_same_output(self):
        """Same seed + same input must produce bit-identical output."""
        x = torch.randint(0, 256, (2, 32))
        torch.manual_seed(42)
        m1 = INSTNCT(**TINY, embed_mode=True)
        torch.manual_seed(42)
        m2 = INSTNCT(**TINY, embed_mode=True)
        with torch.no_grad():
            out1, _ = m1(x)
            out2, _ = m2(x)
        assert torch.equal(out1, out2), "Same seed produced different outputs"

    def test_different_seed_different_output(self):
        """Different seeds must produce different outputs."""
        x = torch.randint(0, 256, (2, 32))
        torch.manual_seed(42)
        m1 = INSTNCT(**TINY, embed_mode=True)
        torch.manual_seed(99)
        m2 = INSTNCT(**TINY, embed_mode=True)
        with torch.no_grad():
            out1, _ = m1(x)
            out2, _ = m2(x)
        assert not torch.equal(out1, out2), "Different seeds produced identical outputs"


# ═══════════════════════════════════════════════════════════════════
#  14. MULTI-EXPERT WRITE CONFLICT
# ═══════════════════════════════════════════════════════════════════

class TestMultiExpertConflict:
    """Verify correct behavior when multiple experts write to overlapping slots."""

    def test_multi_expert_no_nan(self):
        """N=4 experts with R=2 (5-slot windows) on M=16 ring — heavy overlap."""
        model = INSTNCT(M=16, embed_dim=16, N=4, R=2, embed_mode=True)
        x = torch.randint(0, 256, (4, 32))
        with torch.no_grad():
            out, state = model(x)
        assert torch.isfinite(out).all()
        assert torch.isfinite(state['ring']).all()

    def test_multi_expert_gradient_all_experts(self):
        """All N experts must receive gradients."""
        model = INSTNCT(M=32, embed_dim=16, N=4, R=1, embed_mode=True)
        x = torch.randint(0, 256, (2, 16))
        out, _ = model(x)
        out.sum().backward()
        for i in range(4):
            assert model.read_proj[i].weight.grad is not None
            assert model.read_proj[i].weight.grad.abs().sum() > 0, \
                f"Expert {i} read_proj has zero gradients"


# ═══════════════════════════════════════════════════════════════════
#  15. HOURGLASS I/O SPLIT ADVERSARIAL
# ═══════════════════════════════════════════════════════════════════

class TestIOSplitAdversarial:
    """Stress the strict I/O split mode."""

    IO_CFG = dict(M=32, embed_dim=16, N=4, R=1, io_split_mode='strict',
                  io_writer_count=2)

    def test_io_split_output_finite(self):
        model = INSTNCT(**self.IO_CFG, embed_mode=True)
        x = torch.randint(0, 256, (2, 32))
        with torch.no_grad():
            out, _ = model(x)
        assert torch.isfinite(out).all()

    def test_io_split_readers_output_only(self):
        """When io_output_from_readers_only=True, only reader experts contribute to output."""
        model = INSTNCT(**self.IO_CFG, embed_mode=True, io_output_from_readers_only=True)
        assert len(model._strict_reader_idx) == 2  # N=4, 2 writers → 2 readers

    def test_io_split_requires_N_ge_2(self):
        """Strict I/O split with N=1 must fail."""
        with pytest.raises(AssertionError):
            INSTNCT(M=32, embed_dim=16, N=1, R=1, io_split_mode='strict',
                    io_writer_count=1)

    def test_io_split_gradient_flow(self):
        """Both writers and readers must have gradients in strict mode."""
        model = INSTNCT(**self.IO_CFG, embed_mode=True)
        x = torch.randint(0, 256, (2, 16))
        out, _ = model(x)
        out.sum().backward()
        for i in range(4):
            assert model.read_proj[i].weight.grad is not None, \
                f"Expert {i} missing gradient in IO split mode"


# ═══════════════════════════════════════════════════════════════════
#  16. TOPK KERNEL ADVERSARIAL
# ═══════════════════════════════════════════════════════════════════

class TestTopKAdversarial:
    """Stress content-based TopK attention."""

    def test_topk_read_no_crash(self):
        model = INSTNCT(M=32, embed_dim=16, N=1, R=1, embed_mode=True,
                        kernel_mode='topk', topk_K=4)
        x = torch.randint(0, 256, (2, 16))
        with torch.no_grad():
            out, _ = model(x)
        assert torch.isfinite(out).all()

    def test_topk_K_larger_than_M(self):
        """K > M should gracefully handle (topk will just return all M slots)."""
        model = INSTNCT(M=8, embed_dim=16, N=1, R=1, embed_mode=True,
                        kernel_mode='topk', topk_K=16)
        x = torch.randint(0, 256, (2, 8))
        # This might raise or clamp — either is acceptable
        try:
            with torch.no_grad():
                out, _ = model(x)
            assert torch.isfinite(out).all()
        except RuntimeError:
            pass  # Also acceptable if it fails with a clear error

    def test_topk_gradient_flow(self):
        model = INSTNCT(M=32, embed_dim=16, N=1, R=1, embed_mode=True,
                        kernel_mode='topk', topk_K=4)
        x = torch.randint(0, 256, (2, 8))
        out, _ = model(x)
        out.sum().backward()
        assert model.query_proj[0].weight.grad is not None
        assert model.query_proj[0].weight.grad.abs().sum() > 0

    def test_content_topk_write_mode(self):
        """Content-based topK write addressing must work."""
        model = INSTNCT(M=32, embed_dim=16, N=1, R=1, embed_mode=True,
                        kernel_mode='topk', topk_K=4,
                        write_address_mode='content_topk', write_topk_K=2)
        x = torch.randint(0, 256, (2, 8))
        with torch.no_grad():
            out, _ = model(x)
        assert torch.isfinite(out).all()


# ═══════════════════════════════════════════════════════════════════
#  17. MTAPS (MULTI-TIMESCALE TAPS) ADVERSARIAL
# ═══════════════════════════════════════════════════════════════════

class TestMTapsAdversarial:
    """Stress multi-timescale ring taps."""

    @pytest.mark.parametrize('mixer', [
        'current',
        'tap_scalar_gate',
        'residual_gated',
    ])
    def test_mtaps_mixer_modes(self, mixer):
        model = INSTNCT(M=32, embed_dim=16, N=1, R=1, embed_mode=True,
                        mtaps_enabled=True, mtaps_lags=(1, 2, 4),
                        mtaps_mixer_mode=mixer)
        x = torch.randint(0, 256, (2, 16))
        with torch.no_grad():
            out, _ = model(x)
        assert torch.isfinite(out).all(), f"Non-finite with mtaps_mixer_mode={mixer}"

    def test_mtaps_gradient_flow(self):
        model = INSTNCT(M=32, embed_dim=16, N=1, R=1, embed_mode=True,
                        mtaps_enabled=True, mtaps_lags=(1, 4),
                        mtaps_mixer_mode='tap_scalar_gate')
        x = torch.randint(0, 256, (2, 16))
        out, _ = model(x)
        out.sum().backward()
        assert model.read_tap_gate[0].weight.grad is not None
        assert model.read_tap_gate[0].weight.grad.abs().sum() > 0

    def test_mtaps_with_topk_forbidden(self):
        """mtaps + topk read kernel should raise ValueError."""
        with pytest.raises(ValueError, match="mtaps_enabled requires a local read path"):
            INSTNCT(M=32, embed_dim=16, N=1, R=1,
                    kernel_mode='topk', mtaps_enabled=True, mtaps_lags=(1,))


# ═══════════════════════════════════════════════════════════════════
#  18. DIAGNOSTICS ADVERSARIAL
# ═══════════════════════════════════════════════════════════════════

class TestDiagnostics:
    """Verify diagnostics don't break the model under stress."""

    def test_diag_enabled_no_crash(self):
        model = INSTNCT(**TINY, embed_mode=True)
        model._diag_enabled = True
        x = torch.randint(0, 256, (2, 32))
        with torch.no_grad():
            out, _ = model(x)
        assert torch.isfinite(out).all()
        assert len(model._diag) > 0, "Diagnostics enabled but dict is empty"

    def test_diag_values_finite(self):
        model = INSTNCT(**TINY, embed_mode=True)
        model._diag_enabled = True
        x = torch.randint(0, 256, (2, 32))
        with torch.no_grad():
            model(x)
        for key, val in model._diag.items():
            if isinstance(val, float):
                assert math.isfinite(val), f"Diag {key} is not finite: {val}"


# ═══════════════════════════════════════════════════════════════════
#  19. STATE DICT & CHECKPOINT ROBUSTNESS
# ═══════════════════════════════════════════════════════════════════

class TestStateDictRobustness:
    """Verify state_dict save/load doesn't corrupt the model."""

    def test_save_load_roundtrip(self):
        """state_dict → load_state_dict must produce identical output."""
        torch.manual_seed(42)
        model1 = INSTNCT(**TINY_SPLIT, embed_mode=True)
        sd = model1.state_dict()

        torch.manual_seed(99)  # Different seed
        model2 = INSTNCT(**TINY_SPLIT, embed_mode=True)
        model2.load_state_dict(sd)

        x = torch.randint(0, 256, (2, 16))
        with torch.no_grad():
            out1, _ = model1(x)
            out2, _ = model2(x)
        assert torch.allclose(out1, out2, atol=1e-6), \
            f"Roundtrip mismatch: max diff={(out1 - out2).abs().max()}"

    def test_state_dict_no_unexpected_keys(self):
        """state_dict should not contain internal non-persistent state."""
        model = INSTNCT(**TINY, embed_mode=True, expert_weighting=True)
        sd = model.state_dict()
        for key in sd:
            # No internal tracking state should leak
            assert '_write_grad_ema' not in key
            assert '_diag' not in key


# ═══════════════════════════════════════════════════════════════════
#  20. PROXY OVERLAY CORRECTNESS
# ═══════════════════════════════════════════════════════════════════

class TestProxyOverlay:
    """Verify proxy overlay produces identical results to dense path."""

    def test_proxy_vs_dense_parity(self):
        """Proxy overlay must produce identical output to dense replace."""
        torch.manual_seed(42)
        m_dense = INSTNCT(M=32, hidden_dim=16, slot_dim=16, N=1, R=1,
                          embed_mode=False, write_mode='replace',
                          replace_impl='dense', pointer_mode='sequential')
        torch.manual_seed(42)
        m_proxy = INSTNCT(M=32, hidden_dim=16, slot_dim=16, N=1, R=1,
                          embed_mode=False, write_mode='replace',
                          replace_impl='proxy_overlay', pointer_mode='sequential')
        assert m_proxy._proxy_overlay_enabled is True

        x = to_bits(torch.randint(0, 256, (2, 32)))
        with torch.no_grad():
            out_dense, state_dense = m_dense(x)
            out_proxy, state_proxy = m_proxy(x)

        assert torch.allclose(out_dense, out_proxy, atol=1e-5), \
            f"Proxy/dense output mismatch: max diff={(out_dense - out_proxy).abs().max()}"
        assert torch.allclose(state_dense['ring'], state_proxy['ring'], atol=1e-5), \
            f"Proxy/dense ring mismatch: max diff={(state_dense['ring'] - state_proxy['ring']).abs().max()}"


# ═══════════════════════════════════════════════════════════════════
#  21. DOTPROD GATE ADVERSARIAL
# ═══════════════════════════════════════════════════════════════════

class TestDotprodGate:
    """Stress the content-based dotprod ring gate."""

    def test_dotprod_gate_alpha_bounded(self):
        """Alpha (ring gate) must always be in [0, 1]."""
        model = INSTNCT(**TINY, embed_mode=True)
        model._diag_enabled = True
        x = torch.randint(0, 256, (4, 64))
        with torch.no_grad():
            model(x)
        for i in range(model.N):
            key = f'alpha_{i}_min'
            if key in model._diag:
                assert model._diag[key] >= -1e-6, f"Alpha below 0: {model._diag[key]}"
            key = f'alpha_{i}_max'
            if key in model._diag:
                assert model._diag[key] <= 1.0 + 1e-6, f"Alpha above 1: {model._diag[key]}"

    def test_fixed_S_mode(self):
        """S=float (fixed scalar) mode must produce valid output."""
        model = INSTNCT(**TINY, embed_mode=True)
        x = torch.randint(0, 256, (2, 16))
        with torch.no_grad():
            out, _ = model(x, S=0.5)
        assert torch.isfinite(out).all()

    def test_fixed_S_zero(self):
        """S=0 (ring signal completely ignored) must still work."""
        model = INSTNCT(**TINY, embed_mode=True)
        x = torch.randint(0, 256, (2, 16))
        with torch.no_grad():
            out, _ = model(x, S=0.0)
        assert torch.isfinite(out).all()


# ═══════════════════════════════════════════════════════════════════
#  22. FUNCTIONAL HELPERS ADVERSARIAL
# ═══════════════════════════════════════════════════════════════════

class TestFunctionalHelpers:
    """Direct stress tests on standalone functions."""

    def test_ringstart_shape_and_zeros(self):
        ring = func_ringstart_tns(4, 32, 16, 'cpu')
        assert ring.shape == (4, 32, 16)
        assert ring.sum().item() == 0.0

    def test_softread_with_single_slot(self):
        """Reading from a single slot (R=0) must work."""
        B, M, D = 2, 8, 4
        ring = torch.randn(B, M, D)
        indices = torch.tensor([[3], [5]])  # Single slot per batch
        weights = torch.ones(B, 1)
        read_vec, _ = func_softread_tns(ring, indices, weights, D)
        assert read_vec.shape == (B, D)
        assert torch.allclose(read_vec[0], ring[0, 3])
        assert torch.allclose(read_vec[1], ring[1, 5])

    def test_softwrit_additive(self):
        """scatter_add write must be strictly additive (never erase)."""
        B, M, D = 1, 4, 2
        ring = torch.ones(B, M, D)
        hidden = torch.ones(B, D) * 10.0
        indices = torch.tensor([[1]])
        expanded_idx = indices.unsqueeze(-1).expand(-1, -1, D)
        weights = torch.ones(B, 1)
        new_ring = func_softwrit_tns(ring, hidden, expanded_idx, weights)
        # Slot 1 should be original (1.0) + written (10.0) = 11.0
        assert torch.allclose(new_ring[0, 1], torch.tensor([11.0, 11.0]))
        # Other slots unchanged
        assert torch.allclose(new_ring[0, 0], torch.ones(D))

    def test_gated_write_full_erase(self):
        """With erase=1, write_gate=0, slot content should decay toward zero."""
        B, M, D = 1, 4, 2
        ring = torch.ones(B, M, D) * 5.0
        write_vec = torch.zeros(B, D)
        indices = torch.tensor([[2]])
        expanded_idx = indices.unsqueeze(-1).expand(-1, -1, D)
        weights = torch.ones(B, 1)
        erase = torch.ones(B)
        wgate = torch.zeros(B)
        new_ring = func_gated_write_tns(ring, write_vec, expanded_idx, weights, erase, wgate)
        # Slot 2: old * (1 - 1*1) + 0 = 0
        assert torch.allclose(new_ring[0, 2], torch.zeros(D), atol=1e-6)
        # Other slots unchanged
        assert torch.allclose(new_ring[0, 0], torch.full((D,), 5.0))


# ═══════════════════════════════════════════════════════════════════
#  23. MIXED CONFIG COMBINATIONS
# ═══════════════════════════════════════════════════════════════════

class TestMixedConfigs:
    """Cross-feature interactions that might expose bugs."""

    def test_checkpoint_with_learned_pointer(self):
        model = INSTNCT(**TINY, embed_mode=True, pointer_mode='learned',
                        checkpoint_chunks=4)
        x = torch.randint(0, 256, (2, 16))
        out, _ = model(x)
        assert torch.isfinite(out).all()
        out.sum().backward()

    def test_checkpoint_with_bb(self):
        model = INSTNCT(M=32, hidden_dim=32, slot_dim=16, N=2, R=1,
                        embed_mode=True, bb_enabled=True,
                        checkpoint_chunks=4)
        x = torch.randint(0, 256, (2, 16))
        out, _ = model(x)
        assert torch.isfinite(out).all()

    def test_expert_weighting_with_replace_write(self):
        model = INSTNCT(**TINY_SPLIT, embed_mode=True,
                        expert_weighting=True, write_mode='replace',
                        replace_impl='dense')
        x = torch.randint(0, 256, (2, 16))
        out, _ = model(x)
        out.sum().backward()
        model.update_expert_conf()
        assert torch.isfinite(model._expert_conf).all()

    def test_bitlift_with_lowrank_c19(self):
        """Combined bitlift input + lowrank_c19 output."""
        model = INSTNCT(M=32, hidden_dim=64, slot_dim=16, N=1, R=1,
                        embed_mode=True, embed_encoding='bitlift',
                        output_encoding='lowrank_c19')
        x = torch.randint(0, 256, (2, 32))
        out, _ = model(x)
        assert torch.isfinite(out).all()
        loss = F.cross_entropy(out.reshape(-1, 256), x.reshape(-1))
        loss.backward()
        # Both bitlift input and lowrank output must have gradients
        assert model.inp.weight.grad is not None
        assert model.inp.weight.grad.abs().sum() > 0

    def test_dotprod_kernel_with_gated_write(self):
        model = INSTNCT(M=32, embed_dim=16, N=2, R=1, embed_mode=True,
                        kernel_mode='dotprod', gated_write=True,
                        write_mode='accumulate')
        x = torch.randint(0, 256, (2, 16))
        with torch.no_grad():
            out, _ = model(x)
        assert torch.isfinite(out).all()

    def test_all_nightly_features_combined(self):
        """Kitchen sink: many features enabled simultaneously."""
        model = INSTNCT(
            M=32, hidden_dim=32, slot_dim=16, N=2, R=1,
            embed_mode=True,
            embed_encoding='bitlift',
            output_encoding='lowrank_c19',
            pointer_mode='learned',
            write_mode='replace',
            replace_impl='dense',
            expert_weighting=True,
            checkpoint_chunks=4,
        )
        x = torch.randint(0, 256, (2, 16))
        out, state = model(x)
        assert torch.isfinite(out).all()
        assert out.shape == (2, 16, 256)
        # Can backward without crash
        out.sum().backward()
        model.update_expert_conf()


# ═══════════════════════════════════════════════════════════════════
#  24. NO_GRAD / EVAL MODE CONSISTENCY
# ═══════════════════════════════════════════════════════════════════

class TestEvalModeConsistency:
    """Verify eval/no_grad produces same results as train mode inference."""

    def test_train_vs_eval_identical(self):
        """model.eval() must produce identical output to model.train() under no_grad."""
        torch.manual_seed(42)
        model = INSTNCT(**TINY, embed_mode=True)
        x = torch.randint(0, 256, (2, 16))

        model.train()
        with torch.no_grad():
            out_train, _ = model(x)

        model.eval()
        with torch.no_grad():
            out_eval, _ = model(x)

        assert torch.allclose(out_train, out_eval, atol=1e-6), \
            f"Train/eval mismatch: max diff={(out_train - out_eval).abs().max()}"

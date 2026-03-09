"""Unit tests for INSTNCT v4 model — instnct.py

Tests cover:
  - forward pass output shapes (binary + embed mode)
  - no NaN / Inf in output
  - phi_destinations table correctness
  - parameter count sanity
"""

import numpy as np
import torch
import pytest
import instnct as instnct_mod
from instnct import INSTNCT, phi_destinations

# Tiny config — same architecture, far fewer parameters → tests run in <1s
TINY = dict(M=32, embed_dim=16, N=2, R=1)


def to_bits(x: torch.Tensor) -> torch.Tensor:
    """Convert (B, T) int byte tensor → (B, T, 8) float bit tensor.

    Binary mode forward() expects 8 explicit bit channels per timestep,
    not a raw integer token index."""
    return ((x.unsqueeze(-1) >> torch.arange(7, -1, -1)) & 1).float()


# ══════════════════════════════════════════════════════════════════
#  to_bits correctness — must match np.unpackbits (used in train.py)
# ══════════════════════════════════════════════════════════════════

def test_to_bits_matches_numpy():
    """to_bits must produce the same MSB-first bit vectors as np.unpackbits,
    which is the canonical conversion used in ByteDataset.sample_batch."""
    vals = torch.tensor([[0, 127, 128, 255]])          # edge cases: 0, max signed, min high-bit, all-ones
    bits = to_bits(vals)                                # (1, 4, 8)
    for i, v in enumerate(vals[0]):
        expected = np.unpackbits(np.array([v.item()], dtype=np.uint8))
        assert list(bits[0, i].int().tolist()) == list(expected), \
            f"to_bits({v.item()}) = {bits[0,i].tolist()} but np.unpackbits = {expected.tolist()}"


# ══════════════════════════════════════════════════════════════════
#  Forward pass — output shape
# ══════════════════════════════════════════════════════════════════

def test_forward_shape_binary():
    """Binary mode output must be [B, T, 8] — 8 predicted bits per position."""
    model = INSTNCT(**TINY, embed_mode=False)
    x = to_bits(torch.randint(0, 256, (2, 16)))
    out, _ = model(x)
    assert out.shape == (2, 16, 8), f"Expected (2, 16, 8), got {out.shape}"


def test_forward_shape_embed():
    """Embed mode output must be [B, T, 256] — one logit per byte value."""
    model = INSTNCT(**TINY, embed_mode=True)
    x = torch.randint(0, 256, (2, 16))
    out, _ = model(x)
    assert out.shape == (2, 16, 256), f"Expected (2, 16, 256), got {out.shape}"


def test_forward_batch_size_one():
    """B=1 must work — single-sequence inference."""
    model = INSTNCT(**TINY, embed_mode=False)
    x = to_bits(torch.randint(0, 256, (1, 32)))
    out, _ = model(x)
    assert out.shape == (1, 32, 8)


def test_forward_single_timestep():
    """T=1 must work — pathological sequence length."""
    model = INSTNCT(**TINY, embed_mode=False)
    x = to_bits(torch.randint(0, 256, (4, 1)))
    out, _ = model(x)
    assert out.shape == (4, 1, 8)


# ══════════════════════════════════════════════════════════════════
#  Numerical stability
# ══════════════════════════════════════════════════════════════════

def test_no_nan_in_output():
    """Forward pass must never produce NaN."""
    model = INSTNCT(**TINY, embed_mode=False)
    x = to_bits(torch.randint(0, 256, (4, 32)))
    out, _ = model(x)
    assert not torch.isnan(out).any(), "NaN detected in model output"


def test_no_inf_in_output():
    """Forward pass must never produce Inf."""
    model = INSTNCT(**TINY, embed_mode=False)
    x = to_bits(torch.randint(0, 256, (4, 32)))
    out, _ = model(x)
    assert not torch.isinf(out).any(), "Inf detected in model output"


def test_no_nan_in_output_embed():
    """Embed mode forward pass must never produce NaN."""
    model = INSTNCT(**TINY, embed_mode=True)
    x = torch.randint(0, 256, (4, 32))
    out, _ = model(x)
    assert not torch.isnan(out).any(), "NaN detected in embed mode output"


def test_no_inf_in_output_embed():
    """Embed mode forward pass must never produce Inf."""
    model = INSTNCT(**TINY, embed_mode=True)
    x = torch.randint(0, 256, (4, 32))
    out, _ = model(x)
    assert not torch.isinf(out).any(), "Inf detected in embed mode output"


# ══════════════════════════════════════════════════════════════════
#  phi_destinations table
# ══════════════════════════════════════════════════════════════════

def test_phi_destinations_shape():
    """Table must have shape (N, M)."""
    dests = phi_destinations(4, 64)
    assert dests.shape == (4, 64)


def test_phi_destinations_in_range():
    """All destination values must be valid slot indices: 0 ≤ dest < M."""
    M, N = 64, 6
    dests = phi_destinations(N, M)
    assert dests.min().item() >= 0
    assert dests.max().item() < M


def test_phi_destinations_beings_differ():
    """From every slot, each being must jump to a DIFFERENT destination.
    This is the whole point of the +n offset — no two beings collide."""
    N, M = 6, 256
    dests = phi_destinations(N, M)
    # check ALL columns — not just slot 0
    for col in range(M):
        vals = dests[:, col]
        assert vals.unique().numel() == N, \
            f"Collision at slot {col}: {vals.tolist()}"


# ══════════════════════════════════════════════════════════════════
#  Parameter sanity
# ══════════════════════════════════════════════════════════════════

def test_model_has_parameters():
    """Model must have trainable parameters."""
    model = INSTNCT(**TINY)
    n = sum(p.numel() for p in model.parameters())
    assert n > 0, "Model has no parameters"


def test_embed_mode_has_more_params_than_binary():
    """Embed mode with 'learned' encoding adds a 256-entry embedding → more params than binary.
    Note: bitlift encoding may have fewer params than binary — only test with 'learned'."""
    binary = INSTNCT(**TINY, embed_mode=False)
    embed  = INSTNCT(**TINY, embed_mode=True, embed_encoding='learned', output_encoding='learned')
    n_bin  = sum(p.numel() for p in binary.parameters())
    n_emb  = sum(p.numel() for p in embed.parameters())
    assert n_emb > n_bin


# ══════════════════════════════════════════════════════════════════
#  kernel_mode — attention kernel shape
# ══════════════════════════════════════════════════════════════════

def test_forward_kernel_gaussian():
    """Gaussian kernel must produce valid output with correct shape."""
    model = INSTNCT(**TINY, embed_mode=False, kernel_mode='gaussian')
    x = to_bits(torch.randint(0, 256, (2, 16)))
    out, _ = model(x)
    assert out.shape == (2, 16, 8)
    assert not torch.isnan(out).any(), "NaN in gaussian kernel output"


def test_forward_kernel_uniform():
    """Uniform kernel must produce valid output with correct shape."""
    model = INSTNCT(**TINY, embed_mode=False, kernel_mode='uniform')
    x = to_bits(torch.randint(0, 256, (2, 16)))
    out, _ = model(x)
    assert out.shape == (2, 16, 8)
    assert not torch.isnan(out).any(), "NaN in uniform kernel output"


def test_kernel_mode_invalid():
    """Invalid kernel_mode must raise AssertionError."""
    with pytest.raises(AssertionError):
        INSTNCT(**TINY, kernel_mode='invalid')


def test_gradient_flows_through_kernel_modes():
    """All kernel modes must produce valid gradients for trainable params."""
    for mode in ('uniform', 'vshape', 'gaussian'):
        model = INSTNCT(**TINY, embed_mode=False, kernel_mode=mode)
        x = to_bits(torch.randint(0, 256, (2, 8)))
        out, _ = model(x)
        loss = out.sum()
        loss.backward()
        # R is now fixed (no R_param), check other trainable params get gradients
        grads = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
        assert len(grads) > 0, f"No gradients at all with kernel_mode='{mode}'"


# ══════════════════════════════════════════════════════════════════
#  hidden_dim / slot_dim split
# ══════════════════════════════════════════════════════════════════

SPLIT = dict(M=32, hidden_dim=64, slot_dim=16, N=2, R=1)


def test_split_forward_shape_binary():
    """Split mode binary forward must produce correct output shape."""
    model = INSTNCT(**SPLIT, embed_mode=False)
    x = to_bits(torch.randint(0, 256, (2, 8)))
    out, _ = model(x)
    assert out.shape == (2, 8, 8)
    assert torch.isfinite(out).all()


def test_split_forward_shape_embed():
    """Split mode embed forward must produce correct output shape."""
    model = INSTNCT(**SPLIT, embed_mode=True)
    x = torch.randint(0, 256, (2, 8))
    out, _ = model(x)
    assert out.shape == (2, 8, 256)
    assert torch.isfinite(out).all()


def test_split_gradient_flows_through_write_proj():
    """write_proj must receive non-zero gradients in split mode."""
    model = INSTNCT(**SPLIT, embed_mode=True)
    x = torch.randint(0, 256, (2, 8))
    model(x)[0].sum().backward()
    assert model.write_proj is not None
    for i in range(model.N):
        assert model.write_proj[i].weight.grad is not None, \
            f"write_proj[{i}].weight.grad is None"
        assert model.write_proj[i].weight.grad.abs().sum() > 0, \
            f"write_proj[{i}].weight.grad is all zeros"


def test_split_gradient_flows_through_read_proj():
    """read_proj must still receive gradients in split mode (slot_dim→hidden_dim)."""
    model = INSTNCT(**SPLIT, embed_mode=True)
    x = torch.randint(0, 256, (2, 8))
    model(x)[0].sum().backward()
    for i in range(model.N):
        assert model.read_proj[i].weight.grad.abs().sum() > 0, \
            f"read_proj[{i}].weight.grad is all zeros"


def test_embed_dim_alias_sets_both():
    """embed_dim=X must set hidden_dim=slot_dim=X."""
    model = INSTNCT(M=32, embed_dim=32, N=2)
    assert model.hidden_dim == 32
    assert model.slot_dim == 32


def test_equal_dims_no_write_proj():
    """When hidden_dim==slot_dim, write_proj must be None (zero overhead)."""
    model = INSTNCT(M=32, embed_dim=16, N=2)
    assert model.write_proj is None


def test_numerical_equivalence_when_equal():
    """When hidden_dim==slot_dim, output must be bit-identical to embed_dim path."""
    torch.manual_seed(42)
    old = INSTNCT(M=32, embed_dim=16, N=2, embed_mode=True)
    torch.manual_seed(42)
    new = INSTNCT(M=32, hidden_dim=16, slot_dim=16, N=2, embed_mode=True)
    # Weights should be identical from same seed
    x = torch.randint(0, 256, (2, 8))
    with torch.no_grad():
        out_old, _ = old(x)
        out_new, _ = new(x)
    assert torch.allclose(out_old, out_new, atol=1e-6), \
        f"max diff = {(out_old - out_new).abs().max().item():.2e}"


def test_split_large_hidden_small_slot():
    """Extreme split (hidden=256, slot=8) must still produce valid output."""
    model = INSTNCT(M=32, hidden_dim=256, slot_dim=8, N=2, embed_mode=True)
    x = torch.randint(0, 256, (2, 4))
    out, _ = model(x)
    assert out.shape == (2, 4, 256)
    assert torch.isfinite(out).all()
    # Gradient check
    out.sum().backward()
    assert model.write_proj[0].weight.grad.abs().sum() > 0


# ══════════════════════════════════════════════════════════════════
#  Gradient Checkpointing
# ══════════════════════════════════════════════════════════════════

CKPT_CFG = dict(M=32, hidden_dim=64, slot_dim=16, N=2, R=1)


def test_checkpoint_numerical_equivalence_embed():
    """Checkpointed forward must produce identical output to non-checkpointed."""
    torch.manual_seed(99)
    ref = INSTNCT(**CKPT_CFG, embed_mode=True, checkpoint_chunks=0)
    torch.manual_seed(99)
    ckpt = INSTNCT(**CKPT_CFG, embed_mode=True, checkpoint_chunks=4)
    x = torch.randint(0, 256, (2, 16))
    with torch.no_grad():
        out_ref, _ = ref(x)
        out_ckpt, _ = ckpt(x)
    assert torch.allclose(out_ref, out_ckpt, atol=1e-6), \
        f"max diff = {(out_ref - out_ckpt).abs().max().item():.2e}"


def test_checkpoint_numerical_equivalence_binary():
    """Checkpointed forward must produce identical output in binary mode."""
    torch.manual_seed(99)
    ref = INSTNCT(**CKPT_CFG, embed_mode=False, checkpoint_chunks=0)
    torch.manual_seed(99)
    ckpt = INSTNCT(**CKPT_CFG, embed_mode=False, checkpoint_chunks=4)
    x = to_bits(torch.randint(0, 256, (2, 16)))
    with torch.no_grad():
        out_ref, _ = ref(x)
        out_ckpt, _ = ckpt(x)
    assert torch.allclose(out_ref, out_ckpt, atol=1e-6), \
        f"max diff = {(out_ref - out_ckpt).abs().max().item():.2e}"


def test_checkpoint_gradient_flow_write_proj():
    """Gradients must flow through write_proj in checkpointed mode."""
    model = INSTNCT(**CKPT_CFG, embed_mode=True, checkpoint_chunks=4)
    x = torch.randint(0, 256, (2, 16))
    model(x)[0].sum().backward()
    for i in range(model.N):
        assert model.write_proj[i].weight.grad.abs().sum() > 0, \
            f"write_proj[{i}] gradient is zero under checkpointing"


def test_checkpoint_R_is_fixed_buffer():
    """R is now a fixed buffer (not learnable), verify it exists and is correct."""
    model = INSTNCT(**CKPT_CFG, embed_mode=True, checkpoint_chunks=4)
    assert hasattr(model, '_R_eff'), "_R_eff buffer missing"
    assert not model._R_eff.requires_grad, "_R_eff should not require gradients"
    expected = CKPT_CFG['R'] + 0.5
    assert torch.allclose(model._R_eff, torch.full_like(model._R_eff, expected))


def test_checkpoint_disabled_at_eval():
    """Checkpointing must not activate during model.eval() / no_grad."""
    model = INSTNCT(**CKPT_CFG, embed_mode=True, checkpoint_chunks=4)
    model.eval()
    x = torch.randint(0, 256, (2, 8))
    with torch.no_grad():
        out, _ = model(x)
    assert out.shape == (2, 8, 256)
    assert torch.isfinite(out).all()


def test_checkpoint_uneven_chunks():
    """T not divisible by checkpoint_chunks must still work (last chunk smaller)."""
    model = INSTNCT(**CKPT_CFG, embed_mode=True, checkpoint_chunks=3)
    x = torch.randint(0, 256, (2, 10))  # 10 / 3 = 3 full + 1 partial
    out, _ = model(x)
    assert out.shape == (2, 10, 256)
    assert torch.isfinite(out).all()


# ══════════════════════════════════════════════════════════════════
#  Expert Write Weighting
# ══════════════════════════════════════════════════════════════════

XPRT_CFG = dict(M=32, hidden_dim=64, slot_dim=16, N=2, R=1)


def test_expert_weighting_disabled_no_overhead():
    """When expert_weighting=False, _expert_conf is None, zero overhead."""
    model = INSTNCT(**XPRT_CFG, embed_mode=True, expert_weighting=False)
    assert model._expert_conf is None
    assert model._write_grad_ema is None


def test_expert_weighting_init_equal():
    """Expert confidence starts at 1.0 each (sum=N)."""
    model = INSTNCT(**XPRT_CFG, embed_mode=True, expert_weighting=True)
    assert model._expert_conf is not None
    assert torch.allclose(model._expert_conf, torch.ones(2))


def test_expert_weighting_output_unchanged_when_equal():
    """When all weights are 1.0, output must match non-weighted baseline."""
    torch.manual_seed(42)
    ref = INSTNCT(**XPRT_CFG, embed_mode=True, expert_weighting=False)
    torch.manual_seed(42)
    wgt = INSTNCT(**XPRT_CFG, embed_mode=True, expert_weighting=True)
    x = torch.randint(0, 256, (2, 16))
    with torch.no_grad():
        out_ref, _ = ref(x)
        out_wgt, _ = wgt(x)
    assert torch.allclose(out_ref, out_wgt, atol=1e-6), \
        f"max diff = {(out_ref - out_wgt).abs().max().item():.2e}"


def test_expert_weighting_gradient_flow():
    """Gradients must flow through weighted write_vec."""
    model = INSTNCT(**XPRT_CFG, embed_mode=True, expert_weighting=True)
    x = torch.randint(0, 256, (2, 16))
    model(x)[0].sum().backward()
    for i in range(model.N):
        assert model.write_proj[i].weight.grad.abs().sum() > 0


def test_expert_weighting_update_changes_conf():
    """update_expert_conf() must change weights after backward."""
    model = INSTNCT(**XPRT_CFG, embed_mode=True, expert_weighting=True)
    x = torch.randint(0, 256, (2, 16))
    model(x)[0].sum().backward()
    old_conf = model._expert_conf.clone()
    model.update_expert_conf()
    assert not torch.allclose(old_conf, model._expert_conf), \
        "Confidence unchanged after update"


def test_expert_weighting_conf_sums_to_N():
    """Confidence weights must sum to N after update."""
    model = INSTNCT(**XPRT_CFG, embed_mode=True, expert_weighting=True)
    x = torch.randint(0, 256, (2, 16))
    model(x)[0].sum().backward()
    model.update_expert_conf()
    assert abs(model._expert_conf.sum().item() - model.N) < 1e-5


def test_expert_weighting_no_state_dict_pollution():
    """_write_grad_ema and _expert_conf must NOT appear in state_dict."""
    model = INSTNCT(**XPRT_CFG, embed_mode=True, expert_weighting=True)
    sd = model.state_dict()
    assert '_write_grad_ema' not in sd
    assert '_expert_conf' not in sd


def test_expert_weighting_with_checkpoint():
    """Expert weighting must work with gradient checkpointing."""
    model = INSTNCT(**XPRT_CFG, embed_mode=True,
                    expert_weighting=True, checkpoint_chunks=4)
    x = torch.randint(0, 256, (2, 16))
    model(x)[0].sum().backward()
    model.update_expert_conf()
    assert model._expert_conf.sum().item() > 0
    assert torch.isfinite(model._expert_conf).all()


def test_expert_weighting_floor_prevents_zero():
    """No expert weight should drop below floor value."""
    model = INSTNCT(**XPRT_CFG, embed_mode=True, expert_weighting=True)
    # Artificially set extreme EMA to test floor
    model._write_grad_ema = torch.tensor([0.001, 100.0])
    model.update_expert_conf(floor=0.1)
    assert model._expert_conf.min().item() >= 0.09  # floor ≈ 0.1


# ══════════════════════════════════════════════════════════════
#  Fixed Encoding (Hadamard / Sincos)
# ══════════════════════════════════════════════════════════════

ENC_CFG = dict(M=32, hidden_dim=64, slot_dim=16, N=2, R=1)


def test_hadamard_encoding_output_shape():
    """Hadamard encoding produces correct output shape."""
    model = INSTNCT(**ENC_CFG, embed_mode=True, embed_encoding='hadamard')
    x = torch.randint(0, 256, (2, 16))
    out, _ = model(x)
    assert out.shape == (2, 16, 256)


def test_sincos_encoding_output_shape():
    """Sincos encoding produces correct output shape."""
    model = INSTNCT(**ENC_CFG, embed_mode=True, embed_encoding='sincos')
    x = torch.randint(0, 256, (2, 16))
    out, _ = model(x)
    assert out.shape == (2, 16, 256)


def test_fixed_encoding_no_inp_parameter():
    """Fixed encoding models should not have self.inp (nn.Embedding)."""
    model = INSTNCT(**ENC_CFG, embed_mode=True, embed_encoding='hadamard')
    assert model.inp is None
    assert hasattr(model, '_fixed_table')
    assert model._fixed_table.shape == (256, ENC_CFG['hidden_dim'])
    assert not model._fixed_table.requires_grad


def test_fixed_encoding_in_state_dict():
    """_fixed_table should be in state_dict (as buffer, for checkpoint compat)."""
    model = INSTNCT(**ENC_CFG, embed_mode=True, embed_encoding='hadamard')
    sd = model.state_dict()
    assert '_fixed_table' in sd


def test_hadamard_orthogonality():
    """Hadamard rows must be pairwise orthogonal (requires hidden_dim >= 256)."""
    cfg = dict(M=32, hidden_dim=256, slot_dim=16, N=2, R=1)
    model = INSTNCT(**cfg, embed_mode=True, embed_encoding='hadamard')
    table = model._fixed_table
    for i in range(0, 10):
        for j in range(i + 1, 10):
            sim = torch.nn.functional.cosine_similarity(
                table[i].unsqueeze(0), table[j].unsqueeze(0))
            assert abs(sim.item()) < 1e-5, f"rows {i},{j} not orthogonal: {sim.item()}"


def test_fixed_encoding_gradient_flow():
    """Gradients must flow through fixed table to downstream params."""
    model = INSTNCT(**ENC_CFG, embed_mode=True, embed_encoding='hadamard',
                    output_encoding='learned')
    x = torch.randint(0, 256, (2, 16))
    model(x)[0].sum().backward()
    assert model.out.weight.grad is not None
    assert model.out.weight.grad.abs().sum() > 0


def test_fixed_encoding_fewer_params():
    """Fixed encoding must have fewer params than learned."""
    learned = INSTNCT(**ENC_CFG, embed_mode=True, embed_encoding='learned')
    fixed = INSTNCT(**ENC_CFG, embed_mode=True, embed_encoding='hadamard')
    p_learned = sum(p.numel() for p in learned.parameters())
    p_fixed = sum(p.numel() for p in fixed.parameters())
    assert p_learned - p_fixed == 256 * ENC_CFG['hidden_dim']


def test_fixed_encoding_with_checkpoint():
    """Fixed encoding must work with gradient checkpointing."""
    model = INSTNCT(**ENC_CFG, embed_mode=True,
                    embed_encoding='hadamard', checkpoint_chunks=4)
    x = torch.randint(0, 256, (2, 16))
    out, _ = model(x)
    out.sum().backward()
    assert torch.isfinite(out).all()


def test_learned_encoding_unchanged():
    """embed_encoding='learned' produces deterministic, valid output."""
    torch.manual_seed(42)
    m1 = INSTNCT(**ENC_CFG, embed_mode=True, embed_encoding='learned', output_encoding='learned')
    torch.manual_seed(42)
    m2 = INSTNCT(**ENC_CFG, embed_mode=True, embed_encoding='learned', output_encoding='learned')
    x = torch.randint(0, 256, (2, 16))
    with torch.no_grad():
        assert torch.allclose(m1(x)[0], m2(x)[0], atol=1e-6)


def test_invalid_encoding_raises():
    """Unknown encoding type must raise ValueError."""
    try:
        INSTNCT(**ENC_CFG, embed_mode=True, embed_encoding='foobar')
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


# ══════════════════════════════════════════════════════════════
#  Fixed Output Encoding (Hadamard / Sincos)
# ══════════════════════════════════════════════════════════════

def test_hadamard_output_encoding_shape():
    """Hadamard output encoding produces correct output shape."""
    model = INSTNCT(**ENC_CFG, embed_mode=True, output_encoding='hadamard')
    x = torch.randint(0, 256, (2, 16))
    out, _ = model(x)
    assert out.shape == (2, 16, 256)


def test_sincos_output_encoding_shape():
    """Sincos output encoding produces correct output shape."""
    model = INSTNCT(**ENC_CFG, embed_mode=True, output_encoding='sincos')
    x = torch.randint(0, 256, (2, 16))
    out, _ = model(x)
    assert out.shape == (2, 16, 256)


def test_fixed_output_no_out_parameter():
    """Fixed output encoding should not have self.out (nn.Linear)."""
    model = INSTNCT(**ENC_CFG, embed_mode=True, output_encoding='hadamard')
    assert model.out is None
    assert hasattr(model, '_fixed_output_table')
    assert model._fixed_output_table.shape == (256, ENC_CFG['hidden_dim'])
    assert not model._fixed_output_table.requires_grad


def test_fixed_output_in_state_dict():
    """_fixed_output_table should be in state_dict (as buffer)."""
    model = INSTNCT(**ENC_CFG, embed_mode=True, output_encoding='hadamard')
    sd = model.state_dict()
    assert '_fixed_output_table' in sd


def test_fixed_output_gradient_flow():
    """Gradients must flow through fixed output to upstream params."""
    model = INSTNCT(**ENC_CFG, embed_mode=True, output_encoding='hadamard')
    x = torch.randint(0, 256, (2, 16))
    model(x)[0].sum().backward()
    # read_proj must have gradients (upstream of output)
    assert model.read_proj[0].weight.grad is not None
    assert model.read_proj[0].weight.grad.abs().sum() > 0


def test_fixed_output_fewer_params():
    """Fixed output encoding must have fewer params than learned."""
    learned = INSTNCT(**ENC_CFG, embed_mode=True, output_encoding='learned')
    fixed   = INSTNCT(**ENC_CFG, embed_mode=True, output_encoding='hadamard')
    p_learned = sum(p.numel() for p in learned.parameters())
    p_fixed   = sum(p.numel() for p in fixed.parameters())
    # Difference should be 256 * hidden_dim + 256 (weight + bias of nn.Linear)
    assert p_learned - p_fixed == 256 * ENC_CFG['hidden_dim'] + 256


def test_both_fixed_encoding_minimal_params():
    """Input + output both fixed → only core params remain."""
    both = INSTNCT(**ENC_CFG, embed_mode=True,
                   embed_encoding='hadamard', output_encoding='hadamard')
    learned = INSTNCT(**ENC_CFG, embed_mode=True,
                      embed_encoding='learned', output_encoding='learned')
    p_both = sum(p.numel() for p in both.parameters())
    p_learned = sum(p.numel() for p in learned.parameters())
    # Both fixed should save input embedding + output linear
    H = ENC_CFG['hidden_dim']
    expected_savings = 256 * H + (256 * H + 256)  # embedding + linear(weight+bias)
    assert p_learned - p_both == expected_savings


def test_invalid_output_encoding_raises():
    """Unknown output encoding type must raise ValueError."""
    try:
        INSTNCT(**ENC_CFG, embed_mode=True, output_encoding='foobar')
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


def test_output_encoding_default_learned():
    """Two models with explicit output_encoding='learned' must match."""
    torch.manual_seed(42)
    ref = INSTNCT(**ENC_CFG, embed_mode=True, embed_encoding='learned', output_encoding='learned')
    torch.manual_seed(42)
    exp = INSTNCT(**ENC_CFG, embed_mode=True, embed_encoding='learned', output_encoding='learned')
    x = torch.randint(0, 256, (2, 16))
    with torch.no_grad():
        assert torch.allclose(ref(x)[0], exp(x)[0], atol=1e-6)


# ══════════════════════════════════════════════════════════════
#  Compile stabilization
# ══════════════════════════════════════════════════════════════

def test_compile_chunk_bypasses_grad_checkpoint(monkeypatch):
    """Chunk compile must take precedence over grad checkpointing."""
    model = INSTNCT(
        M=32,
        hidden_dim=32,
        slot_dim=16,
        N=1,
        R=1,
        embed_mode=True,
        checkpoint_chunks=4,
        expert_weighting=False,
        embed_encoding='learned',
        output_encoding='learned',
    )
    model._compile_mode = 'chunk'
    model._compile_chunks = True
    model.compile_chunk_size = 4

    monkeypatch.setattr(torch, 'compile', lambda fn, mode=None: fn, raising=False)

    def fail_grad_checkpoint(*args, **kwargs):
        raise AssertionError('grad_checkpoint should not run when chunk compile is active')

    monkeypatch.setattr(instnct_mod, 'grad_checkpoint', fail_grad_checkpoint)

    x = torch.randint(0, 256, (2, 8))
    out, state = model(x)

    assert out.shape == (2, 8, 256)
    assert state['ring'].shape == (2, 32, 16)
    assert state['ptr'].shape == (1, 2)
    assert state['hidden'].shape == (1, 2, 32)


def test_compile_chunk_disables_proxy_overlay(monkeypatch):
    """Proxy overlay helpers must be bypassed inside the compiled chunk path."""
    model = INSTNCT(
        M=32,
        hidden_dim=16,
        slot_dim=16,
        N=1,
        R=1,
        embed_mode=False,
        checkpoint_chunks=0,
        expert_weighting=False,
        write_mode='replace',
        replace_impl='proxy_overlay',
    )
    assert model._proxy_overlay_enabled is True
    model._compile_mode = 'chunk'
    model._compile_chunks = True
    model.compile_chunk_size = 4
    model._disable_proxy_overlay_for_compile = True

    monkeypatch.setattr(torch, 'compile', lambda fn, mode=None: fn, raising=False)

    def fail_overlay(*args, **kwargs):
        raise AssertionError('proxy overlay helper should not run in compiled chunk mode')

    monkeypatch.setattr(instnct_mod, 'func_proxy_overlay_read_tns', fail_overlay)
    monkeypatch.setattr(instnct_mod, 'func_proxy_overlay_write_tns', fail_overlay)
    monkeypatch.setattr(instnct_mod, 'func_proxy_overlay_flush_tns', fail_overlay)

    x = to_bits(torch.randint(0, 256, (2, 8)))
    out, _ = model(x)

    assert out.shape == (2, 8, 8)
    assert torch.isfinite(out).all()


def test_compile_chunk_keeps_core_scalar_diags(monkeypatch):
    """Compiled chunks should preserve scalar training diagnostics only."""
    model = INSTNCT(
        M=32,
        hidden_dim=32,
        slot_dim=16,
        N=1,
        R=1,
        embed_mode=False,
        checkpoint_chunks=0,
        expert_weighting=False,
    )
    model._compile_mode = 'chunk'
    model._compile_chunks = True
    model.compile_chunk_size = 4
    model._diag_enabled = True
    model._diag = {}

    monkeypatch.setattr(torch, 'compile', lambda fn, mode=None: fn, raising=False)

    x = to_bits(torch.randint(0, 256, (2, 8)))
    out, _ = model(x)
    diag = model._diag

    assert out.shape == (2, 8, 8)
    for key in (
        'ring_norm',
        'ring_slot_mean',
        'alpha_0_mean',
        'alpha_0_min',
        'alpha_0_max',
        'input_norm_0',
        'ring_signal_norm_0',
        'blended_norm_0',
        'hidden_norm_0',
        'hidden_final_norm_0',
        'ptr_pos_0',
        'write_strength_0_mean',
        'write_strength_0_min',
        'write_strength_0_max',
        'write_gate_logit_0_mean',
        'write_gate_logit_0_min',
        'write_gate_logit_0_max',
    ):
        assert key in diag, f'missing compile-safe diag key: {key}'
        assert isinstance(diag[key], float)

    assert 'topk_mean_abs_circ_dist' not in diag
    assert 'write_topk_mean_abs_circ_dist' not in diag


FASTPATH_CFG = dict(
    M=32,
    hidden_dim=32,
    slot_dim=16,
    N=1,
    R=1,
    checkpoint_chunks=0,
    expert_weighting=False,
    pointer_mode='sequential',
    pointer_interp_mode='off',
    pointer_seam_mode='mod',
    read_kernel_mode='vshape',
    write_address_mode='pointer',
    write_mode='replace',
    replace_impl='dense',
)


def _clone_state(state):
    if state is None:
        return None
    cloned = {}
    for key, value in state.items():
        cloned[key] = value.clone() if torch.is_tensor(value) else value
    return cloned


def _assert_state_close(ref_state, fast_state, *, atol=1e-6, rtol=1e-5):
    assert ref_state.keys() == fast_state.keys()
    for key in ref_state:
        ref_val = ref_state[key]
        fast_val = fast_state[key]
        if torch.is_tensor(ref_val):
            assert torch.allclose(ref_val, fast_val, atol=atol, rtol=rtol), key
        else:
            assert ref_val == fast_val, key


def _make_fastpath_pair(*, embed_mode=True, **overrides):
    cfg = dict(FASTPATH_CFG)
    cfg.update(overrides)
    torch.manual_seed(1234)
    ref = INSTNCT(**cfg, embed_mode=embed_mode)
    torch.manual_seed(1234)
    fast = INSTNCT(**cfg, embed_mode=embed_mode)
    fast.load_state_dict(ref.state_dict())
    ref._fastpath_mode = 'off'
    fast._fastpath_mode = 'force'
    return ref, fast


def test_fastpath_auto_uses_specialized_chunk(monkeypatch):
    """Supported nightly configs should dispatch to the single-expert fast path."""
    model = INSTNCT(**FASTPATH_CFG, embed_mode=True)
    model._process_chunk_generic = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError('generic chunk path should not run for supported auto fast path')
    )

    x = torch.randint(0, 256, (2, 8))
    out, state = model(x)

    assert out.shape == (2, 8, 256)
    assert state['ring'].shape == (2, 32, 16)


@pytest.mark.parametrize(
    'overrides',
    [
        {'N': 2},
        {'R': 2},
        {'bb_enabled': True},
        {'mtaps_enabled': True, 'mtaps_lags': (1, 2)},
        {'N': 2, 'io_split_mode': 'strict', 'io_writer_count': 1},
        {'replace_impl': 'proxy_overlay'},
        {'gated_write': True},
        {'read_kernel_mode': 'topk'},
    ],
)
def test_fastpath_auto_falls_back_for_unsupported_shapes(monkeypatch, overrides):
    """Unsupported configs must stay on the generic path under auto mode."""
    cfg = dict(FASTPATH_CFG)
    cfg.update(overrides)
    model = INSTNCT(**cfg, embed_mode=True)
    model._process_chunk_fast_n1_seqreplace = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError('fast path should not run for unsupported config')
    )

    x = torch.randint(0, 256, (2, 8))
    out, state = model(x)

    assert model._fastpath_reason() is not None
    assert out.shape[0] == 2
    assert state['ptr'].shape[0] == cfg['N']


def test_fastpath_force_rejects_unsupported_config():
    """Force mode must fail loudly when the narrow nightly predicate is false."""
    model = INSTNCT(**FASTPATH_CFG, embed_mode=True, bb_enabled=True)
    model._fastpath_mode = 'force'

    with pytest.raises(RuntimeError, match='bb_enabled=true'):
        model(torch.randint(0, 256, (2, 8)))


def test_fastpath_parity_eager_backward_embed():
    """Force fast path must match generic outputs, states, and grads in training mode."""
    ref, fast = _make_fastpath_pair(embed_mode=True)
    x = torch.randint(0, 256, (2, 8))

    ref_out, ref_state = ref(x)
    fast_out, fast_state = fast(x)
    assert torch.allclose(ref_out, fast_out, atol=1e-6, rtol=1e-5)
    _assert_state_close(ref_state, fast_state)

    ref_loss = ref_out.square().mean()
    fast_loss = fast_out.square().mean()
    ref_loss.backward()
    fast_loss.backward()

    for (ref_name, ref_param), (fast_name, fast_param) in zip(ref.named_parameters(), fast.named_parameters()):
        assert ref_name == fast_name
        if ref_param.grad is None or fast_param.grad is None:
            assert ref_param.grad is None and fast_param.grad is None, ref_name
            continue
        assert torch.allclose(ref_param.grad, fast_param.grad, atol=1e-6, rtol=1e-5), ref_name


def test_fastpath_parity_no_grad_binary():
    """Binary-mode inference should remain numerically identical under the fast path."""
    ref, fast = _make_fastpath_pair(embed_mode=False)
    x = to_bits(torch.randint(0, 256, (2, 8)))

    with torch.no_grad():
        ref_out, ref_state = ref(x)
        fast_out, fast_state = fast(x)

    assert torch.allclose(ref_out, fast_out, atol=1e-6, rtol=1e-5)
    _assert_state_close(ref_state, fast_state)


def test_fastpath_parity_chunk_compile_identity(monkeypatch):
    """Chunk-compiled dispatch should preserve the same results as the generic path."""
    monkeypatch.setattr(torch, 'compile', lambda fn, mode=None: fn, raising=False)
    ref, fast = _make_fastpath_pair(embed_mode=True)
    for model in (ref, fast):
        model._compile_mode = 'chunk'
        model._compile_chunks = True
        model.compile_chunk_size = 4

    x = torch.randint(0, 256, (2, 8))
    ref_out, ref_state = ref(x)
    fast_out, fast_state = fast(x)

    assert torch.allclose(ref_out, fast_out, atol=1e-6, rtol=1e-5)
    _assert_state_close(ref_state, fast_state)


def test_fastpath_parity_sequential_carry():
    """Carry-over state across consecutive batches must stay identical."""
    ref, fast = _make_fastpath_pair(embed_mode=True)
    x0 = torch.randint(0, 256, (2, 8))
    x1 = torch.randint(0, 256, (2, 8))

    ref_out0, ref_state0 = ref(x0)
    fast_out0, fast_state0 = fast(x0)
    assert torch.allclose(ref_out0, fast_out0, atol=1e-6, rtol=1e-5)
    _assert_state_close(ref_state0, fast_state0)

    ref_out1, ref_state1 = ref(x1, state=_clone_state(ref_state0))
    fast_out1, fast_state1 = fast(x1, state=_clone_state(fast_state0))
    assert torch.allclose(ref_out1, fast_out1, atol=1e-6, rtol=1e-5)
    _assert_state_close(ref_state1, fast_state1)


def test_min_write_strength_floor_prevents_noop_ring_write():
    """A small write floor should keep replace-write updates from collapsing to zero."""
    torch.manual_seed(123)
    base = INSTNCT(**FASTPATH_CFG, embed_mode=True, min_write_strength=0.0)
    torch.manual_seed(123)
    floored = INSTNCT(**FASTPATH_CFG, embed_mode=True, min_write_strength=0.002)
    floored.load_state_dict(base.state_dict())
    for model in (base, floored):
        model._fastpath_mode = 'force'
        with torch.no_grad():
            model.write_gate[0].weight.zero_()
            model.write_gate[0].bias.fill_(-25.0)

    x = torch.randint(0, 256, (2, 1))
    state = {
        'ring': torch.randn(2, 32, 16),
        'ptr': torch.zeros(1, 2),
        'hidden': torch.randn(1, 2, 32),
    }

    _, base_state = base(x, state=_clone_state(state))
    _, floored_state = floored(x, state=_clone_state(state))
    base_delta = (base_state['ring'] - state['ring']).norm().item()
    floored_delta = (floored_state['ring'] - state['ring']).norm().item()

    assert base_delta < 1e-4
    assert floored_delta > 1e-3
    assert floored_delta > base_delta * 1000.0


def test_mtaps_spaced_heads_enforce_minimum_gap():
    """Spaced MTAPS heads must keep the enforced >=8-slot separation."""
    model = INSTNCT(
        M=32,
        hidden_dim=32,
        slot_dim=16,
        N=1,
        R=1,
        embed_mode=True,
        checkpoint_chunks=0,
        expert_weighting=False,
        mtaps_enabled=True,
        mtaps_mixer_mode='hybrid_heads_spaced_scalar_gate',
    )
    model._diag_enabled = True
    with torch.no_grad():
        model.read_tap_aux_offset[0].weight.zero_()
        model.read_tap_aux_offset[0].bias.zero_()

    out, _ = model(torch.randint(0, 256, (4, 8)))

    assert out.shape == (4, 8, 256)
    assert model._diag['head_pair_near_frac_0'] == pytest.approx(0.0)
    assert model._diag['head_pair_dist_mean_0'] >= 7.9

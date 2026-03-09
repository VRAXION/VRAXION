"""Unit tests for training utilities — loss functions, accuracy, checkpoint I/O.

Tensor shape conventions (from train.py docstrings):
  Binary mode:  pred/target = (B, T, 8) float,  mask = (B, T, 1) float
  Embed mode:   pred        = (B, T, 256) float, target = (B, T) long,
                                                  mask = (B, T) float
"""

import importlib.util

import random
from pathlib import Path

import numpy as np
import torch
import pytest
from train import (
    ByteDataset,
    _capture_rng_state,
    _configure_compile_policy,
    _restore_rng_state,
    _select_sequential_state,
    func_maskloss_mse,
    func_maskloss_ce,
    func_accuracy_bin,
    func_accuracy_emb,
    func_loadckpt_dct,
    CKPT_VERSION,
)
from instnct import INSTNCT
from model_factory import build_model_from_spec
from model_factory import build_model_spec, load_model_config

V4_ROOT = Path(__file__).resolve().parents[1]


def _load_training_module(rel_path: str, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, V4_ROOT / rel_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _make_instnct_model_record(*, embed_mode: bool, **overrides):
    model_cfg = {
        'M': 32,
        'hidden_dim': 16,
        'slot_dim': 8,
        'N': 1,
        'R': 1,
        'kernel_mode': 'vshape',
        'read_kernel_mode': 'vshape',
        'checkpoint_chunks': 0,
        'expert_weighting': False,
        'embed_encoding': 'learned',
        'output_encoding': 'learned',
        'pointer_mode': 'sequential',
        'pointer_interp_mode': 'off',
        'pointer_seam_mode': 'mod',
        'write_mode': 'replace',
        'replace_impl': 'dense',
        'min_write_strength': 0.0,
        'bb_enabled': False,
        'bb_gate_bias': 0.0,
        'bb_scale': 0.1,
        'bb_tau': 4.0,
        'bb_gate_mode': 'learned',
        'topk_K': 8,
    }
    model_cfg.update(overrides)
    return build_model_spec(
        model_type='instnct',
        embed_mode=embed_mode,
        model_config=model_cfg,
        training_config={},
    )


def _write_v2_checkpoint(tmp_path: Path, *, embed_mode: bool, **model_overrides) -> Path:
    model_record = _make_instnct_model_record(embed_mode=embed_mode, **model_overrides)
    model = build_model_from_spec(model_record, device='cpu')
    ckpt_path = tmp_path / ('embed.pt' if embed_mode else 'binary.pt')
    torch.save({
        'ckpt_version': CKPT_VERSION,
        'run_id': 'unit-test-run',
        'step': 123,
        'best_loss': 0.42,
        'timestamp_utc': '2026-03-01T00:00:00Z',
        'model': {
            **model_record,
            'state_dict': model.state_dict(),
        },
        'optimizer': {
            'class_name': 'Adam',
            'state_dict': {},
        },
        'train_config_resolved': {
            'batch_size': 4,
            'seq_len': 16,
            'embed_mode': embed_mode,
        },
        'model_config_resolved': model_record['build_spec'],
        'data_state': {
            'data_dir': 'training_data',
            'seq_len': 16,
            'batch_size': 4,
            'embed_mode': embed_mode,
            'sequential': False,
            'file_manifest': [],
        },
        'sequence_state': {},
        'rng_state': {
            'python_random_state': random.getstate(),
            'numpy_random_state': np.random.get_state(),
            'torch_cpu_rng_state': torch.get_rng_state(),
            'cudnn_benchmark': False,
            'cudnn_deterministic': False,
            'torch_deterministic_algorithms': False,
            'dataset_rng_state': {},
            'eval_seed': 1337,
        },
        'env': {
            'python': '3.11.0',
            'torch': '2.5.1',
            'cuda': 'cpu',
            'hostname': 'test',
            'platform': 'test',
            'git_commit': 'deadbeef',
        },
    }, ckpt_path)
    return ckpt_path


def _make_compile_policy_model(**kwargs):
    """Small INSTNCT config used by compile policy unit tests."""
    return INSTNCT(
        M=32,
        embed_dim=16,
        N=1,
        R=1,
        embed_mode=False,
        expert_weighting=False,
        **kwargs,
    )


# ══════════════════════════════════════════════════════════════════
#  Masked MSE loss — binary mode
# ══════════════════════════════════════════════════════════════════

def test_maskloss_mse_zero_mask_gives_zero():
    """All mask=0 → masked_loss must be exactly 0 (no supervised positions)."""
    B, T = 2, 8
    pred   = torch.randn(B, T, 8)
    target = torch.randn(B, T, 8)
    mask   = torch.zeros(B, T, 1)
    _, masked = func_maskloss_mse(pred, target, mask)
    assert masked.item() == pytest.approx(0.0)


def test_maskloss_mse_perfect_pred_gives_zero():
    """pred == target everywhere → masked_loss must be 0."""
    B, T = 2, 8
    pred = torch.randn(B, T, 8)
    mask = torch.ones(B, T, 1)
    _, masked = func_maskloss_mse(pred, pred, mask)
    assert masked.item() == pytest.approx(0.0, abs=1e-6)


def test_maskloss_mse_ignores_unsupervised_error():
    """Error at mask=0 position must NOT affect masked_loss."""
    B, T = 1, 4
    pred   = torch.zeros(B, T, 8)
    target = torch.zeros(B, T, 8)
    target[0, 0] = 1.0          # big error at position 0
    mask = torch.zeros(B, T, 1)
    mask[0, 1] = 1.0            # only supervise position 1 (pred==target there)
    _, masked = func_maskloss_mse(pred, target, mask)
    assert masked.item() == pytest.approx(0.0, abs=1e-6)


def test_maskloss_mse_positive_on_error():
    """When pred ≠ target on supervised positions, masked_loss must be > 0."""
    B, T = 2, 8
    pred   = torch.zeros(B, T, 8)
    target = torch.ones(B, T, 8)   # maximally wrong
    mask   = torch.ones(B, T, 1)
    _, masked = func_maskloss_mse(pred, target, mask)
    assert masked.item() > 0.0


# ══════════════════════════════════════════════════════════════════
#  Masked CE loss — embed mode
# ══════════════════════════════════════════════════════════════════

def test_maskloss_ce_zero_mask_gives_zero():
    """All mask=0 → masked_loss must be 0."""
    B, T = 2, 8
    pred   = torch.randn(B, T, 256)
    target = torch.randint(0, 256, (B, T))
    mask   = torch.zeros(B, T)
    _, masked = func_maskloss_ce(pred, target, mask)
    assert masked.item() == pytest.approx(0.0)


def test_maskloss_ce_positive_on_error():
    """Uniform logits on wrong class → masked_loss must be > 0."""
    B, T = 2, 8
    pred   = torch.zeros(B, T, 256)        # uniform logits → random guess
    target = torch.randint(0, 256, (B, T))
    mask   = torch.ones(B, T)
    _, masked = func_maskloss_ce(pred, target, mask)
    assert masked.item() > 0.0


# ══════════════════════════════════════════════════════════════════
#  Bit-level accuracy — binary mode
# ══════════════════════════════════════════════════════════════════

def test_accuracy_bin_perfect():
    """pred == target → masked_acc = 1.0."""
    B, T = 2, 8
    target = torch.zeros(B, T, 8)
    pred   = torch.zeros(B, T, 8)   # exact match
    mask   = torch.ones(B, T, 1)
    _, masked_acc = func_accuracy_bin(pred, target, mask)
    assert masked_acc == pytest.approx(1.0)


def test_accuracy_bin_zero_mask_gives_zero():
    """Empty mask → masked_acc = 0.0 (no supervised positions to be correct on)."""
    B, T = 2, 8
    pred   = torch.randn(B, T, 8)
    target = torch.randn(B, T, 8)
    mask   = torch.zeros(B, T, 1)
    _, masked_acc = func_accuracy_bin(pred, target, mask)
    assert masked_acc == pytest.approx(0.0)


def test_accuracy_bin_completely_wrong():
    """pred=1 when target=0 everywhere → masked_acc = 0.0."""
    B, T = 2, 8
    pred   = torch.ones(B, T, 8)    # predict all 1s
    target = torch.zeros(B, T, 8)   # truth is all 0s
    mask   = torch.ones(B, T, 1)
    _, masked_acc = func_accuracy_bin(pred, target, mask)
    assert masked_acc == pytest.approx(0.0)


# ══════════════════════════════════════════════════════════════════
#  Byte-level accuracy — embed mode
# ══════════════════════════════════════════════════════════════════

def test_accuracy_emb_perfect():
    """High logit at correct class → masked_acc = 1.0."""
    B, T = 2, 8
    target = torch.randint(0, 256, (B, T))
    pred   = torch.full((B, T, 256), -10.0)
    for b in range(B):
        for t in range(T):
            pred[b, t, target[b, t]] = 10.0   # spike at correct class
    mask = torch.ones(B, T)
    _, masked_acc = func_accuracy_emb(pred, target, mask)
    assert masked_acc == pytest.approx(1.0)   # already a float scalar (no .item() needed)


# ══════════════════════════════════════════════════════════════════
#  Checkpoint round-trip
# ══════════════════════════════════════════════════════════════════

def test_checkpoint_roundtrip(tmp_path):
    """Save a v2 checkpoint and reload it — weights and metadata must survive."""
    model = INSTNCT(M=32, embed_dim=16, N=2, R=1, embed_mode=False)
    build_spec = dict(
        M=32,
        embed_dim=16,
        hidden_dim=16,
        slot_dim=16,
        N=2,
        R=1,
        B=8,
        embed_mode=False,
        kernel_mode='vshape',
        checkpoint_chunks=0,
        expert_weighting=False,
        embed_encoding='learned',
        output_encoding='learned',
        pointer_mode='sequential',
        bb_enabled=False,
        bb_gate_bias=0.0,
        bb_scale=0.1,
        bb_tau=4.0,
        bb_gate_mode='learned',
        topk_K=8,
    )

    ckpt_path = str(tmp_path / 'test.pt')
    torch.save({
        'ckpt_version': CKPT_VERSION,
        'run_id': 'unit-test-run',
        'step': 100,
        'best_loss': 0.42,
        'timestamp_utc': '2026-03-01T00:00:00Z',
        'model': {
            'type': 'instnct',
            'module': 'instnct',
            'class_name': 'INSTNCT',
            'build_spec': build_spec,
            'state_dict': model.state_dict(),
        },
        'optimizer': {
            'class_name': 'Adam',
            'state_dict': {},
        },
        'train_config_resolved': {
            'batch_size': 4,
            'seq_len': 16,
            'embed_mode': False,
        },
        'model_config_resolved': {
            'M': 32,
            'N': 2,
            'R': 1,
        },
        'data_state': {
            'data_dir': 'training_data',
            'seq_len': 16,
            'batch_size': 4,
            'embed_mode': False,
            'sequential': False,
            'file_manifest': [],
        },
        'sequence_state': {},
        'rng_state': {
            'python_random_state': random.getstate(),
            'numpy_random_state': np.random.get_state(),
            'torch_cpu_rng_state': torch.get_rng_state(),
            'cudnn_benchmark': False,
            'cudnn_deterministic': False,
            'torch_deterministic_algorithms': False,
            'dataset_rng_state': {},
            'eval_seed': 1337,
        },
        'env': {
            'python': '3.11.0',
            'torch': '2.5.1',
            'cuda': '12.1',
            'hostname': 'test',
            'platform': 'test',
            'git_commit': 'deadbeef',
        },
    }, ckpt_path)

    ckpt = func_loadckpt_dct(ckpt_path, 'cpu')

    model2 = INSTNCT(M=32, embed_dim=16, N=2, R=1, embed_mode=False)
    model2.load_state_dict(ckpt['model']['state_dict'])

    # All parameters must be identical after reload
    for (n, p1), (_, p2) in zip(model.named_parameters(),
                                 model2.named_parameters()):
        assert torch.allclose(p1, p2), f"Parameter '{n}' differs after roundtrip"

    assert ckpt['step'] == 100
    assert ckpt['best_loss'] == pytest.approx(0.42)


def test_checkpoint_v1_rejected(tmp_path):
    """Legacy checkpoints must fail fast with explicit version mismatch."""
    ckpt_path = str(tmp_path / 'legacy.pt')
    torch.save({'ckpt_version': 1, 'step': 1}, ckpt_path)
    with pytest.raises(RuntimeError, match='Checkpoint version mismatch'):
        func_loadckpt_dct(ckpt_path, 'cpu')


def test_model_factory_roundtrip_both_models():
    """Factory must instantiate both supported model types from build specs."""
    inst_spec = {
        'type': 'instnct',
        'module': 'instnct',
        'class_name': 'INSTNCT',
        'build_spec': {
            'M': 32,
            'embed_dim': None,
            'hidden_dim': 16,
            'slot_dim': 8,
            'N': 2,
            'R': 1,
            'B': 8,
            'embed_mode': True,
            'kernel_mode': 'vshape',
            'checkpoint_chunks': 0,
            'expert_weighting': False,
            'embed_encoding': 'learned',
            'output_encoding': 'learned',
            'pointer_mode': 'sequential',
            'min_write_strength': 0.125,
            'bb_enabled': False,
            'bb_gate_bias': 0.0,
            'bb_scale': 0.1,
            'bb_tau': 4.0,
            'bb_gate_mode': 'learned',
            'topk_K': 8,
            },
    }
    tr_spec = {
        'type': 'transformer',
        'module': 'tiny_transformer',
        'class_name': 'TinyTransformer',
        'build_spec': {
            'embed_mode': True,
            'd_model': 64,
            'n_layers': 2,
            'n_heads': 4,
            'd_ff': 128,
            'max_seq': 64,
            'dropout': 0.0,
        },
    }
    inst = build_model_from_spec(inst_spec, device='cpu')
    tr = build_model_from_spec(tr_spec, device='cpu')
    assert inst.__class__.__name__ == 'INSTNCT'
    assert tr.__class__.__name__ == 'TinyTransformer'
    assert inst.min_write_strength == pytest.approx(0.125)


def test_rng_restore_roundtrip_including_dataset(tmp_path):
    """RNG restore must reproduce python/numpy/torch and dataset RNG streams."""
    data_path = Path(tmp_path) / 'mini.traindat'
    mask_path = Path(tmp_path) / 'mini.mask'
    payload = bytes(range(64))
    data_path.write_bytes(payload)
    mask_path.write_bytes(bytes([1] * 64))

    ds = ByteDataset([(data_path, mask_path, 64)], seq_len=8, embed_mode=True, seed=444)

    random.seed(111)
    np.random.seed(222)
    torch.manual_seed(333)

    saved = _capture_rng_state(ds, eval_seed=1337)
    first = (
        random.random(),
        float(np.random.rand()),
        float(torch.rand(1).item()),
        int(ds.rng.integers(0, 1_000_000)),
    )

    # Disturb all RNG streams.
    _ = random.random()
    _ = np.random.rand()
    _ = torch.rand(8)
    _ = ds.rng.integers(0, 1_000_000, size=8)

    _restore_rng_state(saved, ds)
    second = (
        random.random(),
        float(np.random.rand()),
        float(torch.rand(1).item()),
        int(ds.rng.integers(0, 1_000_000)),
    )

    assert second[0] == pytest.approx(first[0], abs=1e-12)
    assert second[1] == pytest.approx(first[1], abs=1e-12)
    assert second[2] == pytest.approx(first[2], abs=1e-12)
    assert second[3] == first[3]


# ══════════════════════════════════════════════════════════════════
#  Compile policy
# ══════════════════════════════════════════════════════════════════

def test_compile_policy_seq48_uses_full_model_compile(monkeypatch):
    """Auto compile should use full-model compile at or below the nightly threshold."""
    calls = []

    def fake_compile(model, mode=None):
        calls.append((model, mode))
        return model

    monkeypatch.setattr(torch, 'compile', fake_compile, raising=False)
    model = _make_compile_policy_model()

    configured = _configure_compile_policy(
        model,
        {'compile': True, 'compile_chunk_size': 32, 'seq_len': 48},
        device='cuda',
    )

    assert configured is model
    assert calls == [(model, 'reduce-overhead')]
    assert model._compile_mode == 'full'
    assert model._compile_chunks is False
    assert model._disable_proxy_overlay_for_compile is False


def test_compile_policy_seq256_enables_chunk_compile(monkeypatch):
    """Auto compile should defer to chunk compile above the full-model threshold."""
    calls = []

    def fake_compile(model, mode=None):
        calls.append((model, mode))
        return model

    monkeypatch.setattr(torch, 'compile', fake_compile, raising=False)
    model = _make_compile_policy_model()

    configured = _configure_compile_policy(
        model,
        {'compile': True, 'compile_chunk_size': 32, 'seq_len': 256},
        device='cuda',
    )

    assert configured is model
    assert calls == []
    assert model._compile_mode == 'chunk'
    assert model._compile_chunks is True
    assert model.compile_chunk_size == 32


def test_compile_policy_bb_enabled_falls_back_to_eager(monkeypatch, capsys):
    """BB-enabled configs are intentionally kept eager in this compile pass."""
    calls = []

    def fake_compile(model, mode=None):
        calls.append((model, mode))
        return model

    monkeypatch.setattr(torch, 'compile', fake_compile, raising=False)
    model = _make_compile_policy_model(bb_enabled=True)

    configured = _configure_compile_policy(
        model,
        {'compile': True, 'compile_chunk_size': 32, 'seq_len': 256},
        device='cuda',
    )
    out = capsys.readouterr().out

    assert configured is model
    assert calls == []
    assert model._compile_mode == 'eager'
    assert model._compile_chunks is False
    assert 'bb_enabled=true' in out


def test_select_sequential_state_rolls_back_on_nonfinite_loss():
    """A bad forward must not advance the persistent carry state."""
    prev_state = {'ring': torch.ones(2, 3)}
    next_state = {'ring': torch.full((2, 3), 7.0)}

    chosen = _select_sequential_state(
        prev_state,
        next_state,
        torch.tensor(float('nan')),
        sequential=True,
    )

    assert chosen is prev_state


def test_select_sequential_state_commits_on_finite_loss():
    """Healthy steps should keep the newly computed carry state."""
    prev_state = {'ring': torch.ones(2, 3)}
    next_state = {'ring': torch.full((2, 3), 7.0)}

    chosen = _select_sequential_state(
        prev_state,
        next_state,
        torch.tensor(1.2345),
        sequential=True,
    )

    assert chosen is next_state


class _TableDataset:
    def __init__(self, x_all, y_all, mask_all):
        self.x_all = x_all
        self.y_all = y_all
        self.mask_all = mask_all
        self.cursor = 0

    def sample_batch(self, bs, device):
        sl = slice(self.cursor, self.cursor + bs)
        self.cursor += bs
        return (
            self.x_all[sl].to(device),
            self.y_all[sl].to(device),
            self.mask_all[sl].to(device),
        )

    def sample_batch_sequential(self, bs, device):
        return self.sample_batch(bs, device)


class _TableEmbedModel(torch.nn.Module):
    def __init__(self, logits_table):
        super().__init__()
        self.register_buffer('logits_table', logits_table)

    def forward(self, x, state=None):
        ids = x.squeeze(-1).long()
        return self.logits_table[ids], state


def _make_embed_eval_fixture(mask_vals):
    n = len(mask_vals)
    vocab = 256
    x_all = torch.arange(n, dtype=torch.long).unsqueeze(-1)
    y_all = torch.tensor([(idx * 7) % vocab for idx in range(n)], dtype=torch.long).unsqueeze(-1)
    mask_all = torch.tensor(mask_vals, dtype=torch.float32).unsqueeze(-1)
    logits_table = torch.full((n, 1, vocab), -4.0)
    for idx in range(n):
        target = int(y_all[idx, 0].item())
        logits_table[idx, 0, target] = 6.0 if idx % 2 == 0 else -1.0
        logits_table[idx, 0, (target + 1) % vocab] = 1.5 if idx % 2 else -2.0
    return x_all, y_all, mask_all, logits_table


def test_eval_metrics_invariant_to_smaller_final_batch():
    eval_mod = _load_training_module('training/eval.py', 'eval_module_batch_invariance')
    x_all, y_all, mask_all, logits_table = _make_embed_eval_fixture([1, 1, 1, 1, 1, 1])
    config_small = {'embed_mode': True, 'batch_size': 3, 'seq_len': 1, 'eval_seed': 1337}
    config_tail = {'embed_mode': True, 'batch_size': 4, 'seq_len': 1, 'eval_seed': 1337}

    metrics_small = eval_mod.evaluate(
        _TableEmbedModel(logits_table),
        _TableDataset(x_all, y_all, mask_all),
        config_small,
        'cpu',
        n_samples=len(x_all),
    )
    metrics_tail = eval_mod.evaluate(
        _TableEmbedModel(logits_table),
        _TableDataset(x_all, y_all, mask_all),
        config_tail,
        'cpu',
        n_samples=len(x_all),
    )

    for key in ('raw_loss', 'masked_loss', 'accuracy', 'masked_acc', 'mask_frac'):
        assert metrics_small[key] == pytest.approx(metrics_tail[key], abs=1e-7)


def test_eval_masked_metrics_invariant_to_mask_skew():
    eval_mod = _load_training_module('training/eval.py', 'eval_module_mask_invariance')
    x_all, y_all, mask_all, logits_table = _make_embed_eval_fixture([1, 1, 1, 0, 0, 0, 1])
    config_a = {'embed_mode': True, 'batch_size': 2, 'seq_len': 1, 'eval_seed': 1337}
    config_b = {'embed_mode': True, 'batch_size': 5, 'seq_len': 1, 'eval_seed': 1337}

    metrics_a = eval_mod.evaluate(
        _TableEmbedModel(logits_table),
        _TableDataset(x_all, y_all, mask_all),
        config_a,
        'cpu',
        n_samples=len(x_all),
    )
    metrics_b = eval_mod.evaluate(
        _TableEmbedModel(logits_table),
        _TableDataset(x_all, y_all, mask_all),
        config_b,
        'cpu',
        n_samples=len(x_all),
    )

    for key in ('raw_loss', 'masked_loss', 'accuracy', 'masked_acc', 'mask_frac'):
        assert metrics_a[key] == pytest.approx(metrics_b[key], abs=1e-7)


def test_inference_main_loads_v2_embed_checkpoint(tmp_path, monkeypatch, capsys):
    inference_mod = _load_training_module('training/inference.py', 'inference_module_embed')
    ckpt_path = _write_v2_checkpoint(tmp_path, embed_mode=True)
    monkeypatch.setattr(
        'sys.argv',
        [
            'inference.py',
            '--checkpoint', str(ckpt_path),
            '--prompt', 'Hi',
            '--length', '2',
            '--samples', '1',
            '--device', 'cpu',
        ],
    )

    inference_mod.main()
    out = capsys.readouterr().out

    assert '[BOOT] Step        123' in out
    assert '== Sample 1 ==' in out


def test_inference_main_rejects_binary_checkpoint(tmp_path, monkeypatch):
    inference_mod = _load_training_module('training/inference.py', 'inference_module_binary')
    ckpt_path = _write_v2_checkpoint(tmp_path, embed_mode=False)
    monkeypatch.setattr(
        'sys.argv',
        [
            'inference.py',
            '--checkpoint', str(ckpt_path),
            '--prompt', 'Hi',
            '--length', '2',
            '--samples', '1',
            '--device', 'cpu',
        ],
    )

    with pytest.raises(RuntimeError, match='embed-mode checkpoints'):
        inference_mod.main()


def test_resume_rebuild_uses_current_yaml_min_write_strength():
    model_cfg = load_model_config(V4_ROOT)
    model_record = build_model_spec(
        model_type='instnct',
        embed_mode=True,
        model_config=model_cfg,
        training_config={},
    )
    model = build_model_from_spec(model_record, device='cpu')

    assert model.min_write_strength == pytest.approx(float(model_cfg.get('min_write_strength', 0.0)))

"""Unit tests for synthetic data generators — generate.py

Tests cover:
  - output size matches requested size
  - mask ratios match task design (echo=87.5%, denoise=50%)
  - eval seed produces different data than training seed
  - all generators run without crashing
"""

import random
import pytest
from generate import (
    func_echorepat_byt,
    func_denoisbyt_byt,
    func_delayecho_byt,
    func_invrtbits_byt,
    func_byterotat_byt,
    func_countbyte_byt,
    func_addandsub_byt,
    func_fiboseqnc_byt,
    BLOCK, ECHO_REPEAT,
    _EVAL_SALT, TASKS,
)

SIZE = 8192   # large enough to average out edge effects at trim boundary


# ══════════════════════════════════════════════════════════════════
#  Output size
# ══════════════════════════════════════════════════════════════════

def test_echo_output_size():
    data, mask = func_echorepat_byt(SIZE)
    assert len(data) == SIZE
    assert len(mask) == SIZE


def test_denoise_output_size():
    data, mask = func_denoisbyt_byt(SIZE)
    assert len(data) == SIZE
    assert len(mask) == SIZE


def test_delay_echo_output_size():
    data, mask = func_delayecho_byt(SIZE)
    assert len(data) == SIZE
    assert len(mask) == SIZE


# ══════════════════════════════════════════════════════════════════
#  Mask ratios (task design correctness)
# ══════════════════════════════════════════════════════════════════

def test_echo_mask_ratio():
    """Echo: first block of each cycle is the seed (mask=0), the rest are
    supervised copies (mask=1).  Ratio = (ECHO_REPEAT-1) / ECHO_REPEAT."""
    random.seed(0)
    _, mask = func_echorepat_byt(SIZE)
    ratio    = mask.count(1) / len(mask)
    expected = (ECHO_REPEAT - 1) / ECHO_REPEAT   # 7/8 = 0.875 with default cfg
    assert abs(ratio - expected) < 0.02, \
        f"Echo mask ratio {ratio:.4f} far from expected {expected:.4f}"


def test_denoise_mask_ratio():
    """Denoise: noisy block (mask=0) followed by clean block (mask=1) → 50%."""
    random.seed(0)
    _, mask = func_denoisbyt_byt(SIZE)
    ratio = mask.count(1) / len(mask)
    assert abs(ratio - 0.5) < 0.02, \
        f"Denoise mask ratio {ratio:.4f} far from expected 0.5"


def test_mask_values_are_binary():
    """All mask bytes must be exactly 0 or 1 — no other values."""
    random.seed(0)
    _, mask = func_echorepat_byt(SIZE)
    assert set(mask) <= {0, 1}, f"Unexpected mask values: {set(mask) - {0, 1}}"


# ══════════════════════════════════════════════════════════════════
#  Eval seed isolation
# ══════════════════════════════════════════════════════════════════

def test_eval_seed_differs_from_train():
    """Training data (seed=42) and eval data (seed=42 XOR salt) must differ.
    This guarantees the eval set is truly held-out."""
    train_seed = 42
    eval_seed  = train_seed ^ _EVAL_SALT

    random.seed(train_seed)
    train_data, _ = func_echorepat_byt(SIZE)

    random.seed(eval_seed)
    eval_data, _ = func_echorepat_byt(SIZE)

    assert train_data != eval_data, \
        "Train and eval data are identical — seed XOR salt is not working"


def test_eval_salt_is_nonzero():
    """The XOR salt must be non-zero, otherwise train_seed == eval_seed."""
    assert _EVAL_SALT != 0


# ══════════════════════════════════════════════════════════════════
#  Smoke test — all generators
# ══════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("name,fn", TASKS)
def test_generator_no_crash(name, fn):
    """Every registered generator must run without raising an exception."""
    random.seed(7)
    data, mask = fn(512)
    assert len(data) > 0, f"{name}: data is empty"
    assert len(mask) > 0, f"{name}: mask is empty"
    assert len(data) == len(mask), f"{name}: data/mask length mismatch"


@pytest.mark.parametrize("name,fn", TASKS)
def test_generator_byte_range(name, fn):
    """All data values must be valid bytes (0-255) and mask values must be 0 or 1."""
    random.seed(7)
    data, mask = fn(512)
    assert all(0 <= b <= 255 for b in data), f"{name}: data byte out of 0-255 range"
    assert set(mask) <= {0, 1}, f"{name}: mask has values other than 0/1: {set(mask) - {0, 1}}"

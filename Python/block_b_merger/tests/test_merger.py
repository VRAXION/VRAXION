"""Lossless round-trip tests for Block B Python SDK (L1 Byte-Pair Merger)."""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))
from Python.block_b_merger import L1Merger


# ---------------------------------------------------------------------------
# Core lossless test

def test_default_load_and_verify_lossless():
    m = L1Merger.load_default()
    matches, total = m.verify_lossless()
    assert total == 65536,   f"expected total=65536, got {total}"
    assert matches == 65536, f"expected 65536/65536 lossless, got {matches}/{total}"


# ---------------------------------------------------------------------------
# Basic shape / type tests

def test_load_returns_l1_merger():
    m = L1Merger.load_default()
    assert isinstance(m, L1Merger)


def test_forward_output_shape():
    m = L1Merger.load_default()
    x = np.zeros(32, dtype=np.float32)
    y = m.forward(x)
    assert y.shape == (32,), f"expected (32,), got {y.shape}"
    assert y.dtype == np.float32


def test_forward_accepts_float64():
    m = L1Merger.load_default()
    x = np.zeros(32, dtype=np.float64)
    y = m.forward(x)
    assert y.shape == (32,)


def test_forward_wrong_shape_rejected():
    m = L1Merger.load_default()
    try:
        m.forward(np.zeros(16))
        assert False, "should have raised for 16-dim input"
    except ValueError:
        pass


def test_batch_forward_shape():
    m = L1Merger.load_default()
    X = np.random.randn(10, 32).astype(np.float32)
    Y = m.forward_batch(X)
    assert Y.shape == (10, 32), f"expected (10, 32), got {Y.shape}"


def test_single_row_batch_matches_forward():
    m = L1Merger.load_default()
    x = np.random.randn(32).astype(np.float32)
    y_single = m.forward(x)
    y_batch  = m.forward_batch(x[np.newaxis, :])[0]
    np.testing.assert_array_equal(y_single, y_batch)


# ---------------------------------------------------------------------------
# Sign-match spot-checks with known-good pairs

def _load_nozero_lut() -> np.ndarray:
    """Load the nozero training LUT (float32, shape (256, 16))."""
    import json
    repo = Path(__file__).resolve().parent.parent.parent.parent
    lut_path = repo / "tools" / "byte_embedder_lut_int8_nozero.json"
    blob = json.loads(lut_path.read_text(encoding="utf-8"))
    scale = float(blob["scale"])
    return np.array(blob["lut"], dtype=np.float32) * scale


def test_sign_match_spot_check_first_pair():
    """Pair (byte 0, byte 0) — forward -> sign matches training LUT."""
    lut = _load_nozero_lut()
    m = L1Merger.load_default()
    x = np.concatenate([lut[0], lut[0]])
    y = m.forward(x)
    sign_match = np.sign(y) == np.sign(x)
    assert sign_match.all(), f"sign mismatch on pair (0,0): {np.where(~sign_match)}"


def test_sign_match_spot_check_last_pair():
    """Pair (byte 255, byte 255) — forward -> sign matches training LUT."""
    lut = _load_nozero_lut()
    m = L1Merger.load_default()
    x = np.concatenate([lut[255], lut[255]])
    y = m.forward(x)
    sign_match = np.sign(y) == np.sign(x)
    assert sign_match.all(), f"sign mismatch on pair (255,255): {np.where(~sign_match)}"


# ---------------------------------------------------------------------------
# Minimal standalone runner

if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    print(f"Running {len(tests)} tests...")
    passed = failed = 0
    for fn in tests:
        try:
            fn()
            print(f"  [OK]   {fn.__name__}")
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {fn.__name__}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)

"""Lossless round-trip tests for Block A Python SDK."""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from Python.block_a_byte_unit import ByteEncoder


def test_default_load_and_self_verify():
    enc = ByteEncoder.load_default()
    matches, total = enc.verify_lossless()
    assert matches == total == 256, f"self-verify expected 256/256, got {matches}/{total}"


def test_encode_single_byte_shape():
    enc = ByteEncoder.load_default()
    vec = enc.encode(0x41)
    assert vec.shape == (16,)
    assert vec.dtype == np.float32


def test_encode_accepts_bytes_input():
    enc = ByteEncoder.load_default()
    vec1 = enc.encode(65)
    vec2 = enc.encode(b"A")
    np.testing.assert_array_equal(vec1, vec2)


def test_encode_decode_round_trip_all_256_bytes():
    enc = ByteEncoder.load_default()
    for b in range(256):
        vec = enc.encode(b)
        back = enc.decode(vec)
        assert back == b, f"byte {b} round-tripped to {back}"


def test_vectorized_round_trip_all_256_bytes():
    enc = ByteEncoder.load_default()
    data = bytes(range(256))
    latents = enc.encode_bytes(data)
    assert latents.shape == (256, 16)
    back = enc.decode_bytes(latents)
    assert back == data


def test_vectorized_ascii_text():
    enc = ByteEncoder.load_default()
    text = b"The quick brown fox jumps over the lazy dog. 0123456789!@#$%^&*()"
    latents = enc.encode_bytes(text)
    back = enc.decode_bytes(latents)
    assert back == text


def test_vectorized_utf8_text():
    enc = ByteEncoder.load_default()
    text = "Péter szépen éneklő bárány 中文 العربية 🐈".encode("utf-8")
    latents = enc.encode_bytes(text)
    back = enc.decode_bytes(latents)
    assert back == text


def test_byte_out_of_range_rejected():
    enc = ByteEncoder.load_default()
    try:
        enc.encode(256)
        assert False, "should have raised for byte=256"
    except AssertionError:
        pass
    try:
        enc.encode(-1)
        assert False, "should have raised for byte=-1"
    except AssertionError:
        pass


def test_multibyte_bytes_input_rejected():
    enc = ByteEncoder.load_default()
    try:
        enc.encode(b"AB")
        assert False, "should have raised for 2-byte input"
    except AssertionError:
        pass


def test_encode_is_deterministic():
    """Block A encode must be a pure LUT lookup — same byte → same vector, every time.

    Guards against accidental RNG / mutation creeping into the deploy path.
    """
    enc = ByteEncoder.load_default()
    for b in (0, 65, 128, 200, 255):
        v1 = enc.encode(b)
        v2 = enc.encode(b)
        np.testing.assert_array_equal(v1, v2)


def test_empty_bytes_round_trip():
    """Empty text round-trips to empty text (no latent rows, no decode drift)."""
    enc = ByteEncoder.load_default()
    assert enc.decode_bytes(enc.encode_bytes(b"")) == b""


if __name__ == "__main__":
    # Minimal runner
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    print(f"Running {len(tests)} tests...")
    passed = 0
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f"  [OK] {fn.__name__}")
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {fn.__name__}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)

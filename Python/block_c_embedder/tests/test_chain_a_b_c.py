"""End-to-end ABC pipeline stress test.

Exercises:
  A (ByteEncoder): bytes -> 16-d latents (per byte), lossless round-trip.
  B (L1Merger):    pair of 16-d latents -> 32-d latent, identity (sign-match
                   lossless over all 65,536 byte pairs).
  C (L2Embedder):  byte-pair id -> 32-d semantic embedding, int4-quantized
                   hot rows + shared OOV for cold rows.

Each test either asserts the invariant or pytest-style reports a diff. The
test runs entirely on numpy; no ML framework.

Run with pytest:
    pytest -xvs Python/block_c_embedder/tests/test_chain_a_b_c.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "Python"))

from block_a_byte_unit import ByteEncoder
from block_b_merger import L1Merger
from block_c_embedder import L2Embedder


# ------------------------------------------------------------------
# Fixtures

@pytest.fixture(scope="module")
def a_encoder():
    return ByteEncoder.load_default()


@pytest.fixture(scope="module")
def b_merger():
    return L1Merger.load_default()


@pytest.fixture(scope="module")
def c_embedder():
    return L2Embedder.load_default()


# ------------------------------------------------------------------
# Block-level properties

def test_a_lossless(a_encoder):
    """Block A: every byte -> latent -> byte must be identity (lossless)."""
    passed, total = a_encoder.verify_lossless()
    assert passed == total == 256, f"Block A not lossless: {passed}/{total}"


def test_b_lossless_sign_match(b_merger):
    """Block B: identity AE, sign-match lossless over all 65,536 pairs."""
    passed, total = b_merger.verify_lossless()
    assert passed == total == 65_536, f"Block B not lossless: {passed}/{total}"


def test_c_header_and_scheme(c_embedder):
    m = c_embedder.meta
    assert m["vocab_size"] == 65_536
    assert m["E"] == 32
    assert m["n_hot"] > 1_000, f"suspiciously few hot rows: {m['n_hot']}"
    assert m["scheme"] == "cold_shared"
    assert m["packed_bytes"] < 80 * 1024, f"packed too big: {m['packed_bytes']}"


# ------------------------------------------------------------------
# ABC chain: encode a byte stream through A -> B -> C and verify

def test_chain_encode_hello_world(a_encoder, b_merger, c_embedder):
    """End-to-end pipeline: bytes -> A -> (pair ID from bytes) -> C.

    Note: Block A default champion and Block B champion were trained on
    different byte-unit LUTs (binary+C19+H=16 vs nozero), so the chained
    A->B sign-match is NOT guaranteed in the current deploy artifacts.
    Both blocks are lossless in isolation (tested separately). The
    canonical pipeline uses the byte-pair ID (hi<<8)|lo directly — B
    proves information equivalence but is not wired as an intermediary
    in the deploy path.
    """
    data = b"Hello world!"
    # A in isolation: bytes -> latents -> bytes is lossless.
    latents = a_encoder.encode_bytes(data)
    assert latents.shape == (len(data), 16)
    assert a_encoder.decode_bytes(latents) == data

    # C: byte-pair id -> embedding (no intermediate B pass at deploy time)
    n_pairs = len(data) // 2
    embs = c_embedder.encode_bytes(data)
    assert embs.shape == (n_pairs, c_embedder.E)
    assert embs.dtype == np.float32
    assert np.all(np.isfinite(embs)), "C produced non-finite embeddings"


def test_c_deterministic(c_embedder):
    """Same input -> same output, always."""
    data = b"The quick brown fox jumps."
    a = c_embedder.encode_bytes(data)
    b = c_embedder.encode_bytes(data)
    assert np.array_equal(a, b)


def test_c_oov_shared(c_embedder):
    """Two distinct cold byte-pairs must produce the same (shared OOV) vector."""
    cold_found = 0
    last_vec = None
    for pair_id in range(65_536):
        if c_embedder.is_hot(pair_id):
            continue
        v = c_embedder.embed_id(pair_id)
        if last_vec is None:
            last_vec = v
        else:
            assert np.array_equal(v, last_vec), (
                f"cold pair {pair_id} diverged from shared OOV"
            )
        cold_found += 1
        if cold_found > 100:
            break
    assert cold_found >= 100


def test_c_hot_diverse(c_embedder):
    """Hot rows should be mostly distinct (near 100% uniqueness)."""
    hot_ids = c_embedder.hot_ids()
    vecs = np.stack([c_embedder.embed_id(int(i)) for i in hot_ids[:2000]])
    rounded = np.round(vecs, 6)
    uniq = len(np.unique(rounded, axis=0))
    assert uniq >= int(len(vecs) * 0.995), (
        f"hot uniqueness too low: {uniq}/{len(vecs)}"
    )


# ------------------------------------------------------------------
# Semantic smoke: cluster preservation

def test_c_semantic_clusters(c_embedder):
    """Key trained clusters should still be closest neighbours after int4 quant.

    Examples:
      ' .' should cluster near '! ', '? ', '.\\n' (sentence terminators).
      ' t' should cluster near '\\nt', '(t', '-t' (word-start positions).
      'in' should cluster near 'In' (case-invariance for word start).
    """
    def pair_id(s: str) -> int:
        return (ord(s[0]) << 8) | ord(s[1])

    checks = [
        (". ", ["! ", "? ", ".\n", "!\n"]),       # sentence terminators
        (", ", ["; ", ": "]),                      # clause punct
        (" t", ["\nt", "\nT", "(t", "-t"]),       # word-start ' t' family
    ]
    for anchor, expected_any_of in checks:
        aid = pair_id(anchor)
        # Top-5 nearest among ALL rows, but we also check hot-restricted
        vecs = np.stack([c_embedder.embed_id(i) for i in range(65_536)])
        v = vecs[aid]
        d2 = np.sum((vecs - v) ** 2, axis=1)
        d2[aid] = np.inf
        # Restrict to hot to get meaningful neighbours
        hot_mask = np.zeros(65_536, dtype=bool)
        hot_mask[c_embedder.hot_ids()] = True
        d2[~hot_mask] = np.inf
        top5 = np.argsort(d2)[:5]
        top5_ids = top5.tolist()
        expected_ids = {pair_id(e) for e in expected_any_of if pair_id(e) < 65_536}
        hits = [i for i in top5_ids if i in expected_ids]
        assert len(hits) >= 1, (
            f"anchor {anchor!r}: expected any of "
            f"{expected_any_of} in top-5, got "
            f"{[bytes([tid>>8, tid&0xFF]) for tid in top5_ids]}"
        )


# ------------------------------------------------------------------
# Stress: run a large corpus chunk through A -> (implicit B) -> C

def test_chain_large_chunk(a_encoder, c_embedder):
    """Large-corpus stress: A round-trip lossless on 100 KB, C hot-rate > 90%."""
    corpus_path = REPO_ROOT / "output" / "data" / "fineweb_edu_100mb.txt"
    if not corpus_path.exists():
        pytest.skip(f"{corpus_path} not present")
    data = corpus_path.read_bytes()[:100_000]

    # A: byte lossless round-trip on real text
    latents = a_encoder.encode_bytes(data)
    recovered = a_encoder.decode_bytes(latents)
    assert recovered == data, "A round-trip failed on real text"

    # C: embedding stream, no NaN, correct shape
    n = len(data) // 2
    embs = c_embedder.encode_bytes(data)
    assert embs.shape == (n, 32)
    assert np.all(np.isfinite(embs))

    # On real English text most pairs should hit the hot set
    hot_ids = c_embedder.hot_ids()
    hot_set = set(hot_ids.tolist())
    arr = np.frombuffer(data[: 2 * n], dtype=np.uint8).reshape(n, 2)
    ids = (arr[:, 0].astype(np.int64) << 8) | arr[:, 1].astype(np.int64)
    hot_fraction = np.mean([int(i) in hot_set for i in ids[:1000]])
    assert hot_fraction > 0.90, (
        f"only {hot_fraction*100:.1f}% of pairs are hot "
        f"(expected > 90% on real English text)"
    )


def test_chain_edge_cases(a_encoder, c_embedder):
    """Edge cases: empty, single byte, odd length, binary data."""
    # empty
    assert a_encoder.decode_bytes(a_encoder.encode_bytes(b"")) == b""
    assert c_embedder.encode_bytes(b"").shape == (0, 32)

    # single byte — C drops the trailing odd byte
    embs = c_embedder.encode_bytes(b"X")
    assert embs.shape == (0, 32)

    # odd length
    data = b"HelloA"  # 6 bytes = 3 pairs
    embs = c_embedder.encode_bytes(data)
    assert embs.shape == (3, 32)
    data7 = b"HelloAB"  # 7 bytes = 3 pairs (trailing byte dropped)
    embs7 = c_embedder.encode_bytes(data7)
    assert embs7.shape == (3, 32)
    assert np.array_equal(embs, embs7)

    # binary data — A lossless on all 256 bytes
    bin_data = bytes(range(256))
    rec = a_encoder.decode_bytes(a_encoder.encode_bytes(bin_data))
    assert rec == bin_data
    # C: every pair ID valid
    embs = c_embedder.encode_bytes(bin_data)
    assert embs.shape == (128, 32)
    assert np.all(np.isfinite(embs))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-xvs"]))

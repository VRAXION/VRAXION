"""Phase 0: L2 geometry probe (GPT's protocol).

Before building any neural L2 merger, check whether the 648-dim L1 hidden
state geometry even fits into a small linear subspace.

Protocol:
  1. Take a natural text corpus (FineWeb sample).
  2. Slide 16-byte windows across it.
  3. For each window: 8 byte pairs -> L1 hidden -> concat 8*81 = 648-dim.
  4. Collect N such 648-dim vectors (~10k-100k samples).
  5. Fit PCA with D in {32, 48, 64, 96, 128}.
  6. For each D: project -> reconstruct -> decode through frozen L1 -> measure
     byte accuracy and held-out 16-byte exact rate.

This is a BASELINE — if linear PCA can't do it at D=64, a tied neural net
won't either (not without far more capacity).

No training. No weight updates. Just linear algebra + L1 forward/inverse.
"""
from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).parent))
from diag_byte_pair_merger_single_w_mirror import DEVICE
from diag_byte_single_w_huffman_pack import COMPONENTS, unpack_component

HUFFMAN_CHAMPION = Path("output/merger_single_w_huffman_pack/packed_model.bin")
LUT_PATH = Path(__file__).with_name("byte_embedder_lut_int8_nozero.json")
CORPUS_CANDIDATES = [
    Path("instnct-core/tests/fixtures/code_corpus.txt"),  # 3.8 MB — biggest fixture
    Path("instnct-core/tests/fixtures/alice_corpus.txt"),  # 100 KB fallback
    Path("S:/AI"),  # last resort — scan for any .txt
]
OUT_DIR = Path("output/merger_l2_phase0")
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_SAMPLES = 20_000      # 16-byte windows to collect
D_SWEEP = [32, 48, 64, 96, 128]


def load_l1_champion():
    """Rebuild the exact L1 merger from the committed Huffman-packed champion."""
    payload = HUFFMAN_CHAMPION.read_bytes()
    if payload[:4] != b"VGH1":
        raise ValueError(f"bad L1 champion magic in {HUFFMAN_CHAMPION}")
    offset = 4
    state = {}
    for name, n, raw_only in COMPONENTS:
        arr, offset = unpack_component(name, n, raw_only, payload, offset)
        state[name] = arr

    from diag_byte_single_w_huffman_pack import build_model_from_state

    return build_model_from_state(state)


def load_l0_lut():
    with open(LUT_PATH) as f:
        blob = json.load(f)
    scale = blob["scale"]
    lut = np.array(blob["lut"], dtype=np.float32) * scale  # (256, 16)
    return torch.tensor(lut, dtype=torch.float32, device=DEVICE)


def byte_pairs_to_l1_hidden(byte_pairs, l1_model):
    """byte_pairs: (N, 32) tensor.  Returns L1 hidden 81-dim per pair: (N, 81)."""
    with torch.no_grad():
        h = l1_model.c19(byte_pairs @ l1_model.W + l1_model.b1)
    return h  # (N, 81)


def load_corpus_bytes(max_bytes=5_000_000):
    """Load some natural text bytes from whatever corpus is available."""
    for candidate in CORPUS_CANDIDATES:
        if candidate.is_file():
            data = candidate.read_bytes()[:max_bytes]
            print(f"  [corpus] {candidate} -> {len(data)} bytes")
            return data
        if candidate.is_dir():
            # glob for text files
            for f in sorted(candidate.rglob("*.txt")):
                data = f.read_bytes()[:max_bytes]
                print(f"  [corpus] {f} -> {len(data)} bytes")
                return data
    # Fallback: synthetic random bytes (for smoke-test only)
    print(f"  [corpus] WARNING: no real corpus found, using random bytes (smoke-test only)")
    rng = np.random.default_rng(0)
    return bytes(rng.integers(0, 256, size=max_bytes, dtype=np.uint8).tobytes())


def collect_l1_hidden_windows(l1_model, lut, corpus_bytes, n_windows=N_SAMPLES):
    """Batched: for each 16-byte window, produce 648-dim L1 hidden concat.

    Returns:
      x_648: (N, 648) ndarray — the L1 hidden representation
      pair_embeds: (N, 8, 32) ndarray — original byte-pair embeds (for sign-match metric)
      window_bytes: (N, 16) ndarray — original bytes (diagnostic only)
    """
    rng = np.random.default_rng(42)
    # Sample with replacement is fine; replace=False chokes on big corpora
    max_off = len(corpus_bytes) - 16
    offsets = rng.integers(0, max_off, size=n_windows)
    arr = np.frombuffer(corpus_bytes, dtype=np.uint8)

    # Build (N, 16) byte windows
    window_bytes = np.stack([arr[o:o + 16] for o in offsets])

    # LUT lookup -> (N, 16, 16) embeds
    bytes_t = torch.tensor(window_bytes, dtype=torch.long, device=DEVICE)
    embeds = lut[bytes_t]  # (N, 16, 16)

    # Form 8 byte pairs per window: concat(embed[2i], embed[2i+1]) -> (N, 8, 32)
    even = embeds[:, 0::2]  # (N, 8, 16) — first byte of each pair
    odd = embeds[:, 1::2]   # (N, 8, 16)
    pair_embeds = torch.cat([even, odd], dim=-1)  # (N, 8, 32)

    # L1 encoder batched: (N*8, 32) -> (N*8, 81)
    flat = pair_embeds.reshape(-1, 32)
    with torch.no_grad():
        h_flat = l1_model.c19(flat @ l1_model.W + l1_model.b1)  # (N*8, 81)
    hiddens = h_flat.reshape(n_windows, 8, 81)
    x_648 = hiddens.reshape(n_windows, -1).cpu().numpy()
    return x_648, pair_embeds.cpu().numpy(), window_bytes


def pca_reconstruct(X, D):
    """Fit PCA to X, project to D dims, reconstruct."""
    mean = X.mean(axis=0, keepdims=True)
    Xc = X - mean
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    proj = Xc @ Vt[:D].T  # (N, D)
    recon = proj @ Vt[:D] + mean  # (N, 648)
    return recon, proj


def measure_sign_match_rate(x_648_recon, pair_embeds, l1_model):
    """New metric (GPT-consistent): does the L1 decoder restore sign-match
    to the original byte-pair embeds after [identity / PCA recon]?

    This matches the L1 lossless definition: sign(decoded) == sign(original)
    on all 32 dimensions of each byte pair.

    Returns: (pair_sign_rate, window_exact_rate, all32_matching_dims_rate)
    """
    N = pair_embeds.shape[0]
    recon_t = torch.tensor(x_648_recon, dtype=torch.float32, device=DEVICE)
    orig_t = torch.tensor(pair_embeds, dtype=torch.float32, device=DEVICE)  # (N, 8, 32)
    with torch.no_grad():
        hiddens = recon_t.reshape(N, 8, 81)
        decoded = hiddens @ l1_model.W.t() + l1_model.b2  # (N, 8, 32)
        per_dim_match = torch.sign(decoded) == torch.sign(orig_t)  # (N, 8, 32)
        per_pair = per_dim_match.all(dim=2)  # (N, 8) — pair ok iff all 32 dim sign-match
        per_dim_rate = per_dim_match.float().mean().item()
        pair_rate = per_pair.float().mean().item()
        window_rate = per_pair.all(dim=1).float().mean().item()
    return pair_rate, window_rate, per_dim_rate


def main():
    print("=" * 70)
    print("L2 PHASE 0 — GEOMETRY PROBE (linear PCA baseline)")
    print("=" * 70)

    print(f"\n[1] Loading L1 champion model and L0 LUT...")
    l1 = load_l1_champion()
    lut = load_l0_lut()
    print(f"    L1: W {tuple(l1.W.shape)}, L0 LUT {tuple(lut.shape)}")

    print(f"\n[2] Loading corpus...")
    corpus = load_corpus_bytes()
    print(f"    {len(corpus)} bytes loaded")

    print(f"\n[3] Collecting {N_SAMPLES} 16-byte windows, converting to 648-dim L1 hiddens...")
    t0 = time.time()
    X, pair_embeds, window_bytes = collect_l1_hidden_windows(l1, lut, corpus)
    print(f"    X shape: {X.shape}  pair_embeds: {pair_embeds.shape}  ({time.time()-t0:.1f}s)")

    print(f"\n[4] Sanity check — identity (no PCA) L1 decoder sign-match rate:")
    pair_rate, exact, per_dim = measure_sign_match_rate(X, pair_embeds, l1)
    print(f"    Identity: per-dim = {100*per_dim:.4f}%  per-pair = {100*pair_rate:.4f}%  exact 16-byte = {100*exact:.4f}%")
    print(f"    (should be ~100% if L1 lossless holds on natural text — verifies the probe itself)")

    print(f"\n[5] PCA sweep (linear compression 648 -> D -> 648, then L1 decode):")
    results = []
    for D in D_SWEEP:
        t0 = time.time()
        X_recon, _ = pca_reconstruct(X, D)
        pair_rate, exact, per_dim = measure_sign_match_rate(X_recon, pair_embeds, l1)
        results.append((D, pair_rate, exact, per_dim))
        print(f"    D={D:4d}: per-dim={100*per_dim:.3f}%  per-pair={100*pair_rate:.3f}%  "
              f"exact-window={100*exact:.3f}%  ({time.time()-t0:.1f}s)")

    print(f"\n{'='*70}")
    print(f"VERDICT: linear PCA knee for 648-dim -> ? dim")
    print(f"{'='*70}")
    print(f"{'D':>6} {'per-dim':>12} {'per-pair':>12} {'exact-window':>14}")
    prev = 0
    for D, pr, er, pd in results:
        knee = " <-- KNEE" if er >= 0.90 and prev < 0.90 else ""
        prev = er
        print(f"{D:>6} {100*pd:>11.3f}% {100*pr:>11.3f}% {100*er:>13.3f}%{knee}")

    with open(OUT_DIR / "phase0_pca_results.json", "w") as f:
        json.dump({
            "n_samples": N_SAMPLES,
            "D_sweep": D_SWEEP,
            "results": [
                {"D": D, "per_dim_rate": pd, "per_pair_rate": pr, "exact_window_rate": er}
                for D, pr, er, pd in results
            ],
        }, f, indent=2)
    print(f"\n  Saved: {OUT_DIR / 'phase0_pca_results.json'}")

    print(f"\nINTERPRETATION:")
    print(f"  - If exact-window >= 90% at D=64: linear geometry is tractable, proceed to L2-v1.")
    print(f"  - If exact-window drops sharply below D=64: L1 hidden space is anisotropic,")
    print(f"    need higher D or non-linear autoencoder.")
    print(f"  - If exact-window >90% even at D=32: bottleneck target can be smaller.")


if __name__ == "__main__":
    main()

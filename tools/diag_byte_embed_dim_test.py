"""Byte embedding dim validation: is dim=16 good and sufficient?

Sweep dim in {8, 16, 32}. Four tests per dim:
  1. Lossless roundtrip (256/256 byte-perfect)
  2. Collision test (min pairwise latent distance — must be > 0)
  3. Semantic cluster ratios (dual-loss, lower = tighter clustering)
  4. Downstream char-LM (frozen embeddings, predict masked char from 8-context)

The downstream test is the real validation: does the embedding help prediction?
"""

from __future__ import annotations
import sys
import time
from pathlib import Path
from itertools import combinations

import torch
import torch.nn.functional as F

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_BITS = 8
SEED = 42
CTX = 8
MASK_POS = CTX // 2
VOCAB = 256


# ── STE int8 quantization ──────────────────────────────────

class Int8STE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w, scale):
        return torch.clamp(torch.round(w / scale), -127, 127) * scale

    @staticmethod
    def backward(ctx, g):
        return g, None


def quantize_int8(W):
    scale = W.abs().max().detach().clamp(min=1e-8) / 127.0
    return Int8STE.apply(W, scale)


def byte_to_bits(b):
    bits = torch.zeros(b.shape[0], N_BITS, device=b.device)
    for i in range(N_BITS):
        bits[:, i] = (b >> i) & 1
    return bits


# ── Load corpus bigrams ────────────────────────────────────

def load_bigrams(path, n, seed=SEED):
    raw = Path(path).read_bytes()
    gen = torch.Generator().manual_seed(seed)
    arr = torch.frombuffer(bytearray(raw), dtype=torch.uint8)
    offs = torch.randint(0, len(raw) - 2, (n,), generator=gen)
    return arr[offs].long(), arr[offs + 1].long()


# ── Train mirror autoencoder (dual loss) ───────────────────

def train_mirror(latent_dim, cur, nxt, epochs=30, batch=8192, lr=0.01):
    torch.manual_seed(SEED)
    W = (torch.randn(N_BITS, latent_dim, device=DEVICE) * 0.3).detach().requires_grad_(True)
    V = (torch.randn(latent_dim, VOCAB, device=DEVICE) * 0.1).detach().requires_grad_(True)
    opt = torch.optim.Adam([W, V], lr=lr)

    cur_d, nxt_d = cur.to(DEVICE), nxt.to(DEVICE)
    N = cur_d.shape[0]

    for ep in range(epochs):
        perm = torch.randperm(N, device=DEVICE)
        for start in range(0, N, batch):
            idx = perm[start:start + batch]
            bits = byte_to_bits(cur_d[idx]).float()
            inp = bits * 2.0 - 1.0
            Wq = quantize_int8(W)
            latent = inp @ Wq
            loss_rec = F.binary_cross_entropy_with_logits(latent @ Wq.t(), bits)
            loss_ctx = F.cross_entropy(latent @ V, nxt_d[idx])
            loss = loss_rec + 0.1 * loss_ctx
            opt.zero_grad(); loss.backward(); opt.step()

    return W.detach()


# ── Test 1: Lossless roundtrip ─────────────────────────────

def test_lossless(W):
    with torch.no_grad():
        all_b = torch.arange(VOCAB, device=DEVICE)
        bits = byte_to_bits(all_b).float()
        inp = bits * 2.0 - 1.0
        Wq = quantize_int8(W)
        pred = (inp @ Wq @ Wq.t() > 0).float()
        byte_acc = (pred == bits).all(dim=1).float().mean().item() * 100
        missed = int((~(pred == bits).all(dim=1)).sum().item())
    return byte_acc, missed


# ── Test 2: Collision test ─────────────────────────────────

def test_collision(W):
    with torch.no_grad():
        all_b = torch.arange(VOCAB, device=DEVICE)
        bits = byte_to_bits(all_b).float()
        inp = bits * 2.0 - 1.0
        Wq = quantize_int8(W)
        latents = inp @ Wq  # (256, dim)
        # Pairwise distances
        dists = torch.cdist(latents.unsqueeze(0), latents.unsqueeze(0)).squeeze(0)
        # Set diagonal to inf
        dists.fill_diagonal_(float('inf'))
        min_dist = dists.min().item()
        # Count unique latent vectors
        unique = len(torch.unique(latents, dim=0))
    return min_dist, unique


# ── Test 3: Semantic cluster ratios ────────────────────────

def test_clusters(W):
    with torch.no_grad():
        all_b = torch.arange(VOCAB, device=DEVICE)
        bits = byte_to_bits(all_b).float()
        inp = bits * 2.0 - 1.0
        Wq = quantize_int8(W)
        latents = inp @ Wq

    def avg_set_dist(byte_set):
        bs = list(byte_set)
        if len(bs) < 2:
            return 0.0
        ds = [(latents[i] - latents[j]).norm().item()
              for i, j in combinations(bs, 2)]
        return sum(ds) / len(ds)

    gen = torch.Generator(device="cpu").manual_seed(SEED)
    ri = torch.randint(0, 256, (1000,), generator=gen)
    rj = torch.randint(0, 256, (1000,), generator=gen)
    mask = ri != rj
    rnd = sum((latents[ri[k]] - latents[rj[k]]).norm().item()
              for k in range(len(ri)) if mask[k]) / mask.sum().item()

    vowels = [ord(c) for c in "aeiouAEIOU"]
    digits = [ord(c) for c in "0123456789"]
    cases = [(ord(c), ord(c.upper())) for c in "abcdefghijklmnopqrstuvwxyz"]
    ws = [32, 9, 10, 13]

    return {
        "vowels": avg_set_dist(vowels) / rnd,
        "digits": avg_set_dist(digits) / rnd,
        "cases": sum((latents[a] - latents[b]).norm().item() for a, b in cases) / 26 / rnd,
        "whitespace": avg_set_dist(ws) / rnd,
    }


# ── Test 4: Downstream char-LM ────────────────────────────

def test_downstream(W, corpus_path, n_train=20000, n_eval=5000):
    """Frozen embedding W, train a linear predictor on masked-char task."""
    raw = Path(corpus_path).read_bytes()
    arr = torch.frombuffer(bytearray(raw), dtype=torch.uint8)

    def sample(n, seed):
        gen = torch.Generator().manual_seed(seed)
        offs = torch.randint(0, len(arr) - CTX - 1, (n,), generator=gen)
        idx = offs.unsqueeze(1) + torch.arange(CTX).unsqueeze(0)
        chunks = arr[idx].long()
        targets = chunks[:, MASK_POS].clone()
        chunks[:, MASK_POS] = ord(' ')  # mask with space byte
        return chunks.to(DEVICE), targets.to(DEVICE)

    train_x, train_y = sample(n_train, 42)
    eval_x, eval_y = sample(n_eval, 99)

    with torch.no_grad():
        Wq = quantize_int8(W)

        def embed(chunks):
            # chunks: (N, CTX) byte values → (N, CTX, 8) bits → (N, CTX, dim) latent
            flat = chunks.flatten()
            bits = byte_to_bits(flat).float() * 2.0 - 1.0
            lat = bits @ Wq  # (N*CTX, dim)
            return lat.view(chunks.shape[0], CTX, -1).reshape(chunks.shape[0], -1)

        train_feat = embed(train_x)  # (N, CTX*dim)
        eval_feat = embed(eval_x)

    D = train_feat.shape[1]
    torch.manual_seed(SEED)
    P = (torch.randn(D, VOCAB, device=DEVICE) * 0.01).detach().requires_grad_(True)
    bias = torch.zeros(VOCAB, device=DEVICE).requires_grad_(True)
    opt = torch.optim.Adam([P, bias], lr=0.005)

    for ep in range(100):
        logits = train_feat @ P + bias
        loss = F.cross_entropy(logits, train_y)
        opt.zero_grad(); loss.backward(); opt.step()

    with torch.no_grad():
        train_acc = (train_feat @ P + bias).argmax(1).eq(train_y).float().mean().item() * 100
        eval_acc = (eval_feat @ P + bias).argmax(1).eq(eval_y).float().mean().item() * 100

    return train_acc, eval_acc


# ── Random embedding baseline ─────────────────────────────

def make_random_W(latent_dim):
    torch.manual_seed(999)
    return (torch.randn(N_BITS, latent_dim, device=DEVICE) * 0.5).detach()


# ── Main ───────────────────────────────────────────────────

def main():
    corpus = sys.argv[1] if len(sys.argv) > 1 else \
        "S:/AI/MESSY TRAINING DATA - INPUT ONLY/Fineweb edu 10B/fineweb_edu_30m.txt"

    print(f"Device: {DEVICE}")
    print(f"Byte embedding dim validation — dim in {{8, 16, 32}}")
    print(f"Corpus: {corpus}\n")

    # Load bigrams for dual-loss training
    print("Loading 200K bigrams for training...")
    cur, nxt = load_bigrams(corpus, 200_000)

    results = []
    for dim in [8, 16, 32]:
        print(f"\n{'='*60}")
        print(f"  DIM = {dim}")
        print(f"{'='*60}")

        # Train
        t0 = time.time()
        W = train_mirror(dim, cur, nxt)
        t_train = time.time() - t0
        print(f"  trained in {t_train:.1f}s")

        # Test 1: lossless
        byte_acc, missed = test_lossless(W)
        print(f"  [1] Lossless: {byte_acc:.2f}%  missed={missed}/256")

        # Test 2: collision
        min_dist, n_unique = test_collision(W)
        print(f"  [2] Collision: min_dist={min_dist:.6f}  unique={n_unique}/256")

        # Test 3: clusters
        cl = test_clusters(W)
        print(f"  [3] Clusters: vowels={cl['vowels']:.4f}  digits={cl['digits']:.4f}  "
              f"cases={cl['cases']:.4f}  ws={cl['whitespace']:.4f}")

        # Test 4: downstream
        train_acc, eval_acc = test_downstream(W, corpus)
        print(f"  [4] Downstream char-LM: train={train_acc:.2f}%  eval={eval_acc:.2f}%")

        # Random baseline downstream
        W_rnd = make_random_W(dim)
        rnd_train, rnd_eval = test_downstream(W_rnd, corpus)
        print(f"      Random baseline:    train={rnd_train:.2f}%  eval={rnd_eval:.2f}%")
        print(f"      Delta (learned - random): {eval_acc - rnd_eval:+.2f}pp")

        results.append({
            "dim": dim, "byte_acc": byte_acc, "missed": missed,
            "min_dist": min_dist, "unique": n_unique,
            "clusters": cl, "eval_acc": eval_acc, "rnd_eval": rnd_eval,
        })

    # Summary
    print(f"\n{'='*72}")
    print(f"{'METRIC':<30} {'dim=8':>12} {'dim=16':>12} {'dim=32':>12}")
    print(f"{'='*72}")
    for key, label in [
        ("byte_acc", "Lossless (%)"),
        ("unique", "Unique latents (/256)"),
        ("min_dist", "Min pairwise dist"),
    ]:
        vals = [f"{r[key]:>12.4f}" if isinstance(r[key], float) else f"{r[key]:>12}" for r in results]
        print(f"{label:<30} {vals[0]} {vals[1]} {vals[2]}")

    for cat in ["vowels", "digits", "cases", "whitespace"]:
        vals = [f"{r['clusters'][cat]:>12.4f}" for r in results]
        print(f"{'Cluster: ' + cat:<30} {vals[0]} {vals[1]} {vals[2]}")

    for key, label in [
        ("eval_acc", "Downstream eval (%)"),
        ("rnd_eval", "Random baseline eval (%)"),
    ]:
        vals = [f"{r[key]:>12.2f}" for r in results]
        print(f"{label:<30} {vals[0]} {vals[1]} {vals[2]}")

    deltas = [f"{r['eval_acc'] - r['rnd_eval']:>+12.2f}" for r in results]
    print(f"{'Delta (learned - random)':<30} {deltas[0]} {deltas[1]} {deltas[2]}")
    print(f"{'='*72}")
    print(f"\nPass criteria:")
    print(f"  [1] Lossless = 100% at dim=16  (hard requirement)")
    print(f"  [2] Unique = 256/256            (no collisions)")
    print(f"  [3] Cluster ratios < 1.0        (semantic structure)")
    print(f"  [4] Delta > 0                   (learned beats random on real task)")


if __name__ == "__main__":
    main()

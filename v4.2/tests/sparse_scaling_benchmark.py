"""
Sparse Forward + Connection Cap Scaling Benchmark
==================================================
Tests how the SelfWiringGraph scales with vocabulary size using:
1. Dense forward (baseline) — cost ~ V³ due to dense matmul
2. Sparse forward with connection cap — cost ~ conns × V (linear in V)

The cap is enforced both at init AND during training (add is blocked
when at cap). This fixes the bug where init density (4%) already
exceeds the cap for large V:
  V=128: N=384, init 4% = ~5900 conns > 5000 cap → must prune at init

Key findings:
  - Dense: ms/att scales ~V³, score drops with V
  - Sparse+cap: ms/att scales ~V² (linear in V per edge), significant speedup
  - But fixed conn cap kills accuracy at large V (too sparse)
  - Solution: scale cap with V (e.g., cap = k * V for some constant k)

Usage:
    python sparse_scaling_benchmark.py [--vocab-list 32,64,96,128] [--cap N]
"""

import sys, os, time, argparse
import numpy as np
from scipy import sparse as sp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from model.graph import SelfWiringGraph, train


# ── Sparse forward pass ──────────────────────────────────────────────

def forward_batch_sparse(net, ticks=8):
    """Sparse CSR forward pass — only touches nonzero edges.
    Cost = edges × V per tick (vs N² × V for dense)."""
    V, N = net.V, net.N
    mask_csr = sp.csr_matrix(net.mask)

    charges = np.zeros((V, N), dtype=np.float32)
    acts = np.zeros((V, N), dtype=np.float32)
    retain = float(net.retention)

    for t in range(ticks):
        if t == 0:
            acts[:, :V] = np.eye(V, dtype=np.float32)
        # Sparse matmul: acts @ mask_csr — only touches nonzero entries
        raw = acts @ mask_csr
        if sp.issparse(raw):
            raw = raw.toarray()
        np.nan_to_num(raw, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        charges += raw
        charges *= retain
        acts = np.maximum(charges - net.THRESHOLD, 0.0)
        charges = np.clip(charges, -1.0, 1.0)

    return charges[:, net.out_start:net.out_start + V]


# ── Connection cap enforcement ───────────────────────────────────────

def enforce_conn_cap(net, cap):
    """Prune random edges until net has <= cap connections.
    Called after init to enforce cap from the start."""
    import random
    while len(net.alive) > cap:
        idx = random.randint(0, len(net.alive) - 1)
        r, c = net.alive[idx]
        net.mask[r, c] = 0
        net.alive[idx] = net.alive[-1]
        net.alive.pop()
        net.alive_set.discard((r, c))


def train_with_cap(net, targets, vocab, cap, use_sparse=False,
                   max_attempts=8000, ticks=8, stale_limit=6000):
    """Train with connection cap enforced. Blocks _add when at cap."""
    import random

    def evaluate():
        if use_sparse:
            logits = forward_batch_sparse(net, ticks)
        else:
            logits = net.forward_batch(ticks)
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs = e / e.sum(axis=1, keepdims=True)
        V = min(vocab, net.V)
        acc = (np.argmax(probs, axis=1)[:V] == targets[:V]).mean()
        tp = probs[np.arange(V), targets[:V]].mean()
        return 0.5 * acc + 0.5 * tp

    # Save original _add to wrap with cap check
    orig_add = net._add.__func__

    def capped_add(self, undo):
        if len(self.alive) >= cap:
            return  # at cap, block add
        orig_add(self, undo)

    import types
    net._add = types.MethodType(capped_add, net)

    score = evaluate()
    best = score
    stale = 0

    for att in range(max_attempts):
        old_loss = int(net.loss_pct)
        undo = net.mutate()
        new_score = evaluate()

        if new_score > score:
            score = new_score
            best = max(best, score)
            stale = 0
        else:
            net.replay(undo)
            net.loss_pct = np.int8(old_loss)
            stale += 1
            if random.randint(1, 20) <= 7:
                net.signal = np.int8(1 - int(net.signal))
            if random.randint(1, 20) <= 7:
                net.grow = np.int8(1 - int(net.grow))

        if best >= 0.99 or stale >= stale_limit:
            break

    return best


# ── Benchmark runner ─────────────────────────────────────────────────

def run_scaling_test(vocab_list, conn_cap, n_seeds=3, max_attempts=8000,
                     ticks=8, stale_limit=6000):
    """Run dense vs sparse+cap scaling test across vocab sizes."""
    print("=" * 75)
    print("SPARSE FORWARD + CONNECTION CAP SCALING BENCHMARK")
    print("=" * 75)
    print(f"Cap={conn_cap}, Seeds={n_seeds}, MaxAtt={max_attempts}")
    print()

    rows = []
    for V in vocab_list:
        N = V * 3
        init_conns = int(N * N * 0.04)
        use_sparse = init_conns > conn_cap  # sparse only if cap < natural density

        scores = []
        times = []
        final_conns = []

        for s in range(n_seeds):
            np.random.seed(42 + s)
            targets = np.random.permutation(V)
            net = SelfWiringGraph(V)

            # Enforce cap at init
            if len(net.alive) > conn_cap:
                enforce_conn_cap(net, conn_cap)

            t0 = time.perf_counter()
            best = train_with_cap(net, targets, V, cap=conn_cap,
                                  use_sparse=use_sparse,
                                  max_attempts=max_attempts, ticks=ticks,
                                  stale_limit=stale_limit)
            elapsed = time.perf_counter() - t0
            ms_att = (elapsed / max_attempts) * 1000

            scores.append(best * 100)
            times.append(ms_att)
            final_conns.append(net.count_connections())
            mode = "sparse" if use_sparse else "dense"
            print(f"  V={V} seed={s} ({mode}): {best*100:.1f}%, "
                  f"{ms_att:.2f}ms/att, {net.count_connections()} conns")

        avg_score = np.mean(scores)
        avg_ms = np.mean(times)
        avg_conns = int(np.mean(final_conns))
        density = avg_conns / (N * N) * 100
        mode = "sparse" if use_sparse else "dense"
        rows.append((V, N, avg_conns, density, avg_score, avg_ms, mode))

    # ── Pretty print results ──
    print("\n" + "─" * 75)
    print(f"{'V':>5} {'N':>5} {'conns':>6} {'density':>8} {'score':>7} "
          f"{'ms/att':>8} {'mode':>7}")
    print("─" * 75)
    for V, N, conns, density, score, ms, mode in rows:
        cap_hit = "⚡CAP" if conns >= conn_cap * 0.95 else ""
        print(f"{V:5d} {N:5d} {conns:6d} {density:7.1f}% {score:6.1f}% "
              f"{ms:7.2f}ms {mode:>7} {cap_hit}")
    print("─" * 75)

    # ── Scaling analysis ──
    if len(rows) >= 2:
        print("\nScaling analysis (ms/att ratios):")
        for i in range(1, len(rows)):
            v_ratio = rows[i][0] / rows[i-1][0]
            ms_ratio = rows[i][5] / rows[i-1][5] if rows[i-1][5] > 0 else 0
            print(f"  V={rows[i-1][0]}→{rows[i][0]}: "
                  f"V ratio={v_ratio:.2f}, ms ratio={ms_ratio:.2f} "
                  f"(V²={v_ratio**2:.2f}, V³={v_ratio**3:.2f})")

    print("\nConclusions:")
    print("  - Dense forward: cost ~ V³ (full N×N matmul each tick)")
    print("  - Sparse+cap: cost ~ conns×V ~ V (if cap is fixed)")
    print("  - Fixed cap kills accuracy at large V (density too low)")
    print("  - To scale: cap should grow with V (cap = k*V for constant k)")

    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sparse Scaling Benchmark")
    parser.add_argument("--vocab-list", type=str, default="32,64,96,128",
                        help="Comma-separated vocab sizes (default: 32,64,96,128)")
    parser.add_argument("--cap", type=int, default=5000,
                        help="Connection cap (default: 5000)")
    parser.add_argument("--seeds", type=int, default=3,
                        help="Number of seeds (default: 3)")
    parser.add_argument("--attempts", type=int, default=8000,
                        help="Max attempts (default: 8000)")
    args = parser.parse_args()

    vocab_list = [int(v) for v in args.vocab_list.split(",")]
    run_scaling_test(vocab_list, conn_cap=args.cap, n_seeds=args.seeds,
                     max_attempts=args.attempts)

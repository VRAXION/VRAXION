"""
Batch Prune: Dense Init → Single-Pass Edge Scoring → Iterative Bulk Remove
===========================================================================
Instead of removing edges one-by-one (slow crystallize), we:
  1. Run ONE forward pass on eval data
  2. Record per-edge contribution to the output distribution
  3. Compare output distribution to actual English bigram distribution
  4. Score each edge: does it push the output TOWARD or AWAY from the target?
  5. Remove the worst N% of edges at once
  6. Repeat until the network is clean

Edge scoring method:
  - For edge (r → c) with weight w:
    signal_pushed = act[r] * w  (summed over ticks)
    This goes into charge[c], which contributes to output via output_proj[c, :]
    The output should match target_dist (English bigram)
  - edge_score = alignment of (signal * output_proj[c]) with (target - pred)
    Positive = edge is helping. Negative = edge is hurting.
  - Remove the most negative edges each round.
"""
import sys, os, time, random
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "model"))
from graph import SelfWiringGraph

IO = 256
H = 256
TICKS = 8
INJ_TICKS = 2
SEED = 42
CKPT_DIR = Path(__file__).resolve().parent / "checkpoints" / "batch_prune"


def make_bp(io_dim, seed=12345):
    rng = np.random.RandomState(seed)
    p = rng.randn(256, io_dim).astype(np.float32)
    p /= np.linalg.norm(p, axis=1, keepdims=True)
    return p


def compute_bigram_from_bytes(data_bytes):
    bigram = np.zeros((256, 256), dtype=np.float64)
    for i in range(len(data_bytes) - 1):
        bigram[data_bytes[i], data_bytes[i + 1]] += 1
    row_sums = bigram.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    bigram /= row_sums
    return bigram.astype(np.float32)


def forward_with_edge_tracking(mask, theta, decay, text_bytes, bp, input_proj, output_proj,
                                bigram):
    """
    Single forward pass that:
      1. Computes accuracy and loss (bigram cosine)
      2. Tracks per-edge cumulative signal contribution
      3. Scores each edge: does it help or hurt the output?
    Returns: (accuracy, avg_cosine, edge_scores_dict)
    """
    rs, cs = np.where(mask != 0)
    sp_vals = mask[rs, cs]
    n_edges = len(rs)
    if n_edges == 0:
        return 0.0, 0.0, {}

    pat_norm = bp / (np.linalg.norm(bp, axis=1, keepdims=True) + 1e-8)
    ret = 1.0 - decay

    state = np.zeros(H, dtype=np.float32)
    charge = np.zeros(H, dtype=np.float32)

    # Per-edge cumulative alignment score
    edge_alignment = np.zeros(n_edges, dtype=np.float64)
    correct = 0
    cos_sum = 0.0
    n = 0

    for i in range(len(text_bytes) - 1):
        act = state.copy()
        # Track per-edge signals across ticks
        edge_signals_total = np.zeros(n_edges, dtype=np.float32)

        for t in range(TICKS):
            if t < INJ_TICKS:
                act = act + bp[text_bytes[i]] @ input_proj

            # Per-edge signal: act[source] * weight
            edge_signals = act[rs] * sp_vals  # (n_edges,)
            edge_signals_total += edge_signals

            # Apply signals
            raw = np.zeros(H, dtype=np.float32)
            if n_edges:
                np.add.at(raw, cs, edge_signals)
            charge += raw
            charge *= ret
            np.clip(charge, -10.0, 10.0, out=charge)
            act = np.maximum(charge, 0.0)
            charge = np.maximum(charge, 0.0)

        state = act.copy()

        # Compute output and target
        out = charge @ output_proj
        out_n = out / (np.linalg.norm(out) + 1e-8)
        sims = out_n @ pat_norm.T
        e = np.exp(sims - sims.max())
        pred = e / e.sum()

        target_dist = bigram[text_bytes[i]]

        # Accuracy
        if np.argmax(pred) == text_bytes[i + 1]:
            correct += 1

        # Bigram cosine
        cos = np.dot(pred, target_dist) / (np.linalg.norm(pred) * np.linalg.norm(target_dist) + 1e-8)
        cos_sum += cos

        # Error signal: where should the output shift?
        error = target_dist - pred  # (256,) — positive = need more, negative = need less

        # For each edge (r→c): what direction does it push the output?
        # Edge pushes signal into charge[c], which projects via output_proj[c, :]
        # edge_output_direction[e] = output_proj[cs[e], :] (256-dim)
        # edge_alignment[e] += dot(edge_output_direction, error) * edge_signal_magnitude

        for e_idx in range(n_edges):
            c_neuron = cs[e_idx]
            # How much this edge pushed (magnitude, can be + or -)
            signal_mag = edge_signals_total[e_idx]
            # Direction this edge pushes the output
            edge_output_dir = output_proj[c_neuron, :]
            # Alignment with what we NEED (error)
            alignment = np.dot(edge_output_dir, error) * signal_mag
            edge_alignment[e_idx] += alignment

        n += 1

    acc = correct / n if n else 0
    avg_cos = cos_sum / n if n else 0

    # Build edge score dict
    edge_scores = {}
    for e_idx in range(n_edges):
        r, c = int(rs[e_idx]), int(cs[e_idx])
        edge_scores[(r, c)] = float(edge_alignment[e_idx] / max(n, 1))

    return acc, avg_cos, edge_scores


def eval_accuracy_simple(mask, theta, decay, text_bytes, bp, input_proj, output_proj):
    rs, cs = np.where(mask != 0)
    sp_vals = mask[rs, cs]
    pat_norm = bp / (np.linalg.norm(bp, axis=1, keepdims=True) + 1e-8)
    ret = 1.0 - decay
    state = np.zeros(H, dtype=np.float32)
    charge = np.zeros(H, dtype=np.float32)
    correct = 0; total = 0
    for i in range(len(text_bytes) - 1):
        act = state.copy()
        for t in range(TICKS):
            if t < INJ_TICKS:
                act = act + bp[text_bytes[i]] @ input_proj
            raw = np.zeros(H, dtype=np.float32)
            if len(rs):
                np.add.at(raw, cs, act[rs] * sp_vals)
            charge += raw; charge *= ret
            np.clip(charge, -10.0, 10.0, out=charge)
            act = np.maximum(charge, 0.0)
            charge = np.maximum(charge, 0.0)
        state = act.copy()
        out = charge @ output_proj
        out_n = out / (np.linalg.norm(out) + 1e-8)
        sims = out_n @ pat_norm.T
        if np.argmax(sims) == text_bytes[i + 1]:
            correct += 1
        total += 1
    return correct / total if total else 0


def network_stats(mask):
    edges = int(np.count_nonzero(mask))
    in_deg = np.count_nonzero(mask, axis=0)
    out_deg = np.count_nonzero(mask, axis=1)
    connected = (in_deg + out_deg) > 0
    has_in = in_deg > 0
    has_out = out_deg > 0
    return {
        'edges': edges,
        'connected': int(connected.sum()),
        'bidirectional': int((has_in & has_out).sum()),
        'max_deg': int(max(in_deg.max(), out_deg.max())) if edges > 0 else 0,
    }


def batch_prune_round(net, eval_seqs, bp, input_proj, output_proj, bigram, prune_pct=10):
    """
    One round of batch pruning:
      1. Score all edges on multiple sequences
      2. Remove the worst prune_pct% of edges
    Returns: (n_removed, accuracy, avg_cosine)
    """
    # Aggregate edge scores across sequences
    all_scores = {}
    total_acc = 0
    total_cos = 0

    for seq in eval_seqs:
        acc, cos, scores = forward_with_edge_tracking(
            net.mask, net.theta, net.decay, seq, bp, input_proj, output_proj, bigram)
        total_acc += acc
        total_cos += cos
        for (r, c), score in scores.items():
            all_scores[(r, c)] = all_scores.get((r, c), 0.0) + score

    avg_acc = total_acc / len(eval_seqs)
    avg_cos = total_cos / len(eval_seqs)

    # Sort edges by score (worst first)
    sorted_edges = sorted(all_scores.items(), key=lambda x: x[1])

    # Remove worst prune_pct%
    n_edges = len(sorted_edges)
    n_remove = max(1, int(n_edges * prune_pct / 100))

    removed = 0
    for (r, c), score in sorted_edges[:n_remove]:
        if score < 0:  # only remove edges that are actually hurting
            net.mask[r, c] = 0.0
            removed += 1

    if removed > 0:
        net.resync_alive()

    return removed, avg_acc, avg_cos


if __name__ == "__main__":
    os.makedirs(CKPT_DIR, exist_ok=True)

    print("Loading alice.txt...")
    with open(DATA_DIR / "alice.txt", "rb") as f:
        all_data = np.frombuffer(f.read(), dtype=np.uint8)
    print(f"  {len(all_data)} bytes")

    print("Computing bigram table...")
    bigram = compute_bigram_from_bytes(all_data)
    bp = make_bp(IO)

    # Deterministic projections
    random.seed(SEED); np.random.seed(SEED)
    ref = SelfWiringGraph(IO, hidden_ratio=1, projection_scale=1.0, seed=SEED)
    input_proj = ref.input_projection
    output_proj = ref.output_projection

    # Eval sequences
    eval_rng = np.random.RandomState(9999)
    eval_seqs = [all_data[off:off + 200]
                 for off in [eval_rng.randint(0, len(all_data) - 200) for _ in range(5)]]

    # ── Test different densities ────────────────────────────────────
    for density_pct in [2, 4, 8]:
        print(f"\n{'='*65}")
        print(f"  BATCH PRUNE: Dense {density_pct}% init, prune worst 10% per round")
        print(f"{'='*65}")

        random.seed(SEED); np.random.seed(SEED)
        net = SelfWiringGraph(IO, hidden_ratio=1, density=density_pct,
                              projection_scale=1.0, seed=SEED)
        net.theta[:] = 0.0
        decay_rng = np.random.RandomState(99)
        net.decay[:] = decay_rng.uniform(0.08, 0.24, H).astype(np.float32)
        net.input_projection = input_proj
        net.output_projection = output_proj

        stats = network_stats(net.mask)
        init_acc = np.mean([eval_accuracy_simple(net.mask, net.theta, net.decay, s, bp,
                            input_proj, output_proj) for s in eval_seqs])
        print(f"  INIT: acc={init_acc*100:.2f}% edges={stats['edges']} "
              f"connected={stats['connected']} bidir={stats['bidirectional']}")

        # Save init checkpoint
        net.save(str(CKPT_DIR / f"dense{density_pct}_init.npz"))

        t0 = time.time()
        round_num = 0
        prune_pct = 10  # remove worst 10% per round

        while True:
            round_num += 1
            n_before = net.count_connections()
            if n_before == 0:
                print(f"  [round {round_num}] No edges left!")
                break

            removed, acc, cos = batch_prune_round(
                net, eval_seqs, bp, input_proj, output_proj, bigram, prune_pct=prune_pct)

            stats = network_stats(net.mask)
            # Also compute fresh accuracy (not from scoring pass)
            fresh_acc = np.mean([eval_accuracy_simple(net.mask, net.theta, net.decay, s, bp,
                                 input_proj, output_proj) for s in eval_seqs])

            elapsed = time.time() - t0
            print(f"  [round {round_num:3d}] removed={removed:4d} edges={stats['edges']:5d} "
                  f"acc={fresh_acc*100:.2f}% cos={cos:.4f} "
                  f"connected={stats['connected']} bidir={stats['bidirectional']} "
                  f"({elapsed:.1f}s)")
            sys.stdout.flush()

            if removed == 0:
                print(f"  Converged: no harmful edges found.")
                break

            # Save periodic checkpoints
            if round_num % 5 == 0:
                ckpt = CKPT_DIR / f"dense{density_pct}_round{round_num}.npz"
                net.save(str(ckpt))

            # Safety: stop if too many rounds
            if round_num >= 100:
                print(f"  Max rounds reached.")
                break

        # Final checkpoint
        net.save(str(CKPT_DIR / f"dense{density_pct}_final.npz"))
        stats = network_stats(net.mask)
        final_acc = np.mean([eval_accuracy_simple(net.mask, net.theta, net.decay, s, bp,
                             input_proj, output_proj) for s in eval_seqs])
        elapsed = time.time() - t0

        print(f"  FINAL: acc={final_acc*100:.2f}% edges={stats['edges']} "
              f"connected={stats['connected']} bidir={stats['bidirectional']} "
              f"{elapsed:.1f}s")
        print(f"  Saved: {CKPT_DIR / f'dense{density_pct}_final.npz'}")

    print(f"\n{'='*65}")
    print(f"  Done!")
    print(f"{'='*65}")

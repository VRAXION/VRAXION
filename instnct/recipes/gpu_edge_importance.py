"""
GPU Batched Edge Importance Map
================================
For every edge: compute how much the bigram cosine score changes when removed.
All edges tested in parallel on GPU via batched forward pass.

Output: per-edge importance score + distribution analysis.
"""
import sys, os, time
import numpy as np
import torch
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "model"))
from graph import SelfWiringGraph
from quaternary_mask import QuaternaryMask
from gpu_forward import GPUForward

TICKS = 16; INPUT_DURATION = 2; EVAL_TOKENS = 20
BATCH_SIZE = 256  # edges tested per GPU call


def build_sdr(n, dim, k, seed):
    rng = np.random.RandomState(seed)
    t = np.zeros((n, dim), np.float32)
    for v in range(n): t[v, rng.choice(dim, size=k, replace=False)] = 1.0
    return t

def build_freq_order(dim, bigram, seed=12345):
    freq = bigram.sum(axis=0) + bigram.sum(axis=1)
    rank = np.argsort(freq)[::-1]
    rng = np.random.RandomState(seed)
    p = np.zeros((256, dim), np.float32)
    for i, byte_idx in enumerate(rank):
        t = i / 255.0
        for d in range(dim):
            p[byte_idx, d] = np.sin(2 * np.pi * t * (d+1) / dim * 3) + rng.randn() * 0.3
    p /= np.linalg.norm(p, axis=1, keepdims=True) + 1e-8
    return p.astype(np.float32)


def eval_bigram_cosine_gpu(gpu, rows, cols, theta, channel, polarity,
                           text_bytes, bp_in, bp_out, bigram,
                           H, in_dim, out_dim):
    """Single network bigram cosine eval on GPU. Returns float score."""
    dev = gpu.device
    state = torch.zeros(H, device=dev)
    charge = torch.zeros(H, device=dev)
    total_cos = 0.0; n = 0

    for i in range(min(EVAL_TOKENS, len(text_bytes) - 1)):
        inj = torch.zeros(H, device=dev)
        inj[:in_dim] = bp_in[text_bytes[i]]
        state, charge = gpu.rollout_token(
            inj, rows, cols, theta, channel, polarity,
            ticks=TICKS, input_duration=INPUT_DURATION,
            state=state, charge=charge)
        logits = bp_out @ charge[H - out_dim:]
        # Softmax
        e = torch.exp(logits - logits.max())
        pred = e / e.sum()
        # Target bigram distribution
        tgt = bigram[text_bytes[i]]
        # Cosine similarity
        cos = (pred * tgt).sum() / (pred.norm() * tgt.norm() + 1e-8)
        total_cos += cos.item()
        n += 1
    return total_cos / n if n else 0.0


def eval_batched_remove_edges(gpu, rows_full, cols_full, edge_indices_to_test,
                              theta, channel, polarity,
                              text_bytes, bp_in, bp_out, bigram,
                              H, in_dim, out_dim):
    """Eval B networks, each missing 1 different edge. Returns B scores."""
    dev = gpu.device
    B = len(edge_indices_to_test)
    n_edges = len(rows_full)

    scores = []
    for i in range(min(EVAL_TOKENS, len(text_bytes) - 1)):
        inj_single = torch.zeros(H, device=dev)
        inj_single[:in_dim] = bp_in[text_bytes[i]]

    # Sequential per-token (state carries over), but batched per-edge
    state_batch = torch.zeros(B, H, device=dev)
    charge_batch = torch.zeros(B, H, device=dev)
    total_cos = torch.zeros(B, device=dev)
    n_tok = 0

    for i in range(min(EVAL_TOKENS, len(text_bytes) - 1)):
        inj = torch.zeros(H, device=dev)
        inj[:in_dim] = bp_in[text_bytes[i]]
        inj_batch = inj.unsqueeze(0).expand(B, -1)

        # For each candidate b: remove edge edge_indices_to_test[b]
        # Build B sparse caches with 1 edge removed each
        # Trick: instead of B different edge lists, mask the removed edge's contribution
        act = state_batch.clone()
        cur_charge = charge_batch.clone()

        for tick in range(TICKS):
            if tick % 6 == 0:
                cur_charge = torch.clamp(cur_charge - 1.0, min=0.0)
            if tick < INPUT_DURATION:
                act = act + inj_batch
            # Propagate: scatter_add for all edges, then SUBTRACT the removed edge
            raw = torch.zeros(B, H, device=dev)
            if n_edges > 0:
                raw.scatter_add_(1, cols_full.unsqueeze(0).expand(B, -1),
                                 act[:, rows_full])
            # Subtract the contribution of the removed edge for each candidate
            for b_idx in range(B):
                e_idx = edge_indices_to_test[b_idx]
                if e_idx < n_edges:
                    src_neuron = rows_full[e_idx]
                    tgt_neuron = cols_full[e_idx]
                    raw[b_idx, tgt_neuron] -= act[b_idx, src_neuron]

            cur_charge = torch.clamp(cur_charge + raw, 0.0, 15.0)
            # Wave gating
            theta_mult = gpu.wave_lut[channel, tick % 8]
            eff_theta = torch.clamp(theta * theta_mult, 1.0, 15.0)
            fired = cur_charge >= eff_theta
            act = fired.float() * polarity
            cur_charge[fired] = 0.0

        state_batch = act
        charge_batch = cur_charge

        # Compute bigram cosine for all B candidates
        out_charges = charge_batch[:, H - out_dim:]  # (B, out_dim)
        logits = out_charges @ bp_out.T  # (B, 256)
        e = torch.exp(logits - logits.max(dim=1, keepdim=True).values)
        pred = e / e.sum(dim=1, keepdim=True)  # (B, 256)
        tgt = bigram[text_bytes[i]].unsqueeze(0)  # (1, 256)
        cos = (pred * tgt).sum(dim=1) / (pred.norm(dim=1) * tgt.norm() + 1e-8)
        total_cos += cos
        n_tok += 1

    return (total_cos / n_tok).cpu().numpy()  # (B,) array of scores


if __name__ == "__main__":
    from lib.data import load_fineweb_bytes, resolve_fineweb_path
    resolve_fineweb_path()
    ALL_DATA = load_fineweb_bytes()
    print(f"Loaded {len(ALL_DATA)/1e6:.1f} MB")

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    bigram_np = np.load(os.path.join(BASE_DIR, "data", "bigram_table.npy"))

    # Try H=1024 checkpoint first, fall back to H=256
    ckpt_path = os.path.join(BASE_DIR, "data", "h1024_build_checkpoint.npz")
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(BASE_DIR, "data", "build_checkpoint.npz")
    ckpt = np.load(ckpt_path)
    qdata = ckpt['qdata']
    theta_np = ckpt['theta']
    channel_np = ckpt['channel']
    pol_f_np = ckpt['pol_f']

    H = int(np.sqrt(len(qdata) * 2) + 1)  # recover H from qmask size
    # More precise H recovery
    for h in [256, 512, 1024, 2048, 4096]:
        if h * (h - 1) // 2 == len(qdata):
            H = h; break
    print(f"Checkpoint H={H}, qdata len={len(qdata)}")

    PHI = (1 + 5**0.5) / 2
    IN_DIM = int(round(H / PHI))
    OUT_DIM = int(round(H / PHI))
    SDR_K = int(round(IN_DIM * 0.20))

    bp_in_np = build_sdr(256, IN_DIM, SDR_K, 42)
    bp_out_np = build_freq_order(OUT_DIM, bigram_np)

    # Get edges
    qm = QuaternaryMask(H, qdata)
    rows_np, cols_np = qm.to_directed_edges()
    n_edges = len(rows_np)
    print(f"Network: {n_edges} edges, {qm.count_bidir()} bidir")

    # Move to GPU
    dev = torch.device('cuda')
    gpu = GPUForward(H, device='cuda')
    rows_t = torch.tensor(rows_np, dtype=torch.long, device=dev)
    cols_t = torch.tensor(cols_np, dtype=torch.long, device=dev)
    theta_t = torch.tensor(theta_np, dtype=torch.float32, device=dev)
    channel_t = torch.tensor(channel_np, dtype=torch.long, device=dev)
    pol_t = torch.tensor(pol_f_np, dtype=torch.float32, device=dev)
    bp_in_t = torch.tensor(bp_in_np, dtype=torch.float32, device=dev)
    bp_out_t = torch.tensor(bp_out_np, dtype=torch.float32, device=dev)
    bigram_t = torch.tensor(bigram_np, dtype=torch.float32, device=dev)

    # Eval sequences
    eval_rng = np.random.RandomState(9999)
    eval_seqs = [ALL_DATA[off:off + 100]
                 for off in [eval_rng.randint(0, len(ALL_DATA) - 100) for _ in range(3)]]

    # 1. Baseline score (full network)
    print(f"\nComputing baseline score ({EVAL_TOKENS} tokens, {len(eval_seqs)} seqs)...")
    baseline_scores = []
    for seq in eval_seqs:
        s = eval_bigram_cosine_gpu(gpu, rows_t, cols_t, theta_t, channel_t, pol_t,
                                    seq, bp_in_t, bp_out_t, bigram_t, H, IN_DIM, OUT_DIM)
        baseline_scores.append(s)
    baseline = np.mean(baseline_scores)
    print(f"  Baseline bigram cosine: {baseline:.6f}")

    # 2. Test all edges in batches
    print(f"\nTesting {n_edges} edges in batches of {BATCH_SIZE}...")
    importance = np.zeros(n_edges, dtype=np.float32)
    t0 = time.time()

    for batch_start in range(0, n_edges, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, n_edges)
        batch_indices = list(range(batch_start, batch_end))

        # Average over eval sequences
        batch_scores = np.zeros(len(batch_indices))
        for seq in eval_seqs:
            scores = eval_batched_remove_edges(
                gpu, rows_t, cols_t, batch_indices,
                theta_t, channel_t, pol_t,
                seq, bp_in_t, bp_out_t, bigram_t, H, IN_DIM, OUT_DIM)
            batch_scores += scores
        batch_scores /= len(eval_seqs)

        importance[batch_start:batch_end] = baseline - batch_scores

        if (batch_start // BATCH_SIZE) % 5 == 0:
            elapsed = time.time() - t0
            pct = batch_end / n_edges * 100
            eta = elapsed / max(batch_end, 1) * (n_edges - batch_end)
            print(f"  [{batch_end}/{n_edges}] ({pct:.0f}%) {elapsed:.0f}s, ETA {eta:.0f}s")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s")

    # 3. Analysis
    print(f"\n{'='*60}")
    print(f"  EDGE IMPORTANCE DISTRIBUTION (H={H}, {n_edges} edges)")
    print(f"{'='*60}")
    print(f"  Baseline bigram cosine: {baseline:.6f}")
    print()
    print(f"  Importance stats:")
    print(f"    mean:   {importance.mean():.6f}")
    print(f"    std:    {importance.std():.6f}")
    print(f"    min:    {importance.min():.6f} (most harmful)")
    print(f"    max:    {importance.max():.6f} (most valuable)")
    print(f"    median: {np.median(importance):.6f}")
    print()

    # Histogram
    bins = [-0.01, -0.005, -0.001, -0.0001, 0, 0.0001, 0.001, 0.005, 0.01, 0.05, 0.1]
    counts, _ = np.histogram(importance, bins=bins)
    print(f"  Distribution:")
    for i in range(len(counts)):
        bar = '#' * min(int(counts[i] / max(n_edges, 1) * 200), 60)
        pct = counts[i] / n_edges * 100
        print(f"    [{bins[i]:+.4f}, {bins[i+1]:+.4f}): {counts[i]:5d} ({pct:5.1f}%) {bar}")

    # Categories
    harmful = (importance < -0.0001).sum()
    neutral = ((importance >= -0.0001) & (importance <= 0.0001)).sum()
    useful = (importance > 0.0001).sum()
    critical = (importance > 0.001).sum()

    print()
    print(f"  Categories:")
    print(f"    HARMFUL  (imp < -0.0001): {harmful:5d} ({harmful/n_edges*100:.1f}%) -- removing IMPROVES score")
    print(f"    NEUTRAL  (|imp| < 0.0001): {neutral:5d} ({neutral/n_edges*100:.1f}%) -- no effect")
    print(f"    USEFUL   (imp > 0.0001): {useful:5d} ({useful/n_edges*100:.1f}%) -- removing hurts")
    print(f"    CRITICAL (imp > 0.001):  {critical:5d} ({critical/n_edges*100:.1f}%) -- core edges")
    print()
    print(f"  Removable (harmful + neutral): {harmful + neutral} ({(harmful+neutral)/n_edges*100:.1f}%)")
    print(f"  If removed, est. new score: {baseline + abs(importance[importance < 0].sum()):.6f}")
    print(f"{'='*60}")

    # Save
    import json
    out_path = os.path.join(BASE_DIR, "data", f"edge_importance_h{H}.json")
    out = {
        'H': H, 'n_edges': n_edges, 'baseline': float(baseline),
        'mean': float(importance.mean()), 'std': float(importance.std()),
        'harmful': int(harmful), 'neutral': int(neutral),
        'useful': int(useful), 'critical': int(critical),
        'time_s': elapsed,
    }
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    # Also save raw importance array
    np.save(os.path.join(BASE_DIR, "data", f"edge_importance_h{H}.npy"), importance)
    print(f"  Saved to {out_path}")

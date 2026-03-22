"""
English Language Probe v2 — Pattern Encoding
=============================================
- I/O=32 or 64, but 256 bytes encoded as random patterns (not one-hot)
- Each byte gets a unique random 32/64-dim fingerprint
- Parallel candidates per step
- Training data: fineweb_edu.traindat (100MB English text)
"""

import sys, os, time, random, json
import numpy as np

from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "model"))
from graph import SelfWiringGraph


# ─── Pattern Encoding ─────────────────────────────────────

def make_byte_patterns(io_dim, seed=12345):
    """Create 256 unique random patterns of dimension io_dim.
    Each byte gets a sparse random fingerprint."""
    rng = np.random.RandomState(seed)
    # Random gaussian, then normalize to unit vectors
    patterns = rng.randn(256, io_dim).astype(np.float32)
    patterns /= np.linalg.norm(patterns, axis=1, keepdims=True)
    return patterns


# ─── Bigram from real text ────────────────────────────────

def load_bigram(path, max_bytes=1_000_000):
    with open(path, "rb") as f:
        data = f.read(max_bytes)
    counts = np.zeros((256, 256), dtype=np.float64)
    for i in range(len(data) - 1):
        counts[data[i], data[i+1]] += 1
    targets = np.argmax(counts, axis=1).astype(int)
    active = (counts.sum(axis=1) > 0).sum()
    total = counts.sum()
    correct = counts.max(axis=1).sum()
    ceiling = correct / total if total > 0 else 0
    return targets, active, ceiling


# ─── Evaluation ───────────────────────────────────────────

def evaluate(net, byte_patterns, targets, ticks=6):
    """Feed all 256 byte patterns through net, decode output via cosine sim."""
    H = net.H
    io_dim = byte_patterns.shape[1]

    charges = np.zeros((256, H), dtype=np.float32)
    acts = np.zeros((256, H), dtype=np.float32)
    projected = byte_patterns @ net.input_projection  # (256, H)
    retain = float(net.retention)

    for t in range(ticks):
        if t == 0:
            acts = acts + projected
        raw = acts @ net.mask
        np.nan_to_num(raw, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        charges += raw
        charges *= retain
        acts = np.maximum(charges - net.THRESHOLD, 0.0)
        charges = np.clip(charges, -1.0, 1.0)

    out = charges @ net.output_projection  # (256, io_dim)

    # Cosine similarity to all byte patterns
    out_norm = out / (np.linalg.norm(out, axis=1, keepdims=True) + 1e-8)
    pat_norm = byte_patterns / (np.linalg.norm(byte_patterns, axis=1, keepdims=True) + 1e-8)
    sim = out_norm @ pat_norm.T  # (256, 256)

    e = np.exp(sim - sim.max(axis=1, keepdims=True))
    probs = e / e.sum(axis=1, keepdims=True)

    # Score active bytes only
    active_mask = (targets != np.arange(256))
    active = np.where(active_mask)[0]
    if len(active) == 0:
        active = np.arange(256)

    acc = (np.argmax(probs, axis=1)[active] == targets[active]).mean()
    tp = probs[active, targets[active]].mean()
    return float(0.5 * acc + 0.5 * tp)


# ─── Parallel Growth ─────────────────────────────────────

def grow_parallel(net, ev_fn, budget, n_cand=16, verbose=True):
    score = ev_fn()
    best = score
    accepts = 0
    t0 = time.time()

    for step in range(1, budget + 1):
        mask_bak = net.mask.copy()
        alive_bak = list(net.alive)
        aset_bak = set(net.alive_set)

        best_score = score
        best_mask = None
        best_alive = None

        for c in range(n_cand):
            net.mask[:] = mask_bak
            net.alive = list(alive_bak)
            net.alive_set = set(aset_bak)

            undo = net.mutate(forced_op='add')
            if not undo:
                continue
            ns = ev_fn()
            if ns > best_score:
                best_score = ns
                best_mask = net.mask.copy()
                best_alive = list(net.alive)

        if best_mask is not None:
            net.mask[:] = best_mask
            net.alive = best_alive
            net.alive_set = set(best_alive)
            score = best_score
            best = max(best, score)
            accepts += 1
        else:
            net.mask[:] = mask_bak
            net.alive = alive_bak
            net.alive_set = aset_bak

        if verbose and step % 500 == 0:
            elapsed = time.time() - t0
            print(f"  [{step:6d}] score={score:.4f} best={best:.4f} "
                  f"edges={net.count_connections()} accepts={accepts} {elapsed:.0f}s")
            sys.stdout.flush()

    return best, accepts


# ─── Main ─────────────────────────────────────────────────

def main():
    from lib.data import resolve_fineweb_path
    DATA = resolve_fineweb_path()

    print(f"Data: {DATA}")
    targets, active_bytes, ceiling = load_bigram(DATA)
    print(f"  Active bytes: {active_bytes}/256, ceiling: {ceiling*100:.1f}%")

    configs = [
        # (label, io_dim, n_cand, budget)
        ("io32_1cand",   32,  1, 16000),
        ("io32_16cand",  32, 16, 16000),
        ("io64_1cand",   64,  1, 32000),
        ("io64_16cand",  64, 16, 32000),
    ]

    results = []

    for label, io_dim, n_cand, budget in configs:
        neurons = io_dim * 3
        byte_patterns = make_byte_patterns(io_dim)

        print(f"\n{'='*60}")
        print(f"  {label} | I/O={io_dim} neurons={neurons} cand={n_cand} budget={budget}")
        print(f"{'='*60}")

        random.seed(42); np.random.seed(42)
        net = SelfWiringGraph(io_dim)
        net.mask[:] = 0.0; net.alive = []; net.alive_set = set()
        net.state *= 0; net.charge *= 0

        ev_fn = lambda: evaluate(net, byte_patterns, targets)
        init = ev_fn()
        print(f"  Init: {init:.4f}")

        t0 = time.time()
        best, accepts = grow_parallel(net, ev_fn, budget, n_cand)

        edges_pre = net.count_connections()
        removed = net.crystallize(ev_fn)
        edges_post = net.count_connections()
        final = ev_fn()
        elapsed = time.time() - t0

        pct = final / ceiling * 100 if ceiling > 0 else 0
        print(f"  Final: {final:.4f} ({pct:.0f}% of ceiling) "
              f"edges={edges_pre}->{edges_post} crystal={removed} "
              f"accepts={accepts} {elapsed:.0f}s")

        results.append({
            "label": label, "io_dim": io_dim, "neurons": neurons,
            "n_cand": n_cand, "budget": budget,
            "final": round(final, 5), "ceiling": round(ceiling, 4),
            "pct_ceiling": round(pct, 1),
            "edges_pre": edges_pre, "edges_post": edges_post,
            "crystal": removed, "accepts": accepts,
            "elapsed": round(elapsed, 1),
        })

    print(f"\n{'='*60}")
    print(f"  SUMMARY (ceiling={ceiling*100:.1f}%)")
    print(f"{'='*60}")
    for r in results:
        print(f"  {r['label']:15s} | {r['final']:.4f} ({r['pct_ceiling']:.0f}%) "
              f"| {r['neurons']} neurons {r['edges_post']} edges | {r['elapsed']:.0f}s")

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "english_probe_v2_results.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()

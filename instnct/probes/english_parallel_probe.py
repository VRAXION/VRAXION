"""
English Language Probe — Binary Encoding + Parallel Candidates
===============================================================
- Binary I/O: each byte = 8 bits → I/O=8, neurons=24
- N candidates per step: try N random adds, keep the BEST
- Training data: fineweb_edu.traindat (100MB English text)
- Task: next-byte bigram prediction (given byte pattern → most likely next byte)
"""

import sys, os, time, random, json
import numpy as np

from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "model"))
from graph import SelfWiringGraph


# ─── Binary Encoding ──────────────────────────────────────

def byte_to_bits(b):
    """Byte (0-255) → 8-element float array."""
    return np.array([(b >> i) & 1 for i in range(8)], dtype=np.float32)

def bits_to_byte(bits):
    """8-element array → byte (0-255). Threshold at 0.5."""
    b = 0
    for i in range(8):
        if bits[i] > 0.0:
            b |= (1 << i)
    return b

BYTE_PATTERNS = np.array([byte_to_bits(b) for b in range(256)], dtype=np.float32)  # (256, 8)


# ─── Bigram from real text ────────────────────────────────

def load_bigram_from_file(path, max_bytes=1_000_000):
    """Load real text, compute bigram distribution."""
    with open(path, "rb") as f:
        data = f.read(max_bytes)

    # Count bigrams
    counts = np.zeros((256, 256), dtype=np.float64)
    for i in range(len(data) - 1):
        counts[data[i], data[i+1]] += 1

    # Most common next byte per byte
    targets = np.argmax(counts, axis=1).astype(int)

    # Coverage: how many bytes have meaningful bigrams
    active = (counts.sum(axis=1) > 0).sum()

    # Top bigram accuracy (theoretical ceiling with perfect prediction)
    total = counts.sum()
    correct = counts.max(axis=1).sum()
    ceiling = correct / total if total > 0 else 0

    return targets, active, ceiling, counts


# ─── Evaluation ───────────────────────────────────────────

def evaluate_binary(net, targets, ticks=6):
    """Evaluate bigram prediction using binary I/O.
    Feed each byte pattern, decode output, compare to target."""
    total_score = 0.0
    n = 0

    # Batch: feed all 256 byte patterns
    # Input: (256, 8) byte patterns
    V_io = 8  # binary I/O size
    H = net.H

    charges = np.zeros((256, H), dtype=np.float32)
    acts = np.zeros((256, H), dtype=np.float32)
    projected = BYTE_PATTERNS @ net.input_projection  # (256, H)
    retain = float(net.retention_mean)

    for t in range(ticks):
        if t == 0:
            acts = acts + projected
        raw = acts @ net.mask
        np.nan_to_num(raw, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        charges += raw
        charges *= retain
        acts = np.maximum(charges - net.theta_mean, 0.0)
        charges = np.clip(charges, -1.0, 1.0)

    # Output: charges @ output_projection → (256, 8) bit predictions
    out_bits = charges @ net.output_projection  # (256, 8)

    # Decode: for each input byte, what byte does the output encode?
    # Compare each output pattern to all 256 byte patterns
    # Use cosine similarity for soft matching
    out_norm = out_bits / (np.linalg.norm(out_bits, axis=1, keepdims=True) + 1e-8)
    pat_norm = BYTE_PATTERNS / (np.linalg.norm(BYTE_PATTERNS, axis=1, keepdims=True) + 1e-8)
    sim = out_norm @ pat_norm.T  # (256, 256) similarity matrix

    # Softmax scoring (like permutation eval)
    e = np.exp(sim - sim.max(axis=1, keepdims=True))
    probs = e / e.sum(axis=1, keepdims=True)

    predictions = np.argmax(probs, axis=1)

    # Score only bytes that have meaningful targets (appear in text)
    active = np.where(targets != np.arange(256))[0]  # bytes where target != self
    if len(active) == 0:
        active = np.arange(256)

    acc = (predictions[active] == targets[active]).mean()
    tp = probs[active, targets[active]].mean()

    return float(0.5 * acc + 0.5 * tp)


# ─── Parallel Growth ─────────────────────────────────────

def grow_parallel(net, ev_fn, budget, n_candidates=16, verbose=True):
    """Try n_candidates random adds per step, keep the best."""
    score = ev_fn()
    best = score
    accepts = 0
    t0 = time.time()

    for step in range(1, budget + 1):
        # Save current state
        mask_backup = net.mask.copy()
        alive_backup = list(net.alive)
        alive_set_backup = set(net.alive_set)

        best_score = score
        best_mask = None
        best_alive = None

        for c in range(n_candidates):
            # Restore to baseline
            net.mask[:] = mask_backup
            net.alive = list(alive_backup)
            net.alive_set = set(alive_set_backup)

            # Try random add
            undo = net.mutate(forced_op='add')
            if not undo:
                continue

            ns = ev_fn()
            if ns > best_score:
                best_score = ns
                best_mask = net.mask.copy()
                best_alive = list(net.alive)

        if best_mask is not None:
            # Apply the best candidate
            net.mask[:] = best_mask
            net.alive = best_alive
            net.alive_set = set(best_alive)
            score = best_score
            best = max(best, score)
            accepts += 1
        else:
            # No improvement, restore
            net.mask[:] = mask_backup
            net.alive = alive_backup
            net.alive_set = alive_set_backup

        if verbose and step % 500 == 0:
            elapsed = time.time() - t0
            edges = net.count_connections()
            print(f"  [{step:6d}] score={score:.4f} best={best:.4f} "
                  f"edges={edges} accepts={accepts} {elapsed:.0f}s")
            sys.stdout.flush()

    return best, accepts


# ─── Main ─────────────────────────────────────────────────

def main():
    from lib.data import resolve_fineweb_path
    DATA = resolve_fineweb_path()

    print(f"Loading bigrams from: {DATA}")
    targets, active_bytes, ceiling, counts = load_bigram_from_file(DATA)
    print(f"  Active bytes: {active_bytes}/256")
    print(f"  Bigram ceiling: {ceiling*100:.1f}% (best possible with argmax prediction)")
    print(f"  Top 5 bigrams: ", end="")
    flat = counts.flatten()
    top5 = np.argsort(flat)[-5:][::-1]
    for idx in top5:
        b1, b2 = idx // 256, idx % 256
        c1 = chr(b1) if 32 <= b1 < 127 else f"x{b1:02x}"
        c2 = chr(b2) if 32 <= b2 < 127 else f"x{b2:02x}"
        print(f"'{c1}{c2}'({int(flat[idx])})", end=" ")
    print()

    # Test configs
    configs = [
        # (label, V_io, n_candidates, budget)
        ("8bit_1cand",   8,  1,  8000),
        ("8bit_16cand",  8, 16,  8000),
    ]

    results = []

    for label, V_io, n_cand, budget in configs:
        print(f"\n{'='*55}")
        print(f"  {label} | I/O={V_io} neurons={V_io*3} candidates={n_cand} budget={budget}")
        print(f"{'='*55}")

        random.seed(42); np.random.seed(42)
        net = SelfWiringGraph(V_io)
        net.mask[:] = 0.0
        net.alive = []
        net.alive_set = set()
        net.state *= 0
        net.charge *= 0

        ev_fn = lambda: evaluate_binary(net, targets)

        t0 = time.time()
        init_score = ev_fn()
        print(f"  Init score: {init_score:.4f}")

        best, accepts = grow_parallel(net, ev_fn, budget, n_candidates=n_cand)

        # Crystal
        edges_pre = net.count_connections()
        removed = net.crystallize(ev_fn)
        edges_post = net.count_connections()
        final_score = ev_fn()
        elapsed = time.time() - t0

        print(f"  Final: score={final_score:.4f} edges={edges_pre}->{edges_post} "
              f"(crystal removed {removed}) accepts={accepts} {elapsed:.0f}s")

        results.append({
            "label": label, "V_io": V_io, "neurons": V_io * 3,
            "n_candidates": n_cand, "budget": budget,
            "best_score": round(best, 5), "final_score": round(final_score, 5),
            "edges_pre_crystal": edges_pre, "edges_post_crystal": edges_post,
            "crystal_removed": removed, "accepts": accepts,
            "elapsed": round(elapsed, 1),
            "ceiling": round(ceiling, 4),
        })

    print(f"\n{'='*55}")
    print(f"  SUMMARY")
    print(f"{'='*55}")
    for r in results:
        pct_of_ceiling = r['final_score'] / r['ceiling'] * 100 if r['ceiling'] > 0 else 0
        print(f"  {r['label']:15s} | score={r['final_score']:.4f} "
              f"({pct_of_ceiling:.0f}% of ceiling) "
              f"| edges={r['edges_post_crystal']} | {r['elapsed']:.0f}s")

    # Save
    out = os.path.join(os.path.dirname(__file__), "english_probe_results.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved: {out}")


if __name__ == "__main__":
    main()


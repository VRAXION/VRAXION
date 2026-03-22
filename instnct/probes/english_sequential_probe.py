"""
English Sequential Probe — Next Byte Prediction
=================================================
Feed bytes one by one, predict next byte from state.
Uses persistent charge/state across bytes (recurrent).
Pattern encoding: 256 bytes → io_dim random fingerprints.
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
    rng = np.random.RandomState(seed)
    patterns = rng.randn(256, io_dim).astype(np.float32)
    patterns /= np.linalg.norm(patterns, axis=1, keepdims=True)
    return patterns


# ─── Sequential Evaluation ────────────────────────────────

def evaluate_sequential(net, byte_patterns, text_bytes, ticks=6):
    """Feed bytes one by one. After each, check if output matches next byte.
    Returns accuracy (fraction of correctly predicted next-bytes)."""
    net.reset()
    correct = 0
    total = 0
    io_dim = byte_patterns.shape[1]

    # Precompute normalized patterns for cosine matching
    pat_norm = byte_patterns / (np.linalg.norm(byte_patterns, axis=1, keepdims=True) + 1e-8)

    for i in range(len(text_bytes) - 1):
        inp = byte_patterns[text_bytes[i]]  # (io_dim,)
        out = net.forward(inp, ticks=ticks)  # (io_dim,) — uses persistent state

        # Cosine sim to all byte patterns
        out_n = out / (np.linalg.norm(out) + 1e-8)
        sims = out_n @ pat_norm.T  # (256,)
        predicted = np.argmax(sims)

        if predicted == text_bytes[i + 1]:
            correct += 1
        total += 1

    return correct / total if total > 0 else 0.0


def evaluate_sequential_scored(net, byte_patterns, text_bytes, ticks=6):
    """Like above but returns blended score (acc + target_prob) for smoother signal."""
    net.reset()
    correct = 0
    total = 0
    prob_sum = 0.0
    io_dim = byte_patterns.shape[1]
    pat_norm = byte_patterns / (np.linalg.norm(byte_patterns, axis=1, keepdims=True) + 1e-8)

    for i in range(len(text_bytes) - 1):
        inp = byte_patterns[text_bytes[i]]
        out = net.forward(inp, ticks=ticks)

        out_n = out / (np.linalg.norm(out) + 1e-8)
        sims = out_n @ pat_norm.T
        # Softmax
        e = np.exp(sims - sims.max())
        probs = e / e.sum()

        target = text_bytes[i + 1]
        if np.argmax(probs) == target:
            correct += 1
        prob_sum += probs[target]
        total += 1

    acc = correct / total if total > 0 else 0.0
    avg_prob = prob_sum / total if total > 0 else 0.0
    return 0.5 * acc + 0.5 * avg_prob


# ─── Load text ────────────────────────────────────────────

def load_text(path, max_bytes=2000):
    with open(path, "rb") as f:
        data = f.read(max_bytes)
    return np.frombuffer(data, dtype=np.uint8)


# ─── Training ────────────────────────────────────────────

def train_sequential(net, byte_patterns, train_bytes, eval_bytes,
                     budget=8000, n_cand=1, ticks=6, verbose=True):
    """Grow network by evaluating on sequential text prediction."""

    def ev():
        return evaluate_sequential_scored(net, byte_patterns, train_bytes, ticks)

    score = ev()
    best = score
    accepts = 0
    t0 = time.time()

    for step in range(1, budget + 1):
        if n_cand == 1:
            undo = net.mutate(forced_op='add')
            if not undo:
                continue
            ns = ev()
            if ns > score:
                score = ns
                best = max(best, score)
                accepts += 1
            else:
                net.replay(undo)
        else:
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
                ns = ev()
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

        if verbose and step % 200 == 0:
            elapsed = time.time() - t0
            # Quick eval on eval set
            eval_acc = evaluate_sequential(net, byte_patterns, eval_bytes, ticks)
            print(f"  [{step:6d}] train={score:.4f} eval_acc={eval_acc*100:.1f}% "
                  f"edges={net.count_connections()} accepts={accepts} {elapsed:.0f}s")
            sys.stdout.flush()

    return best, accepts


# ─── Main ─────────────────────────────────────────────────

def main():
    from lib.data import resolve_fineweb_path
    DATA = resolve_fineweb_path()

    # Short sequences for tractable eval (sequential is slow per-byte)
    print(f"Data: {DATA}")
    raw = load_text(DATA, max_bytes=5000)
    train_bytes = raw[:500]   # train on first 500 bytes
    eval_bytes = raw[500:1000]  # eval on next 500 bytes
    print(f"  Train: {len(train_bytes)} bytes, Eval: {len(eval_bytes)} bytes")
    print(f"  Train text: {bytes(train_bytes[:80])}")

    # Random baseline
    print(f"  Random baseline: {1/256*100:.2f}% (1/256)")

    configs = [
        # (label, io_dim, n_cand, budget)
        ("io32_seq",   32,  1, 4000),
        ("io64_seq",   64,  1, 4000),
    ]

    results = []

    for label, io_dim, n_cand, budget in configs:
        neurons = io_dim * 3
        byte_patterns = make_byte_patterns(io_dim)

        print(f"\n{'='*60}")
        print(f"  {label} | {neurons} neurons, cand={n_cand}, budget={budget}")
        print(f"{'='*60}")

        random.seed(42); np.random.seed(42)
        net = SelfWiringGraph(io_dim)
        net.mask[:] = 0.0; net.alive = []; net.alive_set = set()
        net.state *= 0; net.charge *= 0

        init_acc = evaluate_sequential(net, byte_patterns, train_bytes)
        print(f"  Init accuracy: {init_acc*100:.1f}%")

        t0 = time.time()
        best, accepts = train_sequential(
            net, byte_patterns, train_bytes, eval_bytes,
            budget=budget, n_cand=n_cand)

        # Crystal
        ev_fn = lambda: evaluate_sequential_scored(net, byte_patterns, train_bytes)
        edges_pre = net.count_connections()
        removed = net.crystallize(ev_fn)
        edges_post = net.count_connections()

        final_train = evaluate_sequential(net, byte_patterns, train_bytes)
        final_eval = evaluate_sequential(net, byte_patterns, eval_bytes)
        elapsed = time.time() - t0

        print(f"\n  RESULT: train={final_train*100:.1f}% eval={final_eval*100:.1f}% "
              f"edges={edges_pre}->{edges_post} crystal={removed} "
              f"accepts={accepts} {elapsed:.0f}s")

        results.append({
            "label": label, "neurons": neurons,
            "train_acc": round(final_train * 100, 2),
            "eval_acc": round(final_eval * 100, 2),
            "edges_pre": edges_pre, "edges_post": edges_post,
            "crystal": removed, "accepts": accepts,
            "elapsed": round(elapsed, 1),
        })

    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    for r in results:
        print(f"  {r['label']:15s} | train={r['train_acc']:.1f}% eval={r['eval_acc']:.1f}% "
              f"| {r['neurons']} neurons {r['edges_post']} edges | {r['elapsed']:.0f}s")

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "english_seq_results.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()

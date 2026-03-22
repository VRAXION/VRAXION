"""
English Sequential v2 — Parallel Candidates + Short Eval
=========================================================
- 100 byte train/eval sequences (fast eval)
- 16 parallel candidates per step
- I/O=64 (192 neurons)
- Multiple train sequences for robustness
"""

import sys, os, time, random
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "model"))
from graph import SelfWiringGraph


def make_byte_patterns(io_dim, seed=12345):
    rng = np.random.RandomState(seed)
    patterns = rng.randn(256, io_dim).astype(np.float32)
    patterns /= np.linalg.norm(patterns, axis=1, keepdims=True)
    return patterns


def eval_seq(net, bp, text_bytes, ticks=6):
    """Sequential next-byte prediction. Returns blended score."""
    net.reset()
    pat_norm = bp / (np.linalg.norm(bp, axis=1, keepdims=True) + 1e-8)
    correct = 0
    prob_sum = 0.0
    total = 0

    for i in range(len(text_bytes) - 1):
        out = net.forward(bp[text_bytes[i]], ticks=ticks)
        out_n = out / (np.linalg.norm(out) + 1e-8)
        sims = out_n @ pat_norm.T
        e = np.exp(sims - sims.max())
        probs = e / e.sum()
        target = text_bytes[i + 1]
        if np.argmax(probs) == target:
            correct += 1
        prob_sum += probs[target]
        total += 1

    acc = correct / total if total > 0 else 0.0
    avg_p = prob_sum / total if total > 0 else 0.0
    return 0.5 * acc + 0.5 * avg_p


def eval_multi(net, bp, sequences, ticks=6):
    """Average score over multiple short sequences."""
    total = 0.0
    for seq in sequences:
        total += eval_seq(net, bp, seq, ticks)
    return total / len(sequences)


def eval_accuracy(net, bp, text_bytes, ticks=6):
    """Pure accuracy (no softmax blend)."""
    net.reset()
    pat_norm = bp / (np.linalg.norm(bp, axis=1, keepdims=True) + 1e-8)
    correct = 0
    total = 0
    for i in range(len(text_bytes) - 1):
        out = net.forward(bp[text_bytes[i]], ticks=ticks)
        out_n = out / (np.linalg.norm(out) + 1e-8)
        sims = out_n @ pat_norm.T
        if np.argmax(sims) == text_bytes[i + 1]:
            correct += 1
        total += 1
    return correct / total if total > 0 else 0.0


def main():
    from lib.data import load_fineweb_bytes, resolve_fineweb_path
    DATA = resolve_fineweb_path()
    raw = load_fineweb_bytes(max_bytes=10000)

    # 5 short train sequences, 2 eval sequences (non-overlapping)
    SEQ_LEN = 100
    train_seqs = [raw[i*SEQ_LEN:(i+1)*SEQ_LEN] for i in range(5)]
    eval_seqs = [raw[5*SEQ_LEN:6*SEQ_LEN], raw[6*SEQ_LEN:7*SEQ_LEN]]

    print(f"Data: {DATA}")
    print(f"  Train: {len(train_seqs)} sequences x {SEQ_LEN} bytes")
    print(f"  Eval:  {len(eval_seqs)} sequences x {SEQ_LEN} bytes")
    print(f"  Sample: {bytes(train_seqs[0][:60])}")

    IO = 64
    NEURONS = IO * 3
    N_CAND = 16
    BUDGET = 8000

    bp = make_byte_patterns(IO)

    print(f"\n  {NEURONS} neurons, {N_CAND} candidates/step, budget={BUDGET}")
    print(f"  Random baseline: {1/256*100:.2f}%")

    random.seed(42); np.random.seed(42)
    net = SelfWiringGraph(IO)
    net.mask[:] = 0.0; net.alive = []; net.alive_set = set()
    net.state *= 0; net.charge *= 0

    def ev():
        return eval_multi(net, bp, train_seqs)

    init = ev()
    init_acc = np.mean([eval_accuracy(net, bp, s) for s in eval_seqs])
    print(f"  Init: score={init:.4f} eval_acc={init_acc*100:.1f}%")

    score = init
    best = score
    accepts = 0
    t0 = time.time()

    for step in range(1, BUDGET + 1):
        mask_bak = net.mask.copy()
        alive_bak = list(net.alive)
        aset_bak = set(net.alive_set)

        best_score = score
        best_mask = None
        best_alive = None

        for c in range(N_CAND):
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

        if step % 100 == 0:
            elapsed = time.time() - t0
            ea = np.mean([eval_accuracy(net, bp, s) for s in eval_seqs])
            print(f"  [{step:5d}] train={score:.4f} eval={ea*100:.1f}% "
                  f"edges={net.count_connections()} acc={accepts} {elapsed:.0f}s")
            sys.stdout.flush()

    # Final eval
    final_eval = np.mean([eval_accuracy(net, bp, s) for s in eval_seqs])
    edges = net.count_connections()
    elapsed = time.time() - t0
    print(f"\n  FINAL: eval={final_eval*100:.1f}% edges={edges} accepts={accepts} {elapsed:.0f}s")

    # Crystal
    removed = net.crystallize(ev)
    post_eval = np.mean([eval_accuracy(net, bp, s) for s in eval_seqs])
    print(f"  CRYSTAL: eval={post_eval*100:.1f}% edges={net.count_connections()} removed={removed}")


if __name__ == "__main__":
    main()

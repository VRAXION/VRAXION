"""
English Sequential — TRUE Multiprocessing
==========================================
12 worker processes, each tries a random add + eval.
Master collects results, applies best.
Real parallel CPU usage.
"""

import sys, os, time, random
import numpy as np
from multiprocessing import Pool, cpu_count

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "model"))
from graph import SelfWiringGraph

# ─── Globals set once per worker ──────────────────────────

_worker_bp = None
_worker_seqs = None
_worker_ticks = 6


def init_worker(bp, seqs, ticks):
    global _worker_bp, _worker_seqs, _worker_ticks
    _worker_bp = bp
    _worker_seqs = seqs
    _worker_ticks = ticks


def make_byte_patterns(io_dim, seed=12345):
    rng = np.random.RandomState(seed)
    p = rng.randn(256, io_dim).astype(np.float32)
    p /= np.linalg.norm(p, axis=1, keepdims=True)
    return p


# ─── Eval (runs in worker) ───────────────────────────────

def eval_seq_single(mask, H, V, input_projection, output_projection, retention, threshold, text_bytes, bp, ticks):
    """Stateless sequential eval — no SWG object needed."""
    pat_norm = bp / (np.linalg.norm(bp, axis=1, keepdims=True) + 1e-8)
    state = np.zeros(H, dtype=np.float32)
    charge = np.zeros(H, dtype=np.float32)
    correct = 0
    prob_sum = 0.0
    total = 0

    for i in range(len(text_bytes) - 1):
        inp = bp[text_bytes[i]]
        act = state.copy()
        for t in range(ticks):
            if t == 0:
                act = act + inp @ input_projection
            raw = act @ mask
            charge += raw
            charge *= retention
            act = np.maximum(charge - threshold, 0.0)
            charge = np.clip(charge, -1.0, 1.0)
        state = act.copy()
        out = charge @ output_projection

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


def worker_try_add(args):
    """Worker: add random edge to mask copy, evaluate, return (score, r, c, val)."""
    mask_flat, H, V, input_projection, output_projection, retention, threshold, seed = args
    bp = _worker_bp
    seqs = _worker_seqs
    ticks = _worker_ticks

    rng = random.Random(seed)
    mask = mask_flat.reshape(H, H)

    # Random add
    r = rng.randint(0, H - 1)
    c = rng.randint(0, H - 1)
    if r == c or mask[r, c] != 0:
        return (-1e9, -1, -1, 0.0)

    val = 0.6 if rng.random() < 0.5 else -0.6
    new_mask = mask.copy()
    new_mask[r, c] = val

    # Eval on all train sequences
    total = 0.0
    for seq in seqs:
        total += eval_seq_single(new_mask, H, V, input_projection, output_projection, retention, threshold, seq, bp, ticks)
    score = total / len(seqs)

    return (score, r, c, val)


def eval_accuracy_stateless(mask, H, V, input_projection, output_projection, retention, threshold, text_bytes, bp, ticks=6):
    """Pure accuracy eval."""
    pat_norm = bp / (np.linalg.norm(bp, axis=1, keepdims=True) + 1e-8)
    state = np.zeros(H, dtype=np.float32)
    charge = np.zeros(H, dtype=np.float32)
    correct = 0
    total = 0
    for i in range(len(text_bytes) - 1):
        inp = bp[text_bytes[i]]
        act = state.copy()
        for t in range(ticks):
            if t == 0:
                act = act + inp @ input_projection
            raw = act @ mask
            charge += raw
            charge *= retention
            act = np.maximum(charge - threshold, 0.0)
            charge = np.clip(charge, -1.0, 1.0)
        state = act.copy()
        out = charge @ output_projection
        out_n = out / (np.linalg.norm(out) + 1e-8)
        sims = out_n @ pat_norm.T
        if np.argmax(sims) == text_bytes[i + 1]:
            correct += 1
        total += 1
    return correct / total if total > 0 else 0.0


# ─── Main ────────────────────────────────────────────────

def main():
    from lib.data import load_fineweb_bytes, resolve_fineweb_path
    DATA = resolve_fineweb_path()
    raw = load_fineweb_bytes(max_bytes=5000)

    SEQ_LEN = 80
    train_seqs = [raw[i*SEQ_LEN:(i+1)*SEQ_LEN] for i in range(3)]
    eval_seqs = [raw[4*SEQ_LEN:5*SEQ_LEN], raw[5*SEQ_LEN:6*SEQ_LEN]]

    IO = 64
    NEURONS = IO * 3
    N_WORKERS = min(12, cpu_count() - 2)
    BUDGET = 4000
    TICKS = 6

    bp = make_byte_patterns(IO)

    print(f"Data: {DATA}")
    print(f"  Train: {len(train_seqs)}x{SEQ_LEN} bytes, Eval: {len(eval_seqs)}x{SEQ_LEN} bytes")
    print(f"  {NEURONS} neurons, {N_WORKERS} workers (true parallel), budget={BUDGET}")
    print(f"  Sample: {bytes(train_seqs[0][:60])}")
    sys.stdout.flush()

    random.seed(42); np.random.seed(42)
    net = SelfWiringGraph(IO)
    net.mask[:] = 0.0; net.alive = []; net.alive_set = set()
    net.state *= 0; net.charge *= 0

    H = net.H
    V = net.V
    input_projection = net.input_projection
    output_projection = net.output_projection
    retention = float(net.retention)
    threshold = float(net.THRESHOLD)

    # Baseline score
    base_score = 0.0
    for seq in train_seqs:
        base_score += eval_seq_single(net.mask, H, V, input_projection, output_projection, retention, threshold, seq, bp, TICKS)
    base_score /= len(train_seqs)

    init_acc = np.mean([eval_accuracy_stateless(net.mask, H, V, input_projection, output_projection, retention, threshold, s, bp, TICKS) for s in eval_seqs])
    print(f"  Init: score={base_score:.4f} eval={init_acc*100:.1f}%")
    sys.stdout.flush()

    score = base_score
    best = score
    accepts = 0
    t0 = time.time()
    seed_counter = 1000

    pool = Pool(N_WORKERS, initializer=init_worker, initargs=(bp, train_seqs, TICKS))

    try:
        for step in range(1, BUDGET + 1):
            # Prepare args for workers
            mask_flat = net.mask.flatten()
            args = []
            for w in range(N_WORKERS):
                args.append((mask_flat, H, V, input_projection, output_projection, retention, threshold, seed_counter))
                seed_counter += 1

            # TRUE PARALLEL: all workers eval simultaneously
            results = pool.map(worker_try_add, args)

            # Pick best
            best_result = max(results, key=lambda x: x[0])
            if best_result[0] > score:
                _, r, c, val = best_result
                net.mask[r, c] = val
                net.alive.append((r, c))
                net.alive_set.add((r, c))
                score = best_result[0]
                best = max(best, score)
                accepts += 1

            if step % 50 == 0:
                elapsed = time.time() - t0
                ea = np.mean([eval_accuracy_stateless(net.mask, H, V, input_projection, output_projection, retention, threshold, s, bp, TICKS) for s in eval_seqs])
                rate = step / elapsed
                print(f"  [{step:5d}] train={score:.4f} eval={ea*100:.1f}% "
                      f"edges={net.count_connections()} acc={accepts} "
                      f"rate={rate:.1f}/s {elapsed:.0f}s")
                sys.stdout.flush()

    finally:
        pool.terminate()
        pool.join()

    # Final
    final_acc = np.mean([eval_accuracy_stateless(net.mask, H, V, input_projection, output_projection, retention, threshold, s, bp, TICKS) for s in eval_seqs])
    elapsed = time.time() - t0
    print(f"\n  FINAL: eval={final_acc*100:.1f}% edges={net.count_connections()} "
          f"accepts={accepts} {elapsed:.0f}s")

    # Crystal
    def ev_crystal():
        total = 0.0
        for seq in train_seqs:
            total += eval_seq_single(net.mask, H, V, input_projection, output_projection, retention, threshold, seq, bp, TICKS)
        return total / len(train_seqs)

    removed = net.crystallize(ev_crystal)
    post_acc = np.mean([eval_accuracy_stateless(net.mask, H, V, input_projection, output_projection, retention, threshold, s, bp, TICKS) for s in eval_seqs])
    print(f"  CRYSTAL: eval={post_acc*100:.1f}% edges={net.count_connections()} removed={removed}")


if __name__ == "__main__":
    main()

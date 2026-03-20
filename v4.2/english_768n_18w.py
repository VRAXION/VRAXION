"""
English 768 neurons, 18 workers, sparse forward
================================================
Full byte-range (256 I/O), pattern encoding, real English text.
"""
import sys, os, time, random
import numpy as np
from multiprocessing import Pool

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "model"))
from graph import SelfWiringGraph

_bp = None
_all_data = None
_seq_len = 200
_n_train = 5

def init_w(b, d, sl, nt):
    global _bp, _all_data, _seq_len, _n_train
    _bp, _all_data, _seq_len, _n_train = b, d, sl, nt

def make_bp(io_dim, seed=12345):
    rng = np.random.RandomState(seed)
    p = rng.randn(256, io_dim).astype(np.float32)
    p /= np.linalg.norm(p, axis=1, keepdims=True)
    return p

def _eval_mask_on_seqs(mask, H, W_in, W_out, retention, threshold, seqs):
    """Eval a mask on given sequences. Stateless."""
    rs, cs = np.where(mask != 0)
    sp_vals = mask[rs, cs]
    pat_norm = _bp / (np.linalg.norm(_bp, axis=1, keepdims=True) + 1e-8)
    total = 0.0
    for text_bytes in seqs:
        state = np.zeros(H, dtype=np.float32)
        charge = np.zeros(H, dtype=np.float32)
        correct = 0; prob_sum = 0.0; n = 0
        for i in range(len(text_bytes)-1):
            act = state.copy()
            for t in range(6):
                if t == 0:
                    act = act + _bp[text_bytes[i]] @ W_in
                raw = np.zeros(H, dtype=np.float32)
                if len(rs):
                    np.add.at(raw, cs, act[rs] * sp_vals)
                charge += raw; charge *= retention
                act = np.maximum(charge - threshold, 0.0)
                charge = np.clip(charge, -1.0, 1.0)
            state = act.copy()
            out = charge @ W_out
            out_n = out / (np.linalg.norm(out) + 1e-8)
            sims = out_n @ pat_norm.T
            e = np.exp(sims - sims.max())
            probs = e / e.sum()
            target = text_bytes[i+1]
            if np.argmax(probs) == target: correct += 1
            prob_sum += probs[target]; n += 1
        acc = correct/n if n else 0
        avg_p = prob_sum/n if n else 0
        total += 0.5*acc + 0.5*avg_p
    return total / len(seqs)

def worker_eval(args):
    mask_flat, H, W_in, W_out, retention, threshold, seed = args
    rng = random.Random(seed)
    np_rng = np.random.RandomState(seed)
    mask = mask_flat.reshape(H, H)
    r = rng.randint(0, H-1)
    c = rng.randint(0, H-1)
    if r == c or mask[r, c] != 0:
        return (-1e9, -1, -1, 0.0)
    val = 0.6 if rng.random() < 0.5 else -0.6
    new_mask = mask.copy()
    new_mask[r, c] = val

    # Sample random sequences (same for old and new mask = fair comparison)
    data_len = len(_all_data)
    seqs = []
    for _ in range(_n_train):
        off = np_rng.randint(0, data_len - _seq_len)
        seqs.append(_all_data[off:off+_seq_len])

    # Eval BOTH masks on SAME random sequences
    old_score = _eval_mask_on_seqs(mask, H, W_in, W_out, retention, threshold, seqs)
    new_score = _eval_mask_on_seqs(new_mask, H, W_in, W_out, retention, threshold, seqs)
    delta = new_score - old_score

    return (delta, r, c, val)

def eval_accuracy(mask, H, W_in, W_out, retention, threshold, text_bytes, bp, ticks=6):
    pat_norm = bp / (np.linalg.norm(bp, axis=1, keepdims=True) + 1e-8)
    rs, cs = np.where(mask != 0)
    sp_vals = mask[rs, cs]
    state = np.zeros(H, dtype=np.float32)
    charge = np.zeros(H, dtype=np.float32)
    correct = 0; total = 0
    for i in range(len(text_bytes)-1):
        act = state.copy()
        for t in range(ticks):
            if t == 0:
                act = act + bp[text_bytes[i]] @ W_in
            raw = np.zeros(H, dtype=np.float32)
            if len(rs):
                np.add.at(raw, cs, act[rs] * sp_vals)
            charge += raw; charge *= retention
            act = np.maximum(charge - threshold, 0.0)
            charge = np.clip(charge, -1.0, 1.0)
        state = act.copy()
        out = charge @ W_out
        out_n = out / (np.linalg.norm(out) + 1e-8)
        sims = out_n @ pat_norm.T
        if np.argmax(sims) == text_bytes[i+1]: correct += 1
        total += 1
    return correct/total if total else 0

if __name__ == "__main__":
    IO = 256; H = IO * 3; N_WORKERS = 18; BUDGET = 20000
    SEQ_LEN = 200; N_TRAIN_SEQS = 5; N_EVAL_SEQS = 3

    bp = make_bp(IO)
    DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "..", "Diamond Code", "data", "traindat", "fineweb_edu.traindat")
    with open(DATA, 'rb') as f:
        ALL_DATA = np.frombuffer(f.read(), dtype=np.uint8)
    DATA_LEN = len(ALL_DATA)
    print(f"Loaded {DATA_LEN / 1e6:.1f} MB text")

    # Fixed eval sequences (always the same for consistent reporting)
    eval_rng = np.random.RandomState(9999)
    eval_seqs = []
    for _ in range(N_EVAL_SEQS):
        off = eval_rng.randint(0, DATA_LEN - SEQ_LEN)
        eval_seqs.append(ALL_DATA[off:off+SEQ_LEN])

    print(f"{H} neurons, I/O={IO}, {N_WORKERS} workers, budget={BUDGET}")
    print(f"Train: {N_TRAIN_SEQS}x{SEQ_LEN} RANDOM per step from {DATA_LEN/1e6:.1f}MB | Eval: {N_EVAL_SEQS}x{SEQ_LEN} fixed")
    print(f"Sample: {bytes(ALL_DATA[:60])}")
    sys.stdout.flush()

    random.seed(42); np.random.seed(42)
    net = SelfWiringGraph(IO)

    # Resume from checkpoint if exists
    CKPT_RESUME = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "checkpoints", "english_768n_final.npz")
    if os.path.exists(CKPT_RESUME):
        d = np.load(CKPT_RESUME)
        net.mask[:] = 0
        net.mask[d['rows'], d['cols']] = d['vals']
        net.resync_alive()
        print(f"RESUMED from {CKPT_RESUME} ({net.count_connections()} edges)")
    else:
        net.mask[:]=0; net.alive=[]; net.alive_set=set(); net._sync_sparse_idx()
        print("Starting from empty network")

    net.state *= 0; net.charge *= 0
    W_in=net.W_in; W_out=net.W_out
    retention=float(net.retention); threshold=float(net.THRESHOLD)

    # Log file
    LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "english_768n_live.txt")
    with open(LOG, "a") as f:
        f.write(f"\n--- RESUMED {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")
        f.write(f"768n, {N_WORKERS}w, {N_TRAIN_SEQS}x{SEQ_LEN}b random train, budget={BUDGET}\n")

    CKPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
    os.makedirs(CKPT_DIR, exist_ok=True)

    score = 0.0
    best = 0.0
    accepts = 0
    seed_c = 1000
    t0 = time.time()

    pool = Pool(N_WORKERS, initializer=init_w, initargs=(bp, ALL_DATA, SEQ_LEN, N_TRAIN_SEQS))
    try:
        for step in range(1, BUDGET+1):
            mask_flat = net.mask.flatten()
            args = [(mask_flat, H, W_in, W_out, retention, threshold, seed_c+step*50+w)
                    for w in range(N_WORKERS)]
            results = pool.map(worker_eval, args)
            # Delta-based: accept best positive delta
            best_r = max(results, key=lambda x: x[0])
            if best_r[0] > 0 and best_r[1] >= 0:
                net.mask[best_r[1], best_r[2]] = best_r[3]
                net.alive.append((best_r[1], best_r[2]))
                net.alive_set.add((best_r[1], best_r[2]))
                net._sync_sparse_idx()
                accepts += 1

            if step % 50 == 0:
                elapsed = time.time() - t0
                ea = np.mean([eval_accuracy(net.mask, H, W_in, W_out, retention, threshold, s, bp)
                              for s in eval_seqs])
                edges = net.count_connections()
                line = (f"[{step:5d}] eval={ea*100:.1f}% "
                        f"edges={edges} accepts={accepts} {elapsed:.0f}s")
                print(f"  {line}")
                with open(LOG, "a") as f:
                    f.write(line + "\n")
                sys.stdout.flush()

            # Checkpoint every 500 steps
            if step % 500 == 0:
                ckpt = os.path.join(CKPT_DIR, f"english_768n_step{step}.npz")
                net.save(ckpt)
                print(f"  SAVED: {ckpt}")
                sys.stdout.flush()

    finally:
        pool.terminate(); pool.join()
        # Always save final state
        final_ckpt = os.path.join(CKPT_DIR, "english_768n_final.npz")
        net.save(final_ckpt)
        print(f"  SAVED FINAL: {final_ckpt}")

    elapsed = time.time() - t0
    final_ea = np.mean([eval_accuracy(net.mask, H, W_in, W_out, retention, threshold, s, bp)
                        for s in eval_seqs])
    print(f"\nFINAL: eval={final_ea*100:.1f}% edges={net.count_connections()} "
          f"accepts={accepts} {elapsed:.0f}s")

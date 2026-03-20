"""Sweep neuron count with ACTUAL parallel Pool training pattern.
Runs 20 steps at each size, measures real step/s with 18 workers.
"""
import sys, os, time, random
import numpy as np
from multiprocessing import Pool

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "model"))

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

def _eval_on_seqs(mask, H, W_in, W_out, theta, decay, seqs):
    rs, cs = np.where(mask != 0)
    sp_vals = mask[rs, cs]
    ret = 1.0 - decay
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
                charge += raw; charge *= ret
                act = np.maximum(charge - theta, 0.0)
                charge = np.clip(charge, -1.0, 1.0)
            state = act.copy()
            pat_norm = _bp / (np.linalg.norm(_bp, axis=1, keepdims=True) + 1e-8)
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
    mask_flat, theta, decay, H, W_in, W_out, seed, proposal_type = args
    rng = random.Random(seed)
    np_rng = np.random.RandomState(seed)
    mask = mask_flat.reshape(H, H)
    new_mask = mask
    new_theta = theta
    new_decay = decay

    if proposal_type == 'add':
        r = rng.randint(0, H-1)
        c = rng.randint(0, H-1)
        if r == c or mask[r, c] != 0:
            return {'delta': -1e9, 'type': 'add'}
        val = 0.6 if rng.random() < 0.5 else -0.6
        new_mask = mask.copy()
        new_mask[r, c] = val

    data_len = len(_all_data)
    seqs = []
    for _ in range(_n_train):
        off = np_rng.randint(0, data_len - _seq_len)
        seqs.append(_all_data[off:off+_seq_len])

    old_score = _eval_on_seqs(mask, H, W_in, W_out, theta, decay, seqs)
    new_score = _eval_on_seqs(new_mask, H, W_in, W_out, new_theta, new_decay, seqs)
    return {'delta': new_score - old_score, 'type': proposal_type}


if __name__ == "__main__":
    IO = 256
    N_WORKERS = 18
    N_STEPS = 20  # steps per config

    bp = make_bp(IO)

    DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "..", "Diamond Code", "data", "traindat", "fineweb_edu.traindat")
    with open(DATA, 'rb') as f:
        ALL_DATA = np.frombuffer(f.read(), dtype=np.uint8)

    print("=" * 65)
    print("Neuron Count Sweep (18 workers, Pool, 20 steps each)")
    print("=" * 65)
    print(f"{'Neurons':>8} {'H':>6} {'Edges':>7} {'step/s':>8} {'sec/step':>10} {'verdict':>10}")

    # Sweep: neurons = IO * multiplier
    for mult in [3, 4, 5, 6, 8, 10, 12, 16]:
        H = IO * mult
        neurons = H

        # Create network with some edges (simulate mid-training ~5K edges)
        np.random.seed(42)
        mask = np.zeros((H, H), dtype=np.float32)
        n_target_edges = min(5000, H * H // 10)
        placed = 0
        while placed < n_target_edges:
            r, c = np.random.randint(0, H), np.random.randint(0, H)
            if r != c and mask[r, c] == 0:
                mask[r, c] = 0.6 if np.random.random() < 0.5 else -0.6
                placed += 1

        theta = np.full(H, 0.1, dtype=np.float32)
        decay = np.full(H, 0.15, dtype=np.float32)

        proj_rng = np.random.RandomState(42)
        W_in = proj_rng.randn(IO, H).astype(np.float32)
        W_in /= np.linalg.norm(W_in, axis=0, keepdims=True)
        W_in *= 3.0
        W_out = proj_rng.randn(H, IO).astype(np.float32)
        W_out /= np.linalg.norm(W_out, axis=0, keepdims=True)
        W_out *= 3.0

        pool = Pool(N_WORKERS, initializer=init_w, initargs=(bp, ALL_DATA, 200, 5))

        # Warmup 1 step
        mask_flat = mask.flatten()
        args = [(mask_flat, theta.copy(), decay.copy(), H, W_in, W_out,
                 1000+w, 'add') for w in range(N_WORKERS)]
        pool.map(worker_eval, args)

        # Benchmark N_STEPS
        t0 = time.perf_counter()
        for step in range(N_STEPS):
            args = [(mask_flat, theta.copy(), decay.copy(), H, W_in, W_out,
                     2000+step*50+w, 'add') for w in range(N_WORKERS)]
            pool.map(worker_eval, args)
        elapsed = time.perf_counter() - t0

        pool.terminate()
        pool.join()

        sps = N_STEPS / elapsed
        sec_per = elapsed / N_STEPS

        if sps > 1.0:
            verdict = "FAST"
        elif sps > 0.5:
            verdict = "OK"
        elif sps > 0.2:
            verdict = "SLOW"
        else:
            verdict = "TOO SLOW"

        print(f"{neurons:>8} {H:>6} {n_target_edges:>7} {sps:>7.2f} {sec_per:>9.2f}s {verdict:>10}")
        sys.stdout.flush()

    print("\n" + "=" * 65)
    print("FAST >1 step/s | OK 0.5-1 | SLOW 0.2-0.5 | TOO SLOW <0.2")
    print("=" * 65)

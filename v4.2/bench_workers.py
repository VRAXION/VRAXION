"""Benchmark worker count scaling at 768 neurons."""
import sys, os, time, random
import numpy as np
from multiprocessing import Pool

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "model"))
from graph import SelfWiringGraph

_bp = None
_seqs = None

def init_w(b, s):
    global _bp, _seqs
    _bp, _seqs = b, s

def make_bp(io_dim, seed=12345):
    rng = np.random.RandomState(seed)
    p = rng.randn(256, io_dim).astype(np.float32)
    p /= np.linalg.norm(p, axis=1, keepdims=True)
    return p

def worker_eval(args):
    mask_flat, H, input_projection, output_projection, retention, threshold, seed = args
    rng = random.Random(seed)
    mask = mask_flat.reshape(H, H)
    r = rng.randint(0, H-1)
    c = rng.randint(0, H-1)
    if r == c or mask[r, c] != 0:
        return (-1e9, -1, -1, 0.0)
    val = 0.6 if rng.random() < 0.5 else -0.6
    new_mask = mask.copy()
    new_mask[r, c] = val
    rs, cs = np.where(new_mask != 0)
    sp_vals = new_mask[rs, cs]
    pat_norm = _bp / (np.linalg.norm(_bp, axis=1, keepdims=True) + 1e-8)
    total = 0.0
    for text_bytes in _seqs:
        state = np.zeros(H, dtype=np.float32)
        charge = np.zeros(H, dtype=np.float32)
        correct = 0; prob_sum = 0.0; n = 0
        for i in range(len(text_bytes)-1):
            act = state.copy()
            for t in range(6):
                if t == 0:
                    act = act + _bp[text_bytes[i]] @ input_projection
                raw = np.zeros(H, dtype=np.float32)
                if len(rs):
                    np.add.at(raw, cs, act[rs] * sp_vals)
                charge += raw; charge *= retention
                act = np.maximum(charge - threshold, 0.0)
                charge = np.clip(charge, -1.0, 1.0)
            state = act.copy()
            out = charge @ output_projection
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
    return (total/len(_seqs), r, c, val)

if __name__ == "__main__":
    IO = 256; H = IO * 3
    bp = make_bp(IO)
    from lib.data import load_fineweb_bytes, resolve_fineweb_path
    DATA = resolve_fineweb_path()
    raw = load_fineweb_bytes(max_bytes=3000)
    train_seqs = [raw[i*80:(i+1)*80] for i in range(3)]

    net = SelfWiringGraph(IO)
    net.mask[:]=0; net.alive=[]; net.alive_set=set(); net._sync_sparse_idx()
    input_projection=net.input_projection; output_projection=net.output_projection
    retention=float(net.retention); threshold=float(net.THRESHOLD)

    print(f'{H} neurons (I/O={IO}), 3x80 byte seqs')
    print(f'Workers | Steps | Time  | Steps/s | Candidates/s')
    print('-' * 55)
    sys.stdout.flush()

    STEPS = 30
    for n_workers in [4, 8, 12, 18, 22]:
        net.mask[:]=0; net.alive=[]; net.alive_set=set(); net._sync_sparse_idx()
        pool = Pool(n_workers, initializer=init_w, initargs=(bp, train_seqs))
        seed_c = 1000
        t0 = time.time()
        for step in range(STEPS):
            mask_flat = net.mask.flatten()
            args = [(mask_flat, H, input_projection, output_projection, retention, threshold, seed_c+step*50+w)
                    for w in range(n_workers)]
            results = pool.map(worker_eval, args)
            best = max(results, key=lambda x: x[0])
            if best[0] > -1e8 and best[1] >= 0:
                net.mask[best[1], best[2]] = best[3]
                net.alive.append((best[1], best[2]))
                net.alive_set.add((best[1], best[2]))
                net._sync_sparse_idx()
        dt = time.time() - t0
        pool.terminate(); pool.join()
        cand_s = STEPS * n_workers / dt
        print(f'  {n_workers:5d}   | {STEPS:5d} | {dt:5.1f}s | {STEPS/dt:5.1f}/s | {cand_s:.0f}/s')
        sys.stdout.flush()
    print('DONE')

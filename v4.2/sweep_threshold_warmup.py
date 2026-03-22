"""
Threshold Warmup + Bigram 2seq — best of both worlds?
======================================================
Bigram cosine 2seq (3.5x faster) + warmup threshold (loose→tight).
"""
import sys, os, time, random
import numpy as np
from multiprocessing import Pool

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "model"))
from graph import SelfWiringGraph

_bp = None; _all_data = None; _seq_len = 200; _n_train = 2
_input_projection = None; _output_projection = None; _bigram = None

def init_w(b, d, sl, nt, wi, wo, bg):
    global _bp, _all_data, _seq_len, _n_train, _input_projection, _output_projection, _bigram
    _bp, _all_data, _seq_len, _n_train = b, d, sl, nt
    _input_projection, _output_projection, _bigram = wi, wo, bg

def make_bp(io_dim, seed=12345):
    rng = np.random.RandomState(seed)
    p = rng.randn(256, io_dim).astype(np.float32)
    p /= np.linalg.norm(p, axis=1, keepdims=True)
    return p

def _eval_on_seqs(mask, H, theta, decay, seqs):
    rs, cs = np.where(mask != 0)
    sp_vals = mask[rs, cs]
    pat_norm = _bp / (np.linalg.norm(_bp, axis=1, keepdims=True) + 1e-8)
    ret = 1.0 - decay
    total = 0.0
    for text_bytes in seqs:
        state = np.zeros(H, dtype=np.float32)
        charge = np.zeros(H, dtype=np.float32)
        seq_score = 0.0; n = 0
        for i in range(len(text_bytes)-1):
            act = state.copy()
            for t in range(6):
                if t == 0:
                    act = act + _bp[text_bytes[i]] @ _input_projection
                raw = np.zeros(H, dtype=np.float32)
                if len(rs):
                    np.add.at(raw, cs, act[rs] * sp_vals)
                charge += raw; charge *= ret
                act = np.maximum(charge - theta, 0.0)
                charge = np.clip(charge, -1.0, 1.0)
            state = act.copy()
            out = charge @ _output_projection
            out_n = out / (np.linalg.norm(out) + 1e-8)
            sims = out_n @ pat_norm.T
            e = np.exp(sims - sims.max())
            pred = e / e.sum()
            target_dist = _bigram[text_bytes[i]]
            cos = np.dot(pred, target_dist) / (np.linalg.norm(pred) * np.linalg.norm(target_dist) + 1e-8)
            seq_score += cos
            n += 1
        total += seq_score / n if n else 0
    return total / len(seqs)

def worker_eval(args):
    mask_flat, theta, decay, H, seed, proposal_type = args
    rng = random.Random(seed)
    np_rng = np.random.RandomState(seed)
    mask = mask_flat.reshape(H, H)
    new_mask = mask; new_theta = theta

    if proposal_type == 'add':
        r = rng.randint(0, H-1); c = rng.randint(0, H-1)
        if r == c or mask[r, c] != 0:
            return {'delta': -1e9, 'type': 'add'}
        val = 0.6 if rng.random() < 0.5 else -0.6
        new_mask = mask.copy(); new_mask[r, c] = val
    elif proposal_type == 'flip':
        alive = list(zip(*np.where(mask != 0)))
        if not alive:
            return {'delta': -1e9, 'type': 'flip'}
        r, c = alive[rng.randint(0, len(alive)-1)]
        new_mask = mask.copy(); new_mask[r, c] = -mask[r, c]
    elif proposal_type == 'theta':
        idx = rng.randint(0, H-1)
        new_theta = theta.copy()
        new_theta[idx] = max(0.0, min(1.0, theta[idx] + rng.uniform(-0.05, 0.05)))

    seqs = []
    data_len = len(_all_data)
    for _ in range(_n_train):
        off = np_rng.randint(0, data_len - _seq_len)
        seqs.append(_all_data[off:off+_seq_len])

    old_score = _eval_on_seqs(mask, H, theta, decay, seqs)
    new_score = _eval_on_seqs(new_mask, H, new_theta, decay, seqs)

    return {'delta': new_score - old_score, 'type': proposal_type,
            'new_mask_flat': new_mask.flatten() if new_score > old_score else None,
            'new_theta': new_theta if proposal_type == 'theta' else None}

def eval_accuracy_classic(mask, H, input_projection, output_projection, theta, decay, text_bytes, bp):
    pat_norm = bp / (np.linalg.norm(bp, axis=1, keepdims=True) + 1e-8)
    rs, cs = np.where(mask != 0); sp_vals = mask[rs, cs]
    ret = 1.0 - decay
    state = np.zeros(H, dtype=np.float32); charge = np.zeros(H, dtype=np.float32)
    correct = 0; total = 0
    for i in range(len(text_bytes)-1):
        act = state.copy()
        for t in range(6):
            if t == 0: act = act + bp[text_bytes[i]] @ input_projection
            raw = np.zeros(H, dtype=np.float32)
            if len(rs): np.add.at(raw, cs, act[rs] * sp_vals)
            charge += raw; charge *= ret
            act = np.maximum(charge - theta, 0.0)
            charge = np.clip(charge, -1.0, 1.0)
        state = act.copy()
        out = charge @ output_projection
        out_n = out / (np.linalg.norm(out) + 1e-8)
        sims = out_n @ pat_norm.T
        if np.argmax(sims) == text_bytes[i+1]: correct += 1
        total += 1
    return correct/total if total else 0


def get_threshold(step, n_steps, schedule):
    """Return threshold at given step."""
    if schedule == 'fix_low':
        return 0.0001
    elif schedule == 'fix_mid':
        return 0.0005
    elif schedule == 'warmup_low_high':
        # 0.0001 → 0.001 linearly
        t = step / n_steps
        return 0.0001 + t * (0.001 - 0.0001)
    elif schedule == 'warmup_zero_mid':
        # 0.0 → 0.0005 linearly
        t = step / n_steps
        return t * 0.0005
    elif schedule == 'warmup_zero_high':
        # 0.0 → 0.001 linearly
        t = step / n_steps
        return t * 0.001
    return 0.0005


def run_config(name, schedule, bp, ALL_DATA, bigram, eval_seqs, H, input_projection, output_projection,
               n_steps=100, n_workers=18):
    mask = np.zeros((H, H), dtype=np.float32)
    theta = np.full(H, 0.03, dtype=np.float32)
    decay = np.full(H, 0.15, dtype=np.float32)

    print(f"\n--- {name} (schedule={schedule}) ---")
    sys.stdout.flush()

    sched = ['add', 'add', 'add', 'flip', 'theta', 'add']
    accepts = {'add': 0, 'flip': 0, 'theta': 0}
    t0 = time.time()

    pool = Pool(n_workers, initializer=init_w,
                initargs=(bp, ALL_DATA, 200, 2, input_projection, output_projection, bigram))
    try:
        for step in range(1, n_steps+1):
            ptype = sched[(step-1) % len(sched)]
            if ptype in ('flip', 'theta') and np.count_nonzero(mask) == 0:
                ptype = 'add'
            mask_flat = mask.flatten()
            args = [(mask_flat, theta.copy(), decay.copy(), H,
                     8000+step*50+w, ptype) for w in range(n_workers)]
            results = pool.map(worker_eval, args)

            thresh = get_threshold(step, n_steps, schedule)
            best_r = max(results, key=lambda x: x['delta'])
            if best_r['delta'] > thresh:
                if best_r['type'] in ('add', 'flip') and best_r['new_mask_flat'] is not None:
                    mask = best_r['new_mask_flat'].reshape(H, H)
                    accepts[best_r['type']] += 1
                elif best_r['type'] == 'theta' and best_r['new_theta'] is not None:
                    theta = best_r['new_theta']
                    accepts['theta'] += 1

            if step % 25 == 0:
                elapsed = time.time() - t0
                edges = int(np.count_nonzero(mask))
                ea = np.mean([eval_accuracy_classic(mask, H, input_projection, output_projection, theta, decay, s, bp)
                              for s in eval_seqs])
                tot = sum(accepts.values())
                cur_thresh = get_threshold(step, n_steps, schedule)
                print(f"  [{step:3d}] acc={ea*100:.2f}% edges={edges} "
                      f"accepts={tot} thresh={cur_thresh:.5f} "
                      f"{step/elapsed:.1f} step/s {elapsed:.0f}s")
                sys.stdout.flush()
    finally:
        pool.terminate(); pool.join()

    edges = int(np.count_nonzero(mask))
    ea = np.mean([eval_accuracy_classic(mask, H, input_projection, output_projection, theta, decay, s, bp)
                  for s in eval_seqs])
    elapsed = time.time() - t0
    print(f"  FINAL: acc={ea*100:.2f}% edges={edges} accepts={sum(accepts.values())} "
          f"{n_steps/elapsed:.1f} step/s {elapsed:.0f}s")
    sys.stdout.flush()
    return {'name': name, 'acc': ea, 'edges': edges, 'time': elapsed,
            'accepts': dict(accepts), 'sps': n_steps/elapsed}


if __name__ == "__main__":
    IO = 256; NV = 4; H = IO * NV
    SelfWiringGraph.NV_RATIO = NV
    bp = make_bp(IO)

    from lib.data import load_fineweb_bytes, resolve_fineweb_path
    DATA = resolve_fineweb_path()
    ALL_DATA = load_fineweb_bytes()
    print(f"Loaded {len(ALL_DATA)/1e6:.1f} MB text")

    bigram = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "data", "bigram_table.npy"))

    eval_rng = np.random.RandomState(9999)
    eval_seqs = [ALL_DATA[off:off+200] for off in
                 [eval_rng.randint(0, len(ALL_DATA)-200) for _ in range(10)]]

    random.seed(42); np.random.seed(42)
    proj_rng = np.random.RandomState(np.random.randint(0, 2**31))
    input_projection = proj_rng.randn(IO, H).astype(np.float32)
    input_projection /= np.linalg.norm(input_projection, axis=1, keepdims=True)
    output_projection = proj_rng.randn(H, IO).astype(np.float32)
    output_projection /= np.linalg.norm(output_projection, axis=0, keepdims=True)

    results = []

    # A: Fix threshold 0.0005 (previous baseline)
    results.append(run_config("FIX 0.0005", 'fix_mid',
                              bp, ALL_DATA, bigram, eval_seqs, H, input_projection, output_projection))

    # B: Fix very low (almost everything accepted)
    results.append(run_config("FIX 0.0001", 'fix_low',
                              bp, ALL_DATA, bigram, eval_seqs, H, input_projection, output_projection))

    # C: Warmup 0.0001 → 0.001
    results.append(run_config("WARM 0.0001->0.001", 'warmup_low_high',
                              bp, ALL_DATA, bigram, eval_seqs, H, input_projection, output_projection))

    # D: Warmup 0.0 → 0.0005
    results.append(run_config("WARM 0->0.0005", 'warmup_zero_mid',
                              bp, ALL_DATA, bigram, eval_seqs, H, input_projection, output_projection))

    # E: Warmup 0.0 → 0.001
    results.append(run_config("WARM 0->0.001", 'warmup_zero_high',
                              bp, ALL_DATA, bigram, eval_seqs, H, input_projection, output_projection))

    print(f"\n{'='*70}")
    print(f"  SUMMARY — THRESHOLD WARMUP + BIGRAM 2seq")
    print(f"{'='*70}")
    print(f"  {'Name':<22} {'Acc%':>6} {'Edges':>6} {'Accepts':>8} {'Step/s':>7} {'Time':>6}")
    print(f"  {'-'*22} {'-'*6} {'-'*6} {'-'*8} {'-'*7} {'-'*6}")
    for r in results:
        tot = sum(r['accepts'].values())
        print(f"  {r['name']:<22} {r['acc']*100:6.2f} {r['edges']:6d} {tot:8d} {r['sps']:7.1f} {r['time']:5.0f}s")
    sys.stdout.flush()

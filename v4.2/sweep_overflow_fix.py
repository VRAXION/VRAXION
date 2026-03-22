"""
Overflow Fix — nan_to_num vs soft cap vs baseline
===================================================
The overnight run had overflow warnings (charge explodes).
Test fixes: nan_to_num guard, soft cap, or both.
1000 steps, 18 workers, same config as overnight.
"""
import sys, os, time, random
import numpy as np
from multiprocessing import Pool

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "model"))
from graph import SelfWiringGraph

_bp = None; _all_data = None; _seq_len = 200; _n_train = 2
_W_out_f = None; _bigram = None; _inj_table = None
_overflow_mode = 'none'

def init_w(b, d, sl, nt, wof, bg, it, om):
    global _bp, _all_data, _seq_len, _n_train, _W_out_f, _bigram, _inj_table, _overflow_mode
    _bp, _all_data, _seq_len, _n_train = b, d, sl, nt
    _W_out_f, _bigram, _inj_table = wof, bg, it
    _overflow_mode = om

def make_bp(io_dim, seed=12345):
    rng = np.random.RandomState(seed)
    p = rng.randn(256, io_dim).astype(np.float32)
    p /= np.linalg.norm(p, axis=1, keepdims=True)
    return p

def _eval_bigram(msign, mmag, H, seqs):
    rs, cs = np.where(mmag > 0)
    s = msign[rs, cs].astype(np.float32) * 2 - 1
    sp_vals = s * mmag[rs, cs].astype(np.float32) / 128.0
    pat_norm = _bp / (np.linalg.norm(_bp, axis=1, keepdims=True) + 1e-8)
    ret = 217.0 / 256.0
    total = 0.0
    for text_bytes in seqs:
        state = np.zeros(H, dtype=np.float32)
        charge = np.zeros(H, dtype=np.float32)
        seq_score = 0.0; n = 0
        for i in range(len(text_bytes)-1):
            act = state.copy()
            for t in range(8):
                if t < 2:
                    act = act + _inj_table[text_bytes[i]].astype(np.float32) / 128.0
                raw = np.zeros(H, dtype=np.float32)
                if len(rs):
                    np.add.at(raw, cs, act[rs] * sp_vals)
                charge += raw; charge *= ret
                act = np.maximum(charge, 0.0)
                charge = np.maximum(charge, 0.0)
                # Overflow protection
                if _overflow_mode == 'nan_to_num':
                    np.nan_to_num(charge, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
                    np.nan_to_num(act, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
                elif _overflow_mode == 'soft_cap':
                    charge = np.minimum(charge, 10.0)
                    act = np.minimum(act, 10.0)
                elif _overflow_mode == 'both':
                    np.nan_to_num(charge, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
                    charge = np.minimum(charge, 10.0)
                    act = np.minimum(act, 10.0)
            state = act.copy()
            out = charge @ _W_out_f
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
    msign_flat, mmag_flat, H, seed, ptype = args
    rng = random.Random(seed)
    np_rng = np.random.RandomState(seed)
    msign = msign_flat.reshape(H, H); mmag = mmag_flat.reshape(H, H)
    new_s = msign.copy(); new_m = mmag.copy()

    if ptype == 'add':
        r = rng.randint(0, H-1); c = rng.randint(0, H-1)
        if r == c or mmag[r, c] > 0: return {'delta': -1e9, 'type': 'add'}
        new_s[r, c] = rng.random() < 0.5
        new_m[r, c] = rng.randint(1, 255)
    elif ptype == 'flip':
        rs, cs = np.where(mmag > 0)
        if len(rs) == 0: return {'delta': -1e9, 'type': 'flip'}
        idx = rng.randint(0, len(rs)-1)
        new_s[rs[idx], cs[idx]] = not msign[rs[idx], cs[idx]]
    elif ptype == 'mag_resample':
        rs, cs = np.where(mmag > 0)
        if len(rs) == 0: return {'delta': -1e9, 'type': 'mag_resample'}
        idx = rng.randint(0, len(rs)-1)
        new_m[rs[idx], cs[idx]] = rng.randint(1, 255)

    seqs = []
    for _ in range(_n_train):
        off = np_rng.randint(0, len(_all_data) - _seq_len)
        seqs.append(_all_data[off:off+_seq_len])

    old = _eval_bigram(msign, mmag, H, seqs)
    new = _eval_bigram(new_s, new_m, H, seqs)
    return {'delta': new - old, 'type': ptype,
            'new_s': new_s.flatten() if new > old else None,
            'new_m': new_m.flatten() if new > old else None}

def eval_accuracy(msign, mmag, H, W_out_f, text_bytes, bp, inj_table, overflow_mode):
    pat_norm = bp / (np.linalg.norm(bp, axis=1, keepdims=True) + 1e-8)
    rs, cs = np.where(mmag > 0)
    s = msign[rs, cs].astype(np.float32) * 2 - 1
    sp_vals = s * mmag[rs, cs].astype(np.float32) / 128.0
    ret = 217.0 / 256.0
    state = np.zeros(H, dtype=np.float32); charge = np.zeros(H, dtype=np.float32)
    correct = 0; total = 0
    for i in range(len(text_bytes)-1):
        act = state.copy()
        for t in range(8):
            if t < 2: act = act + inj_table[text_bytes[i]].astype(np.float32) / 128.0
            raw = np.zeros(H, dtype=np.float32)
            if len(rs): np.add.at(raw, cs, act[rs] * sp_vals)
            charge += raw; charge *= ret
            act = np.maximum(charge, 0.0); charge = np.maximum(charge, 0.0)
            if overflow_mode in ('nan_to_num', 'both'):
                np.nan_to_num(charge, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
                np.nan_to_num(act, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
            if overflow_mode in ('soft_cap', 'both'):
                charge = np.minimum(charge, 10.0); act = np.minimum(act, 10.0)
        state = act.copy()
        out = charge @ W_out_f
        out_n = out / (np.linalg.norm(out) + 1e-8)
        sims = out_n @ pat_norm.T
        if np.argmax(sims) == text_bytes[i+1]: correct += 1
        total += 1
    return correct/total if total else 0


def run_config(name, overflow_mode, bp, ALL_DATA, bigram, eval_seqs, H, W_out_f, inj_table,
               max_steps=1000, n_workers=18, threshold=0.00005):
    msign = np.zeros((H, H), dtype=np.bool_)
    mmag = np.zeros((H, H), dtype=np.uint8)
    schedule = ['add', 'add', 'flip', 'mag_resample', 'add', 'add']

    print(f"\n--- {name} (overflow={overflow_mode}) ---")
    sys.stdout.flush()

    accepts = {'add': 0, 'flip': 0, 'mag_resample': 0}
    t0 = time.time()

    pool = Pool(n_workers, initializer=init_w,
                initargs=(bp, ALL_DATA, 200, 2, W_out_f, bigram, inj_table, overflow_mode))
    try:
        for step in range(1, max_steps+1):
            ptype = schedule[(step-1) % len(schedule)]
            if ptype in ('flip', 'mag_resample') and (mmag > 0).sum() == 0: ptype = 'add'

            args = [(msign.flatten(), mmag.flatten(), H,
                     42000+step*50+w, ptype) for w in range(n_workers)]
            results = pool.map(worker_eval, args)

            best = max(results, key=lambda x: x['delta'])
            if best['delta'] > threshold and best.get('new_s') is not None:
                msign = best['new_s'].reshape(H, H)
                mmag = best['new_m'].reshape(H, H)
                accepts[best['type']] += 1

            if step % 200 == 0:
                elapsed = time.time() - t0
                edges = int((mmag > 0).sum())
                ea = np.mean([eval_accuracy(msign, mmag, H, W_out_f, s, bp, inj_table, overflow_mode)
                              for s in eval_seqs])
                print(f"  [{step:4d}] acc={ea*100:.2f}% edges={edges} "
                      f"A={accepts['add']}|F={accepts['flip']}|M={accepts['mag_resample']} {elapsed:.0f}s")
                sys.stdout.flush()
    finally:
        pool.terminate(); pool.join()

    edges = int((mmag > 0).sum())
    ea = np.mean([eval_accuracy(msign, mmag, H, W_out_f, s, bp, inj_table, overflow_mode)
                  for s in eval_seqs])
    elapsed = time.time() - t0
    print(f"  FINAL: acc={ea*100:.2f}% edges={edges} {elapsed:.0f}s")
    sys.stdout.flush()
    return {'name': name, 'acc': ea, 'edges': edges, 'time': elapsed}


if __name__ == "__main__":
    IO = 256; NV = 4; H = IO * NV
    SelfWiringGraph.NV_RATIO = NV
    bp = make_bp(IO)

    DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "..", "Diamond Code", "data", "traindat", "fineweb_edu.traindat")
    with open(DATA, 'rb') as f:
        ALL_DATA = np.frombuffer(f.read(), dtype=np.uint8)
    print(f"Loaded {len(ALL_DATA)/1e6:.1f} MB text")

    bigram = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "data", "bigram_table.npy"))

    eval_rng = np.random.RandomState(9999)
    eval_seqs = [ALL_DATA[off:off+200] for off in
                 [eval_rng.randint(0, len(ALL_DATA)-200) for _ in range(10)]]

    random.seed(42); np.random.seed(42)
    ref = SelfWiringGraph(IO)
    W_in = ref.W_in / ref.INJ_SCALE * 1.0
    W_out = ref.W_out / ref.INJ_SCALE * 1.0

    inj_table = np.clip(bp @ W_in * 128, -128, 127).astype(np.int8)
    W_out_int8 = np.clip(W_out * 128, -128, 127).astype(np.int8)
    W_out_f = W_out_int8.astype(np.float32) / 128.0

    results = []

    results.append(run_config("NO FIX (baseline)", 'none',
                              bp, ALL_DATA, bigram, eval_seqs, H, W_out_f, inj_table))
    results.append(run_config("NAN_TO_NUM", 'nan_to_num',
                              bp, ALL_DATA, bigram, eval_seqs, H, W_out_f, inj_table))
    results.append(run_config("SOFT CAP 10", 'soft_cap',
                              bp, ALL_DATA, bigram, eval_seqs, H, W_out_f, inj_table))
    results.append(run_config("BOTH", 'both',
                              bp, ALL_DATA, bigram, eval_seqs, H, W_out_f, inj_table))

    print(f"\n{'='*55}")
    print(f"  OVERFLOW FIX (1000 steps)")
    print(f"{'='*55}")
    for r in results:
        print(f"  {r['name']:<20} {r['acc']*100:6.2f}% {r['edges']} edges {r['time']:.0f}s")
    sys.stdout.flush()

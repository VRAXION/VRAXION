"""
Threshold=0 test — accept ALL positive deltas, then crystal
"""
import sys, os, time, random
import numpy as np
from multiprocessing import Pool

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "model"))
from graph import SelfWiringGraph

_bp = None; _all_data = None; _seq_len = 200; _n_train = 2
_W_out_f = None; _bigram = None; _inj_table = None

def init_w(b, d, sl, nt, wof, bg, it):
    global _bp, _all_data, _seq_len, _n_train, _W_out_f, _bigram, _inj_table
    _bp, _all_data, _seq_len, _n_train = b, d, sl, nt
    _W_out_f, _bigram, _inj_table = wof, bg, it

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
                if len(rs): np.add.at(raw, cs, act[rs] * sp_vals)
                charge += raw; charge *= ret
                act = np.maximum(charge, 0.0); charge = np.maximum(charge, 0.0)
            state = act.copy()
            out = charge @ _W_out_f
            out_n = out / (np.linalg.norm(out) + 1e-8)
            sims = out_n @ pat_norm.T
            e = np.exp(sims - sims.max())
            pred = e / e.sum()
            target_dist = _bigram[text_bytes[i]]
            cos = np.dot(pred, target_dist) / (np.linalg.norm(pred) * np.linalg.norm(target_dist) + 1e-8)
            seq_score += cos; n += 1
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
    seqs = []
    for _ in range(_n_train):
        off = np_rng.randint(0, len(_all_data) - _seq_len)
        seqs.append(_all_data[off:off+_seq_len])
    old = _eval_bigram(msign, mmag, H, seqs)
    new = _eval_bigram(new_s, new_m, H, seqs)
    return {'delta': new - old, 'type': ptype,
            'new_s': new_s.flatten() if new > old else None,
            'new_m': new_m.flatten() if new > old else None}

def eval_acc(msign, mmag, H, W_out_f, eval_seqs, bp, inj_table):
    pat_norm = bp / (np.linalg.norm(bp, axis=1, keepdims=True) + 1e-8)
    rs, cs = np.where(mmag > 0)
    s = msign[rs, cs].astype(np.float32) * 2 - 1
    sp_vals = s * mmag[rs, cs].astype(np.float32) / 128.0
    ret = 217.0 / 256.0
    correct = 0; total = 0
    for text_bytes in eval_seqs:
        state = np.zeros(H, dtype=np.float32); charge = np.zeros(H, dtype=np.float32)
        for i in range(len(text_bytes)-1):
            act = state.copy()
            for t in range(8):
                if t < 2: act = act + inj_table[text_bytes[i]].astype(np.float32) / 128.0
                raw = np.zeros(H, dtype=np.float32)
                if len(rs): np.add.at(raw, cs, act[rs] * sp_vals)
                charge += raw; charge *= ret
                act = np.maximum(charge, 0.0); charge = np.maximum(charge, 0.0)
            state = act.copy()
            out = charge @ W_out_f
            out_n = out / (np.linalg.norm(out) + 1e-8)
            sims = out_n @ pat_norm.T
            if np.argmax(sims) == text_bytes[i+1]: correct += 1
            total += 1
    return correct / total

if __name__ == "__main__":
    IO = 256; NV = 4; H = IO * NV
    bp = make_bp(IO)
    DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "..", "Diamond Code", "data", "traindat", "fineweb_edu.traindat")
    with open(DATA, 'rb') as f:
        ALL_DATA = np.frombuffer(f.read(), dtype=np.uint8)
    bigram = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "data", "bigram_table.npy"))
    eval_rng = np.random.RandomState(9999)
    eval_seqs = [ALL_DATA[off:off+200] for off in
                 [eval_rng.randint(0, len(ALL_DATA)-200) for _ in range(10)]]
    random.seed(42); np.random.seed(42)
    ref = SelfWiringGraph(IO, hidden_ratio=NV)
    W_in = ref.W_in / ref.INJ_SCALE * 1.0
    W_out = ref.W_out / ref.INJ_SCALE * 1.0
    inj_table = np.clip(bp @ W_in * 128, -128, 127).astype(np.int8)
    W_out_int8 = np.clip(W_out * 128, -128, 127).astype(np.int8)
    W_out_f = W_out_int8.astype(np.float32) / 128.0

    print(f"Loaded {len(ALL_DATA)/1e6:.1f} MB text")
    print(f"THRESHOLD=0 — accept ALL positive deltas")

    msign = np.zeros((H, H), dtype=np.bool_)
    mmag = np.zeros((H, H), dtype=np.uint8)
    schedule = ['add', 'add', 'flip', 'add', 'add', 'add']
    accepts = {'add': 0, 'flip': 0}
    t0 = time.time()

    pool = Pool(18, initializer=init_w,
                initargs=(bp, ALL_DATA, 200, 2, W_out_f, bigram, inj_table))
    try:
        for step in range(1, 1001):
            ptype = schedule[(step-1) % len(schedule)]
            if ptype == 'flip' and (mmag > 0).sum() == 0: ptype = 'add'
            args = [(msign.flatten(), mmag.flatten(), H,
                     45000+step*50+w, ptype) for w in range(18)]
            results = pool.map(worker_eval, args)
            best = max(results, key=lambda x: x['delta'])
            if best['delta'] > 0 and best.get('new_s') is not None:
                msign = best['new_s'].reshape(H, H)
                mmag = best['new_m'].reshape(H, H)
                accepts[best['type']] += 1
            if step % 100 == 0:
                elapsed = time.time() - t0
                edges = int((mmag > 0).sum())
                ea = eval_acc(msign, mmag, H, W_out_f, eval_seqs, bp, inj_table)
                print(f"  [{step:4d}] acc={ea*100:.2f}% edges={edges} "
                      f"A={accepts['add']}|F={accepts['flip']} {elapsed:.0f}s")
                sys.stdout.flush()
    finally:
        pool.terminate(); pool.join()

    edges = int((mmag > 0).sum())
    ea = eval_acc(msign, mmag, H, W_out_f, eval_seqs, bp, inj_table)
    elapsed = time.time() - t0
    print(f"\nPRE-CRYSTAL: acc={ea*100:.2f}% edges={edges} "
          f"A={accepts['add']}|F={accepts['flip']} {elapsed:.0f}s")
    np.savez_compressed(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "checkpoints", "thresh0_pre_crystal.npz"),
        msign=msign, mmag=mmag, inj_table=inj_table, W_out_int8=W_out_int8)
    print("Saved pre-crystal checkpoint")


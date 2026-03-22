"""
Bool Mask Test — verify bool mask == float mask
=================================================
Same network, same eval, same mutations.
A: Float mask (current), B: Bool mask (2 arrays).
500 steps, must produce IDENTICAL results.
"""
import sys, os, time, random
import numpy as np
from multiprocessing import Pool

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "model"))
from graph import SelfWiringGraph

_bp = None; _all_data = None; _seq_len = 200; _n_train = 2
_W_in = None; _W_out = None; _bigram = None
_use_bool = False
DRIVE = 0.6

def init_w(b, d, sl, nt, wi, wo, bg, ub):
    global _bp, _all_data, _seq_len, _n_train, _W_in, _W_out, _bigram, _use_bool
    _bp, _all_data, _seq_len, _n_train = b, d, sl, nt
    _W_in, _W_out, _bigram = wi, wo, bg
    _use_bool = ub

def make_bp(io_dim, seed=12345):
    rng = np.random.RandomState(seed)
    p = rng.randn(256, io_dim).astype(np.float32)
    p /= np.linalg.norm(p, axis=1, keepdims=True)
    return p

def _eval_bigram_float(mask, H, ret, seqs):
    rs, cs = np.where(mask != 0)
    sp_vals = mask[rs, cs]
    pat_norm = _bp / (np.linalg.norm(_bp, axis=1, keepdims=True) + 1e-8)
    total = 0.0
    for text_bytes in seqs:
        state = np.zeros(H, dtype=np.float32)
        charge = np.zeros(H, dtype=np.float32)
        seq_score = 0.0; n = 0
        for i in range(len(text_bytes)-1):
            act = state.copy()
            injection = _bp[text_bytes[i]] @ _W_in
            for t in range(8):
                if t < 2: act = act + injection
                raw = np.zeros(H, dtype=np.float32)
                if len(rs): np.add.at(raw, cs, act[rs] * sp_vals)
                charge += raw; charge *= ret
                act = np.maximum(charge, 0.0)
                charge = np.maximum(charge, 0.0)
            state = act.copy()
            out = charge @ _W_out
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

def _eval_bigram_bool(exists, sign, H, ret, seqs):
    rs, cs = np.where(exists)
    sp_signs = sign[rs, cs].astype(np.float32) * 2 - 1  # bool -> {-1, +1}
    sp_vals = sp_signs * DRIVE  # {-0.6, +0.6}
    pat_norm = _bp / (np.linalg.norm(_bp, axis=1, keepdims=True) + 1e-8)
    total = 0.0
    for text_bytes in seqs:
        state = np.zeros(H, dtype=np.float32)
        charge = np.zeros(H, dtype=np.float32)
        seq_score = 0.0; n = 0
        for i in range(len(text_bytes)-1):
            act = state.copy()
            injection = _bp[text_bytes[i]] @ _W_in
            for t in range(8):
                if t < 2: act = act + injection
                raw = np.zeros(H, dtype=np.float32)
                if len(rs): np.add.at(raw, cs, act[rs] * sp_vals)
                charge += raw; charge *= ret
                act = np.maximum(charge, 0.0)
                charge = np.maximum(charge, 0.0)
            state = act.copy()
            out = charge @ _W_out
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
    if _use_bool:
        exists_flat, sign_flat, ret, H, seed, proposal_type = args
        exists = exists_flat.reshape(H, H)
        sign = sign_flat.reshape(H, H)
    else:
        mask_flat, ret, H, seed, proposal_type = args
        mask = mask_flat.reshape(H, H)

    rng = random.Random(seed)
    np_rng = np.random.RandomState(seed)

    if _use_bool:
        new_exists = exists.copy(); new_sign = sign.copy()
        new_ret = ret.copy()

        if proposal_type == 'add':
            r = rng.randint(0, H-1); c = rng.randint(0, H-1)
            if r == c or exists[r, c]:
                return {'delta': -1e9, 'type': 'add'}
            new_exists[r, c] = True
            new_sign[r, c] = rng.random() < 0.5  # True=+, False=-
        elif proposal_type == 'flip':
            rs, cs = np.where(exists)
            if len(rs) == 0:
                return {'delta': -1e9, 'type': 'flip'}
            idx = rng.randint(0, len(rs)-1)
            r, c = rs[idx], cs[idx]
            new_sign[r, c] = not sign[r, c]
        elif proposal_type == 'decay':
            idx = rng.randint(0, H-1)
            new_ret = ret.copy()
            new_ret[idx] = rng.randint(0, 256)

        seqs = []
        data_len = len(_all_data)
        for _ in range(_n_train):
            off = np_rng.randint(0, data_len - _seq_len)
            seqs.append(_all_data[off:off+_seq_len])

        ret_f = ret.astype(np.float32) / 256.0
        new_ret_f = new_ret.astype(np.float32) / 256.0
        old_score = _eval_bigram_bool(exists, sign, H, ret_f, seqs)
        new_score = _eval_bigram_bool(new_exists, new_sign, H, new_ret_f, seqs)

        return {'delta': new_score - old_score, 'type': proposal_type,
                'new_exists_flat': new_exists.flatten() if new_score > old_score else None,
                'new_sign_flat': new_sign.flatten() if new_score > old_score else None,
                'new_ret': new_ret if new_score > old_score else None}
    else:
        new_mask = mask.copy(); new_ret = ret.copy()

        if proposal_type == 'add':
            r = rng.randint(0, H-1); c = rng.randint(0, H-1)
            if r == c or mask[r, c] != 0:
                return {'delta': -1e9, 'type': 'add'}
            val = DRIVE if rng.random() < 0.5 else -DRIVE
            new_mask[r, c] = val
        elif proposal_type == 'flip':
            alive = list(zip(*np.where(mask != 0)))
            if not alive:
                return {'delta': -1e9, 'type': 'flip'}
            r, c = alive[rng.randint(0, len(alive)-1)]
            new_mask[r, c] = -mask[r, c]
        elif proposal_type == 'decay':
            idx = rng.randint(0, H-1)
            new_ret = ret.copy()
            new_ret[idx] = rng.randint(0, 256)

        seqs = []
        data_len = len(_all_data)
        for _ in range(_n_train):
            off = np_rng.randint(0, data_len - _seq_len)
            seqs.append(_all_data[off:off+_seq_len])

        ret_f = ret.astype(np.float32) / 256.0
        new_ret_f = new_ret.astype(np.float32) / 256.0
        old_score = _eval_bigram_float(mask, H, ret_f, seqs)
        new_score = _eval_bigram_float(new_mask, H, new_ret_f, seqs)

        return {'delta': new_score - old_score, 'type': proposal_type,
                'new_mask_flat': new_mask.flatten() if new_score > old_score else None,
                'new_ret': new_ret if new_score > old_score else None}

def eval_accuracy(exists, sign, mask, H, W_in, W_out, ret_int, use_bool, text_bytes, bp):
    pat_norm = bp / (np.linalg.norm(bp, axis=1, keepdims=True) + 1e-8)
    if use_bool:
        rs, cs = np.where(exists)
        sp_vals = (sign[rs, cs].astype(np.float32) * 2 - 1) * DRIVE
    else:
        rs, cs = np.where(mask != 0)
        sp_vals = mask[rs, cs]
    ret = ret_int.astype(np.float32) / 256.0
    state = np.zeros(H, dtype=np.float32); charge = np.zeros(H, dtype=np.float32)
    correct = 0; total = 0
    for i in range(len(text_bytes)-1):
        act = state.copy()
        injection = bp[text_bytes[i]] @ W_in
        for t in range(8):
            if t < 2: act = act + injection
            raw = np.zeros(H, dtype=np.float32)
            if len(rs): np.add.at(raw, cs, act[rs] * sp_vals)
            charge += raw; charge *= ret
            act = np.maximum(charge, 0.0)
            charge = np.maximum(charge, 0.0)
        state = act.copy()
        out = charge @ W_out
        out_n = out / (np.linalg.norm(out) + 1e-8)
        sims = out_n @ pat_norm.T
        if np.argmax(sims) == text_bytes[i+1]: correct += 1
        total += 1
    return correct/total if total else 0


def run_config(name, use_bool, bp, ALL_DATA, bigram, eval_seqs, H, W_in, W_out,
               max_steps=500, n_workers=18, threshold=0.00005):
    # Init
    exists = np.zeros((H, H), dtype=np.bool_)
    sign = np.zeros((H, H), dtype=np.bool_)
    mask = np.zeros((H, H), dtype=np.float32)
    ret_int = np.full(H, 217, dtype=np.int32)

    schedule = ['add', 'add', 'flip', 'decay', 'decay', 'decay', 'decay', 'decay']

    print(f"\n--- {name} (use_bool={use_bool}) ---")
    sys.stdout.flush()

    accepts = {'add': 0, 'flip': 0, 'decay': 0}
    t0 = time.time()

    pool = Pool(n_workers, initializer=init_w,
                initargs=(bp, ALL_DATA, 200, 2, W_in, W_out, bigram, use_bool))
    try:
        for step in range(1, max_steps+1):
            ptype = schedule[(step-1) % len(schedule)]
            edge_count = int(exists.sum()) if use_bool else int(np.count_nonzero(mask))
            if ptype in ('flip', 'decay') and edge_count == 0:
                ptype = 'add'

            if use_bool:
                args = [(exists.flatten(), sign.flatten(), ret_int.copy(), H,
                         35000+step*50+w, ptype) for w in range(n_workers)]
            else:
                args = [(mask.flatten(), ret_int.copy(), H,
                         35000+step*50+w, ptype) for w in range(n_workers)]

            results = pool.map(worker_eval, args)
            best_r = max(results, key=lambda x: x['delta'])

            if best_r['delta'] > threshold:
                if use_bool:
                    if best_r['type'] in ('add', 'flip') and best_r['new_exists_flat'] is not None:
                        exists = best_r['new_exists_flat'].reshape(H, H)
                        sign = best_r['new_sign_flat'].reshape(H, H)
                        accepts[best_r['type']] += 1
                    elif best_r['type'] == 'decay' and best_r['new_ret'] is not None:
                        ret_int = best_r['new_ret']
                        accepts['decay'] += 1
                else:
                    if best_r['type'] in ('add', 'flip') and best_r['new_mask_flat'] is not None:
                        mask = best_r['new_mask_flat'].reshape(H, H)
                        accepts[best_r['type']] += 1
                    elif best_r['type'] == 'decay' and best_r['new_ret'] is not None:
                        ret_int = best_r['new_ret']
                        accepts['decay'] += 1

            if step % 100 == 0:
                elapsed = time.time() - t0
                edges = int(exists.sum()) if use_bool else int(np.count_nonzero(mask))
                ea = np.mean([eval_accuracy(exists, sign, mask, H, W_in, W_out, ret_int,
                              use_bool, s, bp) for s in eval_seqs])
                quality = ea / max(edges, 1) * 100
                print(f"  [{step:3d}] acc={ea*100:.2f}% edges={edges} q={quality:.3f}%/e "
                      f"A={accepts['add']}|F={accepts['flip']}|D={accepts['decay']} {elapsed:.0f}s")
                sys.stdout.flush()
    finally:
        pool.terminate(); pool.join()

    edges = int(exists.sum()) if use_bool else int(np.count_nonzero(mask))
    ea = np.mean([eval_accuracy(exists, sign, mask, H, W_in, W_out, ret_int,
                  use_bool, s, bp) for s in eval_seqs])
    elapsed = time.time() - t0
    quality = ea / max(edges, 1) * 100
    print(f"  FINAL: acc={ea*100:.2f}% edges={edges} q={quality:.3f}%/e {elapsed:.0f}s")
    sys.stdout.flush()
    return {'name': name, 'acc': ea, 'edges': edges, 'quality': quality, 'time': elapsed}


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

    results = []

    # A: Float mask (current)
    results.append(run_config("FLOAT mask", False,
                              bp, ALL_DATA, bigram, eval_seqs, H, W_in, W_out))

    # B: Bool mask (2 arrays: exists + sign)
    results.append(run_config("BOOL mask", True,
                              bp, ALL_DATA, bigram, eval_seqs, H, W_in, W_out))

    print(f"\n{'='*55}")
    print(f"  FLOAT vs BOOL MASK (500 steps, int8 retention)")
    print(f"{'='*55}")
    print(f"  {'Name':<15} {'Acc%':>6} {'Edges':>6} {'Q(%/e)':>8} {'Time':>6}")
    print(f"  {'-'*15} {'-'*6} {'-'*6} {'-'*8} {'-'*6}")
    for r in results:
        print(f"  {r['name']:<15} {r['acc']*100:6.2f} {r['edges']:6d} {r['quality']:8.3f} {r['time']:5.0f}s")
    sys.stdout.flush()

"""
Int8 Retention — pure int [0-256] with bitshift
=================================================
retention_int [0-256]: charge = charge * ret_int >> 8
0 = full forget, 256 = perfect memory.
Sweet spot: int 194-235 (= retention 0.76-0.92 = decay 0.08-0.24).
A: Float baseline, B: Int8 sweet spot, C: Int8 fix 217 (≈0.85)
18 workers, 1000 steps each.
"""
import sys, os, time, random
import numpy as np
from multiprocessing import Pool

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "model"))
from graph import SelfWiringGraph

_bp = None; _all_data = None; _seq_len = 200; _n_train = 2
_input_projection = None; _output_projection = None; _bigram = None
_use_int = False

def init_w(b, d, sl, nt, wi, wo, bg, ui):
    global _bp, _all_data, _seq_len, _n_train, _input_projection, _output_projection, _bigram, _use_int
    _bp, _all_data, _seq_len, _n_train = b, d, sl, nt
    _input_projection, _output_projection, _bigram = wi, wo, bg
    _use_int = ui

def make_bp(io_dim, seed=12345):
    rng = np.random.RandomState(seed)
    p = rng.randn(256, io_dim).astype(np.float32)
    p /= np.linalg.norm(p, axis=1, keepdims=True)
    return p

def _eval_bigram(mask, H, ret_param, seqs):
    rs, cs = np.where(mask != 0)
    sp_vals = mask[rs, cs]
    pat_norm = _bp / (np.linalg.norm(_bp, axis=1, keepdims=True) + 1e-8)
    if _use_int:
        # int [0-256]: ret = param / 256.0 (equivalent to >> 8 in int math)
        ret = ret_param.astype(np.float32) / 256.0
    else:
        ret = ret_param  # already float retention (1 - decay)
    total = 0.0
    for text_bytes in seqs:
        state = np.zeros(H, dtype=np.float32)
        charge = np.zeros(H, dtype=np.float32)
        seq_score = 0.0; n = 0
        for i in range(len(text_bytes)-1):
            act = state.copy()
            injection = _bp[text_bytes[i]] @ _input_projection
            for t in range(8):
                if t < 2:
                    act = act + injection
                raw = np.zeros(H, dtype=np.float32)
                if len(rs):
                    np.add.at(raw, cs, act[rs] * sp_vals)
                charge += raw; charge *= ret
                act = np.maximum(charge, 0.0)
                charge = np.maximum(charge, 0.0)
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
    mask_flat, ret_param, H, seed, proposal_type = args
    rng = random.Random(seed)
    np_rng = np.random.RandomState(seed)
    mask = mask_flat.reshape(H, H)
    new_mask = mask.copy()
    new_ret = ret_param.copy()

    if proposal_type == 'add':
        r = rng.randint(0, H-1); c = rng.randint(0, H-1)
        if r == c or mask[r, c] != 0:
            return {'delta': -1e9, 'type': 'add'}
        val = 0.6 if rng.random() < 0.5 else -0.6
        new_mask[r, c] = val
    elif proposal_type == 'flip':
        alive = list(zip(*np.where(mask != 0)))
        if not alive:
            return {'delta': -1e9, 'type': 'flip'}
        r, c = alive[rng.randint(0, len(alive)-1)]
        new_mask[r, c] = -mask[r, c]
    elif proposal_type == 'decay':
        idx = rng.randint(0, H-1)
        if _use_int:
            new_ret[idx] = rng.randint(0, 256)  # resample [0-256]
        else:
            new_ret[idx] = 1.0 - rng.uniform(0.01, 0.50)  # float retention

    seqs = []
    data_len = len(_all_data)
    for _ in range(_n_train):
        off = np_rng.randint(0, data_len - _seq_len)
        seqs.append(_all_data[off:off+_seq_len])

    old_score = _eval_bigram(mask, H, ret_param, seqs)
    new_score = _eval_bigram(new_mask, H, new_ret, seqs)

    return {'delta': new_score - old_score, 'type': proposal_type,
            'new_mask_flat': new_mask.flatten() if new_score > old_score else None,
            'new_ret': new_ret if new_score > old_score else None}

def eval_accuracy_classic(mask, H, input_projection, output_projection, ret_param, use_int, text_bytes, bp):
    pat_norm = bp / (np.linalg.norm(bp, axis=1, keepdims=True) + 1e-8)
    rs, cs = np.where(mask != 0); sp_vals = mask[rs, cs]
    if use_int:
        ret = ret_param.astype(np.float32) / 256.0
    else:
        ret = ret_param
    state = np.zeros(H, dtype=np.float32); charge = np.zeros(H, dtype=np.float32)
    correct = 0; total = 0
    for i in range(len(text_bytes)-1):
        act = state.copy()
        injection = bp[text_bytes[i]] @ input_projection
        for t in range(8):
            if t < 2: act = act + injection
            raw = np.zeros(H, dtype=np.float32)
            if len(rs): np.add.at(raw, cs, act[rs] * sp_vals)
            charge += raw; charge *= ret
            act = np.maximum(charge, 0.0)
            charge = np.maximum(charge, 0.0)
        state = act.copy()
        out = charge @ output_projection
        out_n = out / (np.linalg.norm(out) + 1e-8)
        sims = out_n @ pat_norm.T
        if np.argmax(sims) == text_bytes[i+1]: correct += 1
        total += 1
    return correct/total if total else 0


def run_config(name, use_int, init_ret, bp, ALL_DATA, bigram, eval_seqs, H, input_projection, output_projection,
               max_steps=1000, n_workers=18, threshold=0.00005):
    mask = np.zeros((H, H), dtype=np.float32)
    ret_param = init_ret.copy()

    schedule = ['add', 'add', 'flip', 'decay', 'decay', 'decay', 'decay', 'decay']

    if use_int:
        desc = f"int8 mean={ret_param.mean():.0f} [{ret_param.min()}-{ret_param.max()}] (ret={ret_param.mean()/256:.3f})"
    else:
        desc = f"float mean={ret_param.mean():.3f} [{ret_param.min():.3f}-{ret_param.max():.3f}]"
    print(f"\n--- {name} ({desc}) ---")
    sys.stdout.flush()

    accepts = {'add': 0, 'flip': 0, 'decay': 0}
    acc_history = []
    t0 = time.time()

    pool = Pool(n_workers, initializer=init_w,
                initargs=(bp, ALL_DATA, 200, 2, input_projection, output_projection, bigram, use_int))
    try:
        for step in range(1, max_steps+1):
            ptype = schedule[(step-1) % len(schedule)]
            if ptype in ('flip', 'decay') and np.count_nonzero(mask) == 0:
                ptype = 'add'

            mask_flat = mask.flatten()
            args = [(mask_flat, ret_param.copy(), H,
                     34000+step*50+w, ptype) for w in range(n_workers)]
            results = pool.map(worker_eval, args)

            best_r = max(results, key=lambda x: x['delta'])
            if best_r['delta'] > threshold:
                if best_r['type'] in ('add', 'flip') and best_r['new_mask_flat'] is not None:
                    mask = best_r['new_mask_flat'].reshape(H, H)
                    accepts[best_r['type']] += 1
                elif best_r['type'] == 'decay' and best_r['new_ret'] is not None:
                    ret_param = best_r['new_ret']
                    accepts['decay'] += 1

            if step % 200 == 0:
                elapsed = time.time() - t0
                edges = int(np.count_nonzero(mask))
                ea = np.mean([eval_accuracy_classic(mask, H, input_projection, output_projection, ret_param, use_int, s, bp)
                              for s in eval_seqs])
                acc_history.append((step, ea))
                quality = ea / max(edges, 1) * 100
                if use_int:
                    rstr = f"ret_int={ret_param.mean():.0f}+/-{ret_param.std():.0f} [{ret_param.min()}-{ret_param.max()}]"
                else:
                    rstr = f"ret={ret_param.mean():.3f}+/-{ret_param.std():.3f}"

                print(f"  [{step:4d}] acc={ea*100:.2f}% edges={edges} q={quality:.3f}%/e "
                      f"A={accepts['add']}|F={accepts['flip']}|D={accepts['decay']} {rstr} {elapsed:.0f}s")
                sys.stdout.flush()
    finally:
        pool.terminate(); pool.join()

    edges = int(np.count_nonzero(mask))
    ea = np.mean([eval_accuracy_classic(mask, H, input_projection, output_projection, ret_param, use_int, s, bp)
                  for s in eval_seqs])
    elapsed = time.time() - t0
    quality = ea / max(edges, 1) * 100

    print(f"  FINAL: acc={ea*100:.2f}% edges={edges} q={quality:.3f}%/e D={accepts['decay']} {elapsed:.0f}s")
    sys.stdout.flush()
    return {'name': name, 'acc': ea, 'edges': edges, 'quality': quality,
            'decay_accepts': accepts['decay'], 'time': elapsed}


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
    ref = SelfWiringGraph(IO)
    input_projection = ref.input_projection / ref.INJ_SCALE * 1.0
    output_projection = ref.output_projection / ref.INJ_SCALE * 1.0

    results = []

    # A: Float baseline (retention = 1 - decay, init [0.76, 0.92])
    drng = np.random.RandomState(99)
    float_ret = 1.0 - drng.uniform(0.08, 0.24, H).astype(np.float32)
    results.append(run_config("FLOAT baseline", False, float_ret,
                              bp, ALL_DATA, bigram, eval_seqs, H, input_projection, output_projection))

    # B: Int8 sweet spot [194-235] (= retention 0.758-0.918)
    drng2 = np.random.RandomState(99)
    int_ret_sweet = drng2.randint(194, 236, size=H).astype(np.int32)
    results.append(run_config("INT8 [194-235]", True, int_ret_sweet,
                              bp, ALL_DATA, bigram, eval_seqs, H, input_projection, output_projection))

    # C: Int8 fix 217 (= retention 0.848 ≈ 0.85)
    results.append(run_config("INT8 fix=217", True,
                              np.full(H, 217, dtype=np.int32),
                              bp, ALL_DATA, bigram, eval_seqs, H, input_projection, output_projection))

    print(f"\n{'='*65}")
    print(f"  SUMMARY -- INT8 vs FLOAT RETENTION (1000 steps, 2a/1f/5d)")
    print(f"{'='*65}")
    print(f"  {'Name':<20} {'Acc%':>6} {'Edges':>6} {'Q(%/e)':>8} {'D_acc':>6}")
    print(f"  {'-'*20} {'-'*6} {'-'*6} {'-'*8} {'-'*6}")
    for r in results:
        print(f"  {r['name']:<20} {r['acc']*100:6.2f} {r['edges']:6d} {r['quality']:8.3f} {r['decay_accepts']:6d}")
    sys.stdout.flush()

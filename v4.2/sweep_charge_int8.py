"""
Int8 Charge/State — quantize runtime to int8
==============================================
charge and state as int8 [0-255] (since ReLU, never negative).
Forward pass: all ops in int8 where possible.
A: Float (baseline)
B: Int8 charge/state (scale=256)
500 steps, 18 workers, sign+mag mask, int8 injection table, ret=217.
"""
import sys, os, time, random
import numpy as np
from multiprocessing import Pool

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "model"))
from graph import SelfWiringGraph

_bp = None; _all_data = None; _seq_len = 200; _n_train = 2
_input_projection = None; _output_projection = None; _bigram = None
_inj_table = None
_use_int_charge = False

def init_w(b, d, sl, nt, wi, wo, bg, it, uic):
    global _bp, _all_data, _seq_len, _n_train, _input_projection, _output_projection, _bigram, _inj_table, _use_int_charge
    _bp, _all_data, _seq_len, _n_train = b, d, sl, nt
    _input_projection, _output_projection, _bigram = wi, wo, bg
    _inj_table, _use_int_charge = it, uic

def make_bp(io_dim, seed=12345):
    rng = np.random.RandomState(seed)
    p = rng.randn(256, io_dim).astype(np.float32)
    p /= np.linalg.norm(p, axis=1, keepdims=True)
    return p

def _eval_bigram(mask_sign, mask_mag, H, seqs):
    rs, cs = np.where(mask_mag > 0)
    s = mask_sign[rs, cs].astype(np.float32) * 2 - 1
    sp_vals = s * mask_mag[rs, cs].astype(np.float32) / 128.0
    pat_norm = _bp / (np.linalg.norm(_bp, axis=1, keepdims=True) + 1e-8)
    ret_int = 217
    total = 0.0
    for text_bytes in seqs:
        if _use_int_charge:
            # Int8 charge/state — scaled by 256
            state = np.zeros(H, dtype=np.int32)  # int32 to avoid overflow during add
            charge = np.zeros(H, dtype=np.int32)
        else:
            state = np.zeros(H, dtype=np.float32)
            charge = np.zeros(H, dtype=np.float32)
        seq_score = 0.0; n = 0
        for i in range(len(text_bytes)-1):
            if _use_int_charge:
                act = state.copy()
                for t in range(8):
                    if t < 2:
                        # injection table is int8, scale to match charge scale
                        act = act + _inj_table[text_bytes[i]].astype(np.int32) * 2
                    raw = np.zeros(H, dtype=np.int32)
                    if len(rs):
                        # sp_vals is float, but we can compute in int:
                        # raw = act[rs] * sign * mag / 128
                        edge_s = mask_sign[rs, cs].astype(np.int32) * 2 - 1
                        edge_m = mask_mag[rs, cs].astype(np.int32)
                        np.add.at(raw, cs, act[rs] * edge_s * edge_m >> 7)  # /128 via shift
                    charge = charge + raw
                    charge = charge * ret_int >> 8  # retention via shift
                    act = np.maximum(charge, 0)
                    charge = np.maximum(charge, 0)
                state = act.copy()
                # Output: convert back to float for output_projection matmul
                charge_f = charge.astype(np.float32) / 256.0
            else:
                act = state.copy()
                ret_f = ret_int / 256.0
                for t in range(8):
                    if t < 2:
                        act = act + _inj_table[text_bytes[i]].astype(np.float32) / 128.0
                    raw = np.zeros(H, dtype=np.float32)
                    if len(rs):
                        np.add.at(raw, cs, act[rs] * sp_vals)
                    charge += raw; charge *= ret_f
                    act = np.maximum(charge, 0.0)
                    charge = np.maximum(charge, 0.0)
                state = act.copy()
                charge_f = charge

            out = charge_f @ _output_projection
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
    msign_flat, mmag_flat, H, seed, proposal_type = args
    rng = random.Random(seed)
    np_rng = np.random.RandomState(seed)
    msign = msign_flat.reshape(H, H)
    mmag = mmag_flat.reshape(H, H)
    new_msign = msign.copy(); new_mmag = mmag.copy()

    if proposal_type == 'add':
        r = rng.randint(0, H-1); c = rng.randint(0, H-1)
        if r == c or mmag[r, c] > 0:
            return {'delta': -1e9, 'type': 'add'}
        new_msign[r, c] = rng.random() < 0.5
        new_mmag[r, c] = rng.randint(1, 255)
    elif proposal_type == 'flip':
        rs, cs = np.where(mmag > 0)
        if len(rs) == 0:
            return {'delta': -1e9, 'type': 'flip'}
        idx = rng.randint(0, len(rs)-1)
        new_msign[rs[idx], cs[idx]] = not msign[rs[idx], cs[idx]]

    seqs = []
    data_len = len(_all_data)
    for _ in range(_n_train):
        off = np_rng.randint(0, data_len - _seq_len)
        seqs.append(_all_data[off:off+_seq_len])

    old_score = _eval_bigram(msign, mmag, H, seqs)
    new_score = _eval_bigram(new_msign, new_mmag, H, seqs)

    return {'delta': new_score - old_score, 'type': proposal_type,
            'new_msign': new_msign.flatten() if new_score > old_score else None,
            'new_mmag': new_mmag.flatten() if new_score > old_score else None}

def eval_acc(msign, mmag, H, input_projection, output_projection, text_bytes, bp, inj_table, use_int):
    pat_norm = bp / (np.linalg.norm(bp, axis=1, keepdims=True) + 1e-8)
    rs, cs = np.where(mmag > 0)
    s = msign[rs, cs].astype(np.float32) * 2 - 1
    sp_vals = s * mmag[rs, cs].astype(np.float32) / 128.0
    ret_f = 217.0 / 256.0
    state = np.zeros(H, dtype=np.float32); charge = np.zeros(H, dtype=np.float32)
    correct = 0; total = 0
    for i in range(len(text_bytes)-1):
        act = state.copy()
        for t in range(8):
            if t < 2: act = act + inj_table[text_bytes[i]].astype(np.float32) / 128.0
            raw = np.zeros(H, dtype=np.float32)
            if len(rs): np.add.at(raw, cs, act[rs] * sp_vals)
            charge += raw; charge *= ret_f
            act = np.maximum(charge, 0.0); charge = np.maximum(charge, 0.0)
        state = act.copy()
        out = charge @ output_projection
        out_n = out / (np.linalg.norm(out) + 1e-8)
        sims = out_n @ pat_norm.T
        if np.argmax(sims) == text_bytes[i+1]: correct += 1
        total += 1
    return correct/total if total else 0


def run_config(name, use_int, bp, ALL_DATA, bigram, eval_seqs, H, input_projection, output_projection, inj_table,
               max_steps=500, n_workers=18, threshold=0.00005):
    msign = np.zeros((H, H), dtype=np.bool_)
    mmag = np.zeros((H, H), dtype=np.uint8)
    schedule = ['add', 'add', 'flip', 'add', 'add', 'add']

    print(f"\n--- {name} (int_charge={use_int}) ---")
    sys.stdout.flush()

    accepts = {'add': 0, 'flip': 0}
    t0 = time.time()

    pool = Pool(n_workers, initializer=init_w,
                initargs=(bp, ALL_DATA, 200, 2, input_projection, output_projection, bigram, inj_table, use_int))
    try:
        for step in range(1, max_steps+1):
            ptype = schedule[(step-1) % len(schedule)]
            if ptype == 'flip' and (mmag > 0).sum() == 0: ptype = 'add'

            args = [(msign.flatten(), mmag.flatten(), H,
                     40000+step*50+w, ptype) for w in range(n_workers)]
            results = pool.map(worker_eval, args)

            best_r = max(results, key=lambda x: x['delta'])
            if best_r['delta'] > threshold and best_r.get('new_msign') is not None:
                msign = best_r['new_msign'].reshape(H, H)
                mmag = best_r['new_mmag'].reshape(H, H)
                accepts[best_r['type']] += 1

            if step % 100 == 0:
                elapsed = time.time() - t0
                edges = int((mmag > 0).sum())
                ea = np.mean([eval_acc(msign, mmag, H, input_projection, output_projection, s, bp, inj_table, use_int)
                              for s in eval_seqs])
                print(f"  [{step:3d}] acc={ea*100:.2f}% edges={edges} "
                      f"A={accepts['add']}|F={accepts['flip']} {elapsed:.0f}s")
                sys.stdout.flush()
    finally:
        pool.terminate(); pool.join()

    edges = int((mmag > 0).sum())
    ea = np.mean([eval_acc(msign, mmag, H, input_projection, output_projection, s, bp, inj_table, use_int)
                  for s in eval_seqs])
    elapsed = time.time() - t0
    print(f"  FINAL: acc={ea*100:.2f}% edges={edges} {elapsed:.0f}s")
    sys.stdout.flush()
    return {'name': name, 'acc': ea, 'edges': edges, 'time': elapsed}


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

    inj_table = np.clip(bp @ input_projection * 128, -128, 127).astype(np.int8)
    print(f"Injection table: range=[{inj_table.min()},{inj_table.max()}]")

    results = []

    # A: Float charge (int8 injection table, float charge/state)
    results.append(run_config("FLOAT charge", False,
                              bp, ALL_DATA, bigram, eval_seqs, H, input_projection, output_projection, inj_table))

    # B: Int charge (int32 internal, int8 output)
    results.append(run_config("INT charge", True,
                              bp, ALL_DATA, bigram, eval_seqs, H, input_projection, output_projection, inj_table))

    print(f"\n{'='*55}")
    print(f"  FLOAT vs INT CHARGE (500 steps)")
    print(f"{'='*55}")
    for r in results:
        print(f"  {r['name']:<15} {r['acc']*100:6.2f}% {r['edges']} edges {r['time']:.0f}s")
    sys.stdout.flush()

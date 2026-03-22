"""
Unified Input — dense float vs int8 table vs sparse learnable
==============================================================
A: Dense float input_projection (current baseline)
B: Precomputed int8 injection table (same mapping, quantized)
C: Sparse learnable input edges (byte -> neuron, built by mutation)
500 steps, 18 workers, sign+mag mask, ret=217.
"""
import sys, os, time, random
import numpy as np
from multiprocessing import Pool

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "model"))
from graph import SelfWiringGraph

_bp = None; _all_data = None; _seq_len = 200; _n_train = 2
_input_projection = None; _output_projection = None; _bigram = None
_inj_table = None  # int8 precomputed
_mode = 'dense'  # 'dense', 'int8_table', 'sparse'

def init_w(b, d, sl, nt, wi, wo, bg, it, m):
    global _bp, _all_data, _seq_len, _n_train, _input_projection, _output_projection, _bigram, _inj_table, _mode
    _bp, _all_data, _seq_len, _n_train = b, d, sl, nt
    _input_projection, _output_projection, _bigram = wi, wo, bg
    _inj_table, _mode = it, m

def make_bp(io_dim, seed=12345):
    rng = np.random.RandomState(seed)
    p = rng.randn(256, io_dim).astype(np.float32)
    p /= np.linalg.norm(p, axis=1, keepdims=True)
    return p

def get_injection(byte_val, input_sign, input_mag, H):
    """Sparse injection from learnable input edges."""
    inj = np.zeros(H, dtype=np.float32)
    neuron_ids = np.where(input_mag[byte_val] > 0)[0]
    if len(neuron_ids):
        signs = input_sign[byte_val, neuron_ids].astype(np.float32) * 2 - 1
        mags = input_mag[byte_val, neuron_ids].astype(np.float32) / 128.0
        inj[neuron_ids] = signs * mags
    return inj

def _eval_bigram(mask_sign, mask_mag, input_sign, input_mag, H, ret, seqs):
    rs, cs = np.where(mask_mag > 0)
    s = mask_sign[rs, cs].astype(np.float32) * 2 - 1
    sp_vals = s * mask_mag[rs, cs].astype(np.float32) / 128.0
    pat_norm = _bp / (np.linalg.norm(_bp, axis=1, keepdims=True) + 1e-8)
    total = 0.0
    for text_bytes in seqs:
        state = np.zeros(H, dtype=np.float32)
        charge = np.zeros(H, dtype=np.float32)
        seq_score = 0.0; n = 0
        for i in range(len(text_bytes)-1):
            act = state.copy()
            for t in range(8):
                if t < 2:
                    if _mode == 'dense':
                        act = act + _bp[text_bytes[i]] @ _input_projection
                    elif _mode == 'int8_table':
                        act = act + _inj_table[text_bytes[i]].astype(np.float32) / 128.0
                    elif _mode == 'sparse':
                        act = act + get_injection(text_bytes[i], input_sign, input_mag, H)
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
    mask_sign_flat, mask_mag_flat, input_sign_flat, input_mag_flat, H, seed, proposal_type = args
    rng = random.Random(seed)
    np_rng = np.random.RandomState(seed)
    mask_sign = mask_sign_flat.reshape(H, H)
    mask_mag = mask_mag_flat.reshape(H, H)
    input_sign = input_sign_flat.reshape(256, H) if input_sign_flat is not None else None
    input_mag = input_mag_flat.reshape(256, H) if input_mag_flat is not None else None

    new_msign = mask_sign.copy(); new_mmag = mask_mag.copy()
    new_isign = input_sign.copy() if input_sign is not None else None
    new_imag = input_mag.copy() if input_mag is not None else None
    ret = 217.0 / 256.0

    if proposal_type == 'add':
        r = rng.randint(0, H-1); c = rng.randint(0, H-1)
        if r == c or mask_mag[r, c] > 0:
            return {'delta': -1e9, 'type': 'add'}
        new_msign[r, c] = rng.random() < 0.5
        new_mmag[r, c] = rng.randint(1, 255)
    elif proposal_type == 'flip':
        rs, cs = np.where(mask_mag > 0)
        if len(rs) == 0:
            return {'delta': -1e9, 'type': 'flip'}
        idx = rng.randint(0, len(rs)-1)
        r, c = rs[idx], cs[idx]
        new_msign[r, c] = not mask_sign[r, c]
    elif proposal_type == 'input_add':
        if input_mag is None:
            return {'delta': -1e9, 'type': 'input_add'}
        b = rng.randint(0, 255); n = rng.randint(0, H-1)
        if input_mag[b, n] > 0:
            return {'delta': -1e9, 'type': 'input_add'}
        new_isign[b, n] = rng.random() < 0.5
        new_imag[b, n] = rng.randint(1, 255)
    elif proposal_type == 'input_flip':
        if input_mag is None:
            return {'delta': -1e9, 'type': 'input_flip'}
        alive = np.where(input_mag > 0)
        if len(alive[0]) == 0:
            return {'delta': -1e9, 'type': 'input_flip'}
        idx = rng.randint(0, len(alive[0])-1)
        b, n = alive[0][idx], alive[1][idx]
        new_isign[b, n] = not input_sign[b, n]
    elif proposal_type == 'decay':
        # placeholder — not used in this sweep
        return {'delta': -1e9, 'type': 'decay'}

    seqs = []
    data_len = len(_all_data)
    for _ in range(_n_train):
        off = np_rng.randint(0, data_len - _seq_len)
        seqs.append(_all_data[off:off+_seq_len])

    old_score = _eval_bigram(mask_sign, mask_mag, input_sign, input_mag, H, ret, seqs)
    new_score = _eval_bigram(new_msign, new_mmag, new_isign, new_imag, H, ret, seqs)

    return {'delta': new_score - old_score, 'type': proposal_type,
            'new_msign': new_msign.flatten() if new_score > old_score else None,
            'new_mmag': new_mmag.flatten() if new_score > old_score else None,
            'new_isign': new_isign.flatten() if new_score > old_score and new_isign is not None else None,
            'new_imag': new_imag.flatten() if new_score > old_score and new_imag is not None else None}

def eval_accuracy(mask_sign, mask_mag, input_sign, input_mag, H, input_projection, output_projection, text_bytes, bp, mode, inj_table):
    pat_norm = bp / (np.linalg.norm(bp, axis=1, keepdims=True) + 1e-8)
    rs, cs = np.where(mask_mag > 0)
    s = mask_sign[rs, cs].astype(np.float32) * 2 - 1
    sp_vals = s * mask_mag[rs, cs].astype(np.float32) / 128.0
    ret = 217.0 / 256.0
    state = np.zeros(H, dtype=np.float32); charge = np.zeros(H, dtype=np.float32)
    correct = 0; total = 0
    for i in range(len(text_bytes)-1):
        act = state.copy()
        for t in range(8):
            if t < 2:
                if mode == 'dense':
                    act = act + bp[text_bytes[i]] @ input_projection
                elif mode == 'int8_table':
                    act = act + inj_table[text_bytes[i]].astype(np.float32) / 128.0
                elif mode == 'sparse':
                    act = act + get_injection(text_bytes[i], input_sign, input_mag, H)
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


def run_config(name, mode, bp, ALL_DATA, bigram, eval_seqs, H, input_projection, output_projection, inj_table,
               max_steps=500, n_workers=18, threshold=0.00005):
    mask_sign = np.zeros((H, H), dtype=np.bool_)
    mask_mag = np.zeros((H, H), dtype=np.uint8)

    if mode == 'sparse':
        input_sign = np.zeros((256, H), dtype=np.bool_)
        input_mag = np.zeros((256, H), dtype=np.uint8)
        schedule = ['add', 'input_add', 'input_add', 'flip', 'input_flip', 'add']
    else:
        input_sign = None; input_mag = None
        schedule = ['add', 'add', 'flip', 'add', 'add', 'add']

    print(f"\n--- {name} (mode={mode}) ---")
    sys.stdout.flush()

    accepts = {'add': 0, 'flip': 0, 'input_add': 0, 'input_flip': 0, 'decay': 0}
    t0 = time.time()

    pool = Pool(n_workers, initializer=init_w,
                initargs=(bp, ALL_DATA, 200, 2, input_projection, output_projection, bigram, inj_table, mode))
    try:
        for step in range(1, max_steps+1):
            ptype = schedule[(step-1) % len(schedule)]
            edges = int((mask_mag > 0).sum())
            if ptype == 'flip' and edges == 0: ptype = 'add'
            if ptype == 'input_flip' and (input_mag is None or (input_mag > 0).sum() == 0):
                ptype = 'input_add'

            isf = input_sign.flatten() if input_sign is not None else None
            imf = input_mag.flatten() if input_mag is not None else None

            args = [(mask_sign.flatten(), mask_mag.flatten(), isf, imf,
                     H, 39000+step*50+w, ptype) for w in range(n_workers)]
            results = pool.map(worker_eval, args)

            best_r = max(results, key=lambda x: x['delta'])
            if best_r['delta'] > threshold:
                if best_r.get('new_msign') is not None:
                    mask_sign = best_r['new_msign'].reshape(H, H)
                    mask_mag = best_r['new_mmag'].reshape(H, H)
                if best_r.get('new_isign') is not None:
                    input_sign = best_r['new_isign'].reshape(256, H)
                    input_mag = best_r['new_imag'].reshape(256, H)
                accepts[best_r['type']] += 1

            if step % 100 == 0:
                elapsed = time.time() - t0
                edges = int((mask_mag > 0).sum())
                iedges = int((input_mag > 0).sum()) if input_mag is not None else 0
                ea = np.mean([eval_accuracy(mask_sign, mask_mag, input_sign, input_mag,
                              H, input_projection, output_projection, s, bp, mode, inj_table) for s in eval_seqs])
                quality = ea / max(edges + iedges, 1) * 100

                print(f"  [{step:3d}] acc={ea*100:.2f}% edges={edges} iedges={iedges} "
                      f"q={quality:.3f}%/e A={accepts['add']}|F={accepts['flip']}"
                      f"|IA={accepts['input_add']}|IF={accepts['input_flip']} {elapsed:.0f}s")
                sys.stdout.flush()
    finally:
        pool.terminate(); pool.join()

    edges = int((mask_mag > 0).sum())
    iedges = int((input_mag > 0).sum()) if input_mag is not None else 0
    ea = np.mean([eval_accuracy(mask_sign, mask_mag, input_sign, input_mag,
                  H, input_projection, output_projection, s, bp, mode, inj_table) for s in eval_seqs])
    elapsed = time.time() - t0
    quality = ea / max(edges + iedges, 1) * 100

    print(f"  FINAL: acc={ea*100:.2f}% edges={edges} iedges={iedges} q={quality:.3f}%/e {elapsed:.0f}s")
    sys.stdout.flush()
    return {'name': name, 'acc': ea, 'edges': edges, 'iedges': iedges,
            'quality': quality, 'time': elapsed}


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

    # Precompute int8 injection table
    inj_table = np.clip(bp @ input_projection * 128, -128, 127).astype(np.int8)
    print(f"Injection table: {inj_table.shape}, range=[{inj_table.min()},{inj_table.max()}]")

    results = []

    # A: Dense float (current)
    results.append(run_config("A: Dense float", 'dense',
                              bp, ALL_DATA, bigram, eval_seqs, H, input_projection, output_projection, inj_table))

    # B: Int8 table (precomputed, no matmul)
    results.append(run_config("B: Int8 table", 'int8_table',
                              bp, ALL_DATA, bigram, eval_seqs, H, input_projection, output_projection, inj_table))

    # C: Sparse learnable (no input_projection, edges byte->neuron)
    results.append(run_config("C: Sparse learnable", 'sparse',
                              bp, ALL_DATA, bigram, eval_seqs, H, input_projection, output_projection, inj_table))

    print(f"\n{'='*65}")
    print(f"  SUMMARY -- UNIFIED INPUT (500 steps, sign+mag, ret=217)")
    print(f"{'='*65}")
    print(f"  {'Name':<25} {'Acc%':>6} {'Edges':>6} {'IEdges':>7} {'Q(%/e)':>8} {'Time':>6}")
    print(f"  {'-'*25} {'-'*6} {'-'*6} {'-'*7} {'-'*8} {'-'*6}")
    for r in results:
        print(f"  {r['name']:<25} {r['acc']*100:6.2f} {r['edges']:6d} {r['iedges']:7d} "
              f"{r['quality']:8.3f} {r['time']:5.0f}s")
    sys.stdout.flush()

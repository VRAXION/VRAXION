"""
Sign + Magnitude mask — bool sign + uint8 magnitude
=====================================================
sign (bool): True=+, False=-
magnitude (uint8): 0=no edge, 1-255=weight strength
val = magnitude/128 * sign_to_float

A: INT8 free add (previous winner, signed int)
B: Sign+Mag free add (bool + uint8)
C: Sign+Mag + magnitude resample on existing edges
500 steps, 18 workers, ret=217.
"""
import sys, os, time, random
import numpy as np
from multiprocessing import Pool

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "model"))
from graph import SelfWiringGraph

_bp = None; _all_data = None; _seq_len = 200; _n_train = 2
_input_projection = None; _output_projection = None; _bigram = None
_mode = 'int8'  # 'int8' or 'signmag'

def init_w(b, d, sl, nt, wi, wo, bg, m):
    global _bp, _all_data, _seq_len, _n_train, _input_projection, _output_projection, _bigram, _mode
    _bp, _all_data, _seq_len, _n_train = b, d, sl, nt
    _input_projection, _output_projection, _bigram = wi, wo, bg
    _mode = m

def make_bp(io_dim, seed=12345):
    rng = np.random.RandomState(seed)
    p = rng.randn(256, io_dim).astype(np.float32)
    p /= np.linalg.norm(p, axis=1, keepdims=True)
    return p

def get_sp_vals(sign, mag, rs, cs):
    """sign+mag -> float edge values."""
    s = sign[rs, cs].astype(np.float32) * 2 - 1  # bool -> {-1, +1}
    m = mag[rs, cs].astype(np.float32) / 128.0
    return s * m

def _eval_bigram(mask_or_sign, mag_or_none, H, ret, seqs):
    if _mode == 'signmag':
        sign, mag = mask_or_sign, mag_or_none
        rs, cs = np.where(mag > 0)
        sp_vals = get_sp_vals(sign, mag, rs, cs)
    else:
        mask = mask_or_sign
        rs, cs = np.where(mask != 0)
        sp_vals = mask[rs, cs].astype(np.float32) / 128.0

    pat_norm = _bp / (np.linalg.norm(_bp, axis=1, keepdims=True) + 1e-8)
    total = 0.0
    for text_bytes in seqs:
        state = np.zeros(H, dtype=np.float32)
        charge = np.zeros(H, dtype=np.float32)
        seq_score = 0.0; n = 0
        for i in range(len(text_bytes)-1):
            act = state.copy()
            injection = _bp[text_bytes[i]] @ _input_projection
            for t in range(8):
                if t < 2: act = act + injection
                raw = np.zeros(H, dtype=np.float32)
                if len(rs): np.add.at(raw, cs, act[rs] * sp_vals)
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
    if _mode == 'signmag':
        sign_flat, mag_flat, H, seed, proposal_type = args
        sign = sign_flat.reshape(H, H)
        mag = mag_flat.reshape(H, H)
    else:
        mask_flat, H, seed, proposal_type = args
        mask = mask_flat.reshape(H, H)

    rng = random.Random(seed)
    np_rng = np.random.RandomState(seed)
    ret = 217.0 / 256.0

    if _mode == 'signmag':
        new_sign = sign.copy(); new_mag = mag.copy()

        if proposal_type == 'add':
            r = rng.randint(0, H-1); c = rng.randint(0, H-1)
            if r == c or mag[r, c] > 0:
                return {'delta': -1e9, 'type': 'add'}
            new_sign[r, c] = rng.random() < 0.5
            new_mag[r, c] = rng.randint(1, 255)
        elif proposal_type == 'flip':
            rs, cs = np.where(mag > 0)
            if len(rs) == 0:
                return {'delta': -1e9, 'type': 'flip'}
            idx = rng.randint(0, len(rs)-1)
            r, c = rs[idx], cs[idx]
            new_sign[r, c] = not sign[r, c]
        elif proposal_type == 'mag_resample':
            rs, cs = np.where(mag > 0)
            if len(rs) == 0:
                return {'delta': -1e9, 'type': 'mag_resample'}
            idx = rng.randint(0, len(rs)-1)
            r, c = rs[idx], cs[idx]
            new_mag[r, c] = rng.randint(1, 255)

        seqs = []
        data_len = len(_all_data)
        for _ in range(_n_train):
            off = np_rng.randint(0, data_len - _seq_len)
            seqs.append(_all_data[off:off+_seq_len])

        old_score = _eval_bigram(sign, mag, H, ret, seqs)
        new_score = _eval_bigram(new_sign, new_mag, H, ret, seqs)

        return {'delta': new_score - old_score, 'type': proposal_type,
                'new_sign_flat': new_sign.flatten() if new_score > old_score else None,
                'new_mag_flat': new_mag.flatten() if new_score > old_score else None}
    else:
        new_mask = mask.copy()

        if proposal_type == 'add':
            r = rng.randint(0, H-1); c = rng.randint(0, H-1)
            if r == c or mask[r, c] != 0:
                return {'delta': -1e9, 'type': 'add'}
            val = rng.randint(-128, 127)
            while val == 0: val = rng.randint(-128, 127)
            new_mask[r, c] = val
        elif proposal_type == 'flip':
            alive = list(zip(*np.where(mask != 0)))
            if not alive:
                return {'delta': -1e9, 'type': 'flip'}
            r, c = alive[rng.randint(0, len(alive)-1)]
            new_mask[r, c] = -mask[r, c]

        seqs = []
        data_len = len(_all_data)
        for _ in range(_n_train):
            off = np_rng.randint(0, data_len - _seq_len)
            seqs.append(_all_data[off:off+_seq_len])

        old_score = _eval_bigram(mask, None, H, ret, seqs)
        new_score = _eval_bigram(new_mask, None, H, ret, seqs)

        return {'delta': new_score - old_score, 'type': proposal_type,
                'new_mask_flat': new_mask.flatten() if new_score > old_score else None}

def eval_accuracy(sign, mag, mask, H, input_projection, output_projection, use_signmag, text_bytes, bp):
    pat_norm = bp / (np.linalg.norm(bp, axis=1, keepdims=True) + 1e-8)
    if use_signmag:
        rs, cs = np.where(mag > 0)
        sp_vals = get_sp_vals(sign, mag, rs, cs)
    else:
        rs, cs = np.where(mask != 0)
        sp_vals = mask[rs, cs].astype(np.float32) / 128.0
    ret = 217.0 / 256.0
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


def run_config(name, mode, schedule, bp, ALL_DATA, bigram, eval_seqs, H, input_projection, output_projection,
               max_steps=500, n_workers=18, threshold=0.00005):
    sign = np.zeros((H, H), dtype=np.bool_)
    mag = np.zeros((H, H), dtype=np.uint8)
    mask = np.zeros((H, H), dtype=np.int32)

    print(f"\n--- {name} (mode={mode}) ---")
    sys.stdout.flush()

    accepts = {'add': 0, 'flip': 0, 'mag_resample': 0}
    t0 = time.time()

    pool = Pool(n_workers, initializer=init_w,
                initargs=(bp, ALL_DATA, 200, 2, input_projection, output_projection, bigram, mode))
    try:
        for step in range(1, max_steps+1):
            ptype = schedule[(step-1) % len(schedule)]
            edges = int((mag > 0).sum()) if mode == 'signmag' else int(np.count_nonzero(mask))
            if ptype in ('flip', 'mag_resample') and edges == 0:
                ptype = 'add'

            if mode == 'signmag':
                args = [(sign.flatten(), mag.flatten(), H,
                         38000+step*50+w, ptype) for w in range(n_workers)]
            else:
                args = [(mask.flatten(), H,
                         38000+step*50+w, ptype) for w in range(n_workers)]

            results = pool.map(worker_eval, args)
            best_r = max(results, key=lambda x: x['delta'])

            if best_r['delta'] > threshold:
                if mode == 'signmag':
                    if best_r['type'] in ('add', 'flip', 'mag_resample'):
                        if best_r.get('new_sign_flat') is not None:
                            sign = best_r['new_sign_flat'].reshape(H, H)
                            mag = best_r['new_mag_flat'].reshape(H, H)
                            accepts[best_r['type']] += 1
                else:
                    if best_r.get('new_mask_flat') is not None:
                        mask = best_r['new_mask_flat'].reshape(H, H).astype(np.int32)
                        accepts[best_r['type']] += 1

            if step % 100 == 0:
                elapsed = time.time() - t0
                edges = int((mag > 0).sum()) if mode == 'signmag' else int(np.count_nonzero(mask))
                ea = np.mean([eval_accuracy(sign, mag, mask, H, input_projection, output_projection,
                              mode == 'signmag', s, bp) for s in eval_seqs])
                quality = ea / max(edges, 1) * 100
                if mode == 'signmag':
                    vals = mag[mag > 0]
                    vstr = f"mag=[{vals.min()},{vals.max()}] std={vals.std():.0f}" if len(vals) else "none"
                else:
                    vals = mask[mask != 0]
                    vstr = f"vals=[{vals.min()},{vals.max()}] std={vals.std():.0f}" if len(vals) else "none"

                print(f"  [{step:3d}] acc={ea*100:.2f}% edges={edges} q={quality:.3f}%/e "
                      f"A={accepts['add']}|F={accepts['flip']}|M={accepts.get('mag_resample',0)} "
                      f"{vstr} {elapsed:.0f}s")
                sys.stdout.flush()
    finally:
        pool.terminate(); pool.join()

    edges = int((mag > 0).sum()) if mode == 'signmag' else int(np.count_nonzero(mask))
    ea = np.mean([eval_accuracy(sign, mag, mask, H, input_projection, output_projection,
                  mode == 'signmag', s, bp) for s in eval_seqs])
    elapsed = time.time() - t0
    quality = ea / max(edges, 1) * 100
    print(f"  FINAL: acc={ea*100:.2f}% edges={edges} q={quality:.3f}%/e {elapsed:.0f}s")
    sys.stdout.flush()
    return {'name': name, 'acc': ea, 'edges': edges, 'quality': quality, 'time': elapsed}


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

    # A: INT8 free add (previous winner)
    results.append(run_config("INT8 free", 'int8',
                              ['add', 'add', 'flip', 'add'],
                              bp, ALL_DATA, bigram, eval_seqs, H, input_projection, output_projection))

    # B: Sign+Mag free add
    results.append(run_config("SIGN+MAG free", 'signmag',
                              ['add', 'add', 'flip', 'add'],
                              bp, ALL_DATA, bigram, eval_seqs, H, input_projection, output_projection))

    # C: Sign+Mag free + magnitude resample
    results.append(run_config("SIGN+MAG + resample", 'signmag',
                              ['add', 'add', 'flip', 'mag_resample'],
                              bp, ALL_DATA, bigram, eval_seqs, H, input_projection, output_projection))

    print(f"\n{'='*60}")
    print(f"  SUMMARY -- SIGN+MAG vs INT8 (500 steps, ret=217)")
    print(f"{'='*60}")
    print(f"  {'Name':<25} {'Acc%':>6} {'Edges':>6} {'Q(%/e)':>8} {'Time':>6}")
    print(f"  {'-'*25} {'-'*6} {'-'*6} {'-'*8} {'-'*6}")
    for r in results:
        print(f"  {r['name']:<25} {r['acc']*100:6.2f} {r['edges']:6d} {r['quality']:8.3f} {r['time']:5.0f}s")
    sys.stdout.flush()

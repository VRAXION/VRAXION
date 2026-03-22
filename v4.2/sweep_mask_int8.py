"""
Int8 Mask — edge weights as int [-128..+127]
=============================================
0 = no edge. Nonzero = edge with weight val/128.
A: Float ternary (current +-0.6)
B: Int8 init +-77 (= +-0.60, same as current)
C: Int8 init random [-128..+127]
All with int8 retention fix=217.
500 steps, 18 workers.
"""
import sys, os, time, random
import numpy as np
from multiprocessing import Pool

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "model"))
from graph import SelfWiringGraph

_bp = None; _all_data = None; _seq_len = 200; _n_train = 2
_input_projection = None; _output_projection = None; _bigram = None
_use_int_mask = False

def init_w(b, d, sl, nt, wi, wo, bg, uim):
    global _bp, _all_data, _seq_len, _n_train, _input_projection, _output_projection, _bigram, _use_int_mask
    _bp, _all_data, _seq_len, _n_train = b, d, sl, nt
    _input_projection, _output_projection, _bigram = wi, wo, bg
    _use_int_mask = uim

def make_bp(io_dim, seed=12345):
    rng = np.random.RandomState(seed)
    p = rng.randn(256, io_dim).astype(np.float32)
    p /= np.linalg.norm(p, axis=1, keepdims=True)
    return p

def _eval_bigram(mask_param, H, ret, seqs):
    if _use_int_mask:
        rs, cs = np.where(mask_param != 0)
        sp_vals = mask_param[rs, cs].astype(np.float32) / 128.0
    else:
        rs, cs = np.where(mask_param != 0)
        sp_vals = mask_param[rs, cs]
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
    mask_flat, H, seed, proposal_type = args
    rng = random.Random(seed)
    np_rng = np.random.RandomState(seed)
    if _use_int_mask:
        mask = mask_flat.reshape(H, H).copy()
    else:
        mask = mask_flat.reshape(H, H).copy()
    new_mask = mask.copy()

    ret = 217.0 / 256.0  # fix int8 retention

    if proposal_type == 'add':
        r = rng.randint(0, H-1); c = rng.randint(0, H-1)
        if r == c or mask[r, c] != 0:
            return {'delta': -1e9, 'type': 'add'}
        if _use_int_mask:
            val = 77 if rng.random() < 0.5 else -77  # ≈ +-0.60
            new_mask[r, c] = val
        else:
            val = 0.6 if rng.random() < 0.5 else -0.6
            new_mask[r, c] = val
    elif proposal_type == 'flip':
        alive = list(zip(*np.where(mask != 0)))
        if not alive:
            return {'delta': -1e9, 'type': 'flip'}
        r, c = alive[rng.randint(0, len(alive)-1)]
        new_mask[r, c] = -mask[r, c]
    elif proposal_type == 'weight':
        # Int8: step +-1. Float: skip (already tested, doesn't work)
        if _use_int_mask:
            alive = list(zip(*np.where(mask != 0)))
            if not alive:
                return {'delta': -1e9, 'type': 'weight'}
            r, c = alive[rng.randint(0, len(alive)-1)]
            step = 1 if rng.random() < 0.5 else -1
            new_val = int(mask[r, c]) + step
            # Clamp and skip 0 (0 = no edge)
            if new_val == 0:
                new_val = step  # skip over 0
            new_val = max(-128, min(127, new_val))
            new_mask[r, c] = new_val
        else:
            return {'delta': -1e9, 'type': 'weight'}

    seqs = []
    data_len = len(_all_data)
    for _ in range(_n_train):
        off = np_rng.randint(0, data_len - _seq_len)
        seqs.append(_all_data[off:off+_seq_len])

    old_score = _eval_bigram(mask, H, ret, seqs)
    new_score = _eval_bigram(new_mask, H, ret, seqs)

    return {'delta': new_score - old_score, 'type': proposal_type,
            'new_mask_flat': new_mask.flatten() if new_score > old_score else None}

def eval_accuracy(mask, H, input_projection, output_projection, use_int, text_bytes, bp):
    pat_norm = bp / (np.linalg.norm(bp, axis=1, keepdims=True) + 1e-8)
    if use_int:
        rs, cs = np.where(mask != 0)
        sp_vals = mask[rs, cs].astype(np.float32) / 128.0
    else:
        rs, cs = np.where(mask != 0)
        sp_vals = mask[rs, cs]
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


def run_config(name, use_int, schedule, bp, ALL_DATA, bigram, eval_seqs, H, input_projection, output_projection,
               max_steps=500, n_workers=18, threshold=0.00005):
    if use_int:
        mask = np.zeros((H, H), dtype=np.int32)
    else:
        mask = np.zeros((H, H), dtype=np.float32)

    print(f"\n--- {name} (int_mask={use_int}, schedule={schedule}) ---")
    sys.stdout.flush()

    accepts = {'add': 0, 'flip': 0, 'weight': 0}
    t0 = time.time()

    pool = Pool(n_workers, initializer=init_w,
                initargs=(bp, ALL_DATA, 200, 2, input_projection, output_projection, bigram, use_int))
    try:
        for step in range(1, max_steps+1):
            ptype = schedule[(step-1) % len(schedule)]
            if ptype in ('flip', 'weight') and np.count_nonzero(mask) == 0:
                ptype = 'add'

            mask_flat = mask.flatten()
            args = [(mask_flat, H, 36000+step*50+w, ptype) for w in range(n_workers)]
            results = pool.map(worker_eval, args)

            best_r = max(results, key=lambda x: x['delta'])
            if best_r['delta'] > threshold:
                if best_r['new_mask_flat'] is not None:
                    mask = best_r['new_mask_flat'].reshape(H, H)
                    if use_int:
                        mask = mask.astype(np.int32)
                    accepts[best_r['type']] += 1

            if step % 100 == 0:
                elapsed = time.time() - t0
                edges = int(np.count_nonzero(mask))
                ea = np.mean([eval_accuracy(mask, H, input_projection, output_projection, use_int, s, bp)
                              for s in eval_seqs])
                quality = ea / max(edges, 1) * 100
                if use_int:
                    vals = mask[mask != 0]
                    vstr = f"vals=[{vals.min()},{vals.max()}]" if len(vals) else "no edges"
                else:
                    vstr = "float +-0.6"

                print(f"  [{step:3d}] acc={ea*100:.2f}% edges={edges} q={quality:.3f}%/e "
                      f"A={accepts['add']}|F={accepts['flip']}|W={accepts['weight']} "
                      f"{vstr} {elapsed:.0f}s")
                sys.stdout.flush()
    finally:
        pool.terminate(); pool.join()

    edges = int(np.count_nonzero(mask))
    ea = np.mean([eval_accuracy(mask, H, input_projection, output_projection, use_int, s, bp)
                  for s in eval_seqs])
    elapsed = time.time() - t0
    quality = ea / max(edges, 1) * 100
    print(f"  FINAL: acc={ea*100:.2f}% edges={edges} q={quality:.3f}%/e {elapsed:.0f}s")
    sys.stdout.flush()
    return {'name': name, 'acc': ea, 'edges': edges, 'quality': quality, 'time': elapsed,
            'accepts': dict(accepts)}


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
    input_projection = ref.input_projection / ref.INJ_SCALE * 1.0
    output_projection = ref.output_projection / ref.INJ_SCALE * 1.0

    results = []

    # A: Float ternary +-0.6 (current baseline)
    results.append(run_config("FLOAT +-0.6", False,
                              ['add', 'add', 'flip', 'add'],
                              bp, ALL_DATA, bigram, eval_seqs, H, input_projection, output_projection))

    # B: Int8 +-77 (= +-0.60, same as ternary but int)
    results.append(run_config("INT8 +-77 (ternary)", True,
                              ['add', 'add', 'flip', 'add'],
                              bp, ALL_DATA, bigram, eval_seqs, H, input_projection, output_projection))

    # C: Int8 +-77 + weight step +-1 (can the int weights evolve?)
    results.append(run_config("INT8 + weight step", True,
                              ['add', 'add', 'flip', 'weight'],
                              bp, ALL_DATA, bigram, eval_seqs, H, input_projection, output_projection))

    print(f"\n{'='*65}")
    print(f"  SUMMARY -- INT8 MASK (500 steps, ret=217)")
    print(f"{'='*65}")
    print(f"  {'Name':<25} {'Acc%':>6} {'Edges':>6} {'Q(%/e)':>8} {'W_acc':>6}")
    print(f"  {'-'*25} {'-'*6} {'-'*6} {'-'*8} {'-'*6}")
    for r in results:
        print(f"  {r['name']:<25} {r['acc']*100:6.2f} {r['edges']:6d} {r['quality']:8.3f} "
              f"{r['accepts'].get('weight', 0):6d}")
    sys.stdout.flush()

"""
Int8 Free Weight — add with random weight vs fixed +-77
=========================================================
A: Int8 ternary +-77 (baseline, previous winner)
B: Int8 free weight — add picks random [-128..-1] or [+1..127]
C: Int8 free weight + weight resample (existing edge -> random new weight)
500 steps, 18 workers, ret=217.
"""
import sys, os, time, random
import numpy as np
from multiprocessing import Pool

from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "model"))
from graph import SelfWiringGraph

_bp = None; _all_data = None; _seq_len = 200; _n_train = 2
_input_projection = None; _output_projection = None; _bigram = None
_add_mode = 'ternary'  # 'ternary', 'free'

def init_w(b, d, sl, nt, wi, wo, bg, am):
    global _bp, _all_data, _seq_len, _n_train, _input_projection, _output_projection, _bigram, _add_mode
    _bp, _all_data, _seq_len, _n_train = b, d, sl, nt
    _input_projection, _output_projection, _bigram = wi, wo, bg
    _add_mode = am

def make_bp(io_dim, seed=12345):
    rng = np.random.RandomState(seed)
    p = rng.randn(256, io_dim).astype(np.float32)
    p /= np.linalg.norm(p, axis=1, keepdims=True)
    return p

def _eval_bigram(mask, H, ret, seqs):
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
    mask_flat, H, seed, proposal_type = args
    rng = random.Random(seed)
    np_rng = np.random.RandomState(seed)
    mask = mask_flat.reshape(H, H)
    new_mask = mask.copy()
    ret = 217.0 / 256.0

    if proposal_type == 'add':
        r = rng.randint(0, H-1); c = rng.randint(0, H-1)
        if r == c or mask[r, c] != 0:
            return {'delta': -1e9, 'type': 'add'}
        if _add_mode == 'ternary':
            val = 77 if rng.random() < 0.5 else -77
        else:
            # Free: random weight, skip 0
            val = rng.randint(-128, 127)
            while val == 0:
                val = rng.randint(-128, 127)
        new_mask[r, c] = val
    elif proposal_type == 'flip':
        alive = list(zip(*np.where(mask != 0)))
        if not alive:
            return {'delta': -1e9, 'type': 'flip'}
        r, c = alive[rng.randint(0, len(alive)-1)]
        new_mask[r, c] = -mask[r, c]
    elif proposal_type == 'weight_resample':
        alive = list(zip(*np.where(mask != 0)))
        if not alive:
            return {'delta': -1e9, 'type': 'weight_resample'}
        r, c = alive[rng.randint(0, len(alive)-1)]
        val = rng.randint(-128, 127)
        while val == 0:
            val = rng.randint(-128, 127)
        new_mask[r, c] = val

    seqs = []
    data_len = len(_all_data)
    for _ in range(_n_train):
        off = np_rng.randint(0, data_len - _seq_len)
        seqs.append(_all_data[off:off+_seq_len])

    old_score = _eval_bigram(mask, H, ret, seqs)
    new_score = _eval_bigram(new_mask, H, ret, seqs)

    return {'delta': new_score - old_score, 'type': proposal_type,
            'new_mask_flat': new_mask.flatten() if new_score > old_score else None}

def eval_accuracy(mask, H, input_projection, output_projection, text_bytes, bp):
    pat_norm = bp / (np.linalg.norm(bp, axis=1, keepdims=True) + 1e-8)
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


def run_config(name, add_mode, schedule, bp, ALL_DATA, bigram, eval_seqs, H, input_projection, output_projection,
               max_steps=500, n_workers=18, threshold=0.00005):
    mask = np.zeros((H, H), dtype=np.int32)

    print(f"\n--- {name} (add_mode={add_mode}) ---")
    sys.stdout.flush()

    accepts = {'add': 0, 'flip': 0, 'weight_resample': 0}
    t0 = time.time()

    pool = Pool(n_workers, initializer=init_w,
                initargs=(bp, ALL_DATA, 200, 2, input_projection, output_projection, bigram, add_mode))
    try:
        for step in range(1, max_steps+1):
            ptype = schedule[(step-1) % len(schedule)]
            if ptype in ('flip', 'weight_resample') and np.count_nonzero(mask) == 0:
                ptype = 'add'

            mask_flat = mask.flatten()
            args = [(mask_flat, H, 37000+step*50+w, ptype) for w in range(n_workers)]
            results = pool.map(worker_eval, args)

            best_r = max(results, key=lambda x: x['delta'])
            if best_r['delta'] > threshold:
                if best_r['new_mask_flat'] is not None:
                    mask = best_r['new_mask_flat'].reshape(H, H).astype(np.int32)
                    accepts[best_r['type']] += 1

            if step % 100 == 0:
                elapsed = time.time() - t0
                edges = int(np.count_nonzero(mask))
                ea = np.mean([eval_accuracy(mask, H, input_projection, output_projection, s, bp)
                              for s in eval_seqs])
                quality = ea / max(edges, 1) * 100
                vals = mask[mask != 0]
                vstr = f"[{vals.min()},{vals.max()}] std={vals.std():.0f}" if len(vals) else "none"

                print(f"  [{step:3d}] acc={ea*100:.2f}% edges={edges} q={quality:.3f}%/e "
                      f"A={accepts['add']}|F={accepts['flip']}|WR={accepts['weight_resample']} "
                      f"vals={vstr} {elapsed:.0f}s")
                sys.stdout.flush()
    finally:
        pool.terminate(); pool.join()

    edges = int(np.count_nonzero(mask))
    ea = np.mean([eval_accuracy(mask, H, input_projection, output_projection, s, bp) for s in eval_seqs])
    elapsed = time.time() - t0
    quality = ea / max(edges, 1) * 100
    vals = mask[mask != 0]

    print(f"  FINAL: acc={ea*100:.2f}% edges={edges} q={quality:.3f}%/e")
    if len(vals):
        print(f"  Weight dist: mean={vals.mean():.1f} std={vals.std():.1f} "
              f"[{vals.min()},{vals.max()}] unique={len(np.unique(vals))}")
    print(f"  {elapsed:.0f}s")
    sys.stdout.flush()
    return {'name': name, 'acc': ea, 'edges': edges, 'quality': quality,
            'accepts': dict(accepts), 'time': elapsed}


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

    # A: Ternary +-77 (baseline)
    results.append(run_config("INT8 ternary +-77", 'ternary',
                              ['add', 'add', 'flip', 'add'],
                              bp, ALL_DATA, bigram, eval_seqs, H, input_projection, output_projection))

    # B: Free weight — random [-128..+127] on add
    results.append(run_config("INT8 free add", 'free',
                              ['add', 'add', 'flip', 'add'],
                              bp, ALL_DATA, bigram, eval_seqs, H, input_projection, output_projection))

    # C: Free weight + weight resample on existing edges
    results.append(run_config("INT8 free + resample", 'free',
                              ['add', 'add', 'flip', 'weight_resample'],
                              bp, ALL_DATA, bigram, eval_seqs, H, input_projection, output_projection))

    print(f"\n{'='*65}")
    print(f"  SUMMARY -- INT8 FREE WEIGHT (500 steps, ret=217)")
    print(f"{'='*65}")
    print(f"  {'Name':<25} {'Acc%':>6} {'Edges':>6} {'Q(%/e)':>8} {'Time':>6}")
    print(f"  {'-'*25} {'-'*6} {'-'*6} {'-'*8} {'-'*6}")
    for r in results:
        print(f"  {r['name']:<25} {r['acc']*100:6.2f} {r['edges']:6d} {r['quality']:8.3f} {r['time']:5.0f}s")
    sys.stdout.flush()

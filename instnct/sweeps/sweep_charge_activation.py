"""
Charge Activation Sweep — ReLU vs Leaky ReLU vs none
======================================================
The clip [-1,+1] is dead code. But clip [0,+2] won (+3.7%).
Now test: hard ReLU vs leaky ReLU (different leak rates) on charge.
Bigram 2seq, thresh=0.00005, 200 steps from empty.
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
_charge_mode = 'none'  # 'none', 'relu', 'leaky_001', 'leaky_01', 'leaky_03'

def init_w(b, d, sl, nt, wi, wo, bg, mode):
    global _bp, _all_data, _seq_len, _n_train, _input_projection, _output_projection, _bigram, _charge_mode
    _bp, _all_data, _seq_len, _n_train = b, d, sl, nt
    _input_projection, _output_projection, _bigram = wi, wo, bg
    _charge_mode = mode

def make_bp(io_dim, seed=12345):
    rng = np.random.RandomState(seed)
    p = rng.randn(256, io_dim).astype(np.float32)
    p /= np.linalg.norm(p, axis=1, keepdims=True)
    return p

def apply_charge_activation(charge):
    if _charge_mode == 'none':
        return charge  # no clip at all (baseline)
    elif _charge_mode == 'relu':
        return np.maximum(charge, 0.0)
    elif _charge_mode == 'leaky_001':
        return np.where(charge > 0, charge, charge * 0.01)
    elif _charge_mode == 'leaky_01':
        return np.where(charge > 0, charge, charge * 0.1)
    elif _charge_mode == 'leaky_03':
        return np.where(charge > 0, charge, charge * 0.3)
    elif _charge_mode == 'leaky_05':
        return np.where(charge > 0, charge, charge * 0.5)
    return charge

def _eval_bigram(mask, H, theta, decay, seqs):
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
                charge = apply_charge_activation(charge)
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

    old_score = _eval_bigram(mask, H, theta, decay, seqs)
    new_score = _eval_bigram(new_mask, H, new_theta, decay, seqs)

    return {'delta': new_score - old_score, 'type': proposal_type,
            'new_mask_flat': new_mask.flatten() if new_score > old_score else None,
            'new_theta': new_theta if proposal_type == 'theta' else None}

def eval_accuracy_classic(mask, H, input_projection, output_projection, theta, decay, text_bytes, bp, charge_mode):
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
            if charge_mode == 'relu':
                charge = np.maximum(charge, 0.0)
            elif charge_mode == 'leaky_001':
                charge = np.where(charge > 0, charge, charge * 0.01)
            elif charge_mode == 'leaky_01':
                charge = np.where(charge > 0, charge, charge * 0.1)
            elif charge_mode == 'leaky_03':
                charge = np.where(charge > 0, charge, charge * 0.3)
            elif charge_mode == 'leaky_05':
                charge = np.where(charge > 0, charge, charge * 0.5)
        state = act.copy()
        out = charge @ output_projection
        out_n = out / (np.linalg.norm(out) + 1e-8)
        sims = out_n @ pat_norm.T
        if np.argmax(sims) == text_bytes[i+1]: correct += 1
        total += 1
    return correct/total if total else 0


def run_config(name, mode, bp, ALL_DATA, bigram, eval_seqs, H, input_projection, output_projection,
               n_steps=200, n_workers=18, threshold=0.00005):
    mask = np.zeros((H, H), dtype=np.float32)
    theta = np.full(H, 0.03, dtype=np.float32)
    decay = np.full(H, 0.15, dtype=np.float32)

    print(f"\n--- {name} (mode={mode}) ---")
    sys.stdout.flush()

    schedule = ['add', 'add', 'add', 'flip', 'theta', 'add']
    accepts = {'add': 0, 'flip': 0, 'theta': 0}
    t0 = time.time()

    pool = Pool(n_workers, initializer=init_w,
                initargs=(bp, ALL_DATA, 200, 2, input_projection, output_projection, bigram, mode))
    try:
        for step in range(1, n_steps+1):
            ptype = schedule[(step-1) % len(schedule)]
            if ptype in ('flip', 'theta') and np.count_nonzero(mask) == 0:
                ptype = 'add'
            mask_flat = mask.flatten()
            args = [(mask_flat, theta.copy(), decay.copy(), H,
                     12000+step*50+w, ptype) for w in range(n_workers)]
            results = pool.map(worker_eval, args)

            best_r = max(results, key=lambda x: x['delta'])
            if best_r['delta'] > threshold:
                if best_r['type'] in ('add', 'flip') and best_r['new_mask_flat'] is not None:
                    mask = best_r['new_mask_flat'].reshape(H, H)
                    accepts[best_r['type']] += 1
                elif best_r['type'] == 'theta' and best_r['new_theta'] is not None:
                    theta = best_r['new_theta']
                    accepts['theta'] += 1

            if step % 50 == 0:
                elapsed = time.time() - t0
                edges = int(np.count_nonzero(mask))
                ea = np.mean([eval_accuracy_classic(mask, H, input_projection, output_projection, theta, decay, s, bp, mode)
                              for s in eval_seqs])
                tot = sum(accepts.values())
                print(f"  [{step:3d}] acc={ea*100:.2f}% edges={edges} "
                      f"accepts={tot} [A={accepts['add']}|F={accepts['flip']}|T={accepts['theta']}] "
                      f"{elapsed:.0f}s")
                sys.stdout.flush()
    finally:
        pool.terminate(); pool.join()

    edges = int(np.count_nonzero(mask))
    ea = np.mean([eval_accuracy_classic(mask, H, input_projection, output_projection, theta, decay, s, bp, mode)
                  for s in eval_seqs])
    elapsed = time.time() - t0
    print(f"  FINAL: acc={ea*100:.2f}% edges={edges} accepts={sum(accepts.values())} {elapsed:.0f}s")
    sys.stdout.flush()
    return {'name': name, 'acc': ea, 'edges': edges, 'time': elapsed, 'accepts': dict(accepts)}


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

    # 1: No activation (current baseline — clip is dead code)
    results.append(run_config("NONE (baseline)", 'none',
                              bp, ALL_DATA, bigram, eval_seqs, H, input_projection, output_projection))

    # 2: Hard ReLU (what [0,+2] effectively did)
    results.append(run_config("RELU (hard)", 'relu',
                              bp, ALL_DATA, bigram, eval_seqs, H, input_projection, output_projection))

    # 3: Leaky 0.01 (almost ReLU, tiny negative)
    results.append(run_config("LEAKY 0.01", 'leaky_001',
                              bp, ALL_DATA, bigram, eval_seqs, H, input_projection, output_projection))

    # 4: Leaky 0.1 (some negative signal)
    results.append(run_config("LEAKY 0.1", 'leaky_01',
                              bp, ALL_DATA, bigram, eval_seqs, H, input_projection, output_projection))

    # 5: Leaky 0.3 (moderate negative)
    results.append(run_config("LEAKY 0.3", 'leaky_03',
                              bp, ALL_DATA, bigram, eval_seqs, H, input_projection, output_projection))

    # 6: Leaky 0.5 (half negative)
    results.append(run_config("LEAKY 0.5", 'leaky_05',
                              bp, ALL_DATA, bigram, eval_seqs, H, input_projection, output_projection))

    print(f"\n{'='*60}")
    print(f"  SUMMARY -- CHARGE ACTIVATION (200 steps, bigram 2seq)")
    print(f"{'='*60}")
    print(f"  {'Name':<18} {'Acc%':>6} {'Edges':>6} {'Flips':>6} {'Time':>6}")
    print(f"  {'-'*18} {'-'*6} {'-'*6} {'-'*6} {'-'*6}")
    for r in results:
        print(f"  {r['name']:<18} {r['acc']*100:6.2f} {r['edges']:6d} "
              f"{r['accepts'].get('flip',0):6d} {r['time']:5.0f}s")
    sys.stdout.flush()

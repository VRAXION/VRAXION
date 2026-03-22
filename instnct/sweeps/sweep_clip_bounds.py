"""
Clip Bounds Sweep — per-neuron learnable vs fixed
===================================================
Current: charge = clip(charge, -1, +1) for all neurons.
Test: different fixed bounds + per-neuron learnable bounds.
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
_clip_lo = None; _clip_hi = None  # per-neuron or scalar

def init_w(b, d, sl, nt, wi, wo, bg, clo, chi):
    global _bp, _all_data, _seq_len, _n_train, _input_projection, _output_projection, _bigram, _clip_lo, _clip_hi
    _bp, _all_data, _seq_len, _n_train = b, d, sl, nt
    _input_projection, _output_projection, _bigram = wi, wo, bg
    _clip_lo, _clip_hi = clo, chi

def make_bp(io_dim, seed=12345):
    rng = np.random.RandomState(seed)
    p = rng.randn(256, io_dim).astype(np.float32)
    p /= np.linalg.norm(p, axis=1, keepdims=True)
    return p

def _eval_bigram(mask, H, theta, decay, clip_lo, clip_hi, seqs):
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
                charge = np.clip(charge, clip_lo, clip_hi)
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
    mask_flat, theta, decay, clip_lo, clip_hi, H, seed, proposal_type = args
    rng = random.Random(seed)
    np_rng = np.random.RandomState(seed)
    mask = mask_flat.reshape(H, H)
    new_mask = mask; new_theta = theta
    new_clip_lo = clip_lo; new_clip_hi = clip_hi

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
    elif proposal_type == 'clip':
        # Mutate one neuron's clip bounds
        idx = rng.randint(0, H-1)
        new_clip_lo = clip_lo.copy()
        new_clip_hi = clip_hi.copy()
        if rng.random() < 0.5:
            new_clip_lo[idx] = max(-5.0, min(-0.1, clip_lo[idx] + rng.uniform(-0.2, 0.2)))
        else:
            new_clip_hi[idx] = max(0.1, min(5.0, clip_hi[idx] + rng.uniform(-0.2, 0.2)))

    seqs = []
    data_len = len(_all_data)
    for _ in range(_n_train):
        off = np_rng.randint(0, data_len - _seq_len)
        seqs.append(_all_data[off:off+_seq_len])

    old_score = _eval_bigram(mask, H, theta, decay, clip_lo, clip_hi, seqs)
    new_score = _eval_bigram(new_mask, H, new_theta, decay, new_clip_lo, new_clip_hi, seqs)

    return {'delta': new_score - old_score, 'type': proposal_type,
            'new_mask_flat': new_mask.flatten() if new_score > old_score else None,
            'new_theta': new_theta if proposal_type == 'theta' else None,
            'new_clip_lo': new_clip_lo if proposal_type == 'clip' else None,
            'new_clip_hi': new_clip_hi if proposal_type == 'clip' else None}

def eval_accuracy_classic(mask, H, input_projection, output_projection, theta, decay, clip_lo, clip_hi, text_bytes, bp):
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
            charge = np.clip(charge, clip_lo, clip_hi)
        state = act.copy()
        out = charge @ output_projection
        out_n = out / (np.linalg.norm(out) + 1e-8)
        sims = out_n @ pat_norm.T
        if np.argmax(sims) == text_bytes[i+1]: correct += 1
        total += 1
    return correct/total if total else 0


def run_config(name, clip_lo_init, clip_hi_init, learnable_clip,
               bp, ALL_DATA, bigram, eval_seqs, H, input_projection, output_projection,
               n_steps=200, n_workers=18, threshold=0.00005):
    mask = np.zeros((H, H), dtype=np.float32)
    theta = np.full(H, 0.03, dtype=np.float32)
    decay = np.full(H, 0.15, dtype=np.float32)

    if isinstance(clip_lo_init, (int, float)):
        clip_lo = np.full(H, clip_lo_init, dtype=np.float32)
        clip_hi = np.full(H, clip_hi_init, dtype=np.float32)
    else:
        clip_lo = clip_lo_init.copy()
        clip_hi = clip_hi_init.copy()

    print(f"\n--- {name} (lo={clip_lo[0]:.1f}, hi={clip_hi[0]:.1f}, learnable={learnable_clip}) ---")
    sys.stdout.flush()

    if learnable_clip:
        schedule = ['add', 'add', 'add', 'flip', 'theta', 'clip']
    else:
        schedule = ['add', 'add', 'add', 'flip', 'theta', 'add']

    accepts = {'add': 0, 'flip': 0, 'theta': 0, 'clip': 0}
    t0 = time.time()

    pool = Pool(n_workers, initializer=init_w,
                initargs=(bp, ALL_DATA, 200, 2, input_projection, output_projection, bigram, clip_lo, clip_hi))
    try:
        for step in range(1, n_steps+1):
            ptype = schedule[(step-1) % len(schedule)]
            if ptype in ('flip', 'theta', 'clip') and np.count_nonzero(mask) == 0:
                ptype = 'add'
            mask_flat = mask.flatten()
            args = [(mask_flat, theta.copy(), decay.copy(), clip_lo.copy(), clip_hi.copy(),
                     H, 11000+step*50+w, ptype) for w in range(n_workers)]
            results = pool.map(worker_eval, args)

            best_r = max(results, key=lambda x: x['delta'])
            if best_r['delta'] > threshold:
                if best_r['type'] in ('add', 'flip') and best_r['new_mask_flat'] is not None:
                    mask = best_r['new_mask_flat'].reshape(H, H)
                    accepts[best_r['type']] += 1
                elif best_r['type'] == 'theta' and best_r['new_theta'] is not None:
                    theta = best_r['new_theta']
                    accepts['theta'] += 1
                elif best_r['type'] == 'clip' and best_r['new_clip_lo'] is not None:
                    clip_lo = best_r['new_clip_lo']
                    clip_hi = best_r['new_clip_hi']
                    accepts['clip'] += 1

            if step % 50 == 0:
                elapsed = time.time() - t0
                edges = int(np.count_nonzero(mask))
                ea = np.mean([eval_accuracy_classic(mask, H, input_projection, output_projection, theta, decay,
                              clip_lo, clip_hi, s, bp) for s in eval_seqs])
                tot = sum(accepts.values())
                extra = ""
                if learnable_clip:
                    extra = f" clip=[{clip_lo.mean():.2f},{clip_hi.mean():.2f}]"
                print(f"  [{step:3d}] acc={ea*100:.2f}% edges={edges} "
                      f"accepts={tot} [A={accepts['add']}|F={accepts['flip']}|T={accepts['theta']}|C={accepts['clip']}]"
                      f"{extra} {elapsed:.0f}s")
                sys.stdout.flush()
    finally:
        pool.terminate(); pool.join()

    edges = int(np.count_nonzero(mask))
    ea = np.mean([eval_accuracy_classic(mask, H, input_projection, output_projection, theta, decay,
                  clip_lo, clip_hi, s, bp) for s in eval_seqs])
    elapsed = time.time() - t0
    print(f"  FINAL: acc={ea*100:.2f}% edges={edges} {elapsed:.0f}s")
    if learnable_clip:
        print(f"  CLIP: lo=[{clip_lo.mean():.3f}+/-{clip_lo.std():.3f}] "
              f"hi=[{clip_hi.mean():.3f}+/-{clip_hi.std():.3f}]")
    sys.stdout.flush()
    return {'name': name, 'acc': ea, 'edges': edges, 'time': elapsed,
            'clip_lo_mean': float(clip_lo.mean()), 'clip_hi_mean': float(clip_hi.mean())}


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

    # A: Baseline [-1, +1]
    results.append(run_config("FIX [-1,+1]", -1.0, 1.0, False,
                              bp, ALL_DATA, bigram, eval_seqs, H, input_projection, output_projection))

    # B: Bigger tank [-2, +2]
    results.append(run_config("FIX [-2,+2]", -2.0, 2.0, False,
                              bp, ALL_DATA, bigram, eval_seqs, H, input_projection, output_projection))

    # C: Smaller tank [-0.5, +0.5]
    results.append(run_config("FIX [-0.5,+0.5]", -0.5, 0.5, False,
                              bp, ALL_DATA, bigram, eval_seqs, H, input_projection, output_projection))

    # D: No clip (effectively infinite)
    results.append(run_config("FIX [-100,+100]", -100.0, 100.0, False,
                              bp, ALL_DATA, bigram, eval_seqs, H, input_projection, output_projection))

    # E: Asymmetric [0, +2] (only positive charge)
    results.append(run_config("FIX [0,+2]", 0.0, 2.0, False,
                              bp, ALL_DATA, bigram, eval_seqs, H, input_projection, output_projection))

    # F: Learnable per-neuron (start at [-1,+1])
    results.append(run_config("LEARN [-1,+1]", -1.0, 1.0, True,
                              bp, ALL_DATA, bigram, eval_seqs, H, input_projection, output_projection))

    print(f"\n{'='*65}")
    print(f"  SUMMARY -- CLIP BOUNDS (200 steps, bigram 2seq, thresh=5e-05)")
    print(f"{'='*65}")
    print(f"  {'Name':<22} {'Acc%':>6} {'Edges':>6} {'ClipLo':>7} {'ClipHi':>7} {'Time':>6}")
    print(f"  {'-'*22} {'-'*6} {'-'*6} {'-'*7} {'-'*7} {'-'*6}")
    for r in results:
        print(f"  {r['name']:<22} {r['acc']*100:6.2f} {r['edges']:6d} "
              f"{r['clip_lo_mean']:7.2f} {r['clip_hi_mean']:7.2f} {r['time']:5.0f}s")
    sys.stdout.flush()

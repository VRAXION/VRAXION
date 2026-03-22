"""
INJ_SCALE sweep — fair test from EMPTY network
================================================
3 configs × 100 steps, 10 seq (fast), empty start.
"""
import sys, os, time, random
import numpy as np
from multiprocessing import Pool

from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "model"))
from graph import SelfWiringGraph

_bp = None; _all_data = None; _seq_len = 200; _n_train = 10
_input_projection = None; _output_projection = None

def init_w(b, d, sl, nt, wi, wo):
    global _bp, _all_data, _seq_len, _n_train, _input_projection, _output_projection
    _bp, _all_data, _seq_len, _n_train = b, d, sl, nt
    _input_projection, _output_projection = wi, wo

def make_bp(io_dim, seed=12345):
    rng = np.random.RandomState(seed)
    p = rng.randn(256, io_dim).astype(np.float32)
    p /= np.linalg.norm(p, axis=1, keepdims=True)
    return p

def _eval_on_seqs(mask, H, theta, decay, seqs):
    rs, cs = np.where(mask != 0)
    sp_vals = mask[rs, cs]
    pat_norm = _bp / (np.linalg.norm(_bp, axis=1, keepdims=True) + 1e-8)
    ret = 1.0 - decay
    total = 0.0
    for text_bytes in seqs:
        state = np.zeros(H, dtype=np.float32)
        charge = np.zeros(H, dtype=np.float32)
        correct = 0; prob_sum = 0.0; n = 0
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
                charge = np.maximum(charge, 0.0)
            state = act.copy()
            out = charge @ _output_projection
            out_n = out / (np.linalg.norm(out) + 1e-8)
            sims = out_n @ pat_norm.T
            e = np.exp(sims - sims.max())
            probs = e / e.sum()
            target = text_bytes[i+1]
            if np.argmax(probs) == target: correct += 1
            prob_sum += probs[target]; n += 1
        acc = correct/n if n else 0
        avg_p = prob_sum/n if n else 0
        total += 0.5*acc + 0.5*avg_p
    return total / len(seqs)

def worker_eval(args):
    mask_flat, theta, decay, H, seed, proposal_type, drive_mag = args
    rng = random.Random(seed)
    np_rng = np.random.RandomState(seed)
    mask = mask_flat.reshape(H, H)
    new_mask = mask; new_theta = theta; new_decay = decay

    if proposal_type == 'add':
        r = rng.randint(0, H-1); c = rng.randint(0, H-1)
        if r == c or mask[r, c] != 0:
            return {'delta': -1e9, 'type': 'add'}
        val = drive_mag if rng.random() < 0.5 else -drive_mag
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

    old_score = _eval_on_seqs(mask, H, theta, decay, seqs)
    new_score = _eval_on_seqs(new_mask, H, new_theta, new_decay, seqs)

    return {'delta': new_score - old_score, 'type': proposal_type,
            'new_mask_flat': new_mask.flatten() if new_score > old_score else None,
            'new_theta': new_theta if proposal_type == 'theta' else None}

def eval_accuracy(mask, H, input_projection, output_projection, theta, decay, text_bytes, bp):
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
            charge = np.maximum(charge, 0.0)
        state = act.copy()
        out = charge @ output_projection
        out_n = out / (np.linalg.norm(out) + 1e-8)
        sims = out_n @ pat_norm.T
        if np.argmax(sims) == text_bytes[i+1]: correct += 1
        total += 1
    return correct/total if total else 0


def run_config(name, scale, theta_init, drive_mag, bp, ALL_DATA, eval_seqs, H, input_projection_raw, output_projection_raw,
               n_steps=100, n_workers=18, threshold=0.0005):
    input_projection = input_projection_raw * scale
    output_projection = output_projection_raw * scale

    # Empty network
    mask = np.zeros((H, H), dtype=np.float32)
    theta = np.full(H, theta_init, dtype=np.float32)
    decay = np.full(H, 0.15, dtype=np.float32)

    print(f"\n--- {name} (scale={scale}, theta_init={theta_init}, drive={drive_mag}) ---")
    sys.stdout.flush()

    schedule = ['add', 'add', 'add', 'flip', 'theta', 'add']
    accepts = {'add': 0, 'flip': 0, 'theta': 0}
    t0 = time.time()

    pool = Pool(n_workers, initializer=init_w, initargs=(bp, ALL_DATA, 200, 10, input_projection, output_projection))
    try:
        for step in range(1, n_steps+1):
            ptype = schedule[(step-1) % len(schedule)]
            # Skip flip/theta if no edges yet
            if ptype in ('flip', 'theta') and np.count_nonzero(mask) == 0:
                ptype = 'add'
            mask_flat = mask.flatten()
            args = [(mask_flat, theta.copy(), decay.copy(), H,
                     4000+step*50+w, ptype, drive_mag) for w in range(n_workers)]
            results = pool.map(worker_eval, args)

            best_r = max(results, key=lambda x: x['delta'])
            if best_r['delta'] > threshold:
                if best_r['type'] in ('add', 'flip') and best_r['new_mask_flat'] is not None:
                    mask = best_r['new_mask_flat'].reshape(H, H)
                    accepts[best_r['type']] += 1
                elif best_r['type'] == 'theta' and best_r['new_theta'] is not None:
                    theta = best_r['new_theta']
                    accepts['theta'] += 1

            if step % 25 == 0:
                elapsed = time.time() - t0
                edges = int(np.count_nonzero(mask))
                ea = np.mean([eval_accuracy(mask, H, input_projection, output_projection, theta, decay, s, bp)
                              for s in eval_seqs])
                tot = sum(accepts.values())
                print(f"  [{step:3d}] eval={ea*100:.2f}% edges={edges} "
                      f"acc={tot} [A={accepts['add']}|F={accepts['flip']}|T={accepts['theta']}] "
                      f"theta={theta.mean():.4f} {elapsed:.0f}s")
                sys.stdout.flush()
    finally:
        pool.terminate(); pool.join()

    edges = int(np.count_nonzero(mask))
    ea = np.mean([eval_accuracy(mask, H, input_projection, output_projection, theta, decay, s, bp)
                  for s in eval_seqs])
    elapsed = time.time() - t0
    vals = mask[mask != 0]
    print(f"  FINAL: eval={ea*100:.2f}% edges={edges} "
          f"theta={theta.mean():.4f}+/-{theta.std():.4f} {elapsed:.0f}s")
    sys.stdout.flush()
    return {'name': name, 'eval': ea, 'edges': edges, 'time': elapsed, 'accepts': dict(accepts)}


if __name__ == "__main__":
    IO = 256; NV = 4; H = IO * NV
    bp = make_bp(IO)

    from lib.data import load_fineweb_bytes, resolve_fineweb_path
    DATA = resolve_fineweb_path()
    ALL_DATA = load_fineweb_bytes()
    print(f"Loaded {len(ALL_DATA)/1e6:.1f} MB text")

    eval_rng = np.random.RandomState(9999)
    eval_seqs = [ALL_DATA[off:off+200] for off in
                 [eval_rng.randint(0, len(ALL_DATA)-200) for _ in range(10)]]

    # Fixed raw projections (unit-norm, no scaling)
    random.seed(42); np.random.seed(42)
    proj_rng = np.random.RandomState(np.random.randint(0, 2**31))
    input_projection_raw = proj_rng.randn(IO, H).astype(np.float32)
    input_projection_raw /= np.linalg.norm(input_projection_raw, axis=1, keepdims=True)
    output_projection_raw = proj_rng.randn(H, IO).astype(np.float32)
    output_projection_raw /= np.linalg.norm(output_projection_raw, axis=0, keepdims=True)

    results = []

    # Config A: current (scale=3, theta=0.1, drive=0.6)
    results.append(run_config("A: scale=3 theta=0.1", 3.0, 0.1, 0.6,
                              bp, ALL_DATA, eval_seqs, H, input_projection_raw, output_projection_raw))

    # Config B: no hack (scale=1, theta=0.03, drive=0.6)
    results.append(run_config("B: scale=1 theta=0.03", 1.0, 0.03, 0.6,
                              bp, ALL_DATA, eval_seqs, H, input_projection_raw, output_projection_raw))

    # Config C: no hack, stronger drive (scale=1, theta=0.03, drive=1.0)
    results.append(run_config("C: scale=1 drive=1.0", 1.0, 0.03, 1.0,
                              bp, ALL_DATA, eval_seqs, H, input_projection_raw, output_projection_raw))

    # Config D: big scale (scale=5, theta=0.1, drive=0.6)
    results.append(run_config("D: scale=5 theta=0.1", 5.0, 0.1, 0.6,
                              bp, ALL_DATA, eval_seqs, H, input_projection_raw, output_projection_raw))

    # Config E: scale=1, theta=0.003 (super sensitive)
    results.append(run_config("E: scale=1 theta=0.003", 1.0, 0.003, 0.6,
                              bp, ALL_DATA, eval_seqs, H, input_projection_raw, output_projection_raw))

    print(f"\n{'='*60}")
    print(f"  SUMMARY — INJ_SCALE from EMPTY (100 steps, 10 seq)")
    print(f"{'='*60}")
    print(f"  {'Name':<25} {'Eval%':>6} {'Edges':>6} {'Time':>6}")
    print(f"  {'-'*25} {'-'*6} {'-'*6} {'-'*6}")
    for r in results:
        print(f"  {r['name']:<25} {r['eval']*100:6.2f} {r['edges']:6d} {r['time']:5.0f}s")
    sys.stdout.flush()



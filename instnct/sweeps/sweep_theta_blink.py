"""
Theta Blink Sweep — dark matter neuron oscillation
====================================================
3 configs from EMPTY: fix theta vs min/max blink vs phi rotation.
Scale=1.0 (no hack), 100 steps, 10 seq.
"""
import sys, os, time, random, math
import numpy as np
from multiprocessing import Pool

from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "model"))
from graph import SelfWiringGraph

_bp = None; _all_data = None; _seq_len = 200; _n_train = 10
_input_projection = None; _output_projection = None
_theta_mode = 'fixed'  # 'fixed', 'blink', 'phi'
_theta_base = 0.03
_theta_min = 0.01
_theta_max = 0.15
PHI = (1 + math.sqrt(5)) / 2  # golden ratio

def init_w(b, d, sl, nt, wi, wo, mode, tbase, tmin, tmax):
    global _bp, _all_data, _seq_len, _n_train, _input_projection, _output_projection
    global _theta_mode, _theta_base, _theta_min, _theta_max
    _bp, _all_data, _seq_len, _n_train = b, d, sl, nt
    _input_projection, _output_projection = wi, wo
    _theta_mode, _theta_base, _theta_min, _theta_max = mode, tbase, tmin, tmax

def make_bp(io_dim, seed=12345):
    rng = np.random.RandomState(seed)
    p = rng.randn(256, io_dim).astype(np.float32)
    p /= np.linalg.norm(p, axis=1, keepdims=True)
    return p

def get_effective_theta(theta_learned, active_mask, tick, H):
    """Get effective theta: learned for active neurons, oscillating for dark matter."""
    if _theta_mode == 'fixed':
        return theta_learned

    # Which neurons have at least one incoming edge?
    has_incoming = active_mask
    theta_eff = theta_learned.copy()

    if _theta_mode == 'blink':
        # Min/max alternation
        if tick % 2 == 0:
            theta_eff[~has_incoming] = _theta_min
        else:
            theta_eff[~has_incoming] = _theta_max

    elif _theta_mode == 'phi':
        # Irrational rotation per neuron — vectorized
        dark_idx = np.where(~has_incoming)[0]
        phases = (dark_idx * PHI + tick * PHI) % 1.0
        theta_eff[dark_idx] = _theta_min + (_theta_max - _theta_min) * phases

    elif _theta_mode == 'phi_global':
        # Same phi phase for ALL dark neurons (sync attempt)
        phase = (tick * PHI) % 1.0
        theta_eff[~has_incoming] = _theta_min + (_theta_max - _theta_min) * phase

    elif _theta_mode == 'random':
        # Fresh random each tick
        dark_idx = np.where(~has_incoming)[0]
        theta_eff[dark_idx] = np.random.uniform(_theta_min, _theta_max, len(dark_idx)).astype(np.float32)

    return theta_eff

def _eval_on_seqs(mask, H, theta, decay, seqs):
    rs, cs = np.where(mask != 0)
    sp_vals = mask[rs, cs]
    # Precompute which neurons have incoming edges
    has_incoming = np.zeros(H, dtype=bool)
    if len(cs):
        has_incoming[np.unique(cs)] = True
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
                # Use effective theta (blink/phi for dark neurons)
                th_eff = get_effective_theta(theta, has_incoming, t + i*6, H)
                act = np.maximum(charge - th_eff, 0.0)
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

    old_score = _eval_on_seqs(mask, H, theta, decay, seqs)
    new_score = _eval_on_seqs(new_mask, H, new_theta, decay, seqs)

    return {'delta': new_score - old_score, 'type': proposal_type,
            'new_mask_flat': new_mask.flatten() if new_score > old_score else None,
            'new_theta': new_theta if proposal_type == 'theta' else None}

def eval_accuracy_with_blink(mask, H, input_projection, output_projection, theta, decay, text_bytes, bp, mode, tmin, tmax):
    pat_norm = bp / (np.linalg.norm(bp, axis=1, keepdims=True) + 1e-8)
    rs, cs = np.where(mask != 0); sp_vals = mask[rs, cs]
    has_incoming = np.zeros(H, dtype=bool)
    if len(cs): has_incoming[np.unique(cs)] = True
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
            th_eff = theta.copy()
            dark = ~has_incoming
            if mode == 'blink':
                th_eff[dark] = tmin if (t + i*6) % 2 == 0 else tmax
            elif mode == 'phi':
                dark_idx = np.where(dark)[0]
                phases = (dark_idx * PHI + (t + i*6) * PHI) % 1.0
                th_eff[dark_idx] = tmin + (tmax - tmin) * phases
            elif mode == 'phi_global':
                phase = ((t + i*6) * PHI) % 1.0
                th_eff[dark] = tmin + (tmax - tmin) * phase
            elif mode == 'random':
                dark_idx = np.where(dark)[0]
                th_eff[dark_idx] = np.random.uniform(tmin, tmax, len(dark_idx)).astype(np.float32)
            act = np.maximum(charge - th_eff, 0.0)
            charge = np.maximum(charge, 0.0)
        state = act.copy()
        out = charge @ output_projection
        out_n = out / (np.linalg.norm(out) + 1e-8)
        sims = out_n @ pat_norm.T
        if np.argmax(sims) == text_bytes[i+1]: correct += 1
        total += 1
    return correct/total if total else 0


def run_config(name, mode, theta_init, theta_min, theta_max,
               bp, ALL_DATA, eval_seqs, H, input_projection, output_projection,
               n_steps=100, n_workers=18, threshold=0.0005):
    mask = np.zeros((H, H), dtype=np.float32)
    theta = np.full(H, theta_init, dtype=np.float32)
    decay = np.full(H, 0.15, dtype=np.float32)

    print(f"\n--- {name} (mode={mode}, init={theta_init}, range=[{theta_min},{theta_max}]) ---")
    sys.stdout.flush()

    schedule = ['add', 'add', 'add', 'flip', 'theta', 'add']
    accepts = {'add': 0, 'flip': 0, 'theta': 0}
    t0 = time.time()

    pool = Pool(n_workers, initializer=init_w,
                initargs=(bp, ALL_DATA, 200, 10, input_projection, output_projection, mode, theta_init, theta_min, theta_max))
    try:
        for step in range(1, n_steps+1):
            ptype = schedule[(step-1) % len(schedule)]
            if ptype in ('flip', 'theta') and np.count_nonzero(mask) == 0:
                ptype = 'add'
            mask_flat = mask.flatten()
            args = [(mask_flat, theta.copy(), decay.copy(), H,
                     5000+step*50+w, ptype) for w in range(n_workers)]
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
                ea = np.mean([eval_accuracy_with_blink(mask, H, input_projection, output_projection, theta, decay, s, bp,
                              mode, theta_min, theta_max) for s in eval_seqs])
                tot = sum(accepts.values())
                print(f"  [{step:3d}] eval={ea*100:.2f}% edges={edges} "
                      f"acc={tot} [A={accepts['add']}|F={accepts['flip']}|T={accepts['theta']}] "
                      f"theta_mean={theta.mean():.4f} {elapsed:.0f}s")
                sys.stdout.flush()
    finally:
        pool.terminate(); pool.join()

    edges = int(np.count_nonzero(mask))
    ea = np.mean([eval_accuracy_with_blink(mask, H, input_projection, output_projection, theta, decay, s, bp,
                  mode, theta_min, theta_max) for s in eval_seqs])
    elapsed = time.time() - t0
    print(f"  FINAL: eval={ea*100:.2f}% edges={edges} theta_mean={theta.mean():.4f} {elapsed:.0f}s")
    sys.stdout.flush()
    return {'name': name, 'eval': ea, 'edges': edges, 'time': elapsed}


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

    # Fixed raw projections, scale=1.0 (winner from prev sweep)
    random.seed(42); np.random.seed(42)
    proj_rng = np.random.RandomState(np.random.randint(0, 2**31))
    input_projection_raw = proj_rng.randn(IO, H).astype(np.float32)
    input_projection_raw /= np.linalg.norm(input_projection_raw, axis=1, keepdims=True)
    output_projection_raw = proj_rng.randn(H, IO).astype(np.float32)
    output_projection_raw /= np.linalg.norm(output_projection_raw, axis=0, keepdims=True)
    input_projection = input_projection_raw * 1.0  # NO HACK
    output_projection = output_projection_raw * 1.0

    results = []

    # 1. Baseline: fixed theta=0.03
    results.append(run_config("FIXED 0.03", 'fixed', 0.03, 0.01, 0.15,
                              bp, ALL_DATA, eval_seqs, H, input_projection, output_projection))

    # 2. Blink: min/max alternation
    results.append(run_config("BLINK min/max", 'blink', 0.03, 0.01, 0.15,
                              bp, ALL_DATA, eval_seqs, H, input_projection, output_projection))

    # 3. Phi rotation: per-neuron unique phase
    results.append(run_config("PHI per-neuron", 'phi', 0.03, 0.01, 0.15,
                              bp, ALL_DATA, eval_seqs, H, input_projection, output_projection))

    # 4. Phi global: all dark neurons same phase (sync)
    results.append(run_config("PHI global sync", 'phi_global', 0.03, 0.01, 0.15,
                              bp, ALL_DATA, eval_seqs, H, input_projection, output_projection))

    # 5. Random: fresh random theta each tick
    results.append(run_config("RANDOM each tick", 'random', 0.03, 0.01, 0.15,
                              bp, ALL_DATA, eval_seqs, H, input_projection, output_projection))

    print(f"\n{'='*60}")
    print(f"  SUMMARY — THETA BLINK SWEEP (scale=1.0, 100 steps)")
    print(f"{'='*60}")
    print(f"  {'Name':<22} {'Eval%':>6} {'Edges':>6} {'Time':>6}")
    print(f"  {'-'*22} {'-'*6} {'-'*6} {'-'*6}")
    for r in results:
        print(f"  {r['name']:<22} {r['eval']*100:6.2f} {r['edges']:6d} {r['time']:5.0f}s")
    sys.stdout.flush()



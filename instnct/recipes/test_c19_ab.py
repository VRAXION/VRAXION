"""
C19 Activation A/B Test
=======================
Head-to-head comparison:
  A) ReLU baseline: act = max(charge - theta, 0)
  B) C19 periodic:  act = c19(charge - theta, C)  with learnable C, rho=4.0 fixed

Uses alice.txt (170KB) — no fineweb dependency.
256 neurons, 4 workers, 200 steps — quick smoke test.
Same seed, same data, same projections for both conditions.
"""
import sys, os, time, random, json
import numpy as np
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

# ── C19 activation ──────────────────────────────────────────────────
_RHO = 4.0  # fixed wave shape

def c19_activation(x, C):
    """Periodic parabolic wave. rho=4.0 fixed, C per-neuron."""
    inv_c = 1.0 / C
    l = 6.0 * C
    scaled = x * inv_c
    n = np.floor(scaled)
    t = scaled - n
    h = t * (1.0 - t)
    sgn = np.where(np.remainder(n, 2.0) < 1.0, 1.0, -1.0)
    core = C * (sgn * h + _RHO * h * h)
    return np.where(x >= l, x - l, np.where(x <= -l, x + l, core))

# ── Shared eval globals (for multiprocessing) ───────────────────────
_bp = None; _all_data = None; _seq_len = 200; _n_train = 2
_bigram = None; _use_c19 = False

def init_w(bp, data, sl, nt, bg, use_c19):
    global _bp, _all_data, _seq_len, _n_train, _bigram, _use_c19
    _bp, _all_data, _seq_len, _n_train = bp, data, sl, nt
    _bigram, _use_c19 = bg, use_c19

def make_bp(io_dim, seed=12345):
    rng = np.random.RandomState(seed)
    p = rng.randn(256, io_dim).astype(np.float32)
    p /= np.linalg.norm(p, axis=1, keepdims=True)
    return p

def compute_bigram_from_bytes(data_bytes):
    """Compute 256x256 bigram probability table from raw bytes."""
    bigram = np.zeros((256, 256), dtype=np.float64)
    for i in range(len(data_bytes) - 1):
        bigram[data_bytes[i], data_bytes[i + 1]] += 1
    row_sums = bigram.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    bigram /= row_sums
    return bigram.astype(np.float32)

# ── Eval function ───────────────────────────────────────────────────
def _eval_bigram(mask, H, input_proj, output_proj, theta, decay, C_param, seqs):
    rs, cs = np.where(mask != 0)
    sp_vals = mask[rs, cs]
    pat_norm = _bp / (np.linalg.norm(_bp, axis=1, keepdims=True) + 1e-8)
    ret = 1.0 - decay
    total = 0.0
    for text_bytes in seqs:
        state = np.zeros(H, dtype=np.float32)
        charge = np.zeros(H, dtype=np.float32)
        seq_score = 0.0; n = 0
        for i in range(len(text_bytes) - 1):
            act = state.copy()
            for t in range(8):
                if t < 2:
                    act = act + _bp[text_bytes[i]] @ input_proj
                raw = np.zeros(H, dtype=np.float32)
                if len(rs):
                    np.add.at(raw, cs, act[rs] * sp_vals)
                charge += raw; charge *= ret
                if _use_c19:
                    act = c19_activation(charge - theta, C_param)
                else:
                    act = np.maximum(charge - theta, 0.0)
                charge = np.maximum(charge, 0.0)
            state = act.copy()
            out = charge @ output_proj
            out_n = out / (np.linalg.norm(out) + 1e-8)
            sims = out_n @ pat_norm.T
            e = np.exp(sims - sims.max())
            pred = e / e.sum()
            target_dist = _bigram[text_bytes[i]]
            cos = np.dot(pred, target_dist) / (np.linalg.norm(pred) * np.linalg.norm(target_dist) + 1e-8)
            seq_score += cos; n += 1
        total += seq_score / n if n else 0
    return total / len(seqs)

# ── Worker ──────────────────────────────────────────────────────────
def worker_eval(args):
    mask_flat, theta, decay, C_param, H, input_proj, output_proj, seed, proposal_type = args
    rng = random.Random(seed)
    np_rng = np.random.RandomState(seed)
    mask = mask_flat.reshape(H, H)
    new_mask = mask; new_theta = theta; new_decay = decay; new_C = C_param

    if proposal_type == 'add':
        r = rng.randint(0, H - 1); c = rng.randint(0, H - 1)
        if r == c or mask[r, c] != 0:
            return {'delta': -1e9, 'type': 'add'}
        val = 0.6 if rng.random() < 0.5 else -0.6
        new_mask = mask.copy(); new_mask[r, c] = val
    elif proposal_type == 'flip':
        alive = list(zip(*np.where(mask != 0)))
        if not alive:
            return {'delta': -1e9, 'type': 'flip'}
        r, c = alive[rng.randint(0, len(alive) - 1)]
        new_mask = mask.copy(); new_mask[r, c] = -mask[r, c]
    elif proposal_type == 'decay':
        idx = rng.randint(0, H - 1)
        new_decay = decay.copy()
        new_decay[idx] = max(0.01, min(0.5, decay[idx] + rng.uniform(-0.03, 0.03)))
    elif proposal_type == 'cparam':
        idx = rng.randint(0, H - 1)
        new_C = C_param.copy()
        new_C[idx] = rng.uniform(0.2, 5.0)

    seqs = []
    data_len = len(_all_data)
    for _ in range(_n_train):
        off = np_rng.randint(0, data_len - _seq_len)
        seqs.append(_all_data[off:off + _seq_len])

    old_score = _eval_bigram(mask, H, input_proj, output_proj, theta, decay, C_param, seqs)
    new_score = _eval_bigram(new_mask, H, input_proj, output_proj, new_theta, new_decay, new_C, seqs)

    return {
        'delta': new_score - old_score, 'type': proposal_type,
        'new_mask_flat': new_mask.flatten() if proposal_type in ('add', 'flip') and new_score > old_score else None,
        'new_theta': new_theta if proposal_type == 'theta' else None,
        'new_decay': new_decay if proposal_type == 'decay' else None,
        'new_C': new_C if proposal_type == 'cparam' else None,
    }

# ── Accuracy eval (reporting only) ─────────────────────────────────
def eval_accuracy(mask, H, input_proj, output_proj, theta, decay, C_param, text_bytes, bp, use_c19):
    pat_norm = bp / (np.linalg.norm(bp, axis=1, keepdims=True) + 1e-8)
    rs, cs = np.where(mask != 0); sp_vals = mask[rs, cs]
    ret = 1.0 - decay
    state = np.zeros(H, dtype=np.float32); charge = np.zeros(H, dtype=np.float32)
    correct = 0; total = 0
    for i in range(len(text_bytes) - 1):
        act = state.copy()
        for t in range(8):
            if t < 2:
                act = act + bp[text_bytes[i]] @ input_proj
            raw = np.zeros(H, dtype=np.float32)
            if len(rs):
                np.add.at(raw, cs, act[rs] * sp_vals)
            charge += raw; charge *= ret
            if use_c19:
                act = c19_activation(charge - theta, C_param)
            else:
                act = np.maximum(charge - theta, 0.0)
            charge = np.maximum(charge, 0.0)
        state = act.copy()
        out = charge @ output_proj
        out_n = out / (np.linalg.norm(out) + 1e-8)
        sims = out_n @ pat_norm.T
        if np.argmax(sims) == text_bytes[i + 1]:
            correct += 1
        total += 1
    return correct / total if total else 0

# ── Single training run ─────────────────────────────────────────────
def run_condition(label, use_c19, all_data, bigram, bp, H, IO, n_workers, budget, threshold, seed):
    print(f"\n{'='*60}")
    print(f"  Condition: {label}")
    print(f"  Activation: {'C19 (rho=4.0, C learnable)' if use_c19 else 'ReLU (max(charge-theta, 0))'}")
    print(f"  {H}n, {n_workers}w, {budget} steps, threshold={threshold}")
    print(f"{'='*60}")
    sys.stdout.flush()

    random.seed(seed); np.random.seed(seed)

    # Build projections (scale=1.0)
    proj_rng = np.random.RandomState(seed)
    input_proj = proj_rng.randn(256, H).astype(np.float32)
    input_proj /= np.linalg.norm(input_proj, axis=1, keepdims=True)
    output_proj = proj_rng.randn(H, 256).astype(np.float32)
    output_proj /= np.linalg.norm(output_proj, axis=0, keepdims=True)

    # Empty mask
    mask = np.zeros((H, H), dtype=np.float32)
    theta = np.zeros(H, dtype=np.float32)
    decay_rng = np.random.RandomState(99)
    decay = decay_rng.uniform(0.08, 0.24, H).astype(np.float32)
    C_param = np.full(H, 1.0, dtype=np.float32)

    # Eval sequences (fixed)
    eval_rng = np.random.RandomState(9999)
    eval_seqs = [all_data[off:off + 200]
                 for off in [eval_rng.randint(0, len(all_data) - 200) for _ in range(5)]]

    # Schedule: C19 gets cparam steps, ReLU uses decay instead
    if use_c19:
        schedule = ['add', 'add', 'flip', 'decay', 'decay', 'decay', 'add', 'add', 'cparam']
    else:
        schedule = ['add', 'add', 'flip', 'decay', 'decay', 'decay', 'decay', 'decay']

    add_acc = 0; flip_acc = 0; decay_acc = 0; c_acc = 0
    accepts = 0
    log_data = []
    c_snapshots = {}
    t0 = time.time()

    pool = Pool(n_workers, initializer=init_w,
                initargs=(bp, all_data, 200, 2, bigram, use_c19))
    try:
        for step in range(1, budget + 1):
            ptype = schedule[(step - 1) % len(schedule)]
            edges = int(np.count_nonzero(mask))
            if ptype in ('flip', 'decay', 'cparam') and edges == 0:
                ptype = 'add'

            mask_flat = mask.flatten()
            args = [(mask_flat, theta.copy(), decay.copy(), C_param.copy(), H,
                     input_proj, output_proj, 1000 + step * 50 + w, ptype)
                    for w in range(n_workers)]
            results = pool.map(worker_eval, args)

            best_r = max(results, key=lambda x: x['delta'])
            if best_r['delta'] > threshold:
                if best_r['type'] in ('add', 'flip') and best_r['new_mask_flat'] is not None:
                    mask[:] = best_r['new_mask_flat'].reshape(H, H)
                    if best_r['type'] == 'add': add_acc += 1
                    else: flip_acc += 1
                elif best_r['type'] == 'decay' and best_r['new_decay'] is not None:
                    decay[:] = best_r['new_decay']
                    decay_acc += 1
                elif best_r['type'] == 'cparam' and best_r['new_C'] is not None:
                    C_param[:] = best_r['new_C']
                    c_acc += 1
                accepts += 1

            # Snapshots of C at key steps
            if use_c19 and step in (1, 50, 100, 150, 200):
                c_snapshots[step] = C_param.copy()

            if step % 20 == 0:
                elapsed = time.time() - t0
                ea = np.mean([eval_accuracy(mask, H, input_proj, output_proj, theta, decay,
                                            C_param, s, bp, use_c19) for s in eval_seqs])
                edges = int(np.count_nonzero(mask))
                sps = step / elapsed

                c_info = ""
                if use_c19:
                    c_info = (f" C={C_param.mean():.3f}+/-{C_param.std():.3f}"
                              f" [min={C_param.min():.2f} max={C_param.max():.2f}]")

                line = (f"  [{step:4d}] eval={ea*100:.2f}% edges={edges} "
                        f"[A={add_acc}|F={flip_acc}|D={decay_acc}|C={c_acc}]"
                        f" decay={decay.mean():.4f}{c_info}"
                        f" ({sps:.1f} step/s)")
                print(line)
                sys.stdout.flush()

                log_data.append({
                    'step': step, 'eval_pct': round(ea * 100, 2), 'edges': edges,
                    'add_acc': add_acc, 'flip_acc': flip_acc, 'decay_acc': decay_acc, 'c_acc': c_acc,
                    'decay_mean': round(float(decay.mean()), 4),
                    'C_mean': round(float(C_param.mean()), 4) if use_c19 else None,
                    'C_std': round(float(C_param.std()), 4) if use_c19 else None,
                    'C_min': round(float(C_param.min()), 2) if use_c19 else None,
                    'C_max': round(float(C_param.max()), 2) if use_c19 else None,
                    'sps': round(sps, 1),
                })

    finally:
        pool.terminate(); pool.join()

    elapsed = time.time() - t0
    final_ea = np.mean([eval_accuracy(mask, H, input_proj, output_proj, theta, decay,
                                      C_param, s, bp, use_c19) for s in eval_seqs])
    edges = int(np.count_nonzero(mask))

    result = {
        'label': label,
        'final_eval_pct': round(final_ea * 100, 2),
        'edges': edges,
        'accepts': accepts,
        'add_acc': add_acc, 'flip_acc': flip_acc, 'decay_acc': decay_acc, 'c_acc': c_acc,
        'decay_mean': round(float(decay.mean()), 4),
        'elapsed': round(elapsed, 1),
        'log': log_data,
    }
    if use_c19:
        result['C_mean'] = round(float(C_param.mean()), 4)
        result['C_std'] = round(float(C_param.std()), 4)
        result['C_min'] = round(float(C_param.min()), 2)
        result['C_max'] = round(float(C_param.max()), 2)
        result['C_final_vector'] = C_param.tolist()
        result['C_snapshots'] = {str(k): v.tolist() for k, v in c_snapshots.items()}

        # C distribution histogram
        bins = [0.2, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0]
        hist, _ = np.histogram(C_param, bins=bins)
        result['C_histogram'] = {f"{bins[i]:.1f}-{bins[i+1]:.1f}": int(hist[i]) for i in range(len(hist))}

        # C vs connectivity correlation
        degrees = np.count_nonzero(mask, axis=0) + np.count_nonzero(mask, axis=1)
        connected = degrees > 0
        if connected.sum() > 2:
            corr = np.corrcoef(C_param[connected], degrees[connected])[0, 1]
            result['C_degree_correlation'] = round(float(corr), 3)
            result['C_connected_mean'] = round(float(C_param[connected].mean()), 3)
            result['C_disconnected_mean'] = round(float(C_param[~connected].mean()), 3) if (~connected).sum() > 0 else None

    return result


# ── Main ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    IO = 256
    H = 256          # hidden_ratio=1
    N_WORKERS = 4
    BUDGET = 200
    THRESHOLD = 0.00005
    SEED = 42

    print("Loading alice.txt...")
    alice_path = DATA_DIR / "alice.txt"
    with open(alice_path, "rb") as f:
        all_data = np.frombuffer(f.read(), dtype=np.uint8)
    print(f"  {len(all_data)} bytes loaded")

    print("Computing bigram table from alice.txt...")
    bigram = compute_bigram_from_bytes(all_data)
    print(f"  bigram shape: {bigram.shape}")

    bp = make_bp(IO)

    # ── Run both conditions ─────────────────────────────────────────
    result_relu = run_condition("A: ReLU baseline", use_c19=False,
                                all_data=all_data, bigram=bigram, bp=bp,
                                H=H, IO=IO, n_workers=N_WORKERS,
                                budget=BUDGET, threshold=THRESHOLD, seed=SEED)

    result_c19 = run_condition("B: C19 (rho=4, C learnable)", use_c19=True,
                               all_data=all_data, bigram=bigram, bp=bp,
                               H=H, IO=IO, n_workers=N_WORKERS,
                               budget=BUDGET, threshold=THRESHOLD, seed=SEED)

    # ── Final comparison ────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  FINAL COMPARISON")
    print(f"{'='*60}")
    print(f"  {'Metric':<25} {'ReLU':>12} {'C19':>12}")
    print(f"  {'-'*25} {'-'*12} {'-'*12}")
    print(f"  {'Eval accuracy':<25} {result_relu['final_eval_pct']:>11.2f}% {result_c19['final_eval_pct']:>11.2f}%")
    print(f"  {'Edges':<25} {result_relu['edges']:>12} {result_c19['edges']:>12}")
    print(f"  {'Total accepts':<25} {result_relu['accepts']:>12} {result_c19['accepts']:>12}")
    print(f"  {'Add accepts':<25} {result_relu['add_acc']:>12} {result_c19['add_acc']:>12}")
    print(f"  {'Flip accepts':<25} {result_relu['flip_acc']:>12} {result_c19['flip_acc']:>12}")
    print(f"  {'Decay accepts':<25} {result_relu['decay_acc']:>12} {result_c19['decay_acc']:>12}")
    print(f"  {'C accepts':<25} {'N/A':>12} {result_c19['c_acc']:>12}")
    print(f"  {'Decay mean':<25} {result_relu['decay_mean']:>12.4f} {result_c19['decay_mean']:>12.4f}")
    print(f"  {'Time (s)':<25} {result_relu['elapsed']:>12.1f} {result_c19['elapsed']:>12.1f}")

    if 'C_mean' in result_c19:
        print(f"\n  C19 Parameter Analysis:")
        print(f"    C mean:  {result_c19['C_mean']:.4f}")
        print(f"    C std:   {result_c19['C_std']:.4f}")
        print(f"    C range: [{result_c19['C_min']:.2f}, {result_c19['C_max']:.2f}]")
        if 'C_histogram' in result_c19:
            print(f"    C distribution:")
            for bucket, count in result_c19['C_histogram'].items():
                bar = '#' * count
                print(f"      [{bucket}]: {count:3d} {bar}")
        if 'C_degree_correlation' in result_c19:
            print(f"    C vs degree correlation: {result_c19['C_degree_correlation']:.3f}")
            print(f"    C mean (connected neurons):    {result_c19['C_connected_mean']:.3f}")
            if result_c19['C_disconnected_mean'] is not None:
                print(f"    C mean (disconnected neurons): {result_c19['C_disconnected_mean']:.3f}")

    # Save results
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "c19_ab_results.json")
    with open(out_path, 'w') as f:
        # Don't save full C vector to keep JSON readable
        save_relu = {k: v for k, v in result_relu.items() if k != 'log'}
        save_c19 = {k: v for k, v in result_c19.items() if k not in ('log', 'C_final_vector', 'C_snapshots')}
        json.dump({'relu': save_relu, 'c19': save_c19}, f, indent=2)
    print(f"\n  Results saved to: {out_path}")
    print(f"{'='*60}")

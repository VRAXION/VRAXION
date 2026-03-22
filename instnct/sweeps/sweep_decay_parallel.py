"""
Parallel Decay Perturbation Sweep — 6 configs × 3 workers
==========================================================
Each config runs on 3 cores simultaneously, testing different
decay mutation strategies. 8 ticks, bigram 2seq, charge ReLU.
"""
import sys, os, time, random, json
import numpy as np
from multiprocessing import Pool, Process, Queue
import queue

from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "model"))
from graph import SelfWiringGraph

_bp = None; _all_data = None; _seq_len = 200; _n_train = 2
_input_projection = None; _output_projection = None; _bigram = None
_decay_step = 0.03
_theta_step = 0.05

def init_w(b, d, sl, nt, wi, wo, bg, ds, ts):
    global _bp, _all_data, _seq_len, _n_train, _input_projection, _output_projection, _bigram, _decay_step, _theta_step
    _bp, _all_data, _seq_len, _n_train = b, d, sl, nt
    _input_projection, _output_projection, _bigram = wi, wo, bg
    _decay_step, _theta_step = ds, ts

def make_bp(io_dim, seed=12345):
    rng = np.random.RandomState(seed)
    p = rng.randn(256, io_dim).astype(np.float32)
    p /= np.linalg.norm(p, axis=1, keepdims=True)
    return p

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
            for t in range(8):
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
    new_mask = mask; new_theta = theta; new_decay = decay

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
        new_theta[idx] = max(0.0, min(1.0, theta[idx] + rng.uniform(-_theta_step, _theta_step)))
    elif proposal_type == 'decay':
        idx = rng.randint(0, H-1)
        new_decay = decay.copy()
        new_decay[idx] = max(0.01, min(0.5, decay[idx] + rng.uniform(-_decay_step, _decay_step)))

    seqs = []
    data_len = len(_all_data)
    for _ in range(_n_train):
        off = np_rng.randint(0, data_len - _seq_len)
        seqs.append(_all_data[off:off+_seq_len])

    old_score = _eval_bigram(mask, H, theta, decay, seqs)
    new_score = _eval_bigram(new_mask, H, new_theta, new_decay, seqs)

    return {'delta': new_score - old_score, 'type': proposal_type,
            'new_mask_flat': new_mask.flatten() if new_score > old_score else None,
            'new_theta': new_theta if proposal_type == 'theta' else None,
            'new_decay': new_decay if proposal_type == 'decay' else None}

def eval_accuracy_classic(mask, H, input_projection, output_projection, theta, decay, text_bytes, bp):
    pat_norm = bp / (np.linalg.norm(bp, axis=1, keepdims=True) + 1e-8)
    rs, cs = np.where(mask != 0); sp_vals = mask[rs, cs]
    ret = 1.0 - decay
    state = np.zeros(H, dtype=np.float32); charge = np.zeros(H, dtype=np.float32)
    correct = 0; total = 0
    for i in range(len(text_bytes)-1):
        act = state.copy()
        for t in range(8):
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

def compute_deepness(mask, H, input_projection):
    """BFS from input projection: how many hops to reach each neuron."""
    # Which neurons get direct input? Those with high input_projection magnitude
    input_strength = np.abs(input_projection).sum(axis=0)  # (H,)
    # Start BFS from neurons with above-median input strength
    threshold = np.median(input_strength)
    visited = input_strength > threshold
    depth = np.zeros(H, dtype=np.int32)
    depth[~visited] = 99  # unvisited = far away

    # BFS through edges
    for d in range(1, 16):
        rs, cs = np.where(mask != 0)
        newly_reached = np.zeros(H, dtype=bool)
        for r, c in zip(rs, cs):
            if visited[r] and not visited[c]:
                newly_reached[c] = True
                depth[c] = d
        if not newly_reached.any():
            break
        visited |= newly_reached

    depth[depth == 99] = 0  # unreachable = treat as shallow
    return depth


def run_config(name, decay_step, theta_step, init_decay, topo_init,
               bp, ALL_DATA, bigram, eval_seqs, H, input_projection, output_projection,
               max_steps=800, n_workers=3, threshold=0.00005):
    mask = np.zeros((H, H), dtype=np.float32)
    theta = np.full(H, 0.03, dtype=np.float32)

    if topo_init:
        # Will be set after some edges exist — start with default
        decay = np.full(H, 0.15, dtype=np.float32)
    elif isinstance(init_decay, (int, float)):
        decay = np.full(H, init_decay, dtype=np.float32)
    else:
        decay = init_decay.copy()

    schedule = ['add', 'add', 'add', 'flip', 'theta', 'decay']
    accepts = {'add': 0, 'flip': 0, 'theta': 0, 'decay': 0}
    acc_history = []
    topo_applied = False
    t0 = time.time()

    pool = Pool(n_workers, initializer=init_w,
                initargs=(bp, ALL_DATA, 200, 2, input_projection, output_projection, bigram, decay_step, theta_step))
    try:
        for step in range(1, max_steps+1):
            ptype = schedule[(step-1) % len(schedule)]
            if ptype in ('flip', 'theta', 'decay') and np.count_nonzero(mask) == 0:
                ptype = 'add'

            # Topo init: after 50 edges, set decay based on graph depth
            if topo_init and not topo_applied and np.count_nonzero(mask) > 50:
                depth = compute_deepness(mask, H, input_projection)
                # deeper neurons get lower decay (longer memory)
                decay = np.clip(0.20 - depth * 0.02, 0.01, 0.5).astype(np.float32)
                topo_applied = True

            mask_flat = mask.flatten()
            args = [(mask_flat, theta.copy(), decay.copy(), H,
                     23000+step*50+w, ptype) for w in range(n_workers)]
            results = pool.map(worker_eval, args)

            best_r = max(results, key=lambda x: x['delta'])
            if best_r['delta'] > threshold:
                if best_r['type'] in ('add', 'flip') and best_r['new_mask_flat'] is not None:
                    mask = best_r['new_mask_flat'].reshape(H, H)
                    accepts[best_r['type']] += 1
                elif best_r['type'] == 'theta' and best_r['new_theta'] is not None:
                    theta = best_r['new_theta']
                    accepts['theta'] += 1
                elif best_r['type'] == 'decay' and best_r['new_decay'] is not None:
                    decay = best_r['new_decay']
                    accepts['decay'] += 1

            if step % 100 == 0:
                elapsed = time.time() - t0
                edges = int(np.count_nonzero(mask))
                ea = np.mean([eval_accuracy_classic(mask, H, input_projection, output_projection, theta, decay, s, bp)
                              for s in eval_seqs])
                acc_history.append((step, ea))
                quality = ea / max(edges, 1) * 100
                dm = decay.mean(); ds_val = decay.std()

                print(f"  {name:.<25} [{step:4d}] acc={ea*100:.2f}% edges={edges} "
                      f"A={accepts['add']}|F={accepts['flip']}|T={accepts['theta']}|D={accepts['decay']} "
                      f"decay={dm:.3f}+/-{ds_val:.3f} {elapsed:.0f}s")
                sys.stdout.flush()

                if len(acc_history) >= 4:
                    last4 = [a for _, a in acc_history[-4:]]
                    if max(last4) - min(last4) < 0.01:
                        print(f"  {name:.<25} PLATEAU @ step {step}")
                        break
    finally:
        pool.terminate(); pool.join()

    edges = int(np.count_nonzero(mask))
    ea = np.mean([eval_accuracy_classic(mask, H, input_projection, output_projection, theta, decay, s, bp)
                  for s in eval_seqs])
    elapsed = time.time() - t0
    quality = ea / max(edges, 1) * 100

    result = {
        'name': name, 'acc': ea, 'edges': edges, 'quality': quality,
        'decay_mean': float(decay.mean()), 'decay_std': float(decay.std()),
        'accepts': dict(accepts), 'time': elapsed
    }
    print(f"  {name:.<25} FINAL: acc={ea*100:.2f}% edges={edges} q={quality:.3f}%/e "
          f"decay={decay.mean():.3f}+/-{decay.std():.3f} D_acc={accepts['decay']} {elapsed:.0f}s")
    sys.stdout.flush()
    return result


if __name__ == "__main__":
    IO = 256; NV = 4; H = IO * NV
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
    ref = SelfWiringGraph(IO, hidden_ratio=NV, projection_scale=1.0)
    input_projection = ref.input_projection
    output_projection = ref.output_projection

    configs = [
        ("SMALL d=0.03",       0.03, 0.05, 0.15, False),
        ("MEDIUM d=0.1",       0.10, 0.05, 0.15, False),
        ("BIG d=0.2",          0.20, 0.05, 0.15, False),
        ("HUGE d=0.3",         0.30, 0.05, 0.15, False),
        ("TOPO+MED d=0.1",    0.10, 0.05, 0.15, True),
        ("AGGRO d=0.2 t=0.1",  0.20, 0.10, 0.15, False),
    ]

    print(f"\nRunning {len(configs)} configs in PARALLEL (3 workers each)")
    print(f"{'='*70}")
    sys.stdout.flush()

    # Run ALL configs in parallel using Process
    from concurrent.futures import ProcessPoolExecutor, as_completed

    def run_one(args):
        name, ds, ts, init_d, topo = args
        return run_config(name, ds, ts, init_d, topo,
                          bp, ALL_DATA, bigram, eval_seqs, H, input_projection, output_projection,
                          max_steps=800, n_workers=3)

    results = []
    # Run sequentially but with small pools (3 workers each)
    # True parallel would need subprocess spawning which is complex on Windows
    for cfg in configs:
        r = run_one(cfg)
        results.append(r)

    print(f"\n{'='*75}")
    print(f"  SUMMARY -- DECAY PERTURBATION (8t, bigram 2seq, ReLU, 3w)")
    print(f"{'='*75}")
    print(f"  {'Name':<25} {'Acc%':>6} {'Edges':>6} {'Q(%/e)':>8} {'Decay':>10} {'D_acc':>6} {'Time':>6}")
    print(f"  {'-'*25} {'-'*6} {'-'*6} {'-'*8} {'-'*10} {'-'*6} {'-'*6}")
    for r in results:
        print(f"  {r['name']:<25} {r['acc']*100:6.2f} {r['edges']:6d} {r['quality']:8.3f} "
              f"{r['decay_mean']:.3f}+/-{r['decay_std']:.3f} {r['accepts']['decay']:6d} {r['time']:5.0f}s")
    sys.stdout.flush()


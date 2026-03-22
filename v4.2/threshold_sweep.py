"""
Threshold + Architecture Sweep — 6 variants, 1000 steps each
=============================================================
Tests which threshold/signal approach unlocks recurrent dynamics.

Variants:
  A: threshold=0 (ReLU, no gate)
  B: threshold=0.1 (low gate)
  C: threshold=adaptive (mean(|charge|)*1.5 per tick)
  D: INJ_SCALE=10 (stronger injection, threshold=0.5)
  E: retention=0.99 (minimal decay, charge accumulates)
  F: baseline (threshold=0.5, INJ_SCALE=3, retention=0.85)
"""

import sys, os, time, random, json
import numpy as np
from multiprocessing import Pool

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "model"))
from graph import SelfWiringGraph

# ─── Globals per worker ──────────────────────────────────
_bp = None
_all_data = None
_cfg = None

def init_w(bp, data, cfg):
    global _bp, _all_data, _cfg
    _bp, _all_data, _cfg = bp, data, cfg

def make_bp(io_dim, seed=12345):
    rng = np.random.RandomState(seed)
    p = rng.randn(256, io_dim).astype(np.float32)
    p /= np.linalg.norm(p, axis=1, keepdims=True)
    return p

# ─── Configurable forward pass ───────────────────────────
def eval_mask(mask, H, input_projection, output_projection, seqs, cfg):
    """Forward pass with configurable threshold/retention/scale."""
    threshold = cfg['threshold']
    adaptive_thresh = cfg.get('adaptive_threshold', False)
    retention = cfg['retention']

    rs, cs = np.where(mask != 0)
    sp_vals = mask[rs, cs]
    pat_norm = _bp / (np.linalg.norm(_bp, axis=1, keepdims=True) + 1e-8)

    total = 0.0
    charge_stats = {'max': 0.0, 'fired': 0}

    for text_bytes in seqs:
        state = np.zeros(H, dtype=np.float32)
        charge = np.zeros(H, dtype=np.float32)
        correct = 0; prob_sum = 0.0; n = 0

        for i in range(len(text_bytes) - 1):
            act = state.copy()
            for t in range(6):
                if t == 0:
                    act = act + _bp[text_bytes[i]] @ input_projection
                raw = np.zeros(H, dtype=np.float32)
                if len(rs):
                    np.add.at(raw, cs, act[rs] * sp_vals)
                charge += raw
                charge *= retention
                # Threshold logic
                if adaptive_thresh:
                    thresh = float(np.mean(np.abs(charge))) * 1.5
                    act = np.maximum(charge - thresh, 0.0)
                else:
                    act = np.maximum(charge - threshold, 0.0)
                charge = np.clip(charge, -1.0, 1.0)

                cmax = float(np.abs(charge).max())
                if cmax > charge_stats['max']:
                    charge_stats['max'] = cmax
                if float(act.max()) > 0:
                    charge_stats['fired'] += 1

            state = act.copy()
            out = charge @ output_projection
            out_n = out / (np.linalg.norm(out) + 1e-8)
            sims = out_n @ pat_norm.T
            e = np.exp(sims - sims.max())
            probs = e / e.sum()
            target = text_bytes[i + 1]
            if np.argmax(probs) == target:
                correct += 1
            prob_sum += probs[target]
            n += 1

        acc = correct / n if n else 0
        avg_p = prob_sum / n if n else 0
        total += 0.5 * acc + 0.5 * avg_p

    return total / len(seqs), charge_stats

# ─── Worker ───────────────────────────────────────────────
def worker_eval(args):
    mask_flat, H, input_projection, output_projection, seed = args
    rng = random.Random(seed)
    np_rng = np.random.RandomState(seed)
    mask = mask_flat.reshape(H, H)

    r = rng.randint(0, H - 1)
    c = rng.randint(0, H - 1)
    if r == c or mask[r, c] != 0:
        return (0.0, -1, -1, 0.0, {})

    val = 0.6 if rng.random() < 0.5 else -0.6
    new_mask = mask.copy()
    new_mask[r, c] = val

    # Random sequences
    data_len = len(_all_data)
    seqs = []
    for _ in range(5):
        off = np_rng.randint(0, data_len - 200)
        seqs.append(_all_data[off:off + 200])

    old_score, _ = eval_mask(mask, H, input_projection, output_projection, seqs, _cfg)
    new_score, stats = eval_mask(new_mask, H, input_projection, output_projection, seqs, _cfg)
    delta = new_score - old_score

    return (delta, r, c, val, stats)

# ─── Eval accuracy (fixed seqs) ──────────────────────────
def eval_accuracy(mask, H, input_projection, output_projection, eval_seqs, cfg, bp):
    pat_norm = bp / (np.linalg.norm(bp, axis=1, keepdims=True) + 1e-8)
    threshold = cfg['threshold']
    adaptive_thresh = cfg.get('adaptive_threshold', False)
    retention = cfg['retention']

    rs, cs = np.where(mask != 0)
    sp_vals = mask[rs, cs]
    correct = 0; total = 0

    for seq in eval_seqs:
        state = np.zeros(H, dtype=np.float32)
        charge = np.zeros(H, dtype=np.float32)
        for i in range(len(seq) - 1):
            act = state.copy()
            for t in range(6):
                if t == 0:
                    act = act + bp[seq[i]] @ input_projection
                raw = np.zeros(H, dtype=np.float32)
                if len(rs):
                    np.add.at(raw, cs, act[rs] * sp_vals)
                charge += raw
                charge *= retention
                if adaptive_thresh:
                    thresh = float(np.mean(np.abs(charge))) * 1.5
                    act = np.maximum(charge - thresh, 0.0)
                else:
                    act = np.maximum(charge - threshold, 0.0)
                charge = np.clip(charge, -1.0, 1.0)
            state = act.copy()
            out = charge @ output_projection
            out_n = out / (np.linalg.norm(out) + 1e-8)
            sims = out_n @ pat_norm.T
            if np.argmax(sims) == seq[i + 1]:
                correct += 1
            total += 1

    return correct / total if total else 0

# ─── Main ─────────────────────────────────────────────────
if __name__ == "__main__":
    IO = 256; H = IO * 3; N_WORKERS = 18; BUDGET = 1000

    bp = make_bp(IO)
    DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "..", "Diamond Code", "data", "traindat", "fineweb_edu.traindat")
    with open(DATA, 'rb') as f:
        ALL_DATA = np.frombuffer(f.read(), dtype=np.uint8)

    # Fixed eval sequences
    eval_rng = np.random.RandomState(9999)
    eval_seqs = [ALL_DATA[eval_rng.randint(0, len(ALL_DATA)-200):] [:200] for _ in range(3)]

    # Define variants
    VARIANTS = {
        'A_thresh0':    {'threshold': 0.0,  'adaptive_threshold': False, 'retention': 0.85, 'inj_scale': 3.0},
        'B_thresh01':   {'threshold': 0.1,  'adaptive_threshold': False, 'retention': 0.85, 'inj_scale': 3.0},
        'C_adaptive':   {'threshold': 0.0,  'adaptive_threshold': True,  'retention': 0.85, 'inj_scale': 3.0},
        'D_injscale10': {'threshold': 0.5,  'adaptive_threshold': False, 'retention': 0.85, 'inj_scale': 10.0},
        'E_retain99':   {'threshold': 0.5,  'adaptive_threshold': False, 'retention': 0.99, 'inj_scale': 3.0},
        'F_baseline':   {'threshold': 0.5,  'adaptive_threshold': False, 'retention': 0.85, 'inj_scale': 3.0},
    }

    LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "threshold_sweep_live.txt")
    RESULTS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "threshold_sweep_results.json")

    with open(LOG, "w") as f:
        f.write(f"Threshold Sweep | {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"{len(VARIANTS)} variants, {BUDGET} steps each, {N_WORKERS} workers\n\n")

    all_results = {}

    for name, cfg in VARIANTS.items():
        print(f"\n{'='*60}")
        print(f"  {name}: threshold={cfg['threshold']} adaptive={cfg.get('adaptive_threshold',False)} "
              f"retention={cfg['retention']} inj_scale={cfg['inj_scale']}")
        print(f"{'='*60}")
        sys.stdout.flush()

        # Fresh network
        random.seed(42); np.random.seed(42)
        net = SelfWiringGraph(IO)
        net.mask[:] = 0; net.alive = []; net.alive_set = set(); net._sync_sparse_idx()
        net.state *= 0; net.charge *= 0

        # Apply INJ_SCALE override
        if cfg['inj_scale'] != 3.0:
            scale_factor = cfg['inj_scale'] / 3.0
            input_projection = net.input_projection * scale_factor
            output_projection = net.output_projection * scale_factor
        else:
            input_projection = net.input_projection
            output_projection = net.output_projection

        pool = Pool(N_WORKERS, initializer=init_w, initargs=(bp, ALL_DATA, cfg))
        accepts = 0
        last_stats = {}
        t0 = time.time()

        try:
            for step in range(1, BUDGET + 1):
                mask_flat = net.mask.flatten()
                args = [(mask_flat, H, input_projection, output_projection, 1000 + step * 50 + w) for w in range(N_WORKERS)]
                results = pool.map(worker_eval, args)

                best_r = max(results, key=lambda x: x[0])
                if best_r[0] > 0 and best_r[1] >= 0:
                    net.mask[best_r[1], best_r[2]] = best_r[3]
                    net.alive.append((best_r[1], best_r[2]))
                    net.alive_set.add((best_r[1], best_r[2]))
                    net._sync_sparse_idx()
                    accepts += 1
                    last_stats = best_r[4]

                if step % 200 == 0:
                    elapsed = time.time() - t0
                    ea = eval_accuracy(net.mask, H, input_projection, output_projection, eval_seqs, cfg, bp)
                    edges = net.count_connections()
                    fired = last_stats.get('fired', 0)
                    cmax = last_stats.get('max', 0)
                    line = (f"  [{step:5d}] eval={ea*100:.1f}% edges={edges} "
                            f"accepts={accepts} charge_max={cmax:.3f} fired={fired} "
                            f"{elapsed:.0f}s")
                    print(line)
                    with open(LOG, "a") as f:
                        f.write(f"[{name}] {line}\n")
                    sys.stdout.flush()
        finally:
            pool.terminate(); pool.join()

        # Final eval
        elapsed = time.time() - t0
        final_ea = eval_accuracy(net.mask, H, input_projection, output_projection, eval_seqs, cfg, bp)
        edges = net.count_connections()

        result = {
            'name': name,
            'eval': round(final_ea * 100, 2),
            'edges': edges,
            'accepts': accepts,
            'charge_max': round(last_stats.get('max', 0), 4),
            'fired_ticks': last_stats.get('fired', 0),
            'elapsed': round(elapsed, 1),
            'config': cfg,
        }
        all_results[name] = result

        summary = f"  DONE: eval={final_ea*100:.1f}% edges={edges} accepts={accepts} {elapsed:.0f}s"
        print(summary)
        with open(LOG, "a") as f:
            f.write(f"[{name}] {summary}\n\n")

        # Save checkpoint
        CKPT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "checkpoints", f"sweep_{name}.npz")
        os.makedirs(os.path.dirname(CKPT), exist_ok=True)
        net.save(CKPT)

    # Final comparison
    print(f"\n{'='*70}")
    print(f"  SWEEP RESULTS")
    print(f"{'='*70}")
    print(f"  {'Variant':20s} {'Eval%':>7s} {'Edges':>6s} {'Accept':>7s} {'ChgMax':>7s} {'Fired':>6s} {'Time':>6s}")
    print(f"  {'-'*20} {'-'*7} {'-'*6} {'-'*7} {'-'*7} {'-'*6} {'-'*6}")

    with open(LOG, "a") as f:
        f.write(f"\n{'='*70}\n  SWEEP RESULTS\n{'='*70}\n")

    sorted_results = sorted(all_results.values(), key=lambda x: -x['eval'])
    for r in sorted_results:
        line = (f"  {r['name']:20s} {r['eval']:7.1f} {r['edges']:6d} {r['accepts']:7d} "
                f"{r['charge_max']:7.3f} {r['fired_ticks']:6d} {r['elapsed']:6.0f}s")
        print(line)
        with open(LOG, "a") as f:
            f.write(line + "\n")

    winner = sorted_results[0]['name']
    print(f"\n  WINNER: {winner} ({sorted_results[0]['eval']:.1f}%)")

    with open(RESULTS_FILE, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"  Saved: {RESULTS_FILE}")

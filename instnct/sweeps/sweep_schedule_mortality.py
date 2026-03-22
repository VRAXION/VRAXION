"""
Schedule Mortality — use-it-or-lose-it
=======================================
Every N steps, all schedule counts decay by -1 (min 1).
The network must actively maintain what's important.
Compare: no mortality vs mortality at different decay rates.
8 ticks, charge ReLU, bigram 2seq, run to plateau.
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

def init_w(b, d, sl, nt, wi, wo, bg):
    global _bp, _all_data, _seq_len, _n_train, _input_projection, _output_projection, _bigram
    _bp, _all_data, _seq_len, _n_train = b, d, sl, nt
    _input_projection, _output_projection, _bigram = wi, wo, bg

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
        new_theta[idx] = max(0.0, min(1.0, theta[idx] + rng.uniform(-0.05, 0.05)))
    elif proposal_type == 'decay':
        idx = rng.randint(0, H-1)
        new_decay = decay.copy()
        new_decay[idx] = max(0.01, min(0.5, decay[idx] + rng.uniform(-0.03, 0.03)))

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

def build_schedule(counts):
    s = []
    for op in ['add', 'flip', 'theta', 'decay']:
        s.extend([op] * counts.get(op, 0))
    return s

def mutate_counts(counts, rng):
    ops = list(counts.keys())
    op = ops[rng.randint(0, len(ops)-1)]
    new_counts = dict(counts)
    if rng.random() < 0.5:
        new_counts[op] = counts[op] + 1
    else:
        new_counts[op] = max(1, counts[op] - 1)
    return new_counts

def apply_mortality(counts):
    """All counts decay by 1 (min 1). Use-it-or-lose-it."""
    return {k: max(1, v - 1) for k, v in counts.items()}


def run_config(name, init_counts, mortality_every, bp, ALL_DATA, bigram, eval_seqs, H, input_projection, output_projection,
               max_steps=1500, n_workers=18, threshold=0.00005):
    mask = np.zeros((H, H), dtype=np.float32)
    theta = np.full(H, 0.03, dtype=np.float32)
    decay = np.full(H, 0.15, dtype=np.float32)
    counts = dict(init_counts)
    schedule = build_schedule(counts)
    sched_rng = random.Random(42)
    mortality_events = 0

    print(f"\n{'='*70}")
    print(f"  {name}")
    print(f"  Init: {counts}, mortality every {mortality_every} steps")
    print(f"{'='*70}")
    sys.stdout.flush()

    accepts = {'add': 0, 'flip': 0, 'theta': 0, 'decay': 0}
    acc_history = []
    t0 = time.time()

    pool = Pool(n_workers, initializer=init_w,
                initargs=(bp, ALL_DATA, 200, 2, input_projection, output_projection, bigram))
    try:
        for step in range(1, max_steps+1):
            ptype = schedule[(step - 1) % len(schedule)]
            if ptype in ('flip', 'theta', 'decay') and np.count_nonzero(mask) == 0:
                ptype = 'add'

            mask_flat = mask.flatten()
            args = [(mask_flat, theta.copy(), decay.copy(), H,
                     17000+step*50+w, ptype) for w in range(n_workers)]
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

            # Mutate schedule every 20 steps
            if step % 20 == 0:
                counts = mutate_counts(counts, sched_rng)
                schedule = build_schedule(counts)

            # MORTALITY: decay all counts every N steps
            if mortality_every > 0 and step % mortality_every == 0:
                old_counts = dict(counts)
                counts = apply_mortality(counts)
                schedule = build_schedule(counts)
                mortality_events += 1

            if step % 100 == 0:
                elapsed = time.time() - t0
                edges = int(np.count_nonzero(mask))
                ea = np.mean([eval_accuracy_classic(mask, H, input_projection, output_projection, theta, decay, s, bp)
                              for s in eval_seqs])
                acc_history.append((step, ea))
                total_slots = sum(counts.values())
                ratios = {k: f"{v/total_slots*100:.0f}%" for k, v in counts.items() if v > 0}
                quality = ea / max(edges, 1) * 100

                print(f"  [{step:4d}] acc={ea*100:.2f}% edges={edges} q={quality:.3f}%/e "
                      f"A={accepts['add']}|F={accepts['flip']}|T={accepts['theta']}|D={accepts['decay']} "
                      f"counts={counts} mort={mortality_events} {elapsed:.0f}s")
                sys.stdout.flush()

                # Plateau: last 4 evals within 1%
                if len(acc_history) >= 4:
                    last4 = [a for _, a in acc_history[-4:]]
                    if max(last4) - min(last4) < 0.01:
                        print(f"  PLATEAU @ step {step}")
                        break

    finally:
        pool.terminate(); pool.join()

    edges = int(np.count_nonzero(mask))
    ea = np.mean([eval_accuracy_classic(mask, H, input_projection, output_projection, theta, decay, s, bp)
                  for s in eval_seqs])
    elapsed = time.time() - t0
    quality = ea / max(edges, 1) * 100

    print(f"\n  FINAL: acc={ea*100:.2f}% edges={edges} quality={quality:.3f}%/edge")
    print(f"  Accepts: A={accepts['add']}|F={accepts['flip']}|T={accepts['theta']}|D={accepts['decay']}")
    print(f"  Schedule: {counts} (mortality events: {mortality_events})")
    print(f"  Time: {elapsed:.0f}s")
    sys.stdout.flush()
    return {'name': name, 'acc': ea, 'edges': edges, 'time': elapsed,
            'quality': quality, 'accepts': dict(accepts), 'final_counts': dict(counts)}


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

    # A: No mortality (baseline, just learnable schedule)
    results.append(run_config("NO MORTALITY (baseline)",
                              {'add': 3, 'flip': 1, 'theta': 1, 'decay': 1}, 0,
                              bp, ALL_DATA, bigram, eval_seqs, H, input_projection, output_projection))

    # B: Mortality every 50 steps (moderate pressure)
    results.append(run_config("MORTALITY every 50",
                              {'add': 3, 'flip': 1, 'theta': 1, 'decay': 1}, 50,
                              bp, ALL_DATA, bigram, eval_seqs, H, input_projection, output_projection))

    # C: Mortality every 20 steps (high pressure — same as mutation rate)
    results.append(run_config("MORTALITY every 20",
                              {'add': 3, 'flip': 1, 'theta': 1, 'decay': 1}, 20,
                              bp, ALL_DATA, bigram, eval_seqs, H, input_projection, output_projection))

    # D: Mortality every 100 steps (gentle)
    results.append(run_config("MORTALITY every 100",
                              {'add': 3, 'flip': 1, 'theta': 1, 'decay': 1}, 100,
                              bp, ALL_DATA, bigram, eval_seqs, H, input_projection, output_projection))

    # E: High init + mortality (start big, see what survives)
    results.append(run_config("HIGH START + MORT 50",
                              {'add': 8, 'flip': 4, 'theta': 4, 'decay': 4}, 50,
                              bp, ALL_DATA, bigram, eval_seqs, H, input_projection, output_projection))

    print(f"\n{'='*75}")
    print(f"  SUMMARY -- SCHEDULE MORTALITY (8t, bigram 2seq, ReLU)")
    print(f"{'='*75}")
    print(f"  {'Name':<25} {'Acc%':>6} {'Edges':>6} {'Q(%/e)':>8} {'Final sched':<25}")
    print(f"  {'-'*25} {'-'*6} {'-'*6} {'-'*8} {'-'*25}")
    for r in results:
        fc = r['final_counts']
        sched_str = '/'.join(f"{fc.get(k,0)}{k[0]}" for k in ['add','flip','theta','decay'] if fc.get(k,0) > 0)
        print(f"  {r['name']:<25} {r['acc']*100:6.2f} {r['edges']:6d} {r['quality']:8.3f} {sched_str:<25}")
    sys.stdout.flush()

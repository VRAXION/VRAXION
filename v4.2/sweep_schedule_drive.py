"""
Schedule Drive — continuous decay with 10x multiplier
=======================================================
Internal counters decay -1 every step. Mutation adds +10.
Schedule = counter // 10 (min 1). The AI must actively maintain
what's important — use-it-or-lose-it with fine-grained control.

Compare: no drive (baseline) vs drive with different divisors.
8 ticks, charge ReLU, bigram 2seq, run to plateau.
"""
import sys, os, time, random
import numpy as np
from multiprocessing import Pool

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "model"))
from graph import SelfWiringGraph

_bp = None; _all_data = None; _seq_len = 200; _n_train = 2
_W_in = None; _W_out = None; _bigram = None

def init_w(b, d, sl, nt, wi, wo, bg):
    global _bp, _all_data, _seq_len, _n_train, _W_in, _W_out, _bigram
    _bp, _all_data, _seq_len, _n_train = b, d, sl, nt
    _W_in, _W_out, _bigram = wi, wo, bg

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
                    act = act + _bp[text_bytes[i]] @ _W_in
                raw = np.zeros(H, dtype=np.float32)
                if len(rs):
                    np.add.at(raw, cs, act[rs] * sp_vals)
                charge += raw; charge *= ret
                act = np.maximum(charge - theta, 0.0)
                charge = np.maximum(charge, 0.0)
            state = act.copy()
            out = charge @ _W_out
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

def eval_accuracy_classic(mask, H, W_in, W_out, theta, decay, text_bytes, bp):
    pat_norm = bp / (np.linalg.norm(bp, axis=1, keepdims=True) + 1e-8)
    rs, cs = np.where(mask != 0); sp_vals = mask[rs, cs]
    ret = 1.0 - decay
    state = np.zeros(H, dtype=np.float32); charge = np.zeros(H, dtype=np.float32)
    correct = 0; total = 0
    for i in range(len(text_bytes)-1):
        act = state.copy()
        for t in range(8):
            if t == 0: act = act + bp[text_bytes[i]] @ W_in
            raw = np.zeros(H, dtype=np.float32)
            if len(rs): np.add.at(raw, cs, act[rs] * sp_vals)
            charge += raw; charge *= ret
            act = np.maximum(charge - theta, 0.0)
            charge = np.maximum(charge, 0.0)
        state = act.copy()
        out = charge @ W_out
        out_n = out / (np.linalg.norm(out) + 1e-8)
        sims = out_n @ pat_norm.T
        if np.argmax(sims) == text_bytes[i+1]: correct += 1
        total += 1
    return correct/total if total else 0


def build_schedule(counters, divisor):
    """counters = {add: 30, flip: 10, ...}, divisor = 10 -> schedule from counts//divisor."""
    s = []
    for op in ['add', 'flip', 'theta', 'decay']:
        n = max(1, counters.get(op, 0) // divisor)
        s.extend([op] * n)
    return s

def get_effective_counts(counters, divisor):
    return {k: max(1, v // divisor) for k, v in counters.items()}


def run_config(name, init_counters, divisor, boost, use_drive,
               bp, ALL_DATA, bigram, eval_seqs, H, W_in, W_out,
               max_steps=1500, n_workers=18, threshold=0.00005):
    mask = np.zeros((H, H), dtype=np.float32)
    theta = np.full(H, 0.03, dtype=np.float32)
    decay_arr = np.full(H, 0.15, dtype=np.float32)
    counters = dict(init_counters)
    sched_rng = random.Random(42)

    schedule = build_schedule(counters, divisor)

    print(f"\n{'='*70}")
    print(f"  {name}")
    print(f"  Init counters: {counters}, divisor={divisor}, boost={boost}")
    print(f"  Schedule: {get_effective_counts(counters, divisor)} (len={len(schedule)})")
    print(f"{'='*70}")
    sys.stdout.flush()

    accepts = {'add': 0, 'flip': 0, 'theta': 0, 'decay': 0}
    acc_history = []
    t0 = time.time()

    pool = Pool(n_workers, initializer=init_w,
                initargs=(bp, ALL_DATA, 200, 2, W_in, W_out, bigram))
    try:
        for step in range(1, max_steps+1):
            ptype = schedule[(step - 1) % len(schedule)]
            if ptype in ('flip', 'theta', 'decay') and np.count_nonzero(mask) == 0:
                ptype = 'add'

            mask_flat = mask.flatten()
            args = [(mask_flat, theta.copy(), decay_arr.copy(), H,
                     18000+step*50+w, ptype) for w in range(n_workers)]
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
                    decay_arr = best_r['new_decay']
                    accepts['decay'] += 1

            if use_drive:
                # DRIVE: every step, all counters -= 1 (min 0)
                for k in counters:
                    counters[k] = max(0, counters[k] - 1)

                # MUTATION: every 20 steps, one random counter gets +boost
                if step % 20 == 0:
                    ops = list(counters.keys())
                    op = ops[sched_rng.randint(0, len(ops)-1)]
                    counters[op] += boost

                # Rebuild schedule from counters
                schedule = build_schedule(counters, divisor)
            else:
                # No drive: just random walk every 20 steps (like before)
                if step % 20 == 0:
                    ops = list(counters.keys())
                    op = ops[sched_rng.randint(0, len(ops)-1)]
                    if sched_rng.random() < 0.5:
                        counters[op] += divisor
                    else:
                        counters[op] = max(divisor, counters[op] - divisor)
                    schedule = build_schedule(counters, divisor)

            if step % 100 == 0:
                elapsed = time.time() - t0
                edges = int(np.count_nonzero(mask))
                ea = np.mean([eval_accuracy_classic(mask, H, W_in, W_out, theta, decay_arr, s, bp)
                              for s in eval_seqs])
                acc_history.append((step, ea))
                eff = get_effective_counts(counters, divisor)
                quality = ea / max(edges, 1) * 100

                print(f"  [{step:4d}] acc={ea*100:.2f}% edges={edges} q={quality:.3f}%/e "
                      f"A={accepts['add']}|F={accepts['flip']}|T={accepts['theta']}|D={accepts['decay']} "
                      f"raw={counters} eff={eff} {elapsed:.0f}s")
                sys.stdout.flush()

                if len(acc_history) >= 4:
                    last4 = [a for _, a in acc_history[-4:]]
                    if max(last4) - min(last4) < 0.01:
                        print(f"  PLATEAU @ step {step}")
                        break

    finally:
        pool.terminate(); pool.join()

    edges = int(np.count_nonzero(mask))
    ea = np.mean([eval_accuracy_classic(mask, H, W_in, W_out, theta, decay_arr, s, bp)
                  for s in eval_seqs])
    elapsed = time.time() - t0
    quality = ea / max(edges, 1) * 100
    eff = get_effective_counts(counters, divisor)

    print(f"\n  FINAL: acc={ea*100:.2f}% edges={edges} quality={quality:.3f}%/edge")
    print(f"  Accepts: A={accepts['add']}|F={accepts['flip']}|T={accepts['theta']}|D={accepts['decay']}")
    print(f"  Counters: {counters} -> effective: {eff}")
    print(f"  Time: {elapsed:.0f}s")
    sys.stdout.flush()
    return {'name': name, 'acc': ea, 'edges': edges, 'time': elapsed,
            'quality': quality, 'accepts': dict(accepts),
            'final_counters': dict(counters), 'final_eff': dict(eff)}


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
    W_in = ref.W_in / ref.INJ_SCALE * 1.0
    W_out = ref.W_out / ref.INJ_SCALE * 1.0

    results = []

    # A: No drive baseline (learnable random walk, same as before)
    results.append(run_config("NO DRIVE (baseline)",
                              {'add': 30, 'flip': 10, 'theta': 10, 'decay': 10},
                              10, 0, False,
                              bp, ALL_DATA, bigram, eval_seqs, H, W_in, W_out))

    # B: Drive div=10, boost=10 (every 20 steps one counter gets +10, -1/step)
    results.append(run_config("DRIVE div=10 boost=10",
                              {'add': 30, 'flip': 10, 'theta': 10, 'decay': 10},
                              10, 10, True,
                              bp, ALL_DATA, bigram, eval_seqs, H, W_in, W_out))

    # C: Drive div=10, boost=20 (stronger life force)
    results.append(run_config("DRIVE div=10 boost=20",
                              {'add': 30, 'flip': 10, 'theta': 10, 'decay': 10},
                              10, 20, True,
                              bp, ALL_DATA, bigram, eval_seqs, H, W_in, W_out))

    # D: Drive div=5, boost=5 (finer granularity)
    results.append(run_config("DRIVE div=5 boost=5",
                              {'add': 15, 'flip': 5, 'theta': 5, 'decay': 5},
                              5, 5, True,
                              bp, ALL_DATA, bigram, eval_seqs, H, W_in, W_out))

    # E: Drive div=10, boost=10, high add start
    results.append(run_config("DRIVE high-add start",
                              {'add': 50, 'flip': 10, 'theta': 10, 'decay': 10},
                              10, 10, True,
                              bp, ALL_DATA, bigram, eval_seqs, H, W_in, W_out))

    print(f"\n{'='*80}")
    print(f"  SUMMARY -- SCHEDULE DRIVE (8t, bigram 2seq, ReLU)")
    print(f"{'='*80}")
    print(f"  {'Name':<25} {'Acc%':>6} {'Edges':>6} {'Q(%/e)':>8} {'Final eff':<25}")
    print(f"  {'-'*25} {'-'*6} {'-'*6} {'-'*8} {'-'*25}")
    for r in results:
        eff = r['final_eff']
        eff_str = '/'.join(f"{eff.get(k,1)}{k[0]}" for k in ['add','flip','theta','decay'])
        print(f"  {r['name']:<25} {r['acc']*100:6.2f} {r['edges']:6d} {r['quality']:8.3f} {eff_str:<25}")
    sys.stdout.flush()

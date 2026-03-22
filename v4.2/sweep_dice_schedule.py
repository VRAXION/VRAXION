"""
Int4 Dice Schedule — independent % per mutation type
=====================================================
3 int4 counters [0-15]: add, flip, decay.
Each step: roll d16 per type, fire if roll < counter.
Multiple can fire per step. None can fire (skip).
No normalization needed — independent probabilities.

counter=0: 0% (never), counter=15: 93.75% (almost always)
"""
import sys, os, time, random
import numpy as np
from multiprocessing import Pool

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "model"))
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
            injection = _bp[text_bytes[i]] @ _input_projection
            for t in range(8):
                if t < 2:
                    act = act + injection
                raw = np.zeros(H, dtype=np.float32)
                if len(rs):
                    np.add.at(raw, cs, act[rs] * sp_vals)
                charge += raw; charge *= ret
                act = np.maximum(charge, 0.0)  # theta=0, just ReLU
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
    mask_flat, decay, H, seed, proposal_type = args
    rng = random.Random(seed)
    np_rng = np.random.RandomState(seed)
    mask = mask_flat.reshape(H, H)
    new_mask = mask; new_decay = decay

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
    elif proposal_type == 'decay':
        idx = rng.randint(0, H-1)
        new_decay = decay.copy()
        new_decay[idx] = rng.uniform(0.01, 0.50)

    seqs = []
    data_len = len(_all_data)
    for _ in range(_n_train):
        off = np_rng.randint(0, data_len - _seq_len)
        seqs.append(_all_data[off:off+_seq_len])

    theta = np.zeros(H, dtype=np.float32)
    old_score = _eval_bigram(mask, H, theta, decay, seqs)
    new_score = _eval_bigram(new_mask, H, theta, new_decay, seqs)

    return {'delta': new_score - old_score, 'type': proposal_type,
            'new_mask_flat': new_mask.flatten() if new_score > old_score else None,
            'new_decay': new_decay if proposal_type == 'decay' else None}

def eval_accuracy_classic(mask, H, input_projection, output_projection, decay, text_bytes, bp):
    pat_norm = bp / (np.linalg.norm(bp, axis=1, keepdims=True) + 1e-8)
    rs, cs = np.where(mask != 0); sp_vals = mask[rs, cs]
    ret = 1.0 - decay
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


def run_config(name, counters, learnable,
               bp, ALL_DATA, bigram, eval_seqs, H, input_projection, output_projection,
               max_steps=800, n_workers=18, threshold=0.00005):
    mask = np.zeros((H, H), dtype=np.float32)
    decay_rng = np.random.RandomState(99)
    decay = decay_rng.uniform(0.08, 0.24, H).astype(np.float32)
    counters = dict(counters)
    dice_rng = random.Random(42)
    meta_rng = random.Random(77)

    ops = ['add', 'flip', 'decay']
    print(f"\n--- {name} (counters={counters}, learnable={learnable}) ---")
    probs = {k: f"{v/16*100:.0f}%" for k, v in counters.items()}
    print(f"  Probs: {probs}")
    sys.stdout.flush()

    accepts = {'add': 0, 'flip': 0, 'decay': 0}
    fires = {'add': 0, 'flip': 0, 'decay': 0}
    skips = 0
    acc_history = []
    t0 = time.time()

    pool = Pool(n_workers, initializer=init_w,
                initargs=(bp, ALL_DATA, 200, 2, input_projection, output_projection, bigram))
    try:
        for step in range(1, max_steps+1):
            # Roll dice for each type independently
            fired = []
            for op in ops:
                if dice_rng.randint(0, 15) < counters[op]:
                    fired.append(op)

            if not fired:
                skips += 1
                # Learnable: mutate counters on skip steps
                if learnable:
                    op = ops[meta_rng.randint(0, len(ops)-1)]
                    counters[op] = meta_rng.randint(0, 15)
                continue

            # For each fired type, run workers and accept best
            for ptype in fired:
                if ptype in ('flip', 'decay') and np.count_nonzero(mask) == 0:
                    ptype = 'add'

                mask_flat = mask.flatten()
                args = [(mask_flat, decay.copy(), H,
                         30000+step*50+w, ptype) for w in range(n_workers)]
                results = pool.map(worker_eval, args)

                best_r = max(results, key=lambda x: x['delta'])
                if best_r['delta'] > threshold:
                    if best_r['type'] in ('add', 'flip') and best_r['new_mask_flat'] is not None:
                        mask = best_r['new_mask_flat'].reshape(H, H)
                        accepts[best_r['type']] += 1
                    elif best_r['type'] == 'decay' and best_r['new_decay'] is not None:
                        decay = best_r['new_decay']
                        accepts['decay'] += 1

                fires[ptype] = fires.get(ptype, 0) + 1

            # Learnable: mutate one random counter every 20 steps
            if learnable and step % 20 == 0:
                op = ops[meta_rng.randint(0, len(ops)-1)]
                counters[op] = meta_rng.randint(0, 15)

            if step % 100 == 0:
                elapsed = time.time() - t0
                edges = int(np.count_nonzero(mask))
                ea = np.mean([eval_accuracy_classic(mask, H, input_projection, output_projection, decay, s, bp)
                              for s in eval_seqs])
                acc_history.append((step, ea))
                quality = ea / max(edges, 1) * 100
                probs = {k: f"{v/16*100:.0f}%" for k, v in counters.items()}

                print(f"  [{step:4d}] acc={ea*100:.2f}% edges={edges} q={quality:.3f}%/e "
                      f"A={accepts['add']}|F={accepts['flip']}|D={accepts['decay']} "
                      f"fires={fires} skips={skips} {probs} {elapsed:.0f}s")
                sys.stdout.flush()

                if len(acc_history) >= 4:
                    last4 = [a for _, a in acc_history[-4:]]
                    if max(last4) - min(last4) < 0.01:
                        print(f"  PLATEAU @ step {step}")
                        break
    finally:
        pool.terminate(); pool.join()

    edges = int(np.count_nonzero(mask))
    ea = np.mean([eval_accuracy_classic(mask, H, input_projection, output_projection, decay, s, bp)
                  for s in eval_seqs])
    elapsed = time.time() - t0
    quality = ea / max(edges, 1) * 100

    print(f"  FINAL: acc={ea*100:.2f}% edges={edges} q={quality:.3f}%/e "
          f"counters={counters} fires={fires} skips={skips} {elapsed:.0f}s")
    sys.stdout.flush()
    return {'name': name, 'acc': ea, 'edges': edges, 'quality': quality,
            'counters': dict(counters), 'fires': dict(fires), 'skips': skips,
            'time': elapsed}


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

    # A: Hand-tuned (add-heavy, moderate flip, light decay)
    results.append(run_config("FIX 8/5/3",
                              {'add': 8, 'flip': 5, 'decay': 3}, False,
                              bp, ALL_DATA, bigram, eval_seqs, H, input_projection, output_projection))

    # B: Equal
    results.append(run_config("FIX equal 8/8/8",
                              {'add': 8, 'flip': 8, 'decay': 8}, False,
                              bp, ALL_DATA, bigram, eval_seqs, H, input_projection, output_projection))

    # C: Learnable from hand-tuned
    results.append(run_config("LEARN 8/5/3",
                              {'add': 8, 'flip': 5, 'decay': 3}, True,
                              bp, ALL_DATA, bigram, eval_seqs, H, input_projection, output_projection))

    # D: Learnable from equal
    results.append(run_config("LEARN 8/8/8",
                              {'add': 8, 'flip': 8, 'decay': 8}, True,
                              bp, ALL_DATA, bigram, eval_seqs, H, input_projection, output_projection))

    print(f"\n{'='*75}")
    print(f"  SUMMARY -- DICE SCHEDULE (int4, 8t, dur=2, bigram, theta=0)")
    print(f"{'='*75}")
    print(f"  {'Name':<18} {'Acc%':>6} {'Edges':>6} {'Q(%/e)':>8} {'Skips':>6} {'Counters':<20}")
    print(f"  {'-'*18} {'-'*6} {'-'*6} {'-'*8} {'-'*6} {'-'*20}")
    for r in results:
        c = r['counters']
        cstr = f"a={c['add']}/f={c['flip']}/d={c['decay']}"
        print(f"  {r['name']:<18} {r['acc']*100:6.2f} {r['edges']:6d} {r['quality']:8.3f} "
              f"{r['skips']:6d} {cstr:<20}")
    sys.stdout.flush()

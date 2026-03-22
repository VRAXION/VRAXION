"""
Triangle Schedule — 3 relational axes, learnable
==================================================
3 int4 params encode pairwise relations between mutation types.
Total always 45 — forced tradeoff. No random dice, pure learnable.

A: Fix 2a/1f/1d (best from previous)
B: Triangle learnable init add-heavy
C: Triangle learnable init equal
All 1000 steps, 18 workers, bigram 2seq, charge ReLU, theta=0, decay [.08,.24].
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

def _eval_bigram(mask, H, decay, seqs):
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
                act = np.maximum(charge, 0.0)
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
    new_mask = mask.copy()
    new_decay = decay.copy()

    if proposal_type == 'add':
        r = rng.randint(0, H-1); c = rng.randint(0, H-1)
        if r == c or mask[r, c] != 0:
            return {'delta': -1e9, 'type': 'add'}
        val = 0.6 if rng.random() < 0.5 else -0.6
        new_mask[r, c] = val
    elif proposal_type == 'flip':
        alive = list(zip(*np.where(mask != 0)))
        if not alive:
            return {'delta': -1e9, 'type': 'flip'}
        r, c = alive[rng.randint(0, len(alive)-1)]
        new_mask[r, c] = -mask[r, c]
    elif proposal_type == 'decay':
        idx = rng.randint(0, H-1)
        new_decay[idx] = rng.uniform(0.01, 0.50)

    seqs = []
    data_len = len(_all_data)
    for _ in range(_n_train):
        off = np_rng.randint(0, data_len - _seq_len)
        seqs.append(_all_data[off:off+_seq_len])

    old_score = _eval_bigram(mask, H, decay, seqs)
    new_score = _eval_bigram(new_mask, H, new_decay, seqs)

    return {'delta': new_score - old_score, 'type': proposal_type,
            'new_mask_flat': new_mask.flatten() if new_score > old_score else None,
            'new_decay': new_decay if new_score > old_score else None}

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


def triangle_probs(a, b, c):
    """3 int4 params -> mutation probabilities."""
    add_s = a + b
    flip_s = (15 - a) + c
    decay_s = (15 - b) + (15 - c)
    total = 45.0
    return add_s / total, flip_s / total, decay_s / total

def pick_from_triangle(a, b, c, rng):
    """Pick mutation type from triangle probabilities."""
    pa, pf, pd = triangle_probs(a, b, c)
    r = rng.random()
    if r < pa:
        return 'add'
    elif r < pa + pf:
        return 'flip'
    else:
        return 'decay'


def run_config(name, use_triangle, tri_a, tri_b, tri_c, learnable,
               bp, ALL_DATA, bigram, eval_seqs, H, input_projection, output_projection,
               max_steps=1000, n_workers=18, threshold=0.00005):
    mask = np.zeros((H, H), dtype=np.float32)
    decay_rng_init = np.random.RandomState(99)
    decay = decay_rng_init.uniform(0.08, 0.24, H).astype(np.float32)

    a, b, c = tri_a, tri_b, tri_c
    pa, pf, pd = triangle_probs(a, b, c)
    fix_schedule = ['add', 'add', 'flip', 'decay']

    print(f"\n--- {name} ---")
    if use_triangle:
        print(f"  Triangle: a={a} b={b} c={c} -> add={pa*100:.0f}% flip={pf*100:.0f}% decay={pd*100:.0f}%")
    else:
        print(f"  Fix schedule: {fix_schedule}")
    sys.stdout.flush()

    accepts = {'add': 0, 'flip': 0, 'decay': 0, 'tri': 0}
    acc_history = []
    sched_rng = random.Random(42)
    t0 = time.time()

    pool = Pool(n_workers, initializer=init_w,
                initargs=(bp, ALL_DATA, 200, 2, input_projection, output_projection, bigram))
    try:
        for step in range(1, max_steps+1):
            # Triangle mutation every 6th step
            if use_triangle and learnable and step % 6 == 0:
                # Try mutating one axis +-1
                axis = sched_rng.randint(0, 2)
                direction = sched_rng.choice([-1, 1])
                old_a, old_b, old_c = a, b, c
                if axis == 0: a = max(0, min(15, a + direction))
                elif axis == 1: b = max(0, min(15, b + direction))
                else: c = max(0, min(15, c + direction))

                # Eval: does the new ratio help?
                # We can't directly eval the ratio — accept unconditionally
                # and let the network mutations under new ratio be the test
                accepts['tri'] += 1
                continue  # skip this step's mutation

            # Pick mutation type
            if use_triangle:
                ptype = pick_from_triangle(a, b, c, sched_rng)
            else:
                ptype = fix_schedule[(step-1) % len(fix_schedule)]

            if ptype in ('flip', 'decay') and np.count_nonzero(mask) == 0:
                ptype = 'add'

            mask_flat = mask.flatten()
            args = [(mask_flat, decay.copy(), H,
                     32000+step*50+w, ptype) for w in range(n_workers)]
            results = pool.map(worker_eval, args)

            best_r = max(results, key=lambda x: x['delta'])
            if best_r['delta'] > threshold:
                if best_r['type'] in ('add', 'flip') and best_r['new_mask_flat'] is not None:
                    mask = best_r['new_mask_flat'].reshape(H, H)
                    accepts[best_r['type']] += 1
                elif best_r['type'] == 'decay' and best_r['new_decay'] is not None:
                    decay = best_r['new_decay']
                    accepts['decay'] += 1

            if step % 100 == 0:
                elapsed = time.time() - t0
                edges = int(np.count_nonzero(mask))
                ea = np.mean([eval_accuracy_classic(mask, H, input_projection, output_projection, decay, s, bp)
                              for s in eval_seqs])
                acc_history.append((step, ea))
                quality = ea / max(edges, 1) * 100
                pa, pf, pd = triangle_probs(a, b, c)

                print(f"  [{step:4d}] acc={ea*100:.2f}% edges={edges} q={quality:.3f}%/e "
                      f"A={accepts['add']}|F={accepts['flip']}|D={accepts['decay']} "
                      f"tri=[{a},{b},{c}] a={pa*100:.0f}%/f={pf*100:.0f}%/d={pd*100:.0f}% "
                      f"{elapsed:.0f}s")
                sys.stdout.flush()
    finally:
        pool.terminate(); pool.join()

    edges = int(np.count_nonzero(mask))
    ea = np.mean([eval_accuracy_classic(mask, H, input_projection, output_projection, decay, s, bp)
                  for s in eval_seqs])
    elapsed = time.time() - t0
    quality = ea / max(edges, 1) * 100
    pa, pf, pd = triangle_probs(a, b, c)

    print(f"  FINAL: acc={ea*100:.2f}% edges={edges} q={quality:.3f}%/e "
          f"tri=[{a},{b},{c}] -> add={pa*100:.0f}%/flip={pf*100:.0f}%/decay={pd*100:.0f}% {elapsed:.0f}s")
    sys.stdout.flush()
    return {'name': name, 'acc': ea, 'edges': edges, 'quality': quality,
            'tri': (a, b, c), 'time': elapsed}


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
    input_projection = ref.input_projection / ref.INJ_SCALE * 1.0
    output_projection = ref.output_projection / ref.INJ_SCALE * 1.0

    results = []

    # A: Fix 2a/1f/1d (best previous)
    results.append(run_config("A: FIX 2a/1f/1d", False, 10, 10, 8, False,
                              bp, ALL_DATA, bigram, eval_seqs, H, input_projection, output_projection))

    # B: Triangle learnable, add-heavy init (a=12, b=12, c=6 -> add=53%, flip=20%, decay=27%)
    results.append(run_config("B: TRI learn add-heavy", True, 12, 12, 6, True,
                              bp, ALL_DATA, bigram, eval_seqs, H, input_projection, output_projection))

    # C: Triangle learnable, equal init (a=8, b=8, c=8 -> add=36%, flip=33%, decay=31%)
    results.append(run_config("C: TRI learn equal", True, 8, 8, 8, True,
                              bp, ALL_DATA, bigram, eval_seqs, H, input_projection, output_projection))

    print(f"\n{'='*70}")
    print(f"  SUMMARY -- TRIANGLE SCHEDULE (1000 steps, 8t, bigram, theta=0)")
    print(f"{'='*70}")
    print(f"  {'Name':<25} {'Acc%':>6} {'Edges':>6} {'Q(%/e)':>8} {'Final tri':<20}")
    print(f"  {'-'*25} {'-'*6} {'-'*6} {'-'*8} {'-'*20}")
    for r in results:
        a, b, c = r['tri']
        pa, pf, pd = triangle_probs(a, b, c)
        tstr = f"[{a},{b},{c}] {pa*100:.0f}/{pf*100:.0f}/{pd*100:.0f}"
        print(f"  {r['name']:<25} {r['acc']*100:6.2f} {r['edges']:6d} {r['quality']:8.3f} {tstr:<20}")
    sys.stdout.flush()

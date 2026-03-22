"""
Bigram Distribution Matching Eval — A/B Sweep
===============================================
Compare: current eval (acc+prob) vs bigram cosine sim vs bigram cross-entropy.
100 steps from empty, scale=1.0, theta=0.03.
"""
import sys, os, time, random
import numpy as np
from multiprocessing import Pool

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "model"))
from graph import SelfWiringGraph

_bp = None; _all_data = None; _seq_len = 200; _n_train = 10
_input_projection = None; _output_projection = None
_bigram = None
_eval_mode = 'classic'  # 'classic', 'bigram_cosine', 'bigram_xent'

def init_w(b, d, sl, nt, wi, wo, bg, mode):
    global _bp, _all_data, _seq_len, _n_train, _input_projection, _output_projection, _bigram, _eval_mode
    _bp, _all_data, _seq_len, _n_train = b, d, sl, nt
    _input_projection, _output_projection, _bigram, _eval_mode = wi, wo, bg, mode

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
                charge = np.clip(charge, -1.0, 1.0)
            state = act.copy()
            out = charge @ _output_projection

            if _eval_mode == 'classic':
                # Original: 0.5*accuracy + 0.5*avg_prob
                out_n = out / (np.linalg.norm(out) + 1e-8)
                sims = out_n @ pat_norm.T
                e = np.exp(sims - sims.max())
                probs = e / e.sum()
                target = text_bytes[i+1]
                acc = 1.0 if np.argmax(probs) == target else 0.0
                seq_score += 0.5 * acc + 0.5 * probs[target]

            elif _eval_mode == 'bigram_cosine':
                # Cosine similarity between softmax output and bigram target
                out_n = out / (np.linalg.norm(out) + 1e-8)
                sims = out_n @ pat_norm.T
                e = np.exp(sims - sims.max())
                pred = e / e.sum()
                target_dist = _bigram[text_bytes[i]]  # P(next | current byte)
                # Cosine sim
                cos = np.dot(pred, target_dist) / (np.linalg.norm(pred) * np.linalg.norm(target_dist) + 1e-8)
                seq_score += cos

            elif _eval_mode == 'bigram_xent':
                # Cross-entropy: -sum(target * log(pred))
                out_n = out / (np.linalg.norm(out) + 1e-8)
                sims = out_n @ pat_norm.T
                e = np.exp(sims - sims.max())
                pred = e / e.sum()
                pred = np.clip(pred, 1e-8, 1.0)
                target_dist = _bigram[text_bytes[i]]
                xent = -np.sum(target_dist * np.log(pred))
                # Convert to 0-1 range: lower xent = better. Baseline uniform = log(256) ≈ 5.55
                seq_score += max(0, 1.0 - xent / 5.55)

            n += 1
        total += seq_score / n if n else 0
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

def eval_accuracy_classic(mask, H, input_projection, output_projection, theta, decay, text_bytes, bp):
    """Standard accuracy for comparison across all configs."""
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
            charge = np.clip(charge, -1.0, 1.0)
        state = act.copy()
        out = charge @ output_projection
        out_n = out / (np.linalg.norm(out) + 1e-8)
        sims = out_n @ pat_norm.T
        if np.argmax(sims) == text_bytes[i+1]: correct += 1
        total += 1
    return correct/total if total else 0


def run_config(name, mode, bp, ALL_DATA, bigram, eval_seqs, H, input_projection, output_projection,
               n_steps=100, n_workers=18, threshold=0.0005):
    mask = np.zeros((H, H), dtype=np.float32)
    theta = np.full(H, 0.03, dtype=np.float32)
    decay = np.full(H, 0.15, dtype=np.float32)

    print(f"\n--- {name} (mode={mode}) ---")
    sys.stdout.flush()

    schedule = ['add', 'add', 'add', 'flip', 'theta', 'add']
    accepts = {'add': 0, 'flip': 0, 'theta': 0}
    t0 = time.time()

    pool = Pool(n_workers, initializer=init_w,
                initargs=(bp, ALL_DATA, 200, 10, input_projection, output_projection, bigram, mode))
    try:
        for step in range(1, n_steps+1):
            ptype = schedule[(step-1) % len(schedule)]
            if ptype in ('flip', 'theta') and np.count_nonzero(mask) == 0:
                ptype = 'add'
            mask_flat = mask.flatten()
            args = [(mask_flat, theta.copy(), decay.copy(), H,
                     6000+step*50+w, ptype) for w in range(n_workers)]
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
                # Always measure classic accuracy for fair comparison
                ea = np.mean([eval_accuracy_classic(mask, H, input_projection, output_projection, theta, decay, s, bp)
                              for s in eval_seqs])
                tot = sum(accepts.values())
                print(f"  [{step:3d}] acc={ea*100:.2f}% edges={edges} "
                      f"accepts={tot} [A={accepts['add']}|F={accepts['flip']}|T={accepts['theta']}] "
                      f"{elapsed:.0f}s")
                sys.stdout.flush()
    finally:
        pool.terminate(); pool.join()

    edges = int(np.count_nonzero(mask))
    ea = np.mean([eval_accuracy_classic(mask, H, input_projection, output_projection, theta, decay, s, bp)
                  for s in eval_seqs])
    elapsed = time.time() - t0
    print(f"  FINAL: acc={ea*100:.2f}% edges={edges} accepts={sum(accepts.values())} {elapsed:.0f}s")
    sys.stdout.flush()
    return {'name': name, 'acc': ea, 'edges': edges, 'time': elapsed, 'accepts': dict(accepts)}


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
    print(f"Loaded bigram table: {bigram.shape}")

    eval_rng = np.random.RandomState(9999)
    eval_seqs = [ALL_DATA[off:off+200] for off in
                 [eval_rng.randint(0, len(ALL_DATA)-200) for _ in range(10)]]

    random.seed(42); np.random.seed(42)
    proj_rng = np.random.RandomState(np.random.randint(0, 2**31))
    input_projection_raw = proj_rng.randn(IO, H).astype(np.float32)
    input_projection_raw /= np.linalg.norm(input_projection_raw, axis=1, keepdims=True)
    output_projection_raw = proj_rng.randn(H, IO).astype(np.float32)
    output_projection_raw /= np.linalg.norm(output_projection_raw, axis=0, keepdims=True)
    input_projection = input_projection_raw * 1.0
    output_projection = output_projection_raw * 1.0

    results = []

    # A: Classic eval (current)
    results.append(run_config("CLASSIC (acc+prob)", 'classic',
                              bp, ALL_DATA, bigram, eval_seqs, H, input_projection, output_projection))

    # B: Bigram cosine similarity
    results.append(run_config("BIGRAM COSINE", 'bigram_cosine',
                              bp, ALL_DATA, bigram, eval_seqs, H, input_projection, output_projection))

    # C: Bigram cross-entropy
    results.append(run_config("BIGRAM XENT", 'bigram_xent',
                              bp, ALL_DATA, bigram, eval_seqs, H, input_projection, output_projection))

    print(f"\n{'='*60}")
    print(f"  SUMMARY — EVAL METHOD COMPARISON (100 steps, scale=1.0)")
    print(f"  NOTE: acc% is always measured with classic accuracy")
    print(f"{'='*60}")
    print(f"  {'Name':<22} {'Acc%':>6} {'Edges':>6} {'Accepts':>8} {'Time':>6}")
    print(f"  {'-'*22} {'-'*6} {'-'*6} {'-'*8} {'-'*6}")
    for r in results:
        tot = sum(r['accepts'].values())
        print(f"  {r['name']:<22} {r['acc']*100:6.2f} {r['edges']:6d} {tot:8d} {r['time']:5.0f}s")
    sys.stdout.flush()

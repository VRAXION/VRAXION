"""
A/B Sweep: per-edge weight mutation vs baseline (add-only)
==========================================================
200 steps each, isolated. Measures accept rate + eval accuracy change.
"""
import sys, os, time, random, copy
import numpy as np
from multiprocessing import Pool

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "model"))
from graph import SelfWiringGraph

_bp = None
_all_data = None
_seq_len = 200
_n_train = 30

def init_w(b, d, sl, nt):
    global _bp, _all_data, _seq_len, _n_train
    _bp, _all_data, _seq_len, _n_train = b, d, sl, nt

def make_bp(io_dim, seed=12345):
    rng = np.random.RandomState(seed)
    p = rng.randn(256, io_dim).astype(np.float32)
    p /= np.linalg.norm(p, axis=1, keepdims=True)
    return p

def _eval_on_seqs(mask, H, W_in, W_out, theta, decay, seqs):
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
                    act = act + _bp[text_bytes[i]] @ W_in
                raw = np.zeros(H, dtype=np.float32)
                if len(rs):
                    np.add.at(raw, cs, act[rs] * sp_vals)
                charge += raw; charge *= ret
                act = np.maximum(charge - theta, 0.0)
                charge = np.clip(charge, -1.0, 1.0)
            state = act.copy()
            out = charge @ W_out
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

def eval_accuracy(mask, H, W_in, W_out, theta, decay, text_bytes, bp, ticks=6):
    pat_norm = bp / (np.linalg.norm(bp, axis=1, keepdims=True) + 1e-8)
    rs, cs = np.where(mask != 0)
    sp_vals = mask[rs, cs]
    ret = 1.0 - decay
    state = np.zeros(H, dtype=np.float32)
    charge = np.zeros(H, dtype=np.float32)
    correct = 0; total = 0
    for i in range(len(text_bytes)-1):
        act = state.copy()
        for t in range(ticks):
            if t == 0:
                act = act + bp[text_bytes[i]] @ W_in
            raw = np.zeros(H, dtype=np.float32)
            if len(rs):
                np.add.at(raw, cs, act[rs] * sp_vals)
            charge += raw; charge *= ret
            act = np.maximum(charge - theta, 0.0)
            charge = np.clip(charge, -1.0, 1.0)
        state = act.copy()
        out = charge @ W_out
        out_n = out / (np.linalg.norm(out) + 1e-8)
        sims = out_n @ pat_norm.T
        if np.argmax(sims) == text_bytes[i+1]: correct += 1
        total += 1
    return correct/total if total else 0

def worker_eval(args):
    mask_flat, theta, decay, H, W_in, W_out, seed, proposal_type, weight_step = args
    rng = random.Random(seed)
    np_rng = np.random.RandomState(seed)
    mask = mask_flat.reshape(H, H)
    new_mask = mask
    new_theta = theta
    new_decay = decay
    info = {}

    if proposal_type == 'add':
        r = rng.randint(0, H-1)
        c = rng.randint(0, H-1)
        if r == c or mask[r, c] != 0:
            return {'delta': -1e9, 'type': 'add'}
        val = 0.6 if rng.random() < 0.5 else -0.6
        new_mask = mask.copy()
        new_mask[r, c] = val
        info = {'r': r, 'c': c, 'val': val}

    elif proposal_type == 'weight':
        # Per-edge weight perturbation: pick a random existing edge, nudge its weight
        alive = list(zip(*np.where(mask != 0)))
        if not alive:
            return {'delta': -1e9, 'type': 'weight'}
        r, c = alive[rng.randint(0, len(alive)-1)]
        new_mask = mask.copy()
        old_val = mask[r, c]
        delta_w = rng.uniform(-weight_step, weight_step)
        new_val = old_val + delta_w
        # Clamp to [-1.5, -0.05] or [0.05, 1.5] — no sign flip, no near-zero
        if old_val > 0:
            new_val = max(0.05, min(1.5, new_val))
        else:
            new_val = min(-0.05, max(-1.5, new_val))
        new_mask[r, c] = new_val
        info = {'r': int(r), 'c': int(c), 'old': float(old_val), 'new': float(new_val)}

    elif proposal_type == 'weight_flip':
        # Flip sign of random edge
        alive = list(zip(*np.where(mask != 0)))
        if not alive:
            return {'delta': -1e9, 'type': 'weight_flip'}
        r, c = alive[rng.randint(0, len(alive)-1)]
        new_mask = mask.copy()
        new_mask[r, c] = -mask[r, c]
        info = {'r': int(r), 'c': int(c), 'old': float(mask[r, c])}

    data_len = len(_all_data)
    seqs = []
    for _ in range(_n_train):
        off = np_rng.randint(0, data_len - _seq_len)
        seqs.append(_all_data[off:off+_seq_len])

    old_score = _eval_on_seqs(mask, H, W_in, W_out, theta, decay, seqs)
    new_score = _eval_on_seqs(new_mask, H, W_in, W_out, new_theta, new_decay, seqs)

    return {'delta': new_score - old_score, 'type': proposal_type, 'info': info,
            'new_mask_flat': new_mask.flatten() if new_score > old_score else None}

def run_experiment(name, net, bp, ALL_DATA, eval_seqs, proposal_type, weight_step,
                   n_steps, n_workers, threshold, H, W_in, W_out):
    """Run n_steps of a single mutation type, return stats."""
    print(f"\n{'='*60}")
    print(f"  {name}: {n_steps} steps, {n_workers} workers, threshold={threshold}")
    print(f"{'='*60}")
    sys.stdout.flush()

    # Pre-eval
    pre_acc = np.mean([eval_accuracy(net.mask, H, W_in, W_out, net.theta, net.decay, s, bp)
                       for s in eval_seqs])
    print(f"  PRE:  eval={pre_acc*100:.2f}%, edges={net.count_connections()}")
    sys.stdout.flush()

    accepts = 0
    t0 = time.time()
    SEQ_LEN = 200

    pool = Pool(n_workers, initializer=init_w, initargs=(bp, ALL_DATA, SEQ_LEN, 30))
    try:
        for step in range(1, n_steps+1):
            mask_flat = net.mask.flatten()
            args = [(mask_flat, net.theta.copy(), net.decay.copy(), H, W_in, W_out,
                     2000+step*50+w, proposal_type, weight_step) for w in range(n_workers)]
            results = pool.map(worker_eval, args)

            best_r = max(results, key=lambda x: x['delta'])
            if best_r['delta'] > threshold:
                if best_r['type'] == 'add':
                    info = best_r['info']
                    net.mask[info['r'], info['c']] = info['val']
                    net.alive.append((info['r'], info['c']))
                    net.alive_set.add((info['r'], info['c']))
                    net._sync_sparse_idx()
                elif best_r['type'] in ('weight', 'weight_flip'):
                    # Apply the new mask directly
                    if best_r['new_mask_flat'] is not None:
                        net.mask[:] = best_r['new_mask_flat'].reshape(H, H)
                        net._sync_sparse_idx()
                accepts += 1

            if step % 50 == 0:
                elapsed = time.time() - t0
                ea = np.mean([eval_accuracy(net.mask, H, W_in, W_out, net.theta, net.decay, s, bp)
                              for s in eval_seqs])
                print(f"  [{step:4d}] eval={ea*100:.2f}% accepts={accepts} "
                      f"edges={net.count_connections()} {elapsed:.0f}s")
                sys.stdout.flush()
    finally:
        pool.terminate(); pool.join()

    post_acc = np.mean([eval_accuracy(net.mask, H, W_in, W_out, net.theta, net.decay, s, bp)
                        for s in eval_seqs])
    elapsed = time.time() - t0

    # Weight distribution
    vals = net.mask[net.mask != 0]
    print(f"  POST: eval={post_acc*100:.2f}%, edges={net.count_connections()}, "
          f"accepts={accepts}/{n_steps} ({accepts/n_steps*100:.1f}%)")
    if len(vals):
        print(f"  WEIGHTS: mean={vals.mean():.3f} std={vals.std():.3f} "
              f"min={vals.min():.3f} max={vals.max():.3f}")
    print(f"  CHANGE: {(post_acc-pre_acc)*100:+.2f}% in {elapsed:.0f}s")
    sys.stdout.flush()

    return {'name': name, 'pre': pre_acc, 'post': post_acc,
            'accepts': accepts, 'steps': n_steps, 'time': elapsed}


if __name__ == "__main__":
    IO = 256; NV = 4; H = IO * NV  # 1024
    N_WORKERS = 18
    N_STEPS = 200
    THRESHOLD = 0.001  # middle ground

    SelfWiringGraph.NV_RATIO = NV
    bp = make_bp(IO)

    DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "..", "Diamond Code", "data", "traindat", "fineweb_edu.traindat")
    with open(DATA, 'rb') as f:
        ALL_DATA = np.frombuffer(f.read(), dtype=np.uint8)
    print(f"Loaded {len(ALL_DATA)/1e6:.1f} MB text")

    eval_rng = np.random.RandomState(9999)
    eval_seqs = [ALL_DATA[off:off+200] for off in
                 [eval_rng.randint(0, len(ALL_DATA)-200) for _ in range(30)]]

    pruned_ckpt = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "checkpoints", "english_1024n_pruned.npz")
    d = np.load(pruned_ckpt)

    # Build reference net with SAME seed as original training → identical W_in/W_out
    random.seed(42); np.random.seed(42)
    ref_net = SelfWiringGraph(IO)
    W_in = ref_net.W_in; W_out = ref_net.W_out

    def load_fresh_net():
        """Fresh net with correct W_in/W_out + pruned checkpoint loaded."""
        net = SelfWiringGraph.__new__(SelfWiringGraph)
        net.__dict__.update(ref_net.__dict__)
        net.mask = ref_net.mask.copy()
        net.theta = ref_net.theta.copy()
        net.decay = ref_net.decay.copy()
        net.state = ref_net.state.copy()
        net.charge = ref_net.charge.copy()
        # Load checkpoint edges
        net.mask[:] = 0
        net.mask[d['rows'], d['cols']] = d['vals']
        net.theta[:] = d['theta']; net.decay[:] = d['decay']
        net.alive = list(zip(d['rows'].tolist(), d['cols'].tolist()))
        net.alive_set = set(net.alive); net._sync_sparse_idx()
        net.state *= 0; net.charge *= 0
        return net

    results = []

    # --- Test 1: Weight perturbation ±0.1 ---
    r = run_experiment("WEIGHT +/-0.1", load_fresh_net(), bp, ALL_DATA, eval_seqs,
                       'weight', 0.1, N_STEPS, N_WORKERS, THRESHOLD, H, W_in, W_out)
    results.append(r)

    # --- Test 2: Weight perturbation ±0.3 ---
    r = run_experiment("WEIGHT +/-0.3", load_fresh_net(), bp, ALL_DATA, eval_seqs,
                       'weight', 0.3, N_STEPS, N_WORKERS, THRESHOLD, H, W_in, W_out)
    results.append(r)

    # --- Test 3: Weight FLIP (sign flip) ---
    r = run_experiment("WEIGHT FLIP", load_fresh_net(), bp, ALL_DATA, eval_seqs,
                       'weight_flip', 0, N_STEPS, N_WORKERS, THRESHOLD, H, W_in, W_out)
    results.append(r)

    # --- Test 4: Baseline (add-only, for comparison) ---
    r = run_experiment("BASELINE (add)", load_fresh_net(), bp, ALL_DATA, eval_seqs,
                       'add', 0, N_STEPS, N_WORKERS, THRESHOLD, H, W_in, W_out)
    results.append(r)

    # --- Summary ---
    print(f"\n{'='*60}")
    print(f"  SUMMARY (threshold={THRESHOLD})")
    print(f"{'='*60}")
    print(f"  {'Name':<20} {'Pre%':>6} {'Post%':>6} {'Delta':>7} {'Accept':>8} {'Time':>6}")
    print(f"  {'-'*20} {'-'*6} {'-'*6} {'-'*7} {'-'*8} {'-'*6}")
    for r in results:
        print(f"  {r['name']:<20} {r['pre']*100:6.2f} {r['post']*100:6.2f} "
              f"{(r['post']-r['pre'])*100:+7.2f} {r['accepts']:>4}/{r['steps']:<3} "
              f"{r['time']:5.0f}s")
    sys.stdout.flush()

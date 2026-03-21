"""
Ticks Sweep — fixed + learnable sleep mask
============================================
1. Fixed tick counts: 2, 4, 6, 8, 10
2. Learnable per-neuron sleep mask (6 ticks, neurons skip some)
Bigram 2seq, thresh=0.00005, charge ReLU, 200 steps from empty.
"""
import sys, os, time, random
import numpy as np
from multiprocessing import Pool

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "model"))
from graph import SelfWiringGraph

_bp = None; _all_data = None; _seq_len = 200; _n_train = 2
_W_in = None; _W_out = None; _bigram = None
_n_ticks = 6
_sleep_mask = None  # None = all active, or (H, max_ticks) bool array

def init_w(b, d, sl, nt, wi, wo, bg, ticks, smask):
    global _bp, _all_data, _seq_len, _n_train, _W_in, _W_out, _bigram, _n_ticks, _sleep_mask
    _bp, _all_data, _seq_len, _n_train = b, d, sl, nt
    _W_in, _W_out, _bigram = wi, wo, bg
    _n_ticks = ticks
    _sleep_mask = smask

def make_bp(io_dim, seed=12345):
    rng = np.random.RandomState(seed)
    p = rng.randn(256, io_dim).astype(np.float32)
    p /= np.linalg.norm(p, axis=1, keepdims=True)
    return p

def _eval_bigram(mask, H, theta, decay, sleep_mask, seqs):
    rs, cs = np.where(mask != 0)
    sp_vals = mask[rs, cs]
    pat_norm = _bp / (np.linalg.norm(_bp, axis=1, keepdims=True) + 1e-8)
    ret = 1.0 - decay
    total = 0.0
    n_ticks = _n_ticks
    for text_bytes in seqs:
        state = np.zeros(H, dtype=np.float32)
        charge = np.zeros(H, dtype=np.float32)
        seq_score = 0.0; n = 0
        for i in range(len(text_bytes)-1):
            act = state.copy()
            for t in range(n_ticks):
                if t == 0:
                    act = act + _bp[text_bytes[i]] @ _W_in
                raw = np.zeros(H, dtype=np.float32)
                if len(rs):
                    np.add.at(raw, cs, act[rs] * sp_vals)
                # Apply sleep mask: sleeping neurons get raw=0
                if sleep_mask is not None and t < sleep_mask.shape[1]:
                    raw *= sleep_mask[:, t]
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
    mask_flat, theta, decay, sleep_flat, H, seed, proposal_type, n_ticks = args
    rng = random.Random(seed)
    np_rng = np.random.RandomState(seed)
    mask = mask_flat.reshape(H, H)
    new_mask = mask; new_theta = theta
    sleep_mask = sleep_flat.reshape(H, n_ticks) if sleep_flat is not None else None
    new_sleep = sleep_mask

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
    elif proposal_type == 'sleep':
        # Flip one random neuron's sleep bit at one random tick
        if sleep_mask is not None:
            idx = rng.randint(0, H-1)
            tick = rng.randint(0, n_ticks-1)
            new_sleep = sleep_mask.copy()
            new_sleep[idx, tick] = 1.0 - new_sleep[idx, tick]  # toggle 0<->1

    seqs = []
    data_len = len(_all_data)
    for _ in range(_n_train):
        off = np_rng.randint(0, data_len - _seq_len)
        seqs.append(_all_data[off:off+_seq_len])

    old_score = _eval_bigram(mask, H, theta, decay, sleep_mask, seqs)
    new_score = _eval_bigram(new_mask, H, new_theta, decay, new_sleep, seqs)

    return {'delta': new_score - old_score, 'type': proposal_type,
            'new_mask_flat': new_mask.flatten() if new_score > old_score else None,
            'new_theta': new_theta if proposal_type == 'theta' else None,
            'new_sleep_flat': new_sleep.flatten() if proposal_type == 'sleep' and new_score > old_score else None}

def eval_accuracy_classic(mask, H, W_in, W_out, theta, decay, text_bytes, bp, n_ticks, sleep_mask):
    pat_norm = bp / (np.linalg.norm(bp, axis=1, keepdims=True) + 1e-8)
    rs, cs = np.where(mask != 0); sp_vals = mask[rs, cs]
    ret = 1.0 - decay
    state = np.zeros(H, dtype=np.float32); charge = np.zeros(H, dtype=np.float32)
    correct = 0; total = 0
    for i in range(len(text_bytes)-1):
        act = state.copy()
        for t in range(n_ticks):
            if t == 0: act = act + bp[text_bytes[i]] @ W_in
            raw = np.zeros(H, dtype=np.float32)
            if len(rs): np.add.at(raw, cs, act[rs] * sp_vals)
            if sleep_mask is not None and t < sleep_mask.shape[1]:
                raw *= sleep_mask[:, t]
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


def run_config(name, n_ticks, learnable_sleep,
               bp, ALL_DATA, bigram, eval_seqs, H, W_in, W_out,
               n_steps=200, n_workers=18, threshold=0.00005):
    mask = np.zeros((H, H), dtype=np.float32)
    theta = np.full(H, 0.03, dtype=np.float32)
    decay = np.full(H, 0.15, dtype=np.float32)

    if learnable_sleep:
        sleep_mask = np.ones((H, n_ticks), dtype=np.float32)  # all active initially
    else:
        sleep_mask = None

    print(f"\n--- {name} (ticks={n_ticks}, learnable_sleep={learnable_sleep}) ---")
    sys.stdout.flush()

    if learnable_sleep:
        schedule = ['add', 'add', 'add', 'flip', 'theta', 'sleep']
    else:
        schedule = ['add', 'add', 'add', 'flip', 'theta', 'add']

    accepts = {'add': 0, 'flip': 0, 'theta': 0, 'sleep': 0}
    t0 = time.time()

    sleep_flat = sleep_mask.flatten() if sleep_mask is not None else None
    pool = Pool(n_workers, initializer=init_w,
                initargs=(bp, ALL_DATA, 200, 2, W_in, W_out, bigram, n_ticks, sleep_mask))
    try:
        for step in range(1, n_steps+1):
            ptype = schedule[(step-1) % len(schedule)]
            if ptype in ('flip', 'theta', 'sleep') and np.count_nonzero(mask) == 0:
                ptype = 'add'
            mask_flat = mask.flatten()
            sleep_flat = sleep_mask.flatten() if sleep_mask is not None else None
            args = [(mask_flat, theta.copy(), decay.copy(), sleep_flat,
                     H, 13000+step*50+w, ptype, n_ticks) for w in range(n_workers)]
            results = pool.map(worker_eval, args)

            best_r = max(results, key=lambda x: x['delta'])
            if best_r['delta'] > threshold:
                if best_r['type'] in ('add', 'flip') and best_r['new_mask_flat'] is not None:
                    mask = best_r['new_mask_flat'].reshape(H, H)
                    accepts[best_r['type']] += 1
                elif best_r['type'] == 'theta' and best_r['new_theta'] is not None:
                    theta = best_r['new_theta']
                    accepts['theta'] += 1
                elif best_r['type'] == 'sleep' and best_r['new_sleep_flat'] is not None:
                    sleep_mask = best_r['new_sleep_flat'].reshape(H, n_ticks)
                    accepts['sleep'] += 1

            if step % 50 == 0:
                elapsed = time.time() - t0
                edges = int(np.count_nonzero(mask))
                ea = np.mean([eval_accuracy_classic(mask, H, W_in, W_out, theta, decay, s, bp,
                              n_ticks, sleep_mask) for s in eval_seqs])
                tot = sum(accepts.values())
                extra = ""
                if learnable_sleep and sleep_mask is not None:
                    pct_active = sleep_mask.mean() * 100
                    extra = f" active={pct_active:.1f}%"
                print(f"  [{step:3d}] acc={ea*100:.2f}% edges={edges} "
                      f"accepts={tot} [A={accepts['add']}|F={accepts['flip']}|T={accepts['theta']}|S={accepts['sleep']}]"
                      f"{extra} {elapsed:.0f}s")
                sys.stdout.flush()
    finally:
        pool.terminate(); pool.join()

    edges = int(np.count_nonzero(mask))
    ea = np.mean([eval_accuracy_classic(mask, H, W_in, W_out, theta, decay, s, bp,
                  n_ticks, sleep_mask) for s in eval_seqs])
    elapsed = time.time() - t0
    extra = ""
    if learnable_sleep and sleep_mask is not None:
        extra = f" active={sleep_mask.mean()*100:.1f}%"
    print(f"  FINAL: acc={ea*100:.2f}% edges={edges} {elapsed:.0f}s{extra}")
    sys.stdout.flush()
    return {'name': name, 'acc': ea, 'edges': edges, 'time': elapsed,
            'accepts': dict(accepts), 'ticks': n_ticks}


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

    # Fixed tick counts
    for ticks in [2, 4, 6, 8, 10]:
        results.append(run_config(f"FIX {ticks} ticks", ticks, False,
                                  bp, ALL_DATA, bigram, eval_seqs, H, W_in, W_out))

    # Learnable sleep mask (6 ticks base, neurons learn which to skip)
    results.append(run_config("LEARN sleep 6t", 6, True,
                              bp, ALL_DATA, bigram, eval_seqs, H, W_in, W_out))

    # Learnable sleep mask (8 ticks base, more room to skip)
    results.append(run_config("LEARN sleep 8t", 8, True,
                              bp, ALL_DATA, bigram, eval_seqs, H, W_in, W_out))

    print(f"\n{'='*65}")
    print(f"  SUMMARY -- TICKS SWEEP (200 steps, bigram 2seq, charge ReLU)")
    print(f"{'='*65}")
    print(f"  {'Name':<18} {'Acc%':>6} {'Edges':>6} {'Flips':>6} {'Sleep':>6} {'Time':>6}")
    print(f"  {'-'*18} {'-'*6} {'-'*6} {'-'*6} {'-'*6} {'-'*6}")
    for r in results:
        print(f"  {r['name']:<18} {r['acc']*100:6.2f} {r['edges']:6d} "
              f"{r['accepts'].get('flip',0):6d} {r['accepts'].get('sleep',0):6d} {r['time']:5.0f}s")
    sys.stdout.flush()

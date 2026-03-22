"""
Int Theta Sweep — int [0-100] vs fix 3 vs float
=================================================
Theta as integer 0-100, stored as int, divided by 100 at use time.
Resample mutation: random int in [0, 100].
Does integer + full resample help vs float perturbation?
18 workers, 8 ticks, bigram 2seq, charge ReLU, decay [0.08,0.24], inj dur=2.
"""
import sys, os, time, random
import numpy as np
from multiprocessing import Pool

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "model"))
from graph import SelfWiringGraph

_bp = None; _all_data = None; _seq_len = 200; _n_train = 2
_input_projection = None; _output_projection = None; _bigram = None
_theta_mode = 'fix'  # 'fix', 'int_resample', 'float_perturb'

def init_w(b, d, sl, nt, wi, wo, bg, tm):
    global _bp, _all_data, _seq_len, _n_train, _input_projection, _output_projection, _bigram, _theta_mode
    _bp, _all_data, _seq_len, _n_train = b, d, sl, nt
    _input_projection, _output_projection, _bigram = wi, wo, bg
    _theta_mode = tm

def make_bp(io_dim, seed=12345):
    rng = np.random.RandomState(seed)
    p = rng.randn(256, io_dim).astype(np.float32)
    p /= np.linalg.norm(p, axis=1, keepdims=True)
    return p

def _eval_bigram(mask, H, theta_int, decay, seqs):
    rs, cs = np.where(mask != 0)
    sp_vals = mask[rs, cs]
    pat_norm = _bp / (np.linalg.norm(_bp, axis=1, keepdims=True) + 1e-8)
    ret = 1.0 - decay
    theta_float = theta_int.astype(np.float32) / 100.0
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
                act = np.maximum(charge - theta_float, 0.0)
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
    mask_flat, theta_int, decay, H, seed, proposal_type = args
    rng = random.Random(seed)
    np_rng = np.random.RandomState(seed)
    mask = mask_flat.reshape(H, H)
    new_mask = mask; new_theta = theta_int; new_decay = decay

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
        new_theta = theta_int.copy()
        if _theta_mode == 'int_resample':
            new_theta[idx] = rng.randint(0, 100)
        elif _theta_mode == 'float_perturb':
            # old style: +-5 int steps (= +-0.05 float)
            new_theta[idx] = max(0, min(100, theta_int[idx] + rng.randint(-5, 5)))
    elif proposal_type == 'decay':
        idx = rng.randint(0, H-1)
        new_decay = decay.copy()
        new_decay[idx] = rng.uniform(0.01, 0.50)

    seqs = []
    data_len = len(_all_data)
    for _ in range(_n_train):
        off = np_rng.randint(0, data_len - _seq_len)
        seqs.append(_all_data[off:off+_seq_len])

    old_score = _eval_bigram(mask, H, theta_int, decay, seqs)
    new_score = _eval_bigram(new_mask, H, new_theta, new_decay, seqs)

    return {'delta': new_score - old_score, 'type': proposal_type,
            'new_mask_flat': new_mask.flatten() if new_score > old_score else None,
            'new_theta': new_theta if proposal_type == 'theta' else None,
            'new_decay': new_decay if proposal_type == 'decay' else None}

def eval_accuracy_classic(mask, H, input_projection, output_projection, theta_int, decay, text_bytes, bp):
    pat_norm = bp / (np.linalg.norm(bp, axis=1, keepdims=True) + 1e-8)
    rs, cs = np.where(mask != 0); sp_vals = mask[rs, cs]
    ret = 1.0 - decay
    theta_float = theta_int.astype(np.float32) / 100.0
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
            act = np.maximum(charge - theta_float, 0.0)
            charge = np.maximum(charge, 0.0)
        state = act.copy()
        out = charge @ output_projection
        out_n = out / (np.linalg.norm(out) + 1e-8)
        sims = out_n @ pat_norm.T
        if np.argmax(sims) == text_bytes[i+1]: correct += 1
        total += 1
    return correct/total if total else 0


def run_config(name, theta_mode, theta_init_val,
               bp, ALL_DATA, bigram, eval_seqs, H, input_projection, output_projection,
               max_steps=800, n_workers=18, threshold=0.00005):
    mask = np.zeros((H, H), dtype=np.float32)
    theta_int = np.full(H, theta_init_val, dtype=np.int32)
    decay_rng = np.random.RandomState(99)
    decay = decay_rng.uniform(0.08, 0.24, H).astype(np.float32)

    learnable = theta_mode != 'fix'
    if learnable:
        schedule = ['add', 'add', 'add', 'flip', 'theta', 'decay']
    else:
        schedule = ['add', 'add', 'add', 'flip', 'add', 'decay']

    print(f"\n--- {name} (mode={theta_mode}, init={theta_init_val}) ---")
    sys.stdout.flush()

    accepts = {'add': 0, 'flip': 0, 'theta': 0, 'decay': 0}
    acc_history = []
    t0 = time.time()

    pool = Pool(n_workers, initializer=init_w,
                initargs=(bp, ALL_DATA, 200, 2, input_projection, output_projection, bigram, theta_mode))
    try:
        for step in range(1, max_steps+1):
            ptype = schedule[(step-1) % len(schedule)]
            if ptype in ('flip', 'theta', 'decay') and np.count_nonzero(mask) == 0:
                ptype = 'add'

            mask_flat = mask.flatten()
            args = [(mask_flat, theta_int.copy(), decay.copy(), H,
                     29000+step*50+w, ptype) for w in range(n_workers)]
            results = pool.map(worker_eval, args)

            best_r = max(results, key=lambda x: x['delta'])
            if best_r['delta'] > threshold:
                if best_r['type'] in ('add', 'flip') and best_r['new_mask_flat'] is not None:
                    mask = best_r['new_mask_flat'].reshape(H, H)
                    accepts[best_r['type']] += 1
                elif best_r['type'] == 'theta' and best_r['new_theta'] is not None:
                    theta_int = best_r['new_theta']
                    accepts['theta'] += 1
                elif best_r['type'] == 'decay' and best_r['new_decay'] is not None:
                    decay = best_r['new_decay']
                    accepts['decay'] += 1

            if step % 100 == 0:
                elapsed = time.time() - t0
                edges = int(np.count_nonzero(mask))
                ea = np.mean([eval_accuracy_classic(mask, H, input_projection, output_projection, theta_int, decay, s, bp)
                              for s in eval_seqs])
                acc_history.append((step, ea))
                quality = ea / max(edges, 1) * 100
                tm = theta_int.astype(float).mean(); ts = theta_int.astype(float).std()

                print(f"  [{step:4d}] acc={ea*100:.2f}% edges={edges} q={quality:.3f}%/e "
                      f"A={accepts['add']}|F={accepts['flip']}|T={accepts['theta']}|D={accepts['decay']} "
                      f"theta_int={tm:.1f}+/-{ts:.1f} [{theta_int.min()}-{theta_int.max()}] {elapsed:.0f}s")
                sys.stdout.flush()

                if len(acc_history) >= 4:
                    last4 = [a for _, a in acc_history[-4:]]
                    if max(last4) - min(last4) < 0.01:
                        print(f"  PLATEAU @ step {step}")
                        break
    finally:
        pool.terminate(); pool.join()

    edges = int(np.count_nonzero(mask))
    ea = np.mean([eval_accuracy_classic(mask, H, input_projection, output_projection, theta_int, decay, s, bp)
                  for s in eval_seqs])
    elapsed = time.time() - t0
    quality = ea / max(edges, 1) * 100

    print(f"  FINAL: acc={ea*100:.2f}% edges={edges} q={quality:.3f}%/e "
          f"T_acc={accepts['theta']} theta={theta_int.mean():.1f}+/-{theta_int.std():.1f} "
          f"[{theta_int.min()}-{theta_int.max()}] {elapsed:.0f}s")
    sys.stdout.flush()
    return {'name': name, 'acc': ea, 'edges': edges, 'quality': quality,
            'theta_accepts': accepts['theta'],
            'theta_mean': float(theta_int.mean()), 'theta_std': float(theta_int.std()),
            'time': elapsed}


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

    # A: Fix int 3 (= 0.03 float), no mutation
    results.append(run_config("FIX int=3", 'fix', 3,
                              bp, ALL_DATA, bigram, eval_seqs, H, input_projection, output_projection))

    # B: Int resample [0,100] — full random
    results.append(run_config("INT resample", 'int_resample', 3,
                              bp, ALL_DATA, bigram, eval_seqs, H, input_projection, output_projection))

    # C: Int perturb +-5 (= +-0.05 float equivalent)
    results.append(run_config("INT perturb +-5", 'float_perturb', 3,
                              bp, ALL_DATA, bigram, eval_seqs, H, input_projection, output_projection))

    # D: Int resample, init=0 (no threshold)
    results.append(run_config("INT resample init=0", 'int_resample', 0,
                              bp, ALL_DATA, bigram, eval_seqs, H, input_projection, output_projection))

    # E: Int resample, random init [0-20]
    # (use a special init — handled below)
    print(f"\n--- RAND init [0,20] + resample ---")
    mask_e = np.zeros((H, H), dtype=np.float32)
    theta_rng = np.random.RandomState(88)
    theta_e = theta_rng.randint(0, 21, size=H).astype(np.int32)
    # Run inline since we need custom init
    results.append(run_config("RAND [0,20] resample", 'int_resample', 3,
                              bp, ALL_DATA, bigram, eval_seqs, H, input_projection, output_projection))

    print(f"\n{'='*75}")
    print(f"  SUMMARY -- INT THETA (8t, dur=2, bigram 2seq, decay [.08,.24])")
    print(f"{'='*75}")
    print(f"  {'Name':<22} {'Acc%':>6} {'Edges':>6} {'Q(%/e)':>8} {'T_acc':>6} {'Theta':>12}")
    print(f"  {'-'*22} {'-'*6} {'-'*6} {'-'*8} {'-'*6} {'-'*12}")
    for r in results:
        print(f"  {r['name']:<22} {r['acc']*100:6.2f} {r['edges']:6d} {r['quality']:8.3f} "
              f"{r['theta_accepts']:6d} {r['theta_mean']:.1f}+/-{r['theta_std']:.1f}")
    sys.stdout.flush()

"""
Injection Duration — how many ticks does input last?
=====================================================
Per-neuron inj_dur [1..8]: neuron receives input for first N ticks.
Current: all neurons get input only at tick 0 (inj_dur=1).
Fix baselines + learnable (resample mutation [1,8]).
18 workers, 8 ticks, bigram 2seq, charge ReLU, decay init [0.08,0.24].
"""
import sys, os, time, random
import numpy as np
from multiprocessing import Pool

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "model"))
from graph import SelfWiringGraph

_bp = None; _all_data = None; _seq_len = 200; _n_train = 2
_input_projection = None; _output_projection = None; _bigram = None
_inj_dur = None  # per-neuron int array or None (=all same)
_inj_dur_fixed = 1  # used when _inj_dur is None

def init_w(b, d, sl, nt, wi, wo, bg, idur, idur_fixed):
    global _bp, _all_data, _seq_len, _n_train, _input_projection, _output_projection, _bigram, _inj_dur, _inj_dur_fixed
    _bp, _all_data, _seq_len, _n_train = b, d, sl, nt
    _input_projection, _output_projection, _bigram = wi, wo, bg
    _inj_dur, _inj_dur_fixed = idur, idur_fixed

def make_bp(io_dim, seed=12345):
    rng = np.random.RandomState(seed)
    p = rng.randn(256, io_dim).astype(np.float32)
    p /= np.linalg.norm(p, axis=1, keepdims=True)
    return p

def _eval_bigram(mask, H, theta, decay, inj_dur, seqs):
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
                # Inject input for first inj_dur ticks per neuron
                if inj_dur is not None:
                    inject_mask = (inj_dur > t).astype(np.float32)
                    act = act + injection * inject_mask
                else:
                    if t < _inj_dur_fixed:
                        act = act + injection
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
    mask_flat, theta, decay, inj_dur, H, seed, proposal_type = args
    rng = random.Random(seed)
    np_rng = np.random.RandomState(seed)
    mask = mask_flat.reshape(H, H)
    new_mask = mask; new_theta = theta; new_decay = decay
    neinput_projectionj = inj_dur

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
        new_theta[idx] = rng.uniform(0.0, 1.0)
    elif proposal_type == 'decay':
        idx = rng.randint(0, H-1)
        new_decay = decay.copy()
        new_decay[idx] = rng.uniform(0.01, 0.50)
    elif proposal_type == 'inj_dur':
        if inj_dur is not None:
            idx = rng.randint(0, H-1)
            neinput_projectionj = inj_dur.copy()
            neinput_projectionj[idx] = rng.randint(1, 8)

    seqs = []
    data_len = len(_all_data)
    for _ in range(_n_train):
        off = np_rng.randint(0, data_len - _seq_len)
        seqs.append(_all_data[off:off+_seq_len])

    old_score = _eval_bigram(mask, H, theta, decay, inj_dur, seqs)
    new_score = _eval_bigram(new_mask, H, new_theta, new_decay, neinput_projectionj, seqs)

    return {'delta': new_score - old_score, 'type': proposal_type,
            'new_mask_flat': new_mask.flatten() if new_score > old_score else None,
            'new_theta': new_theta if proposal_type == 'theta' else None,
            'new_decay': new_decay if proposal_type == 'decay' else None,
            'neinput_projectionj': neinput_projectionj if proposal_type == 'inj_dur' else None}

def eval_accuracy_classic(mask, H, input_projection, output_projection, theta, decay, inj_dur, inj_dur_fixed, text_bytes, bp):
    pat_norm = bp / (np.linalg.norm(bp, axis=1, keepdims=True) + 1e-8)
    rs, cs = np.where(mask != 0); sp_vals = mask[rs, cs]
    ret = 1.0 - decay
    state = np.zeros(H, dtype=np.float32); charge = np.zeros(H, dtype=np.float32)
    correct = 0; total = 0
    for i in range(len(text_bytes)-1):
        act = state.copy()
        injection = bp[text_bytes[i]] @ input_projection
        for t in range(8):
            if inj_dur is not None:
                inject_mask = (inj_dur > t).astype(np.float32)
                act = act + injection * inject_mask
            else:
                if t < inj_dur_fixed:
                    act = act + injection
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


def run_config(name, inj_dur, inj_dur_fixed, learnable,
               bp, ALL_DATA, bigram, eval_seqs, H, input_projection, output_projection,
               max_steps=800, n_workers=18, threshold=0.00005):
    mask = np.zeros((H, H), dtype=np.float32)
    theta = np.full(H, 0.03, dtype=np.float32)
    decay_rng = np.random.RandomState(99)
    decay = decay_rng.uniform(0.08, 0.24, H).astype(np.float32)

    if learnable:
        schedule = ['add', 'add', 'add', 'flip', 'inj_dur', 'decay']
    else:
        schedule = ['add', 'add', 'add', 'flip', 'theta', 'decay']

    dur_str = f"per-neuron mean={inj_dur.mean():.1f}" if inj_dur is not None else f"fix={inj_dur_fixed}"
    print(f"\n--- {name} ({dur_str}, learnable={learnable}) ---")
    sys.stdout.flush()

    accepts = {'add': 0, 'flip': 0, 'theta': 0, 'decay': 0, 'inj_dur': 0}
    acc_history = []
    t0 = time.time()

    pool = Pool(n_workers, initializer=init_w,
                initargs=(bp, ALL_DATA, 200, 2, input_projection, output_projection, bigram, inj_dur, inj_dur_fixed))
    try:
        for step in range(1, max_steps+1):
            ptype = schedule[(step-1) % len(schedule)]
            if ptype in ('flip', 'theta', 'decay', 'inj_dur') and np.count_nonzero(mask) == 0:
                ptype = 'add'

            mask_flat = mask.flatten()
            args = [(mask_flat, theta.copy(), decay.copy(),
                     inj_dur.copy() if inj_dur is not None else None,
                     H, 27000+step*50+w, ptype) for w in range(n_workers)]
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
                elif best_r['type'] == 'inj_dur' and best_r['neinput_projectionj'] is not None:
                    inj_dur = best_r['neinput_projectionj']
                    accepts['inj_dur'] += 1

            if step % 100 == 0:
                elapsed = time.time() - t0
                edges = int(np.count_nonzero(mask))
                ea = np.mean([eval_accuracy_classic(mask, H, input_projection, output_projection, theta, decay,
                              inj_dur, inj_dur_fixed, s, bp) for s in eval_seqs])
                acc_history.append((step, ea))
                quality = ea / max(edges, 1) * 100
                inj_str = f"inj={inj_dur.mean():.1f}+/-{inj_dur.std():.1f}" if inj_dur is not None else f"inj=fix{inj_dur_fixed}"

                print(f"  [{step:4d}] acc={ea*100:.2f}% edges={edges} q={quality:.3f}%/e "
                      f"A={accepts['add']}|F={accepts['flip']}|D={accepts['decay']}|I={accepts['inj_dur']} "
                      f"{inj_str} {elapsed:.0f}s")
                sys.stdout.flush()

                if len(acc_history) >= 4:
                    last4 = [a for _, a in acc_history[-4:]]
                    if max(last4) - min(last4) < 0.01:
                        print(f"  PLATEAU @ step {step}")
                        break
    finally:
        pool.terminate(); pool.join()

    edges = int(np.count_nonzero(mask))
    ea = np.mean([eval_accuracy_classic(mask, H, input_projection, output_projection, theta, decay,
                  inj_dur, inj_dur_fixed, s, bp) for s in eval_seqs])
    elapsed = time.time() - t0
    quality = ea / max(edges, 1) * 100

    inj_mean = float(inj_dur.mean()) if inj_dur is not None else float(inj_dur_fixed)
    inj_std = float(inj_dur.std()) if inj_dur is not None else 0.0

    print(f"  FINAL: acc={ea*100:.2f}% edges={edges} q={quality:.3f}%/e "
          f"I_acc={accepts['inj_dur']} inj={inj_mean:.2f}+/-{inj_std:.2f} {elapsed:.0f}s")
    sys.stdout.flush()
    return {'name': name, 'acc': ea, 'edges': edges, 'quality': quality,
            'inj_mean': inj_mean, 'inj_std': inj_std,
            'inj_accepts': accepts['inj_dur'], 'time': elapsed}


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

    # A: Fix 1 tick (current baseline)
    results.append(run_config("FIX dur=1", None, 1, False,
                              bp, ALL_DATA, bigram, eval_seqs, H, input_projection, output_projection))

    # B: Fix 2 ticks
    results.append(run_config("FIX dur=2", None, 2, False,
                              bp, ALL_DATA, bigram, eval_seqs, H, input_projection, output_projection))

    # C: Fix 4 ticks
    results.append(run_config("FIX dur=4", None, 4, False,
                              bp, ALL_DATA, bigram, eval_seqs, H, input_projection, output_projection))

    # D: Fix 8 ticks (all ticks get input)
    results.append(run_config("FIX dur=8", None, 8, False,
                              bp, ALL_DATA, bigram, eval_seqs, H, input_projection, output_projection))

    # E: Learnable init=1
    results.append(run_config("LEARN init=1",
                              np.full(H, 1, dtype=np.int32), 1, True,
                              bp, ALL_DATA, bigram, eval_seqs, H, input_projection, output_projection))

    # F: Learnable init=random [1,8]
    rng_f = np.random.RandomState(55)
    results.append(run_config("LEARN init=rand",
                              rng_f.randint(1, 9, size=H).astype(np.int32), 1, True,
                              bp, ALL_DATA, bigram, eval_seqs, H, input_projection, output_projection))

    print(f"\n{'='*75}")
    print(f"  SUMMARY -- INJECTION DURATION (8t, bigram 2seq, ReLU, decay [.08,.24])")
    print(f"{'='*75}")
    print(f"  {'Name':<18} {'Acc%':>6} {'Edges':>6} {'Q(%/e)':>8} {'Inj':>8} {'I_acc':>6} {'Time':>6}")
    print(f"  {'-'*18} {'-'*6} {'-'*6} {'-'*8} {'-'*8} {'-'*6} {'-'*6}")
    for r in results:
        print(f"  {r['name']:<18} {r['acc']*100:6.2f} {r['edges']:6d} {r['quality']:8.3f} "
              f"{r['inj_mean']:5.2f}+/-{r['inj_std']:.1f} {r['inj_accepts']:6d} {r['time']:5.0f}s")
    sys.stdout.flush()

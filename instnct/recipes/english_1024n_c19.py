"""
English 1024 neurons, 18 workers — C19 ACTIVATION with learnable C
===================================================================
Same as english_1024n_18w.py but replaces ReLU(charge-theta) with c19.
- rho = 4.0 FIXED (wave shape)
- C = per-neuron learnable (period/amplitude scaling), evolved like theta/decay
Round-robin: [A, A, T, A, A, D, A, A, C] — C = C-param mutation step.
"""
import sys, os, time, random, json, math
import numpy as np
from multiprocessing import Pool

from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "model"))
from graph import SelfWiringGraph

_bp = None
_all_data = None
_seq_len = 200
_n_train = 5
_RHO = 4.0  # fixed

def init_w(b, d, sl, nt):
    global _bp, _all_data, _seq_len, _n_train
    _bp, _all_data, _seq_len, _n_train = b, d, sl, nt

def make_bp(io_dim, seed=12345):
    rng = np.random.RandomState(seed)
    p = rng.randn(256, io_dim).astype(np.float32)
    p /= np.linalg.norm(p, axis=1, keepdims=True)
    return p

def c19_activation(x, C):
    """Numpy c19: periodic parabolic wave. rho=4.0 fixed, C per-neuron."""
    inv_c = 1.0 / C
    l = 6.0 * C
    scaled = x * inv_c
    n = np.floor(scaled)
    t = scaled - n
    h = t * (1.0 - t)
    sgn = np.where(np.remainder(n, 2.0) < 1.0, 1.0, -1.0)
    core = C * (sgn * h + _RHO * h * h)
    return np.where(x >= l, x - l, np.where(x <= -l, x + l, core))

def _eval_on_seqs(mask, H, input_projection, output_projection, theta, decay, C_param, seqs):
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
                    act = act + _bp[text_bytes[i]] @ input_projection
                raw = np.zeros(H, dtype=np.float32)
                if len(rs):
                    np.add.at(raw, cs, act[rs] * sp_vals)
                charge += raw; charge *= ret
                # C19 activation: c19(charge - theta, C)
                act = c19_activation(charge - theta, C_param)
                charge = np.clip(charge, -1.0, 1.0)
            state = act.copy()
            out = charge @ output_projection
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

def worker_eval(args):
    mask_flat, theta, decay, C_param, H, input_projection, output_projection, seed, proposal_type = args
    rng = random.Random(seed)
    np_rng = np.random.RandomState(seed)
    mask = mask_flat.reshape(H, H)
    new_mask = mask
    new_theta = theta
    new_decay = decay
    new_C = C_param
    info = {'r': -1, 'c': -1, 'val': 0.0}

    if proposal_type == 'add':
        r = rng.randint(0, H-1)
        c = rng.randint(0, H-1)
        if r == c or mask[r, c] != 0:
            return {'delta': -1e9, 'type': 'add'}
        val = 0.6 if rng.random() < 0.5 else -0.6
        new_mask = mask.copy()
        new_mask[r, c] = val
        info = {'r': r, 'c': c, 'val': val}
    elif proposal_type == 'theta':
        idx = rng.randint(0, H-1)
        new_theta = theta.copy()
        new_theta[idx] = rng.random()
        info = {'idx': idx}
    elif proposal_type == 'decay':
        idx = rng.randint(0, H-1)
        new_decay = decay.copy()
        new_decay[idx] = rng.uniform(0.01, 0.5)
        info = {'idx': idx}
    elif proposal_type == 'cparam':
        idx = rng.randint(0, H-1)
        new_C = C_param.copy()
        # C controls period+amplitude: range [0.2, 5.0]
        new_C[idx] = rng.uniform(0.2, 5.0)
        info = {'idx': idx}

    data_len = len(_all_data)
    seqs = []
    for _ in range(_n_train):
        off = np_rng.randint(0, data_len - _seq_len)
        seqs.append(_all_data[off:off+_seq_len])

    old_score = _eval_on_seqs(mask, H, input_projection, output_projection, theta, decay, C_param, seqs)
    new_score = _eval_on_seqs(new_mask, H, input_projection, output_projection, new_theta, new_decay, new_C, seqs)

    return {'delta': new_score - old_score, 'type': proposal_type, 'info': info,
            'new_theta': new_theta if proposal_type == 'theta' else None,
            'new_decay': new_decay if proposal_type == 'decay' else None,
            'new_C': new_C if proposal_type == 'cparam' else None}

def eval_accuracy(mask, H, input_projection, output_projection, theta, decay, C_param, text_bytes, bp, ticks=6):
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
                act = act + bp[text_bytes[i]] @ input_projection
            raw = np.zeros(H, dtype=np.float32)
            if len(rs):
                np.add.at(raw, cs, act[rs] * sp_vals)
            charge += raw; charge *= ret
            act = c19_activation(charge - theta, C_param)
            charge = np.clip(charge, -1.0, 1.0)
        state = act.copy()
        out = charge @ output_projection
        out_n = out / (np.linalg.norm(out) + 1e-8)
        sims = out_n @ pat_norm.T
        if np.argmax(sims) == text_bytes[i+1]: correct += 1
        total += 1
    return correct/total if total else 0

if __name__ == "__main__":
    IO = 256
    NV = 4  # 1024 neurons
    N_WORKERS = 18
    BUDGET = 50000
    SEQ_LEN = 200
    N_TRAIN_SEQS = 5
    N_EVAL_SEQS = 3

    SelfWiringGraph.NV_RATIO = NV
    H = IO * NV

    bp = make_bp(IO)
    from lib.data import load_fineweb_bytes, resolve_fineweb_path
    DATA = resolve_fineweb_path()
    ALL_DATA = load_fineweb_bytes()
    DATA_LEN = len(ALL_DATA)
    print(f"Loaded {DATA_LEN / 1e6:.1f} MB text")

    eval_rng = np.random.RandomState(9999)
    eval_seqs = []
    for _ in range(N_EVAL_SEQS):
        off = eval_rng.randint(0, DATA_LEN - SEQ_LEN)
        eval_seqs.append(ALL_DATA[off:off+SEQ_LEN])

    print(f"{H} neurons, I/O={IO}, {N_WORKERS} workers, budget={BUDGET}")
    print(f"ACTIVATION: c19 (rho=4.0 fixed, C per-neuron learnable)")
    print(f"Schedule: [A,A,T,A,A,D,A,A,C]")
    sys.stdout.flush()

    random.seed(42); np.random.seed(42)
    net = SelfWiringGraph(IO)

    net.mask[:]=0; net.alive=[]; net.alive_set=set(); net._sync_sparse_idx()
    # Per-neuron C parameter, init 1.0 (unit scale)
    C_param = np.full(H, 1.0, dtype=np.float32)
    print(f"Empty network, H={net.H}, theta={net.theta.mean():.3f}, C={C_param.mean():.1f}")
    net.state *= 0; net.charge *= 0
    input_projection=net.input_projection; output_projection=net.output_projection

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    LOG = os.path.join(BASE_DIR, "english_1024n_c19_live.txt")
    JSON_LOG = os.path.join(BASE_DIR, "training_c19_live_data.json")
    CKPT_DIR = os.path.join(BASE_DIR, "checkpoints")
    os.makedirs(CKPT_DIR, exist_ok=True)

    with open(LOG, "w") as f:
        f.write(f"--- START {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")
        f.write(f"{H}n, c19(rho=4 fix, C learnable), {N_WORKERS}w, budget={BUDGET}\n")

    SCHEDULE = ['add', 'add', 'theta', 'add', 'add', 'decay', 'add', 'add', 'cparam']
    add_accepts = 0; theta_accepts = 0; decay_accepts = 0; c_accepts = 0
    accepts = 0
    log_data = []
    t0 = time.time()

    pool = Pool(N_WORKERS, initializer=init_w, initargs=(bp, ALL_DATA, SEQ_LEN, N_TRAIN_SEQS))
    try:
        for step in range(1, BUDGET+1):
            ptype = SCHEDULE[(step - 1) % len(SCHEDULE)]
            mask_flat = net.mask.flatten()
            args = [(mask_flat, net.theta.copy(), net.decay.copy(), C_param.copy(), H, input_projection, output_projection,
                     1000+step*50+w, ptype) for w in range(N_WORKERS)]
            results = pool.map(worker_eval, args)

            best_r = max(results, key=lambda x: x['delta'])
            if best_r['delta'] > 0:
                if best_r['type'] == 'add':
                    info = best_r['info']
                    net.mask[info['r'], info['c']] = info['val']
                    net.alive.append((info['r'], info['c']))
                    net.alive_set.add((info['r'], info['c']))
                    net._sync_sparse_idx()
                    add_accepts += 1
                elif best_r['type'] == 'theta' and best_r['new_theta'] is not None:
                    idx = best_r['info']['idx']
                    old_v = net.theta[idx]
                    net.theta[:] = best_r['new_theta']
                    print(f"    T accept: neuron {idx} {old_v:.3f} -> {net.theta[idx]:.3f}")
                    theta_accepts += 1
                elif best_r['type'] == 'decay' and best_r['new_decay'] is not None:
                    idx = best_r['info']['idx']
                    old_v = net.decay[idx]
                    net.decay[:] = best_r['new_decay']
                    print(f"    D accept: neuron {idx} {old_v:.3f} -> {net.decay[idx]:.3f}")
                    decay_accepts += 1
                elif best_r['type'] == 'cparam' and best_r['new_C'] is not None:
                    idx = best_r['info']['idx']
                    old_c = C_param[idx]
                    C_param[:] = best_r['new_C']
                    new_c = C_param[idx]
                    print(f"    C accept: neuron {idx} {old_c:.3f} -> {new_c:.3f} (delta={best_r['delta']:.5f})")
                    c_accepts += 1
                accepts += 1

            if step % 50 == 0:
                elapsed = time.time() - t0
                ea = np.mean([eval_accuracy(net.mask, H, input_projection, output_projection, net.theta, net.decay, C_param, s, bp)
                              for s in eval_seqs])
                edges = net.count_connections()
                th_m = float(net.theta.mean()); th_s = float(net.theta.std())
                dc_m = float(net.decay.mean()); dc_s = float(net.decay.std())
                c_m = float(C_param.mean()); c_s = float(C_param.std())
                sps = step / elapsed

                line = (f"[{step:5d}] eval={ea*100:.1f}% edges={edges} "
                        f"[A={add_accepts}|T={theta_accepts}|D={decay_accepts}|C={c_accepts}] "
                        f"theta={th_m:.3f}+/-{th_s:.3f} decay={dc_m:.3f}+/-{dc_s:.3f} "
                        f"C={c_m:.2f}+/-{c_s:.2f} "
                        f"{elapsed:.0f}s ({sps:.2f} step/s)")
                print(f"  {line}")
                with open(LOG, "a") as f:
                    f.write(line + "\n")

                log_data.append({
                    'step': step, 'eval': round(ea * 100, 1), 'edges': edges,
                    'A': add_accepts, 'T': theta_accepts, 'D': decay_accepts, 'C': c_accepts,
                    'theta_m': round(th_m, 4), 'theta_s': round(th_s, 4),
                    'decay_m': round(dc_m, 4), 'decay_s': round(dc_s, 4),
                    'C_m': round(c_m, 4), 'C_s': round(c_s, 4),
                    'sps': round(sps, 2), 'time': int(elapsed)
                })
                with open(JSON_LOG, 'w') as f:
                    json.dump(log_data, f, separators=(',', ':'))
                sys.stdout.flush()

            if step % 500 == 0:
                ckpt = os.path.join(CKPT_DIR, f"english_1024n_c19_step{step}.npz")
                net.save(ckpt)
                print(f"  SAVED: {ckpt}")
                sys.stdout.flush()

    finally:
        pool.terminate(); pool.join()
        final_ckpt = os.path.join(CKPT_DIR, "english_1024n_c19_final.npz")
        net.save(final_ckpt)
        print(f"  SAVED FINAL: {final_ckpt}")

    elapsed = time.time() - t0
    final_ea = np.mean([eval_accuracy(net.mask, H, input_projection, output_projection, net.theta, net.decay, C_param, s, bp)
                        for s in eval_seqs])
    print(f"\nFINAL: eval={final_ea*100:.1f}% edges={net.count_connections()} "
          f"accepts={accepts} {elapsed:.0f}s ({BUDGET/elapsed:.2f} step/s)")

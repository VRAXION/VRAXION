"""
Sparse forward propagation: while-true style edge loop vs current
================================================================
The "edge native" approach: iterate ONLY over connections that exist,
propagate ONLY from neurons that fired. Pure integer arithmetic.

A: current float32 rollout (channel LUT, baseline)
B: sparse forward int (connection list + fired-only propagation)
"""
import sys, os, time, random
import numpy as np
from multiprocessing import Pool
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR)); sys.path.insert(0, str(ROOT_DIR / "model"))
from graph import SelfWiringGraph

H = 256; N_WORKERS = 18; TICKS = 8; INPUT_DURATION = 2
THRESHOLD = 0.00005; EVAL_EVERY = 20
INIT_DENSITY = 0.05
PHI = (1 + 5**0.5) / 2
IN_DIM = int(round(H / PHI))
OUT_DIM = int(round(H / PHI))
SDR_K = int(round(IN_DIM * 0.20))
SCHEDULE = ['add','flip','theta','channel','theta','channel','flip','remove']

# Precompute EFF_THETA_LUT[theta][channel][tick] — integer thresholds
WAVE_LUT = SelfWiringGraph.WAVE_LUT
EFF_THETA_LUT = np.zeros((16, 9, 8), dtype=np.uint8)
for th in range(1, 16):
    for ch in range(1, 9):
        for t in range(8):
            EFF_THETA_LUT[th, ch, t] = max(1, min(15, round(th * WAVE_LUT[ch, t])))

def build_sdr(n, dim, k, seed):
    rng = np.random.RandomState(seed)
    t = np.zeros((n, dim), np.float32)
    for v in range(n): t[v, rng.choice(dim, size=k, replace=False)] = 1.0
    return t

def build_sdr_int(n, dim, k, seed):
    rng = np.random.RandomState(seed)
    t = np.zeros((n, dim), np.int8)
    for v in range(n): t[v, rng.choice(dim, size=k, replace=False)] = 1
    return t

def build_freq_order(dim, bigram, seed=12345):
    freq = bigram.sum(axis=0) + bigram.sum(axis=1)
    rank = np.argsort(freq)[::-1]
    rng = np.random.RandomState(seed)
    p = np.zeros((256, dim), np.float32)
    for i, byte_idx in enumerate(rank):
        t = i / 255.0
        for d in range(dim):
            p[byte_idx, d] = np.sin(2 * np.pi * t * (d+1) / dim * 3) + rng.randn() * 0.3
    p /= np.linalg.norm(p, axis=1, keepdims=True) + 1e-8
    return p.astype(np.float32)

BP_IN = build_sdr(256, IN_DIM, SDR_K, 42)
BP_IN_INT = build_sdr_int(256, IN_DIM, SDR_K, 42)

def build_connection_list(mask, polarity_bool):
    """Build edge-native connection list: (source, target, sign)."""
    rows, cols = np.where(mask)
    signs = np.where(polarity_bool[rows], np.int8(1), np.int8(-1))
    return rows.astype(np.uint16), cols.astype(np.uint16), signs

# === A: FLOAT ROLLOUT (current graph.py style with channel LUT) ===
def rollout_float(injected, mask, theta_f, channel, sparse_cache, polarity_f):
    H = mask.shape[0]
    act = np.zeros(H, np.float32)
    charge = np.zeros(H, np.float32)
    rows, cols = sparse_cache
    for tick in range(TICKS):
        if tick % 6 == 0:
            charge = np.maximum(charge - 1.0, 0.0)
        if tick < INPUT_DURATION:
            act = act + injected
        raw = np.zeros(H, np.float32)
        if len(rows):
            np.add.at(raw, cols, act[rows])
        charge += raw
        np.clip(charge, 0.0, 15.0, out=charge)
        theta_mult = WAVE_LUT[channel, tick % 8]
        eff = np.clip(theta_f * theta_mult, 1.0, 15.0)
        fired = charge >= eff
        act = fired.astype(np.float32) * polarity_f
        charge[fired] = 0.0
    return act, charge

# === B: SPARSE FORWARD INT (edge-native while-true style) ===
def rollout_sparse_int(sdr_neurons, mask, theta_u8, channel, conn_src, conn_tgt, conn_sign):
    """
    Pure integer, connection-list-only propagation.
    Only iterates over existing connections. Only propagates from fired neurons.
    """
    H = mask.shape[0]
    spike = np.zeros(H, np.int8)       # 0 or 1
    charge = np.zeros(H, np.int16)     # [0, 15]
    polarity_out = conn_sign            # precomputed ±1 per connection

    # Precompute: for each source neuron, which connections go out?
    # Group connections by source for fast "fired neuron → targets" lookup
    src_starts = np.zeros(H + 1, np.int32)
    for s in conn_src:
        src_starts[s + 1] += 1
    np.cumsum(src_starts, out=src_starts)

    decay_counter = 0
    tick_counter = 0

    # WHILE tick_counter < 8
    while tick_counter < 8:
        tick_counter += 1
        decay_counter += 1

        # DECAY
        if decay_counter >= 6:
            charge = np.maximum(charge - np.int16(1), np.int16(0))
            decay_counter = 0

        # INPUT (first 2 ticks)
        if tick_counter <= INPUT_DURATION:
            charge[sdr_neurons] += np.int16(1)

        # PROPAGATE: only from fired neurons, only through existing connections
        new_spike = np.zeros(H, np.int8)

        # Find which neurons fired (have spike=1)
        if tick_counter <= INPUT_DURATION:
            # First ticks: SDR neurons are "active"
            fired_mask = np.zeros(H, np.bool_)
            fired_mask[sdr_neurons] = True
            if tick_counter > 1:
                fired_mask |= (spike != 0)
            fired_idx = np.where(fired_mask)[0]
        else:
            fired_idx = np.where(spike != 0)[0]

        # For each fired neuron, propagate through its connections
        for src in fired_idx:
            start = src_starts[src]
            end = src_starts[src + 1]
            if start == end:
                continue
            targets = conn_tgt[start:end]
            signs = conn_sign[start:end]
            charge[targets] += signs.astype(np.int16)

        # CLAMP
        np.clip(charge, 0, 15, out=charge)

        # SPIKE DECISION
        eff_th = EFF_THETA_LUT[theta_u8, channel, (tick_counter - 1) % 8]
        fired_now = charge >= eff_th.astype(np.int16)
        new_spike[fired_now] = np.int8(1)

        # RESET fired
        charge[fired_now] = 0

        spike = new_spike

    return spike, charge

# === EVAL ===
_bp_out=None;_all_data=None;_bigram=None;_pol_f=None;_pol_bool=None;_mode=None

def init_w(bpo,data,bg,pf,pb,mode):
    global _bp_out,_all_data,_bigram,_pol_f,_pol_bool,_mode
    _bp_out=bpo;_all_data=data;_bigram=bg;_pol_f=pf;_pol_bool=pb;_mode=mode

def _eval_bigram(mask, theta_u8, theta_f, channel, seqs):
    rows, cols = np.where(mask)
    sparse = (rows.astype(np.int32), cols.astype(np.int32))
    c_src, c_tgt, c_sign = build_connection_list(mask, _pol_bool)
    # Sort by source for grouped access
    sort_idx = np.argsort(c_src)
    c_src = c_src[sort_idx]; c_tgt = c_tgt[sort_idx]; c_sign = c_sign[sort_idx]

    total = 0.0
    for tb in seqs:
        charge_f = np.zeros(H, np.float32)
        charge_i = np.zeros(H, np.int16)
        s = 0.0; n = 0
        for i in range(len(tb)-1):
            if _mode == 'float':
                inj = np.zeros(H, np.float32); inj[:IN_DIM] = BP_IN[tb[i]]
                _, charge_f = rollout_float(inj, mask, theta_f, channel, sparse, _pol_f)
                logits = np.dot(_bp_out, charge_f[H-OUT_DIM:])
            else:
                sdr_active = np.where(BP_IN_INT[tb[i]] > 0)[0].astype(np.uint16)
                _, charge_i = rollout_sparse_int(sdr_active, mask, theta_u8, channel,
                                                  c_src, c_tgt, c_sign)
                logits = np.dot(_bp_out, charge_i[H-OUT_DIM:].astype(np.float32))
            e = np.exp(logits - logits.max()); pred = e / e.sum()
            tgt = _bigram[tb[i]]
            cos = np.dot(pred, tgt) / (np.linalg.norm(pred) * np.linalg.norm(tgt) + 1e-8)
            s += cos; n += 1
        total += s / n if n else 0
    return total / len(seqs)

def worker_eval(args):
    mf, theta_u8, theta_f, channel, seed, pt = args
    rng = random.Random(seed); nrng = np.random.RandomState(seed)
    mask = mf.reshape(H, H); nm = mask; nt_u8 = theta_u8; nt_f = theta_f; nc = channel
    if pt == 'add':
        r = rng.randint(0, H-1); c = rng.randint(0, H-1)
        if r == c or mask[r, c]: return {'delta': -1e9, 'type': 'add'}
        nm = mask.copy(); nm[r, c] = True
    elif pt == 'remove':
        alive = list(zip(*np.where(mask)))
        if len(alive) < 1: return {'delta': -1e9, 'type': 'remove'}
        r, c = alive[rng.randint(0, len(alive)-1)]; nm = mask.copy(); nm[r, c] = False
    elif pt == 'flip':
        alive = list(zip(*np.where(mask)))
        if len(alive) < 1: return {'delta': -1e9, 'type': 'flip'}
        r, c = alive[rng.randint(0, len(alive)-1)]; nc2 = rng.randint(0, H-1)
        if nc2 == r or nc2 == c or mask[r, nc2]: return {'delta': -1e9, 'type': 'flip'}
        nm = mask.copy(); nm[r, c] = False; nm[r, nc2] = True
    elif pt == 'theta':
        idx = rng.randint(0, H-1)
        nt_u8 = theta_u8.copy(); nt_u8[idx] = np.uint8(rng.randint(1, 15))
        nt_f = nt_u8.astype(np.float32)
    elif pt == 'channel':
        idx = rng.randint(0, H-1)
        nc = channel.copy(); nc[idx] = np.uint8(rng.randint(1, 8))
    seqs = []
    for _ in range(2):
        off = nrng.randint(0, len(_all_data) - 100); seqs.append(_all_data[off:off+100])
    old = _eval_bigram(mask, theta_u8, theta_f, channel, seqs)
    new = _eval_bigram(nm, nt_u8, nt_f, nc, seqs)
    return {'delta': float(new - old), 'type': pt,
            'new_mask_flat': nm.flatten() if new > old else None,
            'new_theta_u8': nt_u8 if pt == 'theta' else None,
            'new_theta_f': nt_f if pt == 'theta' else None,
            'new_channel': nc if pt == 'channel' else None}

if __name__ == "__main__":
    sys.path.insert(0, str(ROOT_DIR))
    from lib.data import load_fineweb_bytes, resolve_fineweb_path
    resolve_fineweb_path(); ALL_DATA = load_fineweb_bytes()
    print(f"Loaded {len(ALL_DATA)/1e6:.1f} MB")

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    bigram = np.load(os.path.join(BASE_DIR, "data", "bigram_table.npy"))
    bp_out = build_freq_order(OUT_DIM, bigram)
    eval_rng = np.random.RandomState(9999)
    eval_seqs = [ALL_DATA[o:o+100] for o in [eval_rng.randint(0, len(ALL_DATA)-100) for _ in range(5)]]

    MODES = [
        ('A', 'float32 current', 'float'),
        ('B', 'sparse forward int', 'int'),
    ]

    print(f"\n{'='*60}")
    print(f"  SPARSE FORWARD INT vs FLOAT at H={H}")
    print(f"{'='*60}")
    sys.stdout.flush()

    results = []
    for mk, label, mode in MODES:
        print(f"\n>> {mk}: {label}")
        sys.stdout.flush()

        random.seed(42); np.random.seed(42)
        ref = SelfWiringGraph(max(IN_DIM, 16), hidden=H, projection_scale=1.0)
        pol_f32 = np.where(ref.polarity, 1.0, -1.0).astype(np.float32)
        pol_bool = ref.polarity.copy()
        irng = np.random.RandomState(42)
        mask = (irng.rand(H, H) < INIT_DENSITY).astype(bool); np.fill_diagonal(mask, False)
        theta_u8 = np.full(H, 1, dtype=np.uint8)
        theta_f32 = theta_u8.astype(np.float32)
        nrng = np.random.RandomState(42)
        channel = nrng.randint(1, 9, size=H).astype(np.uint8)

        init_w(bp_out, ALL_DATA, bigram, pol_f32, pol_bool, mode)
        pool = Pool(N_WORKERS, initializer=init_w,
            initargs=(bp_out, ALL_DATA, bigram, pol_f32, pol_bool, mode))

        best = 0; acc = 0; t0 = time.time()
        for step in range(1, 2001):
            pt = SCHEDULE[(step-1) % len(SCHEDULE)]
            if pt in ('flip', 'remove', 'theta', 'channel') and mask.sum() == 0: pt = 'add'
            args = [(mask.flatten(), theta_u8.copy(), theta_f32.copy(), channel.copy(),
                     1000+step*50+w, pt) for w in range(N_WORKERS)]
            res = pool.map(worker_eval, args)
            br = max(res, key=lambda x: x['delta'])
            if br['delta'] > THRESHOLD:
                if br['type'] in ('add', 'remove', 'flip') and br['new_mask_flat'] is not None:
                    mask[:] = br['new_mask_flat'].reshape(H, H); acc += 1
                elif br['type'] == 'theta' and br['new_theta_u8'] is not None:
                    theta_u8[:] = br['new_theta_u8']; theta_f32[:] = br['new_theta_f']; acc += 1
                elif br['type'] == 'channel' and br['new_channel'] is not None:
                    channel[:] = br['new_channel']; acc += 1
            if step % EVAL_EVERY == 0:
                ea_list = []
                for s in eval_seqs:
                    sp = (np.where(mask)[0].astype(np.int32), np.where(mask)[1].astype(np.int32))
                    c_src, c_tgt, c_sign = build_connection_list(mask, pol_bool)
                    sort_idx = np.argsort(c_src)
                    c_src_s = c_src[sort_idx]; c_tgt_s = c_tgt[sort_idx]; c_sign_s = c_sign[sort_idx]
                    cor = 0; tot = 0
                    ch_f = np.zeros(H, np.float32)
                    for i in range(len(s)-1):
                        if mode == 'float':
                            inj = np.zeros(H, np.float32); inj[:IN_DIM] = BP_IN[s[i]]
                            _, ch_f = rollout_float(inj, mask, theta_f32, channel, sp, pol_f32)
                            logits = np.dot(bp_out, ch_f[H-OUT_DIM:])
                        else:
                            sdr_a = np.where(BP_IN_INT[s[i]] > 0)[0].astype(np.uint16)
                            _, ch_i = rollout_sparse_int(sdr_a, mask, theta_u8, channel,
                                                          c_src_s, c_tgt_s, c_sign_s)
                            logits = np.dot(bp_out, ch_i[H-OUT_DIM:].astype(np.float32))
                        if np.argmax(logits) == s[i+1]: cor += 1
                        tot += 1
                    ea_list.append(cor / tot if tot else 0)
                ea = np.mean(ea_list)
                if ea > best: best = ea
                if step % 400 == 0:
                    elapsed = time.time() - t0
                    sps = step / elapsed
                    print(f"  [{mk}:{step:4d}] eval={ea*100:.1f}% best={best*100:.1f}% "
                          f"th={theta_u8.mean():.1f} edges={mask.sum()} {elapsed:.0f}s ({sps:.1f} sps)")
                    sys.stdout.flush()
        pool.terminate(); pool.join()
        elapsed = time.time() - t0
        sps = 2000 / elapsed
        results.append({'mode': mk, 'label': label, 'best': float(best), 'time': elapsed, 'sps': sps})
        print(f"  >> DONE {mk}: best={best*100:.1f}% {elapsed:.0f}s ({sps:.1f} sps)")
        sys.stdout.flush()

    print(f"\n{'='*60}")
    print(f"  SPARSE FORWARD RESULTS (H={H})")
    print(f"{'='*60}")
    for r in results:
        print(f"  {r['mode']}: {r['label']:35s} best={r['best']*100:5.1f}%  {r['sps']:.1f} sps")
    if len(results) == 2:
        speedup = results[1]['sps'] / results[0]['sps']
        acc_diff = results[1]['best'] - results[0]['best']
        print(f"  Speedup: {speedup:.2f}x  Accuracy diff: {acc_diff*100:+.1f}%")
    print(f"{'='*60}")

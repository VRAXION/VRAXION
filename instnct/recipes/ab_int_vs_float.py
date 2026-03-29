"""
Integer tick loop vs float32: A/B test
======================================
Can the entire tick loop run with ZERO float operations?
A: float32 (current graph.py rollout_token with channel LUT)
B: int8/int16 (pure integer, precomputed EFF_THETA_LUT)
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

# === PRECOMPUTE: EFF_THETA_LUT[theta][channel][tick] ===
# theta: 0-15 (0 unused), channel: 0-8 (0 unused), tick: 0-7
WAVE_LUT = SelfWiringGraph.WAVE_LUT  # (9, 8) float32
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
# Int version of BP_IN: 0/1 as int8
BP_IN_INT = BP_IN.astype(np.int8)

def build_sparse_idx(mask):
    rows, cols = np.where(mask)
    return rows.astype(np.int32), cols.astype(np.int32)

# === FLOAT ROLLOUT (current) ===
def rollout_float(injected_f, mask, theta_f, channel, sparse_idx, polarity_f):
    H = mask.shape[0]
    rows, cols = sparse_idx
    act = np.zeros(H, np.float32)
    charge = np.zeros(H, np.float32)
    for tick in range(TICKS):
        if tick % 6 == 0:
            charge = np.maximum(charge - 1.0, 0.0)
        if tick < INPUT_DURATION:
            act = act + injected_f
        raw = np.zeros(H, np.float32)
        if len(rows):
            np.add.at(raw, cols, act[rows])
        charge += raw
        np.clip(charge, 0.0, 15.0, out=charge)
        theta_mult = WAVE_LUT[channel, tick % 8]
        eff_theta = np.clip(theta_f * theta_mult, 1.0, 15.0)
        fired = charge >= eff_theta
        act = fired.astype(np.float32) * polarity_f
        charge[fired] = 0.0
    return act, charge

# === INTEGER ROLLOUT (zero float in loop) ===
def rollout_int(injected_i, mask, theta_u8, channel, sparse_idx, polarity_i):
    H = mask.shape[0]
    rows, cols = sparse_idx
    act = np.zeros(H, np.int8)
    charge = np.zeros(H, np.int16)
    for tick in range(TICKS):
        if tick % 6 == 0:
            charge = np.maximum(charge - np.int16(1), np.int16(0))
        if tick < INPUT_DURATION:
            act = np.clip(act.astype(np.int16) + injected_i.astype(np.int16), -127, 127).astype(np.int8)
        raw = np.zeros(H, np.int16)
        if len(rows):
            np.add.at(raw, cols, act[rows].astype(np.int16))
        charge += raw
        np.clip(charge, 0, 15, out=charge)
        eff_theta = EFF_THETA_LUT[theta_u8, channel, tick % 8]  # uint8 lookup
        fired = charge >= eff_theta.astype(np.int16)
        act = fired.astype(np.int8) * polarity_i
        charge[fired] = 0
    return act, charge

# === EVAL ===
_bp_out=None;_all_data=None;_bigram=None;_pol_f=None;_pol_i=None;_mode=None

def init_w(bpo,data,bg,pf,pi,mode):
    global _bp_out,_all_data,_bigram,_pol_f,_pol_i,_mode
    _bp_out=bpo;_all_data=data;_bigram=bg;_pol_f=pf;_pol_i=pi;_mode=mode

def _eval_bigram(mask, theta_u8, theta_f, channel, seqs):
    sp = build_sparse_idx(mask)
    total = 0.0
    for tb in seqs:
        charge_f = np.zeros(H, np.float32)
        charge_i = np.zeros(H, np.int16)
        act_f = np.zeros(H, np.float32)
        act_i = np.zeros(H, np.int8)
        s = 0.0; n = 0
        for i in range(len(tb)-1):
            if _mode == 'float':
                inj = np.zeros(H, np.float32); inj[:IN_DIM] = BP_IN[tb[i]]
                act_f, charge_f = rollout_float(inj, mask, theta_f, channel, sp, _pol_f)
                logits = np.dot(_bp_out, charge_f[H-OUT_DIM:])
            else:
                inj = np.zeros(H, np.int8); inj[:IN_DIM] = BP_IN_INT[tb[i]]
                act_i, charge_i = rollout_int(inj, mask, theta_u8, channel, sp, _pol_i)
                # Output: convert charge to float for dot product (outside tick loop)
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

    # Verify EFF_THETA_LUT
    print(f"EFF_THETA_LUT shape: {EFF_THETA_LUT.shape}")
    print(f"  theta=6, ch=1: {EFF_THETA_LUT[6, 1, :]}")
    print(f"  theta=6, ch=5: {EFF_THETA_LUT[6, 5, :]}")
    print(f"  Float equiv:    {[f'{6*WAVE_LUT[1,t]:.1f}' for t in range(8)]}")

    # Quick correctness check: do both paths fire same neurons?
    print("\nCorrectness check:")
    random.seed(42); np.random.seed(42)
    ref = SelfWiringGraph(max(IN_DIM, 16), hidden=H, projection_scale=1.0)
    pol_f32 = np.where(ref.polarity, 1.0, -1.0).astype(np.float32)
    pol_i8 = np.where(ref.polarity, np.int8(1), np.int8(-1))
    irng = np.random.RandomState(42)
    mask = (irng.rand(H, H) < INIT_DENSITY).astype(bool); np.fill_diagonal(mask, False)
    theta_u8 = np.full(H, 6, dtype=np.uint8)
    theta_f32 = theta_u8.astype(np.float32)
    nrng = np.random.RandomState(42)
    channel = nrng.randint(1, 9, size=H).astype(np.uint8)

    inj_f = np.zeros(H, np.float32); inj_f[:IN_DIM] = BP_IN[65]
    inj_i = np.zeros(H, np.int8); inj_i[:IN_DIM] = BP_IN_INT[65]
    sp = build_sparse_idx(mask)

    act_f, ch_f = rollout_float(inj_f, mask, theta_f32, channel, sp, pol_f32)
    act_i, ch_i = rollout_int(inj_i, mask, theta_u8, channel, sp, pol_i8)

    fired_f = (act_f != 0).sum()
    fired_i = (act_i != 0).sum()
    charge_match = (np.round(ch_f).astype(int) == ch_i.astype(int)).mean() * 100
    print(f"  Float fired: {fired_f}, Int fired: {fired_i}")
    print(f"  Charge match: {charge_match:.1f}%")

    MODES = [
        ('A', 'float32 (current)', 'float'),
        ('B', 'int8/int16 (zero float)', 'int'),
    ]

    print(f"\n{'='*60}")
    print(f"  INTEGER vs FLOAT TICK LOOP at H={H}")
    print(f"{'='*60}")
    sys.stdout.flush()

    results = []
    for mk, label, mode in MODES:
        print(f"\n>> {mk}: {label}")
        sys.stdout.flush()

        random.seed(42); np.random.seed(42)
        ref = SelfWiringGraph(max(IN_DIM, 16), hidden=H, projection_scale=1.0)
        pol_f32 = np.where(ref.polarity, 1.0, -1.0).astype(np.float32)
        pol_i8 = np.where(ref.polarity, np.int8(1), np.int8(-1))
        irng = np.random.RandomState(42)
        mask = (irng.rand(H, H) < INIT_DENSITY).astype(bool); np.fill_diagonal(mask, False)
        theta_u8 = np.full(H, 1, dtype=np.uint8)
        theta_f32 = theta_u8.astype(np.float32)
        nrng = np.random.RandomState(42)
        channel = nrng.randint(1, 9, size=H).astype(np.uint8)

        init_w(bp_out, ALL_DATA, bigram, pol_f32, pol_i8, mode)
        pool = Pool(N_WORKERS, initializer=init_w,
            initargs=(bp_out, ALL_DATA, bigram, pol_f32, pol_i8, mode))

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
                    theta_u8[:] = br['new_theta_u8']
                    theta_f32[:] = br['new_theta_f']
                    acc += 1
                elif br['type'] == 'channel' and br['new_channel'] is not None:
                    channel[:] = br['new_channel']; acc += 1
            if step % EVAL_EVERY == 0:
                ea_list = []
                for s in eval_seqs:
                    sp = build_sparse_idx(mask)
                    cor = 0; tot = 0
                    ch_f = np.zeros(H, np.float32)
                    ch_i = np.zeros(H, np.int16)
                    act_f = np.zeros(H, np.float32)
                    act_i = np.zeros(H, np.int8)
                    for i in range(len(s)-1):
                        if mode == 'float':
                            inj = np.zeros(H, np.float32); inj[:IN_DIM] = BP_IN[s[i]]
                            act_f, ch_f = rollout_float(inj, mask, theta_f32, channel, sp, pol_f32)
                            logits = np.dot(bp_out, ch_f[H-OUT_DIM:])
                        else:
                            inj = np.zeros(H, np.int8); inj[:IN_DIM] = BP_IN_INT[s[i]]
                            act_i, ch_i = rollout_int(inj, mask, theta_u8, channel, sp, pol_i8)
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
                          f"th={theta_u8.mean():.1f} acc={acc} {elapsed:.0f}s ({sps:.1f} sps)")
                    sys.stdout.flush()
        pool.terminate(); pool.join()
        elapsed = time.time() - t0
        sps = 2000 / elapsed
        results.append({'mode': mk, 'label': label, 'best': float(best),
                        'time': elapsed, 'sps': sps})
        print(f"  >> DONE {mk}: best={best*100:.1f}% {elapsed:.0f}s ({sps:.1f} sps)")
        sys.stdout.flush()

    print(f"\n{'='*60}")
    print(f"  INT vs FLOAT RESULTS (H={H})")
    print(f"{'='*60}")
    for r in results:
        print(f"  {r['mode']}: {r['label']:35s} best={r['best']*100:5.1f}%  {r['sps']:.1f} sps")
    if len(results) == 2:
        speedup = results[1]['sps'] / results[0]['sps']
        acc_diff = results[1]['best'] - results[0]['best']
        print(f"  Speedup: {speedup:.2f}x  Accuracy diff: {acc_diff*100:+.1f}%")
    print(f"{'='*60}")

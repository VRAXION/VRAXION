"""
B/C test: two damping approaches (control group from previous runs ~10-14%)

B: Damping score — 1 number per neuron (0-8), blocks weakest N ticks
   channel picks WHICH ticks, damping picks HOW MANY blocked
   Spike: tick_value > damping_cutoff && charge >= theta+1

C: Per-tick negative multiplier — 1 number per tick per neuron (0 to -8)
   Each tick has its own damping. 0=normal, -1..-8=blocked (charge×negative < threshold)
   Spike: charge × tick_mult[neuron][tick] >= theta+1
   tick_mult range: 0 (off) to -8 (strong block) to +1 (normal, init)

Run: python instnct/recipes/ab_damping_vs_pertick.py
"""
import sys, os, time, json, random
from collections import defaultdict
import numpy as np
from multiprocessing import Pool
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "model"))
from graph import SelfWiringGraph
from quaternary_mask import QuaternaryMask

H = 256; N_WORKERS = 12; TICKS = 12; INPUT_DURATION = 2
THRESHOLD = 0.00005; EVAL_EVERY = 50
INIT_DENSITY = 0.05; BUILD_STEPS = 1800
PHI = (1 + 5**0.5) / 2
IN_DIM = int(round(H / PHI))
OUT_DIM = int(round(H / PHI))
SDR_K = int(round(IN_DIM * 0.20))
SEQ_LEN = 100; N_TRAIN_SEQS = 2; N_EVAL_SEQS = 5
TICK_PERIOD = 8

# Cosine LUT values sorted for damping cutoff logic
COSINE_BASE = [7, 8, 10, 12, 13, 12, 10, 8]  # x10 scale, channel 1

def build_cosine_lut_x10():
    """Build 9x8 LUT with integer x10 cosine values."""
    lut = np.full((9, TICK_PERIOD), 10, dtype=np.int32)
    for ch in range(8):
        for t in range(TICK_PERIOD):
            lut[ch + 1, t] = COSINE_BASE[(t - ch) % 8]
    return lut

COSINE_LUT = build_cosine_lut_x10()

BP_IN = None
_bp_out = None; _all_data = None; _bigram = None; _pol = None
_mode = 'B'

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

def init_w(bpi, bpo, data, bg, pol, mode):
    global BP_IN, _bp_out, _all_data, _bigram, _pol, _mode
    BP_IN = bpi; _bp_out = bpo; _all_data = data; _bigram = bg; _pol = pol
    _mode = mode


def _eval_bigram_B(qdata, theta, channel, damping, pol_f, seqs):
    """Mode B: damping score blocks weakest N ticks per neuron."""
    qm = QuaternaryMask(H, qdata)
    sc = qm.to_directed_edges()
    src_list, tgt_list = sc
    MAX_CHARGE = 15.0
    total = 0.0
    # Precompute sorted cutoff values per damping level
    # damping=0: nothing blocked (cutoff=-1, everything passes)
    # damping=N: the N weakest tick values are blocked
    # Sorted unique values: 7, 8, 10, 12, 13
    # damping=1: block ticks with value<=7, damping=2: block <=8, etc.
    # We use the Nth smallest value as cutoff
    for tb in seqs:
        state = np.zeros(H, np.float32); charge = np.zeros(H, np.float32)
        s = 0.0; n = 0
        for i in range(len(tb) - 1):
            inj = np.zeros(H, np.float32); inj[0:IN_DIM] = BP_IN[tb[i]]
            for tick in range(TICKS):
                if tick % 6 == 0:
                    charge[:] = np.maximum(charge - 1.0, 0.0)
                if tick < INPUT_DURATION:
                    state[:] += inj
                incoming = np.zeros(H, np.float32)
                if len(src_list) > 0:
                    np.add.at(incoming, tgt_list, state[src_list])
                charge[:] = np.clip(charge + incoming, 0.0, MAX_CHARGE)
                # Get tick value for each neuron's channel
                tip = tick % TICK_PERIOD
                tick_val = COSINE_LUT[channel, tip]  # (H,) array
                # Damping cutoff: block if tick_val <= cutoff
                # cutoff = damping value (0-13 scale, maps to x10 LUT values)
                # damping=0: cutoff=0, nothing blocked (all LUT vals >= 7 > 0)
                # damping=7: cutoff=7, blocks ticks with value 7
                # damping=8: cutoff=8, blocks ticks with value 7 or 8
                # damping=10: cutoff=10, blocks 7,8,10
                # damping=13: cutoff=13, blocks everything
                active = tick_val > damping  # (H,) bool
                fired = active & (charge >= (theta + 1.0))
                state[:] = 0.0
                state[fired] = pol_f[fired]
                charge[fired] = 0.0
            logits = np.dot(_bp_out, charge[H - OUT_DIM:])
            e = np.exp(logits - logits.max()); pred = e / e.sum()
            tgt = _bigram[tb[i]]
            cos = np.dot(pred, tgt) / (np.linalg.norm(pred) * np.linalg.norm(tgt) + 1e-8)
            s += cos; n += 1
        total += s / n if n else 0
    return total / len(seqs)


def _eval_bigram_C(qdata, theta, channel, tick_damp, pol_f, seqs):
    """Mode C: per-tick damping multiplier per neuron. 0=off, 1=normal (init), -N=blocked."""
    qm = QuaternaryMask(H, qdata)
    sc = qm.to_directed_edges()
    src_list, tgt_list = sc
    MAX_CHARGE = 15.0
    total = 0.0
    for tb in seqs:
        state = np.zeros(H, np.float32); charge = np.zeros(H, np.float32)
        s = 0.0; n = 0
        for i in range(len(tb) - 1):
            inj = np.zeros(H, np.float32); inj[0:IN_DIM] = BP_IN[tb[i]]
            for tick in range(TICKS):
                if tick % 6 == 0:
                    charge[:] = np.maximum(charge - 1.0, 0.0)
                if tick < INPUT_DURATION:
                    state[:] += inj
                incoming = np.zeros(H, np.float32)
                if len(src_list) > 0:
                    np.add.at(incoming, tgt_list, state[src_list])
                charge[:] = np.clip(charge + incoming, 0.0, MAX_CHARGE)
                # Per-tick multiplier: charge × mult >= theta+1
                mult = tick_damp[:, tick % TICK_PERIOD].astype(np.float32)
                weighted = charge * mult
                bias = theta + 1.0
                fired = weighted >= bias
                state[:] = 0.0
                state[fired] = pol_f[fired]
                charge[fired] = 0.0
            logits = np.dot(_bp_out, charge[H - OUT_DIM:])
            e = np.exp(logits - logits.max()); pred = e / e.sum()
            tgt = _bigram[tb[i]]
            cos = np.dot(pred, tgt) / (np.linalg.norm(pred) * np.linalg.norm(tgt) + 1e-8)
            s += cos; n += 1
        total += s / n if n else 0
    return total / len(seqs)


def worker_eval(args):
    qdata_bytes, theta, channel, extra_bytes, extra_shape, pol_f, seed, pt = args
    rng = random.Random(seed); nrng = np.random.RandomState(seed)
    qm = QuaternaryMask(H, np.frombuffer(qdata_bytes, dtype=np.uint8).copy())
    nq = qm.copy(); nt = theta.copy(); nc = channel.copy(); npf = pol_f.copy()

    if _mode == 'B':
        ne = np.frombuffer(extra_bytes, dtype=np.int32).copy()  # damping per neuron
    else:
        ne = np.frombuffer(extra_bytes, dtype=np.int8).copy().reshape(extra_shape)  # tick_damp

    if pt == 'add':
        undo = []; nq.mutate_add(rng, undo)
        if not undo: return {'delta': -1e9, 'type': pt}
    elif pt == 'enhance':
        bm = qm.to_bool_mask()
        in_deg = bm.sum(axis=0).astype(np.float64) + 1.0
        top = np.argsort(in_deg)[::-1][:H // 4]
        c = int(top[rng.randint(0, len(top) - 1)]); r = rng.randint(0, H - 1)
        if r == c or nq.get_pair(r, c) != 0: return {'delta': -1e9, 'type': pt}
        nq.set_pair(r, c, 1)
    elif pt == 'reverse':
        undo = []; nq.mutate_flip(rng, undo)
        if not undo: return {'delta': -1e9, 'type': pt}
    elif pt == 'mirror':
        undo = []; nq.mutate_upgrade(rng, undo)
        if not undo: return {'delta': -1e9, 'type': pt}
        if undo:
            idx = undo[-1][1]
            ii, jj = nq._triu_i, nq._triu_j
            i_n, j_n = int(ii[idx]), int(jj[idx])
            if (pol_f[i_n] > 0) == (pol_f[j_n] > 0):
                npf = pol_f.copy(); npf[j_n] *= -1
    elif pt in ('loop3', 'loop5', 'loop8'):
        loop_len = int(pt[4:])
        nodes = [rng.randint(0, H - 1)]
        for _ in range(loop_len - 1):
            n = rng.randint(0, H - 1)
            if n in nodes: return {'delta': -1e9, 'type': pt}
            nodes.append(n)
        for k in range(loop_len):
            r, c = nodes[k], nodes[(k + 1) % loop_len]
            if nq.get_pair(r, c) != 0: return {'delta': -1e9, 'type': pt}
        for k in range(loop_len):
            r, c = nodes[k], nodes[(k + 1) % loop_len]
            nq.set_pair(r, c, 1)
    elif pt == 'remove':
        undo = []; nq.mutate_remove(rng, undo)
        if not undo: return {'delta': -1e9, 'type': pt}
    elif pt == 'flip':
        idx = rng.randint(0, H - 1)
        npf = pol_f.copy(); npf[idx] *= -1
    elif pt == 'theta':
        idx = rng.randint(0, H - 1); nt = theta.copy()
        nt[idx] = float(rng.randint(1, 15))
    elif pt == 'channel':
        idx = rng.randint(0, H - 1); nc = channel.copy()
        nc[idx] = np.uint8(rng.randint(1, 8))
    elif pt == 'damping':
        idx = rng.randint(0, H - 1)
        if _mode == 'B':
            # Damping score: 0 (no blocking) to 13 (block all)
            # Useful range: 0, 7, 8, 10, 12, 13 (matching LUT values)
            ne[idx] = rng.choice([0, 7, 8, 10, 12, 13])
        else:
            # Per-tick: pick neuron + tick, set multiplier
            tick_idx = rng.randint(0, TICK_PERIOD - 1)
            # Range: -8 to 1. -8..-1 = blocked, 0 = off, 1 = normal
            ne[idx, tick_idx] = rng.randint(-8, 1)

    seqs = []
    for _ in range(N_TRAIN_SEQS):
        off = nrng.randint(0, len(_all_data) - SEQ_LEN)
        seqs.append(_all_data[off:off + SEQ_LEN])

    if _mode == 'B':
        old_extra = np.frombuffer(extra_bytes, dtype=np.int32)
        old = _eval_bigram_B(qm.data, theta, channel, old_extra, pol_f, seqs)
        new = _eval_bigram_B(nq.data, nt, nc, ne, npf, seqs)
    else:
        old_extra = np.frombuffer(extra_bytes, dtype=np.int8).reshape(extra_shape)
        old = _eval_bigram_C(qm.data, theta, channel, old_extra, pol_f, seqs)
        new = _eval_bigram_C(nq.data, nt, nc, ne, npf, seqs)

    return {
        'delta': float(new - old), 'type': pt,
        'new_qdata': nq.data.tobytes() if new > old else None,
        'new_theta': nt if new > old else None,
        'new_channel': nc if new > old else None,
        'new_extra': ne.tobytes() if new > old else None,
        'new_pol': npf if new > old else None,
    }


def eval_accuracy(qdata, theta, channel, extra, pol_f, text_bytes, bp_in, bp_out):
    qm = QuaternaryMask(H, qdata)
    sc = qm.to_directed_edges()
    src_list, tgt_list = sc
    MAX_CHARGE = 15.0
    state = np.zeros(H, np.float32); charge = np.zeros(H, np.float32)
    cor = 0; tot = 0
    for i in range(len(text_bytes) - 1):
        inj = np.zeros(H, np.float32); inj[0:IN_DIM] = bp_in[text_bytes[i]]
        for tick in range(TICKS):
            if tick % 6 == 0:
                charge[:] = np.maximum(charge - 1.0, 0.0)
            if tick < INPUT_DURATION:
                state[:] += inj
            incoming = np.zeros(H, np.float32)
            if len(src_list) > 0:
                np.add.at(incoming, tgt_list, state[src_list])
            charge[:] = np.clip(charge + incoming, 0.0, MAX_CHARGE)
            if _mode == 'B':
                tip = tick % TICK_PERIOD
                tick_val = COSINE_LUT[channel, tip]
                active = tick_val > extra
                fired = active & (charge >= (theta + 1.0))
            else:
                mult = extra[:, tick % TICK_PERIOD].astype(np.float32)
                fired = (charge * mult) >= (theta + 1.0)
            state[:] = 0.0
            state[fired] = pol_f[fired]
            charge[fired] = 0.0
        logits = np.dot(bp_out, charge[H - OUT_DIM:])
        if np.argmax(logits) == text_bytes[i + 1]: cor += 1
        tot += 1
    return cor / tot if tot else 0


def run_config(label, mode, bp_in, bp_out, all_data, bigram, eval_seqs):
    global _mode
    _mode = mode
    init_rng = np.random.RandomState(42)
    ref = SelfWiringGraph(max(IN_DIM, 16), hidden=H, projection_scale=1.0)
    init_mask = (init_rng.rand(H, H) < INIT_DENSITY).astype(bool)
    np.fill_diagonal(init_mask, False)
    qm = QuaternaryMask.from_bool_mask(init_mask)
    qdata = qm.data.copy()
    theta = np.full(H, 1.0, np.float32)
    channel = init_rng.randint(1, 9, size=H).astype(np.uint8)
    pol_f = np.where(ref.polarity, 1.0, -1.0).astype(np.float32)

    if mode == 'B':
        extra = np.zeros(H, dtype=np.int32)  # damping=0 = no blocking
        extra_shape = (H,)
        schedule = [
            'add', 'enhance', 'reverse', 'mirror',
            'loop3', 'loop5', 'loop8',
            'flip', 'theta', 'damping', 'channel', 'remove',
        ]
    else:
        extra = np.ones((H, TICK_PERIOD), dtype=np.int8)  # all +1 = normal
        extra_shape = (H, TICK_PERIOD)
        schedule = [
            'add', 'enhance', 'reverse', 'mirror',
            'loop3', 'loop5', 'loop8',
            'flip', 'theta', 'damping', 'damping', 'remove',
        ]

    print(f"\n{'='*60}")
    print(f"  {label} (mode={mode})")
    print(f"  Schedule: {schedule}")
    print(f"{'='*60}")

    pool = Pool(N_WORKERS, initializer=init_w,
                initargs=(bp_in, bp_out, all_data, bigram, pol_f, mode))
    acc_by_type = defaultdict(int)
    t0 = time.time()
    best_eval = 0

    try:
        for step in range(1, BUILD_STEPS + 1):
            pt = schedule[(step - 1) % len(schedule)]
            edges = QuaternaryMask(H, qdata).count_edges()
            if pt in ('remove', 'reverse') and edges < 50: pt = 'add'

            extra_bytes = extra.tobytes()
            args = [(qdata.tobytes(), theta.copy(), channel.copy(),
                     extra_bytes, extra_shape, pol_f.copy(),
                     1000 + step * 50 + w, pt) for w in range(N_WORKERS)]
            results = pool.map(worker_eval, args)
            best_r = max(results, key=lambda x: x['delta'])

            if best_r['delta'] > THRESHOLD:
                if best_r['new_qdata'] is not None:
                    qdata[:] = np.frombuffer(best_r['new_qdata'], dtype=np.uint8)
                if best_r['new_theta'] is not None: theta[:] = best_r['new_theta']
                if best_r['new_channel'] is not None: channel[:] = best_r['new_channel']
                if best_r['new_extra'] is not None:
                    if mode == 'B':
                        extra[:] = np.frombuffer(best_r['new_extra'], dtype=np.int32)
                    else:
                        extra[:] = np.frombuffer(best_r['new_extra'], dtype=np.int8).reshape(extra_shape)
                if best_r['new_pol'] is not None: pol_f[:] = best_r['new_pol']
                acc_by_type[best_r['type']] += 1

            if step % EVAL_EVERY == 0:
                ea = np.mean([eval_accuracy(qdata, theta, channel, extra, pol_f, s, bp_in, bp_out)
                              for s in eval_seqs])
                if ea > best_eval: best_eval = ea
                elapsed = time.time() - t0
                if mode == 'B':
                    d_vals = np.unique(extra, return_counts=True)
                    d_str = ' '.join(f'{v}:{c}' for v, c in zip(d_vals[0], d_vals[1]))
                    print(f"  [{step:5d}] eval={ea*100:.1f}% best={best_eval*100:.1f}% "
                          f"edges={edges} damping=[{d_str}] {elapsed:.0f}s")
                else:
                    n_neg = (extra < 0).sum()
                    n_zero = (extra == 0).sum()
                    n_pos = (extra > 0).sum()
                    changed = (extra != 1).any(axis=1).sum()
                    print(f"  [{step:5d}] eval={ea*100:.1f}% best={best_eval*100:.1f}% "
                          f"edges={edges} changed={changed}/{H} neg={n_neg} zero={n_zero} {elapsed:.0f}s")
                sys.stdout.flush()
    finally:
        pool.terminate(); pool.join()

    elapsed = time.time() - t0
    final_eval = np.mean([eval_accuracy(qdata, theta, channel, extra, pol_f, s, bp_in, bp_out)
                          for s in eval_seqs])

    print(f"\n  DONE: {label} -> final={final_eval*100:.1f}% best={best_eval*100:.1f}% {elapsed:.0f}s")
    print(f"  Accepts: {dict(acc_by_type)}")
    return {'label': label, 'mode': mode,
            'final': round(final_eval*100, 2), 'best': round(best_eval*100, 2),
            'accepts': {k: v for k, v in acc_by_type.items()}}


if __name__ == "__main__":
    from lib.data import load_fineweb_bytes, resolve_fineweb_path
    resolve_fineweb_path()
    ALL_DATA = load_fineweb_bytes()
    print(f"Loaded {len(ALL_DATA)/1e6:.1f} MB")

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    bigram = np.load(os.path.join(BASE_DIR, "data", "bigram_table.npy"))
    bp_in = build_sdr(256, IN_DIM, SDR_K, 42)
    BP_IN = bp_in
    bp_out = build_freq_order(OUT_DIM, bigram)

    eval_rng = np.random.RandomState(9999)
    eval_seqs = [ALL_DATA[off:off + SEQ_LEN]
                 for off in [eval_rng.randint(0, len(ALL_DATA) - SEQ_LEN)
                             for _ in range(N_EVAL_SEQS)]]

    results = []
    # B already completed: best=16.8%, final=12.5%
    results.append({
        'label': 'B: Damping score (1 val/neuron, blocks weakest ticks)',
        'best': 16.8, 'final': 12.5,
        'accepts': {'theta': 134, 'damping': 124, 'mirror': 118, 'loop3': 120,
                    'loop5': 115, 'flip': 120, 'loop8': 98, 'reverse': 125,
                    'enhance': 95, 'remove': 107, 'channel': 50, 'add': 104},
    })

    r = run_config("C: Per-tick damping (-8..+1 per neuron per tick)", 'C',
                   bp_in, bp_out, ALL_DATA, bigram, eval_seqs)
    results.append(r)

    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    for r in results:
        print(f"  {r['label'][:55]:55s}  best={r['best']:5.1f}%  final={r['final']:5.1f}%")

    out_path = os.path.join(BASE_DIR, "data", "damping_vs_pertick_result.json")
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")

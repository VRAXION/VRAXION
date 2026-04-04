"""
D: Per-tick threshold multiplier — each neuron has 8 uint8 values (one per tick),
applied on the THRESHOLD side for a proper gradient.

Spike: charge * 10 >= (theta + 1) * tick_mult[neuron, tick % 8]
  tick_mult range: 7 (easy, threshold x0.7) to 13 (hard, threshold x1.3)

Init: cosine pattern seeded from random channel per neuron (same starting point
as standard phase gating). Channel is then discarded — tick_mult replaces it.

Control: B (damping score) best=16.8% from ab_damping_vs_pertick.py

Run: python instnct/recipes/ab_pertick_multiplier.py
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

COSINE_BASE = [7, 8, 10, 12, 13, 12, 10, 8]  # x10 scale cosine values
TICK_MULT_MIN = 7
TICK_MULT_MAX = 13

SCHEDULE = [
    'add', 'enhance', 'reverse', 'mirror',
    'loop3', 'loop5', 'loop8',
    'flip', 'theta', 'tick_mut', 'tick_mut', 'remove',
]

BP_IN = None
_bp_out = None; _all_data = None; _bigram = None; _pol = None

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

def init_w(bpi, bpo, data, bg, pol):
    global BP_IN, _bp_out, _all_data, _bigram, _pol
    BP_IN = bpi; _bp_out = bpo; _all_data = data; _bigram = bg; _pol = pol


def _eval_bigram_D(qdata, theta, tick_mult, pol_f, seqs):
    """Mode D: per-tick threshold multiplier. tick_mult: (H, 8) uint8 [7..13]."""
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
                # D: charge * 10 >= (theta+1) * tick_mult
                tm = tick_mult[:, tick % TICK_PERIOD].astype(np.float32)
                lhs = charge * 10.0
                rhs = (theta + 1.0) * tm
                fired = lhs >= rhs
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
    qdata_bytes, theta, tm_bytes, tm_shape, pol_f, seed, pt = args
    rng = random.Random(seed); nrng = np.random.RandomState(seed)
    qm = QuaternaryMask(H, np.frombuffer(qdata_bytes, dtype=np.uint8).copy())
    nq = qm.copy(); nt = theta.copy()
    ntm = np.frombuffer(tm_bytes, dtype=np.uint8).copy().reshape(tm_shape)
    npf = pol_f.copy()

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
    elif pt == 'tick_mut':
        idx = rng.randint(0, H - 1)
        tick_idx = rng.randint(0, TICK_PERIOD - 1)
        ntm[idx, tick_idx] = rng.randint(TICK_MULT_MIN, TICK_MULT_MAX)

    seqs = []
    for _ in range(N_TRAIN_SEQS):
        off = nrng.randint(0, len(_all_data) - SEQ_LEN)
        seqs.append(_all_data[off:off + SEQ_LEN])

    old = _eval_bigram_D(qm.data, theta,
                         np.frombuffer(tm_bytes, dtype=np.uint8).reshape(tm_shape),
                         pol_f, seqs)
    new = _eval_bigram_D(nq.data, nt, ntm, npf, seqs)
    return {
        'delta': float(new - old), 'type': pt,
        'new_qdata': nq.data.tobytes() if new > old else None,
        'new_theta': nt if new > old else None,
        'new_tm': ntm.tobytes() if new > old else None,
        'new_pol': npf if new > old else None,
    }


def eval_accuracy(qdata, theta, tick_mult, pol_f, text_bytes, bp_in, bp_out):
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
            tm = tick_mult[:, tick % TICK_PERIOD].astype(np.float32)
            lhs = charge * 10.0
            rhs = (theta + 1.0) * tm
            fired = lhs >= rhs
            state[:] = 0.0
            state[fired] = pol_f[fired]
            charge[fired] = 0.0
        logits = np.dot(bp_out, charge[H - OUT_DIM:])
        if np.argmax(logits) == text_bytes[i + 1]: cor += 1
        tot += 1
    return cor / tot if tot else 0


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

    init_rng = np.random.RandomState(42)
    ref = SelfWiringGraph(max(IN_DIM, 16), hidden=H, projection_scale=1.0)
    init_mask = (init_rng.rand(H, H) < INIT_DENSITY).astype(bool)
    np.fill_diagonal(init_mask, False)
    qm = QuaternaryMask.from_bool_mask(init_mask)
    qdata = qm.data.copy()
    theta = np.full(H, 1.0, np.float32)
    pol_f = np.where(ref.polarity, 1.0, -1.0).astype(np.float32)

    # Init tick_mult: all 10 (neutral, no gating — same flat start as B)
    tick_mult = np.full((H, TICK_PERIOD), 10, dtype=np.uint8)
    init_tick_mult = tick_mult.copy()  # keep reference for change tracking

    print(f"\n{'='*60}")
    print(f"  D: Per-tick threshold multiplier (uint8, {TICK_MULT_MIN}-{TICK_MULT_MAX})")
    print(f"  Spike: charge*10 >= (theta+1) * tick_mult[n, tick%8]")
    print(f"  Init: all 10 (neutral, no gating)")
    print(f"  Schedule: {SCHEDULE}")
    print(f"{'='*60}")

    pool = Pool(N_WORKERS, initializer=init_w,
                initargs=(bp_in, bp_out, ALL_DATA, bigram, pol_f))
    acc_by_type = defaultdict(int)
    t0 = time.time()
    best_eval = 0

    try:
        for step in range(1, BUILD_STEPS + 1):
            pt = SCHEDULE[(step - 1) % len(SCHEDULE)]
            edges = QuaternaryMask(H, qdata).count_edges()
            if pt in ('remove', 'reverse') and edges < 50: pt = 'add'

            tm_bytes = tick_mult.tobytes()
            args = [(qdata.tobytes(), theta.copy(), tm_bytes, (H, TICK_PERIOD),
                     pol_f.copy(), 1000 + step * 50 + w, pt) for w in range(N_WORKERS)]
            results = pool.map(worker_eval, args)
            best_r = max(results, key=lambda x: x['delta'])

            if best_r['delta'] > THRESHOLD:
                if best_r['new_qdata'] is not None:
                    qdata[:] = np.frombuffer(best_r['new_qdata'], dtype=np.uint8)
                if best_r['new_theta'] is not None: theta[:] = best_r['new_theta']
                if best_r['new_tm'] is not None:
                    tick_mult[:] = np.frombuffer(best_r['new_tm'], dtype=np.uint8).reshape(H, TICK_PERIOD)
                if best_r['new_pol'] is not None: pol_f[:] = best_r['new_pol']
                acc_by_type[best_r['type']] += 1

            if step % EVAL_EVERY == 0:
                ea = np.mean([eval_accuracy(qdata, theta, tick_mult, pol_f, s, bp_in, bp_out)
                              for s in eval_seqs])
                if ea > best_eval: best_eval = ea
                elapsed = time.time() - t0
                n_changed = (tick_mult != init_tick_mult).any(axis=1).sum()
                cells_changed = int((tick_mult != init_tick_mult).sum())
                val_counts = {v: int((tick_mult == v).sum()) for v in range(7, 14)}
                val_str = ' '.join(f'{v}:{c}' for v, c in val_counts.items() if c > 0)
                print(f"  [{step:5d}] eval={ea*100:.1f}% best={best_eval*100:.1f}% "
                      f"edges={edges} changed={n_changed}/{H} cells={cells_changed}/{H*TICK_PERIOD} "
                      f"vals=[{val_str}] {elapsed:.0f}s")
                sys.stdout.flush()
    finally:
        pool.terminate(); pool.join()

    elapsed = time.time() - t0
    final_eval = np.mean([eval_accuracy(qdata, theta, tick_mult, pol_f, s, bp_in, bp_out)
                          for s in eval_seqs])

    print(f"\n{'='*60}")
    print(f"  RESULT: D (per-tick threshold multiplier)")
    print(f"{'='*60}")
    print(f"  Final: {final_eval*100:.1f}%  Best: {best_eval*100:.1f}%")
    print(f"  Accepts: {dict(acc_by_type)}")
    print(f"  Elapsed: {elapsed:.0f}s")

    # Value distribution
    print(f"\n  Tick multiplier distribution:")
    for v in range(TICK_MULT_MIN, TICK_MULT_MAX + 1):
        count = int((tick_mult == v).sum())
        pct = count / tick_mult.size * 100
        bar = '#' * int(pct)
        print(f"    tm={v:2d}: {count:>5d} ({pct:4.1f}%) {bar}")

    # Example neuron patterns
    print(f"\n  Example neuron tick patterns (init -> final):")
    for i in [0, 1, 50, 100, 200]:
        changed = '*' if (tick_mult[i] != init_tick_mult[i]).any() else ' '
        print(f"   {changed}neuron {i:>3d}: {list(init_tick_mult[i])} -> {list(tick_mult[i])}")

    # Summary comparison with B
    print(f"\n  Comparison:")
    print(f"    B (damping score):           best=16.8%  final=12.5%")
    print(f"    D (per-tick threshold mult): best={best_eval*100:.1f}%  final={final_eval*100:.1f}%")

    out_path = os.path.join(BASE_DIR, "data", "pertick_multiplier_result.json")
    with open(out_path, 'w') as f:
        json.dump({
            'label': 'D: Per-tick threshold multiplier',
            'final': round(final_eval * 100, 2),
            'best': round(best_eval * 100, 2),
            'accepts': {k: v for k, v in acc_by_type.items()},
            'val_distribution': {str(v): int((tick_mult == v).sum()) for v in range(7, 14)},
        }, f, indent=2)
    print(f"\nSaved to {out_path}")

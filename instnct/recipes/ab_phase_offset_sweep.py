"""
Phase gating offset sweep: how should the LUT modulate thresholds?

Configs:
  E1: Additive symmetric   offset: -3, -2, 0, +2, +3
  E2: Additive positive    offset:  0, +1, +2, +4, +5 (only harder)
  E3: Additive negative    offset: -5, -4, -2, -1,  0 (only easier)
  E4: Multiplicative x10   values:  7,  8, 10, 12, 13 (current cosine)
  E5: Exponential offset   offset: -4, -1, 0, +1, +4 (steep edges, flat middle)
  E6: No gating baseline   all zeros (control)

Run: python instnct/recipes/ab_phase_offset_sweep.py
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

SCHEDULE = [
    'add', 'enhance', 'reverse', 'mirror',
    'loop3', 'loop5', 'loop8',
    'flip', 'theta', 'channel', 'channel', 'remove',
]

# =========================================================================
# Build LUTs for each config
# =========================================================================

def build_cosine_phase_pattern():
    """Returns the 8 phase-shifted patterns as column indices into value array.
    Pattern[ch][tick] = which of the 5 values to use."""
    # Cosine at 8 angles: cos(0)=1, cos(pi/4)=0.707, cos(pi/2)=0, ...
    # Maps to value indices: 0=strongest, 1=strong, 2=neutral, 3=weak, 4=weakest
    # The pattern for ch0 (peak at tick 0): 0,1,2,3,4,3,2,1
    base = [0, 1, 2, 3, 4, 3, 2, 1]
    patterns = []
    for ch in range(8):
        patterns.append([base[(t - ch) % 8] for t in range(8)])
    return patterns

PATTERNS = build_cosine_phase_pattern()

def build_lut_from_values(values_5):
    """Build 9x8 float32 LUT from 5 values using cosine phase pattern.
    values_5[0]=peak effect, values_5[4]=trough effect."""
    lut = np.ones((9, 8), dtype=np.float32)
    for ch in range(8):
        for t in range(8):
            lut[ch + 1, t] = values_5[PATTERNS[ch][t]]
    return lut


def build_additive_lut(offsets_5):
    """Build LUT where values are threshold OFFSETS (added to theta).
    Comparison: charge >= theta + offset
    We store as float multiplier: (theta + offset) / theta = 1 + offset/theta
    But since theta varies per neuron, we store raw offsets and handle in eval."""
    # Store offsets directly as floats for simplicity in Python
    lut = np.zeros((9, 8), dtype=np.float32)
    for ch in range(8):
        for t in range(8):
            lut[ch + 1, t] = offsets_5[PATTERNS[ch][t]]
    return lut


# E1: Additive symmetric -3..+3
E1_OFFSETS = [-3.0, -2.0, 0.0, 2.0, 3.0]

# E2: Positive only 0..+5
E2_OFFSETS = [0.0, 1.0, 2.0, 4.0, 5.0]

# E3: Negative only -5..0
E3_OFFSETS = [-5.0, -4.0, -2.0, -1.0, 0.0]

# E4: Multiplicative (current cosine x10 as float)
E4_MULTS = [0.70, 0.788, 1.0, 1.212, 1.30]

# E5: Exponential (steep edges, flat middle)
E5_OFFSETS = [-4.0, -1.0, 0.0, 1.0, 4.0]

CONFIGS = {
    'E1_add_sym': {
        'label': 'Additive symmetric (-3..+3)',
        'mode': 'additive',
        'lut': build_additive_lut(E1_OFFSETS),
    },
    'E2_add_pos': {
        'label': 'Additive positive only (0..+5)',
        'mode': 'additive',
        'lut': build_additive_lut(E2_OFFSETS),
    },
    'E3_add_neg': {
        'label': 'Additive negative only (-5..0)',
        'mode': 'additive',
        'lut': build_additive_lut(E3_OFFSETS),
    },
    'E4_mult': {
        'label': 'Multiplicative cosine (0.7..1.3)',
        'mode': 'multiplicative',
        'lut': build_lut_from_values(E4_MULTS),
    },
    'E5_exp': {
        'label': 'Exponential offset (-4,0,+4)',
        'mode': 'additive',
        'lut': build_additive_lut(E5_OFFSETS),
    },
    'E6_none': {
        'label': 'No gating (control)',
        'mode': 'none',
        'lut': np.zeros((9, 8), dtype=np.float32),
    },
}

# =========================================================================
# Globals
# =========================================================================
BP_IN = None
_bp_out = None; _all_data = None; _bigram = None; _pol = None
_active_lut = None; _active_mode = 'multiplicative'

_ORIG_WAVE_LUT = SelfWiringGraph.WAVE_LUT.copy()

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

def init_w(bpi, bpo, data, bg, pol, lut, mode):
    global BP_IN, _bp_out, _all_data, _bigram, _pol, _active_lut, _active_mode
    BP_IN = bpi; _bp_out = bpo; _all_data = data; _bigram = bg; _pol = pol
    _active_lut = lut; _active_mode = mode
    if mode == 'multiplicative':
        SelfWiringGraph.WAVE_LUT = lut

def _eval_bigram(qdata, theta, channel, pol_f, seqs):
    qm = QuaternaryMask(H, qdata)
    sc = qm.to_directed_edges()
    _dummy = np.zeros((H, H), dtype=bool)
    total = 0.0
    MAX_CHARGE = 15.0
    for tb in seqs:
        state = np.zeros(H, np.float32); charge = np.zeros(H, np.float32)
        s = 0.0; n = 0
        for i in range(len(tb) - 1):
            inj = np.zeros(H, np.float32); inj[0:IN_DIM] = BP_IN[tb[i]]
            if _active_mode == 'multiplicative':
                state, charge = SelfWiringGraph.rollout_token(
                    inj, mask=_dummy, theta=theta,
                    decay=np.float32(0.16), ticks=TICKS, input_duration=INPUT_DURATION,
                    state=state, charge=charge, sparse_cache=sc,
                    polarity=pol_f, channel=channel)
            elif _active_mode == 'additive':
                # Custom additive forward pass (vectorized)
                src_list, tgt_list = sc
                for tick in range(TICKS):
                    if tick % 6 == 0:
                        charge[:] = np.maximum(charge - 1.0, 0.0)
                    if tick < INPUT_DURATION:
                        state[:] += inj
                    incoming = np.zeros(H, np.float32)
                    if len(src_list) > 0:
                        np.add.at(incoming, tgt_list, state[src_list])
                    charge[:] = np.clip(charge + incoming, 0.0, MAX_CHARGE)
                    offset = _active_lut[channel, tick % 8]
                    effective_theta = np.clip(theta + offset, 1.0, MAX_CHARGE)
                    fired = charge >= effective_theta
                    state[:] = 0.0
                    state[fired] = pol_f[fired]
                    charge[fired] = 0.0
            else:
                # No gating
                state, charge = SelfWiringGraph.rollout_token(
                    inj, mask=_dummy, theta=theta,
                    decay=np.float32(0.16), ticks=TICKS, input_duration=INPUT_DURATION,
                    state=state, charge=charge, sparse_cache=sc,
                    polarity=pol_f)
            logits = np.dot(_bp_out, charge[H - OUT_DIM:])
            e = np.exp(logits - logits.max()); pred = e / e.sum()
            tgt = _bigram[tb[i]]
            cos = np.dot(pred, tgt) / (np.linalg.norm(pred) * np.linalg.norm(tgt) + 1e-8)
            s += cos; n += 1
        total += s / n if n else 0
    return total / len(seqs)

def worker_eval(args):
    qdata_bytes, theta, channel, pol_f, seed, pt = args
    rng = random.Random(seed); nrng = np.random.RandomState(seed)
    qm = QuaternaryMask(H, np.frombuffer(qdata_bytes, dtype=np.uint8).copy())
    nq = qm.copy(); nt = theta.copy(); nc = channel.copy(); npf = pol_f.copy()

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

    seqs = []
    for _ in range(N_TRAIN_SEQS):
        off = nrng.randint(0, len(_all_data) - SEQ_LEN)
        seqs.append(_all_data[off:off + SEQ_LEN])

    old = _eval_bigram(qm.data, theta, channel, pol_f, seqs)
    new = _eval_bigram(nq.data, nt, nc, npf, seqs)
    return {
        'delta': float(new - old), 'type': pt,
        'new_qdata': nq.data.tobytes() if new > old else None,
        'new_theta': nt if new > old else None,
        'new_channel': nc if new > old else None,
        'new_pol': npf if new > old else None,
    }

def eval_accuracy(qdata, theta, channel, pol_f, text_bytes, bp_in, bp_out):
    qm = QuaternaryMask(H, qdata)
    sc = qm.to_directed_edges()
    _dummy = np.zeros((H, H), dtype=bool)
    state = np.zeros(H, np.float32); charge = np.zeros(H, np.float32)
    MAX_CHARGE = 15.0
    cor = 0; tot = 0
    for i in range(len(text_bytes) - 1):
        inj = np.zeros(H, np.float32); inj[0:IN_DIM] = bp_in[text_bytes[i]]
        if _active_mode == 'multiplicative':
            state, charge = SelfWiringGraph.rollout_token(
                inj, mask=_dummy, theta=theta,
                decay=np.float32(0.16), ticks=TICKS, input_duration=INPUT_DURATION,
                state=state, charge=charge, sparse_cache=sc,
                polarity=pol_f, channel=channel)
        elif _active_mode == 'additive':
            src_list, tgt_list = sc
            for tick in range(TICKS):
                if tick % 6 == 0:
                    charge[:] = np.maximum(charge - 1.0, 0.0)
                if tick < INPUT_DURATION:
                    state[:] += inj
                incoming = np.zeros(H, np.float32)
                if len(src_list) > 0:
                    np.add.at(incoming, tgt_list, state[src_list])
                charge[:] = np.clip(charge + incoming, 0.0, MAX_CHARGE)
                offset = _active_lut[channel, tick % 8]
                effective_theta = np.clip(theta + offset, 1.0, MAX_CHARGE)
                fired = charge >= effective_theta
                state[:] = 0.0
                state[fired] = pol_f[fired]
                charge[fired] = 0.0
        else:
            state, charge = SelfWiringGraph.rollout_token(
                inj, mask=_dummy, theta=theta,
                decay=np.float32(0.16), ticks=TICKS, input_duration=INPUT_DURATION,
                state=state, charge=charge, sparse_cache=sc,
                polarity=pol_f)
        logits = np.dot(bp_out, charge[H - OUT_DIM:])
        if np.argmax(logits) == text_bytes[i + 1]: cor += 1
        tot += 1
    return cor / tot if tot else 0


def run_config(name, cfg, init_state, bp_in, bp_out, all_data, bigram, eval_seqs):
    qdata, theta, channel, pol_f = [x.copy() for x in init_state]

    lut = cfg['lut']
    mode = cfg['mode']

    if mode == 'multiplicative':
        SelfWiringGraph.WAVE_LUT = lut
    elif mode == 'none':
        SelfWiringGraph.WAVE_LUT = _ORIG_WAVE_LUT

    print(f"\n{'='*60}")
    print(f"  Config: {name} -- {cfg['label']}")
    print(f"  Mode: {mode}")
    if mode == 'additive':
        vals = [lut[1, t] for t in range(8)]
        print(f"  Ch1 offsets: {vals}")
    elif mode == 'multiplicative':
        vals = [f'{lut[1, t]:.3f}' for t in range(8)]
        print(f"  Ch1 multipliers: {vals}")
    print(f"{'='*60}")

    pool = Pool(N_WORKERS, initializer=init_w,
                initargs=(bp_in, bp_out, all_data, bigram, pol_f, lut, mode))
    acc_by_type = defaultdict(int)
    t0 = time.time()
    best_eval = 0
    history = []

    try:
        for step in range(1, BUILD_STEPS + 1):
            pt = SCHEDULE[(step - 1) % len(SCHEDULE)]
            edges = QuaternaryMask(H, qdata).count_edges()
            if pt in ('remove', 'reverse') and edges < 50: pt = 'add'

            args = [(qdata.tobytes(), theta.copy(), channel.copy(), pol_f.copy(),
                     1000 + step * 50 + w, pt) for w in range(N_WORKERS)]
            results = pool.map(worker_eval, args)
            best_r = max(results, key=lambda x: x['delta'])

            if best_r['delta'] > THRESHOLD:
                if best_r['new_qdata'] is not None:
                    qdata[:] = np.frombuffer(best_r['new_qdata'], dtype=np.uint8)
                if best_r['new_theta'] is not None: theta[:] = best_r['new_theta']
                if best_r['new_channel'] is not None: channel[:] = best_r['new_channel']
                if best_r['new_pol'] is not None: pol_f[:] = best_r['new_pol']
                acc_by_type[best_r['type']] += 1

            if step % EVAL_EVERY == 0:
                ea = np.mean([eval_accuracy(qdata, theta, channel, pol_f, s, bp_in, bp_out)
                              for s in eval_seqs])
                if ea > best_eval: best_eval = ea
                elapsed = time.time() - t0
                print(f"  [{step:5d}] eval={ea*100:.1f}% best={best_eval*100:.1f}% "
                      f"edges={QuaternaryMask(H, qdata).count_edges()} {elapsed:.0f}s")
                sys.stdout.flush()
                history.append({'step': step, 'eval': round(ea * 100, 2),
                                'best': round(best_eval * 100, 2)})
    finally:
        pool.terminate(); pool.join()

    elapsed = time.time() - t0
    final_eval = np.mean([eval_accuracy(qdata, theta, channel, pol_f, s, bp_in, bp_out)
                          for s in eval_seqs])
    result = {
        'config': name, 'label': cfg['label'],
        'final_eval': round(final_eval * 100, 2),
        'best_eval': round(best_eval * 100, 2),
        'edges': int(QuaternaryMask(H, qdata).count_edges()),
        'elapsed_sec': round(elapsed, 1),
        'accepts': dict(acc_by_type),
        'history': history,
    }
    print(f"\n  DONE: {name} -> final={final_eval*100:.1f}% best={best_eval*100:.1f}% "
          f"time={elapsed:.0f}s")
    return result


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
    qdata_init = qm.data.copy()
    theta_init = np.full(H, 1.0, np.float32)
    channel_init = init_rng.randint(1, 9, size=H).astype(np.uint8)
    pol_f_init = np.where(ref.polarity, 1.0, -1.0).astype(np.float32)

    init_state = (qdata_init, theta_init, channel_init, pol_f_init)

    all_results = {}
    for name, cfg in CONFIGS.items():
        result = run_config(name, cfg, init_state, bp_in, bp_out,
                            ALL_DATA, bigram, eval_seqs)
        all_results[name] = result

    SelfWiringGraph.WAVE_LUT = _ORIG_WAVE_LUT

    print(f"\n{'='*60}")
    print(f"  SWEEP SUMMARY")
    print(f"{'='*60}")
    for name, r in sorted(all_results.items(), key=lambda x: -x[1]['best_eval']):
        print(f"  {r['label']:40s}  final={r['final_eval']:5.1f}%  "
              f"best={r['best_eval']:5.1f}%  edges={r['edges']}")

    out_path = os.path.join(BASE_DIR, "data", "phase_offset_sweep_result.json")
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")

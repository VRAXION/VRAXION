"""
A/B/C/D: Loop strategy sweep at H=512
======================================
A: baseline (random add, no explicit loops)
B: bidir focus (50% of adds become upgrade → 2-loops)
C: triangle focus (50% of adds become add_loop(3) → 3-cycles)
D: long loop focus (50% of adds become add_loop(6) → 4-6 cycles)

All on real FineWeb data, quaternary mask, 18 workers.
"""
import sys, os, time, random, json
from collections import deque
import numpy as np
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "model"))
from graph import SelfWiringGraph
from quaternary_mask import QuaternaryMask

# === CONFIG ===
H = 512; N_WORKERS = 18; TICKS = 8; INPUT_DURATION = 2
THRESHOLD = 0.00005; EVAL_EVERY = 20
INIT_DENSITY = 0.05; MAX_STEPS = 3000
PHI = (1 + 5**0.5) / 2
IN_DIM = int(round(H / PHI))    # 316
OUT_DIM = int(round(H / PHI))   # 316
SDR_K = int(round(IN_DIM * 0.20))  # 63
SEQ_LEN = 100; N_TRAIN_SEQS = 2; N_EVAL_SEQS = 5
PLATEAU_WINDOW = 8; PLATEAU_THRESH_PCT = 0.5
PLATEAU_STRIKES_MAX = 3; PLATEAU_MIN_STEPS = 500

SCHEDULE = ['add', 'flip', 'theta', 'channel', 'theta', 'channel', 'flip', 'remove']

# SDR patterns
BP_IN = None

# Worker globals
_bp_out = None; _all_data = None; _bigram = None; _pol_f = None; _mode = None

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
    global BP_IN, _bp_out, _all_data, _bigram, _pol_f, _mode
    BP_IN = bpi; _bp_out = bpo; _all_data = data; _bigram = bg; _pol_f = pol; _mode = mode

def _eval_bigram_quat(qdata, theta, channel, pol_f, seqs):
    qm = QuaternaryMask(H, qdata)
    sc = qm.to_directed_edges()
    _dummy = np.zeros((H, H), dtype=bool)
    total = 0.0
    for tb in seqs:
        state = np.zeros(H, np.float32); charge = np.zeros(H, np.float32)
        s = 0.0; n = 0
        for i in range(len(tb) - 1):
            inj = np.zeros(H, np.float32); inj[0:IN_DIM] = BP_IN[tb[i]]
            state, charge = SelfWiringGraph.rollout_token(
                inj, mask=_dummy, theta=theta,
                decay=np.float32(0.16), ticks=TICKS, input_duration=INPUT_DURATION,
                state=state, charge=charge, sparse_cache=sc,
                polarity=pol_f, channel=channel)
            logits = np.dot(_bp_out, charge[H - OUT_DIM:])
            e = np.exp(logits - logits.max()); pred = e / e.sum()
            tgt = _bigram[tb[i]]
            cos = np.dot(pred, tgt) / (np.linalg.norm(pred) * np.linalg.norm(tgt) + 1e-8)
            s += cos; n += 1
        total += s / n if n else 0
    return total / len(seqs)

# ------------------------------------------------------------------
# Mutation logic per mode
# ------------------------------------------------------------------

def _apply_add_baseline(qm, rng):
    """Standard add: random edge."""
    undo = []
    qm.mutate_add(rng, undo)
    return undo

def _apply_add_bidir(qm, rng):
    """50% random add, 50% upgrade (create bidir pair = 2-loop)."""
    undo = []
    if rng.random() < 0.5 and len(qm._alive) > 0:
        qm.mutate_upgrade(rng, undo)
    else:
        qm.mutate_add(rng, undo)
    return undo

def _apply_add_tri3(qm, rng):
    """50% random add, 50% add_loop(max_len=3) = triangle."""
    undo = []
    if rng.random() < 0.5:
        _add_loop_quat(qm, rng, undo, max_len=3)
        if not undo:
            qm.mutate_add(rng, undo)  # fallback
    else:
        qm.mutate_add(rng, undo)
    return undo

def _apply_add_loop6(qm, rng):
    """50% random add, 50% add_loop(max_len=6) = longer circuits."""
    undo = []
    if rng.random() < 0.5:
        _add_loop_quat(qm, rng, undo, max_len=6)
        if not undo:
            qm.mutate_add(rng, undo)
        return undo
    else:
        qm.mutate_add(rng, undo)
    return undo

def _add_loop_quat(qm, rng, undo, max_len=6):
    """Add a complete directed loop of random length [2, max_len] on QuaternaryMask."""
    loop_len = rng.randint(2, max(2, max_len))
    nodes = [rng.randint(0, qm.H - 1)]
    for _ in range(loop_len - 1):
        n = rng.randint(0, qm.H - 1)
        if n in nodes:
            return  # collision
        nodes.append(n)
    # Check all edges free
    edges = []
    for i in range(loop_len):
        r, c = nodes[i], nodes[(i + 1) % loop_len]
        if qm.get_pair(r, c) != 0:
            return
        edges.append((r, c))
    # Commit
    for r, c in edges:
        qm.set_pair(r, c, 1)
        undo.append(('QA', qm._pair_index(r, c), 0))

ADD_FNS = {
    'A': _apply_add_baseline,
    'B': _apply_add_bidir,
    'C': _apply_add_tri3,
    'D': _apply_add_loop6,
}

# ------------------------------------------------------------------
# Worker
# ------------------------------------------------------------------

def worker_eval(args):
    qdata_bytes, theta, channel, seed, pt = args
    rng = random.Random(seed); nrng = np.random.RandomState(seed)
    qm = QuaternaryMask(H, np.frombuffer(qdata_bytes, dtype=np.uint8).copy())
    nq = qm; nt = theta; nc = channel

    if pt == 'add':
        nq = qm.copy()
        undo = ADD_FNS[_mode](nq, rng)
        if not undo: return {'delta': -1e9, 'type': 'add'}
    elif pt == 'flip':
        nq = qm.copy(); undo = []; nq.mutate_flip(rng, undo)
        if not undo: return {'delta': -1e9, 'type': 'flip'}
    elif pt == 'remove':
        nq = qm.copy(); undo = []; nq.mutate_remove(rng, undo)
        if not undo: return {'delta': -1e9, 'type': 'remove'}
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

    old = _eval_bigram_quat(qm.data, theta, channel, _pol_f, seqs)
    new = _eval_bigram_quat(nq.data, nt, nc, _pol_f, seqs)
    return {'delta': float(new - old), 'type': pt,
            'new_qdata': nq.data.tobytes() if new > old else None,
            'new_theta': nt if pt == 'theta' and new > old else None,
            'new_channel': nc if pt == 'channel' and new > old else None}

# ------------------------------------------------------------------
# Eval
# ------------------------------------------------------------------

def eval_accuracy(qdata, theta, channel, pol_f, text_bytes, bp_in, bp_out):
    qm = QuaternaryMask(H, qdata)
    sc = qm.to_directed_edges()
    _dummy = np.zeros((H, H), dtype=bool)
    state = np.zeros(H, np.float32); charge = np.zeros(H, np.float32)
    cor = 0; tot = 0
    for i in range(len(text_bytes) - 1):
        inj = np.zeros(H, np.float32); inj[0:IN_DIM] = bp_in[text_bytes[i]]
        state, charge = SelfWiringGraph.rollout_token(
            inj, mask=_dummy, theta=theta,
            decay=np.float32(0.16), ticks=TICKS, input_duration=INPUT_DURATION,
            state=state, charge=charge, sparse_cache=sc,
            polarity=pol_f, channel=channel)
        logits = np.dot(bp_out, charge[H - OUT_DIM:])
        if np.argmax(logits) == text_bytes[i + 1]:
            cor += 1
        tot += 1
    return cor / tot if tot else 0

# ------------------------------------------------------------------
# Run one mode
# ------------------------------------------------------------------

def run_mode(mode_label, init_qdata, init_theta, init_channel, pol_f,
             ALL_DATA, bigram, eval_seqs, bp_in, bp_out):
    desc = {'A': 'baseline (random add)', 'B': 'bidir (2-loop upgrade)',
            'C': 'triangle (add_loop 3)', 'D': 'long loop (add_loop 6)'}[mode_label]
    print(f"\n{'=' * 70}")
    print(f"  Mode {mode_label}: {desc}")
    print(f"  H={H}, workers={N_WORKERS}, max_steps={MAX_STEPS}")

    qdata = init_qdata.copy()
    theta = init_theta.copy()
    channel = init_channel.copy()
    qm0 = QuaternaryMask(H, qdata)
    print(f"  Init: {qm0.count_edges()} edges, {qm0.count_bidir()} bidir")
    print(f"{'=' * 70}")

    pool = Pool(N_WORKERS, initializer=init_w,
                initargs=(bp_in, bp_out, ALL_DATA, bigram, pol_f, mode_label))

    accepts = 0; acc_by_type = {}
    eval_history = []; plateau_strikes = 0
    recent_deltas = deque(maxlen=50)
    t0 = time.time(); best_eval = 0; stall_steps = 0

    try:
        for step in range(1, MAX_STEPS + 1):
            pt = SCHEDULE[(step - 1) % len(SCHEDULE)]
            qm_tmp = QuaternaryMask(H, qdata)
            edges = qm_tmp.count_edges()
            if pt in ('flip', 'remove') and edges < 100: pt = 'add'
            if edges == 0: pt = 'add'

            args = [(qdata.tobytes(), theta.copy(), channel.copy(),
                     1000 + step * 50 + w, pt) for w in range(N_WORKERS)]
            results = pool.map(worker_eval, args)
            best_r = max(results, key=lambda x: x['delta'])

            accepted = False
            if best_r['delta'] > THRESHOLD:
                if best_r['new_qdata'] is not None:
                    qdata[:] = np.frombuffer(best_r['new_qdata'], dtype=np.uint8)
                    accepted = True
                if best_r['new_theta'] is not None:
                    theta[:] = best_r['new_theta']; accepted = True
                if best_r['new_channel'] is not None:
                    channel[:] = best_r['new_channel']; accepted = True
                if accepted:
                    accepts += 1
                    acc_by_type[best_r['type']] = acc_by_type.get(best_r['type'], 0) + 1

            recent_deltas.append(best_r['delta'] if best_r['delta'] > -1e8 else 0.0)

            if step % EVAL_EVERY == 0:
                elapsed = time.time() - t0
                ea_list = [eval_accuracy(qdata, theta, channel, pol_f, s, bp_in, bp_out)
                           for s in eval_seqs]
                ea = np.mean(ea_list)

                if ea > best_eval:
                    best_eval = ea; stall_steps = 0
                else:
                    stall_steps += EVAL_EVERY

                qm_now = QuaternaryMask(H, qdata)
                bidir_s, tri_s = qm_now.loop_levels()
                cur_edges = qm_now.count_edges()
                cur_bidir = qm_now.count_bidir()

                eval_history.append({
                    'step': step, 'eval': round(ea * 100, 2),
                    'edges': cur_edges, 'bidir': cur_bidir,
                    'tri_neurons': int(tri_s.sum()),
                    'bidir_neurons': int(bidir_s.sum()),
                })

                rate = sum(1 for d in recent_deltas if d > THRESHOLD) / max(len(recent_deltas), 1)
                sps = step / elapsed
                at = ' '.join(f"{k}={v}" for k, v in sorted(acc_by_type.items()))
                print(f"  [{mode_label}][{step:4d}] eval={ea*100:.1f}% best={best_eval*100:.1f}% "
                      f"edges={cur_edges} bidir={cur_bidir} tri_n={tri_s.sum()} "
                      f"[{at}] rate={rate*100:.0f}% {elapsed:.0f}s ({sps:.1f}sps)")
                sys.stdout.flush()

                if len(eval_history) >= PLATEAU_WINDOW and step >= PLATEAU_MIN_STEPS:
                    window = [e['eval'] for e in eval_history[-PLATEAU_WINDOW:]]
                    spread = max(window) - min(window)
                    if spread < PLATEAU_THRESH_PCT:
                        plateau_strikes += 1
                        if plateau_strikes >= PLATEAU_STRIKES_MAX:
                            print(f"  ** PLATEAU STOP at step {step}")
                            break
                    else:
                        plateau_strikes = 0

                if stall_steps >= 400 and step >= PLATEAU_MIN_STEPS:
                    print(f"  ** STALL STOP at step {step}")
                    break
    finally:
        pool.terminate(); pool.join()

    elapsed = time.time() - t0
    qm_final = QuaternaryMask(H, qdata)
    bidir_f, tri_f = qm_final.loop_levels()
    print(f"\n  [{mode_label}] DONE: best={best_eval*100:.1f}% edges={qm_final.count_edges()} "
          f"bidir={qm_final.count_bidir()} tri_n={tri_f.sum()} {elapsed:.0f}s")

    return {
        'mode': mode_label, 'desc': desc, 'best': best_eval,
        'edges': qm_final.count_edges(), 'bidir': qm_final.count_bidir(),
        'tri_neurons': int(tri_f.sum()), 'bidir_neurons': int(bidir_f.sum()),
        'accepts': accepts, 'acc_by_type': dict(acc_by_type),
        'steps': eval_history[-1]['step'] if eval_history else 0,
        'time': elapsed, 'history': eval_history,
    }


if __name__ == "__main__":
    sys.stdout.reconfigure(line_buffering=True)
    sys.path.insert(0, str(ROOT))
    from lib.data import load_fineweb_bytes, resolve_fineweb_path
    resolve_fineweb_path()
    ALL_DATA = load_fineweb_bytes()
    print(f"Loaded {len(ALL_DATA)/1e6:.1f} MB text")

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    bigram_path = os.path.join(BASE_DIR, "data", "bigram_table.npy")
    if os.path.exists(bigram_path):
        bigram = np.load(bigram_path)
    else:
        os.makedirs(os.path.join(BASE_DIR, "data"), exist_ok=True)
        counts = np.zeros((256, 256), dtype=np.float64)
        for i in range(len(ALL_DATA) - 1):
            counts[ALL_DATA[i], ALL_DATA[i + 1]] += 1
        row_sums = counts.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        bigram = (counts / row_sums).astype(np.float32)
        np.save(bigram_path, bigram)

    bp_in = build_sdr(256, IN_DIM, SDR_K, 42)
    BP_IN = bp_in  # module level for workers
    bp_out = build_freq_order(OUT_DIM, bigram)

    eval_rng = np.random.RandomState(9999)
    eval_seqs = [ALL_DATA[off:off + SEQ_LEN]
                 for off in [eval_rng.randint(0, len(ALL_DATA) - SEQ_LEN)
                             for _ in range(N_EVAL_SEQS)]]

    # Shared init
    init_rng = np.random.RandomState(42)
    ref = SelfWiringGraph(max(IN_DIM, 16), hidden=H, projection_scale=1.0)
    init_mask = (init_rng.rand(H, H) < INIT_DENSITY).astype(bool)
    np.fill_diagonal(init_mask, False)
    init_qm = QuaternaryMask.from_bool_mask(init_mask)
    init_theta = np.full(H, 1.0, np.float32)
    init_channel = init_rng.randint(1, 9, size=H).astype(np.uint8)
    pol_f = np.where(ref.polarity, 1.0, -1.0).astype(np.float32)

    bidir0, tri0 = init_qm.loop_levels()
    print(f"Init: {init_qm.count_edges()} edges, {init_qm.count_bidir()} bidir, "
          f"{tri0.sum()} tri neurons, {bidir0.sum()} bidir neurons")

    # Run all 4 modes
    results = []
    for label in ['A', 'B', 'C', 'D']:
        r = run_mode(label, init_qm.data, init_theta, init_channel,
                     pol_f, ALL_DATA, bigram, eval_seqs, bp_in, bp_out)
        results.append(r)

    # Final report
    print(f"\n{'=' * 70}")
    print(f"  LOOP STRATEGY SWEEP — H={H}")
    print(f"{'=' * 70}")
    for r in results:
        print(f"  {r['mode']} ({r['desc']:30s}): best={r['best']*100:.1f}%  "
              f"edges={r['edges']}  bidir={r['bidir']}  tri_n={r['tri_neurons']}  "
              f"acc={r['accepts']}  {r['time']:.0f}s")
        if r['acc_by_type']:
            print(f"       accepts: {r['acc_by_type']}")

    print()
    base = results[0]['best']
    for r in results[1:]:
        diff = r['best'] - base
        print(f"  {r['mode']} vs A: {diff*100:+.2f}%")
    print(f"{'=' * 70}")

    out_path = os.path.join(BASE_DIR, "data", "ab_loop_strategy_h512.json")
    os.makedirs(os.path.join(BASE_DIR, "data"), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved to {out_path}")

"""
A/B Test: Quaternary Upper-Triangle Mask vs Binary H×H Mask
============================================================
Mode A (BOOL):   Current H×H boolean mask, flip = rewire (baseline)
Mode B (QUAT):   Quaternary upper-triangle, flip = atomic direction reversal
Mode C (QUAT+):  Quaternary + upgrade op (explicit bidir creation)

Forward pass is IDENTICAL — all modes produce (rows, cols) sparse cache.
Only the mask representation and mutation operators differ.
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

# === CONFIG (canonical) ===
H = 256; N_WORKERS = 9; TICKS = 8; INPUT_DURATION = 2
THRESHOLD = 0.00005; EVAL_EVERY = 20; INIT_DENSITY = 0.05
MAX_STEPS = 5000
PHI = (1 + 5**0.5) / 2
IN_DIM = int(round(H / PHI))
OUT_DIM = int(round(H / PHI))
SDR_K = int(round(IN_DIM * 0.20))
WAVE_LUT = SelfWiringGraph.WAVE_LUT
SEQ_LEN = 100; N_TRAIN_SEQS = 2; N_EVAL_SEQS = 5
PLATEAU_WINDOW = 8; PLATEAU_THRESH_PCT = 0.5
PLATEAU_STRIKES_MAX = 3; PLATEAU_MIN_STEPS = 300

SCHEDULE_A = ['add', 'flip', 'theta', 'channel', 'theta', 'channel', 'flip', 'remove']
SCHEDULE_B = ['add', 'flip', 'theta', 'channel', 'theta', 'channel', 'flip', 'remove']
SCHEDULE_C = ['add', 'flip', 'upgrade', 'channel', 'theta', 'channel', 'flip', 'remove']

# SDR input patterns (canonical)
BP_IN = None  # set in main

# --- Worker globals ---
_bp_out = None; _all_data = None; _bigram = None; _pol_f = None; _mode = None


def build_sdr(n, dim, k, seed):
    rng = np.random.RandomState(seed)
    t = np.zeros((n, dim), np.float32)
    for v in range(n):
        t[v, rng.choice(dim, size=k, replace=False)] = 1.0
    return t


def build_freq_order(dim, bigram, seed=12345):
    freq = bigram.sum(axis=0) + bigram.sum(axis=1)
    rank = np.argsort(freq)[::-1]
    rng = np.random.RandomState(seed)
    p = np.zeros((256, dim), np.float32)
    for i, byte_idx in enumerate(rank):
        t = i / 255.0
        for d in range(dim):
            p[byte_idx, d] = np.sin(2 * np.pi * t * (d + 1) / dim * 3) + rng.randn() * 0.3
    p /= np.linalg.norm(p, axis=1, keepdims=True) + 1e-8
    return p.astype(np.float32)


def init_w(bpi, bpo, data, bg, pol, mode):
    global BP_IN, _bp_out, _all_data, _bigram, _pol_f, _mode
    BP_IN = bpi; _bp_out = bpo; _all_data = data; _bigram = bg; _pol_f = pol; _mode = mode


def _eval_bigram_from_cache(sparse_cache, theta, channel, pol_f, seqs):
    """Evaluate bigram cosine — takes sparse cache directly."""
    total = 0.0
    for tb in seqs:
        state = np.zeros(H, np.float32); charge = np.zeros(H, np.float32)
        s = 0.0; n = 0
        for i in range(len(tb) - 1):
            inj = np.zeros(H, np.float32); inj[0:IN_DIM] = BP_IN[tb[i]]
            state, charge = SelfWiringGraph.rollout_token(
                inj, mask=np.zeros((H, H), dtype=bool), theta=theta,
                decay=np.float32(0.16), ticks=TICKS, input_duration=INPUT_DURATION,
                state=state, charge=charge, sparse_cache=sparse_cache,
                polarity=pol_f, channel=channel)
            logits = np.dot(_bp_out, charge[H - OUT_DIM:])
            e = np.exp(logits - logits.max()); pred = e / e.sum()
            tgt = _bigram[tb[i]]
            cos = np.dot(pred, tgt) / (np.linalg.norm(pred) * np.linalg.norm(tgt) + 1e-8)
            s += cos; n += 1
        total += s / n if n else 0
    return total / len(seqs)


def _eval_bigram_bool(mask, theta, channel, pol_f, seqs):
    sc = SelfWiringGraph.build_sparse_cache(mask)
    return _eval_bigram_from_cache(sc, theta, channel, pol_f, seqs)


def _eval_bigram_quat(qdata, theta, channel, pol_f, seqs):
    qm = QuaternaryMask(H, qdata)
    sc = qm.to_directed_edges()
    return _eval_bigram_from_cache(sc, theta, channel, pol_f, seqs)


# ------------------------------------------------------------------
# Workers
# ------------------------------------------------------------------

def worker_eval_bool(args):
    """Mode A worker: H×H bool mask mutations."""
    mf, theta, channel, seed, pt = args
    rng = random.Random(seed); nrng = np.random.RandomState(seed)
    mask = mf.reshape(H, H); nm = mask; nt = theta; nc = channel; npf = _pol_f

    if pt == 'add':
        r = rng.randint(0, H - 1); c = rng.randint(0, H - 1)
        if r == c or mask[r, c]: return {'delta': -1e9, 'type': 'add'}
        nm = mask.copy(); nm[r, c] = True
    elif pt == 'flip':
        alive = list(zip(*np.where(mask)))
        if not alive: return {'delta': -1e9, 'type': 'flip'}
        r, c = alive[rng.randint(0, len(alive) - 1)]
        nc2 = rng.randint(0, H - 1)
        if nc2 == r or nc2 == c or mask[r, nc2]: return {'delta': -1e9, 'type': 'flip'}
        nm = mask.copy(); nm[r, c] = False; nm[r, nc2] = True
    elif pt == 'remove':
        alive = list(zip(*np.where(mask)))
        if not alive: return {'delta': -1e9, 'type': 'remove'}
        r, c = alive[rng.randint(0, len(alive) - 1)]
        nm = mask.copy(); nm[r, c] = False
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

    old = _eval_bigram_bool(mask, theta, channel, _pol_f, seqs)
    new = _eval_bigram_bool(nm, nt, nc, npf, seqs)
    return {'delta': float(new - old), 'type': pt,
            'new_mask_flat': nm.flatten() if new > old else None,
            'new_theta': nt if pt == 'theta' and new > old else None,
            'new_channel': nc if pt == 'channel' and new > old else None}


def worker_eval_quat(args):
    """Mode B/C worker: quaternary mask mutations."""
    qdata_bytes, theta, channel, seed, pt = args
    rng = random.Random(seed); nrng = np.random.RandomState(seed)
    qm = QuaternaryMask(H, np.frombuffer(qdata_bytes, dtype=np.uint8).copy())
    nq = qm; nt = theta; nc = channel

    if pt == 'add':
        nq = qm.copy(); undo = []; nq.mutate_add(rng, undo)
        if not undo: return {'delta': -1e9, 'type': 'add'}
    elif pt == 'flip':
        nq = qm.copy(); undo = []; nq.mutate_flip(rng, undo)
        if not undo: return {'delta': -1e9, 'type': 'flip'}
    elif pt == 'remove':
        nq = qm.copy(); undo = []; nq.mutate_remove(rng, undo)
        if not undo: return {'delta': -1e9, 'type': 'remove'}
    elif pt == 'upgrade':
        nq = qm.copy(); undo = []; nq.mutate_upgrade(rng, undo)
        if not undo: return {'delta': -1e9, 'type': 'upgrade'}
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
# Eval accuracy
# ------------------------------------------------------------------

def eval_accuracy_bool(mask, theta, channel, pol_f, text_bytes, bp_in, bp_out):
    sc = SelfWiringGraph.build_sparse_cache(mask)
    return _eval_acc(sc, theta, channel, pol_f, text_bytes, bp_in, bp_out)


def eval_accuracy_quat(qdata, theta, channel, pol_f, text_bytes, bp_in, bp_out):
    qm = QuaternaryMask(H, qdata)
    sc = qm.to_directed_edges()
    return _eval_acc(sc, theta, channel, pol_f, text_bytes, bp_in, bp_out)


def _eval_acc(sparse_cache, theta, channel, pol_f, text_bytes, bp_in, bp_out):
    state = np.zeros(H, np.float32); charge = np.zeros(H, np.float32)
    cor = 0; tot = 0
    for i in range(len(text_bytes) - 1):
        inj = np.zeros(H, np.float32); inj[0:IN_DIM] = bp_in[text_bytes[i]]
        state, charge = SelfWiringGraph.rollout_token(
            inj, mask=np.zeros((H, H), dtype=bool), theta=theta,
            decay=np.float32(0.16), ticks=TICKS, input_duration=INPUT_DURATION,
            state=state, charge=charge, sparse_cache=sparse_cache,
            polarity=pol_f, channel=channel)
        logits = np.dot(bp_out, charge[H - OUT_DIM:])
        if np.argmax(logits) == text_bytes[i + 1]:
            cor += 1
        tot += 1
    return cor / tot if tot else 0


# ------------------------------------------------------------------
# Bidir counting for Mode A (for cross-mode comparison)
# ------------------------------------------------------------------

def count_bidir_bool(mask):
    return int(np.triu(mask & mask.T, k=1).sum())


# ------------------------------------------------------------------
# Run one mode
# ------------------------------------------------------------------

def run_mode(mode_label, schedule, init_mask, init_theta, init_channel,
             pol_f, ALL_DATA, bigram, eval_seqs, bp_out_local, seed=42):
    print(f"\n{'=' * 60}")
    is_quat = mode_label in ('B', 'C')
    desc = {'A': 'BOOL H×H (baseline)', 'B': 'QUATERNARY (same schedule)',
            'C': 'QUATERNARY + UPGRADE'}[mode_label]
    print(f"  Mode {mode_label}: {desc}")
    print(f"  H={H}, schedule={schedule}")

    # Init
    if is_quat:
        qm = QuaternaryMask.from_bool_mask(init_mask)
        qdata = qm.data.copy()
        init_edges = qm.count_edges()
        init_bidir = qm.count_bidir()
        mem = qm.memory_bytes
    else:
        mask = init_mask.copy()
        init_edges = int(mask.sum())
        init_bidir = count_bidir_bool(mask)
        mem = mask.nbytes

    theta = init_theta.copy()
    channel = init_channel.copy()
    print(f"  Init: {init_edges} edges, {init_bidir} bidir pairs, {mem} bytes")
    print(f"{'=' * 60}")

    worker_fn = worker_eval_quat if is_quat else worker_eval_bool
    pool = Pool(N_WORKERS, initializer=init_w,
                initargs=(BP_IN, bp_out_local, ALL_DATA, bigram, pol_f, mode_label))

    accepts = 0
    acc_by_type = {}
    eval_history = []
    plateau_strikes = 0
    recent_deltas = deque(maxlen=50)
    t0 = time.time()
    best_eval = 0
    stall_steps = 0

    try:
        for step in range(1, MAX_STEPS + 1):
            pt = schedule[(step - 1) % len(schedule)]
            # Edge count safety
            if is_quat:
                edges = QuaternaryMask(H, qdata).count_edges()
            else:
                edges = int(mask.sum())
            if pt in ('flip', 'remove', 'upgrade') and edges < 50:
                pt = 'add'
            if edges == 0:
                pt = 'add'

            # Build args
            if is_quat:
                mask_arg = qdata.tobytes()
                args = [(mask_arg, theta.copy(), channel.copy(),
                         1000 + step * 50 + w, pt) for w in range(N_WORKERS)]
            else:
                mask_flat = mask.flatten()
                args = [(mask_flat, theta.copy(), channel.copy(),
                         1000 + step * 50 + w, pt) for w in range(N_WORKERS)]

            results = pool.map(worker_fn, args)
            best_r = max(results, key=lambda x: x['delta'])

            accepted = False
            if best_r['delta'] > THRESHOLD:
                if is_quat and best_r['new_qdata'] is not None:
                    qdata[:] = np.frombuffer(best_r['new_qdata'], dtype=np.uint8)
                    accepted = True
                elif not is_quat and best_r['new_mask_flat'] is not None:
                    mask[:] = best_r['new_mask_flat'].reshape(H, H)
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
                ea_list = []
                for s in eval_seqs:
                    if is_quat:
                        ea_list.append(eval_accuracy_quat(qdata, theta, channel, pol_f, s, BP_IN, bp_out_local))
                    else:
                        ea_list.append(eval_accuracy_bool(mask, theta, channel, pol_f, s, BP_IN, bp_out_local))
                ea = np.mean(ea_list)

                if ea > best_eval:
                    best_eval = ea; stall_steps = 0
                else:
                    stall_steps += EVAL_EVERY

                # Bidir count
                if is_quat:
                    bidir = QuaternaryMask(H, qdata).count_bidir()
                    cur_edges = QuaternaryMask(H, qdata).count_edges()
                else:
                    bidir = count_bidir_bool(mask)
                    cur_edges = int(mask.sum())

                eval_history.append({
                    'step': step, 'eval': round(ea * 100, 2),
                    'edges': cur_edges, 'bidir': bidir,
                    'accepts': accepts, 'acc_by_type': dict(acc_by_type),
                })

                rate = sum(1 for d in recent_deltas if d > THRESHOLD) / max(len(recent_deltas), 1)
                sps = step / elapsed
                at = ' '.join(f"{k}={v}" for k, v in sorted(acc_by_type.items()))
                print(f"  [{mode_label}][{step:4d}] eval={ea*100:.1f}% best={best_eval*100:.1f}% "
                      f"edges={cur_edges} bidir={bidir} acc={accepts} [{at}] "
                      f"rate={rate*100:.0f}% {elapsed:.0f}s ({sps:.1f}sps)")
                sys.stdout.flush()

                # Plateau
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
    if is_quat:
        final_edges = QuaternaryMask(H, qdata).count_edges()
        final_bidir = QuaternaryMask(H, qdata).count_bidir()
    else:
        final_edges = int(mask.sum())
        final_bidir = count_bidir_bool(mask)

    print(f"\n  [{mode_label}] DONE: best={best_eval*100:.1f}% edges={final_edges} "
          f"bidir={final_bidir} accepts={accepts} {elapsed:.0f}s")

    return {
        'mode': mode_label, 'desc': desc, 'best': best_eval,
        'edges': final_edges, 'bidir': final_bidir, 'accepts': accepts,
        'acc_by_type': dict(acc_by_type),
        'steps': eval_history[-1]['step'] if eval_history else 0,
        'time': elapsed, 'memory_bytes': mem,
        'history': eval_history,
    }


# ------------------------------------------------------------------
# Sanity check: verify edge-set equivalence at step 0
# ------------------------------------------------------------------

def sanity_check(mask):
    """Verify bool mask and quaternary produce identical edge sets."""
    qm = QuaternaryMask.from_bool_mask(mask)
    ref_r, ref_c = np.where(mask)
    ref_set = set(zip(ref_r.tolist(), ref_c.tolist()))
    q_r, q_c = qm.to_directed_edges()
    q_set = set(zip(q_r.tolist(), q_c.tolist()))
    assert ref_set == q_set, f"SANITY FAIL: {len(ref_set ^ q_set)} edges differ!"
    print(f"  Sanity check OK: {len(ref_set)} directed edges match")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

if __name__ == "__main__":
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

    BP_IN = build_sdr(256, IN_DIM, SDR_K, 42)
    bp_out = build_freq_order(OUT_DIM, bigram)

    eval_rng = np.random.RandomState(9999)
    eval_seqs = [ALL_DATA[off:off + SEQ_LEN]
                 for off in [eval_rng.randint(0, len(ALL_DATA) - SEQ_LEN)
                             for _ in range(N_EVAL_SEQS)]]

    # Shared initial state (seeded)
    init_rng = np.random.RandomState(42)
    ref = SelfWiringGraph(max(IN_DIM, 16), hidden=H, projection_scale=1.0)
    init_mask = (init_rng.rand(H, H) < INIT_DENSITY).astype(bool)
    np.fill_diagonal(init_mask, False)
    init_theta = np.full(H, 1.0, np.float32)
    init_channel = init_rng.randint(1, 9, size=H).astype(np.uint8)
    pol_f = np.where(ref.polarity, 1.0, -1.0).astype(np.float32)

    # Sanity check
    sanity_check(init_mask)

    # Run all 3 modes
    results = []
    for label, sched in [('A', SCHEDULE_A), ('B', SCHEDULE_B), ('C', SCHEDULE_C)]:
        r = run_mode(label, sched, init_mask, init_theta, init_channel,
                     pol_f, ALL_DATA, bigram, eval_seqs, bp_out)
        results.append(r)

    # Final report
    print(f"\n{'=' * 70}")
    print(f"  A/B/C: BOOL vs QUATERNARY vs QUATERNARY+UPGRADE")
    print(f"{'=' * 70}")
    for r in results:
        print(f"  {r['mode']} ({r['desc']:30s}): best={r['best']*100:.1f}%  "
              f"edges={r['edges']}  bidir={r['bidir']}  mem={r['memory_bytes']}B  "
              f"accepts={r['accepts']}  {r['time']:.0f}s")
        if r['acc_by_type']:
            print(f"       accepts: {r['acc_by_type']}")

    # Pairwise comparison
    print()
    a, b, c = results[0], results[1], results[2]
    for x, y in [(a, b), (a, c), (b, c)]:
        diff = y['best'] - x['best']
        winner = y['mode'] if diff > 0 else x['mode']
        print(f"  {x['mode']} vs {y['mode']}: delta={diff*100:+.2f}%  winner={winner}")

    mem_ratio = b['memory_bytes'] / a['memory_bytes']
    print(f"\n  Memory: A={a['memory_bytes']}B  B={b['memory_bytes']}B  ratio={mem_ratio:.1%}")
    print(f"{'=' * 70}")

    # Save results
    out_path = os.path.join(BASE_DIR, "data", "ab_quaternary_results.json")
    os.makedirs(os.path.join(BASE_DIR, "data"), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved to {out_path}")

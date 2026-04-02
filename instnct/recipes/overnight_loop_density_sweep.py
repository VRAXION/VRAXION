"""
Overnight exhaustive sweep: loop strategy × density at H=512
=============================================================
4 densities: 0.5%, 1%, 2%, 5%
4 strategies: A (baseline), B (bidir/upgrade), C (tri-3), D (loop-6)
= 16 runs, ~5-6 hours total

Results saved incrementally to data/overnight_loop_density.json
"""
import sys, os, time, random, json, datetime
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
MAX_STEPS = 5000
PHI = (1 + 5**0.5) / 2
IN_DIM = int(round(H / PHI))
OUT_DIM = int(round(H / PHI))
SDR_K = int(round(IN_DIM * 0.20))
SEQ_LEN = 100; N_TRAIN_SEQS = 2; N_EVAL_SEQS = 5
PLATEAU_WINDOW = 10; PLATEAU_THRESH_PCT = 0.3
PLATEAU_STRIKES_MAX = 3; PLATEAU_MIN_STEPS = 800

SCHEDULE = ['add', 'flip', 'theta', 'channel', 'theta', 'channel', 'flip', 'remove']

DENSITIES = [0.005, 0.01, 0.02, 0.05]
STRATEGIES = ['A', 'B', 'C', 'D']

BP_IN = None
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
# Loop mutation helpers
# ------------------------------------------------------------------

def _apply_add_baseline(qm, rng):
    undo = []; qm.mutate_add(rng, undo); return undo

def _apply_add_bidir(qm, rng):
    undo = []
    if rng.random() < 0.5 and len(qm._alive) > 0:
        qm.mutate_upgrade(rng, undo)
    else:
        qm.mutate_add(rng, undo)
    return undo

def _apply_add_tri3(qm, rng):
    undo = []
    if rng.random() < 0.5:
        _add_loop_quat(qm, rng, undo, max_len=3)
        if not undo: qm.mutate_add(rng, undo)
    else:
        qm.mutate_add(rng, undo)
    return undo

def _apply_add_loop6(qm, rng):
    undo = []
    if rng.random() < 0.5:
        _add_loop_quat(qm, rng, undo, max_len=6)
        if not undo: qm.mutate_add(rng, undo)
    else:
        qm.mutate_add(rng, undo)
    return undo

def _add_loop_quat(qm, rng, undo, max_len=6):
    loop_len = rng.randint(2, max(2, max_len))
    nodes = [rng.randint(0, qm.H - 1)]
    for _ in range(loop_len - 1):
        n = rng.randint(0, qm.H - 1)
        if n in nodes: return
        nodes.append(n)
    edges = []
    for i in range(loop_len):
        r, c = nodes[i], nodes[(i + 1) % loop_len]
        if qm.get_pair(r, c) != 0: return
        edges.append((r, c))
    for r, c in edges:
        qm.set_pair(r, c, 1)
        undo.append(('QA', qm._pair_index(r, c), 0))

ADD_FNS = {'A': _apply_add_baseline, 'B': _apply_add_bidir,
           'C': _apply_add_tri3, 'D': _apply_add_loop6}

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
        if np.argmax(logits) == text_bytes[i + 1]: cor += 1
        tot += 1
    return cor / tot if tot else 0

# ------------------------------------------------------------------
# Run single config
# ------------------------------------------------------------------

def run_config(strategy, density, init_theta, init_channel, pol_f,
               ALL_DATA, bigram, eval_seqs, bp_in, bp_out):
    desc = {'A': 'baseline', 'B': 'bidir', 'C': 'tri-3', 'D': 'loop-6'}[strategy]
    tag = f"{strategy}_d{density}"

    # Init mask at this density
    init_rng = np.random.RandomState(42)
    bool_mask = (init_rng.rand(H, H) < density).astype(bool)
    np.fill_diagonal(bool_mask, False)
    qm_init = QuaternaryMask.from_bool_mask(bool_mask)
    qdata = qm_init.data.copy()

    theta = init_theta.copy()
    channel = init_channel.copy()

    bidir0, tri0 = qm_init.loop_levels()
    print(f"\n  [{tag}] {desc} density={density*100:.1f}%: "
          f"{qm_init.count_edges()} edges, {qm_init.count_bidir()} bidir, "
          f"{tri0.sum()} tri_neurons")

    pool = Pool(N_WORKERS, initializer=init_w,
                initargs=(bp_in, bp_out, ALL_DATA, bigram, pol_f, strategy))

    accepts = 0; acc_by_type = {}
    eval_history = []; plateau_strikes = 0
    recent_deltas = deque(maxlen=50)
    t0 = time.time(); best_eval = 0; stall_steps = 0

    try:
        for step in range(1, MAX_STEPS + 1):
            pt = SCHEDULE[(step - 1) % len(SCHEDULE)]
            qm_tmp = QuaternaryMask(H, qdata)
            edges = qm_tmp.count_edges()
            if pt in ('flip', 'remove') and edges < 20: pt = 'add'
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

                eval_history.append({
                    'step': step, 'eval': round(ea * 100, 2),
                    'edges': qm_now.count_edges(), 'bidir': qm_now.count_bidir(),
                    'tri_neurons': int(tri_s.sum()), 'bidir_neurons': int(bidir_s.sum()),
                    'acc_by_type': dict(acc_by_type),
                })

                rate = sum(1 for d in recent_deltas if d > THRESHOLD) / max(len(recent_deltas), 1)
                sps = step / elapsed
                at = ' '.join(f"{k}={v}" for k, v in sorted(acc_by_type.items()))
                print(f"    [{tag}][{step:4d}] eval={ea*100:.1f}% best={best_eval*100:.1f}% "
                      f"edges={qm_now.count_edges()} bidir={qm_now.count_bidir()} "
                      f"tri_n={tri_s.sum()} [{at}] {elapsed:.0f}s ({sps:.1f}sps)")
                sys.stdout.flush()

                if len(eval_history) >= PLATEAU_WINDOW and step >= PLATEAU_MIN_STEPS:
                    window = [e['eval'] for e in eval_history[-PLATEAU_WINDOW:]]
                    spread = max(window) - min(window)
                    if spread < PLATEAU_THRESH_PCT:
                        plateau_strikes += 1
                        if plateau_strikes >= PLATEAU_STRIKES_MAX:
                            print(f"    ** PLATEAU STOP at step {step}")
                            break
                    else:
                        plateau_strikes = 0

                if stall_steps >= 600 and step >= PLATEAU_MIN_STEPS:
                    print(f"    ** STALL STOP at step {step}")
                    break
    finally:
        pool.terminate(); pool.join()

    elapsed = time.time() - t0
    qm_final = QuaternaryMask(H, qdata)
    bidir_f, tri_f = qm_final.loop_levels()

    result = {
        'tag': tag, 'strategy': strategy, 'desc': desc,
        'density': density, 'H': H,
        'best': best_eval, 'best_pct': round(best_eval * 100, 2),
        'edges_init': qm_init.count_edges(), 'edges_final': qm_final.count_edges(),
        'bidir_init': qm_init.count_bidir(), 'bidir_final': qm_final.count_bidir(),
        'tri_neurons_init': int(tri0.sum()), 'tri_neurons_final': int(tri_f.sum()),
        'accepts': accepts, 'acc_by_type': dict(acc_by_type),
        'steps': eval_history[-1]['step'] if eval_history else 0,
        'time_s': round(elapsed, 1),
        'history': eval_history,
    }

    print(f"    [{tag}] DONE: best={best_eval*100:.1f}% edges={qm_final.count_edges()} "
          f"bidir={qm_final.count_bidir()} tri_n={tri_f.sum()} "
          f"accepts={accepts} {elapsed:.0f}s")
    return result


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

if __name__ == "__main__":
    sys.stdout.reconfigure(line_buffering=True)
    start_time = datetime.datetime.now()
    print(f"=== OVERNIGHT LOOP × DENSITY SWEEP ===")
    print(f"Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"H={H}, workers={N_WORKERS}, max_steps={MAX_STEPS}")
    print(f"Densities: {DENSITIES}")
    print(f"Strategies: {STRATEGIES}")
    print(f"Total configs: {len(DENSITIES) * len(STRATEGIES)}")
    print()

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
    BP_IN = bp_in
    bp_out = build_freq_order(OUT_DIM, bigram)

    eval_rng = np.random.RandomState(9999)
    eval_seqs = [ALL_DATA[off:off + SEQ_LEN]
                 for off in [eval_rng.randint(0, len(ALL_DATA) - SEQ_LEN)
                             for _ in range(N_EVAL_SEQS)]]

    # Shared params (same across all configs)
    init_rng = np.random.RandomState(42)
    ref = SelfWiringGraph(max(IN_DIM, 16), hidden=H, projection_scale=1.0)
    init_theta = np.full(H, 1.0, np.float32)
    init_channel = init_rng.randint(1, 9, size=H).astype(np.uint8)
    pol_f = np.where(ref.polarity, 1.0, -1.0).astype(np.float32)

    # Run all configs
    OUT_PATH = os.path.join(BASE_DIR, "data", "overnight_loop_density.json")
    os.makedirs(os.path.join(BASE_DIR, "data"), exist_ok=True)
    all_results = []

    for density in DENSITIES:
        print(f"\n{'#' * 70}")
        print(f"  DENSITY = {density*100:.1f}%")
        print(f"{'#' * 70}")
        for strategy in STRATEGIES:
            r = run_config(strategy, density, init_theta, init_channel, pol_f,
                           ALL_DATA, bigram, eval_seqs, bp_in, bp_out)
            all_results.append(r)
            # Save incrementally
            with open(OUT_PATH, 'w') as f:
                json.dump(all_results, f, indent=2, default=str)

    # Final summary
    end_time = datetime.datetime.now()
    elapsed_total = (end_time - start_time).total_seconds()

    print(f"\n\n{'=' * 70}")
    print(f"  FINAL SUMMARY — {len(all_results)} configs")
    print(f"  {start_time.strftime('%H:%M')} → {end_time.strftime('%H:%M')} ({elapsed_total/3600:.1f}h)")
    print(f"{'=' * 70}")
    print(f"{'Density':>8s} | {'Strategy':>10s} | {'Best%':>6s} | {'Edges':>7s} | {'Bidir':>6s} | {'Tri_n':>6s} | {'Accepts':>7s} | {'Add':>4s} | {'Steps':>5s}")
    print(f"{'-'*8:>8s}-+-{'-'*10:>10s}-+-{'-'*6:>6s}-+-{'-'*7:>7s}-+-{'-'*6:>6s}-+-{'-'*6:>6s}-+-{'-'*7:>7s}-+-{'-'*4:>4s}-+-{'-'*5:>5s}")

    for r in all_results:
        add_acc = r['acc_by_type'].get('add', 0)
        print(f"  {r['density']*100:5.1f}%  | {r['desc']:>10s} | {r['best_pct']:5.1f}% | "
              f"{r['edges_final']:6d}  | {r['bidir_final']:5d} | {r['tri_neurons_final']:5d} | "
              f"{r['accepts']:6d}  | {add_acc:3d} | {r['steps']:5d}")

    # Best per density
    print(f"\n  BEST PER DENSITY:")
    for d in DENSITIES:
        group = [r for r in all_results if r['density'] == d]
        best = max(group, key=lambda x: x['best'])
        others = [r for r in group if r != best]
        diffs = ', '.join(f"{r['desc']}={r['best_pct']:.1f}%" for r in others)
        print(f"    d={d*100:.1f}%: WINNER={best['desc']} ({best['best_pct']:.1f}%)  others: {diffs}")

    # Best per strategy
    print(f"\n  BEST PER STRATEGY:")
    for s in STRATEGIES:
        group = [r for r in all_results if r['strategy'] == s]
        best = max(group, key=lambda x: x['best'])
        desc = group[0]['desc']
        scores = ', '.join(f"d{r['density']*100:.0f}%={r['best_pct']:.1f}%" for r in group)
        print(f"    {desc}: {scores}")

    print(f"\n{'=' * 70}")
    print(f"  Results: {OUT_PATH}")
    print(f"  Total time: {elapsed_total/3600:.1f}h")
    print(f"{'=' * 70}")

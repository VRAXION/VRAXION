"""
Quick test: theta+1 shift (0-15 stored, 1-16 effective).
3 seeds, 1200 steps each. Single config, no A/B needed.

Run: python instnct/recipes/test_theta_plus1.py
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
THRESHOLD = 0.00005; EVAL_EVERY = 100
INIT_DENSITY = 0.05; BUILD_STEPS = 1200
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

def _eval_bigram(qdata, theta_stored, channel, pol_f, seqs):
    """theta_stored is 0-15, effective = theta_stored + 1 = 1-16"""
    qm = QuaternaryMask(H, qdata)
    sc = qm.to_directed_edges()
    _dummy = np.zeros((H, H), dtype=bool)
    # Use theta+1 as effective threshold
    theta_effective = theta_stored + 1.0
    total = 0.0
    for tb in seqs:
        state = np.zeros(H, np.float32); charge = np.zeros(H, np.float32)
        s = 0.0; n = 0
        for i in range(len(tb) - 1):
            inj = np.zeros(H, np.float32); inj[0:IN_DIM] = BP_IN[tb[i]]
            state, charge = SelfWiringGraph.rollout_token(
                inj, mask=_dummy, theta=theta_effective,
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
        nt[idx] = float(rng.randint(0, 15))  # 0-15 stored, +1 at eval time
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
    theta_effective = theta + 1.0
    state = np.zeros(H, np.float32); charge = np.zeros(H, np.float32)
    cor = 0; tot = 0
    for i in range(len(text_bytes) - 1):
        inj = np.zeros(H, np.float32); inj[0:IN_DIM] = bp_in[text_bytes[i]]
        state, charge = SelfWiringGraph.rollout_token(
            inj, mask=_dummy, theta=theta_effective,
            decay=np.float32(0.16), ticks=TICKS, input_duration=INPUT_DURATION,
            state=state, charge=charge, sparse_cache=sc,
            polarity=pol_f, channel=channel)
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

    SEEDS = [42, 123, 777]
    all_results = []

    for seed in SEEDS:
        init_rng = np.random.RandomState(seed)
        ref = SelfWiringGraph(max(IN_DIM, 16), hidden=H, projection_scale=1.0)
        init_mask = (init_rng.rand(H, H) < INIT_DENSITY).astype(bool)
        np.fill_diagonal(init_mask, False)
        qm = QuaternaryMask.from_bool_mask(init_mask)
        qdata = qm.data.copy()
        # Init theta=0 stored (effective=1, same as current init)
        theta = np.full(H, 0.0, np.float32)
        channel = init_rng.randint(1, 9, size=H).astype(np.uint8)
        pol_f = np.where(ref.polarity, 1.0, -1.0).astype(np.float32)

        print(f"\n  Starting seed={seed}")
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

                args = [(qdata.tobytes(), theta.copy(), channel.copy(), pol_f.copy(),
                         seed * 10000 + step * 50 + w, pt) for w in range(N_WORKERS)]
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
                    # Theta distribution (stored 0-15)
                    t_stored = theta.astype(int)
                    n15 = int(np.sum(t_stored == 15))
                    t_mean = float(np.mean(theta))
                    print(f"  [seed={seed}][{step:5d}] eval={ea*100:.1f}% best={best_eval*100:.1f}% "
                          f"edges={QuaternaryMask(H, qdata).count_edges()} "
                          f"theta_mean={t_mean:.1f} theta15={n15} {elapsed:.0f}s")
                    sys.stdout.flush()
        finally:
            pool.terminate(); pool.join()

        elapsed = time.time() - t0
        final_eval = np.mean([eval_accuracy(qdata, theta, channel, pol_f, s, bp_in, bp_out)
                              for s in eval_seqs])
        t_stored = theta.astype(int)
        hist = np.bincount(t_stored, minlength=16)

        result = {
            'seed': seed,
            'final': round(final_eval * 100, 2),
            'best': round(best_eval * 100, 2),
            'theta_hist': hist.tolist(),
            'theta_mean': round(float(np.mean(theta)), 2),
            'n_theta15': int(np.sum(t_stored == 15)),
            'elapsed': round(elapsed, 1),
        }
        all_results.append(result)
        print(f"  DONE: seed={seed} -> best={result['best']}% theta15={result['n_theta15']} "
              f"mean_theta={result['theta_mean']}")
        print(f"  Theta hist: {hist.tolist()}")

    # Summary
    print(f"\n{'='*60}")
    print(f"  THETA+1 SUMMARY (3 seeds)")
    print(f"{'='*60}")
    bests = [r['best'] for r in all_results]
    finals = [r['final'] for r in all_results]
    t15s = [r['n_theta15'] for r in all_results]
    print(f"  best={np.mean(bests):.1f}% +/- {np.std(bests):.1f}%")
    print(f"  final={np.mean(finals):.1f}% +/- {np.std(finals):.1f}%")
    print(f"  theta=15 neurons (almost-off): {np.mean(t15s):.0f} avg")
    print(f"\n  Compare to baseline theta 1-15:")
    print(f"  (from neverfire test) best=17.3% +/- 1.0%  final=17.2% +/- 1.0%")

    out_path = os.path.join(BASE_DIR, "data", "theta_plus1_result.json")
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")

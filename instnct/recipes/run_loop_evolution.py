"""
Loop evolution: canonical schedule + all loop sizes, track acceptance per op.
Ticks=12 (more than default 8 — gives loop-8 one full pass + margin).
H=256 canonical size. Real FineWeb reading.
"""
import sys, os, time, random, json
from collections import defaultdict
import numpy as np
from multiprocessing import Pool
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "model"))
from graph import SelfWiringGraph
from quaternary_mask import QuaternaryMask

H = 256; N_WORKERS = 18; TICKS = 12; INPUT_DURATION = 2
THRESHOLD = 0.00005; EVAL_EVERY = 25
INIT_DENSITY = 0.05; MAX_STEPS = 8000
PHI = (1 + 5**0.5) / 2
IN_DIM = int(round(H / PHI))
OUT_DIM = int(round(H / PHI))
SDR_K = int(round(IN_DIM * 0.20))
SEQ_LEN = 100; N_TRAIN_SEQS = 2; N_EVAL_SEQS = 5

# Full schedule: canonical ops + loop variants (3,4,5,6,8)
# Each loop size gets 1 slot — the eval delta decides which survive.
SCHEDULE = [
    'add', 'enhance', 'reverse', 'mirror',
    'loop3', 'loop5', 'loop8',              # small / medium / large loops
    'flip', 'theta', 'channel', 'channel',
    'remove',
]

BP_IN = None
_bp_out = None; _all_data = None; _bigram = None; _pol = None; _channel_g = None

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

def init_w(bpi, bpo, data, bg, pol, ch):
    global BP_IN, _bp_out, _all_data, _bigram, _pol, _channel_g
    BP_IN = bpi; _bp_out = bpo; _all_data = data; _bigram = bg; _pol = pol; _channel_g = ch

def _eval_bigram(qdata, theta, channel, pol_f, seqs):
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


def worker_eval(args):
    qdata_bytes, theta, channel, pol_f, seed, pt = args
    rng = random.Random(seed); nrng = np.random.RandomState(seed)
    qm = QuaternaryMask(H, np.frombuffer(qdata_bytes, dtype=np.uint8).copy())
    nq = qm.copy(); nt = theta.copy(); nc = channel.copy(); npf = pol_f.copy()

    # --- Structural ops ---
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
        # Polarity flip if same polarity
        if undo:
            idx = undo[-1][1]
            ii, jj = nq._triu_i, nq._triu_j
            i_n, j_n = int(ii[idx]), int(jj[idx])
            if (pol_f[i_n] > 0) == (pol_f[j_n] > 0):
                npf = pol_f.copy()
                npf[j_n] *= -1
    elif pt in ('loop3', 'loop4', 'loop5', 'loop6', 'loop7', 'loop8'):
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
    # --- Parameter ops ---
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

    # Init
    init_rng = np.random.RandomState(42)
    ref = SelfWiringGraph(max(IN_DIM, 16), hidden=H, projection_scale=1.0)
    init_mask = (init_rng.rand(H, H) < INIT_DENSITY).astype(bool)
    np.fill_diagonal(init_mask, False)
    qm = QuaternaryMask.from_bool_mask(init_mask)
    qdata = qm.data.copy()
    theta = np.full(H, 1.0, np.float32)
    channel = init_rng.randint(1, 9, size=H).astype(np.uint8)
    pol_f = np.where(ref.polarity, 1.0, -1.0).astype(np.float32)

    bidir0, tri0 = qm.loop_levels()
    print(f"H={H}, ticks={TICKS}, schedule={SCHEDULE}")
    print(f"Init: {qm.count_edges()} edges, {qm.count_bidir()} bidir, "
          f"{tri0.sum()} tri neurons")
    print()

    pool = Pool(N_WORKERS, initializer=init_w,
                initargs=(bp_in, bp_out, ALL_DATA, bigram, pol_f, channel))

    acc_by_type = defaultdict(int)
    proposed_by_type = defaultdict(int)
    eval_history = []
    t0 = time.time()
    best_eval = 0

    try:
        for step in range(1, MAX_STEPS + 1):
            pt = SCHEDULE[(step - 1) % len(SCHEDULE)]
            edges = QuaternaryMask(H, qdata).count_edges()
            if pt in ('remove', 'reverse', 'mirror') and edges < 50: pt = 'add'
            if edges == 0: pt = 'add'

            proposed_by_type[pt] += 1
            args = [(qdata.tobytes(), theta.copy(), channel.copy(), pol_f.copy(),
                     1000 + step * 50 + w, pt) for w in range(N_WORKERS)]
            results = pool.map(worker_eval, args)
            best_r = max(results, key=lambda x: x['delta'])

            if best_r['delta'] > THRESHOLD:
                if best_r['new_qdata'] is not None:
                    qdata[:] = np.frombuffer(best_r['new_qdata'], dtype=np.uint8)
                if best_r['new_theta'] is not None:
                    theta[:] = best_r['new_theta']
                if best_r['new_channel'] is not None:
                    channel[:] = best_r['new_channel']
                if best_r['new_pol'] is not None:
                    pol_f[:] = best_r['new_pol']
                acc_by_type[best_r['type']] += 1

            if step % EVAL_EVERY == 0:
                elapsed = time.time() - t0
                ea = np.mean([eval_accuracy(qdata, theta, channel, pol_f, s, bp_in, bp_out) for s in eval_seqs])
                if ea > best_eval: best_eval = ea

                qm_now = QuaternaryMask(H, qdata)
                bidir_s, tri_s = qm_now.loop_levels()
                cur_edges = qm_now.count_edges()

                # Per-op accept rates
                rates = []
                for op in SCHEDULE:
                    if op not in rates:
                        proposed = proposed_by_type[op]
                        accepted = acc_by_type[op]
                        if proposed > 0:
                            rates.append(f"{op}={accepted}/{proposed}({accepted/proposed*100:.0f}%)")
                rate_str = ' '.join(dict.fromkeys(rates))  # dedup

                eval_history.append({
                    'step': step, 'eval': round(ea * 100, 2),
                    'best': round(best_eval * 100, 2),
                    'edges': cur_edges, 'bidir': qm_now.count_bidir(),
                    'tri_n': int(tri_s.sum()),
                })

                sps = step / elapsed
                print(f"[{step:5d}] eval={ea*100:.1f}% best={best_eval*100:.1f}% "
                      f"edges={cur_edges} bidir={qm_now.count_bidir()} tri={tri_s.sum()} "
                      f"| {rate_str} "
                      f"| {elapsed:.0f}s ({sps:.1f}sps)")
                sys.stdout.flush()

    except KeyboardInterrupt:
        print("\n  Interrupted by user")
    finally:
        pool.terminate(); pool.join()

    elapsed = time.time() - t0
    qm_final = QuaternaryMask(H, qdata)
    bidir_f, tri_f = qm_final.loop_levels()

    print(f"\n{'=' * 70}")
    print(f"  FINAL: best={best_eval*100:.1f}% edges={qm_final.count_edges()} "
          f"bidir={qm_final.count_bidir()} tri={tri_f.sum()} {elapsed:.0f}s")
    print(f"\n  OP ACCEPTANCE RATES:")
    for op in sorted(set(SCHEDULE)):
        p = proposed_by_type[op]; a = acc_by_type[op]
        print(f"    {op:10s}: {a:4d}/{p:4d} = {a/p*100 if p else 0:5.1f}%")
    print(f"{'=' * 70}")

    # Save
    out = {
        'config': {'H': H, 'ticks': TICKS, 'schedule': SCHEDULE, 'max_steps': MAX_STEPS},
        'result': {'best': best_eval, 'edges': qm_final.count_edges(),
                   'bidir': qm_final.count_bidir(), 'tri_neurons': int(tri_f.sum())},
        'acceptance': {op: {'accepted': acc_by_type[op], 'proposed': proposed_by_type[op]}
                       for op in sorted(set(SCHEDULE))},
        'history': eval_history,
    }
    out_path = os.path.join(BASE_DIR, "data", "loop_evolution_result.json")
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"  Saved to {out_path}")

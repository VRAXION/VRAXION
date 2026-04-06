"""
Instrumented theta mutation analysis.
Logs every theta proposal: old value, new value, accepted/rejected, delta.
Short run (600 steps) to collect statistics.

Run: python instnct/recipes/analyze_theta_acceptance.py
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
INIT_DENSITY = 0.05; BUILD_STEPS = 600
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

    theta_info = None

    if pt == 'add':
        undo = []; nq.mutate_add(rng, undo)
        if not undo: return {'delta': -1e9, 'type': pt, 'theta_info': None}
    elif pt == 'enhance':
        bm = qm.to_bool_mask()
        in_deg = bm.sum(axis=0).astype(np.float64) + 1.0
        top = np.argsort(in_deg)[::-1][:H // 4]
        c = int(top[rng.randint(0, len(top) - 1)]); r = rng.randint(0, H - 1)
        if r == c or nq.get_pair(r, c) != 0: return {'delta': -1e9, 'type': pt, 'theta_info': None}
        nq.set_pair(r, c, 1)
    elif pt == 'reverse':
        undo = []; nq.mutate_flip(rng, undo)
        if not undo: return {'delta': -1e9, 'type': pt, 'theta_info': None}
    elif pt == 'mirror':
        undo = []; nq.mutate_upgrade(rng, undo)
        if not undo: return {'delta': -1e9, 'type': pt, 'theta_info': None}
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
            if n in nodes: return {'delta': -1e9, 'type': pt, 'theta_info': None}
            nodes.append(n)
        for k in range(loop_len):
            r, c = nodes[k], nodes[(k + 1) % loop_len]
            if nq.get_pair(r, c) != 0: return {'delta': -1e9, 'type': pt, 'theta_info': None}
        for k in range(loop_len):
            r, c = nodes[k], nodes[(k + 1) % loop_len]
            nq.set_pair(r, c, 1)
    elif pt == 'remove':
        undo = []; nq.mutate_remove(rng, undo)
        if not undo: return {'delta': -1e9, 'type': pt, 'theta_info': None}
    elif pt == 'flip':
        idx = rng.randint(0, H - 1)
        npf = pol_f.copy(); npf[idx] *= -1
    elif pt == 'theta':
        idx = rng.randint(0, H - 1); nt = theta.copy()
        old_val = float(theta[idx])
        new_val = float(rng.randint(1, 15))
        nt[idx] = new_val
        theta_info = {'idx': idx, 'old': old_val, 'new': new_val,
                      'step': abs(new_val - old_val)}
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
        'theta_info': theta_info,
        'new_qdata': nq.data.tobytes() if new > old else None,
        'new_theta': nt if new > old else None,
        'new_channel': nc if new > old else None,
        'new_pol': npf if new > old else None,
    }


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
    channel = init_rng.randint(1, 9, size=H).astype(np.uint8)
    pol_f = np.where(ref.polarity, 1.0, -1.0).astype(np.float32)

    pool = Pool(N_WORKERS, initializer=init_w,
                initargs=(bp_in, bp_out, ALL_DATA, bigram, pol_f))

    # Collect all theta mutation data
    all_theta_proposals = []  # (old, new, step, delta, accepted)
    accepted_steps = []
    rejected_steps = []
    acc_by_type = defaultdict(int)
    t0 = time.time()

    for step in range(1, BUILD_STEPS + 1):
        pt = SCHEDULE[(step - 1) % len(SCHEDULE)]
        edges = QuaternaryMask(H, qdata).count_edges()
        if pt in ('remove', 'reverse') and edges < 50: pt = 'add'

        args = [(qdata.tobytes(), theta.copy(), channel.copy(), pol_f.copy(),
                 1000 + step * 50 + w, pt) for w in range(N_WORKERS)]
        results = pool.map(worker_eval, args)
        best_r = max(results, key=lambda x: x['delta'])

        # Log theta mutations from ALL workers
        if pt == 'theta':
            for r in results:
                if r['theta_info'] is not None:
                    info = r['theta_info']
                    accepted = (r is best_r and r['delta'] > THRESHOLD)
                    all_theta_proposals.append({
                        'old': info['old'], 'new': info['new'],
                        'step_size': info['step'], 'delta': r['delta'],
                        'accepted': accepted
                    })
                    if accepted:
                        accepted_steps.append(info['step'])
                    else:
                        rejected_steps.append(info['step'])

        if best_r['delta'] > THRESHOLD:
            if best_r['new_qdata'] is not None:
                qdata[:] = np.frombuffer(best_r['new_qdata'], dtype=np.uint8)
            if best_r['new_theta'] is not None: theta[:] = best_r['new_theta']
            if best_r['new_channel'] is not None: channel[:] = best_r['new_channel']
            if best_r['new_pol'] is not None: pol_f[:] = best_r['new_pol']
            acc_by_type[best_r['type']] += 1

        if step % EVAL_EVERY == 0:
            elapsed = time.time() - t0
            print(f"  [{step:5d}] {elapsed:.0f}s  theta_proposals={len(all_theta_proposals)}")

    pool.terminate(); pool.join()

    # =========================================================================
    # ANALYSIS
    # =========================================================================
    print(f"\n{'='*60}")
    print(f"  THETA MUTATION ANALYSIS ({len(all_theta_proposals)} proposals)")
    print(f"{'='*60}")

    # Acceptance rate by step size
    print(f"\n  Step size acceptance rates:")
    print(f"  {'Step':>6s} {'Proposed':>10s} {'Accepted':>10s} {'Rate':>8s}")
    for step_size in range(1, 15):
        proposed = sum(1 for p in all_theta_proposals if int(p['step_size']) == step_size)
        accepted = sum(1 for p in all_theta_proposals
                       if int(p['step_size']) == step_size and p['accepted'])
        rate = accepted / proposed * 100 if proposed > 0 else 0
        bar = '#' * int(rate)
        print(f"  {step_size:>6d} {proposed:>10d} {accepted:>10d} {rate:>7.1f}% {bar}")

    # Acceptance rate by direction (up vs down)
    print(f"\n  Direction analysis:")
    up = [p for p in all_theta_proposals if p['new'] > p['old']]
    down = [p for p in all_theta_proposals if p['new'] < p['old']]
    same = [p for p in all_theta_proposals if p['new'] == p['old']]
    up_acc = sum(1 for p in up if p['accepted'])
    down_acc = sum(1 for p in down if p['accepted'])
    print(f"    UP   (theta increases): {len(up):>6d} proposed, {up_acc:>4d} accepted ({up_acc/max(len(up),1)*100:.1f}%)")
    print(f"    DOWN (theta decreases): {len(down):>6d} proposed, {down_acc:>4d} accepted ({down_acc/max(len(down),1)*100:.1f}%)")
    print(f"    SAME (no change):       {len(same):>6d}")

    # Average accepted vs rejected step size
    if accepted_steps:
        print(f"\n  Average step sizes:")
        print(f"    Accepted: {np.mean(accepted_steps):.1f} (median: {np.median(accepted_steps):.0f})")
    if rejected_steps:
        print(f"    Rejected: {np.mean(rejected_steps):.1f} (median: {np.median(rejected_steps):.0f})")

    # Final theta distribution
    print(f"\n  Final theta distribution:")
    hist, _ = np.histogram(theta, bins=range(0, 17))
    for val in range(1, 16):
        bar = '#' * (hist[val] * 2)
        print(f"    theta={val:>2d}: {hist[val]:>4d} neurons {bar}")

    # Transition matrix (which old→new transitions get accepted)
    print(f"\n  Top 10 accepted transitions (old -> new):")
    transitions = defaultdict(int)
    for p in all_theta_proposals:
        if p['accepted']:
            key = (int(p['old']), int(p['new']))
            transitions[key] += 1
    for (old, new), count in sorted(transitions.items(), key=lambda x: -x[1])[:10]:
        print(f"    {old:>2d} -> {new:>2d}: {count:>3d} times (step={abs(new-old)})")

    # Save
    out_path = os.path.join(BASE_DIR, "data", "theta_acceptance_analysis.json")
    with open(out_path, 'w') as f:
        json.dump({
            'total_proposals': len(all_theta_proposals),
            'accepted_step_mean': float(np.mean(accepted_steps)) if accepted_steps else 0,
            'rejected_step_mean': float(np.mean(rejected_steps)) if rejected_steps else 0,
            'final_theta_hist': hist.tolist(),
            'top_transitions': dict(sorted(transitions.items(), key=lambda x: -x[1])[:20]),
        }, f, indent=2, default=str)
    print(f"\nSaved to {out_path}")

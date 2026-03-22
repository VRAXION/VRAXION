"""
INSTNCT Adaptive Schedule — learnable mutation weights
========================================================
Each mutation type gets an int4 weight (0-15).
Schedule is built from weights: type appears weight times per cycle.

Mutation types: add, flip, mag_resample, ret (remove is AUTO)
Schedule mutation: pick random type, set weight to random(0-15).

A/B: fixed schedule vs adaptive, 3000 steps, word pairs, LL eval, w=2.
"""
import sys, os, time, random
import numpy as np
from multiprocessing import Pool

from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "model"))
from graph import SelfWiringGraph

_bp = None; _all_data = None; _seq_len = 120; _n_train = 2
_output_projection_f = None; _inj_table = None

def init_w(b, d, sl, nt, wof, it):
    global _bp, _all_data, _seq_len, _n_train, _output_projection_f, _inj_table
    _bp, _all_data, _seq_len, _n_train = b, d, sl, nt
    _output_projection_f, _inj_table = wof, it

def make_bp(io_dim, seed=12345):
    rng = np.random.RandomState(seed)
    p = rng.randn(256, io_dim).astype(np.float32)
    p /= np.linalg.norm(p, axis=1, keepdims=True)
    return p

def _eval_loglik(msign, mmag, ret_vec, H, seqs):
    rs, cs = np.where(mmag > 0)
    s = msign[rs, cs].astype(np.float32) * 2 - 1
    sp_vals = s * mmag[rs, cs].astype(np.float32) / 128.0
    pat_norm = _bp / (np.linalg.norm(_bp, axis=1, keepdims=True) + 1e-8)
    total = 0.0
    for text_bytes in seqs:
        state = np.zeros(H, dtype=np.float32)
        charge = np.zeros(H, dtype=np.float32)
        seq_score = 0.0; n = 0
        for i in range(len(text_bytes)-1):
            act = state.copy()
            inj = _inj_table[text_bytes[i]].astype(np.float32) / 128.0
            if i > 0:
                inj = inj + _inj_table[text_bytes[i-1]].astype(np.float32) / 128.0 * 0.5
            for t in range(8):
                if t < 2: act = act + inj
                raw = np.zeros(H, dtype=np.float32)
                if len(rs): np.add.at(raw, cs, act[rs] * sp_vals)
                charge += raw
                charge *= ret_vec
                act = np.maximum(charge, 0.0)
                charge = np.maximum(charge, 0.0)
            state = act.copy()
            out = charge @ _output_projection_f
            out_n = out / (np.linalg.norm(out) + 1e-8)
            sims = out_n @ pat_norm.T
            e = np.exp(sims - sims.max())
            probs = e / e.sum()
            seq_score += np.log(probs[text_bytes[i+1]] + 1e-10)
            n += 1
        total += seq_score / n if n else 0
    return total / len(seqs)

def worker_eval(args):
    msign_flat, mmag_flat, ret_int4, H, seed, ptype = args
    rng = random.Random(seed)
    np_rng = np.random.RandomState(seed)
    msign = msign_flat.reshape(H, H); mmag = mmag_flat.reshape(H, H)
    new_s = msign.copy(); new_m = mmag.copy()
    new_ret = ret_int4.copy()
    ret_vec = (1.0 - ret_int4.astype(np.float32) * 0.01)

    if ptype == 'add':
        alive_rs, alive_cs = np.where(mmag > 0)
        if len(alive_rs) > 0 and rng.random() < 0.5:
            if rng.random() < 0.5:
                r = alive_rs[rng.randint(0, len(alive_rs)-1)]
                c = rng.randint(0, H-1)
            else:
                r = rng.randint(0, H-1)
                c = alive_cs[rng.randint(0, len(alive_cs)-1)]
        else:
            r = rng.randint(0, H-1); c = rng.randint(0, H-1)
        if r == c or mmag[r, c] > 0:
            return {'delta': -1e9, 'type': ptype}
        new_s[r, c] = rng.random() < 0.5
        new_m[r, c] = rng.randint(1, 255)
    elif ptype == 'flip':
        rs, cs = np.where(mmag > 0)
        if len(rs) == 0: return {'delta': -1e9, 'type': ptype}
        idx = rng.randint(0, len(rs)-1)
        new_s[rs[idx], cs[idx]] = not msign[rs[idx], cs[idx]]
    elif ptype == 'mag':
        rs, cs = np.where(mmag > 0)
        if len(rs) == 0: return {'delta': -1e9, 'type': ptype}
        idx = rng.randint(0, len(rs)-1)
        new_m[rs[idx], cs[idx]] = rng.randint(1, 255)
    elif ptype == 'ret':
        idx = rng.randint(0, H-1)
        new_ret[idx] = rng.randint(0, 15)

    seqs = []
    for _ in range(2):
        off = np_rng.randint(0, len(_all_data) - 120)
        seqs.append(_all_data[off:off+120])

    new_ret_vec = (1.0 - new_ret.astype(np.float32) * 0.01)
    old = _eval_loglik(msign, mmag, ret_vec, H, seqs)
    new = _eval_loglik(new_s, new_m, new_ret_vec, H, seqs)
    improved = new > old
    return {'delta': new - old, 'type': ptype,
            'new_s': new_s.flatten() if improved else None,
            'new_m': new_m.flatten() if improved else None,
            'new_ret': new_ret if improved else None}


def compute_edge_alignment(msign, mmag, ret_int4, H, wof, bp, bigram, inj_table, eval_seqs):
    """Score each edge by alignment with target. 1 forward pass."""
    rs, cs = np.where(mmag > 0)
    n = len(rs)
    if n == 0: return np.array([]), rs, cs
    s = msign[rs, cs].astype(np.float32) * 2 - 1
    sp_vals = s * mmag[rs, cs].astype(np.float32) / 128.0
    target_vecs = bigram @ bp
    target_norms = target_vecs / (np.linalg.norm(target_vecs, axis=1, keepdims=True) + 1e-8)
    w_align = wof @ target_norms.T
    ret_vec = 1.0 - ret_int4.astype(np.float32) * 0.01
    edge_scores = np.zeros(n, dtype=np.float64)
    n_pos = 0
    for text_bytes in eval_seqs:
        state = np.zeros(H, dtype=np.float32)
        charge = np.zeros(H, dtype=np.float32)
        for i in range(len(text_bytes)-1):
            act = state.copy()
            inj = inj_table[text_bytes[i]].astype(np.float32) / 128.0
            if i > 0: inj = inj + inj_table[text_bytes[i-1]].astype(np.float32) / 128.0 * 0.5
            byte_align = w_align[:, text_bytes[i]]
            for t in range(8):
                if t < 2: act = act + inj
                edge_act = act[rs] * sp_vals
                edge_scores += edge_act * byte_align[cs]
                raw = np.zeros(H, dtype=np.float32)
                if n: np.add.at(raw, cs, edge_act)
                charge += raw; charge *= ret_vec
                act = np.maximum(charge, 0.0); charge = np.maximum(charge, 0.0)
            state = act.copy()
            n_pos += 1
    if n_pos: edge_scores /= n_pos
    return edge_scores, rs, cs


def find_knee(scores):
    """Find the knee: biggest gap in sorted scores. Returns cutoff index."""
    if len(scores) < 3: return 0
    sorted_s = np.sort(scores)
    gaps = np.diff(sorted_s)
    # Knee = biggest gap in bottom half (we want to cut the weak tail)
    half = len(gaps) // 2
    if half == 0: return 0
    knee_idx = int(np.argmax(gaps[:half]))
    return knee_idx + 1  # number of edges below the knee


def soft_decay(msign, mmag, ret_int4, H, wof, bp, bigram, inj_table, eval_seqs):
    """Remove only HARMFUL edges (score < 0)."""
    scores, e_rs, e_cs = compute_edge_alignment(
        msign, mmag, ret_int4, H, wof, bp, bigram, inj_table, eval_seqs)
    if len(scores) == 0: return msign, mmag, 0
    harmful = scores < 0
    n_remove = int(harmful.sum())
    if n_remove > 0:
        mmag[e_rs[harmful], e_cs[harmful]] = 0
        msign[e_rs[harmful], e_cs[harmful]] = False
    return msign, mmag, n_remove


def hard_prune(msign, mmag, ret_int4, H, wof, bp, bigram, inj_table, eval_seqs):
    """Find knee in score distribution, kill everything below."""
    scores, e_rs, e_cs = compute_edge_alignment(
        msign, mmag, ret_int4, H, wof, bp, bigram, inj_table, eval_seqs)
    if len(scores) < 5: return msign, mmag, 0, 0
    # Remove harmful first
    harmful = scores < 0
    n_harmful = int(harmful.sum())
    mmag[e_rs[harmful], e_cs[harmful]] = 0
    msign[e_rs[harmful], e_cs[harmful]] = False
    # Find knee in remaining
    keep = ~harmful
    if keep.sum() < 3: return msign, mmag, n_harmful, 0
    kept_scores = scores[keep]
    kept_rs = e_rs[keep]; kept_cs = e_cs[keep]
    n_below_knee = find_knee(kept_scores)
    # Kill below knee
    if n_below_knee > 0:
        weak_idx = np.argsort(kept_scores)[:n_below_knee]
        for idx in weak_idx:
            mmag[kept_rs[idx], kept_cs[idx]] = 0
            msign[kept_rs[idx], kept_cs[idx]] = False
    return msign, mmag, n_harmful, n_below_knee


FACTS = [
    ("the sky is", "blue"), ("the sun is", "yellow and bright"),
    ("grass color is", "green"), ("fire is very", "hot and red"),
    ("snow is always", "white and cold"), ("coal color is", "black"),
    ("gold looks very", "shiny"), ("ice feels very", "cold"),
    ("milk color is", "white"), ("a cat says", "meow"),
    ("a dog says", "bark bark"), ("a cow says", "moo"),
    ("a bee says", "buzz"), ("the sea is", "salt water"),
    ("rain makes things", "wet"), ("iron feels very", "hard"),
    ("silk feels very", "soft"), ("a red flower is", "a rose"),
    ("leaves are always", "green"), ("the moon is", "round and bright"),
    ("stars look very", "bright at night"), ("a fish can", "swim fast"),
    ("a bird can", "fly high"), ("a frog can", "jump far"),
    ("a bear is", "big and strong"), ("an ant is", "tiny and small"),
    ("paris is in", "france"), ("tokyo is in", "japan"),
    ("london is in", "england"), ("rome is in", "italy"),
]

def eval_facts(msign, mmag, ret_int4, H, wof, bp, it):
    pat_norm = bp / (np.linalg.norm(bp, axis=1, keepdims=True) + 1e-8)
    rs, cs = np.where(mmag > 0)
    if len(rs) == 0: return 0.0
    s = msign[rs, cs].astype(np.float32) * 2 - 1
    sp = s * mmag[rs, cs].astype(np.float32) / 128.0
    ret_vec = (1.0 - ret_int4.astype(np.float32) * 0.01)
    tc = 0; ta = 0
    for key, val in FACTS:
        text = (key + '=' + val + '\n').encode('ascii')
        tb = np.frombuffer(text, dtype=np.uint8)
        st = np.zeros(H, dtype=np.float32); ch = np.zeros(H, dtype=np.float32)
        eq = len(key)
        for i in range(len(tb)-1):
            act = st.copy()
            inj = it[tb[i]].astype(np.float32) / 128.0
            if i > 0: inj = inj + it[tb[i-1]].astype(np.float32) / 128.0 * 0.5
            for t in range(8):
                if t < 2: act = act + inj
                raw = np.zeros(H, dtype=np.float32)
                if len(rs): np.add.at(raw, cs, act[rs] * sp)
                ch += raw; ch *= ret_vec
                act = np.maximum(ch, 0.0); ch = np.maximum(ch, 0.0)
            st = act.copy()
            if i >= eq:
                out = ch @ wof
                out_n = out / (np.linalg.norm(out) + 1e-8)
                sims = out_n @ pat_norm.T
                pb = int(np.argmax(sims))
                actual = int(tb[i+1])
                if actual != ord('\n'):
                    ta += 1
                    if pb == actual: tc += 1
    return tc/ta if ta else 0


MUT_TYPES = ['add', 'flip', 'mag', 'ret']

def build_schedule(weights):
    """Build schedule from int4 weights. Each type appears weight times."""
    schedule = []
    for i, t in enumerate(MUT_TYPES):
        schedule.extend([t] * int(weights[i]))
    if not schedule:
        schedule = ['add']  # fallback
    return schedule


def run_config(name, adaptive, H, bp, inj_table, wof, ALL_DATA, bigram, n_steps, n_workers):
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    sys.stdout.flush()

    msign = np.zeros((H, H), dtype=np.bool_)
    mmag = np.zeros((H, H), dtype=np.uint8)
    ret_int4 = np.full(H, 15, dtype=np.uint8)  # default retention

    # Schedule weights: int4 per type
    if adaptive:
        sched_weights = np.array([4, 1, 1, 1], dtype=np.uint8)  # start: 4a/1f/1m/1ret
    else:
        sched_weights = np.array([4, 1, 1, 0], dtype=np.uint8)  # fixed: 4a/1f/1m/0ret

    schedule = build_schedule(sched_weights)
    accepts = {t: 0 for t in MUT_TYPES}
    accepts['sched'] = 0
    t0 = time.time()

    pool = Pool(n_workers, initializer=init_w,
                initargs=(bp, ALL_DATA, 120, 2, wof, inj_table))
    try:
        for step in range(1, n_steps+1):
            # Schedule mutation (adaptive only): every 50 steps, try changing a weight
            if adaptive and step % 50 == 0:
                # Try mutating one schedule weight
                test_weights = sched_weights.copy()
                idx = random.randint(0, len(MUT_TYPES)-1)
                test_weights[idx] = random.randint(0, 15)
                # Can't eval schedule directly, just accept and see
                # (the LL eval over next 50 steps will show if it helped)
                sched_weights = test_weights
                schedule = build_schedule(sched_weights)
                accepts['sched'] += 1

            ptype = schedule[(step-1) % len(schedule)]
            if ptype in ('flip', 'mag') and (mmag > 0).sum() == 0:
                ptype = 'add'

            args = [(msign.flatten(), mmag.flatten(), ret_int4.copy(), H,
                     10000+step*50+w, ptype) for w in range(n_workers)]
            results = pool.map(worker_eval, args)
            best = max(results, key=lambda x: x['delta'])
            if best['delta'] > 0.00005:
                if best.get('new_s') is not None:
                    msign = best['new_s'].reshape(H, H)
                    mmag = best['new_m'].reshape(H, H)
                if best.get('new_ret') is not None:
                    ret_int4 = best['new_ret']
                accepts[best['type']] += 1

            # AUTO DECAY: every 100 steps, kill bottom 10%
            if step % 100 == 0 and (mmag > 0).sum() > 10:
                eval_rng2 = np.random.RandomState(step)
                decay_seqs = [ALL_DATA[off:off+120] for off in
                              [eval_rng2.randint(0, len(ALL_DATA)-120) for _ in range(3)]]
                msign, mmag, n_decayed = soft_decay(
                    msign, mmag, ret_int4, H, wof, bp, bigram, inj_table, decay_seqs)
                accepts['decay'] = accepts.get('decay', 0) + n_decayed

            # HARD PRUNE: every 5000 steps, knee detection
            if step % 1000 == 0 and (mmag > 0).sum() > 10:
                eval_rng2 = np.random.RandomState(step + 99999)
                prune_seqs = [ALL_DATA[off:off+120] for off in
                              [eval_rng2.randint(0, len(ALL_DATA)-120) for _ in range(5)]]
                msign, mmag, n_harm, n_knee = hard_prune(
                    msign, mmag, ret_int4, H, wof, bp, bigram, inj_table, prune_seqs)
                accepts['hard_prune'] = accepts.get('hard_prune', 0) + 1
                edges_after = int((mmag > 0).sum())
                print(f"  [HARD PRUNE @{step}] removed {n_harm} harmful + {n_knee} below knee = {n_harm+n_knee} total, {edges_after} remain")
                sys.stdout.flush()

            if step % 500 == 0:
                elapsed = time.time() - t0
                edges = int((mmag > 0).sum())
                overall = eval_facts(msign, mmag, ret_int4, H, wof, bp, inj_table)
                decay_total = accepts.get('decay', 0)
                w_str = '/'.join(f"{MUT_TYPES[i][0]}={sched_weights[i]}" for i in range(len(MUT_TYPES)))
                acc_str = '/'.join(f"{t[0]}={accepts[t]}" for t in MUT_TYPES)
                print(f"  [{step:4d}] answer={overall*100:.1f}% edges={edges} decayed={decay_total} "
                      f"weights=[{w_str}] accepts=[{acc_str}] {elapsed:.0f}s")
                sys.stdout.flush()
    finally:
        pool.terminate(); pool.join()

    elapsed = time.time() - t0
    edges = int((mmag > 0).sum())
    overall = eval_facts(msign, mmag, ret_int4, H, wof, bp, inj_table)
    return {'name': name, 'acc': overall, 'edges': edges, 'time': elapsed,
            'weights': sched_weights.copy(), 'accepts': dict(accepts)}


if __name__ == "__main__":
    IO = 256; NV = 4; H = IO * NV
    N_WORKERS = 18; BUDGET = 3000

    lines = [f"{k}={v}\n" for k, v in FACTS]
    random.seed(42)
    corpus_text = ''
    while len(corpus_text) < 50_000:
        random.shuffle(lines)
        corpus_text += ''.join(lines)
    ALL_DATA = np.frombuffer(corpus_text.encode('ascii'), dtype=np.uint8).copy()

    SelfWiringGraph.NV_RATIO = 4; bp = make_bp(IO)
    random.seed(42); np.random.seed(42)
    ref = SelfWiringGraph(IO)
    inp = ref.input_projection / ref.INJ_SCALE * 1.0
    outp = ref.output_projection / ref.INJ_SCALE * 1.0
    inj_table = np.clip(bp @ inp * 128, -128, 127).astype(np.int8)
    woi = np.clip(outp * 128, -128, 127).astype(np.int8)
    wof = woi.astype(np.float32) / 128.0

    # Bigram for alignment scoring
    bigram = np.zeros((256, 256), dtype=np.float64)
    for i in range(len(ALL_DATA) - 1):
        bigram[ALL_DATA[i], ALL_DATA[i+1]] += 1
    row_sums = bigram.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    bigram = (bigram / row_sums).astype(np.float32)

    print(f"ADAPTIVE SCHEDULE + AUTO DECAY/PRUNE")
    print(f"Types: {MUT_TYPES} (remove is AUTO)")
    print(f"Soft decay: bottom 10% every 100 steps")
    print(f"Hard prune: knee detection every 5000 steps")
    print(f"A: fixed [4a/1f/1m/0ret] + auto decay/prune")
    print(f"B: adaptive (starts 4a/1f/1m/1ret) + auto decay/prune")
    print(f"{BUDGET} steps, LL eval, w=2, PNR, threshold=0.00005")

    results = []
    results.append(run_config("A: fixed + auto prune", False, H, bp, inj_table, wof,
                              ALL_DATA, bigram, BUDGET, N_WORKERS))
    results.append(run_config("B: adaptive + auto prune", True, H, bp, inj_table, wof,
                              ALL_DATA, bigram, BUDGET, N_WORKERS))

    print(f"\n{'='*60}")
    print(f"  ADAPTIVE + AUTO PRUNE RESULTS")
    print(f"{'='*60}")
    for r in results:
        w_str = '/'.join(f"{MUT_TYPES[i][0]}={r['weights'][i]}" for i in range(len(MUT_TYPES)))
        print(f"  {r['name']:<25} acc={r['acc']*100:.2f}% edges={r['edges']} "
              f"final_weights=[{w_str}] {r['time']:.0f}s")
    delta = (results[1]['acc'] - results[0]['acc']) * 100
    print(f"\n  Delta: {delta:+.2f}%")
    print(f"{'='*60}")
    sys.stdout.flush()

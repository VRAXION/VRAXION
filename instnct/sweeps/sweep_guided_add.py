"""
INSTNCT Guided Add Sweep
==========================
A: random add (baseline)
B: guided add (score potential edges, pick from top candidates)

The guided add precomputes: sensitivity[src] * alignment[tgt]
to find the best (src, tgt) pairs for new edges.
Score computation: 0.003s. Recomputed every 100 steps.

w=2 superposition, LL eval, threshold=0.00005, 2000 steps.
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
_guided_src = None; _guided_tgt = None  # top candidate lists

def init_w(b, d, sl, nt, wof, it, gs, gt):
    global _bp, _all_data, _seq_len, _n_train, _output_projection_f, _inj_table
    global _guided_src, _guided_tgt
    _bp, _all_data, _seq_len, _n_train = b, d, sl, nt
    _output_projection_f, _inj_table = wof, it
    _guided_src, _guided_tgt = gs, gt

def make_bp(io_dim, seed=12345):
    rng = np.random.RandomState(seed)
    p = rng.randn(256, io_dim).astype(np.float32)
    p /= np.linalg.norm(p, axis=1, keepdims=True)
    return p

def _eval_loglik(msign, mmag, H, seqs):
    rs, cs = np.where(mmag > 0)
    s = msign[rs, cs].astype(np.float32) * 2 - 1
    sp_vals = s * mmag[rs, cs].astype(np.float32) / 128.0
    pat_norm = _bp / (np.linalg.norm(_bp, axis=1, keepdims=True) + 1e-8)
    ret = 217.0 / 256.0
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
                charge += raw; charge *= ret
                act = np.maximum(charge, 0.0); charge = np.maximum(charge, 0.0)
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
    msign_flat, mmag_flat, H, seed, ptype, use_guided = args
    rng = random.Random(seed)
    np_rng = np.random.RandomState(seed)
    msign = msign_flat.reshape(H, H); mmag = mmag_flat.reshape(H, H)
    new_s = msign.copy(); new_m = mmag.copy()

    if ptype == 'add':
        if use_guided and _guided_src is not None and len(_guided_src) > 0:
            # GUIDED: pick from precomputed good (src, tgt) pairs
            idx = rng.randint(0, len(_guided_src)-1)
            r, c = int(_guided_src[idx]), int(_guided_tgt[idx])
        else:
            # Random (with targeted add)
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
            return {'delta': -1e9, 'type': 'add'}
        new_s[r, c] = rng.random() < 0.5
        new_m[r, c] = rng.randint(1, 255)
    elif ptype == 'flip':
        rs, cs = np.where(mmag > 0)
        if len(rs) == 0: return {'delta': -1e9, 'type': 'flip'}
        idx = rng.randint(0, len(rs)-1)
        new_s[rs[idx], cs[idx]] = not msign[rs[idx], cs[idx]]
    elif ptype == 'mag_resample':
        rs, cs = np.where(mmag > 0)
        if len(rs) == 0: return {'delta': -1e9, 'type': 'mag_resample'}
        idx = rng.randint(0, len(rs)-1)
        new_m[rs[idx], cs[idx]] = rng.randint(1, 255)

    seqs = []
    for _ in range(_n_train):
        off = np_rng.randint(0, len(_all_data) - _seq_len)
        seqs.append(_all_data[off:off+_seq_len])
    old = _eval_loglik(msign, mmag, H, seqs)
    new = _eval_loglik(new_s, new_m, H, seqs)
    return {'delta': new - old, 'type': ptype,
            'new_s': new_s.flatten() if new > old else None,
            'new_m': new_m.flatten() if new > old else None}


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

def eval_facts(msign, mmag, H, wof, bp, it):
    pat_norm = bp / (np.linalg.norm(bp, axis=1, keepdims=True) + 1e-8)
    rs, cs = np.where(mmag > 0)
    if len(rs) == 0: return 0.0
    s = msign[rs, cs].astype(np.float32) * 2 - 1
    sp = s * mmag[rs, cs].astype(np.float32) / 128.0
    ret = 217.0 / 256.0; tc = 0; ta = 0
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
                ch += raw; ch *= ret
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


def compute_guided_candidates(mmag, inj_table, wof, bigram, bp, H, top_n=200):
    """Find top N potential (src, tgt) pairs by alignment score."""
    inj_sensitivity = np.abs(inj_table.astype(np.float32)).mean(axis=0)
    target_vecs = bigram @ bp
    target_norms = target_vecs / (np.linalg.norm(target_vecs, axis=1, keepdims=True) + 1e-8)
    w_align = wof @ target_norms.T
    out_alignment = np.abs(w_align).mean(axis=1)

    scores = inj_sensitivity[:, None] * out_alignment[None, :]
    np.fill_diagonal(scores, 0)
    scores[mmag > 0] = 0  # exclude existing edges

    flat_idx = np.argsort(scores.ravel())[::-1][:top_n]
    return flat_idx // H, flat_idx % H


def run_config(name, use_guided, H, bp, inj_table, wof, ALL_DATA, bigram,
               n_steps, n_workers, schedule):
    SEQ_LEN = 120
    eval_rng = np.random.RandomState(9999)
    eval_seqs = [ALL_DATA[off:off+SEQ_LEN] for off in
                 [eval_rng.randint(0, len(ALL_DATA)-SEQ_LEN) for _ in range(20)]]

    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    sys.stdout.flush()

    msign = np.zeros((H, H), dtype=np.bool_)
    mmag = np.zeros((H, H), dtype=np.uint8)
    accepts = {'add': 0, 'flip': 0, 'mag_resample': 0}
    t0 = time.time()

    guided_src = np.array([], dtype=np.int64)
    guided_tgt = np.array([], dtype=np.int64)

    pool = Pool(n_workers, initializer=init_w,
                initargs=(bp, ALL_DATA, SEQ_LEN, 2, wof, inj_table, guided_src, guided_tgt))

    try:
        for step in range(1, n_steps+1):
            # Refresh guided candidates every 100 steps
            if use_guided and step % 100 == 1:
                guided_src, guided_tgt = compute_guided_candidates(
                    mmag, inj_table, wof, bigram, bp, H)
                pool.terminate(); pool.join()
                pool = Pool(n_workers, initializer=init_w,
                            initargs=(bp, ALL_DATA, SEQ_LEN, 2, wof, inj_table,
                                      guided_src, guided_tgt))

            ptype = schedule[(step-1) % len(schedule)]
            if ptype in ('flip', 'mag_resample') and (mmag > 0).sum() == 0:
                ptype = 'add'

            args = [(msign.flatten(), mmag.flatten(), H,
                     10000+step*50+w, ptype, use_guided) for w in range(n_workers)]
            results = pool.map(worker_eval, args)
            best = max(results, key=lambda x: x['delta'])
            if best['delta'] > 0.00005 and best.get('new_s') is not None:
                msign = best['new_s'].reshape(H, H)
                mmag = best['new_m'].reshape(H, H)
                accepts[best['type']] += 1

            if step % 500 == 0:
                elapsed = time.time() - t0
                edges = int((mmag > 0).sum())
                overall = eval_facts(msign, mmag, H, wof, bp, inj_table)
                print(f"  [{step:4d}] answer={overall*100:.1f}% edges={edges} "
                      f"A={accepts['add']}|F={accepts['flip']}|M={accepts['mag_resample']} {elapsed:.0f}s")
                sys.stdout.flush()
    finally:
        pool.terminate(); pool.join()

    elapsed = time.time() - t0
    edges = int((mmag > 0).sum())
    overall = eval_facts(msign, mmag, H, wof, bp, inj_table)
    return {'name': name, 'acc': overall, 'edges': edges, 'time': elapsed}


if __name__ == "__main__":
    IO = 256; NV = 4; H = IO * NV
    N_WORKERS = 18; BUDGET = 2000
    SCHEDULE = ['add', 'add', 'flip', 'mag_resample', 'add', 'add']

    lines = [f"{k}={v}\n" for k, v in FACTS]
    random.seed(42)
    corpus_text = ''
    while len(corpus_text) < 80_000:
        random.shuffle(lines)
        corpus_text += ''.join(lines)
    ALL_DATA = np.frombuffer(corpus_text.encode('ascii'), dtype=np.uint8).copy()

    bigram = np.zeros((256, 256), dtype=np.float64)
    for i in range(len(ALL_DATA) - 1):
        bigram[ALL_DATA[i], ALL_DATA[i+1]] += 1
    row_sums = bigram.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    bigram = (bigram / row_sums).astype(np.float32)

    SelfWiringGraph.NV_RATIO = 4; bp = make_bp(IO)
    random.seed(42); np.random.seed(42)
    ref = SelfWiringGraph(IO)
    inp = ref.input_projection / ref.INJ_SCALE * 1.0
    outp = ref.output_projection / ref.INJ_SCALE * 1.0
    inj_table = np.clip(bp @ inp * 128, -128, 127).astype(np.int8)
    woi = np.clip(outp * 128, -128, 127).astype(np.int8)
    wof = woi.astype(np.float32) / 128.0

    print(f"GUIDED ADD SWEEP")
    print(f"A: random add (baseline)")
    print(f"B: guided add (top 200 candidates from alignment scoring)")

    results = []
    results.append(run_config("A: random add", False, H, bp, inj_table, wof,
                              ALL_DATA, bigram, BUDGET, N_WORKERS, SCHEDULE))
    results.append(run_config("B: guided add", True, H, bp, inj_table, wof,
                              ALL_DATA, bigram, BUDGET, N_WORKERS, SCHEDULE))

    print(f"\n{'='*60}")
    print(f"  GUIDED ADD RESULTS")
    print(f"{'='*60}")
    print(f"  {'Config':<25} {'Acc%':>6} {'Edges':>6} {'Time':>6}")
    print(f"  {'-'*25} {'-'*6} {'-'*6} {'-'*6}")
    for r in results:
        print(f"  {r['name']:<25} {r['acc']*100:6.2f} {r['edges']:6d} {r['time']:5.0f}s")
    delta = (results[1]['acc'] - results[0]['acc']) * 100
    print(f"\n  Delta: {delta:+.2f}%")
    print(f"{'='*60}")
    sys.stdout.flush()

"""
INSTNCT Adaptive v2 — learnable schedule + learnable prune rate
================================================================
Mutation types: add, flip, mag, ret (4x int4 weights)
Auto prune: every 250 steps, remove bottom N edges (N = learnable int4, 1-16)
Hard prune: every 1000 steps, knee detection

Saves checkpoints every 1000 steps.
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

def _eval_loglik(msign, mmag, ret_int4, H, seqs):
    rs, cs = np.where(mmag > 0)
    s = msign[rs, cs].astype(np.float32) * 2 - 1
    sp_vals = s * mmag[rs, cs].astype(np.float32) / 128.0
    pat_norm = _bp / (np.linalg.norm(_bp, axis=1, keepdims=True) + 1e-8)
    ret_vec = 1.0 - ret_int4.astype(np.float32) * 0.01
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
                charge += raw; charge *= ret_vec
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
    msign_flat, mmag_flat, ret_int4, H, seed, ptype = args
    rng = random.Random(seed)
    np_rng = np.random.RandomState(seed)
    msign = msign_flat.reshape(H, H); mmag = mmag_flat.reshape(H, H)
    new_s = msign.copy(); new_m = mmag.copy()
    new_ret = ret_int4.copy()

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

    old = _eval_loglik(msign, mmag, ret_int4, H, seqs)
    new = _eval_loglik(new_s, new_m, new_ret, H, seqs)
    improved = new > old
    return {'delta': new - old, 'type': ptype,
            'new_s': new_s.flatten() if improved else None,
            'new_m': new_m.flatten() if improved else None,
            'new_ret': new_ret if improved else None}


def compute_edge_scores(msign, mmag, ret_int4, H, wof, bp, bigram, inj_table, eval_seqs):
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
    if len(scores) < 3: return 0
    sorted_s = np.sort(scores)
    gaps = np.diff(sorted_s)
    half = len(gaps) // 2
    if half == 0: return 0
    return int(np.argmax(gaps[:half])) + 1


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
    ret_vec = 1.0 - ret_int4.astype(np.float32) * 0.01
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
    schedule = []
    for i, t in enumerate(MUT_TYPES):
        schedule.extend([t] * int(weights[i]))
    if not schedule:
        schedule = ['add']
    return schedule


if __name__ == "__main__":
    IO = 256; NV = 4; H = IO * NV
    N_WORKERS = 18; BUDGET = 5000
    PRUNE_EVERY = 250
    HARD_PRUNE_EVERY = 1000

    lines = [f"{k}={v}\n" for k, v in FACTS]
    random.seed(42)
    corpus_text = ''
    while len(corpus_text) < 50_000:
        random.shuffle(lines)
        corpus_text += ''.join(lines)
    ALL_DATA = np.frombuffer(corpus_text.encode('ascii'), dtype=np.uint8).copy()

    bigram = np.zeros((256, 256), dtype=np.float64)
    for i in range(len(ALL_DATA) - 1):
        bigram[ALL_DATA[i], ALL_DATA[i+1]] += 1
    row_sums = bigram.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    bigram = (bigram / row_sums).astype(np.float32)

    bp = make_bp(IO)
    random.seed(42); np.random.seed(42)
    ref = SelfWiringGraph(IO, hidden_ratio=4, projection_scale=1.0)
    inp = ref.input_projection
    outp = ref.output_projection
    inj_table = np.clip(bp @ inp * 128, -128, 127).astype(np.int8)
    woi = np.clip(outp * 128, -128, 127).astype(np.int8)
    wof = woi.astype(np.float32) / 128.0

    CKPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
    os.makedirs(CKPT_DIR, exist_ok=True)

    print(f"ADAPTIVE v2 — learnable schedule + learnable prune rate")
    print(f"Types: {MUT_TYPES}")
    print(f"Prune: every {PRUNE_EVERY} steps, remove bottom N (N=learnable int4)")
    print(f"Hard prune: every {HARD_PRUNE_EVERY} steps, knee detection")
    print(f"{BUDGET} steps, LL eval, w=2, PNR, threshold=0.00005")
    print(f"{'='*70}")
    sys.stdout.flush()

    msign = np.zeros((H, H), dtype=np.bool_)
    mmag = np.zeros((H, H), dtype=np.uint8)
    ret_int4 = np.full(H, 15, dtype=np.uint8)

    # Learnable params
    sched_weights = np.array([4, 1, 1, 1], dtype=np.uint8)
    prune_rate = np.uint8(3)  # int4: remove bottom 3 edges per prune cycle

    schedule = build_schedule(sched_weights)
    accepts = {t: 0 for t in MUT_TYPES}
    accepts['sched'] = 0
    accepts['prune_rate'] = 0
    decayed_total = 0
    t0 = time.time()

    eval_rng = np.random.RandomState(9999)
    eval_seqs = [ALL_DATA[off:off+120] for off in
                 [eval_rng.randint(0, len(ALL_DATA)-120) for _ in range(5)]]

    pool = Pool(N_WORKERS, initializer=init_w,
                initargs=(bp, ALL_DATA, 120, 2, wof, inj_table))
    try:
        for step in range(1, BUDGET+1):
            # Schedule mutation: every 50 steps
            if step % 50 == 0:
                test_weights = sched_weights.copy()
                idx = random.randint(0, len(MUT_TYPES)-1)
                test_weights[idx] = random.randint(0, 15)
                sched_weights = test_weights
                schedule = build_schedule(sched_weights)
                accepts['sched'] += 1

            # Prune rate mutation: every 100 steps
            if step % 100 == 0:
                new_pr = max(0, min(15, int(prune_rate) + random.randint(-2, 2)))
                prune_rate = np.uint8(new_pr)
                accepts['prune_rate'] += 1

            ptype = schedule[(step-1) % len(schedule)]
            if ptype in ('flip', 'mag') and (mmag > 0).sum() == 0:
                ptype = 'add'

            args = [(msign.flatten(), mmag.flatten(), ret_int4.copy(), H,
                     10000+step*50+w, ptype) for w in range(N_WORKERS)]
            results = pool.map(worker_eval, args)
            best = max(results, key=lambda x: x['delta'])
            if best['delta'] > 0.00005:
                if best.get('new_s') is not None:
                    msign = best['new_s'].reshape(H, H)
                    mmag = best['new_m'].reshape(H, H)
                if best.get('new_ret') is not None:
                    ret_int4 = best['new_ret']
                accepts[best['type']] += 1

            # AUTO PRUNE: every PRUNE_EVERY steps, remove bottom N
            if step % PRUNE_EVERY == 0 and (mmag > 0).sum() > int(prune_rate) + 5:
                scores, e_rs, e_cs = compute_edge_scores(
                    msign, mmag, ret_int4, H, wof, bp, bigram, inj_table, eval_seqs[:3])
                if len(scores) > 0:
                    n_remove = min(int(prune_rate), len(scores) - 5)
                    if n_remove > 0:
                        worst_idx = np.argsort(scores)[:n_remove]
                        for idx in worst_idx:
                            mmag[e_rs[idx], e_cs[idx]] = 0
                            msign[e_rs[idx], e_cs[idx]] = False
                        decayed_total += n_remove

            # HARD PRUNE: every HARD_PRUNE_EVERY steps
            if step % HARD_PRUNE_EVERY == 0 and (mmag > 0).sum() > 10:
                scores, e_rs, e_cs = compute_edge_scores(
                    msign, mmag, ret_int4, H, wof, bp, bigram, inj_table, eval_seqs)
                if len(scores) > 0:
                    harmful = scores < 0
                    n_harm = int(harmful.sum())
                    mmag[e_rs[harmful], e_cs[harmful]] = 0
                    msign[e_rs[harmful], e_cs[harmful]] = False
                    keep = ~harmful
                    n_knee = 0
                    if keep.sum() > 3:
                        kept_scores = scores[keep]
                        kept_rs = e_rs[keep]; kept_cs = e_cs[keep]
                        n_knee = find_knee(kept_scores)
                        if n_knee > 0:
                            weak_idx = np.argsort(kept_scores)[:n_knee]
                            for idx in weak_idx:
                                mmag[kept_rs[idx], kept_cs[idx]] = 0
                                msign[kept_rs[idx], kept_cs[idx]] = False
                    edges_after = int((mmag > 0).sum())
                    print(f"  [HARD PRUNE @{step}] {n_harm} harmful + {n_knee} knee = "
                          f"{n_harm+n_knee} removed, {edges_after} remain")
                    sys.stdout.flush()

            # REPORT
            if step % 500 == 0:
                elapsed = time.time() - t0
                edges = int((mmag > 0).sum())
                overall = eval_facts(msign, mmag, ret_int4, H, wof, bp, inj_table)
                n_mem = int((ret_int4 <= 5).sum())
                w_str = '/'.join(f"{MUT_TYPES[i][0]}={sched_weights[i]}" for i in range(len(MUT_TYPES)))
                acc_str = '/'.join(f"{t[0]}={accepts[t]}" for t in MUT_TYPES)
                print(f"  [{step:4d}] acc={overall*100:.1f}% edges={edges} prune_rate={int(prune_rate)} "
                      f"decayed={decayed_total} mem={n_mem} "
                      f"w=[{w_str}] [{acc_str}] {elapsed:.0f}s")
                sys.stdout.flush()

            # CHECKPOINT
            if step % 1000 == 0:
                ckpt_path = os.path.join(CKPT_DIR, f"adaptive_v2_step{step}.npz")
                np.savez_compressed(ckpt_path,
                    msign=msign, mmag=mmag, ret_int4=ret_int4,
                    inj_table=inj_table, output_projection_int8=woi,
                    sched_weights=sched_weights, prune_rate=prune_rate)
                print(f"  SAVED: {ckpt_path}")
                sys.stdout.flush()

    finally:
        pool.terminate(); pool.join()

    elapsed = time.time() - t0
    edges = int((mmag > 0).sum())
    overall = eval_facts(msign, mmag, ret_int4, H, wof, bp, inj_table)
    n_mem = int((ret_int4 <= 5).sum())

    print(f"\n{'='*70}")
    print(f"  ADAPTIVE v2 FINAL ({BUDGET} steps)")
    print(f"{'='*70}")
    print(f"  Accuracy: {overall*100:.2f}%")
    print(f"  Edges: {edges}")
    print(f"  Memory neurons: {n_mem}")
    print(f"  Prune rate: {int(prune_rate)}")
    print(f"  Decayed total: {decayed_total}")
    print(f"  Schedule: {'/'.join(f'{MUT_TYPES[i][0]}={sched_weights[i]}' for i in range(len(MUT_TYPES)))}")
    print(f"  Time: {elapsed:.0f}s")
    print(f"{'='*70}")

    # Final checkpoint
    np.savez_compressed(os.path.join(CKPT_DIR, "adaptive_v2_final.npz"),
        msign=msign, mmag=mmag, ret_int4=ret_int4,
        inj_table=inj_table, output_projection_int8=woi,
        sched_weights=sched_weights, prune_rate=prune_rate)
    sys.stdout.flush()


"""
INSTNCT Word Pairs — log-likelihood eval (no bigram!)
=====================================================
Same 30 word pairs, but now the worker eval uses
log P(actual_next_byte) instead of bigram cosine.

This is context-aware: rewards predicting the RIGHT byte
for THIS specific context, not matching the average distribution.
"""
import sys, os, time, random
import numpy as np
from multiprocessing import Pool

from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "model"))
from graph import SelfWiringGraph

_bp = None; _all_data = None; _seq_len = 80; _n_train = 2
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

def _eval_loglik(msign, mmag, H, seqs):
    """Log-likelihood eval: score = mean log P(actual_next_byte)"""
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
            for t in range(8):
                if t < 2:
                    act = act + _inj_table[text_bytes[i]].astype(np.float32) / 128.0
                raw = np.zeros(H, dtype=np.float32)
                if len(rs):
                    np.add.at(raw, cs, act[rs] * sp_vals)
                charge += raw; charge *= ret
                act = np.maximum(charge, 0.0)
                charge = np.maximum(charge, 0.0)
            state = act.copy()
            out = charge @ _output_projection_f
            out_n = out / (np.linalg.norm(out) + 1e-8)
            sims = out_n @ pat_norm.T
            e = np.exp(sims - sims.max())
            probs = e / e.sum()
            # Log-likelihood of the ACTUAL next byte
            seq_score += np.log(probs[text_bytes[i+1]] + 1e-10)
            n += 1
        total += seq_score / n if n else 0
    return total / len(seqs)

def worker_eval(args):
    msign_flat, mmag_flat, H, seed, ptype = args
    rng = random.Random(seed)
    np_rng = np.random.RandomState(seed)
    msign = msign_flat.reshape(H, H); mmag = mmag_flat.reshape(H, H)
    new_s = msign.copy(); new_m = mmag.copy()

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
    ("sky", "blue"), ("sun", "yellow"), ("grass", "green"), ("fire", "red"),
    ("snow", "white"), ("coal", "black"), ("gold", "shiny"), ("ice", "cold"),
    ("lava", "hot"), ("milk", "white"), ("cat", "meow"), ("dog", "bark"),
    ("cow", "moo"), ("bee", "buzz"), ("owl", "hoot"), ("sea", "salt"),
    ("rain", "wet"), ("sand", "dry"), ("iron", "hard"), ("silk", "soft"),
    ("rose", "red"), ("leaf", "green"), ("moon", "round"), ("star", "bright"),
    ("fish", "swim"), ("bird", "fly"), ("frog", "jump"), ("bear", "big"),
    ("ant", "tiny"), ("fox", "sly"),
]

EQ_BYTE = ord('=')
NL_BYTE = ord('\n')

def eval_facts_detailed(msign, mmag, H, output_projection_f, bp, inj_table):
    """Run each fact individually, report per-byte predictions."""
    pat_norm = bp / (np.linalg.norm(bp, axis=1, keepdims=True) + 1e-8)
    rs, cs = np.where(mmag > 0)
    if len(rs) == 0:
        return 0.0, {}
    s = msign[rs, cs].astype(np.float32) * 2 - 1
    sp_vals = s * mmag[rs, cs].astype(np.float32) / 128.0
    ret = 217.0 / 256.0

    total_correct = 0; total_answer = 0
    fact_results = {}

    for key, val in FACTS:
        text = (key + '=' + val + '\n').encode('ascii')
        text_bytes = np.frombuffer(text, dtype=np.uint8)

        state = np.zeros(H, dtype=np.float32)
        charge = np.zeros(H, dtype=np.float32)

        eq_pos = len(key)
        answer_correct = 0; answer_total = 0
        pred_chars = []

        for i in range(len(text_bytes)-1):
            act = state.copy()
            for t in range(8):
                if t < 2: act = act + inj_table[text_bytes[i]].astype(np.float32) / 128.0
                raw = np.zeros(H, dtype=np.float32)
                if len(rs): np.add.at(raw, cs, act[rs] * sp_vals)
                charge += raw; charge *= ret
                act = np.maximum(charge, 0.0); charge = np.maximum(charge, 0.0)
            state = act.copy()
            out = charge @ output_projection_f
            out_n = out / (np.linalg.norm(out) + 1e-8)
            sims = out_n @ pat_norm.T
            pred_byte = int(np.argmax(sims))

            if i >= eq_pos:  # after and including '='
                actual = int(text_bytes[i+1])
                correct = pred_byte == actual
                if i > eq_pos and actual != NL_BYTE:  # answer bytes only
                    answer_total += 1
                    if correct:
                        answer_correct += 1
                    pred_ch = chr(pred_byte) if 32 <= pred_byte < 127 else '?'
                    actual_ch = chr(actual)
                    pred_chars.append(f"{pred_ch}({'+'if correct else 'x'})")
                elif i == eq_pos:  # prediction at '=' position = first answer byte
                    actual = int(text_bytes[i+1])
                    correct = pred_byte == actual
                    answer_total += 1
                    if correct:
                        answer_correct += 1
                    pred_ch = chr(pred_byte) if 32 <= pred_byte < 127 else '?'
                    pred_chars.append(f"{pred_ch}({'+'if correct else 'x'})")

        total_correct += answer_correct
        total_answer += answer_total
        acc = answer_correct / answer_total if answer_total else 0
        fact_results[key] = {'acc': acc, 'correct': answer_correct, 'total': answer_total,
                             'preds': ' '.join(pred_chars)}

    overall = total_correct / total_answer if total_answer else 0
    return overall, fact_results


if __name__ == "__main__":
    IO = 256; NV = 4; H = IO * NV
    N_WORKERS = 18
    BUDGET = 2000
    THRESHOLD = 0  # accept ALL improvements (log-lik is noisy, need sensitivity)
    SEQ_LEN = 80
    SCHEDULE = ['add', 'add', 'flip', 'mag_resample', 'add', 'add']

    # Build corpus
    lines = [f"{k}={v}\n" for k, v in FACTS]
    random.seed(42)
    corpus_text = ''
    while len(corpus_text) < 80_000:
        random.shuffle(lines)
        corpus_text += ''.join(lines)
    ALL_DATA = np.frombuffer(corpus_text.encode('ascii'), dtype=np.uint8).copy()

    SelfWiringGraph.NV_RATIO = 4
    bp = make_bp(IO)

    random.seed(42); np.random.seed(42)
    ref = SelfWiringGraph(IO)
    input_projection = ref.input_projection / ref.INJ_SCALE * 1.0
    output_projection = ref.output_projection / ref.INJ_SCALE * 1.0

    inj_table = np.clip(bp @ input_projection * 128, -128, 127).astype(np.int8)
    output_projection_int8 = np.clip(output_projection * 128, -128, 127).astype(np.int8)
    output_projection_f = output_projection_int8.astype(np.float32) / 128.0

    CKPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
    os.makedirs(CKPT_DIR, exist_ok=True)

    print(f"WORD PAIRS with LOG-LIKELIHOOD eval (no bigram!)")
    print(f"{len(FACTS)} facts, corpus {len(ALL_DATA)/1024:.0f} KB")
    print(f"H={H}, {N_WORKERS}w, seq_len={SEQ_LEN}, threshold={THRESHOLD}")
    print(f"Random baseline: ~4% (1/26)")
    print(f"{'='*60}")
    sys.stdout.flush()

    eval_rng = np.random.RandomState(9999)
    eval_seqs = [ALL_DATA[off:off+SEQ_LEN] for off in
                 [eval_rng.randint(0, len(ALL_DATA)-SEQ_LEN) for _ in range(20)]]

    msign = np.zeros((H, H), dtype=np.bool_)
    mmag = np.zeros((H, H), dtype=np.uint8)
    accepts = {'add': 0, 'flip': 0, 'mag_resample': 0}
    t0 = time.time()

    pool = Pool(N_WORKERS, initializer=init_w,
                initargs=(bp, ALL_DATA, SEQ_LEN, 2, output_projection_f, inj_table))
    try:
        for step in range(1, BUDGET+1):
            ptype = SCHEDULE[(step-1) % len(SCHEDULE)]
            if ptype in ('flip', 'mag_resample') and (mmag > 0).sum() == 0:
                ptype = 'add'

            args = [(msign.flatten(), mmag.flatten(), H,
                     10000+step*50+w, ptype) for w in range(N_WORKERS)]
            results = pool.map(worker_eval, args)
            best = max(results, key=lambda x: x['delta'])
            if best['delta'] > THRESHOLD and best.get('new_s') is not None:
                msign = best['new_s'].reshape(H, H)
                mmag = best['new_m'].reshape(H, H)
                accepts[best['type']] += 1

            if step % 200 == 0:
                elapsed = time.time() - t0
                edges = int((mmag > 0).sum())

                overall, fact_res = eval_facts_detailed(
                    msign, mmag, H, output_projection_f, bp, inj_table)

                # Top 5 best
                scored = sorted(fact_res.items(), key=lambda x: -x[1]['acc'])[:5]
                top_str = ', '.join(f"{k}={dict(FACTS)[k]}:{v['acc']*100:.0f}%" for k, v in scored)

                # Mean log-lik (compute inline, not using worker globals)
                ll = 0.0

                print(f"  [{step:4d}] answer={overall*100:.1f}% LL={ll:.2f} "
                      f"edges={edges} A={accepts['add']}|F={accepts['flip']}|M={accepts['mag_resample']} "
                      f"{elapsed:.0f}s")
                print(f"         best: {top_str}")
                sys.stdout.flush()
    finally:
        pool.terminate(); pool.join()

    elapsed = time.time() - t0
    edges = int((mmag > 0).sum())
    overall, fact_res = eval_facts_detailed(msign, mmag, H, output_projection_f, bp, inj_table)

    print(f"\n{'='*60}")
    print(f"  WORD PAIRS — LOG-LIKELIHOOD RESULTS ({BUDGET} steps)")
    print(f"{'='*60}")
    print(f"  Answer accuracy: {overall*100:.2f}% (random ~4%)")
    print(f"  Edges: {edges}")
    print(f"  Time: {elapsed:.0f}s")
    print(f"\n  Per-fact detail:")
    for key, val in FACTS:
        if key in fact_res:
            r = fact_res[key]
            bar = '#' * int(r['acc'] * 30)
            print(f"    {key:5s}={val:6s}  {r['acc']*100:5.1f}% {r['preds']}")
    print(f"{'='*60}")

    np.savez_compressed(os.path.join(CKPT_DIR, "wordpairs_ll_final.npz"),
        msign=msign, mmag=mmag, inj_table=inj_table,
        output_projection_int8=output_projection_int8)
    sys.stdout.flush()

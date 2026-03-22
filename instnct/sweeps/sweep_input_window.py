"""
INSTNCT Input Window Sweep
===========================
Does a wider input context window help?

A: window=1 (current byte only — baseline)
B: window=2 (current + prev*0.5)
C: window=3 (current + prev*0.5 + prev2*0.25)
D: window=4 (current + prev*0.5 + prev2*0.25 + prev3*0.125)

Longer sentences as facts for real context test.
Log-likelihood eval, 2000 steps each.
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
_output_projection_f = None; _inj_table = None; _window = 1

def init_w(b, d, sl, nt, wof, it, win):
    global _bp, _all_data, _seq_len, _n_train, _output_projection_f, _inj_table, _window
    _bp, _all_data, _seq_len, _n_train = b, d, sl, nt
    _output_projection_f, _inj_table, _window = wof, it, win

def make_bp(io_dim, seed=12345):
    rng = np.random.RandomState(seed)
    p = rng.randn(256, io_dim).astype(np.float32)
    p /= np.linalg.norm(p, axis=1, keepdims=True)
    return p

def _make_injection(text_bytes, pos, inj_table, window):
    """Build injection vector from sliding window of bytes."""
    inj = inj_table[text_bytes[pos]].astype(np.float32) / 128.0
    decay = 0.5
    for w in range(1, window):
        if pos - w >= 0:
            inj = inj + inj_table[text_bytes[pos - w]].astype(np.float32) / 128.0 * decay
        decay *= 0.5
    return inj

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
            inj = _make_injection(text_bytes, i, _inj_table, _window)
            for t in range(8):
                if t < 2:
                    act = act + inj
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


# --- Longer sentence facts ---
FACTS = [
    ("the sky is", "blue"),
    ("the sun is", "yellow and bright"),
    ("grass color is", "green"),
    ("fire is very", "hot and red"),
    ("snow is always", "white and cold"),
    ("coal color is", "black"),
    ("gold looks very", "shiny"),
    ("ice feels very", "cold"),
    ("milk color is", "white"),
    ("a cat says", "meow"),
    ("a dog says", "bark bark"),
    ("a cow says", "moo"),
    ("a bee says", "buzz"),
    ("the sea is", "salt water"),
    ("rain makes things", "wet"),
    ("iron feels very", "hard"),
    ("silk feels very", "soft"),
    ("a red flower is", "a rose"),
    ("leaves are always", "green"),
    ("the moon is", "round and bright"),
    ("stars look very", "bright at night"),
    ("a fish can", "swim fast"),
    ("a bird can", "fly high"),
    ("a frog can", "jump far"),
    ("a bear is", "big and strong"),
    ("an ant is", "tiny and small"),
    ("paris is in", "france"),
    ("tokyo is in", "japan"),
    ("london is in", "england"),
    ("rome is in", "italy"),
]

EQ_BYTE = ord('=')

def eval_facts_detail(msign, mmag, H, output_projection_f, bp, inj_table, window):
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
        state = np.zeros(H, dtype=np.float32); charge = np.zeros(H, dtype=np.float32)
        eq_pos = len(key)
        answer_correct = 0; answer_total = 0
        pred_chars = []

        for i in range(len(text_bytes)-1):
            act = state.copy()
            inj = _make_injection(text_bytes, i, inj_table, window)
            for t in range(8):
                if t < 2: act = act + inj
                raw = np.zeros(H, dtype=np.float32)
                if len(rs): np.add.at(raw, cs, act[rs] * sp_vals)
                charge += raw; charge *= ret
                act = np.maximum(charge, 0.0); charge = np.maximum(charge, 0.0)
            state = act.copy()
            out = charge @ output_projection_f
            out_n = out / (np.linalg.norm(out) + 1e-8)
            sims = out_n @ pat_norm.T
            pred_byte = int(np.argmax(sims))

            if i >= eq_pos:
                actual = int(text_bytes[i+1])
                if actual != ord('\n'):
                    correct = pred_byte == actual
                    answer_total += 1
                    if correct: answer_correct += 1
                    pred_ch = chr(pred_byte) if 32 <= pred_byte < 127 else '?'
                    mark = '+' if correct else 'x'
                    pred_chars.append(f"{pred_ch}({mark})")

        total_correct += answer_correct
        total_answer += answer_total
        acc = answer_correct / answer_total if answer_total else 0
        short_key = key[:15]
        fact_results[short_key] = {'acc': acc, 'preds': ' '.join(pred_chars), 'val': val}

    overall = total_correct / total_answer if total_answer else 0
    return overall, fact_results


def run_config(window, H, bp, inj_table, output_projection_f, ALL_DATA, n_steps, n_workers, schedule):
    SEQ_LEN = 120
    eval_rng = np.random.RandomState(9999)
    eval_seqs = [ALL_DATA[off:off+SEQ_LEN] for off in
                 [eval_rng.randint(0, len(ALL_DATA)-SEQ_LEN) for _ in range(20)]]

    print(f"\n{'='*60}")
    print(f"  Window={window} ({'current only' if window==1 else f'current + {window-1} prev'})")
    print(f"{'='*60}")
    sys.stdout.flush()

    msign = np.zeros((H, H), dtype=np.bool_)
    mmag = np.zeros((H, H), dtype=np.uint8)
    accepts = {'add': 0, 'flip': 0, 'mag_resample': 0}
    t0 = time.time()

    pool = Pool(n_workers, initializer=init_w,
                initargs=(bp, ALL_DATA, SEQ_LEN, 2, output_projection_f, inj_table, window))
    try:
        for step in range(1, n_steps+1):
            ptype = schedule[(step-1) % len(schedule)]
            if ptype in ('flip', 'mag_resample') and (mmag > 0).sum() == 0:
                ptype = 'add'

            args = [(msign.flatten(), mmag.flatten(), H,
                     10000+step*50+w, ptype) for w in range(n_workers)]
            results = pool.map(worker_eval, args)
            best = max(results, key=lambda x: x['delta'])
            if best['delta'] > 0.00005 and best.get('new_s') is not None:
                msign = best['new_s'].reshape(H, H)
                mmag = best['new_m'].reshape(H, H)
                accepts[best['type']] += 1

            if step % 500 == 0:
                elapsed = time.time() - t0
                edges = int((mmag > 0).sum())
                overall, _ = eval_facts_detail(msign, mmag, H, output_projection_f, bp, inj_table, window)
                print(f"  [{step:4d}] answer={overall*100:.1f}% edges={edges} "
                      f"A={accepts['add']}|F={accepts['flip']}|M={accepts['mag_resample']} {elapsed:.0f}s")
                sys.stdout.flush()
    finally:
        pool.terminate(); pool.join()

    elapsed = time.time() - t0
    edges = int((mmag > 0).sum())
    overall, fact_res = eval_facts_detail(msign, mmag, H, output_projection_f, bp, inj_table, window)

    return {
        'window': window, 'acc': overall, 'edges': edges, 'time': elapsed,
        'fact_res': fact_res, 'accepts': accepts
    }


if __name__ == "__main__":
    IO = 256; NV = 4; H = IO * NV
    N_WORKERS = 18
    BUDGET = 2000
    SCHEDULE = ['add', 'add', 'flip', 'mag_resample', 'add', 'add']

    # Build corpus from longer facts
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

    print(f"INPUT WINDOW SWEEP")
    print(f"{len(FACTS)} longer facts, corpus {len(ALL_DATA)/1024:.0f} KB")
    print(f"H={H}, {N_WORKERS}w, {BUDGET} steps each")
    print(f"Example: '{FACTS[0][0]}' -> '{FACTS[0][1]}'")
    print(f"Example: '{FACTS[3][0]}' -> '{FACTS[3][1]}'")
    print(f"Random baseline: ~4%")

    results = []
    for w in [1, 2, 3, 4]:
        results.append(run_config(w, H, bp, inj_table, output_projection_f,
                                  ALL_DATA, BUDGET, N_WORKERS, SCHEDULE))

    # Summary
    print(f"\n{'='*60}")
    print(f"  INPUT WINDOW SWEEP RESULTS")
    print(f"{'='*60}")
    print(f"  Random baseline: ~4%")
    print(f"  {'Window':<10} {'Answer%':>8} {'Edges':>6} {'Time':>6}")
    print(f"  {'-'*10} {'-'*8} {'-'*6} {'-'*6}")
    for r in results:
        label = f"w={r['window']}"
        print(f"  {label:<10} {r['acc']*100:8.2f} {r['edges']:6d} {r['time']:5.0f}s")

    best = max(results, key=lambda x: x['acc'])
    print(f"\n  Best: window={best['window']} ({best['acc']*100:.2f}%)")

    # Show per-fact detail for best
    print(f"\n  Top facts (window={best['window']}):")
    scored = sorted(best['fact_res'].items(), key=lambda x: -x[1]['acc'])
    for key, v in scored[:10]:
        print(f"    {key:15s} -> {v['val']:20s}  {v['acc']*100:5.1f}%  {v['preds'][:60]}")

    print(f"{'='*60}")
    sys.stdout.flush()

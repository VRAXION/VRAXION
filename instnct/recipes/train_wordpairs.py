"""
INSTNCT Word Pairs — associative memory test
=============================================
Can the SWG memorize key=value word pairs?

sky=blue
sun=yellow
cat=animal
...

Simplest possible fact learning.
Bigram is VERY strong here (sky= always followed by blue).
Random baseline: ~4% (1/26 letters).
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
_output_projection_f = None; _bigram = None; _inj_table = None

def init_w(b, d, sl, nt, wof, bg, it):
    global _bp, _all_data, _seq_len, _n_train, _output_projection_f, _bigram, _inj_table
    _bp, _all_data, _seq_len, _n_train = b, d, sl, nt
    _output_projection_f, _bigram, _inj_table = wof, bg, it

def make_bp(io_dim, seed=12345):
    rng = np.random.RandomState(seed)
    p = rng.randn(256, io_dim).astype(np.float32)
    p /= np.linalg.norm(p, axis=1, keepdims=True)
    return p

def _eval_bigram(msign, mmag, H, seqs):
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
            pred = e / e.sum()
            target_dist = _bigram[text_bytes[i]]
            cos = np.dot(pred, target_dist) / (np.linalg.norm(pred) * np.linalg.norm(target_dist) + 1e-8)
            seq_score += cos; n += 1
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

    old = _eval_bigram(msign, mmag, H, seqs)
    new = _eval_bigram(new_s, new_m, H, seqs)
    return {'delta': new - old, 'type': ptype,
            'new_s': new_s.flatten() if new > old else None,
            'new_m': new_m.flatten() if new > old else None}


# --- Word pairs ---
FACTS = [
    ("sky", "blue"),
    ("sun", "yellow"),
    ("grass", "green"),
    ("fire", "red"),
    ("snow", "white"),
    ("coal", "black"),
    ("gold", "shiny"),
    ("ice", "cold"),
    ("lava", "hot"),
    ("milk", "white"),
    ("cat", "meow"),
    ("dog", "bark"),
    ("cow", "moo"),
    ("bee", "buzz"),
    ("owl", "hoot"),
    ("sea", "salt"),
    ("rain", "wet"),
    ("sand", "dry"),
    ("iron", "hard"),
    ("silk", "soft"),
    ("rose", "red"),
    ("leaf", "green"),
    ("moon", "round"),
    ("star", "bright"),
    ("fish", "swim"),
    ("bird", "fly"),
    ("frog", "jump"),
    ("bear", "big"),
    ("ant", "tiny"),
    ("fox", "sly"),
]

EQ_BYTE = ord('=')
NL_BYTE = ord('\n')


def eval_facts(msign, mmag, H, output_projection_f, bp, inj_table, corpus_bytes):
    """Eval each fact: how many answer bytes correct after '='?"""
    pat_norm = bp / (np.linalg.norm(bp, axis=1, keepdims=True) + 1e-8)
    rs, cs = np.where(mmag > 0)
    if len(rs) == 0:
        return 0.0, 0.0, {}
    s = msign[rs, cs].astype(np.float32) * 2 - 1
    sp_vals = s * mmag[rs, cs].astype(np.float32) / 128.0
    ret = 217.0 / 256.0

    # Run through entire corpus
    state = np.zeros(H, dtype=np.float32)
    charge = np.zeros(H, dtype=np.float32)
    text_bytes = corpus_bytes

    answer_correct = 0; answer_total = 0
    all_correct = 0; all_total = 0
    in_answer = False

    # Per-fact tracking
    current_key = []
    fact_results = {}

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
        actual_next = int(text_bytes[i+1])

        if pred_byte == actual_next:
            all_correct += 1
        all_total += 1

        if text_bytes[i] == NL_BYTE:
            in_answer = False
            current_key = []
        elif text_bytes[i] == EQ_BYTE:
            in_answer = True
            key_str = bytes(current_key).decode('ascii', errors='replace')
            if key_str not in fact_results:
                fact_results[key_str] = {'correct': 0, 'total': 0}
            # First answer byte prediction (at '=' position)
            if pred_byte == actual_next:
                answer_correct += 1
                fact_results[key_str]['correct'] += 1
            answer_total += 1
            fact_results[key_str]['total'] += 1
        elif in_answer:
            if actual_next != NL_BYTE:
                if pred_byte == actual_next:
                    answer_correct += 1
                    key_str = bytes(current_key).decode('ascii', errors='replace')
                    if key_str in fact_results:
                        fact_results[key_str]['correct'] += 1
                answer_total += 1
                if key_str in fact_results:
                    fact_results[key_str]['total'] += 1
        else:
            current_key.append(text_bytes[i])

    ans_acc = answer_correct / answer_total if answer_total else 0
    all_acc = all_correct / all_total if all_total else 0
    return ans_acc, all_acc, fact_results


if __name__ == "__main__":
    IO = 256; NV = 4; H = IO * NV
    N_WORKERS = 18
    BUDGET = 2000
    THRESHOLD = 0.00005
    SCHEDULE = ['add', 'add', 'flip', 'mag_resample', 'add', 'add']

    # Build corpus
    lines = [f"{k}={v}\n" for k, v in FACTS]
    one_cycle = ''.join(lines)
    print(f"{len(FACTS)} word pairs, {len(one_cycle)} bytes per cycle")
    print(f"Examples: {lines[0].strip()}, {lines[1].strip()}, {lines[2].strip()}")

    random.seed(42)
    corpus_text = ''
    while len(corpus_text) < 80_000:
        random.shuffle(lines)
        corpus_text += ''.join(lines)

    ALL_DATA = np.frombuffer(corpus_text.encode('ascii'), dtype=np.uint8).copy()

    # Bigram
    bigram = np.zeros((256, 256), dtype=np.float64)
    for i in range(len(ALL_DATA) - 1):
        bigram[ALL_DATA[i], ALL_DATA[i+1]] += 1
    row_sums = bigram.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    bigram = (bigram / row_sums).astype(np.float32)

    # Bigram sanity
    print(f"\nBigram after '=':")
    top = np.argsort(bigram[EQ_BYTE])[::-1][:8]
    for b in top:
        if b >= 32:
            print(f"  '{chr(b)}' = {bigram[EQ_BYTE, b]:.3f}")

    SEQ_LEN = 80
    eval_rng = np.random.RandomState(9999)
    eval_seqs = [ALL_DATA[off:off+SEQ_LEN] for off in
                 [eval_rng.randint(0, len(ALL_DATA)-SEQ_LEN) for _ in range(20)]]

    # Use a fixed eval corpus (first 2000 bytes) for per-fact eval
    eval_corpus = ALL_DATA[:2000]

    SelfWiringGraph.NV_RATIO = NV
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

    print(f"\nH={H}, {N_WORKERS}w, seq_len={SEQ_LEN}, {BUDGET} steps")
    print(f"Corpus: {len(ALL_DATA)/1024:.0f} KB")
    print(f"Random baseline: ~4% (1/26 letters)")
    print(f"{'='*60}")
    sys.stdout.flush()

    msign = np.zeros((H, H), dtype=np.bool_)
    mmag = np.zeros((H, H), dtype=np.uint8)
    accepts = {'add': 0, 'flip': 0, 'mag_resample': 0}
    t0 = time.time()

    pool = Pool(N_WORKERS, initializer=init_w,
                initargs=(bp, ALL_DATA, SEQ_LEN, 2, output_projection_f, bigram, inj_table))
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

                ans_acc, all_acc, fact_res = eval_facts(
                    msign, mmag, H, output_projection_f, bp, inj_table, eval_corpus)

                print(f"  [{step:4d}] answer={ans_acc*100:.1f}% all={all_acc*100:.1f}% "
                      f"edges={edges} A={accepts['add']}|F={accepts['flip']}|M={accepts['mag_resample']} "
                      f"{elapsed:.0f}s")

                # Show top 5 best-learned facts
                if fact_res:
                    scored = [(k, v['correct']/v['total'] if v['total'] else 0, v['total'])
                              for k, v in fact_res.items()]
                    scored.sort(key=lambda x: -x[1])
                    top5 = scored[:5]
                    facts_str = ', '.join(f"{k}={next((v for kk,v in FACTS if kk==k),'?')}:{acc*100:.0f}%"
                                          for k, acc, _ in top5)
                    print(f"         best: {facts_str}")
                sys.stdout.flush()
    finally:
        pool.terminate(); pool.join()

    elapsed = time.time() - t0
    edges = int((mmag > 0).sum())

    ans_acc, all_acc, fact_res = eval_facts(
        msign, mmag, H, output_projection_f, bp, inj_table, eval_corpus)

    print(f"\n{'='*60}")
    print(f"  WORD PAIRS RESULTS ({BUDGET} steps)")
    print(f"{'='*60}")
    print(f"  Answer accuracy: {ans_acc*100:.2f}% (random ~4%)")
    print(f"  All accuracy:    {all_acc*100:.2f}%")
    print(f"  Edges: {edges}")
    print(f"  Time: {elapsed:.0f}s")
    print(f"\n  Per-fact accuracy:")
    for k, v in FACTS:
        if k in fact_res and fact_res[k]['total'] > 0:
            acc = fact_res[k]['correct'] / fact_res[k]['total']
            bar = '#' * int(acc * 30)
            print(f"    {k:5s}={v:6s}  {acc*100:5.1f}% {bar}")
    print(f"{'='*60}")

    np.savez_compressed(os.path.join(CKPT_DIR, "wordpairs_final.npz"),
        msign=msign, mmag=mmag, inj_table=inj_table,
        output_projection_int8=output_projection_int8)
    sys.stdout.flush()

"""
INSTNCT Variable Sorting — the REAL test
==========================================
3 digits (0-9, repeats OK) -> sorted output.
"352=235\n" "871=178\n" "999=999\n"
Output VARIES — network must actually SORT, not memorize.

10^3 = 1000 unique combos. 8 bytes per example.
2000 steps, per-position accuracy.
"""
import sys, os, time, random
import numpy as np
from multiprocessing import Pool

from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "model"))
from graph import SelfWiringGraph

_bp = None; _all_data = None; _seq_len = 48; _n_train = 2
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


EQ_BYTE = ord('=')
NL_BYTE = ord('\n')

def eval_varsort(msign, mmag, H, output_projection_f, text_bytes, bp, inj_table):
    pat_norm = bp / (np.linalg.norm(bp, axis=1, keepdims=True) + 1e-8)
    rs, cs = np.where(mmag > 0)
    s = msign[rs, cs].astype(np.float32) * 2 - 1
    sp_vals = s * mmag[rs, cs].astype(np.float32) / 128.0
    ret = 217.0 / 256.0
    state = np.zeros(H, dtype=np.float32); charge = np.zeros(H, dtype=np.float32)

    pos_correct = np.zeros(3, dtype=np.int32)
    pos_total = np.zeros(3, dtype=np.int32)
    sort_correct = 0; sort_total = 0
    all_correct = 0; all_total = 0
    output_pos = -1

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
        pred_byte = np.argmax(sims)
        actual_next = text_bytes[i+1]

        if pred_byte == actual_next:
            all_correct += 1
        all_total += 1

        if text_bytes[i] == EQ_BYTE:
            output_pos = 0
        elif output_pos >= 0:
            if actual_next == NL_BYTE or output_pos >= 3:
                output_pos = -1
            else:
                if pred_byte == actual_next:
                    sort_correct += 1
                    pos_correct[output_pos] += 1
                pos_total[output_pos] += 1
                sort_total += 1
                output_pos += 1

    sort_acc = sort_correct / sort_total if sort_total else 0
    all_acc = all_correct / all_total if all_total else 0
    return sort_acc, all_acc, pos_correct, pos_total


if __name__ == "__main__":
    IO = 256; NV = 4; H = IO * NV
    N_WORKERS = 18
    BUDGET = 2000
    THRESHOLD = 0.00005
    SEQ_LEN = 48  # 6 examples per window
    SCHEDULE = ['add', 'add', 'flip', 'mag_resample', 'add', 'add']

    SelfWiringGraph.NV_RATIO = NV
    bp = make_bp(IO)
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    # --- Generate 3-digit sorting data inline ---
    print("Generating 3-digit variable sorting data...")
    examples = []
    for a in range(10):
        for b in range(10):
            for c in range(10):
                digits = [a, b, c]
                inp = ''.join(str(x) for x in digits)
                out = ''.join(str(x) for x in sorted(digits))
                examples.append(f"{inp}={out}\n")
    print(f"  {len(examples)} unique combos, example: {examples[352].strip()}")

    random.seed(42)
    random.shuffle(examples)
    corpus_text = ''
    while len(corpus_text) < 200_000:
        random.shuffle(examples)
        corpus_text += ''.join(examples)

    ALL_DATA = np.frombuffer(corpus_text.encode('ascii'), dtype=np.uint8).copy()

    # Bigram
    bigram = np.zeros((256, 256), dtype=np.float64)
    for i in range(len(ALL_DATA) - 1):
        bigram[ALL_DATA[i], ALL_DATA[i+1]] += 1
    row_sums = bigram.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    bigram = (bigram / row_sums).astype(np.float32)

    print(f"  Corpus: {len(ALL_DATA)//8} examples, {len(ALL_DATA)/1024:.0f} KB")
    print(f"  Bigram '=' -> '0'={bigram[ord('='), ord('0')]:.3f}, "
          f"'1'={bigram[ord('='), ord('1')]:.3f}, "
          f"'5'={bigram[ord('='), ord('5')]:.3f}")

    eval_rng = np.random.RandomState(9999)
    eval_seqs = [ALL_DATA[off:off+SEQ_LEN] for off in
                 [eval_rng.randint(0, len(ALL_DATA)-SEQ_LEN) for _ in range(20)]]

    random.seed(42); np.random.seed(42)
    ref = SelfWiringGraph(IO)
    input_projection = ref.input_projection / ref.INJ_SCALE * 1.0
    output_projection = ref.output_projection / ref.INJ_SCALE * 1.0

    inj_table = np.clip(bp @ input_projection * 128, -128, 127).astype(np.int8)
    output_projection_int8 = np.clip(output_projection * 128, -128, 127).astype(np.int8)
    output_projection_f = output_projection_int8.astype(np.float32) / 128.0

    CKPT_DIR = os.path.join(BASE_DIR, "checkpoints")
    os.makedirs(CKPT_DIR, exist_ok=True)

    print(f"\nH={H}, {N_WORKERS}w, seq_len={SEQ_LEN}")
    print(f"Budget: {BUDGET} steps, threshold={THRESHOLD}")
    print(f"Random baseline: ~10% per position (10 possible digits)")
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

                all_sa, all_aa = [], []
                all_pc, all_pt = np.zeros(3), np.zeros(3)
                for s in eval_seqs:
                    sa, aa, pc, pt = eval_varsort(msign, mmag, H, output_projection_f, s, bp, inj_table)
                    all_sa.append(sa); all_aa.append(aa)
                    all_pc += pc; all_pt += pt

                sort_acc = np.mean(all_sa)
                all_acc = np.mean(all_aa)
                pos_acc = all_pc / np.maximum(all_pt, 1)
                pos_str = ' '.join(f"{a*100:.0f}" for a in pos_acc)

                print(f"  [{step:4d}] sort={sort_acc*100:.1f}% all={all_acc*100:.1f}% "
                      f"edges={edges} A={accepts['add']}|F={accepts['flip']}|M={accepts['mag_resample']} "
                      f"{elapsed:.0f}s")
                print(f"         pos[min,mid,max]: [{pos_str}]%")
                sys.stdout.flush()
    finally:
        pool.terminate(); pool.join()

    elapsed = time.time() - t0
    edges = int((mmag > 0).sum())

    all_sa, all_aa = [], []
    all_pc, all_pt = np.zeros(3), np.zeros(3)
    for s in eval_seqs:
        sa, aa, pc, pt = eval_varsort(msign, mmag, H, output_projection_f, s, bp, inj_table)
        all_sa.append(sa); all_aa.append(aa)
        all_pc += pc; all_pt += pt

    sort_acc = np.mean(all_sa)
    all_acc = np.mean(all_aa)
    pos_acc = all_pc / np.maximum(all_pt, 1)

    print(f"\n{'='*60}")
    print(f"  VARIABLE 3-DIGIT SORTING RESULTS ({BUDGET} steps)")
    print(f"{'='*60}")
    print(f"  Sort accuracy: {sort_acc*100:.2f}% (random=10%)")
    print(f"  All accuracy:  {all_acc*100:.2f}%")
    print(f"  Edges: {edges}")
    print(f"  Time: {elapsed:.0f}s")
    print(f"\n  Per-position accuracy:")
    labels = ['min', 'mid', 'max']
    for i in range(3):
        bar = '#' * int(pos_acc[i] * 50)
        print(f"    [{labels[i]:3s}] {pos_acc[i]*100:5.1f}% {bar}")
    above = sort_acc > 0.10
    print(f"\n  {'>>> ABOVE RANDOM — NETWORK IS SORTING!' if above else '--- At or below random baseline'}")
    print(f"{'='*60}")

    np.savez_compressed(os.path.join(CKPT_DIR, "varsort3_final.npz"),
        msign=msign, mmag=mmag, inj_table=inj_table,
        output_projection_int8=output_projection_int8)
    sys.stdout.flush()

"""
INSTNCT Framed Sorting — chatbot format + thinking tokens
==========================================================
A: [352]{235}\n         — no thinking time
B: [352].....{235}\n    — 5 thinking tokens (40 extra ticks)

Frame tokens give the network context (input vs output phase).
Thinking tokens give charge dynamics time to settle.

2000 steps each, per-position accuracy on sorted digits.
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


OPEN_BRACE = ord('{')
CLOSE_BRACE = ord('}')

def eval_framed_sort(msign, mmag, H, output_projection_f, text_bytes, bp, inj_table):
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

        # FIX: measure at '{' position (predicting first sorted digit)
        if text_bytes[i] == OPEN_BRACE:
            if pred_byte == actual_next:
                sort_correct += 1
                pos_correct[0] += 1
            pos_total[0] += 1
            output_pos = 1
        elif output_pos >= 1 and output_pos < 3:
            if actual_next == CLOSE_BRACE:
                output_pos = -1
            else:
                if pred_byte == actual_next:
                    sort_correct += 1
                    pos_correct[output_pos] += 1
                pos_total[output_pos] += 1
                sort_total += 1
                output_pos += 1
        elif output_pos >= 3:
            output_pos = -1

    sort_total += int(pos_total[0])  # include pos[0] in sort_total
    sort_acc = sort_correct / sort_total if sort_total else 0
    all_acc = all_correct / all_total if all_total else 0
    return sort_acc, all_acc, pos_correct, pos_total


def generate_corpus(n_pad):
    """Generate 3-digit sorting corpus with frame tokens."""
    examples = []
    for a in range(10):
        for b in range(10):
            for c in range(10):
                digits = [a, b, c]
                inp = ''.join(str(x) for x in digits)
                out = ''.join(str(x) for x in sorted(digits))
                pad = '.' * n_pad
                examples.append(f"[{inp}]{pad}{{{out}}}\n")
    return examples


def build_data(n_pad):
    """Build corpus + bigram for a given padding level."""
    examples = generate_corpus(n_pad)
    random.seed(42)
    random.shuffle(examples)
    corpus_text = ''
    while len(corpus_text) < 80_000:
        random.shuffle(examples)
        corpus_text += ''.join(examples)

    data = np.frombuffer(corpus_text.encode('ascii'), dtype=np.uint8).copy()

    bigram = np.zeros((256, 256), dtype=np.float64)
    for i in range(len(data) - 1):
        bigram[data[i], data[i+1]] += 1
    row_sums = bigram.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    bigram = (bigram / row_sums).astype(np.float32)

    return data, bigram, examples[0]


def run_config(name, n_pad, H, bp, inj_table, output_projection_f, n_steps, n_workers, schedule):
    ALL_DATA, bigram, ex = build_data(n_pad)
    bytes_per_ex = 8 + n_pad  # [XXX]{YYY}\n
    seq_len = max(48, bytes_per_ex * 4)  # at least 4 examples per window

    eval_rng = np.random.RandomState(9999)
    eval_seqs = [ALL_DATA[off:off+seq_len] for off in
                 [eval_rng.randint(0, len(ALL_DATA)-seq_len) for _ in range(20)]]

    print(f"\n{'='*60}")
    print(f"  {name} (pad={n_pad}, {bytes_per_ex} bytes/ex, seq_len={seq_len})")
    print(f"  Example: {ex.strip()}")
    print(f"  Bigram '{{' -> '0'={bigram[OPEN_BRACE, ord('0')]:.3f}, "
          f"'1'={bigram[OPEN_BRACE, ord('1')]:.3f}, "
          f"'5'={bigram[OPEN_BRACE, ord('5')]:.3f}")
    print(f"{'='*60}")
    sys.stdout.flush()

    msign = np.zeros((H, H), dtype=np.bool_)
    mmag = np.zeros((H, H), dtype=np.uint8)
    accepts = {'add': 0, 'flip': 0, 'mag_resample': 0}
    t0 = time.time()

    pool = Pool(n_workers, initializer=init_w,
                initargs=(bp, ALL_DATA, seq_len, 2, output_projection_f, bigram, inj_table))
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

            if step % 200 == 0:
                elapsed = time.time() - t0
                edges = int((mmag > 0).sum())

                all_sa, all_aa = [], []
                all_pc, all_pt = np.zeros(3), np.zeros(3)
                for s in eval_seqs:
                    sa, aa, pc, pt = eval_framed_sort(msign, mmag, H, output_projection_f, s, bp, inj_table)
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
        sa, aa, pc, pt = eval_framed_sort(msign, mmag, H, output_projection_f, s, bp, inj_table)
        all_sa.append(sa); all_aa.append(aa)
        all_pc += pc; all_pt += pt

    sort_acc = np.mean(all_sa)
    all_acc = np.mean(all_aa)
    pos_acc = all_pc / np.maximum(all_pt, 1)

    return {
        'name': name, 'sort': sort_acc, 'all': all_acc, 'edges': edges,
        'time': elapsed, 'pos': pos_acc, 'n_pad': n_pad
    }


if __name__ == "__main__":
    IO = 256; NV = 4; H = IO * NV
    N_WORKERS = 18
    BUDGET = 2000
    SCHEDULE = ['add', 'add', 'flip', 'mag_resample', 'add', 'add']

    bp = make_bp(IO)

    random.seed(42); np.random.seed(42)
    ref = SelfWiringGraph(IO, hidden_ratio=NV, projection_scale=1.0)
    input_projection = ref.input_projection
    output_projection = ref.output_projection

    inj_table = np.clip(bp @ input_projection * 128, -128, 127).astype(np.int8)
    output_projection_int8 = np.clip(output_projection * 128, -128, 127).astype(np.int8)
    output_projection_f = output_projection_int8.astype(np.float32) / 128.0

    print("INSTNCT Framed Sorting A/B Test")
    print(f"H={H}, {N_WORKERS}w, {BUDGET} steps each")
    print(f"Random baseline: ~10%")

    results = []

    # A: No padding
    results.append(run_config("A: No padding", 0, H, bp, inj_table, output_projection_f,
                              BUDGET, N_WORKERS, SCHEDULE))

    # B: 5 thinking tokens
    results.append(run_config("B: 5 thinking", 5, H, bp, inj_table, output_projection_f,
                              BUDGET, N_WORKERS, SCHEDULE))

    # Summary
    print(f"\n{'='*60}")
    print(f"  FRAMED SORTING COMPARISON")
    print(f"{'='*60}")
    print(f"  Random baseline: ~10%")
    print(f"  {'Config':<20} {'Sort%':>6} {'All%':>6} {'Edges':>6} {'Time':>6} {'pos[min,mid,max]'}")
    print(f"  {'-'*20} {'-'*6} {'-'*6} {'-'*6} {'-'*6} {'-'*20}")
    for r in results:
        pos_str = ' '.join(f"{a*100:.0f}" for a in r['pos'])
        print(f"  {r['name']:<20} {r['sort']*100:6.2f} {r['all']*100:6.2f} {r['edges']:6d} {r['time']:5.0f}s [{pos_str}]")

    delta = (results[1]['sort'] - results[0]['sort']) * 100
    print(f"\n  Thinking time effect: {delta:+.2f}%")
    best = max(results, key=lambda x: x['sort'])
    if best['sort'] > 0.10:
        print(f"  >>> {best['name']} ABOVE RANDOM — NETWORK IS SORTING!")
    print(f"{'='*60}")
    sys.stdout.flush()


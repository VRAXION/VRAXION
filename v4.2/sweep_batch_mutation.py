"""
Batch Mutation — counter = how many edges to mutate at once
=============================================================
add=N: add N edges, flip=N: flip N edges, decay=N: resample N decays.
All in one eval. Int8 [0-255]. Quick test vs single-mutation baseline.
18 workers, 8 ticks, bigram 2seq, charge ReLU, theta=0, 500 steps.
"""
import sys, os, time, random
import numpy as np
from multiprocessing import Pool

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "model"))
from graph import SelfWiringGraph

_bp = None; _all_data = None; _seq_len = 200; _n_train = 2
_input_projection = None; _output_projection = None; _bigram = None
_batch_add = 1; _batch_flip = 1; _batch_decay = 1

def init_w(b, d, sl, nt, wi, wo, bg, ba, bf, bd):
    global _bp, _all_data, _seq_len, _n_train, _input_projection, _output_projection, _bigram
    global _batch_add, _batch_flip, _batch_decay
    _bp, _all_data, _seq_len, _n_train = b, d, sl, nt
    _input_projection, _output_projection, _bigram = wi, wo, bg
    _batch_add, _batch_flip, _batch_decay = ba, bf, bd

def make_bp(io_dim, seed=12345):
    rng = np.random.RandomState(seed)
    p = rng.randn(256, io_dim).astype(np.float32)
    p /= np.linalg.norm(p, axis=1, keepdims=True)
    return p

def _eval_bigram(mask, H, decay, seqs):
    rs, cs = np.where(mask != 0)
    sp_vals = mask[rs, cs]
    pat_norm = _bp / (np.linalg.norm(_bp, axis=1, keepdims=True) + 1e-8)
    ret = 1.0 - decay
    total = 0.0
    for text_bytes in seqs:
        state = np.zeros(H, dtype=np.float32)
        charge = np.zeros(H, dtype=np.float32)
        seq_score = 0.0; n = 0
        for i in range(len(text_bytes)-1):
            act = state.copy()
            injection = _bp[text_bytes[i]] @ _input_projection
            for t in range(8):
                if t < 2:
                    act = act + injection
                raw = np.zeros(H, dtype=np.float32)
                if len(rs):
                    np.add.at(raw, cs, act[rs] * sp_vals)
                charge += raw; charge *= ret
                act = np.maximum(charge, 0.0)
                charge = np.maximum(charge, 0.0)
            state = act.copy()
            out = charge @ _output_projection
            out_n = out / (np.linalg.norm(out) + 1e-8)
            sims = out_n @ pat_norm.T
            e = np.exp(sims - sims.max())
            pred = e / e.sum()
            target_dist = _bigram[text_bytes[i]]
            cos = np.dot(pred, target_dist) / (np.linalg.norm(pred) * np.linalg.norm(target_dist) + 1e-8)
            seq_score += cos
            n += 1
        total += seq_score / n if n else 0
    return total / len(seqs)

def worker_eval(args):
    mask_flat, decay, H, seed, proposal_type = args
    rng = random.Random(seed)
    np_rng = np.random.RandomState(seed)
    mask = mask_flat.reshape(H, H)
    new_mask = mask.copy()
    new_decay = decay.copy()

    if proposal_type == 'add':
        for _ in range(_batch_add):
            r = rng.randint(0, H-1); c = rng.randint(0, H-1)
            if r != c and new_mask[r, c] == 0:
                new_mask[r, c] = 0.6 if rng.random() < 0.5 else -0.6
    elif proposal_type == 'flip':
        alive = list(zip(*np.where(mask != 0)))
        if alive:
            for _ in range(min(_batch_flip, len(alive))):
                r, c = alive[rng.randint(0, len(alive)-1)]
                new_mask[r, c] = -new_mask[r, c]
    elif proposal_type == 'decay':
        for _ in range(_batch_decay):
            idx = rng.randint(0, H-1)
            new_decay[idx] = rng.uniform(0.01, 0.50)

    seqs = []
    data_len = len(_all_data)
    for _ in range(_n_train):
        off = np_rng.randint(0, data_len - _seq_len)
        seqs.append(_all_data[off:off+_seq_len])

    old_score = _eval_bigram(mask, H, decay, seqs)
    new_score = _eval_bigram(new_mask, H, new_decay, seqs)

    return {'delta': new_score - old_score, 'type': proposal_type,
            'new_mask_flat': new_mask.flatten() if new_score > old_score else None,
            'new_decay': new_decay if new_score > old_score else None}

def eval_accuracy_classic(mask, H, input_projection, output_projection, decay, text_bytes, bp):
    pat_norm = bp / (np.linalg.norm(bp, axis=1, keepdims=True) + 1e-8)
    rs, cs = np.where(mask != 0); sp_vals = mask[rs, cs]
    ret = 1.0 - decay
    state = np.zeros(H, dtype=np.float32); charge = np.zeros(H, dtype=np.float32)
    correct = 0; total = 0
    for i in range(len(text_bytes)-1):
        act = state.copy()
        injection = bp[text_bytes[i]] @ input_projection
        for t in range(8):
            if t < 2: act = act + injection
            raw = np.zeros(H, dtype=np.float32)
            if len(rs): np.add.at(raw, cs, act[rs] * sp_vals)
            charge += raw; charge *= ret
            act = np.maximum(charge, 0.0)
            charge = np.maximum(charge, 0.0)
        state = act.copy()
        out = charge @ output_projection
        out_n = out / (np.linalg.norm(out) + 1e-8)
        sims = out_n @ pat_norm.T
        if np.argmax(sims) == text_bytes[i+1]: correct += 1
        total += 1
    return correct/total if total else 0


def run_config(name, batch_add, batch_flip, batch_decay,
               bp, ALL_DATA, bigram, eval_seqs, H, input_projection, output_projection,
               max_steps=500, n_workers=18, threshold=0.00005):
    mask = np.zeros((H, H), dtype=np.float32)
    decay_rng = np.random.RandomState(99)
    decay = decay_rng.uniform(0.08, 0.24, H).astype(np.float32)

    schedule = ['add', 'flip', 'flip', 'decay', 'decay', 'decay']

    print(f"\n--- {name} (batch: add={batch_add}, flip={batch_flip}, decay={batch_decay}) ---")
    sys.stdout.flush()

    accepts = {'add': 0, 'flip': 0, 'decay': 0}
    acc_history = []
    t0 = time.time()

    pool = Pool(n_workers, initializer=init_w,
                initargs=(bp, ALL_DATA, 200, 2, input_projection, output_projection, bigram,
                          batch_add, batch_flip, batch_decay))
    try:
        for step in range(1, max_steps+1):
            ptype = schedule[(step-1) % len(schedule)]
            if ptype in ('flip', 'decay') and np.count_nonzero(mask) == 0:
                ptype = 'add'

            mask_flat = mask.flatten()
            args = [(mask_flat, decay.copy(), H,
                     31000+step*50+w, ptype) for w in range(n_workers)]
            results = pool.map(worker_eval, args)

            best_r = max(results, key=lambda x: x['delta'])
            if best_r['delta'] > threshold:
                if best_r['new_mask_flat'] is not None:
                    mask = best_r['new_mask_flat'].reshape(H, H)
                    accepts[best_r['type']] += 1
                if best_r['new_decay'] is not None and best_r['type'] == 'decay':
                    decay = best_r['new_decay']
                    accepts['decay'] += 1

            if step % 100 == 0:
                elapsed = time.time() - t0
                edges = int(np.count_nonzero(mask))
                ea = np.mean([eval_accuracy_classic(mask, H, input_projection, output_projection, decay, s, bp)
                              for s in eval_seqs])
                acc_history.append((step, ea))
                quality = ea / max(edges, 1) * 100

                print(f"  [{step:3d}] acc={ea*100:.2f}% edges={edges} q={quality:.3f}%/e "
                      f"A={accepts['add']}|F={accepts['flip']}|D={accepts['decay']} {elapsed:.0f}s")
                sys.stdout.flush()
    finally:
        pool.terminate(); pool.join()

    edges = int(np.count_nonzero(mask))
    ea = np.mean([eval_accuracy_classic(mask, H, input_projection, output_projection, decay, s, bp)
                  for s in eval_seqs])
    elapsed = time.time() - t0
    quality = ea / max(edges, 1) * 100

    print(f"  FINAL: acc={ea*100:.2f}% edges={edges} q={quality:.3f}%/e {elapsed:.0f}s")
    sys.stdout.flush()
    return {'name': name, 'acc': ea, 'edges': edges, 'quality': quality, 'time': elapsed}


if __name__ == "__main__":
    IO = 256; NV = 4; H = IO * NV
    SelfWiringGraph.NV_RATIO = NV
    bp = make_bp(IO)

    DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "..", "Diamond Code", "data", "traindat", "fineweb_edu.traindat")
    with open(DATA, 'rb') as f:
        ALL_DATA = np.frombuffer(f.read(), dtype=np.uint8)
    print(f"Loaded {len(ALL_DATA)/1e6:.1f} MB text")

    bigram = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "data", "bigram_table.npy"))

    eval_rng = np.random.RandomState(9999)
    eval_seqs = [ALL_DATA[off:off+200] for off in
                 [eval_rng.randint(0, len(ALL_DATA)-200) for _ in range(10)]]

    random.seed(42); np.random.seed(42)
    ref = SelfWiringGraph(IO)
    input_projection = ref.input_projection / ref.INJ_SCALE * 1.0
    output_projection = ref.output_projection / ref.INJ_SCALE * 1.0

    results = []

    # A: Baseline (1/1/1 — current single mutation)
    results.append(run_config("SINGLE 1/1/1", 1, 1, 1,
                              bp, ALL_DATA, bigram, eval_seqs, H, input_projection, output_projection))

    # B: Batch 3 (3 edges per op)
    results.append(run_config("BATCH 3/3/3", 3, 3, 3,
                              bp, ALL_DATA, bigram, eval_seqs, H, input_projection, output_projection))

    # C: Batch 6
    results.append(run_config("BATCH 6/6/6", 6, 6, 6,
                              bp, ALL_DATA, bigram, eval_seqs, H, input_projection, output_projection))

    # D: Asymmetric (more add, less flip/decay)
    results.append(run_config("BATCH 6/2/1", 6, 2, 1,
                              bp, ALL_DATA, bigram, eval_seqs, H, input_projection, output_projection))

    # E: Big batch add, single flip/decay
    results.append(run_config("BATCH 10/1/1", 10, 1, 1,
                              bp, ALL_DATA, bigram, eval_seqs, H, input_projection, output_projection))

    print(f"\n{'='*60}")
    print(f"  SUMMARY -- BATCH MUTATION (500 steps, 8t, bigram)")
    print(f"{'='*60}")
    print(f"  {'Name':<18} {'Acc%':>6} {'Edges':>6} {'Q(%/e)':>8} {'Time':>6}")
    print(f"  {'-'*18} {'-'*6} {'-'*6} {'-'*8} {'-'*6}")
    for r in results:
        print(f"  {r['name']:<18} {r['acc']*100:6.2f} {r['edges']:6d} {r['quality']:8.3f} {r['time']:5.0f}s")
    sys.stdout.flush()

"""
Grow -> Crystal -> Grow with Bigram eval
========================================
Phase 1: 100 steps grow (bigram 2seq, thresh=0.0001)
Phase 2: Crystal (bigram eval, remove edges that don't improve distribution)
Phase 3: 100 steps grow again
Compare: with crystal vs without crystal (200 straight steps)
"""
import sys, os, time, random
import numpy as np
from multiprocessing import Pool

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "model"))
from graph import SelfWiringGraph

_bp = None; _all_data = None; _seq_len = 200; _n_train = 2
_W_in = None; _W_out = None; _bigram = None

def init_w(b, d, sl, nt, wi, wo, bg):
    global _bp, _all_data, _seq_len, _n_train, _W_in, _W_out, _bigram
    _bp, _all_data, _seq_len, _n_train = b, d, sl, nt
    _W_in, _W_out, _bigram = wi, wo, bg

def make_bp(io_dim, seed=12345):
    rng = np.random.RandomState(seed)
    p = rng.randn(256, io_dim).astype(np.float32)
    p /= np.linalg.norm(p, axis=1, keepdims=True)
    return p

def _eval_on_seqs(mask, H, theta, decay, seqs):
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
            for t in range(6):
                if t == 0:
                    act = act + _bp[text_bytes[i]] @ _W_in
                raw = np.zeros(H, dtype=np.float32)
                if len(rs):
                    np.add.at(raw, cs, act[rs] * sp_vals)
                charge += raw; charge *= ret
                act = np.maximum(charge - theta, 0.0)
                charge = np.clip(charge, -1.0, 1.0)
            state = act.copy()
            out = charge @ _W_out
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
    mask_flat, theta, decay, H, seed, proposal_type = args
    rng = random.Random(seed)
    np_rng = np.random.RandomState(seed)
    mask = mask_flat.reshape(H, H)
    new_mask = mask; new_theta = theta

    if proposal_type == 'add':
        r = rng.randint(0, H-1); c = rng.randint(0, H-1)
        if r == c or mask[r, c] != 0:
            return {'delta': -1e9, 'type': 'add'}
        val = 0.6 if rng.random() < 0.5 else -0.6
        new_mask = mask.copy(); new_mask[r, c] = val
    elif proposal_type == 'flip':
        alive = list(zip(*np.where(mask != 0)))
        if not alive:
            return {'delta': -1e9, 'type': 'flip'}
        r, c = alive[rng.randint(0, len(alive)-1)]
        new_mask = mask.copy(); new_mask[r, c] = -mask[r, c]
    elif proposal_type == 'theta':
        idx = rng.randint(0, H-1)
        new_theta = theta.copy()
        new_theta[idx] = max(0.0, min(1.0, theta[idx] + rng.uniform(-0.05, 0.05)))

    seqs = []
    data_len = len(_all_data)
    for _ in range(_n_train):
        off = np_rng.randint(0, data_len - _seq_len)
        seqs.append(_all_data[off:off+_seq_len])

    old_score = _eval_on_seqs(mask, H, theta, decay, seqs)
    new_score = _eval_on_seqs(new_mask, H, new_theta, decay, seqs)

    return {'delta': new_score - old_score, 'type': proposal_type,
            'new_mask_flat': new_mask.flatten() if new_score > old_score else None,
            'new_theta': new_theta if proposal_type == 'theta' else None}

def eval_accuracy_classic(mask, H, W_in, W_out, theta, decay, text_bytes, bp):
    pat_norm = bp / (np.linalg.norm(bp, axis=1, keepdims=True) + 1e-8)
    rs, cs = np.where(mask != 0); sp_vals = mask[rs, cs]
    ret = 1.0 - decay
    state = np.zeros(H, dtype=np.float32); charge = np.zeros(H, dtype=np.float32)
    correct = 0; total = 0
    for i in range(len(text_bytes)-1):
        act = state.copy()
        for t in range(6):
            if t == 0: act = act + bp[text_bytes[i]] @ W_in
            raw = np.zeros(H, dtype=np.float32)
            if len(rs): np.add.at(raw, cs, act[rs] * sp_vals)
            charge += raw; charge *= ret
            act = np.maximum(charge - theta, 0.0)
            charge = np.clip(charge, -1.0, 1.0)
        state = act.copy()
        out = charge @ W_out
        out_n = out / (np.linalg.norm(out) + 1e-8)
        sims = out_n @ pat_norm.T
        if np.argmax(sims) == text_bytes[i+1]: correct += 1
        total += 1
    return correct/total if total else 0

def bigram_crystal(mask, H, theta, decay, W_in, W_out, bp, bigram, ALL_DATA, n_eval_seqs=5):
    """Crystal pruning using bigram cosine eval."""
    eval_rng = np.random.RandomState(7777)
    eval_seqs = [ALL_DATA[off:off+200] for off in
                 [eval_rng.randint(0, len(ALL_DATA)-200) for _ in range(n_eval_seqs)]]

    pat_norm = bp / (np.linalg.norm(bp, axis=1, keepdims=True) + 1e-8)

    def bigram_score():
        rs, cs = np.where(mask != 0)
        sp_vals = mask[rs, cs]
        ret = 1.0 - decay
        total = 0.0
        for text_bytes in eval_seqs:
            state = np.zeros(H, dtype=np.float32)
            charge = np.zeros(H, dtype=np.float32)
            seq_score = 0.0; n = 0
            for i in range(len(text_bytes)-1):
                act = state.copy()
                for t in range(6):
                    if t == 0:
                        act = act + bp[text_bytes[i]] @ W_in
                    raw = np.zeros(H, dtype=np.float32)
                    if len(rs):
                        np.add.at(raw, cs, act[rs] * sp_vals)
                    charge += raw; charge *= ret
                    act = np.maximum(charge - theta, 0.0)
                    charge = np.clip(charge, -1.0, 1.0)
                state = act.copy()
                out = charge @ W_out
                out_n = out / (np.linalg.norm(out) + 1e-8)
                sims = out_n @ pat_norm.T
                e = np.exp(sims - sims.max())
                pred = e / e.sum()
                target_dist = bigram[text_bytes[i]]
                cos = np.dot(pred, target_dist) / (np.linalg.norm(pred) * np.linalg.norm(target_dist) + 1e-8)
                seq_score += cos
                n += 1
            total += seq_score / n if n else 0
        return total / len(eval_seqs)

    score = bigram_score()
    total_removed = 0
    pass_num = 0
    while True:
        alive = list(zip(*np.where(mask != 0)))
        random.shuffle(alive)
        removed_this_pass = 0
        for r, c in alive:
            if mask[r, c] == 0:
                continue
            old_val = mask[r, c]
            mask[r, c] = 0.0
            new_score = bigram_score()
            if new_score >= score - 1e-6:
                score = new_score
                removed_this_pass += 1
                total_removed += 1
            else:
                mask[r, c] = old_val
        pass_num += 1
        remaining = int(np.count_nonzero(mask))
        print(f"    crystal pass {pass_num}: removed {removed_this_pass}, remaining {remaining}")
        sys.stdout.flush()
        if removed_this_pass == 0:
            break
    return total_removed


def grow_phase(name, mask, theta, decay, bp, ALL_DATA, bigram, eval_seqs, H, W_in, W_out,
               n_steps=100, n_workers=18, threshold=0.0001):
    print(f"\n  === {name}: {n_steps} steps ===")
    sys.stdout.flush()

    schedule = ['add', 'add', 'add', 'flip', 'theta', 'add']
    accepts = {'add': 0, 'flip': 0, 'theta': 0}
    t0 = time.time()

    pool = Pool(n_workers, initializer=init_w,
                initargs=(bp, ALL_DATA, 200, 2, W_in, W_out, bigram))
    try:
        for step in range(1, n_steps+1):
            ptype = schedule[(step-1) % len(schedule)]
            if ptype in ('flip', 'theta') and np.count_nonzero(mask) == 0:
                ptype = 'add'
            mask_flat = mask.flatten()
            args = [(mask_flat, theta.copy(), decay.copy(), H,
                     9000+step*50+w, ptype) for w in range(n_workers)]
            results = pool.map(worker_eval, args)

            best_r = max(results, key=lambda x: x['delta'])
            if best_r['delta'] > threshold:
                if best_r['type'] in ('add', 'flip') and best_r['new_mask_flat'] is not None:
                    mask[:] = best_r['new_mask_flat'].reshape(H, H)
                    accepts[best_r['type']] += 1
                elif best_r['type'] == 'theta' and best_r['new_theta'] is not None:
                    theta[:] = best_r['new_theta']
                    accepts['theta'] += 1

            if step % 25 == 0:
                elapsed = time.time() - t0
                edges = int(np.count_nonzero(mask))
                ea = np.mean([eval_accuracy_classic(mask, H, W_in, W_out, theta, decay, s, bp)
                              for s in eval_seqs])
                tot = sum(accepts.values())
                print(f"    [{step:3d}] acc={ea*100:.2f}% edges={edges} "
                      f"accepts={tot} {elapsed:.0f}s")
                sys.stdout.flush()
    finally:
        pool.terminate(); pool.join()

    edges = int(np.count_nonzero(mask))
    ea = np.mean([eval_accuracy_classic(mask, H, W_in, W_out, theta, decay, s, bp)
                  for s in eval_seqs])
    elapsed = time.time() - t0
    print(f"    DONE: acc={ea*100:.2f}% edges={edges} accepts={sum(accepts.values())} {elapsed:.0f}s")
    sys.stdout.flush()
    return ea


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
    proj_rng = np.random.RandomState(np.random.randint(0, 2**31))
    W_in = proj_rng.randn(IO, H).astype(np.float32)
    W_in /= np.linalg.norm(W_in, axis=1, keepdims=True)
    W_out = proj_rng.randn(H, IO).astype(np.float32)
    W_out /= np.linalg.norm(W_out, axis=0, keepdims=True)

    t0_total = time.time()

    # ===== CONFIG A: Grow -> Crystal -> Grow =====
    print(f"\n{'='*60}")
    print(f"  CONFIG A: Grow(100) -> Crystal -> Grow(100)")
    print(f"{'='*60}")

    mask_a = np.zeros((H, H), dtype=np.float32)
    theta_a = np.full(H, 0.03, dtype=np.float32)
    decay_a = np.full(H, 0.15, dtype=np.float32)

    # Phase 1: Grow
    grow_phase("GROW-1", mask_a, theta_a, decay_a, bp, ALL_DATA, bigram, eval_seqs, H, W_in, W_out)

    pre_crystal = int(np.count_nonzero(mask_a))
    pre_acc = np.mean([eval_accuracy_classic(mask_a, H, W_in, W_out, theta_a, decay_a, s, bp) for s in eval_seqs])
    print(f"\n  --- CRYSTAL (bigram eval) ---")
    print(f"  Pre-crystal: {pre_acc*100:.2f}% acc, {pre_crystal} edges")
    sys.stdout.flush()

    t_crystal = time.time()
    removed = bigram_crystal(mask_a, H, theta_a, decay_a, W_in, W_out, bp, bigram, ALL_DATA)
    crystal_time = time.time() - t_crystal

    post_crystal = int(np.count_nonzero(mask_a))
    post_acc = np.mean([eval_accuracy_classic(mask_a, H, W_in, W_out, theta_a, decay_a, s, bp) for s in eval_seqs])
    print(f"  Post-crystal: {post_acc*100:.2f}% acc, {post_crystal} edges "
          f"(removed {removed}, {removed/max(pre_crystal,1)*100:.0f}%) {crystal_time:.0f}s")
    sys.stdout.flush()

    # Phase 3: Grow again
    grow_phase("GROW-2", mask_a, theta_a, decay_a, bp, ALL_DATA, bigram, eval_seqs, H, W_in, W_out,
               n_steps=100)

    final_a_edges = int(np.count_nonzero(mask_a))
    final_a_acc = np.mean([eval_accuracy_classic(mask_a, H, W_in, W_out, theta_a, decay_a, s, bp) for s in eval_seqs])

    # ===== CONFIG B: Straight 200 steps (no crystal) =====
    print(f"\n{'='*60}")
    print(f"  CONFIG B: Grow(200) straight — no crystal")
    print(f"{'='*60}")

    mask_b = np.zeros((H, H), dtype=np.float32)
    theta_b = np.full(H, 0.03, dtype=np.float32)
    decay_b = np.full(H, 0.15, dtype=np.float32)

    grow_phase("GROW-200", mask_b, theta_b, decay_b, bp, ALL_DATA, bigram, eval_seqs, H, W_in, W_out,
               n_steps=200)

    final_b_edges = int(np.count_nonzero(mask_b))
    final_b_acc = np.mean([eval_accuracy_classic(mask_b, H, W_in, W_out, theta_b, decay_b, s, bp) for s in eval_seqs])

    total_time = time.time() - t0_total

    print(f"\n{'='*60}")
    print(f"  SUMMARY — GROW-CRYSTAL-GROW vs STRAIGHT")
    print(f"{'='*60}")
    print(f"  {'Config':<30} {'Acc%':>6} {'Edges':>6}")
    print(f"  {'-'*30} {'-'*6} {'-'*6}")
    print(f"  {'A: Grow->Crystal->Grow':<30} {final_a_acc*100:6.2f} {final_a_edges:6d}")
    print(f"  {'B: Grow 200 straight':<30} {final_b_acc*100:6.2f} {final_b_edges:6d}")
    print(f"\n  Crystal stats: {pre_crystal}->{post_crystal} edges "
          f"({removed} removed, {removed/max(pre_crystal,1)*100:.0f}%)")
    print(f"  Total time: {total_time:.0f}s")
    sys.stdout.flush()

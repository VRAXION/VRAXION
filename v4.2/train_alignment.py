"""
INSTNCT Alignment Training Loop
================================
Grow -> Score -> Prune -> Repeat

Each cycle: grow N steps (threshold=0), 1-pass alignment score,
prune harmful + bottom 20%.

A/B test:
  A: Baseline (threshold=0.00005, 1000 steps, no pruning)
  B: Alignment (2 cycles: grow 500 + prune)
"""
import sys, os, time, random, json
import numpy as np
from multiprocessing import Pool

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "model"))
from graph import SelfWiringGraph

# --- Worker globals ---
_bp = None; _all_data = None; _seq_len = 200; _n_train = 2
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


# --- Alignment scoring (1 forward pass) ---
def compute_edge_alignment(msign, mmag, H, output_projection_f, bp, bigram, inj_table, eval_seqs):
    """Score each edge: does it push output toward the target bigram distribution?

    Per-edge alignment = sum over (bytes, ticks) of:
        act[source] * weight * dot(W_out[target_neuron, :], desired_output_dir[byte])

    1 forward pass, vectorized over edges.
    """
    rs, cs = np.where(mmag > 0)
    n_edges = len(rs)
    if n_edges == 0:
        return np.array([]), rs, cs

    s = msign[rs, cs].astype(np.float32) * 2 - 1
    sp_vals = s * mmag[rs, cs].astype(np.float32) / 128.0

    # target output direction for each input byte
    target_vecs = bigram @ bp  # (256, IO)
    target_norms = target_vecs / (np.linalg.norm(target_vecs, axis=1, keepdims=True) + 1e-8)

    # per-neuron, per-byte alignment: how much neuron c's W_out pushes toward target
    w_align = output_projection_f @ target_norms.T  # (H, 256)

    ret = 217.0 / 256.0
    edge_scores = np.zeros(n_edges, dtype=np.float64)
    n_positions = 0

    for text_bytes in eval_seqs:
        state = np.zeros(H, dtype=np.float32)
        charge = np.zeros(H, dtype=np.float32)
        for i in range(len(text_bytes)-1):
            act = state.copy()
            byte_align = w_align[:, text_bytes[i]]  # (H,)
            for t in range(8):
                if t < 2:
                    act = act + inj_table[text_bytes[i]].astype(np.float32) / 128.0
                # per-edge alignment contribution
                edge_act = act[rs] * sp_vals
                edge_scores += edge_act * byte_align[cs]
                # normal forward
                raw = np.zeros(H, dtype=np.float32)
                if n_edges:
                    np.add.at(raw, cs, edge_act)
                charge += raw; charge *= ret
                act = np.maximum(charge, 0.0)
                charge = np.maximum(charge, 0.0)
            state = act.copy()
            n_positions += 1

    if n_positions > 0:
        edge_scores /= n_positions
    return edge_scores, rs, cs


# --- Eval accuracy ---
def eval_accuracy(msign, mmag, H, output_projection_f, text_bytes, bp, inj_table):
    pat_norm = bp / (np.linalg.norm(bp, axis=1, keepdims=True) + 1e-8)
    rs, cs = np.where(mmag > 0)
    s = msign[rs, cs].astype(np.float32) * 2 - 1
    sp_vals = s * mmag[rs, cs].astype(np.float32) / 128.0
    ret = 217.0 / 256.0
    state = np.zeros(H, dtype=np.float32); charge = np.zeros(H, dtype=np.float32)
    correct = 0; total = 0
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
        if np.argmax(sims) == text_bytes[i+1]: correct += 1
        total += 1
    return correct / total if total else 0


# --- Grow phase ---
def grow_phase(msign, mmag, H, bp, ALL_DATA, bigram, output_projection_f, inj_table,
               eval_seqs, n_steps, threshold, n_workers, schedule, seed_offset):
    accepts = {'add': 0, 'flip': 0, 'mag_resample': 0}
    pool = Pool(n_workers, initializer=init_w,
                initargs=(bp, ALL_DATA, 200, 2, output_projection_f, bigram, inj_table))
    t0 = time.time()
    try:
        for step in range(1, n_steps+1):
            ptype = schedule[(step-1) % len(schedule)]
            if ptype in ('flip', 'mag_resample') and (mmag > 0).sum() == 0:
                ptype = 'add'

            args = [(msign.flatten(), mmag.flatten(), H,
                     seed_offset+step*50+w, ptype) for w in range(n_workers)]
            results = pool.map(worker_eval, args)
            best = max(results, key=lambda x: x['delta'])
            if best['delta'] > threshold and best.get('new_s') is not None:
                msign = best['new_s'].reshape(H, H)
                mmag = best['new_m'].reshape(H, H)
                accepts[best['type']] += 1

            if step % 100 == 0:
                elapsed = time.time() - t0
                edges = int((mmag > 0).sum())
                print(f"    [{step:4d}/{n_steps}] edges={edges} "
                      f"A={accepts['add']}|F={accepts['flip']}|M={accepts['mag_resample']} "
                      f"{elapsed:.0f}s")
                sys.stdout.flush()
    finally:
        pool.terminate(); pool.join()
    elapsed = time.time() - t0
    return msign, mmag, accepts, elapsed


# --- Main ---
if __name__ == "__main__":
    IO = 256; NV = 4; H = IO * NV
    N_WORKERS = 18
    STEPS_PER_CYCLE = 500
    N_CYCLES = 2
    SCHEDULE = ['add', 'add', 'flip', 'mag_resample', 'add', 'add']

    SelfWiringGraph.NV_RATIO = NV
    bp = make_bp(IO)
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    DATA = os.path.join(BASE_DIR, "..", "Diamond Code", "data", "traindat", "fineweb_edu.traindat")
    with open(DATA, 'rb') as f:
        ALL_DATA = np.frombuffer(f.read(), dtype=np.uint8)
    bigram = np.load(os.path.join(BASE_DIR, "data", "bigram_table.npy"))

    eval_rng = np.random.RandomState(9999)
    eval_seqs = [ALL_DATA[off:off+200] for off in
                 [eval_rng.randint(0, len(ALL_DATA)-200) for _ in range(10)]]

    random.seed(42); np.random.seed(42)
    ref = SelfWiringGraph(IO)
    input_projection = ref.input_projection / ref.INJ_SCALE * 1.0
    output_projection = ref.output_projection / ref.INJ_SCALE * 1.0

    inj_table = np.clip(bp @ input_projection * 128, -128, 127).astype(np.int8)
    output_projection_int8 = np.clip(output_projection * 128, -128, 127).astype(np.int8)
    output_projection_f = output_projection_int8.astype(np.float32) / 128.0

    CKPT_DIR = os.path.join(BASE_DIR, "checkpoints")
    os.makedirs(CKPT_DIR, exist_ok=True)
    LOG = os.path.join(BASE_DIR, "alignment_live.txt")

    print(f"Loaded {len(ALL_DATA)/1e6:.1f} MB text")
    print(f"H={H}, {N_WORKERS}w, sign+mag, int8 pipeline, ret=217")
    print(f"Schedule: {SCHEDULE}")

    # ============================================================
    # A: BASELINE -- 1000 steps, threshold=0.00005, no pruning
    # ============================================================
    print(f"\n{'='*60}")
    print(f"  A: BASELINE (1000 steps, thresh=0.00005)")
    print(f"{'='*60}")
    sys.stdout.flush()

    msign_a = np.zeros((H, H), dtype=np.bool_)
    mmag_a = np.zeros((H, H), dtype=np.uint8)

    msign_a, mmag_a, acc_a, elapsed_a = grow_phase(
        msign_a, mmag_a, H, bp, ALL_DATA, bigram, output_projection_f, inj_table,
        eval_seqs, n_steps=1000, threshold=0.00005, n_workers=N_WORKERS,
        schedule=SCHEDULE, seed_offset=10000)

    edges_a = int((mmag_a > 0).sum())
    ea_a = np.mean([eval_accuracy(msign_a, mmag_a, H, output_projection_f, s, bp, inj_table)
                    for s in eval_seqs])
    qa_a = ea_a / max(edges_a, 1) * 100

    print(f"\n  BASELINE RESULT: acc={ea_a*100:.2f}% edges={edges_a} q={qa_a:.3f}%/e "
          f"A={acc_a['add']}|F={acc_a['flip']}|M={acc_a['mag_resample']} {elapsed_a:.0f}s")
    sys.stdout.flush()

    np.savez_compressed(os.path.join(CKPT_DIR, "align_baseline.npz"),
        msign=msign_a, mmag=mmag_a, inj_table=inj_table,
        output_projection_int8=output_projection_int8)

    # ============================================================
    # B: ALIGNMENT LOOP -- N cycles of (grow + score + prune)
    # ============================================================
    print(f"\n{'='*60}")
    print(f"  B: ALIGNMENT LOOP ({N_CYCLES} x {STEPS_PER_CYCLE} grow + prune)")
    print(f"{'='*60}")
    sys.stdout.flush()

    msign_b = np.zeros((H, H), dtype=np.bool_)
    mmag_b = np.zeros((H, H), dtype=np.uint8)
    barrier = 0.0
    t0_b = time.time()

    with open(LOG, "w") as f:
        f.write(f"--- ALIGNMENT TRAINING {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")

    for cycle in range(1, N_CYCLES+1):
        print(f"\n  --- Cycle {cycle}/{N_CYCLES} ---")

        # GROW
        print(f"  Growing {STEPS_PER_CYCLE} steps (threshold=0)...")
        sys.stdout.flush()
        msign_b, mmag_b, acc_b, grow_time = grow_phase(
            msign_b, mmag_b, H, bp, ALL_DATA, bigram, output_projection_f, inj_table,
            eval_seqs, n_steps=STEPS_PER_CYCLE, threshold=0.0, n_workers=N_WORKERS,
            schedule=SCHEDULE, seed_offset=20000+cycle*10000)

        edges_pre = int((mmag_b > 0).sum())
        ea_pre = np.mean([eval_accuracy(msign_b, mmag_b, H, output_projection_f, s, bp, inj_table)
                          for s in eval_seqs])
        print(f"  PRE-PRUNE: acc={ea_pre*100:.2f}% edges={edges_pre} "
              f"A={acc_b['add']}|F={acc_b['flip']}|M={acc_b['mag_resample']} {grow_time:.0f}s")
        sys.stdout.flush()

        # SCORE (use 3 eval seqs for speed)
        print(f"  Scoring edges (1-pass alignment)...")
        sys.stdout.flush()
        t_score = time.time()
        edge_scores, e_rs, e_cs = compute_edge_alignment(
            msign_b, mmag_b, H, output_projection_f, bp, bigram, inj_table, eval_seqs[:3])
        score_time = time.time() - t_score

        if len(edge_scores) == 0:
            print(f"  No edges to score, skipping prune")
            continue

        n_harmful = int((edge_scores < 0).sum())
        n_helpful = int((edge_scores > 0).sum())
        print(f"  Scores: harmful={n_harmful} helpful={n_helpful} ({score_time:.1f}s)")

        # Distribution
        if len(edge_scores) > 10:
            pcts = np.percentile(edge_scores, [10, 25, 50, 75, 90])
            print(f"  Distribution: p10={pcts[0]:.6f} p25={pcts[1]:.6f} "
                  f"med={pcts[2]:.6f} p75={pcts[3]:.6f} p90={pcts[4]:.6f}")

        # PRUNE harmful
        harmful = edge_scores < 0
        mmag_b[e_rs[harmful], e_cs[harmful]] = 0
        msign_b[e_rs[harmful], e_cs[harmful]] = False

        # PRUNE bottom 20% of remaining
        keep = ~harmful
        n_weak = 0
        if keep.sum() > 0:
            kept_scores = edge_scores[keep]
            kept_rs = e_rs[keep]
            kept_cs = e_cs[keep]
            cutoff_20 = np.percentile(kept_scores, 20)
            weak = kept_scores < cutoff_20
            n_weak = int(weak.sum())
            mmag_b[kept_rs[weak], kept_cs[weak]] = 0
            msign_b[kept_rs[weak], kept_cs[weak]] = False

            # Update barrier = 10th percentile of survivors
            final_scores = kept_scores[~weak]
            if len(final_scores) > 0:
                barrier = float(np.percentile(final_scores, 10))

        print(f"  PRUNED: {n_harmful} harmful + {n_weak} weak = {n_harmful + n_weak} removed")

        edges_post = int((mmag_b > 0).sum())
        ea_post = np.mean([eval_accuracy(msign_b, mmag_b, H, output_projection_f, s, bp, inj_table)
                           for s in eval_seqs])
        q_post = ea_post / max(edges_post, 1) * 100
        print(f"  POST-PRUNE: acc={ea_post*100:.2f}% edges={edges_post} "
              f"q={q_post:.3f}%/e barrier={barrier:.6f}")
        sys.stdout.flush()

        # Log
        line = (f"Cycle {cycle}: pre={edges_pre}e/{ea_pre*100:.2f}% -> "
                f"post={edges_post}e/{ea_post*100:.2f}% q={q_post:.3f} barrier={barrier:.6f}")
        with open(LOG, "a") as f:
            f.write(line + "\n")

        # Save checkpoint
        np.savez_compressed(os.path.join(CKPT_DIR, f"align_cycle{cycle}.npz"),
            msign=msign_b, mmag=mmag_b, inj_table=inj_table,
            output_projection_int8=output_projection_int8)

    total_time_b = time.time() - t0_b
    edges_b = int((mmag_b > 0).sum())
    ea_b = np.mean([eval_accuracy(msign_b, mmag_b, H, output_projection_f, s, bp, inj_table)
                    for s in eval_seqs])
    qa_b = ea_b / max(edges_b, 1) * 100

    # ============================================================
    # COMPARISON
    # ============================================================
    print(f"\n{'='*60}")
    print(f"  ALIGNMENT vs BASELINE ({N_CYCLES}x{STEPS_PER_CYCLE} = {N_CYCLES*STEPS_PER_CYCLE} total steps)")
    print(f"{'='*60}")
    print(f"  {'Method':<20} {'Acc%':>6} {'Edges':>6} {'Q(%/e)':>8} {'Time':>6}")
    print(f"  {'-'*20} {'-'*6} {'-'*6} {'-'*8} {'-'*6}")
    print(f"  {'Baseline':<20} {ea_a*100:6.2f} {edges_a:6d} {qa_a:8.3f} {elapsed_a:5.0f}s")
    print(f"  {'Alignment loop':<20} {ea_b*100:6.2f} {edges_b:6d} {qa_b:8.3f} {total_time_b:5.0f}s")
    print(f"  {'Barrier':<20} {barrier:.6f}")
    delta = (ea_b - ea_a) * 100
    print(f"  {'Delta':<20} {delta:+.2f}%")
    print(f"{'='*60}")
    sys.stdout.flush()

    with open(LOG, "a") as f:
        f.write(f"\n--- FINAL ---\n")
        f.write(f"Baseline: {ea_a*100:.2f}% {edges_a}e {qa_a:.3f}q {elapsed_a:.0f}s\n")
        f.write(f"Alignment: {ea_b*100:.2f}% {edges_b}e {qa_b:.3f}q {total_time_b:.0f}s\n")
        f.write(f"Delta: {delta:+.2f}%\n")

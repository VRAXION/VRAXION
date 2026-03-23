"""
Dense Init → Crystallize Prune Smoke Test
==========================================
Hypothesis: start from a DENSE random graph (4% density = ~40K edges on 1024n)
and prune down. Multi-hop paths already exist — pruning should discover them
by removing harmful edges while keeping useful loops.

Conditions tested:
  A) Empty init → grow (current canonical approach, 200 steps)
  B) Dense 4% init → crystallize prune (log-likelihood loss)
  C) Dense 4% init → crystallize prune (bigram cosine)
  D) Dense 4% init → random thin to 2% → crystallize

Uses alice.txt. 256 neurons for speed. All 4 cores.
Saves checkpoints after each phase.
"""
import sys, os, time, random
import numpy as np
from pathlib import Path
from multiprocessing import Pool

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "model"))
from graph import SelfWiringGraph

# ── Globals ─────────────────────────────────────────────────────────
IO = 256
H = 256  # hidden_ratio=1, fast smoke test
TICKS = 8
INJ_TICKS = 2
SEED = 42
CKPT_DIR = Path(__file__).resolve().parent / "checkpoints" / "dense_crystal"

def make_bp(io_dim, seed=12345):
    rng = np.random.RandomState(seed)
    p = rng.randn(256, io_dim).astype(np.float32)
    p /= np.linalg.norm(p, axis=1, keepdims=True)
    return p

def compute_bigram_from_bytes(data_bytes):
    bigram = np.zeros((256, 256), dtype=np.float64)
    for i in range(len(data_bytes) - 1):
        bigram[data_bytes[i], data_bytes[i + 1]] += 1
    row_sums = bigram.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    bigram /= row_sums
    return bigram.astype(np.float32)

def forward_pass(mask, theta, decay, text_bytes, bp, input_proj, output_proj, ticks=TICKS):
    """Run forward, return (charge_out_sequence, predictions)."""
    rs, cs = np.where(mask != 0)
    sp_vals = mask[rs, cs]
    pat_norm = bp / (np.linalg.norm(bp, axis=1, keepdims=True) + 1e-8)
    ret = 1.0 - decay
    state = np.zeros(H, dtype=np.float32)
    charge = np.zeros(H, dtype=np.float32)
    preds = []
    for i in range(len(text_bytes) - 1):
        act = state.copy()
        for t in range(ticks):
            if t < INJ_TICKS:
                act = act + bp[text_bytes[i]] @ input_proj
            raw = np.zeros(H, dtype=np.float32)
            if len(rs):
                np.add.at(raw, cs, act[rs] * sp_vals)
            charge += raw
            charge *= ret
            np.clip(charge, -10.0, 10.0, out=charge)  # prevent overflow in dense graphs
            act = np.maximum(charge, 0.0)
            charge = np.maximum(charge, 0.0)
        state = act.copy()
        out = charge @ output_proj
        out_n = out / (np.linalg.norm(out) + 1e-8)
        sims = out_n @ pat_norm.T
        e = np.exp(sims - sims.max())
        probs = e / e.sum()
        preds.append(probs)
    return preds

def eval_accuracy(mask, theta, decay, text_bytes, bp, input_proj, output_proj):
    preds = forward_pass(mask, theta, decay, text_bytes, bp, input_proj, output_proj)
    correct = 0
    for i, probs in enumerate(preds):
        if np.argmax(probs) == text_bytes[i + 1]:
            correct += 1
    return correct / len(preds) if preds else 0

def eval_loglik(mask, theta, decay, seqs, bp, input_proj, output_proj):
    """Average log-likelihood across sequences."""
    total = 0.0
    for text_bytes in seqs:
        preds = forward_pass(mask, theta, decay, text_bytes, bp, input_proj, output_proj)
        ll = 0.0
        for i, probs in enumerate(preds):
            ll += np.log(probs[text_bytes[i + 1]] + 1e-10)
        total += ll / len(preds) if preds else 0
    return total / len(seqs)

def eval_bigram_cos(mask, theta, decay, seqs, bp, input_proj, output_proj, bigram):
    """Average bigram cosine similarity."""
    total = 0.0
    for text_bytes in seqs:
        preds = forward_pass(mask, theta, decay, text_bytes, bp, input_proj, output_proj)
        cos_sum = 0.0
        for i, pred in enumerate(preds):
            target_dist = bigram[text_bytes[i]]
            cos = np.dot(pred, target_dist) / (np.linalg.norm(pred) * np.linalg.norm(target_dist) + 1e-8)
            cos_sum += cos
        total += cos_sum / len(preds) if preds else 0
    return total / len(seqs)

def network_stats(mask):
    """Return connectivity stats."""
    edges = int(np.count_nonzero(mask))
    in_deg = np.count_nonzero(mask, axis=0)
    out_deg = np.count_nonzero(mask, axis=1)
    connected = (in_deg + out_deg) > 0
    n_connected = int(connected.sum())
    max_deg = int(max(in_deg.max(), out_deg.max())) if edges > 0 else 0
    # Check for loops: does any neuron have both in and out edges?
    has_in = in_deg > 0
    has_out = out_deg > 0
    n_bidirectional = int((has_in & has_out).sum())
    return {
        'edges': edges, 'connected': n_connected, 'max_deg': max_deg,
        'bidirectional': n_bidirectional,
        'density_pct': round(edges / (H * H) * 100, 2),
    }

def crystallize_with_scoring(net, score_fn, eps=0.0, verbose=True):
    """Crystallize using arbitrary score function. Returns edges removed."""
    score = score_fn()
    total_removed = 0
    pass_num = 0
    while True:
        alive_snapshot = list(net.alive)
        random.shuffle(alive_snapshot)
        removed_this_pass = 0
        for r, c in alive_snapshot:
            if net.mask[r, c] == 0:
                continue
            old_val = net.mask[r, c]
            net.mask[r, c] = 0.0
            net.alive_set.discard((r, c))
            new_score = score_fn()
            if new_score >= score - eps:
                score = new_score
                removed_this_pass += 1
                total_removed += 1
            else:
                net.mask[r, c] = old_val
                net.alive_set.add((r, c))
        net.resync_alive()
        pass_num += 1
        stats = network_stats(net.mask)
        if verbose:
            print(f"    pass {pass_num}: removed {removed_this_pass}, "
                  f"remaining {stats['edges']} edges, "
                  f"{stats['connected']} connected neurons, "
                  f"{stats['bidirectional']} bidirectional, "
                  f"score={score:.4f}")
        if removed_this_pass == 0:
            break
    return total_removed

# ── Grow-from-empty worker (condition A baseline) ──────────────────
_bp = None; _all_data = None; _bigram = None
_input_proj = None; _output_proj = None

def init_grow_w(bp, data, bg, ip, op):
    global _bp, _all_data, _bigram, _input_proj, _output_proj
    _bp, _all_data, _bigram = bp, data, bg
    _input_proj, _output_proj = ip, op

def _worker_eval_bigram(mask, theta, decay, seqs):
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
            for t in range(TICKS):
                if t < INJ_TICKS:
                    act = act + _bp[text_bytes[i]] @ _input_proj
                raw = np.zeros(H, dtype=np.float32)
                if len(rs):
                    np.add.at(raw, cs, act[rs] * sp_vals)
                charge += raw; charge *= ret
                np.clip(charge, -10.0, 10.0, out=charge)
                act = np.maximum(charge, 0.0)
                charge = np.maximum(charge, 0.0)
            state = act.copy()
            out = charge @ _output_proj
            out_n = out / (np.linalg.norm(out) + 1e-8)
            sims = out_n @ pat_norm.T
            e = np.exp(sims - sims.max())
            pred = e / e.sum()
            target_dist = _bigram[text_bytes[i]]
            cos = np.dot(pred, target_dist) / (np.linalg.norm(pred) * np.linalg.norm(target_dist) + 1e-8)
            seq_score += cos; n += 1
        total += seq_score / n if n else 0
    return total / len(seqs)

def grow_worker(args):
    mask_flat, theta, decay, seed, ptype = args
    rng = random.Random(seed)
    np_rng = np.random.RandomState(seed)
    mask = mask_flat.reshape(H, H)
    new_mask = mask; new_decay = decay

    if ptype == 'add':
        r = rng.randint(0, H-1); c = rng.randint(0, H-1)
        if r == c or mask[r, c] != 0:
            return {'delta': -1e9, 'type': 'add'}
        val = 0.6 if rng.random() < 0.5 else -0.6
        new_mask = mask.copy(); new_mask[r, c] = val
    elif ptype == 'flip':
        alive = list(zip(*np.where(mask != 0)))
        if not alive:
            return {'delta': -1e9, 'type': 'flip'}
        r, c = alive[rng.randint(0, len(alive)-1)]
        new_mask = mask.copy(); new_mask[r, c] = -mask[r, c]
    elif ptype == 'decay':
        idx = rng.randint(0, H-1)
        new_decay = decay.copy()
        new_decay[idx] = max(0.01, min(0.5, decay[idx] + rng.uniform(-0.03, 0.03)))

    seqs = []
    data_len = len(_all_data)
    for _ in range(2):
        off = np_rng.randint(0, data_len - 200)
        seqs.append(_all_data[off:off+200])

    old_score = _worker_eval_bigram(mask, theta, decay, seqs)
    new_score = _worker_eval_bigram(new_mask, theta, new_decay, seqs)

    return {
        'delta': new_score - old_score, 'type': ptype,
        'new_mask_flat': new_mask.flatten() if ptype in ('add', 'flip') and new_score > old_score else None,
        'new_decay': new_decay if ptype == 'decay' else None,
    }


def run_grow_from_empty(all_data, bigram, bp, input_proj, output_proj, eval_seqs, budget=200):
    """Condition A: canonical grow-from-empty."""
    print(f"\n{'='*60}")
    print(f"  Condition A: Grow from empty (canonical, {budget} steps)")
    print(f"{'='*60}")

    random.seed(SEED); np.random.seed(SEED)
    net = SelfWiringGraph(IO, hidden_ratio=1, projection_scale=1.0, seed=SEED)
    net.mask[:] = 0; net.alive = []; net.alive_set = set(); net._sync_sparse_idx()
    net.theta[:] = 0.0
    decay_rng = np.random.RandomState(99)
    net.decay[:] = decay_rng.uniform(0.08, 0.24, H).astype(np.float32)

    schedule = ['add', 'add', 'flip', 'decay', 'decay', 'decay', 'decay', 'decay']
    n_workers = 4
    threshold = 0.00005

    pool = Pool(n_workers, initializer=init_grow_w,
                initargs=(bp, all_data, bigram, input_proj, output_proj))
    t0 = time.time()
    add_acc = 0; flip_acc = 0; decay_acc = 0

    try:
        for step in range(1, budget + 1):
            ptype = schedule[(step - 1) % len(schedule)]
            if ptype in ('flip', 'decay') and net.count_connections() == 0:
                ptype = 'add'

            mask_flat = net.mask.flatten()
            args = [(mask_flat, net.theta.copy(), net.decay.copy(),
                     1000 + step * 50 + w, ptype) for w in range(n_workers)]
            results = pool.map(grow_worker, args)

            best_r = max(results, key=lambda x: x['delta'])
            if best_r['delta'] > threshold:
                if best_r['type'] in ('add', 'flip') and best_r['new_mask_flat'] is not None:
                    net.mask[:] = best_r['new_mask_flat'].reshape(H, H)
                    net.resync_alive()
                    if best_r['type'] == 'add': add_acc += 1
                    else: flip_acc += 1
                elif best_r['type'] == 'decay' and best_r['new_decay'] is not None:
                    net.decay[:] = best_r['new_decay']
                    decay_acc += 1

            if step % 50 == 0:
                ea = np.mean([eval_accuracy(net.mask, net.theta, net.decay, s, bp,
                              input_proj, output_proj) for s in eval_seqs])
                stats = network_stats(net.mask)
                print(f"  [{step:4d}] acc={ea*100:.2f}% edges={stats['edges']} "
                      f"connected={stats['connected']} bidir={stats['bidirectional']} "
                      f"[A={add_acc}|F={flip_acc}|D={decay_acc}]")
    finally:
        pool.terminate(); pool.join()

    # Save checkpoint
    net.input_projection = input_proj
    net.output_projection = output_proj
    ckpt = CKPT_DIR / "condA_grow_final.npz"
    net.save(str(ckpt))

    final_acc = np.mean([eval_accuracy(net.mask, net.theta, net.decay, s, bp,
                         input_proj, output_proj) for s in eval_seqs])
    final_ll = eval_loglik(net.mask, net.theta, net.decay, eval_seqs, bp, input_proj, output_proj)
    stats = network_stats(net.mask)
    elapsed = time.time() - t0

    print(f"  FINAL: acc={final_acc*100:.2f}% ll={final_ll:.3f} "
          f"edges={stats['edges']} connected={stats['connected']} "
          f"bidir={stats['bidirectional']} {elapsed:.1f}s")
    print(f"  Saved: {ckpt}")

    return {'acc': round(final_acc * 100, 2), 'll': round(final_ll, 3),
            **stats, 'time': round(elapsed, 1)}


def run_dense_crystallize(all_data, bigram, bp, input_proj, output_proj, eval_seqs,
                          label, density, score_mode='loglik', thin_to=None):
    """Conditions B/C/D: dense init → (optional thin) → crystallize."""
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"  density={density}%, score={score_mode}"
          + (f", thin_to={thin_to}%" if thin_to else ""))
    print(f"{'='*60}")

    random.seed(SEED); np.random.seed(SEED)
    net = SelfWiringGraph(IO, hidden_ratio=1, density=density, projection_scale=1.0, seed=SEED)
    net.theta[:] = 0.0
    decay_rng = np.random.RandomState(99)
    net.decay[:] = decay_rng.uniform(0.08, 0.24, H).astype(np.float32)
    # Use same projections
    net.input_projection = input_proj
    net.output_projection = output_proj

    stats = network_stats(net.mask)
    init_acc = np.mean([eval_accuracy(net.mask, net.theta, net.decay, s, bp,
                        input_proj, output_proj) for s in eval_seqs])
    init_ll = eval_loglik(net.mask, net.theta, net.decay, eval_seqs, bp, input_proj, output_proj)
    print(f"  INIT: acc={init_acc*100:.2f}% ll={init_ll:.3f} "
          f"edges={stats['edges']} connected={stats['connected']} "
          f"bidir={stats['bidirectional']} density={stats['density_pct']}%")

    # Optional: thin randomly first
    if thin_to is not None:
        target_edges = int(H * H * thin_to / 100)
        current = list(net.alive)
        random.shuffle(current)
        remove_count = len(current) - target_edges
        if remove_count > 0:
            for r, c in current[:remove_count]:
                net.mask[r, c] = 0.0
            net.resync_alive()
        stats = network_stats(net.mask)
        thin_acc = np.mean([eval_accuracy(net.mask, net.theta, net.decay, s, bp,
                            input_proj, output_proj) for s in eval_seqs])
        thin_ll = eval_loglik(net.mask, net.theta, net.decay, eval_seqs, bp, input_proj, output_proj)
        print(f"  AFTER THIN: acc={thin_acc*100:.2f}% ll={thin_ll:.3f} "
              f"edges={stats['edges']} connected={stats['connected']} "
              f"bidir={stats['bidirectional']}")

    # Save pre-crystal checkpoint
    pre_ckpt = CKPT_DIR / f"{label.split(':')[0].strip().replace(' ', '_')}_pre_crystal.npz"
    net.save(str(pre_ckpt))

    # Define scoring function
    # Use more eval sequences for crystal stability
    crystal_seqs = eval_seqs[:5]
    if score_mode == 'loglik':
        def score_fn():
            return eval_loglik(net.mask, net.theta, net.decay, crystal_seqs, bp,
                               input_proj, output_proj)
    elif score_mode == 'bigram':
        def score_fn():
            return eval_bigram_cos(net.mask, net.theta, net.decay, crystal_seqs, bp,
                                   input_proj, output_proj, bigram)

    # Crystallize!
    t0 = time.time()
    print(f"  Crystallizing ({score_mode})...")
    removed = crystallize_with_scoring(net, score_fn, eps=0.0, verbose=True)
    elapsed = time.time() - t0

    # Save post-crystal checkpoint
    post_ckpt = CKPT_DIR / f"{label.split(':')[0].strip().replace(' ', '_')}_post_crystal.npz"
    net.save(str(post_ckpt))

    final_acc = np.mean([eval_accuracy(net.mask, net.theta, net.decay, s, bp,
                         input_proj, output_proj) for s in eval_seqs])
    final_ll = eval_loglik(net.mask, net.theta, net.decay, eval_seqs, bp, input_proj, output_proj)
    stats = network_stats(net.mask)

    print(f"  FINAL: acc={final_acc*100:.2f}% ll={final_ll:.3f} "
          f"edges={stats['edges']} connected={stats['connected']} "
          f"bidir={stats['bidirectional']} removed={removed} {elapsed:.1f}s")
    print(f"  Saved: {post_ckpt}")

    return {'acc': round(final_acc * 100, 2), 'll': round(final_ll, 3),
            **stats, 'time': round(elapsed, 1), 'removed': removed,
            'init_acc': round(init_acc * 100, 2)}


if __name__ == "__main__":
    os.makedirs(CKPT_DIR, exist_ok=True)

    print("Loading alice.txt...")
    with open(DATA_DIR / "alice.txt", "rb") as f:
        all_data = np.frombuffer(f.read(), dtype=np.uint8)
    print(f"  {len(all_data)} bytes")

    print("Computing bigram table...")
    bigram = compute_bigram_from_bytes(all_data)

    bp = make_bp(IO)

    # Deterministic projections (scale=1.0)
    random.seed(SEED); np.random.seed(SEED)
    ref = SelfWiringGraph(IO, hidden_ratio=1, projection_scale=1.0, seed=SEED)
    input_proj = ref.input_projection
    output_proj = ref.output_projection

    # Fixed eval sequences
    eval_rng = np.random.RandomState(9999)
    eval_seqs = [all_data[off:off + 200]
                 for off in [eval_rng.randint(0, len(all_data) - 200) for _ in range(10)]]

    results = {}

    # Condition A: grow from empty (canonical baseline)
    results['A_grow'] = run_grow_from_empty(all_data, bigram, bp, input_proj, output_proj, eval_seqs)

    # Note: SelfWiringGraph density: >1.0 is divided by 100 (so 4 → 4%)
    #       ≤1.0 is treated as raw fraction (so 0.5 → 50%!)
    # Use >1.0 values to get percentage interpretation.

    # Condition B: dense 2% (~1300 edges) → crystallize with log-likelihood
    results['B_d2_ll'] = run_dense_crystallize(
        all_data, bigram, bp, input_proj, output_proj, eval_seqs,
        label="B: Dense 2% → crystal (loglik)", density=2, score_mode='loglik')

    # Condition C: dense 4% (~2600 edges) → crystallize with log-likelihood
    results['C_d4_ll'] = run_dense_crystallize(
        all_data, bigram, bp, input_proj, output_proj, eval_seqs,
        label="C: Dense 4% → crystal (loglik)", density=4, score_mode='loglik')

    # Condition D: dense 4% → crystallize with bigram cosine
    results['D_d4_bigram'] = run_dense_crystallize(
        all_data, bigram, bp, input_proj, output_proj, eval_seqs,
        label="D: Dense 4% → crystal (bigram)", density=4, score_mode='bigram')

    # Condition E: dense 4% → thin to 1% → crystallize (loglik)
    results['E_d4_thin1_ll'] = run_dense_crystallize(
        all_data, bigram, bp, input_proj, output_proj, eval_seqs,
        label="E: Dense 4% → thin 1% → crystal (ll)", density=4,
        score_mode='loglik', thin_to=1)

    # ── Summary ─────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  FINAL COMPARISON")
    print(f"{'='*70}")
    print(f"  {'Condition':<35} {'Acc%':>6} {'LL':>8} {'Edges':>7} {'Conn':>6} {'Bidir':>6} {'Time':>6}")
    print(f"  {'-'*35} {'-'*6} {'-'*8} {'-'*7} {'-'*6} {'-'*6} {'-'*6}")
    for key, r in results.items():
        init_info = f" (init {r['init_acc']}%)" if 'init_acc' in r else ""
        print(f"  {key:<35} {r['acc']:>6.2f} {r['ll']:>8.3f} {r['edges']:>7} "
              f"{r['connected']:>6} {r['bidirectional']:>6} {r['time']:>5.0f}s{init_info}")

    print(f"\n  Key question: do crystallized networks retain multi-hop paths?")
    for key, r in results.items():
        if r['bidirectional'] > 0:
            print(f"    {key}: {r['bidirectional']} bidirectional neurons (potential loops)")
        else:
            print(f"    {key}: no bidirectional neurons (no loops)")
    print(f"{'='*70}")

"""
A/B test: Loop injection vs Random edges (fair comparison)
==========================================================
From crystallized plateau checkpoint (21.64%, 2-4 edges):

Arms:
  A) BASELINE — continue normal mutation only (control)
  B) RANDOM EDGES — inject N random edges, then continue mutation
  C) LOOP-3 — inject 3-edge loops, then continue mutation
  D) LOOP-5 — inject 5-edge loops, then continue mutation
  E) LOOP-7 — inject 7-edge loops, then continue mutation
  F) MIXED LOOPS — alternate 3,4,5,6,7-edge loops

Each non-baseline arm gets EXACTLY the same total edge budget.
After injection phase, all arms continue with normal mutation.

Tick sweep: test each winner at 8, 16, 24 ticks.

Uses real English data (fineweb), not synthetic.
"""
import sys, os, time, random, copy
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'model'))
from graph import SelfWiringGraph
from lib.data import load_fineweb_bytes

# ── Config ────────────────────────────────────────────────────────────────────
CKPT = ROOT / 'recipes' / 'checkpoints' / 'variant_seed42_crystal.npz'
INJECTION_BUDGET = 30       # total edges injected (same for all non-baseline arms)
TRAINING_STEPS = 2000       # mutation steps after injection
SEQ_LEN = 150
N_TRAIN = 2
N_EVAL = 8
REPORT_EVERY = 200
THRESHOLD = 0.00005
TICKS_DEFAULT = 8
INPUT_DUR = 2
SEED = 42
SCHEDULE = ['add', 'add', 'flip', 'decay', 'decay', 'decay', 'decay', 'decay']

TICK_SWEEP = [8, 16, 24]

H = 1024
IO = 256

# ── Helpers ───────────────────────────────────────────────────────────────────
def make_bp(seed=12345):
    rng = np.random.RandomState(seed)
    p = rng.randn(256, IO).astype(np.float32)
    p /= np.linalg.norm(p, axis=1, keepdims=True)
    return p

def rollout(inj, mask, theta, decay, state, charge, sp, pol, ref, ticks):
    act = state.copy(); cur = charge.copy(); r = ref.copy()
    rows, cols = sp; df = np.asarray(decay, dtype=np.float32)
    is_sc = df.ndim == 0 or df.shape == ()
    dp = max(1, int(round(1.0 / max(float(df), 0.001)))) if is_sc else 0
    for tick in range(ticks):
        if dp > 0:
            if tick % dp == 0: cur = np.maximum(cur - 1.0, 0.0)
        else: cur = np.maximum(cur - df, 0.0)
        if tick < INPUT_DUR: act = act + inj
        raw = np.zeros(H, dtype=np.float32)
        if len(rows): np.add.at(raw, cols, act[rows])
        np.nan_to_num(raw, copy=False); cur += raw
        np.clip(cur, 0.0, 15.0, out=cur)
        can = (r == 0); fired = (cur >= theta) & can
        r[r > 0] -= 1; r[fired] = 1
        act = fired.astype(np.float32) * pol; cur[fired] = 0.0
    return act, cur, r

def get_sparse(mask):
    rows, cols = np.where(mask)
    return rows.astype(np.intp), cols.astype(np.intp)

def eval_score(mask, theta, decay, pol, seqs, bp, ip, op, bigram, ticks):
    sp = get_sparse(mask)
    pn = bp / (np.linalg.norm(bp, axis=1, keepdims=True) + 1e-8)
    tot = 0.0
    for seq in seqs:
        st = np.zeros(H, dtype=np.float32); ch = np.zeros(H, dtype=np.float32)
        rf = np.zeros(H, dtype=np.int8); ss = 0.0; n = 0
        for i in range(len(seq) - 1):
            inj = bp[seq[i]] @ ip
            st, ch, rf = rollout(inj, mask, theta, decay, st, ch, sp, pol, rf, ticks)
            out = ch @ op; on = out / (np.linalg.norm(out) + 1e-8)
            sims = on @ pn.T; e = np.exp(sims - sims.max()); pred = e / e.sum()
            tgt = bigram[seq[i]]
            cos = np.dot(pred, tgt) / (np.linalg.norm(pred) * np.linalg.norm(tgt) + 1e-8)
            ss += cos; n += 1
        tot += ss / n if n else 0.0
    return tot / len(seqs)

def eval_acc(mask, theta, decay, pol, seqs, bp, ip, op, ticks):
    sp = get_sparse(mask)
    pn = bp / (np.linalg.norm(bp, axis=1, keepdims=True) + 1e-8)
    ok = 0; tot = 0
    for seq in seqs:
        st = np.zeros(H, dtype=np.float32); ch = np.zeros(H, dtype=np.float32)
        rf = np.zeros(H, dtype=np.int8)
        for i in range(len(seq) - 1):
            inj = bp[seq[i]] @ ip
            st, ch, rf = rollout(inj, mask, theta, decay, st, ch, sp, pol, rf, ticks)
            out = ch @ op; on = out / (np.linalg.norm(out) + 1e-8)
            if np.argmax(on @ pn.T) == seq[i + 1]: ok += 1
            tot += 1
    return ok / tot if tot else 0.0

# ── Injection functions ───────────────────────────────────────────────────────
def inject_random_edges(mask, n_edges, rng):
    """Add n random edges (no self-connections, no duplicates)."""
    added = 0
    attempts = 0
    while added < n_edges and attempts < n_edges * 20:
        r = rng.randint(0, H)
        c = rng.randint(0, H)
        if r != c and not mask[r, c]:
            mask[r, c] = True
            added += 1
        attempts += 1
    return added

def inject_loops(mask, loop_length, total_budget, rng):
    """Inject loops of given length until budget is used. Return edges added."""
    added = 0
    attempts = 0
    while added < total_budget and attempts < total_budget * 10:
        # Pick random distinct neurons for the loop
        if loop_length > H:
            break
        nodes = rng.choice(H, size=loop_length, replace=False).tolist()
        # Check all edges are free
        edges_to_add = []
        ok = True
        for i in range(loop_length):
            r, c = nodes[i], nodes[(i + 1) % loop_length]
            if mask[r, c]:
                ok = False
                break
            edges_to_add.append((r, c))
        if ok and added + len(edges_to_add) <= total_budget:
            for r, c in edges_to_add:
                mask[r, c] = True
                added += 1
        attempts += 1
    return added

def inject_mixed_loops(mask, total_budget, rng):
    """Alternate loop sizes 3,4,5,6,7 until budget used."""
    added = 0
    sizes = [3, 4, 5, 6, 7]
    idx = 0
    attempts = 0
    while added < total_budget and attempts < total_budget * 10:
        sz = sizes[idx % len(sizes)]
        remaining = total_budget - added
        if sz > remaining:
            sz = remaining
        if sz < 3:
            # Fill remainder with random edges
            added += inject_random_edges(mask, remaining, rng)
            break
        nodes = rng.choice(H, size=sz, replace=False).tolist()
        edges_to_add = []
        ok = True
        for i in range(sz):
            r, c = nodes[i], nodes[(i + 1) % sz]
            if mask[r, c]:
                ok = False
                break
            edges_to_add.append((r, c))
        if ok and added + len(edges_to_add) <= total_budget:
            for r, c in edges_to_add:
                mask[r, c] = True
                added += 1
            idx += 1
        attempts += 1
    return added

# ── Training loop ─────────────────────────────────────────────────────────────
def run_arm(label, mask_init, theta, decay, pol, ip, op, bp, bigram,
            all_data, eval_seqs, ticks, np_rng_seed):
    """Run training from given mask, return final accuracy and edge count."""
    rng = random.Random(np_rng_seed)
    np_rng = np.random.RandomState(np_rng_seed)
    mask = mask_init.copy()

    best_acc = eval_acc(mask, theta, decay, pol, eval_seqs, bp, ip, op, ticks)
    init_edges = int(np.sum(mask))

    for step in range(1, TRAINING_STEPS + 1):
        pt = SCHEDULE[(step - 1) % len(SCHEDULE)]
        na = int(np.sum(mask))
        if pt in ('flip', 'decay') and na == 0: pt = 'add'

        nm = mask; nd = decay
        if pt == 'add':
            r2 = rng.randint(0, H-1); c2 = rng.randint(0, H-1)
            if r2 == c2 or mask[r2, c2]: continue
            nm = mask.copy(); nm[r2, c2] = True
        elif pt == 'flip':
            al = list(zip(*np.where(mask))) if na > 0 else []
            if not al: continue
            r2, c2 = al[rng.randint(0, len(al)-1)]
            nc2 = rng.randint(0, H-1)
            if nc2 == r2 or nc2 == c2 or mask[r2, nc2]: continue
            nm = mask.copy(); nm[r2, c2] = False; nm[r2, nc2] = True
        elif pt == 'decay':
            idx = rng.randint(0, H-1); nd = decay.copy()
            nd[idx] = max(0.01, min(0.5, decay[idx] + rng.uniform(-0.03, 0.03)))

        tr = [all_data[o:o+SEQ_LEN]
              for o in [np_rng.randint(0, len(all_data)-SEQ_LEN) for _ in range(N_TRAIN)]]
        os_ = eval_score(mask, theta, decay, pol, tr, bp, ip, op, bigram, ticks)
        ns_ = eval_score(nm, theta, nd, pol, tr, bp, ip, op, bigram, ticks)
        if ns_ - os_ > THRESHOLD:
            mask = nm; decay = nd

        if step % REPORT_EVERY == 0:
            acc = eval_acc(mask, theta, decay, pol, eval_seqs, bp, ip, op, ticks)
            edges = int(np.sum(mask))
            print(f'    [{step:4d}] {label}: {acc*100:.2f}% edges={edges}')
            sys.stdout.flush()

    final_acc = eval_acc(mask, theta, decay, pol, eval_seqs, bp, ip, op, ticks)
    final_edges = int(np.sum(mask))
    return final_acc, final_edges, mask

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print('Loading...')
    all_data = load_fineweb_bytes()
    bigram = np.load(ROOT / 'recipes' / 'data' / 'bigram_table.npy')
    bp = make_bp()

    net = SelfWiringGraph.load(str(CKPT))
    ip = net.input_projection; op = net.output_projection
    pol = np.where(net.polarity, 1.0, -1.0).astype(np.float32)
    theta = net.theta.astype(np.float32)
    decay = net.decay
    base_mask = net.mask.copy()
    base_edges = int(np.sum(base_mask))

    eval_rng = np.random.RandomState(9999)
    eval_seqs = [all_data[o:o+SEQ_LEN]
                 for o in [eval_rng.randint(0, len(all_data)-SEQ_LEN) for _ in range(N_EVAL)]]

    base_acc = eval_acc(base_mask, theta, decay, pol, eval_seqs, bp, ip, op, TICKS_DEFAULT)
    print(f'Checkpoint: {base_edges} edges, {base_acc*100:.2f}%')
    print(f'Injection budget: {INJECTION_BUDGET} edges')
    print(f'Training steps after injection: {TRAINING_STEPS}')

    ARMS = [
        ('A: baseline (no inject)', 'none', {}),
        ('B: random edges',         'random', {}),
        ('C: loop-3',               'loop', {'length': 3}),
        ('D: loop-5',               'loop', {'length': 5}),
        ('E: loop-7',               'loop', {'length': 7}),
        ('F: mixed loops (3-7)',    'mixed', {}),
    ]

    results = {}

    for ticks in TICK_SWEEP:
        print(f'\n{"="*60}')
        print(f'  TICK = {ticks}')
        print(f'{"="*60}')

        for label, inject_type, params in ARMS:
            print(f'\n  {label} (ticks={ticks})')

            mask = base_mask.copy()
            inject_rng = np.random.RandomState(SEED + hash(label) % 10000)

            if inject_type == 'random':
                added = inject_random_edges(mask, INJECTION_BUDGET, inject_rng)
            elif inject_type == 'loop':
                added = inject_loops(mask, params['length'], INJECTION_BUDGET, inject_rng)
            elif inject_type == 'mixed':
                added = inject_mixed_loops(mask, INJECTION_BUDGET, inject_rng)
            else:
                added = 0

            post_inject_edges = int(np.sum(mask))
            post_inject_acc = eval_acc(mask, theta, decay, pol, eval_seqs, bp, ip, op, ticks)
            print(f'    After inject: {added} edges added -> {post_inject_edges} total, acc={post_inject_acc*100:.2f}%')

            final_acc, final_edges, final_mask = run_arm(
                label, mask, theta, decay, pol, ip, op, bp, bigram,
                all_data, eval_seqs, ticks, SEED)

            key = (label, ticks)
            results[key] = {
                'acc': final_acc, 'edges': final_edges,
                'injected': added, 'post_inject_acc': post_inject_acc
            }
            print(f'    FINAL: {final_acc*100:.2f}% edges={final_edges}')
            sys.stdout.flush()

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f'\n{"="*60}')
    print(f'  RESULTS SUMMARY')
    print(f'{"="*60}')

    for ticks in TICK_SWEEP:
        print(f'\n  Ticks={ticks}:')
        for label, _, _ in ARMS:
            key = (label, ticks)
            r = results[key]
            bar = '#' * int(r['acc'] * 200)
            print(f'    {label:30s} {r["acc"]*100:6.2f}% (edges={r["edges"]}) {bar}')

    # Best per tick
    print(f'\n  Best per tick:')
    for ticks in TICK_SWEEP:
        best_label = max(
            [(label, results[(label, ticks)]['acc']) for label, _, _ in ARMS],
            key=lambda x: x[1])
        print(f'    ticks={ticks}: {best_label[0]} @ {best_label[1]*100:.2f}%')

    print(f'\n{"="*60}')

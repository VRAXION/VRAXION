"""
Fresh training for topology comparison.
Same projections as checkpoint A, different random seed for mask init.
Train until plateau, then compare topology with checkpoint A.
"""
import sys, os, time, random
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'model'))
from graph import SelfWiringGraph
from lib.data import load_fineweb_bytes

IO   = 256
NV   = 4
H    = IO * NV   # 1024
TICKS       = 8
INPUT_DUR   = 2
THRESHOLD   = 0.00005
SEQ_LEN     = 150
N_TRAIN     = 2
N_EVAL      = 8
REPORT_EVERY = 100
BUDGET       = 5000
SEED_FRESH   = 777     # different seed for mask RNG
SEED_PROJ    = 42      # SAME seed for projections (must match checkpoint A)
SCHEDULE = ['add','add','flip','decay','decay','decay','decay','decay']

# Checkpoint A for comparison
CKPT_A = ROOT / 'recipes' / 'checkpoints' / 'standard_step4500.npz'
CKPT_DIR = ROOT / 'recipes' / 'checkpoints'

def make_bp(seed=12345):
    rng = np.random.RandomState(seed)
    p = rng.randn(256, IO).astype(np.float32)
    p /= np.linalg.norm(p, axis=1, keepdims=True)
    return p

def get_sparse(mask):
    rows, cols = np.where(mask)
    return rows.astype(np.intp), cols.astype(np.intp)

def rollout(inj, *, mask, theta, decay, state, charge, sp, pol, ref):
    act = state.copy(); cur = charge.copy(); r = ref.copy()
    tf = theta; df = np.asarray(decay, dtype=np.float32)
    rows, cols = sp
    is_sc = df.ndim == 0 or df.shape == ()
    dp = max(1, int(round(1.0 / max(float(df), 0.001)))) if is_sc else 0
    for tick in range(TICKS):
        if dp > 0:
            if tick % dp == 0: cur = np.maximum(cur - 1.0, 0.0)
        else: cur = np.maximum(cur - df, 0.0)
        if tick < INPUT_DUR: act = act + inj
        raw = np.zeros(H, dtype=np.float32)
        if len(rows): np.add.at(raw, cols, act[rows])
        np.nan_to_num(raw, copy=False)
        cur += raw; np.clip(cur, 0.0, 15.0, out=cur)
        can = (r == 0); fired = (cur >= tf) & can
        r[r > 0] -= 1; r[fired] = 1
        act = fired.astype(np.float32) * pol; cur[fired] = 0.0
    return act, cur, r

def eval_score(mask, theta, decay, pol, seqs, bp, ip, op, bigram):
    sp = get_sparse(mask)
    pn = bp / (np.linalg.norm(bp, axis=1, keepdims=True) + 1e-8)
    tot = 0.0
    for seq in seqs:
        st = np.zeros(H, dtype=np.float32); ch = np.zeros(H, dtype=np.float32)
        rf = np.zeros(H, dtype=np.int8); ss = 0.0; n = 0
        for i in range(len(seq) - 1):
            inj = bp[seq[i]] @ ip
            st, ch, rf = rollout(inj, mask=mask, theta=theta, decay=decay,
                                  state=st, charge=ch, sp=sp, pol=pol, ref=rf)
            out = ch @ op; on = out / (np.linalg.norm(out) + 1e-8)
            sims = on @ pn.T; e = np.exp(sims - sims.max()); pred = e / e.sum()
            tgt = bigram[seq[i]]
            cos = np.dot(pred, tgt) / (np.linalg.norm(pred) * np.linalg.norm(tgt) + 1e-8)
            ss += cos; n += 1
        tot += ss / n if n else 0.0
    return tot / len(seqs)

def eval_acc(mask, theta, decay, pol, seqs, bp, ip, op):
    sp = get_sparse(mask)
    pn = bp / (np.linalg.norm(bp, axis=1, keepdims=True) + 1e-8)
    ok = 0; tot = 0
    for seq in seqs:
        st = np.zeros(H, dtype=np.float32); ch = np.zeros(H, dtype=np.float32)
        rf = np.zeros(H, dtype=np.int8)
        for i in range(len(seq) - 1):
            inj = bp[seq[i]] @ ip
            st, ch, rf = rollout(inj, mask=mask, theta=theta, decay=decay,
                                  state=st, charge=ch, sp=sp, pol=pol, ref=rf)
            out = ch @ op; on = out / (np.linalg.norm(out) + 1e-8)
            if np.argmax(on @ pn.T) == seq[i + 1]: ok += 1
            tot += 1
    return ok / tot if tot else 0.0

def compare_topologies(mask_a, mask_b):
    """Compare two masks: overlap, hamming, shared neurons."""
    edges_a = set(zip(*np.where(mask_a)))
    edges_b = set(zip(*np.where(mask_b)))
    shared = edges_a & edges_b
    only_a = edges_a - edges_b
    only_b = edges_b - edges_a
    hamming = len(only_a) + len(only_b)

    # Active neurons (nodes with at least 1 edge)
    neurons_a = set(r for r, c in edges_a) | set(c for r, c in edges_a)
    neurons_b = set(r for r, c in edges_b) | set(c for r, c in edges_b)
    shared_neurons = neurons_a & neurons_b

    # Jaccard similarity
    union = edges_a | edges_b
    jaccard = len(shared) / len(union) if union else 0

    return {
        'edges_a': len(edges_a), 'edges_b': len(edges_b),
        'shared_edges': len(shared), 'hamming': hamming,
        'jaccard': jaccard,
        'neurons_a': len(neurons_a), 'neurons_b': len(neurons_b),
        'shared_neurons': len(shared_neurons),
    }

if __name__ == '__main__':
    print('Loading data...')
    all_data = load_fineweb_bytes()
    bp = make_bp()

    bigram_path = ROOT / 'recipes' / 'data' / 'bigram_table.npy'
    bigram = np.load(bigram_path) if bigram_path.exists() else None
    if bigram is None:
        os.makedirs(bigram_path.parent, exist_ok=True)
        counts = np.zeros((256, 256), dtype=np.float64)
        for i in range(len(all_data) - 1):
            counts[all_data[i], all_data[i + 1]] += 1
        rs = counts.sum(axis=1, keepdims=True); rs[rs == 0] = 1
        bigram = (counts / rs).astype(np.float32)
        np.save(bigram_path, bigram)
    print(f'  data={len(all_data)//1000}K bytes')

    # Load checkpoint A (trained) — get its projections
    print(f'\nLoading checkpoint A: {CKPT_A}')
    net_a = SelfWiringGraph.load(str(CKPT_A))
    mask_a = net_a.mask.copy()
    ip = net_a.input_projection
    op = net_a.output_projection
    pol = getattr(net_a, '_polarity_f32',
                  np.where(net_a.polarity, 1.0, -1.0).astype(np.float32))
    theta_a = getattr(net_a, '_theta_f32', net_a.theta.astype(np.float32))
    decay_a = net_a.decay.copy()
    print(f'  A: edges={int(np.sum(mask_a))}')

    eval_rng = np.random.RandomState(9999)
    eval_seqs = [all_data[o:o + SEQ_LEN]
                 for o in [eval_rng.randint(0, len(all_data) - SEQ_LEN)
                           for _ in range(N_EVAL)]]

    acc_a = eval_acc(mask_a, theta_a, decay_a, pol, eval_seqs, bp, ip, op)
    print(f'  A accuracy: {acc_a*100:.2f}%')

    # ── Train fresh network B ────────────────────────────────────────────────
    print(f'\nTraining fresh network B (seed={SEED_FRESH})...')
    rng = random.Random(SEED_FRESH)
    np_rng = np.random.RandomState(SEED_FRESH)

    mask_b = np.zeros((H, H), dtype=np.bool_)
    theta_b = np.full(H, 0.0, dtype=np.float32)
    decay_b = np_rng.uniform(0.08, 0.24, H).astype(np.float32)

    accepts = 0; t0 = time.time()
    print(f'{"step":>6}  {"acc%":>6}  {"edges":>6}  {"accepts":>7}  {"sps":>5}')

    for step in range(1, BUDGET + 1):
        pt = SCHEDULE[(step - 1) % len(SCHEDULE)]
        na = int(np.sum(mask_b))
        if pt in ('flip', 'decay') and na == 0: pt = 'add'

        nm = mask_b; nt = theta_b; nd = decay_b
        if pt == 'add':
            r2 = rng.randint(0, H-1); c2 = rng.randint(0, H-1)
            if r2 == c2 or mask_b[r2, c2]: continue
            nm = mask_b.copy(); nm[r2, c2] = True
        elif pt == 'flip':
            al = list(zip(*np.where(mask_b))) if na > 0 else []
            if not al: continue
            r2, c2 = al[rng.randint(0, len(al)-1)]
            nc2 = rng.randint(0, H-1)
            if nc2 == r2 or nc2 == c2 or mask_b[r2, nc2]: continue
            nm = mask_b.copy(); nm[r2, c2] = False; nm[r2, nc2] = True
        elif pt == 'decay':
            idx = rng.randint(0, H-1); nd = decay_b.copy()
            nd[idx] = max(0.01, min(0.5, decay_b[idx] + rng.uniform(-0.03, 0.03)))

        tr = [all_data[o:o + SEQ_LEN]
              for o in [np_rng.randint(0, len(all_data) - SEQ_LEN)
                         for _ in range(N_TRAIN)]]
        os_ = eval_score(mask_b, theta_b, decay_b, pol, tr, bp, ip, op, bigram)
        ns_ = eval_score(nm, nt, nd, pol, tr, bp, ip, op, bigram)
        if ns_ - os_ > THRESHOLD:
            mask_b = nm; theta_b = nt; decay_b = nd
            accepts += 1

        if step % REPORT_EVERY == 0:
            acc = eval_acc(mask_b, theta_b, decay_b, pol, eval_seqs, bp, ip, op)
            elapsed = time.time() - t0
            sps = step / elapsed
            edges = int(np.sum(mask_b))
            print(f'{step:6d}  {acc*100:6.2f}%  {edges:6d}  {accepts:7d}  {sps:5.2f}')
            sys.stdout.flush()

    acc_b = eval_acc(mask_b, theta_b, decay_b, pol, eval_seqs, bp, ip, op)
    edges_b = int(np.sum(mask_b))

    # ── Compare A vs B ───────────────────────────────────────────────────────
    print(f'\n{"="*60}')
    print(f'  TOPOLOGY COMPARISON: A (trained) vs B (fresh)')
    print(f'{"="*60}')
    print(f'  A: {acc_a*100:.2f}%, {int(np.sum(mask_a))} edges')
    print(f'  B: {acc_b*100:.2f}%, {edges_b} edges')

    comp = compare_topologies(mask_a, mask_b)
    print(f'\n  Shared edges:   {comp["shared_edges"]}')
    print(f'  Only in A:      {comp["edges_a"] - comp["shared_edges"]}')
    print(f'  Only in B:      {comp["edges_b"] - comp["shared_edges"]}')
    print(f'  Hamming dist:   {comp["hamming"]}')
    print(f'  Jaccard sim:    {comp["jaccard"]:.4f}')
    print(f'\n  Active neurons A: {comp["neurons_a"]}')
    print(f'  Active neurons B: {comp["neurons_b"]}')
    print(f'  Shared neurons:   {comp["shared_neurons"]}')

    # Expected overlap if random
    total_possible = H * (H - 1)
    p_edge = comp["edges_a"] / total_possible
    expected_shared = comp["edges_b"] * p_edge
    print(f'\n  Expected random overlap: {expected_shared:.1f} edges')
    print(f'  Actual overlap:          {comp["shared_edges"]} edges')

    if comp["shared_edges"] > expected_shared * 2:
        print(f'  → CONVERGE: topologies share structure beyond chance')
    elif comp["shared_edges"] > expected_shared * 1.3:
        print(f'  → WEAK CONVERGENCE: some shared structure')
    else:
        print(f'  → INDEPENDENT: different topologies, same performance = degenerate space')
    print(f'{"="*60}')

    # Save B checkpoint
    ckpt_b = CKPT_DIR / 'fresh_B_final.npz'
    rows, cols = np.where(mask_b)
    np.savez_compressed(ckpt_b,
        V=IO, H=H, hidden_ratio=NV,
        rows=rows.astype(np.uint16), cols=cols.astype(np.uint16),
        vals=np.ones(len(rows), dtype=np.bool_),
        theta=theta_b, decay=decay_b,
        polarity=np.where(pol > 0, np.int8(1), np.int8(-1)),
        loss_pct=np.int8(15), mutation_drive=np.int8(1),
        projection_scale=np.float32(1.0), edge_magnitude=np.float32(1.0),
        cap_ratio=np.int32(120),
        input_projection=ip, output_projection=op,
        channel=np.random.RandomState(SEED_FRESH).randint(1,9,H).astype(np.uint8),
    )
    print(f'\nSaved B: {ckpt_b}')

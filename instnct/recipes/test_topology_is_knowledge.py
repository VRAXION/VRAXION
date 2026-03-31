"""
Test: Is the topology the knowledge?
=====================================
Compare:
  A) Trained checkpoint (step 4500, 21.64%, 128 edges)
  B) N fresh random networks with SAME density, SAME projections

If A >> B → the specific edge pattern IS the learned knowledge.
If A ≈ B → the topology is irrelevant, something else matters.
"""
import sys, os, time
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'model'))
from graph import SelfWiringGraph
from lib.data import load_fineweb_bytes

CKPT     = ROOT / 'recipes' / 'checkpoints' / 'standard_step4500.npz'
TICKS    = 8
INPUT_DUR = 2
SEQ_LEN  = 150
N_EVAL   = 10
N_RANDOM = 50   # number of random topologies to compare against

def make_bp(seed=12345):
    rng = np.random.RandomState(seed)
    p = rng.randn(256, 256).astype(np.float32)
    p /= np.linalg.norm(p, axis=1, keepdims=True)
    return p

def rollout(inj, *, mask, theta, decay, state, charge, sp, pol, ref):
    H = len(theta)
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

def eval_acc(mask, theta, decay, pol, seqs, bp, ip, op):
    rows, cols = np.where(mask)
    sp = (rows.astype(np.intp), cols.astype(np.intp))
    pn = bp / (np.linalg.norm(bp, axis=1, keepdims=True) + 1e-8)
    H = mask.shape[0]
    ok = 0; tot = 0
    for seq in seqs:
        st = np.zeros(H, dtype=np.float32)
        ch = np.zeros(H, dtype=np.float32)
        rf = np.zeros(H, dtype=np.int8)
        for i in range(len(seq) - 1):
            inj = bp[seq[i]] @ ip
            st, ch, rf = rollout(inj, mask=mask, theta=theta, decay=decay,
                                  state=st, charge=ch, sp=sp, pol=pol, ref=rf)
            out = ch @ op; on = out / (np.linalg.norm(out) + 1e-8)
            if np.argmax(on @ pn.T) == seq[i + 1]: ok += 1
            tot += 1
    return ok / tot if tot else 0.0

def random_mask_same_density(H, n_edges, rng):
    """Generate random mask with exactly n_edges edges, no self-connections."""
    mask = np.zeros((H, H), dtype=np.bool_)
    possible = [(i, j) for i in range(H) for j in range(H) if i != j]
    chosen = rng.choice(len(possible), size=min(n_edges, len(possible)), replace=False)
    for idx in chosen:
        i, j = possible[idx]
        mask[i, j] = True
    return mask

if __name__ == '__main__':
    print('Loading checkpoint...')
    net = SelfWiringGraph.load(str(CKPT))
    H = net.H
    trained_mask = net.mask.copy()
    n_edges = int(np.sum(trained_mask))
    theta = getattr(net, '_theta_f32', net.theta.astype(np.float32))
    decay = net.decay
    pol = getattr(net, '_polarity_f32',
                  np.where(net.polarity, 1.0, -1.0).astype(np.float32))
    ip = net.input_projection
    op = net.output_projection
    print(f'  H={H}, trained edges={n_edges}')

    print('Loading data...')
    all_data = load_fineweb_bytes()
    bp = make_bp()
    eval_rng = np.random.RandomState(9999)
    eval_seqs = [all_data[o:o + SEQ_LEN]
                 for o in [eval_rng.randint(0, len(all_data) - SEQ_LEN)
                           for _ in range(N_EVAL)]]

    # ── A) Trained topology ──────────────────────────────────────────────────
    print('\nEvaluating trained topology...')
    t0 = time.time()
    trained_acc = eval_acc(trained_mask, theta, decay, pol, eval_seqs, bp, ip, op)
    t_trained = time.time() - t0
    print(f'  TRAINED: {trained_acc*100:.2f}% ({t_trained:.1f}s)')

    # ── B) Random topologies (same density, same everything else) ────────────
    print(f'\nEvaluating {N_RANDOM} random topologies (same density={n_edges} edges)...')
    random_accs = []
    for i in range(N_RANDOM):
        rng = np.random.RandomState(i + 1000)
        rand_mask = random_mask_same_density(H, n_edges, rng)
        acc = eval_acc(rand_mask, theta, decay, pol, eval_seqs, bp, ip, op)
        random_accs.append(acc)
        if (i + 1) % 10 == 0:
            avg = np.mean(random_accs)
            best = max(random_accs)
            print(f'  [{i+1:3d}/{N_RANDOM}] avg={avg*100:.2f}% best={best*100:.2f}%')
            sys.stdout.flush()

    # ── Results ──────────────────────────────────────────────────────────────
    avg_random = np.mean(random_accs)
    std_random = np.std(random_accs)
    best_random = max(random_accs)
    worst_random = min(random_accs)

    print(f'\n{"="*60}')
    print(f'  RESULTS')
    print(f'{"="*60}')
    print(f'  TRAINED topology:  {trained_acc*100:.2f}%')
    print(f'  RANDOM topologies: {avg_random*100:.2f}% ± {std_random*100:.2f}%')
    print(f'    best random:     {best_random*100:.2f}%')
    print(f'    worst random:    {worst_random*100:.2f}%')
    print(f'  Edges (both):      {n_edges}')
    print()

    delta = trained_acc - avg_random
    sigma = delta / std_random if std_random > 0 else 0
    print(f'  Delta: {delta*100:+.2f}%')
    print(f'  Sigma: {sigma:.1f}σ above random mean')
    print()

    if sigma > 3:
        print(f'  → TOPOLOGY IS KNOWLEDGE (trained >> random)')
    elif sigma > 1.5:
        print(f'  → TOPOLOGY MATTERS (trained > random, moderate)')
    elif sigma > 0:
        print(f'  → WEAK SIGNAL (trained slightly > random)')
    else:
        print(f'  → NO SIGNAL (trained ≈ random)')
    print(f'{"="*60}')

"""
Angle superposition test
========================
Loads a frozen checkpoint, freezes ALL mutations (mask, theta, decay),
and only mutates the per-neuron angle parameter.

If accuracy breaks out of plateau → angle superposition did it.

Mechanism:
  angles[H] = float32 [0, 360) per neuron

  After each tick's spike decision:
    1. Compute circular mean angle of all fired neurons
    2. Coupling = cos(angle_i - mean_angle) for each neuron i
    3. Excitatory zone (|delta| < 90°):  probabilistic fire with P ∝ coupling
    4. Inhibitory zone (|delta| > 90°):  reduce charge ∝ -coupling

  Polarity (Dale's Law) stays unchanged — different mechanism, orthogonal.

Mutation (only this is active):
  - Pick random neuron, shift angle by ±{1, 5, 10, 30} degrees
  - Or full resample to random angle (exploration)
  - Accept/reject via bigram score delta
"""
import sys, os, time, random
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'model'))
from graph import SelfWiringGraph
from lib.data import load_fineweb_bytes

# ── Config ────────────────────────────────────────────────────────────────────
CKPT         = ROOT / 'recipes' / 'checkpoints' / 'standard_step4500.npz'
INIT_DENSITY = 0.05  # random edge fill before angle test (5% of H*H)
BUDGET       = 2000
SEQ_LEN   = 150
N_TRAIN   = 2
N_EVAL    = 8
REPORT_EVERY = 100
THRESHOLD = 0.00005
TICKS     = 8
INPUT_DUR = 2
SEED      = 42

# Angle coupling scales — tune these
EXC_SCALE = 1.0   # max P of excitatory synchrony fire
INH_SCALE = 0.5   # charge reduction per inhibitory unit

# Angle mutation options
# Full random resample — let the network find natural clusters

# ── Helpers ───────────────────────────────────────────────────────────────────
def make_bp(seed=12345):
    rng = np.random.RandomState(seed)
    p = rng.randn(256, 256).astype(np.float32)
    p /= np.linalg.norm(p, axis=1, keepdims=True)
    return p

def get_sparse(mask):
    rows, cols = np.where(mask)
    return rows.astype(np.intp), cols.astype(np.intp)

# ── Forward pass with angle superposition ────────────────────────────────────
def rollout_angle(inj, *, mask, theta, decay, state, charge, sp, pol, ref,
                  angles, exc_scale, inh_scale):
    H = len(angles)
    act = state.copy(); cur = charge.copy(); r = ref.copy()
    tf  = theta.astype(np.float32)
    df  = np.asarray(decay, dtype=np.float32)
    rows, cols = sp
    is_sc = df.ndim == 0 or df.shape == ()
    dp = max(1, int(round(1.0 / max(float(df), 0.001)))) if is_sc else 0

    ang_rad = np.radians(angles)  # precompute once per rollout

    for tick in range(TICKS):
        # 1. DECAY
        if dp > 0:
            if tick % dp == 0: cur = np.maximum(cur - 1.0, 0.0)
        else: cur = np.maximum(cur - df, 0.0)

        # 2. INPUT
        if tick < INPUT_DUR: act = act + inj

        # 3. PROPAGATE (edges, multiply-free)
        raw = np.zeros(H, dtype=np.float32)
        if len(rows): np.add.at(raw, cols, act[rows])
        np.nan_to_num(raw, copy=False)
        cur += raw; np.clip(cur, 0.0, 15.0, out=cur)

        # 4. SPIKE DECISION + REFRACTORY
        can   = (r == 0)
        fired = (cur >= tf) & can
        r[r > 0] -= 1; r[fired] = 1

        # 5. ANGLE SUPERPOSITION
        if np.any(fired):
            fired_rad = ang_rad[fired]

            # Circular mean of fired neurons' angles
            mean_sin = np.mean(np.sin(fired_rad))
            mean_cos = np.mean(np.cos(fired_rad))
            mean_rad = np.arctan2(mean_sin, mean_cos)

            # Coupling to every neuron: cos(angle_i - mean_angle)
            delta = ang_rad - mean_rad
            coupling = np.cos(delta)   # [-1, 1]

            # Excitatory zone (coupling > 0): probabilistic extra fires
            exc = np.maximum(coupling, 0.0)
            p_fire = exc * exc_scale
            new_fires = ~fired & can & (np.random.rand(H) < p_fire)
            fired |= new_fires
            r[new_fires] = 1

            # Inhibitory zone (coupling < 0): drain charge
            inh = np.maximum(-coupling, 0.0)
            cur -= inh * inh_scale
            np.maximum(cur, 0.0, out=cur)

        # 6. ACTIVATIONS + HARD RESET
        act = fired.astype(np.float32) * pol
        cur[fired] = 0.0

    return act, cur, r

# ── Evaluation ────────────────────────────────────────────────────────────────
def eval_score(mask, theta, decay, pol, angles, seqs, bp, ip, op, bigram,
               exc_scale, inh_scale):
    sp = get_sparse(mask)
    pn = bp / (np.linalg.norm(bp, axis=1, keepdims=True) + 1e-8)
    tot = 0.0
    for seq in seqs:
        st = np.zeros(len(angles), dtype=np.float32)
        ch = np.zeros(len(angles), dtype=np.float32)
        rf = np.zeros(len(angles), dtype=np.int8)
        ss = 0.0; n = 0
        for i in range(len(seq) - 1):
            inj = bp[seq[i]] @ ip
            st, ch, rf = rollout_angle(inj, mask=mask, theta=theta, decay=decay,
                state=st, charge=ch, sp=sp, pol=pol, ref=rf,
                angles=angles, exc_scale=exc_scale, inh_scale=inh_scale)
            out = ch @ op; on = out / (np.linalg.norm(out) + 1e-8)
            sims = on @ pn.T; e = np.exp(sims - sims.max()); pred = e / e.sum()
            tgt = bigram[seq[i]]
            cos = np.dot(pred, tgt) / (np.linalg.norm(pred) * np.linalg.norm(tgt) + 1e-8)
            ss += cos; n += 1
        tot += ss / n if n else 0.0
    return tot / len(seqs)

def eval_acc(mask, theta, decay, pol, angles, seqs, bp, ip, op,
             exc_scale, inh_scale):
    sp = get_sparse(mask)
    pn = bp / (np.linalg.norm(bp, axis=1, keepdims=True) + 1e-8)
    ok = 0; tot = 0
    H = len(angles)
    for seq in seqs:
        st = np.zeros(H, dtype=np.float32)
        ch = np.zeros(H, dtype=np.float32)
        rf = np.zeros(H, dtype=np.int8)
        for i in range(len(seq) - 1):
            inj = bp[seq[i]] @ ip
            st, ch, rf = rollout_angle(inj, mask=mask, theta=theta, decay=decay,
                state=st, charge=ch, sp=sp, pol=pol, ref=rf,
                angles=angles, exc_scale=exc_scale, inh_scale=inh_scale)
            out = ch @ op; on = out / (np.linalg.norm(out) + 1e-8)
            if np.argmax(on @ pn.T) == seq[i + 1]: ok += 1
            tot += 1
    return ok / tot if tot else 0.0

# ── Angle mutation ────────────────────────────────────────────────────────────
def mutate_angle(angles, rng):
    new_angles = angles.copy()
    idx = rng.randint(0, len(angles))
    new_angles[idx] = np.int16(rng.randint(0, 360))
    return new_angles

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print(f'Loading checkpoint: {CKPT}')
    net = SelfWiringGraph.load(str(CKPT))
    H = net.H
    print(f'  H={H}, edges={net.count_connections()}, V={net.V}')

    print('Loading data...')
    all_data = load_fineweb_bytes()
    bp = make_bp()

    bigram_path = ROOT / 'recipes' / 'data' / 'bigram_table.npy'
    if bigram_path.exists():
        bigram = np.load(bigram_path)
    else:
        os.makedirs(bigram_path.parent, exist_ok=True)
        counts = np.zeros((256, 256), dtype=np.float64)
        for i in range(len(all_data) - 1):
            counts[all_data[i], all_data[i+1]] += 1
        rs = counts.sum(axis=1, keepdims=True); rs[rs==0] = 1
        bigram = (counts / rs).astype(np.float32)
        np.save(bigram_path, bigram)

    ip = net.input_projection; op = net.output_projection
    pol = getattr(net, '_polarity_f32',
                  np.where(net.polarity, 1.0, -1.0).astype(np.float32))
    mask = net.mask
    theta = getattr(net, '_theta_f32', net.theta.astype(np.float32))
    decay = net.decay

    # Init angles: random scatter 0-360
    ang_rng = np.random.RandomState(SEED + 99)
    angles = ang_rng.randint(0, 360, H).astype(np.int16)

    eval_rng = np.random.RandomState(9999)
    eval_seqs = [all_data[o:o+SEQ_LEN]
                 for o in [eval_rng.randint(0, len(all_data)-SEQ_LEN)
                           for _ in range(N_EVAL)]]

    rng = random.Random(SEED); np_rng = np.random.RandomState(SEED)

    # ── WARMUP: train normally until WARMUP_EDGES ─────────────────────────────
    print(f'\nWarmup: training until {WARMUP_EDGES} edges (no angle, standard mutations)...')
    SCHEDULE = ['add','add','flip','decay','decay','decay','decay','decay']
    warmup_step = 0
    while int(np.sum(mask)) < WARMUP_EDGES:
        warmup_step += 1
        pt = SCHEDULE[(warmup_step - 1) % len(SCHEDULE)]
        na = int(np.sum(mask))
        if pt in ('flip', 'decay') and na == 0: pt = 'add'
        nm = mask; nt = theta; nd = decay
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
        # No angle during warmup
        dummy = np.zeros(H, dtype=np.int16)
        os_ = eval_score(mask, theta, decay, pol, dummy, tr, bp, ip, op, bigram, 0.0, 0.0)
        ns_ = eval_score(nm, nt, nd, pol, dummy, tr, bp, ip, op, bigram, 0.0, 0.0)
        if ns_ - os_ > THRESHOLD:
            mask = nm; theta = nt; decay = nd
        if warmup_step % 500 == 0:
            edges = int(np.sum(mask))
            print(f'  warmup step {warmup_step}: edges={edges}')
            sys.stdout.flush()

    edges_after_warmup = int(np.sum(mask))
    warmup_acc = eval_acc(mask, theta, decay, pol, np.zeros(H, dtype=np.int16),
                          eval_seqs, bp, ip, op, 0.0, 0.0)
    print(f'  Warmup done: edges={edges_after_warmup} acc={warmup_acc*100:.2f}%')

    # Baseline: frozen network, no angles
    print('\nBaseline (frozen, no angle superposition)...')
    base_acc = warmup_acc
    print(f'  Baseline acc = {base_acc*100:.2f}%')

    print(f'\nAngle superposition training (frozen mask/theta/decay)')
    print(f'  exc_scale={EXC_SCALE}  inh_scale={INH_SCALE}')
    print(f'  budget={BUDGET}  threshold={THRESHOLD}')
    print(f'{"step":>6}  {"acc%":>7}  {"accepts":>8}  {"angle_std":>10}  {"sps":>5}')

    accepts = 0; t0 = time.time()

    for step in range(1, BUDGET + 1):
        new_angles = mutate_angle(angles, np_rng)

        tr = [all_data[o:o+SEQ_LEN]
              for o in [np_rng.randint(0, len(all_data)-SEQ_LEN)
                        for _ in range(N_TRAIN)]]

        old_s = eval_score(mask, theta, decay, pol, angles,
                           tr, bp, ip, op, bigram, EXC_SCALE, INH_SCALE)
        new_s = eval_score(mask, theta, decay, pol, new_angles,
                           tr, bp, ip, op, bigram, EXC_SCALE, INH_SCALE)

        if new_s - old_s > THRESHOLD:
            angles = new_angles
            accepts += 1

        if step % REPORT_EVERY == 0:
            acc = eval_acc(mask, theta, decay, pol, angles,
                           eval_seqs, bp, ip, op, EXC_SCALE, INH_SCALE)
            elapsed = time.time() - t0
            sps = step / elapsed
            ang_std = float(np.std(angles))
            # Show angle distribution clusters
            hist, _ = np.histogram(angles, bins=8, range=(0, 360))
            print(f'{step:6d}  {acc*100:7.2f}%  {accepts:8d}  {ang_std:10.1f}  {sps:5.2f}'
                  f'  hist={hist.tolist()}')
            sys.stdout.flush()

    final_acc = eval_acc(mask, theta, decay, pol, angles,
                         eval_seqs, bp, ip, op, EXC_SCALE, INH_SCALE)
    print(f'\nFINAL: {final_acc*100:.2f}%  (baseline was {base_acc*100:.2f}%)')
    delta = final_acc - base_acc
    print(f'Delta: {delta*100:+.2f}%  '
          f'{"↑ SUPERPOSITION HELPS" if delta > 0.01 else "→ no significant change" if abs(delta) < 0.01 else "↓ hurts"}')

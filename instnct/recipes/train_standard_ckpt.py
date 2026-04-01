"""
Standard single-process training with checkpoint saving.
No superposition — clean baseline to freeze and then test superposition on.
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
CKPT_EVERY   = 500
SEED         = 42
SCHEDULE = ['add','add','flip','decay','decay','decay','decay','decay']

CKPT_DIR = ROOT / 'recipes' / 'checkpoints'
os.makedirs(CKPT_DIR, exist_ok=True)

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
    tf = theta.astype(np.float32)
    df = np.asarray(decay, dtype=np.float32)
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

if __name__ == '__main__':
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
            counts[all_data[i], all_data[i + 1]] += 1
        rs = counts.sum(axis=1, keepdims=True); rs[rs == 0] = 1
        bigram = (counts / rs).astype(np.float32)
        np.save(bigram_path, bigram)
    print(f'  data={len(all_data)//1000}K bytes  bigram OK')

    ref_net = SelfWiringGraph(IO, hidden_ratio=NV, projection_scale=1.0, seed=SEED)
    ip = ref_net.input_projection; op = ref_net.output_projection
    pol = ref_net._polarity_f32

    rng = random.Random(SEED); np_rng = np.random.RandomState(SEED)
    mask  = np.zeros((H, H), dtype=np.bool_)
    theta = np.full(H, 0.0, dtype=np.float32)
    decay = np_rng.uniform(0.08, 0.24, H).astype(np.float32)

    eval_rng = np.random.RandomState(9999)
    eval_seqs = [all_data[o:o + SEQ_LEN]
                 for o in [eval_rng.randint(0, len(all_data) - SEQ_LEN)
                           for _ in range(N_EVAL)]]

    print(f'Training H={H}, schedule={SCHEDULE}')
    print(f'Checkpoints → {CKPT_DIR}')
    print(f'{"step":>6}  {"acc%":>6}  {"edges":>6}  {"accepts":>7}  {"sps":>5}')

    accepts = 0; t0 = time.time(); step = 0

    try:
        while True:
            step += 1
            pt = SCHEDULE[(step - 1) % len(SCHEDULE)]
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

            tr = [all_data[o:o + SEQ_LEN]
                  for o in [np_rng.randint(0, len(all_data) - SEQ_LEN)
                             for _ in range(N_TRAIN)]]
            os_ = eval_score(mask, theta, decay, pol, tr, bp, ip, op, bigram)
            ns_ = eval_score(nm, nt, nd, pol, tr, bp, ip, op, bigram)
            if ns_ - os_ > THRESHOLD:
                mask = nm; theta = nt; decay = nd
                accepts += 1

            if step % REPORT_EVERY == 0:
                acc = eval_acc(mask, theta, decay, pol, eval_seqs, bp, ip, op)
                elapsed = time.time() - t0
                sps = step / elapsed
                edges = int(np.sum(mask))
                print(f'{step:6d}  {acc*100:6.2f}%  {edges:6d}  {accepts:7d}  {sps:5.2f}')
                sys.stdout.flush()

            if step % CKPT_EVERY == 0:
                ckpt = CKPT_DIR / f'standard_step{step}.npz'
                # Manual save since we manage state directly
                rows, cols = np.where(mask)
                np.savez_compressed(ckpt,
                    V=IO, H=H, hidden_ratio=NV,
                    rows=rows.astype(np.uint16), cols=cols.astype(np.uint16),
                    vals=np.ones(len(rows), dtype=np.bool_),
                    theta=theta, decay=decay,
                    polarity=np.where(pol > 0, np.int8(1), np.int8(-1)),
                    loss_pct=np.int8(15), mutation_drive=np.int8(1),
                    projection_scale=np.float32(1.0),
                    edge_magnitude=np.float32(1.0),
                    cap_ratio=np.int32(120),
                    input_projection=ip, output_projection=op,
                    channel=np.random.RandomState(SEED).randint(1,9,H).astype(np.uint8),
                )
                print(f'  CKPT saved: {ckpt}')
                sys.stdout.flush()

    except KeyboardInterrupt:
        print(f'\nStopped at step {step}.')
        ckpt = CKPT_DIR / f'standard_step{step}_final.npz'
        rows, cols = np.where(mask)
        np.savez_compressed(ckpt,
            V=IO, H=H, hidden_ratio=NV,
            rows=rows.astype(np.uint16), cols=cols.astype(np.uint16),
            vals=np.ones(len(rows), dtype=np.bool_),
            theta=theta, decay=decay,
            polarity=np.where(pol > 0, np.int8(1), np.int8(-1)),
            loss_pct=np.int8(15), mutation_drive=np.int8(1),
            projection_scale=np.float32(1.0),
            edge_magnitude=np.float32(1.0),
            cap_ratio=np.int32(120),
            input_projection=ip, output_projection=op,
            channel=np.random.RandomState(SEED).randint(1,9,H).astype(np.uint8),
        )
        acc = eval_acc(mask, theta, decay, pol, eval_seqs, bp, ip, op)
        print(f'Final acc={acc*100:.2f}% edges={int(np.sum(mask))} accepts={accepts}')
        print(f'Saved: {ckpt}')

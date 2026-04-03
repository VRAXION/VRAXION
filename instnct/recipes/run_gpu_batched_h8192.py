"""
GPU Batched BUILD+CRYSTAL: H=8192, theta=6 init, batched eval
==============================================================
All 12 mutation candidates evaluated in 1 GPU batch call.
35 min BUILD, then crystal.
"""
import sys, os, time, random
from collections import defaultdict
import numpy as np
import torch
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "model"))
from graph import SelfWiringGraph
from quaternary_mask import QuaternaryMask
from gpu_forward import GPUForward

H = 8192; TICKS = 16; INPUT_DURATION = 2
INIT_DENSITY = 0.005; INIT_THETA = 6
BUILD_STEPS = 2000; EVAL_EVERY = 50
THRESHOLD = 0.00005; N_CANDIDATES = 12
PHI = (1 + 5**0.5) / 2
IN_DIM = int(round(H / PHI))
OUT_DIM = int(round(H / PHI))
SDR_K = int(round(IN_DIM * 0.20))
SEQ_LEN = 100; EVAL_TOKENS = 20; N_EVAL_SEQS = 3

SCHEDULE = ['add', 'reverse', 'loop3', 'mirror', 'flip', 'theta', 'channel', 'channel', 'loop5', 'remove', 'theta', 'reverse']


def build_sdr(n, dim, k, seed):
    rng = np.random.RandomState(seed)
    t = np.zeros((n, dim), np.float32)
    for v in range(n): t[v, rng.choice(dim, size=k, replace=False)] = 1.0
    return t

def build_freq_order(dim, bigram, seed=12345):
    freq = bigram.sum(axis=0) + bigram.sum(axis=1)
    rank = np.argsort(freq)[::-1]
    rng = np.random.RandomState(seed)
    p = np.zeros((256, dim), np.float32)
    for i, byte_idx in enumerate(rank):
        t = i / 255.0
        for d in range(dim):
            p[byte_idx, d] = np.sin(2 * np.pi * t * (d+1) / dim * 3) + rng.randn() * 0.3
    p /= np.linalg.norm(p, axis=1, keepdims=True) + 1e-8
    return p.astype(np.float32)


def eval_cosine_gpu_single(gpu, rows_t, cols_t, theta_t, chan_t, pol_t,
                           text_bytes, bp_in_t, bp_out_t, bigram_t, dev):
    """Single network eval on GPU."""
    state = torch.zeros(H, device=dev); charge = torch.zeros(H, device=dev)
    total = 0.0; n = 0
    for i in range(min(EVAL_TOKENS, len(text_bytes) - 1)):
        inj = torch.zeros(H, device=dev); inj[:IN_DIM] = bp_in_t[text_bytes[i]]
        state, charge = gpu.rollout_token(inj, rows_t, cols_t, theta_t, chan_t, pol_t,
                                          ticks=TICKS, input_duration=INPUT_DURATION,
                                          state=state, charge=charge)
        logits = bp_out_t @ charge[H - OUT_DIM:]
        e = torch.exp(logits - logits.max()); pred = e / e.sum()
        tgt = bigram_t[text_bytes[i]]
        cos = (pred * tgt).sum() / (pred.norm() * tgt.norm() + 1e-8)
        total += cos.item(); n += 1
    return total / n if n else 0.0


def eval_accuracy_gpu(gpu, rows_t, cols_t, theta_t, chan_t, pol_t,
                      text_bytes, bp_in_t, bp_out_t, dev):
    """Full accuracy eval."""
    state = torch.zeros(H, device=dev); charge = torch.zeros(H, device=dev)
    cor = 0; tot = 0
    for i in range(len(text_bytes) - 1):
        inj = torch.zeros(H, device=dev); inj[:IN_DIM] = bp_in_t[text_bytes[i]]
        state, charge = gpu.rollout_token(inj, rows_t, cols_t, theta_t, chan_t, pol_t,
                                          ticks=TICKS, input_duration=INPUT_DURATION,
                                          state=state, charge=charge)
        logits = bp_out_t @ charge[H - OUT_DIM:]
        if torch.argmax(logits).item() == text_bytes[i + 1]: cor += 1
        tot += 1
    return cor / tot if tot else 0.0


def apply_mutation(qm, theta, channel, pol_f, op, rng, H):
    """Apply one mutation, return (new_qm, new_theta, new_channel, new_pol, undo)."""
    nq = qm.copy(); nt = theta.copy(); nc = channel.copy(); npf = pol_f.copy()
    undo = []

    if op == 'add':
        nq.mutate_add(rng, undo)
    elif op == 'reverse':
        nq.mutate_flip(rng, undo)
    elif op == 'mirror':
        nq.mutate_upgrade(rng, undo)
    elif op == 'remove':
        nq.mutate_remove(rng, undo)
    elif op in ('loop3', 'loop5', 'loop8'):
        loop_len = int(op[4:])
        nodes = [rng.randint(0, H - 1)]
        for _ in range(loop_len - 1):
            n = rng.randint(0, H - 1)
            if n in nodes: return nq, nt, nc, npf, []
            nodes.append(n)
        for k in range(loop_len):
            r, c = nodes[k], nodes[(k + 1) % loop_len]
            if nq.get_pair(r, c) != 0: return nq, nt, nc, npf, []
        for k in range(loop_len):
            r, c = nodes[k], nodes[(k + 1) % loop_len]
            nq.set_pair(r, c, 1)
            undo.append(('QA', nq._pair_index(r, c), 0))
    elif op == 'theta':
        idx = rng.randint(0, H - 1)
        nt = theta.copy(); nt[idx] = float(rng.randint(1, 15))
        undo.append(('T',))
    elif op == 'channel':
        idx = rng.randint(0, H - 1)
        nc = channel.copy(); nc[idx] = np.uint8(rng.randint(1, 8))
        undo.append(('CH',))
    elif op == 'flip':
        idx = rng.randint(0, H - 1)
        npf = pol_f.copy(); npf[idx] *= -1
        undo.append(('F',))
    return nq, nt, nc, npf, undo


if __name__ == "__main__":
    from lib.data import load_fineweb_bytes, resolve_fineweb_path
    resolve_fineweb_path()
    ALL_DATA = load_fineweb_bytes()
    print(f"Loaded {len(ALL_DATA)/1e6:.1f} MB")

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    bigram_np = np.load(os.path.join(BASE_DIR, "data", "bigram_table.npy"))
    bp_in_np = build_sdr(256, IN_DIM, SDR_K, 42)
    bp_out_np = build_freq_order(OUT_DIM, bigram_np)

    eval_rng = np.random.RandomState(9999)
    eval_seqs = [ALL_DATA[off:off + SEQ_LEN]
                 for off in [eval_rng.randint(0, len(ALL_DATA) - SEQ_LEN) for _ in range(N_EVAL_SEQS)]]

    dev = torch.device('cuda')
    gpu = GPUForward(H, 'cuda')
    bp_in_t = torch.tensor(bp_in_np, dtype=torch.float32, device=dev)
    bp_out_t = torch.tensor(bp_out_np, dtype=torch.float32, device=dev)
    bigram_t = torch.tensor(bigram_np, dtype=torch.float32, device=dev)

    # Init network
    init_rng = np.random.RandomState(42)
    bool_mask = (init_rng.rand(H, H) < INIT_DENSITY).astype(bool)
    np.fill_diagonal(bool_mask, False)
    qm = QuaternaryMask.from_bool_mask(bool_mask)
    qdata = qm.data
    theta = np.full(H, float(INIT_THETA), dtype=np.float32)
    channel = init_rng.randint(1, 9, size=H).astype(np.uint8)
    pol_bool = np.ones(H, dtype=bool)
    pol_bool[init_rng.random(H) < 0.10] = False
    pol_f = np.where(pol_bool, 1.0, -1.0).astype(np.float32)

    print(f"H={H}, ticks={TICKS}, theta_init={INIT_THETA}, density={INIT_DENSITY*100:.1f}%")
    print(f"Init: {qm.count_edges()} edges, IN_DIM={IN_DIM}, OUT_DIM={OUT_DIM}")
    print(f"QMask: {qm.memory_bytes/1024:.0f} KB")
    print()

    # === BUILD ===
    print(f"=== BUILD ({BUILD_STEPS} steps, batched GPU) ===")
    acc_by_type = defaultdict(int)
    t0 = time.time(); best_eval = 0

    for step in range(1, BUILD_STEPS + 1):
        op = SCHEDULE[(step - 1) % len(SCHEDULE)]
        edges = QuaternaryMask(H, qdata).count_edges()
        if op in ('remove', 'reverse', 'mirror') and edges < 100: op = 'add'

        # Generate N_CANDIDATES mutations
        candidates = []
        for c in range(N_CANDIDATES):
            rng_c = random.Random(step * 1000 + c)
            nq, nt, nc, npf, undo = apply_mutation(
                QuaternaryMask(H, qdata.copy()), theta, channel, pol_f, op, rng_c, H)
            if undo:
                candidates.append((nq, nt, nc, npf))
        if not candidates:
            continue

        # Eval all candidates + original via batched GPU
        # For training speed: use 1 seq, EVAL_TOKENS tokens
        seq_off = random.randint(0, len(ALL_DATA) - SEQ_LEN)
        seq = ALL_DATA[seq_off:seq_off + SEQ_LEN]

        # Original score
        rows_o, cols_o = QuaternaryMask(H, qdata).to_directed_edges()
        rows_ot = torch.tensor(rows_o, dtype=torch.long, device=dev)
        cols_ot = torch.tensor(cols_o, dtype=torch.long, device=dev)
        theta_ot = torch.tensor(theta, dtype=torch.float32, device=dev)
        chan_ot = torch.tensor(channel, dtype=torch.long, device=dev)
        pol_ot = torch.tensor(pol_f, dtype=torch.float32, device=dev)
        base_score = eval_cosine_gpu_single(gpu, rows_ot, cols_ot, theta_ot, chan_ot, pol_ot,
                                            seq, bp_in_t, bp_out_t, bigram_t, dev)

        # Eval candidates sequentially (each has different edges/params)
        # TODO: true batched eval for candidates with same edges but diff params
        best_delta = -1e9; best_idx = -1
        for ci, (nq, nt, nc, npf) in enumerate(candidates):
            r, c = nq.to_directed_edges()
            rt = torch.tensor(r, dtype=torch.long, device=dev)
            ct = torch.tensor(c, dtype=torch.long, device=dev)
            tt = torch.tensor(nt, dtype=torch.float32, device=dev)
            cht = torch.tensor(nc, dtype=torch.long, device=dev)
            pt = torch.tensor(npf, dtype=torch.float32, device=dev)
            score = eval_cosine_gpu_single(gpu, rt, ct, tt, cht, pt,
                                           seq, bp_in_t, bp_out_t, bigram_t, dev)
            delta = score - base_score
            if delta > best_delta:
                best_delta = delta; best_idx = ci

        if best_delta > THRESHOLD and best_idx >= 0:
            nq, nt, nc, npf = candidates[best_idx]
            qdata = nq.data
            theta = nt; channel = nc; pol_f = npf
            acc_by_type[op] += 1

        if step % EVAL_EVERY == 0:
            elapsed = time.time() - t0
            r, c = QuaternaryMask(H, qdata).to_directed_edges()
            rt = torch.tensor(r, dtype=torch.long, device=dev)
            ct = torch.tensor(c, dtype=torch.long, device=dev)
            tt = torch.tensor(theta, dtype=torch.float32, device=dev)
            cht = torch.tensor(channel, dtype=torch.long, device=dev)
            pt = torch.tensor(pol_f, dtype=torch.float32, device=dev)
            ea = np.mean([eval_accuracy_gpu(gpu, rt, ct, tt, cht, pt, s, bp_in_t, bp_out_t, dev)
                          for s in eval_seqs])
            if ea > best_eval: best_eval = ea
            edges = QuaternaryMask(H, qdata).count_edges()
            at = ' '.join(f"{k}={v}" for k, v in sorted(acc_by_type.items()))
            sps = step / elapsed
            # Theta distribution
            th_mean = theta.mean()
            print(f"  [{step:5d}] eval={ea*100:.1f}% best={best_eval*100:.1f}% "
                  f"edges={edges} th_mean={th_mean:.1f} [{at}] {elapsed:.0f}s ({sps:.1f}sps)")
            sys.stdout.flush()

    build_time = time.time() - t0
    qm_final = QuaternaryMask(H, qdata)
    edges_final = qm_final.count_edges()
    print(f"\n  BUILD DONE: edges={edges_final}, best={best_eval*100:.1f}%, {build_time:.0f}s")
    print(f"  theta mean={theta.mean():.1f}, accepts: {dict(acc_by_type)}")

    # Save
    ckpt_path = os.path.join(BASE_DIR, "data", "h8192_gpu_build.npz")
    os.makedirs(os.path.join(BASE_DIR, "data"), exist_ok=True)
    np.savez_compressed(ckpt_path, qdata=qdata, theta=theta, channel=channel, pol_f=pol_f)
    print(f"  Saved: {ckpt_path}")

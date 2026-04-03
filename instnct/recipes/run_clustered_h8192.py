"""
Clustered H=8192 evolution: 11 clusters + IO layer, GPU eval
=============================================================
Each cluster mutates independently, GPU evaluates the full network.
12 candidates per step (11 cluster mutations + 1 IO mutation).
"""
import sys, os, time, random
from collections import defaultdict
import numpy as np
import torch
from multiprocessing import Pool
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "model"))
from graph import SelfWiringGraph
from quaternary_mask import QuaternaryMask
from clustered_network import ClusteredNetwork
from gpu_forward import GPUForward

H = 8192; N_CLUSTERS = 11; TICKS = 16; INPUT_DURATION = 2
INIT_DENSITY = 0.005; IO_DENSITY = 0.001
BUILD_STEPS = 2000; EVAL_EVERY = 25
THRESHOLD = 0.00005
PHI = (1 + 5**0.5) / 2
IN_DIM = int(round(H / PHI))
OUT_DIM = int(round(H / PHI))
SDR_K = int(round(IN_DIM * 0.20))
SEQ_LEN = 100; EVAL_TOKENS = 20; N_EVAL_SEQS = 3

CLUSTER_OPS = ['add', 'reverse', 'loop3', 'flip', 'theta', 'channel', 'remove', 'mirror']
IO_OPS = ['add', 'remove', 'reverse', 'rewire']


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


def eval_gpu(gpu, net, text_bytes, bp_in_t, bp_out_t, bigram_t, dev):
    """Evaluate network on GPU using bigram cosine."""
    rows, cols = net.get_global_edges()
    rows_t = torch.tensor(rows, dtype=torch.long, device=dev)
    cols_t = torch.tensor(cols, dtype=torch.long, device=dev)
    theta_t = torch.tensor(net.theta, dtype=torch.float32, device=dev)
    channel_t = torch.tensor(net.channel, dtype=torch.long, device=dev)
    pol_t = torch.tensor(net.polarity_f32, dtype=torch.float32, device=dev)

    state = torch.zeros(H, device=dev)
    charge = torch.zeros(H, device=dev)
    total_cos = 0.0; n = 0

    for i in range(min(EVAL_TOKENS, len(text_bytes) - 1)):
        inj = torch.zeros(H, device=dev)
        inj[:IN_DIM] = bp_in_t[text_bytes[i]]
        state, charge = gpu.rollout_token(
            inj, rows_t, cols_t, theta_t, channel_t, pol_t,
            ticks=TICKS, input_duration=INPUT_DURATION,
            state=state, charge=charge)
        logits = bp_out_t @ charge[H - OUT_DIM:]
        e = torch.exp(logits - logits.max())
        pred = e / e.sum()
        tgt = bigram_t[text_bytes[i]]
        cos = (pred * tgt).sum() / (pred.norm() * tgt.norm() + 1e-8)
        total_cos += cos.item()
        n += 1
    return total_cos / n if n else 0.0


def eval_accuracy_gpu(gpu, net, text_bytes, bp_in_t, bp_out_t, dev):
    """Full accuracy eval on GPU."""
    rows, cols = net.get_global_edges()
    rows_t = torch.tensor(rows, dtype=torch.long, device=dev)
    cols_t = torch.tensor(cols, dtype=torch.long, device=dev)
    theta_t = torch.tensor(net.theta, dtype=torch.float32, device=dev)
    channel_t = torch.tensor(net.channel, dtype=torch.long, device=dev)
    pol_t = torch.tensor(net.polarity_f32, dtype=torch.float32, device=dev)

    state = torch.zeros(H, device=dev)
    charge = torch.zeros(H, device=dev)
    cor = 0; tot = 0

    for i in range(len(text_bytes) - 1):
        inj = torch.zeros(H, device=dev)
        inj[:IN_DIM] = bp_in_t[text_bytes[i]]
        state, charge = gpu.rollout_token(
            inj, rows_t, cols_t, theta_t, channel_t, pol_t,
            ticks=TICKS, input_duration=INPUT_DURATION,
            state=state, charge=charge)
        logits = bp_out_t @ charge[H - OUT_DIM:]
        if torch.argmax(logits).item() == text_bytes[i + 1]:
            cor += 1
        tot += 1
    return cor / tot if tot else 0.0


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
                 for off in [eval_rng.randint(0, len(ALL_DATA) - SEQ_LEN)
                             for _ in range(N_EVAL_SEQS)]]

    # GPU setup
    dev = torch.device('cuda')
    gpu = GPUForward(H, device='cuda')
    bp_in_t = torch.tensor(bp_in_np, dtype=torch.float32, device=dev)
    bp_out_t = torch.tensor(bp_out_np, dtype=torch.float32, device=dev)
    bigram_t = torch.tensor(bigram_np, dtype=torch.float32, device=dev)

    # Create clustered network
    net = ClusteredNetwork(H, n_clusters=N_CLUSTERS, density=INIT_DENSITY,
                           io_density=IO_DENSITY, seed=42)
    print(net.summary())
    print(f"Ticks={TICKS}, eval_tokens={EVAL_TOKENS}")
    print()

    # === BUILD ===
    print(f"=== BUILD ({BUILD_STEPS} steps) ===")
    acc_by_type = defaultdict(int)
    t0 = time.time(); best_eval = 0

    for step in range(1, BUILD_STEPS + 1):
        # Pick ops for each cluster + IO
        cl_op = CLUSTER_OPS[(step - 1) % len(CLUSTER_OPS)]
        io_op = IO_OPS[(step - 1) % len(IO_OPS)]

        # Get baseline score (1 seq, EVAL_TOKENS tokens)
        seq_off = random.randint(0, len(ALL_DATA) - SEQ_LEN)
        seq = ALL_DATA[seq_off:seq_off + SEQ_LEN]
        base_score = eval_gpu(gpu, net, seq, bp_in_t, bp_out_t, bigram_t, dev)

        # Try mutations: 1 per cluster + 1 IO = 12 candidates
        best_delta = -1e9; best_candidate = None

        for ci in range(N_CLUSTERS):
            rng = random.Random(1000 + step * 100 + ci)
            undo = net.mutate_cluster(ci, cl_op, rng)
            if undo:
                new_score = eval_gpu(gpu, net, seq, bp_in_t, bp_out_t, bigram_t, dev)
                delta = new_score - base_score
                if delta > best_delta:
                    best_delta = delta
                    best_candidate = ('cluster', ci, cl_op, undo)
                net.undo_cluster(ci, undo)

        # IO mutation
        rng_io = random.Random(2000 + step * 100)
        io_undo = net.mutate_io(io_op, rng_io)
        if io_undo:
            new_score = eval_gpu(gpu, net, seq, bp_in_t, bp_out_t, bigram_t, dev)
            delta = new_score - base_score
            if delta > best_delta:
                best_delta = delta
                best_candidate = ('io', -1, io_op, io_undo)
            net.undo_io(io_undo)

        # Accept best
        if best_delta > THRESHOLD and best_candidate is not None:
            kind, ci, op, undo = best_candidate
            # Re-apply the winning mutation
            if kind == 'cluster':
                rng = random.Random(1000 + step * 100 + ci)
                net.mutate_cluster(ci, op, rng)
            else:
                rng_io = random.Random(2000 + step * 100)
                net.mutate_io(op, rng_io)
            key = f"{kind}:{op}" if kind == 'io' else f"c{ci}:{op}"
            acc_by_type[op] += 1

        if step % EVAL_EVERY == 0:
            elapsed = time.time() - t0
            ea = np.mean([eval_accuracy_gpu(gpu, net, s, bp_in_t, bp_out_t, dev)
                          for s in eval_seqs])
            if ea > best_eval: best_eval = ea
            edges = net.count_edges()
            at = ' '.join(f"{k}={v}" for k, v in sorted(acc_by_type.items()))
            sps = step / elapsed
            print(f"  [{step:5d}] eval={ea*100:.1f}% best={best_eval*100:.1f}% "
                  f"edges={edges} [{at}] {elapsed:.0f}s ({sps:.1f}sps)")
            sys.stdout.flush()

    build_time = time.time() - t0
    print(f"\n  BUILD DONE: edges={net.count_edges()}, best={best_eval*100:.1f}%, {build_time:.0f}s")
    print(f"  accepts: {dict(acc_by_type)}")

    # Save
    ckpt_path = os.path.join(BASE_DIR, "data", "h8192_clustered_checkpoint.npz")
    os.makedirs(os.path.join(BASE_DIR, "data"), exist_ok=True)
    net.save(ckpt_path)
    print(f"  Saved: {ckpt_path}")

    print(f"\n{'='*60}")
    print(f"  H={H} CLUSTERED RESULT")
    print(f"  {net.summary()}")
    print(f"  best eval: {best_eval*100:.1f}%")
    print(f"  Compare: H=256 = 21.4%, H=1024 = 18.4% (BUILD only)")
    print(f"{'='*60}")

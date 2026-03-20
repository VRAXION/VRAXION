"""Benchmark: NumPy sparse CPU vs PyTorch GPU forward pass at various scales.

Tests the core SWG forward operation:
  for each tick:
    act = max(charge - theta, 0)
    charge = charge * (1 - decay) + W @ act

Measures: single sequence (200 bytes) and batched (64/256 sequences).
"""
import numpy as np
import torch
import time

def bench_numpy_sparse(H, V, n_edges, seq_len, ticks, n_seqs, repeats=5):
    """Current SWG approach: sparse numpy, one sequence at a time."""
    # Build random sparse W
    total = H * 3  # input + hidden + output (H neurons, V=input/output each)
    N = H * 3
    src = np.random.randint(0, N, n_edges, dtype=np.int32)
    dst = np.random.randint(0, N, n_edges, dtype=np.int32)
    wts = np.random.randn(n_edges).astype(np.float32) * 0.1
    theta = np.random.uniform(0, 0.3, N).astype(np.float32)
    decay = np.random.uniform(0.01, 0.3, N).astype(np.float32)

    # W_in: V -> N encoding
    W_in = np.random.randn(N, V).astype(np.float32) * 0.01
    W_out = np.random.randn(V, N).astype(np.float32) * 0.01

    # Sparse forward (current approach from graph.py)
    def forward_one_seq():
        charge = np.zeros(N, dtype=np.float32)
        for b in range(seq_len):
            # inject input
            inp = np.zeros(V, dtype=np.float32)
            inp[np.random.randint(0, V)] = 1.0
            charge += W_in @ inp
            # ticks
            for t in range(ticks):
                act = np.maximum(charge - theta, 0.0)
                # sparse matmul: scatter_add equivalent
                out = np.zeros(N, dtype=np.float32)
                np.add.at(out, dst, act[src] * wts)
                charge = charge * (1.0 - decay) + out
            # read output (not timed separately)
            logits = W_out @ charge
        return logits

    # Warmup
    forward_one_seq()

    t0 = time.perf_counter()
    for _ in range(repeats):
        for _ in range(n_seqs):
            forward_one_seq()
    elapsed = (time.perf_counter() - t0) / repeats
    return elapsed


def bench_torch_gpu_dense(H, V, n_edges, seq_len, ticks, n_seqs, repeats=5):
    """PyTorch GPU: dense matmul, batched sequences."""
    N = H * 3
    device = torch.device('cuda')

    # Build dense W from sparse edges (on GPU the overhead of sparse is worse than dense at small N)
    W = torch.zeros(N, N, device=device)
    src = torch.randint(0, N, (n_edges,))
    dst = torch.randint(0, N, (n_edges,))
    wts = torch.randn(n_edges) * 0.1
    W[dst, src] = wts.to(device)  # W[i,j] = connection from j to i

    theta = torch.rand(N, device=device) * 0.3
    decay = torch.rand(N, device=device) * 0.3
    W_in = torch.randn(N, V, device=device) * 0.01
    W_out = torch.randn(V, N, device=device) * 0.01

    # Generate random input sequences: (n_seqs, seq_len) of byte indices
    inputs = torch.randint(0, V, (n_seqs, seq_len), device=device)

    # One-hot lookup
    eye = torch.eye(V, device=device)

    torch.cuda.synchronize()

    def forward_batch():
        # charge: (n_seqs, N)
        charge = torch.zeros(n_seqs, N, device=device)
        for b in range(seq_len):
            # inject: (n_seqs, V) -> (n_seqs, N)
            inp = eye[inputs[:, b]]  # (n_seqs, V)
            charge = charge + inp @ W_in.T  # (n_seqs, N)
            for t in range(ticks):
                act = torch.clamp(charge - theta, min=0.0)  # (n_seqs, N)
                charge = charge * (1.0 - decay) + act @ W.T  # (n_seqs, N)
            logits = charge @ W_out.T  # (n_seqs, V)
        return logits

    # Warmup
    for _ in range(3):
        forward_batch()
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(repeats):
        forward_batch()
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - t0) / repeats
    return elapsed


def bench_torch_gpu_sparse(H, V, n_edges, seq_len, ticks, n_seqs, repeats=5):
    """PyTorch GPU: sparse matmul, batched sequences."""
    N = H * 3
    device = torch.device('cuda')

    # Build sparse W
    src = torch.randint(0, N, (n_edges,))
    dst = torch.randint(0, N, (n_edges,))
    wts = torch.randn(n_edges) * 0.1
    indices = torch.stack([dst, src])  # COO format
    W_sparse = torch.sparse_coo_tensor(indices, wts, (N, N)).to(device).coalesce()

    theta = torch.rand(N, device=device) * 0.3
    decay = torch.rand(N, device=device) * 0.3
    W_in = torch.randn(N, V, device=device) * 0.01
    W_out = torch.randn(V, N, device=device) * 0.01

    inputs = torch.randint(0, V, (n_seqs, seq_len), device=device)
    eye = torch.eye(V, device=device)

    torch.cuda.synchronize()

    def forward_batch():
        charge = torch.zeros(n_seqs, N, device=device)
        for b in range(seq_len):
            inp = eye[inputs[:, b]]
            charge = charge + inp @ W_in.T
            for t in range(ticks):
                act = torch.clamp(charge - theta, min=0.0)
                # sparse mm: W_sparse @ act.T -> (N, n_seqs) -> transpose
                charge = charge * (1.0 - decay) + torch.sparse.mm(W_sparse, act.T).T
            logits = charge @ W_out.T
        return logits

    # Warmup
    for _ in range(3):
        forward_batch()
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(repeats):
        forward_batch()
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - t0) / repeats
    return elapsed


print("=" * 80)
print("SWG Forward Pass Benchmark: NumPy CPU vs PyTorch GPU")
print("=" * 80)

configs = [
    # (H, V, n_edges, label)
    (256,  256,  2000,   "768N / 2K edges (current)"),
    (256,  256,  10000,  "768N / 10K edges (projected)"),
    (1024, 256,  10000,  "3072N / 10K edges"),
    (1024, 256,  50000,  "3072N / 50K edges"),
    (2048, 256,  50000,  "6144N / 50K edges"),
    (4096, 256,  100000, "12288N / 100K edges (big)"),
]

seq_len = 200
ticks = 4

for H, V, n_edges, label in configs:
    N = H * 3
    density = n_edges / (N * N) * 100
    print(f"\n--- {label} (density={density:.2f}%, N={N}) ---")

    # CPU: only 1 sequence (that's how current training works per worker)
    t_cpu = bench_numpy_sparse(H, V, n_edges, seq_len, ticks, n_seqs=1, repeats=3)
    print(f"  CPU numpy sparse (1 seq):    {t_cpu*1000:8.1f} ms")

    # GPU dense: batch 1 (fair comparison)
    t_gpu1 = bench_torch_gpu_dense(H, V, n_edges, seq_len, ticks, n_seqs=1, repeats=3)
    print(f"  GPU dense (1 seq):           {t_gpu1*1000:8.1f} ms  ({t_cpu/t_gpu1:.1f}x vs CPU)")

    # GPU dense: batch 18 (= 18 workers batched)
    t_gpu18 = bench_torch_gpu_dense(H, V, n_edges, seq_len, ticks, n_seqs=18, repeats=3)
    print(f"  GPU dense (18 seq batch):    {t_gpu18*1000:8.1f} ms  ({t_cpu*18/t_gpu18:.1f}x vs 18×CPU)")

    # GPU dense: batch 64
    t_gpu64 = bench_torch_gpu_dense(H, V, n_edges, seq_len, ticks, n_seqs=64, repeats=3)
    print(f"  GPU dense (64 seq batch):    {t_gpu64*1000:8.1f} ms  ({t_cpu*64/t_gpu64:.1f}x vs 64×CPU)")

    # GPU sparse: batch 18
    try:
        t_sp18 = bench_torch_gpu_sparse(H, V, n_edges, seq_len, ticks, n_seqs=18, repeats=3)
        print(f"  GPU sparse (18 seq batch):   {t_sp18*1000:8.1f} ms  ({t_cpu*18/t_sp18:.1f}x vs 18×CPU)")
    except Exception as e:
        print(f"  GPU sparse (18 seq batch):   FAILED ({e})")

    # Memory estimate
    mem_dense_mb = N * N * 4 / 1e6  # float32
    mem_sparse_mb = n_edges * 12 / 1e6  # 2 int32 + 1 float32 per edge
    print(f"  Memory: dense W = {mem_dense_mb:.1f} MB, sparse = {mem_sparse_mb:.2f} MB")

print("\n" + "=" * 80)
print("Key question: at what scale does GPU batch > 18× CPU single?")
print("(Because current training uses 18 CPU workers in parallel)")
print("=" * 80)

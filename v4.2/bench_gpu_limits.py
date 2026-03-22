"""Find GPU limits: where does VRAM run out? Where does compute plateau?

RTX 4070 Ti SUPER: 16GB VRAM, 66 SMs, Ada Lovelace
"""
import torch
import time

device = torch.device('cuda')

print("=" * 80)
print("GPU Limit Finder: RTX 4070 Ti SUPER (16 GB VRAM)")
print("=" * 80)

V = 256  # byte vocab
ticks = 4
seq_len = 200

# ---- PART 1: VRAM usage by network size ----
print("\n--- PART 1: VRAM vs Network Size (dense W) ---")
print(f"{'Neurons':>10} {'N':>8} {'W size':>10} {'+batch18':>10} {'+batch64':>10} {'Total(64)':>10} {'Fits?':>6}")

for H in [256, 512, 1024, 2048, 4096, 8192, 16384, 32768]:
    N = H * 3
    w_bytes = N * N * 4  # dense W, float32
    # charge + act buffers per batch
    batch18_bytes = 18 * N * 4 * 3  # charge, act, out (3 buffers)
    batch64_bytes = 64 * N * 4 * 3
    # input_projection + output_projection
    io_bytes = 2 * N * V * 4
    total_64 = w_bytes + batch64_bytes + io_bytes
    fits = total_64 < 16e9
    print(f"{H:>10} {N:>8} {w_bytes/1e9:>9.2f}G {batch18_bytes/1e6:>9.1f}M {batch64_bytes/1e6:>9.1f}M {total_64/1e9:>9.2f}G {'YES' if fits else 'NO':>6}")

# ---- PART 2: VRAM vs Network Size (sparse W) ----
print("\n--- PART 2: VRAM vs Network Size (sparse W, 1% density) ---")
print(f"{'Neurons':>10} {'N':>8} {'Edges':>10} {'W sparse':>10} {'+batch64':>10} {'Total(64)':>10} {'Fits?':>6}")

for H in [256, 1024, 4096, 8192, 16384, 32768, 65536]:
    N = H * 3
    n_edges = int(N * N * 0.01)  # 1% density
    sparse_bytes = n_edges * 12  # 2 int32 + 1 float32
    batch64_bytes = 64 * N * 4 * 3
    io_bytes = 2 * N * V * 4
    total_64 = sparse_bytes + batch64_bytes + io_bytes
    fits = total_64 < 16e9
    print(f"{H:>10} {N:>8} {n_edges:>10} {sparse_bytes/1e9:>9.3f}G {batch64_bytes/1e6:>9.1f}M {total_64/1e9:>9.2f}G {'YES' if fits else 'NO':>6}")

# ---- PART 3: Actual benchmark - find compute ceiling ----
print("\n--- PART 3: Actual GPU Speed vs Size (dense, batch=18) ---")
print(f"{'Config':>30} {'ms/step':>10} {'seq/sec':>10} {'VRAM used':>10}")

for H, n_edges in [(256, 2000), (512, 5000), (1024, 20000), (2048, 50000), (4096, 100000), (8192, 200000)]:
    N = H * 3
    n_seqs = 18

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    try:
        # Dense W (simpler, faster for moderate N)
        W = torch.zeros(N, N, device=device)
        src = torch.randint(0, N, (n_edges,))
        dst = torch.randint(0, N, (n_edges,))
        W[dst, src] = torch.randn(n_edges).to(device) * 0.1

        theta = torch.rand(N, device=device) * 0.3
        decay = torch.rand(N, device=device) * 0.3
        input_projection = torch.randn(N, V, device=device) * 0.01
        output_projection = torch.randn(V, N, device=device) * 0.01
        inputs = torch.randint(0, V, (n_seqs, seq_len), device=device)
        eye = torch.eye(V, device=device)

        # Warmup
        charge = torch.zeros(n_seqs, N, device=device)
        for b in range(min(10, seq_len)):
            inp = eye[inputs[:, b]]
            charge = charge + inp @ input_projection.T
            for t in range(ticks):
                act = torch.clamp(charge - theta, min=0.0)
                charge = charge * (1.0 - decay) + act @ W.T
        torch.cuda.synchronize()

        # Benchmark
        t0 = time.perf_counter()
        repeats = 3
        for _ in range(repeats):
            charge = torch.zeros(n_seqs, N, device=device)
            for b in range(seq_len):
                inp = eye[inputs[:, b]]
                charge = charge + inp @ input_projection.T
                for t in range(ticks):
                    act = torch.clamp(charge - theta, min=0.0)
                    charge = charge * (1.0 - decay) + act @ W.T
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - t0) / repeats

        vram = torch.cuda.max_memory_allocated() / 1e9
        seq_per_sec = n_seqs / elapsed

        label = f"{H}H ({N}N) / {n_edges//1000}K edges"
        print(f"{label:>30} {elapsed*1000:>9.0f}ms {seq_per_sec:>9.1f} {vram:>9.2f}G")

        del W, theta, decay, input_projection, output_projection, inputs, charge

    except torch.cuda.OutOfMemoryError:
        label = f"{H}H ({N}N) / {n_edges//1000}K edges"
        print(f"{label:>30}    ** OUT OF MEMORY **")
        break
    except Exception as e:
        label = f"{H}H ({N}N) / {n_edges//1000}K edges"
        print(f"{label:>30}    ERROR: {e}")

# ---- PART 4: Batch size scaling at sweet spot ----
print("\n--- PART 4: Batch Size Scaling at 1024H (N=3072) ---")
print(f"{'Batch':>8} {'ms/step':>10} {'seq/sec':>10} {'VRAM':>10} {'vs batch1':>10}")

H = 1024
N = H * 3
n_edges = 20000
torch.cuda.empty_cache()

W = torch.zeros(N, N, device=device)
src = torch.randint(0, N, (n_edges,))
dst = torch.randint(0, N, (n_edges,))
W[dst, src] = torch.randn(n_edges).to(device) * 0.1
theta = torch.rand(N, device=device) * 0.3
decay = torch.rand(N, device=device) * 0.3
input_projection = torch.randn(N, V, device=device) * 0.01
output_projection = torch.randn(V, N, device=device) * 0.01
eye = torch.eye(V, device=device)

base_sps = None
for batch in [1, 4, 8, 16, 32, 64, 128, 256, 512]:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    try:
        inputs = torch.randint(0, V, (batch, seq_len), device=device)

        # Warmup
        charge = torch.zeros(batch, N, device=device)
        for b in range(10):
            inp = eye[inputs[:, b]]
            charge = charge + inp @ input_projection.T
            for t in range(ticks):
                act = torch.clamp(charge - theta, min=0.0)
                charge = charge * (1.0 - decay) + act @ W.T
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(3):
            charge = torch.zeros(batch, N, device=device)
            for b in range(seq_len):
                inp = eye[inputs[:, b]]
                charge = charge + inp @ input_projection.T
                for t in range(ticks):
                    act = torch.clamp(charge - theta, min=0.0)
                    charge = charge * (1.0 - decay) + act @ W.T
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - t0) / 3

        sps = batch / elapsed
        vram = torch.cuda.max_memory_allocated() / 1e9
        if base_sps is None:
            base_sps = sps
        print(f"{batch:>8} {elapsed*1000:>9.0f}ms {sps:>9.1f} {vram:>9.2f}G {sps/base_sps:>9.1f}x")

        del inputs, charge
    except torch.cuda.OutOfMemoryError:
        print(f"{batch:>8}    ** OUT OF MEMORY **")
        break

print("\n" + "=" * 80)
print("Summary: VRAM is the wall for dense W (N^2). Compute is the wall for sparse.")
print("Sweet spot: largest N where dense W fits, with max batch that fits remaining VRAM.")
print("=" * 80)

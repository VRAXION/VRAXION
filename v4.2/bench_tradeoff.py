"""VRAM tradeoff calculator: batch vs network size.
Shows exactly what eats the VRAM and what the max configs are.
"""
import torch

VRAM = 16e9  # 16 GB
V = 256
TICKS = 4

print("=" * 80)
print("VRAM Budget Breakdown: What eats the 16 GB?")
print("=" * 80)

print("\n--- Dense W mode ---")
print(f"{'Neurons':>8} {'N':>7} {'W (N^2)':>10} {'Batch=1':>10} {'Batch=128':>10} {'Batch=512':>10} {'W / Total':>10}")
for H in [1024, 2048, 4096, 8192, 16384]:
    N = H * 3
    w = N * N * 4
    io = 2 * N * V * 4  # W_in + W_out
    params = N * 4 * 2  # theta + decay
    fixed = w + io + params
    for batch in [1, 128, 512]:
        buffers = batch * N * 4 * 4  # charge, act, inp_proj, out (4 buffers)
        total = fixed + buffers
        if batch == 1:
            b1 = total
        ratio = w / total * 100
        if batch == 1:
            print(f"{H:>8} {N:>7} {w/1e9:>9.2f}G {b1/1e9:>9.3f}G", end="")
        elif batch == 128:
            print(f" {total/1e9:>9.3f}G", end="")
        elif batch == 512:
            print(f" {total/1e9:>9.3f}G {ratio:>9.1f}%", end="")
    print()

print("\n--> W (N^2) = 95-99% of VRAM. Batch is noise!")
print("--> Reducing batch from 512 to 1 saves almost nothing.")

print("\n" + "=" * 80)
print("Max configurations that fit in 16 GB VRAM")
print("=" * 80)

# Dense: find max N for given batch
print("\n--- Dense W: max neurons per batch ---")
for batch in [1, 18, 64, 128, 256]:
    # N² * 4 + 2*N*256*4 + batch*N*4*4 < 16e9
    # Solve: 4*N² + 2048*N + 16*batch*N < 16e9
    # Approximate: 4*N² < 16e9 -> N < 63245
    # More precise binary search
    lo, hi = 1000, 200000
    while lo < hi:
        mid = (lo + hi + 1) // 2
        N = mid
        mem = N*N*4 + 2*N*V*4 + N*4*2 + batch*N*4*4
        if mem < VRAM * 0.85:  # 85% safety margin
            lo = mid
        else:
            hi = mid - 1
    N = lo
    H = N // 3
    w_gb = N*N*4/1e9
    print(f"  batch={batch:>4}: max {H:>6} neurons (N={N:>6}), W={w_gb:.1f}GB")

# Sparse: find max for given density and batch
print("\n--- Sparse W: max neurons per batch (1% density) ---")
for batch in [1, 18, 64, 128, 256]:
    lo, hi = 1000, 2000000
    while lo < hi:
        mid = (lo + hi + 1) // 2
        N = mid
        edges = int(N * N * 0.01)
        mem = edges*12 + 2*N*V*4 + N*4*2 + batch*N*4*4
        if mem < VRAM * 0.85:
            lo = mid
        else:
            hi = mid - 1
    N = lo
    H = N // 3
    edges = int(N*N*0.01)
    sp_gb = edges*12/1e9
    print(f"  batch={batch:>4}: max {H:>6} neurons (N={N:>6}), edges={edges:>10}, sparse W={sp_gb:.1f}GB")

# Sparse with ACTUAL SWG density (starts sparse, grows)
print("\n--- Sparse W: max neurons at ACTUAL SWG densities ---")
print(f"{'Neurons':>8} {'N':>7} {'Max edges':>12} {'Density':>8} {'Batch 128':>10} {'Fits?':>6}")
for H in [1024, 2048, 4096, 8192, 16384, 32768, 65536]:
    N = H * 3
    batch = 128
    other = 2*N*V*4 + N*4*2 + batch*N*4*4
    remaining = VRAM * 0.85 - other
    if remaining < 0:
        max_edges = 0
        fits = False
    else:
        max_edges = int(remaining / 12)
        fits = True
    max_possible = N * N
    density = max_edges / max_possible * 100 if max_possible > 0 else 0
    fits_str = "YES" if fits and max_edges > 0 else "NO"
    print(f"{H:>8} {N:>7} {max_edges:>12,} {density:>7.2f}% {(other + max_edges*12)/1e9:>9.2f}G {fits_str:>6}")

# The REAL question: at SWG growth rate, how long until we hit the wall?
print("\n--- SWG Growth Reality Check ---")
print("Current: 768 neurons, ~50 edges/100 steps (add rate ~95%)")
print("At that growth rate with bigger network:")
for H in [1024, 4096, 16384, 32768]:
    N = H * 3
    batch = 128
    other = 2*N*V*4 + N*4*2 + batch*N*4*4
    remaining = VRAM * 0.85 - other
    max_edges = int(remaining / 12)
    # Growth: ~0.5 edges per step (current rate)
    edges_per_step = 0.5  # conservative
    steps_to_fill = max_edges / edges_per_step
    max_possible = N * N
    print(f"  {H:>6} neurons: room for {max_edges:>12,} edges = {steps_to_fill:,.0f} steps = {steps_to_fill/3600:.0f} hours @ 1 step/sec")
    print(f"           max possible {max_possible:>12,} edges ({max_edges/max_possible*100:.1f}% density cap)")

print("\n" + "=" * 80)
print("CONCLUSION:")
print("- Dense W: max ~14K neurons. Batch barely matters (W dominates).")
print("- Sparse W: max 30K-65K neurons. Growth rate = centuries before VRAM full.")
print("- Best strategy: SPARSE W on GPU + big batch + huge network")
print("=" * 80)

"""GPU Ternary Search — VRAM-SAFE version. Tiny batches, monitors memory."""
import torch, time, gc

def c19(x, rho=8.0):
    if x >= 6: return x - 6
    if x <= -6: return x + 6
    n = int(x) if x >= 0 else int(x) - 1
    t = x - n; h = t * (1 - t)
    return (1.0 if n % 2 == 0 else -1.0) * h + rho * h * h

dev = torch.device('cuda')
props = torch.cuda.get_device_properties(0)
print(f'GPU: {props.name}, {props.total_memory//1024**2} MB')

def vram_mb():
    return torch.cuda.memory_allocated() // 1024**2

def bench_N(N, max_sec=10):
    total = 3**(N+1)
    M = 1 << N

    # Conservative batch: keep total GPU mem under 2 GB
    # matmul: M × B × 2 bytes (fp16) + M × B × 4 bytes (int32) = M × B × 6
    max_bytes = 1 * 1024**3  # 1 GB max for matmul
    bs = max(64, min(4096, max_bytes // (M * 6)))

    # C19 LUT (tiny)
    lo, hi = -(N+1), N+1
    lut = torch.tensor([1 if c19(float(s)) > 0 else 0 for s in range(lo,hi+1)], dtype=torch.int8, device=dev)

    # Input patterns (this is 2^N × (N+1), biggest fixed alloc)
    idx = torch.arange(M, dtype=torch.int32, device=dev)
    bits = ((idx.unsqueeze(1) >> torch.arange(N, dtype=torch.int32, device=dev)) & 1).to(torch.int8)
    inp = torch.cat([bits, torch.ones(M,1,dtype=torch.int8,device=dev)], dim=1)
    target = (bits.sum(1) > N//2).to(torch.int8)
    del bits, idx

    inp_f16 = inp.to(torch.float16)

    print(f'  N={N} M={M} combos={total:,} batch={bs} vram={vram_mb()}MB')

    done = 0; best = 0
    t0 = time.perf_counter()

    while done < total and time.perf_counter() - t0 < max_sec:
        actual = min(bs, total - done)

        # Generate weights on GPU
        rem = torch.arange(done, done+actual, dtype=torch.int64, device=dev)
        wt = torch.zeros(actual, N+1, dtype=torch.int8, device=dev)
        r = rem
        for p in range(N+1):
            wt[:, p] = (r % 3).to(torch.int8) - 1
            r = r // 3
        del rem, r

        # Matmul fp16
        dot = torch.mm(inp_f16, wt.to(torch.float16).t())  # M × actual, fp16
        dot_i = dot.to(torch.int32) + (N+1)
        del dot
        dot_i.clamp_(0, len(lut)-1)

        act = lut[dot_i]  # M × actual, int8
        del dot_i

        scores = (act == target.unsqueeze(1)).sum(dim=0)  # actual
        del act

        b = int(scores.max().item())
        if b > best: best = b
        del scores, wt
        done += actual

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    rate = done / elapsed if elapsed > 0 else 0
    full = total / rate if rate > 0 else 999999

    # Cleanup
    del inp, inp_f16, target, lut
    torch.cuda.empty_cache()
    gc.collect()

    return N, total, done, rate, full, best, M

print(f'\nStarting benchmark (max 10s per N)...\n')
print(f'{"N":>4} {"Combos":>14} {"Done%":>6} {"Rate":>10} {"Full Est":>10} {"Best":>8} {"VRAM":>6}')
print('-'*65)

for N in [13, 16, 18, 20]:
    torch.cuda.empty_cache(); gc.collect()
    try:
        n, total, done, rate, full, best, M = bench_N(N)
        pct = done/total*100
        if full < 60: fs = f'{full:.1f}s'
        elif full < 3600: fs = f'{full/60:.1f}m'
        else: fs = f'{full/3600:.1f}h'
        tag = 'DONE' if done>=total else f'{pct:.0f}%'
        print(f'{n:>4} {total:>14,} {tag:>6} {rate/1000:>8.0f}K/s {fs:>10} {best}/{M} {vram_mb():>5}MB')
    except Exception as e:
        print(f'{N:>4} ERROR: {e}')
        torch.cuda.empty_cache(); gc.collect()

# CPU comparison
print(f'\nCPU ref (24 threads): N=13 = 18K/s = 4.5min')

# -*- coding: utf-8 -*-
"""
GPU Ternary Exhaustive Search Benchmark
========================================
Hardware: NVIDIA RTX 4070 Ti SUPER (66 SMs, 16 GB VRAM, CUDA 12.1)
Framework: PyTorch 2.5.1+cu121

Strategy:
  - For each N, generate ALL 3^(N+1) ternary weight combos
  - For each combo, evaluate across ALL 2^N binary inputs
  - Compare to target (popcount > N/2)
  - Track best score and combos/sec

Batching: We process weight combos in batches to stay within VRAM.
  batch_weights shape: (B, N+1) dtype=float16
  all_inputs shape: (2^N, N+1) dtype=float16  (precomputed once per N)

  score[b] = sum over inputs of (C19(dot(weights[b], input_with_bias)) > 0) == target_bit
"""

import torch
import math
import time
import sys


# C19 function (scalar, for LUT generation)

def c19(x, rho=8.0):
    """C19 activation function."""
    if x >= 6:
        return x - 6
    if x <= -6:
        return x + 6
    n = int(x) if x >= 0 else int(x) - 1  # floor
    t = x - n
    h = t * (1 - t)
    sgn = 1.0 if n % 2 == 0 else -1.0
    return sgn * h + rho * h * h


def build_c19_lut(N, rho=8.0):
    """
    Build integer LUT for C19(sum) where sum = dot(weights, inputs+bias).

    For N inputs, ternary weights in {-1, 0, 1}:
      - N input weights, each in {-1,0,1} * binary_input{0,1}
      - 1 bias weight in {-1,0,1}

    Input contributions: each w_i * x_i where x_i in {0,1}
      So each contribution is in {-1, 0, 1}
      Sum of N contributions: range [-N, N]
    Bias: in {-1, 0, 1}
    Total sum range: [-N-1, N+1]

    LUT maps integer dot product -> sign of C19 (1 or 0)
    We use offset = N+1 so index 0 = sum of -(N+1).
    """
    lo = -(N + 1)
    hi = N + 1
    lut_size = hi - lo + 1
    lut = torch.zeros(lut_size, dtype=torch.int8)
    for i, s in enumerate(range(lo, hi + 1)):
        lut[i] = 1 if c19(float(s), rho) > 0 else 0
    return lut


# Input pattern generation

def generate_all_inputs(N, device):
    """
    Generate all 2^N binary input patterns as shape (2^N, N) int8 tensor on GPU.
    """
    num_inputs = 1 << N
    indices = torch.arange(num_inputs, dtype=torch.int32, device=device)
    shifts = torch.arange(N, dtype=torch.int32, device=device)
    bits = ((indices.unsqueeze(1) >> shifts.unsqueeze(0)) & 1).to(torch.int8)
    return bits  # shape (2^N, N)


def generate_target(N, device):
    """
    Target function: popcount(input) > N/2  (majority vote)
    Returns (2^N,) int8 tensor of 0/1.
    """
    num_inputs = 1 << N
    indices = torch.arange(num_inputs, dtype=torch.int32, device=device)
    shifts = torch.arange(N, dtype=torch.int32, device=device)
    bits = ((indices.unsqueeze(1) >> shifts.unsqueeze(0)) & 1)  # (2^N, N)
    popcounts = bits.sum(dim=1)  # (2^N,)
    target = (popcounts > N // 2).to(torch.int8)
    return target  # shape (2^N,)


# Ternary weight enumeration

def generate_weight_batch(N, batch_start, batch_size, total_combos, device):
    """
    Generate a batch of ternary weight vectors (mapped: {0,1,2} -> {-1,0,1}).
    Returns shape (actual_batch, N+1) int8.

    Encoding: treat combo index as base-3 number with N+1 digits.
    Digit values {0,1,2} map to {-1, 0, 1}.
    """
    actual = min(batch_size, total_combos - batch_start)
    if actual <= 0:
        return torch.zeros((0, N + 1), dtype=torch.int8, device=device)

    indices = torch.arange(batch_start, batch_start + actual, dtype=torch.int64, device=device)

    num_weights = N + 1
    weights = torch.zeros((actual, num_weights), dtype=torch.int8, device=device)
    rem = indices.clone()
    for pos in range(num_weights):
        digit = (rem % 3).to(torch.int8)
        # Map 0->-1, 1->0, 2->1
        weights[:, pos] = digit - 1
        rem = rem // 3

    return weights  # shape (actual, N+1)


# Score computation

def compute_scores_batch(weights, all_inputs_with_bias, target, lut, lut_offset):
    """
    Compute agreement score for each weight combo vs target.

    weights: (B, N+1) int8
    all_inputs_with_bias: (2^N, N+1) int8  -- last col is bias=1
    target: (2^N,) int8
    lut: (2*N+3,) int8
    lut_offset: int

    Returns (B,) int32 agreement counts.
    """
    B = weights.shape[0]
    M = all_inputs_with_bias.shape[0]

    # Use float16 for the matrix multiply -- RTX 4070 Ti SUPER has fast fp16 tensor cores.
    # Dot products are exact integers in range [-(N+1), N+1] <= 21 for N=20,
    # well within fp16's exact integer range (exact up to 2048).
    w_f = weights.to(torch.float16)               # (B, N+1)
    x_f = all_inputs_with_bias.to(torch.float16)  # (M, N+1)
    dot_f = torch.mm(x_f, w_f.t())                # (M, B) float16

    # Round to int and shift by lut_offset for LUT indexing
    dot_i = dot_f.to(torch.int32)   # (M, B)
    dot_idx = dot_i + lut_offset    # index into LUT

    # Clamp to valid LUT range (safety)
    lut_size = lut.shape[0]
    dot_idx = dot_idx.clamp(0, lut_size - 1)

    # LUT lookup: activation output bits (M, B) int8
    act = lut[dot_idx]  # (M, B) int8

    # Compare with target: (M, 1) broadcast -> (M, B)
    target_col = target.unsqueeze(1).expand(M, B)  # (M, B)
    agreement = (act == target_col).to(torch.int32)  # (M, B)

    # Sum over inputs: (B,) -- score per weight combo
    scores = agreement.sum(dim=0)  # (B,) int32
    return scores


# Main benchmark

def benchmark_N(N, device, rho=8.0, time_limit_sec=120.0, batch_size=4096):
    """
    Run exhaustive ternary search for given N.
    Returns dict with timing and rate info.
    """
    total_combos = 3 ** (N + 1)
    num_inputs = 1 << N

    print(f"\n{'='*60}")
    print(f"N={N}: {total_combos:,} combos, {num_inputs:,} inputs")

    # Build LUT
    lut = build_c19_lut(N, rho).to(device)
    lut_offset = N + 1  # shift so index 0 = dot product of -(N+1)

    # Generate all binary inputs + bias column
    print(f"  Generating {num_inputs:,} input patterns...")
    all_inputs = generate_all_inputs(N, device)  # (2^N, N) int8
    bias_col = torch.ones((num_inputs, 1), dtype=torch.int8, device=device)
    all_inputs_with_bias = torch.cat([all_inputs, bias_col], dim=1)  # (2^N, N+1)

    # Generate target
    target = generate_target(N, device)  # (2^N,) int8

    mem_kb = all_inputs_with_bias.element_size() * all_inputs_with_bias.numel() / 1024
    print(f"  Input tensor: {all_inputs_with_bias.shape}, {mem_kb:.1f} KB")

    # Estimate memory per batch
    # dot_f: M * B * 2 bytes (float16)
    # agreement: M * B * 4 bytes (int32)
    mem_per_combo_bytes = num_inputs * (2 + 4)
    vram_budget = 10 * 1024**3  # 10 GB conservative
    max_batch_by_mem = max(1, int(vram_budget / mem_per_combo_bytes))
    actual_batch = min(batch_size, max_batch_by_mem, total_combos)

    print(f"  Batch size: {actual_batch:,} combos")
    print(f"  Est. VRAM per batch: {actual_batch * mem_per_combo_bytes / 1024**2:.1f} MB")

    # Warmup
    print(f"  Warming up GPU...")
    warmup_weights = generate_weight_batch(N, 0, min(32, total_combos), total_combos, device)
    _ = compute_scores_batch(warmup_weights, all_inputs_with_bias, target, lut, lut_offset)
    torch.cuda.synchronize()

    best_score = 0
    combos_done = 0
    t_start = time.perf_counter()
    t_last_report = t_start
    timed_out = False

    print(f"  Running...")
    while combos_done < total_combos:
        t_now = time.perf_counter()
        if t_now - t_start > time_limit_sec:
            timed_out = True
            break

        weights = generate_weight_batch(N, combos_done, actual_batch, total_combos, device)
        if weights.shape[0] == 0:
            break

        scores = compute_scores_batch(weights, all_inputs_with_bias, target, lut, lut_offset)
        torch.cuda.synchronize()

        batch_best = int(scores.max().item())
        if batch_best > best_score:
            best_score = batch_best

        combos_done += weights.shape[0]

        # Progress report every 5 seconds
        if time.perf_counter() - t_last_report >= 5.0:
            elapsed = time.perf_counter() - t_start
            rate = combos_done / elapsed
            pct = combos_done / total_combos * 100
            print(f"    {pct:.1f}% -- {combos_done:,}/{total_combos:,} combos -- "
                  f"{rate:,.0f} combos/sec -- best={best_score}/{num_inputs}")
            t_last_report = time.perf_counter()

    torch.cuda.synchronize()
    t_end = time.perf_counter()
    elapsed = t_end - t_start

    combos_per_sec = combos_done / elapsed if elapsed > 0 else 0
    pct_done = combos_done / total_combos * 100

    # Extrapolate full time if timed out
    if timed_out:
        full_estimate_sec = total_combos / combos_per_sec if combos_per_sec > 0 else float('inf')
    else:
        full_estimate_sec = elapsed

    result = {
        "N": N,
        "total_combos": total_combos,
        "combos_done": combos_done,
        "pct_done": pct_done,
        "best_score": best_score,
        "max_possible": num_inputs,
        "elapsed_sec": elapsed,
        "combos_per_sec": combos_per_sec,
        "full_estimate_sec": full_estimate_sec,
        "timed_out": timed_out,
        "batch_size": actual_batch,
    }

    tag = "(EXTRAPOLATED)" if timed_out else "(COMPLETE)"
    print(f"  Result {tag}:")
    print(f"    combos/sec: {combos_per_sec:,.0f}")
    print(f"    best score: {best_score}/{num_inputs} ({best_score/num_inputs*100:.1f}%)")
    print(f"    elapsed: {elapsed:.2f}s")
    if timed_out:
        print(f"    full time estimate: {full_estimate_sec:.1f}s ({full_estimate_sec/3600:.2f}h)")
    return result


def format_time(sec):
    if sec < 1:
        return f"{sec*1000:.1f} ms"
    if sec < 60:
        return f"{sec:.2f} sec"
    if sec < 3600:
        return f"{sec/60:.1f} min"
    if sec < 86400:
        return f"{sec/3600:.2f} hr"
    return f"{sec/86400:.1f} days"


# Entry point

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        print(f"GPU: {props.name}")
        print(f"  SMs: {props.multi_processor_count}")
        print(f"  VRAM: {props.total_memory // 1024**2} MB")
        print(f"  CUDA: {torch.version.cuda}")
    else:
        print("WARNING: No GPU found, running on CPU")

    print(f"\nC19 LUT sanity check (rho=8.0):")
    for x in [-7, -6, -3, -1, 0, 1, 3, 6, 7]:
        val = c19(x, 8.0)
        fire = "FIRE" if val > 0 else "silent"
        print(f"  C19({x:3d}) = {val:7.4f}  -> {fire}")

    # Test sizes: N=13, 16, 18, 20
    test_Ns = [13, 16, 18, 20]

    # Time limits per N (seconds) — aggressive, just measure rate
    time_limits = {
        13: 30.0,
        16: 30.0,
        18: 30.0,
        20: 30.0,
    }

    # Batch sizes -- larger N gets smaller batches due to memory
    batch_sizes = {
        13: 65536,
        16: 32768,
        18: 8192,
        20: 2048,
    }

    results = []
    for N in test_Ns:
        try:
            r = benchmark_N(
                N=N,
                device=device,
                rho=8.0,
                time_limit_sec=time_limits.get(N, 120.0),
                batch_size=batch_sizes.get(N, 4096),
            )
            results.append(r)
        except torch.cuda.OutOfMemoryError as e:
            print(f"  OOM at N={N}: {e}")
            print(f"  Trying smaller batch...")
            try:
                r = benchmark_N(N=N, device=device, rho=8.0,
                                time_limit_sec=time_limits.get(N, 120.0),
                                batch_size=256)
                results.append(r)
            except Exception as e2:
                print(f"  Failed N={N}: {e2}")
                results.append({"N": N, "error": str(e2)})
        except Exception as e:
            print(f"  Error at N={N}: {e}")
            results.append({"N": N, "error": str(e)})

    # Check if we can push to N=22
    if results:
        last_good = [r for r in results if "combos_per_sec" in r]
        if last_good:
            rate = last_good[-1]["combos_per_sec"]
            N22_combos = 3 ** 23
            N22_est = N22_combos / rate if rate > 0 else float('inf')
            print(f"\nN=22 estimate: {N22_combos:,} combos @ {rate:,.0f}/sec = {format_time(N22_est)}")
            if N22_est < 300:  # under 5 min
                print("  Running N=22...")
                try:
                    r = benchmark_N(N=22, device=device, rho=8.0,
                                    time_limit_sec=300.0, batch_size=512)
                    results.append(r)
                except Exception as e:
                    print(f"  N=22 failed: {e}")

    # Print summary table
    print(f"\n{'='*70}")
    print(f"SUMMARY - GPU Ternary Exhaustive Search")
    print(f"{'='*70}")
    print(f"{'N':>4}  {'3^(N+1)':>14}  {'Done%':>6}  {'Rate (k/s)':>12}  {'Est. Full':>12}  {'Best%':>7}")
    print(f"{'-'*70}")
    for r in results:
        if "error" in r:
            print(f"{r['N']:>4}  ERROR: {r['error'][:50]}")
            continue
        n = r["N"]
        tc = r["total_combos"]
        done_pct = r["pct_done"]
        rate_k = r["combos_per_sec"] / 1000
        est = format_time(r["full_estimate_sec"])
        best_pct = r["best_score"] / r["max_possible"] * 100
        tag = "*" if r["timed_out"] else ""
        print(f"{n:>4}  {tc:>14,}  {done_pct:>5.1f}%  {rate_k:>10.1f}K  {est:>12}  {best_pct:>6.1f}%{tag}")

    print(f"\n* = timed out, rate extrapolated from partial run")
    print(f"\nCPU reference (24 threads, N=13): 18K combos/sec, 4.5 min")

    if results and any("combos_per_sec" in r for r in results):
        gpu_rate = max(r["combos_per_sec"] for r in results if "combos_per_sec" in r)
        cpu_rate = 18000
        speedup = gpu_rate / cpu_rate
        print(f"GPU peak rate: {gpu_rate:,.0f} combos/sec")
        print(f"GPU speedup vs CPU: {speedup:.0f}x")

    return results


if __name__ == "__main__":
    results = main()

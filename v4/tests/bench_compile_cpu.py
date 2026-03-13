#!/usr/bin/env python3
"""
INSTNCT CPU torch.compile Benchmark
=====================================
Measures eager vs torch.compile speedup on CPU across compile modes,
dtypes, sequence lengths, and thread counts. Includes 10 adversarial
checks: numerical match, NaN/Inf, statistical rigor (N reps),
warmup verification, graph break counting, compile time measurement,
determinism, regression guard, silent fallback detection, memory tracking.

Usage:
  python v4/tests/bench_compile_cpu.py --quick          # smoke test (~2 min)
  python v4/tests/bench_compile_cpu.py --threads 4,8    # focused
  python v4/tests/bench_compile_cpu.py                  # full matrix
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import platform
import resource
import statistics
import sys
import time
from contextlib import nullcontext
from dataclasses import dataclass, field, asdict
from pathlib import Path

# ── Pre-import env var setup ──────────────────────────────────────────
# Must happen before torch import so OpenMP/MKL pick them up.
def _set_cpu_thread_limit(n: int) -> None:
    v = str(n)
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
              "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[k] = v

# Set to max available cores at startup; per-config override via torch.set_num_threads()
_set_cpu_thread_limit(os.cpu_count() or 8)

# ── Path setup ────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
for subdir in ("model", "training"):
    p = str(ROOT / subdir)
    if p not in sys.path:
        sys.path.insert(0, p)

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

try:
    import torch._dynamo  # noqa: E402
    HAS_DYNAMO = True
except ImportError:
    HAS_DYNAMO = False

from instnct import INSTNCT  # noqa: E402

# ── Constants ─────────────────────────────────────────────────────────
SEED = 42
COMPILE_WARMUP = 3   # steps for dynamo tracing (excluded from measurement)
MEAS_WARMUP = 2      # additional warmup steps (excluded)
HAS_COMPILE = hasattr(torch, 'compile')


# ── Data classes ──────────────────────────────────────────────────────
@dataclass
class ConfigSpec:
    compile_mode: str   # 'eager', 'default', 'max-autotune'
    dtype_mode: str     # 'fp32', 'bf16'
    seq_len: int
    num_threads: int
    batch_size: int = 0  # auto-filled

    @property
    def label(self) -> str:
        base = self.compile_mode if self.compile_mode == 'eager' else f"compile({self.compile_mode})"
        if self.dtype_mode == 'bf16':
            base += '+bf16'
        return base


@dataclass
class BenchResult:
    config: ConfigSpec
    ms_per_step: list[float] = field(default_factory=list)
    losses: list[float] = field(default_factory=list)
    compile_call_s: float = 0.0
    tracing_s: float = 0.0
    mem_rss_kb: int = 0
    graph_breaks: int = -1  # -1 = N/A (eager)
    nan_detected: bool = False
    fallback_detected: bool = False
    warmup_outlier: bool = False
    error: str = ''
    actual_batch_size: int = 0

    @property
    def mean_ms(self) -> float:
        return statistics.mean(self.ms_per_step) if self.ms_per_step else 0.0

    @property
    def std_ms(self) -> float:
        return statistics.stdev(self.ms_per_step) if len(self.ms_per_step) > 1 else 0.0

    @property
    def mem_mb(self) -> float:
        return self.mem_rss_kb / 1024.0


# ── Platform detection ────────────────────────────────────────────────
def check_bf16_support() -> bool:
    try:
        torch.ones(1, dtype=torch.bfloat16)
        return True
    except Exception:
        return False


def print_platform_header() -> None:
    bf16 = check_bf16_support()
    mkl = torch.backends.mkl.is_available() if hasattr(torch.backends, 'mkl') else False
    print("=" * 70)
    print("INSTNCT CPU Compile Benchmark")
    print("=" * 70)
    print(f"  PyTorch:        {torch.__version__}")
    print(f"  CPU:            {platform.processor() or platform.machine()} ({os.cpu_count()} cores)")
    print(f"  MKL:            {mkl}")
    print(f"  BFloat16:       {bf16}")
    print(f"  torch.compile:  {HAS_COMPILE}")
    print(f"  torch._dynamo:  {HAS_DYNAMO}")
    print(f"  OMP_NUM_THREADS: {os.environ.get('OMP_NUM_THREADS', 'unset')}")
    print(f"  Note: mode='reduce-overhead' skipped (uses CUDA graphs, CPU-only here)")
    print("=" * 70)
    print()


# ── Model construction ────────────────────────────────────────────────
def build_model() -> INSTNCT:
    model = INSTNCT(
        M=128, hidden_dim=4096, slot_dim=128, N=1, R=2,
        embed_mode=True, kernel_mode='vshape',
        pointer_mode='sequential', write_mode='replace',
        embed_encoding='learned', output_encoding='lowrank_c19',
        c19_mode='dualphi',
        pointer_interp_mode='linear',
        pointer_seam_mode='shortest_arc',
    )
    model._diag_enabled = False
    assert not model._proxy_overlay_enabled, \
        "proxy_overlay must be disabled for compile benchmark (causes .item() graph breaks)"
    model.train()
    return model


# ── Config matrix generation ──────────────────────────────────────────
def generate_config_matrix(args) -> list[ConfigSpec]:
    compile_modes = [m.strip() for m in args.compile_modes.split(',')]
    dtypes = [d.strip() for d in args.dtypes.split(',')]
    seq_lens = [int(s) for s in args.seq_lens.split(',')]
    threads = sorted(set(int(t) for t in args.threads.split(',')))

    bf16_ok = check_bf16_support()
    configs = []
    for cm in compile_modes:
        if cm != 'eager' and not HAS_COMPILE:
            print(f"  [WARN] torch.compile not available, skipping compile_mode='{cm}'")
            continue
        for dt in dtypes:
            if dt == 'bf16' and not bf16_ok:
                print(f"  [WARN] BFloat16 not supported on this CPU, skipping dtype='bf16'")
                continue
            for sl in seq_lens:
                bs = args.batch_size if args.batch_size > 0 else (64 if sl <= 48 else 16)
                for thr in threads:
                    if thr > (os.cpu_count() or 1):
                        print(f"  [WARN] threads={thr} exceeds cpu_count={os.cpu_count()}, proceeding anyway")
                    configs.append(ConfigSpec(
                        compile_mode=cm, dtype_mode=dt,
                        seq_len=sl, num_threads=thr, batch_size=bs,
                    ))
    return configs


# ── Single step execution ─────────────────────────────────────────────
def _run_step(model, optimizer, x, tgt, autocast_ctx):
    with autocast_ctx:
        out, state_out = model(x, state=getattr(model, '_bench_state', None))
        loss = F.cross_entropy(out.reshape(-1, out.size(-1)), tgt.reshape(-1))
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    # Detach state
    if state_out is not None:
        state_out = {k: v.detach() if isinstance(v, torch.Tensor) else v
                     for k, v in state_out.items()}
    model._bench_state = state_out
    return loss.item()


# ── Single benchmark run ──────────────────────────────────────────────
def run_single_benchmark(config: ConfigSpec, measure_steps: int,
                         seed: int, verbose: bool) -> BenchResult:
    result = BenchResult(config=config, actual_batch_size=config.batch_size)
    B, T = config.batch_size, config.seq_len

    # Thread setup
    torch.set_num_threads(config.num_threads)

    # Build model
    torch.manual_seed(seed)
    try:
        model = build_model()
    except Exception as e:
        # OOM or other failure — try halved batch
        if B > 4:
            B = B // 2
            result.actual_batch_size = B
            try:
                model = build_model()
            except Exception as e2:
                result.error = str(e2)
                return result
        else:
            result.error = str(e)
            return result

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model._bench_state = None

    # Graph break counter snapshot (before compile)
    breaks_before = 0
    frames_before = 0
    ok_before = 0
    if HAS_DYNAMO:
        breaks_before = sum(torch._dynamo.utils.counters["graph_break"].values())
        frames_before = sum(torch._dynamo.utils.counters.get("frames", {}).values())
        ok_before = sum(torch._dynamo.utils.counters.get("ok", {}).values())

    # Apply compile
    if config.compile_mode != 'eager':
        if HAS_DYNAMO:
            torch._dynamo.reset()
            breaks_before = sum(torch._dynamo.utils.counters["graph_break"].values())
            frames_before = sum(torch._dynamo.utils.counters.get("frames", {}).values())
            ok_before = sum(torch._dynamo.utils.counters.get("ok", {}).values())
        t_comp = time.perf_counter()
        model = torch.compile(model, mode=config.compile_mode)
        result.compile_call_s = time.perf_counter() - t_comp

    # Autocast context
    if config.dtype_mode == 'bf16':
        autocast_ctx = torch.autocast('cpu', dtype=torch.bfloat16)
    else:
        autocast_ctx = nullcontext()

    # Memory snapshot before
    rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    # ── Compile warmup (tracing) ──
    t_trace = time.perf_counter()
    torch.manual_seed(seed)
    for i in range(COMPILE_WARMUP):
        x = torch.randint(0, 256, (B, T))
        tgt = torch.randint(0, 256, (B, T))
        try:
            _run_step(model, optimizer, x, tgt, autocast_ctx)
        except Exception as e:
            result.error = f"compile warmup step {i}: {e}"
            del model, optimizer
            gc.collect()
            return result
    result.tracing_s = time.perf_counter() - t_trace

    # ── Measurement warmup ──
    for _ in range(MEAS_WARMUP):
        x = torch.randint(0, 256, (B, T))
        tgt = torch.randint(0, 256, (B, T))
        _run_step(model, optimizer, x, tgt, autocast_ctx)

    # ── Measurement ──
    step_times = []
    losses = []
    torch.manual_seed(seed + 1000)  # different data seed for measurement
    for i in range(measure_steps):
        x = torch.randint(0, 256, (B, T))
        tgt = torch.randint(0, 256, (B, T))
        t0 = time.perf_counter()
        loss_val = _run_step(model, optimizer, x, tgt, autocast_ctx)
        t1 = time.perf_counter()
        ms = (t1 - t0) * 1000
        step_times.append(ms)
        losses.append(loss_val)
        if math.isnan(loss_val) or math.isinf(loss_val):
            result.nan_detected = True
        if verbose:
            print(f"    step {i}: {ms:.1f} ms, loss={loss_val:.4f}")

    result.ms_per_step = step_times
    result.losses = losses

    # Memory snapshot after
    rss_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result.mem_rss_kb = rss_after - rss_before

    # Graph break counting
    if HAS_DYNAMO and config.compile_mode != 'eager':
        breaks_after = sum(torch._dynamo.utils.counters["graph_break"].values())
        frames_after = sum(torch._dynamo.utils.counters.get("frames", {}).values())
        ok_after = sum(torch._dynamo.utils.counters.get("ok", {}).values())
        result.graph_breaks = breaks_after - breaks_before
        # Silent fallback detection
        frames_delta = frames_after - frames_before
        ok_delta = ok_after - ok_before
        if frames_delta > ok_delta:
            result.fallback_detected = True

    # Warmup outlier check: first measurement step > 3σ from rest
    if len(step_times) > 3:
        rest = step_times[1:]
        mu = statistics.mean(rest)
        sigma = statistics.stdev(rest) if len(rest) > 1 else 0.0
        if sigma > 0 and abs(step_times[0] - mu) > 3 * sigma:
            result.warmup_outlier = True

    # Cleanup
    del model, optimizer
    gc.collect()
    return result


# ── Determinism verification ──────────────────────────────────────────
def verify_determinism(config: ConfigSpec, measure_steps: int,
                       seed: int, verbose: bool) -> tuple[bool, str]:
    """Run same config twice with same seed, compare losses."""
    r1 = run_single_benchmark(config, measure_steps, seed, verbose=False)
    r2 = run_single_benchmark(config, measure_steps, seed, verbose=False)
    if r1.error or r2.error:
        return True, "skipped (error)"
    if not r1.losses or not r2.losses:
        return True, "skipped (no losses)"
    rtol = 1e-7 if config.compile_mode != 'eager' else 0.0
    for i, (a, b) in enumerate(zip(r1.losses, r2.losses)):
        if rtol == 0.0:
            if a != b:
                return False, f"step {i}: {a} != {b} (bitwise)"
        else:
            if abs(a - b) > rtol * max(abs(a), 1e-8):
                return False, f"step {i}: {a} vs {b} (rtol={rtol})"
    return True, "OK"


# ── Numerical match (eager vs compile) ────────────────────────────────
def verify_numerical_match(eager_losses: list[float],
                           compile_losses: list[float],
                           dtype_mode: str) -> tuple[bool, str]:
    rtol = 1e-3 if dtype_mode == 'fp32' else 5e-2
    for i, (a, b) in enumerate(zip(eager_losses, compile_losses)):
        rel = abs(a - b) / max(abs(a), 1e-8)
        if rel > rtol:
            return False, f"step {i}: eager={a:.6f} vs compile={b:.6f} (rel={rel:.4f}, rtol={rtol})"
    return True, "OK"


# ── Multi-rep runner ──────────────────────────────────────────────────
@dataclass
class AggResult:
    config: ConfigSpec
    mean_ms: float = 0.0
    std_ms: float = 0.0
    min_ms: float = 0.0
    max_ms: float = 0.0
    losses: list[float] = field(default_factory=list)
    compile_call_s: float = 0.0
    tracing_s: float = 0.0
    mem_mb: float = 0.0
    graph_breaks: int = -1
    nan_detected: bool = False
    fallback_detected: bool = False
    warmup_outlier: bool = False
    error: str = ''
    actual_batch_size: int = 0


def run_multi_rep(config: ConfigSpec, measure_steps: int, reps: int,
                  seed: int, verbose: bool) -> AggResult:
    agg = AggResult(config=config)
    all_means = []
    for rep in range(reps):
        rep_seed = seed + rep * 777
        r = run_single_benchmark(config, measure_steps, rep_seed, verbose)
        if r.error:
            agg.error = r.error
            return agg
        all_means.append(r.mean_ms)
        agg.nan_detected = agg.nan_detected or r.nan_detected
        agg.fallback_detected = agg.fallback_detected or r.fallback_detected
        agg.warmup_outlier = agg.warmup_outlier or r.warmup_outlier
        agg.graph_breaks = r.graph_breaks
        agg.compile_call_s = r.compile_call_s
        agg.tracing_s = r.tracing_s
        agg.mem_mb = r.mem_mb
        agg.actual_batch_size = r.actual_batch_size
        # Keep losses from last rep for numerical comparison
        agg.losses = r.losses

    agg.mean_ms = statistics.mean(all_means)
    agg.std_ms = statistics.stdev(all_means) if len(all_means) > 1 else 0.0
    agg.min_ms = min(all_means)
    agg.max_ms = max(all_means)
    return agg


# ── Output formatting ─────────────────────────────────────────────────
def print_summary_table(results: list[AggResult], warnings: list[str]) -> None:
    print()
    print("=" * 110)
    print("  RESULTS")
    print("=" * 110)
    header = (f"  {'Config':<28} {'T':>4} {'Thr':>4} {'ms/step':>8} {'±std':>7} "
              f"{'speedup':>8} {'mem_MB':>7} {'breaks':>7} {'compile_s':>10} {'NaN':>4}")
    print(header)
    print(f"  {'-'*28} {'-'*4} {'-'*4} {'-'*8} {'-'*7} {'-'*8} {'-'*7} {'-'*7} {'-'*10} {'-'*4}")

    # Group by (T, threads) for speedup calculation
    baselines: dict[tuple[int, int], float] = {}
    for r in results:
        c = r.config
        key = (c.seq_len, c.num_threads)
        if c.compile_mode == 'eager' and c.dtype_mode == 'fp32':
            baselines[key] = r.mean_ms

    for r in results:
        if r.error:
            print(f"  {r.config.label:<28} {r.config.seq_len:>4} {r.config.num_threads:>4} "
                  f"{'ERROR':>8} {'-':>7} {'-':>8} {'-':>7} {'-':>7} {'-':>10} {'-':>4}  {r.error}")
            continue

        c = r.config
        key = (c.seq_len, c.num_threads)
        baseline = baselines.get(key, 0.0)

        # Speedup
        if baseline > 0 and r.mean_ms > 0:
            pct = (baseline - r.mean_ms) / baseline * 100
            if c.compile_mode == 'eager' and c.dtype_mode == 'fp32':
                speedup_str = "1.00x"
            elif pct >= 0:
                speedup_str = f"+{pct:.1f}%"
            else:
                speedup_str = f"{pct:.1f}%"
                warnings.append(f"[REGRESSION] {c.label} @ T={c.seq_len}, {c.num_threads} threads: {pct:.1f}%")
        else:
            speedup_str = "-"

        breaks_str = str(r.graph_breaks) if r.graph_breaks >= 0 else "-"
        compile_str = f"{r.compile_call_s + r.tracing_s:.1f}" if c.compile_mode != 'eager' else "-"
        nan_str = "FAIL" if r.nan_detected else "OK"

        if r.fallback_detected:
            warnings.append(f"[FALLBACK] {c.label} @ T={c.seq_len}, {c.num_threads} threads: "
                            f"silent eager fallback detected")
        if r.warmup_outlier:
            warnings.append(f"[WARMUP] {c.label} @ T={c.seq_len}, {c.num_threads} threads: "
                            f"first measurement step is outlier (warmup may be insufficient)")
        if r.nan_detected:
            warnings.append(f"[NaN] {c.label} @ T={c.seq_len}, {c.num_threads} threads")
        if r.graph_breaks > 0:
            warnings.append(f"[GRAPH_BREAK] {c.label} @ T={c.seq_len}, {c.num_threads} threads: "
                            f"{r.graph_breaks} graph breaks")

        print(f"  {c.label:<28} {c.seq_len:>4} {c.num_threads:>4} "
              f"{r.mean_ms:>8.1f} {r.std_ms:>6.1f}s "
              f"{speedup_str:>8} {r.mem_mb:>7.0f} {breaks_str:>7} "
              f"{compile_str:>10} {nan_str:>4}")

    print("=" * 110)


def print_best_config(results: list[AggResult]) -> None:
    valid = [r for r in results if not r.error and not r.nan_detected]
    if not valid:
        print("\n  No valid results to determine best config.")
        return

    # Group by seq_len
    by_sl: dict[int, list[AggResult]] = {}
    for r in valid:
        by_sl.setdefault(r.config.seq_len, []).append(r)

    print()
    for sl, group in sorted(by_sl.items()):
        best = min(group, key=lambda r: r.mean_ms)
        c = best.config
        baselines = [r for r in group
                     if r.config.compile_mode == 'eager' and r.config.dtype_mode == 'fp32'
                     and r.config.num_threads == c.num_threads]
        speedup_str = ""
        if baselines:
            bl = baselines[0].mean_ms
            pct = (bl - best.mean_ms) / bl * 100
            speedup_str = f" ({pct:+.1f}% vs eager/fp32)"
        print(f"  BEST @ T={sl}: {c.label}, {c.num_threads} threads "
              f"-> {best.mean_ms:.1f} ms/step{speedup_str}")


def print_warnings(warnings: list[str]) -> None:
    if not warnings:
        print("\n  No warnings. All checks passed.")
        return
    print()
    print("WARNINGS:")
    for w in warnings:
        print(f"  {w}")


# ── CLI ───────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="INSTNCT CPU torch.compile benchmark")
    p.add_argument('--batch-size', type=int, default=0,
                   help="Batch size (0=auto: 64 for T<=48, 16 for T=256)")
    p.add_argument('--measure-steps', type=int, default=10,
                   help="Measurement steps per rep (default: 10)")
    p.add_argument('--reps', type=int, default=3,
                   help="Repetitions per config (default: 3)")
    p.add_argument('--threads', type=str, default="1,2,4,8,16",
                   help="Comma-separated thread counts (default: 1,2,4,8,16)")
    p.add_argument('--seq-lens', type=str, default="48,256",
                   help="Comma-separated sequence lengths (default: 48,256)")
    p.add_argument('--compile-modes', type=str, default="eager,default,max-autotune",
                   help="Comma-separated compile modes (default: eager,default,max-autotune)")
    p.add_argument('--dtypes', type=str, default="fp32,bf16",
                   help="Comma-separated dtypes (default: fp32,bf16)")
    p.add_argument('--quick', action='store_true',
                   help="Quick mode: threads=4,8 T=48 reps=1 steps=5")
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--json-out', type=str, default='',
                   help="Path to write JSON results")
    p.add_argument('--verbose', action='store_true',
                   help="Print per-step timings")
    p.add_argument('--skip-determinism', action='store_true',
                   help="Skip determinism check")
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    # Quick mode overrides
    if args.quick:
        args.threads = "4,8"
        args.seq_lens = "48"
        args.reps = 1
        args.measure_steps = 5

    print_platform_header()

    configs = generate_config_matrix(args)
    total = len(configs)
    print(f"  Running {total} configs × {args.reps} reps × {args.measure_steps} steps each")
    print(f"  Compile warmup: {COMPILE_WARMUP} steps, measurement warmup: {MEAS_WARMUP} steps")
    print()

    # ── Run benchmarks ──
    results: list[AggResult] = []
    warnings: list[str] = []
    t_total = time.perf_counter()

    for idx, cfg in enumerate(configs, 1):
        print(f"[{idx}/{total}] {cfg.label}  T={cfg.seq_len}  threads={cfg.num_threads}  "
              f"B={cfg.batch_size} ...", end="", flush=True)
        t0 = time.perf_counter()
        agg = run_multi_rep(cfg, args.measure_steps, args.reps, args.seed, args.verbose)
        elapsed = time.perf_counter() - t0
        if agg.error:
            print(f" ERROR ({elapsed:.0f}s): {agg.error}")
        else:
            print(f" {agg.mean_ms:.1f} ±{agg.std_ms:.1f} ms/step ({elapsed:.0f}s)")
        results.append(agg)

    # ── Numerical match: eager vs compile ──
    # Group by (T, threads, dtype) and compare
    eager_map: dict[tuple[int, int, str], AggResult] = {}
    for r in results:
        c = r.config
        if c.compile_mode == 'eager' and not r.error:
            eager_map[(c.seq_len, c.num_threads, c.dtype_mode)] = r

    for r in results:
        c = r.config
        if c.compile_mode == 'eager' or r.error:
            continue
        key = (c.seq_len, c.num_threads, c.dtype_mode)
        eager = eager_map.get(key)
        if eager and eager.losses and r.losses:
            ok, msg = verify_numerical_match(eager.losses, r.losses, c.dtype_mode)
            if not ok:
                warnings.append(f"[NUMERICAL] {c.label} @ T={c.seq_len}, {c.num_threads} threads: {msg}")

    # ── Determinism check ──
    if not args.skip_determinism:
        print("\n  Running determinism checks...", flush=True)
        # Pick one config per compile_mode to check (save time)
        checked_modes = set()
        for cfg in configs:
            if cfg.compile_mode in checked_modes:
                continue
            checked_modes.add(cfg.compile_mode)
            print(f"    {cfg.label} T={cfg.seq_len} threads={cfg.num_threads} ...", end="", flush=True)
            ok, msg = verify_determinism(cfg, min(args.measure_steps, 5), args.seed, False)
            if ok:
                print(f" {msg}")
            else:
                print(f" NONDETERMINISTIC: {msg}")
                warnings.append(f"[DETERMINISM] {cfg.label}: {msg}")

    # ── Output ──
    elapsed_total = time.perf_counter() - t_total
    print_summary_table(results, warnings)
    print_best_config(results)
    print_warnings(warnings)
    print(f"\n  Total benchmark time: {elapsed_total:.0f}s")

    # ── JSON output ──
    if args.json_out:
        json_data = []
        for r in results:
            entry = {
                'compile_mode': r.config.compile_mode,
                'dtype_mode': r.config.dtype_mode,
                'seq_len': r.config.seq_len,
                'num_threads': r.config.num_threads,
                'batch_size': r.actual_batch_size or r.config.batch_size,
                'mean_ms': r.mean_ms,
                'std_ms': r.std_ms,
                'min_ms': r.min_ms,
                'max_ms': r.max_ms,
                'compile_call_s': r.compile_call_s,
                'tracing_s': r.tracing_s,
                'mem_mb': r.mem_mb,
                'graph_breaks': r.graph_breaks,
                'nan_detected': r.nan_detected,
                'fallback_detected': r.fallback_detected,
                'warmup_outlier': r.warmup_outlier,
                'error': r.error,
                'losses': r.losses,
            }
            json_data.append(entry)
        with open(args.json_out, 'w') as f:
            json.dump({'results': json_data, 'warnings': warnings,
                       'torch_version': torch.__version__,
                       'total_time_s': elapsed_total}, f, indent=2)
        print(f"  JSON results written to: {args.json_out}")

    # Exit code: 1 if any NaN detected
    if any(r.nan_detected for r in results):
        sys.exit(1)


if __name__ == '__main__':
    main()

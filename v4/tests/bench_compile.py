"""Benchmark: sequential vs parallel expert ring access in INSTNCT.

Sequential (current): experts read/write one by one, expert i+1 sees expert i's write.
  → N ring clones per timestep (N=6 → 384 clones for T=64)

Parallel (new): all experts read same ring state, then coalesced write.
  → 1 ring clone per timestep (64 clones for T=64) — 6× fewer

Also tests torch.compile() on top of each mode.

Usage:
    python v4/tests/bench_compile.py
    python v4/tests/bench_compile.py --tokens 128 --runs 10
"""

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for subdir in ('model', 'training', 'datagen'):
    p = str(ROOT / subdir)
    if p not in sys.path:
        sys.path.insert(0, p)

import torch
from instnct import INSTNCT


# ── Model configs ──
CONFIGS = {
    'small_N1': dict(
        M=64, hidden_dim=128, slot_dim=32, N=1, R=1,
        embed_mode=True, kernel_mode='vshape', pointer_mode='pilot',
        write_mode='replace', embed_encoding='bitlift',
        output_encoding='lowrank_c19', checkpoint_chunks=0,
    ),
    'prod_N6': dict(
        M=64, hidden_dim=128, slot_dim=32, N=6, R=1,
        embed_mode=True, kernel_mode='vshape', pointer_mode='pilot',
        write_mode='replace', embed_encoding='bitlift',
        output_encoding='lowrank_c19', checkpoint_chunks=0,
    ),
}


def make_model(cfg, device, parallel=False):
    torch.manual_seed(42)
    model = INSTNCT(**cfg, parallel_experts=parallel).to(device)
    model.eval()
    return model


def bench(model, x, warmup, n_runs):
    """Benchmark forward pass latency."""
    for _ in range(warmup):
        with torch.no_grad():
            model(x)

    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        with torch.no_grad():
            out, _ = model(x)
        times.append(time.perf_counter() - t0)
    return times


def bench_compiled(model, x, warmup, n_runs, backend='inductor'):
    """Benchmark compiled forward pass."""
    try:
        compiled = torch.compile(model, backend=backend, fullgraph=False)
    except Exception as e:
        return None, str(e)

    for _ in range(warmup + 2):
        with torch.no_grad():
            compiled(x)

    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        with torch.no_grad():
            out, _ = compiled(x)
        times.append(time.perf_counter() - t0)
    return times, None


def stats(times):
    avg = sum(times) / len(times)
    std = (sum((t - avg)**2 for t in times) / len(times)) ** 0.5
    mn = min(times)
    return avg, std, mn


def check_output_equivalence(cfg, device, tokens):
    """Check that parallel produces similar outputs to sequential."""
    x = torch.randint(0, 256, (1, tokens), dtype=torch.long, device=device)

    torch.manual_seed(42)
    m_seq = INSTNCT(**cfg, parallel_experts=False).to(device)
    m_seq.eval()

    torch.manual_seed(42)
    m_par = INSTNCT(**cfg, parallel_experts=True).to(device)
    m_par.eval()

    with torch.no_grad():
        out_seq, state_seq = m_seq(x)
        out_par, state_par = m_par(x)

    # Output logits comparison
    diff = (out_seq - out_par).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    # Ring state comparison
    ring_diff = (state_seq['ring'] - state_par['ring']).abs().max().item()

    return max_diff, mean_diff, ring_diff


def run():
    import argparse
    parser = argparse.ArgumentParser(description='Sequential vs Parallel expert benchmark')
    parser.add_argument('--tokens', type=int, default=64, help='sequence length')
    parser.add_argument('--warmup', type=int, default=3, help='warmup iterations')
    parser.add_argument('--runs', type=int, default=10, help='timed runs')
    parser.add_argument('--device', default='cpu', help='cpu or cuda')
    parser.add_argument('--config', default='prod_N6', choices=CONFIGS.keys())
    parser.add_argument('--compile', action='store_true', help='also benchmark torch.compile')
    args = parser.parse_args()

    device = args.device
    cfg = CONFIGS[args.config]
    N = cfg['N']
    tokens = args.tokens

    print(f"{'=' * 70}")
    print(f"  Sequential vs Parallel Expert Benchmark — INSTNCT")
    print(f"  Config: {args.config} (N={N}, M={cfg['M']}, hidden={cfg['hidden_dim']})")
    print(f"  Tokens: {tokens}, Device: {device}")
    print(f"  PyTorch: {torch.__version__}")
    print(f"  Ring clones per forward: sequential={N}×{tokens}={N*tokens}, parallel=1×{tokens}={tokens}")
    print(f"{'=' * 70}")

    x = torch.randint(0, 256, (1, tokens), dtype=torch.long, device=device)

    # ── Output equivalence check ──
    print(f"\n  Checking output equivalence...")
    max_d, mean_d, ring_d = check_output_equivalence(cfg, device, tokens)
    print(f"    Logit max diff:  {max_d:.6f}")
    print(f"    Logit mean diff: {mean_d:.6f}")
    print(f"    Ring max diff:   {ring_d:.6f}")
    if max_d > 1.0:
        print(f"    WARNING: outputs diverge significantly (expected for different semantics)")
    else:
        print(f"    OK: outputs are close (same weights, slightly different expert ordering)")

    # ── Sequential baseline ──
    print(f"\n  [1/2] Sequential experts (baseline)...")
    model_seq = make_model(cfg, device, parallel=False)
    times_seq = bench(model_seq, x, args.warmup, args.runs)
    avg_s, std_s, min_s = stats(times_seq)
    per_tok_s = avg_s / tokens * 1000
    print(f"        avg={avg_s*1000:.1f}ms  std={std_s*1000:.1f}ms  "
          f"min={min_s*1000:.1f}ms  ({per_tok_s:.2f} ms/tok)")

    # ── Parallel experts ──
    print(f"\n  [2/2] Parallel experts (coalesced write)...")
    model_par = make_model(cfg, device, parallel=True)
    times_par = bench(model_par, x, args.warmup, args.runs)
    avg_p, std_p, min_p = stats(times_par)
    per_tok_p = avg_p / tokens * 1000
    speedup = avg_s / avg_p
    print(f"        avg={avg_p*1000:.1f}ms  std={std_p*1000:.1f}ms  "
          f"min={min_p*1000:.1f}ms  ({per_tok_p:.2f} ms/tok)")
    print(f"        Speedup: {speedup:.2f}x vs sequential")

    # ── Optional: torch.compile on top ──
    if args.compile:
        print(f"\n  [bonus] Sequential + torch.compile...")
        model_seq_c = make_model(cfg, device, parallel=False)
        times_sc, err = bench_compiled(model_seq_c, x, args.warmup, args.runs)
        if err:
            print(f"        FAILED: {err}")
        else:
            avg_sc, std_sc, min_sc = stats(times_sc)
            print(f"        avg={avg_sc*1000:.1f}ms ({avg_s/avg_sc:.2f}x vs seq eager)")

        print(f"\n  [bonus] Parallel + torch.compile...")
        model_par_c = make_model(cfg, device, parallel=True)
        times_pc, err = bench_compiled(model_par_c, x, args.warmup, args.runs)
        if err:
            print(f"        FAILED: {err}")
        else:
            avg_pc, std_pc, min_pc = stats(times_pc)
            print(f"        avg={avg_pc*1000:.1f}ms ({avg_s/avg_pc:.2f}x vs seq eager)")

    # ── Summary ──
    print(f"\n{'=' * 70}")
    print(f"  SUMMARY")
    print(f"{'=' * 70}")
    print(f"  {'Method':<30} {'ms/tok':>10} {'Speedup':>10}")
    print(f"  {'-'*50}")
    print(f"  {'Sequential (baseline)':<30} {per_tok_s:>9.2f}ms {'1.00x':>10}")
    print(f"  {'Parallel (coalesced)':<30} {per_tok_p:>9.2f}ms {f'{speedup:.2f}x':>10}")
    print()
    saved_ms = per_tok_s - per_tok_p
    saved_pct = saved_ms / per_tok_s * 100 if per_tok_s > 0 else 0
    if saved_ms > 0:
        print(f"  Parallel saves {saved_ms:.2f} ms/tok ({saved_pct:.1f}% of total)")
        print(f"  Ring clones reduced: {N*tokens} → {tokens} ({N}× fewer)")
    else:
        print(f"  No speedup from parallel mode ({saved_pct:.1f}%)")
        print(f"  Bottleneck is elsewhere (likely the Python for-t-in-range(C) loop)")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    run()

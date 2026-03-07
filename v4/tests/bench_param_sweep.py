#!/usr/bin/env python3
"""
INSTNCT CPU Parameter Sweep Benchmark
======================================
Measures steps/sec for each hyperparameter in isolation and in pairwise
combinations to find the optimal CPU training config.

Output: tables showing individual cost scaling + interaction effects.
"""

import sys, os, time, gc, itertools
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'model'))

import torch
from instnct import INSTNCT

torch.set_num_threads(16)

# ── Baseline config (fast, ~1-2 steps/sec) ──────────────────────
BASELINE = dict(
    hidden_dim=256, slot_dim=64, M=128, N=1, R=1, B=16,
    kernel_mode='vshape', pointer_mode='sequential',
    embed_mode=True, embed_encoding='bitlift', output_encoding='lowrank_c19',
    write_mode='replace',
)
BASELINE_SEQ = 64
BASELINE_BATCH = 16

WARMUP_STEPS = 2
BENCH_STEPS = 5


def bench_config(model_kwargs, seq_len, batch_size, label=""):
    """Time a config, return dict with timing info."""
    try:
        model_kwargs = {**model_kwargs, 'B': batch_size}
        model = INSTNCT(**model_kwargs)
        params = sum(p.numel() for p in model.parameters())
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        x = torch.randint(0, 256, (batch_size, seq_len))
        targets = torch.randint(0, 256, (batch_size, seq_len))

        # Warmup
        for _ in range(WARMUP_STEPS):
            logits, _ = model(x, state=None)
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, 256), targets.reshape(-1))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # Timed
        t0 = time.time()
        for _ in range(BENCH_STEPS):
            logits, _ = model(x, state=None)
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, 256), targets.reshape(-1))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        elapsed = time.time() - t0

        sps = BENCH_STEPS / elapsed
        ms = elapsed / BENCH_STEPS * 1000

        del model, optimizer, logits, loss, x, targets
        gc.collect()

        return dict(label=label, params=params, sps=sps, ms=ms, ok=True)
    except Exception as e:
        gc.collect()
        return dict(label=label, params=0, sps=0, ms=0, ok=False, err=str(e))


def print_table(title, rows):
    """Print a formatted results table."""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")
    print(f"  {'Config':<42} {'Params':>8} {'steps/s':>8} {'ms/step':>8}")
    print(f"  {'-'*42} {'-'*8} {'-'*8} {'-'*8}")
    for r in rows:
        if r['ok']:
            print(f"  {r['label']:<42} {r['params']:>8,} {r['sps']:>8.2f} {r['ms']:>8.0f}")
        else:
            print(f"  {r['label']:<42} {'ERROR':>8} {'-':>8} {'-':>8}")


def run_single_sweeps():
    """Vary one parameter at a time, everything else at baseline."""
    all_results = {}

    # ── hidden_dim ──
    rows = []
    for v in [128, 256, 512, 1024, 2048]:
        kw = {**BASELINE, 'hidden_dim': v}
        rows.append(bench_config(kw, BASELINE_SEQ, BASELINE_BATCH,
                                 label=f"hidden_dim={v}"))
    print_table("SWEEP: hidden_dim (others at baseline)", rows)
    all_results['hidden_dim'] = rows

    # ── slot_dim ──
    rows = []
    for v in [32, 64, 128, 256]:
        kw = {**BASELINE, 'slot_dim': v}
        rows.append(bench_config(kw, BASELINE_SEQ, BASELINE_BATCH,
                                 label=f"slot_dim={v}"))
    print_table("SWEEP: slot_dim", rows)
    all_results['slot_dim'] = rows

    # ── M (ring slots) ──
    rows = []
    for v in [64, 128, 256, 512, 1024]:
        kw = {**BASELINE, 'M': v}
        rows.append(bench_config(kw, BASELINE_SEQ, BASELINE_BATCH,
                                 label=f"M={v}"))
    print_table("SWEEP: M (ring slots) — vshape kernel", rows)
    all_results['M'] = rows

    # ── seq_len ──
    rows = []
    for v in [32, 64, 128, 256]:
        kw = {**BASELINE}
        rows.append(bench_config(kw, v, BASELINE_BATCH,
                                 label=f"seq_len={v}"))
    print_table("SWEEP: seq_len", rows)
    all_results['seq_len'] = rows

    # ── batch_size ──
    rows = []
    for v in [4, 8, 16, 32, 64]:
        kw = {**BASELINE, 'B': v}
        rows.append(bench_config(kw, BASELINE_SEQ, v,
                                 label=f"batch_size={v}"))
    print_table("SWEEP: batch_size", rows)
    all_results['batch_size'] = rows

    # ── N (experts) ──
    rows = []
    for v in [1, 2, 4, 6]:
        kw = {**BASELINE, 'N': v}
        rows.append(bench_config(kw, BASELINE_SEQ, BASELINE_BATCH,
                                 label=f"N={v}"))
    print_table("SWEEP: N (experts)", rows)
    all_results['N'] = rows

    # ── R (attention radius) ──
    rows = []
    for v in [0, 1, 2, 4]:
        kw = {**BASELINE, 'R': v}
        rows.append(bench_config(kw, BASELINE_SEQ, BASELINE_BATCH,
                                 label=f"R={v}"))
    print_table("SWEEP: R (attention radius)", rows)
    all_results['R'] = rows

    # ── pointer_mode ──
    rows = []
    for v in ['sequential', 'learned', 'pilot']:
        kw = {**BASELINE, 'pointer_mode': v}
        rows.append(bench_config(kw, BASELINE_SEQ, BASELINE_BATCH,
                                 label=f"pointer_mode={v}"))
    print_table("SWEEP: pointer_mode", rows)
    all_results['pointer_mode'] = rows

    return all_results


def run_interaction_sweeps():
    """Vary pairs of parameters to detect non-additive interactions."""
    all_results = {}

    # ── 1. slot_dim × seq_len ──
    rows = []
    for sd, sl in itertools.product([32, 64, 128], [32, 64, 128]):
        kw = {**BASELINE, 'slot_dim': sd}
        rows.append(bench_config(kw, sl, BASELINE_BATCH,
                                 label=f"slot_dim={sd:>3} × seq_len={sl:>3}"))
    print_table("INTERACTION: slot_dim × seq_len", rows)
    all_results['slot_dim×seq_len'] = rows

    # ── 2. hidden_dim × slot_dim ──
    rows = []
    for hd, sd in itertools.product([128, 256, 512, 1024], [32, 64, 128]):
        kw = {**BASELINE, 'hidden_dim': hd, 'slot_dim': sd}
        rows.append(bench_config(kw, BASELINE_SEQ, BASELINE_BATCH,
                                 label=f"hidden={hd:>4} × slot={sd:>3}"))
    print_table("INTERACTION: hidden_dim × slot_dim", rows)
    all_results['hidden_dim×slot_dim'] = rows

    # ── 3. M × kernel_mode (vshape vs topk) ──
    rows = []
    for m_val in [64, 128, 256, 512]:
        for km in ['vshape', 'dotprod']:
            kw = {**BASELINE, 'M': m_val, 'kernel_mode': km}
            rows.append(bench_config(kw, BASELINE_SEQ, BASELINE_BATCH,
                                     label=f"M={m_val:>4} × kernel={km}"))
    print_table("INTERACTION: M × kernel_mode", rows)
    all_results['M×kernel'] = rows

    # ── 4. N × seq_len ──
    rows = []
    for n_val, sl in itertools.product([1, 2, 4], [32, 64, 128]):
        kw = {**BASELINE, 'N': n_val}
        rows.append(bench_config(kw, sl, BASELINE_BATCH,
                                 label=f"N={n_val} × seq_len={sl:>3}"))
    print_table("INTERACTION: N × seq_len", rows)
    all_results['N×seq_len'] = rows

    # ── 5. hidden_dim × pointer_mode ──
    rows = []
    for hd in [128, 256, 512, 1024]:
        for pm in ['sequential', 'pilot']:
            kw = {**BASELINE, 'hidden_dim': hd, 'pointer_mode': pm}
            rows.append(bench_config(kw, BASELINE_SEQ, BASELINE_BATCH,
                                     label=f"hidden={hd:>4} × ptr={pm}"))
    print_table("INTERACTION: hidden_dim × pointer_mode", rows)
    all_results['hidden_dim×pointer'] = rows

    # ── 6. batch_size × hidden_dim ──
    rows = []
    for bs, hd in itertools.product([8, 16, 32], [128, 256, 512, 1024]):
        kw = {**BASELINE, 'hidden_dim': hd, 'B': bs}
        rows.append(bench_config(kw, BASELINE_SEQ, bs,
                                 label=f"batch={bs:>2} × hidden={hd:>4}"))
    print_table("INTERACTION: batch_size × hidden_dim", rows)
    all_results['batch×hidden'] = rows

    return all_results


def find_optimal(single_results):
    """Based on single sweep data, propose top configs to verify."""
    print(f"\n{'='*70}")
    print(f"  OPTIMAL CONFIG CANDIDATES")
    print(f"{'='*70}")

    # Find sweet spots from each sweep
    print("\n  Per-parameter sweet spots (best speed with reasonable capacity):")
    for param, rows in single_results.items():
        ok_rows = [r for r in rows if r['ok']]
        if not ok_rows:
            continue
        # Find fastest
        fastest = max(ok_rows, key=lambda r: r['sps'])
        # Find best capacity/speed ratio (params × sps)
        best_ratio = max(ok_rows, key=lambda r: r['params'] * r['sps'])
        print(f"  {param:>15}: fastest={fastest['label']} ({fastest['sps']:.2f} sps)"
              f"  |  best ratio={best_ratio['label']}")

    # Test a few promising combos
    print("\n  Testing top candidate configs...")
    candidates = [
        ("MaxSpeed: H=128 SD=32 M=256 seq=32 B=32 N=1",
         dict(hidden_dim=128, slot_dim=32, M=256, N=1, R=1,
              pointer_mode='sequential'), 32, 32),
        ("Balanced: H=512 SD=64 M=512 seq=64 B=16 N=1",
         dict(hidden_dim=512, slot_dim=64, M=512, N=1, R=1,
              pointer_mode='sequential'), 64, 16),
        ("HighCap: H=1024 SD=64 M=512 seq=64 B=8 N=1",
         dict(hidden_dim=1024, slot_dim=64, M=512, N=1, R=1,
              pointer_mode='sequential'), 64, 8),
        ("HighCap+Pilot: H=1024 SD=64 M=512 seq=64 B=8 N=1",
         dict(hidden_dim=1024, slot_dim=64, M=512, N=1, R=1,
              pointer_mode='pilot'), 64, 8),
        ("BigRing: H=512 SD=64 M=1024 seq=64 B=16 N=1",
         dict(hidden_dim=512, slot_dim=64, M=1024, N=1, R=1,
              pointer_mode='sequential'), 64, 16),
        ("2Experts: H=512 SD=64 M=256 seq=64 B=16 N=2",
         dict(hidden_dim=512, slot_dim=64, M=256, N=2, R=1,
              pointer_mode='sequential'), 64, 16),
        ("LongSeq: H=512 SD=32 M=256 seq=128 B=8 N=1",
         dict(hidden_dim=512, slot_dim=32, M=256, N=1, R=1,
              pointer_mode='sequential'), 128, 8),
        ("Prod-Small: H=2048 SD=128 M=1024 seq=128 B=4 N=1",
         dict(hidden_dim=2048, slot_dim=128, M=1024, N=1, R=1,
              pointer_mode='pilot'), 128, 4),
    ]

    rows = []
    for label, overrides, seq, bs in candidates:
        kw = {**BASELINE, **overrides}
        rows.append(bench_config(kw, seq, bs, label=label))
    print_table("CANDIDATE CONFIGS — Final Comparison", rows)


def main():
    print("INSTNCT CPU Parameter Sweep Benchmark")
    print(f"Threads: {torch.get_num_threads()}, Device: cpu")
    print(f"Warmup: {WARMUP_STEPS} steps, Bench: {BENCH_STEPS} steps\n")

    t0 = time.time()
    single = run_single_sweeps()
    t1 = time.time()
    print(f"\n[Single sweeps done in {t1-t0:.0f}s]")

    interaction = run_interaction_sweeps()
    t2 = time.time()
    print(f"\n[Interaction sweeps done in {t2-t1:.0f}s]")

    find_optimal(single)
    t3 = time.time()
    print(f"\n[Candidates done in {t3-t2:.0f}s]")
    print(f"\nTotal time: {t3-t0:.0f}s")


if __name__ == '__main__':
    main()

"""Multi-seed H (neuron count) sweep for SCT dimensionality validation.

Runs the two byte-pair prediction fixtures across a grid of H values and seeds,
parses the `SUMMARY {...}` JSON line each Rust run emits, and aggregates per-cell
mean +- std.

Fixtures:
  * mutual_inhibition  -> evolve_mutual_inhibition (Law I + II, flat 20k steps)
  * bytepair_proj      -> evolve_bytepair_proj     (Law I only, grow-prune cycle)

Both binaries were patched to accept `--seed <u64>` and `--H <usize>` and to
print a single machine-readable `SUMMARY {..}` JSON line at the end of each run.

Run (background-friendly):
    python3 tools/diag_dimensionality_sweep.py \\
        --seeds 10 --H-values 128,192,256,384 \\
        --out output/dimensionality_sweep/$(date +%s)

Resume a partial run (skips already-completed (fixture, H, seed) cells):
    python3 tools/diag_dimensionality_sweep.py --out <existing-dir> --resume
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CORPUS = REPO_ROOT / "instnct-core" / "tests" / "fixtures" / "alice_corpus.txt"
DEFAULT_PACKED = REPO_ROOT / "output" / "block_c_bytepair_champion" / "packed.bin"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fixtures", default="mutual_inhibition,bytepair_proj",
                   help="comma-separated list of example binaries to run")
    p.add_argument("--H-values", default="128,192,256,384",
                   help="comma-separated list of H (neuron count) values to sweep")
    p.add_argument("--seeds", type=int, default=10, help="number of seeds per (fixture, H) cell")
    p.add_argument("--steps", type=int, default=20000, help="mutation steps per run")
    p.add_argument("--corpus", default=str(DEFAULT_CORPUS), help="corpus text file")
    p.add_argument("--packed", default=str(DEFAULT_PACKED), help="VCBP packed embedding table")
    p.add_argument("--out", required=True, help="output directory (created if missing)")
    p.add_argument("--resume", action="store_true",
                   help="skip (fixture, H, seed) cells already present in results.json")
    p.add_argument("--dry-run", action="store_true", help="print cells but don't execute")
    return p.parse_args()


def seed_from_idx(idx: int) -> int:
    return 42 + idx * 1000


def parse_summary_line(stdout: str) -> dict | None:
    for line in stdout.splitlines():
        if line.startswith("SUMMARY "):
            try:
                return json.loads(line[len("SUMMARY "):])
            except json.JSONDecodeError as e:
                print(f"  !! failed to decode SUMMARY line: {e}", file=sys.stderr)
                print(f"     line: {line!r}", file=sys.stderr)
                return None
    return None


def run_cell(fixture: str, h: int, seed: int, steps: int, corpus: str, packed: str) -> tuple[dict | None, int, float]:
    """Invoke cargo for one (fixture, H, seed) cell. Returns (summary, exit_code, wall_clock_s)."""
    cmd = [
        "cargo", "run", "--release", "--example", f"evolve_{fixture}",
        "--manifest-path", str(REPO_ROOT / "instnct-core" / "Cargo.toml"),
        "--", corpus, packed,
        "--steps", str(steps),
        "--seed", str(seed),
        "--H", str(h),
    ]
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    wall = time.time() - t0
    if proc.returncode != 0:
        print(f"  !! cargo exited {proc.returncode}", file=sys.stderr)
        print(proc.stderr[-2000:], file=sys.stderr)
        return None, proc.returncode, wall
    summary = parse_summary_line(proc.stdout)
    return summary, 0, wall


def write_artifacts(out_dir: Path, results: list[dict]) -> None:
    (out_dir / "results.json").write_text(json.dumps({"results": results}, indent=2))
    csv_path = out_dir / "results.csv"
    if not results:
        return
    fieldnames = sorted({k for r in results for k in r.keys()})
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(results)


def mean_std(xs: list[float]) -> tuple[float, float]:
    if not xs:
        return float("nan"), float("nan")
    n = len(xs)
    m = sum(xs) / n
    if n == 1:
        return m, 0.0
    var = sum((x - m) ** 2 for x in xs) / (n - 1)
    return m, math.sqrt(var)


def print_aggregate(results: list[dict]) -> None:
    by_cell: dict[tuple[str, int], list[dict]] = {}
    for r in results:
        by_cell.setdefault((r["fixture"], int(r["H"])), []).append(r)
    fixtures = sorted({k[0] for k in by_cell.keys()})
    print("\n" + "=" * 80)
    print("AGGREGATE (mean +- std, n = # seeds)")
    print("=" * 80)
    for fx in fixtures:
        print(f"\nfixture={fx}:")
        hs = sorted({k[1] for k in by_cell.keys() if k[0] == fx})
        print(f"  {'H':>4} {'peak_acc%':>18} {'final_acc%':>18} {'accept%':>14} {'alive_frac':>14} {'edges':>10} {'wall/s':>9} {'n':>3}")
        for h in hs:
            rows = by_cell[(fx, h)]
            peak = [r["peak_acc"] * 100 for r in rows]
            final = [r["final_acc"] * 100 for r in rows]
            acc = [r["accept_rate_pct"] for r in rows]
            alive = [r["alive_frac_mean"] for r in rows]
            edges = [r["edges"] for r in rows]
            wall = [r["wall_clock_s"] for r in rows]
            pm, ps = mean_std(peak)
            fm, fs = mean_std(final)
            am, as_ = mean_std(acc)
            lm, ls = mean_std(alive)
            em, _ = mean_std(edges)
            wm, _ = mean_std(wall)
            print(f"  {h:>4} {pm:>8.2f} +- {ps:>5.2f}  {fm:>8.2f} +- {fs:>5.2f}  {am:>6.2f} +- {as_:>4.2f}  {lm:>5.3f} +- {ls:>5.3f}  {em:>10.0f}  {wm:>8.1f}  {len(rows):>3}")


def main() -> int:
    args = parse_args()
    fixtures = [f.strip() for f in args.fixtures.split(",") if f.strip()]
    h_values = [int(x) for x in args.H_values.split(",")]

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    done_cells: set[tuple[str, int, int]] = set()
    rj = out_dir / "results.json"
    if args.resume and rj.exists():
        results = json.loads(rj.read_text()).get("results", [])
        done_cells = {(r["fixture"], int(r["H"]), int(r["seed"])) for r in results}
        print(f"  resume: loaded {len(results)} previous results, skipping {len(done_cells)} cells")

    cells = [(fx, h, seed_from_idx(i)) for fx in fixtures for h in h_values for i in range(args.seeds)]
    todo = [c for c in cells if c not in done_cells]
    print(f"  plan: {len(cells)} total cells, {len(todo)} to run, {len(done_cells)} already done")
    print(f"  fixtures: {fixtures}")
    print(f"  H values: {h_values}")
    print(f"  seeds:    {args.seeds} per (fixture, H) cell  -> seed pattern 42 + i*1000")
    print(f"  steps:    {args.steps}")
    print(f"  out:      {out_dir}")

    if args.dry_run:
        for c in todo:
            print(f"  DRY-RUN  fixture={c[0]:22s} H={c[1]:>4} seed={c[2]}")
        return 0

    t_sweep = time.time()
    for idx, (fx, h, seed) in enumerate(todo, 1):
        elapsed = time.time() - t_sweep
        print(f"\n[{idx}/{len(todo)}] (elapsed {elapsed/60:.1f}m) fixture={fx} H={h} seed={seed}", flush=True)
        summary, rc, wall = run_cell(fx, h, seed, args.steps, args.corpus, args.packed)
        if summary is None:
            print(f"  skipped (rc={rc}, wall={wall:.1f}s)", file=sys.stderr)
            continue
        # Overwrite wall with driver-measured elapsed if Rust's value is missing
        summary.setdefault("wall_clock_s", wall)
        results.append(summary)
        write_artifacts(out_dir, results)
        print(f"  done: peak={summary['peak_acc']*100:.2f}%  final={summary['final_acc']*100:.2f}%  "
              f"accept={summary['accept_rate_pct']:.2f}%  alive={summary['alive_frac_mean']:.3f}  "
              f"edges={summary['edges']}  wall={summary['wall_clock_s']:.1f}s", flush=True)

    print_aggregate(results)
    print(f"\nSweep total wall clock: {(time.time() - t_sweep) / 60:.1f} min")
    print(f"Artifacts: {out_dir}/results.json  {out_dir}/results.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Multi-seed H sweep and Phase B microprobe driver.

Default mode preserves the original H-dimensionality sweep. `--phase-b` runs
the preregistered H=384 confound-vs-intrinsic arms for `evolve_mutual_inhibition`
and writes candidate logs, checkpoints, run metadata, and panel summaries.
`--phase-b1` runs the horizon x accept_ties follow-up on the same fixture.
`--phase-d1` runs the acceptance-aperture zero-p policy follow-up.
`--phase-d2` runs the cross-H validation for the D1 activation winner.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import json
import math
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
                   help="comma-separated list of fixtures to run")
    p.add_argument("--H-values", default="128,192,256,384",
                   help="comma-separated list of H values to sweep")
    p.add_argument("--seeds", type=int, default=10, help="number of seeds per cell")
    p.add_argument("--steps", type=int, default=20000, help="base mutation steps per run")
    p.add_argument("--corpus", default=str(DEFAULT_CORPUS), help="corpus text file")
    p.add_argument("--packed", default=str(DEFAULT_PACKED), help="VCBP packed embedding table")
    p.add_argument("--out", required=True, help="output directory")
    p.add_argument("--resume", action="store_true", help="skip cells already in results.json")
    p.add_argument("--dry-run", action="store_true", help="print cells but do not execute")
    p.add_argument("--phase-b", action="store_true",
                   help="run preregistered Phase B arms for evolve_mutual_inhibition")
    p.add_argument("--phase-b1", action="store_true",
                   help="run Phase B.1 horizon x accept_ties arms for evolve_mutual_inhibition")
    p.add_argument("--phase-d1", action="store_true",
                   help="run Phase D1 acceptance-aperture zero-p arms for evolve_mutual_inhibition")
    p.add_argument("--phase-d2", action="store_true",
                   help="run Phase D2 cross-H activation validation arms for evolve_mutual_inhibition")
    p.add_argument("--arms", default="B0,B1,B2,B3,B4",
                   help="comma-separated Phase B arms to run")
    p.add_argument("--panel-interval", type=int, default=None,
                   help="write Phase B panel_timeseries.csv every N steps")
    p.add_argument("--jobs", type=int, default=1,
                   help="parallel Phase B cells to run at once")
    return p.parse_args()


def seed_from_idx(idx: int) -> int:
    return 42 + idx * 1000


def parse_summary_line(stdout: str) -> dict | None:
    summaries = []
    for line in stdout.splitlines():
        if line.startswith("SUMMARY "):
            summaries.append(line[len("SUMMARY "):])
    if len(summaries) != 1:
        print(f"  !! expected exactly one SUMMARY line, got {len(summaries)}", file=sys.stderr)
        return None
    try:
        return json.loads(summaries[0])
    except json.JSONDecodeError as e:
        print(f"  !! failed to decode SUMMARY line: {e}", file=sys.stderr)
        print(f"     line: {summaries[0]!r}", file=sys.stderr)
        return None


def example_binary_path(example: str) -> Path:
    suffix = ".exe" if sys.platform.startswith("win") else ""
    return REPO_ROOT / "target" / "release" / "examples" / f"{example}{suffix}"


def build_release_example(example: str) -> int:
    cmd = [
        "cargo", "build", "--release", "--example", example,
        "--manifest-path", str(REPO_ROOT / "instnct-core" / "Cargo.toml"),
    ]
    proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    if proc.returncode != 0:
        print(f"  !! cargo build --example {example} exited {proc.returncode}", file=sys.stderr)
        print(proc.stderr[-4000:], file=sys.stderr)
    return proc.returncode


def prebuild_phase_examples(fixtures: list[str]) -> int:
    examples = [f"evolve_{fixture}" for fixture in fixtures] + ["diag_phase_b_panel"]
    for example in dict.fromkeys(examples):
        rc = build_release_example(example)
        if rc != 0:
            return rc
    return 0


def cargo_example_cmd(example: str, corpus: str, packed: str, *, prebuilt: bool = False) -> list[str]:
    if prebuilt:
        exe = example_binary_path(example)
        if not exe.exists():
            raise FileNotFoundError(f"prebuilt example missing: {exe}")
        return [str(exe), corpus, packed]
    return [
        "cargo", "run", "--release", "--example", example,
        "--manifest-path", str(REPO_ROOT / "instnct-core" / "Cargo.toml"),
        "--", corpus, packed,
    ]


def run_cell(fixture: str, h: int, seed: int, steps: int, corpus: str, packed: str) -> tuple[dict | None, int, float]:
    cmd = cargo_example_cmd(f"evolve_{fixture}", corpus, packed) + [
        "--steps", str(steps),
        "--seed", str(seed),
        "--H", str(h),
    ]
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    wall = time.time() - t0
    if proc.returncode != 0:
        print(f"  !! cargo exited {proc.returncode}", file=sys.stderr)
        print(proc.stderr[-2000:], file=sys.stderr)
        return None, proc.returncode, wall
    summary = parse_summary_line(proc.stdout)
    return summary, 0, wall


def phase_b_arm_config(arm: str, base_steps: int) -> dict:
    configs = {
        "B0": {"steps": base_steps, "jackpot": 9, "ticks": None, "input_scatter": False, "accept_ties": None},
        "B1": {"steps": base_steps * 2, "jackpot": 9, "ticks": None, "input_scatter": False, "accept_ties": None},
        "B2": {"steps": base_steps, "jackpot": 18, "ticks": None, "input_scatter": False, "accept_ties": None},
        "B3": {"steps": base_steps, "jackpot": 9, "ticks": 12, "input_scatter": False, "accept_ties": None},
        "B4": {"steps": base_steps, "jackpot": 9, "ticks": None, "input_scatter": True, "accept_ties": None},
    }
    if arm not in configs:
        raise ValueError(f"unknown Phase B arm: {arm}")
    return configs[arm]


def phase_b1_arm_config(arm: str, base_steps: int) -> dict:
    configs = {}
    for label, multiplier in [("S20", 1), ("S40", 2), ("S80", 4)]:
        configs[f"B1_{label}_STRICT"] = {
            "steps": base_steps * multiplier,
            "jackpot": 9,
            "ticks": None,
            "input_scatter": False,
            "accept_ties": False,
        }
        configs[f"B1_{label}_TIES"] = {
            "steps": base_steps * multiplier,
            "jackpot": 9,
            "ticks": None,
            "input_scatter": False,
            "accept_ties": True,
        }
    if arm not in configs:
        raise ValueError(f"unknown Phase B.1 arm: {arm}")
    return configs[arm]


def phase_d1_arm_config(arm: str, base_steps: int) -> dict:
    def cfg(jackpot: int, accept_policy: str, neutral_p: float | None = None) -> dict:
        return {
            "steps": base_steps,
            "jackpot": jackpot,
            "ticks": None,
            "input_scatter": False,
            "accept_ties": False,
            "accept_policy": accept_policy,
            "neutral_p": neutral_p,
            "accept_epsilon": None,
        }

    configs = {}
    for jackpot in [1, 3, 9]:
        configs[f"D1_K{jackpot}_STRICT"] = cfg(jackpot, "strict")
        configs[f"D1_K{jackpot}_ZERO_P03"] = cfg(jackpot, "zero-p", 0.3)
        configs[f"D1_K{jackpot}_ZERO_P10"] = cfg(jackpot, "zero-p", 1.0)

    # Backward-compatible aliases for older Phase D1 probes. New default D1
    # runs use the explicit K-qualified arms above.
    configs.update({
        "D1_STRICT": configs["D1_K9_STRICT"],
        "D1_ZERO_P01": cfg(9, "zero-p", 0.1),
        "D1_ZERO_P03": configs["D1_K9_ZERO_P03"],
        "D1_ZERO_P06": cfg(9, "zero-p", 0.6),
        "D1_ZERO_P10": configs["D1_K9_ZERO_P10"],
        "D1_TIES_LEGACY": {
            "steps": base_steps,
            "jackpot": 9,
            "ticks": None,
            "input_scatter": False,
            "accept_ties": True,
            "accept_policy": None,
            "neutral_p": None,
            "accept_epsilon": None,
        },
    })
    if arm not in configs:
        raise ValueError(f"unknown Phase D1 arm: {arm}")
    return configs[arm]


def phase_d2_arm_config(arm: str, base_steps: int) -> dict:
    configs = {
        "D2_K1_STRICT": phase_d1_arm_config("D1_K1_STRICT", base_steps),
        "D2_K1_ZERO_P10": phase_d1_arm_config("D1_K1_ZERO_P10", base_steps),
        "D2_K3_STRICT": phase_d1_arm_config("D1_K3_STRICT", base_steps),
        "D2_K3_ZERO_P10": phase_d1_arm_config("D1_K3_ZERO_P10", base_steps),
        "D2_K9_STRICT": phase_d1_arm_config("D1_K9_STRICT", base_steps),
        "D2_K9_ZERO_P10": phase_d1_arm_config("D1_K9_ZERO_P10", base_steps),
    }
    if arm not in configs:
        raise ValueError(f"unknown Phase D2 arm: {arm}")
    return configs[arm]


def run_panel_analyzer(run_dir: Path) -> int:
    exe = example_binary_path("diag_phase_b_panel")
    if exe.exists():
        cmd = [str(exe), str(run_dir)]
    else:
        cmd = [
            "cargo", "run", "--release", "--example", "diag_phase_b_panel",
            "--manifest-path", str(REPO_ROOT / "instnct-core" / "Cargo.toml"),
            "--", str(run_dir),
        ]
    proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    (run_dir / "panel_stdout.txt").write_text(proc.stdout)
    (run_dir / "panel_stderr.txt").write_text(proc.stderr)
    if proc.returncode != 0:
        print(f"  !! panel analyzer exited {proc.returncode}", file=sys.stderr)
        print(proc.stderr[-2000:], file=sys.stderr)
    return proc.returncode


def run_phase_b_cell(
    fixture: str,
    phase: str,
    arm: str,
    h: int,
    seed: int,
    base_steps: int,
    corpus: str,
    packed: str,
    out_dir: Path,
    panel_interval: int | None,
    cfg: dict | None = None,
) -> tuple[dict | None, int, float]:
    cfg = cfg or phase_b_arm_config(arm, base_steps)
    run_id = f"phase_{phase.lower()}_{fixture}_{arm}_H{h}_seed{seed}"
    if phase == "D2":
        run_dir = out_dir / f"H_{h}" / arm / f"seed_{seed}"
    else:
        run_dir = out_dir / arm / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = run_dir / "stdout.txt"
    stderr_path = run_dir / "stderr.txt"
    candidate_log = run_dir / "candidates.csv"
    checkpoint = run_dir / "final.ckpt"
    panel_timeseries = run_dir / "panel_timeseries.csv"

    cmd = cargo_example_cmd(f"evolve_{fixture}", corpus, packed, prebuilt=True) + [
        "--steps", str(cfg["steps"]),
        "--seed", str(seed),
        "--H", str(h),
        "--jackpot", str(cfg["jackpot"]),
        "--phase", phase,
        "--arm", arm,
        "--run-id", run_id,
        "--candidate-log", str(candidate_log),
        "--checkpoint-at-end", str(checkpoint),
    ]
    if panel_interval is not None:
        cmd += [
            "--panel-interval", str(panel_interval),
            "--panel-log", str(panel_timeseries),
        ]
    if cfg["ticks"] is not None:
        cmd += ["--ticks", str(cfg["ticks"])]
    if cfg["accept_ties"] is not None:
        cmd += ["--accept-ties", "true" if cfg["accept_ties"] else "false"]
    if cfg.get("accept_policy") is not None:
        cmd += ["--accept-policy", str(cfg["accept_policy"])]
    if cfg.get("neutral_p") is not None:
        cmd += ["--neutral-p", str(cfg["neutral_p"])]
    if cfg.get("accept_epsilon") is not None:
        cmd += ["--accept-epsilon", str(cfg["accept_epsilon"])]
    if cfg["input_scatter"]:
        cmd += ["--input-scatter"]

    (run_dir / "run_cmd.json").write_text(json.dumps({"cmd": cmd}, indent=2))
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    wall = time.time() - t0
    stdout_path.write_text(proc.stdout)
    stderr_path.write_text(proc.stderr)
    if proc.returncode != 0:
        print(f"  !! cargo exited {proc.returncode}", file=sys.stderr)
        print(proc.stderr[-2000:], file=sys.stderr)
        return None, proc.returncode, wall
    summary = parse_summary_line(proc.stdout)
    if summary is None:
        return None, 2, wall
    if not checkpoint.exists() or not (run_dir / "run_meta.json").exists():
        print("  !! missing checkpoint or run_meta.json", file=sys.stderr)
        return None, 3, wall
    panel_rc = run_panel_analyzer(run_dir)
    if panel_rc != 0 or not (run_dir / "panel_summary.json").exists():
        return None, panel_rc or 4, wall
    if panel_interval is not None and not panel_timeseries.exists():
        print("  !! missing panel_timeseries.csv", file=sys.stderr)
        return None, 5, wall

    summary.update({
        "phase": phase,
        "arm": arm,
        "run_id": run_id,
        "configured_steps": cfg["steps"],
        "horizon_steps": cfg["steps"],
        "jackpot": cfg["jackpot"],
        "ticks": cfg["ticks"] or 6,
        "accept_ties": cfg["accept_ties"] if cfg["accept_ties"] is not None else summary.get("accept_ties", ""),
        "accept_policy": cfg.get("accept_policy") or summary.get("accept_policy", ""),
        "neutral_p": cfg.get("neutral_p") if cfg.get("neutral_p") is not None else summary.get("neutral_p", ""),
        "accept_epsilon": cfg.get("accept_epsilon") if cfg.get("accept_epsilon") is not None else summary.get("accept_epsilon", ""),
        "input_scatter": cfg["input_scatter"],
        "run_dir": str(run_dir),
        "candidate_log": str(candidate_log),
        "checkpoint": str(checkpoint),
        "panel_summary": str(run_dir / "panel_summary.json"),
        "panel_timeseries": str(panel_timeseries) if panel_interval is not None else "",
        "panel_window_size": panel_interval or "",
        "expected_candidate_rows": cfg["steps"] * cfg["jackpot"],
    })
    return summary, 0, wall


def write_artifacts(out_dir: Path, results: list[dict]) -> None:
    results = sorted(
        results,
        key=lambda r: (
            r.get("fixture", ""),
            r.get("arm", ""),
            int(r.get("H", 0)),
            int(r.get("seed", 0)),
        ),
    )
    (out_dir / "results.json").write_text(json.dumps({"results": results}, indent=2))
    if not results:
        return
    csv_path = out_dir / "results.csv"
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
    by_cell: dict[tuple[str, int, str], list[dict]] = {}
    for r in results:
        by_cell.setdefault((r["fixture"], int(r["H"]), r.get("arm", "")), []).append(r)
    print("\n" + "=" * 80)
    print("AGGREGATE (mean +- std, n = # seeds)")
    print("=" * 80)
    for fixture, h, arm in sorted(by_cell):
        rows = by_cell[(fixture, h, arm)]
        peak = [r["peak_acc"] * 100 for r in rows]
        final = [r["final_acc"] * 100 for r in rows]
        acc = [r["accept_rate_pct"] for r in rows]
        alive = [r["alive_frac_mean"] for r in rows]
        wall = [r["wall_clock_s"] for r in rows]
        pm, ps = mean_std(peak)
        fm, fs = mean_std(final)
        am, as_ = mean_std(acc)
        lm, ls = mean_std(alive)
        wm, _ = mean_std(wall)
        label = f"{fixture} H={h}" + (f" arm={arm}" if arm else "")
        print(f"  {label:<36} peak={pm:>6.2f} +- {ps:>5.2f}  final={fm:>6.2f} +- {fs:>5.2f}  "
              f"accept={am:>6.2f} +- {as_:>4.2f}  alive={lm:>5.3f} +- {ls:>5.3f}  "
              f"wall={wm:>7.1f}s  n={len(rows)}")


def run_constructability_analysis(out_dir: Path) -> int:
    script = REPO_ROOT / "tools" / "diag_constructability_analysis.py"
    cmd = [sys.executable, str(script), "--root", str(out_dir)]
    proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    (out_dir / "constructability_stdout.txt").write_text(proc.stdout)
    (out_dir / "constructability_stderr.txt").write_text(proc.stderr)
    if proc.returncode != 0:
        print(f"  !! constructability analysis exited {proc.returncode}", file=sys.stderr)
        print(proc.stderr[-2000:], file=sys.stderr)
    else:
        print(proc.stdout)
    return proc.returncode


def load_resume(out_dir: Path, phase_b: bool) -> tuple[list[dict], set[tuple]]:
    rj = out_dir / "results.json"
    if not rj.exists():
        return [], set()
    results = json.loads(rj.read_text()).get("results", [])
    if phase_b:
        done = {(r["fixture"], r["arm"], int(r["H"]), int(r["seed"])) for r in results}
    else:
        done = {(r["fixture"], int(r["H"]), int(r["seed"])) for r in results}
    return results, done


def main_phase_b_like(args: argparse.Namespace, phase: str, config_fn, default_arms: list[str]) -> int:
    fixtures = [f.strip() for f in args.fixtures.split(",") if f.strip()]
    if fixtures != ["mutual_inhibition"]:
        raise SystemExit(f"--phase-{phase.lower()} currently supports only --fixtures mutual_inhibition")
    h_values = [int(x) for x in args.H_values.split(",")]
    if phase in {"B1", "D1", "D2"} and args.arms == "B0,B1,B2,B3,B4":
        arms = default_arms
    else:
        arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    done: set[tuple] = set()
    if args.resume:
        results, done = load_resume(out_dir, phase_b=True)
        print(f"  resume: loaded {len(results)} previous results, skipping {len(done)} cells")

    cells = [
        (fx, arm, h, seed_from_idx(i))
        for fx in fixtures
        for arm in arms
        for h in h_values
        for i in range(args.seeds)
    ]
    todo = [cell for cell in cells if cell not in done]
    print(f"  Phase {phase} plan: {len(cells)} total cells, {len(todo)} to run")
    print(f"  fixtures: {fixtures}")
    print(f"  arms:     {arms}")
    print(f"  H values: {h_values}")
    print(f"  seeds:    {args.seeds} per arm -> seed pattern 42 + i*1000")
    print(f"  base steps: {args.steps}")
    if args.panel_interval is not None:
        print(f"  panel interval: {args.panel_interval}")
    print(f"  jobs:     {args.jobs}")
    print(f"  out:      {out_dir}")

    if args.dry_run:
        for fx, arm, h, seed in todo:
            cfg = config_fn(arm, args.steps)
            print(f"  DRY-RUN fixture={fx} arm={arm} H={h} seed={seed} "
                  f"steps={cfg['steps']} jackpot={cfg['jackpot']} ticks={cfg['ticks'] or 6} "
                  f"accept_ties={cfg['accept_ties']} accept_policy={cfg.get('accept_policy')} "
                  f"neutral_p={cfg.get('neutral_p')} input_scatter={cfg['input_scatter']} "
                  f"panel_interval={args.panel_interval}")
        return 0

    prebuild_rc = prebuild_phase_examples(fixtures)
    if prebuild_rc != 0:
        return prebuild_rc

    t_sweep = time.time()
    jobs = max(1, args.jobs)
    if jobs == 1:
        for idx, (fx, arm, h, seed) in enumerate(todo, 1):
            elapsed = time.time() - t_sweep
            print(f"\n[{idx}/{len(todo)}] elapsed={elapsed/60:.1f}m fixture={fx} arm={arm} H={h} seed={seed}", flush=True)
            summary, rc, wall = run_phase_b_cell(
                fx,
                phase,
                arm,
                h,
                seed,
                args.steps,
                args.corpus,
                args.packed,
                out_dir,
                args.panel_interval,
                config_fn(arm, args.steps),
            )
            if summary is None:
                print(f"  FAILED (rc={rc}, wall={wall:.1f}s)", file=sys.stderr)
                write_artifacts(out_dir, results)
                return rc or 1
            summary.setdefault("wall_clock_s", wall)
            results.append(summary)
            write_artifacts(out_dir, results)
            print(f"  done: peak={summary['peak_acc']*100:.2f}% final={summary['final_acc']*100:.2f}% "
                  f"accept={summary['accept_rate_pct']:.2f}% rows={summary['expected_candidate_rows']} "
                  f"wall={summary['wall_clock_s']:.1f}s", flush=True)
    else:
        print(f"\nRunning Phase {phase} with {jobs} parallel jobs", flush=True)
        with ThreadPoolExecutor(max_workers=jobs) as executor:
            futures = {}
            for idx, (fx, arm, h, seed) in enumerate(todo, 1):
                print(f"  queue [{idx}/{len(todo)}] fixture={fx} arm={arm} H={h} seed={seed}", flush=True)
                future = executor.submit(
                    run_phase_b_cell,
                    fx,
                    phase,
                    arm,
                    h,
                    seed,
                    args.steps,
                    args.corpus,
                    args.packed,
                    out_dir,
                    args.panel_interval,
                    config_fn(arm, args.steps),
                )
                futures[future] = (idx, fx, arm, h, seed)

            first_failure = 0
            for future in as_completed(futures):
                idx, fx, arm, h, seed = futures[future]
                elapsed = time.time() - t_sweep
                try:
                    summary, rc, wall = future.result()
                except Exception as exc:
                    summary, rc, wall = None, 1, 0.0
                    print(
                        f"  FAILED [{idx}/{len(todo)}] elapsed={elapsed/60:.1f}m "
                        f"fixture={fx} arm={arm} H={h} seed={seed}: {exc}",
                        file=sys.stderr,
                        flush=True,
                    )
                if summary is None:
                    first_failure = first_failure or (rc or 1)
                    print(
                        f"  FAILED [{idx}/{len(todo)}] elapsed={elapsed/60:.1f}m "
                        f"fixture={fx} arm={arm} H={h} seed={seed} rc={rc} wall={wall:.1f}s",
                        file=sys.stderr,
                        flush=True,
                    )
                    write_artifacts(out_dir, results)
                    continue

                summary.setdefault("wall_clock_s", wall)
                results.append(summary)
                write_artifacts(out_dir, results)
                print(
                    f"  done [{idx}/{len(todo)}] elapsed={elapsed/60:.1f}m "
                    f"fixture={fx} arm={arm} H={h} seed={seed}: "
                    f"peak={summary['peak_acc']*100:.2f}% final={summary['final_acc']*100:.2f}% "
                    f"accept={summary['accept_rate_pct']:.2f}% rows={summary['expected_candidate_rows']} "
                    f"wall={summary['wall_clock_s']:.1f}s",
                    flush=True,
                )
            if first_failure:
                return first_failure

    print_aggregate(results)
    analysis_rc = run_constructability_analysis(out_dir)
    print(f"\nSweep total wall clock: {(time.time() - t_sweep) / 60:.1f} min")
    print(f"Artifacts: {out_dir / 'results.json'}  {out_dir / 'results.csv'}")
    return analysis_rc


def main_phase_b(args: argparse.Namespace) -> int:
    return main_phase_b_like(args, "B", phase_b_arm_config, ["B0", "B1", "B2", "B3", "B4"])


def main_phase_b1(args: argparse.Namespace) -> int:
    return main_phase_b_like(
        args,
        "B1",
        phase_b1_arm_config,
        [
            "B1_S20_STRICT",
            "B1_S20_TIES",
            "B1_S40_STRICT",
            "B1_S40_TIES",
            "B1_S80_STRICT",
            "B1_S80_TIES",
        ],
    )


def main_phase_d1(args: argparse.Namespace) -> int:
    return main_phase_b_like(
        args,
        "D1",
        phase_d1_arm_config,
        [
            "D1_K1_STRICT",
            "D1_K1_ZERO_P03",
            "D1_K1_ZERO_P10",
            "D1_K3_STRICT",
            "D1_K3_ZERO_P03",
            "D1_K3_ZERO_P10",
            "D1_K9_STRICT",
            "D1_K9_ZERO_P03",
            "D1_K9_ZERO_P10",
        ],
    )


def main_phase_d2(args: argparse.Namespace) -> int:
    return main_phase_b_like(
        args,
        "D2",
        phase_d2_arm_config,
        [
            "D2_K1_STRICT",
            "D2_K1_ZERO_P10",
            "D2_K3_STRICT",
            "D2_K3_ZERO_P10",
            "D2_K9_STRICT",
            "D2_K9_ZERO_P10",
        ],
    )


def main_default(args: argparse.Namespace) -> int:
    fixtures = [f.strip() for f in args.fixtures.split(",") if f.strip()]
    h_values = [int(x) for x in args.H_values.split(",")]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    done: set[tuple] = set()
    if args.resume:
        results, done = load_resume(out_dir, phase_b=False)
        print(f"  resume: loaded {len(results)} previous results, skipping {len(done)} cells")

    cells = [(fx, h, seed_from_idx(i)) for fx in fixtures for h in h_values for i in range(args.seeds)]
    todo = [c for c in cells if c not in done]
    print(f"  plan: {len(cells)} total cells, {len(todo)} to run, {len(done)} already done")
    print(f"  fixtures: {fixtures}")
    print(f"  H values: {h_values}")
    print(f"  seeds:    {args.seeds} per cell -> seed pattern 42 + i*1000")
    print(f"  steps:    {args.steps}")
    print(f"  out:      {out_dir}")

    if args.dry_run:
        for c in todo:
            print(f"  DRY-RUN fixture={c[0]:22s} H={c[1]:>4} seed={c[2]}")
        return 0

    t_sweep = time.time()
    for idx, (fx, h, seed) in enumerate(todo, 1):
        elapsed = time.time() - t_sweep
        print(f"\n[{idx}/{len(todo)}] elapsed={elapsed/60:.1f}m fixture={fx} H={h} seed={seed}", flush=True)
        summary, rc, wall = run_cell(fx, h, seed, args.steps, args.corpus, args.packed)
        if summary is None:
            print(f"  skipped (rc={rc}, wall={wall:.1f}s)", file=sys.stderr)
            continue
        summary.setdefault("wall_clock_s", wall)
        results.append(summary)
        write_artifacts(out_dir, results)
        print(f"  done: peak={summary['peak_acc']*100:.2f}% final={summary['final_acc']*100:.2f}% "
              f"accept={summary['accept_rate_pct']:.2f}% alive={summary['alive_frac_mean']:.3f} "
              f"edges={summary['edges']} wall={summary['wall_clock_s']:.1f}s", flush=True)

    print_aggregate(results)
    print(f"\nSweep total wall clock: {(time.time() - t_sweep) / 60:.1f} min")
    print(f"Artifacts: {out_dir / 'results.json'}  {out_dir / 'results.csv'}")
    return 0


def main() -> int:
    args = parse_args()
    if sum([args.phase_b, args.phase_b1, args.phase_d1, args.phase_d2]) > 1:
        raise SystemExit("--phase-b, --phase-b1, --phase-d1, and --phase-d2 are mutually exclusive")
    if args.phase_b:
        return main_phase_b(args)
    if args.phase_b1:
        return main_phase_b1(args)
    if args.phase_d1:
        return main_phase_d1(args)
    if args.phase_d2:
        return main_phase_d2(args)
    return main_default(args)


if __name__ == "__main__":
    raise SystemExit(main())

"""Analyze Phase B per-candidate logs.

Validates candidate CSV invariants and emits run-level constructability metrics:
V_raw, V_sel, M_pos, R_neg, cost_eval_ms, C_K_window_ratio, plus per-operator
breakdowns.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path


EPS_DEFAULT = 1e-4
ACCEPT_TOL_DEFAULT = 1e-9


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True, help="Phase B output root")
    p.add_argument("--eps", type=float, default=EPS_DEFAULT, help="positive-delta threshold")
    p.add_argument(
        "--accept-tol",
        type=float,
        default=ACCEPT_TOL_DEFAULT,
        help="allowed negative tolerance for accepted candidates when accept_ties is enabled",
    )
    return p.parse_args()


def as_bool(value: str) -> bool:
    if value == "true":
        return True
    if value == "false":
        return False
    raise ValueError(f"invalid bool: {value!r}")


def load_meta(csv_path: Path) -> dict | None:
    meta_path = csv_path.parent / "run_meta.json"
    if not meta_path.exists():
        return None
    return json.loads(meta_path.read_text())


def positive_mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def analyze_candidate_log(csv_path: Path, eps: float, accept_tol: float) -> tuple[dict, list[dict]]:
    rows = list(csv.DictReader(csv_path.open(newline="")))
    if not rows:
        raise ValueError(f"{csv_path}: no candidate rows")

    meta = load_meta(csv_path)
    if meta is not None:
        expected = int(meta["steps"]) * int(meta["jackpot"])
        if len(rows) != expected:
            raise ValueError(f"{csv_path}: rows={len(rows)} expected={expected}")

    by_step: dict[int, list[dict]] = defaultdict(list)
    op_rows: dict[str, list[dict]] = defaultdict(list)
    positives: list[float] = []
    negatives: list[float] = []
    eval_costs: list[float] = []
    step_costs: dict[int, float] = {}
    positive_row_count = 0
    evaluated_count = 0

    for idx, row in enumerate(rows, 1):
        step = int(row["step"])
        by_step[step].append(row)
        op_rows[row["operator_id"]].append(row)

        before = float(row["before_U"])
        after = float(row["after_U"])
        delta = float(row["delta_U"])
        evaluated = as_bool(row["evaluated"])
        accepted = as_bool(row["accepted"])
        selected = as_bool(row["selected"])

        if evaluated:
            evaluated_count += 1
            eval_costs.append(float(row["candidate_eval_ms"]))
            if abs(delta - (after - before)) > 1e-12:
                raise ValueError(f"{csv_path}: delta mismatch row={idx}")
        if accepted and not selected:
            raise ValueError(f"{csv_path}: accepted but not selected row={idx}")
        if delta > eps:
            positives.append(delta)
            positive_row_count += 1
        elif delta < -eps:
            negatives.append(-delta)

    if evaluated_count == 0:
        raise ValueError(f"{csv_path}: no evaluated rows")

    useful_sum = 0.0
    selected_positive_steps = 0
    accepted_steps = 0
    accepted_positive_steps = 0
    accepted_nonpositive_steps = 0
    for step, step_rows in by_step.items():
        accepted_rows = [r for r in step_rows if as_bool(r["accepted"])]
        if len(accepted_rows) > 1:
            raise ValueError(f"{csv_path}: step={step} has multiple accepted candidates")
        if accepted_rows:
            accepted = accepted_rows[0]
            accepted_delta = float(accepted["delta_U"])
            if not as_bool(accepted["selected"]):
                raise ValueError(f"{csv_path}: step={step} accepted candidate was not selected")
            if not as_bool(accepted["within_cap"]):
                raise ValueError(f"{csv_path}: step={step} accepted candidate was outside edge cap")
            if accepted_delta < -accept_tol:
                raise ValueError(
                    f"{csv_path}: step={step} accepted negative delta={accepted_delta}"
                )
            if accepted_delta > eps:
                accepted_positive_steps += 1
            else:
                accepted_nonpositive_steps += 1

        eligible = [float(r["delta_U"]) for r in step_rows if as_bool(r["within_cap"])]
        best_delta = max(eligible) if eligible else 0.0
        useful_sum += max(0.0, best_delta - eps)
        if best_delta > eps:
            selected_positive_steps += 1
        if any(as_bool(r["accepted"]) for r in step_rows):
            accepted_steps += 1
        step_costs[step] = max(float(r["step_wall_ms"]) for r in step_rows)

    step_cost_sum = sum(step_costs.values())
    ck_ratio = useful_sum / step_cost_sum if step_cost_sum > 0.0 else 0.0

    run = {
        "csv_path": str(csv_path),
        "run_dir": str(csv_path.parent),
        "run_id": rows[0]["run_id"],
        "arm": rows[0]["arm"],
        "seed": int(rows[0]["seed"]),
        "H": int(rows[0]["H"]),
        "candidate_rows": len(rows),
        "steps": len(by_step),
        "evaluated_rows": evaluated_count,
        "accepted_steps": accepted_steps,
        "accepted_positive_steps": accepted_positive_steps,
        "accepted_nonpositive_steps": accepted_nonpositive_steps,
        "positive_delta_rows": positive_row_count,
        "V_raw": positive_row_count / len(rows),
        "V_sel": selected_positive_steps / len(by_step),
        "M_pos": positive_mean(positives),
        "R_neg": positive_mean(negatives),
        "cost_eval_ms": positive_mean(eval_costs),
        "step_cost_ms_sum": step_cost_sum,
        "useful_delta_sum": useful_sum,
        "C_K_window_ratio": ck_ratio,
    }
    if meta is not None:
        run.update({
            "phase": meta.get("phase", ""),
            "configured_steps": int(meta["steps"]),
            "horizon_steps": int(meta.get("horizon_steps", meta["steps"])),
            "jackpot": int(meta["jackpot"]),
            "ticks": int(meta["ticks"]),
            "accept_ties": bool(meta.get("accept_ties", False)),
            "accept_policy": meta.get("accept_policy", ""),
            "neutral_p": meta.get("neutral_p", ""),
            "accept_epsilon": meta.get("accept_epsilon", ""),
            "input_scatter": bool(meta["input_scatter"]),
        })

    op_summaries = []
    for op, op_data in sorted(op_rows.items()):
        deltas = [float(r["delta_U"]) for r in op_data]
        pos = [d for d in deltas if d > eps]
        neg = [-d for d in deltas if d < -eps]
        evaluated = [r for r in op_data if as_bool(r["evaluated"])]
        op_summaries.append({
            "run_id": run["run_id"],
            "arm": run["arm"],
            "seed": run["seed"],
            "H": run["H"],
            "phase": run.get("phase", ""),
            "horizon_steps": run.get("horizon_steps", run.get("steps", "")),
            "jackpot": run.get("jackpot", ""),
            "accept_ties": run.get("accept_ties", ""),
            "accept_policy": run.get("accept_policy", ""),
            "neutral_p": run.get("neutral_p", ""),
            "accept_epsilon": run.get("accept_epsilon", ""),
            "operator_id": op,
            "candidate_rows": len(op_data),
            "evaluated_rows": len(evaluated),
            "V_raw": len(pos) / len(op_data),
            "M_pos": positive_mean(pos),
            "R_neg": positive_mean(neg),
        })

    return run, op_summaries


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def aggregate_runs(runs: list[dict]) -> list[dict]:
    by_arm: dict[str, list[dict]] = defaultdict(list)
    for run in runs:
        by_arm[run["arm"]].append(run)
    out = []
    for arm, arm_runs in sorted(by_arm.items()):
        out.append({
            "arm": arm,
            "n": len(arm_runs),
            "mean_V_raw": sum(r["V_raw"] for r in arm_runs) / len(arm_runs),
            "mean_V_sel": sum(r["V_sel"] for r in arm_runs) / len(arm_runs),
            "mean_M_pos": sum(r["M_pos"] for r in arm_runs) / len(arm_runs),
            "mean_R_neg": sum(r["R_neg"] for r in arm_runs) / len(arm_runs),
            "mean_cost_eval_ms": sum(r["cost_eval_ms"] for r in arm_runs) / len(arm_runs),
            "mean_C_K_window_ratio": sum(r["C_K_window_ratio"] for r in arm_runs) / len(arm_runs),
            "mean_accepted_nonpositive_steps": sum(r["accepted_nonpositive_steps"] for r in arm_runs) / len(arm_runs),
        })
    return out


def main() -> int:
    args = parse_args()
    root = Path(args.root)
    csv_paths = sorted(root.glob("**/candidates.csv"))
    if not csv_paths:
        raise SystemExit(f"no candidates.csv files found under {root}")

    runs: list[dict] = []
    operators: list[dict] = []
    for csv_path in csv_paths:
        run, op = analyze_candidate_log(csv_path, args.eps, args.accept_tol)
        runs.append(run)
        operators.extend(op)

    aggregate = aggregate_runs(runs)
    payload = {
        "root": str(root),
        "eps": args.eps,
        "accept_tol": args.accept_tol,
        "runs": runs,
        "aggregate_by_arm": aggregate,
        "operator_breakdown": operators,
    }
    (root / "constructability_summary.json").write_text(json.dumps(payload, indent=2))
    write_csv(root / "constructability_summary.csv", runs)
    write_csv(root / "constructability_operator_summary.csv", operators)

    print(json.dumps({
        "status": "PASS",
        "runs": len(runs),
        "candidate_rows": sum(r["candidate_rows"] for r in runs),
        "arms": sorted({r["arm"] for r in runs}),
        "constructability_summary": str(root / "constructability_summary.json"),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


def _slope(points: List[Dict[str, Any]], key: str) -> float:
    xs: List[float] = []
    ys: List[float] = []
    for point in points:
        if key not in point:
            continue
        try:
            xs.append(float(point["step"]))
            ys.append(float(point[key]))
        except Exception:
            continue
    n = len(xs)
    if n < 2:
        return 0.0
    mean_x = sum(xs) / float(n)
    mean_y = sum(ys) / float(n)
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    den = sum((x - mean_x) ** 2 for x in xs)
    if den == 0.0:
        return 0.0
    return float(num / den)


def _arm_meta() -> Dict[str, Dict[str, str]]:
    return {
        "arm_a_route_equal_capacity_equal": {"router_mode": "equal", "capacity_mode": "equal"},
        "arm_b_route_fibo_capacity_equal": {"router_mode": "fibo", "capacity_mode": "equal"},
        "arm_c_route_equal_capacity_fibo": {"router_mode": "equal", "capacity_mode": "fibo"},
        "arm_d_route_fibo_capacity_fibo": {"router_mode": "fibo", "capacity_mode": "fibo"},
    }


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Consolidate matrix summary with final checkpoint eval points.")
    ap.add_argument("--run-root", required=True)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--val-range", type=int, default=256)
    return ap.parse_args()


def main() -> int:
    args = _parse_args()
    run_root = Path(args.run_root).resolve()
    summary_path = run_root / "matrix_summary.json"
    csv_path = run_root / "matrix_summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"missing summary: {summary_path}")

    old = json.loads(summary_path.read_text(encoding="utf-8"))
    old_arms = old.get("arms", []) if isinstance(old.get("arms"), list) else []
    old_map: Dict[str, Dict[str, Any]] = {}
    for arm in old_arms:
        if isinstance(arm, dict) and "arm" in arm:
            old_map[str(arm["arm"])] = arm

    out_arms: List[Dict[str, Any]] = []
    meta = _arm_meta()
    for arm_name, arm_meta in meta.items():
        arm_root = run_root / arm_name
        train_root = arm_root / "train"
        report_path = train_root / "report.json"
        checkpoint_path = train_root / "checkpoint.pt"
        if not report_path.exists():
            continue
        report = json.loads(report_path.read_text(encoding="utf-8"))
        eval_block = report.get("eval", {}) if isinstance(report.get("eval"), dict) else {}
        eval_acc = float(eval_block.get("eval_acc", 0.0))
        eval_loss = float(eval_block.get("eval_loss", 0.0))
        eval_n = int(eval_block.get("eval_n", 0))
        acc_delta = float(eval_acc - (1.0 / float(int(args.val_range))))

        old_arm = old_map.get(arm_name, {})
        old_points = old_arm.get("points", []) if isinstance(old_arm.get("points"), list) else []
        points: List[Dict[str, Any]] = [p for p in old_points if isinstance(p, dict)]
        steps_present = {int(p.get("step", -1)) for p in points}
        if int(args.steps) not in steps_present and checkpoint_path.exists():
            proxy_eval_seconds = 0.0
            if points:
                vals = [float(p.get("eval_seconds", 0.0)) for p in points if float(p.get("eval_seconds", 0.0)) > 0.0]
                if vals:
                    proxy_eval_seconds = float(sum(vals) / float(len(vals)))
            points.append(
                {
                    "step": int(args.steps),
                    "checkpoint": str(checkpoint_path),
                    "eval_acc": float(eval_acc),
                    "acc_delta": float(acc_delta),
                    "eval_loss": float(eval_loss),
                    "eval_n": int(eval_n),
                    "eval_seconds": float(proxy_eval_seconds),
                }
            )

        points = sorted(points, key=lambda x: int(x.get("step", 0)))
        final = points[-1] if points else {
            "step": int(args.steps),
            "checkpoint": str(checkpoint_path),
            "eval_acc": float(eval_acc),
            "acc_delta": float(acc_delta),
            "eval_loss": float(eval_loss),
            "eval_n": int(eval_n),
            "eval_seconds": 0.0,
        }
        avg_eval_seconds = 0.0
        vals = [float(p.get("eval_seconds", 0.0)) for p in points if float(p.get("eval_seconds", 0.0)) > 0.0]
        if vals:
            avg_eval_seconds = float(sum(vals) / float(len(vals)))

        out_arms.append(
            {
                "arm": arm_name,
                "run_root": str(arm_root),
                "router_mode": arm_meta["router_mode"],
                "capacity_mode": arm_meta["capacity_mode"],
                "points": points,
                "final": final,
                "avg_eval_seconds": float(avg_eval_seconds),
                "slope_acc_delta_per_step": float(_slope(points, "acc_delta")),
                "rewire": old_arm.get("rewire", {}),
            }
        )

    baseline = 0.0
    for arm in out_arms:
        if arm["arm"] == "arm_a_route_equal_capacity_equal":
            baseline = float(arm["final"].get("eval_acc", 0.0))
            break
    for arm in out_arms:
        arm["delta_vs_baseline_final_acc"] = float(float(arm["final"].get("eval_acc", 0.0)) - baseline)

    rows: List[Dict[str, Any]] = []
    for arm in out_arms:
        for point in arm["points"]:
            rows.append(
                {
                    "arm": arm["arm"],
                    "router_mode": arm["router_mode"],
                    "capacity_mode": arm["capacity_mode"],
                    "step": int(point.get("step", 0)),
                    "eval_acc": float(point.get("eval_acc", 0.0)),
                    "acc_delta": float(point.get("acc_delta", 0.0)),
                    "eval_loss": float(point.get("eval_loss", 0.0)),
                    "eval_n": int(point.get("eval_n", 0)),
                    "eval_seconds": float(point.get("eval_seconds", 0.0)),
                }
            )
    rows = sorted(rows, key=lambda x: (str(x["arm"]), int(x["step"])))
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "arm",
                "router_mode",
                "capacity_mode",
                "step",
                "eval_acc",
                "acc_delta",
                "eval_loss",
                "eval_n",
                "eval_seconds",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    winner = None
    if out_arms:
        winner = max(
            out_arms,
            key=lambda arm: (
                float(arm.get("final", {}).get("acc_delta", -1e9)),
                float(arm.get("slope_acc_delta_per_step", -1e9)),
            ),
        )

    out = dict(old)
    out["generated_utc"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    out["run_root"] = str(run_root)
    out["arms"] = out_arms
    out["matrix_summary_csv"] = str(csv_path)
    out["winner_by_final_acc_delta_then_slope"] = winner["arm"] if isinstance(winner, dict) else None
    summary_path.write_text(json.dumps(out, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(out, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

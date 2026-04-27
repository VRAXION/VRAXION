"""Phase D8.3 instrumentation-only audit.

Validates that D8 state logs are present, deterministic, parent-linked, and
consistent with panel_timeseries without judging search quality.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd


DEFAULT_REPORT = Path("docs/research/PHASE_D8_INSTRUMENTATION_AUDIT.md")
REQUIRED_COLUMNS = [
    "schema_version",
    "state_id",
    "parent_id",
    "family_id",
    "root_family_id",
    "run_id",
    "phase",
    "arm",
    "seed",
    "H",
    "panel_index",
    "step",
    "time_pct",
    "accepted_total",
    "rejected_total",
    "current_peak",
    "panel_probe_acc",
    "accept_rate_window",
    "accepted_window",
    "rejected_window",
    "edges",
    "unique_predictions",
    "collision_rate",
    "f_active",
    "stable_rank",
    "kernel_rank",
    "separation_sp",
    "checkpoint_ref",
    "archive_cell_id",
    "psi_pred",
    "cell_confidence",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--report", default=DEFAULT_REPORT, type=Path)
    return parser.parse_args()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def numeric_close(a: pd.Series, b: pd.Series, tol: float = 1e-12) -> bool:
    av = pd.to_numeric(a, errors="coerce")
    bv = pd.to_numeric(b, errors="coerce")
    return bool(((av - bv).abs().fillna(0.0) <= tol).all())


def audit_run(run_dir: Path) -> dict:
    meta = read_json(run_dir / "run_meta.json")
    log_path = Path(str(meta.get("d8_state_log") or run_dir / "accepted_state_log.csv"))
    if not log_path.exists():
        log_path = run_dir / "accepted_state_log.csv"
    panel_path = Path(str(meta.get("panel_timeseries") or run_dir / "panel_timeseries.csv"))
    if not panel_path.exists():
        panel_path = run_dir / "panel_timeseries.csv"
    row = {
        "run_dir": str(run_dir),
        "run_id": str(meta.get("run_id", "")),
        "phase": str(meta.get("phase", "")),
        "arm": str(meta.get("arm", "")),
        "H": int(meta.get("H", meta.get("h", -1))),
        "seed": int(meta.get("seed", -1)),
        "schema_version": str(meta.get("instrumentation_schema_version", "")),
        "state_log_exists": log_path.exists(),
        "panel_exists": panel_path.exists(),
        "rows": 0,
        "expected_rows": 0,
        "missing_columns": "",
        "state_id_ok": False,
        "parent_id_ok": False,
        "family_id_ok": False,
        "panel_consistency_ok": False,
        "checkpoint_ref_ok": False,
        "pass": False,
    }
    if not log_path.exists() or not panel_path.exists():
        return row
    states = pd.read_csv(log_path)
    panel = pd.read_csv(panel_path)
    row["rows"] = int(len(states))
    row["expected_rows"] = int(len(panel))
    missing = [c for c in REQUIRED_COLUMNS if c not in states.columns]
    row["missing_columns"] = ",".join(missing)
    if missing or len(states) != len(panel) or states.empty:
        return row

    run_id = row["run_id"]
    expected_state = [f"{run_id}::{i}" for i in range(len(states))]
    expected_parent = [""] + [f"{run_id}::{i}" for i in range(len(states) - 1)]
    row["state_id_ok"] = states["state_id"].astype(str).tolist() == expected_state
    parent_values = states["parent_id"].fillna("").astype(str).tolist()
    row["parent_id_ok"] = parent_values == expected_parent
    row["family_id_ok"] = bool(
        (states["family_id"].astype(str) == run_id).all()
        and (states["root_family_id"].astype(str) == run_id).all()
    )
    compare_cols = [
        ("step", "step"),
        ("current_peak", "main_peak_acc"),
        ("panel_probe_acc", "panel_probe_acc"),
        ("accept_rate_window", "accept_rate_window"),
        ("accepted_window", "accepted_window"),
        ("rejected_window", "rejected_window"),
        ("edges", "edges"),
        ("unique_predictions", "unique_predictions"),
        ("collision_rate", "collision_rate"),
        ("f_active", "f_active"),
        ("stable_rank", "stable_rank"),
        ("kernel_rank", "kernel_rank"),
        ("separation_sp", "separation_sp"),
    ]
    row["panel_consistency_ok"] = all(numeric_close(states[a], panel[b]) for a, b in compare_cols)
    checkpoint_ref = states["checkpoint_ref"].fillna("").astype(str)
    row["checkpoint_ref_ok"] = bool((checkpoint_ref == str(meta.get("checkpoint", ""))).all())
    row["pass"] = bool(
        row["schema_version"] == "d8_state_log_v1"
        and row["state_id_ok"]
        and row["parent_id_ok"]
        and row["family_id_ok"]
        and row["panel_consistency_ok"]
        and row["checkpoint_ref_ok"]
    )
    return row


def decide(runs: pd.DataFrame) -> tuple[str, dict]:
    if runs.empty:
        return "D8_INSTRUMENTATION_REGRESSION", {"reason": "no_runs"}
    pass_count = int(runs["pass"].sum())
    verdict = "D8_INSTRUMENTATION_LOCK" if pass_count == len(runs) else "D8_INSTRUMENTATION_REGRESSION"
    return verdict, {
        "runs": int(len(runs)),
        "pass_count": pass_count,
        "fail_count": int(len(runs) - pass_count),
        "rows_total": int(runs["rows"].sum()),
        "expected_rows_total": int(runs["expected_rows"].sum()),
        "schema_versions": sorted(runs["schema_version"].dropna().astype(str).unique().tolist()),
    }


def write_report(path: Path, verdict: str, decision: dict, runs: pd.DataFrame, root: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Phase D8.3 Instrumentation Audit",
        "",
        f"Verdict: **{verdict}**",
        "",
        "## Summary",
        "",
        "- D8.3 is instrumentation-only: it validates state logs and panel consistency, not search improvement.",
        "- Passing means state IDs, parent links, family IDs, and checkpoint references are deterministic and archive-compatible.",
        "- Live archive parent selection remains out of scope until this instrumentation is locked.",
        "",
        "## Coverage",
        "",
        "```json",
        json.dumps({"root": str(root), **decision}, indent=2),
        "```",
        "",
        "## Run Audit",
        "",
        "```text",
        runs.to_string(index=False, max_rows=120),
        "```",
        "",
        "## Interpretation",
        "",
    ]
    if verdict == "D8_INSTRUMENTATION_LOCK":
        lines.append("- Instrumentation is stable enough for a later D8.4 live archive-parent microprobe.")
    else:
        lines.append("- Instrumentation failed; do not run live archive selection until row/link consistency is fixed.")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    run_dirs = sorted(p for p in args.root.glob("H_*/D8*/seed_*") if p.is_dir())
    if not run_dirs:
        run_dirs = sorted(p for p in args.root.glob("D8*/seed_*") if p.is_dir())
    runs = pd.DataFrame([audit_run(p) for p in run_dirs])
    analysis_dir = args.root / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    verdict, decision = decide(runs)
    (analysis_dir / "summary.json").write_text(json.dumps({"verdict": verdict, "decision": decision}, indent=2), encoding="utf-8")
    runs.to_csv(analysis_dir / "instrumentation_run_audit.csv", index=False)
    write_report(args.report, verdict, decision, runs, args.root)
    print(f"Verdict: {verdict}")
    print(f"Runs: {decision.get('runs', 0)}")
    print(f"Wrote: {analysis_dir / 'summary.json'}")
    print(f"Wrote: {args.report}")
    return 0 if verdict == "D8_INSTRUMENTATION_LOCK" else 1


if __name__ == "__main__":
    raise SystemExit(main())

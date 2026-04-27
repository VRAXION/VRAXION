"""Phase D8.4a archive-parent microprobe analyzer.

This is a live-search microprobe audit, but it only evaluates whether archive
parent switching infrastructure is valid and whether the first parent-selection
arms show a paired signal versus current-best continuation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


DEFAULT_REPORT = Path("docs/research/PHASE_D8_ARCHIVE_PARENT_MICROPROBE.md")
BASELINE_ARM = "D8A_CURRENT_BEST"
TREATMENT_ARMS = ["D8A_RANDOM_ARCHIVE_PARENT", "D8A_SCORE_ARCHIVE_PARENT"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--report", default=DEFAULT_REPORT, type=Path)
    return parser.parse_args()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def count_csv_rows(path: Path) -> int:
    if not path.exists():
        return -1
    with path.open("r", encoding="utf-8", errors="replace") as f:
        return max(0, sum(1 for _ in f) - 1)


def audit_run(row: pd.Series) -> dict:
    run_dir = Path(str(row.get("run_dir", "")))
    meta = read_json(run_dir / "run_meta.json")
    state_path = Path(str(meta.get("d8_state_log") or row.get("d8_state_log") or run_dir / "accepted_state_log.csv"))
    parent_path = Path(str(meta.get("archive_parent_log") or row.get("archive_parent_log") or run_dir / "archive_parent_choice_log.csv"))
    panel_path = Path(str(meta.get("panel_timeseries") or row.get("panel_timeseries") or run_dir / "panel_timeseries.csv"))
    candidate_path = Path(str(row.get("candidate_log") or run_dir / "candidates.csv"))
    expected_candidates = int(float(row.get("expected_candidate_rows", 0)))
    out = {
        "run_dir": str(run_dir),
        "arm": str(row.get("arm", "")),
        "H": int(float(row.get("H", -1))),
        "seed": int(float(row.get("seed", -1))),
        "archive_parent_policy": str(meta.get("archive_parent_policy", row.get("archive_parent_policy", ""))),
        "candidate_rows": count_csv_rows(candidate_path),
        "expected_candidate_rows": expected_candidates,
        "state_log_exists": state_path.exists(),
        "archive_parent_log_exists": parent_path.exists(),
        "panel_exists": panel_path.exists(),
        "state_rows": 0,
        "parent_rows": 0,
        "panel_rows": 0,
        "restored_count": 0,
        "selected_ids_valid": False,
        "parent_links_valid": False,
        "pass": False,
    }
    if not state_path.exists() or not parent_path.exists() or not panel_path.exists():
        return out
    try:
        states = pd.read_csv(state_path)
        parents = pd.read_csv(parent_path)
        panel = pd.read_csv(panel_path)
    except Exception:
        return out
    out["state_rows"] = int(len(states))
    out["parent_rows"] = int(len(parents))
    out["panel_rows"] = int(len(panel))
    if states.empty or len(states) != len(panel) or len(parents) != len(panel):
        return out

    restored = parents[parents.get("restored", False).astype(str).str.lower().isin(["true", "1"])]
    out["restored_count"] = int(len(restored))
    state_ids = set(states["state_id"].astype(str))
    selected_ids = restored.get("selected_parent_state_id", pd.Series(dtype=str)).dropna().astype(str)
    selected_ids = selected_ids[selected_ids.str.len() > 0]
    out["selected_ids_valid"] = bool(selected_ids.map(lambda x: x in state_ids).all())

    parent_values = states["parent_id"].fillna("").astype(str).tolist()
    if out["arm"] == BASELINE_ARM:
        expected_parent = [""] + [f"{states['run_id'].iloc[0]}::{i}" for i in range(len(states) - 1)]
        out["parent_links_valid"] = parent_values == expected_parent and out["restored_count"] == 0
    else:
        # First panel starts from initial current-best lineage; later panels should
        # have either sequential or restored archive-parent links.
        valid_parent_refs = all((p == "" and i == 0) or p in state_ids for i, p in enumerate(parent_values))
        out["parent_links_valid"] = bool(valid_parent_refs and out["restored_count"] >= max(0, len(states) - 1))

    out["pass"] = bool(
        out["candidate_rows"] == expected_candidates
        and out["selected_ids_valid"]
        and out["parent_links_valid"]
        and out["state_rows"] == out["panel_rows"] == out["parent_rows"]
    )
    return out


def summarize_results(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (h, arm), g in results.groupby(["H", "arm"], dropna=False):
        rows.append({
            "H": int(h),
            "arm": str(arm),
            "n": int(len(g)),
            "peak_mean_pct": float(g["peak_acc"].mean() * 100.0),
            "peak_median_pct": float(g["peak_acc"].median() * 100.0),
            "final_mean_pct": float(g["final_acc"].mean() * 100.0),
            "accept_mean_pct": float(g["accept_rate_pct"].mean()),
            "wall_mean_s": float(g["wall_clock_s"].mean()),
        })
    return pd.DataFrame(rows).sort_values(["H", "arm"]).reset_index(drop=True)


def paired_deltas(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    baseline = results[results["arm"] == BASELINE_ARM].set_index(["H", "seed"])
    for _, row in results[results["arm"].isin(TREATMENT_ARMS)].iterrows():
        key = (row["H"], row["seed"])
        if key not in baseline.index:
            continue
        base = baseline.loc[key]
        rows.append({
            "H": int(row["H"]),
            "seed": int(row["seed"]),
            "arm": str(row["arm"]),
            "peak_delta_pp": float((row["peak_acc"] - base["peak_acc"]) * 100.0),
            "final_delta_pp": float((row["final_acc"] - base["final_acc"]) * 100.0),
            "accept_delta_pp": float(row["accept_rate_pct"] - base["accept_rate_pct"]),
        })
    return pd.DataFrame(rows)


def decide(run_audit: pd.DataFrame, deltas: pd.DataFrame) -> tuple[str, dict]:
    if run_audit.empty or not bool(run_audit["pass"].all()):
        return "D8_ARCHIVE_PARENT_INFRA_FAIL", {
            "runs": int(len(run_audit)),
            "pass_count": int(run_audit["pass"].sum()) if not run_audit.empty else 0,
        }
    if deltas.empty:
        return "D8_ARCHIVE_PARENT_WEAK_SIGNAL", {"reason": "no_paired_deltas"}

    arm_stats = []
    for arm, g in deltas.groupby("arm"):
        by_h = g.groupby("H")["peak_delta_pp"].mean()
        overall = float(g["peak_delta_pp"].median())
        positive_h = int((by_h > 0.0).sum())
        worst_h = float(by_h.min())
        arm_stats.append({
            "arm": arm,
            "overall_median_peak_delta_pp": overall,
            "positive_h_count": positive_h,
            "worst_h_peak_delta_pp": worst_h,
        })
    passing = [
        s for s in arm_stats
        if s["overall_median_peak_delta_pp"] > 0.0
        and s["positive_h_count"] >= 2
        and s["worst_h_peak_delta_pp"] >= -0.5
    ]
    if passing:
        return "D8_LIVE_ARCHIVE_PARENT_MICROPROBE_PASS", {"arm_stats": arm_stats, "passing_arms": passing}
    regressing = any(s["worst_h_peak_delta_pp"] < -0.5 for s in arm_stats)
    if regressing:
        return "D8_ARCHIVE_PARENT_REGRESSION", {"arm_stats": arm_stats}
    return "D8_ARCHIVE_PARENT_WEAK_SIGNAL", {"arm_stats": arm_stats}


def write_report(path: Path, verdict: str, decision: dict, summary: pd.DataFrame, deltas: pd.DataFrame, audit: pd.DataFrame, root: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Phase D8.4a Archive-Parent Microprobe",
        "",
        f"Verdict: **{verdict}**",
        "",
        "## Scope",
        "",
        "- Live search microprobe over SAF v1.",
        "- K(H), strict gate, operator schedule, horizon, and fixture are unchanged.",
        "- Only the parent source changes: current-best, random archive parent, or score archive parent.",
        "- This is not the full Ψ live controller; P2_PSI_CONF remains offline until model export/import is instrumented.",
        "",
        "## Decision",
        "",
        "```json",
        json.dumps({"root": str(root), **decision}, indent=2),
        "```",
        "",
        "## Per-H Arm Summary",
        "",
        "```text",
        summary.to_string(index=False, max_rows=120),
        "```",
        "",
        "## Paired Peak Deltas vs Current-Best",
        "",
        "```text",
        (deltas.to_string(index=False, max_rows=160) if not deltas.empty else "no paired deltas"),
        "```",
        "",
        "## Infrastructure Audit",
        "",
        "```text",
        audit.to_string(index=False, max_rows=160),
        "```",
        "",
        "## Interpretation",
        "",
    ]
    if verdict == "D8_LIVE_ARCHIVE_PARENT_MICROPROBE_PASS":
        lines.append("- Archive parent switching produced a positive paired live signal; run a wider D8.4 with more seeds before promotion.")
    elif verdict == "D8_ARCHIVE_PARENT_WEAK_SIGNAL":
        lines.append("- Infrastructure works, but this first parent-selection policy is not strong enough to promote.")
    elif verdict == "D8_ARCHIVE_PARENT_REGRESSION":
        lines.append("- Archive parent switching regressed at least one H materially; do not expand without changing the selector.")
    else:
        lines.append("- Infrastructure failed; fix state restoration/log consistency before any live archive tests.")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    results_path = args.root / "results.csv"
    if not results_path.exists():
        raise SystemExit(f"missing results.csv: {results_path}")
    results = pd.read_csv(results_path)
    run_audit = pd.DataFrame([audit_run(row) for _, row in results.iterrows()])
    summary = summarize_results(results)
    deltas = paired_deltas(results)
    verdict, decision = decide(run_audit, deltas)

    analysis_dir = args.root / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    (analysis_dir / "summary.json").write_text(
        json.dumps({"verdict": verdict, "decision": decision}, indent=2),
        encoding="utf-8",
    )
    summary.to_csv(analysis_dir / "per_H_arm_summary.csv", index=False)
    deltas.to_csv(analysis_dir / "seed_paired_deltas.csv", index=False)
    run_audit.to_csv(analysis_dir / "archive_parent_run_audit.csv", index=False)
    write_report(args.report, verdict, decision, summary, deltas, run_audit, args.root)
    print(f"Verdict: {verdict}")
    print(f"Runs: {len(results)}")
    print(f"Wrote: {analysis_dir / 'summary.json'}")
    print(f"Wrote: {args.report}")
    return 0 if verdict != "D8_ARCHIVE_PARENT_INFRA_FAIL" else 1


if __name__ == "__main__":
    raise SystemExit(main())

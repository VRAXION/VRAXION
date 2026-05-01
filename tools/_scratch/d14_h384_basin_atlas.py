#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[2]
BASELINE = Path("output/phase_d7_operator_bandit_20260427/H_384/D7_BASELINE/seed_2042/final.ckpt")
TOP01 = Path("output/releases/v5.0.0-beta.10/seed2042_top01_h384_research.ckpt")
D9_ROOT = Path("output/phase_d9_3a_quadtree_scan_20260429")
D10U_CANDIDATES = Path("output/phase_d10u_focused_ladder_20260430/bounded/candidate_summary.csv")
D13_DOC = Path("docs/research/PHASE_D13_H384_TOP01_RESEARCH_RELEASE_PACKAGE.md")
DEFAULT_OUT = Path("output/phase_d14_h384_state_anchored_basin_atlas_20260501")
DEFAULT_TILES = "9_26,12_29,11_16"
DEFAULT_CONTROLS = "random_projection_null,state_shuffle_shared,state_shuffle_projection_consistent,no_network_random_state"


def rel(path: Path) -> Path:
    return path if path.is_absolute() else REPO / path


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def parse_float(row: dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


def run(cmd: list[str], log_path: Path | None = None) -> None:
    safe_print("RUN " + " ".join(cmd))
    started = time.time()
    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.Popen(
                cmd,
                cwd=REPO,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
            assert process.stdout is not None
            last_heartbeat = started
            for line in process.stdout:
                log.write(line)
                log.flush()
                safe_print(line.rstrip())
                now = time.time()
                if now - last_heartbeat >= 60:
                    safe_print(f"[d14 heartbeat] elapsed_s={now - started:.1f} cmd={' '.join(cmd[:3])}")
                    last_heartbeat = now
            code = process.wait()
    else:
        code = subprocess.call(cmd, cwd=REPO)
    if code != 0:
        raise RuntimeError(f"command failed with code {code}: {' '.join(cmd)}")


def safe_print(text: str) -> None:
    encoding = sys.stdout.encoding or "utf-8"
    sys.stdout.write(text.encode(encoding, errors="replace").decode(encoding, errors="replace") + "\n")
    sys.stdout.flush()


def require(path: Path, name: str) -> None:
    if not rel(path).exists():
        raise FileNotFoundError(f"missing {name}: {path}")


def consolidate_atlas(out: Path) -> dict[str, Any]:
    child_tiles = read_csv(rel(D9_ROOT / "child_tiles.csv"))
    child_candidates = read_csv(rel(D9_ROOT / "child_candidates.csv"))
    d10u_rows = read_csv(rel(D10U_CANDIDATES))
    d13_text = rel(D13_DOC).read_text(encoding="utf-8") if rel(D13_DOC).exists() else ""

    full_rows = [row for row in child_tiles if row.get("mo_class") == "FULL_GENERALIST"]
    child_full = [row for row in full_rows if row.get("is_control") == "false"]
    control_full = [row for row in full_rows if row.get("is_control") == "true"]

    priority: dict[str, dict[str, Any]] = {}
    for row in child_full + control_full:
        child = row["child_tile_id"]
        entry = priority.setdefault(
            child,
            {
                "child_tile_id": child,
                "parent_tile_id": row.get("parent_tile_id", ""),
                "is_control": row.get("is_control", ""),
                "full_count": 0,
                "best_mo_score": -999.0,
                "best_smooth_delta": 0.0,
                "best_accuracy_delta": 0.0,
                "best_echo_delta": 0.0,
                "best_unigram_delta": 0.0,
                "priority_reason": "",
            },
        )
        entry["full_count"] += 1
        mo = parse_float(row, "mo_score")
        if mo > float(entry["best_mo_score"]):
            entry["best_mo_score"] = mo
            entry["best_smooth_delta"] = parse_float(row, "smooth_delta")
            entry["best_accuracy_delta"] = parse_float(row, "accuracy_delta")
            entry["best_echo_delta"] = parse_float(row, "echo_delta")
            entry["best_unigram_delta"] = parse_float(row, "unigram_delta")
            entry["priority_reason"] = "d9_3a_full_generalist_control" if row.get("is_control") == "true" else "d9_3a_full_generalist_child"

    priority_rows = sorted(
        priority.values(),
        key=lambda row: (
            row["is_control"] == "false",
            float(row["best_mo_score"]),
            int(row["full_count"]),
        ),
        reverse=True,
    )
    for idx, row in enumerate(priority_rows, start=1):
        row["rank"] = idx

    d10u_top = []
    for row in d10u_rows[:20]:
        d10u_top.append(
            {
                "source": "d10u_focused_ladder",
                "seed_label": row.get("seed_label", ""),
                "arm": row.get("arm", ""),
                "candidate_class": row.get("candidate_class", ""),
                "candidate_label": row.get("candidate_label", ""),
                "checkpoint_path": row.get("checkpoint_path", ""),
                "smooth_delta": row.get("smooth_delta", ""),
                "accuracy_delta": row.get("accuracy_delta", ""),
                "unigram_delta": row.get("unigram_delta", ""),
                "hardened_selectivity": row.get("hardened_selectivity", ""),
            }
        )

    atlas_rows = []
    for row in priority_rows:
        atlas_rows.append(
            {
                "source": "d9_3a_quadtree",
                "rank": row["rank"],
                "parent_tile_id": row["parent_tile_id"],
                "child_tile_id": row["child_tile_id"],
                "is_control": row["is_control"],
                "full_count": row["full_count"],
                "best_mo_score": row["best_mo_score"],
                "best_smooth_delta": row["best_smooth_delta"],
                "best_accuracy_delta": row["best_accuracy_delta"],
                "best_echo_delta": row["best_echo_delta"],
                "best_unigram_delta": row["best_unigram_delta"],
                "priority_reason": row["priority_reason"],
            }
        )

    write_csv(
        out / "basin_priority_tiles.csv",
        priority_rows,
        [
            "rank",
            "parent_tile_id",
            "child_tile_id",
            "is_control",
            "full_count",
            "best_mo_score",
            "best_smooth_delta",
            "best_accuracy_delta",
            "best_echo_delta",
            "best_unigram_delta",
            "priority_reason",
        ],
    )
    write_csv(
        out / "atlas_seed_map.csv",
        atlas_rows,
        [
            "source",
            "rank",
            "parent_tile_id",
            "child_tile_id",
            "is_control",
            "full_count",
            "best_mo_score",
            "best_smooth_delta",
            "best_accuracy_delta",
            "best_echo_delta",
            "best_unigram_delta",
            "priority_reason",
        ],
    )
    write_csv(
        out / "d10u_reference_candidates.csv",
        d10u_top,
        [
            "source",
            "seed_label",
            "arm",
            "candidate_class",
            "candidate_label",
            "checkpoint_path",
            "smooth_delta",
            "accuracy_delta",
            "unigram_delta",
            "hardened_selectivity",
        ],
    )

    summary = {
        "d9_child_tile_rows": len(child_tiles),
        "d9_candidate_rows": len(child_candidates),
        "d9_full_child_rows": len(child_full),
        "d9_full_control_rows": len(control_full),
        "priority_tiles": len(priority_rows),
        "d10u_reference_rows": len(d10u_top),
        "d13_packaged": "D10U_TOP01_16K_SHARDED_PASS" in d13_text,
    }
    (out / "atlas_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def run_quadtree(args: argparse.Namespace, out: Path) -> Path:
    scan_out = out / "quadtree_hot"
    if args.skip_existing_scan and (scan_out / "child_candidates.csv").exists():
        print(f"Using existing quadtree scan: {scan_out}", flush=True)
        return scan_out
    cmd = [
        str(rel(Path("target/release/examples/d9_direct_landscape.exe"))),
        "--checkpoint",
        str(BASELINE),
        "--repair-start",
        str(TOP01),
        "--H",
        "384",
        "--mode",
        "quadtree-scout",
        "--tiles",
        args.tiles,
        "--mutation-types",
        "edge,threshold",
        "--radii",
        args.radii,
        "--samples-per-tile",
        str(args.samples_per_tile),
        "--eval-len",
        str(args.eval_len),
        "--mo-eval-seeds",
        args.mo_eval_seeds,
        "--mo-export-top",
        str(args.mo_export_top),
        "--out",
        str(scan_out),
    ]
    run(cmd, out / "quadtree_hot.log")
    return scan_out


def run_d10r_filter(
    args: argparse.Namespace,
    scan_out: Path,
    out: Path,
    confirm: bool = False,
    ranks: set[int] | None = None,
) -> list[dict[str, Any]]:
    candidates = read_csv(scan_out / "child_candidates.csv")
    if ranks:
        top = [row for row in candidates if int(row["rank"]) in ranks]
    else:
        top = candidates[: args.confirm_top if confirm else args.filter_top]
    filter_root = out / ("state_anchor_confirm" if confirm else "state_anchor_filter")
    rows: list[dict[str, Any]] = []
    for row in top:
        rank = row["rank"]
        ckpt = Path(row["checkpoint"])
        eval_len = args.confirm_eval_len if confirm else args.filter_eval_len
        eval_seeds = args.confirm_eval_seeds if confirm else args.filter_eval_seeds
        repeats = args.confirm_control_repeats if confirm else args.filter_control_repeats
        candidate_out = filter_root / f"rank_{int(rank):02}"
        cmd = [
            sys.executable,
            "tools/_scratch/d10r_hardened_eval.py",
            "--device",
            args.device,
            "--eval-len",
            str(eval_len),
            "--eval-seeds",
            eval_seeds,
            "--control-repeats",
            str(repeats),
            "--artifact-controls",
            DEFAULT_CONTROLS,
            "--baseline",
            str(BASELINE),
            "--positive",
            str(ckpt),
            "--positive-label",
            f"d14_rank_{int(rank):02}",
            "--alternate-baseline-checkpoints",
            "",
            "--no-state-zone-diagnostics",
            "--out",
            str(candidate_out),
        ]
        run(cmd, candidate_out / "d10r.log")
        summary_path = candidate_out / "d10r_run_summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        positive = summary["checkpoint_summaries"][0]
        filter_row = {
            "phase": "confirm" if confirm else "filter",
            "rank": rank,
            "checkpoint": str(ckpt),
            "parent_tile_id": row.get("parent_tile_id", ""),
            "child_tile_id": row.get("child_tile_id", ""),
            "is_control": row.get("is_control", ""),
            "mo_class": row.get("mo_class", ""),
            "quadtree_mo_score": row.get("mo_score", ""),
            "verdict": summary.get("verdict", ""),
            "artifact_gate_pass": summary.get("artifact_gate_pass", False),
            "real_mo_delta_mean": positive.get("real_mo_delta_mean", 0.0),
            "real_mo_delta_ci_low": positive.get("real_mo_delta_ci_low", 0.0),
            "trusted_mo_mean": positive.get("trusted_mo_mean", 0.0),
            "trusted_mo_ci_low": positive.get("trusted_mo_ci_low", 0.0),
            "worst_control": ",".join(positive.get("blocking_controls", [])),
            "blocking_control_families": ",".join(summary.get("blocking_control_families", [])),
            "d10r_out": str(candidate_out),
        }
        rows.append(filter_row)
        write_csv(
            out / ("state_anchor_confirm.csv" if confirm else "state_anchor_filter.csv"),
            rows,
            state_anchor_fields(),
        )
    return rows


def state_anchor_fields() -> list[str]:
    return [
        "phase",
        "rank",
        "checkpoint",
        "parent_tile_id",
        "child_tile_id",
        "is_control",
        "mo_class",
        "quadtree_mo_score",
        "verdict",
        "artifact_gate_pass",
        "real_mo_delta_mean",
        "real_mo_delta_ci_low",
        "trusted_mo_mean",
        "trusted_mo_ci_low",
        "worst_control",
        "blocking_control_families",
        "d10r_out",
    ]


def reload_candidates(args: argparse.Namespace, scan_out: Path, out: Path) -> None:
    if args.reload_top <= 0:
        return
    rows = read_csv(scan_out / "child_candidates.csv")[: args.reload_top]
    reload_rows = []
    for row in rows:
        ckpt = Path(row["checkpoint"])
        log_path = out / "reload_smoke" / f"rank_{int(row['rank']):02}.log"
        cmd = [
            str(rel(Path("target/release/examples/chain_diagnosis.exe"))),
            str(ckpt),
            "output/block_c_bytepair_champion/packed.bin",
        ]
        run(cmd, log_path)
        reload_rows.append({"rank": row["rank"], "checkpoint": str(ckpt), "log": str(log_path), "reload_ok": True})
    write_csv(out / "reload_smoke.csv", reload_rows, ["rank", "checkpoint", "log", "reload_ok"])


def decide_verdict(filter_rows: list[dict[str, Any]], confirm_rows: list[dict[str, Any]]) -> str:
    rows = confirm_rows or filter_rows
    pass_rows = [row for row in rows if str(row.get("artifact_gate_pass")).lower() == "true" and float(row.get("trusted_mo_ci_low", 0.0)) > 0.0]
    if not pass_rows:
        return "D14_NO_NEW_BASIN"
    child_keys = {(row["parent_tile_id"], row["child_tile_id"], row["is_control"]) for row in pass_rows}
    control_pass = any(str(row["is_control"]).lower() == "true" for row in pass_rows)
    non_control_keys = {(p, c) for p, c, is_control in child_keys if str(is_control).lower() != "true"}
    if control_pass and len(non_control_keys) <= 1:
        return "D14_CONTROL_DOMINATED"
    if len(non_control_keys) >= 2:
        return "D14_MULTI_BASIN_SIGNAL"
    return "D14_TOP01_ONLY"


def select_confirm_ranks(filter_rows: list[dict[str, Any]], limit: int) -> set[int]:
    pass_rows = [
        row
        for row in filter_rows
        if str(row.get("artifact_gate_pass")).lower() == "true"
        and float(row.get("trusted_mo_ci_low", 0.0)) > 0.0
    ]
    selected: list[dict[str, Any]] = []
    seen_regions: set[tuple[str, str, str]] = set()
    non_controls = [row for row in pass_rows if str(row.get("is_control")).lower() != "true"]
    controls = [row for row in pass_rows if str(row.get("is_control")).lower() == "true"]
    for group in (non_controls, controls):
        group.sort(
            key=lambda row: (
                float(row.get("trusted_mo_ci_low", 0.0)),
                float(row.get("real_mo_delta_ci_low", 0.0)),
            ),
            reverse=True,
        )

    target_non_controls = limit if not controls else max(1, limit - 1)
    for row in non_controls:
        region = (str(row["parent_tile_id"]), str(row["child_tile_id"]), str(row["is_control"]))
        if region in seen_regions:
            continue
        selected.append(row)
        seen_regions.add(region)
        if len(selected) >= target_non_controls:
            break
    if controls and len(selected) < limit:
        selected.append(controls[0])
    for row in non_controls:
        if len(selected) >= limit:
            break
        region = (str(row["parent_tile_id"]), str(row["child_tile_id"]), str(row["is_control"]))
        if region not in seen_regions:
            selected.append(row)
            seen_regions.add(region)
    return {int(row["rank"]) for row in selected}


def write_report(out: Path, atlas_summary: dict[str, Any], filter_rows: list[dict[str, Any]], confirm_rows: list[dict[str, Any]]) -> None:
    verdict = decide_verdict(filter_rows, confirm_rows)
    lines = [
        "# D14 H384 State-Anchored Basin Atlas Report",
        "",
        f"Verdict: `{verdict}`",
        "",
        "## Summary",
        "",
        f"- D9.3a child tile rows loaded: {atlas_summary.get('d9_child_tile_rows', 0)}",
        f"- D9.3a FULL_GENERALIST child rows: {atlas_summary.get('d9_full_child_rows', 0)}",
        f"- D9.3a FULL_GENERALIST control rows: {atlas_summary.get('d9_full_control_rows', 0)}",
        f"- D13 top_01 package present: {atlas_summary.get('d13_packaged', False)}",
        f"- State-anchor filter rows: {len(filter_rows)}",
        f"- Focused confirm rows: {len(confirm_rows)}",
        "",
        "## State-Anchor Filter",
        "",
        "| phase | rank | parent | child | control | verdict | trusted_ci_low | real_ci_low |",
        "|---|---:|---|---|---|---|---:|---:|",
    ]
    for row in filter_rows:
        lines.append(
            f"| {row['phase']} | {row['rank']} | {row['parent_tile_id']} | {row['child_tile_id']} | {row['is_control']} | {row['verdict']} | {float(row['trusted_mo_ci_low']):.6f} | {float(row['real_mo_delta_ci_low']):.6f} |"
        )
    if confirm_rows:
        lines += [
            "",
            "## Focused Confirm",
            "",
            "| phase | rank | parent | child | control | verdict | trusted_ci_low | real_ci_low |",
            "|---|---:|---|---|---|---|---:|---:|",
        ]
        for row in confirm_rows:
            lines.append(
                f"| {row['phase']} | {row['rank']} | {row['parent_tile_id']} | {row['child_tile_id']} | {row['is_control']} | {row['verdict']} | {float(row['trusted_mo_ci_low']):.6f} | {float(row['real_mo_delta_ci_low']):.6f} |"
            )
    lines += [
        "",
        "## Progress Map",
        "",
        "```text",
        "GLOBAL RELEASE-READY AI MAP",
        "",
        "[1] One proven H384 checkpoint",
        "    DONE: D13 top_01 package",
        "",
        "[2] H384 basin atlas",
        f"    CURRENT VERDICT: {verdict}",
        "",
        "[3] Next gate",
        "    if D14_MULTI_BASIN_SIGNAL: run 16k/30-seed confirm for best new basin",
        "    else: stop atlas expansion and shift to capability/context improvement",
        "```",
        "",
        "Generated output data remains uncommitted by default.",
    ]
    (out / "D14_H384_BASIN_ATLAS_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out / "run_summary.json").write_text(
        json.dumps(
            {
                "verdict": verdict,
                "atlas_summary": atlas_summary,
                "filter_rows": filter_rows,
                "confirm_rows": confirm_rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="D14 H384 state-anchored basin atlas harness")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--tiles", default=DEFAULT_TILES)
    parser.add_argument("--radii", default="4,8,16,32")
    parser.add_argument("--samples-per-tile", type=int, default=2)
    parser.add_argument("--eval-len", type=int, default=1000)
    parser.add_argument("--mo-eval-seeds", default="971001,971002,971003,971004")
    parser.add_argument("--mo-export-top", type=int, default=12)
    parser.add_argument("--run-scan", action="store_true")
    parser.add_argument("--skip-existing-scan", action="store_true")
    parser.add_argument("--run-filter", action="store_true")
    parser.add_argument("--filter-top", type=int, default=6)
    parser.add_argument("--filter-eval-len", type=int, default=256)
    parser.add_argument("--filter-eval-seeds", default="971011,971012,971013,971014")
    parser.add_argument("--filter-control-repeats", type=int, default=1)
    parser.add_argument("--run-confirm", action="store_true")
    parser.add_argument("--confirm-top", type=int, default=3)
    parser.add_argument("--confirm-eval-len", type=int, default=4000)
    parser.add_argument("--confirm-eval-seeds", default="971021,971022,971023,971024,971025,971026,971027,971028")
    parser.add_argument("--confirm-control-repeats", type=int, default=2)
    parser.add_argument("--reload-top", type=int, default=3)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out = rel(Path(args.out))
    out.mkdir(parents=True, exist_ok=True)
    require(BASELINE, "H384 seed2042 baseline")
    require(TOP01, "D13 top_01 release checkpoint")
    require(D9_ROOT / "child_tiles.csv", "D9.3a child_tiles.csv")
    require(D9_ROOT / "child_candidates.csv", "D9.3a child_candidates.csv")
    require(Path("target/release/examples/d9_direct_landscape.exe"), "d9_direct_landscape release example")
    require(Path("target/release/examples/chain_diagnosis.exe"), "chain_diagnosis release example")

    atlas_summary = consolidate_atlas(out)
    scan_out = out / "quadtree_hot"
    if args.run_scan:
        scan_out = run_quadtree(args, out)
    elif not (scan_out / "child_candidates.csv").exists():
        # For dry consolidation runs, fall back to the existing D9.3 scan.
        scan_out = rel(D9_ROOT)

    reload_candidates(args, scan_out, out)

    filter_rows: list[dict[str, Any]] = []
    confirm_rows: list[dict[str, Any]] = []
    if args.run_filter:
        filter_rows = run_d10r_filter(args, scan_out, out, confirm=False)
    if args.run_confirm:
        confirm_ranks = select_confirm_ranks(filter_rows, args.confirm_top) if filter_rows else None
        if confirm_ranks:
            (out / "confirm_rank_selection.json").write_text(
                json.dumps({"confirm_ranks": sorted(confirm_ranks)}, indent=2),
                encoding="utf-8",
            )
        confirm_rows = run_d10r_filter(args, scan_out, out, confirm=True, ranks=confirm_ranks)
    write_report(out, atlas_summary, filter_rows, confirm_rows)
    print(json.dumps(json.loads((out / "run_summary.json").read_text(encoding="utf-8")), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

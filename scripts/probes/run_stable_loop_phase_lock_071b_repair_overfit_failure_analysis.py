#!/usr/bin/env python3
"""Analysis-only failure attribution for STABLE_LOOP_PHASE_LOCK_071B."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]

REQUIRED_071_ARTIFACTS = [
    "summary.json",
    "per_family_metrics.json",
    "human_readable_samples.jsonl",
    "failure_case_samples.jsonl",
    "baseline_metrics.json",
    "no_route_feature_control_metrics.json",
]

TARGET_FAMILIES = [
    "FRESH_COUNTERFACTUAL_BINDING",
    "FRESH_CONTEXT_ENTITY_EXTRACTION",
    "FRESH_IRRELEVANT_POCKET_SUPPRESSION",
]

WRONG_SOURCE_LABELS = [
    "old_scenario_value",
    "distractor_scenario_value",
    "first_ledger_value",
    "side_note_value",
    "inactive_pocket_value",
    "stale_pocket_value",
    "copy_first_match_value",
    "no_route_control_value",
    "unknown_label",
]

POSITIVE_VERDICTS = [
    "REPAIR_OVERFIT_FAILURE_ANALYSIS_POSITIVE",
    "UPSTREAM_071_FAILURE_PROFILE_LOADED",
    "COUNTERFACTUAL_FAILURE_CLUSTERS_WRITTEN",
    "CONTEXT_EXTRACTION_FAILURE_CLUSTERS_WRITTEN",
    "POCKET_SUPPRESSION_FAILURE_CLUSTERS_WRITTEN",
    "WRONG_ANSWER_SOURCE_ATTRIBUTION_WRITTEN",
    "ACTIVE_SCENARIO_MISS_RATE_RECORDED",
    "DISTRACTOR_SCENARIO_SELECTION_RECORDED",
    "KEY_COLLISION_REPORT_WRITTEN",
    "CURRICULUM_PATCH_RECOMMENDED",
    "NO_TRAINING_PERFORMED",
    "PRODUCTION_TRAINING_NOT_CLAIMED",
]


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def normalized_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT)).replace("\\", "/")
    except ValueError:
        return str(path.resolve()).replace("\\", "/")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSONL row: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no}: JSONL row is not an object")
            rows.append(row)
    return rows


def rate(numerator: int, denominator: int) -> float:
    return 0.0 if denominator == 0 else numerator / denominator


def init_run(out: Path, args: argparse.Namespace) -> None:
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "queue.json", {
        "schema_version": "repair_overfit_failure_analysis_queue_v1",
        "probe": "STABLE_LOOP_PHASE_LOCK_071B_REPAIR_OVERFIT_FAILURE_ANALYSIS",
        "analysis_only": True,
        "no_training": True,
        "created_at": now_iso(),
    })
    write_json(out / "analysis_config.json", {
        "schema_version": "repair_overfit_failure_analysis_config_v1",
        "out": normalized_path(out),
        "upstream_071_root": normalized_path(args.upstream_071_root),
        "upstream_070_root": normalized_path(args.upstream_070_root),
        "benchmark_069_root": normalized_path(args.benchmark_069_root),
        "heartbeat_sec": args.heartbeat_sec,
        "target_families": TARGET_FAMILIES,
        "unknown_label_hard_limit": 0.20,
        "analysis_only": True,
        "train_step_count": 0,
        "checkpoint_mutation_possible": False,
    })
    write_json(out / "summary.json", {
        "schema_version": "repair_overfit_failure_analysis_summary_v1",
        "status": "running",
        "analysis_only": True,
        "model_capability_improved": False,
        "training_performed": False,
        "checkpoint_repaired": False,
        "verdicts": [],
    })
    write_report(out, {
        "status": "running",
        "verdicts": [],
        "unknown_label_rate": None,
        "dominant_findings": [],
    })
    append_progress(out, "start", {"analysis_only": True})


def append_progress(out: Path, event: str, extra: dict[str, Any] | None = None) -> None:
    payload = {
        "ts": now_iso(),
        "event": event,
    }
    if extra:
        payload.update(extra)
    append_jsonl(out / "progress.jsonl", payload)


def fail(out: Path, verdict: str, message: str, extra: dict[str, Any] | None = None) -> int:
    payload = {
        "schema_version": "repair_overfit_failure_analysis_summary_v1",
        "status": "failed",
        "analysis_only": True,
        "model_capability_improved": False,
        "training_performed": False,
        "checkpoint_repaired": False,
        "error": {"code": verdict, "message": message},
        "extra": extra or {},
        "verdicts": ["REPAIR_OVERFIT_FAILURE_ANALYSIS_FAILS", verdict, "NO_TRAINING_PERFORMED"],
    }
    write_json(out / "summary.json", payload)
    write_report(out, {
        "status": "failed",
        "verdicts": payload["verdicts"],
        "unknown_label_rate": (extra or {}).get("unknown_label_rate"),
        "dominant_findings": [message],
    })
    append_progress(out, "done", {"status": "failed", "verdict": verdict})
    print(json.dumps(payload, separators=(",", ":")))
    return 1


def upstream_manifest(root071: Path, root070: Path, root069: Path) -> tuple[list[str], dict[str, Any]]:
    missing: list[str] = []
    files: list[dict[str, Any]] = []
    for name in REQUIRED_071_ARTIFACTS:
        path = root071 / name
        if not path.exists():
            missing.append(normalized_path(path))
            continue
        stat = path.stat()
        files.append({
            "path": normalized_path(path),
            "size_bytes": stat.st_size,
            "mtime_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(stat.st_mtime)),
            "sha256": sha256_file(path),
        })
    for root, label in [(root070, "upstream_070_root"), (root069, "benchmark_069_root")]:
        if not root.exists():
            missing.append(f"{label}:{normalized_path(root)}")
    manifest = {
        "schema_version": "repair_overfit_upstream_071_manifest_v1",
        "upstream_071_root": normalized_path(root071),
        "upstream_070_root": normalized_path(root070),
        "benchmark_069_root": normalized_path(root069),
        "required_071_artifacts": REQUIRED_071_ARTIFACTS,
        "files": files,
        "missing": missing,
    }
    return missing, manifest


CTX_RE = re.compile(
    r"Requested anchor (?P<requested>\w+)\. Ledger first lists (?P<first_key>\w+) as "
    r"(?P<first_value>\w+); later lists exact anchor (?P<exact_key>\w+) as "
    r"(?P<exact_value>\w+); side note lists (?P<side_key>\w+) as (?P<side_value>\w+)\.",
)
CF_RE = re.compile(
    r"Old scenario: (?P<old_key>\w+) equals (?P<old_value>\w+)\. Active scenario: "
    r"(?P<active_key>\w+) equals (?P<active_value>\w+)\. Distractor scenario: "
    r"(?P<distractor_key>\w+) equals (?P<distractor_value>\w+)\.",
)
POCKET_RE = re.compile(
    r"Pocket red has (?P<red_key>\w+)->(?P<red_value>\w+)\. Pocket blue has "
    r"(?P<blue_key>\w+)->(?P<blue_value>\w+)\. Active pocket has "
    r"(?P<active_key>\w+)->(?P<active_value>\w+)\.",
)


def parse_template(row: dict[str, Any]) -> dict[str, str]:
    text = str(row.get("input", ""))
    family = row.get("task_family")
    regex = {
        "FRESH_CONTEXT_ENTITY_EXTRACTION": CTX_RE,
        "FRESH_COUNTERFACTUAL_BINDING": CF_RE,
        "FRESH_IRRELEVANT_POCKET_SUPPRESSION": POCKET_RE,
    }.get(family)
    if regex is None:
        return {}
    match = regex.search(text)
    return match.groupdict() if match else {}


def copy_first(row: dict[str, Any]) -> str | None:
    baselines = row.get("baseline_outputs")
    if isinstance(baselines, dict):
        value = baselines.get("COPY_FIRST_MATCH")
        return str(value) if value is not None else None
    return None


def classify_source(row: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    family = row.get("task_family")
    output = str(row.get("model_output", ""))
    expected = str(row.get("expected_output", ""))
    no_route = str(row.get("no_route_output", ""))
    first = copy_first(row)
    parsed = parse_template(row)
    if output == expected:
        return "expected_active_value", {"parsed": parsed}
    if family == "FRESH_COUNTERFACTUAL_BINDING" and parsed:
        if output == parsed.get("old_value"):
            return "old_scenario_value", {"parsed": parsed}
        if output == parsed.get("distractor_value"):
            return "distractor_scenario_value", {"parsed": parsed}
        if first is not None and output == first:
            return "copy_first_match_value", {"parsed": parsed}
        if output == no_route:
            return "no_route_control_value", {"parsed": parsed}
        return "unknown_label", {"parsed": parsed}
    if family == "FRESH_CONTEXT_ENTITY_EXTRACTION" and parsed:
        if output == parsed.get("first_value"):
            return "first_ledger_value", {"parsed": parsed}
        if output == parsed.get("side_value"):
            return "side_note_value", {"parsed": parsed}
        if first is not None and output == first:
            return "copy_first_match_value", {"parsed": parsed}
        if output == no_route:
            return "no_route_control_value", {"parsed": parsed}
        return "unknown_label", {"parsed": parsed}
    if family == "FRESH_IRRELEVANT_POCKET_SUPPRESSION" and parsed:
        if output == parsed.get("red_value"):
            return "inactive_pocket_value", {"parsed": parsed}
        if output == parsed.get("blue_value"):
            return "stale_pocket_value", {"parsed": parsed}
        if first is not None and output == first:
            return "copy_first_match_value", {"parsed": parsed}
        if output == no_route:
            return "no_route_control_value", {"parsed": parsed}
        return "unknown_label", {"parsed": parsed}
    if first is not None and output == first:
        return "copy_first_match_value", {"parsed": parsed}
    if output == no_route:
        return "no_route_control_value", {"parsed": parsed}
    return "unknown_label", {"parsed": parsed}


def row_records(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row in rows:
        family = str(row.get("task_family", ""))
        if family not in TARGET_FAMILIES:
            continue
        source, meta = classify_source(row)
        expected = str(row.get("expected_output", ""))
        model = str(row.get("model_output", ""))
        no_route = str(row.get("no_route_output", ""))
        first = copy_first(row)
        records.append({
            "task_family": family,
            "input": str(row.get("input", "")),
            "expected": expected,
            "model_output": model,
            "pass_fail": str(row.get("pass_fail", "")),
            "classified_wrong_source": source,
            "parsed": meta.get("parsed", {}),
            "no_route_output": no_route,
            "copy_first_match": first,
            "no_route_agreement": model == no_route,
            "copy_first_match_agreement": first is not None and model == first,
        })
    return records


def aggregate_family(records: list[dict[str, Any]], family: str) -> dict[str, Any]:
    subset = [row for row in records if row["task_family"] == family]
    failures = [row for row in subset if row["pass_fail"] == "fail"]
    source_counts = Counter(row["classified_wrong_source"] for row in failures)
    total = len(subset)
    fail_total = len(failures)
    correct = total - fail_total
    return {
        "task_family": family,
        "total_rows": total,
        "failure_rows": fail_total,
        "correct_rows": correct,
        "accuracy": rate(correct, total),
        "wrong_answer_source_counts": dict(source_counts),
        "unknown_label_rate": rate(source_counts.get("unknown_label", 0), fail_total),
        "copy_first_match_agreement_rate": rate(sum(1 for row in failures if row["copy_first_match_agreement"]), fail_total),
        "no_route_agreement_rate": rate(sum(1 for row in failures if row["no_route_agreement"]), fail_total),
    }


def counterfactual_report(records: list[dict[str, Any]]) -> dict[str, Any]:
    family = "FRESH_COUNTERFACTUAL_BINDING"
    base = aggregate_family(records, family)
    failures = [row for row in records if row["task_family"] == family and row["pass_fail"] == "fail"]
    counts = Counter(row["classified_wrong_source"] for row in failures)
    total = len(failures)
    first_ledger = sum(1 for row in failures if row["copy_first_match_agreement"])
    same_key_count = 0
    for row in [r for r in records if r["task_family"] == family]:
        parsed = row["parsed"]
        if parsed and parsed.get("old_key") == parsed.get("active_key"):
            same_key_count += 1
    base.update({
        "active_scenario_miss_rate": rate(total, base["total_rows"]),
        "old_scenario_selection_rate": rate(counts.get("old_scenario_value", 0), total),
        "distractor_scenario_selection_rate": rate(counts.get("distractor_scenario_value", 0), total),
        "first_ledger_value_selection_rate": rate(first_ledger, total),
        "inactive_pocket_selection_rate": rate(counts.get("inactive_pocket_value", 0), total),
        "stale_pocket_selection_rate": rate(counts.get("stale_pocket_value", 0), total),
        "same_key_different_scenario_rate": rate(same_key_count, base["total_rows"]),
    })
    return base


def context_report(records: list[dict[str, Any]]) -> dict[str, Any]:
    family = "FRESH_CONTEXT_ENTITY_EXTRACTION"
    base = aggregate_family(records, family)
    failures = [row for row in records if row["task_family"] == family and row["pass_fail"] == "fail"]
    counts = Counter(row["classified_wrong_source"] for row in failures)
    collisions = 0
    for row in [r for r in records if r["task_family"] == family]:
        parsed = row["parsed"]
        if not parsed:
            continue
        values = [parsed.get("first_value"), parsed.get("exact_value"), parsed.get("side_value")]
        keys = [parsed.get("first_key"), parsed.get("exact_key"), parsed.get("side_key")]
        if len(values) != len(set(values)) or len(keys) != len(set(keys)):
            collisions += 1
    base.update({
        "exact_anchor_success_rate": base["accuracy"],
        "key_collision_rate": rate(collisions, base["total_rows"]),
        "side_note_value_selection_rate": rate(counts.get("side_note_value", 0), len(failures)),
        "copy_first_match_agreement_rate": base["copy_first_match_agreement_rate"],
        "no_route_agreement_rate": base["no_route_agreement_rate"],
        "first_ledger_value_selection_rate": rate(counts.get("first_ledger_value", 0), len(failures)),
    })
    return base


def pocket_report(records: list[dict[str, Any]]) -> dict[str, Any]:
    family = "FRESH_IRRELEVANT_POCKET_SUPPRESSION"
    base = aggregate_family(records, family)
    failures = [row for row in records if row["task_family"] == family and row["pass_fail"] == "fail"]
    counts = Counter(row["classified_wrong_source"] for row in failures)
    base.update({
        "irrelevant_pocket_selection_rate": rate(counts.get("inactive_pocket_value", 0) + counts.get("stale_pocket_value", 0), len(failures)),
        "inactive_pocket_selection_rate": rate(counts.get("inactive_pocket_value", 0), len(failures)),
        "stale_pocket_selection_rate": rate(counts.get("stale_pocket_value", 0), len(failures)),
        "side_note_value_selection_rate": rate(counts.get("side_note_value", 0), len(failures)),
        "no_route_agreement_rate": base["no_route_agreement_rate"],
        "copy_first_match_agreement_rate": base["copy_first_match_agreement_rate"],
    })
    return base


def template_failure_matrix(records: list[dict[str, Any]]) -> dict[str, Any]:
    matrix: dict[str, Any] = {"schema_version": "template_failure_matrix_v1", "families": {}}
    for family in TARGET_FAMILIES:
        rows = [row for row in records if row["task_family"] == family]
        source_counts = Counter(row["classified_wrong_source"] for row in rows if row["pass_fail"] == "fail")
        matrix["families"][family] = {
            "rows": len(rows),
            "failures": sum(source_counts.values()),
            "source_counts": dict(source_counts),
        }
    return matrix


def wrong_answer_source_matrix(records: list[dict[str, Any]]) -> dict[str, Any]:
    failures = [row for row in records if row["pass_fail"] == "fail"]
    by_family: dict[str, dict[str, int]] = {}
    totals = Counter(row["classified_wrong_source"] for row in failures)
    for family in TARGET_FAMILIES:
        by_family[family] = dict(Counter(
            row["classified_wrong_source"] for row in failures if row["task_family"] == family
        ))
    return {
        "schema_version": "wrong_answer_source_matrix_v1",
        "labels": WRONG_SOURCE_LABELS,
        "total_failed_supported_rows": len(failures),
        "unknown_label_count": totals.get("unknown_label", 0),
        "unknown_label_rate": rate(totals.get("unknown_label", 0), len(failures)),
        "source_counts": dict(totals),
        "by_family": by_family,
    }


def key_collision_report(records: list[dict[str, Any]]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for row in records:
        parsed = row["parsed"]
        if not parsed:
            continue
        family = row["task_family"]
        keys = [value for key, value in parsed.items() if key.endswith("_key")]
        values = [value for key, value in parsed.items() if key.endswith("_value")]
        rows.append({
            "task_family": family,
            "input": row["input"],
            "pass_fail": row["pass_fail"],
            "duplicate_key_count": len(keys) - len(set(keys)),
            "duplicate_value_count": len(values) - len(set(values)),
            "model_output": row["model_output"],
            "expected": row["expected"],
            "classified_wrong_source": row["classified_wrong_source"],
        })
    by_family: dict[str, Any] = {}
    for family in TARGET_FAMILIES:
        subset = [row for row in rows if row["task_family"] == family]
        by_family[family] = {
            "rows": len(subset),
            "rows_with_key_collision": sum(1 for row in subset if row["duplicate_key_count"] > 0),
            "rows_with_value_collision": sum(1 for row in subset if row["duplicate_value_count"] > 0),
            "failed_rows_with_key_collision": sum(1 for row in subset if row["pass_fail"] == "fail" and row["duplicate_key_count"] > 0),
            "failed_rows_with_value_collision": sum(1 for row in subset if row["pass_fail"] == "fail" and row["duplicate_value_count"] > 0),
        }
    return {
        "schema_version": "key_collision_report_v1",
        "by_family": by_family,
        "sample_rows": rows[:50],
    }


def no_route_comparison(records: list[dict[str, Any]], no_route_metrics: dict[str, Any]) -> dict[str, Any]:
    by_family: dict[str, Any] = {}
    for family in TARGET_FAMILIES:
        rows = [row for row in records if row["task_family"] == family]
        failures = [row for row in rows if row["pass_fail"] == "fail"]
        no_route_family = no_route_metrics.get("per_family", {}).get(family, {})
        by_family[family] = {
            "model_rows": len(rows),
            "model_failure_rows": len(failures),
            "model_no_route_agreement_on_failures": rate(sum(1 for row in failures if row["no_route_agreement"]), len(failures)),
            "no_route_accuracy": no_route_family.get("accuracy"),
            "model_accuracy": rate(len(rows) - len(failures), len(rows)),
            "model_minus_no_route_accuracy": None if no_route_family.get("accuracy") is None else rate(len(rows) - len(failures), len(rows)) - float(no_route_family["accuracy"]),
        }
    return {
        "schema_version": "no_route_control_comparison_v1",
        "eval_row_hash": no_route_metrics.get("eval_row_hash"),
        "no_route_control_present": no_route_metrics.get("no_route_control_present"),
        "by_family": by_family,
    }


def short_diagnosis(row: dict[str, Any]) -> str:
    source = row["classified_wrong_source"]
    family = row["task_family"]
    if source == "old_scenario_value":
        return "selected old scenario value instead of active scenario binding"
    if source == "distractor_scenario_value":
        return "selected distractor scenario value instead of active scenario binding"
    if source == "first_ledger_value":
        return "selected first ledger value instead of requested exact anchor"
    if source == "side_note_value":
        return "selected side-note value instead of requested exact anchor"
    if source == "inactive_pocket_value":
        return "selected red/inactive pocket value instead of active pocket value"
    if source == "stale_pocket_value":
        return "selected blue/stale pocket value instead of active pocket value"
    if row["no_route_agreement"]:
        return "model agreed with no-route control on this failure"
    if row["copy_first_match_agreement"]:
        return "model agreed with copy-first baseline on this failure"
    return f"unclassified {family} failure"


def write_human_digest(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    failures = [row for row in records if row["pass_fail"] == "fail"]
    with path.open("w", encoding="utf-8") as handle:
        for row in failures:
            payload = {
                "task_family": row["task_family"],
                "input": row["input"],
                "expected": row["expected"],
                "model_output": row["model_output"],
                "classified_wrong_source": row["classified_wrong_source"],
                "no_route_output": row["no_route_output"],
                "copy_first_match": row["copy_first_match"],
                "short_diagnosis": short_diagnosis(row),
            }
            handle.write(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")


def curriculum_patch() -> dict[str, Any]:
    return {
        "schema_version": "recommended_curriculum_patch_v1",
        "next_milestone": "072_COUNTERFACTUAL_SCENARIO_BINDING_REPAIR",
        "primary_diagnosis": "fresh counterfactual and pocket failures require stronger scenario-state selection and stale pocket suppression, not immediate FineWeb scale-up",
        "required_patches": [
            "active scenario marker strengthening",
            "same key / different scenario training",
            "stale scenario suppression",
            "inactive pocket negative examples",
            "explicit scenario:active / scenario:old / scenario:distractor trace fields",
            "answer-only plus trace-mixed variants",
            "no-route and copy-first controls retained",
            "no FineWeb scale-up as immediate fix",
        ],
        "training_families_for_072": [
            "ACTIVE_SCENARIO_MARKER_BINDING",
            "SAME_KEY_DIFFERENT_SCENARIO_SWITCH",
            "STALE_SCENARIO_SUPPRESSION",
            "DISTRACTOR_SCENARIO_REJECTION",
            "INACTIVE_POCKET_NEGATIVE_ROUTE",
            "ANSWER_ONLY_SCENARIO_BINDING",
            "TRACE_MIXED_SCENARIO_BINDING",
        ],
        "must_keep_controls": [
            "NO_ROUTE_FEATURE_CONTROL",
            "COPY_FIRST_MATCH",
            "COPY_LAST_TOKEN",
            "SHUFFLED_CONTEXT",
            "SHUFFLED_LABELS",
        ],
        "not_recommended_as_immediate_fix": [
            "FineWeb scale-up",
            "full corpus training",
            "open-ended assistant training",
            "perplexity LM training",
        ],
    }


def write_report(out: Path, summary: dict[str, Any]) -> None:
    verdicts = "\n".join(f"- {verdict}" for verdict in summary.get("verdicts", [])) or "- none yet"
    findings = "\n".join(f"- {item}" for item in summary.get("dominant_findings", [])) or "- pending"
    unknown = summary.get("unknown_label_rate")
    unknown_text = "pending" if unknown is None else f"{unknown:.6f}"
    text = f"""# STABLE_LOOP_PHASE_LOCK_071B_REPAIR_OVERFIT_FAILURE_ANALYSIS Report

Status: {summary.get("status", "unknown")}

This is analysis only.

no training
no inference
no checkpoint repair
no checkpoint mutation
no 069/070/071 rerun
no model capability improved
no production training
no open-ended assistant
no language grounding

## Verdicts

{verdicts}

## Unknown Attribution Rate

```text
unknown_label_rate = {unknown_text}
```

## Dominant Findings

{findings}

## Next Curriculum Direction

071B points to `072_COUNTERFACTUAL_SCENARIO_BINDING_REPAIR` if the attribution
is sufficiently classified.
"""
    (out / "report.md").write_text(text, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--upstream-071-root", type=Path, required=True)
    parser.add_argument("--upstream-070-root", type=Path, required=True)
    parser.add_argument("--benchmark-069-root", type=Path, required=True)
    parser.add_argument("--heartbeat-sec", type=int, default=20)
    args = parser.parse_args()

    out = args.out
    init_run(out, args)

    missing, manifest = upstream_manifest(args.upstream_071_root, args.upstream_070_root, args.benchmark_069_root)
    write_json(out / "upstream_071_manifest.json", manifest)
    append_progress(out, "upstream_manifest_written", {"missing_count": len(missing)})
    if missing:
        return fail(out, "UPSTREAM_071_ARTIFACT_MISSING", "required upstream artifacts are missing", {"missing": missing})

    try:
        summary071 = read_json(args.upstream_071_root / "summary.json")
        per_family071 = read_json(args.upstream_071_root / "per_family_metrics.json")
        baseline_metrics = read_json(args.upstream_071_root / "baseline_metrics.json")
        no_route_metrics = read_json(args.upstream_071_root / "no_route_feature_control_metrics.json")
        human_rows = read_jsonl(args.upstream_071_root / "human_readable_samples.jsonl")
        failure_rows = read_jsonl(args.upstream_071_root / "failure_case_samples.jsonl")
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return fail(out, "UPSTREAM_071_ARTIFACT_MISSING", str(exc))

    if not human_rows or not failure_rows:
        return fail(out, "FAILURE_CASE_INPUT_MISSING", "071 human-readable or failure sample rows are missing")

    append_progress(out, "upstream_071_failure_profile_loaded", {
        "human_rows": len(human_rows),
        "failure_rows": len(failure_rows),
    })

    records = row_records(human_rows)
    failure_records = [row for row in records if row["pass_fail"] == "fail"]
    if not failure_records:
        return fail(out, "FAILURE_CASE_INPUT_MISSING", "071 target-family failure rows are missing")

    counter = counterfactual_report(records)
    context = context_report(records)
    pocket = pocket_report(records)
    write_json(out / "counterfactual_source_attribution.json", {"schema_version": "counterfactual_source_attribution_v1", **counter})
    write_json(out / "active_scenario_miss_report.json", {"schema_version": "active_scenario_miss_report_v1", **counter})
    write_json(out / "distractor_scenario_selection_report.json", {"schema_version": "distractor_scenario_selection_report_v1", **counter})
    write_json(out / "context_extraction_source_attribution.json", {"schema_version": "context_extraction_source_attribution_v1", **context})
    write_json(out / "pocket_suppression_source_attribution.json", {"schema_version": "pocket_suppression_source_attribution_v1", **pocket})
    write_json(out / "stale_pocket_selection_report.json", {"schema_version": "stale_pocket_selection_report_v1", **pocket})
    append_progress(out, "source_attribution_completed", {
        "counterfactual_failures": counter["failure_rows"],
        "context_failures": context["failure_rows"],
        "pocket_failures": pocket["failure_rows"],
    })

    cluster = {
        "schema_version": "failure_cluster_report_v1",
        "upstream_071_status": summary071.get("status"),
        "target_families": TARGET_FAMILIES,
        "per_family_071_metrics": {family: per_family071.get(family) for family in TARGET_FAMILIES},
        "families": {
            "FRESH_COUNTERFACTUAL_BINDING": counter,
            "FRESH_CONTEXT_ENTITY_EXTRACTION": context,
            "FRESH_IRRELEVANT_POCKET_SUPPRESSION": pocket,
        },
    }
    write_json(out / "failure_cluster_report.json", cluster)
    write_json(out / "template_failure_matrix.json", template_failure_matrix(records))
    wrong_matrix = wrong_answer_source_matrix(records)
    write_json(out / "wrong_answer_source_matrix.json", wrong_matrix)
    write_json(out / "key_collision_report.json", key_collision_report(records))
    write_json(out / "no_route_control_comparison.json", no_route_comparison(records, no_route_metrics))
    write_json(out / "recommended_curriculum_patch.json", curriculum_patch())
    write_human_digest(out / "human_failure_digest.jsonl", records)
    append_progress(out, "failure_reports_written", {"unknown_label_rate": wrong_matrix["unknown_label_rate"]})

    if counter["failure_rows"] == 0:
        return fail(out, "COUNTERFACTUAL_ANALYSIS_INCOMPLETE", "counterfactual failure rows were not analyzed")
    if context["total_rows"] == 0:
        return fail(out, "CONTEXT_ANALYSIS_INCOMPLETE", "context extraction rows were not analyzed")
    if pocket["total_rows"] == 0:
        return fail(out, "POCKET_ANALYSIS_INCOMPLETE", "pocket suppression rows were not analyzed")
    if wrong_matrix["unknown_label_rate"] > 0.20:
        return fail(out, "WRONG_ANSWER_SOURCE_UNCLASSIFIED_TOO_HIGH", "too many failed rows have unknown attribution", {
            "unknown_label_rate": wrong_matrix["unknown_label_rate"],
            "unknown_label_count": wrong_matrix["unknown_label_count"],
            "total_failed_supported_rows": wrong_matrix["total_failed_supported_rows"],
        })

    dominant_findings = [
        f"counterfactual failures mostly select {max(counter['wrong_answer_source_counts'], key=counter['wrong_answer_source_counts'].get) if counter['wrong_answer_source_counts'] else 'none'}",
        f"context extraction first-ledger selection rate on failures = {context['first_ledger_value_selection_rate']:.3f}",
        f"pocket irrelevant/stale selection rate on failures = {pocket['irrelevant_pocket_selection_rate']:.3f}",
        "072 should target scenario-state binding and stale/inactive pocket suppression before FineWeb scale-up",
    ]

    final_summary = {
        "schema_version": "repair_overfit_failure_analysis_summary_v1",
        "status": "passed",
        "analysis_only": True,
        "training_performed": False,
        "train_step_count": 0,
        "checkpoint_repaired": False,
        "checkpoint_mutated": False,
        "model_capability_improved": False,
        "inference_performed": False,
        "reran_069_070_071": False,
        "unknown_label_rate": wrong_matrix["unknown_label_rate"],
        "counterfactual": counter,
        "context_extraction": context,
        "pocket_suppression": pocket,
        "baseline_metric_keys": sorted(baseline_metrics.keys()),
        "recommended_next_milestone": "072_COUNTERFACTUAL_SCENARIO_BINDING_REPAIR",
        "dominant_findings": dominant_findings,
        "verdicts": POSITIVE_VERDICTS,
    }
    write_json(out / "summary.json", final_summary)
    write_report(out, final_summary)
    append_progress(out, "done", {"status": "passed"})
    print(json.dumps(final_summary, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        raise

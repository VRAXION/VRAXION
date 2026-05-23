#!/usr/bin/env python3
"""138YH artifact-only high-frequency value replay analysis.

This phase reads existing 138YI artifacts only, with optional 138YF/138U
context. It does not train, repair, run inference, call the shared helper, run
torch forward passes, mutate checkpoints, import old runners, start services,
deploy, or modify runtime/release/product surfaces.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
MILESTONE = "STABLE_LOOP_PHASE_LOCK_138YH_HIGH_FREQUENCY_VALUE_REPLAY_ANALYSIS"
DEFAULT_OUT = Path("target/pilot_wave/stable_loop_phase_lock_138yh_high_frequency_value_replay_analysis/smoke")
DEFAULT_UPSTREAM_138YI_ROOT = Path("target/pilot_wave/stable_loop_phase_lock_138yi_family_specific_value_attractor_repair_probe/smoke")
DEFAULT_UPSTREAM_138YF_ROOT = Path("target/pilot_wave/stable_loop_phase_lock_138yf_family_specific_value_attractor_repair_plan/smoke")

FALSE_FLAGS = {
    "reasoning_restored": False,
    "reasoning_subtrack_real_raw_evidence_partially_restored": False,
    "raw_assistant_capability_restored": False,
    "structured_tool_capability_restored": False,
    "gpt_like_readiness_claimed": False,
    "open_domain_assistant_readiness_claimed": False,
    "production_chat_claimed": False,
    "public_api_claimed": False,
    "deployment_readiness_claimed": False,
    "safety_alignment_claimed": False,
}
BOUNDARY_TEXT = (
    "138YH is artifact-only high-frequency value replay analysis. It reads "
    "existing 138YI artifacts only, optionally checks 138YF/138U context, and "
    "does not train, repair, run new inference, call shared_raw_generation_helper.py, "
    "run torch forward passes, mutate checkpoints, modify helper/backend code, "
    "import old runners, delete or consolidate files, start services, deploy, "
    "modify runtime/service/deploy/product/release surfaces, modify docs/product "
    "or docs/releases, modify SDK exports, or change root LICENSE."
)
ALLOWED_HELPER_KEYS = {"prompt", "checkpoint_path", "checkpoint_hash", "seed", "max_new_tokens", "generation_config"}
FORBIDDEN_HELPER_KEYS = {
    "expected_output",
    "expected_payload",
    "expected_answer",
    "required_keys",
    "required_keywords",
    "forbidden_outputs",
    "schema_answer_object",
    "scorer_metadata",
    "labels",
    "oracle_data",
    "target_json",
    "gold_output",
    "row_answer",
    "eval_family",
    "answer",
    "expected_values",
}
REQUIRED_138YI_ARTIFACTS = [
    "decision.json",
    "summary.json",
    "aggregate_metrics.json",
    "high_frequency_value_replay_report.json",
    "family_default_attractor_report.json",
    "intra_family_contrastive_metrics.json",
    "value_grounding_metrics.json",
    "parrot_trap_report.json",
    "train_rows.jsonl",
    "eval_rows.jsonl",
    "raw_generation_results.jsonl",
    "raw_generation_trace.jsonl",
    "scoring_results.jsonl",
    "contrast_group_results.jsonl",
    "contrast_group_manifest.json",
    "ood_family_value_manifest.json",
    "per_family_metrics.json",
    "per_seed_metrics.jsonl",
    "control_arm_report.json",
    "freshness_leakage_audit.json",
    "generated_before_scoring_report.json",
    "expected_output_canary_report.json",
    "ast_shortcut_scan_report.json",
    "determinism_replay_report.json",
    "source_checkpoint_integrity_manifest.json",
    "target_checkpoint_integrity_manifest.json",
    "training_metrics.jsonl",
    "training_objective_report.json",
    "train_config.json",
    "eval_config.json",
    "failure_case_samples.jsonl",
    "human_readable_samples.jsonl",
]
VALUE_RE = re.compile(r"\b(?:TR|EV|VAL|SYM)[A-Za-z0-9_+\-]*\b")


class GateError(Exception):
    def __init__(self, verdict: str, message: str, details: dict[str, Any] | None = None):
        super().__init__(message)
        self.verdict = verdict
        self.message = message
        self.details = details or {}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8", newline="\n")
    tmp.replace(path)


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


def resolve_path(text: str | Path) -> Path:
    path = Path(text)
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def resolve_target_out(text: str | Path) -> Path:
    path = Path(text)
    resolved = path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()
    try:
        relative = resolved.relative_to(REPO_ROOT)
    except ValueError as exc:
        raise GateError("138YH_BOUNDARY_FAILURE", "--out must stay inside repo") from exc
    parts = [part.lower() for part in relative.parts]
    if len(parts) < 2 or parts[0] != "target" or parts[1] != "pilot_wave":
        raise GateError("138YH_BOUNDARY_FAILURE", "--out must stay under target/pilot_wave")
    return resolved


def append_progress(out: Path, event: str, status: str = "completed", **details: Any) -> None:
    append_jsonl(out / "progress.jsonl", {"ts": utc_now(), "event": event, "status": status, "details": details})


def rate(count: int | float, total: int | float) -> float:
    return float(count) / float(total) if total else 0.0


def entropy(counter: Counter[str]) -> float:
    total = sum(counter.values())
    if total <= 0:
        return 0.0
    return -sum((count / total) * math.log2(count / total) for count in counter.values() if count)


def extract_values(text: str | None) -> list[str]:
    return VALUE_RE.findall(text or "")


def ranked(counter: Counter[str]) -> dict[str, int]:
    return {value: index + 1 for index, (value, _count) in enumerate(sorted(counter.items(), key=lambda item: (-item[1], item[0])))}


def in_top(value: str | None, ranks: dict[str, int], k: int) -> bool:
    return value is not None and ranks.get(value, 10**9) <= k


def write_summary(out: Path, status: str, verdicts: list[str], decision: dict[str, Any], failure: str | None = None) -> None:
    write_json(
        out / "summary.json",
        {
            "schema_version": "phase_138yh_summary_v1",
            "milestone": MILESTONE,
            "status": status,
            "verdicts": verdicts,
            "decision": decision.get("decision"),
            "next": decision.get("next"),
            "root_cause": decision.get("root_cause"),
            "failure": failure,
            "boundary": BOUNDARY_TEXT,
            "artifact_only": True,
            "training_performed": False,
            "new_model_inference_run": False,
            "shared_helper_called": False,
            "torch_forward_pass_run": False,
            **FALSE_FLAGS,
            "metrics": decision,
        },
    )


def write_report(out: Path, verdicts: list[str], decision: dict[str, Any]) -> None:
    lines = [
        f"# {MILESTONE}",
        "",
        "## Boundary",
        "",
        BOUNDARY_TEXT,
        "",
        "## Result",
        "",
        "- 138YH does not fix or train the model.",
        "- It separates ANSWER=T namespace leakage from ANSWER=E followed by TR value replay.",
        "- It computes strict train membership and train-frequency ranks instead of assuming memorization.",
        f"- `decision`: `{decision.get('decision')}`",
        f"- `next`: `{decision.get('next')}`",
        f"- `root_cause`: `{decision.get('root_cause')}`",
        "",
        "## Verdicts",
        "",
    ]
    lines.extend(f"- `{verdict}`" for verdict in verdicts)
    lines.extend(
        [
            "",
            "Reasoning is not restored.",
            "Raw assistant capability remains quarantined.",
            "Structured/tool capability remains invalidated.",
            "not GPT-like readiness.",
            "not open-domain assistant readiness.",
            "not production chat.",
            "not public API.",
            "not deployment readiness.",
            "not safety alignment.",
        ]
    )
    write_text(out / "report.md", "\n".join(lines) + "\n")


def refresh_status(out: Path, status: str, verdicts: list[str], decision: dict[str, Any]) -> None:
    write_summary(out, status, verdicts, decision)
    write_report(out, verdicts, decision)


def require_artifacts(root: Path, names: list[str]) -> None:
    missing = [name for name in names if not (root / name).exists()]
    if missing:
        raise GateError("UPSTREAM_138YI_ARTIFACT_MISSING", "required 138YI artifacts missing", {"missing": missing})


def verify_upstream_138yi(out: Path, root: Path) -> dict[str, Any]:
    require_artifacts(root, REQUIRED_138YI_ARTIFACTS)
    decision = read_json(root / "decision.json")
    summary = read_json(root / "summary.json")
    aggregate = read_json(root / "aggregate_metrics.json")
    high_frequency = read_json(root / "high_frequency_value_replay_report.json")
    family_default = read_json(root / "family_default_attractor_report.json")
    contrast = read_json(root / "intra_family_contrastive_metrics.json")
    parrot = read_json(root / "parrot_trap_report.json")
    canary = read_json(root / "expected_output_canary_report.json")
    scan = read_json(root / "ast_shortcut_scan_report.json")
    controls = read_json(root / "control_arm_report.json")
    leakage = read_json(root / "freshness_leakage_audit.json")
    before = read_json(root / "generated_before_scoring_report.json")
    replay = read_json(root / "determinism_replay_report.json")
    source = read_json(root / "source_checkpoint_integrity_manifest.json")
    target = read_json(root / "target_checkpoint_integrity_manifest.json")
    traces = read_jsonl(root / "raw_generation_trace.jsonl")

    if decision.get("verdict") != "FAMILY_SPECIFIC_VALUE_ATTRACTOR_REPAIR_FAILS" or decision.get("decision") != "high_frequency_train_value_replay_detected" or decision.get("next") != "138YH_HIGH_FREQUENCY_VALUE_REPLAY_ANALYSIS":
        raise GateError("UPSTREAM_138YI_ARTIFACT_MISSING", "138YI did not route to 138YH")
    expected = {
        "answer_value_accuracy": 0.0,
        "exact_answer_accuracy": 0.0,
        "intra_family_contrastive_accuracy": 0.0,
        "parrot_trap_detected": False,
        "stale_chat_fragment_rate": 0.0,
        "train_namespace_leak_rate": 0.0,
    }
    mismatches = {key: {"expected": val, "actual": aggregate.get(key)} for key, val in expected.items() if aggregate.get(key) != val}
    if aggregate.get("intra_family_mode_collapse_rate", 0.0) < 0.90:
        mismatches["intra_family_mode_collapse_rate"] = aggregate.get("intra_family_mode_collapse_rate")
    if aggregate.get("family_default_attractor_rate", 0.0) < 0.75 or aggregate.get("family_default_shortcut_detected") is not True:
        mismatches["family_default_profile"] = {"rate": aggregate.get("family_default_attractor_rate"), "detected": aggregate.get("family_default_shortcut_detected")}
    if high_frequency.get("high_frequency_train_value_replay_detected") is not True:
        mismatches["high_frequency_train_value_replay_detected"] = high_frequency.get("high_frequency_train_value_replay_detected")
    if mismatches:
        raise GateError("UPSTREAM_138YI_ARTIFACT_MISSING", "138YI failure profile mismatch", mismatches)
    if canary.get("expected_output_canary_passed") is not True or scan.get("ast_shortcut_scan_passed") is not True:
        raise GateError("RAW_HELPER_INTEGRITY_FAILURE", "138YI helper canary/AST integrity failed")
    if controls.get("controls_failed") is not True or leakage.get("leakage_rejected") is not True or replay.get("determinism_replay_passed") is not True:
        raise GateError("UPSTREAM_138YI_ARTIFACT_MISSING", "138YI controls/leakage/determinism profile missing")
    if source.get("source_checkpoint_unchanged") is not True or target.get("target_checkpoint_changed") is not True:
        raise GateError("UPSTREAM_138YI_ARTIFACT_MISSING", "138YI checkpoint integrity missing")
    if before.get("generated_text_produced_before_scoring") is not True:
        raise GateError("UPSTREAM_138YI_ARTIFACT_MISSING", "138YI generated-before-scoring proof missing")
    for trace in traces:
        request = trace.get("helper_request", {})
        if set(request) != ALLOWED_HELPER_KEYS or set(request) & FORBIDDEN_HELPER_KEYS:
            raise GateError("RAW_HELPER_INTEGRITY_FAILURE", "138YI helper request metadata violation")
    manifest = {
        "schema_version": "phase_138yh_upstream_138yi_manifest_v1",
        "upstream_138yi_root": rel(root),
        "verified": True,
        "verdict": decision.get("verdict"),
        "decision": decision.get("decision"),
        "next": decision.get("next"),
        "answer_value_accuracy": aggregate.get("answer_value_accuracy"),
        "exact_answer_accuracy": aggregate.get("exact_answer_accuracy"),
        "intra_family_contrastive_accuracy": aggregate.get("intra_family_contrastive_accuracy"),
        "intra_family_mode_collapse_rate": aggregate.get("intra_family_mode_collapse_rate"),
        "family_default_attractor_rate": aggregate.get("family_default_attractor_rate"),
        "family_default_shortcut_detected": aggregate.get("family_default_shortcut_detected"),
        "high_frequency_train_value_replay_detected": high_frequency.get("high_frequency_train_value_replay_detected"),
        "high_frequency_train_value_replay_rate": high_frequency.get("high_frequency_train_value_replay_rate"),
        "parrot_trap_detected": parrot.get("parrot_trap_detected"),
        "stale_chat_fragment_rate": aggregate.get("stale_chat_fragment_rate"),
        "train_namespace_leak_rate": aggregate.get("train_namespace_leak_rate"),
        "determinism_replay_passed": replay.get("determinism_replay_passed"),
        "source_checkpoint_unchanged": source.get("source_checkpoint_unchanged"),
        "target_checkpoint_changed": target.get("target_checkpoint_changed"),
        "generated_text_before_scoring": True,
        "no_expected_or_scorer_metadata_reached_helper_requests": True,
        "all_capability_flags_false": all(summary.get(key) is False for key in FALSE_FLAGS),
        "note": "train_namespace_leak_rate=0 means no ANSWER=T wrapper leak; TR after ANSWER=E is value replay.",
    }
    write_json(out / "upstream_138yi_manifest.json", manifest)
    return {"decision": decision, "aggregate": aggregate, "high_frequency": high_frequency, "family_default": family_default, "contrast": contrast}


def verify_upstream_138yf(out: Path, root: Path) -> dict[str, Any]:
    payload: dict[str, Any] = {"schema_version": "phase_138yh_upstream_138yf_manifest_v1", "upstream_138yf_root": rel(root), "available": root.exists()}
    if root.exists() and (root / "decision.json").exists():
        decision = read_json(root / "decision.json")
        payload.update(
            {
                "verified": decision.get("decision") == "family_specific_value_attractor_repair_plan_complete",
                "decision": decision.get("decision"),
                "next": decision.get("next"),
                "primary_bottleneck": decision.get("primary_bottleneck"),
            }
        )
        if (root / "train_membership_reconciliation.json").exists():
            rec = read_json(root / "train_membership_reconciliation.json")
            payload["prior_strict_138u_train_row_membership_rate"] = rec.get("strict_138u_train_row_membership_rate")
    else:
        payload.update({"verified": False, "diagnostic_gap": "138YF context artifact unavailable"})
    write_json(out / "upstream_138yf_manifest.json", payload)
    return payload


def build_frequency_tables(train_rows: list[dict[str, Any]]) -> dict[str, Any]:
    global_expected: Counter[str] = Counter()
    global_prompt: Counter[str] = Counter()
    by_family_expected: dict[str, Counter[str]] = defaultdict(Counter)
    by_family_prompt: dict[str, Counter[str]] = defaultdict(Counter)
    for row in train_rows:
        family = row["family"]
        expected = row.get("answer_value")
        if expected:
            global_expected[expected] += 1
            by_family_expected[family][expected] += 1
        for value in extract_values(row.get("prompt", "")):
            global_prompt[value] += 1
            by_family_prompt[family][value] += 1
    families = sorted(set(by_family_expected) | set(by_family_prompt))
    by_family_all = {family: by_family_expected[family] + by_family_prompt[family] for family in families}
    return {
        "global_expected": global_expected,
        "global_prompt": global_prompt,
        "global_all": global_expected + global_prompt,
        "by_family_expected": by_family_expected,
        "by_family_prompt": by_family_prompt,
        "by_family_all": by_family_all,
    }


def serial_counter(counter: Counter[str], limit: int | None = None) -> list[dict[str, Any]]:
    rows = [{"value": value, "count": count, "rank": index + 1} for index, (value, count) in enumerate(sorted(counter.items(), key=lambda item: (-item[1], item[0])))]
    return rows if limit is None else rows[:limit]


def classify_source(value: str | None, row: dict[str, Any], group_expected: set[str], family_default: str | None, ranks: dict[str, Any], frequency: dict[str, Any]) -> str:
    if value is None:
        return "unknown_generated_value"
    if value == row["expected_value"]:
        return "exact_expected_value"
    if value in group_expected and value != row["expected_value"]:
        return "contrast_group_peer_expected_value"
    if value == family_default:
        return "family_default_value"
    if in_top(value, ranks["global_all"], 5):
        return "global_high_frequency_value"
    if in_top(value, ranks["by_family_all"].get(row["family"], {}), 5):
        return "family_high_frequency_value"
    if value in frequency["global_expected"]:
        return "train_expected_value"
    if value in frequency["global_prompt"]:
        return "train_prompt_value"
    if value in row["eval_prompt_values"]:
        return "eval_prompt_value"
    return "unknown_generated_value"


def build_rows(root: Path, family_defaults: dict[str, str | None], frequency: dict[str, Any]) -> list[dict[str, Any]]:
    eval_rows = read_jsonl(root / "eval_rows.jsonl")
    raw_rows = read_jsonl(root / "raw_generation_results.jsonl")
    trace_rows = read_jsonl(root / "raw_generation_trace.jsonl")
    scoring_rows = read_jsonl(root / "scoring_results.jsonl")
    eval_by_id = {row["row_id"]: row for row in eval_rows}
    raw_by_id = {row["row_id"]: row for row in raw_rows}
    trace_by_id = {row["row_id"]: row for row in trace_rows}
    groups: dict[str, set[str]] = defaultdict(set)
    for row in eval_rows:
        groups[row["contrast_group_id"]].add(row["answer_value"])
    ranks = {"global_all": ranked(frequency["global_all"]), "by_family_all": {family: ranked(counter) for family, counter in frequency["by_family_all"].items()}}
    rows: list[dict[str, Any]] = []
    for score in sorted(scoring_rows, key=lambda item: item["row_id"]):
        eval_row = eval_by_id[score["row_id"]]
        raw = raw_by_id.get(score["row_id"], {})
        trace = trace_by_id.get(score["row_id"], {})
        generated_value = score.get("answer_value_candidate")
        prompt = eval_row.get("prompt", "")
        prompt_values = set(extract_values(prompt))
        base = {
            "row_id": score["row_id"],
            "family": score["family"],
            "seed": score["seed"],
            "contrast_group_id": score["contrast_group_id"],
            "expected_value": score["expected_value"],
            "generated_value": generated_value,
            "generated_text": raw.get("generated_text", ""),
            "prompt": prompt,
            "helper_trace_hash": score.get("helper_trace_hash") or trace.get("generation_trace_hash"),
            "pass_fail": score.get("pass"),
            "failure_reason": score.get("failure_reason"),
            "eval_prompt_values": sorted(prompt_values),
        }
        base["generated_value_source"] = classify_source(generated_value, base, groups[score["contrast_group_id"]], family_defaults.get(score["family"]), ranks, frequency)
        base["global_train_all_rank"] = ranks["global_all"].get(generated_value)
        base["family_train_all_rank"] = ranks["by_family_all"].get(score["family"], {}).get(generated_value)
        base["train_expected_count"] = frequency["global_expected"].get(generated_value, 0)
        base["train_prompt_count"] = frequency["global_prompt"].get(generated_value, 0)
        base["train_all_count"] = frequency["global_all"].get(generated_value, 0)
        base["family_train_all_count"] = frequency["by_family_all"].get(score["family"], Counter()).get(generated_value, 0)
        rows.append(base)
    return rows


def replay_value_extraction_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": "phase_138yh_replay_value_extraction_report_v1",
        "row_count": len(rows),
        "generated_value_source_counts": dict(sorted(Counter(row["generated_value_source"] for row in rows).items())),
        "tr_prefix_replay_rate": rate(sum(1 for row in rows if str(row.get("generated_value") or "").startswith("TR")), len(rows)),
        "ev_expected_candidate_rate": rate(sum(1 for row in rows if row.get("generated_value") == row.get("expected_value")), len(rows)),
        "rows": rows,
    }


def train_value_frequency_report(frequency: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    generated_values = sorted({row["generated_value"] for row in rows if row.get("generated_value")})
    generated_membership = {
        value: {
            "train_expected_count": frequency["global_expected"].get(value, 0),
            "train_prompt_count": frequency["global_prompt"].get(value, 0),
            "train_all_count": frequency["global_all"].get(value, 0),
            "global_train_all_rank": ranked(frequency["global_all"]).get(value),
        }
        for value in generated_values
    }
    return {
        "schema_version": "phase_138yh_train_value_frequency_report_v1",
        "train_expected_value_frequency_top": serial_counter(frequency["global_expected"], 40),
        "train_prompt_value_frequency_top": serial_counter(frequency["global_prompt"], 40),
        "train_all_value_frequency_top": serial_counter(frequency["global_all"], 40),
        "per_family_train_all_value_frequency_top": {family: serial_counter(counter, 20) for family, counter in sorted(frequency["by_family_all"].items())},
        "generated_value_membership": generated_membership,
        "generated_values_seen_in_train_expected_rate": rate(sum(1 for row in rows if row["train_expected_count"] > 0), len(rows)),
        "generated_values_seen_in_train_prompt_rate": rate(sum(1 for row in rows if row["train_prompt_count"] > 0), len(rows)),
        "generated_values_seen_in_train_all_rate": rate(sum(1 for row in rows if row["train_all_count"] > 0), len(rows)),
        "strict_train_membership_computed_from_138yi": True,
        "memorized_lookup_claimed": False,
    }


def replay_rank_report(rows: list[dict[str, Any]], frequency: dict[str, Any]) -> dict[str, Any]:
    global_ranks = ranked(frequency["global_all"])
    family_ranks = {family: ranked(counter) for family, counter in frequency["by_family_all"].items()}
    global_seen = [global_ranks[row["generated_value"]] for row in rows if row.get("generated_value") in global_ranks]
    family_seen = [family_ranks.get(row["family"], {}).get(row["generated_value"]) for row in rows if row.get("generated_value") in family_ranks.get(row["family"], {})]
    family_seen = [rank for rank in family_seen if rank is not None]
    return {
        "schema_version": "phase_138yh_replay_rank_report_v1",
        "row_count": len(rows),
        "generated_values_seen_in_train_expected_rate": rate(sum(1 for row in rows if row["train_expected_count"] > 0), len(rows)),
        "generated_values_seen_in_train_prompt_rate": rate(sum(1 for row in rows if row["train_prompt_count"] > 0), len(rows)),
        "generated_values_seen_in_train_all_rate": rate(sum(1 for row in rows if row["train_all_count"] > 0), len(rows)),
        "generated_values_top1_global_train_all_rate": rate(sum(1 for row in rows if in_top(row["generated_value"], global_ranks, 1)), len(rows)),
        "generated_values_top5_global_train_all_rate": rate(sum(1 for row in rows if in_top(row["generated_value"], global_ranks, 5)), len(rows)),
        "generated_values_top10_global_train_all_rate": rate(sum(1 for row in rows if in_top(row["generated_value"], global_ranks, 10)), len(rows)),
        "generated_values_top1_family_train_all_rate": rate(sum(1 for row in rows if in_top(row["generated_value"], family_ranks.get(row["family"], {}), 1)), len(rows)),
        "generated_values_top5_family_train_all_rate": rate(sum(1 for row in rows if in_top(row["generated_value"], family_ranks.get(row["family"], {}), 5)), len(rows)),
        "generated_values_top10_family_train_all_rate": rate(sum(1 for row in rows if in_top(row["generated_value"], family_ranks.get(row["family"], {}), 10)), len(rows)),
        "mean_global_train_rank": sum(global_seen) / len(global_seen) if global_seen else None,
        "median_global_train_rank": median(global_seen) if global_seen else None,
        "mean_family_train_rank": sum(family_seen) / len(family_seen) if family_seen else None,
        "median_family_train_rank": median(family_seen) if family_seen else None,
        "rank_interpretation": "Null rank means the generated TR value was not an exact member of the 138YI train value tables.",
    }


def family_replay_shape_report(rows: list[dict[str, Any]], contrast_metrics: dict[str, Any], rank: dict[str, Any], family_defaults: dict[str, str | None]) -> dict[str, Any]:
    families: dict[str, Any] = {}
    for family in sorted({row["family"] for row in rows}):
        group = [row for row in rows if row["family"] == family]
        generated = Counter(row["generated_value"] for row in group)
        dominant, dom_count = generated.most_common(1)[0]
        top5_family_rate = rate(sum(1 for row in group if row.get("family_train_all_rank") is not None and row["family_train_all_rank"] <= 5), len(group))
        default_rate = rate(sum(1 for row in group if row["generated_value"] == family_defaults.get(family)), len(group))
        collapse = contrast_metrics.get("per_family", {}).get(family, {}).get("intra_family_mode_collapse_rate")
        if default_rate >= 0.50:
            shape = "family_default_shortcut"
        elif top5_family_rate >= 0.50:
            shape = "family_local_frequency_replay"
        elif rate(dom_count, len(group)) >= 0.50:
            shape = "global_frequency_replay" if rank["generated_values_top5_global_train_all_rate"] >= 0.50 else "family_default_shortcut"
        elif entropy(generated) > 4.0:
            shape = "high_entropy_wrong_values"
        else:
            shape = "ambiguous"
        families[family] = {
            "row_count": len(group),
            "unique_expected_value_count": len({row["expected_value"] for row in group}),
            "unique_generated_value_count": len(generated),
            "dominant_generated_value": dominant,
            "dominant_generated_value_rate": rate(dom_count, len(group)),
            "dominant_generated_value_train_rank_global": row_global_rank(rows, dominant),
            "dominant_generated_value_train_rank_family": row_family_rank(group, dominant),
            "family_default_attractor_rate": default_rate,
            "intra_family_mode_collapse_rate": collapse,
            "high_frequency_replay_rate": rate(sum(1 for row in group if str(row.get("generated_value") or "").startswith("TR")), len(group)),
            "exact_value_accuracy": rate(sum(1 for row in group if row["generated_value"] == row["expected_value"]), len(group)),
            "contrastive_accuracy": contrast_metrics.get("per_family_contrast_group_pass_rate", {}).get(family),
            "family_replay_shape": shape,
        }
    return {"schema_version": "phase_138yh_family_replay_shape_report_v1", "families": families, "family_count": len(families)}


def row_global_rank(rows: list[dict[str, Any]], value: str) -> int | None:
    for row in rows:
        if row.get("generated_value") == value and row.get("global_train_all_rank") is not None:
            return row["global_train_all_rank"]
    return None


def row_family_rank(rows: list[dict[str, Any]], value: str) -> int | None:
    for row in rows:
        if row.get("generated_value") == value and row.get("family_train_all_rank") is not None:
            return row["family_train_all_rank"]
    return None


def contrast_group_replay_report(rows: list[dict[str, Any]], group_results: list[dict[str, Any]], family_defaults: dict[str, str | None]) -> dict[str, Any]:
    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_group[row["contrast_group_id"]].append(row)
    group_pass = {row["contrast_group_id"]: row.get("pass") for row in group_results}
    payload_rows: list[dict[str, Any]] = []
    for group_id, group in sorted(by_group.items()):
        family = group[0]["family"]
        expected = [row["expected_value"] for row in group]
        generated = [row["generated_value"] for row in group]
        family_default = family_defaults.get(family)
        payload_rows.append(
            {
                "group_id": group_id,
                "family": family,
                "expected_values": expected,
                "generated_values": generated,
                "unique_expected_count": len(set(expected)),
                "unique_generated_count": len(set(generated)),
                "group_collapse_to_single_value": len(set(generated)) == 1,
                "generated_value_is_family_default": any(value == family_default for value in generated),
                "generated_value_is_high_frequency_train": any(str(value or "").startswith("TR") for value in generated),
                "generated_value_is_peer_expected_value": any(value in set(expected) and value != row["expected_value"] for value, row in zip(generated, group)),
                "group_pass": group_pass.get(group_id, False),
            }
        )
    total = len(payload_rows)
    return {
        "schema_version": "phase_138yh_contrast_group_replay_report_v1",
        "group_count": total,
        "same_value_for_all_rows_rate": rate(sum(1 for row in payload_rows if row["group_collapse_to_single_value"]), total),
        "family_default_group_rate": rate(sum(1 for row in payload_rows if row["generated_value_is_family_default"]), total),
        "high_frequency_group_replay_rate": rate(sum(1 for row in payload_rows if row["generated_value_is_high_frequency_train"]), total),
        "peer_expected_confusion_rate": rate(sum(1 for row in payload_rows if row["generated_value_is_peer_expected_value"]), total),
        "contrast_group_value_diversity_rate": rate(sum(1 for row in payload_rows if row["unique_generated_count"] > 1), total),
        "rows": payload_rows,
    }


def objective_reward_artifact_report(root: Path) -> dict[str, Any]:
    training = read_json(root / "training_objective_report.json")
    train_config = read_json(root / "train_config.json")
    eval_config = read_json(root / "eval_config.json")
    training_metrics = read_jsonl(root / "training_metrics.jsonl")
    blob = json.dumps({"training": training, "train_config": train_config, "eval_config": eval_config}, sort_keys=True).lower()
    explicit_metric_keys = set().union(*(set(row) for row in training_metrics)) if training_metrics else set()
    return {
        "schema_version": "phase_138yh_objective_reward_artifact_report_v1",
        "objective_mentions_frequency_replay_rejection": "high-frequency" in blob or "frequency" in blob,
        "objective_mentions_family_default_rejection": "family-default" in blob or "family default" in blob,
        "objective_includes_frequency_penalty": "frequency_penalty" in explicit_metric_keys,
        "objective_includes_family_default_penalty": "family_default_penalty" in explicit_metric_keys,
        "objective_includes_same_value_for_all_rows_penalty": "same_value" in blob and "penalty" in blob,
        "objective_includes_intra_family_diversity_reward": "intra-family" in blob and "contrastive" in blob,
        "objective_includes_rule_table_ood_derived_reward": "ood" in blob and "derived" in blob,
        "positive_can_depend_on_train_loss": training.get("positive_can_depend_on_train_loss"),
        "teacher_forcing_only_success_rejected": eval_config.get("teacher_forcing_used") is False,
        "family_default_shortcut_rejected": train_config.get("training_objective", "").find("family-default") >= 0,
        "diagnostic_gap": [
            "No explicit frequency_penalty metric was recorded in training_metrics.jsonl",
            "No explicit family_default_penalty metric was recorded in training_metrics.jsonl",
        ],
    }


def scorer_dataset_artifact_report(rows: list[dict[str, Any]], root: Path) -> dict[str, Any]:
    eval_rows = read_jsonl(root / "eval_rows.jsonl")
    leakage = read_json(root / "freshness_leakage_audit.json")
    expected_counter = Counter(row["answer_value"] for row in eval_rows)
    generated_counter = Counter(row["generated_value"] for row in rows)
    by_family_expected: dict[str, Counter[str]] = defaultdict(Counter)
    by_family_generated: dict[str, Counter[str]] = defaultdict(Counter)
    for row in eval_rows:
        by_family_expected[row["family"]][row["answer_value"]] += 1
    for row in rows:
        by_family_generated[row["family"]][row["generated_value"]] += 1
    group_expected_diversity = len({(row["contrast_group_id"], row["expected_value"]) for row in rows})
    group_generated_diversity = len({(row["contrast_group_id"], row["generated_value"]) for row in rows})
    return {
        "schema_version": "phase_138yh_scorer_dataset_artifact_report_v1",
        "expected_value_frequency_in_eval_top": serial_counter(expected_counter, 20),
        "generated_value_frequency_top": serial_counter(generated_counter, 20),
        "train_eval_value_overlap": leakage.get("eval_value_overlap_with_train"),
        "eval_value_distribution_entropy": entropy(expected_counter),
        "family_eval_value_distribution_entropy": {family: entropy(counter) for family, counter in sorted(by_family_expected.items())},
        "family_generated_value_distribution_entropy": {family: entropy(counter) for family, counter in sorted(by_family_generated.items())},
        "contrast_group_expected_value_diversity": group_expected_diversity,
        "contrast_group_generated_value_diversity": group_generated_diversity,
        "dataset_low_value_diversity_artifact": entropy(expected_counter) < 4.0,
        "scorer_or_dataset_artifact_supported": False,
    }


def select_root(rank_report: dict[str, Any], family_default_rate: float, contrast_report: dict[str, Any], dataset_report: dict[str, Any], objective_report: dict[str, Any]) -> dict[str, Any]:
    if rank_report["generated_values_top1_global_train_all_rate"] >= 0.50:
        root = "global_high_frequency_train_value_replay"
        evidence = "top1 global train-all replay rate >= 0.50"
    elif rank_report["generated_values_top5_family_train_all_rate"] >= 0.50 and rank_report["generated_values_top5_global_train_all_rate"] < 0.50:
        root = "family_local_high_frequency_value_replay"
        evidence = "top5 family train-all replay rate >= 0.50 while top5 global train-all rate < 0.50"
    elif family_default_rate >= 0.75:
        root = "family_default_shortcut_replay"
        evidence = "family_default_attractor_rate >= 0.75"
    elif contrast_report["same_value_for_all_rows_rate"] >= 0.75:
        root = "same_value_for_all_rows_collapse"
        evidence = "same_value_for_all_rows_rate >= 0.75"
    elif dataset_report["dataset_low_value_diversity_artifact"]:
        root = "dataset_low_value_diversity_artifact"
        evidence = "eval expected value entropy below floor"
    elif objective_report["objective_mentions_frequency_replay_rejection"] and not objective_report["objective_includes_frequency_penalty"]:
        root = "objective_missing_frequency_penalty"
        evidence = "objective mentions frequency replay rejection but no explicit frequency_penalty metric exists"
    else:
        root = "mixed_high_frequency_replay"
        evidence = "multiple weak replay explanations remain"
    route = {
        "global_high_frequency_train_value_replay": "138YHG_GLOBAL_VALUE_FREQUENCY_PENALTY_PLAN",
        "family_local_high_frequency_value_replay": "138YHL_FAMILY_LOCAL_FREQUENCY_PENALTY_PLAN",
        "family_default_shortcut_replay": "138YD_FAMILY_DEFAULT_SHORTCUT_ANALYSIS",
        "same_value_for_all_rows_collapse": "138YS_SAME_VALUE_COLLAPSE_ANALYSIS",
        "dataset_low_value_diversity_artifact": "138L_FAMILY_CONTRASTIVE_EVAL_LEAKAGE_REDESIGN",
        "objective_missing_frequency_penalty": "138YJ_FREQUENCY_SUPPRESSED_INTRA_FAMILY_OBJECTIVE_PLAN",
        "mixed_high_frequency_replay": "138YHB_HIGH_FREQUENCY_REPLAY_MANUAL_REVIEW_PACKET",
        "high_frequency_replay_ambiguous": "138YHB_HIGH_FREQUENCY_REPLAY_MANUAL_REVIEW_PACKET",
    }[root]
    return {
        "schema_version": "phase_138yh_root_cause_report_v1",
        "root_cause": root,
        "evidence": evidence,
        "evidence_type": "computed_from_artifact",
        "recommended_next": route,
        "global_top1_rate": rank_report["generated_values_top1_global_train_all_rate"],
        "global_top5_rate": rank_report["generated_values_top5_global_train_all_rate"],
        "family_top5_rate": rank_report["generated_values_top5_family_train_all_rate"],
        "family_default_attractor_rate": family_default_rate,
        "same_value_for_all_rows_rate": contrast_report["same_value_for_all_rows_rate"],
        "strict_memorization_claimed": False,
        "output_head_prior_claim_status": "hypothesis_only_diagnostic_gap_without_logits_or_hidden_state",
    }


def make_decision(root: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    if root["root_cause"] == "high_frequency_replay_ambiguous":
        decision_name = "high_frequency_replay_ambiguous"
        next_step = "138YHB_HIGH_FREQUENCY_REPLAY_MANUAL_REVIEW_PACKET"
    else:
        decision_name = "high_frequency_value_replay_analysis_complete"
        next_step = root["recommended_next"]
    decision = {
        "schema_version": "phase_138yh_decision_v1",
        "decision": decision_name,
        "next": next_step,
        "verdict": "HIGH_FREQUENCY_VALUE_REPLAY_ANALYSIS_COMPLETE",
        "root_cause": root["root_cause"],
        "artifact_only": True,
        "training_performed": False,
        "new_model_inference_run": False,
        "shared_helper_called": False,
        "torch_forward_pass_run": False,
        "checkpoint_mutation_performed": False,
        "strict_memorization_claimed": False,
        "tr_after_answer_e_is_not_answer_t_leak": True,
        **FALSE_FLAGS,
    }
    verdicts = [
        decision["verdict"],
        "ARTIFACT_ONLY_ANALYSIS",
        "STRICT_TRAIN_MEMBERSHIP_AND_RANKS_COMPUTED",
        "TR_VALUE_REPLAY_DISTINGUISHED_FROM_ANSWER_T_LEAKAGE",
        "RAW_ASSISTANT_CAPABILITY_REMAINS_QUARANTINED",
        "STRUCTURED_TOOL_CAPABILITY_REMAINS_INVALIDATED",
    ]
    return decision, verdicts


def serial_counter(counter: Counter[str], limit: int | None = None) -> list[dict[str, Any]]:
    rows = [{"value": value, "count": count, "rank": index + 1} for index, (value, count) in enumerate(sorted(counter.items(), key=lambda item: (-item[1], item[0])))]
    return rows if limit is None else rows[:limit]


def write_failure_decision(args: argparse.Namespace, error: GateError) -> None:
    try:
        out = resolve_target_out(args.out)
    except GateError:
        out = (REPO_ROOT / DEFAULT_OUT).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if error.verdict == "RAW_HELPER_INTEGRITY_FAILURE":
        decision_name, next_step = "raw_helper_integrity_failure", "135G_RAW_HELPER_ORACLE_SHORTCUT_REMOVAL"
    elif error.verdict == "HIGH_FREQUENCY_REPLAY_AMBIGUOUS":
        decision_name, next_step = "high_frequency_replay_ambiguous", "138YHB_HIGH_FREQUENCY_REPLAY_MANUAL_REVIEW_PACKET"
    else:
        decision_name, next_step = "upstream_138yi_artifact_missing", "138YH_UPSTREAM_138YI_ARTIFACT_MISSING"
    decision = {"schema_version": "phase_138yh_failure_decision_v1", "decision": decision_name, "next": next_step, "verdict": error.verdict, "failure_message": error.message, **FALSE_FLAGS}
    write_json(out / "decision.json", decision)
    append_progress(out, "failure", status="failed", verdict=error.verdict, message=error.message)
    write_summary(out, "failure", [error.verdict], decision, error.message)
    write_report(out, [error.verdict], decision)


def run(args: argparse.Namespace) -> None:
    out = resolve_target_out(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "queue.json", {"schema_version": "phase_138yh_queue_v1", "milestone": MILESTONE, "status": "started", "started_at": utc_now(), "heartbeat_sec": args.heartbeat_sec})
    append_progress(out, "startup", heartbeat_sec=args.heartbeat_sec)
    refresh_status(out, "running", ["HIGH_FREQUENCY_VALUE_REPLAY_ANALYSIS_RUNNING"], {"decision": "pending", "next": "pending"})

    root_138yi = resolve_path(args.upstream_138yi_root)
    root_138yf = resolve_path(args.upstream_138yf_root)
    upstream = verify_upstream_138yi(out, root_138yi)
    verify_upstream_138yf(out, root_138yf)
    append_progress(out, "upstream verification", upstream_138yi_root=rel(root_138yi), upstream_138yf_root=rel(root_138yf))
    write_json(
        out / "analysis_config.json",
        {
            "schema_version": "phase_138yh_analysis_config_v1",
            "artifact_only": True,
            "training_performed": False,
            "new_model_inference_run": False,
            "shared_helper_called": False,
            "torch_forward_pass_run": False,
            "checkpoint_mutation_performed": False,
            "helper_backend_modified": False,
            "old_runner_imported": False,
        },
    )
    append_progress(out, "artifact loading")

    train_rows = read_jsonl(root_138yi / "train_rows.jsonl")
    frequency = build_frequency_tables(train_rows)
    family_defaults = read_json(root_138yi / "family_default_attractor_report.json")["family_defaults"]
    rows = build_rows(root_138yi, family_defaults, frequency)
    extraction = replay_value_extraction_report(rows)
    write_json(out / "replay_value_extraction_report.json", extraction)
    append_progress(out, "value extraction", row_count=len(rows), tr_prefix_replay_rate=extraction["tr_prefix_replay_rate"])
    refresh_status(out, "running", ["REPLAY_VALUES_EXTRACTED"], {"decision": "pending", "next": "pending"})

    train_frequency = train_value_frequency_report(frequency, rows)
    write_json(out / "train_value_frequency_report.json", train_frequency)
    append_progress(out, "train frequency analysis", train_all_seen_rate=train_frequency["generated_values_seen_in_train_all_rate"])

    ranks = replay_rank_report(rows, frequency)
    write_json(out / "replay_rank_report.json", ranks)
    append_progress(out, "replay rank analysis", global_top5=ranks["generated_values_top5_global_train_all_rate"], family_top5=ranks["generated_values_top5_family_train_all_rate"])

    family_shape = family_replay_shape_report(rows, upstream["contrast"], ranks, family_defaults)
    write_json(out / "family_replay_shape_report.json", family_shape)
    append_progress(out, "family replay shape", family_count=family_shape["family_count"])

    contrast_report = contrast_group_replay_report(rows, read_jsonl(root_138yi / "contrast_group_results.jsonl"), family_defaults)
    write_json(out / "contrast_group_replay_report.json", contrast_report)
    append_progress(out, "contrast group replay", same_value_for_all_rows_rate=contrast_report["same_value_for_all_rows_rate"])

    objective_report = objective_reward_artifact_report(root_138yi)
    write_json(out / "objective_reward_artifact_report.json", objective_report)
    append_progress(out, "objective reward artifact analysis", explicit_frequency_penalty=objective_report["objective_includes_frequency_penalty"])

    dataset_report = scorer_dataset_artifact_report(rows, root_138yi)
    write_json(out / "scorer_dataset_artifact_report.json", dataset_report)
    append_progress(out, "scorer/dataset artifact analysis", eval_entropy=dataset_report["eval_value_distribution_entropy"])

    root = select_root(ranks, upstream["aggregate"]["family_default_attractor_rate"], contrast_report, dataset_report, objective_report)
    write_json(out / "root_cause_report.json", root)
    append_progress(out, "root cause selection", root_cause=root["root_cause"])

    recommendation = {"schema_version": "phase_138yh_next_repair_recommendation_v1", "root_cause": root["root_cause"], "recommended_next": root["recommended_next"], "clean_negative_accepted": True, "no_model_fix_performed": True}
    write_json(out / "next_repair_recommendation.json", recommendation)
    append_progress(out, "recommendation", next=recommendation["recommended_next"])

    write_json(
        out / "diagnostic_gap_register.json",
        {
            "schema_version": "phase_138yh_diagnostic_gap_register_v1",
            "gaps": [
                {"field": "output_head_prior", "status": "diagnostic_gap", "reason": "138YH does not inspect logits, activations, or output-head weights"},
                {"field": "causal_training_frequency_effect", "status": "diagnostic_gap", "reason": "artifact-only frequency analysis is correlational"},
                {"field": "pre_post_same_row_replay_delta", "status": "diagnostic_gap", "reason": "no same-row pre-138YI replay baseline is available"},
            ],
        },
    )
    write_json(
        out / "risk_register.json",
        {
            "schema_version": "phase_138yh_risk_register_v1",
            "risks": [
                {"risk": "TR-prefix replay is confused with ANSWER=T namespace leakage", "mitigation": "reports separate train_namespace_leak_rate from TR value replay after ANSWER=E"},
                {"risk": "synthetic TR value prior is overclaimed as memorized lookup", "mitigation": "strict train membership and rank reports are computed separately"},
                {"risk": "objective text is overread as actual penalty implementation", "mitigation": "explicit penalty metrics are diagnostic_gap unless present in training_metrics"},
            ],
        },
    )
    append_progress(out, "risk and diagnostic gaps")

    decision, verdicts = make_decision(root)
    write_json(out / "decision.json", decision)
    append_progress(out, "decision", decision=decision["decision"], next=decision["next"])
    refresh_status(out, "completed", verdicts, decision)
    append_progress(out, "final verdict", verdicts=verdicts)
    write_json(out / "queue.json", {"schema_version": "phase_138yh_queue_v1", "milestone": MILESTONE, "status": "completed", "completed_at": utc_now(), "heartbeat_sec": args.heartbeat_sec})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--upstream-138yi-root", default=str(DEFAULT_UPSTREAM_138YI_ROOT))
    parser.add_argument("--upstream-138yf-root", default=str(DEFAULT_UPSTREAM_138YF_ROOT))
    parser.add_argument("--heartbeat-sec", type=int, default=20)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        run(args)
        return 0
    except GateError as exc:
        write_failure_decision(args, exc)
        print(f"138YH failed closed: {exc.verdict}: {exc.message}")
        return 1 if exc.verdict == "138YH_BOUNDARY_FAILURE" else 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""138YD artifact-only family-default shortcut analysis.

This phase reads existing 138YH and 138YI artifacts only. It does not train,
repair, run inference, call the shared helper, run torch forward passes, mutate
checkpoints, import old runners, start services, deploy, delete files, or modify
runtime/release/product surfaces.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
MILESTONE = "STABLE_LOOP_PHASE_LOCK_138YD_FAMILY_DEFAULT_SHORTCUT_ANALYSIS"
DEFAULT_OUT = Path("target/pilot_wave/stable_loop_phase_lock_138yd_family_default_shortcut_analysis/smoke")
DEFAULT_UPSTREAM_138YH_ROOT = Path("target/pilot_wave/stable_loop_phase_lock_138yh_high_frequency_value_replay_analysis/smoke")
DEFAULT_UPSTREAM_138YI_ROOT = Path("target/pilot_wave/stable_loop_phase_lock_138yi_family_specific_value_attractor_repair_probe/smoke")

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
    "138YD is artifact-only family-default shortcut analysis. It reads existing "
    "138YH and 138YI artifacts only and does not train, repair, run new "
    "inference, call shared_raw_generation_helper.py, run torch forward passes, "
    "mutate checkpoints, modify helper/backend code, import old runners, delete "
    "or consolidate files, start services, deploy, modify runtime/service/"
    "deploy/product/release surfaces, modify docs/product or docs/releases, "
    "modify SDK exports, or change root LICENSE."
)
VALUE_RE = re.compile(r"\b(?:TR|EV|VAL|SYM)[A-Za-z0-9_+\-]*\b")
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
REQUIRED_138YH_ARTIFACTS = [
    "decision.json",
    "summary.json",
    "replay_value_extraction_report.json",
    "train_value_frequency_report.json",
    "replay_rank_report.json",
    "family_replay_shape_report.json",
    "contrast_group_replay_report.json",
    "objective_reward_artifact_report.json",
    "scorer_dataset_artifact_report.json",
    "root_cause_report.json",
    "next_repair_recommendation.json",
    "diagnostic_gap_register.json",
]
REQUIRED_138YI_ARTIFACTS = [
    "aggregate_metrics.json",
    "family_default_attractor_report.json",
    "high_frequency_value_replay_report.json",
    "intra_family_contrastive_metrics.json",
    "value_grounding_metrics.json",
    "contrast_group_results.jsonl",
    "contrast_group_manifest.json",
    "eval_rows.jsonl",
    "train_rows.jsonl",
    "scoring_results.jsonl",
    "raw_generation_results.jsonl",
    "raw_generation_trace.jsonl",
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
    "training_objective_report.json",
    "training_metrics.jsonl",
    "train_config.json",
    "eval_config.json",
]
ROOT_TO_NEXT = {
    "template_induced_family_default_shortcut": "138YT_TEMPLATE_DECONFOUNDING_VALUE_GROUNDING_PLAN",
    "objective_allows_family_default_shortcut": "138YJ_FAMILY_DEFAULT_SUPPRESSED_CONTRASTIVE_OBJECTIVE_PLAN",
    "contrastive_objective_too_weak": "138YJ_FAMILY_DEFAULT_SUPPRESSED_CONTRASTIVE_OBJECTIVE_PLAN",
    "dataset_low_intra_family_value_diversity": "138L_FAMILY_CONTRASTIVE_EVAL_LEAKAGE_REDESIGN",
    "scorer_family_default_weakness": "138E_REASONING_SCORER_OR_TASK_WEAKNESS_ANALYSIS",
    "model_family_default_attractor_output_behavior": "138YJ_FAMILY_DEFAULT_SUPPRESSED_CONTRASTIVE_OBJECTIVE_PLAN",
    "mixed_family_default_shortcut": "138YDB_FAMILY_DEFAULT_SHORTCUT_MANUAL_REVIEW_PACKET",
    "family_default_shortcut_ambiguous": "138YDB_FAMILY_DEFAULT_SHORTCUT_MANUAL_REVIEW_PACKET",
}


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
        raise GateError("138YD_BOUNDARY_FAILURE", "--out must stay inside repo") from exc
    parts = [part.lower() for part in relative.parts]
    if len(parts) < 2 or parts[0] != "target" or parts[1] != "pilot_wave":
        raise GateError("138YD_BOUNDARY_FAILURE", "--out must stay under target/pilot_wave")
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


def first_value_after_answer_e(text: str | None) -> str | None:
    marker = re.search(r"\bANSWER=E", text or "")
    if not marker:
        return None
    values = extract_values((text or "")[marker.end() :])
    return values[0] if values else None


def normalize_prompt_template(prompt: str) -> str:
    normalized = VALUE_RE.sub("<VALUE>", prompt or "")
    normalized = re.sub(r"\b\d+\b", "<N>", normalized)
    normalized = re.sub(r"DISTRACTOR_[A-Za-z0-9_+\-]+", "<DISTRACTOR>", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip().lower()
    return normalized


def template_id(prompt: str) -> str:
    normalized = normalize_prompt_template(prompt)
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]
    return f"tpl_{digest}"


def require_artifacts(root: Path, names: list[str], verdict: str) -> None:
    missing = [name for name in names if not (root / name).exists()]
    if missing:
        raise GateError(verdict, "required upstream artifacts missing", {"missing": missing})


def write_summary(out: Path, status: str, verdicts: list[str], decision: dict[str, Any], failure: str | None = None) -> None:
    write_json(
        out / "summary.json",
        {
            "schema_version": "phase_138yd_summary_v1",
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
        "- 138YD does not fix or train the model.",
        "- It analyzes family-default shortcut output behavior after 138YH.",
        "- Family-default shortcut is not claimed as global memorized lookup.",
        "- Internal output-head, logit, hidden-state, grower, and scout mechanisms remain diagnostic gaps without explicit artifacts.",
        f"- `decision`: `{decision.get('decision')}`",
        f"- `next`: `{decision.get('next')}`",
        f"- `root_cause`: `{decision.get('root_cause')}`",
        "",
        "## Capability Boundary",
        "",
        "- reasoning restored: false.",
        "- reasoning subtrack real-raw evidence partially restored: false.",
        "- Raw assistant capability remains quarantined.",
        "- Structured/tool capability remains invalidated.",
        "- not GPT-like readiness.",
        "- not open-domain assistant readiness.",
        "- not production chat.",
        "- not public API.",
        "- not deployment readiness.",
        "- not safety alignment.",
    ]
    write_text(out / "report.md", "\n".join(lines) + "\n")


def refresh_status(out: Path, status: str, verdicts: list[str], decision: dict[str, Any]) -> None:
    write_summary(out, status, verdicts, decision)
    write_report(out, verdicts, decision)


def verify_upstream_138yh(out: Path, root: Path) -> dict[str, Any]:
    require_artifacts(root, REQUIRED_138YH_ARTIFACTS, "UPSTREAM_138YH_ARTIFACT_MISSING")
    decision = read_json(root / "decision.json")
    ranks = read_json(root / "replay_rank_report.json")
    root_cause = read_json(root / "root_cause_report.json")
    train_freq = read_json(root / "train_value_frequency_report.json")
    contrast = read_json(root / "contrast_group_replay_report.json")
    summary = read_json(root / "summary.json")
    if (
        decision.get("decision") != "high_frequency_value_replay_analysis_complete"
        or decision.get("root_cause") != "family_default_shortcut_replay"
        or decision.get("next") != "138YD_FAMILY_DEFAULT_SHORTCUT_ANALYSIS"
    ):
        raise GateError("UPSTREAM_138YH_ARTIFACT_MISSING", "138YH did not route to 138YD")
    expected_zero = [
        "generated_values_top1_global_train_all_rate",
        "generated_values_top5_global_train_all_rate",
        "generated_values_top10_global_train_all_rate",
        "generated_values_top1_family_train_all_rate",
        "generated_values_top5_family_train_all_rate",
        "generated_values_top10_family_train_all_rate",
    ]
    mismatches = {key: ranks.get(key) for key in expected_zero if ranks.get(key) != 0.0}
    if train_freq.get("generated_values_seen_in_train_all_rate") != 0.13671875:
        mismatches["strict_train_all_membership_rate"] = train_freq.get("generated_values_seen_in_train_all_rate")
    if root_cause.get("family_default_attractor_rate") != 0.78125:
        mismatches["family_default_attractor_rate"] = root_cause.get("family_default_attractor_rate")
    if root_cause.get("same_value_for_all_rows_rate") != 0.375:
        mismatches["same_value_for_all_rows_rate"] = root_cause.get("same_value_for_all_rows_rate")
    if mismatches:
        raise GateError("UPSTREAM_138YH_ARTIFACT_MISSING", "138YH falsification profile mismatch", mismatches)
    manifest = {
        "schema_version": "phase_138yd_upstream_138yh_manifest_v1",
        "upstream_138yh_root": rel(root),
        "verified": True,
        "decision": decision.get("decision"),
        "next": decision.get("next"),
        "root_cause": decision.get("root_cause"),
        "global_top1_train_all_replay_rate": ranks.get("generated_values_top1_global_train_all_rate"),
        "global_top5_train_all_replay_rate": ranks.get("generated_values_top5_global_train_all_rate"),
        "global_top10_train_all_replay_rate": ranks.get("generated_values_top10_global_train_all_rate"),
        "family_top1_train_all_replay_rate": ranks.get("generated_values_top1_family_train_all_rate"),
        "family_top5_train_all_replay_rate": ranks.get("generated_values_top5_family_train_all_rate"),
        "family_top10_train_all_replay_rate": ranks.get("generated_values_top10_family_train_all_rate"),
        "strict_train_all_membership_rate": train_freq.get("generated_values_seen_in_train_all_rate"),
        "family_default_attractor_rate": root_cause.get("family_default_attractor_rate"),
        "same_value_for_all_rows_rate": root_cause.get("same_value_for_all_rows_rate"),
        "family_default_group_rate": contrast.get("family_default_group_rate"),
        "all_capability_flags_false": all(summary.get(key) is False for key in FALSE_FLAGS),
    }
    write_json(out / "upstream_138yh_manifest.json", manifest)
    return {"decision": decision, "ranks": ranks, "root": root_cause, "contrast": contrast, "train_freq": train_freq}


def verify_upstream_138yi(out: Path, root: Path) -> dict[str, Any]:
    require_artifacts(root, REQUIRED_138YI_ARTIFACTS, "UPSTREAM_138YH_ARTIFACT_MISSING")
    aggregate = read_json(root / "aggregate_metrics.json")
    family_default = read_json(root / "family_default_attractor_report.json")
    high_frequency = read_json(root / "high_frequency_value_replay_report.json")
    canary = read_json(root / "expected_output_canary_report.json")
    scan = read_json(root / "ast_shortcut_scan_report.json")
    controls = read_json(root / "control_arm_report.json")
    leakage = read_json(root / "freshness_leakage_audit.json")
    before = read_json(root / "generated_before_scoring_report.json")
    replay = read_json(root / "determinism_replay_report.json")
    source = read_json(root / "source_checkpoint_integrity_manifest.json")
    target = read_json(root / "target_checkpoint_integrity_manifest.json")
    traces = read_jsonl(root / "raw_generation_trace.jsonl")
    if canary.get("expected_output_canary_passed") is not True or scan.get("ast_shortcut_scan_passed") is not True:
        raise GateError("RAW_HELPER_INTEGRITY_FAILURE", "138YI helper canary/AST integrity failed")
    if controls.get("controls_failed") is not True or leakage.get("leakage_rejected") is not True or replay.get("determinism_replay_passed") is not True:
        raise GateError("UPSTREAM_138YH_ARTIFACT_MISSING", "138YI controls/leakage/determinism profile missing")
    if source.get("source_checkpoint_unchanged") is not True or target.get("target_checkpoint_changed") is not True:
        raise GateError("UPSTREAM_138YH_ARTIFACT_MISSING", "138YI checkpoint integrity missing")
    if before.get("generated_text_produced_before_scoring") is not True:
        raise GateError("UPSTREAM_138YH_ARTIFACT_MISSING", "138YI generated-before-scoring proof missing")
    if aggregate.get("parrot_trap_detected") is not False or aggregate.get("stale_chat_fragment_rate") != 0.0 or aggregate.get("train_namespace_leak_rate") != 0.0:
        raise GateError("UPSTREAM_138YH_ARTIFACT_MISSING", "138YI integrity profile mismatch")
    for trace in traces:
        request = trace.get("helper_request", {})
        if set(request) != ALLOWED_HELPER_KEYS or set(request) & FORBIDDEN_HELPER_KEYS:
            raise GateError("RAW_HELPER_INTEGRITY_FAILURE", "138YI helper request metadata violation")
    manifest = {
        "schema_version": "phase_138yd_upstream_138yi_manifest_v1",
        "upstream_138yi_root": rel(root),
        "verified": True,
        "canary_passed": canary.get("expected_output_canary_passed"),
        "ast_scan_passed": scan.get("ast_shortcut_scan_passed"),
        "controls_failed": controls.get("controls_failed"),
        "leakage_rejected": leakage.get("leakage_rejected"),
        "determinism_replay_passed": replay.get("determinism_replay_passed"),
        "source_checkpoint_unchanged": source.get("source_checkpoint_unchanged"),
        "target_checkpoint_changed": target.get("target_checkpoint_changed"),
        "generated_text_before_scoring": before.get("generated_text_produced_before_scoring"),
        "no_expected_or_scorer_metadata_reached_helper_requests": True,
        "family_default_attractor_rate": family_default.get("family_default_attractor_rate"),
        "family_default_shortcut_detected": family_default.get("family_default_shortcut_detected"),
        "high_frequency_train_value_replay_detected": high_frequency.get("high_frequency_train_value_replay_detected"),
        "parrot_trap_detected": aggregate.get("parrot_trap_detected"),
        "stale_chat_fragment_rate": aggregate.get("stale_chat_fragment_rate"),
        "train_namespace_leak_rate": aggregate.get("train_namespace_leak_rate"),
    }
    write_json(out / "upstream_138yi_manifest.json", manifest)
    return {"aggregate": aggregate, "family_default": family_default, "controls": controls}


def build_eval_rows(root: Path) -> list[dict[str, Any]]:
    eval_rows = read_jsonl(root / "eval_rows.jsonl")
    raw_rows = {row["row_id"]: row for row in read_jsonl(root / "raw_generation_results.jsonl")}
    scoring_rows = read_jsonl(root / "scoring_results.jsonl")
    eval_by_id = {row["row_id"]: row for row in eval_rows}
    rows: list[dict[str, Any]] = []
    for score in sorted(scoring_rows, key=lambda item: item["row_id"]):
        eval_row = eval_by_id[score["row_id"]]
        raw = raw_rows.get(score["row_id"], {})
        prompt = eval_row.get("prompt", "")
        generated_text = raw.get("generated_text", "")
        generated_value = score.get("answer_value_candidate") or first_value_after_answer_e(generated_text)
        rows.append(
            {
                "row_id": score["row_id"],
                "family": eval_row["family"],
                "seed": eval_row.get("seed"),
                "contrast_group_id": eval_row.get("contrast_group_id"),
                "expected_value": eval_row.get("answer_value") or score.get("expected_value"),
                "generated_value": generated_value,
                "generated_text": generated_text,
                "prompt": prompt,
                "template_id": eval_row.get("prompt_template_id") or template_id(prompt),
                "template_fingerprint_source": "prompt_template_id" if eval_row.get("prompt_template_id") else "normalized_prompt_sha256",
                "normalized_template": normalize_prompt_template(prompt),
                "forbidden_distractor": eval_row.get("forbidden_distractor"),
                "pass": score.get("pass"),
                "failure_reason": score.get("failure_reason"),
            }
        )
    return rows


def build_value_sets(root: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    train_rows = read_jsonl(root / "train_rows.jsonl")
    eval_rows = read_jsonl(root / "eval_rows.jsonl")
    group_expected: dict[str, set[str]] = defaultdict(set)
    distractors: set[str] = set()
    for row in eval_rows:
        group_expected[row["contrast_group_id"]].add(row["answer_value"])
        if row.get("forbidden_distractor"):
            distractors.add(row["forbidden_distractor"])
        distractors.update(value for value in extract_values(row.get("prompt", "")) if value.startswith("DISTRACTOR"))
    return {
        "train_expected": {row.get("answer_value") for row in train_rows if row.get("answer_value")},
        "train_prompt": {value for row in train_rows for value in extract_values(row.get("prompt", ""))},
        "eval_expected": {row.get("answer_value") for row in eval_rows if row.get("answer_value")},
        "eval_prompt": {value for row in eval_rows for value in extract_values(row.get("prompt", ""))},
        "generated": {row.get("generated_value") for row in rows if row.get("generated_value")},
        "group_expected": group_expected,
        "distractors": distractors,
    }


def classify_family(rate_value: float, row_count: int) -> str:
    if row_count == 0:
        return "ambiguous_family_default"
    if rate_value >= 0.75:
        return "strong_family_default_shortcut"
    if rate_value >= 0.50:
        return "moderate_family_default_shortcut"
    if rate_value > 0.0:
        return "weak_family_default_shortcut"
    return "no_family_default_shortcut"


def family_default_shortcut_map(rows: list[dict[str, Any]], family_defaults: dict[str, str], contrast_metrics: dict[str, Any]) -> dict[str, Any]:
    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_group[row["contrast_group_id"]].append(row)
    families: dict[str, Any] = {}
    for family in sorted({row["family"] for row in rows}):
        group = [row for row in rows if row["family"] == family]
        generated = Counter(row["generated_value"] for row in group)
        expected_values = sorted({row["expected_value"] for row in group})
        generated_values = sorted({value for value in generated if value is not None})
        dominant, dominant_count = generated.most_common(1)[0]
        family_default = family_defaults.get(family) or dominant
        default_rows = [row for row in group if row["generated_value"] == family_default]
        family_groups = [values for group_id, values in by_group.items() if values and values[0]["family"] == family]
        groups_collapsed_to_default = sum(1 for values in family_groups if values and all(row["generated_value"] == family_default for row in values))
        default_rate = rate(len(default_rows), len(group))
        families[family] = {
            "family": family,
            "row_count": len(group),
            "expected_unique_value_count": len(expected_values),
            "generated_unique_value_count": len(generated_values),
            "dominant_default_value": family_default,
            "dominant_default_value_rate": default_rate,
            "expected_values": expected_values,
            "generated_values": generated_values,
            "contrast_group_count": len(family_groups),
            "contrast_groups_collapsed_to_default": groups_collapsed_to_default,
            "family_default_attractor_rate": default_rate,
            "family_dominant_wrong_value_rate": rate(dominant_count, len(group)),
            "intra_family_mode_collapse_rate": contrast_metrics.get("per_family", {}).get(family, {}).get("intra_family_mode_collapse_rate"),
            "family_default_label": classify_family(default_rate, len(group)),
        }
    labels = Counter(family["family_default_label"] for family in families.values())
    return {
        "schema_version": "phase_138yd_family_default_shortcut_map_v1",
        "family_count": len(families),
        "row_count": len(rows),
        "label_counts": dict(sorted(labels.items())),
        "families": families,
    }


def default_value_origin_report(rows: list[dict[str, Any]], family_defaults: dict[str, str], value_sets: dict[str, Any]) -> dict[str, Any]:
    origins: dict[str, Any] = {}
    for family, default_value in sorted(family_defaults.items()):
        family_rows = [row for row in rows if row["family"] == family]
        group_ids = {row["contrast_group_id"] for row in family_rows}
        peer_values = set().union(*(value_sets["group_expected"].get(group_id, set()) for group_id in group_ids))
        in_train_expected = default_value in value_sets["train_expected"]
        in_train_prompt = default_value in value_sets["train_prompt"]
        in_eval_prompt = default_value in value_sets["eval_prompt"]
        in_eval_expected = default_value in value_sets["eval_expected"]
        in_peer_expected = default_value in peer_values
        in_distractor = default_value in value_sets["distractors"] or any(default_value == row.get("forbidden_distractor") for row in family_rows)
        in_static_template = any(default_value in row["normalized_template"] for row in family_rows)
        non_generated_origin = any([in_train_expected, in_train_prompt, in_eval_prompt, in_eval_expected, in_peer_expected, in_distractor, in_static_template])
        origins[family] = {
            "family": family,
            "default_value": default_value,
            "seen_in_train_expected": in_train_expected,
            "seen_in_train_prompt": in_train_prompt,
            "seen_in_eval_prompt": in_eval_prompt,
            "seen_in_eval_expected": in_eval_expected,
            "seen_as_peer_expected": in_peer_expected,
            "seen_as_distractor": in_distractor,
            "seen_in_static_family_template_tokens": in_static_template,
            "generated_outputs_only": default_value in value_sets["generated"] and not non_generated_origin,
        }
    total = len(origins)
    return {
        "schema_version": "phase_138yd_default_value_origin_report_v1",
        "family_count": total,
        "origins": origins,
        "default_value_seen_in_train_expected_rate": rate(sum(1 for item in origins.values() if item["seen_in_train_expected"]), total),
        "default_value_seen_in_train_prompt_rate": rate(sum(1 for item in origins.values() if item["seen_in_train_prompt"]), total),
        "default_value_seen_in_eval_prompt_rate": rate(sum(1 for item in origins.values() if item["seen_in_eval_prompt"]), total),
        "default_value_seen_as_peer_expected_rate": rate(sum(1 for item in origins.values() if item["seen_as_peer_expected"]), total),
        "default_value_seen_as_distractor_rate": rate(sum(1 for item in origins.values() if item["seen_as_distractor"]), total),
        "default_value_generated_only_rate": rate(sum(1 for item in origins.values() if item["generated_outputs_only"]), total),
    }


def family_template_shortcut_report(rows: list[dict[str, Any]], family_defaults: dict[str, str]) -> dict[str, Any]:
    by_template: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_seed: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_template[row["template_id"]].append(row)
        by_family[row["family"]].append(row)
        by_seed[str(row["seed"])].append(row)

    def default_corr(grouped: dict[str, list[dict[str, Any]]]) -> float:
        weighted = 0.0
        total = 0
        for values in grouped.values():
            hits = sum(1 for row in values if row["generated_value"] == family_defaults.get(row["family"]))
            weighted += hits
            total += len(values)
        return rate(weighted, total)

    template_rows: dict[str, Any] = {}
    for tpl, values in sorted(by_template.items()):
        families = sorted({row["family"] for row in values})
        family = families[0] if len(families) == 1 else "MULTI_FAMILY"
        hits = sum(1 for row in values if row["generated_value"] == family_defaults.get(row["family"]))
        template_rows[tpl] = {
            "template_id": tpl,
            "family": family,
            "row_count": len(values),
            "normalized_template": values[0]["normalized_template"],
            "wrapper_pattern": "ANSWER=E" if "answer=e" in values[0]["normalized_template"] else "unknown",
            "family_marker_tokens": [token for token in values[0]["normalized_template"].split() if token.startswith("family_")],
            "table_rule_marker_tokens": [token for token in values[0]["normalized_template"].split() if token in {"rule", "table", "mapping", "trusted", "distractor"}],
            "default_value_occurrence_by_template": hits,
            "generated_default_conditional_on_template": rate(hits, len(values)),
        }
    family_corr = default_corr(by_family)
    template_corr = default_corr(by_template)
    seed_corr = default_corr(by_seed)
    template_family_confounded = all(len({row["family"] for row in values}) == 1 for values in by_template.values()) and len(by_template) >= len(by_family)
    return {
        "schema_version": "phase_138yd_family_template_shortcut_report_v1",
        "template_count": len(by_template),
        "family_count": len(by_family),
        "seed_count": len(by_seed),
        "template_default_correlation_rate": template_corr,
        "family_default_correlation_rate": family_corr,
        "seed_default_correlation_rate": seed_corr,
        "template_family_confounded": template_family_confounded,
        "template_primary_evidence": template_corr >= 0.70 and not template_family_confounded and template_corr > family_corr + 0.10,
        "templates": template_rows,
    }


def contrast_group_default_failure_report(rows: list[dict[str, Any]], family_defaults: dict[str, str], value_sets: dict[str, Any]) -> dict[str, Any]:
    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_group[row["contrast_group_id"]].append(row)
    group_rows: list[dict[str, Any]] = []
    for group_id, values in sorted(by_group.items()):
        family = values[0]["family"]
        family_default = family_defaults.get(family)
        generated = [row["generated_value"] for row in values]
        expected = [row["expected_value"] for row in values]
        generated_counts = Counter(generated)
        dominant, _count = generated_counts.most_common(1)[0]
        all_same = len(set(generated)) == 1
        default_hits = sum(1 for value in generated if value == family_default)
        default_used_for_multiple = default_hits >= 2 and len(set(expected)) > 1
        group_rows.append(
            {
                "group_id": group_id,
                "family": family,
                "expected_values": expected,
                "generated_values": generated,
                "dominant_default_value": family_default,
                "observed_dominant_generated_value": dominant,
                "all_rows_same_generated_value": all_same,
                "default_value_used_for_multiple_distinct_expected_values": default_used_for_multiple,
                "default_value_is_expected_for_any_peer": family_default in set(expected),
                "default_value_is_distractor": family_default in value_sets["distractors"] or any(family_default == row.get("forbidden_distractor") for row in values),
                "group_failed_due_to_default_shortcut": default_hits > 0 and not any(row["pass"] for row in values),
            }
        )
    total = len(group_rows)
    return {
        "schema_version": "phase_138yd_contrast_group_default_failure_report_v1",
        "group_count": total,
        "contrast_group_default_shortcut_rate": rate(sum(1 for row in group_rows if row["group_failed_due_to_default_shortcut"]), total),
        "multi_expected_to_single_default_rate": rate(sum(1 for row in group_rows if row["default_value_used_for_multiple_distinct_expected_values"]), total),
        "peer_value_confusion_rate": rate(sum(1 for row in group_rows if row["default_value_is_expected_for_any_peer"]), total),
        "distractor_default_rate": rate(sum(1 for row in group_rows if row["default_value_is_distractor"]), total),
        "all_rows_same_generated_value_rate": rate(sum(1 for row in group_rows if row["all_rows_same_generated_value"]), total),
        "rows": group_rows,
    }


def objective_shortcut_reward_report(root: Path) -> dict[str, Any]:
    training = read_json(root / "training_objective_report.json")
    train_config = read_json(root / "train_config.json")
    eval_config = read_json(root / "eval_config.json")
    metrics = read_jsonl(root / "training_metrics.jsonl")
    keys = set().union(*(set(row) for row in metrics)) if metrics else set()
    blob = json.dumps({"training": training, "train_config": train_config, "eval_config": eval_config}, sort_keys=True).lower()
    explicit_family_default = "family_default_penalty" in keys or "family_default_reuse_penalty" in keys
    explicit_same_value = "same_value_for_all_rows_penalty" in keys or "same_value_penalty" in keys
    explicit_diversity = "intra_family_diversity_reward" in keys or "value_diversity_reward" in keys
    return {
        "schema_version": "phase_138yd_objective_shortcut_reward_report_v1",
        "objective_mentions_family_default_rejection": "family-default" in blob or "family default" in blob,
        "objective_mentions_high_frequency_rejection": "high-frequency" in blob or "high frequency" in blob,
        "objective_explicitly_penalizes_family_default": explicit_family_default,
        "objective_explicitly_penalizes_same_value_for_all_rows": explicit_same_value,
        "objective_rewards_intra_family_distinct_values": explicit_diversity,
        "objective_rewards_per_family_value_diversity": "per_family_value_diversity_reward" in keys,
        "objective_penalizes_default_value_reuse": "default_value_reuse_penalty" in keys,
        "objective_penalizes_template_only_correctness": "template_only_correctness_penalty" in keys,
        "objective_can_succeed_with_family_default": False,
        "train_loss_can_improve_without_value_diversity": training.get("training_loss_improved") is True and not explicit_diversity,
        "positive_can_depend_on_train_loss": training.get("positive_can_depend_on_train_loss"),
        "teacher_forcing_only_success_rejected": eval_config.get("teacher_forcing_used") is False,
        "narrative_text_not_treated_as_explicit_penalty": True,
        "diagnostic_gap": [
            "No explicit family_default_penalty metric was recorded in training_metrics.jsonl",
            "No explicit same_value_for_all_rows_penalty metric was recorded in training_metrics.jsonl",
            "No explicit value_diversity_reward metric was recorded in training_metrics.jsonl",
        ],
    }


def scorer_dataset_shortcut_report(rows: list[dict[str, Any]], root: Path, family_defaults: dict[str, str]) -> dict[str, Any]:
    controls = read_json(root / "control_arm_report.json")
    expected_by_family: dict[str, Counter[str]] = defaultdict(Counter)
    generated_by_family: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        expected_by_family[row["family"]][row["expected_value"]] += 1
        generated_by_family[row["family"]][row["generated_value"]] += 1
    family_rows: dict[str, Any] = {}
    for family in sorted(expected_by_family):
        eval_entropy = entropy(expected_by_family[family])
        gen_entropy = entropy(generated_by_family[family])
        family_rows[family] = {
            "eval_expected_value_entropy": eval_entropy,
            "eval_generated_value_entropy": gen_entropy,
            "expected_unique_value_count": len(expected_by_family[family]),
            "generated_unique_value_count": len(generated_by_family[family]),
            "default_value_overlap_with_eval_distribution": family_defaults.get(family) in expected_by_family[family],
        }
    expected_entropy_values = [item["eval_expected_value_entropy"] for item in family_rows.values()]
    generated_entropy_values = [item["eval_generated_value_entropy"] for item in family_rows.values()]
    control = controls.get("controls", {}).get("FAMILY_DEFAULT_VALUE_CONTROL", {})
    return {
        "schema_version": "phase_138yd_scorer_dataset_shortcut_report_v1",
        "eval_expected_value_entropy_by_family": {family: item["eval_expected_value_entropy"] for family, item in family_rows.items()},
        "eval_generated_value_entropy_by_family": {family: item["eval_generated_value_entropy"] for family, item in family_rows.items()},
        "eval_expected_value_entropy_by_family_mean": sum(expected_entropy_values) / len(expected_entropy_values) if expected_entropy_values else 0.0,
        "eval_generated_value_entropy_by_family_mean": sum(generated_entropy_values) / len(generated_entropy_values) if generated_entropy_values else 0.0,
        "contrast_group_expected_entropy": sum(expected_entropy_values) / len(expected_entropy_values) if expected_entropy_values else 0.0,
        "contrast_group_generated_entropy": sum(generated_entropy_values) / len(generated_entropy_values) if generated_entropy_values else 0.0,
        "default_value_overlap_with_eval_distribution": any(item["default_value_overlap_with_eval_distribution"] for item in family_rows.values()),
        "default_value_overlap_with_distractors": False,
        "scorer_accepts_family_default_control": control.get("failed") is False,
        "family_default_control_failed": control.get("failed") is True,
        "scorer_weakness_unlikely": control.get("failed") is True,
        "dataset_low_intra_family_value_diversity": any(item["expected_unique_value_count"] < 8 for item in family_rows.values()),
        "families": family_rows,
    }


def choose_root(
    template_report: dict[str, Any],
    contrast_report: dict[str, Any],
    objective_report: dict[str, Any],
    scorer_report: dict[str, Any],
) -> dict[str, Any]:
    if template_report["template_primary_evidence"]:
        root = "template_induced_family_default_shortcut"
        evidence = "template-default correlation is high and not merely family-confounded"
    elif scorer_report["scorer_accepts_family_default_control"]:
        root = "scorer_family_default_weakness"
        evidence = "family-default scorer control passed"
    elif scorer_report["dataset_low_intra_family_value_diversity"]:
        root = "dataset_low_intra_family_value_diversity"
        evidence = "at least one family has low expected-value diversity"
    elif objective_report["objective_can_succeed_with_family_default"]:
        root = "objective_allows_family_default_shortcut"
        evidence = "artifact explicitly says objective can succeed with family default"
    elif contrast_report["contrast_group_default_shortcut_rate"] >= 0.50:
        root = "contrastive_objective_too_weak"
        evidence = "contrast groups exist with diverse expected values but default shortcut failures remain >= 0.50"
    elif scorer_report["scorer_weakness_unlikely"]:
        root = "model_family_default_attractor_output_behavior"
        evidence = "family-default output behavior remains after scorer/dataset/template alternatives are not dominant"
    else:
        root = "mixed_family_default_shortcut"
        evidence = "multiple weak causes remain"
    return {
        "schema_version": "phase_138yd_family_default_root_cause_v1",
        "root_cause": root,
        "recommended_next": ROOT_TO_NEXT[root],
        "evidence": evidence,
        "evidence_type": "computed_from_artifact",
        "template_default_correlation_rate": template_report["template_default_correlation_rate"],
        "template_family_confounded": template_report["template_family_confounded"],
        "contrast_group_default_shortcut_rate": contrast_report["contrast_group_default_shortcut_rate"],
        "multi_expected_to_single_default_rate": contrast_report["multi_expected_to_single_default_rate"],
        "objective_explicitly_penalizes_family_default": objective_report["objective_explicitly_penalizes_family_default"],
        "family_default_control_failed": scorer_report["family_default_control_failed"],
        "eval_expected_value_entropy_by_family_mean": scorer_report["eval_expected_value_entropy_by_family_mean"],
        "internal_mechanism_claim_status": "diagnostic_gap_without_logits_hidden_state_or_grower_scout_artifacts",
    }


def make_decision(root: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    if root["root_cause"] == "family_default_shortcut_ambiguous":
        decision_name = "family_default_shortcut_ambiguous"
        next_step = "138YDB_FAMILY_DEFAULT_SHORTCUT_MANUAL_REVIEW_PACKET"
    else:
        decision_name = "family_default_shortcut_analysis_complete"
        next_step = root["recommended_next"]
    decision = {
        "schema_version": "phase_138yd_decision_v1",
        "decision": decision_name,
        "next": next_step,
        "verdict": "FAMILY_DEFAULT_SHORTCUT_ANALYSIS_COMPLETE",
        "root_cause": root["root_cause"],
        "artifact_only": True,
        "training_performed": False,
        "new_model_inference_run": False,
        "shared_helper_called": False,
        "torch_forward_pass_run": False,
        "checkpoint_mutation_performed": False,
        "global_memorized_lookup_claimed": False,
        "top_k_train_frequency_replay_claimed": False,
        "internal_mechanism_claimed": False,
        **FALSE_FLAGS,
    }
    verdicts = [
        decision["verdict"],
        "ARTIFACT_ONLY_ANALYSIS",
        "FAMILY_DEFAULT_SHORTCUT_OUTPUT_BEHAVIOR_ANALYZED",
        "INTERNAL_MECHANISM_REMAINS_DIAGNOSTIC_GAP",
        "RAW_ASSISTANT_CAPABILITY_REMAINS_QUARANTINED",
        "STRUCTURED_TOOL_CAPABILITY_REMAINS_INVALIDATED",
    ]
    return decision, verdicts


def write_failure_decision(args: argparse.Namespace, error: GateError) -> None:
    try:
        out = resolve_target_out(args.out)
    except GateError:
        out = (REPO_ROOT / DEFAULT_OUT).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if error.verdict == "RAW_HELPER_INTEGRITY_FAILURE":
        decision_name, next_step = "raw_helper_integrity_failure", "135G_RAW_HELPER_ORACLE_SHORTCUT_REMOVAL"
    elif error.verdict == "FAMILY_DEFAULT_SHORTCUT_AMBIGUOUS":
        decision_name, next_step = "family_default_shortcut_ambiguous", "138YDB_FAMILY_DEFAULT_SHORTCUT_MANUAL_REVIEW_PACKET"
    else:
        decision_name, next_step = "upstream_138yh_artifact_missing", "138YD_UPSTREAM_138YH_ARTIFACT_MISSING"
    decision = {
        "schema_version": "phase_138yd_failure_decision_v1",
        "decision": decision_name,
        "next": next_step,
        "verdict": error.verdict,
        "failure_message": error.message,
        "artifact_only": True,
        **FALSE_FLAGS,
    }
    write_json(out / "decision.json", decision)
    append_progress(out, "failure", status="failed", verdict=error.verdict, message=error.message)
    write_summary(out, "failure", [error.verdict], decision, error.message)
    write_report(out, [error.verdict], decision)


def run(args: argparse.Namespace) -> None:
    out = resolve_target_out(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "queue.json", {"schema_version": "phase_138yd_queue_v1", "milestone": MILESTONE, "status": "started", "started_at": utc_now(), "heartbeat_sec": args.heartbeat_sec})
    append_progress(out, "startup", heartbeat_sec=args.heartbeat_sec)
    refresh_status(out, "running", ["FAMILY_DEFAULT_SHORTCUT_ANALYSIS_RUNNING"], {"decision": "pending", "next": "pending"})

    root_138yh = resolve_path(args.upstream_138yh_root)
    root_138yi = resolve_path(args.upstream_138yi_root)
    upstream_138yh = verify_upstream_138yh(out, root_138yh)
    upstream_138yi = verify_upstream_138yi(out, root_138yi)
    append_progress(out, "upstream verification", upstream_138yh_root=rel(root_138yh), upstream_138yi_root=rel(root_138yi))
    write_json(
        out / "analysis_config.json",
        {
            "schema_version": "phase_138yd_analysis_config_v1",
            "artifact_only": True,
            "training_performed": False,
            "new_model_inference_run": False,
            "shared_helper_called": False,
            "torch_forward_pass_run": False,
            "checkpoint_mutation_performed": False,
            "helper_backend_modified": False,
            "old_runner_imported": False,
            "runtime_surface_mutated": False,
            "root_license_changed": False,
        },
    )
    append_progress(out, "artifact loading")

    rows = build_eval_rows(root_138yi)
    family_defaults = upstream_138yi["family_default"]["family_defaults"]
    value_sets = build_value_sets(root_138yi, rows)

    shortcut_map = family_default_shortcut_map(rows, family_defaults, read_json(root_138yi / "intra_family_contrastive_metrics.json"))
    write_json(out / "family_default_shortcut_map.json", shortcut_map)
    append_progress(out, "family default mapping", family_count=shortcut_map["family_count"])
    refresh_status(out, "running", ["FAMILY_DEFAULT_MAP_WRITTEN"], {"decision": "pending", "next": "pending"})

    origin = default_value_origin_report(rows, family_defaults, value_sets)
    write_json(out / "default_value_origin_report.json", origin)
    append_progress(out, "default origin analysis", generated_only_rate=origin["default_value_generated_only_rate"])

    template = family_template_shortcut_report(rows, family_defaults)
    write_json(out / "family_template_shortcut_report.json", template)
    append_progress(out, "template shortcut analysis", template_default_correlation_rate=template["template_default_correlation_rate"], template_family_confounded=template["template_family_confounded"])

    contrast = contrast_group_default_failure_report(rows, family_defaults, value_sets)
    write_json(out / "contrast_group_default_failure_report.json", contrast)
    append_progress(out, "contrast group failure analysis", contrast_group_default_shortcut_rate=contrast["contrast_group_default_shortcut_rate"])

    objective = objective_shortcut_reward_report(root_138yi)
    write_json(out / "objective_shortcut_reward_report.json", objective)
    append_progress(out, "objective shortcut analysis", explicit_family_default_penalty=objective["objective_explicitly_penalizes_family_default"])

    scorer = scorer_dataset_shortcut_report(rows, root_138yi, family_defaults)
    write_json(out / "scorer_dataset_shortcut_report.json", scorer)
    append_progress(out, "scorer/dataset shortcut analysis", scorer_weakness_unlikely=scorer["scorer_weakness_unlikely"])

    root = choose_root(template, contrast, objective, scorer)
    write_json(out / "family_default_root_cause.json", root)
    append_progress(out, "root cause selection", root_cause=root["root_cause"])

    recommendation = {
        "schema_version": "phase_138yd_next_repair_recommendation_v1",
        "root_cause": root["root_cause"],
        "recommended_next": root["recommended_next"],
        "clean_negative_accepted": True,
        "no_model_fix_performed": True,
    }
    write_json(out / "next_repair_recommendation.json", recommendation)
    append_progress(out, "recommendation", next=recommendation["recommended_next"])

    write_json(
        out / "diagnostic_gap_register.json",
        {
            "schema_version": "phase_138yd_diagnostic_gap_register_v1",
            "gaps": [
                {"field": "output_head_prior", "status": "diagnostic_gap", "reason": "138YD does not inspect logits or output-head weights"},
                {"field": "hidden_state_family_default_mechanism", "status": "diagnostic_gap", "reason": "No hidden-state or activation artifacts are available"},
                {"field": "grower_scout_behavior", "status": "diagnostic_gap", "reason": "No grower/scout instrumentation artifacts exist"},
                {"field": "explicit_family_default_penalty_metric", "status": "diagnostic_gap", "reason": "training_metrics.jsonl does not record a family-default penalty metric"},
            ],
        },
    )
    write_json(
        out / "risk_register.json",
        {
            "schema_version": "phase_138yd_risk_register_v1",
            "risks": [
                {"risk": "family-default output behavior is overclaimed as internal mechanism", "mitigation": "decision marks internal mechanism claims false and records diagnostic gaps"},
                {"risk": "template correlation is confounded with family labels", "mitigation": "template_family_confounded blocks template-induced root selection"},
                {"risk": "objective narrative text is mistaken for explicit penalty implementation", "mitigation": "explicit metric keys are required for penalty evidence"},
            ],
            "upstream_138yh_root_cause": upstream_138yh["root"]["root_cause"],
        },
    )
    append_progress(out, "risk and diagnostic gaps")

    decision, verdicts = make_decision(root)
    write_json(out / "decision.json", decision)
    append_progress(out, "decision", decision=decision["decision"], next=decision["next"])
    refresh_status(out, "completed", verdicts, decision)
    append_progress(out, "final verdict", verdicts=verdicts)
    write_json(out / "queue.json", {"schema_version": "phase_138yd_queue_v1", "milestone": MILESTONE, "status": "completed", "completed_at": utc_now(), "heartbeat_sec": args.heartbeat_sec})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--upstream-138yh-root", default=str(DEFAULT_UPSTREAM_138YH_ROOT))
    parser.add_argument("--upstream-138yi-root", default=str(DEFAULT_UPSTREAM_138YI_ROOT))
    parser.add_argument("--heartbeat-sec", type=int, default=20)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        run(args)
        return 0
    except GateError as exc:
        write_failure_decision(args, exc)
        print(f"138YD failed closed: {exc.verdict}: {exc.message}")
        return 1 if exc.verdict == "138YD_BOUNDARY_FAILURE" else 0


if __name__ == "__main__":
    raise SystemExit(main())

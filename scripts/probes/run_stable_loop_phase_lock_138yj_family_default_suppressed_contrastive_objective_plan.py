#!/usr/bin/env python3
"""138YJ artifact-only family-default-suppressed objective plan.

This phase reads existing 138YD/138YH/138YI artifacts only. It does not train,
repair, run inference, call the shared helper, run torch forward passes, mutate
checkpoints, import old runners, start services, deploy, delete files, or modify
runtime/release/product surfaces.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
MILESTONE = "STABLE_LOOP_PHASE_LOCK_138YJ_FAMILY_DEFAULT_SUPPRESSED_CONTRASTIVE_OBJECTIVE_PLAN"
DEFAULT_OUT = Path("target/pilot_wave/stable_loop_phase_lock_138yj_family_default_suppressed_contrastive_objective_plan/smoke")
DEFAULT_UPSTREAM_138YD_ROOT = Path("target/pilot_wave/stable_loop_phase_lock_138yd_family_default_shortcut_analysis/smoke")
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
    "138YJ is artifact-only planning for a family-default-suppressed contrastive "
    "objective. It reads existing 138YD/138YH/138YI artifacts only and does not "
    "train, repair, run new inference, call shared_raw_generation_helper.py, run "
    "torch forward passes, mutate checkpoints, modify helper/backend code, import "
    "old runners, delete or consolidate files, start services, deploy, modify "
    "runtime/service/deploy/product/release surfaces, modify docs/product or "
    "docs/releases, modify SDK exports, or change root LICENSE."
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
REQUIRED_138YD_ARTIFACTS = [
    "decision.json",
    "summary.json",
    "family_default_shortcut_map.json",
    "default_value_origin_report.json",
    "family_template_shortcut_report.json",
    "contrast_group_default_failure_report.json",
    "objective_shortcut_reward_report.json",
    "scorer_dataset_shortcut_report.json",
    "family_default_root_cause.json",
    "next_repair_recommendation.json",
    "diagnostic_gap_register.json",
]
REQUIRED_138YH_ARTIFACTS = [
    "decision.json",
    "replay_value_extraction_report.json",
    "train_value_frequency_report.json",
    "replay_rank_report.json",
    "family_replay_shape_report.json",
    "contrast_group_replay_report.json",
    "objective_reward_artifact_report.json",
    "scorer_dataset_artifact_report.json",
    "root_cause_report.json",
]
REQUIRED_138YI_ARTIFACTS = [
    "decision.json",
    "aggregate_metrics.json",
    "family_default_attractor_report.json",
    "high_frequency_value_replay_report.json",
    "intra_family_contrastive_metrics.json",
    "value_grounding_metrics.json",
    "parrot_trap_report.json",
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
    "train_config.json",
    "eval_config.json",
]


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
        raise GateError("138YJ_BOUNDARY_FAILURE", "--out must stay inside repo") from exc
    parts = [part.lower() for part in relative.parts]
    if len(parts) < 2 or parts[0] != "target" or parts[1] != "pilot_wave":
        raise GateError("138YJ_BOUNDARY_FAILURE", "--out must stay under target/pilot_wave")
    return resolved


def append_progress(out: Path, event: str, status: str = "completed", **details: Any) -> None:
    append_jsonl(out / "progress.jsonl", {"ts": utc_now(), "event": event, "status": status, "details": details})


def require_artifacts(root: Path, names: list[str], verdict: str) -> None:
    missing = [name for name in names if not (root / name).exists()]
    if missing:
        raise GateError(verdict, "required upstream artifacts missing", {"missing": missing})


def write_summary(out: Path, status: str, verdicts: list[str], decision: dict[str, Any], failure: str | None = None) -> None:
    write_json(
        out / "summary.json",
        {
            "schema_version": "phase_138yj_summary_v1",
            "milestone": MILESTONE,
            "status": status,
            "verdicts": verdicts,
            "decision": decision.get("decision"),
            "next": decision.get("next"),
            "failure": failure,
            "boundary": BOUNDARY_TEXT,
            "artifact_only": True,
            "planning_only": True,
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
        "- 138YJ does not fix or train the model.",
        "- It designs 138YK as a family-default-suppressed contrastive repair/probe.",
        "- Family Default Shortcut is observed output behavior, not an internal mechanism claim.",
        "- Scout-First Laziness and Missing Intra-Family Variance remain planning hypotheses unless instrumented.",
        f"- `decision`: `{decision.get('decision')}`",
        f"- `next`: `{decision.get('next')}`",
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


def verify_upstream_138yd(out: Path, root: Path) -> dict[str, Any]:
    require_artifacts(root, REQUIRED_138YD_ARTIFACTS, "UPSTREAM_138YD_ARTIFACT_MISSING")
    decision = read_json(root / "decision.json")
    root_cause = read_json(root / "family_default_root_cause.json")
    contrast = read_json(root / "contrast_group_default_failure_report.json")
    objective = read_json(root / "objective_shortcut_reward_report.json")
    scorer = read_json(root / "scorer_dataset_shortcut_report.json")
    template = read_json(root / "family_template_shortcut_report.json")
    if (
        decision.get("decision") != "family_default_shortcut_analysis_complete"
        or decision.get("root_cause") != "contrastive_objective_too_weak"
        or decision.get("next") != "138YJ_FAMILY_DEFAULT_SUPPRESSED_CONTRASTIVE_OBJECTIVE_PLAN"
    ):
        raise GateError("UPSTREAM_138YD_ARTIFACT_MISSING", "138YD did not route to 138YJ")
    expected = {
        "contrast_group_default_shortcut_rate": 0.78125,
        "multi_expected_to_single_default_rate": 0.6822916666666666,
    }
    mismatches = {key: {"expected": value, "actual": contrast.get(key)} for key, value in expected.items() if contrast.get(key) != value}
    if root_cause.get("eval_expected_value_entropy_by_family_mean") != 6.574545834054494:
        mismatches["eval_expected_value_entropy_by_family_mean"] = root_cause.get("eval_expected_value_entropy_by_family_mean")
    if root_cause.get("family_default_control_failed") is not True:
        mismatches["family_default_control_failed"] = root_cause.get("family_default_control_failed")
    if root_cause.get("template_family_confounded") is not True:
        mismatches["template_family_confounded"] = root_cause.get("template_family_confounded")
    for key in [
        "objective_explicitly_penalizes_family_default",
        "objective_explicitly_penalizes_same_value_for_all_rows",
        "objective_rewards_intra_family_distinct_values",
    ]:
        if objective.get(key) is not False:
            mismatches[key] = objective.get(key)
    if mismatches:
        raise GateError("UPSTREAM_138YD_ARTIFACT_MISSING", "138YD evidence profile mismatch", mismatches)
    manifest = {
        "schema_version": "phase_138yj_upstream_138yd_manifest_v1",
        "upstream_138yd_root": rel(root),
        "verified": True,
        "decision": decision.get("decision"),
        "root_cause": decision.get("root_cause"),
        "next": decision.get("next"),
        "contrast_group_default_shortcut_rate": contrast.get("contrast_group_default_shortcut_rate"),
        "multi_expected_to_single_default_rate": contrast.get("multi_expected_to_single_default_rate"),
        "eval_expected_value_entropy_by_family_mean": root_cause.get("eval_expected_value_entropy_by_family_mean"),
        "family_default_control_failed": root_cause.get("family_default_control_failed"),
        "template_family_confounded": template.get("template_family_confounded"),
        "objective_explicitly_penalizes_family_default": objective.get("objective_explicitly_penalizes_family_default"),
        "objective_explicitly_penalizes_same_value_for_all_rows": objective.get("objective_explicitly_penalizes_same_value_for_all_rows"),
        "objective_rewards_intra_family_distinct_values": objective.get("objective_rewards_intra_family_distinct_values"),
        "scorer_weakness_unlikely": scorer.get("scorer_weakness_unlikely"),
    }
    write_json(out / "upstream_138yd_manifest.json", manifest)
    return {"decision": decision, "root": root_cause, "contrast": contrast, "objective": objective, "scorer": scorer, "template": template}


def verify_upstream_138yh(out: Path, root: Path) -> dict[str, Any]:
    require_artifacts(root, REQUIRED_138YH_ARTIFACTS, "UPSTREAM_138YD_ARTIFACT_MISSING")
    decision = read_json(root / "decision.json")
    ranks = read_json(root / "replay_rank_report.json")
    train_freq = read_json(root / "train_value_frequency_report.json")
    root_cause = read_json(root / "root_cause_report.json")
    if decision.get("decision") != "high_frequency_value_replay_analysis_complete" or decision.get("root_cause") != "family_default_shortcut_replay":
        raise GateError("UPSTREAM_138YD_ARTIFACT_MISSING", "138YH route mismatch")
    zero_fields = [
        "generated_values_top1_global_train_all_rate",
        "generated_values_top5_global_train_all_rate",
        "generated_values_top10_global_train_all_rate",
        "generated_values_top1_family_train_all_rate",
        "generated_values_top5_family_train_all_rate",
        "generated_values_top10_family_train_all_rate",
    ]
    if any(ranks.get(field) != 0.0 for field in zero_fields):
        raise GateError("UPSTREAM_138YD_ARTIFACT_MISSING", "138YH frequency falsification mismatch")
    if train_freq.get("generated_values_seen_in_train_all_rate") != 0.13671875 or root_cause.get("family_default_attractor_rate") != 0.78125:
        raise GateError("UPSTREAM_138YD_ARTIFACT_MISSING", "138YH strict membership/default profile mismatch")
    manifest = {
        "schema_version": "phase_138yj_upstream_138yh_manifest_v1",
        "upstream_138yh_root": rel(root),
        "verified": True,
        "decision": decision.get("decision"),
        "root_cause": decision.get("root_cause"),
        "global_top1_train_all_replay_rate": ranks.get("generated_values_top1_global_train_all_rate"),
        "global_top5_train_all_replay_rate": ranks.get("generated_values_top5_global_train_all_rate"),
        "global_top10_train_all_replay_rate": ranks.get("generated_values_top10_global_train_all_rate"),
        "family_top1_train_all_replay_rate": ranks.get("generated_values_top1_family_train_all_rate"),
        "family_top5_train_all_replay_rate": ranks.get("generated_values_top5_family_train_all_rate"),
        "family_top10_train_all_replay_rate": ranks.get("generated_values_top10_family_train_all_rate"),
        "strict_train_all_membership_rate": train_freq.get("generated_values_seen_in_train_all_rate"),
        "family_default_attractor_rate": root_cause.get("family_default_attractor_rate"),
    }
    write_json(out / "upstream_138yh_manifest.json", manifest)
    return {"decision": decision, "ranks": ranks, "train_freq": train_freq, "root": root_cause}


def verify_upstream_138yi(out: Path, root: Path) -> dict[str, Any]:
    require_artifacts(root, REQUIRED_138YI_ARTIFACTS, "UPSTREAM_138YD_ARTIFACT_MISSING")
    decision = read_json(root / "decision.json")
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
    if decision.get("verdict") != "FAMILY_SPECIFIC_VALUE_ATTRACTOR_REPAIR_FAILS" or decision.get("decision") != "high_frequency_train_value_replay_detected":
        raise GateError("UPSTREAM_138YD_ARTIFACT_MISSING", "138YI result mismatch")
    expected = {
        "answer_value_accuracy": 0.0,
        "exact_answer_accuracy": 0.0,
        "intra_family_contrastive_accuracy": 0.0,
        "family_default_attractor_rate": 0.78125,
        "parrot_trap_detected": False,
        "stale_chat_fragment_rate": 0.0,
        "train_namespace_leak_rate": 0.0,
    }
    mismatches = {key: {"expected": value, "actual": aggregate.get(key)} for key, value in expected.items() if aggregate.get(key) != value}
    if aggregate.get("intra_family_mode_collapse_rate") != 0.9427083333333334:
        mismatches["intra_family_mode_collapse_rate"] = aggregate.get("intra_family_mode_collapse_rate")
    if family_default.get("family_default_shortcut_detected") is not True or high_frequency.get("high_frequency_train_value_replay_detected") is not True:
        mismatches["shortcut_flags"] = {"family_default": family_default.get("family_default_shortcut_detected"), "high_frequency": high_frequency.get("high_frequency_train_value_replay_detected")}
    if mismatches:
        raise GateError("UPSTREAM_138YD_ARTIFACT_MISSING", "138YI metric profile mismatch", mismatches)
    if canary.get("expected_output_canary_passed") is not True or scan.get("ast_shortcut_scan_passed") is not True:
        raise GateError("RAW_HELPER_INTEGRITY_FAILURE", "138YI canary/AST failed")
    if controls.get("controls_failed") is not True or leakage.get("leakage_rejected") is not True or replay.get("determinism_replay_passed") is not True:
        raise GateError("UPSTREAM_138YD_ARTIFACT_MISSING", "138YI controls/leakage/determinism mismatch")
    if before.get("generated_text_produced_before_scoring") is not True or source.get("source_checkpoint_unchanged") is not True or target.get("target_checkpoint_changed") is not True:
        raise GateError("UPSTREAM_138YD_ARTIFACT_MISSING", "138YI generation/checkpoint integrity mismatch")
    for trace in traces:
        request = trace.get("helper_request", {})
        if set(request) != ALLOWED_HELPER_KEYS or set(request) & FORBIDDEN_HELPER_KEYS:
            raise GateError("RAW_HELPER_INTEGRITY_FAILURE", "138YI helper request metadata violation")
    manifest = {
        "schema_version": "phase_138yj_upstream_138yi_manifest_v1",
        "upstream_138yi_root": rel(root),
        "verified": True,
        "verdict": decision.get("verdict"),
        "decision": decision.get("decision"),
        "answer_value_accuracy": aggregate.get("answer_value_accuracy"),
        "exact_answer_accuracy": aggregate.get("exact_answer_accuracy"),
        "intra_family_contrastive_accuracy": aggregate.get("intra_family_contrastive_accuracy"),
        "intra_family_mode_collapse_rate": aggregate.get("intra_family_mode_collapse_rate"),
        "family_default_attractor_rate": aggregate.get("family_default_attractor_rate"),
        "family_default_shortcut_detected": family_default.get("family_default_shortcut_detected"),
        "high_frequency_train_value_replay_detected": high_frequency.get("high_frequency_train_value_replay_detected"),
        "parrot_trap_detected": aggregate.get("parrot_trap_detected"),
        "stale_chat_fragment_rate": aggregate.get("stale_chat_fragment_rate"),
        "train_namespace_leak_rate": aggregate.get("train_namespace_leak_rate"),
        "determinism_replay_passed": replay.get("determinism_replay_passed"),
        "source_checkpoint_unchanged": source.get("source_checkpoint_unchanged"),
        "target_checkpoint_changed": target.get("target_checkpoint_changed"),
        "generated_text_before_scoring": before.get("generated_text_produced_before_scoring"),
        "no_expected_or_scorer_metadata_reached_helper_requests": True,
    }
    write_json(out / "upstream_138yi_manifest.json", manifest)
    return {"decision": decision, "aggregate": aggregate, "family_default": family_default, "high_frequency": high_frequency}


def build_family_default_value_bank(root_138yi: Path) -> dict[str, Any]:
    scoring = read_jsonl(root_138yi / "scoring_results.jsonl")
    rows_by_family: dict[str, Counter[str]] = defaultdict(Counter)
    for row in scoring:
        value = row.get("answer_value_candidate")
        expected = row.get("expected_value")
        if value and value != expected:
            rows_by_family[row["family"]][value] += 1
    bank: dict[str, list[dict[str, Any]]] = {}
    for family, counter in sorted(rows_by_family.items()):
        total = sum(counter.values())
        bank[family] = [
            {"value": value, "count": count, "rate": count / total if total else 0.0, "rank": index + 1}
            for index, (value, count) in enumerate(sorted(counter.items(), key=lambda item: (-item[1], item[0])))
        ]
    return {"schema_version": "phase_138yj_family_default_value_bank_v1", "families": bank}


def write_planning_artifacts(out: Path, root_138yd: Path, root_138yh: Path, root_138yi: Path, yd: dict[str, Any], yh: dict[str, Any], yi: dict[str, Any]) -> dict[str, Any]:
    yd_root = yd["root"]
    yd_contrast = yd["contrast"]
    yd_objective = yd["objective"]
    yi_agg = yi["aggregate"]
    family_bank = build_family_default_value_bank(root_138yi)
    family_default_bank_fields = {
        "family_default_value_bank": family_bank,
        "per_family_forbidden_default_values": {family: [item["value"] for item in entries[:5]] for family, entries in family_bank["families"].items()},
    }
    failure_summary = {
        "schema_version": "phase_138yj_family_default_failure_summary_v1",
        "root_cause": "contrastive_objective_too_weak",
        "contrast_group_default_shortcut_rate": yd_contrast["contrast_group_default_shortcut_rate"],
        "multi_expected_to_single_default_rate": yd_contrast["multi_expected_to_single_default_rate"],
        "family_default_attractor_rate": yi_agg["family_default_attractor_rate"],
        "intra_family_mode_collapse_rate": yi_agg["intra_family_mode_collapse_rate"],
        "answer_value_accuracy": yi_agg["answer_value_accuracy"],
        "exact_answer_accuracy": yi_agg["exact_answer_accuracy"],
        "intra_family_contrastive_accuracy": yi_agg["intra_family_contrastive_accuracy"],
        "dataset_low_intra_family_value_diversity_selected": False,
        "scorer_family_default_weakness_selected": False,
        "global_train_frequency_replay_selected": False,
        "prompt_copy_parrot_trap": False,
        "stale_chat_failure": False,
        "train_namespace_failure": False,
        "evidence_source": "138YD/138YH/138YI artifacts",
    }
    write_json(out / "family_default_failure_summary.json", failure_summary)
    append_progress(out, "family default failure summary", root_cause=failure_summary["root_cause"])

    weakness = {
        "schema_version": "phase_138yj_objective_weakness_diagnosis_v1",
        "current_contrastive_objective_created_same_family_groups": True,
        "same_family_groups_had_diverse_expected_values": True,
        "model_still_collapsed_to_family_defaults": True,
        "scorer_controls_did_not_accept_family_default_cheating": True,
        "objective_pressure_against_family_default_reuse_insufficient": True,
        "objective_currently_penalizes_family_default": yd_objective["objective_explicitly_penalizes_family_default"],
        "objective_currently_penalizes_same_value_for_distinct_expected_values": yd_objective["objective_explicitly_penalizes_same_value_for_all_rows"],
        "objective_currently_rewards_intra_family_value_diversity": yd_objective["objective_rewards_intra_family_distinct_values"],
        "objective_currently_rewards_derived_ood_values": True,
        "objective_currently_penalizes_family_level_format_only": False,
        "missing_or_weak_pressure_terms": [
            "family_default_reuse_penalty",
            "same_value_for_distinct_expected_values_penalty",
            "intra_family_value_diversity_reward",
            "hard_negative_default_rows",
            "family_default_shortcut_control_as_decision_gate",
        ],
        "diagnostic_gaps": [
            "output_head_prior",
            "hidden_state_carrier",
            "grower_scout_behavior",
            "topological_inhibition",
        ],
    }
    write_json(out / "objective_weakness_diagnosis.json", weakness)
    append_progress(out, "objective weakness diagnosis", missing_terms=len(weakness["missing_or_weak_pressure_terms"]))

    suppression = {
        "schema_version": "phase_138yj_family_default_suppression_requirements_v1",
        **family_default_bank_fields,
        "required_suppression_mechanisms": [
            "family_default_value_bank",
            "per_family_forbidden_default_values",
            "same_family_different_value_pairs",
            "same_family_different_value_groups",
            "hard_negative_default_rows",
            "family_default_reuse_penalty",
            "multi_expected_to_single_generated_penalty",
            "dominant_wrong_value_penalty",
            "family_default_shortcut_control",
            "same_value_for_all_rows_control",
        ],
        "required_138yk_metrics": [
            "family_default_attractor_rate",
            "family_default_reuse_rate",
            "family_dominant_wrong_value_rate",
            "multi_expected_to_single_default_rate",
            "same_value_for_all_rows_rate",
            "intra_family_mode_collapse_rate",
            "intra_family_unique_correct_value_rate",
            "intra_family_contrastive_accuracy",
            "family_default_control_passed_or_failed",
            "family_default_shortcut_detected",
        ],
    }
    write_json(out / "family_default_suppression_requirements.json", suppression)
    append_progress(out, "suppression requirement drafting")

    contrast_design = {
        "schema_version": "phase_138yj_strengthened_contrast_group_design_v1",
        "group_requirements": [
            "same family",
            "same template shape",
            "same wrapper",
            "different expected values",
            "different distractors",
            "different table/rule bindings",
            "at least one hard negative default value from prior failure artifacts",
            "held-out expected values",
            "held-out family/value combinations where possible",
        ],
        "group_pass_requires": [
            "all rows emit ANSWER=E",
            "all rows emit correct distinct values",
            "no row emits family default",
            "no row emits dominant wrong value",
            "no row emits high-frequency train value",
            "no row emits peer expected value incorrectly",
            "no row emits prompt-copy-only shortcut",
            "no train namespace leak",
            "no stale chat",
            "no post-generation repair",
        ],
        "hard_negative_bank_source": "family_default_suppression_requirements.family_default_value_bank",
    }
    write_json(out / "strengthened_contrast_group_design.json", contrast_design)
    append_progress(out, "contrast group redesign")

    hard_negative_policy = {
        "schema_version": "phase_138yj_hard_negative_family_default_policy_v1",
        "default_value_bank_construction": "Build from 138YD/138YI generated wrong values ranked per family.",
        "default_value_suppression_during_training_objective": "Penalize generated values that match per-family forbidden defaults.",
        "default_value_hard_negative_eval_rows": "Each eval row carries forbidden_family_default_values from its family bank.",
        "default_value_scorer_controls": ["FAMILY_DEFAULT_VALUE_CONTROL", "SAME_VALUE_FOR_ALL_ROWS_CONTROL"],
        "default_value_replay_failure_routes": {
            "family_default_shortcut_persists": "138YD_FAMILY_DEFAULT_SHORTCUT_ANALYSIS",
            "contrastive_objective_still_too_weak": "138YJ_CONTRASTIVE_OBJECTIVE_WEAKNESS_REVIEW",
        },
        "row_failure_rule": "If generated value equals a forbidden family default, fail row even when wrapper is correct.",
        "group_failure_rule": "If a forbidden family default appears across multiple distinct expected values, fail group and increment family_default_reuse_rate.",
    }
    write_json(out / "hard_negative_family_default_policy.json", hard_negative_policy)
    append_progress(out, "hard negative policy")

    anti_shortcut = {
        "schema_version": "phase_138yj_anti_shortcut_requirements_v1",
        "reject": [
            "family-level format success only",
            "ANSWER=E prefix-only success",
            "namespace-only success",
            "train-loss-only success",
            "teacher-forcing-only success",
            "prompt-copy-only success",
            "family default value replay",
            "high-frequency train value replay",
            "target checkpoint changed only",
            "expected-output construction",
            "old runner imports",
            "oracle/rerank/verifier/LLM judge",
            "constrained decoding",
            "JSON mode",
            "regex fixer",
            "post-generation repair",
            "retry loop",
            "best-of-n",
            "threshold weakening",
        ],
    }
    write_json(out / "anti_shortcut_requirements.json", anti_shortcut)
    append_progress(out, "anti-shortcut design")

    training_spec = {
        "schema_version": "phase_138yj_target_138yk_training_objective_spec_v1",
        "milestone": "138YK_FAMILY_DEFAULT_SUPPRESSED_CONTRASTIVE_REPAIR_PROBE",
        "objective_rewards": [
            "correct value after ANSWER=E",
            "same-family distinct correct values",
            "rule-derived correct value",
            "table-derived correct value",
            "composition-derived correct value",
            "OOD symbol correct value",
            "value-after-prefix correctness",
            "reduction of family-default reuse",
        ],
        "objective_penalizes": [
            "family default wrong value",
            "dominant wrong value",
            "same generated value for multiple distinct expected values",
            "high-frequency train value replay",
            "prompt-copy-only success",
            "prefix-only success",
            "namespace-only success",
            "train-loss-only success",
            "teacher-forcing-only success",
            "stale chat",
            "train namespace leak",
            "off-prompt continuation",
        ],
        "positive_cannot_depend_on": [
            "train loss alone",
            "target checkpoint changed alone",
            "family-level classification alone",
            "wrapper correctness alone",
            "namespace correctness alone",
        ],
    }
    write_json(out / "target_138yk_training_objective_spec.json", training_spec)

    eval_spec = {
        "schema_version": "phase_138yj_target_138yk_eval_gate_spec_v1",
        "final_eval_helper_only": "scripts/probes/shared_raw_generation_helper.py",
        "generated_text_before_scoring": True,
        "required_integrity_gates": [
            "expected-output canary",
            "AST shortcut scan over helper/runner/checker",
            "helper provenance verification",
            "checkpoint hash verification",
            "leakage audit",
            "scorer controls",
            "deterministic replay",
            "source checkpoint unchanged",
            "target checkpoint under target only",
            "no expected/scorer metadata in helper requests",
        ],
        "recommended_positive_gates": {
            "answer_value_accuracy": ">= 0.25",
            "exact_answer_accuracy": ">= 0.20",
            "intra_family_contrastive_accuracy": ">= 0.30",
            "intra_family_unique_correct_value_rate": ">= 0.25",
            "family_default_attractor_rate": "<= 0.35",
            "family_default_reuse_rate": "<= 0.35",
            "multi_expected_to_single_default_rate": "<= 0.30",
            "same_value_for_all_rows_rate": "<= 0.20",
            "rule_derived_value_accuracy": ">= 0.20",
            "table_derived_value_accuracy": ">= 0.20",
            "ood_symbol_value_accuracy": ">= 0.15",
            "family_default_shortcut_detected": False,
            "high_frequency_train_value_replay_detected": False,
            "parrot_trap_detected": False,
        },
        "every_seed_must_independently_pass": True,
    }
    write_json(out / "target_138yk_eval_gate_spec.json", eval_spec)

    failure_routes = {
        "schema_version": "phase_138yj_target_138yk_failure_routes_v1",
        "clean_negative_routes": {
            "no_value_improvement": "138YK_FAILURE_ANALYSIS",
            "family_default_shortcut_persists": "138YD_FAMILY_DEFAULT_SHORTCUT_ANALYSIS",
            "contrastive_objective_still_too_weak": "138YJ_CONTRASTIVE_OBJECTIVE_WEAKNESS_REVIEW",
            "high_frequency_value_replay_detected": "138YH_HIGH_FREQUENCY_VALUE_REPLAY_ANALYSIS",
            "parrot_trap_copy_shortcut_detected": "138P_PARROT_TRAP_VALUE_COPY_ANALYSIS",
            "stale_chat_rollout_failure": "138S_STALE_CHAT_ROLLOUT_FAILURE_ANALYSIS",
            "namespace_rollout_failure": "138S_NAMESPACE_ROLLOUT_FAILURE_ANALYSIS",
            "nondeterministic_probe": "138N_DETERMINISM_FAILURE_ANALYSIS",
            "eval_leakage": "138L_FAMILY_CONTRASTIVE_EVAL_LEAKAGE_REDESIGN",
            "raw_helper_integrity_failure": "135G_RAW_HELPER_ORACLE_SHORTCUT_REMOVAL",
        },
        "clean_negative_accepted": True,
        "threshold_weakening_forbidden": True,
    }
    write_json(out / "target_138yk_failure_routes.json", failure_routes)

    milestone = {
        "schema_version": "phase_138yj_next_138yk_milestone_plan_v1",
        "milestone": "138YK_FAMILY_DEFAULT_SUPPRESSED_CONTRASTIVE_REPAIR_PROBE",
        "type": "targeted repair/probe",
        "train_allowed": True,
        "target_checkpoint_under_target_only": True,
        "source_checkpoint_mutation_allowed": False,
        "helper_modification_allowed": False,
        "old_runner_import_allowed": False,
        "service_deploy_allowed": False,
        "clean_negative_accepted": True,
        "shared_raw_generation_helper_only_for_final_eval": True,
        "generated_text_before_scoring_required": True,
        "family_default_value_bank_required": True,
        "hard_negative_default_rows_required": True,
        "deterministic_replay_required": True,
        "required_artifacts": [
            "queue.json",
            "progress.jsonl",
            "upstream_138yj_manifest.json",
            "upstream_138yd_manifest.json",
            "determinism_manifest.json",
            "train_config.json",
            "eval_config.json",
            "source_checkpoint_integrity_manifest.json",
            "target_checkpoint_integrity_manifest.json",
            "helper_provenance_verification.json",
            "forbidden_input_rejection_report.json",
            "expected_output_canary_report.json",
            "ast_shortcut_scan_report.json",
            "train_dataset_manifest.json",
            "eval_dataset_manifest.json",
            "train_rows.jsonl",
            "eval_rows.jsonl",
            "eval_row_hashes.json",
            "family_default_value_bank.json",
            "hard_negative_default_rows.jsonl",
            "contrast_group_manifest.json",
            "ood_family_value_manifest.json",
            "freshness_leakage_audit.json",
            "training_metrics.jsonl",
            "training_objective_report.json",
            "raw_generation_trace.jsonl",
            "raw_generation_results.jsonl",
            "scoring_results.jsonl",
            "contrast_group_results.jsonl",
            "family_default_suppression_metrics.json",
            "intra_family_contrastive_metrics.json",
            "value_grounding_metrics.json",
            "parrot_trap_report.json",
            "family_default_attractor_report.json",
            "high_frequency_value_replay_report.json",
            "control_results.jsonl",
            "control_arm_report.json",
            "generated_before_scoring_report.json",
            "per_family_metrics.json",
            "per_seed_metrics.jsonl",
            "aggregate_metrics.json",
            "determinism_replay_report.json",
            "failure_case_samples.jsonl",
            "human_readable_samples.jsonl",
            "evidence_rebuild_status.json",
            "decision.json",
            "summary.json",
            "report.md",
        ],
    }
    write_json(out / "next_138yk_milestone_plan.json", milestone)
    append_progress(out, "target 138YK plan writing")
    return {"failure_summary": failure_summary, "milestone": milestone}


def make_decision() -> tuple[dict[str, Any], list[str]]:
    decision = {
        "schema_version": "phase_138yj_decision_v1",
        "decision": "family_default_suppressed_contrastive_objective_plan_complete",
        "next": "138YK_FAMILY_DEFAULT_SUPPRESSED_CONTRASTIVE_REPAIR_PROBE",
        "verdict": "FAMILY_DEFAULT_SUPPRESSED_CONTRASTIVE_OBJECTIVE_PLAN_COMPLETE",
        "artifact_only": True,
        "planning_only": True,
        "training_performed": False,
        "new_model_inference_run": False,
        "shared_helper_called": False,
        "torch_forward_pass_run": False,
        "checkpoint_mutation_performed": False,
        "internal_mechanism_claimed": False,
        **FALSE_FLAGS,
    }
    verdicts = [
        decision["verdict"],
        "ARTIFACT_ONLY_PLANNING",
        "NEXT_138YK_PLAN_MACHINE_READABLE",
        "FAMILY_DEFAULT_HARD_NEGATIVES_REQUIRED",
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
    elif error.verdict == "FAMILY_DEFAULT_OBJECTIVE_EVIDENCE_RECHECK":
        decision_name, next_step = "family_default_objective_evidence_recheck", "138YJ_FAMILY_DEFAULT_OBJECTIVE_EVIDENCE_RECHECK"
    else:
        decision_name, next_step = "upstream_138yd_artifact_missing", "138YJ_UPSTREAM_138YD_ARTIFACT_MISSING"
    decision = {
        "schema_version": "phase_138yj_failure_decision_v1",
        "decision": decision_name,
        "next": next_step,
        "verdict": error.verdict,
        "failure_message": error.message,
        "artifact_only": True,
        "planning_only": True,
        **FALSE_FLAGS,
    }
    write_json(out / "decision.json", decision)
    append_progress(out, "failure", status="failed", verdict=error.verdict, message=error.message)
    write_summary(out, "failure", [error.verdict], decision, error.message)
    write_report(out, [error.verdict], decision)


def run(args: argparse.Namespace) -> None:
    out = resolve_target_out(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "queue.json", {"schema_version": "phase_138yj_queue_v1", "milestone": MILESTONE, "status": "started", "started_at": utc_now(), "heartbeat_sec": args.heartbeat_sec})
    append_progress(out, "startup", heartbeat_sec=args.heartbeat_sec)
    refresh_status(out, "running", ["FAMILY_DEFAULT_SUPPRESSED_OBJECTIVE_PLAN_RUNNING"], {"decision": "pending", "next": "pending"})

    root_138yd = resolve_path(args.upstream_138yd_root)
    root_138yh = resolve_path(args.upstream_138yh_root)
    root_138yi = resolve_path(args.upstream_138yi_root)
    yd = verify_upstream_138yd(out, root_138yd)
    yh = verify_upstream_138yh(out, root_138yh)
    yi = verify_upstream_138yi(out, root_138yi)
    append_progress(out, "upstream verification", upstream_138yd_root=rel(root_138yd), upstream_138yh_root=rel(root_138yh), upstream_138yi_root=rel(root_138yi))
    write_json(
        out / "analysis_config.json",
        {
            "schema_version": "phase_138yj_analysis_config_v1",
            "artifact_only": True,
            "planning_only": True,
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
    planning = write_planning_artifacts(out, root_138yd, root_138yh, root_138yi, yd, yh, yi)

    write_json(
        out / "diagnostic_gap_register.json",
        {
            "schema_version": "phase_138yj_diagnostic_gap_register_v1",
            "gaps": [
                {"field": "output_head_prior", "status": "diagnostic_gap", "reason": "138YJ does not inspect logits or output-head weights"},
                {"field": "hidden_state_carrier", "status": "diagnostic_gap", "reason": "No hidden-state or activation artifacts are available"},
                {"field": "grower_scout_behavior", "status": "diagnostic_gap", "reason": "Scout/grower concepts are planning hypotheses only"},
                {"field": "topological_inhibition", "status": "diagnostic_gap", "reason": "No graph/topology instrumentation artifacts exist"},
            ],
        },
    )
    write_json(
        out / "risk_register.json",
        {
            "schema_version": "phase_138yj_risk_register_v1",
            "risks": [
                {"risk": "138YK repeats vanilla contrastive objective", "mitigation": "plan requires family default value bank and hard negative rows"},
                {"risk": "family-default shortcut is overclaimed as internal mechanism", "mitigation": "diagnostic gaps forbid internal mechanism claims"},
                {"risk": "positive is obtained from wrapper/namespace only", "mitigation": "positive gates require value, exact, contrastive, and default suppression metrics"},
            ],
            "planned_next": planning["milestone"]["milestone"],
        },
    )

    decision, verdicts = make_decision()
    write_json(out / "decision.json", decision)
    append_progress(out, "decision", decision=decision["decision"], next=decision["next"])
    refresh_status(out, "completed", verdicts, decision)
    append_progress(out, "final verdict", verdicts=verdicts)
    write_json(out / "queue.json", {"schema_version": "phase_138yj_queue_v1", "milestone": MILESTONE, "status": "completed", "completed_at": utc_now(), "heartbeat_sec": args.heartbeat_sec})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--upstream-138yd-root", default=str(DEFAULT_UPSTREAM_138YD_ROOT))
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
        print(f"138YJ failed closed: {exc.verdict}: {exc.message}")
        return 1 if exc.verdict == "138YJ_BOUNDARY_FAILURE" else 0


if __name__ == "__main__":
    raise SystemExit(main())

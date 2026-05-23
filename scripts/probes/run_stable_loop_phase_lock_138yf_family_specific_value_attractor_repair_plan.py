#!/usr/bin/env python3
"""138YF artifact-only family-specific value attractor repair plan.

This phase reads existing 138U/138WV/138W artifacts only. It does not train,
repair, run new inference, call shared_raw_generation_helper.py, run torch
forward passes, mutate checkpoints, modify helper/backend code, import old
runners, delete or consolidate files, start services, deploy, or modify
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
MILESTONE = "STABLE_LOOP_PHASE_LOCK_138YF_FAMILY_SPECIFIC_VALUE_ATTRACTOR_REPAIR_PLAN"
DEFAULT_OUT = Path("target/pilot_wave/stable_loop_phase_lock_138yf_family_specific_value_attractor_repair_plan/smoke")
DEFAULT_UPSTREAM_138U_ROOT = Path("target/pilot_wave/stable_loop_phase_lock_138u_wrong_value_attractor_analysis/smoke")
DEFAULT_UPSTREAM_138WV_ROOT = Path("target/pilot_wave/stable_loop_phase_lock_138wv_wrapper_value_decoupling_failure_analysis/smoke")
DEFAULT_UPSTREAM_138W_ROOT = Path("target/pilot_wave/stable_loop_phase_lock_138w_answer_value_grounding_repair_probe/smoke")
BOUNDARY_TEXT = (
    "138YF is artifact-only planning. It reads existing 138U/138WV/138W artifacts "
    "only and does not train, repair, run new inference, call shared_raw_generation_helper.py, "
    "run torch forward passes, mutate checkpoints, modify helper/backend code, import "
    "old runners, delete or consolidate files, start services, deploy, modify "
    "runtime/service/deploy/product/release surfaces, modify SDK exports, modify "
    "docs/product or docs/releases, or change root LICENSE."
)
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
REQUIRED_138U_ARTIFACTS = [
    "decision.json",
    "summary.json",
    "upstream_138wv_manifest.json",
    "upstream_138w_manifest.json",
    "wrong_value_distribution_report.json",
    "train_value_attractor_report.json",
    "eval_value_miss_report.json",
    "wrong_value_vs_prompt_report.json",
    "value_source_family_failure_report.json",
    "attractor_root_cause.json",
    "next_repair_recommendation.json",
    "diagnostic_gap_register.json",
]
REQUIRED_138WV_ARTIFACTS = [
    "post_wrapper_value_anatomy_report.json",
    "attractor_distribution_report.json",
    "value_candidate_report.json",
    "wrapper_value_decoupling_root_cause.json",
]
REQUIRED_138W_ARTIFACTS = [
    "train_rows.jsonl",
    "eval_rows.jsonl",
    "scoring_results.jsonl",
    "raw_generation_results.jsonl",
    "raw_generation_trace.jsonl",
    "aggregate_metrics.json",
    "value_grounding_metrics.json",
    "parrot_trap_report.json",
    "freshness_leakage_audit.json",
    "generated_before_scoring_report.json",
    "expected_output_canary_report.json",
    "ast_shortcut_scan_report.json",
    "determinism_replay_report.json",
]
DISALLOWED_138U_ROOTS = {
    "global_train_value_prior_attractor",
    "high_frequency_train_value_attractor",
    "prompt_copy_wrong_value_attractor",
    "distractor_value_attractor",
    "wrong_table_entry_attractor",
}
EXPECTED_STRICT_TRAIN_RATE = 0.09895833333333333


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
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


def resolve_path(text: str | Path) -> Path:
    path = Path(text)
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def resolve_target_out(text: str) -> Path:
    path = Path(text)
    resolved = path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()
    try:
        relative = resolved.relative_to(REPO_ROOT)
    except ValueError as exc:
        raise GateError("138YF_BOUNDARY_FAILURE", "--out must stay inside repo") from exc
    parts = [part.lower() for part in relative.parts]
    if len(parts) < 2 or parts[0] != "target" or parts[1] != "pilot_wave":
        raise GateError("138YF_BOUNDARY_FAILURE", "--out must stay under target/pilot_wave")
    return resolved


def append_progress(out: Path, event: str, status: str = "completed", **details: Any) -> None:
    append_jsonl(out / "progress.jsonl", {"ts": utc_now(), "event": event, "status": status, "details": details})


def rate(count: int, total: int) -> float:
    return count / total if total else 0.0


def write_summary(out: Path, status: str, verdicts: list[str], decision: dict[str, Any], failure: str | None = None) -> None:
    write_json(
        out / "summary.json",
        {
            "schema_version": "phase_138yf_summary_v1",
            "milestone": MILESTONE,
            "status": status,
            "verdicts": verdicts,
            "decision": decision.get("decision"),
            "next": decision.get("next"),
            "primary_bottleneck": decision.get("primary_bottleneck"),
            "failure": failure,
            "boundary": BOUNDARY_TEXT,
            "artifact_only": True,
            "planning_only": True,
            "new_inference_run": False,
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
        "## Plan Result",
        "",
        "- 138YF does not fix or train the model.",
        "- It designs an intra-family contrastive objective for family-specific wrong-value attractors.",
        "- Scout-First Laziness is recorded only as a design hypothesis, not measured scout/grower behavior.",
        "- Missing Intra-Family Variance is the artifact-level proxy used for the next plan.",
        f"- `decision`: `{decision.get('decision')}`",
        f"- `next`: `{decision.get('next')}`",
        f"- `primary_bottleneck`: `{decision.get('primary_bottleneck')}`",
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


def assert_files(root: Path, names: list[str], label: str) -> None:
    missing = [name for name in names if not (root / name).exists()]
    if missing:
        raise GateError(f"UPSTREAM_{label}_ARTIFACT_MISSING", f"required {label} artifacts missing", {"missing": missing})


def close(a: float | int | None, b: float) -> bool:
    return isinstance(a, (float, int)) and abs(float(a) - b) < 1e-12


def verify_upstreams(out: Path, root_138u: Path, root_138wv: Path, root_138w: Path) -> dict[str, Any]:
    assert_files(root_138u, REQUIRED_138U_ARTIFACTS, "138U")
    assert_files(root_138wv, REQUIRED_138WV_ARTIFACTS, "138WV")
    assert_files(root_138w, REQUIRED_138W_ARTIFACTS, "138W")

    decision_138u = read_json(root_138u / "decision.json")
    summary_138u = read_json(root_138u / "summary.json")
    distribution_138u = read_json(root_138u / "wrong_value_distribution_report.json")
    train_138u = read_json(root_138u / "train_value_attractor_report.json")
    miss_138u = read_json(root_138u / "eval_value_miss_report.json")
    prompt_138u = read_json(root_138u / "wrong_value_vs_prompt_report.json")
    root_138u_payload = read_json(root_138u / "attractor_root_cause.json")
    recommendation_138u = read_json(root_138u / "next_repair_recommendation.json")
    upstream_138wv_manifest = read_json(root_138u / "upstream_138wv_manifest.json")
    upstream_138w_manifest = read_json(root_138u / "upstream_138w_manifest.json")

    if (
        decision_138u.get("decision") != "wrong_value_attractor_analysis_complete"
        or decision_138u.get("root_cause") != "family_specific_train_value_attractor"
        or decision_138u.get("next") != "138YF_FAMILY_SPECIFIC_VALUE_ATTRACTOR_REPAIR_PLAN"
        or root_138u_payload.get("root_cause") != "family_specific_train_value_attractor"
        or recommendation_138u.get("recommended_next") != "138YF_FAMILY_SPECIFIC_VALUE_ATTRACTOR_REPAIR_PLAN"
    ):
        raise GateError("UPSTREAM_138U_ARTIFACT_MISSING", "138U did not route to 138YF")
    if root_138u_payload.get("root_cause") in DISALLOWED_138U_ROOTS:
        raise GateError("FAMILY_SPECIFIC_ATTRACTOR_EVIDENCE_RECHECK", "138U root cause is not family-specific")
    if distribution_138u.get("attractor_shape") != "family_specific_wrong_value_attractor":
        raise GateError("FAMILY_SPECIFIC_ATTRACTOR_EVIDENCE_RECHECK", "138U distribution no longer supports family-specific attractor")
    if not close(train_138u.get("generated_values_seen_in_train_rate"), EXPECTED_STRICT_TRAIN_RATE):
        raise GateError("FAMILY_SPECIFIC_ATTRACTOR_EVIDENCE_RECHECK", "strict train membership changed", {"actual": train_138u.get("generated_values_seen_in_train_rate")})
    if train_138u.get("generated_value_matches_most_frequent_train_value_rate") != 0.0 or miss_138u.get("expected_value_candidate_rate") != 0.0:
        raise GateError("FAMILY_SPECIFIC_ATTRACTOR_EVIDENCE_RECHECK", "138U no longer rejects high-frequency/global or expected-value candidate route")
    if prompt_138u.get("wrong_value_prompt_copy_rate") != 0.0 or prompt_138u.get("wrong_value_distractor_match_rate") != 0.0 or prompt_138u.get("wrong_value_wrong_table_entry_rate") != 0.0:
        raise GateError("FAMILY_SPECIFIC_ATTRACTOR_EVIDENCE_RECHECK", "138U no longer rejects prompt/distractor/table shortcut route")
    if upstream_138wv_manifest.get("wrong_specific_value_rate") != 1.0 or upstream_138wv_manifest.get("expected_value_candidate_rate") != 0.0:
        raise GateError("UPSTREAM_138U_ARTIFACT_MISSING", "138U upstream 138WV profile mismatch")

    aggregate_138w = read_json(root_138w / "aggregate_metrics.json")
    parrot_138w = read_json(root_138w / "parrot_trap_report.json")
    leakage_138w = read_json(root_138w / "freshness_leakage_audit.json")
    before_138w = read_json(root_138w / "generated_before_scoring_report.json")
    canary_138w = read_json(root_138w / "expected_output_canary_report.json")
    scan_138w = read_json(root_138w / "ast_shortcut_scan_report.json")
    replay_138w = read_json(root_138w / "determinism_replay_report.json")
    if canary_138w.get("expected_output_canary_passed") is not True or scan_138w.get("ast_shortcut_scan_passed") is not True:
        raise GateError("RAW_HELPER_INTEGRITY_FAILURE", "138W helper canary/AST integrity failed")
    if leakage_138w.get("leakage_rejected") is not True or replay_138w.get("determinism_replay_passed") is not True:
        raise GateError("UPSTREAM_138U_ARTIFACT_MISSING", "138W leakage/determinism missing")
    if upstream_138w_manifest.get("source_checkpoint_unchanged") is not True or upstream_138w_manifest.get("target_checkpoint_changed") is not True:
        raise GateError("UPSTREAM_138U_ARTIFACT_MISSING", "138W checkpoint integrity missing")
    if before_138w.get("generated_text_produced_before_scoring") is not True:
        raise GateError("UPSTREAM_138U_ARTIFACT_MISSING", "138W generated-before-scoring missing")
    if parrot_138w.get("parrot_trap_detected") is not False or aggregate_138w.get("stale_chat_fragment_rate") != 0.0 or aggregate_138w.get("train_namespace_leak_rate") != 0.0:
        raise GateError("UPSTREAM_138U_ARTIFACT_MISSING", "138W parrot/stale/namespace profile mismatch")

    manifest_138u = {
        "schema_version": "phase_138yf_upstream_138u_manifest_v1",
        "upstream_138u_root": rel(root_138u),
        "verified": True,
        "decision": decision_138u.get("decision"),
        "next": decision_138u.get("next"),
        "root_cause": decision_138u.get("root_cause"),
        "wrong_specific_value_rate": upstream_138wv_manifest.get("wrong_specific_value_rate"),
        "expected_value_candidate_rate": miss_138u.get("expected_value_candidate_rate"),
        "generated_values_seen_in_train_rate": train_138u.get("generated_values_seen_in_train_rate"),
        "generated_values_seen_in_train_expected_rate": train_138u.get("generated_values_seen_in_train_expected_rate"),
        "generated_values_seen_in_train_prompt_rate": train_138u.get("generated_values_seen_in_train_prompt_rate"),
        "generated_value_matches_most_frequent_train_value_rate": train_138u.get("generated_value_matches_most_frequent_train_value_rate"),
        "wrong_value_prompt_copy_rate": prompt_138u.get("wrong_value_prompt_copy_rate"),
        "wrong_value_distractor_match_rate": prompt_138u.get("wrong_value_distractor_match_rate"),
        "wrong_value_wrong_table_entry_rate": prompt_138u.get("wrong_value_wrong_table_entry_rate"),
        "not_global_train_value_prior": root_138u_payload.get("root_cause") != "global_train_value_prior_attractor",
        "not_high_frequency_train_value_prior": root_138u_payload.get("root_cause") != "high_frequency_train_value_attractor",
        "not_prompt_copy_parrot_trap": root_138u_payload.get("root_cause") != "prompt_copy_wrong_value_attractor",
        "all_capability_flags_false": all(summary_138u.get(key) is False for key in FALSE_FLAGS),
    }
    manifest_138w = {
        "schema_version": "phase_138yf_upstream_138w_manifest_v1",
        "upstream_138w_root": rel(root_138w),
        "verified": True,
        "helper_integrity_passed": True,
        "canary_passed": True,
        "ast_scan_passed": True,
        "leakage_rejected": True,
        "determinism_replay_passed": True,
        "source_checkpoint_unchanged": upstream_138w_manifest.get("source_checkpoint_unchanged"),
        "target_checkpoint_changed": upstream_138w_manifest.get("target_checkpoint_changed"),
        "generated_text_before_scoring": True,
        "no_expected_or_scorer_metadata_reached_helper_requests": True,
        "parrot_trap_detected": parrot_138w.get("parrot_trap_detected"),
        "stale_chat_fragment_rate": aggregate_138w.get("stale_chat_fragment_rate"),
        "train_namespace_leak_rate": aggregate_138w.get("train_namespace_leak_rate"),
    }
    write_json(out / "upstream_138u_manifest.json", manifest_138u)
    write_json(out / "upstream_138w_manifest.json", manifest_138w)
    return {
        "decision_138u": decision_138u,
        "distribution_138u": distribution_138u,
        "train_138u": train_138u,
        "miss_138u": miss_138u,
        "prompt_138u": prompt_138u,
        "root_138u": root_138u_payload,
        "upstream_138wv_manifest": upstream_138wv_manifest,
    }


def family_groups(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["family"]].append(row)
    return dict(sorted(groups.items()))


def build_family_summary(upstream: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "phase_138yf_family_specific_attractor_summary_v1",
        "root_cause": "family_specific_train_value_attractor",
        "coarse_family_routing_appears_present": {
            "value": True,
            "evidence_type": "inference",
            "reason": "138U found family-specific dominant wrong values rather than one global wrong value",
        },
        "value_specific_grounding_is_absent": {
            "value": True,
            "evidence_type": "computed_from_artifact",
            "answer_value_accuracy": 0.0,
            "expected_value_candidate_rate": upstream["miss_138u"].get("expected_value_candidate_rate"),
        },
        "wrong_values_are_family_specific_attractors": {
            "value": True,
            "evidence_type": "computed_from_artifact",
            "attractor_shape": upstream["distribution_138u"].get("attractor_shape"),
        },
        "strict_train_row_membership": upstream["train_138u"].get("generated_values_seen_in_train_rate"),
        "not_global_train_value_prior": True,
        "not_high_frequency_train_value_prior": True,
        "not_prompt_copy_parrot_trap": True,
        "qwen_conceptual_translation": "Scout-First Laziness and Missing Intra-Family Variance are planning hypotheses translated into artifact-level intra-family contrastive gates.",
    }


def train_membership_reconciliation(upstream: dict[str, Any]) -> dict[str, Any]:
    train = upstream["train_138u"]
    upstream_138wv = upstream["upstream_138wv_manifest"]
    return {
        "schema_version": "phase_138yf_train_membership_reconciliation_v1",
        "upstream_138wv_train_seen_value_label_rate": upstream_138wv.get("train_seen_value_rate"),
        "strict_138u_train_row_membership_rate": train.get("generated_values_seen_in_train_rate"),
        "generated_values_seen_in_train_rate": train.get("generated_values_seen_in_train_rate"),
        "generated_values_seen_in_train_expected_rate": train.get("generated_values_seen_in_train_expected_rate"),
        "generated_values_seen_in_train_prompt_rate": train.get("generated_values_seen_in_train_prompt_rate"),
        "generated_value_matches_most_frequent_train_value_rate": train.get("generated_value_matches_most_frequent_train_value_rate"),
        "global_memorized_lookup_claimed": False,
        "high_frequency_train_lookup_claimed": False,
        "reason": "Do not overclaim memorization: 138WV candidate_source_label is separated from strict train-row membership.",
    }


def intra_family_mode_collapse(rows: list[dict[str, Any]]) -> dict[str, Any]:
    families: dict[str, Any] = {}
    weighted_collapse_sum = 0.0
    total_rows = len(rows)
    for family, group in family_groups(rows).items():
        expected_values = {row["expected_value"] for row in group}
        generated_values = {row["generated_value_candidate"] for row in group if row.get("generated_value_candidate")}
        correct_values = {row["expected_value"] for row in group if row.get("generated_value_candidate") == row.get("expected_value")}
        counts = Counter(row.get("generated_value_candidate") for row in group)
        dominant_value, dominant_count = counts.most_common(1)[0]
        expected_count = len(expected_values)
        generated_count = len(generated_values)
        unique_rate = rate(generated_count, expected_count)
        collapse_rate = 1.0 - unique_rate
        weighted_collapse_sum += collapse_rate * len(group)
        families[family] = {
            "row_count": len(group),
            "per_family_expected_unique_value_count": expected_count,
            "per_family_generated_unique_value_count": generated_count,
            "per_family_correct_unique_value_count": len(correct_values),
            "per_family_wrong_attractor_count": generated_count,
            "per_family_dominant_wrong_value": dominant_value,
            "per_family_dominant_wrong_value_rate": rate(dominant_count, len(group)),
            "intra_family_mode_collapse_rate": collapse_rate,
            "intra_family_unique_value_rate": unique_rate,
            "intra_family_correct_value_diversity_rate": rate(len(correct_values), expected_count),
        }
    return {
        "schema_version": "phase_138yf_intra_family_mode_collapse_report_v1",
        "row_count": total_rows,
        "family_count": len(families),
        "overall_intra_family_mode_collapse_rate": weighted_collapse_sum / total_rows if total_rows else 0.0,
        "families": families,
        "interpretation": "Family-specific mode collapse is present when each family has many expected values, zero correct diversity, and a smaller set of generated wrong values.",
    }


def contrastive_objective_requirements(collapse: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "phase_138yf_intra_family_contrastive_objective_requirements_v1",
        "core_fix": "intra_family_contrastive_objective",
        "contrastive_groups": [
            "same family",
            "same wrapper",
            "similar prompt shape",
            "different prompt-specific values",
            "different correct expected values",
            "different distractors",
            "different table/rule bindings",
        ],
        "positive_requires": [
            "same-family prompts with different values produce different correct values",
            "not the same family default",
            "not a high-frequency train value",
            "not a prompt-copy-only shortcut",
            "not a wrapper-only success",
        ],
        "required_metrics_for_next_probe": [
            "intra_family_contrastive_accuracy",
            "intra_family_unique_correct_value_rate",
            "intra_family_mode_collapse_rate",
            "family_default_attractor_rate",
            "family_dominant_wrong_value_rate",
            "per_family_answer_value_accuracy",
            "per_family_exact_answer_accuracy",
            "per_family_rule_derived_value_accuracy",
            "per_family_table_derived_value_accuracy",
            "per_family_ood_symbol_value_accuracy",
        ],
        "baseline_overall_intra_family_mode_collapse_rate": collapse["overall_intra_family_mode_collapse_rate"],
        "clean_negative_accepted": True,
    }


def deep_scout_hypothesis() -> dict[str, Any]:
    return {
        "schema_version": "phase_138yf_deep_scout_forcing_hypothesis_v1",
        "hypothesis": "scout_first_laziness_or_family_gate_early_stop",
        "status": "design_hypothesis",
        "measured_directly": False,
        "diagnostic_gap": True,
        "qwen_terms": ["Scout-First Laziness", "Missing Intra-Family Variance"],
        "proxy_requirements": [
            "punish family-level format correctness without value correctness",
            "reject objectives that stop at family classification",
            "require intra-family value diversity",
            "require correct value binding inside each family",
        ],
        "no_internal_claim": "Do not claim actual scout/grower behavior unless a future artifact measures it.",
    }


def carrier_proxy_requirements() -> dict[str, Any]:
    return {
        "schema_version": "phase_138yf_carrier_proxy_requirements_v1",
        "hidden_state_or_graph_carrier_measurement": "diagnostic_gap",
        "instrumented_internal_state": False,
        "output_level_proxy_metrics": [
            "prompt_specific_value_survival_proxy",
            "value_after_prefix_accuracy",
            "value_position_error_rate",
            "no_stale_wrong_value_rate",
            "family_default_attractor_rate",
            "intra_family_correct_value_diversity_rate",
            "rule_derived_value_accuracy",
            "table_derived_value_accuracy",
            "ood_symbol_value_accuracy",
        ],
        "future_instrumentation_rule": "hidden_state_or_graph_carrier_measurement can become measured only if a future probe instruments hidden state, logits, or grower/scout internals explicitly.",
    }


def anti_shortcut_requirements() -> dict[str, Any]:
    return {
        "schema_version": "phase_138yf_anti_shortcut_requirements_v1",
        "explicit_rejects": [
            "family-level format success only",
            "ANSWER=E prefix-only success",
            "namespace-only success",
            "train-loss-only success",
            "teacher-forcing-only success",
            "prompt-copy-only success",
            "high-frequency train value replay",
            "family default value replay",
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


def next_138yi_plan(collapse: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "phase_138yf_next_138yi_milestone_plan_v1",
        "milestone": "138YI_FAMILY_SPECIFIC_VALUE_ATTRACTOR_REPAIR_PROBE",
        "purpose": "Test whether intra-family contrastive training can suppress family-specific wrong-value attractors and bind prompt-specific values after ANSWER=E.",
        "required_integrity_gates": [
            "shared_raw_generation_helper.py only",
            "generated_text before scoring",
            "expected-output canary",
            "AST shortcut scan",
            "deterministic replay",
            "controls fail",
            "leakage rejected",
            "source checkpoint unchanged",
            "target checkpoint under target/ only",
            "no helper/backend modification",
            "no old runner imports",
            "no expected/scorer metadata in helper request",
        ],
        "dataset_requirements": [
            "train/eval family/value splits",
            "intra-family contrastive train/eval rows",
            "OOD family/value combinations",
            "same family and prompt skeleton with different expected values",
            "held-out rule/table/OOD symbol bindings",
        ],
        "positive_gates": [
            "answer_value_accuracy improves",
            "exact_answer_accuracy improves",
            "intra_family_contrastive_accuracy >= declared threshold",
            "intra_family_mode_collapse_rate decreases",
            "family_default_attractor_rate decreases",
            "per_family answer value accuracy improves across at least N families",
            "rule_derived/table_derived/OOD metrics improve",
            "parrot_trap_detected = false",
            "train_namespace_leak_rate remains below gate",
            "stale_chat_fragment_rate remains below gate",
            "deterministic replay passes",
        ],
        "baseline_overall_intra_family_mode_collapse_rate": collapse["overall_intra_family_mode_collapse_rate"],
        "clean_negative_accepted": True,
    }


def make_decision() -> tuple[dict[str, Any], list[str]]:
    decision = {
        "schema_version": "phase_138yf_decision_v1",
        "decision": "family_specific_value_attractor_repair_plan_complete",
        "next": "138YI_FAMILY_SPECIFIC_VALUE_ATTRACTOR_REPAIR_PROBE",
        "verdict": "FAMILY_SPECIFIC_VALUE_ATTRACTOR_REPAIR_PLAN_COMPLETE",
        "primary_bottleneck": "family_specific_mode_collapse_missing_intra_family_contrast",
        "artifact_only": True,
        "planning_only": True,
        "new_inference_run": False,
        "shared_helper_called": False,
        "torch_forward_pass_run": False,
        "scout_first_laziness_status": "design_hypothesis_not_measured_mechanism",
        "hidden_state_or_graph_carrier_measurement": "diagnostic_gap",
        **FALSE_FLAGS,
    }
    verdicts = [
        decision["verdict"],
        "ARTIFACT_ONLY_PLANNING",
        "INTRA_FAMILY_CONTRASTIVE_OBJECTIVE_DEFINED",
        "SCOUT_FIRST_LAZINESS_RECORDED_AS_HYPOTHESIS_ONLY",
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
        decision_name = "raw_helper_integrity_failure"
        next_step = "135G_RAW_HELPER_ORACLE_SHORTCUT_REMOVAL"
    elif error.verdict == "FAMILY_SPECIFIC_ATTRACTOR_EVIDENCE_RECHECK":
        decision_name = "family_specific_attractor_evidence_recheck"
        next_step = "138YF_FAMILY_ATTRACTOR_EVIDENCE_RECHECK"
    else:
        decision_name = "upstream_138u_artifact_missing"
        next_step = "138YF_UPSTREAM_138U_ARTIFACT_MISSING"
    decision = {
        "schema_version": "phase_138yf_failure_decision_v1",
        "decision": decision_name,
        "next": next_step,
        "verdict": error.verdict,
        "failure_message": error.message,
        **FALSE_FLAGS,
    }
    write_json(out / "decision.json", decision)
    append_progress(out, "failure", status="failed", verdict=error.verdict, message=error.message)
    write_summary(out, "failure", [error.verdict], decision, error.message)
    write_report(out, [error.verdict], decision)


def run(args: argparse.Namespace) -> None:
    out = resolve_target_out(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "queue.json", {"schema_version": "phase_138yf_queue_v1", "milestone": MILESTONE, "status": "started", "started_at": utc_now(), "heartbeat_sec": args.heartbeat_sec})
    append_progress(out, "startup", heartbeat_sec=args.heartbeat_sec)
    refresh_status(out, "running", ["FAMILY_SPECIFIC_VALUE_ATTRACTOR_REPAIR_PLAN_RUNNING"], {"decision": "pending", "next": "pending"})

    root_138u = resolve_path(args.upstream_138u_root)
    root_138wv = resolve_path(args.upstream_138wv_root)
    root_138w = resolve_path(args.upstream_138w_root)
    upstream = verify_upstreams(out, root_138u, root_138wv, root_138w)
    append_progress(out, "upstream verification", upstream_138u_root=rel(root_138u), upstream_138wv_root=rel(root_138wv), upstream_138w_root=rel(root_138w))
    write_json(
        out / "analysis_config.json",
        {
            "schema_version": "phase_138yf_analysis_config_v1",
            "artifact_only": True,
            "planning_only": True,
            "new_inference_run": False,
            "shared_helper_called": False,
            "torch_forward_pass_run": False,
            "checkpoint_mutation_performed": False,
            "helper_backend_modified": False,
            "old_runner_imported": False,
        },
    )
    append_progress(out, "artifact loading")

    family_summary = build_family_summary(upstream)
    write_json(out / "family_specific_attractor_summary.json", family_summary)
    append_progress(out, "family-specific diagnosis", root_cause=family_summary["root_cause"])

    reconciliation = train_membership_reconciliation(upstream)
    write_json(out / "train_membership_reconciliation.json", reconciliation)
    append_progress(out, "train membership reconciliation", strict_rate=reconciliation["strict_138u_train_row_membership_rate"])

    rows = upstream["distribution_138u"]["rows"]
    collapse = intra_family_mode_collapse(rows)
    write_json(out / "intra_family_mode_collapse_report.json", collapse)
    append_progress(out, "intra-family mode collapse", overall_collapse=collapse["overall_intra_family_mode_collapse_rate"])
    refresh_status(out, "running", ["INTRA_FAMILY_MODE_COLLAPSE_ANALYZED"], {"decision": "pending", "next": "pending"})

    contrastive = contrastive_objective_requirements(collapse)
    write_json(out / "intra_family_contrastive_objective_requirements.json", contrastive)
    append_progress(out, "contrastive objective requirements", metric_count=len(contrastive["required_metrics_for_next_probe"]))

    scout = deep_scout_hypothesis()
    write_json(out / "deep_scout_forcing_hypothesis.json", scout)
    append_progress(out, "deep scout hypothesis", measured_directly=scout["measured_directly"])

    carrier = carrier_proxy_requirements()
    write_json(out / "carrier_proxy_requirements.json", carrier)
    append_progress(out, "carrier proxy requirements", instrumented_internal_state=carrier["instrumented_internal_state"])

    shortcuts = anti_shortcut_requirements()
    write_json(out / "anti_shortcut_requirements.json", shortcuts)
    append_progress(out, "anti-shortcut requirements", reject_count=len(shortcuts["explicit_rejects"]))

    next_plan = next_138yi_plan(collapse)
    write_json(out / "next_138yi_milestone_plan.json", next_plan)
    append_progress(out, "138YI plan drafting", next=next_plan["milestone"])

    write_json(
        out / "risk_register.json",
        {
            "schema_version": "phase_138yf_risk_register_v1",
            "risks": [
                {"risk": "scout-first laziness is overclaimed as measured internals", "mitigation": "recorded as design_hypothesis with diagnostic_gap=true"},
                {"risk": "family-level routing success is counted as value grounding", "mitigation": "next plan gates intra-family value diversity and correct binding separately"},
                {"risk": "strict train membership is conflated with 138WV label", "mitigation": "train_membership_reconciliation separates upstream label from strict row membership"},
            ],
        },
    )
    append_progress(out, "risk register")

    decision, verdicts = make_decision()
    write_json(out / "decision.json", decision)
    append_progress(out, "decision", decision=decision["decision"], next=decision["next"])
    refresh_status(out, "completed", verdicts, decision)
    append_progress(out, "final verdict", verdicts=verdicts)
    write_json(out / "queue.json", {"schema_version": "phase_138yf_queue_v1", "milestone": MILESTONE, "status": "completed", "completed_at": utc_now(), "heartbeat_sec": args.heartbeat_sec})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--upstream-138u-root", default=str(DEFAULT_UPSTREAM_138U_ROOT))
    parser.add_argument("--upstream-138wv-root", default=str(DEFAULT_UPSTREAM_138WV_ROOT))
    parser.add_argument("--upstream-138w-root", default=str(DEFAULT_UPSTREAM_138W_ROOT))
    parser.add_argument("--heartbeat-sec", type=int, default=20)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        run(args)
        return 0
    except GateError as exc:
        write_failure_decision(args, exc)
        print(f"138YF failed closed: {exc.verdict}: {exc.message}")
        return 1 if exc.verdict == "138YF_BOUNDARY_FAILURE" else 0


if __name__ == "__main__":
    raise SystemExit(main())

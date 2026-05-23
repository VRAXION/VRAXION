#!/usr/bin/env python3
"""138V artifact-only answer-value grounding objective redesign plan.

This phase reads existing 138S and 138I artifacts only. It does not train,
repair, run inference, call shared_raw_generation_helper.py, run torch forward
passes, mutate checkpoints, modify helper/backend code, import old runners,
start services, deploy, delete or consolidate files, or modify runtime,
release, or product surfaces.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
MILESTONE = "STABLE_LOOP_PHASE_LOCK_138V_ANSWER_VALUE_GROUNDING_OBJECTIVE_REDESIGN_PLAN"
DEFAULT_OUT = Path("target/pilot_wave/stable_loop_phase_lock_138v_answer_value_grounding_objective_redesign_plan/smoke")
DEFAULT_UPSTREAM_138S_ROOT = Path("target/pilot_wave/stable_loop_phase_lock_138s_stale_chat_rollout_failure_analysis/smoke")
DEFAULT_UPSTREAM_138I_ROOT = Path("target/pilot_wave/stable_loop_phase_lock_138i_real_raw_reasoning_rollout_aligned_repair_probe/smoke")
BOUNDARY_TEXT = (
    "138V is planning-only answer-value grounding objective redesign. It does "
    "not train, repair, run inference, call shared_raw_generation_helper.py, "
    "run torch forward passes, mutate checkpoints, modify helper/backend code, "
    "import old runners, delete or consolidate files, start services, deploy, "
    "modify runtime/release/product surfaces, or change root LICENSE. It does "
    "not restore reasoning, raw assistant capability, structured/tool "
    "capability, GPT-like readiness, open-domain readiness, production chat, "
    "public API, deployment readiness, or safety alignment."
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
REQUIRED_138S_ARTIFACTS = [
    "decision.json",
    "summary.json",
    "upstream_138i_manifest.json",
    "stale_chat_distribution_report.json",
    "value_grounding_failure_report.json",
    "prefix_vs_value_decoupling_report.json",
    "source_prior_vs_training_objective_report.json",
    "stale_chat_value_coupling_report.json",
    "diagnostic_gap_register.json",
    "next_repair_recommendation.json",
]
REQUIRED_138I_ARTIFACTS = [
    "decision.json",
    "expected_output_canary_report.json",
    "ast_shortcut_scan_report.json",
    "control_arm_report.json",
    "freshness_leakage_audit.json",
    "determinism_replay_report.json",
    "generated_before_scoring_report.json",
    "source_checkpoint_integrity_manifest.json",
    "target_checkpoint_integrity_manifest.json",
    "raw_generation_trace.jsonl",
]
REJECTED_SHORTCUTS = [
    "more teacher-forcing",
    "more loss weighting",
    "train-loss-only success",
    "prefix-only success",
    "namespace-only success",
    "expected-output construction",
    "old runner imports",
    "oracle/rerank/verifier/LLM judge",
    "constrained decoding",
    "JSON mode",
    "regex fixer",
    "post-generation repair",
    "retry loop",
    "best-of-n",
    "threshold weakening to force positive",
]
OUTPUT_PROXY_METRICS = [
    "value_token_emission_accuracy",
    "answer_value_accuracy",
    "exact_answer_accuracy",
    "prefix_success_value_failure_rate",
    "eval_namespace_success_value_failure_rate",
    "no_stale_wrong_value_rate",
    "value_after_prefix_accuracy",
    "value_position_error_rate",
    "empty_value_after_prefix_rate",
    "generic_value_after_prefix_rate",
    "prompt_value_copy_accuracy",
    "rule_derived_value_accuracy",
    "table_derived_value_accuracy",
]
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
        raise GateError("138V_BOUNDARY_FAILURE", "--out must stay inside the repo") from exc
    parts = [part.lower() for part in relative.parts]
    if len(parts) < 2 or parts[0] != "target" or parts[1] != "pilot_wave":
        raise GateError("138V_BOUNDARY_FAILURE", "--out must stay under target/pilot_wave")
    if any(part == ".." for part in relative.parts):
        raise GateError("138V_BOUNDARY_FAILURE", "--out must not escape target/pilot_wave")
    return resolved


def append_progress(out: Path, event: str, status: str = "completed", **details: Any) -> None:
    append_jsonl(out / "progress.jsonl", {"ts": utc_now(), "event": event, "status": status, "details": details})


def require_files(root: Path, names: list[str], verdict: str) -> None:
    missing = [name for name in names if not (root / name).exists()]
    if missing:
        raise GateError(verdict, "required upstream artifacts missing", {"root": rel(root), "missing": missing})


def write_summary(out: Path, status: str, verdicts: list[str], decision: dict[str, Any], failure: str | None = None) -> None:
    write_json(
        out / "summary.json",
        {
            "schema_version": "phase_138v_summary_v1",
            "milestone": MILESTONE,
            "status": status,
            "verdicts": verdicts,
            "decision": decision.get("decision"),
            "next": decision.get("next"),
            "failure": failure,
            "boundary": BOUNDARY_TEXT,
            "planning_only": True,
            "training_performed": False,
            "new_model_inference_run": False,
            "shared_helper_called": False,
            "torch_forward_pass_run": False,
            "checkpoint_mutated": False,
            "runtime_surface_mutated": False,
            "root_license_changed": False,
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
        "## Verdicts",
        "",
    ]
    lines.extend(f"- `{verdict}`" for verdict in verdicts)
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- `decision`: `{decision.get('decision')}`",
            f"- `next`: `{decision.get('next')}`",
            f"- `primary_bottleneck`: `{decision.get('primary_bottleneck')}`",
            f"- `hidden_state_residual_signal_measurement`: `{decision.get('hidden_state_residual_signal_measurement')}`",
            "",
            "138V is planning-only objective redesign.",
            "Residual Signal Carrier is a design concept here, not a measured hidden-state fact.",
            "Reasoning is not restored.",
            "The reasoning subtrack real-raw evidence is not partially restored.",
            "Raw assistant capability remains quarantined.",
            "Structured/tool capability remains invalidated as model evidence.",
            "Not GPT-like readiness.",
            "Not open-domain assistant readiness.",
            "Not production chat.",
            "Not public API.",
            "Not deployment readiness.",
            "Not safety alignment.",
        ]
    )
    write_text(out / "report.md", "\n".join(lines) + "\n")


def refresh_status(out: Path, status: str, verdicts: list[str], decision: dict[str, Any]) -> None:
    write_summary(out, status, verdicts, decision)
    write_report(out, verdicts, decision)


def verify_upstreams(out: Path, root_138s: Path, root_138i: Path) -> dict[str, Any]:
    require_files(root_138s, REQUIRED_138S_ARTIFACTS, "UPSTREAM_138S_ARTIFACT_MISSING")
    require_files(root_138i, REQUIRED_138I_ARTIFACTS, "UPSTREAM_138S_ARTIFACT_MISSING")
    d138s = read_json(root_138s / "decision.json")
    value = read_json(root_138s / "value_grounding_failure_report.json")
    prefix = read_json(root_138s / "prefix_vs_value_decoupling_report.json")
    coupling = read_json(root_138s / "stale_chat_value_coupling_report.json")
    recommendation = read_json(root_138s / "next_repair_recommendation.json")
    d138i = read_json(root_138i / "decision.json")
    canary = read_json(root_138i / "expected_output_canary_report.json")
    scan = read_json(root_138i / "ast_shortcut_scan_report.json")
    controls = read_json(root_138i / "control_arm_report.json")
    leakage = read_json(root_138i / "freshness_leakage_audit.json")
    replay = read_json(root_138i / "determinism_replay_report.json")
    before = read_json(root_138i / "generated_before_scoring_report.json")
    source = read_json(root_138i / "source_checkpoint_integrity_manifest.json")
    target = read_json(root_138i / "target_checkpoint_integrity_manifest.json")
    traces = read_jsonl(root_138i / "raw_generation_trace.jsonl")

    if d138s.get("decision") != "stale_chat_rollout_failure_analysis_complete" or d138s.get("next") != "138V_ANSWER_VALUE_GROUNDING_OBJECTIVE_REDESIGN_PLAN":
        raise GateError("UPSTREAM_138S_ARTIFACT_MISSING", "138S did not route to 138V")
    if d138s.get("primary_diagnosis") != "answer_value_grounding_failure_decoupled_from_stale_chat":
        raise GateError("ANSWER_VALUE_GROUNDING_REDESIGN_BLOCKED", "138S does not support value grounding as primary bottleneck")
    if recommendation.get("recommended_next") != "138V_ANSWER_VALUE_GROUNDING_OBJECTIVE_REDESIGN_PLAN":
        raise GateError("ANSWER_VALUE_GROUNDING_REDESIGN_BLOCKED", "138S recommendation does not support 138V")
    if value.get("answer_prefix_accuracy") != 1.0 or value.get("eval_namespace_emission_accuracy") != 1.0:
        raise GateError("ANSWER_VALUE_GROUNDING_REDESIGN_BLOCKED", "wrapper/namespace success evidence missing")
    if value.get("answer_value_accuracy") != 0.0 or prefix.get("wrapper_prefix_learned_without_value_grounding") is not True:
        raise GateError("ANSWER_VALUE_GROUNDING_REDESIGN_BLOCKED", "value failure evidence missing")
    if coupling.get("P_wrong_value_given_stale_chat") != 1.0 or coupling.get("P_wrong_value_given_no_stale_chat") != 1.0:
        raise GateError("ANSWER_VALUE_GROUNDING_REDESIGN_BLOCKED", "no-stale value failure evidence missing")
    if d138i.get("expected_output_canary_passed") is not True or canary.get("expected_output_canary_passed") is not True:
        raise GateError("RAW_HELPER_INTEGRITY_FAILURE", "138I canary integrity failed")
    if d138i.get("ast_shortcut_scan_passed") is not True or scan.get("ast_shortcut_scan_passed") is not True:
        raise GateError("RAW_HELPER_INTEGRITY_FAILURE", "138I AST integrity failed")
    if controls.get("controls_failed") is not True or leakage.get("leakage_rejected") is not True or replay.get("determinism_replay_passed") is not True:
        raise GateError("UPSTREAM_138S_ARTIFACT_MISSING", "138I controls/leakage/replay gates missing")
    if before.get("generated_text_produced_before_scoring") is not True:
        raise GateError("UPSTREAM_138S_ARTIFACT_MISSING", "138I generated-before-scoring proof missing")
    if source.get("source_checkpoint_unchanged") is not True or target.get("target_checkpoint_changed") is not True:
        raise GateError("UPSTREAM_138S_ARTIFACT_MISSING", "138I checkpoint integrity missing")
    for trace in traces:
        helper_request = trace.get("helper_request", {})
        if set(helper_request) != ALLOWED_HELPER_KEYS or set(helper_request) & FORBIDDEN_HELPER_KEYS:
            raise GateError("RAW_HELPER_INTEGRITY_FAILURE", "138I helper request metadata violation")

    manifest = {
        "schema_version": "phase_138v_upstream_138s_manifest_v1",
        "upstream_138s_root": rel(root_138s),
        "upstream_138i_root": rel(root_138i),
        "upstream_138s_verified": True,
        "upstream_138i_helper_integrity_verified": True,
        "decision": d138s.get("decision"),
        "next": d138s.get("next"),
        "primary_diagnosis": d138s.get("primary_diagnosis"),
        "answer_prefix_accuracy": value.get("answer_prefix_accuracy"),
        "eval_namespace_emission_accuracy": value.get("eval_namespace_emission_accuracy"),
        "answer_value_accuracy": value.get("answer_value_accuracy"),
        "prefix_success_value_failure_rate": prefix.get("prefix_success_value_failure_rate"),
        "wrapper_prefix_learned_without_value_grounding": prefix.get("wrapper_prefix_learned_without_value_grounding"),
        "P_wrong_value_given_stale_chat": coupling.get("P_wrong_value_given_stale_chat"),
        "P_wrong_value_given_no_stale_chat": coupling.get("P_wrong_value_given_no_stale_chat"),
        "canary_ast_controls_leakage_determinism_passed": True,
        "source_checkpoint_unchanged": True,
        "target_checkpoint_changed": True,
        "no_expected_or_scorer_metadata_reached_helper_requests": True,
    }
    write_json(out / "upstream_138s_manifest.json", manifest)
    return {"manifest": manifest, "value": value, "prefix": prefix, "coupling": coupling, "d138s": d138s}


def wrapper_value_gap_summary(upstream: dict[str, Any]) -> dict[str, Any]:
    value = upstream["value"]
    prefix = upstream["prefix"]
    return {
        "schema_version": "phase_138v_wrapper_value_gap_summary_v1",
        "answer_prefix_accuracy": value["answer_prefix_accuracy"],
        "eval_namespace_emission_accuracy": value["eval_namespace_emission_accuracy"],
        "answer_value_accuracy": value["answer_value_accuracy"],
        "exact_answer_accuracy": value["exact_answer_accuracy"],
        "prefix_success_value_failure_rate": prefix["prefix_success_value_failure_rate"],
        "eval_namespace_success_value_failure_rate": prefix["eval_namespace_success_value_failure_rate"],
        "wrapper_prefix_learned_without_value_grounding": prefix["wrapper_prefix_learned_without_value_grounding"],
        "interpretation": "wrapper reflex succeeded while prompt-derived value grounding failed",
        "evidence_type": "computed_from_artifact",
        "source": "138S value_grounding_failure_report and prefix_vs_value_decoupling_report",
    }


def no_stale_value_failure_summary(upstream: dict[str, Any]) -> dict[str, Any]:
    coupling = upstream["coupling"]
    return {
        "schema_version": "phase_138v_no_stale_value_failure_summary_v1",
        "P_wrong_value_given_stale_chat": coupling["P_wrong_value_given_stale_chat"],
        "P_wrong_value_given_no_stale_chat": coupling["P_wrong_value_given_no_stale_chat"],
        "value_failure_occurs_without_stale_chat": coupling["value_failure_occurs_without_stale_chat"],
        "stale_chat_is_sufficient_explanation_for_value_failure": coupling["stale_chat_is_sufficient_explanation_for_value_failure"],
        "conclusion": "stale chat is a secondary failure and is not a sufficient root cause for answer-value failure",
        "evidence_type": "computed_from_artifact",
        "source": "138S stale_chat_value_coupling_report",
    }


def residual_signal_carrier_requirements() -> dict[str, Any]:
    return {
        "schema_version": "phase_138v_residual_signal_carrier_requirements_v1",
        "design_concept": "Residual Signal Carrier",
        "not_a_hidden_state_claim": True,
        "hidden_state_residual_signal_measurement": "diagnostic_gap",
        "diagnostic_gap_reason": "138V does not inspect hidden states or activations; any hidden-state collapse claim requires future instrumentation.",
        "objective_requirement": "The prompt-derived answer value must survive through wrapper generation and appear after ANSWER=E in helper-only free rollout.",
        "layer_separation": {
            "wrapper_reflex": {
                "status": "already_observed",
                "evidence": ["answer_prefix_accuracy = 1.0", "eval_namespace_emission_accuracy = 1.0"],
                "must_not_count_as_grounding": True,
            },
            "value_carrier": {
                "status": "design_requirement",
                "measurement": "output_level_proxy_metrics",
                "hidden_state_claim_status": "diagnostic_gap_without_instrumentation",
            },
            "value_grounding": {
                "status": "next_probe_target",
                "requirement": "The generated value after ANSWER=E must match prompt-provided or rule-derived held-out values.",
            },
        },
        "success_rejections": {
            "prefix_emission_alone_is_success": False,
            "namespace_emission_alone_is_success": False,
            "train_loss_alone_is_success": False,
            "teacher_forcing_only_success": False,
        },
        "output_level_proxy_metrics": OUTPUT_PROXY_METRICS,
    }


def wrapper_induced_amnesia_hypothesis() -> dict[str, Any]:
    return {
        "schema_version": "phase_138v_wrapper_induced_amnesia_hypothesis_v1",
        "hypothesis": "The model emits the shallow wrapper/prefix ANSWER=E, then loses or fails to carry the prompt-derived value into the value position.",
        "status": "planning_hypothesis",
        "artifact_support": [
            "answer_prefix_accuracy = 1.0",
            "eval_namespace_emission_accuracy = 1.0",
            "answer_value_accuracy = 0.0",
            "P(wrong_value | no_stale_chat) = 1.0",
        ],
        "hidden_state_claim_status": "diagnostic_gap_without_instrumentation",
        "must_be_tested_by": "138W helper-only free-rollout value grounding probe",
    }


def value_grounding_objective_requirements() -> dict[str, Any]:
    return {
        "schema_version": "phase_138v_value_grounding_objective_requirements_v1",
        "must_directly_reward": [
            "correct value after ANSWER=E",
            "correct value on OOD eval rows",
            "value grounded from prompt-provided facts",
            "value correct even without stale chat",
            "value correct across families and seeds",
        ],
        "must_directly_penalize": [
            "ANSWER=E with wrong value",
            "ANSWER=E with empty value",
            "ANSWER=E with generic/random value",
            "prompt-independent value emission",
            "train namespace replay",
            "stale User:/Assistant fragments",
            "off-prompt continuation",
        ],
        "positive_cannot_depend_on": [
            "train loss alone",
            "teacher-forcing-only success",
            "prefix-only success",
            "namespace-only success",
            "target checkpoint changed",
        ],
    }


def ood_value_grounding_eval_requirements() -> dict[str, Any]:
    return {
        "schema_version": "phase_138v_ood_value_grounding_eval_requirements_v1",
        "requirements": [
            "train/eval value namespaces disjoint",
            "train/eval row hashes disjoint",
            "no exact prompt overlap",
            "no near duplicate prompt overlap",
            "eval values novel relative to train where possible",
            "same wrapper with novel required values",
            "held-out value ranges",
            "held-out symbol tokens",
            "held-out table mappings",
            "held-out rule chains",
        ],
        "purpose": "prevent fake grounding by memorized prompt-value pairs",
    }


def stale_secondary_gate_requirements() -> dict[str, Any]:
    return {
        "schema_version": "phase_138v_stale_secondary_gate_requirements_v1",
        "stale_chat_is_primary_root_cause": False,
        "stale_chat_is_secondary_hard_gate": True,
        "required_metrics": [
            "stale_chat_fragment_rate",
            "stale_user_rate",
            "stale_assistant_rate",
            "P_wrong_value_given_stale_chat",
            "P_wrong_value_given_no_stale_chat",
        ],
        "positive_requires": {
            "stale_chat_fragment_rate_below_declared_gate": True,
            "value_accuracy_improves_on_non_stale_rows": True,
        },
    }


def next_138w_milestone_plan() -> dict[str, Any]:
    return {
        "schema_version": "phase_138v_next_138w_milestone_plan_v1",
        "milestone": "138W_ANSWER_VALUE_GROUNDING_REPAIR_PROBE",
        "purpose": "test whether the prompt-derived answer value can be carried through ANSWER=E wrapper generation under helper-only free rollout",
        "required_path": [
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
        "positive_gates": [
            "answer_value_accuracy improves from 0.0",
            "exact_answer_accuracy improves from 0.0",
            "prefix_success_value_failure_rate decreases",
            "eval_namespace_success_value_failure_rate decreases",
            "P(wrong_value | no_stale_chat) decreases from 1.0",
            "value_after_prefix_accuracy improves from 0.0",
            "value_position_error_rate decreases",
            "empty_value_after_prefix_rate remains low",
            "generic_value_after_prefix_rate remains low",
            "prompt_value_copy_accuracy is tracked",
            "rule_derived_value_accuracy is tracked",
            "table_derived_value_accuracy is tracked",
            "stale_chat_fragment_rate remains tracked and below declared gate",
            "train_namespace_leak_rate remains below gate",
            "eval_namespace_emission_accuracy remains high",
            "determinism replay passes",
        ],
        "clean_negative_accepted": True,
        "failure_routes": {
            "no_value_grounding_improvement": "138W_VALUE_GROUNDING_REPAIR_FAILURE_ANALYSIS",
            "stale_chat_regression": "138S_STALE_CHAT_ROLLOUT_FAILURE_ANALYSIS",
            "namespace_leak_regression": "138S_NAMESPACE_ROLLOUT_FAILURE_ANALYSIS",
            "determinism_mismatch": "138N_DETERMINISM_FAILURE_ANALYSIS",
            "helper_integrity_failure": "135G_RAW_HELPER_ORACLE_SHORTCUT_REMOVAL",
        },
    }


def anti_shortcut_requirements() -> dict[str, Any]:
    return {
        "schema_version": "phase_138v_anti_shortcut_requirements_v1",
        "explicit_rejects": REJECTED_SHORTCUTS,
        "expected_output_may_enter_generation": False,
        "scorer_metadata_may_enter_helper_request": False,
        "post_generation_repair_allowed": False,
        "threshold_weakening_to_force_positive_allowed": False,
    }


def risk_register() -> dict[str, Any]:
    return {
        "schema_version": "phase_138v_risk_register_v1",
        "risks": [
            {
                "risk": "future objective repeats wrapper success without value grounding",
                "mitigation": "gate answer_value_accuracy, exact_answer_accuracy, and no-stale wrong-value rate separately from prefix accuracy",
            },
            {
                "risk": "Residual Signal Carrier becomes an unsupported hidden-state claim",
                "mitigation": "record hidden-state measurement as diagnostic_gap unless 138W explicitly instruments activations",
            },
            {
                "risk": "OOD value checks are too close to train values",
                "mitigation": "require held-out value ranges, symbol tokens, table mappings, and rule chains",
            },
            {
                "risk": "stale chat is ignored as secondary issue",
                "mitigation": "keep stale chat below declared gate for any positive 138W route",
            },
        ],
    }


def make_decision() -> tuple[dict[str, Any], list[str]]:
    decision = {
        "schema_version": "phase_138v_decision_v1",
        "decision": "answer_value_grounding_objective_redesign_plan_complete",
        "next": "138W_ANSWER_VALUE_GROUNDING_REPAIR_PROBE",
        "verdict": "ANSWER_VALUE_GROUNDING_OBJECTIVE_REDESIGN_PLAN_COMPLETE",
        "primary_bottleneck": "wrapper_success_without_value_grounding",
        "wrapper_induced_amnesia_hypothesis_recorded": True,
        "residual_signal_carrier_design_concept_recorded": True,
        "hidden_state_residual_signal_measurement": "diagnostic_gap",
        "planning_only": True,
        "training_performed": False,
        "new_model_inference_run": False,
        "shared_helper_called": False,
        "torch_forward_pass_run": False,
        "checkpoint_mutated": False,
        "runtime_surface_mutated": False,
        "root_license_changed": False,
        "no_capability_restored": True,
        **FALSE_FLAGS,
    }
    verdicts = [
        "ANSWER_VALUE_GROUNDING_OBJECTIVE_REDESIGN_PLAN_COMPLETE",
        "WRAPPER_VALUE_GAP_RECORDED",
        "NO_STALE_VALUE_FAILURE_RECORDED",
        "RESIDUAL_SIGNAL_CARRIER_REQUIREMENTS_RECORDED",
        "NEXT_138W_PLAN_WRITTEN",
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
    elif error.verdict == "ANSWER_VALUE_GROUNDING_REDESIGN_BLOCKED":
        decision_name = "answer_value_grounding_redesign_blocked"
        next_step = "138VA_VALUE_GROUNDING_EVIDENCE_RECHECK"
    else:
        decision_name = "upstream_138s_artifact_missing"
        next_step = "138V_UPSTREAM_138S_ARTIFACT_MISSING"
    decision = {
        "schema_version": "phase_138v_failure_decision_v1",
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
    write_json(out / "queue.json", {"schema_version": "phase_138v_queue_v1", "milestone": MILESTONE, "status": "started", "started_at": utc_now(), "heartbeat_sec": args.heartbeat_sec})
    append_progress(out, "startup", heartbeat_sec=args.heartbeat_sec)
    refresh_status(out, "running", ["ANSWER_VALUE_GROUNDING_OBJECTIVE_REDESIGN_RUNNING"], {"decision": "pending", "next": "pending"})

    upstream = verify_upstreams(out, resolve_path(args.upstream_138s_root), resolve_path(args.upstream_138i_root))
    append_progress(out, "upstream verification", upstream_138s_verified=True, upstream_138i_helper_integrity_verified=True)

    wrapper = wrapper_value_gap_summary(upstream)
    write_json(out / "wrapper_value_gap_summary.json", wrapper)
    append_progress(out, "wrapper/value gap summary", answer_value_accuracy=wrapper["answer_value_accuracy"])
    refresh_status(out, "running", ["WRAPPER_VALUE_GAP_SUMMARIZED"], {"decision": "pending", "next": "pending"})

    no_stale = no_stale_value_failure_summary(upstream)
    write_json(out / "no_stale_value_failure_summary.json", no_stale)
    append_progress(out, "no-stale value failure summary", value_failure_occurs_without_stale_chat=no_stale["value_failure_occurs_without_stale_chat"])

    residual = residual_signal_carrier_requirements()
    write_json(out / "residual_signal_carrier_requirements.json", residual)
    append_progress(out, "residual signal carrier requirements", hidden_state_measurement=residual["hidden_state_residual_signal_measurement"])

    amnesia = wrapper_induced_amnesia_hypothesis()
    write_json(out / "wrapper_induced_amnesia_hypothesis.json", amnesia)
    append_progress(out, "wrapper-induced amnesia hypothesis", status=amnesia["status"])

    value_reqs = value_grounding_objective_requirements()
    write_json(out / "value_grounding_objective_requirements.json", value_reqs)
    append_progress(out, "value grounding objective requirements", rewards=len(value_reqs["must_directly_reward"]))

    ood_reqs = ood_value_grounding_eval_requirements()
    write_json(out / "ood_value_grounding_eval_requirements.json", ood_reqs)
    append_progress(out, "OOD value grounding eval requirements", requirements=len(ood_reqs["requirements"]))

    stale_reqs = stale_secondary_gate_requirements()
    write_json(out / "stale_secondary_gate_requirements.json", stale_reqs)
    append_progress(out, "stale secondary gate requirements", stale_secondary_gate=stale_reqs["stale_chat_is_secondary_hard_gate"])

    plan_138w = next_138w_milestone_plan()
    write_json(out / "next_138w_milestone_plan.json", plan_138w)
    append_progress(out, "next 138W plan", milestone=plan_138w["milestone"])

    anti = anti_shortcut_requirements()
    write_json(out / "anti_shortcut_requirements.json", anti)
    write_json(out / "risk_register.json", risk_register())
    append_progress(out, "anti-shortcut and risk requirements", rejects=len(anti["explicit_rejects"]))

    decision, verdicts = make_decision()
    write_json(out / "decision.json", decision)
    append_progress(out, "decision", decision=decision["decision"], next=decision["next"])
    refresh_status(out, "completed", verdicts, decision)
    append_progress(out, "final verdict", verdicts=verdicts)
    write_json(out / "queue.json", {"schema_version": "phase_138v_queue_v1", "milestone": MILESTONE, "status": "completed", "completed_at": utc_now(), "heartbeat_sec": args.heartbeat_sec})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--upstream-138s-root", default=str(DEFAULT_UPSTREAM_138S_ROOT))
    parser.add_argument("--upstream-138i-root", default=str(DEFAULT_UPSTREAM_138I_ROOT))
    parser.add_argument("--heartbeat-sec", type=int, default=20)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        run(args)
        return 0
    except GateError as exc:
        write_failure_decision(args, exc)
        print(f"138V failed closed: {exc.verdict}: {exc.message}")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""138YM artifact-only INSTNCT mutation raw-helper adapter plan.

This phase plans the strict adapter needed before INSTNCT mutation/grower
generation can be compared with the byte-GRU raw-helper route. It does not
modify the helper, train, run inference, run torch forward passes, mutate
checkpoints, or import old phase runners.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
MILESTONE = "STABLE_LOOP_PHASE_LOCK_138YM_INSTNCT_MUTATION_RAW_HELPER_ADAPTER_PLAN"
DEFAULT_OUT = Path("target/pilot_wave/stable_loop_phase_lock_138ym_instnct_mutation_raw_helper_adapter_plan/smoke")
DEFAULT_138YL_ROOT = Path("target/pilot_wave/stable_loop_phase_lock_138yl_instnct_mutation_helper_integration_analysis/smoke")

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
    "138YM is artifact-only planning for a strict INSTNCT mutation raw-helper "
    "adapter. It reads existing 138YL artifacts and repo source metadata only. "
    "It does not train, infer, call shared_raw_generation_helper.py, run torch "
    "forward passes, mutate checkpoints, modify helper/backend/runtime/service/"
    "product/release surfaces, import old phase runners, delete files, deploy, "
    "or change root LICENSE."
)

REQUIRED_ARTIFACTS = [
    "queue.json",
    "progress.jsonl",
    "upstream_138yl_manifest.json",
    "analysis_config.json",
    "adapter_contract.json",
    "helper_surface_change_plan.json",
    "instnct_checkpoint_contract.json",
    "prompt_encoder_contract.json",
    "iterative_propagation_schedule.json",
    "output_decoder_contract.json",
    "forbidden_metadata_policy.json",
    "canary_and_ast_gate_plan.json",
    "determinism_plan.json",
    "comparison_eval_plan.json",
    "target_138yn_milestone_plan.json",
    "diagnostic_gap_register.json",
    "risk_register.json",
    "decision.json",
    "summary.json",
    "report.md",
]


class GateError(Exception):
    def __init__(self, decision: str, next_step: str, message: str, details: dict[str, Any] | None = None):
        super().__init__(message)
        self.decision = decision
        self.next_step = next_step
        self.message = message
        self.details = details or {}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def resolve_repo_path(path: str | Path) -> Path:
    raw = Path(path)
    return raw if raw.is_absolute() else (REPO_ROOT / raw).resolve()


def resolve_target_out(path: str | Path) -> Path:
    resolved = resolve_repo_path(path)
    try:
        relative = resolved.relative_to(REPO_ROOT)
    except ValueError as exc:
        raise GateError("adapter_plan_boundary_failure", "138YM_BOUNDARY_FAILURE", "--out must stay inside repo") from exc
    parts = [part.lower() for part in relative.parts]
    if len(parts) < 2 or parts[0] != "target" or parts[1] != "pilot_wave":
        raise GateError("adapter_plan_boundary_failure", "138YM_BOUNDARY_FAILURE", "--out must stay under target/pilot_wave")
    return resolved


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


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


def append_progress(out: Path, event: str, **details: Any) -> None:
    append_jsonl(out / "progress.jsonl", {"ts": utc_now(), "event": event, "details": details})


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def require_138yl(root: Path) -> dict[str, Any]:
    missing = [
        name
        for name in [
            "decision.json",
            "summary.json",
            "helper_backend_audit.json",
            "generator_contract_gap_analysis.json",
            "target_138ym_milestone_plan.json",
        ]
        if not (root / name).exists()
    ]
    if missing:
        raise GateError(
            "upstream_138yl_artifact_missing",
            "138YM_UPSTREAM_138YL_ARTIFACT_MISSING",
            "required 138YL artifacts missing",
            {"root": rel(root), "missing": missing},
        )
    decision = read_json(root / "decision.json")
    helper = read_json(root / "helper_backend_audit.json")
    gaps = read_json(root / "generator_contract_gap_analysis.json")
    if decision.get("decision") != "instnct_mutation_helper_integration_analysis_complete":
        raise GateError(
            "upstream_138yl_route_mismatch",
            "138YM_UPSTREAM_138YL_ROUTE_RECHECK",
            "138YL decision is not the expected adapter-plan route",
            {"actual": decision.get("decision")},
        )
    if decision.get("next") != "138YM_INSTNCT_MUTATION_RAW_HELPER_ADAPTER_PLAN":
        raise GateError(
            "upstream_138yl_route_mismatch",
            "138YM_UPSTREAM_138YL_ROUTE_RECHECK",
            "138YL next is not 138YM",
            {"actual": decision.get("next")},
        )
    if helper.get("adapter_required_for_instnct_raw_generation") is not True:
        raise GateError(
            "upstream_138yl_adapter_gap_missing",
            "138YM_UPSTREAM_138YL_EVIDENCE_RECHECK",
            "138YL did not preserve the adapter-required gap",
        )
    return {
        "root": rel(root),
        "decision": decision.get("decision"),
        "next": decision.get("next"),
        "instnct_helper_adapter_required": decision.get("instnct_helper_adapter_required"),
        "real_raw_value_grounding_comparison_ready": decision.get("real_raw_value_grounding_comparison_ready"),
        "helper_backend": helper.get("helper_backend"),
        "adapter_required_for_instnct_raw_generation": helper.get("adapter_required_for_instnct_raw_generation"),
        "blocking_gap_count": gaps.get("blocking_gap_count"),
        "toy_mutation_signal_present": gaps.get("toy_mutation_signal_present"),
        "hardened_phase_transport_blocker_present": gaps.get("hardened_phase_transport_blocker_present"),
    }


def build_artifacts(upstream: dict[str, Any]) -> dict[str, Any]:
    adapter_contract = {
        "schema_version": "phase_138ym_adapter_contract_v1",
        "adapter_name": "instnct_mutation_raw_generation_adapter",
        "backend_name": "repo_local_instnct_mutation_graph",
        "same_request_keys_as_shared_helper": [
            "prompt",
            "checkpoint_path",
            "checkpoint_hash",
            "seed",
            "max_new_tokens",
            "generation_config",
        ],
        "forbidden_request_material": [
            "expected_output",
            "expected_payload",
            "expected_answer",
            "labels",
            "oracle_data",
            "scorer_metadata",
            "gold_output",
            "eval_family",
            "answer",
            "expected_values",
        ],
        "generated_text_before_scoring_required": True,
        "deterministic_replay_required": True,
        "no_post_generation_repair": True,
        "no_oracle_rerank": True,
    }

    helper_surface_change_plan = {
        "schema_version": "phase_138ym_helper_surface_change_plan_v1",
        "selected_strategy": "backend_dispatch_extension_after_contract",
        "current_backend": upstream["helper_backend"],
        "new_backend": adapter_contract["backend_name"],
        "allowed_change_scope_for_future_probe": [
            "shared_raw_generation_helper.py backend dispatch only",
            "new loader for INSTNCT checkpoint manifest only",
            "new deterministic generator path only",
            "no expected/scorer/oracle request fields",
            "no public API/service/deploy/product surface changes",
        ],
        "rejected_strategies": {
            "separate_untrusted_helper": "not comparable to the strict shared helper evidence chain",
            "modify_helper_without_canary_plan": "risks corrupting 135E helper integrity",
            "wrap_byte_gru_outputs": "does not test INSTNCT mutation architecture",
        },
    }

    checkpoint_contract = {
        "schema_version": "phase_138ym_instnct_checkpoint_contract_v1",
        "required_manifest_fields": [
            "backend_name",
            "schema_version",
            "network_state_path",
            "projection_state_path",
            "prompt_encoder",
            "output_decoder",
            "propagation_schedule",
            "source_commit",
            "checkpoint_hash",
        ],
        "backend_name": adapter_contract["backend_name"],
        "source_checkpoint_mutation_allowed": False,
        "target_checkpoint_under_target_only": True,
        "must_load_without_training": True,
        "must_fail_closed_on_unknown_schema": True,
    }

    prompt_encoder_contract = {
        "schema_version": "phase_138ym_prompt_encoder_contract_v1",
        "purpose": "Convert prompt bytes into deterministic INSTNCT input events without expected-answer leakage.",
        "allowed_inputs": ["prompt", "seed", "generation_config"],
        "forbidden_inputs": adapter_contract["forbidden_request_material"],
        "encoding_requirements": [
            "byte-stable UTF-8 handling",
            "no expected answer fields",
            "no scorer metadata",
            "deterministic event ordering",
            "records prompt_hash and encoder_config_hash",
        ],
    }

    propagation = {
        "schema_version": "phase_138ym_iterative_propagation_schedule_v1",
        "default_ticks_per_generated_byte": 8,
        "supports_iterative_refinement": True,
        "threshold_gate_behavior": "side pockets may affect state only through explicit gated writeback",
        "required_metrics": [
            "tick_count",
            "pocket_activation_rate",
            "writeback_rate",
            "gate_open_rate",
            "ablation_delta",
            "highway_retention",
        ],
        "determinism_requirements": [
            "fixed seed",
            "fixed mutation/eval order",
            "no wall-clock-dependent sampling",
            "stable serialized trace hashes",
        ],
    }

    decoder = {
        "schema_version": "phase_138ym_output_decoder_contract_v1",
        "purpose": "Convert INSTNCT readout state into raw generated text bytes.",
        "must_emit_raw_generated_text": True,
        "may_not_use_expected_output": True,
        "may_not_use_regex_fixer": True,
        "may_not_use_post_generation_repair": True,
        "required_trace_fields": [
            "generated_byte_ids",
            "generated_text_hash",
            "generation_trace_hash",
            "per_byte_readout_scores_hash",
        ],
    }

    forbidden_policy = {
        "schema_version": "phase_138ym_forbidden_metadata_policy_v1",
        "helper_request_allowed_keys": adapter_contract["same_request_keys_as_shared_helper"],
        "helper_request_forbidden_keys": adapter_contract["forbidden_request_material"],
        "request_rejection_required": True,
        "nested_generation_config_scan_required": True,
        "expected_output_canary_required": True,
    }

    canary_ast = {
        "schema_version": "phase_138ym_canary_and_ast_gate_plan_v1",
        "required_gates": [
            "forbidden-input rejection",
            "expected-output canary",
            "AST shortcut scan over helper, runner, checker, adapter",
            "helper provenance verification",
            "checkpoint hash verification",
            "generated_before_scoring report",
        ],
        "forbidden_implementation_patterns": [
            "old phase runner import",
            "oracle/rerank/verifier",
            "expected-output construction",
            "regex fixer",
            "JSON mode",
            "post-generation repair",
            "best-of-n",
            "threshold weakening",
        ],
    }

    determinism = {
        "schema_version": "phase_138ym_determinism_plan_v1",
        "must_record": [
            "seed",
            "source checkpoint hash",
            "adapter source hash",
            "helper source hash",
            "prompt encoder config hash",
            "propagation schedule hash",
            "output decoder config hash",
        ],
        "replay_must_match": [
            "generated_text_hashes",
            "generation_trace_hashes",
            "per-row pass/fail",
            "aggregate metrics",
            "decision-critical metrics",
        ],
    }

    comparison_eval = {
        "schema_version": "phase_138ym_comparison_eval_plan_v1",
        "byte_gru_baseline_source": "138YK aggregate metrics",
        "instnct_probe_target": "138YN adapter smoke first, then 138YO value-grounding comparison",
        "fair_comparison_requirements": [
            "same eval rows",
            "same helper request key policy",
            "generated_text before scoring",
            "same scorer controls",
            "same deterministic replay standard",
            "separate report for pocket ablation and highway retention",
        ],
        "primary_metrics": [
            "answer_value_accuracy",
            "intra_family_contrastive_accuracy",
            "family_default_attractor_rate",
            "same_value_for_all_rows_rate",
            "pocket_ablation_delta",
            "highway_retention",
            "phase_transport_success_rate",
        ],
    }

    target_138yn = {
        "schema_version": "phase_138ym_target_138yn_milestone_plan_v1",
        "milestone": "138YN_INSTNCT_MUTATION_RAW_HELPER_ADAPTER_PROBE",
        "type": "targeted adapter/probe",
        "helper_backend_modification_allowed": True,
        "allowed_helper_change_scope": helper_surface_change_plan["allowed_change_scope_for_future_probe"],
        "train_allowed": False,
        "source_checkpoint_mutation_allowed": False,
        "public_api_claim_allowed": False,
        "clean_negative_accepted": True,
        "required_artifacts": [
            "queue.json",
            "progress.jsonl",
            "upstream_138ym_manifest.json",
            "adapter_contract.json",
            "helper_provenance_verification.json",
            "forbidden_input_rejection_report.json",
            "expected_output_canary_report.json",
            "ast_shortcut_scan_report.json",
            "instnct_checkpoint_manifest.json",
            "prompt_encoder_trace.jsonl",
            "iterative_propagation_trace.jsonl",
            "raw_generation_trace.jsonl",
            "raw_generation_results.jsonl",
            "generated_before_scoring_report.json",
            "determinism_replay_report.json",
            "decision.json",
            "summary.json",
            "report.md",
        ],
        "success_route": {
            "decision": "instnct_mutation_raw_helper_adapter_probe_complete",
            "next": "138YO_INSTNCT_MUTATION_VALUE_GROUNDING_COMPARISON_PROBE",
        },
        "failure_routes": {
            "helper_integrity_failure": "135G_RAW_HELPER_ORACLE_SHORTCUT_REMOVAL",
            "adapter_generation_missing": "138YNA_INSTNCT_ADAPTER_GENERATION_FAILURE_ANALYSIS",
            "determinism_failure": "138N_DETERMINISM_FAILURE_ANALYSIS",
            "phase_transport_blocker": "138YM_PHASE_TRANSPORT_BLOCKER_ANALYSIS",
        },
    }

    return {
        "adapter_contract": adapter_contract,
        "helper_surface_change_plan": helper_surface_change_plan,
        "instnct_checkpoint_contract": checkpoint_contract,
        "prompt_encoder_contract": prompt_encoder_contract,
        "iterative_propagation_schedule": propagation,
        "output_decoder_contract": decoder,
        "forbidden_metadata_policy": forbidden_policy,
        "canary_and_ast_gate_plan": canary_ast,
        "determinism_plan": determinism,
        "comparison_eval_plan": comparison_eval,
        "target_138yn_milestone_plan": target_138yn,
    }


def write_summary_and_report(out: Path, decision: dict[str, Any], status: str = "complete") -> None:
    summary = {
        "schema_version": "phase_138ym_summary_v1",
        "milestone": MILESTONE,
        "status": status,
        "boundary": BOUNDARY_TEXT,
        "artifact_only": True,
        "planning_only": True,
        "training_performed": False,
        "new_helper_inference_run": False,
        "shared_helper_called": False,
        "torch_forward_pass_run": False,
        "checkpoint_mutated": False,
        "helper_backend_modified": False,
        **FALSE_FLAGS,
        **decision,
    }
    write_json(out / "summary.json", summary)
    lines = [
        f"# {MILESTONE} Result",
        "",
        "## Decision",
        "",
        "```text",
        f"decision = {decision.get('decision')}",
        f"next = {decision.get('next')}",
        "```",
        "",
        "## Adapter Plan",
        "",
        "138YM plans the strict adapter needed before INSTNCT mutation/grower generation can be compared against the byte-GRU helper route.",
        "",
        "Required future backend:",
        "",
        "```text",
        "repo_local_instnct_mutation_graph",
        "```",
        "",
        "Future 138YN may modify helper backend dispatch only inside a separate checked probe. 138YM itself does not modify helper/backend code.",
        "",
        "Raw assistant capability remains quarantined. Structured/tool capability remains invalidated. This is not GPT-like readiness, not open-domain assistant readiness, not production chat, not public API, not deployment readiness, and not safety alignment.",
    ]
    write_text(out / "report.md", "\n".join(lines) + "\n")


def fail(out: Path, error: GateError) -> int:
    decision = {
        "schema_version": "phase_138ym_decision_v1",
        "decision": error.decision,
        "next": error.next_step,
        "error": error.message,
        "details": error.details,
        **FALSE_FLAGS,
    }
    write_json(out / "decision.json", decision)
    write_summary_and_report(out, decision, status="failed")
    append_progress(out, "final verdict", status="failed", decision=error.decision, next=error.next_step)
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description=MILESTONE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--upstream-138yl-root", type=Path, default=DEFAULT_138YL_ROOT)
    parser.add_argument("--heartbeat-sec", type=float, default=20.0)
    args = parser.parse_args()

    out = resolve_target_out(args.out)
    out.mkdir(parents=True, exist_ok=True)
    append_progress(out, "startup", milestone=MILESTONE)

    try:
        queue = {
            "schema_version": "phase_138ym_queue_v1",
            "milestone": MILESTONE,
            "stages": [
                "startup",
                "upstream verification",
                "adapter contract",
                "helper surface plan",
                "checkpoint contract",
                "prompt encoder contract",
                "propagation schedule",
                "decoder contract",
                "forbidden metadata policy",
                "canary and ast gate plan",
                "determinism plan",
                "comparison eval plan",
                "target 138YN plan",
                "decision",
                "final verdict",
            ],
            "required_artifacts": REQUIRED_ARTIFACTS,
        }
        write_json(out / "queue.json", queue)

        config = {
            "schema_version": "phase_138ym_analysis_config_v1",
            "milestone": MILESTONE,
            "boundary": BOUNDARY_TEXT,
            "artifact_only": True,
            "planning_only": True,
            "training_performed": False,
            "new_helper_inference_run": False,
            "shared_helper_called": False,
            "torch_forward_pass_run": False,
            "checkpoint_mutated": False,
            "helper_backend_modified": False,
            "heartbeat_sec": args.heartbeat_sec,
            "upstream_138yl_root": rel(resolve_repo_path(args.upstream_138yl_root)),
        }
        write_json(out / "analysis_config.json", config)

        upstream = require_138yl(resolve_repo_path(args.upstream_138yl_root))
        write_json(out / "upstream_138yl_manifest.json", upstream)
        append_progress(out, "upstream verification", decision=upstream["decision"], adapter_required=upstream["adapter_required_for_instnct_raw_generation"])

        artifacts = build_artifacts(upstream)
        artifact_to_file = {
            "adapter_contract": "adapter_contract.json",
            "helper_surface_change_plan": "helper_surface_change_plan.json",
            "instnct_checkpoint_contract": "instnct_checkpoint_contract.json",
            "prompt_encoder_contract": "prompt_encoder_contract.json",
            "iterative_propagation_schedule": "iterative_propagation_schedule.json",
            "output_decoder_contract": "output_decoder_contract.json",
            "forbidden_metadata_policy": "forbidden_metadata_policy.json",
            "canary_and_ast_gate_plan": "canary_and_ast_gate_plan.json",
            "determinism_plan": "determinism_plan.json",
            "comparison_eval_plan": "comparison_eval_plan.json",
            "target_138yn_milestone_plan": "target_138yn_milestone_plan.json",
        }
        for key, filename in artifact_to_file.items():
            write_json(out / filename, artifacts[key])
            append_progress(out, key.replace("_", " "), artifact=filename)

        gaps = {
            "schema_version": "phase_138ym_diagnostic_gap_register_v1",
            "diagnostic_gaps": [
                "No helper-compatible INSTNCT text-generation backend exists yet.",
                "No measured hidden-state/output-head/grower/scout mechanism is claimed.",
                "No 138YK value-grounding comparison exists for INSTNCT mutation until adapter probe passes.",
                "Phase transport blocker remains preserved from phase-lock 004.",
            ],
        }
        risks = {
            "schema_version": "phase_138ym_risk_register_v1",
            "risks": [
                {"risk": "helper_integrity_regression", "mitigation": "future 138YN must run canary, AST, provenance, and forbidden-input gates"},
                {"risk": "adapter_becomes_oracle", "mitigation": "same allowed request keys and no expected/scorer metadata"},
                {"risk": "toy_signal_overclaim", "mitigation": "138YN adapter smoke precedes any 138YO value-grounding comparison"},
                {"risk": "black_box_run", "mitigation": "continuous progress.jsonl and report refresh in every future probe"},
            ],
        }
        write_json(out / "diagnostic_gap_register.json", gaps)
        write_json(out / "risk_register.json", risks)
        append_progress(out, "risk and diagnostic gaps", gap_count=len(gaps["diagnostic_gaps"]))

        decision = {
            "schema_version": "phase_138ym_decision_v1",
            "decision": "instnct_mutation_raw_helper_adapter_plan_complete",
            "next": "138YN_INSTNCT_MUTATION_RAW_HELPER_ADAPTER_PROBE",
            "adapter_backend_name": "repo_local_instnct_mutation_graph",
            "future_helper_backend_modification_required": True,
            "helper_modified_in_138ym": False,
            "train_allowed_in_138ym": False,
            "real_value_grounding_comparison_ready": False,
            "value_grounding_comparison_after": "138YO_INSTNCT_MUTATION_VALUE_GROUNDING_COMPARISON_PROBE",
            **FALSE_FLAGS,
        }
        write_json(out / "decision.json", decision)
        append_progress(out, "decision", decision=decision["decision"], next=decision["next"])
        write_summary_and_report(out, decision)
        append_progress(out, "final verdict", status="complete", decision=decision["decision"], next=decision["next"])
        return 0
    except GateError as exc:
        return fail(out, exc)


if __name__ == "__main__":
    raise SystemExit(main())

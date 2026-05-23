#!/usr/bin/env python3
"""138YL artifact-only INSTNCT mutation helper integration analysis.

This phase compares the current byte-GRU raw-helper route with the repo-local
INSTNCT mutation/highway-pocket surfaces. It does not train, run helper
inference, call torch forward passes, mutate checkpoints, modify helper/backend
code, or import old phase runners.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
MILESTONE = "STABLE_LOOP_PHASE_LOCK_138YL_INSTNCT_MUTATION_HELPER_INTEGRATION_ANALYSIS"
DEFAULT_OUT = Path("target/pilot_wave/stable_loop_phase_lock_138yl_instnct_mutation_helper_integration_analysis/smoke")
DEFAULT_138YK_ROOT = Path("target/pilot_wave/stable_loop_phase_lock_138yk_family_default_suppressed_contrastive_repair_probe/smoke")
DEFAULT_MUTATION_ROOT = Path("target/pilot_wave/highway_pocket_mutation_001/current_micro_20260523")
DEFAULT_PHASE004_ROOT = Path("target/pilot_wave/stable_loop_phase_lock_004_mutation_credit_assignment/smoke_bounded")

HELPER_PATH = Path("scripts/probes/shared_raw_generation_helper.py")
MUTATION_RUNNER_PATH = Path("tools/diag_highway_pocket_mutation.py")
EVOLUTION_PATH = Path("instnct-core/src/evolution.rs")
NETWORK_PATH = Path("instnct-core/src/network.rs")
GROWER_PATH = Path("instnct-core/examples/neuron_grower.rs")
EVOLVE_LANGUAGE_PATH = Path("instnct-core/examples/evolve_language.rs")
EVOLVE_BYTEPAIR_PATH = Path("instnct-core/examples/evolve_bytepair_proj.rs")

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
    "138YL is artifact-only integration analysis. It reads existing 138YK, "
    "HIGHWAY_POCKET_MUTATION_001, phase-lock 004, and repo source artifacts. "
    "It does not train, run new helper inference, call shared_raw_generation_helper.py, "
    "run torch forward passes, mutate checkpoints, modify helper/backend/runtime/"
    "service/product/release surfaces, import old phase runners, delete files, "
    "deploy, or change root LICENSE."
)

REQUIRED_ARTIFACTS = [
    "queue.json",
    "progress.jsonl",
    "analysis_config.json",
    "upstream_138yk_manifest.json",
    "mutation_microprobe_manifest.json",
    "phase_lock_004_manifest.json",
    "helper_backend_audit.json",
    "instnct_mutation_surface_map.json",
    "generator_contract_gap_analysis.json",
    "gradient_vs_mutation_credit_report.json",
    "external_research_manifest.json",
    "integration_options.json",
    "integration_risk_register.json",
    "target_138ym_milestone_plan.json",
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
        raise GateError(
            "instnct_mutation_integration_boundary_failure",
            "138YL_BOUNDARY_FAILURE",
            "--out must remain inside the repository",
        ) from exc
    parts = [part.lower() for part in relative.parts]
    if len(parts) < 2 or parts[0] != "target" or parts[1] != "pilot_wave":
        raise GateError(
            "instnct_mutation_integration_boundary_failure",
            "138YL_BOUNDARY_FAILURE",
            "--out must stay under target/pilot_wave",
        )
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


def sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_summary(path: Path, patterns: list[str]) -> dict[str, Any]:
    resolved = resolve_repo_path(path)
    text = resolved.read_text(encoding="utf-8") if resolved.exists() else ""
    return {
        "path": rel(resolved),
        "exists": resolved.exists(),
        "sha256": sha256_file(resolved),
        "line_count": len(text.splitlines()),
        "matched_patterns": {pattern: bool(re.search(pattern, text)) for pattern in patterns},
    }


def write_summary_and_report(out: Path, decision: dict[str, Any], status: str = "complete") -> None:
    summary = {
        "schema_version": "phase_138yl_summary_v1",
        "milestone": MILESTONE,
        "status": status,
        "boundary": BOUNDARY_TEXT,
        "artifact_only": True,
        "training_performed": False,
        "new_helper_inference_run": False,
        "shared_helper_called": False,
        "torch_forward_pass_run": False,
        "checkpoint_mutated": False,
        "helper_backend_modified": False,
        "runtime_surface_mutated": False,
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
        f"adapter_required = {decision.get('instnct_helper_adapter_required')}",
        "```",
        "",
        "## Boundary",
        "",
        BOUNDARY_TEXT,
        "",
        "## Readout",
        "",
        "- The current strict raw helper backend remains `repo_local_checkpoint_byte_lm`.",
        "- 138YK used a byte-GRU checkpoint and failed value grounding with a clean negative.",
        "- The runner-local highway/pocket mutation probe shows non-decorative pocket signal on toy tasks.",
        "- The stronger phase-lock 004 artifact still says the hardened spatial credit-assignment problem is not solved.",
        "- INSTNCT mutation is not yet a helper-compatible raw generation backend.",
        "",
        "## Recommendation",
        "",
        "Implement a planning milestone for a strict INSTNCT mutation raw-helper adapter before any value-grounding comparison.",
        "",
        "Raw assistant capability remains quarantined. Structured/tool capability remains invalidated. This is not GPT-like readiness, not open-domain assistant readiness, not production chat, not public API, not deployment readiness, and not safety alignment.",
    ]
    write_text(out / "report.md", "\n".join(lines) + "\n")


def fail(out: Path, error: GateError) -> int:
    decision = {
        "schema_version": "phase_138yl_decision_v1",
        "decision": error.decision,
        "next": error.next_step,
        "error": error.message,
        "details": error.details,
        "instnct_helper_adapter_required": None,
        **FALSE_FLAGS,
    }
    write_json(out / "decision.json", decision)
    write_summary_and_report(out, decision, status="failed")
    append_progress(out, "final verdict", status="failed", decision=error.decision, next=error.next_step)
    return 1


def require_138yk(root: Path) -> dict[str, Any]:
    decision = read_json(root / "decision.json")
    aggregate = read_json(root / "aggregate_metrics.json")
    source_manifest = read_json(root / "source_checkpoint_integrity_manifest.json")
    target_manifest = read_json(root / "target_checkpoint_integrity_manifest.json")
    missing = [
        name
        for name in [
            "decision.json",
            "aggregate_metrics.json",
            "source_checkpoint_integrity_manifest.json",
            "target_checkpoint_integrity_manifest.json",
            "generated_before_scoring_report.json",
            "determinism_replay_report.json",
        ]
        if not (root / name).exists()
    ]
    if missing:
        raise GateError(
            "upstream_138yk_artifact_missing",
            "138YL_UPSTREAM_138YK_ARTIFACT_MISSING",
            "138YK required artifacts are missing",
            {"missing": missing, "root": rel(root)},
        )
    if decision.get("decision") != "family_default_shortcut_persists":
        raise GateError(
            "upstream_138yk_profile_recheck",
            "138YL_UPSTREAM_138YK_PROFILE_RECHECK",
            "138YK no longer has the expected clean-negative route",
            {"decision": decision.get("decision")},
        )
    required = {
        "shared_raw_generation_helper_used": True,
        "source_checkpoint_unchanged": True,
        "target_checkpoint_changed": True,
        "determinism_replay_passed": True,
        "generated_text_produced_before_scoring": True,
        "parrot_trap_detected": False,
        "stale_chat_fragment_rate": 0.0,
        "train_namespace_leak_rate": 0.0,
    }
    mismatches = {key: {"expected": value, "actual": decision.get(key)} for key, value in required.items() if decision.get(key) != value}
    if mismatches:
        raise GateError(
            "upstream_138yk_integrity_recheck",
            "138YL_UPSTREAM_138YK_INTEGRITY_RECHECK",
            "138YK helper/eval integrity fields do not match expected values",
            mismatches,
        )
    return {
        "root": rel(root),
        "decision": decision.get("decision"),
        "next": decision.get("next"),
        "verdict": decision.get("verdict"),
        "helper_used": decision.get("shared_raw_generation_helper_used"),
        "source_backend_name": source_manifest.get("backend_name"),
        "source_checkpoint_path": source_manifest.get("source_checkpoint_path"),
        "target_checkpoint_path": target_manifest.get("target_checkpoint_path"),
        "answer_value_accuracy": aggregate.get("answer_value_accuracy"),
        "intra_family_contrastive_accuracy": aggregate.get("intra_family_contrastive_accuracy"),
        "family_default_attractor_rate": aggregate.get("family_default_attractor_rate"),
        "intra_family_mode_collapse_rate": aggregate.get("intra_family_mode_collapse_rate"),
        "determinism_replay_passed": decision.get("determinism_replay_passed"),
    }


def mutation_manifest(root: Path) -> dict[str, Any]:
    summary = read_json(root / "summary.json")
    missing = [name for name in ["summary.json", "report.md", "progress.jsonl", "queue.json"] if not (root / name).exists()]
    if missing:
        raise GateError(
            "mutation_microprobe_artifact_missing",
            "138YL_MUTATION_MICROPROBE_ARTIFACT_MISSING",
            "mutation microprobe artifacts missing",
            {"missing": missing, "root": rel(root)},
        )
    arm_summary = summary.get("arm_summary", [])
    by_arm = {row.get("arm"): row for row in arm_summary}
    return {
        "root": rel(root),
        "verdicts": summary.get("verdicts", []),
        "completed_jobs": summary.get("completed_jobs"),
        "total_jobs": summary.get("total_jobs"),
        "gated_symbolic_final_answer_accuracy": by_arm.get("HIGHWAY_WITH_GATED_POCKETS", {}).get("final_answer_accuracy"),
        "highway_symbolic_final_answer_accuracy": by_arm.get("HIGHWAY_ONLY", {}).get("final_answer_accuracy"),
        "gated_symbolic_ablation_drop": by_arm.get("HIGHWAY_WITH_GATED_POCKETS", {}).get("pocket_ablation_max_drop"),
        "gated_phase_final_accuracy": by_arm.get("HIGHWAY_WITH_GATED_POCKETS_PHASE", {}).get("phase_final_accuracy"),
        "highway_phase_final_accuracy": by_arm.get("HIGHWAY_ONLY_PHASE", {}).get("phase_final_accuracy"),
        "gated_phase_ablation_drop": by_arm.get("HIGHWAY_WITH_GATED_POCKETS_PHASE", {}).get("pocket_ablation_phase_drop"),
        "claim_boundary": "runner-local mutation smoke; not helper integration and not language/value-grounding proof",
    }


def phase004_manifest(root: Path) -> dict[str, Any]:
    summary = read_json(root / "summary.json")
    missing = [name for name in ["summary.json", "report.md", "queue.json"] if not (root / name).exists()]
    if missing:
        raise GateError(
            "phase004_artifact_missing",
            "138YL_PHASE004_ARTIFACT_MISSING",
            "phase-lock 004 artifacts missing",
            {"missing": missing, "root": rel(root)},
        )
    by_arm = {row.get("arm"): row for row in summary.get("arm_summary", [])}
    return {
        "root": rel(root),
        "verdicts": summary.get("verdicts", []),
        "gated_phase_final_accuracy": by_arm.get("HIGHWAY_WITH_GATED_POCKETS_PHASE", {}).get("phase_final_accuracy"),
        "highway_phase_final_accuracy": by_arm.get("HIGHWAY_ONLY_PHASE", {}).get("phase_final_accuracy"),
        "gated_highway_phase_retention": by_arm.get("HIGHWAY_WITH_GATED_POCKETS_PHASE", {}).get("highway_phase_retention"),
        "phase_transport_blocker_supported": "PHASE_TRANSPORT_IS_BLOCKER" in summary.get("verdicts", []),
        "phase_credit_assignment_not_solved": "PHASE_CREDIT_ASSIGNMENT_NOT_SOLVED" in summary.get("verdicts", []),
    }


def helper_audit() -> dict[str, Any]:
    helper = source_summary(
        HELPER_PATH,
        [
            r'HELPER_BACKEND\s*=\s*"repo_local_checkpoint_byte_lm"',
            r"class ByteRNNLM",
            r"class ByteMLPLM",
            r"ALLOWED_REQUEST_KEYS",
            r"FORBIDDEN_REQUEST_KEYS",
            r"instnct",
            r"Network",
        ],
    )
    patterns = helper["matched_patterns"]
    return {
        **helper,
        "helper_backend": "repo_local_checkpoint_byte_lm" if patterns.get(r'HELPER_BACKEND\s*=\s*"repo_local_checkpoint_byte_lm"') else "unknown",
        "supports_byte_rnn_lm": patterns.get(r"class ByteRNNLM"),
        "supports_byte_mlp_lm": patterns.get(r"class ByteMLPLM"),
        "strict_request_contract_present": patterns.get(r"ALLOWED_REQUEST_KEYS") and patterns.get(r"FORBIDDEN_REQUEST_KEYS"),
        "instnct_backend_detected": patterns.get(r"instnct") or patterns.get(r"Network"),
        "adapter_required_for_instnct_raw_generation": not (patterns.get(r"instnct") or patterns.get(r"Network")),
    }


def surface_map() -> dict[str, Any]:
    return {
        "mutation_runner": source_summary(
            MUTATION_RUNNER_PATH,
            [
                r"HIGHWAY_POCKET_MUTATION_001",
                r"SYMBOLIC_ARMS",
                r"PHASE_ARMS",
                r"MUTATION_OPERATORS",
                r"pocket_add_internal_edge",
                r"pocket_move_writeback",
                r"heartbeat",
            ],
        ),
        "evolution_core": source_summary(
            EVOLUTION_PATH,
            [
                r"evolution_step",
                r"mutate",
                r"Accepted",
                r"Rejected",
                r"restore_state",
                r"fitness_fn",
            ],
        ),
        "network_core": source_summary(
            NETWORK_PATH,
            [
                r"struct Network",
                r"save_state",
                r"restore_state",
                r"apply_undo",
                r"mutate_add_edge",
                r"mutate_add_loop",
            ],
        ),
        "grower_example": source_summary(
            GROWER_PATH,
            [
                r"Neuron Grower",
                r"scout",
                r"proposals",
                r"state",
                r"threshold",
            ],
        ),
        "language_evolution_example": source_summary(
            EVOLVE_LANGUAGE_PATH,
            [
                r"evolution_step",
                r"smooth",
                r"fitness",
                r"projection",
                r"propagate",
            ],
        ),
        "bytepair_evolution_example": source_summary(
            EVOLVE_BYTEPAIR_PATH,
            [
                r"evolution_step",
                r"smooth",
                r"fitness",
                r"Int8Projection",
                r"propagate",
            ],
        ),
    }


def contract_gap(helper: dict[str, Any], mutation: dict[str, Any], phase004: dict[str, Any]) -> dict[str, Any]:
    gaps = [
        {
            "gap": "helper_backend_gap",
            "evidence": "shared helper supports repo-local byte LM checkpoints only",
            "required_resolution": "define strict INSTNCT raw-generation adapter without scorer/oracle metadata",
            "blocks_real_raw_comparison": True,
        },
        {
            "gap": "checkpoint_format_gap",
            "evidence": "138YK helper consumes PyTorch ByteRNN/ByteMLP model.pt checkpoints; INSTNCT mutation artifacts are graph/topology/evolution surfaces",
            "required_resolution": "machine-readable INSTNCT checkpoint/load contract for raw generation",
            "blocks_real_raw_comparison": True,
        },
        {
            "gap": "prompt_to_output_contract_gap",
            "evidence": "mutation probes are toy symbolic/phase tasks, not ANSWER=E text generation over 138YK eval prompts",
            "required_resolution": "deterministic prompt encoder, iterative propagation schedule, and byte/text decoder",
            "blocks_real_raw_comparison": True,
        },
        {
            "gap": "credit_assignment_scope_gap",
            "evidence": "HIGHWAY_POCKET_MUTATION_001 toy bridge positive, but phase-lock 004 says PHASE_CREDIT_ASSIGNMENT_NOT_SOLVED",
            "required_resolution": "bounded helper-compatible probe that tests value grounding and transport separately",
            "blocks_real_raw_comparison": False,
        },
    ]
    return {
        "schema_version": "phase_138yl_generator_contract_gap_analysis_v1",
        "helper_backend": helper.get("helper_backend"),
        "adapter_required_for_instnct_raw_generation": helper.get("adapter_required_for_instnct_raw_generation"),
        "toy_mutation_signal_present": "HIGHWAY_POCKET_MUTATION_POSITIVE" in mutation.get("verdicts", []),
        "hardened_phase_transport_blocker_present": phase004.get("phase_transport_blocker_supported"),
        "gaps": gaps,
        "blocking_gap_count": sum(1 for gap in gaps if gap["blocks_real_raw_comparison"]),
    }


def integration_options() -> dict[str, Any]:
    options = [
        {
            "option": "do_nothing_byte_gru_only",
            "verdict": "reject_for_user_goal",
            "reason": "keeps testing the fallback byte-GRU and does not test the requested mutation architecture",
        },
        {
            "option": "modify_shared_helper_immediately",
            "verdict": "reject_now",
            "reason": "would mutate a trusted evidence surface before the adapter contract and canary requirements are defined",
        },
        {
            "option": "separate_instnct_adapter_plan_then_probe",
            "verdict": "recommended",
            "reason": "keeps 135E helper integrity intact while designing a strict helper-compatible INSTNCT backend with canaries",
        },
        {
            "option": "toy_mutation_benchmark_only",
            "verdict": "insufficient",
            "reason": "proves non-decorative pockets on toy tasks but not 138YK prompt-specific value grounding",
        },
    ]
    return {
        "schema_version": "phase_138yl_integration_options_v1",
        "recommended_option": "separate_instnct_adapter_plan_then_probe",
        "options": options,
    }


def target_plan() -> dict[str, Any]:
    required_artifacts = [
        "queue.json",
        "progress.jsonl",
        "upstream_138yl_manifest.json",
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
        "decision.json",
        "summary.json",
        "report.md",
    ]
    return {
        "schema_version": "phase_138yl_target_138ym_milestone_plan_v1",
        "milestone": "138YM_INSTNCT_MUTATION_RAW_HELPER_ADAPTER_PLAN",
        "type": "planning-only",
        "goal": "Design a strict raw-helper-compatible INSTNCT mutation/grower adapter before any real value-grounding comparison.",
        "train_allowed": False,
        "helper_modification_allowed": False,
        "future_helper_modification_requires_separate_contract": True,
        "source_checkpoint_mutation_allowed": False,
        "old_runner_import_allowed": False,
        "public_api_claim_allowed": False,
        "required_constraints": [
            "no expected/scorer/oracle metadata in generation request",
            "generated_text before scoring",
            "deterministic replay",
            "canary rejection",
            "AST shortcut scan",
            "helper provenance verification",
            "continuous progress artifacts",
            "side-pocket ablation metrics",
            "phase transport metrics",
            "byte-GRU baseline comparison",
        ],
        "required_artifacts": required_artifacts,
        "success_route": {
            "decision": "instnct_mutation_adapter_plan_complete",
            "next": "138YN_INSTNCT_MUTATION_VALUE_GROUNDING_PROBE",
        },
        "clean_negative_routes": {
            "adapter_contract_missing": "138YMA_INSTNCT_ADAPTER_CONTRACT_FAILURE_ANALYSIS",
            "helper_integrity_risk": "135G_RAW_HELPER_ORACLE_SHORTCUT_REMOVAL",
            "mutation_surface_not_text_generating": "138YMB_INSTNCT_TEXT_GENERATION_BRIDGE_DESIGN",
            "phase_transport_blocker_dominates": "138YM_PHASE_TRANSPORT_BLOCKER_ANALYSIS",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=MILESTONE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--upstream-138yk-root", type=Path, default=DEFAULT_138YK_ROOT)
    parser.add_argument("--mutation-root", type=Path, default=DEFAULT_MUTATION_ROOT)
    parser.add_argument("--phase004-root", type=Path, default=DEFAULT_PHASE004_ROOT)
    parser.add_argument("--heartbeat-sec", type=float, default=20.0)
    args = parser.parse_args()

    out = resolve_target_out(args.out)
    out.mkdir(parents=True, exist_ok=True)
    append_progress(out, "startup", milestone=MILESTONE, boundary=BOUNDARY_TEXT)

    try:
        queue = {
            "schema_version": "phase_138yl_queue_v1",
            "milestone": MILESTONE,
            "stages": [
                "startup",
                "upstream verification",
                "source surface audit",
                "contract gap analysis",
                "credit mechanism analysis",
                "integration option selection",
                "target 138YM plan writing",
                "decision",
                "final verdict",
            ],
            "required_artifacts": REQUIRED_ARTIFACTS,
        }
        write_json(out / "queue.json", queue)

        config = {
            "schema_version": "phase_138yl_analysis_config_v1",
            "milestone": MILESTONE,
            "boundary": BOUNDARY_TEXT,
            "artifact_only": True,
            "training_performed": False,
            "new_helper_inference_run": False,
            "shared_helper_called": False,
            "torch_forward_pass_run": False,
            "checkpoint_mutated": False,
            "helper_backend_modified": False,
            "heartbeat_sec": args.heartbeat_sec,
            "upstream_138yk_root": rel(resolve_repo_path(args.upstream_138yk_root)),
            "mutation_root": rel(resolve_repo_path(args.mutation_root)),
            "phase004_root": rel(resolve_repo_path(args.phase004_root)),
        }
        write_json(out / "analysis_config.json", config)

        upstream_138yk = require_138yk(resolve_repo_path(args.upstream_138yk_root))
        mutation = mutation_manifest(resolve_repo_path(args.mutation_root))
        phase004 = phase004_manifest(resolve_repo_path(args.phase004_root))
        write_json(out / "upstream_138yk_manifest.json", upstream_138yk)
        write_json(out / "mutation_microprobe_manifest.json", mutation)
        write_json(out / "phase_lock_004_manifest.json", phase004)
        append_progress(out, "upstream verification", upstream_138yk=upstream_138yk["decision"], mutation_verdicts=mutation["verdicts"])

        helper = helper_audit()
        surfaces = surface_map()
        write_json(out / "helper_backend_audit.json", helper)
        write_json(out / "instnct_mutation_surface_map.json", surfaces)
        append_progress(out, "source surface audit", helper_backend=helper["helper_backend"], adapter_required=helper["adapter_required_for_instnct_raw_generation"])

        gaps = contract_gap(helper, mutation, phase004)
        write_json(out / "generator_contract_gap_analysis.json", gaps)
        append_progress(out, "contract gap analysis", blocking_gap_count=gaps["blocking_gap_count"])

        credit_report = {
            "schema_version": "phase_138yl_gradient_vs_mutation_credit_report_v1",
            "byte_gru_credit_mechanism": "differentiable loss gradient through ByteRNNLM/ByteMLPLM weights",
            "instnct_mutation_credit_mechanism": "fitness delta after topology/gate/writeback mutation with accept-or-rollback",
            "side_pocket_receives_backprop_gradient": False,
            "side_pocket_credit_proxy": [
                "candidate fitness improvement",
                "accepted_operator_rate",
                "pocket_ablation_drop",
                "gate_shuffle_control degradation",
                "highway_retention",
            ],
            "artifact_supported_claim": "side pockets are an output-behavior and fitness-selection mechanism in the runner-local probes, not a measured hidden-state/backprop mechanism",
        }
        write_json(out / "gradient_vs_mutation_credit_report.json", credit_report)

        external = {
            "schema_version": "phase_138yl_external_research_manifest_v1",
            "sources": [
                {
                    "title": "Evolution Strategies as a Scalable Alternative to Reinforcement Learning",
                    "url": "https://arxiv.org/abs/1703.03864",
                    "relevance": "Frames ES as black-box optimization using scalar returns rather than backprop through an environment.",
                },
                {
                    "title": "OpenAI evolution-strategies-starter",
                    "url": "https://github.com/openai/evolution-strategies-starter",
                    "relevance": "Reference implementation showing master-worker scalar-return optimization for ES.",
                },
                {
                    "title": "Efficient Evolution of Neural Network Topologies / NEAT",
                    "url": "https://nn.cs.utexas.edu/downloads/papers/stanley.cec02.pdf",
                    "relevance": "Primary neuroevolution source for evolving topology and protecting structural innovation.",
                },
            ],
            "research_conclusion": "Mutation/highway-pocket credit should be treated as gradient-free fitness selection, not as backpropagation into a side pocket.",
        }
        write_json(out / "external_research_manifest.json", external)
        append_progress(out, "credit mechanism analysis", side_pocket_backprop=False)

        options = integration_options()
        risks = {
            "schema_version": "phase_138yl_integration_risk_register_v1",
            "risks": [
                {
                    "risk": "helper_integrity_regression",
                    "severity": "high",
                    "mitigation": "plan adapter contract before helper changes; preserve strict forbidden metadata policy",
                },
                {
                    "risk": "toy_positive_overclaim",
                    "severity": "high",
                    "mitigation": "carry phase-lock 004 negative evidence and require 138YK-compatible value grounding eval",
                },
                {
                    "risk": "non_text_generating_instnct_surface",
                    "severity": "medium",
                    "mitigation": "define prompt encoder, iterative propagation schedule, and decoder before probe",
                },
                {
                    "risk": "black_box_long_run",
                    "severity": "high",
                    "mitigation": "require progress.jsonl and summary/report refresh every heartbeat in 138YM/138YN",
                },
            ],
        }
        write_json(out / "integration_options.json", options)
        write_json(out / "integration_risk_register.json", risks)
        append_progress(out, "integration option selection", recommended=options["recommended_option"])

        plan = target_plan()
        write_json(out / "target_138ym_milestone_plan.json", plan)
        append_progress(out, "target 138YM plan writing", next=plan["milestone"])

        decision = {
            "schema_version": "phase_138yl_decision_v1",
            "decision": "instnct_mutation_helper_integration_analysis_complete",
            "next": "138YM_INSTNCT_MUTATION_RAW_HELPER_ADAPTER_PLAN",
            "instnct_helper_adapter_required": True,
            "byte_gru_helper_backend_confirmed": helper["helper_backend"] == "repo_local_checkpoint_byte_lm",
            "instnct_backend_currently_in_shared_helper": bool(helper["instnct_backend_detected"]),
            "toy_mutation_signal_present": gaps["toy_mutation_signal_present"],
            "hardened_phase_transport_blocker_present": gaps["hardened_phase_transport_blocker_present"],
            "real_raw_value_grounding_comparison_ready": False,
            "reason_not_ready": "INSTNCT mutation/grower is not currently a shared_raw_generation_helper-compatible raw text generation backend.",
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

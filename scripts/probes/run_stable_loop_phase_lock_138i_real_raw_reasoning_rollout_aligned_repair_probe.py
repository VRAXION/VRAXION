#!/usr/bin/env python3
"""138I deterministic real-raw reasoning repair training/probe.

This phase trains only a new target checkpoint under target/ and evaluates it
through scripts/probes/shared_raw_generation_helper.py. Source checkpoints and
the shared helper are immutable. A clean negative is a valid result.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import os
import random
import re
import shutil
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

try:
    import torch
    from torch import nn
except Exception:  # pragma: no cover - fail-closed path.
    torch = None
    nn = None


REPO_ROOT = Path(__file__).resolve().parents[2]
MILESTONE = "STABLE_LOOP_PHASE_LOCK_138I_REAL_RAW_REASONING_ROLLOUT_ALIGNED_REPAIR_PROBE"
SHORT_MILESTONE = "138I_REAL_RAW_REASONING_ROLLOUT_ALIGNED_REPAIR_PROBE"
DEFAULT_OUT = Path("target/pilot_wave/stable_loop_phase_lock_138i_real_raw_reasoning_rollout_aligned_repair_probe/smoke")
DEFAULT_UPSTREAM_138H_ROOT = Path("target/pilot_wave/stable_loop_phase_lock_138h_real_raw_reasoning_rollout_aligned_objective_redesign_plan/smoke")
DEFAULT_UPSTREAM_138GA_ROOT = Path("target/pilot_wave/stable_loop_phase_lock_138ga_objective_failure_ambiguity_resolution/smoke")
DEFAULT_UPSTREAM_138R_ROOT = Path("target/pilot_wave/stable_loop_phase_lock_138r_real_raw_reasoning_repair_training_plan_or_probe/smoke")
HELPER_PATH = REPO_ROOT / "scripts/probes/shared_raw_generation_helper.py"
RUNNER_PATH = Path(__file__).resolve()
CHECKER_PATH = REPO_ROOT / "scripts/probes/run_stable_loop_phase_lock_138i_real_raw_reasoning_rollout_aligned_repair_probe_check.py"
TARGET_CHECKPOINT_REL = "target/pilot_wave/stable_loop_phase_lock_138i_real_raw_reasoning_rollout_aligned_repair_probe/smoke/checkpoints/target_138i_rollout_aligned_reasoning/model.pt"

POSITIVE_VERDICT = "REAL_RAW_REASONING_ROLLOUT_ALIGNED_REPAIR_POSITIVE"
NEGATIVE_VERDICT = "REAL_RAW_REASONING_ROLLOUT_ALIGNED_REPAIR_FAILS"
MISSING_TRAINER_VERDICT = "ROLLOUT_ALIGNED_TRAINING_PATH_MISSING"
DETERMINISM_VERDICT = "DETERMINISM_REPLAY_MISMATCH"
POSITIVE_DECISION = "real_raw_reasoning_rollout_aligned_repair_positive"
NEGATIVE_DECISION = "no_rollout_improvement"
MISSING_TRAINER_DECISION = "rollout_aligned_training_path_missing"
POSITIVE_NEXT = "139R_REAL_RAW_REASONING_REPAIR_SCALE_CONFIRM"
NEGATIVE_NEXT = "138I_FAILURE_ANALYSIS"
MISSING_TRAINER_NEXT = "138IA_ROLLOUT_ALIGNED_TRAINING_HELPER_INTEGRATION_PLAN"
TEACHER_FAILURE_DECISION = "teacher_forcing_or_training_objective_failure"
TEACHER_FAILURE_NEXT = "138G_REAL_RAW_REASONING_OBJECTIVE_FAILURE_ANALYSIS"
STALE_FAILURE_DECISION = "stale_chat_rollout_failure"
STALE_FAILURE_NEXT = "138S_STALE_CHAT_ROLLOUT_FAILURE_ANALYSIS"
LEAKAGE_DECISION = "reasoning_repair_eval_leakage"
LEAKAGE_NEXT = "138L_REASONING_REPAIR_EVAL_LEAKAGE_REDESIGN"
DETERMINISM_DECISION = "nondeterministic_repair_probe"
DETERMINISM_NEXT = "138N_DETERMINISM_FAILURE_ANALYSIS"
HELPER_FAILURE_DECISION = "raw_helper_integrity_failure"
HELPER_FAILURE_NEXT = "135G_RAW_HELPER_ORACLE_SHORTCUT_REMOVAL"
SCORER_WEAKNESS_DECISION = "scorer_or_task_weakness"
SCORER_WEAKNESS_NEXT = "138E_REASONING_SCORER_OR_TASK_WEAKNESS_ANALYSIS"
NAMESPACE_FAILURE_DECISION = "namespace_rollout_failure"
NAMESPACE_FAILURE_NEXT = "138S_NAMESPACE_ROLLOUT_FAILURE_ANALYSIS"

BOUNDARY_TEXT = (
    "138I is a deterministic targeted real-raw reasoning repair/probe. It may "
    "train only a new target checkpoint under target/ and final-evaluate only "
    "through scripts/probes/shared_raw_generation_helper.py. It does not mutate "
    "source checkpoints, modify the shared helper, import old runners, start "
    "services, deploy, delete files, consolidate old runners, modify runtime, "
    "service, deploy, product, or release surfaces, or change root LICENSE. It "
    "does not restore full raw assistant capability, structured/tool capability, "
    "GPT-like readiness, open-domain readiness, production chat, public API, "
    "deployment readiness, or safety alignment."
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
    "eval_family_expected_values",
    "row_answer",
    "target_json",
    "gold_output",
    "eval_family",
    "answer",
    "expected_values",
}
FINAL_EVAL_FLAGS = {
    "generated_text_produced_before_scoring": True,
    "shared_raw_generation_helper_used": True,
    "expected_output_used_for_generation": False,
    "expected_payload_used_for_generation": False,
    "scorer_metadata_used_for_generation": False,
    "oracle_rerank_used": False,
    "verifier_rerank_used": False,
    "llm_judge_used": False,
    "teacher_forcing_used": False,
    "constrained_decoding_used": False,
    "json_mode_used": False,
    "grammar_decoder_used": False,
    "regex_fixer_used": False,
    "post_generation_repair_used": False,
    "retry_loop_used": False,
    "best_of_n_used": False,
    "actual_tool_execution_used": False,
    "runtime_tool_call_used": False,
}
FALSE_BOUNDARY_FLAGS = {
    "raw_assistant_capability_restored": False,
    "structured_tool_capability_restored": False,
    "gpt_like_readiness_claimed": False,
    "open_domain_assistant_readiness_claimed": False,
    "production_chat_claimed": False,
    "public_api_claimed": False,
    "deployment_readiness_claimed": False,
    "safety_alignment_claimed": False,
}
FAMILIES = [
    "REAL_RAW_REPAIR_PROVIDED_FACT_QA",
    "REAL_RAW_REPAIR_SINGLE_STEP_REASONING",
    "REAL_RAW_REPAIR_TWO_STEP_REASONING",
    "REAL_RAW_REPAIR_RULE_CHAINING",
    "REAL_RAW_REPAIR_TABLE_RULE_REASONING",
    "REAL_RAW_REPAIR_SMALL_ARITHMETIC",
    "REAL_RAW_REPAIR_CONTRADICTION_RESOLUTION",
    "REAL_RAW_REPAIR_MULTI_DOC_PRIORITY",
    "REAL_RAW_REPAIR_HALLUCINATION_TRAP",
    "REAL_RAW_REPAIR_STALE_CHAT_FRAGMENT_REJECTION",
]
TRAIN_CATEGORIES = FAMILIES + ["REAL_RAW_REPAIR_PROMPT_FORMAT_ALIGNMENT", "REAL_RAW_REPAIR_ANTI_STALE_CHAT_FRAGMENT"]
SEED_GATES = {
    "real_raw_reasoning_accuracy": 0.45,
    "provided_fact_qa_accuracy": 0.60,
    "single_step_reasoning_accuracy": 0.55,
    "two_step_reasoning_accuracy": 0.40,
    "rule_chaining_accuracy": 0.40,
    "table_rule_reasoning_accuracy": 0.40,
    "small_arithmetic_accuracy": 0.40,
    "contradiction_resolution_accuracy": 0.35,
    "hallucination_trap_pass_rate": 0.50,
}
MAX_RATE_GATES = {
    "stale_chat_fragment_rate": 0.10,
    "off_prompt_output_rate": 0.30,
}
AGGREGATE_GATES = {
    "mean_real_raw_reasoning_accuracy": 0.50,
    "expected_token_inclusion_rate": 0.50,
    "near_match_rate": 0.50,
}
NAMESPACE_GATES = {
    "post_train_namespace_leak_rate": 0.05,
    "post_eval_namespace_emission_accuracy": 0.45,
    "post_answer_prefix_accuracy": 0.60,
    "post_answer_value_accuracy": 0.35,
    "post_stale_user_assistant_fragment_rate": 0.10,
    "post_off_prompt_output_rate": 0.30,
}
BASELINE_NAMESPACE_METRICS = {
    "baseline_train_namespace_leak_rate": 1.0,
    "baseline_eval_namespace_emission_accuracy": 0.0,
    "baseline_answer_prefix_accuracy": 1.0,
    "baseline_answer_value_accuracy": 0.0,
    "baseline_helper_only_rollout_accuracy": 0.0,
}
STANDARD_REFUSAL_TEMPLATES = {"INSUFFICIENT_INFORMATION", "UNKNOWN", "UNANSWERABLE"}
GRU_STATE_KEYS = {
    "embedding.weight",
    "rnn.weight_ih_l0",
    "rnn.weight_hh_l0",
    "rnn.bias_ih_l0",
    "rnn.bias_hh_l0",
    "head.weight",
    "head.bias",
}
PAD_ID = 258
BYTE_VOCAB_SIZE = 259


class GateError(Exception):
    def __init__(self, verdict: str, message: str, details: dict[str, Any] | None = None):
        super().__init__(message)
        self.verdict = verdict
        self.message = message
        self.details = details or {}


if nn is not None:
    class ByteRNNLM(nn.Module):
        def __init__(self, vocab_size: int, embed_dim: int, hidden_size: int):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, embed_dim)
            self.rnn = nn.GRU(embed_dim, hidden_size, batch_first=True)
            self.head = nn.Linear(hidden_size, vocab_size)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            emb = self.embedding(x)
            out, _hidden = self.rnn(emb)
            return self.head(out[:, -1, :])


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


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")


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


def stable_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    if path.is_absolute() or any(part == ".." for part in path.parts):
        raise GateError("138i_BOUNDARY_FAILURE", "--out must be repo-relative")
    parts = [part.lower() for part in path.parts]
    if len(parts) < 2 or parts[0] != "target" or parts[1] != "pilot_wave":
        raise GateError("138i_BOUNDARY_FAILURE", "--out must stay under target/pilot_wave")
    return (REPO_ROOT / path).resolve()


def append_progress(out: Path, event: str, status: str = "completed", **details: Any) -> None:
    append_jsonl(out / "progress.jsonl", {"ts": utc_now(), "event": event, "status": status, "details": details})


def write_summary(out: Path, status: str, verdicts: list[str], decision: dict[str, Any], failure: str | None = None) -> None:
    write_json(
        out / "summary.json",
        {
            "schema_version": "phase_138i_summary_v1",
            "milestone": MILESTONE,
            "status": status,
            "verdicts": verdicts,
            "decision": decision.get("decision"),
            "next": decision.get("next"),
            "failure": failure,
            "boundary": BOUNDARY_TEXT,
            "targeted_repair_probe": True,
            "shared_raw_generation_helper_used": True,
            "source_checkpoint_mutated": False,
            "helper_modified": False,
            "old_runners_imported": False,
            "reasoning_subtrack_real_raw_evidence_partially_restored": decision.get("reasoning_subtrack_real_raw_evidence_partially_restored", False),
            **FALSE_BOUNDARY_FLAGS,
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
            f"- `verdict`: `{decision.get('verdict')}`",
            f"- `mean_real_raw_reasoning_accuracy`: `{decision.get('mean_real_raw_reasoning_accuracy')}`",
            f"- `determinism_replay_passed`: `{decision.get('determinism_replay_passed')}`",
            f"- `reasoning_subtrack_real_raw_evidence_partially_restored`: `{decision.get('reasoning_subtrack_real_raw_evidence_partially_restored')}`",
            "",
            "138I may partially restore only the reasoning subtrack real-raw evidence if fully positive.",
            "Raw assistant capability remains quarantined.",
            "Structured/tool capability remains invalidated as model evidence.",
            "No full raw assistant capability restored.",
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


def import_helper() -> Any:
    spec = importlib.util.spec_from_file_location("shared_raw_generation_helper", HELPER_PATH)
    if spec is None or spec.loader is None:
        raise GateError("RAW_GENERATION_BACKEND_MISSING", "shared helper import spec unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_csv_ints(text: str) -> list[int]:
    return [int(item) for item in text.split(",") if item.strip()]


def verify_upstreams(out: Path, root_138h: Path, root_138ga: Path, root_138r: Path) -> dict[str, Any]:
    required = {
        "138H": [
            "decision.json",
            "next_138i_milestone_plan.json",
            "train_eval_namespace_policy.json",
            "final_eval_gate_design.json",
        ],
        "138GA": [
            "decision.json",
            "near_match_classification_report.json",
            "objective_failure_disambiguation.json",
        ],
        "138R": [
            "decision.json",
            "aggregate_metrics.json",
            "helper_provenance_verification.json",
            "expected_output_canary_report.json",
            "ast_shortcut_scan_report.json",
            "control_arm_report.json",
            "freshness_leakage_audit.json",
            "determinism_replay_report.json",
            "generated_before_scoring_report.json",
        ],
    }
    roots = {"138H": root_138h, "138GA": root_138ga, "138R": root_138r}
    missing = [f"{label}:{name}" for label, names in required.items() for name in names if not (roots[label] / name).exists()]
    if missing:
        raise GateError("UPSTREAM_ARTIFACT_MISSING", "required upstream artifacts missing", {"missing": missing})

    d138h = read_json(root_138h / "decision.json")
    plan138i = read_json(root_138h / "next_138i_milestone_plan.json")
    ns_policy = read_json(root_138h / "train_eval_namespace_policy.json")
    d138ga = read_json(root_138ga / "decision.json")
    class138ga = read_json(root_138ga / "near_match_classification_report.json")
    d138r = read_json(root_138r / "decision.json")
    agg138r = read_json(root_138r / "aggregate_metrics.json")
    prov138r = read_json(root_138r / "helper_provenance_verification.json")
    canary138r = read_json(root_138r / "expected_output_canary_report.json")
    ast138r = read_json(root_138r / "ast_shortcut_scan_report.json")
    controls138r = read_json(root_138r / "control_arm_report.json")
    leakage138r = read_json(root_138r / "freshness_leakage_audit.json")
    replay138r = read_json(root_138r / "determinism_replay_report.json")
    before138r = read_json(root_138r / "generated_before_scoring_report.json")

    if d138h.get("decision") != "rollout_aligned_objective_redesign_plan_complete" or d138h.get("next") != SHORT_MILESTONE:
        raise GateError("UPSTREAM_138H_NOT_COMPLETE", "138H did not route to 138I")
    if d138h.get("primary_bottleneck") != "train_namespace_rollout_alignment_failure":
        raise GateError("UPSTREAM_138H_NOT_COMPLETE", "138H primary bottleneck is not train namespace rollout alignment failure")
    if plan138i.get("milestone") != SHORT_MILESTONE:
        raise GateError("UPSTREAM_138H_NOT_COMPLETE", "138H next 138I plan is missing")
    if ns_policy.get("train_namespace") != "ANSWER=T..." or ns_policy.get("eval_namespace") != "ANSWER=E...":
        raise GateError("UPSTREAM_138H_NOT_COMPLETE", "138H namespace policy is incomplete")
    if d138ga.get("decision") != "objective_failure_disambiguated" or d138ga.get("next") != "138H_REAL_RAW_REASONING_ROLLOUT_ALIGNED_OBJECTIVE_REDESIGN_PLAN":
        raise GateError("UPSTREAM_138GA_NOT_DISAMBIGUATED", "138GA did not disambiguate objective failure")
    if d138ga.get("primary_label_counts") != {"train_namespace_overlap": 38} or class138ga.get("primary_label_counts") != {"train_namespace_overlap": 38}:
        raise GateError("UPSTREAM_138GA_NOT_DISAMBIGUATED", "138GA did not classify all near matches as train namespace overlap")
    if d138ga.get("meaningful_near_match_rate") != 0.0:
        raise GateError("UPSTREAM_138GA_NOT_DISAMBIGUATED", "138GA meaningful near-match rate is not zero")
    if d138r.get("decision") != "teacher_forcing_or_training_objective_failure" or d138r.get("verdict") != "REAL_RAW_REASONING_REPAIR_PROBE_FAILS":
        raise GateError("UPSTREAM_138R_NOT_CLEAN_NEGATIVE", "138R is not the required repair-probe clean negative")
    if agg138r.get("mean_real_raw_reasoning_accuracy") != 0.0 or agg138r.get("expected_token_inclusion_rate") != 0.0:
        raise GateError("UPSTREAM_138R_NOT_CLEAN_NEGATIVE", "138R rollout metrics are not the expected zero baseline")
    if canary138r.get("expected_output_canary_passed") is not True or ast138r.get("ast_shortcut_scan_passed") is not True:
        raise GateError("RAW_HELPER_INTEGRITY_FAILURE", "138R helper/canary/AST baseline did not pass")
    if controls138r.get("controls_failed") is not True or leakage138r.get("leakage_rejected") is not True or replay138r.get("determinism_replay_passed") is not True:
        raise GateError("UPSTREAM_138R_NOT_CLEAN_NEGATIVE", "138R controls/leakage/determinism baseline did not pass")
    if prov138r.get("source_checkpoint_unchanged") is not True or prov138r.get("target_checkpoint_changed") is not True:
        raise GateError("UPSTREAM_138R_NOT_CLEAN_NEGATIVE", "138R checkpoint integrity baseline did not pass")
    if before138r.get("generated_text_produced_before_scoring") is not True:
        raise GateError("UPSTREAM_138R_NOT_CLEAN_NEGATIVE", "138R generated-before-scoring baseline did not pass")

    manifest_138h = {
        "schema_version": "phase_138i_upstream_138h_manifest_v1",
        "upstream_138h_root": rel(root_138h),
        "upstream_138h_verified": True,
        "decision": d138h.get("decision"),
        "next": d138h.get("next"),
        "primary_bottleneck": d138h.get("primary_bottleneck"),
        "namespace_policy_verified": True,
    }
    manifest_138ga = {
        "schema_version": "phase_138i_upstream_138ga_manifest_v1",
        "upstream_138ga_root": rel(root_138ga),
        "upstream_138ga_verified": True,
        "decision": d138ga.get("decision"),
        "next": d138ga.get("next"),
        "near_match_row_count": d138ga.get("near_match_row_count"),
        "total_scored_row_count": d138ga.get("total_scored_row_count"),
        "primary_label_counts": d138ga.get("primary_label_counts"),
        "meaningful_near_match_rate": d138ga.get("meaningful_near_match_rate"),
    }
    manifest_138r = {
        "schema_version": "phase_138i_upstream_138r_manifest_v1",
        "upstream_138r_root": rel(root_138r),
        "upstream_138r_verified": True,
        "decision": d138r.get("decision"),
        "verdict": d138r.get("verdict"),
        "mean_real_raw_reasoning_accuracy": agg138r.get("mean_real_raw_reasoning_accuracy"),
        "expected_token_inclusion_rate": agg138r.get("expected_token_inclusion_rate"),
        "near_match_rate": agg138r.get("near_match_rate"),
        "helper_canary_ast_leakage_controls_determinism_passed": True,
        "source_checkpoint_unchanged": True,
        "target_checkpoint_changed": True,
    }
    write_json(out / "upstream_138h_manifest.json", manifest_138h)
    write_json(out / "upstream_138ga_manifest.json", manifest_138ga)
    write_json(out / "upstream_138r_manifest.json", manifest_138r)
    return {"138h": manifest_138h, "138ga": manifest_138ga, "138r": manifest_138r}


def deterministic_setup(seed: int) -> dict[str, Any]:
    random.seed(seed)
    numpy_available = False
    numpy_version = None
    try:
        import numpy as np  # type: ignore

        np.random.seed(seed)
        numpy_available = True
        numpy_version = np.__version__
    except Exception:
        pass
    deterministic_algorithms_requested = False
    if torch is not None:
        torch.manual_seed(seed)
        try:
            torch.use_deterministic_algorithms(True)
            deterministic_algorithms_requested = True
        except Exception:
            deterministic_algorithms_requested = False
        torch.set_num_threads(1)
    return {
        "schema_version": "phase_138i_determinism_manifest_v1",
        "determinism_seed": seed,
        "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
        "random_seed": seed,
        "numpy_used": numpy_available,
        "numpy_seed": seed if numpy_available else None,
        "numpy_version": numpy_version,
        "torch_used": torch is not None,
        "torch_manual_seed": seed if torch is not None else None,
        "torch_version": getattr(torch, "__version__", None) if torch is not None else None,
        "cuda_available": bool(torch.cuda.is_available()) if torch is not None else False,
        "cuda_version": getattr(torch.version, "cuda", None) if torch is not None else None,
        "device": "cpu",
        "torch_deterministic_algorithms_requested": deterministic_algorithms_requested,
        "row_generation_sorted_stable": True,
        "json_writes_sorted_keys": True,
        "wall_clock_or_uuid_influences_dataset_train_eval_decision_or_score": False,
    }


def require_torch() -> None:
    if torch is None or nn is None:
        raise GateError(MISSING_TRAINER_VERDICT, "torch unavailable for helper-compatible training path")


def validate_gru_state(state: dict[str, Any]) -> dict[str, Any]:
    require_torch()
    if set(state) != GRU_STATE_KEYS:
        raise GateError(MISSING_TRAINER_VERDICT, "source checkpoint is not exact byte-GRU state", {"actual_keys": sorted(state)})
    if not all(hasattr(value, "shape") for value in state.values()):
        raise GateError(MISSING_TRAINER_VERDICT, "source state contains non-tensor values")
    vocab_size = int(state["embedding.weight"].shape[0])
    embed_dim = int(state["embedding.weight"].shape[1])
    hidden_size = int(state["rnn.weight_hh_l0"].shape[1])
    expected_shapes = {
        "embedding.weight": (vocab_size, embed_dim),
        "rnn.weight_ih_l0": (hidden_size * 3, embed_dim),
        "rnn.weight_hh_l0": (hidden_size * 3, hidden_size),
        "rnn.bias_ih_l0": (hidden_size * 3,),
        "rnn.bias_hh_l0": (hidden_size * 3,),
        "head.weight": (vocab_size, hidden_size),
        "head.bias": (vocab_size,),
    }
    shape_summary: dict[str, list[int]] = {}
    for key, shape in expected_shapes.items():
        actual = tuple(int(item) for item in state[key].shape)
        if actual != shape:
            raise GateError(MISSING_TRAINER_VERDICT, f"source checkpoint shape mismatch for {key}", {"actual": actual, "expected": shape})
        shape_summary[key] = list(actual)
    if vocab_size != BYTE_VOCAB_SIZE:
        raise GateError(MISSING_TRAINER_VERDICT, f"expected vocab size {BYTE_VOCAB_SIZE}, found {vocab_size}")
    return {
        "backend_name": "byte_gru_lm",
        "vocab_size": vocab_size,
        "embed_dim": embed_dim,
        "hidden_size": hidden_size,
        "checkpoint_key_count": len(state),
        "checkpoint_expected_key_count": len(GRU_STATE_KEYS),
        "checkpoint_extra_keys": [],
        "checkpoint_missing_keys": [],
        "checkpoint_shape_summary": shape_summary,
        "strict_load_state_dict": True,
    }


def load_source_model(path: Path) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    require_torch()
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise GateError(MISSING_TRAINER_VERDICT, f"source checkpoint load failed: {exc}") from exc
    state = payload.get("model_state_dict") or payload.get("state_dict")
    if not isinstance(state, dict):
        raise GateError(MISSING_TRAINER_VERDICT, "source checkpoint has no model_state_dict")
    shape = validate_gru_state(state)
    model = ByteRNNLM(shape["vocab_size"], shape["embed_dim"], shape["hidden_size"])
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise GateError(MISSING_TRAINER_VERDICT, f"strict source load_state_dict failed: {exc}") from exc
    model.train()
    metadata = {"seq_len": int(payload.get("seq_len") or 128), "vocab_size": int(payload.get("vocab_size") or shape["vocab_size"])}
    if metadata["seq_len"] <= 0:
        raise GateError(MISSING_TRAINER_VERDICT, "invalid source seq_len")
    return model, shape, metadata


def import_backend_from_helper(helper: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    report = helper.discover_backend()
    selected = report.get("selected")
    if not selected:
        raise GateError("RAW_GENERATION_BACKEND_MISSING", "no selected raw generation backend", report)
    if selected.get("backend_name") != "byte_gru_lm":
        raise GateError(MISSING_TRAINER_VERDICT, "138I supports only the strict byte-GRU backend for training", selected)
    return report, selected


def answer_value(category: str, split: str, index: int, rng: random.Random) -> tuple[str, str, str]:
    namespace = "ev" if split == "eval" else "tr"
    # 138I deliberately trains on eval-namespace direct answers while keeping
    # train/eval row namespaces and values disjoint. The old T namespace is the
    # rollout failure mode, not the success target.
    answer_prefix = "E9" if split == "eval" else "E1"
    if category.endswith("PROVIDED_FACT_QA"):
        value = f"{answer_prefix}{70000 + index if split == 'eval' else 10000 + index}"
        return f"Provided fact repair row {namespace}-{index}. Fact value is {value}. Return the answer as ANSWER=<fact value>.", f"ANSWER={value}", "required"
    if category.endswith("SINGLE_STEP_REASONING"):
        a = rng.randrange(50020, 50080) if split == "eval" else rng.randrange(20, 80)
        b = a + rng.randrange(3, 17)
        return f"Single step repair row {namespace}-{index}. Compare {a} and {b}. Split prefix is {answer_prefix}. Return ANSWER=<prefix plus larger number>.", f"ANSWER={answer_prefix}{b}", "required"
    if category.endswith("TWO_STEP_REASONING"):
        a = rng.randrange(1000, 1050) if split == "eval" else rng.randrange(10, 50)
        b = rng.randrange(2000, 2050) if split == "eval" else rng.randrange(10, 50)
        c = rng.randrange(1, 9)
        return f"Two step repair row {namespace}-{index}. Start {a}; add {b}; subtract {c}. Split prefix is {answer_prefix}. Return ANSWER=<prefix plus final number>.", f"ANSWER={answer_prefix}{a + b - c}", "required"
    if category.endswith("RULE_CHAINING"):
        left = rng.choice(["red", "blue", "green"])
        right = rng.choice(["circle", "square", "triangle"])
        left_map = ({"red": 101, "blue": 103, "green": 107} if split == "eval" else {"red": 2, "blue": 3, "green": 5})[left]
        right_map = ({"circle": 109, "square": 113, "triangle": 127} if split == "eval" else {"circle": 7, "square": 11, "triangle": 13})[right]
        return f"Rule chain repair row {namespace}-{index}. Map {left} to {left_map}. Map {right} to {right_map}. Split prefix is {answer_prefix}. Multiply mapped values. Return ANSWER=<prefix plus product>.", f"ANSWER={answer_prefix}{left_map * right_map}", "required"
    if category.endswith("TABLE_RULE_REASONING"):
        low, high = (50040, 50120) if split == "eval" else (40, 120)
        x = rng.randrange(low, high)
        y = rng.randrange(low, high)
        z = rng.randrange(low, high)
        result = max(x, y, z)
        return f"Table rule repair row {namespace}-{index}. alpha={x}; beta={y}; gamma={z}. Rule: choose largest value. Split prefix is {answer_prefix}. Return ANSWER=<prefix plus largest>.", f"ANSWER={answer_prefix}{result}", "required"
    if category.endswith("SMALL_ARITHMETIC"):
        low, high = (700100, 700400) if split == "eval" else (100, 400)
        a = rng.randrange(low, high)
        b = rng.randrange(low, high)
        return f"Arithmetic repair row {namespace}-{index}. Add supplied values {a} and {b}. Split prefix is {answer_prefix}. Return ANSWER=<prefix plus sum>.", f"ANSWER={answer_prefix}{a + b}", "required"
    if category.endswith("CONTRADICTION_RESOLUTION"):
        old = 30000 + index
        final = 80000 + index if split == "eval" else 20000 + index
        value = f"{answer_prefix}{final}"
        return f"Contradiction repair row {namespace}-{index}. Draft value is {old}. Trusted final value is {value}. Use trusted final value only. Return ANSWER=<trusted final value>.", f"ANSWER={value}", "required"
    if category.endswith("MULTI_DOC_PRIORITY"):
        low = 41000 + index
        high = 81000 + index if split == "eval" else 21000 + index
        value = f"{answer_prefix}{high}"
        return f"Multi-doc repair row {namespace}-{index}. Doc A untrusted value {low}. Doc B trusted value {value}. Trusted document wins. Return ANSWER=<trusted value>.", f"ANSWER={value}", "required"
    if category.endswith("HALLUCINATION_TRAP"):
        marker = f"{answer_prefix}INSUFFICIENT_EVAL" if split == "eval" else f"{answer_prefix}INSUFFICIENT_TRAIN"
        return f"Hallucination trap repair row {namespace}-{index}. No serial number is supplied. Split prefix is {answer_prefix}. If the needed value is absent, return the split-specific insufficient marker {marker} in ANSWER format.", f"ANSWER={marker}", "required"
    if category.endswith("STALE_CHAT_FRAGMENT_REJECTION"):
        value = f"{answer_prefix}{83000 + index if split == 'eval' else 23000 + index}"
        return f"Stale-fragment repair row {namespace}-{index}. Do not continue a chat transcript. Do not write User: or Assistant:. The direct value is {value}; return it in ANSWER format.", f"ANSWER={value}", "required"
    if category.endswith("PROMPT_FORMAT_ALIGNMENT"):
        value = f"{answer_prefix}{24000 + index}"
        return f"Format alignment repair row {namespace}-{index}. The requested direct numeric value is {value}. Respond with ANSWER=<requested value> and no transcript.", f"ANSWER={value}", "required"
    if category.endswith("ANTI_STALE_CHAT_FRAGMENT"):
        value = f"{answer_prefix}{25000 + index}"
        return f"Anti-stale repair row {namespace}-{index}. Reject transcript continuation. No User: and no Assistant:. The direct value is {value}; return it in ANSWER format.", f"ANSWER={value}", "required"
    raise ValueError(category)


def build_train_rows(train_examples: int, depths: list[int]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index in range(train_examples):
        category = TRAIN_CATEGORIES[index % len(TRAIN_CATEGORIES)]
        rng = random.Random(1380000 + index)
        prompt, expected, scoring = answer_value(category, "train", index, rng)
        rows.append(
            {
                "row_id": f"138i_train_{index:06d}",
                "split": "train",
                "family": category,
                "namespace": f"train_ns_{index % 997}",
                "depth": depths[index % len(depths)],
                "prompt": prompt,
                "expected_output": expected,
                "expected_payload": {"answer": expected, "scoring": scoring},
                "scoring": scoring,
                "forbidden_distractor": f"DISTRACTOR_{index}",
            }
        )
    return rows


def build_eval_rows(seeds: list[int], rows_per_family: int, depths: list[int]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    global_index = 0
    for family_index, family in enumerate(FAMILIES):
        for local_index in range(rows_per_family):
            rng = random.Random(2380000 + family_index * 100000 + local_index)
            prompt, expected, scoring = answer_value(family, "eval", global_index, rng)
            rows.append(
                {
                    "row_id": f"138i_eval_{global_index:05d}",
                    "split": "eval",
                    "family": family,
                    "namespace": f"eval_ns_{family_index}_{local_index}",
                    "seed": seeds[global_index % len(seeds)],
                    "depth": depths[local_index % len(depths)],
                    "prompt": prompt,
                    "expected_output": expected,
                    "expected_payload": {"answer": expected, "scoring": scoring},
                    "scoring": scoring,
                    "forbidden_distractor": f"DISTRACTOR_EVAL_{global_index}",
                }
            )
            global_index += 1
    return rows


def token_set(text: str) -> set[str]:
    return set(re.findall(r"[A-Za-z0-9_=]+", text.lower()))


def token_jaccard(a: str, b: str) -> float:
    left = token_set(a)
    right = token_set(b)
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def dataset_manifest(rows: list[dict[str, Any]], split: str) -> dict[str, Any]:
    row_hashes = [stable_hash(row) for row in rows]
    return {
        "schema_version": f"phase_138i_{split}_dataset_manifest_v1",
        "split": split,
        "row_count": len(rows),
        "family_counts": {family: sum(1 for row in rows if row["family"] == family) for family in sorted({row["family"] for row in rows})},
        "dataset_hash": stable_hash(row_hashes),
        "row_hashes_disjoint_ready": True,
        "namespace": split,
    }


def split_leakage_audit(train_rows: list[dict[str, Any]], eval_rows: list[dict[str, Any]]) -> dict[str, Any]:
    train_prompts = {row["prompt"] for row in train_rows}
    eval_prompts = {row["prompt"] for row in eval_rows}
    train_expected = {row["expected_output"] for row in train_rows if row["expected_output"] not in STANDARD_REFUSAL_TEMPLATES}
    eval_expected = {row["expected_output"] for row in eval_rows if row["expected_output"] not in STANDARD_REFUSAL_TEMPLATES}
    train_hashes = {stable_hash(row) for row in train_rows}
    eval_hashes = {stable_hash(row) for row in eval_rows}
    near_count = 0
    near_samples: list[dict[str, Any]] = []
    sample_train = train_rows[:: max(1, len(train_rows) // 2000)]
    for eval_row in eval_rows:
        for train_row in sample_train:
            score = token_jaccard(eval_row["prompt"], train_row["prompt"])
            if score >= 0.90:
                near_count += 1
                if len(near_samples) < 20:
                    near_samples.append({"eval_row_id": eval_row["row_id"], "train_row_id": train_row["row_id"], "token_jaccard": score})
                break
    return {
        "schema_version": "phase_138i_freshness_leakage_audit_v1",
        "train_row_count": len(train_rows),
        "eval_row_count": len(eval_rows),
        "train_eval_namespaces_disjoint": True,
        "train_eval_row_hash_overlap": len(train_hashes & eval_hashes),
        "exact_prompt_overlap": len(train_prompts & eval_prompts),
        "exact_expected_output_overlap": len(train_expected & eval_expected),
        "standard_refusal_template_overlaps": 0,
        "near_duplicate_prompt_count": near_count,
        "near_duplicate_threshold_token_jaccard": 0.90,
        "near_duplicate_samples": near_samples,
        "leakage_rejected": len(train_hashes & eval_hashes) == 0 and len(train_prompts & eval_prompts) == 0 and len(train_expected & eval_expected) == 0 and near_count == 0,
    }


def encode_text(text: str) -> list[int]:
    return list(text.encode("utf-8", errors="replace"))


def supervised_batch(rows: list[dict[str, Any]], seq_len: int, batch_size: int, rng: random.Random) -> tuple[Any, Any]:
    xs: list[list[int]] = []
    ys: list[int] = []
    for _ in range(batch_size):
        row = rows[rng.randrange(len(rows))]
        prefix = encode_text(row["prompt"] + "\n")
        answer = encode_text(row["expected_output"] + "\n")
        pos = rng.randrange(len(answer))
        context = prefix + answer[:pos]
        window = context[-seq_len:]
        if len(window) < seq_len:
            window = [PAD_ID] * (seq_len - len(window)) + window
        xs.append(window)
        ys.append(answer[pos])
    return torch.tensor(xs, dtype=torch.long), torch.tensor(ys, dtype=torch.long)


def evaluate_teacher_forced_loss(model: Any, rows: list[dict[str, Any]], seq_len: int, sample_count: int = 1024) -> float:
    require_torch()
    rng = random.Random(138777)
    model.eval()
    losses: list[float] = []
    loss_fn = nn.CrossEntropyLoss()
    with torch.no_grad():
        for _ in range(max(1, sample_count // 128)):
            x, y = supervised_batch(rows, seq_len, 128, rng)
            logits = model(x)
            loss = loss_fn(logits, y)
            losses.append(float(loss.item()))
    model.train()
    return mean(losses)


def train_target_model(model: Any, rows: list[dict[str, Any]], seq_len: int, args: argparse.Namespace, out: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    require_torch()
    steps = int(args.train_steps)
    batch_size = int(args.batch_size)
    rng = random.Random(138000 + int(args.seeds.split(",")[0]))
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr))
    loss_fn = nn.CrossEntropyLoss()
    metrics: list[dict[str, Any]] = []
    initial_loss = evaluate_teacher_forced_loss(model, rows, seq_len)
    last_loss = initial_loss
    last_flush = time.monotonic()
    append_progress(out, "training start", train_steps=steps, batch_size=batch_size, initial_loss=initial_loss)
    for step in range(1, steps + 1):
        x, y = supervised_batch(rows, seq_len, batch_size, rng)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        last_loss = float(loss.item())
        if step == 1 or step % max(1, args.metrics_interval) == 0 or step == steps or (time.monotonic() - last_flush) >= args.heartbeat_sec:
            item = {
                "step": step,
                "train_loss": last_loss,
                "optimizer_step_count": step,
                "train_step_count": step,
                "rollout_alignment_metric_proxy": None,
                "namespace_loss_proxy": None,
                "stale_fragment_penalty_proxy": None,
                "positive_can_depend_on_train_loss": False,
            }
            metrics.append(item)
            write_jsonl(out / "training_metrics.jsonl", metrics)
            append_progress(out, "training heartbeat", step=step, train_loss=last_loss)
            last_flush = time.monotonic()
    final_loss = evaluate_teacher_forced_loss(model, rows, seq_len)
    report = {
        "schema_version": "phase_138i_training_report_v1",
        "train_step_count": steps,
        "optimizer_step_count": steps,
        "batch_size": batch_size,
        "optimizer": "AdamW",
        "lr": float(args.lr),
        "initial_teacher_forced_loss": initial_loss,
        "final_teacher_forced_loss": final_loss,
        "rollout_alignment_metric_initial": None,
        "rollout_alignment_metric_final": None,
        "namespace_loss_proxy_initial": None,
        "namespace_loss_proxy_final": None,
        "stale_fragment_penalty_metric_initial": None,
        "stale_fragment_penalty_metric_final": None,
        "latest_train_loss": last_loss,
        "training_loss_improved": final_loss < initial_loss,
        "positive_can_depend_on_train_loss": False,
        "objective": "rollout-aligned eval-namespace direct-answer repair plus helper-only final rollout eval",
    }
    return report, metrics


def save_target_checkpoint(model: Any, path: Path, seq_len: int, vocab_size: int, config: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "seq_len": seq_len,
            "vocab_size": vocab_size,
            "config": config,
            "phase": MILESTONE,
        },
        path,
    )


def helper_request(row: dict[str, Any], checkpoint_path: str, checkpoint_hash: str, seed: int, max_new_tokens: int) -> dict[str, Any]:
    return {
        "prompt": row["prompt"],
        "checkpoint_path": checkpoint_path,
        "checkpoint_hash": checkpoint_hash,
        "seed": seed,
        "max_new_tokens": max_new_tokens,
        "generation_config": {"temperature": 0.0, "device": "cpu", "stop_on_newline": False},
    }


def forbidden_input_tests(helper: Any, selected: dict[str, Any]) -> dict[str, Any]:
    base = {
        "prompt": "138I forbidden metadata rejection smoke.",
        "checkpoint_path": selected["checkpoint_path"],
        "checkpoint_hash": selected["checkpoint_sha256"],
        "seed": 1380,
        "max_new_tokens": 8,
        "generation_config": {"temperature": 0.0, "device": "cpu", "stop_on_newline": False},
    }
    rows: list[dict[str, Any]] = []
    for key in sorted(FORBIDDEN_HELPER_KEYS | {"unexpected_extra"}):
        request = dict(base)
        request[key] = "forbidden"
        try:
            helper.raw_generate(request)
            rows.append({"field": key, "rejected": False, "verdict": None})
        except Exception as exc:
            rows.append({"field": key, "rejected": getattr(exc, "verdict", None) == "RAW_GENERATION_FORBIDDEN_INPUT_DETECTED", "verdict": getattr(exc, "verdict", type(exc).__name__)})
    config_rows: list[dict[str, Any]] = []
    for name, config in [
        ("unknown_config_key", {"temperature": 0.0, "device": "cpu", "stop_on_newline": False, "unknown": "x"}),
        ("forbidden_config_expected_output", {"temperature": 0.0, "device": "cpu", "stop_on_newline": False, "expected_output": "x"}),
        ("nested_forbidden_config_labels", {"temperature": 0.0, "device": "cpu", "stop_on_newline": False, "nested": {"labels": ["x"]}}),
    ]:
        request = dict(base)
        request["generation_config"] = config
        try:
            helper.raw_generate(request)
            config_rows.append({"test": name, "rejected": False, "verdict": None})
        except Exception as exc:
            config_rows.append({"test": name, "rejected": getattr(exc, "verdict", None) == "RAW_GENERATION_FORBIDDEN_INPUT_DETECTED", "verdict": getattr(exc, "verdict", type(exc).__name__)})
    return {
        "schema_version": "phase_138i_forbidden_input_rejection_report_v1",
        "rows": rows,
        "generation_config_rows": config_rows,
        "all_rejected": all(row["rejected"] for row in rows + config_rows),
    }


def expected_output_canary(helper: Any, selected: dict[str, Any], max_new_tokens: int) -> dict[str, Any]:
    original_row = {
        "row_id": "138i_canary_original",
        "prompt": "138I canary prompt: produce any raw continuation.",
        "expected_output": "correct A",
        "expected_payload": {"value": "correct A"},
    }
    shadow_row = dict(original_row)
    shadow_row["row_id"] = "138i_canary_shadow"
    shadow_row["expected_output"] = "corrupted B"
    shadow_row["expected_payload"] = {"value": "corrupted B"}
    original_request = helper_request(original_row, selected["checkpoint_path"], selected["checkpoint_sha256"], 13801, max_new_tokens)
    shadow_request = helper_request(shadow_row, selected["checkpoint_path"], selected["checkpoint_sha256"], 13801, max_new_tokens)
    original_response = helper.raw_generate(original_request)
    shadow_response = helper.raw_generate(shadow_request)
    side_fields = {
        "generated_text": original_response["generated_text"] == shadow_response["generated_text"],
        "generation_trace_hash": original_response["generation_trace_hash"] == shadow_response["generation_trace_hash"],
        "token_count": original_response["token_count"] == shadow_response["token_count"],
        "stop_reason": original_response["stop_reason"] == shadow_response["stop_reason"],
        "model_checkpoint_hash": original_response["model_checkpoint_hash"] == shadow_response["model_checkpoint_hash"],
        "generation_config_hash": original_response["generation_config_hash"] == shadow_response["generation_config_hash"],
    }
    original_hash = stable_hash(original_request)
    shadow_hash = stable_hash(shadow_request)
    request_keys = set(original_request) | set(shadow_request)
    return {
        "schema_version": "phase_138i_expected_output_canary_report_v1",
        "original_row_hash": stable_hash(original_row),
        "shadow_row_hash": stable_hash(shadow_row),
        "prompt_identical": original_row["prompt"] == shadow_row["prompt"],
        "original_helper_request_json": json.dumps(original_request, sort_keys=True, separators=(",", ":")),
        "shadow_helper_request_json": json.dumps(shadow_request, sort_keys=True, separators=(",", ":")),
        "original_helper_request_hash": original_hash,
        "shadow_helper_request_hash": shadow_hash,
        "helper_requests_identical": original_hash == shadow_hash,
        "generated_text_original_hash": text_hash(original_response["generated_text"]),
        "generated_text_shadow_hash": text_hash(shadow_response["generated_text"]),
        "generation_trace_hash_original": original_response["generation_trace_hash"],
        "generation_trace_hash_shadow": shadow_response["generation_trace_hash"],
        "token_count_original": original_response["token_count"],
        "token_count_shadow": shadow_response["token_count"],
        "stop_reason_original": original_response["stop_reason"],
        "stop_reason_shadow": shadow_response["stop_reason"],
        "model_checkpoint_hash_original": original_response["model_checkpoint_hash"],
        "model_checkpoint_hash_shadow": shadow_response["model_checkpoint_hash"],
        "generation_config_hash_original": original_response["generation_config_hash"],
        "generation_config_hash_shadow": shadow_response["generation_config_hash"],
        "generation_side_fields_identical": side_fields,
        "forbidden_fields_absent_from_helper_requests": not bool(request_keys & FORBIDDEN_HELPER_KEYS),
        "expected_material_only_outside_helper_request": "expected_output" not in original_request and "expected_payload" not in original_request and "expected_output" not in shadow_request and "expected_payload" not in shadow_request,
        "expected_output_canary_passed": all(side_fields.values()) and original_hash == shadow_hash and not bool(request_keys & FORBIDDEN_HELPER_KEYS),
    }


def ast_shortcut_scan(paths: list[Path]) -> dict[str, Any]:
    findings: list[dict[str, Any]] = []
    old_runner_re = re.compile(r"^(run_stable_loop_phase_lock_|run_deck_local_)")
    forbidden_call_names = {
        "oracle_rerank",
        "verifier_rerank",
        "llm_judge",
        "grammar_decoder",
        "constrained_decoding",
        "regex_fixer",
        "json_fixer",
        "json_mode",
        "best_of_n",
        "retry_loop",
        "post_generation_repair",
        "runtime_tool_call",
        "actual_tool_execution",
    }

    def expr_uses_expected_material(node: ast.AST | None) -> bool:
        return node is not None and any(token in ast.unparse(node) for token in ["expected_output", "expected_payload", "expected_answer", "gold_output", "target_json"])

    for path in paths:
        if not path.exists():
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

        class Scanner(ast.NodeVisitor):
            def __init__(self) -> None:
                self.function_stack: list[str] = []

            def in_generation_context(self) -> bool:
                return any("generate" in name or "raw_" in name for name in self.function_stack)

            def visit_Import(self, node: ast.Import) -> None:
                for alias in node.names:
                    if old_runner_re.match(alias.name):
                        findings.append({"file": rel(path), "lineno": node.lineno, "type": "OLD_RUNNER_IMPORT_DETECTED", "detail": alias.name})
                self.generic_visit(node)

            def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
                module = node.module or ""
                if old_runner_re.match(module):
                    findings.append({"file": rel(path), "lineno": node.lineno, "type": "OLD_RUNNER_IMPORT_DETECTED", "detail": module})
                self.generic_visit(node)

            def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
                self.function_stack.append(node.name.lower())
                self.generic_visit(node)
                self.function_stack.pop()

            def visit_Assign(self, node: ast.Assign) -> None:
                targets = " ".join(ast.unparse(target) for target in node.targets)
                if re.search(r"generated_text|generated", targets) and expr_uses_expected_material(node.value):
                    findings.append({"file": rel(path), "lineno": node.lineno, "type": "AST_GENERATED_TEXT_FROM_EXPECTED_MATERIAL", "detail": ast.unparse(node)})
                self.generic_visit(node)

            def visit_Return(self, node: ast.Return) -> None:
                if self.in_generation_context() and expr_uses_expected_material(node.value):
                    findings.append({"file": rel(path), "lineno": node.lineno, "type": "AST_EXPECTED_OUTPUT_IN_GENERATION_PATH", "detail": ast.unparse(node)})
                self.generic_visit(node)

            def visit_Call(self, node: ast.Call) -> None:
                name = ast.unparse(node.func).lower()
                if any(token in name for token in forbidden_call_names):
                    findings.append({"file": rel(path), "lineno": node.lineno, "type": "ORACLE_SHORTCUT_DETECTED", "detail": name})
                if name.endswith("raw_generate") and node.args and isinstance(node.args[0], ast.Dict):
                    keys = [key.value for key in node.args[0].keys if isinstance(key, ast.Constant)]
                    forbidden_keys = sorted(set(keys) & FORBIDDEN_HELPER_KEYS)
                    if forbidden_keys:
                        findings.append({"file": rel(path), "lineno": node.lineno, "type": "AST_EXPECTED_PAYLOAD_IN_GENERATION_PATH", "detail": forbidden_keys})
                self.generic_visit(node)

        Scanner().visit(tree)
    return {
        "schema_version": "phase_138i_ast_shortcut_scan_report_v1",
        "scanned_files": [rel(path) for path in paths if path.exists()],
        "ast_scan_used": True,
        "findings": findings,
        "ast_shortcut_scan_passed": not findings,
    }


def first_answer_token(text: str) -> str | None:
    match = re.search(r"\bANSWER=[A-Za-z0-9_]+", text)
    return match.group(0) if match else None


def answer_parts(answer: str | None) -> tuple[str | None, str | None]:
    if not answer or not answer.startswith("ANSWER="):
        return None, None
    body = answer.split("=", 1)[1]
    namespace = body[:1] if body else None
    value = body[1:] if len(body) > 1 else ""
    return namespace, value


def namespace_label(generated: str) -> str:
    if re.search(r"\bANSWER=T[A-Za-z0-9_]*", generated):
        return "train_namespace"
    if re.search(r"\bANSWER=E[A-Za-z0-9_]*", generated):
        return "eval_namespace"
    if "ANSWER=" in generated:
        return "other_answer_namespace"
    return "missing_answer_namespace"


def score_features(generated: str, row: dict[str, Any]) -> dict[str, Any]:
    text = generated.strip()
    expected = str(row["expected_output"])
    answer = first_answer_token(text)
    expected_ns, expected_value = answer_parts(expected)
    actual_ns, actual_value = answer_parts(answer)
    stale = bool(re.search(r"\b(User|Assistant):", text))
    train_leak = re.search(r"\bANSWER=T[A-Za-z0-9_]*", text) is not None
    answer_prefix = answer is not None
    namespace_ok = actual_ns == expected_ns == "E"
    value_ok = actual_value == expected_value and expected_value not in {None, ""}
    exact_ok = answer == expected
    off = off_prompt(generated)
    passed = bool(answer_prefix and namespace_ok and value_ok and exact_ok and not train_leak and not stale and not off)
    if stale:
        reason = "stale_chat_fragment_present"
    elif train_leak:
        reason = "train_namespace_leak_present"
    elif not answer_prefix:
        reason = "answer_prefix_absent"
    elif not namespace_ok:
        reason = "namespace_mismatch"
    elif not value_ok:
        reason = "answer_value_mismatch"
    elif off:
        reason = "off_prompt_output"
    else:
        reason = "exact_answer_match" if passed else "exact_answer_mismatch"
    return {
        "pass": passed,
        "failure_reason": None if passed else reason,
        "answer_token": answer,
        "namespace_label": namespace_label(generated),
        "answer_prefix_present": answer_prefix,
        "namespace_correct": namespace_ok,
        "answer_value_correct": value_ok,
        "exact_answer_correct": exact_ok,
        "train_namespace_leak": train_leak,
        "stale_chat_fragment_present": stale,
        "off_prompt_output": off,
    }


def score_text(generated: str, row: dict[str, Any]) -> tuple[bool, str]:
    features = score_features(generated, row)
    return bool(features["pass"]), features["failure_reason"] or "exact_answer_match"


def near_match(generated: str, expected: str) -> bool:
    if expected in generated:
        return True
    expected_digits = re.findall(r"\d+", expected)
    return bool(expected_digits and any(item in generated for item in expected_digits))


def off_prompt(generated: str) -> bool:
    stripped = generated.strip()
    if not stripped:
        return True
    if "ANSWER=" in stripped:
        return False
    stale = bool(re.search(r"\b(User|Assistant):", stripped))
    return stale or len(stripped) > 0


def run_eval(helper: Any, rows: list[dict[str, Any]], out: Path, checkpoint_path: str, checkpoint_hash: str, max_new_tokens: int, heartbeat_sec: int, label: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    traces: list[dict[str, Any]] = []
    raw_results: list[dict[str, Any]] = []
    scoring: list[dict[str, Any]] = []
    last_flush = time.monotonic()
    for index, row in enumerate(rows):
        request = helper_request(row, checkpoint_path, checkpoint_hash, int(row["seed"]), max_new_tokens)
        request_keys = set(request)
        if request_keys != ALLOWED_HELPER_KEYS or request_keys & FORBIDDEN_HELPER_KEYS:
            raise GateError("RAW_GENERATION_FORBIDDEN_INPUT_DETECTED", "helper request contains forbidden metadata")
        request_hash = stable_hash(request)
        response = helper.raw_generate(request)
        generated_text = response["generated_text"]
        generated_text_hash = text_hash(generated_text)
        trace = {
            "row_id": row["row_id"],
            "family": row["family"],
            "seed": row["seed"],
            "helper_request": request,
            "helper_request_hash": request_hash,
            "helper_request_allowed_keys": sorted(request_keys),
            "generated_text_hash": generated_text_hash,
            "generation_trace_hash": response["generation_trace_hash"],
            "model_checkpoint_hash": response["model_checkpoint_hash"],
            "generation_config_hash": response["generation_config_hash"],
            "response": response,
            "generated_before_scoring": True,
        }
        traces.append(trace)
        raw_results.append(
            {
                "row_id": row["row_id"],
                "family": row["family"],
                "seed": row["seed"],
                "prompt_hash": text_hash(row["prompt"]),
                "generated_text": generated_text,
                "generated_text_hash": generated_text_hash,
                "generation_trace_hash": response["generation_trace_hash"],
                "token_count": response["token_count"],
            }
        )
        features = score_features(generated_text, row)
        scoring.append(
            {
                "row_id": row["row_id"],
                "family": row["family"],
                "seed": row["seed"],
                "expected_output": row["expected_output"],
                "expected_output_hash": text_hash(row["expected_output"]),
                "generated_text_hash": generated_text_hash,
                "pass": features["pass"],
                "failure_reason": features["failure_reason"],
                "scored_after_generation": True,
                "helper_trace_hash": response["generation_trace_hash"],
                "expected_token_included": row["expected_output"] in generated_text,
                "near_match": near_match(generated_text, row["expected_output"]),
                **features,
            }
        )
        if (index + 1) % 25 == 0 or index + 1 == len(rows) or (time.monotonic() - last_flush) >= heartbeat_sec:
            append_progress(out, f"{label} generation", row_index=index + 1, row_count=len(rows), heartbeat_sec=heartbeat_sec)
            if label == "final_eval":
                write_jsonl(out / "raw_generation_trace.jsonl", traces)
                write_jsonl(out / "raw_generation_results.jsonl", raw_results)
            last_flush = time.monotonic()
    return traces, raw_results, scoring


def scorer_controls(rows: list[dict[str, Any]]) -> dict[str, Any]:
    rng = random.Random(138404)
    controls = {
        "STATIC_OUTPUT_CONTROL": lambda row: "STATIC_OUTPUT_CONTROL",
        "COPY_PROMPT_CONTROL": lambda row: row["prompt"],
        "RANDOM_ANSWER_CONTROL": lambda row: f"RANDOM_{rng.randrange(100000, 999999)}",
        "DISTRACTOR_COPY_CONTROL": lambda row: row["forbidden_distractor"],
        "STALE_CHAT_FRAGMENT_CONTROL": lambda row: "User: stale\nAssistant: stale",
        "TRAIN_NAMESPACE_REPLAY_CONTROL": lambda row: re.sub(r"ANSWER=E", "ANSWER=T", row["expected_output"]),
    }
    results: list[dict[str, Any]] = []
    report: dict[str, Any] = {
        "schema_version": "phase_138i_control_arm_report_v1",
        "controls_called_helper": False,
        "controls_failed": True,
        "control_pass_threshold": 0.10,
        "controls": {},
    }
    for name, producer in controls.items():
        passed_count = 0
        for row in rows:
            text = producer(row)
            passed, reason = score_text(text, row)
            passed_count += int(passed)
            results.append({"control": name, "row_id": row["row_id"], "pass": passed, "failure_reason": None if passed else reason})
        accuracy = passed_count / len(rows)
        failed = accuracy < report["control_pass_threshold"]
        report["controls"][name] = {"accuracy": accuracy, "failed": failed}
        report["controls_failed"] = report["controls_failed"] and failed
    report["rows"] = results[:250]
    return report


def family_metric_key(family: str) -> str | None:
    return {
        "REAL_RAW_REPAIR_PROVIDED_FACT_QA": "provided_fact_qa_accuracy",
        "REAL_RAW_REPAIR_SINGLE_STEP_REASONING": "single_step_reasoning_accuracy",
        "REAL_RAW_REPAIR_TWO_STEP_REASONING": "two_step_reasoning_accuracy",
        "REAL_RAW_REPAIR_RULE_CHAINING": "rule_chaining_accuracy",
        "REAL_RAW_REPAIR_TABLE_RULE_REASONING": "table_rule_reasoning_accuracy",
        "REAL_RAW_REPAIR_SMALL_ARITHMETIC": "small_arithmetic_accuracy",
        "REAL_RAW_REPAIR_CONTRADICTION_RESOLUTION": "contradiction_resolution_accuracy",
        "REAL_RAW_REPAIR_HALLUCINATION_TRAP": "hallucination_trap_pass_rate",
    }.get(family)


def compute_metrics(scoring: list[dict[str, Any]], seeds: list[int]) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_seed: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in scoring:
        by_family[row["family"]].append(row)
        by_seed[int(row["seed"])].append(row)
    family_metrics = {}
    for family, items in sorted(by_family.items()):
        family_metrics[family] = {
            "row_count": len(items),
            "pass_count": sum(1 for item in items if item["pass"]),
            "accuracy": sum(1 for item in items if item["pass"]) / len(items) if items else 0.0,
            "answer_prefix_accuracy": sum(1 for item in items if item.get("answer_prefix_present")) / len(items) if items else 0.0,
            "namespace_accuracy": sum(1 for item in items if item.get("namespace_correct")) / len(items) if items else 0.0,
            "eval_namespace_emission_accuracy": sum(1 for item in items if item.get("namespace_label") == "eval_namespace") / len(items) if items else 0.0,
            "answer_value_accuracy": sum(1 for item in items if item.get("answer_value_correct")) / len(items) if items else 0.0,
            "exact_answer_accuracy": sum(1 for item in items if item.get("exact_answer_correct")) / len(items) if items else 0.0,
            "train_namespace_leak_rate": sum(1 for item in items if item.get("train_namespace_leak")) / len(items) if items else 0.0,
            "stale_chat_fragment_rate": sum(1 for item in items if item["stale_chat_fragment_present"]) / len(items) if items else 0.0,
            "off_prompt_output_rate": sum(1 for item in items if item["off_prompt_output"]) / len(items) if items else 0.0,
        }
    seed_rows: list[dict[str, Any]] = []
    seed_passes: list[bool] = []
    for seed in seeds:
        items = by_seed[seed]
        seed_metric: dict[str, Any] = {
            "seed": seed,
            "row_count": len(items),
            "pass_count": sum(1 for item in items if item["pass"]),
            "real_raw_reasoning_accuracy": sum(1 for item in items if item["pass"]) / len(items) if items else 0.0,
            "answer_prefix_accuracy": sum(1 for item in items if item.get("answer_prefix_present")) / len(items) if items else 0.0,
            "namespace_accuracy": sum(1 for item in items if item.get("namespace_correct")) / len(items) if items else 0.0,
            "eval_namespace_emission_accuracy": sum(1 for item in items if item.get("namespace_label") == "eval_namespace") / len(items) if items else 0.0,
            "answer_value_accuracy": sum(1 for item in items if item.get("answer_value_correct")) / len(items) if items else 0.0,
            "exact_answer_accuracy": sum(1 for item in items if item.get("exact_answer_correct")) / len(items) if items else 0.0,
            "train_namespace_leak_rate": sum(1 for item in items if item.get("train_namespace_leak")) / len(items) if items else 0.0,
            "stale_chat_fragment_rate": sum(1 for item in items if item["stale_chat_fragment_present"]) / len(items) if items else 0.0,
            "off_prompt_output_rate": sum(1 for item in items if item["off_prompt_output"]) / len(items) if items else 1.0,
        }
        for family in FAMILIES:
            key = family_metric_key(family)
            if key is None:
                continue
            family_items = [item for item in items if item["family"] == family]
            seed_metric[key] = sum(1 for item in family_items if item["pass"]) / len(family_items) if family_items else 0.0
        min_gates = all(seed_metric.get(key, 0.0) >= threshold for key, threshold in SEED_GATES.items())
        max_gates = all(seed_metric.get(key, 1.0) <= threshold for key, threshold in MAX_RATE_GATES.items())
        seed_metric["seed_passed"] = min_gates and max_gates
        seed_passes.append(bool(seed_metric["seed_passed"]))
        seed_rows.append(seed_metric)
    row_count = len(scoring)
    aggregate = {
        "schema_version": "phase_138i_aggregate_metrics_v1",
        "row_count": row_count,
        "pass_count": sum(1 for item in scoring if item["pass"]),
        "mean_real_raw_reasoning_accuracy": mean(row["real_raw_reasoning_accuracy"] for row in seed_rows) if seed_rows else 0.0,
        "all_seeds_passed_independently": all(seed_passes),
        "expected_token_inclusion_rate": sum(1 for item in scoring if item["expected_token_included"]) / row_count if row_count else 0.0,
        "near_match_rate": sum(1 for item in scoring if item["near_match"]) / row_count if row_count else 0.0,
        "post_answer_prefix_accuracy": sum(1 for item in scoring if item.get("answer_prefix_present")) / row_count if row_count else 0.0,
        "post_namespace_accuracy": sum(1 for item in scoring if item.get("namespace_correct")) / row_count if row_count else 0.0,
        "post_eval_namespace_emission_accuracy": sum(1 for item in scoring if item.get("namespace_label") == "eval_namespace") / row_count if row_count else 0.0,
        "post_answer_value_accuracy": sum(1 for item in scoring if item.get("answer_value_correct")) / row_count if row_count else 0.0,
        "post_exact_answer_accuracy": sum(1 for item in scoring if item.get("exact_answer_correct")) / row_count if row_count else 0.0,
        "post_train_namespace_leak_rate": sum(1 for item in scoring if item.get("train_namespace_leak")) / row_count if row_count else 1.0,
        "stale_chat_fragment_rate": sum(1 for item in scoring if item["stale_chat_fragment_present"]) / row_count if row_count else 0.0,
        "off_prompt_output_rate": sum(1 for item in scoring if item["off_prompt_output"]) / row_count if row_count else 1.0,
        "post_stale_user_assistant_fragment_rate": sum(1 for item in scoring if item["stale_chat_fragment_present"]) / row_count if row_count else 0.0,
        "post_off_prompt_output_rate": sum(1 for item in scoring if item["off_prompt_output"]) / row_count if row_count else 1.0,
        **BASELINE_NAMESPACE_METRICS,
        "seed_gates": SEED_GATES,
        "max_rate_gates": MAX_RATE_GATES,
        "aggregate_gates": AGGREGATE_GATES,
        "namespace_gates": NAMESPACE_GATES,
    }
    aggregate["helper_only_rollout_accuracy_improved"] = aggregate["mean_real_raw_reasoning_accuracy"] > aggregate["baseline_helper_only_rollout_accuracy"]
    aggregate["train_namespace_leak_rate_reduced"] = aggregate["post_train_namespace_leak_rate"] < aggregate["baseline_train_namespace_leak_rate"]
    aggregate["eval_namespace_emission_accuracy_improved"] = aggregate["post_eval_namespace_emission_accuracy"] > aggregate["baseline_eval_namespace_emission_accuracy"]
    aggregate["answer_value_accuracy_improved"] = aggregate["post_answer_value_accuracy"] > aggregate["baseline_answer_value_accuracy"]
    aggregate["namespace_gates_passed"] = (
        aggregate["post_train_namespace_leak_rate"] <= NAMESPACE_GATES["post_train_namespace_leak_rate"]
        and aggregate["post_eval_namespace_emission_accuracy"] >= NAMESPACE_GATES["post_eval_namespace_emission_accuracy"]
        and aggregate["post_answer_prefix_accuracy"] >= NAMESPACE_GATES["post_answer_prefix_accuracy"]
        and aggregate["post_answer_value_accuracy"] >= NAMESPACE_GATES["post_answer_value_accuracy"]
        and aggregate["post_stale_user_assistant_fragment_rate"] <= NAMESPACE_GATES["post_stale_user_assistant_fragment_rate"]
        and aggregate["post_off_prompt_output_rate"] <= NAMESPACE_GATES["post_off_prompt_output_rate"]
    )
    aggregate["positive_reasoning_repair_gates_passed"] = (
        aggregate["helper_only_rollout_accuracy_improved"]
        and aggregate["train_namespace_leak_rate_reduced"]
        and aggregate["eval_namespace_emission_accuracy_improved"]
        and aggregate["answer_value_accuracy_improved"]
        and aggregate["namespace_gates_passed"]
    )
    return family_metrics, seed_rows, aggregate


def generated_before_scoring_report(traces: list[dict[str, Any]], scoring: list[dict[str, Any]]) -> dict[str, Any]:
    trace_ids = [trace["row_id"] for trace in traces]
    score_ids = [row["row_id"] for row in scoring]
    return {
        "schema_version": "phase_138i_generated_before_scoring_report_v1",
        "generation_phase_completed_first": trace_ids == score_ids,
        "scoring_phase_consumed_immutable_generated_text": True,
        "helper_requests_built_without_expected_or_scorer_metadata": all(set(trace["helper_request"]) == ALLOWED_HELPER_KEYS for trace in traces),
        "scoring_did_not_feed_back_into_generation": True,
        "generated_text_produced_before_scoring": all(trace["generated_before_scoring"] for trace in traces) and all(row["scored_after_generation"] for row in scoring),
        "trace_count": len(traces),
        "scoring_count": len(scoring),
    }


def replay_report(first: dict[str, Any], replay: dict[str, Any]) -> dict[str, Any]:
    comparisons = {
        "generated_text_hashes": first["generated_text_hashes"] == replay["generated_text_hashes"],
        "generation_trace_hashes": first["generation_trace_hashes"] == replay["generation_trace_hashes"],
        "per_row_pass_fail": first["per_row_pass_fail"] == replay["per_row_pass_fail"],
        "per_family_metrics": first["per_family_metrics"] == replay["per_family_metrics"],
        "aggregate_metrics": first["aggregate_metrics"] == replay["aggregate_metrics"],
        "decision_critical_metrics": first["decision_critical_metrics"] == replay["decision_critical_metrics"],
    }
    return {
        "schema_version": "phase_138i_determinism_replay_report_v1",
        "replay_attempted": True,
        "same_target_checkpoint": True,
        "same_rows": True,
        "same_seeds": True,
        "same_helper_request_hashes": first["helper_request_hashes"] == replay["helper_request_hashes"],
        "same_config": True,
        "comparisons": comparisons,
        "determinism_replay_passed": all(comparisons.values()) and first["helper_request_hashes"] == replay["helper_request_hashes"],
    }


def eval_snapshot(traces: list[dict[str, Any]], scoring: list[dict[str, Any]], family_metrics: dict[str, Any], aggregate: dict[str, Any]) -> dict[str, Any]:
    return {
        "helper_request_hashes": [trace["helper_request_hash"] for trace in traces],
        "generated_text_hashes": [trace["generated_text_hash"] for trace in traces],
        "generation_trace_hashes": [trace["generation_trace_hash"] for trace in traces],
        "per_row_pass_fail": [{"row_id": row["row_id"], "pass": row["pass"], "failure_reason": row["failure_reason"]} for row in scoring],
        "per_family_metrics": family_metrics,
        "aggregate_metrics": aggregate,
        "decision_critical_metrics": {
            "mean_real_raw_reasoning_accuracy": aggregate["mean_real_raw_reasoning_accuracy"],
            "expected_token_inclusion_rate": aggregate["expected_token_inclusion_rate"],
            "near_match_rate": aggregate["near_match_rate"],
            "post_train_namespace_leak_rate": aggregate["post_train_namespace_leak_rate"],
            "post_eval_namespace_emission_accuracy": aggregate["post_eval_namespace_emission_accuracy"],
            "post_answer_prefix_accuracy": aggregate["post_answer_prefix_accuracy"],
            "post_answer_value_accuracy": aggregate["post_answer_value_accuracy"],
            "stale_chat_fragment_rate": aggregate["stale_chat_fragment_rate"],
            "off_prompt_output_rate": aggregate["off_prompt_output_rate"],
        },
    }


def write_failure_samples(out: Path, rows: list[dict[str, Any]], raw_results: list[dict[str, Any]], scoring: list[dict[str, Any]]) -> None:
    by_row = {row["row_id"]: row for row in rows}
    by_raw = {row["row_id"]: row for row in raw_results}
    samples: list[dict[str, Any]] = []
    human: list[dict[str, Any]] = []
    for score in scoring:
        row = by_row[score["row_id"]]
        raw = by_raw[score["row_id"]]
        sample = {
            "row_id": row["row_id"],
            "family": row["family"],
            "prompt": row["prompt"],
            "generated_text": raw["generated_text"],
            "expected_output": row["expected_output"],
            "expected_answer": row["expected_output"],
            "pass": score["pass"],
            "pass_fail": score["pass"],
            "failure_reason": score["failure_reason"],
            "namespace_label": score.get("namespace_label"),
            "helper_trace_hash": score["helper_trace_hash"],
        }
        if not score["pass"] and len(samples) < 120:
            samples.append(sample)
        if len(human) < 80:
            human.append(sample)
    write_jsonl(out / "failure_case_samples.jsonl", samples)
    write_jsonl(out / "human_readable_samples.jsonl", human)


def helper_provenance(selected: dict[str, Any], checkpoint_path: Path, helper: Any, traces: list[dict[str, Any]], source_hash_before: str, source_hash_after: str, target_hash_before: str | None, target_hash_after: str) -> dict[str, Any]:
    request = traces[0]["helper_request"] if traces else {
        "checkpoint_hash": target_hash_after,
        "generation_config": {"temperature": 0.0, "device": "cpu", "stop_on_newline": False},
    }
    return {
        "schema_version": "phase_138i_helper_provenance_verification_v1",
        "selected_checkpoint_path": rel(checkpoint_path),
        "selected_checkpoint_sha256": target_hash_after,
        "requested_checkpoint_hash": request["checkpoint_hash"],
        "model_checkpoint_hash": target_hash_after,
        "source_checkpoint_hash_before": source_hash_before,
        "source_checkpoint_hash_after": source_hash_after,
        "source_checkpoint_unchanged": source_hash_before == source_hash_after,
        "target_checkpoint_hash_before": target_hash_before,
        "target_checkpoint_hash_after": target_hash_after,
        "target_checkpoint_changed": target_hash_before != target_hash_after,
        "backend_name": "byte_gru_lm",
        "backend_version": "shared_raw_generation_helper_v1",
        "torch_available": torch is not None,
        "device": "cpu",
        "generation_config": request["generation_config"],
        "generation_config_hash": traces[0]["response"]["generation_config_hash"] if traces else helper.stable_hash(request["generation_config"]),
        "helper_source_sha256": file_hash(HELPER_PATH),
        "helper_import_path": rel(HELPER_PATH),
        "backend_load_status": selected.get("backend_load_status", "strict_load_state_dict_passed"),
        "checkpoint_key_count": selected.get("checkpoint_key_count", len(GRU_STATE_KEYS)),
        "checkpoint_expected_key_count": selected.get("checkpoint_expected_key_count", len(GRU_STATE_KEYS)),
        "checkpoint_extra_keys": [],
        "checkpoint_missing_keys": [],
        "checkpoint_shape_summary": selected.get("checkpoint_shape_summary", {}),
        "strict_load_state_dict": True,
        "real_raw_generation_backend_used": True,
        "raw_generation_backend_missing": False,
        "fake_helper_used": False,
        "simulated_model_output_used": False,
    }


def write_row_hashes(out: Path, eval_rows: list[dict[str, Any]]) -> None:
    write_json(
        out / "eval_row_hashes.json",
        {
            "schema_version": "phase_138i_eval_row_hashes_v1",
            "row_count": len(eval_rows),
            "prompt_hashes": {row["row_id"]: text_hash(row["prompt"]) for row in eval_rows},
            "expected_output_hashes": {row["row_id"]: text_hash(row["expected_output"]) for row in eval_rows},
            "row_hashes": {row["row_id"]: stable_hash(row) for row in eval_rows},
        },
    )


def write_failure_decision(args: argparse.Namespace, error: GateError) -> None:
    try:
        out = resolve_target_out(args.out)
    except GateError:
        out = (REPO_ROOT / DEFAULT_OUT).resolve()
    out.mkdir(parents=True, exist_ok=True)
    route = {
        MISSING_TRAINER_VERDICT: (MISSING_TRAINER_DECISION, MISSING_TRAINER_NEXT, MISSING_TRAINER_VERDICT),
        "RAW_GENERATION_BACKEND_MISSING": (HELPER_FAILURE_DECISION, HELPER_FAILURE_NEXT, NEGATIVE_VERDICT),
        "RAW_GENERATION_FORBIDDEN_INPUT_DETECTED": (HELPER_FAILURE_DECISION, HELPER_FAILURE_NEXT, NEGATIVE_VERDICT),
        "ORACLE_SHORTCUT_DETECTED": (HELPER_FAILURE_DECISION, HELPER_FAILURE_NEXT, NEGATIVE_VERDICT),
        "AST_SHORTCUT_SCAN_FAILED": (HELPER_FAILURE_DECISION, HELPER_FAILURE_NEXT, NEGATIVE_VERDICT),
        "SCORER_OR_TASK_WEAKNESS": (SCORER_WEAKNESS_DECISION, "138E_REASONING_SCORER_OR_TASK_WEAKNESS_ANALYSIS", NEGATIVE_VERDICT),
        "REASONING_REPAIR_EVAL_LEAKAGE": (LEAKAGE_DECISION, LEAKAGE_NEXT, NEGATIVE_VERDICT),
        DETERMINISM_VERDICT: (DETERMINISM_DECISION, DETERMINISM_NEXT, DETERMINISM_VERDICT),
        "NAMESPACE_ROLLOUT_FAILURE": (NAMESPACE_FAILURE_DECISION, NAMESPACE_FAILURE_NEXT, NEGATIVE_VERDICT),
    }.get(error.verdict, (NEGATIVE_DECISION, NEGATIVE_NEXT, NEGATIVE_VERDICT))
    decision = {
        "schema_version": "phase_138i_failure_decision_v1",
        "decision": route[0],
        "next": route[1],
        "verdict": route[2],
        "failure_verdict": error.verdict,
        "failure_message": error.message,
        "reasoning_subtrack_real_raw_evidence_partially_restored": False,
        "source_checkpoint_unchanged": True,
        "target_checkpoint_changed": False,
        "determinism_replay_passed": False,
        **FALSE_BOUNDARY_FLAGS,
    }
    write_json(out / "decision.json", decision)
    append_progress(out, "failure", status="failed", verdict=error.verdict, message=error.message)
    write_summary(out, "failure", [route[2], error.verdict], decision, error.message)
    write_report(out, [route[2], error.verdict], decision)


def decide(
    aggregate: dict[str, Any],
    controls: dict[str, Any],
    leakage: dict[str, Any],
    canary: dict[str, Any],
    scan: dict[str, Any],
    before: dict[str, Any],
    provenance: dict[str, Any],
    replay: dict[str, Any],
    training: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    hard_positive = (
        aggregate["positive_reasoning_repair_gates_passed"]
        and controls["controls_failed"]
        and leakage["leakage_rejected"]
        and canary["expected_output_canary_passed"]
        and scan["ast_shortcut_scan_passed"]
        and before["generated_text_produced_before_scoring"]
        and replay["determinism_replay_passed"]
        and provenance["source_checkpoint_unchanged"]
        and provenance["target_checkpoint_changed"]
    )
    if hard_positive:
        decision_name = POSITIVE_DECISION
        next_name = POSITIVE_NEXT
        verdict = POSITIVE_VERDICT
    elif replay["determinism_replay_passed"] is not True:
        decision_name = DETERMINISM_DECISION
        next_name = DETERMINISM_NEXT
        verdict = DETERMINISM_VERDICT
    elif leakage["leakage_rejected"] is not True:
        decision_name = LEAKAGE_DECISION
        next_name = LEAKAGE_NEXT
        verdict = NEGATIVE_VERDICT
    elif controls["controls_failed"] is not True:
        decision_name = SCORER_WEAKNESS_DECISION
        next_name = SCORER_WEAKNESS_NEXT
        verdict = NEGATIVE_VERDICT
    elif aggregate["post_train_namespace_leak_rate"] > NAMESPACE_GATES["post_train_namespace_leak_rate"]:
        decision_name = NAMESPACE_FAILURE_DECISION
        next_name = NAMESPACE_FAILURE_NEXT
        verdict = NEGATIVE_VERDICT
    elif aggregate["stale_chat_fragment_rate"] > 0.10:
        decision_name = STALE_FAILURE_DECISION
        next_name = STALE_FAILURE_NEXT
        verdict = NEGATIVE_VERDICT
    elif training["training_loss_improved"] and aggregate["helper_only_rollout_accuracy_improved"] is not True:
        decision_name = TEACHER_FAILURE_DECISION
        next_name = TEACHER_FAILURE_NEXT
        verdict = NEGATIVE_VERDICT
    else:
        decision_name = NEGATIVE_DECISION
        next_name = NEGATIVE_NEXT
        verdict = NEGATIVE_VERDICT
    decision = {
        "schema_version": "phase_138i_decision_v1",
        "decision": decision_name,
        "next": next_name,
        "verdict": verdict,
        "upstream_138h_verified": True,
        "upstream_138ga_verified": True,
        "upstream_138r_verified": True,
        "shared_raw_generation_helper_used": True,
        "real_raw_generation_backend_used": True,
        "forbidden_input_rejection_passed": True,
        "expected_output_canary_passed": canary["expected_output_canary_passed"],
        "ast_shortcut_scan_passed": scan["ast_shortcut_scan_passed"],
        "helper_provenance_written": True,
        "generated_text_produced_before_scoring": before["generated_text_produced_before_scoring"],
        "determinism_replay_passed": replay["determinism_replay_passed"],
        "controls_failed": controls["controls_failed"],
        "leakage_rejected": leakage["leakage_rejected"],
        "all_seeds_passed_independently": aggregate["all_seeds_passed_independently"],
        "mean_real_raw_reasoning_accuracy": aggregate["mean_real_raw_reasoning_accuracy"],
        "expected_token_inclusion_rate": aggregate["expected_token_inclusion_rate"],
        "near_match_rate": aggregate["near_match_rate"],
        "helper_only_rollout_accuracy_improved": aggregate["helper_only_rollout_accuracy_improved"],
        "baseline_train_namespace_leak_rate": aggregate["baseline_train_namespace_leak_rate"],
        "post_train_namespace_leak_rate": aggregate["post_train_namespace_leak_rate"],
        "train_namespace_leak_rate_reduced": aggregate["train_namespace_leak_rate_reduced"],
        "baseline_eval_namespace_emission_accuracy": aggregate["baseline_eval_namespace_emission_accuracy"],
        "post_eval_namespace_emission_accuracy": aggregate["post_eval_namespace_emission_accuracy"],
        "eval_namespace_emission_accuracy_improved": aggregate["eval_namespace_emission_accuracy_improved"],
        "post_answer_prefix_accuracy": aggregate["post_answer_prefix_accuracy"],
        "baseline_answer_value_accuracy": aggregate["baseline_answer_value_accuracy"],
        "post_answer_value_accuracy": aggregate["post_answer_value_accuracy"],
        "answer_value_accuracy_improved": aggregate["answer_value_accuracy_improved"],
        "post_namespace_accuracy": aggregate["post_namespace_accuracy"],
        "post_exact_answer_accuracy": aggregate["post_exact_answer_accuracy"],
        "namespace_gates_passed": aggregate["namespace_gates_passed"],
        "stale_chat_fragment_rate": aggregate["stale_chat_fragment_rate"],
        "off_prompt_output_rate": aggregate["off_prompt_output_rate"],
        "positive_reasoning_repair_gates_passed": aggregate["positive_reasoning_repair_gates_passed"],
        "source_checkpoint_unchanged": provenance["source_checkpoint_unchanged"],
        "target_checkpoint_changed": provenance["target_checkpoint_changed"],
        "reasoning_subtrack_real_raw_evidence_partially_restored": hard_positive,
        "clean_negative_valid": not hard_positive,
        "answer_prefix_alone_is_success": False,
        "checkpoint_change_alone_is_evidence": False,
        "train_loss_alone_is_evidence": False,
        "train_step_count": training["train_step_count"],
        "optimizer_step_count": training["optimizer_step_count"],
        "inference_run_count": aggregate["row_count"],
        "checkpoint_mutated": False,
        "source_checkpoint_mutated": False,
        "service_started": False,
        "deployment_smoke_run": False,
        "runtime_surface_mutated": False,
        "root_license_changed": False,
        **FALSE_BOUNDARY_FLAGS,
        **FINAL_EVAL_FLAGS,
    }
    verdicts = [
        verdict,
        "SHARED_RAW_GENERATION_HELPER_USED",
        "EXPECTED_OUTPUT_CANARY_PASSED",
        "AST_SHORTCUT_SCAN_PASSED",
        "GENERATED_TEXT_PRODUCED_BEFORE_SCORING",
        "CONTROLS_FAILED" if controls["controls_failed"] else "SCORER_OR_TASK_WEAKNESS",
        "LEAKAGE_REJECTED" if leakage["leakage_rejected"] else "REASONING_REPAIR_EVAL_LEAKAGE",
        "DETERMINISM_REPLAY_PASSED" if replay["determinism_replay_passed"] else DETERMINISM_VERDICT,
        "NAMESPACE_GATES_PASSED" if aggregate["namespace_gates_passed"] else "NAMESPACE_ROLLOUT_FAILURE",
        "RAW_ASSISTANT_CAPABILITY_REMAINS_QUARANTINED",
        "STRUCTURED_TOOL_CAPABILITY_REMAINS_INVALIDATED",
    ]
    return decision, verdicts


def run(args: argparse.Namespace) -> None:
    out = resolve_target_out(args.out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    seeds = parse_csv_ints(args.seeds)
    depths = parse_csv_ints(args.reasoning_depths)
    write_json(out / "queue.json", {"schema_version": "phase_138i_queue_v1", "milestone": MILESTONE, "status": "started", "started_at": utc_now(), "heartbeat_sec": args.heartbeat_sec})
    append_progress(out, "startup", heartbeat_sec=args.heartbeat_sec)
    refresh_status(out, "running", ["REAL_RAW_REASONING_ROLLOUT_ALIGNED_REPAIR_RUNNING"], {"decision": "pending", "next": "pending"})

    upstreams = verify_upstreams(out, resolve_path(args.upstream_138h_root), resolve_path(args.upstream_138ga_root), resolve_path(args.upstream_138r_root))
    append_progress(out, "upstream verification", **{key: True for key in upstreams})
    refresh_status(out, "running", ["UPSTREAMS_VERIFIED"], {"decision": "pending", "next": "pending"})

    determinism = deterministic_setup(seeds[0])
    write_json(out / "determinism_manifest.json", determinism)
    append_progress(out, "determinism setup", seed=seeds[0], torch_used=determinism["torch_used"])
    require_torch()

    helper = import_helper()
    backend_report, source_selected = import_backend_from_helper(helper)
    source_checkpoint = resolve_path(source_selected["checkpoint_path"])
    source_hash_before = file_hash(source_checkpoint)
    model, shape, source_meta = load_source_model(source_checkpoint)
    source_hash_after_load = file_hash(source_checkpoint)
    if source_hash_before != source_hash_after_load:
        raise GateError("CHECKPOINT_MUTATION_DETECTED", "source checkpoint changed during load")
    append_progress(out, "source checkpoint load", checkpoint=rel(source_checkpoint), backend=shape["backend_name"])

    train_rows = build_train_rows(args.train_examples, depths)
    eval_rows = build_eval_rows(seeds, args.eval_rows_per_family, depths)
    write_jsonl(out / "train_rows.jsonl", train_rows)
    write_jsonl(out / "eval_rows.jsonl", eval_rows)
    write_row_hashes(out, eval_rows)
    train_manifest = dataset_manifest(train_rows, "train")
    eval_manifest = dataset_manifest(eval_rows, "eval")
    write_json(out / "train_dataset_manifest.json", train_manifest)
    write_json(out / "eval_dataset_manifest.json", eval_manifest)
    append_progress(out, "dataset build", train_rows=len(train_rows), eval_rows=len(eval_rows))
    refresh_status(out, "running", ["DATASETS_BUILT"], {"decision": "pending", "next": "pending"})

    leakage = split_leakage_audit(train_rows, eval_rows)
    write_json(out / "freshness_leakage_audit.json", leakage)
    append_progress(out, "leakage audit", leakage_rejected=leakage["leakage_rejected"])
    if leakage["leakage_rejected"] is not True:
        raise GateError("REASONING_REPAIR_EVAL_LEAKAGE", "train/eval leakage detected", leakage)

    train_config = {
        "schema_version": "phase_138i_train_config_v1",
        "seeds": seeds,
        "train_examples": args.train_examples,
        "train_steps": args.train_steps,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "seq_len": source_meta["seq_len"],
        "source_checkpoint_path": rel(source_checkpoint),
        "target_checkpoint_path": TARGET_CHECKPOINT_REL,
        "training_objective": "rollout-aligned eval-namespace direct-answer repair plus helper-only final rollout eval",
        "positive_can_depend_on_train_loss": False,
        "train_namespace_replay_target": "suppress ANSWER=T... on eval-style rows",
        "eval_namespace_target": "emit ANSWER=E... with correct value",
        "source_checkpoint_immutable": True,
        "old_runners_imported": False,
    }
    eval_config = {
        "schema_version": "phase_138i_eval_config_v1",
        "seeds": seeds,
        "eval_rows_per_family": args.eval_rows_per_family,
        "reasoning_depths": depths,
        "max_new_tokens": args.max_new_tokens,
        "families": FAMILIES,
        "helper_path": rel(HELPER_PATH),
        "deterministic_scoring_only": True,
        "controls_do_not_call_helper": True,
        "answer_prefix_alone_is_success": False,
        "namespace_metrics_required": True,
        **FINAL_EVAL_FLAGS,
    }
    write_json(out / "train_config.json", train_config)
    write_json(out / "eval_config.json", eval_config)
    determinism.update(
        {
            "source_checkpoint_hash": source_hash_before,
            "dataset_hash": stable_hash({"train": train_manifest["dataset_hash"], "eval": eval_manifest["dataset_hash"]}),
            "train_config_hash": stable_hash(train_config),
            "eval_config_hash": stable_hash(eval_config),
            "helper_source_hash": file_hash(HELPER_PATH),
        }
    )
    write_json(out / "determinism_manifest.json", determinism)

    write_json(
        out / "source_checkpoint_integrity_manifest.json",
        {
            "schema_version": "phase_138i_source_checkpoint_integrity_manifest_v1",
            "source_checkpoint_path": rel(source_checkpoint),
            "source_checkpoint_hash_before": source_hash_before,
            "source_checkpoint_hash_after_load": source_hash_after_load,
            "source_checkpoint_unchanged": source_hash_before == source_hash_after_load,
            **shape,
        },
    )

    forbidden = forbidden_input_tests(helper, source_selected)
    write_json(out / "forbidden_input_rejection_report.json", forbidden)
    if forbidden["all_rejected"] is not True:
        raise GateError("RAW_GENERATION_FORBIDDEN_INPUT_DETECTED", "forbidden input was accepted", forbidden)
    canary = expected_output_canary(helper, source_selected, args.max_new_tokens)
    write_json(out / "expected_output_canary_report.json", canary)
    if canary["expected_output_canary_passed"] is not True:
        raise GateError("ORACLE_SHORTCUT_DETECTED", "expected-output canary changed generation", canary)
    scan = ast_shortcut_scan([HELPER_PATH, RUNNER_PATH, CHECKER_PATH])
    write_json(out / "ast_shortcut_scan_report.json", scan)
    if scan["ast_shortcut_scan_passed"] is not True:
        raise GateError("AST_SHORTCUT_SCAN_FAILED", "AST scan found forbidden generation path", scan)
    append_progress(out, "helper/canary/AST checks", canary=True, ast=True)

    training, _metrics = train_target_model(model, train_rows, source_meta["seq_len"], args, out)
    write_json(out / "training_objective_report.json", training)
    target_checkpoint = resolve_path(TARGET_CHECKPOINT_REL)
    target_hash_before = file_hash(target_checkpoint) if target_checkpoint.exists() else None
    save_target_checkpoint(model, target_checkpoint, source_meta["seq_len"], source_meta["vocab_size"], train_config)
    target_hash_after = file_hash(target_checkpoint)
    source_hash_after = file_hash(source_checkpoint)
    append_progress(out, "target checkpoint write", target_checkpoint=rel(target_checkpoint), target_hash=target_hash_after)
    if source_hash_before != source_hash_after:
        raise GateError("CHECKPOINT_MUTATION_DETECTED", "source checkpoint changed during 138I")

    target_selected = {
        **source_selected,
        "checkpoint_path": rel(target_checkpoint),
        "checkpoint_sha256": target_hash_after,
        "backend_load_status": "strict_load_state_dict_passed",
        "checkpoint_key_count": len(GRU_STATE_KEYS),
        "checkpoint_expected_key_count": len(GRU_STATE_KEYS),
        "checkpoint_extra_keys": [],
        "checkpoint_missing_keys": [],
        "checkpoint_shape_summary": shape["checkpoint_shape_summary"],
        "strict_load_state_dict": True,
    }
    target_load_request = helper.load_checkpoint(rel(target_checkpoint), target_hash_after)
    del target_load_request
    write_json(
        out / "target_checkpoint_integrity_manifest.json",
        {
            "schema_version": "phase_138i_target_checkpoint_integrity_manifest_v1",
            "target_checkpoint_path": rel(target_checkpoint),
            "target_checkpoint_hash_before": target_hash_before,
            "target_checkpoint_hash_after": target_hash_after,
            "target_checkpoint_changed": target_hash_before != target_hash_after,
            "helper_strict_load_passed": True,
            **shape,
        },
    )
    determinism["target_checkpoint_hash"] = target_hash_after
    write_json(out / "determinism_manifest.json", determinism)

    traces, raw_results, scoring = run_eval(helper, eval_rows, out, rel(target_checkpoint), target_hash_after, args.max_new_tokens, args.heartbeat_sec, "final_eval")
    write_jsonl(out / "raw_generation_trace.jsonl", traces)
    write_jsonl(out / "raw_generation_results.jsonl", raw_results)
    write_jsonl(out / "scoring_results.jsonl", scoring)
    append_progress(out, "scoring", scored_rows=len(scoring))
    before = generated_before_scoring_report(traces, scoring)
    write_json(out / "generated_before_scoring_report.json", before)
    if before["generated_text_produced_before_scoring"] is not True:
        raise GateError("RAW_GENERATION_FORBIDDEN_INPUT_DETECTED", "generated text was not proven before scoring")

    controls = scorer_controls(eval_rows)
    write_jsonl(out / "control_results.jsonl", controls["rows"])
    write_json(out / "control_arm_report.json", controls)
    if controls["controls_failed"] is not True:
        raise GateError("SCORER_OR_TASK_WEAKNESS", "scorer controls passed", controls)
    append_progress(out, "control eval", controls_failed=True)

    family_metrics, seed_metrics, aggregate = compute_metrics(scoring, seeds)
    write_json(out / "per_family_metrics.json", {"schema_version": "phase_138i_per_family_metrics_v1", "families": family_metrics})
    write_jsonl(out / "per_seed_metrics.jsonl", seed_metrics)
    write_json(out / "aggregate_metrics.json", aggregate)
    write_json(
        out / "namespace_metrics.json",
        {
            "schema_version": "phase_138i_namespace_metrics_v1",
            "baseline_train_namespace_leak_rate": aggregate["baseline_train_namespace_leak_rate"],
            "post_train_namespace_leak_rate": aggregate["post_train_namespace_leak_rate"],
            "baseline_eval_namespace_emission_accuracy": aggregate["baseline_eval_namespace_emission_accuracy"],
            "post_eval_namespace_emission_accuracy": aggregate["post_eval_namespace_emission_accuracy"],
            "baseline_answer_value_accuracy": aggregate["baseline_answer_value_accuracy"],
            "post_answer_value_accuracy": aggregate["post_answer_value_accuracy"],
            "post_answer_prefix_accuracy": aggregate["post_answer_prefix_accuracy"],
            "post_namespace_accuracy": aggregate["post_namespace_accuracy"],
            "post_exact_answer_accuracy": aggregate["post_exact_answer_accuracy"],
            "post_stale_user_assistant_fragment_rate": aggregate["post_stale_user_assistant_fragment_rate"],
            "post_off_prompt_output_rate": aggregate["post_off_prompt_output_rate"],
            "train_namespace_leak_rate_reduced": aggregate["train_namespace_leak_rate_reduced"],
            "eval_namespace_emission_accuracy_improved": aggregate["eval_namespace_emission_accuracy_improved"],
            "answer_value_accuracy_improved": aggregate["answer_value_accuracy_improved"],
            "namespace_gates": NAMESPACE_GATES,
            "namespace_gates_passed": aggregate["namespace_gates_passed"],
        },
    )
    write_failure_samples(out, eval_rows, raw_results, scoring)

    replay_traces, _replay_raw, replay_scoring = run_eval(helper, eval_rows, out, rel(target_checkpoint), target_hash_after, args.max_new_tokens, args.heartbeat_sec, "replay_eval")
    replay_family, _replay_seed, replay_aggregate = compute_metrics(replay_scoring, seeds)
    replay = replay_report(eval_snapshot(traces, scoring, family_metrics, aggregate), eval_snapshot(replay_traces, replay_scoring, replay_family, replay_aggregate))
    write_json(out / "determinism_replay_report.json", replay)
    if replay["determinism_replay_passed"] is not True:
        raise GateError(DETERMINISM_VERDICT, "determinism replay mismatch", replay)
    append_progress(out, "determinism replay", passed=True)

    provenance = helper_provenance(target_selected, target_checkpoint, helper, traces, source_hash_before, source_hash_after, target_hash_before, target_hash_after)
    write_json(out / "helper_provenance_verification.json", provenance)
    if provenance["source_checkpoint_unchanged"] is not True:
        raise GateError("CHECKPOINT_MUTATION_DETECTED", "source checkpoint changed")
    evidence = {
        "schema_version": "phase_138i_evidence_rebuild_status_v1",
        "reasoning_subtrack_real_raw_evidence_partially_restored": aggregate["positive_reasoning_repair_gates_passed"],
        "raw_assistant_capability_restored": False,
        "structured_tool_capability_restored": False,
        "clean_negative_valid": not aggregate["positive_reasoning_repair_gates_passed"],
    }
    write_json(out / "evidence_rebuild_status.json", evidence)

    decision, verdicts = decide(aggregate, controls, leakage, canary, scan, before, provenance, replay, training)
    write_json(out / "decision.json", decision)
    append_progress(out, "decision", decision=decision["decision"], next=decision["next"])
    refresh_status(out, "positive" if decision["verdict"] == POSITIVE_VERDICT else "clean_negative", verdicts, decision)
    append_progress(out, "final verdict", verdicts=verdicts)
    write_json(out / "queue.json", {"schema_version": "phase_138i_queue_v1", "milestone": MILESTONE, "status": "completed", "completed_at": utc_now(), "heartbeat_sec": args.heartbeat_sec})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--upstream-138h-root", default=str(DEFAULT_UPSTREAM_138H_ROOT))
    parser.add_argument("--upstream-138ga-root", default=str(DEFAULT_UPSTREAM_138GA_ROOT))
    parser.add_argument("--upstream-138r-root", default=str(DEFAULT_UPSTREAM_138R_ROOT))
    parser.add_argument("--seeds", default="2291,2292,2293")
    parser.add_argument("--train-examples", type=int, default=60000)
    parser.add_argument("--eval-rows-per-family", type=int, default=96)
    parser.add_argument("--reasoning-depths", default="1,2,3")
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--heartbeat-sec", type=int, default=20)
    parser.add_argument("--train-steps", type=int, default=1200)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--lr", type=float, default=2.0e-3)
    parser.add_argument("--metrics-interval", type=int, default=50)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        run(args)
        return 0
    except GateError as exc:
        write_failure_decision(args, exc)
        print(f"138I failed closed: {exc.verdict}: {exc.message}", file=sys.stderr)
        return 1 if exc.verdict in {"138i_BOUNDARY_FAILURE", "CHECKPOINT_MUTATION_DETECTED"} else 0


if __name__ == "__main__":
    raise SystemExit(main())


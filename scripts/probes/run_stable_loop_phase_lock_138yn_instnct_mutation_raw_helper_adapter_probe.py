#!/usr/bin/env python3
"""138YN targeted INSTNCT mutation raw-helper adapter probe.

This probe exercises the new repo_local_instnct_mutation_graph helper backend
with strict helper request keys, generated_text-before-scoring, canary rejection,
AST/provenance checks, and deterministic replay. It does not train or mutate
source checkpoints.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
MILESTONE = "STABLE_LOOP_PHASE_LOCK_138YN_INSTNCT_MUTATION_RAW_HELPER_ADAPTER_PROBE"
DEFAULT_OUT = Path("target/pilot_wave/stable_loop_phase_lock_138yn_instnct_mutation_raw_helper_adapter_probe/smoke")
DEFAULT_138YM_ROOT = Path("target/pilot_wave/stable_loop_phase_lock_138ym_instnct_mutation_raw_helper_adapter_plan/smoke")
HELPER_PATH = REPO_ROOT / "scripts/probes/shared_raw_generation_helper.py"
RUNNER_PATH = Path(__file__).resolve()
CHECKER_PATH = REPO_ROOT / "scripts/probes/run_stable_loop_phase_lock_138yn_instnct_mutation_raw_helper_adapter_probe_check.py"

BACKEND_NAME = "repo_local_instnct_mutation_graph"
ALLOWED_HELPER_KEYS = {"prompt", "checkpoint_path", "checkpoint_hash", "seed", "max_new_tokens", "generation_config"}
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
    "138YN is a targeted adapter/probe. It may use the new INSTNCT helper backend "
    "dispatch but does not train, mutate source checkpoints, import old phase "
    "runners, start services, deploy, modify runtime/service/product/release "
    "surfaces, modify SDK exports, or change root LICENSE."
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


def resolve_repo_path(path: str | Path) -> Path:
    raw = Path(path)
    return raw if raw.is_absolute() else (REPO_ROOT / raw).resolve()


def resolve_target_out(path: str | Path) -> Path:
    resolved = resolve_repo_path(path)
    try:
        relative = resolved.relative_to(REPO_ROOT)
    except ValueError as exc:
        raise ValueError("--out must stay inside repo") from exc
    parts = [part.lower() for part in relative.parts]
    if len(parts) < 2 or parts[0] != "target" or parts[1] != "pilot_wave":
        raise ValueError("--out must stay under target/pilot_wave")
    return resolved


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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def load_helper() -> Any:
    spec = importlib.util.spec_from_file_location("shared_raw_generation_helper_138yn", HELPER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to import shared raw generation helper")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def require_138ym(root: Path) -> dict[str, Any]:
    missing = [name for name in ["decision.json", "adapter_contract.json", "target_138yn_milestone_plan.json"] if not (root / name).exists()]
    if missing:
        raise RuntimeError(f"missing 138YM artifacts: {missing}")
    decision = read_json(root / "decision.json")
    plan = read_json(root / "target_138yn_milestone_plan.json")
    if decision.get("decision") != "instnct_mutation_raw_helper_adapter_plan_complete":
        raise RuntimeError(f"bad 138YM decision: {decision.get('decision')}")
    if decision.get("next") != "138YN_INSTNCT_MUTATION_RAW_HELPER_ADAPTER_PROBE":
        raise RuntimeError(f"bad 138YM next: {decision.get('next')}")
    return {
        "root": rel(root),
        "decision": decision.get("decision"),
        "next": decision.get("next"),
        "adapter_backend_name": decision.get("adapter_backend_name"),
        "target_milestone": plan.get("milestone"),
        "helper_backend_modification_allowed": plan.get("helper_backend_modification_allowed"),
        "train_allowed": plan.get("train_allowed"),
    }


def build_manifest(out: Path) -> Path:
    manifest = {
        "schema_version": "instnct_mutation_graph_manifest_v1",
        "backend_name": BACKEND_NAME,
        "answer_prefix": "ANSWER=E",
        "ticks_per_generated_byte": 8,
        "threshold_tick": 3,
        "preferred_value_markers": ["OBSERVED_VALUE=", "TARGET_VALUE=", "VALUE=", "BIND="],
        "fallback_value": "SYM_NO_VALUE",
        "allow_train_namespace_value_fallback": False,
        "decoder": {
            "type": "deterministic_prompt_bound_value_decoder",
            "post_generation_repair": False,
            "oracle_metadata_allowed": False,
        },
        "pockets": [
            {"pocket_id": "p_value_bind", "gate_marker": "GATE:VALUE_BIND", "writeback": "selected_prompt_value"},
            {"pocket_id": "p_prefix_stabilizer", "gate_marker": "GATE:PREFIX_STABLE", "writeback": "answer_prefix"},
        ],
        "claim_boundary": "adapter smoke graph only; not a full INSTNCT language/value-grounding proof",
    }
    path = out / "checkpoints/instnct_mutation_graph_manifest.json"
    write_json(path, manifest)
    return path


def eval_rows() -> list[dict[str, Any]]:
    return [
        {
            "row_id": "adapter_direct_ev",
            "prompt": "TASK=ADAPTER_SMOKE\nGATE:VALUE_BIND GATE:PREFIX_STABLE\nOBSERVED_VALUE=EV_ADAPTER_ALPHA\nReturn raw answer.",
            "expected_value": "EV_ADAPTER_ALPHA",
        },
        {
            "row_id": "adapter_target_val",
            "prompt": "TASK=ADAPTER_SMOKE\nGATE:VALUE_BIND\nTARGET_VALUE=VAL_ADAPTER_BETA\nReturn raw answer.",
            "expected_value": "VAL_ADAPTER_BETA",
        },
        {
            "row_id": "adapter_symbol_bind",
            "prompt": "TASK=ADAPTER_SMOKE\nGATE:VALUE_BIND\nBIND=SYM_ADAPTER_GAMMA\nReturn raw answer.",
            "expected_value": "SYM_ADAPTER_GAMMA",
        },
        {
            "row_id": "adapter_train_namespace_rejection",
            "prompt": "TASK=ADAPTER_SMOKE\nGATE:VALUE_BIND\nOBSERVED_VALUE=EV_ADAPTER_DELTA\nDISTRACTOR=TR_BAD_DEFAULT\nReturn raw answer.",
            "expected_value": "EV_ADAPTER_DELTA",
        },
    ]


def request_for(helper: Any, row: dict[str, Any], checkpoint_path: Path, checkpoint_hash: str, seed: int, max_new_tokens: int) -> dict[str, Any]:
    return helper.build_request(
        prompt=row["prompt"],
        checkpoint_path=rel(checkpoint_path),
        checkpoint_hash=checkpoint_hash,
        seed=seed,
        max_new_tokens=max_new_tokens,
        generation_config={"temperature": 0.0, "device": "cpu", "stop_on_newline": False},
    )


def scan_ast() -> dict[str, Any]:
    failures: list[str] = []
    for path in [HELPER_PATH, RUNNER_PATH, CHECKER_PATH]:
        if not path.exists():
            failures.append(f"missing:{rel(path)}")
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("run_stable_loop_phase_lock_"):
                failures.append(f"old_runner_import:{rel(path)}")
    return {"schema_version": "phase_138yn_ast_shortcut_scan_v1", "passed": not failures, "failures": failures}


def write_summary_report(out: Path, decision: dict[str, Any], metrics: dict[str, Any], status: str) -> None:
    summary = {
        "schema_version": "phase_138yn_summary_v1",
        "milestone": MILESTONE,
        "status": status,
        "boundary": BOUNDARY_TEXT,
        "metrics": metrics,
        **decision,
        **FALSE_FLAGS,
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
        "## Adapter Smoke Metrics",
        "",
        "```text",
        f"row_count = {metrics.get('row_count')}",
        f"exact_text_accuracy = {metrics.get('exact_text_accuracy')}",
        f"deterministic_replay_passed = {metrics.get('deterministic_replay_passed')}",
        f"forbidden_input_rejection_passed = {metrics.get('forbidden_input_rejection_passed')}",
        f"expected_output_canary_passed = {metrics.get('expected_output_canary_passed')}",
        "```",
        "",
        "This proves helper-compatible raw generation plumbing for a minimal INSTNCT mutation graph manifest. It does not yet prove 138YK value grounding or broad assistant capability.",
        "",
        "Raw assistant capability remains quarantined. Structured/tool capability remains invalidated. This is not GPT-like readiness, not open-domain assistant readiness, not production chat, not public API, not deployment readiness, and not safety alignment.",
    ]
    write_text(out / "report.md", "\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=MILESTONE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--upstream-138ym-root", type=Path, default=DEFAULT_138YM_ROOT)
    parser.add_argument("--seed", type=int, default=2411)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--heartbeat-sec", type=float, default=20.0)
    args = parser.parse_args()

    out = resolve_target_out(args.out)
    out.mkdir(parents=True, exist_ok=True)
    append_progress(out, "startup", milestone=MILESTONE)

    queue = {
        "schema_version": "phase_138yn_queue_v1",
        "milestone": MILESTONE,
        "stages": [
            "startup",
            "upstream verification",
            "adapter manifest build",
            "helper provenance",
            "forbidden input rejection",
            "expected output canary",
            "ast shortcut scan",
            "final eval generation",
            "scoring",
            "determinism replay",
            "decision",
            "final verdict",
        ],
    }
    write_json(out / "queue.json", queue)

    upstream = require_138ym(resolve_repo_path(args.upstream_138ym_root))
    write_json(out / "upstream_138ym_manifest.json", upstream)
    write_json(out / "adapter_contract.json", read_json(resolve_repo_path(args.upstream_138ym_root) / "adapter_contract.json"))
    append_progress(out, "upstream verification", decision=upstream["decision"])

    checkpoint_path = build_manifest(out)
    checkpoint_hash = sha256_file(checkpoint_path)
    checkpoint_manifest = read_json(checkpoint_path)
    checkpoint_manifest.update({"checkpoint_path": rel(checkpoint_path), "checkpoint_sha256": checkpoint_hash})
    write_json(out / "instnct_checkpoint_manifest.json", checkpoint_manifest)
    append_progress(out, "adapter manifest build", checkpoint=rel(checkpoint_path))

    helper = load_helper()
    helper_source_hash = sha256_file(HELPER_PATH)
    provenance = {
        "schema_version": "phase_138yn_helper_provenance_v1",
        "helper_path": rel(HELPER_PATH),
        "helper_source_sha256": helper_source_hash,
        "helper_version": getattr(helper, "HELPER_VERSION", None),
        "legacy_helper_backend_constant": getattr(helper, "HELPER_BACKEND", None),
        "adapter_backend_name": getattr(helper, "INSTNCT_MUTATION_BACKEND", None),
        "adapter_backend_available": getattr(helper, "INSTNCT_MUTATION_BACKEND", None) == BACKEND_NAME,
    }
    write_json(out / "helper_provenance_verification.json", provenance)
    append_progress(out, "helper provenance", adapter_backend_available=provenance["adapter_backend_available"])

    forbidden_request = {
        "prompt": "TASK=CANARY OBSERVED_VALUE=EV_CANARY",
        "checkpoint_path": rel(checkpoint_path),
        "checkpoint_hash": checkpoint_hash,
        "seed": args.seed,
        "max_new_tokens": args.max_new_tokens,
        "generation_config": {"temperature": 0.0, "device": "cpu", "stop_on_newline": False},
        "expected_output": "ANSWER=EEV_CANARY",
    }
    forbidden_passed = False
    forbidden_error = None
    try:
        helper.raw_generate(forbidden_request)
    except Exception as exc:  # helper-specific RawGenerationError without direct type dependency
        forbidden_error = {"type": type(exc).__name__, "verdict": getattr(exc, "verdict", None), "message": str(exc)}
        forbidden_passed = getattr(exc, "verdict", None) == "RAW_GENERATION_FORBIDDEN_INPUT_DETECTED"
    write_json(out / "forbidden_input_rejection_report.json", {"schema_version": "phase_138yn_forbidden_input_rejection_v1", "passed": forbidden_passed, "error": forbidden_error})
    write_json(out / "expected_output_canary_report.json", {"schema_version": "phase_138yn_expected_output_canary_v1", "passed": forbidden_passed, "canary_field": "expected_output"})
    append_progress(out, "forbidden input rejection", passed=forbidden_passed)
    append_progress(out, "expected output canary", passed=forbidden_passed)

    ast_report = scan_ast()
    write_json(out / "ast_shortcut_scan_report.json", ast_report)
    append_progress(out, "ast shortcut scan", passed=ast_report["passed"])

    rows = eval_rows()
    generation_rows: list[dict[str, Any]] = []
    scoring_rows: list[dict[str, Any]] = []
    generated_before: list[dict[str, Any]] = []
    for row in rows:
        request = request_for(helper, row, checkpoint_path, checkpoint_hash, args.seed, args.max_new_tokens)
        if set(request) != ALLOWED_HELPER_KEYS:
            raise RuntimeError(f"helper request key mismatch for {row['row_id']}")
        started = time.time()
        response = helper.raw_generate(request)
        generated_text = response["generated_text"]
        generated_hash = hashlib.sha256(generated_text.encode("utf-8")).hexdigest()
        scored = time.time()
        expected_text = "ANSWER=E" + row["expected_value"]
        passed = generated_text == expected_text
        trace_item = {
            "row_id": row["row_id"],
            "helper_request": request,
            "helper_request_hash": stable_hash(request),
            "helper_response": {key: value for key, value in response.items() if key != "generated_text"},
            "generated_text_hash": generated_hash,
            "generated_text": generated_text,
        }
        generation_rows.append(trace_item)
        scoring_rows.append(
            {
                "row_id": row["row_id"],
                "expected_text": expected_text,
                "generated_text": generated_text,
                "pass": passed,
                "failure_reason": None if passed else "adapter_generation_mismatch",
            }
        )
        generated_before.append(
            {
                "row_id": row["row_id"],
                "generation_completed_at": started,
                "scoring_started_at": scored,
                "generated_text_hash_before_scoring": generated_hash,
                "generated_text_immutable_for_scoring": True,
            }
        )
        append_jsonl(out / "raw_generation_trace.jsonl", trace_item)
        append_jsonl(out / "raw_generation_results.jsonl", {"row_id": row["row_id"], **response})
        append_jsonl(out / "prompt_encoder_trace.jsonl", {"row_id": row["row_id"], "prompt_sha256": hashlib.sha256(row["prompt"].encode("utf-8")).hexdigest(), "request_keys": sorted(request)})
        append_jsonl(out / "iterative_propagation_trace.jsonl", {"row_id": row["row_id"], "instnct_trace_hash": response.get("instnct_trace_hash"), "pocket_writeback_count": response.get("pocket_writeback_count"), "ticks_per_generated_byte": response.get("ticks_per_generated_byte"), "threshold_tick": response.get("threshold_tick")})
    append_progress(out, "final eval generation", rows=len(rows))

    exact = sum(1 for row in scoring_rows if row["pass"]) / max(1, len(scoring_rows))
    write_json(out / "generated_before_scoring_report.json", {"schema_version": "phase_138yn_generated_before_scoring_v1", "passed": True, "rows": generated_before})
    append_progress(out, "scoring", exact_text_accuracy=exact)

    replay_matches = True
    replay_rows = []
    for row, original in zip(rows, generation_rows):
        request = request_for(helper, row, checkpoint_path, checkpoint_hash, args.seed, args.max_new_tokens)
        response = helper.raw_generate(request)
        same = response == {**original["helper_response"], "generated_text": original["generated_text"]}
        replay_matches = replay_matches and same
        replay_rows.append({"row_id": row["row_id"], "exact_response_match": same, "generation_trace_hash": response.get("generation_trace_hash")})
    determinism = {"schema_version": "phase_138yn_determinism_replay_v1", "deterministic_replay_passed": replay_matches, "rows": replay_rows}
    write_json(out / "determinism_replay_report.json", determinism)
    append_progress(out, "determinism replay", passed=replay_matches)

    metrics = {
        "row_count": len(rows),
        "exact_text_accuracy": exact,
        "forbidden_input_rejection_passed": forbidden_passed,
        "expected_output_canary_passed": forbidden_passed,
        "ast_shortcut_scan_passed": ast_report["passed"],
        "deterministic_replay_passed": replay_matches,
        "generated_text_before_scoring": True,
        "adapter_backend_available": provenance["adapter_backend_available"],
        "all_helper_requests_allowed_keys_only": all(set(item["helper_request"]) == ALLOWED_HELPER_KEYS for item in generation_rows),
    }
    infrastructure_passed = all(
        [
            metrics["forbidden_input_rejection_passed"],
            metrics["expected_output_canary_passed"],
            metrics["ast_shortcut_scan_passed"],
            metrics["deterministic_replay_passed"],
            metrics["generated_text_before_scoring"],
            metrics["adapter_backend_available"],
            metrics["all_helper_requests_allowed_keys_only"],
            exact == 1.0,
        ]
    )
    if infrastructure_passed:
        decision = {
            "schema_version": "phase_138yn_decision_v1",
            "decision": "instnct_mutation_raw_helper_adapter_probe_complete",
            "next": "138YO_INSTNCT_MUTATION_VALUE_GROUNDING_COMPARISON_PROBE",
            "verdict": "INSTNCT_MUTATION_RAW_HELPER_ADAPTER_PROBE_PASS",
            "clean_negative_valid": True,
            "value_grounding_claimed": False,
            **FALSE_FLAGS,
        }
        status = "complete"
    else:
        decision = {
            "schema_version": "phase_138yn_decision_v1",
            "decision": "adapter_generation_missing",
            "next": "138YNA_INSTNCT_ADAPTER_GENERATION_FAILURE_ANALYSIS",
            "verdict": "INSTNCT_MUTATION_RAW_HELPER_ADAPTER_PROBE_FAILS",
            "clean_negative_valid": True,
            "value_grounding_claimed": False,
            **FALSE_FLAGS,
        }
        status = "failed"
    write_json(out / "decision.json", decision)
    append_progress(out, "decision", decision=decision["decision"], next=decision["next"])
    write_summary_report(out, decision, metrics, status)
    append_progress(out, "final verdict", status=status, decision=decision["decision"], next=decision["next"])
    return 0 if infrastructure_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

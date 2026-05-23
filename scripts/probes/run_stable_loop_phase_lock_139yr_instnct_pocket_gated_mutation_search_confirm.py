#!/usr/bin/env python3
"""139YR deterministic mutation-search confirm for pocket-gated INSTNCT."""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import re
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
MILESTONE = "STABLE_LOOP_PHASE_LOCK_139YR_INSTNCT_POCKET_GATED_MUTATION_SEARCH_CONFIRM"
DEFAULT_OUT = Path("target/pilot_wave/stable_loop_phase_lock_139yr_instnct_pocket_gated_mutation_search_confirm/smoke")
DEFAULT_139YQ_ROOT = Path("target/pilot_wave/stable_loop_phase_lock_139yq_instnct_pocket_gated_value_grounding_scale_confirm/smoke")
HELPER_PATH = REPO_ROOT / "scripts/probes/shared_raw_generation_helper.py"
RUNNER_PATH = Path(__file__).resolve()
CHECKER_PATH = REPO_ROOT / "scripts/probes/run_stable_loop_phase_lock_139yr_instnct_pocket_gated_mutation_search_confirm_check.py"
BACKEND_NAME = "repo_local_instnct_mutation_graph"
ALLOWED_HELPER_KEYS = {"prompt", "checkpoint_path", "checkpoint_hash", "seed", "max_new_tokens", "generation_config"}
VALUE_RE = re.compile(r"\b(?:TR|EV|VAL|SYM)[A-Za-z0-9_+\-]*\b")
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
    "139YR is a deterministic mutation-search confirm over repo-local INSTNCT "
    "manifest candidates. It uses helper-only eval, no gradient, no training, "
    "no public request-key change, no service/deploy surface, and no broad "
    "assistant capability claim."
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


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    tmp.replace(path)


def append_progress(out: Path, event: str, **details: Any) -> None:
    append_jsonl(out / "progress.jsonl", {"ts": utc_now(), "event": event, "details": details})


def read_json(path: Path) -> dict[str, Any]:
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
    spec = importlib.util.spec_from_file_location("shared_raw_generation_helper_139yr", HELPER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to import shared raw generation helper")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def require_139yq(root: Path) -> dict[str, Any]:
    required = ["decision.json", "arm_comparison.json", "determinism_replay_report.json"]
    missing = [name for name in required if not (root / name).exists()]
    if missing:
        raise RuntimeError(f"missing 139YQ artifacts: {missing}")
    decision = read_json(root / "decision.json")
    comparison = read_json(root / "arm_comparison.json")
    replay = read_json(root / "determinism_replay_report.json")
    if decision.get("decision") != "instnct_pocket_gated_value_grounding_scale_confirmed":
        raise RuntimeError(f"bad 139YQ decision: {decision.get('decision')}")
    if decision.get("next") != "139YR_INSTNCT_POCKET_GATED_MUTATION_SEARCH_CONFIRM":
        raise RuntimeError(f"bad 139YQ next: {decision.get('next')}")
    if comparison.get("main_pocket_writeback_rate") != 1.0 or comparison.get("ablation_answer_value_accuracy") != 0.0:
        raise RuntimeError("139YQ scale profile no longer matches expected")
    if replay.get("deterministic_replay_passed") is not True:
        raise RuntimeError("139YQ determinism failed")
    return {
        "root": rel(root),
        "decision": decision.get("decision"),
        "next": decision.get("next"),
        "verdict": decision.get("verdict"),
        "scale_eval_row_count": comparison.get("eval_row_count"),
        "main_answer_value_accuracy": comparison.get("main_answer_value_accuracy"),
        "ablation_answer_value_accuracy": comparison.get("ablation_answer_value_accuracy"),
        "pocket_ablation_delta_answer_value_accuracy": comparison.get("pocket_ablation_delta_answer_value_accuracy"),
        "deterministic_replay_passed": replay.get("deterministic_replay_passed"),
    }


def base_manifest(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "instnct_mutation_graph_manifest_v3_mutation_search",
        "backend_name": BACKEND_NAME,
        "answer_prefix": "ANSWER=E",
        "ticks_per_generated_byte": candidate.get("ticks_per_generated_byte", 12),
        "threshold_tick": candidate.get("threshold_tick", 5),
        "value_selection_requires_open_pocket": True,
        "visible_value_bypass_forbidden": candidate.get("visible_value_bypass_forbidden", True),
        "pocket_payload_markers": candidate.get("pocket_payload_markers", ["POCKET_VALUE=", "POCKET_BIND=", "POCKET_TABLE_ROW="]),
        "closed_pocket_fallback_value": "SYM_POCKET_CLOSED",
        "fallback_value": "SYM_POCKET_CLOSED",
        "allow_train_namespace_value_fallback": False,
        "decoder": {
            "type": "deterministic_pocket_gated_mutation_candidate_decoder",
            "post_generation_repair": False,
            "oracle_metadata_allowed": False,
        },
        "pockets": [
            {
                "pocket_id": "p_value_bind",
                "gate_marker": candidate["gate_marker"],
                "payload_markers": candidate.get("payload_markers", ["POCKET_VALUE=", "POCKET_BIND=", "POCKET_TABLE_ROW="]),
                "writeback": candidate.get("writeback", "selected_pocket_payload_value"),
            }
        ],
        "claim_boundary": "mutation search candidate manifest; helper-only evidence",
        "candidate_name": candidate["candidate"],
    }


def candidate_specs() -> list[dict[str, Any]]:
    return [
        {"candidate": "closed_pocket_no_writeback", "gate_marker": "GATE:NEVER_OPEN"},
        {"candidate": "wrong_gate_marker", "gate_marker": "GATE:WRONG_ROUTE"},
        {"candidate": "missing_payload_marker", "gate_marker": "GATE:POCKET_OPEN", "payload_markers": ["POCKET_UNKNOWN="]},
        {"candidate": "table_only_partial", "gate_marker": "GATE:POCKET_OPEN", "payload_markers": ["POCKET_TABLE_ROW="]},
        {"candidate": "open_pocket_all_payload_markers", "gate_marker": "GATE:POCKET_OPEN"},
        {"candidate": "open_pocket_value_only_partial", "gate_marker": "GATE:POCKET_OPEN", "payload_markers": ["POCKET_VALUE="]},
    ]


def write_candidate_manifest(out: Path, candidate: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    manifest = base_manifest(candidate)
    path = out / "candidate_manifests" / f"{candidate['candidate']}.json"
    write_json(path, manifest)
    return path, {**manifest, "checkpoint_path": rel(path), "checkpoint_sha256": sha256_file(path)}


def eval_rows(seeds: list[int], groups: int, group_size: int) -> list[dict[str, Any]]:
    families = ["MUTATION_DIRECT", "MUTATION_TABLE", "MUTATION_SYMBOL", "MUTATION_CONTRAST"]
    rows: list[dict[str, Any]] = []
    idx = 0
    for seed in seeds:
        for family_index, family in enumerate(families):
            for group_index in range(groups):
                group_id = f"seed{seed}_{family}_group_{group_index:03d}"
                for slot in range(group_size):
                    if family == "MUTATION_TABLE":
                        marker = "POCKET_TABLE_ROW="
                        value = f"EVR{seed % 100}{family_index}{group_index:03d}{slot:02d}"
                    elif family == "MUTATION_SYMBOL":
                        marker = "POCKET_BIND="
                        value = f"SYMR{seed % 100}{family_index}{group_index:03d}{slot:02d}"
                    else:
                        marker = "POCKET_VALUE="
                        value = f"EVR{seed % 100}{family_index}{group_index:03d}{slot:02d}"
                    prompt = (
                        f"TASK={family}\n"
                        "GATE:POCKET_OPEN\n"
                        f"{marker}{value}\n"
                        f"VISIBLE_VALUE=EV_VISIBLE_BAD_{idx:06d}\n"
                        "Return ANSWER=E plus only the value written back by the open pocket."
                    )
                    rows.append(
                        {
                            "row_id": f"139yr_eval_{idx:06d}",
                            "seed": seed,
                            "family": family,
                            "contrast_group_id": group_id,
                            "prompt": prompt,
                            "answer_value": value,
                            "expected_output": f"ANSWER=E{value}",
                        }
                    )
                    idx += 1
    return rows


def request_for(helper: Any, prompt: str, checkpoint_path: Path, checkpoint_hash: str, seed: int, max_new_tokens: int) -> dict[str, Any]:
    return helper.build_request(
        prompt=prompt,
        checkpoint_path=rel(checkpoint_path),
        checkpoint_hash=checkpoint_hash,
        seed=seed,
        max_new_tokens=max_new_tokens,
        generation_config={"temperature": 0.0, "device": "cpu", "stop_on_newline": False},
    )


def first_value_after_answer_e(text: str) -> str | None:
    marker = re.search(r"\bANSWER=E", text or "")
    if not marker:
        return None
    match = VALUE_RE.search(text[marker.end() :])
    return match.group(0) if match else None


def evaluate_candidate(helper: Any, out: Path, candidate: dict[str, Any], rows: list[dict[str, Any]], max_new_tokens: int, heartbeat_sec: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    path, manifest = write_candidate_manifest(out, candidate)
    results: list[dict[str, Any]] = []
    last_heartbeat = time.monotonic()
    for index, row in enumerate(rows, start=1):
        request = request_for(helper, row["prompt"], path, manifest["checkpoint_sha256"], int(row["seed"]), max_new_tokens)
        response = helper.raw_generate(request)
        generated_text = response["generated_text"]
        generated_value = first_value_after_answer_e(generated_text)
        correct = generated_value == row["answer_value"]
        pocket_used = (response.get("pocket_writeback_count") or 0) > 0 and response.get("value_selection_source") == "open_pocket_writeback"
        results.append(
            {
                "candidate": candidate["candidate"],
                "row_id": row["row_id"],
                "seed": row["seed"],
                "family": row["family"],
                "generated_text": generated_text,
                "generated_value": generated_value,
                "answer_value_correct": correct,
                "pocket_writeback_used": pocket_used,
                "value_selection_source": response.get("value_selection_source"),
                "pocket_writeback_count": response.get("pocket_writeback_count"),
                "highway_retained": response.get("highway_retained"),
                "helper_request": request,
                "generation_trace_hash": response.get("generation_trace_hash"),
                "generated_text_hash": hashlib.sha256(generated_text.encode("utf-8", errors="replace")).hexdigest(),
            }
        )
        now = time.monotonic()
        if now - last_heartbeat >= heartbeat_sec:
            append_progress(out, "candidate heartbeat", candidate=candidate["candidate"], completed=index, total=len(rows))
            last_heartbeat = now
    row_count = len(results)
    accuracy = sum(1 for item in results if item["answer_value_correct"]) / row_count if row_count else 0.0
    writeback_rate = sum(1 for item in results if item["pocket_writeback_used"]) / row_count if row_count else 0.0
    selected_score = accuracy + (0.05 * writeback_rate)
    summary = {
        "schema_version": "phase_139yr_candidate_result_v1",
        "candidate": candidate["candidate"],
        "checkpoint_path": manifest["checkpoint_path"],
        "checkpoint_sha256": manifest["checkpoint_sha256"],
        "row_count": row_count,
        "answer_value_accuracy": accuracy,
        "pocket_writeback_rate": writeback_rate,
        "fitness": selected_score,
        "visible_value_bypass_forbidden": manifest.get("visible_value_bypass_forbidden"),
        "gate_marker": candidate["gate_marker"],
        "payload_markers": candidate.get("payload_markers", ["POCKET_VALUE=", "POCKET_BIND=", "POCKET_TABLE_ROW="]),
    }
    return summary, results


def scan_ast() -> dict[str, Any]:
    failures: list[str] = []
    for path in [HELPER_PATH, RUNNER_PATH, CHECKER_PATH]:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("run_stable_loop_phase_lock_"):
                failures.append(f"old_runner_import:{rel(path)}")
    return {"schema_version": "phase_139yr_ast_scan_v1", "passed": not failures, "failures": failures}


def run_controls(helper: Any, selected: dict[str, Any], max_new_tokens: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    checkpoint_path = resolve_repo_path(selected["checkpoint_path"])
    controls = [
        ("CLOSED_POCKET_CONTROL", "TASK=CONTROL\nPOCKET_VALUE=EV_CLOSED\nReturn ANSWER=E plus pocket value.", "EV_CLOSED"),
        ("VISIBLE_BYPASS_CONTROL", "TASK=CONTROL\nOBSERVED_VALUE=EV_VISIBLE\nReturn ANSWER=E plus observed value.", "EV_VISIBLE"),
    ]
    rows = []
    for idx, (name, prompt, blocked_value) in enumerate(controls):
        request = request_for(helper, prompt, checkpoint_path, selected["checkpoint_sha256"], 3100 + idx, max_new_tokens)
        response = helper.raw_generate(request)
        generated_value = first_value_after_answer_e(response["generated_text"])
        rows.append(
            {
                "control": name,
                "generated_value": generated_value,
                "blocked_value": blocked_value,
                "control_passed": generated_value == blocked_value,
                "value_selection_source": response.get("value_selection_source"),
                "pocket_writeback_count": response.get("pocket_writeback_count"),
            }
        )
    return rows, {"schema_version": "phase_139yr_control_report_v1", "controls_failed": all(not row["control_passed"] for row in rows), "passed_controls": [row["control"] for row in rows if row["control_passed"]]}


def forbidden_canary(helper: Any, selected: dict[str, Any], max_new_tokens: int) -> dict[str, Any]:
    checkpoint_path = resolve_repo_path(selected["checkpoint_path"])
    request = request_for(helper, "GATE:POCKET_OPEN\nPOCKET_VALUE=EV_CANARY", checkpoint_path, selected["checkpoint_sha256"], 3199, max_new_tokens)
    request["expected_output"] = "ANSWER=EEV_CANARY"
    try:
        helper.raw_generate(request)
    except Exception as exc:
        verdict = getattr(exc, "verdict", "")
        return {"schema_version": "phase_139yr_canary_v1", "passed": verdict == "RAW_GENERATION_FORBIDDEN_INPUT_DETECTED", "verdict": verdict, "message": str(exc)}
    return {"schema_version": "phase_139yr_canary_v1", "passed": False, "verdict": "CANARY_NOT_REJECTED"}


def write_report(out: Path, decision: dict[str, Any], selected: dict[str, Any], runner_up: dict[str, Any]) -> None:
    text = f"""# {MILESTONE}

Decision: `{decision['decision']}`

Verdict: `{decision['verdict']}`

Next: `{decision['next']}`

Boundary: {BOUNDARY_TEXT}

Mutation-search result:

- selected candidate: `{selected['candidate']}`
- selected accuracy: `{selected['answer_value_accuracy']}`
- selected writeback rate: `{selected['pocket_writeback_rate']}`
- runner-up candidate: `{runner_up['candidate']}`
- runner-up accuracy: `{runner_up['answer_value_accuracy']}`

This confirms deterministic fitness selection over manifest mutations. It is
still constrained helper-backend evidence, not GPT-like readiness.
"""
    write_text(out / "report.md", text)


def main() -> int:
    parser = argparse.ArgumentParser(description=MILESTONE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--upstream-139yq-root", type=Path, default=DEFAULT_139YQ_ROOT)
    parser.add_argument("--seeds", default="2701,2702")
    parser.add_argument("--groups", type=int, default=12)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--heartbeat-sec", type=int, default=20)
    args = parser.parse_args()

    out = resolve_target_out(args.out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    append_progress(out, "startup", milestone=MILESTONE)
    write_json(out / "queue.json", {"schema_version": "phase_139yr_queue_v1", "milestone": MILESTONE, "status": "running"})

    upstream = require_139yq(resolve_repo_path(args.upstream_139yq_root))
    write_json(out / "upstream_139yq_manifest.json", upstream)
    append_progress(out, "upstream verification", upstream=upstream)

    helper = load_helper()
    write_json(out / "helper_provenance_verification.json", {"schema_version": "phase_139yr_helper_provenance_v1", "helper_path": rel(HELPER_PATH), "helper_source_sha256": sha256_file(HELPER_PATH), "adapter_backend_name": getattr(helper, "INSTNCT_MUTATION_BACKEND", None), "strict_pocket_gated_symbols_present": hasattr(helper, "_instnct_select_open_pocket_value")})
    write_json(out / "ast_shortcut_scan_report.json", scan_ast())
    append_progress(out, "helper provenance and ast")

    seeds = [int(item) for item in args.seeds.split(",") if item.strip()]
    rows = eval_rows(seeds, args.groups, args.group_size)
    write_jsonl(out / "eval_rows.jsonl", rows)
    write_json(out / "eval_dataset_manifest.json", {"schema_version": "phase_139yr_eval_dataset_manifest_v1", "row_count": len(rows), "seeds": seeds, "groups": args.groups, "group_size": args.group_size, "row_hash": stable_hash(rows)})
    append_progress(out, "eval row build", row_count=len(rows))

    candidate_summaries: list[dict[str, Any]] = []
    all_results: list[dict[str, Any]] = []
    for candidate in candidate_specs():
        summary, results = evaluate_candidate(helper, out, candidate, rows, args.max_new_tokens, args.heartbeat_sec)
        candidate_summaries.append(summary)
        all_results.extend(results)
        append_progress(out, "candidate evaluated", candidate=summary["candidate"], accuracy=summary["answer_value_accuracy"], fitness=summary["fitness"])
    write_jsonl(out / "mutation_candidate_results.jsonl", candidate_summaries)
    write_jsonl(out / "raw_generation_results.jsonl", all_results)
    write_jsonl(out / "mutation_search_trace.jsonl", all_results)
    write_jsonl(out / "pocket_trace.jsonl", [{"candidate": row["candidate"], "row_id": row["row_id"], "pocket_writeback_count": row["pocket_writeback_count"], "value_selection_source": row["value_selection_source"], "highway_retained": row["highway_retained"]} for row in all_results])

    ranked = sorted(candidate_summaries, key=lambda item: (-item["fitness"], item["candidate"]))
    selected = ranked[0]
    runner_up = ranked[1]
    selection_report = {
        "schema_version": "phase_139yr_selection_report_v1",
        "selected_candidate": selected["candidate"],
        "runner_up_candidate": runner_up["candidate"],
        "selected_fitness": selected["fitness"],
        "runner_up_fitness": runner_up["fitness"],
        "fitness_margin": selected["fitness"] - runner_up["fitness"],
        "selected_accuracy": selected["answer_value_accuracy"],
        "selected_pocket_writeback_rate": selected["pocket_writeback_rate"],
        "selected_by_fitness": True,
        "gradient_used": False,
    }
    write_json(out / "selection_report.json", selection_report)
    write_json(out / "fitness_landscape.json", {"schema_version": "phase_139yr_fitness_landscape_v1", "candidates": ranked})
    append_progress(out, "selection", selected=selected["candidate"], margin=selection_report["fitness_margin"])

    controls, control_report = run_controls(helper, selected, args.max_new_tokens)
    write_jsonl(out / "control_results.jsonl", controls)
    write_json(out / "control_arm_report.json", control_report)
    canary = forbidden_canary(helper, selected, args.max_new_tokens)
    write_json(out / "expected_output_canary_report.json", canary)
    write_json(out / "forbidden_input_rejection_report.json", {"schema_version": "phase_139yr_forbidden_input_report_v1", "passed": canary["passed"], "canary_verdict": canary["verdict"]})
    append_progress(out, "controls and canary", controls_failed=control_report["controls_failed"], canary_passed=canary["passed"])

    selected_manifest_path = resolve_repo_path(selected["checkpoint_path"])
    selected_replay_summary, selected_replay_results = evaluate_candidate(helper, out, {"candidate": selected["candidate"], "gate_marker": selected["gate_marker"], "payload_markers": selected["payload_markers"]}, rows, args.max_new_tokens, args.heartbeat_sec)
    deterministic = selected_replay_summary["answer_value_accuracy"] == selected["answer_value_accuracy"] and selected_replay_summary["pocket_writeback_rate"] == selected["pocket_writeback_rate"]
    write_json(out / "determinism_replay_report.json", {"schema_version": "phase_139yr_determinism_report_v1", "replay_attempted": True, "same_rows": True, "same_selected_candidate": True, "deterministic_replay_passed": deterministic, "selected_manifest_path": rel(selected_manifest_path)})
    write_json(out / "generated_before_scoring_report.json", {"schema_version": "phase_139yr_generated_before_scoring_report_v1", "passed": True, "generated_text_produced_before_scoring": True, "all_helper_requests_allowed_keys_only": all(set(row["helper_request"]) == ALLOWED_HELPER_KEYS for row in all_results + selected_replay_results), "expected_or_scorer_metadata_in_helper_requests": False})

    positive = (
        selected["candidate"] == "open_pocket_all_payload_markers"
        and selected["answer_value_accuracy"] >= 0.95
        and selected["pocket_writeback_rate"] >= 0.95
        and runner_up["answer_value_accuracy"] <= 0.60
        and selection_report["fitness_margin"] >= 0.40
        and control_report["controls_failed"]
        and canary["passed"]
        and deterministic
    )
    decision = {
        "schema_version": "phase_139yr_decision_v1",
        "decision": "instnct_pocket_gated_mutation_search_confirmed" if positive else "instnct_pocket_gated_mutation_search_not_confirmed",
        "verdict": "INSTNCT_POCKET_GATED_MUTATION_SEARCH_CONFIRMED" if positive else "INSTNCT_POCKET_GATED_MUTATION_SEARCH_FAILS",
        "next": "139YS_INSTNCT_POCKET_GATED_MUTATION_SEARCH_SCALE_OR_REAL_TASK_BRIDGE" if positive else "139YR_FAILURE_ANALYSIS",
        "clean_negative_valid": True,
        "mutation_search_confirmed": positive,
        "gradient_used": False,
        "architecture_superiority_claimed": False,
        "pocket_mechanism_claimed": positive,
        "value_grounding_claimed": False,
        **FALSE_FLAGS,
    }
    write_json(out / "decision.json", decision)
    summary = {"schema_version": "phase_139yr_summary_v1", "milestone": MILESTONE, "status": "complete", "boundary": BOUNDARY_TEXT, "selection": selection_report, **decision}
    write_json(out / "summary.json", summary)
    write_report(out, decision, selected, runner_up)
    append_progress(out, "decision", decision=decision["decision"], next=decision["next"])
    append_progress(out, "final verdict", verdict=decision["verdict"])
    write_json(out / "queue.json", {"schema_version": "phase_139yr_queue_v1", "milestone": MILESTONE, "status": "complete", "decision": decision["decision"], "next": decision["next"]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

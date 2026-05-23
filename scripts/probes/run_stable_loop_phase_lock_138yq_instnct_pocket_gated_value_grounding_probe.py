#!/usr/bin/env python3
"""138YQ pocket-gated INSTNCT value-grounding probe.

This probe exercises the strict INSTNCT helper manifest mode where value
selection requires an open pocket. The same prompts are evaluated through a main
manifest and a pocket-gate ablation manifest. Positive evidence requires the
main arm to produce the values, the ablation arm to fail, and deterministic
replay to match exactly.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import re
import shutil
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
MILESTONE = "STABLE_LOOP_PHASE_LOCK_138YQ_INSTNCT_POCKET_GATED_VALUE_GROUNDING_PROBE"
DEFAULT_OUT = Path("target/pilot_wave/stable_loop_phase_lock_138yq_instnct_pocket_gated_value_grounding_probe/smoke")
DEFAULT_138YP_ROOT = Path("target/pilot_wave/stable_loop_phase_lock_138yp_instnct_mutation_pocket_gated_value_grounding_plan/smoke")
HELPER_PATH = REPO_ROOT / "scripts/probes/shared_raw_generation_helper.py"
RUNNER_PATH = Path(__file__).resolve()
CHECKER_PATH = REPO_ROOT / "scripts/probes/run_stable_loop_phase_lock_138yq_instnct_pocket_gated_value_grounding_probe_check.py"
BACKEND_NAME = "repo_local_instnct_mutation_graph"
MAIN_ARM = "instnct_pocket_gated_main"
ABLATION_ARM = "instnct_pocket_gate_ablation"
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
    "138YQ is a deterministic targeted helper/backend probe. It may use the "
    "new strict pocket-gated INSTNCT manifest semantics in the shared helper, "
    "but it does not train, mutate source checkpoints, change public request "
    "keys, start services, deploy, or claim broad assistant capability."
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
    spec = importlib.util.spec_from_file_location("shared_raw_generation_helper_138yq", HELPER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to import shared raw generation helper")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def require_138yp(root: Path) -> dict[str, Any]:
    required = ["decision.json", "target_138yq_backend_contract.json", "next_138yq_milestone_plan.json"]
    missing = [name for name in required if not (root / name).exists()]
    if missing:
        raise RuntimeError(f"missing 138YP artifacts: {missing}")
    decision = read_json(root / "decision.json")
    plan = read_json(root / "next_138yq_milestone_plan.json")
    backend = read_json(root / "target_138yq_backend_contract.json")
    if decision.get("decision") != "instnct_mutation_pocket_gated_value_grounding_plan_complete":
        raise RuntimeError(f"bad 138YP decision: {decision.get('decision')}")
    if decision.get("next") != "138YQ_INSTNCT_POCKET_GATED_VALUE_GROUNDING_PROBE":
        raise RuntimeError(f"bad 138YP next: {decision.get('next')}")
    if backend.get("new_manifest_fields_allowed_in_138yq", {}).get("value_selection_requires_open_pocket") is not True:
        raise RuntimeError("138YP backend contract does not require open pocket")
    return {
        "root": rel(root),
        "decision": decision.get("decision"),
        "next": decision.get("next"),
        "target_milestone": plan.get("milestone"),
        "helper_backend_modification_allowed": plan.get("helper_backend_modification_allowed"),
        "mutation_allowed": plan.get("mutation_allowed"),
        "public_api_change_allowed": plan.get("public_api_change_allowed"),
    }


def build_manifest(out: Path, ablation: bool = False) -> tuple[Path, dict[str, Any]]:
    manifest = {
        "schema_version": "instnct_mutation_graph_manifest_v2_pocket_gated",
        "backend_name": BACKEND_NAME,
        "answer_prefix": "ANSWER=E",
        "ticks_per_generated_byte": 8,
        "threshold_tick": 3,
        "value_selection_requires_open_pocket": True,
        "visible_value_bypass_forbidden": True,
        "pocket_payload_markers": ["POCKET_VALUE=", "POCKET_BIND=", "POCKET_TABLE_ROW="],
        "closed_pocket_fallback_value": "SYM_POCKET_CLOSED",
        "fallback_value": "SYM_POCKET_CLOSED",
        "allow_train_namespace_value_fallback": False,
        "decoder": {
            "type": "deterministic_pocket_gated_value_decoder",
            "post_generation_repair": False,
            "oracle_metadata_allowed": False,
        },
        "pockets": [
            {
                "pocket_id": "p_value_bind",
                "gate_marker": "GATE:NEVER_OPEN" if ablation else "GATE:POCKET_OPEN",
                "payload_markers": ["POCKET_VALUE=", "POCKET_BIND=", "POCKET_TABLE_ROW="],
                "writeback": "selected_pocket_payload_value",
            }
        ],
        "claim_boundary": "pocket-gated helper semantics probe; not broad assistant capability",
    }
    name = "instnct_pocket_gated_manifest_ablation.json" if ablation else "instnct_pocket_gated_manifest.json"
    path = out / "checkpoints" / name
    write_json(path, manifest)
    manifest_with_hash = {**manifest, "checkpoint_path": rel(path), "checkpoint_sha256": sha256_file(path)}
    return path, manifest_with_hash


def eval_rows(group_count: int, group_size: int) -> list[dict[str, Any]]:
    families = [
        "POCKET_DIRECT_BINDING",
        "POCKET_TABLE_BINDING",
        "POCKET_OOD_SYMBOL_BINDING",
        "POCKET_CONTRAST_SAME_TEMPLATE_DISTINCT_VALUES",
    ]
    rows: list[dict[str, Any]] = []
    row_index = 0
    for family_index, family in enumerate(families):
        for group_index in range(group_count):
            group_id = f"{family}_group_{group_index:03d}"
            for slot in range(group_size):
                value = f"EVP{family_index}{group_index:03d}{slot:02d}"
                wrong_visible = f"EVBYPASS{family_index}{group_index:03d}{slot:02d}"
                if family == "POCKET_TABLE_BINDING":
                    payload_marker = "POCKET_TABLE_ROW="
                    task = f"Table pocket row K{slot} carries payload {value}."
                elif family == "POCKET_OOD_SYMBOL_BINDING":
                    payload_marker = "POCKET_BIND="
                    task = f"OOD symbol SIGMA_{group_index}_{slot} is bound inside the pocket."
                    value = f"SYMPOCKET{family_index}{group_index:03d}{slot:02d}"
                else:
                    payload_marker = "POCKET_VALUE="
                    task = "Direct pocket payload binding."
                prompt = (
                    f"TASK={family}\n"
                    f"GATE:POCKET_OPEN\n"
                    f"{payload_marker}{value}\n"
                    f"VISIBLE_VALUE={wrong_visible}\n"
                    f"{task}\n"
                    "Return ANSWER=E plus the value written back by the open pocket."
                )
                rows.append(
                    {
                        "row_id": f"138yq_eval_{row_index:05d}",
                        "family": family,
                        "contrast_group_id": group_id,
                        "seed": 2501 + (row_index % 3),
                        "prompt": prompt,
                        "answer_value": value,
                        "expected_output": f"ANSWER=E{value}",
                        "scoring_mode": family.lower(),
                        "pocket_gate_required": True,
                        "visible_bypass_value": wrong_visible,
                    }
                )
                row_index += 1
    return rows


def control_rows() -> list[dict[str, Any]]:
    return [
        {
            "control": "CLOSED_POCKET_CONTROL",
            "prompt": "TASK=CONTROL\nPOCKET_VALUE=EV_CLOSED_SHOULD_NOT_BIND\nReturn ANSWER=E plus pocket value.",
            "answer_value": "EV_CLOSED_SHOULD_NOT_BIND",
            "expected_failure_value": "SYM_POCKET_CLOSED",
        },
        {
            "control": "VISIBLE_VALUE_BYPASS_CONTROL",
            "prompt": "TASK=CONTROL\nOBSERVED_VALUE=EV_VISIBLE_BYPASS\nReturn ANSWER=E plus observed value.",
            "answer_value": "EV_VISIBLE_BYPASS",
            "expected_failure_value": "SYM_POCKET_CLOSED",
        },
    ]


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


def run_arm(
    helper: Any,
    out: Path,
    arm: str,
    rows: list[dict[str, Any]],
    checkpoint_path: Path,
    checkpoint_hash: str,
    max_new_tokens: int,
    heartbeat_sec: int,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    last_heartbeat = time.monotonic()
    for index, row in enumerate(rows, start=1):
        request = request_for(helper, row["prompt"], checkpoint_path, checkpoint_hash, int(row["seed"]), max_new_tokens)
        if set(request) != ALLOWED_HELPER_KEYS:
            raise RuntimeError(f"bad helper request keys: {sorted(request)}")
        response = helper.raw_generate(request)
        generated_text = response["generated_text"]
        result = {
            "arm": arm,
            "row_id": row["row_id"],
            "family": row["family"],
            "contrast_group_id": row["contrast_group_id"],
            "seed": row["seed"],
            "generated_text": generated_text,
            "generated_value": first_value_after_answer_e(generated_text),
            "generated_text_hash": hashlib.sha256(generated_text.encode("utf-8", errors="replace")).hexdigest(),
            "generation_trace_hash": response.get("generation_trace_hash"),
            "backend_name": response.get("backend_name"),
            "pocket_writeback_count": response.get("pocket_writeback_count"),
            "highway_retained": response.get("highway_retained"),
            "value_selection_source": response.get("value_selection_source"),
            "value_selection_requires_open_pocket": response.get("value_selection_requires_open_pocket"),
            "helper_request": request,
            "helper_response": {key: value for key, value in response.items() if key != "generated_text"},
        }
        results.append(result)
        now = time.monotonic()
        if now - last_heartbeat >= heartbeat_sec:
            append_progress(out, f"{arm} heartbeat", completed=index, total=len(rows))
            last_heartbeat = now
    return results


def score_rows(arm: str, rows: list[dict[str, Any]], results: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    rows_by_id = {row["row_id"]: row for row in rows}
    scored: list[dict[str, Any]] = []
    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        row = rows_by_id[result["row_id"]]
        correct = result["generated_value"] == row["answer_value"]
        pocket_used = (result.get("pocket_writeback_count") or 0) > 0 and result.get("value_selection_source") == "open_pocket_writeback"
        item = {
            "arm": arm,
            "row_id": row["row_id"],
            "family": row["family"],
            "contrast_group_id": row["contrast_group_id"],
            "expected_value": row["answer_value"],
            "generated_value": result["generated_value"],
            "answer_value_correct": correct,
            "exact_answer_correct": result["generated_text"].strip() == row["expected_output"],
            "pocket_writeback_used": pocket_used,
            "pocket_writeback_count": result.get("pocket_writeback_count"),
            "value_selection_source": result.get("value_selection_source"),
            "highway_retained": result.get("highway_retained"),
            "failure_reason": None if correct and pocket_used else ("wrong_value" if not correct else "pocket_not_used"),
        }
        scored.append(item)
        by_group[row["contrast_group_id"]].append(item)
    group_results: list[dict[str, Any]] = []
    for group_id, group in sorted(by_group.items()):
        generated_values = [item["generated_value"] for item in group]
        expected_values = [item["expected_value"] for item in group]
        all_correct = all(item["answer_value_correct"] for item in group)
        all_pocket = all(item["pocket_writeback_used"] for item in group)
        group_results.append(
            {
                "arm": arm,
                "group_id": group_id,
                "family": group[0]["family"],
                "row_count": len(group),
                "all_correct": all_correct,
                "all_pocket_writeback_used": all_pocket,
                "distinct_expected": len(set(expected_values)) == len(expected_values),
                "distinct_generated": len(set(generated_values)) == len(generated_values),
                "group_pass": all_correct and all_pocket and len(set(generated_values)) == len(generated_values),
            }
        )
    metrics = {
        "schema_version": "phase_138yq_pocket_gating_metrics_v1",
        "arm": arm,
        "row_count": len(scored),
        "group_count": len(group_results),
        "answer_value_accuracy": rate(sum(1 for item in scored if item["answer_value_correct"]), len(scored)),
        "exact_answer_accuracy": rate(sum(1 for item in scored if item["exact_answer_correct"]), len(scored)),
        "pocket_writeback_rate": rate(sum(1 for item in scored if item["pocket_writeback_used"]), len(scored)),
        "phase_transport_success_rate": rate(sum(1 for item in scored if item["pocket_writeback_used"] and item["highway_retained"] is True), len(scored)),
        "highway_retention_rate": rate(sum(1 for item in scored if item["highway_retained"] is True), len(scored)),
        "contrast_group_accuracy": rate(sum(1 for item in group_results if item["group_pass"]), len(group_results)),
        "same_value_for_all_rows_rate": rate(sum(1 for item in group_results if len(set(item.get("generated_values", []))) == 1), len(group_results)),
        "all_rows_open_pocket_source": all(item["value_selection_source"] == "open_pocket_writeback" for item in scored) if scored else False,
    }
    return scored, metrics, group_results


def rate(count: int, total: int) -> float:
    return count / total if total else 0.0


def run_controls(helper: Any, checkpoint_path: Path, checkpoint_hash: str, max_new_tokens: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for idx, row in enumerate(control_rows()):
        request = request_for(helper, row["prompt"], checkpoint_path, checkpoint_hash, 2600 + idx, max_new_tokens)
        response = helper.raw_generate(request)
        generated_value = first_value_after_answer_e(response["generated_text"])
        control_passed = generated_value == row["answer_value"]
        results.append(
            {
                "control": row["control"],
                "generated_text": response["generated_text"],
                "generated_value": generated_value,
                "expected_failure_value": row["expected_failure_value"],
                "value_selection_source": response.get("value_selection_source"),
                "pocket_writeback_count": response.get("pocket_writeback_count"),
                "control_passed": control_passed,
            }
        )
    return results, {
        "schema_version": "phase_138yq_control_arm_report_v1",
        "control_count": len(results),
        "controls_failed": all(not row["control_passed"] for row in results),
        "passed_controls": [row["control"] for row in results if row["control_passed"]],
    }


def scan_ast() -> dict[str, Any]:
    failures: list[str] = []
    for path in [HELPER_PATH, RUNNER_PATH, CHECKER_PATH]:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("run_stable_loop_phase_lock_"):
                failures.append(f"old_runner_import:{rel(path)}")
    return {"schema_version": "phase_138yq_ast_scan_v1", "passed": not failures, "failures": failures}


def forbidden_canary(helper: Any, checkpoint_path: Path, checkpoint_hash: str, max_new_tokens: int) -> dict[str, Any]:
    request = request_for(helper, "GATE:POCKET_OPEN\nPOCKET_VALUE=EV_CANARY", checkpoint_path, checkpoint_hash, 2699, max_new_tokens)
    request["expected_output"] = "ANSWER=EEV_CANARY"
    try:
        helper.raw_generate(request)
    except Exception as exc:
        verdict = getattr(exc, "verdict", "")
        return {"schema_version": "phase_138yq_canary_v1", "passed": verdict == "RAW_GENERATION_FORBIDDEN_INPUT_DETECTED", "verdict": verdict, "message": str(exc)}
    return {"schema_version": "phase_138yq_canary_v1", "passed": False, "verdict": "CANARY_NOT_REJECTED"}


def compare(main_metrics: dict[str, Any], ablation_metrics: dict[str, Any], row_count: int) -> dict[str, Any]:
    return {
        "schema_version": "phase_138yq_arm_comparison_v1",
        "all_eval_rows_match": True,
        "eval_row_count": row_count,
        "main_arm": MAIN_ARM,
        "ablation_arm": ABLATION_ARM,
        "main_answer_value_accuracy": main_metrics["answer_value_accuracy"],
        "ablation_answer_value_accuracy": ablation_metrics["answer_value_accuracy"],
        "pocket_ablation_delta_answer_value_accuracy": main_metrics["answer_value_accuracy"] - ablation_metrics["answer_value_accuracy"],
        "main_pocket_writeback_rate": main_metrics["pocket_writeback_rate"],
        "ablation_pocket_writeback_rate": ablation_metrics["pocket_writeback_rate"],
        "pocket_ablation_decision_critical": (main_metrics["answer_value_accuracy"] - ablation_metrics["answer_value_accuracy"]) >= 0.20,
        "architecture_superiority_claimed": False,
    }


def write_report(out: Path, decision: dict[str, Any], comparison: dict[str, Any]) -> None:
    text = f"""# {MILESTONE}

Decision: `{decision['decision']}`

Verdict: `{decision['verdict']}`

Next: `{decision['next']}`

Boundary: {BOUNDARY_TEXT}

Pocket-gated evidence:

- main answer value accuracy: `{comparison['main_answer_value_accuracy']}`
- ablation answer value accuracy: `{comparison['ablation_answer_value_accuracy']}`
- ablation delta: `{comparison['pocket_ablation_delta_answer_value_accuracy']}`
- main pocket writeback rate: `{comparison['main_pocket_writeback_rate']}`
- ablation pocket writeback rate: `{comparison['ablation_pocket_writeback_rate']}`

This is constrained pocket-gated adapter evidence, not GPT-like readiness and not
broad architecture superiority.
"""
    write_text(out / "report.md", text)


def main() -> int:
    parser = argparse.ArgumentParser(description=MILESTONE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--upstream-138yp-root", type=Path, default=DEFAULT_138YP_ROOT)
    parser.add_argument("--group-count", type=int, default=8)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--heartbeat-sec", type=int, default=20)
    args = parser.parse_args()

    out = resolve_target_out(args.out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    append_progress(out, "startup", milestone=MILESTONE)
    write_json(out / "queue.json", {"schema_version": "phase_138yq_queue_v1", "milestone": MILESTONE, "status": "running"})

    upstream = require_138yp(resolve_repo_path(args.upstream_138yp_root))
    write_json(out / "upstream_138yp_manifest.json", upstream)
    append_progress(out, "upstream verification", upstream=upstream)

    helper = load_helper()
    provenance = {
        "schema_version": "phase_138yq_helper_provenance_v1",
        "helper_path": rel(HELPER_PATH),
        "helper_source_sha256": sha256_file(HELPER_PATH),
        "helper_version": getattr(helper, "HELPER_VERSION", None),
        "adapter_backend_name": getattr(helper, "INSTNCT_MUTATION_BACKEND", None),
        "strict_pocket_gated_symbols_present": hasattr(helper, "_instnct_select_open_pocket_value"),
    }
    write_json(out / "helper_provenance_verification.json", provenance)
    write_json(out / "adapter_contract.json", read_json(resolve_repo_path(args.upstream_138yp_root) / "target_138yq_backend_contract.json"))
    append_progress(out, "helper provenance", strict_pocket_gated=provenance["strict_pocket_gated_symbols_present"])

    main_checkpoint, main_manifest = build_manifest(out, ablation=False)
    ablation_checkpoint, ablation_manifest = build_manifest(out, ablation=True)
    write_json(out / "instnct_pocket_gated_manifest.json", main_manifest)
    write_json(out / "instnct_pocket_gated_ablation_manifest.json", ablation_manifest)
    append_progress(out, "manifest build", main_hash=main_manifest["checkpoint_sha256"], ablation_hash=ablation_manifest["checkpoint_sha256"])

    ast_report = scan_ast()
    write_json(out / "ast_shortcut_scan_report.json", ast_report)
    canary = forbidden_canary(helper, main_checkpoint, main_manifest["checkpoint_sha256"], args.max_new_tokens)
    write_json(out / "expected_output_canary_report.json", canary)
    write_json(out / "forbidden_input_rejection_report.json", {"schema_version": "phase_138yq_forbidden_input_rejection_v1", "passed": canary["passed"], "canary_verdict": canary["verdict"]})
    append_progress(out, "canary and ast", canary_passed=canary["passed"], ast_passed=ast_report["passed"])

    rows = eval_rows(args.group_count, args.group_size)
    write_jsonl(out / "eval_rows.jsonl", rows)
    write_json(out / "pocket_payload_manifest.json", {"schema_version": "phase_138yq_pocket_payload_manifest_v1", "row_count": len(rows), "payload_markers": ["POCKET_VALUE=", "POCKET_BIND=", "POCKET_TABLE_ROW="], "all_rows_require_gate": True})
    append_progress(out, "eval row build", row_count=len(rows))

    main_results = run_arm(helper, out, MAIN_ARM, rows, main_checkpoint, main_manifest["checkpoint_sha256"], args.max_new_tokens, args.heartbeat_sec)
    ablation_results = run_arm(helper, out, ABLATION_ARM, rows, ablation_checkpoint, ablation_manifest["checkpoint_sha256"], args.max_new_tokens, args.heartbeat_sec)
    write_jsonl(out / "raw_generation_results.jsonl", main_results)
    write_jsonl(out / "pocket_ablation_results.jsonl", ablation_results)
    write_jsonl(out / "raw_generation_trace.jsonl", main_results + ablation_results)
    write_jsonl(out / "pocket_trace.jsonl", [{"row_id": row["row_id"], "arm": row["arm"], "pocket_writeback_count": row["pocket_writeback_count"], "value_selection_source": row["value_selection_source"], "highway_retained": row["highway_retained"]} for row in main_results + ablation_results])
    append_progress(out, "generation", main_rows=len(main_results), ablation_rows=len(ablation_results))

    main_scored, main_metrics, main_groups = score_rows(MAIN_ARM, rows, main_results)
    ablation_scored, ablation_metrics, ablation_groups = score_rows(ABLATION_ARM, rows, ablation_results)
    write_jsonl(out / "scoring_results.jsonl", main_scored + ablation_scored)
    write_jsonl(out / "contrast_group_results.jsonl", main_groups + ablation_groups)
    comparison = compare(main_metrics, ablation_metrics, len(rows))
    write_json(out / "pocket_gating_metrics.json", {"schema_version": "phase_138yq_pocket_gating_metrics_bundle_v1", "main": main_metrics, "ablation": ablation_metrics})
    write_json(out / "arm_comparison.json", comparison)
    append_progress(out, "scoring and comparison", delta=comparison["pocket_ablation_delta_answer_value_accuracy"])

    controls, control_report = run_controls(helper, main_checkpoint, main_manifest["checkpoint_sha256"], args.max_new_tokens)
    write_jsonl(out / "control_results.jsonl", controls)
    write_json(out / "control_arm_report.json", control_report)
    append_progress(out, "controls", controls_failed=control_report["controls_failed"])

    replay = run_arm(helper, out, f"{MAIN_ARM}_replay", rows, main_checkpoint, main_manifest["checkpoint_sha256"], args.max_new_tokens, args.heartbeat_sec)
    deterministic = [row["generated_text_hash"] for row in replay] == [row["generated_text_hash"] for row in main_results]
    write_json(out / "determinism_replay_report.json", {"schema_version": "phase_138yq_determinism_replay_report_v1", "replay_attempted": True, "same_rows": True, "same_checkpoint": True, "generated_text_hashes_equal": deterministic, "deterministic_replay_passed": deterministic})
    append_progress(out, "determinism replay", passed=deterministic)

    generated_before_scoring = {
        "schema_version": "phase_138yq_generated_before_scoring_report_v1",
        "passed": True,
        "generated_text_produced_before_scoring": True,
        "all_helper_requests_allowed_keys_only": all(set(row["helper_request"]) == ALLOWED_HELPER_KEYS for row in main_results + ablation_results + replay),
        "expected_or_scorer_metadata_in_helper_requests": False,
    }
    write_json(out / "generated_before_scoring_report.json", generated_before_scoring)
    mutation_candidates = [
        {"candidate": MAIN_ARM, "fitness": main_metrics["answer_value_accuracy"], "pocket_writeback_rate": main_metrics["pocket_writeback_rate"], "selected": True},
        {"candidate": ABLATION_ARM, "fitness": ablation_metrics["answer_value_accuracy"], "pocket_writeback_rate": ablation_metrics["pocket_writeback_rate"], "selected": False},
    ]
    write_jsonl(out / "mutation_candidate_results.jsonl", mutation_candidates)

    positive = (
        main_metrics["answer_value_accuracy"] >= 0.25
        and main_metrics["pocket_writeback_rate"] >= 0.95
        and main_metrics["phase_transport_success_rate"] >= 0.95
        and ablation_metrics["answer_value_accuracy"] <= 0.05
        and comparison["pocket_ablation_delta_answer_value_accuracy"] >= 0.20
        and deterministic
        and control_report["controls_failed"]
    )
    if positive:
        decision_name = "instnct_pocket_gated_value_grounding_probe_positive"
        verdict = "INSTNCT_POCKET_GATED_VALUE_GROUNDING_PROBE_POSITIVE"
        next_step = "139YQ_INSTNCT_POCKET_GATED_VALUE_GROUNDING_SCALE_CONFIRM"
    else:
        decision_name = "instnct_pocket_gated_value_grounding_probe_failed"
        verdict = "INSTNCT_POCKET_GATED_VALUE_GROUNDING_PROBE_FAILS"
        next_step = "138YQ_FAILURE_ANALYSIS"
    decision = {
        "schema_version": "phase_138yq_decision_v1",
        "decision": decision_name,
        "verdict": verdict,
        "next": next_step,
        "clean_negative_valid": True,
        "architecture_superiority_claimed": False,
        "pocket_mechanism_claimed": positive,
        "pocket_gated_value_grounding_evidence": positive,
        "value_grounding_claimed": False,
        **FALSE_FLAGS,
    }
    write_json(out / "decision.json", decision)
    summary = {"schema_version": "phase_138yq_summary_v1", "milestone": MILESTONE, "status": "complete", "boundary": BOUNDARY_TEXT, "metrics": {"main_answer_value_accuracy": main_metrics["answer_value_accuracy"], "ablation_answer_value_accuracy": ablation_metrics["answer_value_accuracy"], "pocket_ablation_delta_answer_value_accuracy": comparison["pocket_ablation_delta_answer_value_accuracy"], "main_pocket_writeback_rate": main_metrics["pocket_writeback_rate"], "deterministic_replay_passed": deterministic}, **decision}
    write_json(out / "summary.json", summary)
    write_report(out, decision, comparison)
    append_progress(out, "decision", decision=decision_name, next=next_step)
    append_progress(out, "final verdict", verdict=verdict)
    write_json(out / "queue.json", {"schema_version": "phase_138yq_queue_v1", "milestone": MILESTONE, "status": "complete", "decision": decision_name, "next": next_step})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

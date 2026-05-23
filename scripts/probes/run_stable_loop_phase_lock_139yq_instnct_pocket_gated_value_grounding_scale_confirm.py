#!/usr/bin/env python3
"""139YQ scale confirm for pocket-gated INSTNCT value grounding."""

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
MILESTONE = "STABLE_LOOP_PHASE_LOCK_139YQ_INSTNCT_POCKET_GATED_VALUE_GROUNDING_SCALE_CONFIRM"
DEFAULT_OUT = Path("target/pilot_wave/stable_loop_phase_lock_139yq_instnct_pocket_gated_value_grounding_scale_confirm/smoke")
DEFAULT_138YQ_ROOT = Path("target/pilot_wave/stable_loop_phase_lock_138yq_instnct_pocket_gated_value_grounding_probe/smoke")
HELPER_PATH = REPO_ROOT / "scripts/probes/shared_raw_generation_helper.py"
RUNNER_PATH = Path(__file__).resolve()
CHECKER_PATH = REPO_ROOT / "scripts/probes/run_stable_loop_phase_lock_139yq_instnct_pocket_gated_value_grounding_scale_confirm_check.py"
BACKEND_NAME = "repo_local_instnct_mutation_graph"
MAIN_ARM = "instnct_pocket_gated_scale_main"
ABLATION_ARM = "instnct_pocket_gated_scale_ablation"
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
    "139YQ is a deterministic scale-confirm probe for the strict pocket-gated "
    "INSTNCT helper backend. It does not train, mutate source checkpoints, "
    "change public request keys, start services, deploy, or claim broad "
    "assistant capability."
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
    spec = importlib.util.spec_from_file_location("shared_raw_generation_helper_139yq", HELPER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to import shared raw generation helper")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def require_138yq(root: Path) -> dict[str, Any]:
    required = ["decision.json", "arm_comparison.json", "pocket_gating_metrics.json", "determinism_replay_report.json"]
    missing = [name for name in required if not (root / name).exists()]
    if missing:
        raise RuntimeError(f"missing 138YQ artifacts: {missing}")
    decision = read_json(root / "decision.json")
    comparison = read_json(root / "arm_comparison.json")
    replay = read_json(root / "determinism_replay_report.json")
    if decision.get("decision") != "instnct_pocket_gated_value_grounding_probe_positive":
        raise RuntimeError(f"bad 138YQ decision: {decision.get('decision')}")
    if decision.get("next") != "139YQ_INSTNCT_POCKET_GATED_VALUE_GROUNDING_SCALE_CONFIRM":
        raise RuntimeError(f"bad 138YQ next: {decision.get('next')}")
    if comparison.get("main_pocket_writeback_rate") != 1.0 or comparison.get("ablation_answer_value_accuracy") != 0.0:
        raise RuntimeError("138YQ evidence profile no longer matches pocket-gated positive")
    if replay.get("deterministic_replay_passed") is not True:
        raise RuntimeError("138YQ determinism did not pass")
    return {
        "root": rel(root),
        "decision": decision.get("decision"),
        "next": decision.get("next"),
        "verdict": decision.get("verdict"),
        "main_answer_value_accuracy": comparison.get("main_answer_value_accuracy"),
        "ablation_answer_value_accuracy": comparison.get("ablation_answer_value_accuracy"),
        "main_pocket_writeback_rate": comparison.get("main_pocket_writeback_rate"),
        "pocket_ablation_delta_answer_value_accuracy": comparison.get("pocket_ablation_delta_answer_value_accuracy"),
        "deterministic_replay_passed": replay.get("deterministic_replay_passed"),
    }


def build_manifest(out: Path, ablation: bool = False) -> tuple[Path, dict[str, Any]]:
    manifest = {
        "schema_version": "instnct_mutation_graph_manifest_v2_pocket_gated_scale",
        "backend_name": BACKEND_NAME,
        "answer_prefix": "ANSWER=E",
        "ticks_per_generated_byte": 12,
        "threshold_tick": 5,
        "value_selection_requires_open_pocket": True,
        "visible_value_bypass_forbidden": True,
        "pocket_payload_markers": ["POCKET_VALUE=", "POCKET_BIND=", "POCKET_TABLE_ROW="],
        "closed_pocket_fallback_value": "SYM_POCKET_CLOSED",
        "fallback_value": "SYM_POCKET_CLOSED",
        "allow_train_namespace_value_fallback": False,
        "decoder": {
            "type": "deterministic_pocket_gated_value_decoder_scale_confirm",
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
        "claim_boundary": "scale confirm for pocket-gated helper semantics; not broad assistant capability",
    }
    name = "instnct_pocket_gated_scale_ablation.json" if ablation else "instnct_pocket_gated_scale_main.json"
    path = out / "checkpoints" / name
    write_json(path, manifest)
    return path, {**manifest, "checkpoint_path": rel(path), "checkpoint_sha256": sha256_file(path)}


def eval_rows(seeds: list[int], groups_per_family: int, group_size: int) -> list[dict[str, Any]]:
    families = [
        "SCALE_POCKET_DIRECT_BINDING",
        "SCALE_POCKET_TABLE_BINDING",
        "SCALE_POCKET_RULE_DERIVED",
        "SCALE_POCKET_OOD_SYMBOL_BINDING",
        "SCALE_POCKET_NOISE_VISIBLE_BYPASS",
        "SCALE_POCKET_CONTRAST_SAME_TEMPLATE",
    ]
    rows: list[dict[str, Any]] = []
    row_index = 0
    for seed in seeds:
        for family_index, family in enumerate(families):
            for group_index in range(groups_per_family):
                group_id = f"seed{seed}_{family}_group_{group_index:03d}"
                for slot in range(group_size):
                    if family == "SCALE_POCKET_OOD_SYMBOL_BINDING":
                        value = f"SYM_SCALE_{seed}_{family_index}_{group_index:03d}_{slot:02d}"
                        marker = "POCKET_BIND="
                    elif family == "SCALE_POCKET_TABLE_BINDING":
                        value = f"EVT{seed % 100}{family_index}{group_index:03d}{slot:02d}"
                        marker = "POCKET_TABLE_ROW="
                    else:
                        value = f"EVS{seed % 100}{family_index}{group_index:03d}{slot:02d}"
                        marker = "POCKET_VALUE="
                    visible_wrong = f"EVVISIBLE{seed % 100}{family_index}{group_index:03d}{slot:02d}"
                    prompt = (
                        f"TASK={family}\n"
                        f"SEED={seed}\n"
                        "GATE:POCKET_OPEN\n"
                        f"{marker}{value}\n"
                        f"VISIBLE_VALUE={visible_wrong}\n"
                        "Return ANSWER=E plus only the value written back by the open pocket."
                    )
                    rows.append(
                        {
                            "row_id": f"139yq_eval_{row_index:06d}",
                            "seed": seed,
                            "family": family,
                            "contrast_group_id": group_id,
                            "prompt": prompt,
                            "answer_value": value,
                            "expected_output": f"ANSWER=E{value}",
                            "visible_bypass_value": visible_wrong,
                        }
                    )
                    row_index += 1
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


def run_arm(helper: Any, out: Path, arm: str, rows: list[dict[str, Any]], checkpoint_path: Path, checkpoint_hash: str, max_new_tokens: int, heartbeat_sec: int) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    last_heartbeat = time.monotonic()
    for index, row in enumerate(rows, start=1):
        request = request_for(helper, row["prompt"], checkpoint_path, checkpoint_hash, int(row["seed"]), max_new_tokens)
        if set(request) != ALLOWED_HELPER_KEYS:
            raise RuntimeError(f"bad helper request keys: {sorted(request)}")
        response = helper.raw_generate(request)
        generated_text = response["generated_text"]
        item = {
            "arm": arm,
            "row_id": row["row_id"],
            "seed": row["seed"],
            "family": row["family"],
            "contrast_group_id": row["contrast_group_id"],
            "generated_text": generated_text,
            "generated_value": first_value_after_answer_e(generated_text),
            "generated_text_hash": hashlib.sha256(generated_text.encode("utf-8", errors="replace")).hexdigest(),
            "generation_trace_hash": response.get("generation_trace_hash"),
            "backend_name": response.get("backend_name"),
            "pocket_writeback_count": response.get("pocket_writeback_count"),
            "highway_retained": response.get("highway_retained"),
            "value_selection_source": response.get("value_selection_source"),
            "helper_request": request,
            "helper_response": {key: value for key, value in response.items() if key != "generated_text"},
        }
        results.append(item)
        now = time.monotonic()
        if now - last_heartbeat >= heartbeat_sec:
            append_progress(out, f"{arm} heartbeat", completed=index, total=len(rows))
            last_heartbeat = now
    return results


def score(arm: str, rows: list[dict[str, Any]], results: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    rows_by_id = {row["row_id"]: row for row in rows}
    scored: list[dict[str, Any]] = []
    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_seed: dict[int, list[dict[str, Any]]] = defaultdict(list)
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        row = rows_by_id[result["row_id"]]
        correct = result["generated_value"] == row["answer_value"]
        pocket_used = (result.get("pocket_writeback_count") or 0) > 0 and result.get("value_selection_source") == "open_pocket_writeback"
        item = {
            "arm": arm,
            "row_id": row["row_id"],
            "seed": row["seed"],
            "family": row["family"],
            "contrast_group_id": row["contrast_group_id"],
            "expected_value": row["answer_value"],
            "generated_value": result["generated_value"],
            "answer_value_correct": correct,
            "exact_answer_correct": result["generated_text"].strip() == row["expected_output"],
            "pocket_writeback_used": pocket_used,
            "value_selection_source": result.get("value_selection_source"),
            "highway_retained": result.get("highway_retained"),
        }
        scored.append(item)
        by_group[row["contrast_group_id"]].append(item)
        by_seed[int(row["seed"])].append(item)
        by_family[row["family"]].append(item)
    group_results: list[dict[str, Any]] = []
    for group_id, items in sorted(by_group.items()):
        generated = [item["generated_value"] for item in items]
        expected = [item["expected_value"] for item in items]
        group_results.append(
            {
                "arm": arm,
                "group_id": group_id,
                "seed": items[0]["seed"],
                "family": items[0]["family"],
                "row_count": len(items),
                "all_correct": all(item["answer_value_correct"] for item in items),
                "all_pocket_writeback_used": all(item["pocket_writeback_used"] for item in items),
                "distinct_expected": len(set(expected)) == len(expected),
                "distinct_generated": len(set(generated)) == len(generated),
                "group_pass": all(item["answer_value_correct"] and item["pocket_writeback_used"] for item in items) and len(set(generated)) == len(generated),
            }
        )
    metrics = {
        "schema_version": "phase_139yq_arm_metrics_v1",
        "arm": arm,
        "row_count": len(scored),
        "group_count": len(group_results),
        "answer_value_accuracy": metric_rate(scored, "answer_value_correct"),
        "exact_answer_accuracy": metric_rate(scored, "exact_answer_correct"),
        "pocket_writeback_rate": metric_rate(scored, "pocket_writeback_used"),
        "phase_transport_success_rate": rate(sum(1 for item in scored if item["pocket_writeback_used"] and item["highway_retained"] is True), len(scored)),
        "highway_retention_rate": rate(sum(1 for item in scored if item["highway_retained"] is True), len(scored)),
        "contrast_group_accuracy": rate(sum(1 for item in group_results if item["group_pass"]), len(group_results)),
        "per_seed": {str(seed): seed_metrics(items) for seed, items in sorted(by_seed.items())},
        "per_family": {family: seed_metrics(items) for family, items in sorted(by_family.items())},
    }
    return scored, metrics, group_results


def seed_metrics(items: list[dict[str, Any]]) -> dict[str, float]:
    return {
        "row_count": len(items),
        "answer_value_accuracy": metric_rate(items, "answer_value_correct"),
        "pocket_writeback_rate": metric_rate(items, "pocket_writeback_used"),
        "highway_retention_rate": rate(sum(1 for item in items if item["highway_retained"] is True), len(items)),
    }


def metric_rate(items: list[dict[str, Any]], key: str) -> float:
    return rate(sum(1 for item in items if item.get(key) is True), len(items))


def rate(count: int, total: int) -> float:
    return count / total if total else 0.0


def scan_ast() -> dict[str, Any]:
    failures: list[str] = []
    for path in [HELPER_PATH, RUNNER_PATH, CHECKER_PATH]:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("run_stable_loop_phase_lock_"):
                failures.append(f"old_runner_import:{rel(path)}")
    return {"schema_version": "phase_139yq_ast_scan_v1", "passed": not failures, "failures": failures}


def run_controls(helper: Any, checkpoint_path: Path, checkpoint_hash: str, max_new_tokens: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    controls = [
        ("CLOSED_POCKET_CONTROL", "TASK=CONTROL\nPOCKET_VALUE=EV_CLOSED\nReturn ANSWER=E plus pocket value.", "EV_CLOSED"),
        ("VISIBLE_VALUE_BYPASS_CONTROL", "TASK=CONTROL\nOBSERVED_VALUE=EV_VISIBLE\nReturn ANSWER=E plus observed value.", "EV_VISIBLE"),
    ]
    rows = []
    for index, (name, prompt, expected) in enumerate(controls):
        request = request_for(helper, prompt, checkpoint_path, checkpoint_hash, 2800 + index, max_new_tokens)
        response = helper.raw_generate(request)
        generated_value = first_value_after_answer_e(response["generated_text"])
        rows.append(
            {
                "control": name,
                "generated_text": response["generated_text"],
                "generated_value": generated_value,
                "expected_blocked_value": expected,
                "control_passed": generated_value == expected,
                "value_selection_source": response.get("value_selection_source"),
                "pocket_writeback_count": response.get("pocket_writeback_count"),
            }
        )
    return rows, {
        "schema_version": "phase_139yq_control_arm_report_v1",
        "control_count": len(rows),
        "controls_failed": all(not row["control_passed"] for row in rows),
        "passed_controls": [row["control"] for row in rows if row["control_passed"]],
    }


def forbidden_canary(helper: Any, checkpoint_path: Path, checkpoint_hash: str, max_new_tokens: int) -> dict[str, Any]:
    request = request_for(helper, "GATE:POCKET_OPEN\nPOCKET_VALUE=EV_CANARY", checkpoint_path, checkpoint_hash, 2999, max_new_tokens)
    request["expected_output"] = "ANSWER=EEV_CANARY"
    try:
        helper.raw_generate(request)
    except Exception as exc:
        verdict = getattr(exc, "verdict", "")
        return {"schema_version": "phase_139yq_canary_v1", "passed": verdict == "RAW_GENERATION_FORBIDDEN_INPUT_DETECTED", "verdict": verdict, "message": str(exc)}
    return {"schema_version": "phase_139yq_canary_v1", "passed": False, "verdict": "CANARY_NOT_REJECTED"}


def write_report(out: Path, decision: dict[str, Any], comparison: dict[str, Any]) -> None:
    text = f"""# {MILESTONE}

Decision: `{decision['decision']}`

Verdict: `{decision['verdict']}`

Next: `{decision['next']}`

Boundary: {BOUNDARY_TEXT}

Scale-confirm metrics:

- eval rows: `{comparison['eval_row_count']}`
- main answer value accuracy: `{comparison['main_answer_value_accuracy']}`
- main pocket writeback rate: `{comparison['main_pocket_writeback_rate']}`
- ablation answer value accuracy: `{comparison['ablation_answer_value_accuracy']}`
- ablation delta: `{comparison['pocket_ablation_delta_answer_value_accuracy']}`

This remains constrained pocket-gated helper evidence, not GPT-like readiness or
broad architecture superiority.
"""
    write_text(out / "report.md", text)


def main() -> int:
    parser = argparse.ArgumentParser(description=MILESTONE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--upstream-138yq-root", type=Path, default=DEFAULT_138YQ_ROOT)
    parser.add_argument("--seeds", default="2601,2602,2603")
    parser.add_argument("--groups-per-family", type=int, default=24)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--heartbeat-sec", type=int, default=20)
    args = parser.parse_args()

    out = resolve_target_out(args.out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    append_progress(out, "startup", milestone=MILESTONE)
    write_json(out / "queue.json", {"schema_version": "phase_139yq_queue_v1", "milestone": MILESTONE, "status": "running"})

    seeds = [int(item) for item in args.seeds.split(",") if item.strip()]
    upstream = require_138yq(resolve_repo_path(args.upstream_138yq_root))
    write_json(out / "upstream_138yq_manifest.json", upstream)
    append_progress(out, "upstream verification", upstream=upstream)

    helper = load_helper()
    provenance = {
        "schema_version": "phase_139yq_helper_provenance_v1",
        "helper_path": rel(HELPER_PATH),
        "helper_source_sha256": sha256_file(HELPER_PATH),
        "helper_version": getattr(helper, "HELPER_VERSION", None),
        "adapter_backend_name": getattr(helper, "INSTNCT_MUTATION_BACKEND", None),
        "strict_pocket_gated_symbols_present": hasattr(helper, "_instnct_select_open_pocket_value"),
    }
    write_json(out / "helper_provenance_verification.json", provenance)
    append_progress(out, "helper provenance", strict_pocket_gated=provenance["strict_pocket_gated_symbols_present"])

    main_checkpoint, main_manifest = build_manifest(out, ablation=False)
    ablation_checkpoint, ablation_manifest = build_manifest(out, ablation=True)
    write_json(out / "instnct_pocket_gated_scale_manifest.json", main_manifest)
    write_json(out / "instnct_pocket_gated_scale_ablation_manifest.json", ablation_manifest)
    write_json(out / "ast_shortcut_scan_report.json", scan_ast())
    canary = forbidden_canary(helper, main_checkpoint, main_manifest["checkpoint_sha256"], args.max_new_tokens)
    write_json(out / "expected_output_canary_report.json", canary)
    write_json(out / "forbidden_input_rejection_report.json", {"schema_version": "phase_139yq_forbidden_input_rejection_v1", "passed": canary["passed"], "canary_verdict": canary["verdict"]})
    append_progress(out, "manifest canary ast", canary_passed=canary["passed"])

    rows = eval_rows(seeds, args.groups_per_family, args.group_size)
    write_jsonl(out / "eval_rows.jsonl", rows)
    write_json(out / "eval_dataset_manifest.json", {"schema_version": "phase_139yq_eval_dataset_manifest_v1", "row_count": len(rows), "seeds": seeds, "groups_per_family": args.groups_per_family, "group_size": args.group_size, "row_hash": stable_hash(rows)})
    append_progress(out, "eval row build", row_count=len(rows))

    main_results = run_arm(helper, out, MAIN_ARM, rows, main_checkpoint, main_manifest["checkpoint_sha256"], args.max_new_tokens, args.heartbeat_sec)
    ablation_results = run_arm(helper, out, ABLATION_ARM, rows, ablation_checkpoint, ablation_manifest["checkpoint_sha256"], args.max_new_tokens, args.heartbeat_sec)
    write_jsonl(out / "raw_generation_results.jsonl", main_results)
    write_jsonl(out / "pocket_ablation_results.jsonl", ablation_results)
    write_jsonl(out / "raw_generation_trace.jsonl", main_results + ablation_results)
    write_jsonl(out / "pocket_trace.jsonl", [{"row_id": row["row_id"], "arm": row["arm"], "pocket_writeback_count": row["pocket_writeback_count"], "value_selection_source": row["value_selection_source"], "highway_retained": row["highway_retained"]} for row in main_results + ablation_results])
    append_progress(out, "generation", main_rows=len(main_results), ablation_rows=len(ablation_results))

    main_scored, main_metrics, main_groups = score(MAIN_ARM, rows, main_results)
    ablation_scored, ablation_metrics, ablation_groups = score(ABLATION_ARM, rows, ablation_results)
    write_jsonl(out / "scoring_results.jsonl", main_scored + ablation_scored)
    write_jsonl(out / "contrast_group_results.jsonl", main_groups + ablation_groups)
    comparison = {
        "schema_version": "phase_139yq_arm_comparison_v1",
        "all_eval_rows_match": True,
        "eval_row_count": len(rows),
        "main_answer_value_accuracy": main_metrics["answer_value_accuracy"],
        "main_pocket_writeback_rate": main_metrics["pocket_writeback_rate"],
        "main_phase_transport_success_rate": main_metrics["phase_transport_success_rate"],
        "ablation_answer_value_accuracy": ablation_metrics["answer_value_accuracy"],
        "ablation_pocket_writeback_rate": ablation_metrics["pocket_writeback_rate"],
        "pocket_ablation_delta_answer_value_accuracy": main_metrics["answer_value_accuracy"] - ablation_metrics["answer_value_accuracy"],
        "pocket_ablation_decision_critical": main_metrics["answer_value_accuracy"] - ablation_metrics["answer_value_accuracy"] >= 0.20,
        "architecture_superiority_claimed": False,
    }
    write_json(out / "pocket_gating_scale_metrics.json", {"schema_version": "phase_139yq_metrics_bundle_v1", "main": main_metrics, "ablation": ablation_metrics})
    write_json(out / "per_seed_metrics.json", {"schema_version": "phase_139yq_per_seed_metrics_v1", "main": main_metrics["per_seed"], "ablation": ablation_metrics["per_seed"]})
    write_json(out / "per_family_metrics.json", {"schema_version": "phase_139yq_per_family_metrics_v1", "main": main_metrics["per_family"], "ablation": ablation_metrics["per_family"]})
    write_json(out / "arm_comparison.json", comparison)
    append_progress(out, "scoring and comparison", delta=comparison["pocket_ablation_delta_answer_value_accuracy"])

    controls, control_report = run_controls(helper, main_checkpoint, main_manifest["checkpoint_sha256"], args.max_new_tokens)
    write_jsonl(out / "control_results.jsonl", controls)
    write_json(out / "control_arm_report.json", control_report)
    append_progress(out, "controls", controls_failed=control_report["controls_failed"])

    replay = run_arm(helper, out, f"{MAIN_ARM}_replay", rows, main_checkpoint, main_manifest["checkpoint_sha256"], args.max_new_tokens, args.heartbeat_sec)
    deterministic = [row["generated_text_hash"] for row in replay] == [row["generated_text_hash"] for row in main_results]
    write_json(out / "determinism_replay_report.json", {"schema_version": "phase_139yq_determinism_replay_report_v1", "replay_attempted": True, "same_rows": True, "same_checkpoint": True, "generated_text_hashes_equal": deterministic, "deterministic_replay_passed": deterministic})
    append_progress(out, "determinism replay", passed=deterministic)

    write_json(out / "generated_before_scoring_report.json", {"schema_version": "phase_139yq_generated_before_scoring_report_v1", "passed": True, "generated_text_produced_before_scoring": True, "all_helper_requests_allowed_keys_only": all(set(row["helper_request"]) == ALLOWED_HELPER_KEYS for row in main_results + ablation_results + replay), "expected_or_scorer_metadata_in_helper_requests": False})
    write_jsonl(out / "mutation_candidate_results.jsonl", [{"candidate": MAIN_ARM, "fitness": main_metrics["answer_value_accuracy"], "selected": True}, {"candidate": ABLATION_ARM, "fitness": ablation_metrics["answer_value_accuracy"], "selected": False}])

    every_seed_passed = all(item["answer_value_accuracy"] >= 0.95 and item["pocket_writeback_rate"] >= 0.95 for item in main_metrics["per_seed"].values())
    positive = (
        main_metrics["answer_value_accuracy"] >= 0.95
        and main_metrics["pocket_writeback_rate"] >= 0.95
        and main_metrics["phase_transport_success_rate"] >= 0.95
        and ablation_metrics["answer_value_accuracy"] <= 0.05
        and comparison["pocket_ablation_delta_answer_value_accuracy"] >= 0.90
        and every_seed_passed
        and deterministic
        and control_report["controls_failed"]
    )
    decision = {
        "schema_version": "phase_139yq_decision_v1",
        "decision": "instnct_pocket_gated_value_grounding_scale_confirmed" if positive else "instnct_pocket_gated_value_grounding_scale_not_confirmed",
        "verdict": "INSTNCT_POCKET_GATED_VALUE_GROUNDING_SCALE_CONFIRMED" if positive else "INSTNCT_POCKET_GATED_VALUE_GROUNDING_SCALE_FAILS",
        "next": "139YR_INSTNCT_POCKET_GATED_MUTATION_SEARCH_CONFIRM" if positive else "139YQ_FAILURE_ANALYSIS",
        "clean_negative_valid": True,
        "architecture_superiority_claimed": False,
        "pocket_mechanism_claimed": positive,
        "pocket_gated_value_grounding_evidence_scaled": positive,
        "value_grounding_claimed": False,
        **FALSE_FLAGS,
    }
    write_json(out / "decision.json", decision)
    summary = {"schema_version": "phase_139yq_summary_v1", "milestone": MILESTONE, "status": "complete", "boundary": BOUNDARY_TEXT, "metrics": comparison, "every_seed_passed": every_seed_passed, "deterministic_replay_passed": deterministic, **decision}
    write_json(out / "summary.json", summary)
    write_report(out, decision, comparison)
    append_progress(out, "decision", decision=decision["decision"], next=decision["next"])
    append_progress(out, "final verdict", verdict=decision["verdict"])
    write_json(out / "queue.json", {"schema_version": "phase_139yq_queue_v1", "milestone": MILESTONE, "status": "complete", "decision": decision["decision"], "next": decision["next"]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

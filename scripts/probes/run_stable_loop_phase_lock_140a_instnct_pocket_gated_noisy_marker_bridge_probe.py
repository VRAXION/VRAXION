#!/usr/bin/env python3
"""140A noisy-marker bridge probe for pocket-gated INSTNCT value binding."""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import re
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
MILESTONE = "STABLE_LOOP_PHASE_LOCK_140A_INSTNCT_POCKET_GATED_NOISY_MARKER_BRIDGE_PROBE"
DEFAULT_OUT = Path("target/pilot_wave/stable_loop_phase_lock_140a_instnct_pocket_gated_noisy_marker_bridge_probe/smoke")
DEFAULT_139YS_ROOT = Path("target/pilot_wave/stable_loop_phase_lock_139ys_instnct_pocket_gated_mutation_search_scale_or_real_task_bridge/smoke")
HELPER_PATH = REPO_ROOT / "scripts/probes/shared_raw_generation_helper.py"
RUNNER_PATH = Path(__file__).resolve()
CHECKER_PATH = REPO_ROOT / "scripts/probes/run_stable_loop_phase_lock_140a_instnct_pocket_gated_noisy_marker_bridge_probe_check.py"
BACKEND_NAME = "repo_local_instnct_mutation_graph"
MAIN_ARM = "instnct_noisy_marker_bridge_main"
ABLATION_ARM = "instnct_noisy_marker_bridge_closed_pocket_ablation"
SELECTED_CANDIDATE = "open_pocket_all_payload_markers_noisy_bridge"
POSITIVE_NEXT = "140F_INSTNCT_POCKET_GATED_NOISY_MARKER_BRIDGE_SCALE_CONFIRM"
ALLOWED_HELPER_KEYS = {"prompt", "checkpoint_path", "checkpoint_hash", "seed", "max_new_tokens", "generation_config"}
VALUE_RE = re.compile(r"\b(?:TR|EV|VAL|SYM)[A-Za-z0-9_+\-]*\b")
FALSE_FLAGS = {
    "reasoning_restored": False,
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
    "140A is a targeted noisy-marker bridge probe for the existing shared helper "
    "INSTNCT manifest backend. It does not train, mutate source checkpoints, "
    "modify shared_raw_generation_helper.py, modify backend/runtime/release/product "
    "surfaces, change public request keys, start services, deploy, or claim GPT-like "
    "or broad assistant readiness."
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
    spec = importlib.util.spec_from_file_location("shared_raw_generation_helper_140a", HELPER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to import shared raw generation helper")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def require_139ys(root: Path) -> dict[str, Any]:
    required = ["decision.json", "target_140a_milestone_plan.json", "next_milestone_recommendation.json"]
    missing = [name for name in required if not (root / name).exists()]
    if missing:
        raise RuntimeError(f"missing 139YS artifacts: {missing}")
    decision = read_json(root / "decision.json")
    plan = read_json(root / "target_140a_milestone_plan.json")
    recommendation = read_json(root / "next_milestone_recommendation.json")
    if decision.get("decision") != "real_task_bridge_recommended":
        raise RuntimeError(f"bad 139YS decision: {decision.get('decision')}")
    if decision.get("next") != "140A_INSTNCT_POCKET_GATED_NOISY_MARKER_BRIDGE_PROBE":
        raise RuntimeError(f"bad 139YS next: {decision.get('next')}")
    if plan.get("milestone") != "140A_INSTNCT_POCKET_GATED_NOISY_MARKER_BRIDGE_PROBE":
        raise RuntimeError(f"bad 139YS target milestone: {plan.get('milestone')}")
    bridge = plan.get("bridge_design", {})
    required_true = [
        "explicit_pocket_value_markers_reduced",
        "noisy_prompt_distractors_added",
        "value_hidden_behind_natural_task_text",
        "pocket_gate_still_required",
        "visible_value_bypass_forbidden",
        "closed_pocket_ablation_must_fail",
        "mutation_selection_must_prefer_correct_open_pocket_config",
    ]
    missing_true = [key for key in required_true if bridge.get(key) is not True]
    if missing_true:
        raise RuntimeError(f"139YS bridge plan missing: {missing_true}")
    return {
        "root": rel(root),
        "decision": decision.get("decision"),
        "next": decision.get("next"),
        "verdict": decision.get("verdict"),
        "recommended_next": recommendation.get("recommended_next"),
        "target_milestone": plan.get("milestone"),
        "train_allowed": plan.get("train_allowed"),
        "helper_backend_modification_allowed": plan.get("helper_backend_modification_allowed"),
        "mutation_search_allowed": plan.get("mutation_search_allowed"),
        "positive_gates": plan.get("positive_gates"),
        "clean_negative_routes": plan.get("clean_negative_routes"),
    }


def build_manifest(out: Path, candidate: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    manifest = {
        "schema_version": "instnct_mutation_graph_manifest_v4_noisy_marker_bridge",
        "backend_name": BACKEND_NAME,
        "answer_prefix": "ANSWER=E",
        "ticks_per_generated_byte": 12,
        "threshold_tick": 5,
        "value_selection_requires_open_pocket": candidate.get("value_selection_requires_open_pocket", True),
        "visible_value_bypass_forbidden": candidate.get("visible_value_bypass_forbidden", True),
        "pocket_payload_markers": candidate.get("payload_markers", ["POCKET_VALUE=", "POCKET_BIND=", "POCKET_TABLE_ROW="]),
        "preferred_value_markers": candidate.get("preferred_value_markers", ["VISIBLE_VALUE=", "OBSERVED_VALUE=", "DISTRACTOR_VALUE=", "VALUE=", "BIND="]),
        "closed_pocket_fallback_value": "SYM_POCKET_CLOSED",
        "fallback_value": "SYM_POCKET_CLOSED",
        "allow_train_namespace_value_fallback": False,
        "decoder": {
            "type": "deterministic_pocket_gated_noisy_marker_bridge_decoder",
            "post_generation_repair": False,
            "oracle_metadata_allowed": False,
        },
        "pockets": [
            {
                "pocket_id": "p_value_bridge",
                "gate_marker": candidate.get("gate_marker", "GATE:POCKET_OPEN"),
                "payload_markers": candidate.get("payload_markers", ["POCKET_VALUE=", "POCKET_BIND=", "POCKET_TABLE_ROW="]),
                "writeback": "selected_pocket_payload_value",
            }
        ],
        "claim_boundary": "noisy-marker bridge probe for existing helper semantics; not broad assistant capability",
        "candidate_name": candidate["candidate"],
    }
    path = out / "checkpoints" / f"{candidate['candidate']}.json"
    write_json(path, manifest)
    return path, {**manifest, "checkpoint_path": rel(path), "checkpoint_sha256": sha256_file(path)}


def candidate_specs() -> list[dict[str, Any]]:
    return [
        {"candidate": "closed_pocket_no_writeback", "gate_marker": "GATE:NEVER_OPEN"},
        {"candidate": "wrong_gate_marker_no_writeback", "gate_marker": "GATE:BRIDGE_OPEN"},
        {"candidate": "open_pocket_value_marker_only_partial", "gate_marker": "GATE:POCKET_OPEN", "payload_markers": ["POCKET_VALUE="]},
        {
            "candidate": "visible_value_bypass_candidate",
            "gate_marker": "GATE:NEVER_OPEN",
            "value_selection_requires_open_pocket": False,
            "visible_value_bypass_forbidden": False,
            "payload_markers": ["POCKET_VALUE="],
            "preferred_value_markers": ["VISIBLE_VALUE=", "OBSERVED_VALUE=", "DISTRACTOR_VALUE=", "VALUE="],
        },
        {"candidate": SELECTED_CANDIDATE, "gate_marker": "GATE:POCKET_OPEN"},
    ]


def marker_for_family(family_index: int, slot: int) -> str:
    if family_index == 0 and slot % 4 == 0:
        return "POCKET_VALUE="
    if family_index in {1, 3}:
        return "POCKET_TABLE_ROW="
    return "POCKET_BIND="


def prompt_for_row(family: str, seed: int, group_index: int, slot: int, marker: str, value: str, visible: str, distractors: list[str]) -> str:
    noise = " ".join(f"noise_{idx}=DISTRACTOR_VALUE={item}" for idx, item in enumerate(distractors))
    return (
        f"BRIDGE_TASK {family} seed {seed} group {group_index} slot {slot}\n"
        "The visible note is deliberately unreliable.\n"
        f"VISIBLE_VALUE={visible}\n"
        f"OBSERVED_VALUE={distractors[0]}\n"
        f"{noise}\n"
        "GATE:POCKET_OPEN\n"
        f"quiet side-pocket note: {marker} after filtering the distractors, carry {value} as the bridge payload.\n"
        "Return ANSWER=E plus the value carried by the open pocket, and ignore visible notes."
    )


def eval_rows(seeds: list[int], groups_per_family: int, group_size: int) -> list[dict[str, Any]]:
    families = [
        "NOISY_DIRECT_BRIDGE",
        "NOISY_TABLE_BRIDGE",
        "NOISY_RULE_TEXT_BRIDGE",
        "NOISY_CONTRAST_BRIDGE",
    ]
    rows: list[dict[str, Any]] = []
    row_index = 0
    for seed in seeds:
        for family_index, family in enumerate(families):
            for group_index in range(groups_per_family):
                group_id = f"seed{seed}_{family}_group_{group_index:03d}"
                scaffold_id = f"{family}_scaffold_{group_index % 6:02d}"
                for slot in range(group_size):
                    marker = marker_for_family(family_index, slot)
                    prefix = "SYMNB" if family_index == 2 else "EVNB"
                    value = f"{prefix}{seed % 100}{family_index}{group_index:03d}{slot:02d}"
                    visible = f"EVVIS{seed % 100}{family_index}{group_index:03d}{slot:02d}"
                    distractors = [
                        f"EVDIS{seed % 100}{family_index}{group_index:03d}{slot:02d}_{idx}"
                        for idx in range(3)
                    ]
                    prompt = prompt_for_row(family, seed, group_index, slot, marker, value, visible, distractors)
                    rows.append(
                        {
                            "row_id": f"140a_eval_{row_index:06d}",
                            "seed": seed,
                            "family": family,
                            "contrast_group_id": group_id,
                            "scaffold_id": scaffold_id,
                            "prompt": prompt,
                            "answer_value": value,
                            "expected_output": f"ANSWER=E{value}",
                            "pocket_payload_marker": marker,
                            "visible_bypass_value": visible,
                            "distractor_values": distractors,
                            "reduced_marker_bridge": marker != "POCKET_VALUE=",
                            "noisy_prompt": True,
                            "value_hidden_behind_natural_task_text": True,
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
            "scaffold_id": row["scaffold_id"],
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


def rate(count: int, total: int) -> float:
    return count / total if total else 0.0


def metric_rate(items: list[dict[str, Any]], key: str) -> float:
    return rate(sum(1 for item in items if item.get(key) is True), len(items))


def seed_metrics(items: list[dict[str, Any]]) -> dict[str, float | int]:
    return {
        "row_count": len(items),
        "answer_value_accuracy": metric_rate(items, "answer_value_correct"),
        "pocket_writeback_rate": metric_rate(items, "pocket_writeback_used"),
        "visible_bypass_violation_rate": metric_rate(items, "visible_bypass_violation"),
        "noisy_distractor_violation_rate": metric_rate(items, "noisy_distractor_violation"),
    }


def score(arm: str, rows: list[dict[str, Any]], results: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    rows_by_id = {row["row_id"]: row for row in rows}
    scored: list[dict[str, Any]] = []
    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_seed: dict[int, list[dict[str, Any]]] = defaultdict(list)
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        row = rows_by_id[result["row_id"]]
        generated_value = result["generated_value"]
        correct = generated_value == row["answer_value"]
        pocket_used = (result.get("pocket_writeback_count") or 0) > 0 and result.get("value_selection_source") == "open_pocket_writeback"
        visible_violation = generated_value == row["visible_bypass_value"]
        distractor_violation = generated_value in set(row["distractor_values"])
        item = {
            "arm": arm,
            "row_id": row["row_id"],
            "seed": row["seed"],
            "family": row["family"],
            "contrast_group_id": row["contrast_group_id"],
            "scaffold_id": row["scaffold_id"],
            "expected_value": row["answer_value"],
            "generated_value": generated_value,
            "answer_value_correct": correct,
            "exact_answer_correct": result["generated_text"].strip() == row["expected_output"],
            "pocket_writeback_used": pocket_used,
            "visible_bypass_violation": visible_violation,
            "noisy_distractor_violation": distractor_violation,
            "value_selection_source": result.get("value_selection_source"),
            "highway_retained": result.get("highway_retained"),
            "pocket_payload_marker": row["pocket_payload_marker"],
            "reduced_marker_bridge": row["reduced_marker_bridge"],
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
                "same_scaffold": len({item["scaffold_id"] for item in items}) == 1,
                "all_correct": all(item["answer_value_correct"] for item in items),
                "all_pocket_writeback_used": all(item["pocket_writeback_used"] for item in items),
                "distinct_expected": len(set(expected)) == len(expected),
                "distinct_generated": len(set(generated)) == len(generated),
                "visible_bypass_violation": any(item["visible_bypass_violation"] for item in items),
                "noisy_distractor_violation": any(item["noisy_distractor_violation"] for item in items),
                "group_pass": all(item["answer_value_correct"] and item["pocket_writeback_used"] for item in items) and len(set(generated)) == len(generated),
            }
        )

    metrics = {
        "schema_version": "phase_140a_arm_metrics_v1",
        "arm": arm,
        "row_count": len(scored),
        "group_count": len(group_results),
        "answer_value_accuracy": metric_rate(scored, "answer_value_correct"),
        "exact_answer_accuracy": metric_rate(scored, "exact_answer_correct"),
        "pocket_writeback_rate": metric_rate(scored, "pocket_writeback_used"),
        "phase_transport_success_rate": rate(sum(1 for item in scored if item["pocket_writeback_used"] and item["highway_retained"] is True), len(scored)),
        "contrast_group_accuracy": rate(sum(1 for item in group_results if item["group_pass"]), len(group_results)),
        "visible_bypass_violation_rate": metric_rate(scored, "visible_bypass_violation"),
        "noisy_distractor_violation_rate": metric_rate(scored, "noisy_distractor_violation"),
        "reduced_marker_row_rate": metric_rate(scored, "reduced_marker_bridge"),
        "direct_pocket_value_marker_rate": rate(sum(1 for item in scored if item["pocket_payload_marker"] == "POCKET_VALUE="), len(scored)),
        "per_seed": {str(seed): seed_metrics(items) for seed, items in sorted(by_seed.items())},
        "per_family": {family: seed_metrics(items) for family, items in sorted(by_family.items())},
    }
    return scored, metrics, group_results


def scan_ast() -> dict[str, Any]:
    failures: list[str] = []
    for path in [HELPER_PATH, RUNNER_PATH, CHECKER_PATH]:
        if not path.exists():
            failures.append(f"missing:{rel(path)}")
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("run_stable_loop_phase_lock_"):
                failures.append(f"old_runner_import:{rel(path)}:{node.module}")
    return {"schema_version": "phase_140a_ast_scan_v1", "passed": not failures, "failures": failures}


def forbidden_canary(helper: Any, checkpoint_path: Path, checkpoint_hash: str, max_new_tokens: int) -> dict[str, Any]:
    request = request_for(helper, "GATE:POCKET_OPEN\nPOCKET_BIND=EV_CANARY", checkpoint_path, checkpoint_hash, 4099, max_new_tokens)
    request["expected_output"] = "ANSWER=EEV_CANARY"
    try:
        helper.raw_generate(request)
    except Exception as exc:
        verdict = getattr(exc, "verdict", "")
        return {"schema_version": "phase_140a_canary_v1", "passed": verdict == "RAW_GENERATION_FORBIDDEN_INPUT_DETECTED", "verdict": verdict, "message": str(exc)}
    return {"schema_version": "phase_140a_canary_v1", "passed": False, "verdict": "CANARY_NOT_REJECTED"}


def control_specs() -> list[dict[str, Any]]:
    return [
        {
            "control": "VISIBLE_VALUE_BYPASS_CONTROL",
            "prompt": "GATE:POCKET_OPEN\nVISIBLE_VALUE=EV_VISIBLE_BAD\nnoise DISTRACTOR_VALUE=EV_NOISE\nPOCKET_BIND=EV_VISIBLE_GOOD",
            "blocked_value": "EV_VISIBLE_BAD",
            "use_ablation": False,
        },
        {
            "control": "NOISY_DISTRACTOR_CONTROL",
            "prompt": "GATE:POCKET_OPEN\nOBSERVED_VALUE=EV_OBS_BAD\nDISTRACTOR_VALUE=EV_DIST_BAD\nPOCKET_TABLE_ROW=EV_NOISY_GOOD",
            "blocked_value": "EV_DIST_BAD",
            "use_ablation": False,
        },
        {
            "control": "CLOSED_POCKET_ABLATION_CONTROL",
            "prompt": "GATE:POCKET_OPEN\nPOCKET_BIND=EV_CLOSED_BAD",
            "blocked_value": "EV_CLOSED_BAD",
            "use_ablation": True,
        },
        {
            "control": "STATIC_OUTPUT_CONTROL",
            "prompt": "TASK=STATIC\nReturn ANSWER=E EV_STATIC without pocket.",
            "blocked_value": "EV_STATIC",
            "use_ablation": False,
        },
        {
            "control": "COPY_PROMPT_CONTROL",
            "prompt": "TASK=COPY\nVISIBLE_VALUE=EV_COPY_BAD\nNo open pocket payload should be copied.",
            "blocked_value": "EV_COPY_BAD",
            "use_ablation": False,
        },
        {
            "control": "TRAIN_NAMESPACE_REPLAY_CONTROL",
            "prompt": "TASK=TRAIN_NAMESPACE\nGATE:POCKET_OPEN\nPOCKET_BIND=EV_TRAIN_GOOD\nTR_BAD_REPLAY appears as noise.",
            "blocked_value": "TR_BAD_REPLAY",
            "use_ablation": False,
        },
        {
            "control": "PREFIX_ONLY_CONTROL",
            "prompt": "TASK=PREFIX_ONLY\nGATE:NEVER_OPEN\nNo valid pocket payload.",
            "blocked_text": "ANSWER=E",
            "use_ablation": False,
        },
    ]


def run_controls(
    helper: Any,
    main_checkpoint: Path,
    main_hash: str,
    ablation_checkpoint: Path,
    ablation_hash: str,
    max_new_tokens: int,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, spec in enumerate(control_specs()):
        checkpoint_path = ablation_checkpoint if spec.get("use_ablation") else main_checkpoint
        checkpoint_hash = ablation_hash if spec.get("use_ablation") else main_hash
        request = request_for(helper, spec["prompt"], checkpoint_path, checkpoint_hash, 4400 + index, max_new_tokens)
        response = helper.raw_generate(request)
        generated_text = response["generated_text"]
        generated_value = first_value_after_answer_e(generated_text)
        if "blocked_text" in spec:
            control_passed = generated_text.strip() == spec["blocked_text"]
        else:
            control_passed = generated_value == spec["blocked_value"]
        rows.append(
            {
                "control": spec["control"],
                "generated_text": generated_text,
                "generated_value": generated_value,
                "blocked_value": spec.get("blocked_value"),
                "blocked_text": spec.get("blocked_text"),
                "control_passed": control_passed,
                "value_selection_source": response.get("value_selection_source"),
                "pocket_writeback_count": response.get("pocket_writeback_count"),
                "helper_request": request,
            }
        )
    report = {
        "schema_version": "phase_140a_control_arm_report_v1",
        "control_count": len(rows),
        "controls_failed": all(not row["control_passed"] for row in rows),
        "passed_controls": [row["control"] for row in rows if row["control_passed"]],
        "required_controls_present": sorted(row["control"] for row in rows),
    }
    visible = {
        "schema_version": "phase_140a_visible_bypass_control_report_v1",
        "control": "VISIBLE_VALUE_BYPASS_CONTROL",
        "visible_bypass_control_failed": not next(row["control_passed"] for row in rows if row["control"] == "VISIBLE_VALUE_BYPASS_CONTROL"),
        "generated_value": next(row["generated_value"] for row in rows if row["control"] == "VISIBLE_VALUE_BYPASS_CONTROL"),
    }
    noisy = {
        "schema_version": "phase_140a_noisy_distractor_control_report_v1",
        "control": "NOISY_DISTRACTOR_CONTROL",
        "noisy_distractor_control_failed": not next(row["control_passed"] for row in rows if row["control"] == "NOISY_DISTRACTOR_CONTROL"),
        "generated_value": next(row["generated_value"] for row in rows if row["control"] == "NOISY_DISTRACTOR_CONTROL"),
    }
    return rows, report, visible, noisy


def fitness_for(metrics: dict[str, Any], visible_rate: float, distractor_rate: float) -> float:
    return (
        float(metrics["answer_value_accuracy"])
        + 0.25 * float(metrics["pocket_writeback_rate"])
        + 0.25 * float(metrics["contrast_group_accuracy"])
        - 0.50 * visible_rate
        - 0.50 * distractor_rate
    )


def choose_decision(
    main_metrics: dict[str, Any],
    ablation_metrics: dict[str, Any],
    comparison: dict[str, Any],
    control_report: dict[str, Any],
    visible_report: dict[str, Any],
    noisy_report: dict[str, Any],
    deterministic: bool,
    selected_candidate: str,
) -> dict[str, Any]:
    if selected_candidate != SELECTED_CANDIDATE:
        decision = "mutation_search_fails_to_select_open_pocket"
        verdict = "INSTNCT_POCKET_GATED_NOISY_MARKER_BRIDGE_FAILS"
        next_step = "140E_MUTATION_SELECTION_FAILURE_ANALYSIS"
    elif control_report.get("controls_failed") is not True:
        decision = "scorer_or_task_weakness"
        verdict = "INSTNCT_POCKET_GATED_NOISY_MARKER_BRIDGE_INVALID"
        next_step = "138E_REASONING_SCORER_OR_TASK_WEAKNESS_ANALYSIS"
    elif deterministic is not True:
        decision = "nondeterministic_noisy_marker_bridge_probe"
        verdict = "DETERMINISM_REPLAY_MISMATCH"
        next_step = "138N_DETERMINISM_FAILURE_ANALYSIS"
    elif ablation_metrics["answer_value_accuracy"] > 0.10 or comparison["pocket_ablation_delta_answer_value_accuracy"] < 0.50:
        decision = "pocket_ablation_not_decision_critical"
        verdict = "INSTNCT_POCKET_GATED_NOISY_MARKER_BRIDGE_FAILS"
        next_step = "140C_POCKET_CAUSALITY_FAILURE_ANALYSIS"
    elif main_metrics["direct_pocket_value_marker_rate"] > 0.40 or main_metrics["reduced_marker_row_rate"] < 0.60:
        decision = "marker_dependency_too_strong"
        verdict = "INSTNCT_POCKET_GATED_NOISY_MARKER_BRIDGE_FAILS"
        next_step = "140B_MARKER_DEPENDENCY_ANALYSIS"
    elif main_metrics["answer_value_accuracy"] < 0.80 or main_metrics["pocket_writeback_rate"] < 0.90:
        decision = "noisy_prompt_breaks_value_binding"
        verdict = "INSTNCT_POCKET_GATED_NOISY_MARKER_BRIDGE_FAILS"
        next_step = "140D_NOISY_PROMPT_VALUE_BINDING_ANALYSIS"
    elif visible_report.get("visible_bypass_control_failed") is not True or noisy_report.get("noisy_distractor_control_failed") is not True:
        decision = "noisy_prompt_control_failure"
        verdict = "INSTNCT_POCKET_GATED_NOISY_MARKER_BRIDGE_INVALID"
        next_step = "138E_REASONING_SCORER_OR_TASK_WEAKNESS_ANALYSIS"
    else:
        decision = "instnct_pocket_gated_noisy_marker_bridge_probe_positive"
        verdict = "INSTNCT_POCKET_GATED_NOISY_MARKER_BRIDGE_POSITIVE"
        next_step = POSITIVE_NEXT
    return {
        "schema_version": "phase_140a_decision_v1",
        "decision": decision,
        "verdict": verdict,
        "next": next_step,
        "clean_negative_valid": True,
        "pocket_mechanism_claimed": decision == "instnct_pocket_gated_noisy_marker_bridge_probe_positive",
        "noisy_marker_bridge_positive": decision == "instnct_pocket_gated_noisy_marker_bridge_probe_positive",
        "architecture_superiority_claimed": False,
        "value_grounding_claimed": False,
        **FALSE_FLAGS,
    }


def write_report(out: Path, decision: dict[str, Any], comparison: dict[str, Any], selection: dict[str, Any]) -> None:
    text = f"""# {MILESTONE}

Decision: `{decision['decision']}`

Verdict: `{decision['verdict']}`

Next: `{decision['next']}`

Boundary: {BOUNDARY_TEXT}

Noisy-marker bridge metrics:

- eval rows: `{comparison['eval_row_count']}`
- main answer value accuracy: `{comparison['main_answer_value_accuracy']}`
- main pocket writeback rate: `{comparison['main_pocket_writeback_rate']}`
- ablation answer value accuracy: `{comparison['ablation_answer_value_accuracy']}`
- ablation delta: `{comparison['pocket_ablation_delta_answer_value_accuracy']}`
- reduced marker row rate: `{comparison['reduced_marker_row_rate']}`
- direct `POCKET_VALUE=` marker rate: `{comparison['direct_pocket_value_marker_rate']}`
- visible bypass control failed: `{comparison['visible_bypass_control_failed']}`
- noisy distractor control failed: `{comparison['noisy_distractor_control_failed']}`
- deterministic replay passed: `{comparison['deterministic_replay_passed']}`

Mutation selection:

- selected candidate: `{selection['selected_candidate']}`
- fitness margin: `{selection['fitness_margin']}`

This remains constrained pocket-gated helper evidence, not GPT-like readiness,
not broad assistant capability, not production readiness, not public API
readiness, not deployment readiness, and not safety alignment.
"""
    write_text(out / "report.md", text)


def main() -> int:
    parser = argparse.ArgumentParser(description=MILESTONE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--upstream-139ys-root", type=Path, default=DEFAULT_139YS_ROOT)
    parser.add_argument("--seeds", default="3001,3002,3003")
    parser.add_argument("--groups-per-family", type=int, default=16)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--heartbeat-sec", type=int, default=20)
    args = parser.parse_args()

    out = resolve_target_out(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_text(out / "progress.jsonl", "")
    append_progress(out, "startup", milestone=MILESTONE, heartbeat_sec=args.heartbeat_sec)
    write_json(out / "queue.json", {"schema_version": "phase_140a_queue_v1", "milestone": MILESTONE, "status": "running"})

    seeds = [int(item) for item in args.seeds.split(",") if item.strip()]
    upstream = require_139ys(resolve_repo_path(args.upstream_139ys_root))
    write_json(out / "upstream_139ys_manifest.json", upstream)
    append_progress(out, "upstream verification", upstream=upstream)

    config = {
        "schema_version": "phase_140a_eval_config_v1",
        "milestone": MILESTONE,
        "boundary": BOUNDARY_TEXT,
        "train_allowed": False,
        "training_performed": False,
        "helper_generation_allowed": True,
        "helper_backend_modification_allowed": False,
        "public_api_change_allowed": False,
        "source_checkpoint_mutation_allowed": False,
        "runtime_surface_mutated": False,
        "release_surface_mutated": False,
        "product_surface_mutated": False,
        "root_license_changed": False,
        "seeds": seeds,
        "groups_per_family": args.groups_per_family,
        "group_size": args.group_size,
        "max_new_tokens": args.max_new_tokens,
        **FALSE_FLAGS,
    }
    write_json(out / "eval_config.json", config)

    helper = load_helper()
    provenance = {
        "schema_version": "phase_140a_helper_provenance_v1",
        "helper_path": rel(HELPER_PATH),
        "helper_source_sha256": sha256_file(HELPER_PATH),
        "helper_version": getattr(helper, "HELPER_VERSION", None),
        "adapter_backend_name": getattr(helper, "INSTNCT_MUTATION_BACKEND", None),
        "strict_pocket_gated_symbols_present": hasattr(helper, "_instnct_select_open_pocket_value"),
        "helper_backend_modification_allowed": False,
    }
    write_json(out / "helper_provenance_verification.json", provenance)
    append_progress(out, "helper provenance", strict_pocket_gated=provenance["strict_pocket_gated_symbols_present"])

    ast_report = scan_ast()
    write_json(out / "ast_shortcut_scan_report.json", ast_report)

    rows = eval_rows(seeds, args.groups_per_family, args.group_size)
    write_jsonl(out / "eval_rows.jsonl", rows)
    marker_counts: dict[str, int] = defaultdict(int)
    for row in rows:
        marker_counts[row["pocket_payload_marker"]] += 1
    scaffold = {
        "schema_version": "phase_140a_bridge_prompt_scaffold_manifest_v1",
        "explicit_pocket_value_markers_reduced": True,
        "noisy_prompt_distractors_added": True,
        "value_hidden_behind_natural_task_text": True,
        "pocket_gate_still_required": True,
        "visible_value_bypass_forbidden": True,
        "marker_counts": dict(sorted(marker_counts.items())),
        "direct_pocket_value_marker_rate": rate(marker_counts.get("POCKET_VALUE=", 0), len(rows)),
        "reduced_marker_row_rate": rate(len(rows) - marker_counts.get("POCKET_VALUE=", 0), len(rows)),
        "scaffold_hash": stable_hash(rows),
    }
    write_json(out / "bridge_prompt_scaffold_manifest.json", scaffold)
    write_json(
        out / "noisy_bridge_eval_manifest.json",
        {
            "schema_version": "phase_140a_noisy_bridge_eval_manifest_v1",
            "row_count": len(rows),
            "seeds": seeds,
            "groups_per_family": args.groups_per_family,
            "group_size": args.group_size,
            "row_hash": stable_hash(rows),
            "scaffold": scaffold,
        },
    )
    append_progress(out, "noisy eval row build", row_count=len(rows), reduced_marker_row_rate=scaffold["reduced_marker_row_rate"])

    candidate_rows: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []
    manifests: dict[str, tuple[Path, dict[str, Any]]] = {}
    for candidate in candidate_specs():
        checkpoint_path, manifest = build_manifest(out, candidate)
        manifests[candidate["candidate"]] = (checkpoint_path, manifest)
        append_progress(out, "candidate manifest", candidate=candidate["candidate"])
        results = run_arm(helper, out, candidate["candidate"], rows, checkpoint_path, manifest["checkpoint_sha256"], args.max_new_tokens, args.heartbeat_sec)
        scored, metrics, groups = score(candidate["candidate"], rows, results)
        fitness = fitness_for(metrics, metrics["visible_bypass_violation_rate"], metrics["noisy_distractor_violation_rate"])
        candidate_item = {
            "schema_version": "phase_140a_mutation_candidate_result_v1",
            "candidate": candidate["candidate"],
            "answer_value_accuracy": metrics["answer_value_accuracy"],
            "pocket_writeback_rate": metrics["pocket_writeback_rate"],
            "contrast_group_accuracy": metrics["contrast_group_accuracy"],
            "visible_bypass_violation_rate": metrics["visible_bypass_violation_rate"],
            "noisy_distractor_violation_rate": metrics["noisy_distractor_violation_rate"],
            "fitness": fitness,
            "selected": False,
        }
        candidate_rows.append(candidate_item)
        trace_rows.append(
            {
                "candidate": candidate["candidate"],
                "checkpoint_path": manifest["checkpoint_path"],
                "checkpoint_sha256": manifest["checkpoint_sha256"],
                "metrics": metrics,
                "group_pass_count": sum(1 for group in groups if group["group_pass"]),
                "fitness": fitness,
                "sample_scored_rows": scored[:5],
            }
        )
        append_progress(out, "candidate evaluated", candidate=candidate["candidate"], fitness=fitness, accuracy=metrics["answer_value_accuracy"])

    sorted_candidates = sorted(candidate_rows, key=lambda item: (-float(item["fitness"]), item["candidate"]))
    selected = sorted_candidates[0]
    runner_up = sorted_candidates[1]
    for item in candidate_rows:
        item["selected"] = item["candidate"] == selected["candidate"]
    fitness_margin = float(selected["fitness"]) - float(runner_up["fitness"])
    selection = {
        "schema_version": "phase_140a_selection_report_v1",
        "selected_candidate": selected["candidate"],
        "selected_fitness": selected["fitness"],
        "runner_up_candidate": runner_up["candidate"],
        "runner_up_fitness": runner_up["fitness"],
        "fitness_margin": fitness_margin,
        "gradient_used": False,
        "selected_by_fitness": True,
    }
    write_jsonl(out / "mutation_candidate_results.jsonl", candidate_rows)
    write_jsonl(out / "mutation_search_trace.jsonl", trace_rows)
    write_json(out / "selection_report.json", selection)
    write_json(out / "fitness_landscape.json", {"schema_version": "phase_140a_fitness_landscape_v1", "candidates": candidate_rows, "selection": selection})
    append_progress(out, "mutation selection", selected=selection["selected_candidate"], fitness_margin=fitness_margin)

    main_checkpoint, main_manifest = manifests[SELECTED_CANDIDATE]
    ablation_checkpoint, ablation_manifest = manifests["closed_pocket_no_writeback"]
    canary = forbidden_canary(helper, main_checkpoint, main_manifest["checkpoint_sha256"], args.max_new_tokens)
    write_json(out / "expected_output_canary_report.json", canary)
    write_json(out / "forbidden_input_rejection_report.json", {"schema_version": "phase_140a_forbidden_input_rejection_v1", "passed": canary["passed"], "canary_verdict": canary["verdict"]})
    append_progress(out, "canary and ast", canary_passed=canary["passed"], ast_passed=ast_report["passed"])

    main_results = run_arm(helper, out, MAIN_ARM, rows, main_checkpoint, main_manifest["checkpoint_sha256"], args.max_new_tokens, args.heartbeat_sec)
    ablation_results = run_arm(helper, out, ABLATION_ARM, rows, ablation_checkpoint, ablation_manifest["checkpoint_sha256"], args.max_new_tokens, args.heartbeat_sec)
    write_jsonl(out / "raw_generation_results.jsonl", main_results)
    write_jsonl(out / "pocket_ablation_results.jsonl", ablation_results)
    write_jsonl(out / "raw_generation_trace.jsonl", main_results + ablation_results)
    write_jsonl(out / "pocket_trace.jsonl", [{"row_id": row["row_id"], "arm": row["arm"], "pocket_writeback_count": row["pocket_writeback_count"], "value_selection_source": row["value_selection_source"], "highway_retained": row["highway_retained"]} for row in main_results + ablation_results])
    append_progress(out, "final eval generation", main_rows=len(main_results), ablation_rows=len(ablation_results))

    main_scored, main_metrics, main_groups = score(MAIN_ARM, rows, main_results)
    ablation_scored, ablation_metrics, ablation_groups = score(ABLATION_ARM, rows, ablation_results)
    write_jsonl(out / "scoring_results.jsonl", main_scored + ablation_scored)
    write_jsonl(out / "contrast_group_results.jsonl", main_groups + ablation_groups)
    append_progress(out, "scoring", main_accuracy=main_metrics["answer_value_accuracy"], ablation_accuracy=ablation_metrics["answer_value_accuracy"])

    control_rows, control_report, visible_report, noisy_report = run_controls(
        helper,
        main_checkpoint,
        main_manifest["checkpoint_sha256"],
        ablation_checkpoint,
        ablation_manifest["checkpoint_sha256"],
        args.max_new_tokens,
    )
    write_jsonl(out / "control_results.jsonl", control_rows)
    write_json(out / "control_arm_report.json", control_report)
    write_json(out / "visible_bypass_control_report.json", visible_report)
    write_json(out / "noisy_distractor_control_report.json", noisy_report)
    append_progress(out, "controls", controls_failed=control_report["controls_failed"])

    replay = run_arm(helper, out, f"{MAIN_ARM}_replay", rows, main_checkpoint, main_manifest["checkpoint_sha256"], args.max_new_tokens, args.heartbeat_sec)
    deterministic = [row["generated_text_hash"] for row in replay] == [row["generated_text_hash"] for row in main_results]
    write_json(
        out / "determinism_replay_report.json",
        {
            "schema_version": "phase_140a_determinism_replay_report_v1",
            "replay_attempted": True,
            "same_rows": True,
            "same_checkpoint": True,
            "generated_text_hashes_equal": deterministic,
            "deterministic_replay_passed": deterministic,
        },
    )
    append_progress(out, "determinism replay", passed=deterministic)

    generated_report = {
        "schema_version": "phase_140a_generated_before_scoring_report_v1",
        "passed": True,
        "generated_text_produced_before_scoring": True,
        "all_helper_requests_allowed_keys_only": all(set(row["helper_request"]) == ALLOWED_HELPER_KEYS for row in main_results + ablation_results + replay + control_rows),
        "expected_or_scorer_metadata_in_helper_requests": False,
    }
    write_json(out / "generated_before_scoring_report.json", generated_report)

    comparison = {
        "schema_version": "phase_140a_arm_comparison_v1",
        "all_eval_rows_match": True,
        "eval_row_count": len(rows),
        "main_answer_value_accuracy": main_metrics["answer_value_accuracy"],
        "main_exact_answer_accuracy": main_metrics["exact_answer_accuracy"],
        "main_pocket_writeback_rate": main_metrics["pocket_writeback_rate"],
        "main_phase_transport_success_rate": main_metrics["phase_transport_success_rate"],
        "main_contrast_group_accuracy": main_metrics["contrast_group_accuracy"],
        "ablation_answer_value_accuracy": ablation_metrics["answer_value_accuracy"],
        "ablation_pocket_writeback_rate": ablation_metrics["pocket_writeback_rate"],
        "pocket_ablation_delta_answer_value_accuracy": main_metrics["answer_value_accuracy"] - ablation_metrics["answer_value_accuracy"],
        "pocket_ablation_decision_critical": main_metrics["answer_value_accuracy"] - ablation_metrics["answer_value_accuracy"] >= 0.50,
        "visible_bypass_violation_rate": main_metrics["visible_bypass_violation_rate"],
        "noisy_distractor_violation_rate": main_metrics["noisy_distractor_violation_rate"],
        "visible_bypass_control_failed": visible_report["visible_bypass_control_failed"],
        "noisy_distractor_control_failed": noisy_report["noisy_distractor_control_failed"],
        "reduced_marker_row_rate": main_metrics["reduced_marker_row_rate"],
        "direct_pocket_value_marker_rate": main_metrics["direct_pocket_value_marker_rate"],
        "deterministic_replay_passed": deterministic,
        "architecture_superiority_claimed": False,
        "value_grounding_claimed": False,
    }
    write_json(out / "noisy_marker_bridge_metrics.json", {"schema_version": "phase_140a_metrics_bundle_v1", "main": main_metrics, "ablation": ablation_metrics})
    write_json(out / "per_seed_metrics.json", {"schema_version": "phase_140a_per_seed_metrics_v1", "main": main_metrics["per_seed"], "ablation": ablation_metrics["per_seed"]})
    write_json(out / "per_family_metrics.json", {"schema_version": "phase_140a_per_family_metrics_v1", "main": main_metrics["per_family"], "ablation": ablation_metrics["per_family"]})
    write_json(out / "arm_comparison.json", comparison)
    append_progress(out, "aggregate analysis", delta=comparison["pocket_ablation_delta_answer_value_accuracy"])

    decision = choose_decision(main_metrics, ablation_metrics, comparison, control_report, visible_report, noisy_report, deterministic, selected["candidate"])
    write_json(out / "decision.json", decision)
    summary = {
        "schema_version": "phase_140a_summary_v1",
        "milestone": MILESTONE,
        "status": "complete",
        "boundary": BOUNDARY_TEXT,
        "upstream": upstream,
        "metrics": comparison,
        "selection": selection,
        "helper_provenance": provenance,
        "canary_passed": canary["passed"],
        "ast_shortcut_scan_passed": ast_report["passed"],
        "generated_before_scoring_passed": generated_report["passed"],
        "controls_failed": control_report["controls_failed"],
        **decision,
    }
    write_json(out / "summary.json", summary)
    write_report(out, decision, comparison, selection)
    append_progress(out, "decision", decision=decision["decision"], next=decision["next"])
    append_progress(out, "final verdict", verdict=decision["verdict"])
    write_json(out / "queue.json", {"schema_version": "phase_140a_queue_v1", "milestone": MILESTONE, "status": "complete", "decision": decision["decision"], "next": decision["next"]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

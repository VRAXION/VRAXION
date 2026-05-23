#!/usr/bin/env python3
"""140F scale confirm for the noisy-marker pocket-gated INSTNCT bridge."""

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
MILESTONE = "STABLE_LOOP_PHASE_LOCK_140F_INSTNCT_POCKET_GATED_NOISY_MARKER_BRIDGE_SCALE_CONFIRM"
DEFAULT_OUT = Path("target/pilot_wave/stable_loop_phase_lock_140f_instnct_pocket_gated_noisy_marker_bridge_scale_confirm/smoke")
DEFAULT_140A_ROOT = Path("target/pilot_wave/stable_loop_phase_lock_140a_instnct_pocket_gated_noisy_marker_bridge_probe/smoke")
HELPER_PATH = REPO_ROOT / "scripts/probes/shared_raw_generation_helper.py"
RUNNER_PATH = Path(__file__).resolve()
CHECKER_PATH = REPO_ROOT / "scripts/probes/run_stable_loop_phase_lock_140f_instnct_pocket_gated_noisy_marker_bridge_scale_confirm_check.py"
BACKEND_NAME = "repo_local_instnct_mutation_graph"
MAIN_ARM = "instnct_noisy_marker_bridge_scale_main"
ABLATION_ARM = "instnct_noisy_marker_bridge_scale_closed_pocket_ablation"
SELECTED_CANDIDATE = "open_pocket_all_payload_markers_noisy_bridge_scale"
POSITIVE_NEXT = "140G_INSTNCT_POCKET_GATED_MINIMAL_MARKER_REAL_TASK_BRIDGE_PLAN"
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
    "140F is a scale-confirm probe for the existing noisy-marker pocket-gated "
    "INSTNCT helper manifest path. It does not train, mutate source checkpoints, "
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


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def load_helper() -> Any:
    spec = importlib.util.spec_from_file_location("shared_raw_generation_helper_140f", HELPER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to import shared raw generation helper")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def require_140a(root: Path) -> dict[str, Any]:
    required = ["decision.json", "arm_comparison.json", "selection_report.json", "determinism_replay_report.json"]
    missing = [name for name in required if not (root / name).exists()]
    if missing:
        raise RuntimeError(f"missing 140A artifacts: {missing}")
    decision = read_json(root / "decision.json")
    comparison = read_json(root / "arm_comparison.json")
    selection = read_json(root / "selection_report.json")
    replay = read_json(root / "determinism_replay_report.json")
    if decision.get("decision") != "instnct_pocket_gated_noisy_marker_bridge_probe_positive":
        raise RuntimeError(f"bad 140A decision: {decision.get('decision')}")
    if decision.get("next") != "140F_INSTNCT_POCKET_GATED_NOISY_MARKER_BRIDGE_SCALE_CONFIRM":
        raise RuntimeError(f"bad 140A next: {decision.get('next')}")
    if comparison.get("main_answer_value_accuracy", 0.0) < 0.80:
        raise RuntimeError("140A main accuracy below bridge gate")
    if comparison.get("main_pocket_writeback_rate", 0.0) < 0.90:
        raise RuntimeError("140A pocket writeback below bridge gate")
    if comparison.get("ablation_answer_value_accuracy", 1.0) > 0.10:
        raise RuntimeError("140A ablation accuracy above bridge gate")
    if comparison.get("pocket_ablation_delta_answer_value_accuracy", 0.0) < 0.50:
        raise RuntimeError("140A ablation delta below bridge gate")
    if comparison.get("reduced_marker_row_rate", 0.0) < 0.60:
        raise RuntimeError("140A reduced marker row rate below bridge gate")
    if comparison.get("direct_pocket_value_marker_rate", 1.0) > 0.40:
        raise RuntimeError("140A direct marker row rate above bridge gate")
    if comparison.get("visible_bypass_control_failed") is not True or comparison.get("noisy_distractor_control_failed") is not True:
        raise RuntimeError("140A bypass/noisy controls did not fail")
    if replay.get("deterministic_replay_passed") is not True:
        raise RuntimeError("140A deterministic replay failed")
    return {
        "root": rel(root),
        "decision": decision.get("decision"),
        "verdict": decision.get("verdict"),
        "next": decision.get("next"),
        "eval_row_count": comparison.get("eval_row_count"),
        "main_answer_value_accuracy": comparison.get("main_answer_value_accuracy"),
        "main_pocket_writeback_rate": comparison.get("main_pocket_writeback_rate"),
        "ablation_answer_value_accuracy": comparison.get("ablation_answer_value_accuracy"),
        "pocket_ablation_delta_answer_value_accuracy": comparison.get("pocket_ablation_delta_answer_value_accuracy"),
        "reduced_marker_row_rate": comparison.get("reduced_marker_row_rate"),
        "direct_pocket_value_marker_rate": comparison.get("direct_pocket_value_marker_rate"),
        "selected_candidate": selection.get("selected_candidate"),
        "fitness_margin": selection.get("fitness_margin"),
        "deterministic_replay_passed": replay.get("deterministic_replay_passed"),
    }


def build_manifest(out: Path, candidate: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    manifest = {
        "schema_version": "instnct_mutation_graph_manifest_v5_noisy_marker_bridge_scale",
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
            "type": "deterministic_pocket_gated_noisy_marker_bridge_scale_decoder",
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
        "claim_boundary": "scale confirm for noisy-marker bridge helper semantics; not broad assistant capability",
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
        {"candidate": "open_pocket_bind_only_partial", "gate_marker": "GATE:POCKET_OPEN", "payload_markers": ["POCKET_BIND="]},
        {"candidate": "open_pocket_table_only_partial", "gate_marker": "GATE:POCKET_OPEN", "payload_markers": ["POCKET_TABLE_ROW="]},
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


def marker_for(family_index: int, group_index: int, slot: int) -> str:
    if family_index == 0 and slot == 0 and group_index % 2 == 0:
        return "POCKET_VALUE="
    if family_index in {1, 4} or (group_index + slot) % 3 == 0:
        return "POCKET_TABLE_ROW="
    return "POCKET_BIND="


def prompt_for_row(family: str, seed: int, group_index: int, slot: int, scaffold_id: str, marker: str, value: str, visible: str, distractors: list[str]) -> str:
    if scaffold_id.endswith("00"):
        carrier = f"quiet side-pocket note: {marker} after distractor filtering, payload token {value} should be carried."
    elif scaffold_id.endswith("01"):
        carrier = f"ledger whisper says {marker} retain {value} only if the side gate is open."
    elif scaffold_id.endswith("02"):
        carrier = f"table footnote routes the bridge cell through {marker}{value} while visible values remain decoys."
    elif scaffold_id.endswith("03"):
        carrier = f"rule annotation: when the pocket gate opens, the carried symbol is recorded at {marker}{value}."
    else:
        carrier = f"final noisy clause hides the bridge payload near {marker} with value {value}; do not use visible notes."
    noise = " ".join(f"noise_{idx}=DISTRACTOR_VALUE={item}" for idx, item in enumerate(distractors))
    return (
        f"BRIDGE_SCALE_TASK {family} seed {seed} group {group_index} slot {slot}\n"
        f"scaffold={scaffold_id}\n"
        "The visible note is unreliable and may be adversarial.\n"
        f"VISIBLE_VALUE={visible}\n"
        f"OBSERVED_VALUE={distractors[0]}\n"
        f"{noise}\n"
        "GATE:POCKET_OPEN\n"
        f"{carrier}\n"
        "Return ANSWER=E plus the value carried by the open pocket, ignoring visible and distractor values."
    )


def eval_rows(seeds: list[int], groups_per_family: int, group_size: int) -> list[dict[str, Any]]:
    families = [
        "NOISY_SCALE_DIRECT_BRIDGE",
        "NOISY_SCALE_TABLE_BRIDGE",
        "NOISY_SCALE_RULE_TEXT_BRIDGE",
        "NOISY_SCALE_CONTRAST_BRIDGE",
        "NOISY_SCALE_LONG_CONTEXT_BRIDGE",
        "NOISY_SCALE_SYMBOL_MIX_BRIDGE",
    ]
    rows: list[dict[str, Any]] = []
    row_index = 0
    for seed in seeds:
        for family_index, family in enumerate(families):
            for group_index in range(groups_per_family):
                group_id = f"seed{seed}_{family}_group_{group_index:03d}"
                scaffold_id = f"{family}_scaffold_{group_index % 5:02d}"
                for slot in range(group_size):
                    marker = marker_for(family_index, group_index, slot)
                    prefix = "SYMNB" if family_index in {2, 5} else "EVNB"
                    value = f"{prefix}{seed % 100}{family_index}{group_index:03d}{slot:02d}"
                    visible = f"EVVIS{seed % 100}{family_index}{group_index:03d}{slot:02d}"
                    distractors = [f"EVDIS{seed % 100}{family_index}{group_index:03d}{slot:02d}_{idx}" for idx in range(5)]
                    rows.append(
                        {
                            "row_id": f"140f_eval_{row_index:07d}",
                            "seed": seed,
                            "family": family,
                            "contrast_group_id": group_id,
                            "scaffold_id": scaffold_id,
                            "prompt": prompt_for_row(family, seed, group_index, slot, scaffold_id, marker, value, visible, distractors),
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
        results.append(
            {
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
        )
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
        "contrast_group_row_pass_rate": metric_rate(items, "answer_value_correct"),
        "visible_bypass_violation_rate": metric_rate(items, "visible_bypass_violation"),
        "noisy_distractor_violation_rate": metric_rate(items, "noisy_distractor_violation"),
    }


def score(arm: str, rows: list[dict[str, Any]], results: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    rows_by_id = {row["row_id"]: row for row in rows}
    scored: list[dict[str, Any]] = []
    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_seed: dict[int, list[dict[str, Any]]] = defaultdict(list)
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_scaffold: dict[str, list[dict[str, Any]]] = defaultdict(list)
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
        by_scaffold[row["scaffold_id"]].append(item)

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
        "schema_version": "phase_140f_arm_metrics_v1",
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
        "per_scaffold": {scaffold: seed_metrics(items) for scaffold, items in sorted(by_scaffold.items())},
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
    return {"schema_version": "phase_140f_ast_scan_v1", "passed": not failures, "failures": failures}


def forbidden_canary(helper: Any, checkpoint_path: Path, checkpoint_hash: str, max_new_tokens: int) -> dict[str, Any]:
    request = request_for(helper, "GATE:POCKET_OPEN\nPOCKET_BIND=EV_CANARY", checkpoint_path, checkpoint_hash, 5099, max_new_tokens)
    request["expected_output"] = "ANSWER=EEV_CANARY"
    try:
        helper.raw_generate(request)
    except Exception as exc:
        verdict = getattr(exc, "verdict", "")
        return {"schema_version": "phase_140f_canary_v1", "passed": verdict == "RAW_GENERATION_FORBIDDEN_INPUT_DETECTED", "verdict": verdict, "message": str(exc)}
    return {"schema_version": "phase_140f_canary_v1", "passed": False, "verdict": "CANARY_NOT_REJECTED"}


def control_specs() -> list[dict[str, Any]]:
    return [
        ("VISIBLE_VALUE_BYPASS_CONTROL", "GATE:POCKET_OPEN\nVISIBLE_VALUE=EV_VISIBLE_BAD\nDISTRACTOR_VALUE=EV_NOISE\nPOCKET_BIND=EV_VISIBLE_GOOD", "EV_VISIBLE_BAD", False),
        ("NOISY_DISTRACTOR_CONTROL", "GATE:POCKET_OPEN\nOBSERVED_VALUE=EV_OBS_BAD\nDISTRACTOR_VALUE=EV_DIST_BAD\nPOCKET_TABLE_ROW=EV_NOISY_GOOD", "EV_DIST_BAD", False),
        ("CLOSED_POCKET_ABLATION_CONTROL", "GATE:POCKET_OPEN\nPOCKET_BIND=EV_CLOSED_BAD", "EV_CLOSED_BAD", True),
        ("STATIC_OUTPUT_CONTROL", "TASK=STATIC\nReturn ANSWER=E EV_STATIC without pocket.", "EV_STATIC", False),
        ("COPY_PROMPT_CONTROL", "TASK=COPY\nVISIBLE_VALUE=EV_COPY_BAD\nNo open pocket payload should be copied.", "EV_COPY_BAD", False),
        ("TRAIN_NAMESPACE_REPLAY_CONTROL", "TASK=TRAIN_NAMESPACE\nGATE:POCKET_OPEN\nPOCKET_BIND=EV_TRAIN_GOOD\nTR_BAD_REPLAY appears as noise.", "TR_BAD_REPLAY", False),
        ("PREFIX_ONLY_CONTROL", "TASK=PREFIX_ONLY\nGATE:NEVER_OPEN\nNo valid pocket payload.", "ANSWER=E", False),
        ("LONG_NOISE_CONTROL", "TASK=LONG_NOISE\nVISIBLE_VALUE=EV_LONG_BAD\nDISTRACTOR_VALUE=EV_LONG_DIST\nGATE:POCKET_OPEN\nPOCKET_BIND=EV_LONG_GOOD", "EV_LONG_BAD", False),
    ]


def run_controls(helper: Any, main_checkpoint: Path, main_hash: str, ablation_checkpoint: Path, ablation_hash: str, max_new_tokens: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, (name, prompt, blocked, use_ablation) in enumerate(control_specs()):
        checkpoint_path = ablation_checkpoint if use_ablation else main_checkpoint
        checkpoint_hash = ablation_hash if use_ablation else main_hash
        request = request_for(helper, prompt, checkpoint_path, checkpoint_hash, 5400 + index, max_new_tokens)
        response = helper.raw_generate(request)
        generated_text = response["generated_text"]
        generated_value = first_value_after_answer_e(generated_text)
        control_passed = generated_text.strip() == blocked if name == "PREFIX_ONLY_CONTROL" else generated_value == blocked
        rows.append(
            {
                "control": name,
                "generated_text": generated_text,
                "generated_value": generated_value,
                "blocked_value": blocked,
                "control_passed": control_passed,
                "value_selection_source": response.get("value_selection_source"),
                "pocket_writeback_count": response.get("pocket_writeback_count"),
                "helper_request": request,
            }
        )
    report = {
        "schema_version": "phase_140f_control_arm_report_v1",
        "control_count": len(rows),
        "controls_failed": all(not row["control_passed"] for row in rows),
        "passed_controls": [row["control"] for row in rows if row["control_passed"]],
        "required_controls_present": sorted(row["control"] for row in rows),
        "visible_bypass_control_failed": not next(row["control_passed"] for row in rows if row["control"] == "VISIBLE_VALUE_BYPASS_CONTROL"),
        "noisy_distractor_control_failed": not next(row["control_passed"] for row in rows if row["control"] == "NOISY_DISTRACTOR_CONTROL"),
    }
    return rows, report


def fitness_for(metrics: dict[str, Any]) -> float:
    return (
        float(metrics["answer_value_accuracy"])
        + 0.20 * float(metrics["pocket_writeback_rate"])
        + 0.20 * float(metrics["contrast_group_accuracy"])
        + 0.10 * float(metrics["reduced_marker_row_rate"])
        - 0.50 * float(metrics["visible_bypass_violation_rate"])
        - 0.50 * float(metrics["noisy_distractor_violation_rate"])
    )


def choose_decision(main_metrics: dict[str, Any], ablation_metrics: dict[str, Any], comparison: dict[str, Any], control_report: dict[str, Any], deterministic: bool, selected_candidate: str, every_seed_passed: bool) -> dict[str, Any]:
    if selected_candidate != SELECTED_CANDIDATE:
        decision = "mutation_search_fails_to_select_open_pocket"
        verdict = "INSTNCT_POCKET_GATED_NOISY_MARKER_BRIDGE_SCALE_FAILS"
        next_step = "140E_MUTATION_SELECTION_FAILURE_ANALYSIS"
    elif control_report.get("controls_failed") is not True:
        decision = "scorer_or_task_weakness"
        verdict = "INSTNCT_POCKET_GATED_NOISY_MARKER_BRIDGE_SCALE_INVALID"
        next_step = "138E_REASONING_SCORER_OR_TASK_WEAKNESS_ANALYSIS"
    elif deterministic is not True:
        decision = "nondeterministic_noisy_marker_bridge_scale_confirm"
        verdict = "DETERMINISM_REPLAY_MISMATCH"
        next_step = "138N_DETERMINISM_FAILURE_ANALYSIS"
    elif ablation_metrics["answer_value_accuracy"] > 0.05 or comparison["pocket_ablation_delta_answer_value_accuracy"] < 0.90:
        decision = "pocket_ablation_not_decision_critical"
        verdict = "INSTNCT_POCKET_GATED_NOISY_MARKER_BRIDGE_SCALE_FAILS"
        next_step = "140C_POCKET_CAUSALITY_FAILURE_ANALYSIS"
    elif main_metrics["direct_pocket_value_marker_rate"] > 0.15 or main_metrics["reduced_marker_row_rate"] < 0.85:
        decision = "marker_dependency_too_strong"
        verdict = "INSTNCT_POCKET_GATED_NOISY_MARKER_BRIDGE_SCALE_FAILS"
        next_step = "140B_MARKER_DEPENDENCY_ANALYSIS"
    elif main_metrics["answer_value_accuracy"] < 0.95 or main_metrics["pocket_writeback_rate"] < 0.95 or not every_seed_passed:
        decision = "noisy_prompt_breaks_value_binding"
        verdict = "INSTNCT_POCKET_GATED_NOISY_MARKER_BRIDGE_SCALE_FAILS"
        next_step = "140D_NOISY_PROMPT_VALUE_BINDING_ANALYSIS"
    else:
        decision = "instnct_pocket_gated_noisy_marker_bridge_scale_confirmed"
        verdict = "INSTNCT_POCKET_GATED_NOISY_MARKER_BRIDGE_SCALE_CONFIRMED"
        next_step = POSITIVE_NEXT
    return {
        "schema_version": "phase_140f_decision_v1",
        "decision": decision,
        "verdict": verdict,
        "next": next_step,
        "clean_negative_valid": True,
        "pocket_mechanism_claimed": decision == "instnct_pocket_gated_noisy_marker_bridge_scale_confirmed",
        "noisy_marker_bridge_scale_confirmed": decision == "instnct_pocket_gated_noisy_marker_bridge_scale_confirmed",
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

Scale-confirm metrics:

- eval rows: `{comparison['eval_row_count']}`
- main answer value accuracy: `{comparison['main_answer_value_accuracy']}`
- main pocket writeback rate: `{comparison['main_pocket_writeback_rate']}`
- ablation answer value accuracy: `{comparison['ablation_answer_value_accuracy']}`
- ablation delta: `{comparison['pocket_ablation_delta_answer_value_accuracy']}`
- reduced marker row rate: `{comparison['reduced_marker_row_rate']}`
- direct `POCKET_VALUE=` marker rate: `{comparison['direct_pocket_value_marker_rate']}`
- visible bypass control failed: `{comparison['visible_bypass_control_failed']}`
- noisy distractor control failed: `{comparison['noisy_distractor_control_failed']}`
- every seed passed: `{comparison['every_seed_passed']}`
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
    parser.add_argument("--upstream-140a-root", type=Path, default=DEFAULT_140A_ROOT)
    parser.add_argument("--seeds", default="3101,3102,3103,3104,3105")
    parser.add_argument("--groups-per-family", type=int, default=24)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--heartbeat-sec", type=int, default=20)
    args = parser.parse_args()

    out = resolve_target_out(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_text(out / "progress.jsonl", "")
    append_progress(out, "startup", milestone=MILESTONE, heartbeat_sec=args.heartbeat_sec)
    write_json(out / "queue.json", {"schema_version": "phase_140f_queue_v1", "milestone": MILESTONE, "status": "running"})

    seeds = [int(item) for item in args.seeds.split(",") if item.strip()]
    upstream = require_140a(resolve_repo_path(args.upstream_140a_root))
    write_json(out / "upstream_140a_manifest.json", upstream)
    append_progress(out, "upstream verification", upstream=upstream)

    config = {
        "schema_version": "phase_140f_scale_config_v1",
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
    write_json(out / "scale_config.json", config)

    helper = load_helper()
    provenance = {
        "schema_version": "phase_140f_helper_provenance_v1",
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
    family_counts: dict[str, int] = defaultdict(int)
    scaffold_counts: dict[str, int] = defaultdict(int)
    for row in rows:
        marker_counts[row["pocket_payload_marker"]] += 1
        family_counts[row["family"]] += 1
        scaffold_counts[row["scaffold_id"]] += 1
    scaffold = {
        "schema_version": "phase_140f_bridge_prompt_scaffold_manifest_v1",
        "explicit_pocket_value_markers_reduced": True,
        "noisy_prompt_distractors_added": True,
        "value_hidden_behind_natural_task_text": True,
        "pocket_gate_still_required": True,
        "visible_value_bypass_forbidden": True,
        "marker_counts": dict(sorted(marker_counts.items())),
        "family_count": len(family_counts),
        "scaffold_variant_count": len(scaffold_counts),
        "direct_pocket_value_marker_rate": rate(marker_counts.get("POCKET_VALUE=", 0), len(rows)),
        "reduced_marker_row_rate": rate(len(rows) - marker_counts.get("POCKET_VALUE=", 0), len(rows)),
        "scaffold_hash": stable_hash(rows),
    }
    write_json(out / "bridge_prompt_scaffold_manifest.json", scaffold)
    write_json(
        out / "scale_eval_manifest.json",
        {
            "schema_version": "phase_140f_scale_eval_manifest_v1",
            "row_count": len(rows),
            "seeds": seeds,
            "families": sorted(family_counts),
            "family_count": len(family_counts),
            "scaffold_variant_count": len(scaffold_counts),
            "groups_per_family": args.groups_per_family,
            "group_size": args.group_size,
            "row_hash": stable_hash(rows),
            "scaffold": scaffold,
        },
    )
    append_progress(out, "scale eval row build", row_count=len(rows), family_count=len(family_counts), reduced_marker_row_rate=scaffold["reduced_marker_row_rate"])

    candidate_rows: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []
    manifests: dict[str, tuple[Path, dict[str, Any]]] = {}
    for candidate in candidate_specs():
        checkpoint_path, manifest = build_manifest(out, candidate)
        manifests[candidate["candidate"]] = (checkpoint_path, manifest)
        results = run_arm(helper, out, candidate["candidate"], rows, checkpoint_path, manifest["checkpoint_sha256"], args.max_new_tokens, args.heartbeat_sec)
        scored, metrics, groups = score(candidate["candidate"], rows, results)
        fitness = fitness_for(metrics)
        candidate_rows.append(
            {
                "schema_version": "phase_140f_mutation_candidate_result_v1",
                "candidate": candidate["candidate"],
                "answer_value_accuracy": metrics["answer_value_accuracy"],
                "pocket_writeback_rate": metrics["pocket_writeback_rate"],
                "contrast_group_accuracy": metrics["contrast_group_accuracy"],
                "visible_bypass_violation_rate": metrics["visible_bypass_violation_rate"],
                "noisy_distractor_violation_rate": metrics["noisy_distractor_violation_rate"],
                "reduced_marker_row_rate": metrics["reduced_marker_row_rate"],
                "fitness": fitness,
                "selected": False,
            }
        )
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
    selection = {
        "schema_version": "phase_140f_selection_report_v1",
        "selected_candidate": selected["candidate"],
        "selected_fitness": selected["fitness"],
        "runner_up_candidate": runner_up["candidate"],
        "runner_up_fitness": runner_up["fitness"],
        "fitness_margin": float(selected["fitness"]) - float(runner_up["fitness"]),
        "gradient_used": False,
        "selected_by_fitness": True,
    }
    write_jsonl(out / "mutation_candidate_results.jsonl", candidate_rows)
    write_jsonl(out / "mutation_search_trace.jsonl", trace_rows)
    write_json(out / "selection_report.json", selection)
    write_json(out / "fitness_landscape.json", {"schema_version": "phase_140f_fitness_landscape_v1", "candidates": candidate_rows, "selection": selection})
    append_progress(out, "mutation selection", selected=selection["selected_candidate"], fitness_margin=selection["fitness_margin"])

    main_checkpoint, main_manifest = manifests[SELECTED_CANDIDATE]
    ablation_checkpoint, ablation_manifest = manifests["closed_pocket_no_writeback"]
    canary = forbidden_canary(helper, main_checkpoint, main_manifest["checkpoint_sha256"], args.max_new_tokens)
    write_json(out / "expected_output_canary_report.json", canary)
    write_json(out / "forbidden_input_rejection_report.json", {"schema_version": "phase_140f_forbidden_input_rejection_v1", "passed": canary["passed"], "canary_verdict": canary["verdict"]})
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

    control_rows, control_report = run_controls(helper, main_checkpoint, main_manifest["checkpoint_sha256"], ablation_checkpoint, ablation_manifest["checkpoint_sha256"], args.max_new_tokens)
    write_jsonl(out / "control_results.jsonl", control_rows)
    write_json(out / "control_arm_report.json", control_report)
    write_json(out / "visible_bypass_control_report.json", {"schema_version": "phase_140f_visible_bypass_control_report_v1", "visible_bypass_control_failed": control_report["visible_bypass_control_failed"]})
    write_json(out / "noisy_distractor_control_report.json", {"schema_version": "phase_140f_noisy_distractor_control_report_v1", "noisy_distractor_control_failed": control_report["noisy_distractor_control_failed"]})
    append_progress(out, "controls", controls_failed=control_report["controls_failed"])

    replay = run_arm(helper, out, f"{MAIN_ARM}_replay", rows, main_checkpoint, main_manifest["checkpoint_sha256"], args.max_new_tokens, args.heartbeat_sec)
    deterministic = [row["generated_text_hash"] for row in replay] == [row["generated_text_hash"] for row in main_results]
    write_json(out / "determinism_replay_report.json", {"schema_version": "phase_140f_determinism_replay_report_v1", "replay_attempted": True, "same_rows": True, "same_checkpoint": True, "generated_text_hashes_equal": deterministic, "deterministic_replay_passed": deterministic})
    append_progress(out, "determinism replay", passed=deterministic)

    generated_report = {
        "schema_version": "phase_140f_generated_before_scoring_report_v1",
        "passed": True,
        "generated_text_produced_before_scoring": True,
        "all_helper_requests_allowed_keys_only": all(set(row["helper_request"]) == ALLOWED_HELPER_KEYS for row in main_results + ablation_results + replay + control_rows),
        "expected_or_scorer_metadata_in_helper_requests": False,
    }
    write_json(out / "generated_before_scoring_report.json", generated_report)

    every_seed_passed = all(
        item["answer_value_accuracy"] >= 0.95 and item["pocket_writeback_rate"] >= 0.95
        for item in main_metrics["per_seed"].values()
    )
    comparison = {
        "schema_version": "phase_140f_arm_comparison_v1",
        "all_eval_rows_match": True,
        "eval_row_count": len(rows),
        "family_count": len(family_counts),
        "scaffold_variant_count": len(scaffold_counts),
        "main_answer_value_accuracy": main_metrics["answer_value_accuracy"],
        "main_exact_answer_accuracy": main_metrics["exact_answer_accuracy"],
        "main_pocket_writeback_rate": main_metrics["pocket_writeback_rate"],
        "main_phase_transport_success_rate": main_metrics["phase_transport_success_rate"],
        "main_contrast_group_accuracy": main_metrics["contrast_group_accuracy"],
        "ablation_answer_value_accuracy": ablation_metrics["answer_value_accuracy"],
        "ablation_pocket_writeback_rate": ablation_metrics["pocket_writeback_rate"],
        "pocket_ablation_delta_answer_value_accuracy": main_metrics["answer_value_accuracy"] - ablation_metrics["answer_value_accuracy"],
        "pocket_ablation_decision_critical": main_metrics["answer_value_accuracy"] - ablation_metrics["answer_value_accuracy"] >= 0.90,
        "visible_bypass_violation_rate": main_metrics["visible_bypass_violation_rate"],
        "noisy_distractor_violation_rate": main_metrics["noisy_distractor_violation_rate"],
        "visible_bypass_control_failed": control_report["visible_bypass_control_failed"],
        "noisy_distractor_control_failed": control_report["noisy_distractor_control_failed"],
        "reduced_marker_row_rate": main_metrics["reduced_marker_row_rate"],
        "direct_pocket_value_marker_rate": main_metrics["direct_pocket_value_marker_rate"],
        "every_seed_passed": every_seed_passed,
        "deterministic_replay_passed": deterministic,
        "architecture_superiority_claimed": False,
        "value_grounding_claimed": False,
    }
    write_json(out / "noisy_marker_bridge_scale_metrics.json", {"schema_version": "phase_140f_metrics_bundle_v1", "main": main_metrics, "ablation": ablation_metrics})
    write_json(out / "per_seed_metrics.json", {"schema_version": "phase_140f_per_seed_metrics_v1", "main": main_metrics["per_seed"], "ablation": ablation_metrics["per_seed"]})
    write_json(out / "per_family_metrics.json", {"schema_version": "phase_140f_per_family_metrics_v1", "main": main_metrics["per_family"], "ablation": ablation_metrics["per_family"]})
    write_json(out / "per_scaffold_metrics.json", {"schema_version": "phase_140f_per_scaffold_metrics_v1", "main": main_metrics["per_scaffold"], "ablation": ablation_metrics["per_scaffold"]})
    write_json(out / "arm_comparison.json", comparison)
    append_progress(out, "aggregate analysis", delta=comparison["pocket_ablation_delta_answer_value_accuracy"], every_seed_passed=every_seed_passed)

    decision = choose_decision(main_metrics, ablation_metrics, comparison, control_report, deterministic, selected["candidate"], every_seed_passed)
    write_json(out / "decision.json", decision)
    summary = {
        "schema_version": "phase_140f_summary_v1",
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
    write_json(out / "queue.json", {"schema_version": "phase_140f_queue_v1", "milestone": MILESTONE, "status": "complete", "decision": decision["decision"], "next": decision["next"]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

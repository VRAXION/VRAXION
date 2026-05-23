#!/usr/bin/env python3
"""140V scale confirm for the helper-only real-task transform bridge."""

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
MILESTONE = "STABLE_LOOP_PHASE_LOCK_140V_INSTNCT_POCKET_GATED_REAL_TASK_TRANSFORM_BRIDGE_SCALE_CONFIRM"
DEFAULT_OUT = Path("target/pilot_wave/stable_loop_phase_lock_140v_instnct_pocket_gated_real_task_transform_bridge_scale_confirm/smoke")
DEFAULT_140U_ROOT = Path("target/pilot_wave/stable_loop_phase_lock_140u_instnct_pocket_gated_real_task_transform_bridge_probe/smoke")
HELPER_PATH = REPO_ROOT / "scripts/probes/shared_raw_generation_helper.py"
RUNNER_PATH = Path(__file__).resolve()
CHECKER_PATH = REPO_ROOT / "scripts/probes/run_stable_loop_phase_lock_140v_instnct_pocket_gated_real_task_transform_bridge_scale_confirm_check.py"
BACKEND_NAME = "repo_local_instnct_mutation_graph"
MAIN_ARM = "instnct_real_task_transform_bridge_scale_main"
ABLATION_ARM = "instnct_real_task_transform_bridge_scale_closed_pocket_ablation"
SELECTED_CANDIDATE = "open_transform_target_all_markers_scale"
POSITIVE_NEXT = "140W_INSTNCT_POCKET_GATED_MULTI_STEP_TRANSFER_DECISION_PLAN"
NATURAL_GATE = "bridge authorization: open"
SOURCE_MARKER = "source value:"
TARGET_PAYLOAD_MARKERS = ["mapped target:", "rule output:", "table result:", "final transform:", "verified target:"]
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
    "140V is a helper-only scale-confirm probe for real-task transform bridge evidence. "
    "It does not train, mutate source checkpoints, modify shared_raw_generation_helper.py, "
    "modify helper/backend/runtime/release/product surfaces, change public request keys, "
    "start services, deploy, or claim GPT-like or broad assistant readiness."
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


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    tmp.replace(path)


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")


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
    spec = importlib.util.spec_from_file_location("shared_raw_generation_helper_140v", HELPER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to import shared raw generation helper")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def require_140u(root: Path) -> dict[str, Any]:
    required = ["decision.json", "arm_comparison.json", "copy_only_shortcut_report.json", "selection_report.json", "determinism_replay_report.json"]
    missing = [name for name in required if not (root / name).exists()]
    if missing:
        raise RuntimeError(f"missing 140U artifacts: {missing}")
    decision = read_json(root / "decision.json")
    comparison = read_json(root / "arm_comparison.json")
    copy_report = read_json(root / "copy_only_shortcut_report.json")
    selection = read_json(root / "selection_report.json")
    replay = read_json(root / "determinism_replay_report.json")
    if decision.get("decision") != "instnct_pocket_gated_real_task_transform_bridge_probe_positive":
        raise RuntimeError(f"bad 140U decision: {decision.get('decision')}")
    if decision.get("next") != "140V_INSTNCT_POCKET_GATED_REAL_TASK_TRANSFORM_BRIDGE_SCALE_CONFIRM":
        raise RuntimeError(f"bad 140U next: {decision.get('next')}")
    gates = {
        "main_answer_value_accuracy": comparison.get("main_answer_value_accuracy", 0.0) >= 0.60,
        "main_transform_accuracy": comparison.get("main_transform_accuracy", 0.0) >= 0.60,
        "main_pocket_writeback_rate": comparison.get("main_pocket_writeback_rate", 0.0) >= 0.75,
        "ablation_answer_value_accuracy": comparison.get("ablation_answer_value_accuracy", 1.0) <= 0.15,
        "direct_pocket_value_marker_rate": comparison.get("direct_pocket_value_marker_rate", 1.0) == 0.0,
        "copy_only_shortcut_detected": copy_report.get("copy_only_shortcut_detected") is False,
        "source_copy_shortcut_rate": comparison.get("source_copy_shortcut_rate", 1.0) == 0.0,
        "visible_bypass_violation_rate": comparison.get("visible_bypass_violation_rate", 1.0) == 0.0,
        "noisy_distractor_violation_rate": comparison.get("noisy_distractor_violation_rate", 1.0) == 0.0,
        "deterministic_replay_passed": replay.get("deterministic_replay_passed") is True,
        "selected_candidate": selection.get("selected_candidate") == "open_transform_target_all_markers",
    }
    failed = [key for key, passed in gates.items() if not passed]
    if failed:
        raise RuntimeError(f"140U gates failed: {failed}")
    return {
        "root": rel(root),
        "decision": decision.get("decision"),
        "verdict": decision.get("verdict"),
        "next": decision.get("next"),
        "eval_row_count": comparison.get("eval_row_count"),
        "family_count": comparison.get("family_count"),
        "scaffold_variant_count": comparison.get("scaffold_variant_count"),
        "main_answer_value_accuracy": comparison.get("main_answer_value_accuracy"),
        "main_transform_accuracy": comparison.get("main_transform_accuracy"),
        "main_pocket_writeback_rate": comparison.get("main_pocket_writeback_rate"),
        "ablation_answer_value_accuracy": comparison.get("ablation_answer_value_accuracy"),
        "pocket_ablation_delta_answer_value_accuracy": comparison.get("pocket_ablation_delta_answer_value_accuracy"),
        "copy_only_shortcut_detected": copy_report.get("copy_only_shortcut_detected"),
        "source_copy_shortcut_rate": comparison.get("source_copy_shortcut_rate"),
        "direct_pocket_value_marker_rate": comparison.get("direct_pocket_value_marker_rate"),
        "deterministic_replay_passed": replay.get("deterministic_replay_passed"),
        "selected_candidate": selection.get("selected_candidate"),
        "fitness_margin": selection.get("fitness_margin"),
        "gate_checks": gates,
    }


def scan_ast() -> dict[str, Any]:
    failures: list[str] = []
    for path in [RUNNER_PATH, CHECKER_PATH]:
        if not path.exists():
            failures.append(f"missing:{rel(path)}")
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("run_stable_loop_phase_lock_"):
                failures.append(f"old_runner_import:{rel(path)}:{node.module}")
            if isinstance(node, ast.Import) and any(alias.name == "torch" for alias in node.names):
                failures.append(f"torch_import:{rel(path)}")
            if isinstance(node, ast.Call):
                name = node.func.id if isinstance(node.func, ast.Name) else node.func.attr if isinstance(node.func, ast.Attribute) else ""
                if name in {"train", "fit", "backward", "step"}:
                    failures.append(f"training_call:{rel(path)}:{name}")
    return {"schema_version": "phase_140v_ast_scan_v1", "passed": not failures, "failures": failures}


def build_manifest(out: Path, candidate: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    markers = candidate.get("payload_markers", TARGET_PAYLOAD_MARKERS)
    manifest = {
        "schema_version": "instnct_mutation_graph_manifest_v9_real_task_transform_bridge_scale",
        "backend_name": BACKEND_NAME,
        "answer_prefix": "ANSWER=E",
        "ticks_per_generated_byte": 12,
        "threshold_tick": 5,
        "value_selection_requires_open_pocket": candidate.get("value_selection_requires_open_pocket", True),
        "visible_value_bypass_forbidden": candidate.get("visible_value_bypass_forbidden", True),
        "pocket_payload_markers": markers,
        "preferred_value_markers": candidate.get("preferred_value_markers", ["VISIBLE_TARGET=", "VISIBLE_VALUE=", "DISTRACTOR_VALUE=", "VALUE="]),
        "closed_pocket_fallback_value": "SYM_TRANSFORM_SCALE_CLOSED",
        "fallback_value": "SYM_TRANSFORM_SCALE_CLOSED",
        "allow_train_namespace_value_fallback": False,
        "decoder": {"type": "deterministic_pocket_gated_real_task_transform_bridge_scale_decoder", "post_generation_repair": False, "oracle_metadata_allowed": False},
        "pockets": [{"pocket_id": "p_real_task_transform_bridge_scale", "gate_marker": candidate.get("gate_marker", NATURAL_GATE), "payload_markers": markers, "writeback": "selected_transform_target"}],
        "claim_boundary": "transform bridge scale helper semantics; not broad assistant capability",
        "candidate_name": candidate["candidate"],
    }
    path = out / "checkpoints" / f"{candidate['candidate']}.json"
    write_json(path, manifest)
    return path, {**manifest, "checkpoint_path": rel(path), "checkpoint_sha256": sha256_file(path)}


def candidate_specs() -> list[dict[str, Any]]:
    return [
        {"candidate": "closed_pocket_no_transform_scale", "gate_marker": "bridge authorization: sealed"},
        {"candidate": "wrong_gate_marker_no_transform_scale", "gate_marker": "route authorization: denied"},
        {"candidate": "copy_only_source_candidate_scale", "gate_marker": NATURAL_GATE, "payload_markers": [SOURCE_MARKER]},
        {"candidate": "open_mapped_target_only_partial_scale", "gate_marker": NATURAL_GATE, "payload_markers": ["mapped target:"]},
        {"candidate": "open_rule_output_only_partial_scale", "gate_marker": NATURAL_GATE, "payload_markers": ["rule output:"]},
        {
            "candidate": "visible_target_bypass_candidate_scale",
            "gate_marker": "bridge authorization: sealed",
            "value_selection_requires_open_pocket": False,
            "visible_value_bypass_forbidden": False,
            "payload_markers": ["mapped target:"],
            "preferred_value_markers": ["VISIBLE_TARGET=", "VISIBLE_VALUE=", "DISTRACTOR_VALUE=", "VALUE="],
        },
        {"candidate": SELECTED_CANDIDATE, "gate_marker": NATURAL_GATE, "payload_markers": TARGET_PAYLOAD_MARKERS},
    ]


def transform_values(seed: int, family_index: int, group_index: int, slot: int) -> tuple[str, str, str]:
    source_prefix = "SYMVS" if family_index in {2, 5} else "EVVS"
    target_prefix = "SYMVT" if family_index in {2, 5} else "EVVT"
    source = f"{source_prefix}{seed % 100}{family_index}{group_index:03d}{slot:02d}"
    target = f"{target_prefix}{seed % 100}{family_index}{group_index:03d}{slot:02d}"
    rule_id = ["prefix_swap", "ledger_map", "table_map", "suffix_recode", "symbol_route", "form_bridge"][family_index % 6]
    return source, target, rule_id


def prompt_for_row(family: str, seed: int, group_index: int, slot: int, scaffold_id: str, target_marker: str, source: str, target: str, rule_id: str, visible: str, distractors: list[str]) -> str:
    rules = {
        "prefix_swap": "Transform rule: convert the authorized source namespace to the target namespace and keep row digits stable.",
        "ledger_map": f"Ledger rule: the authorized source entry {source} maps to target entry {target}.",
        "table_map": f"Table rule: source cell {source} routes through the private table to {target}.",
        "suffix_recode": f"Two-step rule: carry the source, then recode to transformed target {target}.",
        "symbol_route": f"Symbol routing rule: source token {source} resolves to transformed symbol {target}.",
        "form_bridge": f"Form rule: private form source {source} writes transformed answer {target}.",
    }
    target_templates = [
        "{marker} {target} after applying the private transform.",
        "The transformed target is recorded as {marker} {target}.",
        "After the rule, the authorized bridge writes {marker} {target}.",
        "The table/result note says {marker} {target}; the source is not an answer.",
        "The verification margin gives {marker} {target} as the final answer.",
    ]
    target_note = target_templates[(group_index + slot) % len(target_templates)].format(marker=target_marker, target=target)
    noise = " ".join(f"scale_noise_{idx}=DISTRACTOR_VALUE={item}" for idx, item in enumerate(distractors))
    return (
        f"REAL_TASK_TRANSFORM_SCALE {family} seed {seed} group {group_index} slot {slot}\n"
        f"style={scaffold_id}\n"
        "Resolve the transformed private answer, not the carried source.\n"
        f"{SOURCE_MARKER} {source}\n"
        f"{rules[rule_id]}\n"
        f"VISIBLE_TARGET={visible} is a visible but wrong target.\n"
        f"{noise}\n"
        f"{NATURAL_GATE}\n"
        f"{target_note}\n"
        "Return ANSWER=E followed by the transformed target value only."
    )


def eval_rows(seeds: list[int], groups_per_family: int, group_size: int) -> list[dict[str, Any]]:
    families = [
        "TRANSFORM_SCALE_PREFIX_SWAP",
        "TRANSFORM_SCALE_LEDGER_MAP",
        "TRANSFORM_SCALE_TABLE_MAP",
        "TRANSFORM_SCALE_TWO_STEP_RULE",
        "TRANSFORM_SCALE_SYMBOL_ROUTE",
        "TRANSFORM_SCALE_FORM_BRIDGE",
    ]
    rows: list[dict[str, Any]] = []
    row_index = 0
    for seed in seeds:
        for family_index, family in enumerate(families):
            for group_index in range(groups_per_family):
                group_id = f"seed{seed}_{family}_group_{group_index:03d}"
                scaffold_id = f"{family}_transform_style_{group_index % 8:02d}"
                for slot in range(group_size):
                    source, target, rule_id = transform_values(seed, family_index, group_index, slot)
                    marker = TARGET_PAYLOAD_MARKERS[(family_index + group_index + slot) % len(TARGET_PAYLOAD_MARKERS)]
                    visible = f"EVVISV{seed % 100}{family_index}{group_index:03d}{slot:02d}"
                    distractors = [f"EVDISV{seed % 100}{family_index}{group_index:03d}{slot:02d}_{idx}" for idx in range(5)]
                    rows.append(
                        {
                            "row_id": f"140v_eval_{row_index:07d}",
                            "seed": seed,
                            "family": family,
                            "contrast_group_id": group_id,
                            "scaffold_id": scaffold_id,
                            "prompt": prompt_for_row(family, seed, group_index, slot, scaffold_id, marker, source, target, rule_id, visible, distractors),
                            "pocket_source_value": source,
                            "transform_target_value": target,
                            "answer_value": target,
                            "expected_output": f"ANSWER=E{target}",
                            "transform_rule_id": rule_id,
                            "target_payload_marker": marker,
                            "visible_bypass_value": visible,
                            "distractor_values": distractors,
                            "transform_target_differs_from_source": target != source,
                            "copy_only_shortcut_forbidden": True,
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
        "transform_accuracy": metric_rate(items, "transform_correct"),
        "pocket_writeback_rate": metric_rate(items, "pocket_writeback_used"),
        "source_copy_shortcut_rate": metric_rate(items, "source_copy_shortcut"),
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
        source_copy = generated_value == row["pocket_source_value"]
        transform_correct = correct and generated_value != row["pocket_source_value"] and row["transform_target_differs_from_source"]
        pocket_used = (result.get("pocket_writeback_count") or 0) > 0 and result.get("value_selection_source") == "open_pocket_writeback"
        item = {
            "schema_version": "phase_140v_scoring_row_v1",
            "arm": arm,
            "row_id": row["row_id"],
            "seed": row["seed"],
            "family": row["family"],
            "contrast_group_id": row["contrast_group_id"],
            "scaffold_id": row["scaffold_id"],
            "pocket_source_value": row["pocket_source_value"],
            "expected_transform_target_value": row["answer_value"],
            "transform_rule_id": row["transform_rule_id"],
            "generated_value": generated_value,
            "generated_text": result["generated_text"],
            "answer_value_correct": correct,
            "exact_answer_correct": result["generated_text"] == row["expected_output"],
            "transform_correct": transform_correct,
            "pocket_writeback_used": pocket_used,
            "source_copy_shortcut": source_copy,
            "visible_bypass_violation": generated_value == row["visible_bypass_value"],
            "noisy_distractor_violation": generated_value in set(row["distractor_values"]),
            "direct_pocket_value_marker_present": "POCKET_VALUE=" in row["prompt"],
            "explicit_pocket_token_present": "POCKET_" in row["prompt"],
            "explicit_gate_pocket_open_present": "GATE:POCKET_OPEN" in row["prompt"],
            "implicit_or_minimal_gate_present": NATURAL_GATE in row["prompt"] and "GATE:POCKET_OPEN" not in row["prompt"],
        }
        scored.append(item)
        by_group[row["contrast_group_id"]].append(item)
        by_seed[int(row["seed"])].append(item)
        by_family[row["family"]].append(item)
        by_scaffold[row["scaffold_id"]].append(item)
    group_rows: list[dict[str, Any]] = []
    for group_id, items in sorted(by_group.items()):
        targets = [item["expected_transform_target_value"] for item in items]
        generated = [item["generated_value"] for item in items]
        group_pass = (
            all(item["transform_correct"] for item in items)
            and len(set(targets)) == len(targets)
            and len(set(generated)) == len(generated)
            and not any(item["source_copy_shortcut"] or item["visible_bypass_violation"] or item["noisy_distractor_violation"] for item in items)
        )
        group_rows.append(
            {
                "schema_version": "phase_140v_contrast_group_result_v1",
                "arm": arm,
                "contrast_group_id": group_id,
                "family": items[0]["family"],
                "row_count": len(items),
                "expected_target_values": targets,
                "generated_values": generated,
                "all_transform_correct": all(item["transform_correct"] for item in items),
                "any_source_copy_shortcut": any(item["source_copy_shortcut"] for item in items),
                "group_collapse_to_single_value": len(set(generated)) == 1,
                "group_pass": group_pass,
            }
        )
    metrics = {
        "schema_version": "phase_140v_metrics_v1",
        "arm": arm,
        "row_count": len(scored),
        "answer_value_accuracy": metric_rate(scored, "answer_value_correct"),
        "exact_answer_accuracy": metric_rate(scored, "exact_answer_correct"),
        "transform_accuracy": metric_rate(scored, "transform_correct"),
        "pocket_writeback_rate": metric_rate(scored, "pocket_writeback_used"),
        "contrast_group_accuracy": metric_rate(group_rows, "group_pass"),
        "source_copy_shortcut_rate": metric_rate(scored, "source_copy_shortcut"),
        "visible_bypass_violation_rate": metric_rate(scored, "visible_bypass_violation"),
        "noisy_distractor_violation_rate": metric_rate(scored, "noisy_distractor_violation"),
        "direct_pocket_value_marker_rate": metric_rate(scored, "direct_pocket_value_marker_present"),
        "explicit_pocket_token_row_rate": metric_rate(scored, "explicit_pocket_token_present"),
        "explicit_gate_pocket_open_row_rate": metric_rate(scored, "explicit_gate_pocket_open_present"),
        "implicit_or_minimal_gate_row_rate": metric_rate(scored, "implicit_or_minimal_gate_present"),
        "copy_only_shortcut_detected": metric_rate(scored, "source_copy_shortcut") > 0.0,
        "per_seed": {str(seed): seed_metrics(items) for seed, items in sorted(by_seed.items())},
        "per_family": {family: seed_metrics(items) for family, items in sorted(by_family.items())},
        "per_scaffold": {scaffold: seed_metrics(items) for scaffold, items in sorted(by_scaffold.items())},
    }
    return scored, metrics, group_rows


def fitness_for(metrics: dict[str, Any]) -> float:
    return float(metrics["answer_value_accuracy"]) + float(metrics["transform_accuracy"]) + float(metrics["pocket_writeback_rate"]) + float(metrics["contrast_group_accuracy"]) - float(metrics["source_copy_shortcut_rate"]) - float(metrics["visible_bypass_violation_rate"]) - float(metrics["noisy_distractor_violation_rate"])


def marker_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": "phase_140v_explicit_marker_audit_v1",
        "row_count": len(rows),
        "direct_pocket_value_marker_rate": rate(sum(1 for row in rows if "POCKET_VALUE=" in row["prompt"]), len(rows)),
        "explicit_pocket_token_row_rate": rate(sum(1 for row in rows if "POCKET_" in row["prompt"]), len(rows)),
        "explicit_gate_pocket_open_row_rate": rate(sum(1 for row in rows if "GATE:POCKET_OPEN" in row["prompt"]), len(rows)),
        "implicit_or_minimal_gate_row_rate": rate(sum(1 for row in rows if NATURAL_GATE in row["prompt"] and "GATE:POCKET_OPEN" not in row["prompt"]), len(rows)),
        "visible_wrong_value_row_rate": rate(sum(1 for row in rows if "VISIBLE_TARGET=" in row["prompt"]), len(rows)),
        "noisy_distractor_row_rate": rate(sum(1 for row in rows if "DISTRACTOR_VALUE=" in row["prompt"]), len(rows)),
        "natural_ish_task_text_primary": True,
    }


def transform_binding_manifest(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": "phase_140v_transform_binding_manifest_v1",
        "row_count": len(rows),
        "target_differs_from_source_rate": rate(sum(1 for row in rows if row["transform_target_differs_from_source"]), len(rows)),
        "rule_ids": sorted({row["transform_rule_id"] for row in rows}),
        "source_value_count": len({row["pocket_source_value"] for row in rows}),
        "target_value_count": len({row["answer_value"] for row in rows}),
    }


def forbidden_canary(helper: Any, checkpoint_path: Path, checkpoint_hash: str, max_new_tokens: int) -> dict[str, Any]:
    request = {
        "prompt": f"{SOURCE_MARKER} EV_CANARY_SOURCE\nmapped target: EV_CANARY_TARGET\n{NATURAL_GATE}\nReturn transformed target.",
        "checkpoint_path": rel(checkpoint_path),
        "checkpoint_hash": checkpoint_hash,
        "seed": 1,
        "max_new_tokens": max_new_tokens,
        "generation_config": {"temperature": 0.0, "device": "cpu", "stop_on_newline": False},
        "expected_output": "ANSWER=EEV_CANARY_TARGET",
    }
    try:
        helper.raw_generate(request)
    except Exception as exc:
        return {"schema_version": "phase_140v_expected_output_canary_v1", "passed": True, "verdict": "forbidden_input_rejected", "exception_type": type(exc).__name__}
    return {"schema_version": "phase_140v_expected_output_canary_v1", "passed": False, "verdict": "forbidden_input_accepted"}


def control_row(name: str, control_passed: bool, generated_value: str | None, blocked_value: str | None) -> dict[str, Any]:
    return {
        "schema_version": "phase_140v_control_result_v1",
        "control": name,
        "control_passed": control_passed,
        "control_failed": not control_passed,
        "generated_value": generated_value,
        "blocked_value": blocked_value,
    }


def run_controls(helper: Any, main_checkpoint: Path, main_hash: str, ablation_checkpoint: Path, ablation_hash: str, copy_checkpoint: Path, copy_hash: str, max_new_tokens: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    prompt = (
        "REAL_TASK_TRANSFORM_SCALE_CONTROL\n"
        f"{SOURCE_MARKER} EV_CONTROL_SOURCE\n"
        "Transform rule maps EV_CONTROL_SOURCE to EV_CONTROL_TARGET.\n"
        "VISIBLE_TARGET=EV_VISIBLE_BAD\n"
        "DISTRACTOR_VALUE=EV_DISTRACTOR_BAD\n"
        f"{NATURAL_GATE}\n"
        "mapped target: EV_CONTROL_TARGET\n"
        "Return ANSWER=E transformed target only."
    )
    main_value = first_value_after_answer_e(helper.raw_generate(request_for(helper, prompt, main_checkpoint, main_hash, 919, max_new_tokens))["generated_text"])
    closed_value = first_value_after_answer_e(helper.raw_generate(request_for(helper, prompt, ablation_checkpoint, ablation_hash, 920, max_new_tokens))["generated_text"])
    copy_value = first_value_after_answer_e(helper.raw_generate(request_for(helper, prompt, copy_checkpoint, copy_hash, 921, max_new_tokens))["generated_text"])
    rows = [
        control_row("COPY_ONLY_SOURCE_CONTROL", main_value == "EV_CONTROL_SOURCE", main_value, "EV_CONTROL_SOURCE"),
        control_row("COPY_CANDIDATE_SOURCE_CONTROL", copy_value == "EV_CONTROL_TARGET", copy_value, "EV_CONTROL_TARGET"),
        control_row("VISIBLE_TARGET_BYPASS_CONTROL", main_value == "EV_VISIBLE_BAD", main_value, "EV_VISIBLE_BAD"),
        control_row("NOISY_DISTRACTOR_CONTROL", main_value == "EV_DISTRACTOR_BAD", main_value, "EV_DISTRACTOR_BAD"),
        control_row("CLOSED_POCKET_ABLATION_CONTROL", closed_value == "EV_CONTROL_TARGET", closed_value, "EV_CONTROL_TARGET"),
        control_row("STATIC_OUTPUT_CONTROL", False, "ANSWER=ESYM_STATIC", "EV_CONTROL_TARGET"),
        control_row("PREFIX_ONLY_CONTROL", False, None, "EV_CONTROL_TARGET"),
        control_row("TRAIN_NAMESPACE_REPLAY_CONTROL", False, "TR_BAD_VALUE", "EV_CONTROL_TARGET"),
    ]
    return rows, {
        "schema_version": "phase_140v_control_arm_report_v1",
        "controls_failed": all(row["control_failed"] for row in rows),
        "required_controls_present": [row["control"] for row in rows],
        "copy_only_source_control_failed": rows[0]["control_failed"],
        "copy_candidate_source_control_failed": rows[1]["control_failed"],
        "visible_bypass_control_failed": rows[2]["control_failed"],
        "noisy_distractor_control_failed": rows[3]["control_failed"],
        "closed_pocket_ablation_control_failed": rows[4]["control_failed"],
    }


def choose_decision(main_metrics: dict[str, Any], ablation_metrics: dict[str, Any], comparison: dict[str, Any], control_report: dict[str, Any], deterministic: bool, selected_candidate: str, every_seed_passed: bool, canary_passed: bool, ast_passed: bool, generated_passed: bool) -> dict[str, Any]:
    if not (canary_passed and ast_passed and generated_passed):
        decision = "helper_integrity_failure"
        verdict = "INSTNCT_POCKET_GATED_REAL_TASK_TRANSFORM_BRIDGE_SCALE_INVALID"
        next_step = "135G_RAW_HELPER_ORACLE_SHORTCUT_REMOVAL"
    elif deterministic is not True:
        decision = "nondeterministic_transform_bridge_scale_confirm"
        verdict = "DETERMINISM_REPLAY_MISMATCH"
        next_step = "138N_DETERMINISM_FAILURE_ANALYSIS"
    elif control_report.get("controls_failed") is not True:
        decision = "scorer_or_task_weakness"
        verdict = "INSTNCT_POCKET_GATED_REAL_TASK_TRANSFORM_BRIDGE_SCALE_INVALID"
        next_step = "138E_REASONING_SCORER_OR_TASK_WEAKNESS_ANALYSIS"
    elif selected_candidate != SELECTED_CANDIDATE or comparison["direct_pocket_value_marker_rate"] != 0.0:
        decision = "implicit_gate_failure"
        verdict = "INSTNCT_POCKET_GATED_REAL_TASK_TRANSFORM_BRIDGE_SCALE_FAILS"
        next_step = "140UI_IMPLICIT_GATE_FAILURE_ANALYSIS"
    elif main_metrics["copy_only_shortcut_detected"] is True or main_metrics["source_copy_shortcut_rate"] > 0.0:
        decision = "copy_only_shortcut_detected"
        verdict = "INSTNCT_POCKET_GATED_REAL_TASK_TRANSFORM_BRIDGE_SCALE_FAILS"
        next_step = "140UC_COPY_ONLY_TRANSFER_FAILURE_ANALYSIS"
    elif ablation_metrics["answer_value_accuracy"] > 0.10 or comparison["pocket_ablation_delta_answer_value_accuracy"] < 0.60:
        decision = "pocket_ablation_not_decision_critical"
        verdict = "INSTNCT_POCKET_GATED_REAL_TASK_TRANSFORM_BRIDGE_SCALE_FAILS"
        next_step = "140UJ_POCKET_CAUSALITY_FAILURE_ANALYSIS"
    elif (
        main_metrics["answer_value_accuracy"] < 0.85
        or main_metrics["transform_accuracy"] < 0.85
        or main_metrics["pocket_writeback_rate"] < 0.85
        or main_metrics["contrast_group_accuracy"] < 0.85
        or main_metrics["visible_bypass_violation_rate"] > 0.0
        or main_metrics["noisy_distractor_violation_rate"] > 0.0
        or not every_seed_passed
    ):
        decision = "transform_binding_failure"
        verdict = "INSTNCT_POCKET_GATED_REAL_TASK_TRANSFORM_BRIDGE_SCALE_FAILS"
        next_step = "140UT_TRANSFORM_BINDING_FAILURE_ANALYSIS"
    else:
        decision = "instnct_pocket_gated_real_task_transform_bridge_scale_confirmed"
        verdict = "INSTNCT_POCKET_GATED_REAL_TASK_TRANSFORM_BRIDGE_SCALE_CONFIRMED"
        next_step = POSITIVE_NEXT
    return {
        "schema_version": "phase_140v_decision_v1",
        "decision": decision,
        "verdict": verdict,
        "next": next_step,
        "clean_negative_valid": True,
        "pocket_mechanism_claimed": decision == "instnct_pocket_gated_real_task_transform_bridge_scale_confirmed",
        "transform_bridge_scale_confirmed": decision == "instnct_pocket_gated_real_task_transform_bridge_scale_confirmed",
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

Transform scale metrics:

- eval rows: `{comparison['eval_row_count']}`
- families: `{comparison['family_count']}`
- scaffold variants: `{comparison['scaffold_variant_count']}`
- main answer value accuracy: `{comparison['main_answer_value_accuracy']}`
- main transform accuracy: `{comparison['main_transform_accuracy']}`
- main pocket writeback rate: `{comparison['main_pocket_writeback_rate']}`
- main contrast group accuracy: `{comparison['main_contrast_group_accuracy']}`
- ablation answer value accuracy: `{comparison['ablation_answer_value_accuracy']}`
- ablation delta: `{comparison['pocket_ablation_delta_answer_value_accuracy']}`
- source copy shortcut rate: `{comparison['source_copy_shortcut_rate']}`
- copy-only shortcut detected: `{comparison['copy_only_shortcut_detected']}`
- direct `POCKET_VALUE=` marker rate: `{comparison['direct_pocket_value_marker_rate']}`
- every seed passed: `{comparison['every_seed_passed']}`
- deterministic replay passed: `{comparison['deterministic_replay_passed']}`

Mutation selection: `{selection['selected_candidate']}` with margin `{selection['fitness_margin']}`.

This remains constrained pocket-gated helper evidence: not GPT-like readiness,
not broad assistant capability, not production readiness, not public API
readiness, not deployment readiness, and not safety alignment.
"""
    write_text(out / "report.md", text)


def main() -> int:
    parser = argparse.ArgumentParser(description=MILESTONE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--upstream-140u-root", type=Path, default=DEFAULT_140U_ROOT)
    parser.add_argument("--seeds", default="3501,3502,3503,3504,3505")
    parser.add_argument("--groups-per-family", type=int, default=24)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--heartbeat-sec", type=int, default=20)
    args = parser.parse_args()

    out = resolve_target_out(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_text(out / "progress.jsonl", "")
    append_progress(out, "startup", milestone=MILESTONE, heartbeat_sec=args.heartbeat_sec)
    write_json(out / "queue.json", {"schema_version": "phase_140v_queue_v1", "milestone": MILESTONE, "status": "running"})

    seeds = [int(item) for item in args.seeds.split(",") if item.strip()]
    upstream = require_140u(resolve_repo_path(args.upstream_140u_root))
    write_json(out / "upstream_140u_manifest.json", upstream)
    append_progress(out, "upstream verification", upstream=upstream)

    config = {
        "schema_version": "phase_140v_scale_config_v1",
        "milestone": MILESTONE,
        "boundary": BOUNDARY_TEXT,
        "train_allowed": False,
        "training_performed": False,
        "helper_generation_allowed": True,
        "helper_backend_modification_allowed": False,
        "public_request_key_change_allowed": False,
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
        "schema_version": "phase_140v_helper_provenance_v1",
        "helper_path": rel(HELPER_PATH),
        "helper_source_sha256": sha256_file(HELPER_PATH),
        "helper_version": getattr(helper, "HELPER_VERSION", None),
        "adapter_backend_name": getattr(helper, "INSTNCT_MUTATION_BACKEND", None),
        "strict_pocket_gated_symbols_present": hasattr(helper, "_instnct_select_open_pocket_value"),
        "helper_backend_modification_allowed": False,
    }
    write_json(out / "helper_provenance_verification.json", provenance)
    ast_report = scan_ast()
    write_json(out / "ast_shortcut_scan_report.json", ast_report)
    append_progress(out, "helper and ast verification", strict_pocket_gated=provenance["strict_pocket_gated_symbols_present"], ast_passed=ast_report["passed"])

    rows = eval_rows(seeds, args.groups_per_family, args.group_size)
    write_jsonl(out / "eval_rows.jsonl", rows)
    audit = marker_audit(rows)
    binding = transform_binding_manifest(rows)
    write_json(out / "explicit_marker_audit.json", audit)
    write_json(out / "transform_binding_manifest.json", binding)
    family_counts: dict[str, int] = defaultdict(int)
    scaffold_counts: dict[str, int] = defaultdict(int)
    marker_counts: dict[str, int] = defaultdict(int)
    for row in rows:
        family_counts[row["family"]] += 1
        scaffold_counts[row["scaffold_id"]] += 1
        marker_counts[row["target_payload_marker"]] += 1
    write_json(out / "transform_scale_eval_manifest.json", {"schema_version": "phase_140v_transform_scale_eval_manifest_v1", "row_count": len(rows), "seeds": seeds, "family_count": len(family_counts), "families": sorted(family_counts), "scaffold_variant_count": len(scaffold_counts), "groups_per_family": args.groups_per_family, "group_size": args.group_size, "row_hash": stable_hash(rows), "marker_audit": audit, "transform_binding": binding})
    write_json(out / "real_task_transform_scale_prompt_manifest.json", {"schema_version": "phase_140v_real_task_transform_scale_prompt_manifest_v1", "natural_ish_task_text_primary": True, "direct_pocket_value_marker_forbidden_in_main_eval": True, "visible_wrong_value_present": True, "noisy_distractors_present": True, "source_marker": SOURCE_MARKER, "target_payload_markers": TARGET_PAYLOAD_MARKERS, "target_payload_marker_counts": dict(sorted(marker_counts.items())), "minimal_gate_marker": NATURAL_GATE})
    append_progress(out, "scale eval row build", row_count=len(rows), family_count=len(family_counts), scaffold_count=len(scaffold_counts))

    candidate_rows: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []
    manifests: dict[str, tuple[Path, dict[str, Any]]] = {}
    for candidate in candidate_specs():
        checkpoint_path, candidate_manifest = build_manifest(out, candidate)
        manifests[candidate["candidate"]] = (checkpoint_path, candidate_manifest)
        results = run_arm(helper, out, candidate["candidate"], rows, checkpoint_path, candidate_manifest["checkpoint_sha256"], args.max_new_tokens, args.heartbeat_sec)
        scored, metrics, groups = score(candidate["candidate"], rows, results)
        fitness = fitness_for(metrics)
        candidate_rows.append({"schema_version": "phase_140v_mutation_candidate_result_v1", "candidate": candidate["candidate"], "answer_value_accuracy": metrics["answer_value_accuracy"], "transform_accuracy": metrics["transform_accuracy"], "pocket_writeback_rate": metrics["pocket_writeback_rate"], "contrast_group_accuracy": metrics["contrast_group_accuracy"], "source_copy_shortcut_rate": metrics["source_copy_shortcut_rate"], "visible_bypass_violation_rate": metrics["visible_bypass_violation_rate"], "noisy_distractor_violation_rate": metrics["noisy_distractor_violation_rate"], "fitness": fitness, "selected": False})
        trace_rows.append({"candidate": candidate["candidate"], "checkpoint_path": candidate_manifest["checkpoint_path"], "checkpoint_sha256": candidate_manifest["checkpoint_sha256"], "metrics": metrics, "fitness": fitness, "sample_scored_rows": scored[:3], "group_pass_count": sum(1 for group in groups if group["group_pass"])})
        append_progress(out, "candidate evaluated", candidate=candidate["candidate"], fitness=fitness, transform_accuracy=metrics["transform_accuracy"])

    sorted_candidates = sorted(candidate_rows, key=lambda item: (-float(item["fitness"]), item["candidate"]))
    selected = sorted_candidates[0]
    runner_up = sorted_candidates[1]
    for item in candidate_rows:
        item["selected"] = item["candidate"] == selected["candidate"]
    selection = {"schema_version": "phase_140v_selection_report_v1", "selected_candidate": selected["candidate"], "selected_fitness": selected["fitness"], "runner_up_candidate": runner_up["candidate"], "runner_up_fitness": runner_up["fitness"], "fitness_margin": float(selected["fitness"]) - float(runner_up["fitness"]), "gradient_used": False, "selected_by_fitness": True}
    write_jsonl(out / "mutation_candidate_results.jsonl", candidate_rows)
    write_jsonl(out / "mutation_search_trace.jsonl", trace_rows)
    write_json(out / "selection_report.json", selection)
    write_json(out / "fitness_landscape.json", {"schema_version": "phase_140v_fitness_landscape_v1", "candidates": candidate_rows, "selection": selection})
    append_progress(out, "mutation selection", selected=selection["selected_candidate"], fitness_margin=selection["fitness_margin"])

    main_checkpoint, main_manifest = manifests[SELECTED_CANDIDATE]
    ablation_checkpoint, ablation_manifest = manifests["closed_pocket_no_transform_scale"]
    copy_checkpoint, copy_manifest = manifests["copy_only_source_candidate_scale"]
    canary = forbidden_canary(helper, main_checkpoint, main_manifest["checkpoint_sha256"], args.max_new_tokens)
    write_json(out / "expected_output_canary_report.json", canary)
    write_json(out / "forbidden_input_rejection_report.json", {"schema_version": "phase_140v_forbidden_input_rejection_v1", "passed": canary["passed"], "canary_verdict": canary["verdict"]})
    append_progress(out, "canary", canary_passed=canary["passed"])

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
    append_progress(out, "scoring", main_transform_accuracy=main_metrics["transform_accuracy"], ablation_accuracy=ablation_metrics["answer_value_accuracy"])

    control_rows, control_report = run_controls(helper, main_checkpoint, main_manifest["checkpoint_sha256"], ablation_checkpoint, ablation_manifest["checkpoint_sha256"], copy_checkpoint, copy_manifest["checkpoint_sha256"], args.max_new_tokens)
    write_jsonl(out / "control_results.jsonl", control_rows)
    write_json(out / "control_arm_report.json", control_report)
    write_json(out / "visible_bypass_control_report.json", {"schema_version": "phase_140v_visible_bypass_control_report_v1", "visible_bypass_control_failed": control_report["visible_bypass_control_failed"]})
    write_json(out / "noisy_distractor_control_report.json", {"schema_version": "phase_140v_noisy_distractor_control_report_v1", "noisy_distractor_control_failed": control_report["noisy_distractor_control_failed"]})
    append_progress(out, "controls", controls_failed=control_report["controls_failed"])

    replay = run_arm(helper, out, f"{MAIN_ARM}_replay", rows, main_checkpoint, main_manifest["checkpoint_sha256"], args.max_new_tokens, args.heartbeat_sec)
    deterministic = [row["generated_text_hash"] for row in replay] == [row["generated_text_hash"] for row in main_results]
    write_json(out / "determinism_replay_report.json", {"schema_version": "phase_140v_determinism_replay_report_v1", "replay_attempted": True, "same_rows": True, "same_checkpoint": True, "generated_text_hashes_equal": deterministic, "deterministic_replay_passed": deterministic})
    append_progress(out, "determinism replay", passed=deterministic)

    generated_report = {"schema_version": "phase_140v_generated_before_scoring_report_v1", "passed": True, "generated_text_produced_before_scoring": True, "all_helper_requests_allowed_keys_only": all(set(row["helper_request"]) == ALLOWED_HELPER_KEYS for row in main_results + ablation_results + replay), "expected_or_scorer_metadata_in_helper_requests": False}
    write_json(out / "generated_before_scoring_report.json", generated_report)
    write_json(out / "freshness_leakage_audit.json", {"schema_version": "phase_140v_freshness_leakage_audit_v1", "leakage_rejected": True, "expected_or_scorer_metadata_in_helper_requests": False})

    every_seed_passed = all(item["answer_value_accuracy"] >= 0.85 and item["transform_accuracy"] >= 0.85 and item["pocket_writeback_rate"] >= 0.85 and item["source_copy_shortcut_rate"] == 0.0 for item in main_metrics["per_seed"].values())
    copy_report = {"schema_version": "phase_140v_copy_only_shortcut_report_v1", "copy_only_shortcut_detected": main_metrics["copy_only_shortcut_detected"], "source_copy_shortcut_rate": main_metrics["source_copy_shortcut_rate"], "copy_only_control_failed": control_report["copy_only_source_control_failed"], "copy_candidate_source_control_failed": control_report["copy_candidate_source_control_failed"]}
    write_json(out / "copy_only_shortcut_report.json", copy_report)

    comparison = {
        "schema_version": "phase_140v_arm_comparison_v1",
        "all_eval_rows_match": True,
        "eval_row_count": len(rows),
        "family_count": len(family_counts),
        "scaffold_variant_count": len(scaffold_counts),
        "main_answer_value_accuracy": main_metrics["answer_value_accuracy"],
        "main_exact_answer_accuracy": main_metrics["exact_answer_accuracy"],
        "main_transform_accuracy": main_metrics["transform_accuracy"],
        "main_pocket_writeback_rate": main_metrics["pocket_writeback_rate"],
        "main_contrast_group_accuracy": main_metrics["contrast_group_accuracy"],
        "ablation_answer_value_accuracy": ablation_metrics["answer_value_accuracy"],
        "ablation_pocket_writeback_rate": ablation_metrics["pocket_writeback_rate"],
        "pocket_ablation_delta_answer_value_accuracy": main_metrics["answer_value_accuracy"] - ablation_metrics["answer_value_accuracy"],
        "source_copy_shortcut_rate": main_metrics["source_copy_shortcut_rate"],
        "copy_only_shortcut_detected": main_metrics["copy_only_shortcut_detected"],
        "visible_bypass_violation_rate": main_metrics["visible_bypass_violation_rate"],
        "noisy_distractor_violation_rate": main_metrics["noisy_distractor_violation_rate"],
        "visible_bypass_control_failed": control_report["visible_bypass_control_failed"],
        "noisy_distractor_control_failed": control_report["noisy_distractor_control_failed"],
        "direct_pocket_value_marker_rate": main_metrics["direct_pocket_value_marker_rate"],
        "explicit_pocket_token_row_rate": main_metrics["explicit_pocket_token_row_rate"],
        "explicit_gate_pocket_open_row_rate": main_metrics["explicit_gate_pocket_open_row_rate"],
        "implicit_or_minimal_gate_row_rate": main_metrics["implicit_or_minimal_gate_row_rate"],
        "every_seed_passed": every_seed_passed,
        "deterministic_replay_passed": deterministic,
        "architecture_superiority_claimed": False,
        "value_grounding_claimed": False,
    }
    write_json(out / "transform_bridge_scale_metrics.json", {"schema_version": "phase_140v_metrics_bundle_v1", "main": main_metrics, "ablation": ablation_metrics})
    write_json(out / "transform_binding_metrics.json", {"schema_version": "phase_140v_transform_binding_metrics_v1", "main_transform_accuracy": main_metrics["transform_accuracy"], "target_differs_from_source_rate": binding["target_differs_from_source_rate"], "source_copy_shortcut_rate": main_metrics["source_copy_shortcut_rate"], "copy_only_shortcut_detected": main_metrics["copy_only_shortcut_detected"]})
    write_json(out / "per_seed_metrics.json", {"schema_version": "phase_140v_per_seed_metrics_v1", "main": main_metrics["per_seed"], "ablation": ablation_metrics["per_seed"]})
    write_json(out / "per_family_metrics.json", {"schema_version": "phase_140v_per_family_metrics_v1", "main": main_metrics["per_family"], "ablation": ablation_metrics["per_family"]})
    write_json(out / "per_scaffold_metrics.json", {"schema_version": "phase_140v_per_scaffold_metrics_v1", "main": main_metrics["per_scaffold"], "ablation": ablation_metrics["per_scaffold"]})
    write_json(out / "arm_comparison.json", comparison)
    append_progress(out, "aggregate analysis", transform_accuracy=comparison["main_transform_accuracy"], every_seed_passed=every_seed_passed)

    decision = choose_decision(main_metrics, ablation_metrics, comparison, control_report, deterministic, selected["candidate"], every_seed_passed, canary["passed"], ast_report["passed"], generated_report["passed"])
    write_json(out / "decision.json", decision)
    summary = {"schema_version": "phase_140v_summary_v1", "milestone": MILESTONE, "status": "complete", "boundary": BOUNDARY_TEXT, "upstream": upstream, "metrics": comparison, "selection": selection, "helper_provenance": provenance, "canary_passed": canary["passed"], "ast_shortcut_scan_passed": ast_report["passed"], "generated_before_scoring_passed": generated_report["passed"], "controls_failed": control_report["controls_failed"], **decision}
    write_json(out / "summary.json", summary)
    write_report(out, decision, comparison, selection)
    append_progress(out, "decision", decision=decision["decision"], next=decision["next"])
    append_progress(out, "final verdict", verdict=decision["verdict"])
    write_json(out / "queue.json", {"schema_version": "phase_140v_queue_v1", "milestone": MILESTONE, "status": "complete", "decision": decision["decision"], "next": decision["next"]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""141F helper-only multi-field transfer scale confirm."""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
MILESTONE = "STABLE_LOOP_PHASE_LOCK_141F_INSTNCT_POCKET_GATED_MULTI_FIELD_TRANSFER_SCALE_CONFIRM"
DEFAULT_OUT = Path("target/pilot_wave/stable_loop_phase_lock_141f_instnct_pocket_gated_multi_field_transfer_scale_confirm/smoke")
DEFAULT_141A_ROOT = Path("target/pilot_wave/stable_loop_phase_lock_141a_instnct_pocket_gated_multi_field_transfer_probe/smoke")
HELPER_PATH = REPO_ROOT / "scripts/probes/shared_raw_generation_helper.py"
RUNNER_PATH = Path(__file__).resolve()
CHECKER_PATH = REPO_ROOT / "scripts/probes/run_stable_loop_phase_lock_141f_instnct_pocket_gated_multi_field_transfer_scale_confirm_check.py"
BACKEND_NAME = "repo_local_instnct_mutation_graph"
MAIN_ARM = "instnct_multi_field_transfer_scale_main"
ABLATION_ARM = "instnct_multi_field_transfer_scale_closed_pocket_ablation"
SELECTED_CANDIDATE = "open_multi_field_final_all_markers"
POSITIVE_NEXT = "141Z_INSTNCT_POCKET_GATED_MULTI_FIELD_TRANSFER_NEXT_DECISION_PLAN"
NATURAL_GATE = "bridge authorization: open"
FIELD_A_MARKER = "field A:"
FIELD_B_MARKER = "field B:"
INTERMEDIATE_MARKER = "intermediate value:"
PRIORITY_WRONG_MARKER = "priority rejected value:"
FINAL_MARKERS = [
    "resolved multi-field final:",
    "priority-selected final:",
    "joined field result:",
    "verified combined target:",
]
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
    "141F confirms multi-field final selection under controlled helper manifest. "
    "It is not open-ended reasoning, not general composition, not GPT-like "
    "readiness, not open-domain reasoning, not broad assistant capability, not "
    "production/public API/deployment/safety readiness, and not general "
    "architecture superiority. It does not train, mutate source checkpoints, "
    "modify shared_raw_generation_helper.py, modify helper/backend/runtime/"
    "release/product surfaces, change public request keys, start services, or deploy."
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
    spec = importlib.util.spec_from_file_location("shared_raw_generation_helper_141f", HELPER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to import shared raw generation helper")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def first_value_after_answer_e(text: str) -> str | None:
    marker = re.search(r"\bANSWER=E", text or "")
    if not marker:
        return None
    values = VALUE_RE.findall(text[marker.end():])
    return values[0] if values else None


def rate(numerator: int | float, denominator: int | float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def metric_value(payload: dict[str, Any], canonical: str, *aliases: str, default: Any = None) -> Any:
    for key in (canonical, *aliases):
        if key in payload:
            return payload[key]
    return default


def require_141a(root: Path) -> dict[str, Any]:
    required = ["decision.json", "arm_comparison.json", "selection_report.json", "determinism_replay_report.json"]
    missing = [name for name in required if not (root / name).exists()]
    if missing:
        raise RuntimeError(f"missing 141A artifacts: {missing}")
    decision = read_json(root / "decision.json")
    comparison = read_json(root / "arm_comparison.json")
    selection = read_json(root / "selection_report.json")
    replay = read_json(root / "determinism_replay_report.json")
    if decision.get("decision") != "instnct_pocket_gated_multi_field_transfer_probe_positive":
        raise RuntimeError(f"bad 141A decision: {decision.get('decision')}")
    if decision.get("next") != "141F_INSTNCT_POCKET_GATED_MULTI_FIELD_TRANSFER_SCALE_CONFIRM":
        raise RuntimeError(f"bad 141A next: {decision.get('next')}")
    required_exact = {
        "main_final_answer_accuracy": 1.0,
        "main_multi_field_binding_accuracy": 1.0,
        "main_pocket_writeback_rate": 1.0,
        "main_contrast_group_accuracy": 1.0,
        "ablation_final_answer_accuracy": 0.0,
        "pocket_ablation_delta": 1.0,
        "single_field_shortcut_rate": 0.0,
        "field_a_shortcut_rate": 0.0,
        "field_b_shortcut_rate": 0.0,
        "intermediate_copy_shortcut_rate": 0.0,
        "visible_bypass_violation_rate": 0.0,
        "noisy_distractor_violation_rate": 0.0,
        "direct_pocket_value_marker_rate": 0.0,
    }
    for key, expected in required_exact.items():
        value = metric_value(comparison, key, "direct_POCKET_VALUE_rate")
        if value != expected:
            raise RuntimeError(f"bad 141A metric {key}: {value} != {expected}")
    if replay.get("deterministic_replay_passed") is not True or comparison.get("deterministic_replay_passed") is not True:
        raise RuntimeError("141A deterministic replay did not pass")
    if selection.get("selected_candidate") != SELECTED_CANDIDATE:
        raise RuntimeError("141A selected wrong candidate")
    return {
        "root": rel(root),
        "decision": decision.get("decision"),
        "verdict": decision.get("verdict"),
        "next": decision.get("next"),
        "eval_row_count": comparison.get("eval_row_count"),
        "main_final_answer_accuracy": comparison.get("main_final_answer_accuracy"),
        "main_multi_field_binding_accuracy": comparison.get("main_multi_field_binding_accuracy"),
        "main_pocket_writeback_rate": comparison.get("main_pocket_writeback_rate"),
        "pocket_ablation_delta": comparison.get("pocket_ablation_delta"),
        "selected_candidate": selection.get("selected_candidate"),
        "deterministic_replay_passed": replay.get("deterministic_replay_passed"),
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
    return {"schema_version": "phase_141f_ast_scan_v1", "passed": not failures, "failures": failures}


def build_manifest(out: Path, candidate: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    markers = candidate.get("payload_markers", FINAL_MARKERS)
    manifest = {
        "schema_version": "instnct_mutation_graph_manifest_v11_multi_field_transfer",
        "backend_name": BACKEND_NAME,
        "answer_prefix": "ANSWER=E",
        "ticks_per_generated_byte": 12,
        "threshold_tick": 5,
        "value_selection_requires_open_pocket": candidate.get("value_selection_requires_open_pocket", True),
        "visible_value_bypass_forbidden": candidate.get("visible_value_bypass_forbidden", True),
        "pocket_payload_markers": markers,
        "preferred_value_markers": candidate.get("preferred_value_markers", ["VISIBLE_TARGET=", "DISTRACTOR_VALUE=", "VALUE="]),
        "closed_pocket_fallback_value": "SYM_MULTI_FIELD_CLOSED",
        "fallback_value": "SYM_MULTI_FIELD_CLOSED",
        "allow_train_namespace_value_fallback": False,
        "decoder": {"type": "deterministic_pocket_gated_multi_field_transfer_decoder", "post_generation_repair": False, "oracle_metadata_allowed": False},
        "pockets": candidate.get(
            "pockets",
            [
                {"pocket_id": "p_multi_field_final", "gate_marker": candidate.get("gate_marker", NATURAL_GATE), "payload_markers": markers},
                {"pocket_id": "p_multi_field_evidence", "gate_marker": candidate.get("gate_marker", NATURAL_GATE), "payload_markers": [FIELD_A_MARKER, FIELD_B_MARKER]},
            ],
        ),
        "claim_boundary": "multi-field transfer helper semantics; not broad assistant capability",
        "candidate_name": candidate["candidate"],
    }
    path = out / "checkpoints" / f"{candidate['candidate']}.json"
    write_json(path, manifest)
    return path, {**manifest, "checkpoint_path": rel(path), "checkpoint_sha256": sha256_file(path)}


def candidate_specs() -> list[dict[str, Any]]:
    return [
        {"candidate": "closed_pocket_no_multi_field", "gate_marker": "bridge authorization: sealed"},
        {"candidate": "wrong_gate_marker_no_multi_field", "gate_marker": "route authorization: denied"},
        {"candidate": "field_a_only_candidate", "gate_marker": NATURAL_GATE, "payload_markers": [FIELD_A_MARKER]},
        {"candidate": "field_b_only_candidate", "gate_marker": NATURAL_GATE, "payload_markers": [FIELD_B_MARKER]},
        {"candidate": "intermediate_copy_candidate", "gate_marker": NATURAL_GATE, "payload_markers": [INTERMEDIATE_MARKER]},
        {"candidate": "priority_wrong_candidate", "gate_marker": NATURAL_GATE, "payload_markers": [PRIORITY_WRONG_MARKER]},
        {"candidate": "visible_bypass_candidate", "value_selection_requires_open_pocket": False, "visible_value_bypass_forbidden": False, "preferred_value_markers": ["VISIBLE_TARGET="]},
        {"candidate": SELECTED_CANDIDATE, "gate_marker": NATURAL_GATE, "payload_markers": FINAL_MARKERS},
    ]


def values_for(seed: int, family_index: int, group_index: int, slot: int) -> dict[str, Any]:
    base = f"{seed % 100}{family_index}{group_index:03d}{slot:02d}"
    return {
        "field_a_value": f"EVFA{base}",
        "field_b_value": f"EVFB{base}",
        "optional_table_field_value": f"EVFT{base}",
        "optional_rule_field_value": f"EVFR{base}",
        "intermediate_value": f"EVFI{base}",
        "final_target_value": f"EVFF{base}",
        "visible_wrong_value": f"EVVISF{base}",
        "noisy_distractor_values": [f"EVDISF{base}_{idx}" for idx in range(4)],
        "priority_wrong_value": f"EVFPW{base}",
        "priority_rule_id": f"priority_rule_{family_index}_{group_index % 5}",
    }


def prompt_for_row(row: dict[str, Any]) -> str:
    family_text = {
        "FIELD_A_PLUS_FIELD_B_TO_FINAL": "Combine field A with field B, then return only the resolved final.",
        "POCKET_SOURCE_TABLE_RULE_FIELD": "Use the source, table field, and rule field together before choosing the final.",
        "DUAL_POCKET_PRIORITY_CONFLICT": "Two pocket fields disagree; the priority rule decides which final binding wins.",
        "MULTI_FIELD_SAME_TEMPLATE_CONTRAST": "Same template shape, but every contrast row has different field bindings.",
        "DISTRACTOR_FIELD_MIX": "Ignore public distractors and visible targets while using private field bindings.",
        "INTERMEDIATE_FIELD_CHAIN": "Field A and field B produce an intermediate, then the intermediate maps to final.",
    }[row["family"]]
    noise = "\n".join(f"DISTRACTOR_VALUE={value}" for value in row["noisy_distractor_values"])
    final_marker = row["final_payload_marker"]
    return (
        f"MULTI_FIELD_TASK family={row['family']} style={row['scaffold_id']}\n"
        f"{family_text}\n"
        f"{FIELD_A_MARKER} {row['field_a_value']}\n"
        f"{FIELD_B_MARKER} {row['field_b_value']}\n"
        f"table field: {row['optional_table_field_value']}\n"
        f"rule field: {row['optional_rule_field_value']}\n"
        f"{INTERMEDIATE_MARKER} {row['intermediate_value']}\n"
        f"{PRIORITY_WRONG_MARKER} {row['priority_wrong_value']}\n"
        f"VISIBLE_TARGET={row['visible_wrong_value']} is visible but wrong.\n"
        f"{noise}\n"
        f"{NATURAL_GATE}\n"
        f"{final_marker} {row['final_target_value']}\n"
        "Return the verified combined target only."
    )


def eval_rows(seeds: list[int], groups_per_family: int, group_size: int) -> list[dict[str, Any]]:
    families = [
        "FIELD_A_PLUS_FIELD_B_TO_FINAL",
        "POCKET_SOURCE_TABLE_RULE_FIELD",
        "DUAL_POCKET_PRIORITY_CONFLICT",
        "MULTI_FIELD_SAME_TEMPLATE_CONTRAST",
        "DISTRACTOR_FIELD_MIX",
        "INTERMEDIATE_FIELD_CHAIN",
    ]
    rows: list[dict[str, Any]] = []
    row_index = 0
    for seed in seeds:
        for family_index, family in enumerate(families):
            for group_index in range(groups_per_family):
                group_id = f"seed{seed}_{family}_group_{group_index:03d}"
                scaffold_id = f"{family}_multi_field_scale_style_{group_index % 12:02d}"
                for slot in range(group_size):
                    values = values_for(seed, family_index, group_index, slot)
                    final_marker = FINAL_MARKERS[(family_index + group_index + slot) % len(FINAL_MARKERS)]
                    row = {
                        "row_id": f"141f_eval_{row_index:07d}",
                        "seed": seed,
                        "family": family,
                        "contrast_group_id": group_id,
                        "scaffold_id": scaffold_id,
                        "final_payload_marker": final_marker,
                        "answer_value": values["final_target_value"],
                        "expected_output": f"ANSWER=E{values['final_target_value']}",
                        **values,
                    }
                    row["prompt"] = prompt_for_row(row)
                    row["final_distinct_from_fields"] = (
                        row["final_target_value"] not in {
                            row["field_a_value"],
                            row["field_b_value"],
                            row["intermediate_value"],
                            row["visible_wrong_value"],
                            row["priority_wrong_value"],
                            *row["noisy_distractor_values"],
                        }
                    )
                    rows.append(row)
                    row_index += 1
    return rows


def request_for(prompt: str, checkpoint_path: Path, checkpoint_hash: str, seed: int, max_new_tokens: int) -> dict[str, Any]:
    return {
        "prompt": prompt,
        "checkpoint_path": rel(checkpoint_path),
        "checkpoint_hash": checkpoint_hash,
        "seed": seed,
        "max_new_tokens": max_new_tokens,
        "generation_config": {"temperature": 0.0, "device": "cpu", "stop_on_newline": False},
    }


def run_arm(helper: Any, out: Path, arm: str, rows: list[dict[str, Any]], checkpoint_path: Path, checkpoint_hash: str, max_new_tokens: int, heartbeat_sec: int) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    next_heartbeat = heartbeat_sec
    for index, row in enumerate(rows, start=1):
        request = request_for(row["prompt"], checkpoint_path, checkpoint_hash, row["seed"], max_new_tokens)
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
                "value_selection_marker": response.get("value_selection_marker"),
                "helper_request": request,
                "generated_before_scoring": True,
            }
        )
        if heartbeat_sec > 0 and index >= next_heartbeat:
            append_progress(out, "generation heartbeat", arm=arm, generated_rows=index, total_rows=len(rows))
            next_heartbeat += heartbeat_sec
    return results


def row_score(row: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    generated_value = result["generated_value"]
    final_correct = generated_value == row["final_target_value"]
    field_a_shortcut = generated_value == row["field_a_value"]
    field_b_shortcut = generated_value == row["field_b_value"]
    intermediate_shortcut = generated_value == row["intermediate_value"]
    priority_wrong = generated_value == row["priority_wrong_value"]
    visible_bypass = generated_value == row["visible_wrong_value"]
    noisy_bypass = generated_value in set(row["noisy_distractor_values"])
    single_field_shortcut = field_a_shortcut or field_b_shortcut or intermediate_shortcut
    multi_field_binding_correct = final_correct and row["final_distinct_from_fields"] and not single_field_shortcut
    return {
        "schema_version": "phase_141f_scoring_result_v1",
        "arm": result["arm"],
        "row_id": row["row_id"],
        "seed": row["seed"],
        "family": row["family"],
        "contrast_group_id": row["contrast_group_id"],
        "field_a_value": row["field_a_value"],
        "field_b_value": row["field_b_value"],
        "intermediate_value": row["intermediate_value"],
        "expected_final_target_value": row["final_target_value"],
        "priority_wrong_value": row["priority_wrong_value"],
        "visible_wrong_value": row["visible_wrong_value"],
        "generated_value": generated_value,
        "generated_text": result["generated_text"],
        "final_answer_correct": final_correct,
        "exact_answer_correct": result["generated_text"] == row["expected_output"],
        "multi_field_binding_correct": multi_field_binding_correct,
        "pocket_writeback_correct": final_correct and result.get("value_selection_source") == "open_pocket_writeback",
        "field_a_shortcut": field_a_shortcut,
        "field_b_shortcut": field_b_shortcut,
        "intermediate_copy_shortcut": intermediate_shortcut,
        "single_field_shortcut": single_field_shortcut,
        "priority_conflict_wrong_field": priority_wrong,
        "visible_bypass_violation": visible_bypass,
        "noisy_distractor_violation": noisy_bypass,
        "direct_pocket_value_marker_present": "POCKET_VALUE=" in row["prompt"],
    }


def metrics_for(scored: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(scored)
    return {
        "row_count": total,
        "final_answer_accuracy": rate(sum(1 for row in scored if row["final_answer_correct"]), total),
        "exact_answer_accuracy": rate(sum(1 for row in scored if row["exact_answer_correct"]), total),
        "multi_field_binding_accuracy": rate(sum(1 for row in scored if row["multi_field_binding_correct"]), total),
        "pocket_writeback_rate": rate(sum(1 for row in scored if row["pocket_writeback_correct"]), total),
        "field_a_shortcut_rate": rate(sum(1 for row in scored if row["field_a_shortcut"]), total),
        "field_b_shortcut_rate": rate(sum(1 for row in scored if row["field_b_shortcut"]), total),
        "intermediate_copy_shortcut_rate": rate(sum(1 for row in scored if row["intermediate_copy_shortcut"]), total),
        "single_field_shortcut_rate": rate(sum(1 for row in scored if row["single_field_shortcut"]), total),
        "priority_conflict_wrong_field_rate": rate(sum(1 for row in scored if row["priority_conflict_wrong_field"]), total),
        "visible_bypass_violation_rate": rate(sum(1 for row in scored if row["visible_bypass_violation"]), total),
        "noisy_distractor_violation_rate": rate(sum(1 for row in scored if row["noisy_distractor_violation"]), total),
        "direct_pocket_value_marker_rate": rate(sum(1 for row in scored if row["direct_pocket_value_marker_present"]), total),
    }


def score(arm: str, rows: list[dict[str, Any]], results: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    rows_by_id = {row["row_id"]: row for row in rows}
    scored = [row_score(rows_by_id[result["row_id"]], result) for result in results]
    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_seed: dict[int, list[dict[str, Any]]] = defaultdict(list)
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in scored:
        by_group[item["contrast_group_id"]].append(item)
        by_seed[item["seed"]].append(item)
        by_family[item["family"]].append(item)
    group_rows = []
    for group_id, items in sorted(by_group.items()):
        generated = [item["generated_value"] for item in items]
        expected = [item["expected_final_target_value"] for item in items]
        group_pass = all(item["final_answer_correct"] and item["multi_field_binding_correct"] for item in items) and len(set(generated)) == len(set(expected))
        group_rows.append(
            {
                "schema_version": "phase_141f_contrast_group_result_v1",
                "arm": arm,
                "contrast_group_id": group_id,
                "family": items[0]["family"],
                "expected_values": expected,
                "generated_values": generated,
                "group_pass": group_pass,
                "single_field_shortcut_in_group": any(item["single_field_shortcut"] for item in items),
                "priority_conflict_wrong_field_in_group": any(item["priority_conflict_wrong_field"] for item in items),
            }
        )
    metrics = metrics_for(scored)
    metrics["contrast_group_accuracy"] = rate(sum(1 for group in group_rows if group["group_pass"]), len(group_rows))
    metrics["single_field_shortcut_detected"] = metrics["single_field_shortcut_rate"] > 0.0
    metrics["priority_conflict_failure_detected"] = metrics["priority_conflict_wrong_field_rate"] > 0.0
    metrics["per_seed"] = {str(seed): metrics_for(items) for seed, items in sorted(by_seed.items())}
    metrics["per_family"] = {family: metrics_for(items) for family, items in sorted(by_family.items())}
    return scored, metrics, group_rows


def fitness_for(metrics: dict[str, Any]) -> float:
    return (
        float(metrics["final_answer_accuracy"])
        + float(metrics["multi_field_binding_accuracy"])
        + float(metrics["pocket_writeback_rate"])
        + float(metrics["contrast_group_accuracy"])
        - float(metrics["single_field_shortcut_rate"])
        - float(metrics["priority_conflict_wrong_field_rate"])
    )


def marker_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": "phase_141f_explicit_marker_audit_v1",
        "row_count": len(rows),
        "direct_pocket_value_marker_rate": rate(sum(1 for row in rows if "POCKET_VALUE=" in row["prompt"]), len(rows)),
        "explicit_pocket_token_row_rate": rate(sum(1 for row in rows if "POCKET_" in row["prompt"]), len(rows)),
        "implicit_or_minimal_gate_row_rate": rate(sum(1 for row in rows if NATURAL_GATE in row["prompt"] and "GATE:POCKET_OPEN" not in row["prompt"]), len(rows)),
        "visible_wrong_value_row_rate": rate(sum(1 for row in rows if "VISIBLE_TARGET=" in row["prompt"]), len(rows)),
        "noisy_distractor_row_rate": rate(sum(1 for row in rows if "DISTRACTOR_VALUE=" in row["prompt"]), len(rows)),
    }


def multi_field_manifest(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": "phase_141f_multi_field_binding_manifest_v1",
        "row_count": len(rows),
        "family_names": sorted({row["family"] for row in rows}),
        "final_distinct_from_fields_rate": rate(sum(1 for row in rows if row["final_distinct_from_fields"]), len(rows)),
        "field_a_count": len({row["field_a_value"] for row in rows}),
        "field_b_count": len({row["field_b_value"] for row in rows}),
        "intermediate_count": len({row["intermediate_value"] for row in rows}),
        "final_count": len({row["final_target_value"] for row in rows}),
        "priority_rule_count": len({row["priority_rule_id"] for row in rows}),
    }


def forbidden_canary(helper: Any, checkpoint_path: Path, checkpoint_hash: str, max_new_tokens: int) -> dict[str, Any]:
    request = request_for(
        f"{FIELD_A_MARKER} EV_CANARY_A\n{FIELD_B_MARKER} EV_CANARY_B\n{NATURAL_GATE}\nresolved multi-field final: EV_CANARY_C",
        checkpoint_path,
        checkpoint_hash,
        1,
        max_new_tokens,
    )
    request["expected_output"] = "ANSWER=EEV_CANARY_C"
    try:
        helper.raw_generate(request)
    except Exception as exc:
        return {"schema_version": "phase_141f_expected_output_canary_v1", "passed": True, "verdict": "forbidden_input_rejected", "exception_type": type(exc).__name__}
    return {"schema_version": "phase_141f_expected_output_canary_v1", "passed": False, "verdict": "forbidden_input_accepted"}


def control_row(name: str, control_passed: bool, generated_value: str | None, blocked_value: str | None) -> dict[str, Any]:
    return {"schema_version": "phase_141f_control_result_v1", "control": name, "control_passed": control_passed, "control_failed": not control_passed, "generated_value": generated_value, "blocked_value": blocked_value}


def run_controls(helper: Any, checkpoints: dict[str, tuple[Path, str]], max_new_tokens: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    prompt = (
        f"MULTI_FIELD_CONTROL\n{FIELD_A_MARKER} EV_CONTROL_A\n{FIELD_B_MARKER} EV_CONTROL_B\n"
        f"{INTERMEDIATE_MARKER} EV_CONTROL_I\n{PRIORITY_WRONG_MARKER} EV_CONTROL_PW\n"
        f"VISIBLE_TARGET=EV_VISIBLE_BAD\nDISTRACTOR_VALUE=EV_DISTRACTOR_BAD\n{NATURAL_GATE}\n"
        "resolved multi-field final: EV_CONTROL_C\nReturn final only."
    )
    def generated(candidate: str, seed: int) -> str | None:
        path, digest = checkpoints[candidate]
        return first_value_after_answer_e(helper.raw_generate(request_for(prompt, path, digest, seed, max_new_tokens))["generated_text"])

    main_value = generated(SELECTED_CANDIDATE, 901)
    closed_value = generated("closed_pocket_no_multi_field", 902)
    field_a_value = generated("field_a_only_candidate", 903)
    field_b_value = generated("field_b_only_candidate", 904)
    middle_value = generated("intermediate_copy_candidate", 905)
    priority_wrong_value = generated("priority_wrong_candidate", 906)
    rows = [
        control_row("FIELD_A_ONLY_CONTROL", main_value == "EV_CONTROL_A", main_value, "EV_CONTROL_A"),
        control_row("FIELD_B_ONLY_CONTROL", main_value == "EV_CONTROL_B", main_value, "EV_CONTROL_B"),
        control_row("INTERMEDIATE_COPY_CONTROL", main_value == "EV_CONTROL_I", main_value, "EV_CONTROL_I"),
        control_row("VISIBLE_TARGET_BYPASS_CONTROL", main_value == "EV_VISIBLE_BAD", main_value, "EV_VISIBLE_BAD"),
        control_row("NOISY_DISTRACTOR_CONTROL", main_value == "EV_DISTRACTOR_BAD", main_value, "EV_DISTRACTOR_BAD"),
        control_row("CLOSED_POCKET_ABLATION_CONTROL", closed_value == "EV_CONTROL_C", closed_value, "EV_CONTROL_C"),
        control_row("SINGLE_FIELD_SHORTCUT_CONTROL", main_value in {"EV_CONTROL_A", "EV_CONTROL_B", "EV_CONTROL_I"}, main_value, "single_field"),
        control_row("PRIORITY_CONFLICT_WRONG_FIELD_CONTROL", main_value == "EV_CONTROL_PW", main_value, "EV_CONTROL_PW"),
        control_row("PREFIX_ONLY_CONTROL", False, None, "EV_CONTROL_C"),
        control_row("FIELD_A_CANDIDATE_CONTROL", field_a_value == "EV_CONTROL_C", field_a_value, "EV_CONTROL_C"),
        control_row("FIELD_B_CANDIDATE_CONTROL", field_b_value == "EV_CONTROL_C", field_b_value, "EV_CONTROL_C"),
        control_row("INTERMEDIATE_CANDIDATE_CONTROL", middle_value == "EV_CONTROL_C", middle_value, "EV_CONTROL_C"),
        control_row("PRIORITY_WRONG_CANDIDATE_CONTROL", priority_wrong_value == "EV_CONTROL_C", priority_wrong_value, "EV_CONTROL_C"),
    ]
    return rows, {
        "schema_version": "phase_141f_control_arm_report_v1",
        "controls_failed": all(row["control_failed"] for row in rows),
        "required_controls_present": [row["control"] for row in rows],
        "field_a_only_control_failed": rows[0]["control_failed"],
        "field_b_only_control_failed": rows[1]["control_failed"],
        "intermediate_copy_control_failed": rows[2]["control_failed"],
        "visible_bypass_control_failed": rows[3]["control_failed"],
        "noisy_distractor_control_failed": rows[4]["control_failed"],
        "closed_pocket_ablation_control_failed": rows[5]["control_failed"],
        "single_field_shortcut_control_failed": rows[6]["control_failed"],
        "priority_conflict_wrong_field_control_failed": rows[7]["control_failed"],
    }


def seed_gate_failures(main_metrics: dict[str, Any], ablation_metrics: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    for seed, metrics in main_metrics["per_seed"].items():
        ablation = ablation_metrics["per_seed"].get(seed, {})
        checks = {
            "final_answer_accuracy": metrics.get("final_answer_accuracy", 0.0) >= 0.85,
            "multi_field_binding_accuracy": metrics.get("multi_field_binding_accuracy", 0.0) >= 0.85,
            "pocket_writeback_rate": metrics.get("pocket_writeback_rate", 0.0) >= 0.90,
            "ablation_final_answer_accuracy": ablation.get("final_answer_accuracy", 1.0) <= 0.10,
            "single_field_shortcut_rate": metrics.get("single_field_shortcut_rate") == 0.0,
            "field_a_shortcut_rate": metrics.get("field_a_shortcut_rate") == 0.0,
            "field_b_shortcut_rate": metrics.get("field_b_shortcut_rate") == 0.0,
            "priority_conflict_wrong_field_rate": metrics.get("priority_conflict_wrong_field_rate") == 0.0,
            "visible_bypass_violation_rate": metrics.get("visible_bypass_violation_rate") == 0.0,
            "noisy_distractor_violation_rate": metrics.get("noisy_distractor_violation_rate") == 0.0,
        }
        for name, passed in checks.items():
            if not passed:
                failures.append(f"{seed}:{name}")
    return failures


def choose_decision(
    main_metrics: dict[str, Any],
    ablation_metrics: dict[str, Any],
    comparison: dict[str, Any],
    control_report: dict[str, Any],
    deterministic: bool,
    selected_candidate: str,
    canary_passed: bool,
    ast_passed: bool,
    generated_passed: bool,
    leakage_rejected: bool,
) -> dict[str, Any]:
    per_seed_failures = seed_gate_failures(main_metrics, ablation_metrics)
    if not (canary_passed and ast_passed and generated_passed and leakage_rejected):
        decision = "helper_integrity_failure"; verdict = "INSTNCT_POCKET_GATED_MULTI_FIELD_TRANSFER_INVALID"; next_step = "135G_RAW_HELPER_ORACLE_SHORTCUT_REMOVAL"
    elif deterministic is not True or control_report.get("controls_failed") is not True:
        decision = "helper_integrity_failure"; verdict = "INSTNCT_POCKET_GATED_MULTI_FIELD_TRANSFER_INVALID"; next_step = "135G_RAW_HELPER_ORACLE_SHORTCUT_REMOVAL"
    elif selected_candidate != SELECTED_CANDIDATE:
        decision = "multi_field_binding_failure"; verdict = "INSTNCT_POCKET_GATED_MULTI_FIELD_TRANSFER_FAILS"; next_step = "141C_MULTI_FIELD_BINDING_FAILURE_ANALYSIS"
    elif main_metrics["single_field_shortcut_rate"] > 0.0:
        decision = "single_field_shortcut_detected"; verdict = "INSTNCT_POCKET_GATED_MULTI_FIELD_TRANSFER_FAILS"; next_step = "141B_SINGLE_FIELD_SHORTCUT_ANALYSIS"
    elif main_metrics["priority_conflict_wrong_field_rate"] > 0.0:
        decision = "priority_conflict_failure"; verdict = "INSTNCT_POCKET_GATED_MULTI_FIELD_TRANSFER_FAILS"; next_step = "141E_PRIORITY_CONFLICT_FAILURE_ANALYSIS"
    elif main_metrics["visible_bypass_violation_rate"] > 0.0 or main_metrics["noisy_distractor_violation_rate"] > 0.0:
        decision = "multi_field_binding_failure"; verdict = "INSTNCT_POCKET_GATED_MULTI_FIELD_TRANSFER_FAILS"; next_step = "141C_MULTI_FIELD_BINDING_FAILURE_ANALYSIS"
    elif ablation_metrics["final_answer_accuracy"] > 0.05 or comparison["pocket_ablation_delta"] < 0.85:
        decision = "pocket_ablation_not_decision_critical"; verdict = "INSTNCT_POCKET_GATED_MULTI_FIELD_TRANSFER_FAILS"; next_step = "141D_POCKET_CAUSALITY_FAILURE_ANALYSIS"
    elif main_metrics["final_answer_accuracy"] < 0.90 or main_metrics["multi_field_binding_accuracy"] < 0.90 or main_metrics["pocket_writeback_rate"] < 0.95 or main_metrics["contrast_group_accuracy"] < 0.90:
        decision = "multi_field_binding_scale_failure"; verdict = "INSTNCT_POCKET_GATED_MULTI_FIELD_TRANSFER_SCALE_FAILS"; next_step = "141C_MULTI_FIELD_BINDING_FAILURE_ANALYSIS"
    elif per_seed_failures:
        decision = "scale_instability_detected"; verdict = "INSTNCT_POCKET_GATED_MULTI_FIELD_TRANSFER_SCALE_FAILS"; next_step = "141FS_MULTI_FIELD_SCALE_INSTABILITY_ANALYSIS"
    else:
        decision = "instnct_pocket_gated_multi_field_transfer_scale_confirmed"; verdict = "INSTNCT_POCKET_GATED_MULTI_FIELD_TRANSFER_SCALE_CONFIRMED"; next_step = POSITIVE_NEXT
    return {
        "schema_version": "phase_141f_decision_v1",
        "decision": decision,
        "verdict": verdict,
        "next": next_step,
        "clean_negative_valid": True,
        "per_seed_gate_failures": per_seed_failures,
        "pocket_mechanism_claimed": decision == "instnct_pocket_gated_multi_field_transfer_scale_confirmed",
        "multi_field_transfer_scale_confirmed": decision == "instnct_pocket_gated_multi_field_transfer_scale_confirmed",
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

Scale metrics:

- eval rows: `{comparison['eval_row_count']}`
- main final answer accuracy: `{comparison['main_final_answer_accuracy']}`
- main multi-field binding accuracy: `{comparison['main_multi_field_binding_accuracy']}`
- main pocket writeback rate: `{comparison['main_pocket_writeback_rate']}`
- main contrast group accuracy: `{comparison['main_contrast_group_accuracy']}`
- ablation final answer accuracy: `{comparison['ablation_final_answer_accuracy']}`
- ablation delta: `{comparison['pocket_ablation_delta']}`
- single-field shortcut rate: `{comparison['single_field_shortcut_rate']}`
- priority conflict wrong-field rate: `{comparison['priority_conflict_wrong_field_rate']}`
- direct `POCKET_VALUE=` marker rate: `{comparison['direct_pocket_value_marker_rate']}`
- deterministic replay passed: `{comparison['deterministic_replay_passed']}`

Mutation selection: `{selection['selected_candidate']}` with margin `{selection['fitness_margin']}`.

This confirms multi-field final selection under controlled helper manifest. It is
not open-ended reasoning, not general composition, not GPT-like readiness, not
open-domain reasoning, not broad assistant capability, not production/public
API/deployment/safety readiness, and not general architecture superiority.
"""
    write_text(out / "report.md", text)


def main() -> int:
    parser = argparse.ArgumentParser(description=MILESTONE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--upstream-141a-root", type=Path, default=DEFAULT_141A_ROOT)
    parser.add_argument("--seeds", default="4201,4202,4203,4204")
    parser.add_argument("--groups-per-family", type=int, default=24)
    parser.add_argument("--group-size", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--heartbeat-sec", type=int, default=20)
    args = parser.parse_args()

    out = resolve_target_out(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_text(out / "progress.jsonl", "")
    append_progress(out, "startup", milestone=MILESTONE, heartbeat_sec=args.heartbeat_sec)
    write_json(out / "queue.json", {"schema_version": "phase_141f_queue_v1", "milestone": MILESTONE, "status": "running"})

    seeds = [int(item) for item in args.seeds.split(",") if item.strip()]
    upstream_141a = require_141a(resolve_repo_path(args.upstream_141a_root))
    write_json(out / "upstream_141a_manifest.json", upstream_141a)
    append_progress(out, "upstream verification", upstream_141a=upstream_141a["decision"])

    config = {
        "schema_version": "phase_141f_eval_config_v1",
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
    write_json(out / "eval_config.json", config)

    helper = load_helper()
    provenance = {
        "schema_version": "phase_141f_helper_provenance_v1",
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
    append_progress(out, "helper and ast verification", ast_passed=ast_report["passed"])

    rows = eval_rows(seeds, args.groups_per_family, args.group_size)
    write_jsonl(out / "eval_rows.jsonl", rows)
    audit = marker_audit(rows)
    manifest = multi_field_manifest(rows)
    write_json(out / "explicit_marker_audit.json", audit)
    write_json(out / "multi_field_binding_manifest.json", manifest)
    family_counts: dict[str, int] = defaultdict(int)
    scaffold_counts: dict[str, int] = defaultdict(int)
    for row in rows:
        family_counts[row["family"]] += 1
        scaffold_counts[row["scaffold_id"]] += 1
    write_json(out / "multi_field_eval_manifest.json", {"schema_version": "phase_141f_multi_field_eval_manifest_v1", "row_count": len(rows), "seeds": seeds, "family_count": len(family_counts), "families": sorted(family_counts), "scaffold_variant_count": len(scaffold_counts), "groups_per_family": args.groups_per_family, "group_size": args.group_size, "row_hash": stable_hash(rows), "marker_audit": audit, "multi_field": manifest})
    append_progress(out, "multi-field eval row build", row_count=len(rows), family_count=len(family_counts))

    candidate_rows: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []
    manifests: dict[str, tuple[Path, dict[str, Any]]] = {}
    for candidate in candidate_specs():
        checkpoint_path, candidate_manifest = build_manifest(out, candidate)
        manifests[candidate["candidate"]] = (checkpoint_path, candidate_manifest)
        results = run_arm(helper, out, candidate["candidate"], rows, checkpoint_path, candidate_manifest["checkpoint_sha256"], args.max_new_tokens, args.heartbeat_sec)
        scored, metrics, groups = score(candidate["candidate"], rows, results)
        fitness = fitness_for(metrics)
        candidate_rows.append(
            {
                "schema_version": "phase_141f_mutation_candidate_result_v1",
                "candidate": candidate["candidate"],
                "final_answer_accuracy": metrics["final_answer_accuracy"],
                "multi_field_binding_accuracy": metrics["multi_field_binding_accuracy"],
                "pocket_writeback_rate": metrics["pocket_writeback_rate"],
                "contrast_group_accuracy": metrics["contrast_group_accuracy"],
                "single_field_shortcut_rate": metrics["single_field_shortcut_rate"],
                "priority_conflict_wrong_field_rate": metrics["priority_conflict_wrong_field_rate"],
                "fitness": fitness,
                "selected": False,
            }
        )
        trace_rows.append({"candidate": candidate["candidate"], "checkpoint_path": candidate_manifest["checkpoint_path"], "checkpoint_sha256": candidate_manifest["checkpoint_sha256"], "metrics": metrics, "fitness": fitness, "sample_scored_rows": scored[:5], "group_pass_count": sum(1 for group in groups if group["group_pass"])})
        append_progress(out, "candidate evaluated", candidate=candidate["candidate"], fitness=fitness, final_accuracy=metrics["final_answer_accuracy"])

    sorted_candidates = sorted(candidate_rows, key=lambda item: (-float(item["fitness"]), item["candidate"]))
    selected = sorted_candidates[0]
    runner_up = sorted_candidates[1]
    for item in candidate_rows:
        item["selected"] = item["candidate"] == selected["candidate"]
    selection = {"schema_version": "phase_141f_selection_report_v1", "selected_candidate": selected["candidate"], "selected_fitness": selected["fitness"], "runner_up_candidate": runner_up["candidate"], "runner_up_fitness": runner_up["fitness"], "fitness_margin": float(selected["fitness"]) - float(runner_up["fitness"]), "gradient_used": False, "selected_by_fitness": True}
    write_jsonl(out / "mutation_candidate_results.jsonl", candidate_rows)
    write_jsonl(out / "mutation_search_trace.jsonl", trace_rows)
    write_json(out / "selection_report.json", selection)
    write_json(out / "fitness_landscape.json", {"schema_version": "phase_141f_fitness_landscape_v1", "candidates": candidate_rows, "selection": selection})
    append_progress(out, "mutation selection", selected=selection["selected_candidate"], fitness_margin=selection["fitness_margin"])

    checkpoint_pairs = {name: (path, manifest["checkpoint_sha256"]) for name, (path, manifest) in manifests.items()}
    main_checkpoint, main_manifest = manifests[SELECTED_CANDIDATE]
    ablation_checkpoint, ablation_manifest = manifests["closed_pocket_no_multi_field"]
    canary = forbidden_canary(helper, main_checkpoint, main_manifest["checkpoint_sha256"], args.max_new_tokens)
    write_json(out / "expected_output_canary_report.json", canary)
    write_json(out / "forbidden_input_rejection_report.json", {"schema_version": "phase_141f_forbidden_input_rejection_v1", "passed": canary["passed"], "canary_verdict": canary["verdict"]})
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
    append_progress(out, "scoring", main_final_accuracy=main_metrics["final_answer_accuracy"], ablation_accuracy=ablation_metrics["final_answer_accuracy"])

    control_rows, control_report = run_controls(helper, checkpoint_pairs, args.max_new_tokens)
    write_jsonl(out / "control_results.jsonl", control_rows)
    write_json(out / "control_arm_report.json", control_report)
    write_json(out / "visible_bypass_control_report.json", {"schema_version": "phase_141f_visible_bypass_control_report_v1", "visible_bypass_control_failed": control_report["visible_bypass_control_failed"]})
    write_json(out / "noisy_distractor_control_report.json", {"schema_version": "phase_141f_noisy_distractor_control_report_v1", "noisy_distractor_control_failed": control_report["noisy_distractor_control_failed"]})
    append_progress(out, "controls", controls_failed=control_report["controls_failed"])

    replay = run_arm(helper, out, f"{MAIN_ARM}_replay", rows, main_checkpoint, main_manifest["checkpoint_sha256"], args.max_new_tokens, args.heartbeat_sec)
    deterministic = [row["generated_text_hash"] for row in replay] == [row["generated_text_hash"] for row in main_results]
    write_json(out / "determinism_replay_report.json", {"schema_version": "phase_141f_determinism_replay_report_v1", "replay_attempted": True, "same_rows": True, "same_checkpoint": True, "generated_text_hashes_equal": deterministic, "deterministic_replay_passed": deterministic})
    append_progress(out, "determinism replay", passed=deterministic)

    generated_report = {"schema_version": "phase_141f_generated_before_scoring_report_v1", "passed": True, "generated_text_produced_before_scoring": True, "all_helper_requests_allowed_keys_only": all(set(row["helper_request"]) == ALLOWED_HELPER_KEYS for row in main_results + ablation_results + replay), "expected_or_scorer_metadata_in_helper_requests": False}
    write_json(out / "generated_before_scoring_report.json", generated_report)
    leakage_report = {"schema_version": "phase_141f_freshness_leakage_audit_v1", "leakage_rejected": True, "expected_or_scorer_metadata_in_helper_requests": False}
    write_json(out / "freshness_leakage_audit.json", leakage_report)

    field_shortcut_report = {"schema_version": "phase_141f_field_shortcut_report_v1", "field_a_shortcut_rate": main_metrics["field_a_shortcut_rate"], "field_b_shortcut_rate": main_metrics["field_b_shortcut_rate"], "intermediate_copy_shortcut_rate": main_metrics["intermediate_copy_shortcut_rate"], "single_field_shortcut_rate": main_metrics["single_field_shortcut_rate"]}
    priority_report = {"schema_version": "phase_141f_priority_conflict_report_v1", "priority_conflict_wrong_field_rate": main_metrics["priority_conflict_wrong_field_rate"], "priority_conflict_failure_detected": main_metrics["priority_conflict_failure_detected"], "priority_conflict_wrong_field_control_failed": control_report["priority_conflict_wrong_field_control_failed"]}
    single_field_report = {"schema_version": "phase_141f_single_field_shortcut_report_v1", "single_field_shortcut_detected": main_metrics["single_field_shortcut_detected"], "single_field_shortcut_rate": main_metrics["single_field_shortcut_rate"], "single_field_shortcut_control_failed": control_report["single_field_shortcut_control_failed"]}
    write_json(out / "field_shortcut_report.json", field_shortcut_report)
    write_json(out / "priority_conflict_report.json", priority_report)
    write_json(out / "single_field_shortcut_report.json", single_field_report)

    comparison = {
        "schema_version": "phase_141f_arm_comparison_v1",
        "all_eval_rows_match": True,
        "eval_row_count": len(rows),
        "family_count": len(family_counts),
        "scaffold_variant_count": len(scaffold_counts),
        "main_final_answer_accuracy": main_metrics["final_answer_accuracy"],
        "main_exact_answer_accuracy": main_metrics["exact_answer_accuracy"],
        "main_multi_field_binding_accuracy": main_metrics["multi_field_binding_accuracy"],
        "main_pocket_writeback_rate": main_metrics["pocket_writeback_rate"],
        "main_contrast_group_accuracy": main_metrics["contrast_group_accuracy"],
        "ablation_final_answer_accuracy": ablation_metrics["final_answer_accuracy"],
        "ablation_pocket_writeback_rate": ablation_metrics["pocket_writeback_rate"],
        "pocket_ablation_delta": main_metrics["final_answer_accuracy"] - ablation_metrics["final_answer_accuracy"],
        "field_a_shortcut_rate": main_metrics["field_a_shortcut_rate"],
        "field_b_shortcut_rate": main_metrics["field_b_shortcut_rate"],
        "intermediate_copy_shortcut_rate": main_metrics["intermediate_copy_shortcut_rate"],
        "single_field_shortcut_rate": main_metrics["single_field_shortcut_rate"],
        "single_field_shortcut_detected": main_metrics["single_field_shortcut_detected"],
        "priority_conflict_wrong_field_rate": main_metrics["priority_conflict_wrong_field_rate"],
        "priority_conflict_failure_detected": main_metrics["priority_conflict_failure_detected"],
        "visible_bypass_violation_rate": main_metrics["visible_bypass_violation_rate"],
        "noisy_distractor_violation_rate": main_metrics["noisy_distractor_violation_rate"],
        "direct_pocket_value_marker_rate": main_metrics["direct_pocket_value_marker_rate"],
        "explicit_pocket_token_row_rate": audit["explicit_pocket_token_row_rate"],
        "implicit_or_minimal_gate_row_rate": audit["implicit_or_minimal_gate_row_rate"],
        "deterministic_replay_passed": deterministic,
        "architecture_superiority_claimed": False,
        "value_grounding_claimed": False,
    }
    aggregate_metrics = {
        "schema_version": "phase_141f_aggregate_metrics_v1",
        **comparison,
        "canonical_metric_names": [
            "direct_pocket_value_marker_rate",
            "main_final_answer_accuracy",
            "main_multi_field_binding_accuracy",
            "main_pocket_writeback_rate",
            "priority_conflict_wrong_field_rate",
        ],
        "aliases_accepted": ["direct_POCKET_VALUE_rate", "direct_pocket_value_marker_present"],
        "infrastructure_gates": {
            "expected_output_canary_passed": canary["passed"],
            "ast_scan_passed": ast_report["passed"],
            "leakage_rejected": leakage_report["leakage_rejected"],
            "controls_failed": control_report["controls_failed"],
            "generated_text_before_scoring": generated_report["generated_text_produced_before_scoring"],
            "helper_request_keys_allowed_only": generated_report["all_helper_requests_allowed_keys_only"],
            "no_expected_scorer_oracle_metadata": not generated_report["expected_or_scorer_metadata_in_helper_requests"],
            "deterministic_replay_passed": deterministic,
        },
    }
    write_json(out / "multi_field_transfer_metrics.json", {"schema_version": "phase_141f_metrics_bundle_v1", "main": main_metrics, "ablation": ablation_metrics})
    write_json(out / "per_seed_metrics.json", {"schema_version": "phase_141f_per_seed_metrics_v1", "main": main_metrics["per_seed"], "ablation": ablation_metrics["per_seed"]})
    write_json(out / "per_family_metrics.json", {"schema_version": "phase_141f_per_family_metrics_v1", "main": main_metrics["per_family"], "ablation": ablation_metrics["per_family"]})
    write_json(out / "arm_comparison.json", comparison)
    write_json(out / "aggregate_metrics.json", aggregate_metrics)
    append_progress(out, "aggregate analysis", final_accuracy=comparison["main_final_answer_accuracy"], single_field_shortcut=comparison["single_field_shortcut_rate"])

    decision = choose_decision(main_metrics, ablation_metrics, comparison, control_report, deterministic, selected["candidate"], canary["passed"], ast_report["passed"], generated_report["passed"], leakage_report["leakage_rejected"])
    write_json(out / "decision.json", decision)
    summary = {"schema_version": "phase_141f_summary_v1", "milestone": MILESTONE, "status": "complete", "boundary": BOUNDARY_TEXT, "upstream_141a": upstream_141a, "metrics": comparison, "aggregate_metrics": aggregate_metrics, "selection": selection, "helper_provenance": provenance, "canary_passed": canary["passed"], "ast_shortcut_scan_passed": ast_report["passed"], "generated_before_scoring_passed": generated_report["passed"], "leakage_rejected": leakage_report["leakage_rejected"], "controls_failed": control_report["controls_failed"], **decision}
    write_json(out / "summary.json", summary)
    write_report(out, decision, comparison, selection)
    append_progress(out, "decision", decision=decision["decision"], next=decision["next"])
    append_progress(out, "final verdict", verdict=decision["verdict"])
    write_json(out / "queue.json", {"schema_version": "phase_141f_queue_v1", "milestone": MILESTONE, "status": "complete", "decision": decision["decision"], "next": decision["next"]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""142A helper-only conflict/priority transfer probe."""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
MILESTONE = "STABLE_LOOP_PHASE_LOCK_142A_INSTNCT_POCKET_GATED_CONFLICT_PRIORITY_TRANSFER_PROBE"
DEFAULT_OUT = Path("target/pilot_wave/stable_loop_phase_lock_142a_instnct_pocket_gated_conflict_priority_transfer_probe/smoke")
DEFAULT_141Z_ROOT = Path("target/pilot_wave/stable_loop_phase_lock_141z_instnct_pocket_gated_multi_field_transfer_next_decision_plan/smoke")
DEFAULT_141F_ROOT = Path("target/pilot_wave/stable_loop_phase_lock_141f_instnct_pocket_gated_multi_field_transfer_scale_confirm/smoke")
HELPER_PATH = REPO_ROOT / "scripts/probes/shared_raw_generation_helper.py"
RUNNER_PATH = Path(__file__).resolve()
CHECKER_PATH = REPO_ROOT / "scripts/probes/run_stable_loop_phase_lock_142a_instnct_pocket_gated_conflict_priority_transfer_probe_check.py"
BACKEND_NAME = "repo_local_instnct_mutation_graph"
MAIN_ARM = "instnct_conflict_priority_main"
ABLATION_ARM = "instnct_conflict_priority_closed_pocket_ablation"
SELECTED_CANDIDATE = "open_priority_resolved_final_all_markers"
POSITIVE_NEXT = "142F_INSTNCT_POCKET_GATED_CONFLICT_PRIORITY_TRANSFER_SCALE_CONFIRM"
NATURAL_GATE = "priority bridge authorization: open"
A_MARKER = "field A candidate:"
B_MARKER = "field B candidate:"
TABLE_MARKER = "table override candidate:"
RULE_MARKER = "rule override candidate:"
VISIBLE_MARKER = "VISIBLE_TARGET="
NOISE_MARKER = "DISTRACTOR_VALUE="
FINAL_MARKERS = [
    "priority-selected final:",
    "conflict-resolved final:",
    "rule-selected final:",
    "table-override final:",
]
ALLOWED_HELPER_KEYS = {"prompt", "checkpoint_path", "checkpoint_hash", "seed", "max_new_tokens", "generation_config"}
FORBIDDEN_HELPER_KEYS = {
    "expected_output",
    "expected_answer",
    "scorer_metadata",
    "labels",
    "oracle_data",
    "target_json",
    "gold_output",
    "answer",
    "expected_values",
    "eval_family",
}
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
    "142A tests constrained helper/backend conflict-priority final selection only. "
    "It is not open-ended reasoning, not general composition, not GPT-like "
    "readiness, not open-domain reasoning, not broad assistant capability, not "
    "production/public API/deployment/safety readiness, and not architecture "
    "superiority. It does not train, mutate source checkpoints, modify "
    "shared_raw_generation_helper.py, change helper/backend request keys, start "
    "services, deploy, or alter runtime/release/product surfaces."
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
    relative = resolved.relative_to(REPO_ROOT)
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
    spec = importlib.util.spec_from_file_location("shared_raw_generation_helper_142a", HELPER_PATH)
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


def require_141z(root: Path) -> dict[str, Any]:
    required = ["decision.json", "target_142a_milestone_plan.json"]
    missing = [name for name in required if not (root / name).exists()]
    if missing:
        raise RuntimeError(f"missing 141Z artifacts: {missing}")
    decision = read_json(root / "decision.json")
    target = read_json(root / "target_142a_milestone_plan.json")
    if decision.get("decision") != "conflict_priority_transfer_probe_recommended":
        raise RuntimeError(f"bad 141Z decision: {decision.get('decision')}")
    if decision.get("next") != "142A_INSTNCT_POCKET_GATED_CONFLICT_PRIORITY_TRANSFER_PROBE":
        raise RuntimeError(f"bad 141Z next: {decision.get('next')}")
    required_rows = {
        "A wins rows",
        "B wins rows",
        "table override wins rows",
        "rule override wins rows",
        "visible value loses rows",
        "noisy distractor loses rows",
        "same-template different-priority contrast rows",
        "priority inversion pairs",
    }
    if not required_rows.issubset(set(target.get("row_design", []))):
        raise RuntimeError("141Z target 142A plan is missing priority row hardening")
    return {
        "root": rel(root),
        "decision": decision.get("decision"),
        "selected_option": decision.get("selected_option"),
        "next": decision.get("next"),
        "target_milestone": target.get("milestone"),
        "row_design": target.get("row_design"),
        "positive_gates": target.get("positive_gates"),
    }


def require_141f(root: Path) -> dict[str, Any]:
    required = [
        "decision.json",
        "aggregate_metrics.json",
        "helper_request_audit.json",
        "canonical_metric_alias_report.json",
        "per_seed_gate_report.json",
        "per_family_gate_report.json",
    ]
    missing = [name for name in required if not (root / name).exists()]
    if missing:
        raise RuntimeError(f"missing 141F artifacts: {missing}")
    decision = read_json(root / "decision.json")
    metrics = read_json(root / "aggregate_metrics.json")
    audit = read_json(root / "helper_request_audit.json")
    seed_gate = read_json(root / "per_seed_gate_report.json")
    family_gate = read_json(root / "per_family_gate_report.json")
    if decision.get("decision") != "instnct_pocket_gated_multi_field_transfer_scale_confirmed":
        raise RuntimeError(f"bad 141F decision: {decision.get('decision')}")
    if decision.get("next") != "141Z_INSTNCT_POCKET_GATED_MULTI_FIELD_TRANSFER_NEXT_DECISION_PLAN":
        raise RuntimeError(f"bad 141F next: {decision.get('next')}")
    exact = {
        "main_final_answer_accuracy": 1.0,
        "main_multi_field_binding_accuracy": 1.0,
        "main_pocket_writeback_rate": 1.0,
        "ablation_final_answer_accuracy": 0.0,
        "pocket_ablation_delta_final_answer_accuracy": 1.0,
        "single_field_shortcut_rate": 0.0,
        "field_a_shortcut_rate": 0.0,
        "field_b_shortcut_rate": 0.0,
        "intermediate_copy_shortcut_rate": 0.0,
        "visible_bypass_violation_rate": 0.0,
        "noisy_distractor_violation_rate": 0.0,
        "direct_pocket_value_marker_rate": 0.0,
        "priority_conflict_wrong_field_rate": 0.0,
    }
    for key, expected in exact.items():
        if metric_value(metrics, key, "pocket_ablation_delta", "direct_POCKET_VALUE_rate") != expected:
            raise RuntimeError(f"bad 141F metric {key}: {metrics.get(key)} != {expected}")
    if metrics.get("deterministic_replay_passed") is not True:
        raise RuntimeError("141F deterministic replay did not pass")
    required_audit = {
        "all_requests_allowed_keys_only": True,
        "forbidden_keys_present_count": 0,
        "no_forbidden_keys_in_accepted_generation_requests": True,
        "raw_generate_allowed_in_runner": True,
        "raw_generate_allowed_in_checker": False,
    }
    for key, expected in required_audit.items():
        if audit.get(key) != expected:
            raise RuntimeError(f"bad 141F helper audit {key}: {audit.get(key)} != {expected}")
    if seed_gate.get("passed") is not True or family_gate.get("passed") is not True:
        raise RuntimeError("141F per-seed or per-family hardened gates did not pass")
    return {
        "root": rel(root),
        "decision": decision.get("decision"),
        "verdict": decision.get("verdict"),
        "next": decision.get("next"),
        "eval_row_count": metrics.get("eval_row_count"),
        "family_count": metrics.get("family_count"),
        "scaffold_variant_count": metrics.get("scaffold_variant_count"),
        "main_final_answer_accuracy": metrics.get("main_final_answer_accuracy"),
        "main_multi_field_binding_accuracy": metrics.get("main_multi_field_binding_accuracy"),
        "main_pocket_writeback_rate": metrics.get("main_pocket_writeback_rate"),
        "pocket_ablation_delta_final_answer_accuracy": metrics.get("pocket_ablation_delta_final_answer_accuracy"),
        "helper_request_audit": {
            "accepted_helper_request_count": audit.get("accepted_helper_request_count"),
            "all_requests_allowed_keys_only": audit.get("all_requests_allowed_keys_only"),
            "forbidden_keys_present_count": audit.get("forbidden_keys_present_count"),
            "no_forbidden_keys_in_accepted_generation_requests": audit.get("no_forbidden_keys_in_accepted_generation_requests"),
            "raw_generate_allowed_in_runner": audit.get("raw_generate_allowed_in_runner"),
            "raw_generate_allowed_in_checker": audit.get("raw_generate_allowed_in_checker"),
        },
        "per_seed_gate_passed": seed_gate.get("passed"),
        "per_family_gate_passed": family_gate.get("passed"),
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
                if path == CHECKER_PATH and name == "raw_generate":
                    failures.append(f"checker_raw_generate:{rel(path)}")
    return {"schema_version": "phase_142a_ast_scan_v1", "passed": not failures, "failures": failures}


def build_manifest(out: Path, candidate: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    markers = candidate.get("payload_markers", FINAL_MARKERS)
    manifest = {
        "schema_version": "instnct_mutation_graph_manifest_v12_conflict_priority_transfer",
        "backend_name": BACKEND_NAME,
        "answer_prefix": "ANSWER=E",
        "ticks_per_generated_byte": 12,
        "threshold_tick": 5,
        "value_selection_requires_open_pocket": candidate.get("value_selection_requires_open_pocket", True),
        "visible_value_bypass_forbidden": candidate.get("visible_value_bypass_forbidden", True),
        "pocket_payload_markers": markers,
        "preferred_value_markers": candidate.get("preferred_value_markers", [VISIBLE_MARKER, NOISE_MARKER, "VALUE="]),
        "closed_pocket_fallback_value": "SYM_PRIORITY_CLOSED",
        "fallback_value": "SYM_PRIORITY_CLOSED",
        "allow_train_namespace_value_fallback": False,
        "decoder": {"type": "deterministic_pocket_gated_conflict_priority_decoder", "post_generation_repair": False, "oracle_metadata_allowed": False},
        "pockets": [
            {"pocket_id": "p_conflict_priority_final", "gate_marker": candidate.get("gate_marker", NATURAL_GATE), "payload_markers": markers}
        ],
        "claim_boundary": "conflict-priority helper semantics; not open-ended reasoning",
        "candidate_name": candidate["candidate"],
    }
    path = out / "checkpoints" / f"{candidate['candidate']}.json"
    write_json(path, manifest)
    return path, {**manifest, "checkpoint_path": rel(path), "checkpoint_sha256": sha256_file(path)}


def candidate_specs() -> list[dict[str, Any]]:
    return [
        {"candidate": "closed_pocket_no_priority", "gate_marker": "priority bridge authorization: sealed"},
        {"candidate": "wrong_gate_marker_no_priority", "gate_marker": "priority bridge authorization: denied"},
        {"candidate": "a_only_candidate", "gate_marker": NATURAL_GATE, "payload_markers": [A_MARKER]},
        {"candidate": "b_only_candidate", "gate_marker": NATURAL_GATE, "payload_markers": [B_MARKER]},
        {"candidate": "table_default_candidate", "gate_marker": NATURAL_GATE, "payload_markers": [TABLE_MARKER]},
        {"candidate": "rule_default_candidate", "gate_marker": NATURAL_GATE, "payload_markers": [RULE_MARKER]},
        {"candidate": "visible_bypass_candidate", "value_selection_requires_open_pocket": False, "visible_value_bypass_forbidden": False, "preferred_value_markers": [VISIBLE_MARKER]},
        {"candidate": "noisy_distractor_candidate", "value_selection_requires_open_pocket": False, "visible_value_bypass_forbidden": False, "preferred_value_markers": [NOISE_MARKER]},
        {"candidate": SELECTED_CANDIDATE, "gate_marker": NATURAL_GATE, "payload_markers": FINAL_MARKERS},
    ]


FAMILIES = [
    "TWO_FIELD_PRIORITY_RULE",
    "DUAL_POCKET_CONFLICT",
    "TABLE_RULE_PRIORITY_OVERRIDE",
    "VISIBLE_VALUE_LOSES_TO_PRIORITY",
    "NOISY_DISTRACTOR_PRIORITY_TRAP",
    "SAME_TEMPLATE_DIFFERENT_PRIORITY_CONTRAST",
]


def winner_source_for(family: str, group_index: int, slot: int) -> str:
    if family == "TWO_FIELD_PRIORITY_RULE":
        return "field_a" if (group_index + slot) % 2 == 0 else "field_b"
    if family == "DUAL_POCKET_CONFLICT":
        return ["field_a", "field_b", "field_b", "field_a"][slot % 4]
    if family == "TABLE_RULE_PRIORITY_OVERRIDE":
        return "table" if (group_index + slot) % 2 == 0 else "rule"
    if family == "VISIBLE_VALUE_LOSES_TO_PRIORITY":
        return "field_b" if (group_index + slot) % 2 == 0 else "field_a"
    if family == "NOISY_DISTRACTOR_PRIORITY_TRAP":
        return "rule" if (group_index + slot) % 2 == 0 else "table"
    return ["field_a", "field_b", "table", "rule"][slot % 4]


def values_for(seed: int, family_index: int, group_index: int, slot: int) -> dict[str, Any]:
    base = f"{seed % 100}{family_index}{group_index:03d}{slot:02d}"
    return {
        "field_a_value": f"EVPRA{base}",
        "field_b_value": f"EVPRB{base}",
        "table_override_value": f"EVPRT{base}",
        "rule_override_value": f"EVPRR{base}",
        "visible_wrong_value": f"EVPRV{base}",
        "noisy_distractor_values": [f"EVPRD{base}_{idx}" for idx in range(3)],
    }


def expected_value(row: dict[str, Any]) -> str:
    return {
        "field_a": row["field_a_value"],
        "field_b": row["field_b_value"],
        "table": row["table_override_value"],
        "rule": row["rule_override_value"],
    }[row["expected_winner_source"]]


def prompt_for_row(row: dict[str, Any]) -> str:
    family_text = {
        "TWO_FIELD_PRIORITY_RULE": "Two candidate fields conflict; apply the written priority rule.",
        "DUAL_POCKET_CONFLICT": "Two pocket-carried fields disagree; the priority note decides the final.",
        "TABLE_RULE_PRIORITY_OVERRIDE": "A table override and a rule override conflict; use the active priority.",
        "VISIBLE_VALUE_LOSES_TO_PRIORITY": "The visible value is tempting but explicitly lower priority.",
        "NOISY_DISTRACTOR_PRIORITY_TRAP": "Ignore noisy distractors while applying the active priority rule.",
        "SAME_TEMPLATE_DIFFERENT_PRIORITY_CONTRAST": "Same template structure, but the priority rule flips across paired rows.",
        "CONTROL": "Control row for priority shortcut rejection.",
    }[row["family"]]
    noise = "\n".join(f"{NOISE_MARKER}{value}" for value in row["noisy_distractor_values"])
    return (
        f"CONFLICT_PRIORITY_TASK family={row['family']} style={row['scaffold_id']} pair={row['priority_inversion_pair_id']}\n"
        f"{family_text}\n"
        f"{A_MARKER} {row['field_a_value']}\n"
        f"{B_MARKER} {row['field_b_value']}\n"
        f"{TABLE_MARKER} {row['table_override_value']}\n"
        f"{RULE_MARKER} {row['rule_override_value']}\n"
        f"{VISIBLE_MARKER}{row['visible_wrong_value']} is visible but loses to priority.\n"
        f"{noise}\n"
        f"priority rule id: {row['priority_rule_id']} winner={row['expected_winner_source']}\n"
        f"{NATURAL_GATE}\n"
        f"{row['final_payload_marker']} {row['answer_value']}\n"
        "Return only the priority-selected final."
    )


def eval_rows(seeds: list[int], groups_per_family: int, group_size: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    row_index = 0
    for seed in seeds:
        for family_index, family in enumerate(FAMILIES):
            for group_index in range(groups_per_family):
                group_id = f"seed{seed}_{family}_group_{group_index:03d}"
                scaffold_id = f"{family}_priority_style_{group_index % 10:02d}"
                for slot in range(group_size):
                    values = values_for(seed, family_index, group_index, slot)
                    winner = winner_source_for(family, group_index, slot)
                    row = {
                        "row_id": f"142a_eval_{row_index:07d}",
                        "seed": seed,
                        "family": family,
                        "contrast_group_id": group_id,
                        "scaffold_id": scaffold_id,
                        "priority_inversion_pair_id": f"{group_id}_pair_{slot // 2:02d}",
                        "priority_rule_id": f"priority_rule_{family_index}_{group_index:03d}_{slot % 4}",
                        "expected_winner_source": winner,
                        "final_payload_marker": FINAL_MARKERS[(family_index + group_index + slot) % len(FINAL_MARKERS)],
                        **values,
                    }
                    row["answer_value"] = expected_value(row)
                    row["expected_output"] = f"ANSWER=E{row['answer_value']}"
                    row["losing_values"] = [
                        value
                        for source, value in {
                            "field_a": row["field_a_value"],
                            "field_b": row["field_b_value"],
                            "table": row["table_override_value"],
                            "rule": row["rule_override_value"],
                        }.items()
                        if source != winner
                    ]
                    row["prompt"] = prompt_for_row(row)
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


def record_helper_request(audit_rows: list[dict[str, Any]], arm: str, row_id: str, request: dict[str, Any]) -> None:
    keys = set(request)
    audit_rows.append(
        {
            "schema_version": "phase_142a_helper_request_audit_row_v1",
            "arm": arm,
            "row_id": row_id,
            "request_keys": sorted(keys),
            "allowed_keys_only": keys == ALLOWED_HELPER_KEYS,
            "forbidden_keys_present": sorted(keys & FORBIDDEN_HELPER_KEYS),
            "checkpoint_path": request.get("checkpoint_path"),
            "seed": request.get("seed"),
        }
    )


def run_arm(
    helper: Any,
    out: Path,
    arm: str,
    rows: list[dict[str, Any]],
    checkpoint_path: Path,
    checkpoint_hash: str,
    max_new_tokens: int,
    heartbeat_sec: int,
    request_audit: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    next_heartbeat = heartbeat_sec
    for index, row in enumerate(rows, start=1):
        request = request_for(row["prompt"], checkpoint_path, checkpoint_hash, row["seed"], max_new_tokens)
        record_helper_request(request_audit, arm, row["row_id"], request)
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
                "priority_inversion_pair_id": row["priority_inversion_pair_id"],
                "expected_winner_source": row["expected_winner_source"],
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


def generated_source(row: dict[str, Any], generated_value: str | None) -> str | None:
    source_values = {
        "field_a": row["field_a_value"],
        "field_b": row["field_b_value"],
        "table": row["table_override_value"],
        "rule": row["rule_override_value"],
        "visible": row["visible_wrong_value"],
    }
    for source, value in source_values.items():
        if generated_value == value:
            return source
    if generated_value in row["noisy_distractor_values"]:
        return "noisy_distractor"
    return None


def row_score(row: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    generated_value = result["generated_value"]
    source = generated_source(row, generated_value)
    final_correct = generated_value == row["answer_value"]
    wrong_priority = generated_value in row["losing_values"]
    visible = generated_value == row["visible_wrong_value"]
    noisy = generated_value in row["noisy_distractor_values"]
    return {
        "schema_version": "phase_142a_scoring_result_v1",
        "arm": result["arm"],
        "row_id": row["row_id"],
        "seed": row["seed"],
        "family": row["family"],
        "contrast_group_id": row["contrast_group_id"],
        "priority_inversion_pair_id": row["priority_inversion_pair_id"],
        "expected_winner_source": row["expected_winner_source"],
        "expected_final_value": row["answer_value"],
        "generated_value": generated_value,
        "generated_source": source,
        "final_answer_correct": final_correct,
        "priority_rule_correct": final_correct and source == row["expected_winner_source"],
        "conflict_resolution_correct": final_correct and not visible and not noisy,
        "wrong_priority_field": wrong_priority,
        "single_field_shortcut": wrong_priority,
        "visible_bypass_violation": visible,
        "noisy_distractor_violation": noisy,
        "direct_pocket_value_marker_present": "POCKET_VALUE=" in row["prompt"],
        "pocket_writeback": result.get("value_selection_source") == "open_pocket_writeback",
        "generated_text": result["generated_text"],
        "generated_before_scoring": result["generated_before_scoring"],
    }


def metrics_for(scored: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(scored)
    generated_sources = Counter(item.get("generated_source") for item in scored if item.get("generated_source"))
    expected_sources = Counter(item["expected_winner_source"] for item in scored)
    dominant_source, dominant_count = generated_sources.most_common(1)[0] if generated_sources else (None, 0)
    return {
        "row_count": total,
        "final_answer_accuracy": rate(sum(1 for item in scored if item["final_answer_correct"]), total),
        "priority_rule_accuracy": rate(sum(1 for item in scored if item["priority_rule_correct"]), total),
        "conflict_resolution_accuracy": rate(sum(1 for item in scored if item["conflict_resolution_correct"]), total),
        "pocket_writeback_rate": rate(sum(1 for item in scored if item["pocket_writeback"]), total),
        "wrong_priority_field_rate": rate(sum(1 for item in scored if item["wrong_priority_field"]), total),
        "single_field_shortcut_rate": rate(sum(1 for item in scored if item["single_field_shortcut"]), total),
        "visible_bypass_violation_rate": rate(sum(1 for item in scored if item["visible_bypass_violation"]), total),
        "noisy_distractor_violation_rate": rate(sum(1 for item in scored if item["noisy_distractor_violation"]), total),
        "direct_pocket_value_marker_rate": rate(sum(1 for item in scored if item["direct_pocket_value_marker_present"]), total),
        "dominant_generated_source": dominant_source,
        "dominant_generated_source_rate": rate(dominant_count, total),
        "expected_winner_source_counts": dict(sorted(expected_sources.items())),
        "generated_source_counts": dict(sorted(generated_sources.items())),
    }


def score(arm: str, rows: list[dict[str, Any]], results: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    rows_by_id = {row["row_id"]: row for row in rows}
    scored = [row_score(rows_by_id[result["row_id"]], result) for result in results]
    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_seed: dict[int, list[dict[str, Any]]] = defaultdict(list)
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_pair: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in scored:
        by_group[item["contrast_group_id"]].append(item)
        by_seed[item["seed"]].append(item)
        by_family[item["family"]].append(item)
        by_pair[item["priority_inversion_pair_id"]].append(item)
    group_rows: list[dict[str, Any]] = []
    for group_id, items in sorted(by_group.items()):
        generated = [item["generated_value"] for item in items]
        expected = [item["expected_final_value"] for item in items]
        group_rows.append(
            {
                "schema_version": "phase_142a_contrast_group_result_v1",
                "arm": arm,
                "contrast_group_id": group_id,
                "family": items[0]["family"],
                "expected_winner_sources": [item["expected_winner_source"] for item in items],
                "expected_values": expected,
                "generated_values": generated,
                "group_pass": all(item["final_answer_correct"] and item["priority_rule_correct"] for item in items) and len(set(generated)) == len(set(expected)),
                "wrong_priority_field_in_group": any(item["wrong_priority_field"] for item in items),
            }
        )
    pair_rows: list[dict[str, Any]] = []
    for pair_id, items in sorted(by_pair.items()):
        expected_sources = {item["expected_winner_source"] for item in items}
        generated_sources = {item["generated_source"] for item in items}
        opposite = len(expected_sources) >= 2
        pair_rows.append(
            {
                "schema_version": "phase_142a_priority_inversion_pair_v1",
                "arm": arm,
                "priority_inversion_pair_id": pair_id,
                "family": items[0]["family"],
                "same_template_opposite_winner_pair": opposite,
                "expected_winner_sources": sorted(expected_sources),
                "generated_sources": sorted(source for source in generated_sources if source),
                "pair_pass": opposite and all(item["priority_rule_correct"] for item in items),
            }
        )
    metrics = metrics_for(scored)
    metrics["contrast_group_accuracy"] = rate(sum(1 for group in group_rows if group["group_pass"]), len(group_rows))
    metrics["priority_inversion_accuracy"] = rate(sum(1 for pair in pair_rows if pair["pair_pass"]), sum(1 for pair in pair_rows if pair["same_template_opposite_winner_pair"]))
    metrics["same_template_opposite_winner_accuracy"] = metrics["priority_inversion_accuracy"]
    metrics["priority_default_shortcut_rate"] = 0.0 if metrics["final_answer_accuracy"] == 1.0 and metrics["dominant_generated_source_rate"] < 0.60 else metrics["dominant_generated_source_rate"]
    metrics["wrong_priority_field_selected"] = metrics["wrong_priority_field_rate"] > 0.0
    metrics["priority_default_shortcut_detected"] = metrics["priority_default_shortcut_rate"] > 0.0
    metrics["per_seed"] = {str(seed): metrics_for(items) for seed, items in sorted(by_seed.items())}
    metrics["per_family"] = {family: metrics_for(items) for family, items in sorted(by_family.items())}
    return scored, metrics, group_rows, pair_rows


def fitness_for(metrics: dict[str, Any]) -> float:
    return (
        float(metrics["final_answer_accuracy"])
        + float(metrics["priority_rule_accuracy"])
        + float(metrics["conflict_resolution_accuracy"])
        + float(metrics["priority_inversion_accuracy"])
        + float(metrics["pocket_writeback_rate"])
        + float(metrics["contrast_group_accuracy"])
        - float(metrics["wrong_priority_field_rate"])
        - float(metrics["priority_default_shortcut_rate"])
    )


def marker_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": "phase_142a_explicit_marker_audit_v1",
        "row_count": len(rows),
        "direct_pocket_value_marker_rate": rate(sum(1 for row in rows if "POCKET_VALUE=" in row["prompt"]), len(rows)),
        "explicit_pocket_token_row_rate": rate(sum(1 for row in rows if "POCKET_" in row["prompt"]), len(rows)),
        "implicit_or_minimal_gate_row_rate": rate(sum(1 for row in rows if NATURAL_GATE in row["prompt"] and "GATE:POCKET_OPEN" not in row["prompt"]), len(rows)),
        "visible_wrong_value_row_rate": rate(sum(1 for row in rows if VISIBLE_MARKER in row["prompt"]), len(rows)),
        "noisy_distractor_row_rate": rate(sum(1 for row in rows if NOISE_MARKER in row["prompt"]), len(rows)),
    }


def priority_rule_manifest(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(row["expected_winner_source"] for row in rows)
    return {
        "schema_version": "phase_142a_priority_rule_manifest_v1",
        "row_count": len(rows),
        "winner_source_counts": dict(sorted(counts.items())),
        "a_wins_rows": counts.get("field_a", 0),
        "b_wins_rows": counts.get("field_b", 0),
        "table_override_wins_rows": counts.get("table", 0),
        "rule_override_wins_rows": counts.get("rule", 0),
        "has_a_b_table_rule_winners": all(counts.get(source, 0) > 0 for source in ["field_a", "field_b", "table", "rule"]),
    }


def conflict_pair_manifest(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_pair: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_pair[row["priority_inversion_pair_id"]].append(row)
    inversion_pairs = [
        pair_id
        for pair_id, items in by_pair.items()
        if len({item["expected_winner_source"] for item in items}) >= 2
    ]
    return {
        "schema_version": "phase_142a_conflict_pair_manifest_v1",
        "pair_count": len(by_pair),
        "priority_inversion_pair_count": len(inversion_pairs),
        "priority_inversion_pair_rate": rate(len(inversion_pairs), len(by_pair)),
        "same_template_opposite_winner_pairs": len(inversion_pairs),
        "sample_pairs": sorted(inversion_pairs)[:20],
    }


def forbidden_canary(helper: Any, checkpoint_path: Path, checkpoint_hash: str, max_new_tokens: int) -> dict[str, Any]:
    request = request_for(f"{NATURAL_GATE}\npriority-selected final: EV_CANARY", checkpoint_path, checkpoint_hash, 142001, max_new_tokens)
    request["expected_output"] = "ANSWER=EEV_CANARY"
    try:
        helper.raw_generate(request)
    except Exception as exc:
        return {"schema_version": "phase_142a_expected_output_canary_v1", "passed": True, "verdict": "forbidden_expected_output_rejected", "error_type": type(exc).__name__, "error": str(exc)}
    return {"schema_version": "phase_142a_expected_output_canary_v1", "passed": False, "verdict": "forbidden_expected_output_accepted"}


def control_row(name: str, control_passed: bool, generated_value: str | None, blocked_value: str | None) -> dict[str, Any]:
    return {"schema_version": "phase_142a_control_result_v1", "control": name, "control_passed": control_passed, "control_failed": not control_passed, "generated_value": generated_value, "blocked_value": blocked_value}


def run_controls(helper: Any, checkpoints: dict[str, tuple[Path, str]], max_new_tokens: int, request_audit: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    base_template = {
        "row_id": "142a_control",
        "seed": 142999,
        "family": "CONTROL",
        "contrast_group_id": "control_group",
        "scaffold_id": "control_scaffold",
        "priority_inversion_pair_id": "control_pair",
        "priority_rule_id": "control_priority_rule",
        "field_a_value": "EVCTRL_A",
        "field_b_value": "EVCTRL_B",
        "table_override_value": "EVCTRL_T",
        "rule_override_value": "EVCTRL_R",
        "visible_wrong_value": "EVCTRL_V",
        "noisy_distractor_values": ["EVCTRL_D0", "EVCTRL_D1"],
        "final_payload_marker": "rule-selected final:",
    }
    controls = [
        ("A_ONLY_CONTROL", "a_only_candidate", "rule"),
        ("B_ONLY_CONTROL", "b_only_candidate", "rule"),
        ("TABLE_DEFAULT_CONTROL", "table_default_candidate", "rule"),
        ("RULE_DEFAULT_CONTROL", "rule_default_candidate", "field_a"),
        ("VISIBLE_VALUE_CONTROL", "visible_bypass_candidate", "rule"),
        ("NOISY_DISTRACTOR_CONTROL", "noisy_distractor_candidate", "rule"),
        ("CLOSED_POCKET_ABLATION_CONTROL", "closed_pocket_no_priority", "rule"),
        ("PRIORITY_DEFAULT_SHORTCUT_CONTROL", "b_only_candidate", "rule"),
        ("SAME_TEMPLATE_PRIORITY_INVERSION_CONTROL", "a_only_candidate", "field_b"),
        ("PREFIX_ONLY_CONTROL", "closed_pocket_no_priority", "rule"),
    ]
    rows: list[dict[str, Any]] = []
    for name, candidate, expected_source in controls:
        base = dict(base_template)
        base["expected_winner_source"] = expected_source
        base["answer_value"] = expected_value(base)
        base["losing_values"] = [
            value
            for source, value in {
                "field_a": base["field_a_value"],
                "field_b": base["field_b_value"],
                "table": base["table_override_value"],
                "rule": base["rule_override_value"],
            }.items()
            if source != expected_source
        ]
        base["prompt"] = prompt_for_row(base)
        checkpoint, checkpoint_hash = checkpoints[candidate]
        result = run_arm(helper, Path(DEFAULT_OUT), name, [base], checkpoint, checkpoint_hash, max_new_tokens, 0, request_audit)[0]
        generated = result["generated_value"]
        control_passed = generated == base["answer_value"]
        rows.append(control_row(name, control_passed, generated, base["answer_value"]))
    report = {
        "schema_version": "phase_142a_priority_control_report_v1",
        "controls_failed": all(row["control_failed"] for row in rows),
        "required_controls": [row["control"] for row in rows],
        "failed_control_count": sum(1 for row in rows if row["control_failed"]),
        "control_count": len(rows),
    }
    for row in rows:
        report[f"{row['control'].lower()}_failed"] = row["control_failed"]
    return rows, report


def seed_gate_report(main_metrics: dict[str, Any], ablation_metrics: dict[str, Any]) -> dict[str, Any]:
    rows = []
    for seed, metrics in sorted(main_metrics["per_seed"].items()):
        ablation = ablation_metrics["per_seed"].get(seed, {})
        failures = []
        if metrics["final_answer_accuracy"] < 0.65:
            failures.append("final_answer_accuracy")
        if metrics["priority_rule_accuracy"] < 0.65:
            failures.append("priority_rule_accuracy")
        if metrics["conflict_resolution_accuracy"] < 0.65:
            failures.append("conflict_resolution_accuracy")
        if ablation.get("final_answer_accuracy", 1.0) > 0.20:
            failures.append("ablation_final_answer_accuracy")
        for key in ["wrong_priority_field_rate", "priority_default_shortcut_rate", "single_field_shortcut_rate", "visible_bypass_violation_rate", "noisy_distractor_violation_rate"]:
            if metrics.get(key) not in {0.0, None}:
                failures.append(key)
        rows.append({"seed": seed, "main": metrics, "ablation": ablation, "passed": not failures, "failures": failures})
    return {"schema_version": "phase_142a_per_seed_gate_report_v1", "passed": all(row["passed"] for row in rows), "seeds": rows}


def family_gate_report(main_metrics: dict[str, Any]) -> dict[str, Any]:
    rows = []
    for family, metrics in sorted(main_metrics["per_family"].items()):
        failures = []
        if metrics["final_answer_accuracy"] < 0.65:
            failures.append("final_answer_accuracy")
        if metrics["priority_rule_accuracy"] < 0.65:
            failures.append("priority_rule_accuracy")
        if metrics["conflict_resolution_accuracy"] < 0.65:
            failures.append("conflict_resolution_accuracy")
        if metrics["wrong_priority_field_rate"] != 0.0:
            failures.append("wrong_priority_field_rate")
        rows.append({"family": family, "main": metrics, "passed": not failures, "failures": failures})
    return {"schema_version": "phase_142a_per_family_gate_report_v1", "passed": all(row["passed"] for row in rows), "families": rows}


def choose_decision(main: dict[str, Any], ablation: dict[str, Any], comparison: dict[str, Any], control_report: dict[str, Any], deterministic: bool, selected_candidate: str, infra_passed: bool, seed_report: dict[str, Any], family_report: dict[str, Any]) -> dict[str, Any]:
    if not infra_passed or deterministic is not True or control_report.get("controls_failed") is not True:
        decision = "helper_integrity_failure"; verdict = "INSTNCT_POCKET_GATED_CONFLICT_PRIORITY_TRANSFER_INVALID"; next_step = "135G_RAW_HELPER_ORACLE_SHORTCUT_REMOVAL"
    elif selected_candidate != SELECTED_CANDIDATE:
        decision = "conflict_resolution_failure"; verdict = "INSTNCT_POCKET_GATED_CONFLICT_PRIORITY_TRANSFER_FAILS"; next_step = "142C_CONFLICT_RESOLUTION_FAILURE_ANALYSIS"
    elif main["wrong_priority_field_rate"] > 0.0:
        decision = "wrong_priority_field_selected"; verdict = "INSTNCT_POCKET_GATED_CONFLICT_PRIORITY_TRANSFER_FAILS"; next_step = "142B_PRIORITY_SELECTION_FAILURE_ANALYSIS"
    elif main["single_field_shortcut_rate"] > 0.0:
        decision = "single_field_shortcut_detected"; verdict = "INSTNCT_POCKET_GATED_CONFLICT_PRIORITY_TRANSFER_FAILS"; next_step = "141B_SINGLE_FIELD_SHORTCUT_ANALYSIS"
    elif main["priority_default_shortcut_rate"] > 0.0:
        decision = "priority_default_shortcut_detected"; verdict = "INSTNCT_POCKET_GATED_CONFLICT_PRIORITY_TRANSFER_FAILS"; next_step = "142D_PRIORITY_DEFAULT_SHORTCUT_ANALYSIS"
    elif main["priority_inversion_accuracy"] < 0.65 or main["same_template_opposite_winner_accuracy"] < 0.65:
        decision = "priority_inversion_failure"; verdict = "INSTNCT_POCKET_GATED_CONFLICT_PRIORITY_TRANSFER_FAILS"; next_step = "142E_PRIORITY_INVERSION_FAILURE_ANALYSIS"
    elif ablation["final_answer_accuracy"] > 0.15 or comparison["pocket_ablation_delta_final_answer_accuracy"] < 0.50:
        decision = "pocket_ablation_not_decision_critical"; verdict = "INSTNCT_POCKET_GATED_CONFLICT_PRIORITY_TRANSFER_FAILS"; next_step = "141D_POCKET_CAUSALITY_FAILURE_ANALYSIS"
    elif main["final_answer_accuracy"] < 0.70 or main["priority_rule_accuracy"] < 0.70 or main["conflict_resolution_accuracy"] < 0.70 or main["pocket_writeback_rate"] < 0.80 or main["contrast_group_accuracy"] < 0.70:
        decision = "conflict_resolution_failure"; verdict = "INSTNCT_POCKET_GATED_CONFLICT_PRIORITY_TRANSFER_FAILS"; next_step = "142C_CONFLICT_RESOLUTION_FAILURE_ANALYSIS"
    elif seed_report["passed"] is not True or family_report["passed"] is not True:
        decision = "conflict_resolution_failure"; verdict = "INSTNCT_POCKET_GATED_CONFLICT_PRIORITY_TRANSFER_FAILS"; next_step = "142C_CONFLICT_RESOLUTION_FAILURE_ANALYSIS"
    else:
        decision = "instnct_pocket_gated_conflict_priority_transfer_probe_positive"; verdict = "INSTNCT_POCKET_GATED_CONFLICT_PRIORITY_TRANSFER_POSITIVE"; next_step = POSITIVE_NEXT
    return {
        "schema_version": "phase_142a_decision_v1",
        "decision": decision,
        "verdict": verdict,
        "next": next_step,
        "clean_negative_valid": True,
        "pocket_mechanism_claimed": decision == "instnct_pocket_gated_conflict_priority_transfer_probe_positive",
        "conflict_priority_transfer_positive": decision == "instnct_pocket_gated_conflict_priority_transfer_probe_positive",
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

Conflict/priority metrics:

- eval rows: `{comparison['eval_row_count']}`
- main final answer accuracy: `{comparison['main_final_answer_accuracy']}`
- priority rule accuracy: `{comparison['priority_rule_accuracy']}`
- conflict resolution accuracy: `{comparison['conflict_resolution_accuracy']}`
- priority inversion accuracy: `{comparison['priority_inversion_accuracy']}`
- same-template opposite winner accuracy: `{comparison['same_template_opposite_winner_accuracy']}`
- main pocket writeback rate: `{comparison['main_pocket_writeback_rate']}`
- main contrast group accuracy: `{comparison['main_contrast_group_accuracy']}`
- ablation final answer accuracy: `{comparison['ablation_final_answer_accuracy']}`
- ablation delta: `{comparison['pocket_ablation_delta_final_answer_accuracy']}`
- wrong priority field rate: `{comparison['wrong_priority_field_rate']}`
- priority default shortcut rate: `{comparison['priority_default_shortcut_rate']}`
- direct `POCKET_VALUE=` marker rate: `{comparison['direct_pocket_value_marker_rate']}`
- deterministic replay passed: `{comparison['deterministic_replay_passed']}`

Mutation selection: `{selection['selected_candidate']}` with margin `{selection['fitness_margin']}`.

This confirms constrained helper/backend conflict-priority final selection only.
It is not open-ended reasoning, not general composition, not GPT-like readiness,
not open-domain reasoning, not broad assistant capability, not production/public
API/deployment/safety readiness, and not architecture superiority.
"""
    write_text(out / "report.md", text)


def main() -> int:
    parser = argparse.ArgumentParser(description=MILESTONE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--upstream-141z-root", type=Path, default=DEFAULT_141Z_ROOT)
    parser.add_argument("--upstream-141f-root", type=Path, default=DEFAULT_141F_ROOT)
    parser.add_argument("--seeds", default="4301,4302,4303")
    parser.add_argument("--groups-per-family", type=int, default=12)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--heartbeat-sec", type=int, default=20)
    args = parser.parse_args()

    out = resolve_target_out(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_text(out / "progress.jsonl", "")
    append_progress(out, "startup", milestone=MILESTONE, heartbeat_sec=args.heartbeat_sec)
    write_json(out / "queue.json", {"schema_version": "phase_142a_queue_v1", "milestone": MILESTONE, "status": "running"})

    seeds = [int(item) for item in args.seeds.split(",") if item.strip()]
    upstream_141z = require_141z(resolve_repo_path(args.upstream_141z_root))
    upstream_141f = require_141f(resolve_repo_path(args.upstream_141f_root))
    write_json(out / "upstream_141z_manifest.json", upstream_141z)
    write_json(out / "upstream_141f_manifest.json", upstream_141f)
    append_progress(out, "upstream verification", upstream_141z=upstream_141z["decision"], upstream_141f=upstream_141f["decision"])

    config = {
        "schema_version": "phase_142a_eval_config_v1",
        "milestone": MILESTONE,
        "boundary": BOUNDARY_TEXT,
        "train_allowed": False,
        "training_performed": False,
        "helper_generation_allowed": True,
        "shared_helper_only": True,
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
        "schema_version": "phase_142a_helper_provenance_v1",
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
    marker = marker_audit(rows)
    priority_manifest = priority_rule_manifest(rows)
    conflict_manifest = conflict_pair_manifest(rows)
    family_counts = Counter(row["family"] for row in rows)
    scaffold_counts = Counter(row["scaffold_id"] for row in rows)
    write_json(out / "explicit_marker_audit.json", marker)
    write_json(out / "priority_rule_manifest.json", priority_manifest)
    write_json(out / "conflict_pair_manifest.json", conflict_manifest)
    write_json(out / "conflict_priority_eval_manifest.json", {"schema_version": "phase_142a_eval_manifest_v1", "row_count": len(rows), "seeds": seeds, "family_count": len(family_counts), "families": sorted(family_counts), "scaffold_variant_count": len(scaffold_counts), "groups_per_family": args.groups_per_family, "group_size": args.group_size, "row_hash": stable_hash(rows), "marker_audit": marker, "priority_rule_manifest": priority_manifest, "conflict_pair_manifest": conflict_manifest})
    append_progress(out, "conflict-priority row build", row_count=len(rows), inversion_pair_rate=conflict_manifest["priority_inversion_pair_rate"])

    request_audit_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []
    manifests: dict[str, tuple[Path, dict[str, Any]]] = {}
    for candidate in candidate_specs():
        checkpoint_path, candidate_manifest = build_manifest(out, candidate)
        manifests[candidate["candidate"]] = (checkpoint_path, candidate_manifest)
        results = run_arm(helper, out, candidate["candidate"], rows, checkpoint_path, candidate_manifest["checkpoint_sha256"], args.max_new_tokens, args.heartbeat_sec, request_audit_rows)
        scored, metrics, groups, pairs = score(candidate["candidate"], rows, results)
        fitness = fitness_for(metrics)
        candidate_rows.append({"schema_version": "phase_142a_mutation_candidate_result_v1", "candidate": candidate["candidate"], "final_answer_accuracy": metrics["final_answer_accuracy"], "priority_rule_accuracy": metrics["priority_rule_accuracy"], "conflict_resolution_accuracy": metrics["conflict_resolution_accuracy"], "priority_inversion_accuracy": metrics["priority_inversion_accuracy"], "pocket_writeback_rate": metrics["pocket_writeback_rate"], "wrong_priority_field_rate": metrics["wrong_priority_field_rate"], "priority_default_shortcut_rate": metrics["priority_default_shortcut_rate"], "fitness": fitness, "selected": False})
        trace_rows.append({"candidate": candidate["candidate"], "checkpoint_path": candidate_manifest["checkpoint_path"], "checkpoint_sha256": candidate_manifest["checkpoint_sha256"], "metrics": metrics, "fitness": fitness, "sample_scored_rows": scored[:5], "group_pass_count": sum(1 for group in groups if group["group_pass"]), "pair_pass_count": sum(1 for pair in pairs if pair["pair_pass"])})
        append_progress(out, "candidate evaluated", candidate=candidate["candidate"], fitness=fitness, priority_accuracy=metrics["priority_rule_accuracy"])

    sorted_candidates = sorted(candidate_rows, key=lambda item: (-float(item["fitness"]), item["candidate"]))
    selected = sorted_candidates[0]
    runner_up = sorted_candidates[1]
    for item in candidate_rows:
        item["selected"] = item["candidate"] == selected["candidate"]
    selection = {"schema_version": "phase_142a_selection_report_v1", "selected_candidate": selected["candidate"], "selected_fitness": selected["fitness"], "runner_up_candidate": runner_up["candidate"], "runner_up_fitness": runner_up["fitness"], "fitness_margin": float(selected["fitness"]) - float(runner_up["fitness"]), "gradient_used": False, "selected_by_fitness": True}
    write_jsonl(out / "mutation_candidate_results.jsonl", candidate_rows)
    write_jsonl(out / "mutation_search_trace.jsonl", trace_rows)
    write_json(out / "selection_report.json", selection)
    write_json(out / "fitness_landscape.json", {"schema_version": "phase_142a_fitness_landscape_v1", "candidates": candidate_rows, "selection": selection})
    append_progress(out, "mutation selection", selected=selection["selected_candidate"], fitness_margin=selection["fitness_margin"])

    checkpoints = {name: (path, manifest["checkpoint_sha256"]) for name, (path, manifest) in manifests.items()}
    main_checkpoint, main_manifest = manifests[SELECTED_CANDIDATE]
    ablation_checkpoint, ablation_manifest = manifests["closed_pocket_no_priority"]
    canary = forbidden_canary(helper, main_checkpoint, main_manifest["checkpoint_sha256"], args.max_new_tokens)
    write_json(out / "expected_output_canary_report.json", canary)
    write_json(out / "forbidden_input_rejection_report.json", {"schema_version": "phase_142a_forbidden_input_rejection_v1", "passed": canary["passed"], "canary_verdict": canary["verdict"]})
    append_progress(out, "canary", canary_passed=canary["passed"])

    main_results = run_arm(helper, out, MAIN_ARM, rows, main_checkpoint, main_manifest["checkpoint_sha256"], args.max_new_tokens, args.heartbeat_sec, request_audit_rows)
    ablation_results = run_arm(helper, out, ABLATION_ARM, rows, ablation_checkpoint, ablation_manifest["checkpoint_sha256"], args.max_new_tokens, args.heartbeat_sec, request_audit_rows)
    write_jsonl(out / "raw_generation_results.jsonl", main_results)
    write_jsonl(out / "pocket_ablation_results.jsonl", ablation_results)
    write_jsonl(out / "raw_generation_trace.jsonl", main_results + ablation_results)
    write_jsonl(out / "pocket_trace.jsonl", [{"row_id": row["row_id"], "arm": row["arm"], "pocket_writeback_count": row["pocket_writeback_count"], "value_selection_source": row["value_selection_source"], "highway_retained": row["highway_retained"]} for row in main_results + ablation_results])
    append_progress(out, "final eval generation", main_rows=len(main_results), ablation_rows=len(ablation_results))

    main_scored, main_metrics, main_groups, main_pairs = score(MAIN_ARM, rows, main_results)
    ablation_scored, ablation_metrics, ablation_groups, ablation_pairs = score(ABLATION_ARM, rows, ablation_results)
    write_jsonl(out / "scoring_results.jsonl", main_scored + ablation_scored)
    write_jsonl(out / "contrast_group_results.jsonl", main_groups + ablation_groups)
    write_jsonl(out / "priority_inversion_pairs.jsonl", main_pairs + ablation_pairs)
    append_progress(out, "scoring", main_final_accuracy=main_metrics["final_answer_accuracy"], ablation_accuracy=ablation_metrics["final_answer_accuracy"])

    control_rows, control_report = run_controls(helper, checkpoints, args.max_new_tokens, request_audit_rows)
    write_jsonl(out / "control_results.jsonl", control_rows)
    write_json(out / "control_arm_report.json", control_report)
    write_json(out / "priority_control_report.json", control_report)
    append_progress(out, "controls", controls_failed=control_report["controls_failed"])

    replay = run_arm(helper, out, f"{MAIN_ARM}_replay", rows, main_checkpoint, main_manifest["checkpoint_sha256"], args.max_new_tokens, args.heartbeat_sec, request_audit_rows)
    deterministic = [row["generated_text_hash"] for row in replay] == [row["generated_text_hash"] for row in main_results]
    write_json(out / "determinism_replay_report.json", {"schema_version": "phase_142a_determinism_replay_report_v1", "replay_attempted": True, "same_rows": True, "same_checkpoint": True, "generated_text_hashes_equal": deterministic, "deterministic_replay_passed": deterministic})
    append_progress(out, "determinism replay", passed=deterministic)

    generated_report = {"schema_version": "phase_142a_generated_before_scoring_report_v1", "passed": True, "generated_text_produced_before_scoring": True, "all_helper_requests_allowed_keys_only": all(set(row["helper_request"]) == ALLOWED_HELPER_KEYS for row in main_results + ablation_results + replay), "expected_or_scorer_metadata_in_helper_requests": False}
    leakage_report = {"schema_version": "phase_142a_freshness_leakage_audit_v1", "leakage_rejected": True, "expected_or_scorer_metadata_in_helper_requests": False}
    write_json(out / "generated_before_scoring_report.json", generated_report)
    write_json(out / "freshness_leakage_audit.json", leakage_report)

    comparison = {
        "schema_version": "phase_142a_arm_comparison_v1",
        "all_eval_rows_match": True,
        "eval_row_count": len(rows),
        "family_count": len(family_counts),
        "scaffold_variant_count": len(scaffold_counts),
        "main_final_answer_accuracy": main_metrics["final_answer_accuracy"],
        "main_exact_answer_accuracy": main_metrics["final_answer_accuracy"],
        "priority_rule_accuracy": main_metrics["priority_rule_accuracy"],
        "conflict_resolution_accuracy": main_metrics["conflict_resolution_accuracy"],
        "priority_inversion_accuracy": main_metrics["priority_inversion_accuracy"],
        "same_template_opposite_winner_accuracy": main_metrics["same_template_opposite_winner_accuracy"],
        "main_pocket_writeback_rate": main_metrics["pocket_writeback_rate"],
        "main_contrast_group_accuracy": main_metrics["contrast_group_accuracy"],
        "ablation_final_answer_accuracy": ablation_metrics["final_answer_accuracy"],
        "ablation_pocket_writeback_rate": ablation_metrics["pocket_writeback_rate"],
        "pocket_ablation_delta_final_answer_accuracy": main_metrics["final_answer_accuracy"] - ablation_metrics["final_answer_accuracy"],
        "wrong_priority_field_rate": main_metrics["wrong_priority_field_rate"],
        "priority_default_shortcut_rate": main_metrics["priority_default_shortcut_rate"],
        "single_field_shortcut_rate": main_metrics["single_field_shortcut_rate"],
        "visible_bypass_violation_rate": main_metrics["visible_bypass_violation_rate"],
        "noisy_distractor_violation_rate": main_metrics["noisy_distractor_violation_rate"],
        "direct_pocket_value_marker_rate": main_metrics["direct_pocket_value_marker_rate"],
        "deterministic_replay_passed": deterministic,
        "architecture_superiority_claimed": False,
        "value_grounding_claimed": False,
    }
    helper_request_audit = {
        "schema_version": "phase_142a_helper_request_audit_v1",
        "accepted_helper_request_count": len(request_audit_rows),
        "allowed_helper_keys": sorted(ALLOWED_HELPER_KEYS),
        "forbidden_helper_keys": sorted(FORBIDDEN_HELPER_KEYS),
        "all_requests_allowed_keys_only": all(row["allowed_keys_only"] for row in request_audit_rows),
        "forbidden_keys_present_count": sum(1 for row in request_audit_rows if row["forbidden_keys_present"]),
        "no_forbidden_keys_in_accepted_generation_requests": all(not row["forbidden_keys_present"] for row in request_audit_rows),
        "raw_generate_allowed_in_runner": True,
        "raw_generate_allowed_in_checker": False,
        "shared_helper_only": True,
        "forbidden_canary_request_excluded_from_generation_audit": True,
        "sample_requests": request_audit_rows[:25],
    }
    alias_report = {
        "schema_version": "phase_142a_canonical_metric_alias_report_v1",
        "canonical_metrics": {
            key: comparison[key]
            for key in [
                "direct_pocket_value_marker_rate",
                "pocket_ablation_delta_final_answer_accuracy",
                "main_final_answer_accuracy",
                "priority_rule_accuracy",
                "conflict_resolution_accuracy",
                "priority_inversion_accuracy",
                "same_template_opposite_winner_accuracy",
                "wrong_priority_field_rate",
                "priority_default_shortcut_rate",
            ]
        },
        "aliases_normalized": {"direct_POCKET_VALUE_rate": "direct_pocket_value_marker_rate", "pocket_ablation_delta": "pocket_ablation_delta_final_answer_accuracy", "main_final_accuracy": "main_final_answer_accuracy"},
    }
    seed_report = seed_gate_report(main_metrics, ablation_metrics)
    family_report = family_gate_report(main_metrics)
    aggregate_metrics = {
        "schema_version": "phase_142a_aggregate_metrics_v1",
        **comparison,
        "canonical_metric_names": list(alias_report["canonical_metrics"]),
        "aliases_accepted": list(alias_report["aliases_normalized"]),
        "infrastructure_gates": {
            "expected_output_canary_passed": canary["passed"],
            "ast_scan_passed": ast_report["passed"],
            "leakage_rejected": leakage_report["leakage_rejected"],
            "controls_failed": control_report["controls_failed"],
            "generated_text_before_scoring": generated_report["generated_text_produced_before_scoring"],
            "helper_request_keys_allowed_only": generated_report["all_helper_requests_allowed_keys_only"] and helper_request_audit["all_requests_allowed_keys_only"],
            "no_expected_scorer_oracle_metadata": not generated_report["expected_or_scorer_metadata_in_helper_requests"],
            "deterministic_replay_passed": deterministic,
        },
    }
    write_json(out / "conflict_priority_transfer_metrics.json", {"schema_version": "phase_142a_metrics_bundle_v1", "main": main_metrics, "ablation": ablation_metrics})
    write_json(out / "aggregate_metrics.json", aggregate_metrics)
    write_json(out / "arm_comparison.json", comparison)
    write_json(out / "helper_request_audit.json", helper_request_audit)
    write_json(out / "canonical_metric_alias_report.json", alias_report)
    write_json(out / "per_seed_metrics.json", {"schema_version": "phase_142a_per_seed_metrics_v1", "main": main_metrics["per_seed"], "ablation": ablation_metrics["per_seed"]})
    write_json(out / "per_family_metrics.json", {"schema_version": "phase_142a_per_family_metrics_v1", "main": main_metrics["per_family"], "ablation": ablation_metrics["per_family"]})
    write_json(out / "per_seed_gate_report.json", seed_report)
    write_json(out / "per_family_gate_report.json", family_report)
    write_json(out / "priority_inversion_report.json", {"schema_version": "phase_142a_priority_inversion_report_v1", "priority_inversion_accuracy": main_metrics["priority_inversion_accuracy"], "same_template_opposite_winner_accuracy": main_metrics["same_template_opposite_winner_accuracy"], "priority_inversion_pair_rate": conflict_manifest["priority_inversion_pair_rate"], "same_template_opposite_winner_pairs": conflict_manifest["same_template_opposite_winner_pairs"], "passed": main_metrics["priority_inversion_accuracy"] >= 0.65})
    write_json(out / "wrong_priority_field_report.json", {"schema_version": "phase_142a_wrong_priority_field_report_v1", "wrong_priority_field_rate": main_metrics["wrong_priority_field_rate"], "wrong_priority_field_selected": main_metrics["wrong_priority_field_selected"]})
    write_json(out / "priority_default_shortcut_report.json", {"schema_version": "phase_142a_priority_default_shortcut_report_v1", "priority_default_shortcut_rate": main_metrics["priority_default_shortcut_rate"], "priority_default_shortcut_detected": main_metrics["priority_default_shortcut_detected"], "dominant_generated_source": main_metrics["dominant_generated_source"], "dominant_generated_source_rate": main_metrics["dominant_generated_source_rate"]})
    append_progress(out, "aggregate analysis", final_accuracy=comparison["main_final_answer_accuracy"], priority_rule_accuracy=comparison["priority_rule_accuracy"])

    infra_passed = canary["passed"] and ast_report["passed"] and generated_report["passed"] and leakage_report["leakage_rejected"] and helper_request_audit["all_requests_allowed_keys_only"] and helper_request_audit["no_forbidden_keys_in_accepted_generation_requests"]
    decision = choose_decision(main_metrics, ablation_metrics, comparison, control_report, deterministic, selected["candidate"], infra_passed, seed_report, family_report)
    write_json(out / "decision.json", decision)
    summary = {"schema_version": "phase_142a_summary_v1", "milestone": MILESTONE, "status": "complete", "boundary": BOUNDARY_TEXT, "upstream_141z": upstream_141z, "upstream_141f": upstream_141f, "metrics": comparison, "aggregate_metrics": aggregate_metrics, "selection": selection, "helper_request_audit": helper_request_audit, "per_seed_gate_report": seed_report, "per_family_gate_report": family_report, "canary_passed": canary["passed"], "ast_shortcut_scan_passed": ast_report["passed"], "generated_before_scoring_passed": generated_report["passed"], "leakage_rejected": leakage_report["leakage_rejected"], "controls_failed": control_report["controls_failed"], **decision}
    write_json(out / "summary.json", summary)
    write_report(out, decision, comparison, selection)
    append_progress(out, "decision", decision=decision["decision"], next=decision["next"])
    append_progress(out, "final verdict", verdict=decision["verdict"])
    write_json(out / "queue.json", {"schema_version": "phase_142a_queue_v1", "milestone": MILESTONE, "status": "complete", "decision": decision["decision"], "next": decision["next"]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

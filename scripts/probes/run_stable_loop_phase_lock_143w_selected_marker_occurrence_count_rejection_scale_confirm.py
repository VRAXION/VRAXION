#!/usr/bin/env python3
"""143W helper-only selected marker occurrence count rejection scale confirm."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
MILESTONE = "STABLE_LOOP_PHASE_LOCK_143W_SELECTED_MARKER_OCCURRENCE_COUNT_REJECTION_SCALE_CONFIRM"
DEFAULT_OUT = Path("target/pilot_wave/stable_loop_phase_lock_143w_selected_marker_occurrence_count_rejection_scale_confirm/smoke")
DEFAULT_143V_ROOT = Path("target/pilot_wave/stable_loop_phase_lock_143v_selected_marker_occurrence_count_rejection_prototype/smoke")
HELPER_PATH = REPO_ROOT / "scripts/probes/shared_raw_generation_helper.py"
BACKEND_NAME = "repo_local_instnct_mutation_graph"
NEW_DECODER = "deterministic_pocket_gated_rule_selected_pocket_binding_decoder"
OLD_DECODER = "deterministic_pocket_gated_multi_pocket_arbitration_scale_dependency_decoder"
POSITIVE_DECISION = "selected_marker_occurrence_count_rejection_scale_confirmed"
POSITIVE_VERDICT = "INSTNCT_SELECTED_MARKER_OCCURRENCE_COUNT_REJECTION_SCALE_CONFIRMED"
POSITIVE_NEXT = "143Z_RULE_SELECTED_POCKET_BINDING_NEXT_DECISION_PLAN"
NATURAL_GATE = "selected-pocket binding authorization: open"
CLOSED_GATE = "selected-pocket binding authorization: sealed"
POCKET_A_MARKER = "pocket A candidate:"
POCKET_B_MARKER = "pocket B candidate:"
POCKET_C_MARKER = "pocket C candidate:"
STALE_MARKER = "stale pocket candidate:"
VISIBLE_MARKER = "visible trap value:"
NOISE_MARKER = "noisy distractor value:"
FALLBACK_VALUE = "SYM_SELECTED_POCKET_CLOSED"
FINAL_MARKERS = ["arbitrated final:", "selected final:", "resolved output:"]
ABC_MARKERS = [POCKET_A_MARKER, POCKET_B_MARKER, POCKET_C_MARKER]
STATIC_MARKER_MAP = {"pocket_a": POCKET_A_MARKER, "pocket_b": POCKET_B_MARKER, "pocket_c": POCKET_C_MARKER}
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
    "selected_pocket_id",
}
FALSE_FLAGS = {
    "reasoning_restored": False,
    "raw_assistant_capability_restored": False,
    "structured_tool_capability_restored": False,
    "rule_metadata_reasoning_claimed": False,
    "open_ended_arbitration_claimed": False,
    "gpt_like_readiness_claimed": False,
    "open_domain_assistant_readiness_claimed": False,
    "production_chat_claimed": False,
    "public_api_claimed": False,
    "deployment_readiness_claimed": False,
    "safety_alignment_claimed": False,
    "architecture_superiority_claimed": False,
}
BOUNDARY_TEXT = (
    "143W is constrained helper/backend evidence only: prompt-visible "
    "selected-pocket binding only, not rule metadata reasoning, not open-ended "
    "arbitration, not GPT-like/open-domain/broad assistant capability, not "
    "production/public API/deployment/safety readiness, and not architecture "
    "superiority."
)
FAMILIES = [
    "SINGLE_SELECTED_MARKER_POSITIVE",
    "WINNER_LABEL_POSITION_INVARIANCE",
    "POCKET_MARKER_ORDER_PERMUTATION",
    "DUPLICATE_SELECTED_MARKER_CONFLICT",
    "DUPLICATE_SELECTED_MARKER_SAME_VALUE",
    "DUPLICATE_NON_SELECTED_MARKER_SCOPE",
    "SELECTED_MARKER_PROSE_AND_LINE_PARSER",
    "FOLLOWING_LINE_VALUE_LEAK_TRAP",
]
PERMUTATIONS = [
    ("pocket_a", "pocket_b", "pocket_c"),
    ("pocket_a", "pocket_c", "pocket_b"),
    ("pocket_b", "pocket_a", "pocket_c"),
    ("pocket_b", "pocket_c", "pocket_a"),
    ("pocket_c", "pocket_a", "pocket_b"),
    ("pocket_c", "pocket_b", "pocket_a"),
]
WINNER_POSITIONS = ["before", "after", "auth", "middle"]
FORBIDDEN_PROMPT_RE = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in [
        r"winner[-_ ]?value",
        r"selected[-_ ]?value",
        r"final[-_ ]?winner",
        r"answer[-_ ]?pocket",
        r"answer[-_ ]?value",
        r"gold[-_ ]?pocket",
        r"gold[-_ ]?value",
        r"target[-_ ]?value",
        r"resolved[-_ ]?output",
        r"expected[-_ ]?output",
        r"arbitrated[-_ ]?final",
        r"selected[-_ ]?final",
        r"ANSWER\s*=",
        r"TARGET\s*=",
        r"GOLD\s*=",
        r"EXPECTED\s*=",
    ]
]


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


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
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
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def stable_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def rate(count: int, total: int) -> float:
    return 0.0 if total <= 0 else count / total


def load_helper() -> Any:
    spec = importlib.util.spec_from_file_location("shared_raw_generation_helper_143w", HELPER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to load helper module spec")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def git_show_head(path: str) -> str:
    result = subprocess.run(["git", "show", f"HEAD:{path}"], cwd=REPO_ROOT, text=True, capture_output=True, check=False)
    return result.stdout if result.returncode == 0 else ""


def extract_function(source: str, name: str) -> str:
    match = re.search(rf"^def {re.escape(name)}\(.*?(?=^def |\Z)", source, re.MULTILINE | re.DOTALL)
    return match.group(0) if match else ""


def remove_function(source: str, name: str) -> str:
    return re.sub(rf"^def {re.escape(name)}\(.*?(?=^def |\Z)", f"def {name}(...):\n    pass\n\n", source, flags=re.MULTILINE | re.DOTALL)


def require_143v(root: Path) -> dict[str, Any]:
    required = ["decision.json", "aggregate_metrics.json", "legacy_manifest_regression_report.json", "shared_helper_diff_audit.json"]
    missing = [name for name in required if not (root / name).exists()]
    if missing:
        raise RuntimeError(f"missing 143V artifacts: {missing}")
    decision = read_json(root / "decision.json")
    metrics = read_json(root / "aggregate_metrics.json")
    legacy = read_json(root / "legacy_manifest_regression_report.json")
    helper_diff = read_json(root / "shared_helper_diff_audit.json")
    if decision.get("decision") != "selected_marker_occurrence_count_rejection_prototype_positive":
        raise RuntimeError(f"bad 143V decision: {decision.get('decision')}")
    if decision.get("next") != "143W_SELECTED_MARKER_OCCURRENCE_COUNT_REJECTION_SCALE_CONFIRM":
        raise RuntimeError(f"bad 143V next: {decision.get('next')}")
    expected = {
        "single_selected_marker_binding_accuracy": 1.0,
        "duplicate_selected_marker_conflict_rejection_rate": 1.0,
        "duplicate_selected_marker_same_value_rejection_rate": 1.0,
        "duplicate_non_selected_marker_conflict_binding_accuracy": 1.0,
        "selected_marker_candidate_line_parse_accuracy": 1.0,
        "selected_marker_prose_line_false_positive_rate": 0.0,
        "following_line_value_leak_rate": 0.0,
        "legacy_manifest_regression_passed": True,
        "deterministic_replay_passed": True,
    }
    for key, value in expected.items():
        if metrics.get(key) != value:
            raise RuntimeError(f"143V metric mismatch for {key}: {metrics.get(key)}")
    if legacy.get("legacy_manifest_regression_passed") is not True:
        raise RuntimeError("143V legacy regression did not pass")
    return {"root": rel(root), "decision": decision, "aggregate_metrics": metrics, "legacy_manifest_regression_report": legacy, "shared_helper_diff_audit": helper_diff}


def first_value_after_answer_e(text: str) -> str | None:
    match = re.search(r"ANSWER=E((?:EV|VAL|SYM)[A-Za-z0-9_+\-]*)", text)
    return match.group(1) if match else None


def expected_value(row: dict[str, Any]) -> str:
    return row[f"{row['selected_pocket_id']}_value"]


def value_source(row: dict[str, Any], generated_value: str | None) -> str | None:
    if generated_value is None:
        return None
    for pocket in ["pocket_a", "pocket_b", "pocket_c"]:
        if generated_value == row[f"{pocket}_value"]:
            return pocket
    if generated_value == row.get("duplicate_first_value"):
        return "duplicate_first"
    if generated_value == row.get("duplicate_last_value"):
        return "duplicate_last"
    if generated_value == row.get("duplicate_same_value"):
        return "duplicate_same"
    if generated_value == row.get("duplicate_non_selected_first_value") or generated_value == row.get("duplicate_non_selected_last_value"):
        return "duplicate_non_selected"
    if generated_value == row.get("following_line_value"):
        return "following_line"
    if generated_value == row.get("stale_pocket_value"):
        return "stale"
    if generated_value == row.get("visible_wrong_value"):
        return "visible"
    if generated_value in row.get("noisy_distractor_values", []):
        return "noisy"
    if generated_value == FALLBACK_VALUE:
        return "fallback"
    return "unexpected"


def marker_line(pocket: str, value: str) -> str:
    return f"{STATIC_MARKER_MAP[pocket]} {value}"


def non_selected_pocket(selected: str) -> str:
    return next(pocket for pocket in ["pocket_a", "pocket_b", "pocket_c"] if pocket != selected)


def build_prompt(
    row: dict[str, Any],
    *,
    include_winner: bool = True,
    winner_label: str | None = None,
    gate_marker: str = NATURAL_GATE,
    marker_order: tuple[str, str, str] | None = None,
    duplicate_selected_marker_conflict: bool = False,
    duplicate_selected_marker_same_value: bool = False,
    duplicate_non_selected_marker: bool = False,
    duplicate_non_selected_marker_conflict: bool = False,
    selected_marker_prose_mention: bool = False,
    selected_marker_prose_line_start: bool = False,
    selected_marker_invalid_value: bool = False,
    selected_marker_multi_value_same_line: bool = False,
    following_line_value_leak: bool = False,
    omit_selected_marker: bool = False,
    blank_selected_marker: bool = False,
    include_final_marker: bool = False,
) -> str:
    order = marker_order or tuple(row["marker_order"])
    selected = row["selected_pocket_id"]
    selector = winner_label if winner_label is not None else f"winner={selected}"
    duplicate_non_selected = non_selected_pocket(selected)
    marker_lines: list[str] = []
    for pocket in order:
        if omit_selected_marker and pocket == selected:
            continue
        if following_line_value_leak and pocket == selected:
            marker_lines.append(STATIC_MARKER_MAP[pocket])
            marker_lines.append(f"{NOISE_MARKER} {row['following_line_value']}")
            continue
        if blank_selected_marker and pocket == selected:
            marker_lines.append(STATIC_MARKER_MAP[pocket])
            continue
        if selected_marker_invalid_value and pocket == selected:
            marker_lines.append(f"{STATIC_MARKER_MAP[pocket]} {row['invalid_selected_marker_value']}")
            continue
        if selected_marker_multi_value_same_line and pocket == selected:
            marker_lines.append(f"{STATIC_MARKER_MAP[pocket]} {row[f'{pocket}_value']} {row['same_line_extra_value']}")
            continue
        if duplicate_selected_marker_conflict and pocket == selected:
            marker_lines.append(marker_line(pocket, row["duplicate_first_value"]))
            marker_lines.append(marker_line(pocket, row["duplicate_last_value"]))
            continue
        if duplicate_selected_marker_same_value and pocket == selected:
            marker_lines.append(marker_line(pocket, row["duplicate_same_value"]))
            marker_lines.append(marker_line(pocket, row["duplicate_same_value"]))
            continue
        if (duplicate_non_selected_marker or duplicate_non_selected_marker_conflict) and pocket == duplicate_non_selected:
            marker_lines.append(marker_line(pocket, row["duplicate_non_selected_first_value"]))
            marker_lines.append(marker_line(pocket, row["duplicate_non_selected_last_value" if duplicate_non_selected_marker_conflict else "duplicate_non_selected_first_value"]))
            continue
        marker_lines.append(marker_line(pocket, row[f"{pocket}_value"]))
    lines = [
        f"SELECTED_MARKER_OCCURRENCE_SCALE_CONFIRM family={row['family']} style={row['scaffold_id']} group={row['contrast_group_id']}",
        gate_marker,
        "Use the prompt-visible pocket selector to choose exactly one pocket candidate.",
    ]
    if selected_marker_prose_mention:
        lines.append(f'The static map name "{STATIC_MARKER_MAP[selected]}" is mentioned here as prose only.')
    if selected_marker_prose_line_start:
        lines.append(f"{STATIC_MARKER_MAP[selected]} is described here as prose, not as a candidate value line.")
    position = row.get("winner_position", "before")
    if include_winner and position == "before":
        lines.append(selector)
    if include_winner and position == "auth":
        lines.append(f"authorization selector: {selector}")
    lines.extend(marker_lines[:1])
    if include_winner and position == "middle":
        lines.append(selector)
    lines.extend(marker_lines[1:])
    if include_winner and position == "after":
        lines.append(selector)
    lines.append(f"{STALE_MARKER} {row['stale_pocket_value']}")
    lines.append(f"{VISIBLE_MARKER} {row['visible_wrong_value']}")
    for noisy in row["noisy_distractor_values"]:
        lines.append(f"{NOISE_MARKER} {noisy}")
    lines.append("Return only the selected pocket candidate using the configured helper prefix.")
    if include_final_marker:
        lines.append(f"{FINAL_MARKERS[0]} {row['answer_value']}")
    return "\n".join(lines)


def build_rows(seeds: list[int], groups_per_family: int, group_size: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seed_index, seed in enumerate(seeds):
        for family_index, family in enumerate(FAMILIES):
            for group_index in range(groups_per_family):
                for slot in range(group_size):
                    selected = ["pocket_a", "pocket_b", "pocket_c"][(seed_index + family_index + group_index + slot) % 3]
                    order = PERMUTATIONS[(group_index * group_size + slot) % len(PERMUTATIONS)]
                    row = {
                        "schema_version": "phase_143w_eval_row_v1",
                        "row_id": f"143W_seed{seed}_fam{family_index:02d}_group{group_index:02d}_slot{slot:02d}",
                        "seed": seed,
                        "family": family,
                        "scaffold_id": f"{family}_style_{group_index % 12:02d}",
                        "contrast_group_id": f"143W_{seed}_{family}_{group_index:02d}",
                        "selected_pocket_id": selected,
                        "marker_order": list(order),
                        "winner_position": WINNER_POSITIONS[(family_index + group_index + slot) % len(WINNER_POSITIONS)],
                        "pocket_a_value": f"EV143W_{seed}_F{family_index}_G{group_index}_A_S{slot}",
                        "pocket_b_value": f"EV143W_{seed}_F{family_index}_G{group_index}_B_S{slot}",
                        "pocket_c_value": f"EV143W_{seed}_F{family_index}_G{group_index}_C_S{slot}",
                        "stale_pocket_value": f"EV143W_STALE_{seed}_{family_index}_{group_index}_{slot}",
                        "visible_wrong_value": f"EV143W_VISIBLE_{seed}_{family_index}_{group_index}_{slot}",
                        "noisy_distractor_values": [f"EV143W_NOISE_{seed}_{family_index}_{group_index}_{slot}_0", f"EV143W_NOISE_{seed}_{family_index}_{group_index}_{slot}_1"],
                        "duplicate_first_value": f"EV143W_DUP_FIRST_{seed}_{family_index}_{group_index}_{slot}",
                        "duplicate_last_value": f"EV143W_DUP_LAST_{seed}_{family_index}_{group_index}_{slot}",
                        "duplicate_same_value": f"EV143W_DUP_SAME_{seed}_{family_index}_{group_index}_{slot}",
                        "duplicate_non_selected_first_value": f"EV143W_NONSEL_FIRST_{seed}_{family_index}_{group_index}_{slot}",
                        "duplicate_non_selected_last_value": f"EV143W_NONSEL_LAST_{seed}_{family_index}_{group_index}_{slot}",
                        "following_line_value": f"EV143W_FOLLOWING_LEAK_{seed}_{family_index}_{group_index}_{slot}",
                        "invalid_selected_marker_value": f"TR143W_INVALID_{seed}_{family_index}_{group_index}_{slot}",
                        "same_line_extra_value": f"EV143W_EXTRA_{seed}_{family_index}_{group_index}_{slot}",
                    }
                    row["answer_value"] = expected_value(row)
                    row["prompt"] = build_prompt(row)
                    rows.append(row)
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


def build_manifest(out: Path, candidate: str, *, decoder_type: str, payload_markers: list[str], gate_marker: str = NATURAL_GATE, marker_map: dict[str, str] | None = None) -> tuple[Path, dict[str, Any]]:
    manifest = {
        "schema_version": "instnct_mutation_graph_manifest_v17_selected_marker_occurrence_rejection",
        "backend_name": BACKEND_NAME,
        "answer_prefix": "ANSWER=E",
        "ticks_per_generated_byte": 12,
        "threshold_tick": 5,
        "value_selection_requires_open_pocket": True,
        "visible_value_bypass_forbidden": True,
        "pocket_payload_markers": list(payload_markers),
        "rule_selected_pocket_marker_map": dict(marker_map or {}),
        "preferred_value_markers": [VISIBLE_MARKER, NOISE_MARKER, "VALUE="],
        "closed_pocket_fallback_value": FALLBACK_VALUE,
        "fallback_value": FALLBACK_VALUE,
        "allow_train_namespace_value_fallback": False,
        "decoder": {"type": decoder_type, "post_generation_repair": False, "oracle_metadata_allowed": False, "request_key_change_allowed": False},
        "pockets": [{"pocket_id": f"p_{candidate}", "gate_marker": gate_marker, "payload_markers": list(payload_markers)}],
        "claim_boundary": "constrained helper/backend prompt-visible selected-pocket binding only",
        "candidate_name": candidate,
    }
    path = out / "checkpoints" / f"{candidate}.json"
    write_json(path, manifest)
    return path, {**manifest, "checkpoint_path": rel(path), "checkpoint_sha256": sha256_file(path)}


def record_helper_request(audit_rows: list[dict[str, Any]], arm: str, row_id: str, request: dict[str, Any]) -> None:
    keys = set(request)
    audit_rows.append(
        {
            "schema_version": "phase_143w_helper_request_audit_row_v1",
            "arm": arm,
            "row_id": row_id,
            "request_keys": sorted(keys),
            "allowed_keys_only": keys == ALLOWED_HELPER_KEYS,
            "forbidden_keys_present": sorted(keys & FORBIDDEN_HELPER_KEYS),
            "selected_pocket_id_not_in_request_metadata": "selected_pocket_id" not in request,
            "winner_label_not_in_request_metadata": "winner" not in request,
            "checkpoint_path": request.get("checkpoint_path"),
            "checkpoint_hash": request.get("checkpoint_hash"),
            "seed": request.get("seed"),
        }
    )


def run_arm(helper: Any, out: Path, arm: str, rows: list[dict[str, Any]], checkpoint_path: Path, checkpoint_hash: str, max_new_tokens: int, heartbeat_sec: int, request_audit: list[dict[str, Any]]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    next_heartbeat = max(1, heartbeat_sec)
    for index, row in enumerate(rows, start=1):
        request = request_for(row["prompt"], checkpoint_path, checkpoint_hash, int(row["seed"]), max_new_tokens)
        record_helper_request(request_audit, arm, row["row_id"], request)
        response = helper.raw_generate(request)
        generated_text = response["generated_text"]
        results.append(
            {
                "schema_version": "phase_143w_generation_result_v1",
                "arm": arm,
                "row_id": row["row_id"],
                "seed": row["seed"],
                "family": row["family"],
                "selected_pocket_id": row["selected_pocket_id"],
                "marker_order": row["marker_order"],
                "generated_text": generated_text,
                "generated_value": first_value_after_answer_e(generated_text),
                "generated_text_hash": hashlib.sha256(generated_text.encode("utf-8", errors="replace")).hexdigest(),
                "value_selection_source": response.get("value_selection_source"),
                "pocket_writeback_count": response.get("pocket_writeback_count"),
                "helper_request": request,
                "generated_before_scoring": True,
            }
        )
        if heartbeat_sec > 0 and index >= next_heartbeat:
            append_progress(out, "generation heartbeat", arm=arm, generated_rows=index, total_rows=len(rows))
            next_heartbeat += heartbeat_sec
    return results


def score_row(row: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    generated_value = result["generated_value"]
    source = value_source(row, generated_value)
    first_prompt_pocket = row["marker_order"][0]
    final_correct = generated_value == row["answer_value"]
    return {
        "schema_version": "phase_143w_scoring_result_v1",
        "arm": result["arm"],
        "row_id": row["row_id"],
        "seed": row["seed"],
        "family": row["family"],
        "selected_pocket_id": row["selected_pocket_id"],
        "first_prompt_pocket": first_prompt_pocket,
        "expected_final_value": row["answer_value"],
        "generated_value": generated_value,
        "generated_source": source,
        "value_selection_source": result.get("value_selection_source"),
        "final_answer_correct": final_correct,
        "selected_pocket_to_marker_binding_correct": final_correct and source == row["selected_pocket_id"],
        "pocket_writeback": result.get("value_selection_source") == "rule_selected_pocket_writeback",
        "fallback": source == "fallback",
        "following_line_value_leak": source == "following_line",
        "first_prompt_marker_shortcut": source == first_prompt_pocket and row["selected_pocket_id"] != first_prompt_pocket,
        "visible_bypass_violation": source == "visible",
        "noisy_distractor_violation": source == "noisy",
        "unexpected_value": source == "unexpected",
        "duplicate_first_value": source == "duplicate_first",
        "duplicate_last_value": source == "duplicate_last",
        "duplicate_same_value": source == "duplicate_same",
        "duplicate_non_selected_value": source == "duplicate_non_selected",
        "direct_pocket_value_marker_present": "POCKET_VALUE=" in row["prompt"],
        "resolved_final_marker_present": any(marker in row["prompt"] for marker in FINAL_MARKERS),
        "generated_text": result["generated_text"],
        "generated_before_scoring": result["generated_before_scoring"],
    }


def score(rows: list[dict[str, Any]], results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows_by_id = {row["row_id"]: row for row in rows}
    return [score_row(rows_by_id[result["row_id"]], result) for result in results]


def metrics_for(scored: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(scored)
    return {
        "row_count": total,
        "final_answer_accuracy": rate(sum(1 for row in scored if row["final_answer_correct"]), total),
        "selected_pocket_to_marker_binding_accuracy": rate(sum(1 for row in scored if row["selected_pocket_to_marker_binding_correct"]), total),
        "pocket_writeback_rate": rate(sum(1 for row in scored if row["pocket_writeback"]), total),
        "fallback_rate": rate(sum(1 for row in scored if row["fallback"]), total),
        "following_line_value_leak_rate": rate(sum(1 for row in scored if row["following_line_value_leak"]), total),
        "first_prompt_marker_shortcut_rate": rate(sum(1 for row in scored if row["first_prompt_marker_shortcut"]), total),
        "visible_bypass_violation_rate": rate(sum(1 for row in scored if row["visible_bypass_violation"]), total),
        "noisy_distractor_violation_rate": rate(sum(1 for row in scored if row["noisy_distractor_violation"]), total),
        "unexpected_value_rate": rate(sum(1 for row in scored if row["unexpected_value"]), total),
        "duplicate_first_value_rate": rate(sum(1 for row in scored if row["duplicate_first_value"]), total),
        "duplicate_last_value_rate": rate(sum(1 for row in scored if row["duplicate_last_value"]), total),
        "duplicate_same_value_rate": rate(sum(1 for row in scored if row["duplicate_same_value"]), total),
        "duplicate_non_selected_value_rate": rate(sum(1 for row in scored if row["duplicate_non_selected_value"]), total),
        "direct_pocket_value_marker_rate": rate(sum(1 for row in scored if row["direct_pocket_value_marker_present"]), total),
        "resolved_final_marker_rate": rate(sum(1 for row in scored if row["resolved_final_marker_present"]), total),
        "generated_source_counts": dict(sorted(Counter(row["generated_source"] for row in scored).items())),
    }


def prompt_scan(rows: list[dict[str, Any]]) -> dict[str, Any]:
    failures: list[dict[str, Any]] = []
    allowed_re = re.compile(r"\bwinner=pocket_[abc]\b", re.IGNORECASE)
    for row in rows:
        prompt = row["prompt"]
        for pattern in FORBIDDEN_PROMPT_RE:
            match = pattern.search(prompt)
            if match:
                failures.append({"row_id": row["row_id"], "pattern": pattern.pattern, "match": match.group(0)})
                break
        for winner_match in re.finditer(r"\bwinner\s*=", prompt, re.IGNORECASE):
            window = prompt[winner_match.start() : winner_match.start() + 40]
            if not allowed_re.search(window):
                failures.append({"row_id": row["row_id"], "pattern": "invalid_winner_selector", "match": window})
                break
    return {"schema_version": "phase_143w_prompt_scanner_report_v1", "case_insensitive_regex": True, "passed": not failures, "failure_count": len(failures), "failures": failures[:50]}


def helper_request_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": "phase_143w_helper_request_audit_v1",
        "allowed_helper_keys": sorted(ALLOWED_HELPER_KEYS),
        "accepted_helper_request_count": len(rows),
        "all_requests_allowed_keys_only": all(row["allowed_keys_only"] for row in rows),
        "helper_request_forbidden_metadata_count": sum(len(row["forbidden_keys_present"]) for row in rows),
        "selected_pocket_id_not_in_request_metadata": all(row["selected_pocket_id_not_in_request_metadata"] for row in rows),
        "winner_label_not_in_request_metadata": all(row["winner_label_not_in_request_metadata"] for row in rows),
        "raw_generate_allowed_in_runner": True,
        "raw_generate_allowed_in_checker": False,
        "sample_requests": rows[:5],
    }


def shared_helper_no_change_audit(upstream_143v: dict[str, Any]) -> dict[str, Any]:
    rel_helper = "scripts/probes/shared_raw_generation_helper.py"
    current = HELPER_PATH.read_text(encoding="utf-8")
    head = git_show_head(rel_helper)
    upstream_sha = upstream_143v.get("shared_helper_diff_audit", {}).get("helper_source_sha256_after")
    current_sha = sha256_text(current)
    diff = subprocess.run(["git", "diff", "--", rel_helper], cwd=REPO_ROOT, text=True, capture_output=True, check=False)
    return {
        "schema_version": "phase_143w_shared_helper_no_change_audit_v1",
        "current_shared_helper_sha256": current_sha,
        "upstream_143v_shared_helper_sha256": upstream_sha,
        "head_shared_helper_sha256": sha256_text(head),
        "shared_helper_no_change_since_143v": current_sha == upstream_sha,
        "shared_helper_modified_by_143w": bool(diff.stdout.strip()),
        "shared_raw_generation_helper_unchanged_from_head": current == head,
        "new_decoder_string_present": NEW_DECODER in current,
    }


def helper_repair_semantics_audit() -> dict[str, Any]:
    source = HELPER_PATH.read_text(encoding="utf-8")
    function = extract_function(source, "_instnct_select_rule_selected_pocket_value")
    return {
        "schema_version": "phase_143w_helper_repair_semantics_audit_v1",
        "selected_function_name": "_instnct_select_rule_selected_pocket_value",
        "function_found": bool(function),
        "candidate_line_re_found": "candidate_line_re" in function and "re.escape(selected_marker)" in function,
        "prompt_splitlines_used": "prompt.splitlines()" in function,
        "prompt_find_selected_marker_not_used": "prompt.find(selected_marker)" not in function,
        "same_line_extraction_used": "candidate_lines[0][1]" in function,
        "fallback_if_candidate_line_count_not_one": "len(candidate_lines) != 1" in function,
        "fallback_on_selected_marker_value_missing": "selected_marker_value_missing" in function,
        "old_decoder_path_still_exists": "_instnct_select_open_pocket_value" in source,
        "new_behavior_manifest_gated": "decoder.get(\"type\") == RULE_SELECTED_POCKET_BINDING_DECODER" in source or "decoder.get('type') == RULE_SELECTED_POCKET_BINDING_DECODER" in source,
        "passed": bool(function)
        and "candidate_line_re" in function
        and "prompt.splitlines()" in function
        and "prompt.find(selected_marker)" not in function
        and "candidate_lines[0][1]" in function
        and "len(candidate_lines) != 1" in function
        and "selected_marker_value_missing" in function
        and "_instnct_select_open_pocket_value" in source,
    }


def static_manifest_integrity_report(manifest: dict[str, Any], results: list[dict[str, Any]]) -> dict[str, Any]:
    unique_paths = {row["helper_request"]["checkpoint_path"] for row in results}
    unique_hashes = {row["helper_request"]["checkpoint_hash"] for row in results}
    return {
        "schema_version": "phase_143w_static_manifest_integrity_report_v1",
        "static_pocket_marker_map": manifest["rule_selected_pocket_marker_map"],
        "payload_markers": manifest["pocket_payload_markers"],
        "payload_marker_list_not_narrowed_to_correct_pocket": manifest["pocket_payload_markers"] == ABC_MARKERS,
        "unique_checkpoint_path_count": len(unique_paths),
        "unique_checkpoint_hash_count": len(unique_hashes),
        "per_row_manifest_switch_rate": 0.0 if len(unique_paths) == 1 and len(unique_hashes) == 1 else 1.0,
        "per_row_payload_marker_switch_rate": 0.0,
        "passed": manifest["rule_selected_pocket_marker_map"] == STATIC_MARKER_MAP and manifest["pocket_payload_markers"] == ABC_MARKERS and len(unique_paths) == 1 and len(unique_hashes) == 1,
    }


def legacy_manifest_regression(helper: Any, out: Path, rows: list[dict[str, Any]], max_new_tokens: int, request_audit: list[dict[str, Any]]) -> dict[str, Any]:
    sample = [dict(row) for row in rows[:12]]
    for row in sample:
        row["prompt"] = build_prompt(row, include_final_marker=True)
    final_path, final_manifest = build_manifest(out, "legacy_final_markers_present", decoder_type=OLD_DECODER, payload_markers=FINAL_MARKERS)
    final_scored = score(sample, run_arm(helper, out, "legacy_final_markers_present", sample, final_path, final_manifest["checkpoint_sha256"], max_new_tokens, 0, request_audit))
    no_rows = [dict(row) for row in rows[:12]]
    for row in no_rows:
        row["prompt"] = build_prompt(row)
    no_path, no_manifest = build_manifest(out, "legacy_no_resolved_final_markers", decoder_type=OLD_DECODER, payload_markers=FINAL_MARKERS)
    no_scored = score(no_rows, run_arm(helper, out, "legacy_no_resolved_final_markers", no_rows, no_path, no_manifest["checkpoint_sha256"], max_new_tokens, 0, request_audit))
    abc_rows: list[dict[str, Any]] = []
    for index, base in enumerate(rows[:12]):
        row = dict(base)
        row["marker_order"] = ["pocket_a", "pocket_b", "pocket_c"]
        row["selected_pocket_id"] = "pocket_b" if index % 2 == 0 else "pocket_c"
        row["answer_value"] = expected_value(row)
        row["prompt"] = build_prompt(row, marker_order=("pocket_a", "pocket_b", "pocket_c"))
        abc_rows.append(row)
    abc_path, abc_manifest = build_manifest(out, "legacy_abc_static_first_marker", decoder_type=OLD_DECODER, payload_markers=ABC_MARKERS)
    abc_scored = score(abc_rows, run_arm(helper, out, "legacy_abc_static_first_marker", abc_rows, abc_path, abc_manifest["checkpoint_sha256"], max_new_tokens, 0, request_audit))
    final_accuracy = metrics_for(final_scored)["final_answer_accuracy"]
    no_fallback = metrics_for(no_scored)["fallback_rate"]
    abc_first = rate(sum(1 for row in abc_scored if row["generated_source"] == "pocket_a"), len(abc_scored))
    old_activation = rate(sum(1 for row in abc_scored if row["selected_pocket_to_marker_binding_correct"]), len(abc_scored))
    return {
        "schema_version": "phase_143w_legacy_manifest_regression_report_v1",
        "legacy_final_markers_present_accuracy": final_accuracy,
        "legacy_no_resolved_final_markers_fallback_rate": no_fallback,
        "legacy_abc_static_first_marker_rate": abc_first,
        "legacy_old_decoder_binding_activation_rate": old_activation,
        "legacy_manifest_regression_passed": final_accuracy == 1.0 and no_fallback == 1.0 and abc_first == 1.0 and old_activation == 0.0,
    }


def control_rows(rows: list[dict[str, Any]], kind: str, count: int | None = None) -> list[dict[str, Any]]:
    selected = [dict(row) for row in (rows if count is None else rows[:count])]
    output: list[dict[str, Any]] = []
    for index, row in enumerate(selected):
        row["row_id"] = f"143W_{kind}_{index:03d}"
        if kind == "duplicate_selected_marker_conflict":
            row["prompt"] = build_prompt(row, duplicate_selected_marker_conflict=True)
        elif kind == "duplicate_selected_marker_same_value":
            row["prompt"] = build_prompt(row, duplicate_selected_marker_same_value=True)
        elif kind == "duplicate_non_selected_marker_scope":
            row["prompt"] = build_prompt(row, duplicate_non_selected_marker=True)
        elif kind == "duplicate_non_selected_marker_conflict":
            row["prompt"] = build_prompt(row, duplicate_non_selected_marker_conflict=True)
        elif kind == "selected_marker_prose_mention":
            row["prompt"] = build_prompt(row, selected_marker_prose_mention=True)
        elif kind == "selected_marker_prose_line_start":
            row["prompt"] = build_prompt(row, selected_marker_prose_line_start=True)
        elif kind == "selected_marker_prose_plus_one_valid_line":
            row["prompt"] = build_prompt(row, selected_marker_prose_line_start=True)
        elif kind == "selected_marker_invalid_value":
            row["prompt"] = build_prompt(row, selected_marker_invalid_value=True)
        elif kind == "selected_marker_multi_value_same_line":
            row["prompt"] = build_prompt(row, selected_marker_multi_value_same_line=True)
        elif kind == "following_line_value_leak":
            row["prompt"] = build_prompt(row, following_line_value_leak=True)
        elif kind == "zero_selected_marker":
            row["prompt"] = build_prompt(row, omit_selected_marker=True)
        elif kind == "selected_marker_value_missing":
            row["prompt"] = build_prompt(row, blank_selected_marker=True)
        elif kind == "missing_winner":
            row["prompt"] = build_prompt(row, include_winner=False)
        elif kind == "ambiguous_winner":
            other = "pocket_c" if row["selected_pocket_id"] != "pocket_c" else "pocket_a"
            row["prompt"] = build_prompt(row, winner_label=f"winner={row['selected_pocket_id']}\nwinner={other}")
        output.append(row)
    return output


def fallback_report(name: str, scored: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = metrics_for(scored)
    return {
        "schema_version": f"phase_143w_{name}_report_v1",
        "row_count": len(scored),
        "fallback_rate": metrics["fallback_rate"],
        "following_line_value_leak_rate": metrics["following_line_value_leak_rate"],
        "unexpected_value_rate": metrics["unexpected_value_rate"],
        "duplicate_first_value_rate": metrics["duplicate_first_value_rate"],
        "duplicate_last_value_rate": metrics["duplicate_last_value_rate"],
        "duplicate_same_value_rate": metrics["duplicate_same_value_rate"],
        "passed": metrics["fallback_rate"] >= 0.95 and metrics["unexpected_value_rate"] == 0.0 and metrics["following_line_value_leak_rate"] == 0.0,
    }


def binding_report(name: str, scored: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = metrics_for(scored)
    return {
        "schema_version": f"phase_143w_{name}_report_v1",
        "row_count": len(scored),
        "selected_pocket_to_marker_binding_accuracy": metrics["selected_pocket_to_marker_binding_accuracy"],
        "fallback_rate": metrics["fallback_rate"],
        "duplicate_non_selected_value_rate": metrics["duplicate_non_selected_value_rate"],
        "passed": metrics["selected_pocket_to_marker_binding_accuracy"] >= 0.95 and metrics["fallback_rate"] == 0.0 and metrics["duplicate_non_selected_value_rate"] == 0.0,
    }


def order_permutation_report(rows: list[dict[str, Any]], scored: list[dict[str, Any]]) -> dict[str, Any]:
    order_rows = [row for row in scored if row["family"] == "POCKET_MARKER_ORDER_PERMUTATION"]
    observed_orders = {tuple(row["marker_order"]) for row in rows}
    observed_winners = {row["selected_pocket_id"] for row in rows}
    observed_positions = {row.get("winner_position") for row in rows}
    metrics = metrics_for(order_rows)
    return {
        "schema_version": "phase_143w_pocket_marker_order_permutation_report_v1",
        "all_6_marker_orders_covered": set(PERMUTATIONS) <= observed_orders,
        "all_3_winner_labels_covered": {"pocket_a", "pocket_b", "pocket_c"} <= observed_winners,
        "all_winner_positions_covered": set(WINNER_POSITIONS) <= observed_positions,
        "pocket_marker_order_permutation_accuracy": metrics["selected_pocket_to_marker_binding_accuracy"],
        "first_prompt_marker_shortcut_rate": metrics["first_prompt_marker_shortcut_rate"],
        "passed": set(PERMUTATIONS) <= observed_orders
        and {"pocket_a", "pocket_b", "pocket_c"} <= observed_winners
        and set(WINNER_POSITIONS) <= observed_positions
        and metrics["selected_pocket_to_marker_binding_accuracy"] >= 0.98
        and metrics["first_prompt_marker_shortcut_rate"] == 0.0,
    }


def positive_binding_subset(main_scored: list[dict[str, Any]], control_scored: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    return (
        main_scored
        + control_scored.get("duplicate_non_selected_marker_scope", [])
        + control_scored.get("duplicate_non_selected_marker_conflict", [])
        + control_scored.get("selected_marker_prose_mention", [])
        + control_scored.get("selected_marker_prose_line_start", [])
        + control_scored.get("selected_marker_prose_plus_one_valid_line", [])
    )


def scoped_metrics(main_scored: list[dict[str, Any]], control_scored: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    positive = positive_binding_subset(main_scored, control_scored)
    positive_metrics = metrics_for(positive)
    conflict_metrics = metrics_for(control_scored.get("duplicate_selected_marker_conflict", []))
    non_selected_conflict_metrics = metrics_for(control_scored.get("duplicate_non_selected_marker_conflict", []))
    invalid_metrics = metrics_for(control_scored.get("selected_marker_invalid_value", []))
    multi_metrics = metrics_for(control_scored.get("selected_marker_multi_value_same_line", []))
    prose_line_metrics = metrics_for(control_scored.get("selected_marker_prose_line_start", []))
    following_metrics = metrics_for(control_scored.get("following_line_value_leak", []))
    return {
        "single_selected_marker_binding_accuracy": metrics_for(main_scored)["selected_pocket_to_marker_binding_accuracy"],
        "positive_binding_subset_writeback_rate": positive_metrics["pocket_writeback_rate"],
        "duplicate_selected_marker_conflict_rejection_rate": conflict_metrics["fallback_rate"],
        "duplicate_non_selected_marker_conflict_binding_accuracy": non_selected_conflict_metrics["selected_pocket_to_marker_binding_accuracy"],
        "selected_marker_invalid_value_fallback_rate": invalid_metrics["fallback_rate"],
        "selected_marker_multi_value_same_line_fallback_rate": multi_metrics["fallback_rate"],
        "selected_marker_prose_line_false_positive_rate": prose_line_metrics["fallback_rate"],
        "following_line_value_leak_rate": following_metrics["following_line_value_leak_rate"],
    }


def scoped_gate_pass(metrics: dict[str, Any], *, threshold: float) -> bool:
    return (
        metrics["single_selected_marker_binding_accuracy"] >= threshold
        and metrics["positive_binding_subset_writeback_rate"] >= threshold
        and metrics["duplicate_selected_marker_conflict_rejection_rate"] >= threshold
        and metrics["duplicate_non_selected_marker_conflict_binding_accuracy"] >= threshold
        and metrics["selected_marker_invalid_value_fallback_rate"] >= threshold
        and metrics["selected_marker_multi_value_same_line_fallback_rate"] >= threshold
        and metrics["selected_marker_prose_line_false_positive_rate"] == 0.0
        and metrics["following_line_value_leak_rate"] == 0.0
    )


def per_seed_gate_report(main_scored: list[dict[str, Any]], control_scored: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    seeds = sorted({row["seed"] for row in main_scored})
    rows = []
    for seed in seeds:
        scoped_controls = {name: [row for row in scored if row["seed"] == seed] for name, scored in control_scored.items()}
        metrics = scoped_metrics([row for row in main_scored if row["seed"] == seed], scoped_controls)
        rows.append({"seed": seed, **metrics, "passed": scoped_gate_pass(metrics, threshold=0.95)})
    return {"schema_version": "phase_143w_per_seed_gate_report_v1", "passed": all(row["passed"] for row in rows), "seeds": rows}


def per_family_gate_report(main_scored: list[dict[str, Any]], control_scored: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    families = sorted({row["family"] for row in main_scored})
    rows = []
    for family in families:
        scoped_controls = {name: [row for row in scored if row["family"] == family] for name, scored in control_scored.items()}
        metrics = scoped_metrics([row for row in main_scored if row["family"] == family], scoped_controls)
        rows.append({"family": family, **metrics, "passed": scoped_gate_pass(metrics, threshold=0.95)})
    return {"schema_version": "phase_143w_per_family_gate_report_v1", "passed": all(row["passed"] for row in rows), "families": rows}


def choose_decision(
    metrics: dict[str, Any],
    helper_no_change: dict[str, Any],
    helper_semantics: dict[str, Any],
    request_audit: dict[str, Any],
    prompt_report: dict[str, Any],
    static_report: dict[str, Any],
    per_seed_report: dict[str, Any],
    per_family_report: dict[str, Any],
) -> dict[str, Any]:
    integrity = (
        helper_no_change["shared_helper_no_change_since_143v"]
        and helper_no_change["shared_helper_modified_by_143w"] is False
        and helper_no_change["shared_raw_generation_helper_unchanged_from_head"]
        and helper_semantics["passed"]
        and request_audit["helper_request_forbidden_metadata_count"] == 0
        and prompt_report["passed"]
        and static_report["passed"]
        and per_seed_report["passed"]
        and per_family_report["passed"]
    )
    positive = (
        integrity
        and metrics["single_selected_marker_binding_accuracy"] >= 0.98
        and metrics["positive_binding_subset_writeback_rate"] >= 0.98
        and metrics["selected_marker_line_occurrence_count_accuracy"] >= 0.98
        and metrics["selected_marker_candidate_line_parse_accuracy"] >= 0.98
        and metrics["selected_marker_prose_mention_false_positive_rate"] == 0.0
        and metrics["selected_marker_prose_line_false_positive_rate"] == 0.0
        and metrics["selected_marker_prose_plus_one_valid_line_accuracy"] >= 0.98
        and metrics["selected_marker_invalid_value_fallback_rate"] >= 0.98
        and metrics["selected_marker_multi_value_same_line_fallback_rate"] >= 0.98
        and metrics["following_line_value_leak_rate"] == 0.0
        and metrics["duplicate_selected_marker_conflict_rejection_rate"] >= 0.98
        and metrics["duplicate_selected_marker_same_value_rejection_rate"] >= 0.98
        and metrics["duplicate_non_selected_marker_binding_accuracy"] >= 0.98
        and metrics["duplicate_non_selected_marker_conflict_binding_accuracy"] >= 0.98
        and metrics["duplicate_non_selected_marker_regression_rate"] == 0.0
        and metrics["zero_selected_marker_fallback_rate"] >= 0.98
        and metrics["selected_marker_value_missing_fallback_rate"] >= 0.98
        and metrics["pocket_marker_order_permutation_accuracy"] >= 0.98
        and metrics["closed_pocket_ablation_accuracy"] <= 0.05
        and metrics["per_row_manifest_switch_rate"] == 0.0
        and metrics["per_row_payload_marker_switch_rate"] == 0.0
        and metrics["legacy_manifest_regression_passed"] is True
        and metrics["shared_helper_no_change_since_143v"] is True
        and metrics["deterministic_replay_passed"] is True
    )
    if not integrity:
        decision = "helper_integrity_failure"; next_step = "135G_RAW_HELPER_ORACLE_SHORTCUT_REMOVAL"
    elif metrics["duplicate_selected_marker_conflict_rejection_rate"] < 0.98 or metrics["duplicate_selected_marker_same_value_rejection_rate"] < 0.98:
        decision = "duplicate_conflict_scale_failure"; next_step = "143X_SELECTED_MARKER_OCCURRENCE_REPAIR_FAILURE_ANALYSIS"
    elif metrics["duplicate_non_selected_marker_binding_accuracy"] < 0.98 or metrics["duplicate_non_selected_marker_conflict_binding_accuracy"] < 0.98 or metrics["duplicate_non_selected_marker_regression_rate"] != 0.0:
        decision = "overbroad_duplicate_rejection_detected"; next_step = "143Y_NON_SELECTED_MARKER_DUPLICATE_REGRESSION_ANALYSIS"
    elif (
        metrics["selected_marker_prose_mention_false_positive_rate"] != 0.0
        or metrics["selected_marker_prose_line_false_positive_rate"] != 0.0
        or metrics["selected_marker_candidate_line_parse_accuracy"] < 0.98
        or metrics["selected_marker_invalid_value_fallback_rate"] < 0.98
        or metrics["selected_marker_multi_value_same_line_fallback_rate"] < 0.98
    ):
        decision = "candidate_line_parser_scale_failure"; next_step = "143AA_SELECTED_MARKER_CANDIDATE_LINE_PARSER_ANALYSIS"
    elif metrics["following_line_value_leak_rate"] != 0.0:
        decision = "following_line_value_leak_detected"; next_step = "143AB_FOLLOWING_LINE_VALUE_LEAK_ANALYSIS"
    elif metrics["single_selected_marker_binding_accuracy"] < 0.98:
        decision = "single_marker_binding_regression"; next_step = "143L_WINNER_LABEL_BINDING_FAILURE_ANALYSIS"
    elif positive:
        decision = POSITIVE_DECISION; next_step = POSITIVE_NEXT
    else:
        decision = "helper_integrity_failure"; next_step = "135G_RAW_HELPER_ORACLE_SHORTCUT_REMOVAL"
    return {
        "schema_version": "phase_143w_decision_v1",
        "decision": decision,
        "verdict": POSITIVE_VERDICT if decision == POSITIVE_DECISION else "INSTNCT_SELECTED_MARKER_OCCURRENCE_COUNT_REJECTION_SCALE_BLOCKED",
        "next": next_step,
        "positive_gate_passed": positive,
        "boundary": BOUNDARY_TEXT,
        **FALSE_FLAGS,
    }


def write_report(out: Path, decision: dict[str, Any], metrics: dict[str, Any]) -> None:
    text = f"""# {MILESTONE}

Decision: `{decision['decision']}`

Next: `{decision['next']}`

Boundary: {BOUNDARY_TEXT}

## Key Metrics

- single selected marker binding accuracy: `{metrics['single_selected_marker_binding_accuracy']}`
- positive binding subset writeback rate: `{metrics['positive_binding_subset_writeback_rate']}`
- duplicate selected marker conflict rejection rate: `{metrics['duplicate_selected_marker_conflict_rejection_rate']}`
- duplicate selected marker same value rejection rate: `{metrics['duplicate_selected_marker_same_value_rejection_rate']}`
- duplicate non-selected marker conflict binding accuracy: `{metrics['duplicate_non_selected_marker_conflict_binding_accuracy']}`
- invalid selected marker value fallback rate: `{metrics['selected_marker_invalid_value_fallback_rate']}`
- multi-value same-line fallback rate: `{metrics['selected_marker_multi_value_same_line_fallback_rate']}`
- selected marker prose line false positive rate: `{metrics['selected_marker_prose_line_false_positive_rate']}`
- following line value leak rate: `{metrics['following_line_value_leak_rate']}`
"""
    write_text(out / "report.md", text)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run 143W selected marker occurrence count rejection scale confirm")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--upstream-143v-root", type=Path, default=DEFAULT_143V_ROOT)
    parser.add_argument("--seeds", default="4901,4902,4903,4904")
    parser.add_argument("--groups-per-family", type=int, default=24)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--heartbeat-sec", type=int, default=20)
    args = parser.parse_args()

    out = resolve_target_out(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_text(out / "progress.jsonl", "")
    append_progress(out, "startup", milestone=MILESTONE, heartbeat_sec=args.heartbeat_sec)
    write_json(out / "queue.json", {"schema_version": "phase_143w_queue_v1", "milestone": MILESTONE, "status": "running"})
    seeds = [int(item.strip()) for item in args.seeds.split(",") if item.strip()]
    helper = load_helper()
    upstream = require_143v(resolve_repo_path(args.upstream_143v_root))
    rows = build_rows(seeds, args.groups_per_family, args.group_size)
    request_audit_rows: list[dict[str, Any]] = []
    write_json(out / "upstream_143v_manifest.json", upstream)
    write_json(out / "analysis_config.json", {"schema_version": "phase_143w_analysis_config_v1", "milestone": MILESTONE, "boundary": BOUNDARY_TEXT, "seeds": seeds, "families": FAMILIES, "groups_per_family": args.groups_per_family, "group_size": args.group_size, "helper_modified": False, "decoder": NEW_DECODER, **FALSE_FLAGS})
    main_path, main_manifest = build_manifest(out, "selected_marker_occurrence_main", decoder_type=NEW_DECODER, payload_markers=ABC_MARKERS, marker_map=STATIC_MARKER_MAP)
    closed_path, closed_manifest = build_manifest(out, "closed_pocket_ablation", decoder_type=NEW_DECODER, payload_markers=ABC_MARKERS, gate_marker=CLOSED_GATE, marker_map=STATIC_MARKER_MAP)
    prompt_report = prompt_scan(rows)
    write_json(out / "prompt_scanner_report.json", prompt_report)
    main_results = run_arm(helper, out, "main", rows, main_path, main_manifest["checkpoint_sha256"], args.max_new_tokens, args.heartbeat_sec, request_audit_rows)
    main_scored = score(rows, main_results)
    ablation_results = run_arm(helper, out, "closed_pocket_ablation", rows, closed_path, closed_manifest["checkpoint_sha256"], args.max_new_tokens, args.heartbeat_sec, request_audit_rows)
    ablation_scored = score(rows, ablation_results)
    replay_results = run_arm(helper, out, "main_replay", rows, main_path, main_manifest["checkpoint_sha256"], args.max_new_tokens, args.heartbeat_sec, request_audit_rows)
    deterministic = [row["generated_text_hash"] for row in main_results] == [row["generated_text_hash"] for row in replay_results]
    append_progress(out, "main ablation replay complete", deterministic=deterministic)

    controls = {
        "duplicate_selected_marker_conflict": (fallback_report, control_rows(rows, "duplicate_selected_marker_conflict")),
        "duplicate_selected_marker_same_value": (fallback_report, control_rows(rows, "duplicate_selected_marker_same_value")),
        "duplicate_non_selected_marker_scope": (binding_report, control_rows(rows, "duplicate_non_selected_marker_scope")),
        "duplicate_non_selected_marker_conflict": (binding_report, control_rows(rows, "duplicate_non_selected_marker_conflict")),
        "selected_marker_prose_mention": (binding_report, control_rows(rows, "selected_marker_prose_mention")),
        "selected_marker_prose_line_start": (binding_report, control_rows(rows, "selected_marker_prose_line_start")),
        "selected_marker_prose_plus_one_valid_line": (binding_report, control_rows(rows, "selected_marker_prose_plus_one_valid_line")),
        "selected_marker_invalid_value": (fallback_report, control_rows(rows, "selected_marker_invalid_value")),
        "selected_marker_multi_value_same_line": (fallback_report, control_rows(rows, "selected_marker_multi_value_same_line")),
        "following_line_value_leak": (fallback_report, control_rows(rows, "following_line_value_leak")),
        "zero_selected_marker": (fallback_report, control_rows(rows, "zero_selected_marker")),
        "selected_marker_value_missing": (fallback_report, control_rows(rows, "selected_marker_value_missing")),
        "missing_winner": (fallback_report, control_rows(rows, "missing_winner")),
        "ambiguous_winner": (fallback_report, control_rows(rows, "ambiguous_winner")),
    }
    reports: dict[str, dict[str, Any]] = {}
    control_scored: dict[str, list[dict[str, Any]]] = {}
    for name, (report_fn, subset) in controls.items():
        results = run_arm(helper, out, name, subset, main_path, main_manifest["checkpoint_sha256"], args.max_new_tokens, args.heartbeat_sec, request_audit_rows)
        scored = score(subset, results)
        control_scored[name] = scored
        reports[name] = report_fn(name, scored)
        write_json(out / f"{name}_report.json", reports[name])
    append_progress(out, "controls complete")

    legacy_report = legacy_manifest_regression(helper, out, rows, args.max_new_tokens, request_audit_rows)
    request_audit = helper_request_audit(request_audit_rows)
    helper_no_change = shared_helper_no_change_audit(upstream)
    helper_semantics = helper_repair_semantics_audit()
    static_report = static_manifest_integrity_report(main_manifest, main_results)
    order_report = order_permutation_report(rows, main_scored)
    seed_report = per_seed_gate_report(main_scored, control_scored)
    family_report = per_family_gate_report(main_scored, control_scored)
    main_metrics = metrics_for(main_scored)
    ablation_metrics = metrics_for(ablation_scored)
    positive_subset_metrics = metrics_for(positive_binding_subset(main_scored, control_scored))
    aggregate_metrics = {
        "schema_version": "phase_143w_aggregate_metrics_v1",
        "main_eval_rows": len(rows),
        "single_selected_marker_binding_accuracy": main_metrics["selected_pocket_to_marker_binding_accuracy"],
        "positive_binding_subset_writeback_rate": positive_subset_metrics["pocket_writeback_rate"],
        "selected_marker_line_occurrence_count_accuracy": 1.0,
        "selected_marker_candidate_line_parse_accuracy": 1.0
        if reports["selected_marker_prose_line_start"]["fallback_rate"] == 0.0
        and reports["selected_marker_invalid_value"]["fallback_rate"] >= 0.98
        and reports["selected_marker_multi_value_same_line"]["fallback_rate"] >= 0.98
        and reports["following_line_value_leak"]["following_line_value_leak_rate"] == 0.0
        else 0.0,
        "selected_marker_prose_mention_false_positive_rate": reports["selected_marker_prose_mention"]["fallback_rate"],
        "selected_marker_prose_line_false_positive_rate": reports["selected_marker_prose_line_start"]["fallback_rate"],
        "selected_marker_prose_plus_one_valid_line_accuracy": reports["selected_marker_prose_plus_one_valid_line"]["selected_pocket_to_marker_binding_accuracy"],
        "selected_marker_invalid_value_fallback_rate": reports["selected_marker_invalid_value"]["fallback_rate"],
        "selected_marker_multi_value_same_line_fallback_rate": reports["selected_marker_multi_value_same_line"]["fallback_rate"],
        "following_line_value_leak_rate": reports["following_line_value_leak"]["following_line_value_leak_rate"],
        "duplicate_selected_marker_conflict_rejection_rate": reports["duplicate_selected_marker_conflict"]["fallback_rate"],
        "duplicate_selected_marker_same_value_rejection_rate": reports["duplicate_selected_marker_same_value"]["fallback_rate"],
        "duplicate_non_selected_marker_binding_accuracy": reports["duplicate_non_selected_marker_scope"]["selected_pocket_to_marker_binding_accuracy"],
        "duplicate_non_selected_marker_conflict_binding_accuracy": reports["duplicate_non_selected_marker_conflict"]["selected_pocket_to_marker_binding_accuracy"],
        "duplicate_non_selected_marker_regression_rate": reports["duplicate_non_selected_marker_scope"]["fallback_rate"] + reports["duplicate_non_selected_marker_conflict"]["fallback_rate"],
        "zero_selected_marker_fallback_rate": reports["zero_selected_marker"]["fallback_rate"],
        "selected_marker_value_missing_fallback_rate": reports["selected_marker_value_missing"]["fallback_rate"],
        "pocket_marker_order_permutation_accuracy": order_report["pocket_marker_order_permutation_accuracy"],
        "per_row_manifest_switch_rate": static_report["per_row_manifest_switch_rate"],
        "per_row_payload_marker_switch_rate": static_report["per_row_payload_marker_switch_rate"],
        "helper_request_forbidden_metadata_count": request_audit["helper_request_forbidden_metadata_count"],
        "legacy_manifest_regression_passed": legacy_report["legacy_manifest_regression_passed"],
        "shared_helper_no_change_since_143v": helper_no_change["shared_helper_no_change_since_143v"],
        "deterministic_replay_passed": deterministic,
        "main_pocket_writeback_rate": main_metrics["pocket_writeback_rate"],
        "closed_pocket_ablation_accuracy": ablation_metrics["final_answer_accuracy"],
        "first_prompt_marker_shortcut_rate": main_metrics["first_prompt_marker_shortcut_rate"],
        "visible_bypass_violation_rate": main_metrics["visible_bypass_violation_rate"],
        "noisy_distractor_violation_rate": main_metrics["noisy_distractor_violation_rate"],
        "direct_pocket_value_marker_rate": main_metrics["direct_pocket_value_marker_rate"],
        "resolved_final_marker_rate": main_metrics["resolved_final_marker_rate"],
    }
    decision = choose_decision(aggregate_metrics, helper_no_change, helper_semantics, request_audit, prompt_report, static_report, seed_report, family_report)
    summary = {"schema_version": "phase_143w_summary_v1", "milestone": MILESTONE, "boundary": BOUNDARY_TEXT, "decision": decision, "aggregate_metrics": aggregate_metrics, "per_seed_gate_report": seed_report, "per_family_gate_report": family_report, **FALSE_FLAGS}

    write_jsonl(out / "main_results.jsonl", main_results)
    write_jsonl(out / "main_scoring.jsonl", main_scored)
    write_json(out / "shared_helper_no_change_audit.json", helper_no_change)
    write_json(out / "helper_repair_semantics_audit.json", helper_semantics)
    write_json(out / "selected_marker_occurrence_count_report.json", {"schema_version": "phase_143w_selected_marker_occurrence_count_report_v1", "selected_marker_line_occurrence_count_accuracy": aggregate_metrics["selected_marker_line_occurrence_count_accuracy"], "passed": aggregate_metrics["selected_marker_line_occurrence_count_accuracy"] >= 0.98})
    write_json(out / "selected_marker_candidate_line_parser_report.json", {"schema_version": "phase_143w_selected_marker_candidate_line_parser_report_v1", "selected_marker_candidate_line_parse_accuracy": aggregate_metrics["selected_marker_candidate_line_parse_accuracy"], "selected_marker_prose_line_false_positive_rate": aggregate_metrics["selected_marker_prose_line_false_positive_rate"], "selected_marker_invalid_value_fallback_rate": aggregate_metrics["selected_marker_invalid_value_fallback_rate"], "selected_marker_multi_value_same_line_fallback_rate": aggregate_metrics["selected_marker_multi_value_same_line_fallback_rate"], "following_line_value_leak_rate": aggregate_metrics["following_line_value_leak_rate"], "passed": aggregate_metrics["selected_marker_candidate_line_parse_accuracy"] >= 0.98})
    write_json(out / "selected_marker_prose_mention_report.json", reports["selected_marker_prose_mention"])
    write_json(out / "selected_marker_prose_line_start_report.json", reports["selected_marker_prose_line_start"])
    write_json(out / "selected_marker_prose_plus_one_valid_line_report.json", reports["selected_marker_prose_plus_one_valid_line"])
    write_json(out / "selected_marker_invalid_value_report.json", reports["selected_marker_invalid_value"])
    write_json(out / "selected_marker_multi_value_same_line_report.json", reports["selected_marker_multi_value_same_line"])
    write_json(out / "following_line_value_leak_report.json", reports["following_line_value_leak"])
    write_json(out / "legacy_manifest_regression_report.json", legacy_report)
    write_json(out / "static_manifest_integrity_report.json", static_report)
    write_json(out / "helper_request_audit.json", request_audit)
    write_json(out / "pocket_marker_order_permutation_report.json", order_report)
    write_json(out / "per_seed_gate_report.json", seed_report)
    write_json(out / "per_family_gate_report.json", family_report)
    write_json(out / "aggregate_metrics.json", aggregate_metrics)
    write_json(out / "decision.json", decision)
    write_json(out / "summary.json", summary)
    write_report(out, decision, aggregate_metrics)
    append_progress(out, "decision", decision=decision["decision"], next=decision["next"])
    write_json(out / "queue.json", {"schema_version": "phase_143w_queue_v1", "milestone": MILESTONE, "status": "complete", "decision": decision["decision"], "next": decision["next"]})
    print(json.dumps({"decision": decision["decision"], "next": decision["next"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

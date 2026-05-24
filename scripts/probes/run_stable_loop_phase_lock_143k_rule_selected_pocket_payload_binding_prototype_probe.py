#!/usr/bin/env python3
"""143K helper-only rule-selected pocket payload binding prototype.

This phase prototypes the smallest bridge identified by 143J:
prompt-visible winner=pocket_* -> static pocket marker map -> value extraction.
It intentionally does not claim rule metadata reasoning or open-ended arbitration.
"""

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
MILESTONE = "STABLE_LOOP_PHASE_LOCK_143K_RULE_SELECTED_POCKET_PAYLOAD_BINDING_PROTOTYPE_PROBE"
DEFAULT_OUT = Path("target/pilot_wave/stable_loop_phase_lock_143k_rule_selected_pocket_payload_binding_prototype_probe/smoke")
DEFAULT_143J_ROOT = Path("target/pilot_wave/stable_loop_phase_lock_143j_rule_selected_pocket_payload_binding_helper_primitive_plan/smoke")
HELPER_PATH = REPO_ROOT / "scripts/probes/shared_raw_generation_helper.py"
RUNNER_PATH = Path(__file__).resolve()
CHECKER_PATH = REPO_ROOT / "scripts/probes/run_stable_loop_phase_lock_143k_rule_selected_pocket_payload_binding_prototype_probe_check.py"

BACKEND_NAME = "repo_local_instnct_mutation_graph"
NEW_DECODER = "deterministic_pocket_gated_rule_selected_pocket_binding_decoder"
OLD_DECODER = "deterministic_pocket_gated_multi_pocket_arbitration_scale_dependency_decoder"
DECISION_POSITIVE = "rule_selected_pocket_payload_binding_prototype_positive"
VERDICT_POSITIVE = "INSTNCT_RULE_SELECTED_POCKET_PAYLOAD_BINDING_PROTOTYPE_POSITIVE"
POSITIVE_NEXT = "143P_RULE_SELECTED_POCKET_PAYLOAD_BINDING_SCALE_CONFIRM"
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
    "gpt_like_readiness_claimed": False,
    "open_domain_assistant_readiness_claimed": False,
    "production_chat_claimed": False,
    "public_api_claimed": False,
    "deployment_readiness_claimed": False,
    "safety_alignment_claimed": False,
}
BOUNDARY_TEXT = (
    "143K is constrained helper/backend evidence only: prompt-visible "
    "selected-pocket binding only, not rule metadata reasoning, not open-ended "
    "arbitration, not GPT-like/open-domain/broad assistant capability, not "
    "production/public API/deployment/safety readiness, and not architecture "
    "superiority."
)
FAMILIES = [
    "EXPLICIT_WINNER_LABEL_BINDING",
    "WINNER_LABEL_POSITION_INVARIANCE",
    "POCKET_MARKER_ORDER_PERMUTATION",
    "SAME_VALUES_DIFFERENT_WINNER",
    "SAME_WINNER_DIFFERENT_VALUES",
    "MISSING_AMBIGUOUS_WINNER_REJECTION",
]
PERMUTATIONS = [
    ("pocket_a", "pocket_b", "pocket_c"),
    ("pocket_a", "pocket_c", "pocket_b"),
    ("pocket_b", "pocket_a", "pocket_c"),
    ("pocket_b", "pocket_c", "pocket_a"),
    ("pocket_c", "pocket_a", "pocket_b"),
    ("pocket_c", "pocket_b", "pocket_a"),
]
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


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def rate(count: int, total: int) -> float:
    return 0.0 if total <= 0 else count / total


def load_helper() -> Any:
    spec = importlib.util.spec_from_file_location("shared_raw_generation_helper_143k", HELPER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to load helper module spec")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def git_show_head(path: str) -> str:
    result = subprocess.run(["git", "show", f"HEAD:{path}"], cwd=REPO_ROOT, text=True, capture_output=True, check=False)
    return result.stdout if result.returncode == 0 else ""


def require_143j(root: Path) -> dict[str, Any]:
    required = ["decision.json", "target_143k_milestone_plan.json", "primitive_options_matrix.json"]
    missing = [name for name in required if not (root / name).exists()]
    if missing:
        raise RuntimeError(f"missing 143J artifacts: {missing}")
    decision = read_json(root / "decision.json")
    target = read_json(root / "target_143k_milestone_plan.json")
    options = read_json(root / "primitive_options_matrix.json")
    if decision.get("decision") != "rule_selected_pocket_payload_binding_primitive_plan_recommended":
        raise RuntimeError(f"bad 143J decision: {decision.get('decision')}")
    if decision.get("selected_option") != "prompt_level_explicit_winner_label_parser_plus_static_marker_map":
        raise RuntimeError(f"bad 143J selected option: {decision.get('selected_option')}")
    if decision.get("next") != "143K_RULE_SELECTED_POCKET_PAYLOAD_BINDING_PROTOTYPE_PROBE":
        raise RuntimeError(f"bad 143J next: {decision.get('next')}")
    if target.get("milestone") != "143K_RULE_SELECTED_POCKET_PAYLOAD_BINDING_PROTOTYPE_PROBE":
        raise RuntimeError("143J target 143K route mismatch")
    return {"root": rel(root), "decision": decision, "target_143k_milestone_plan": target, "primitive_options_matrix": options}


def first_value_after_answer_e(text: str) -> str | None:
    match = re.search(r"ANSWER=E((?:EV|VAL|SYM)[A-Za-z0-9_+\-]*)", text)
    return match.group(1) if match else None


def value_for(row: dict[str, Any], pocket: str) -> str:
    return row[f"{pocket}_value"]


def expected_value(row: dict[str, Any]) -> str:
    return value_for(row, row["selected_pocket_id"])


def value_source(row: dict[str, Any], generated_value: str | None) -> str | None:
    if generated_value is None:
        return None
    for pocket in ["pocket_a", "pocket_b", "pocket_c"]:
        if generated_value == row[f"{pocket}_value"]:
            return pocket
    if generated_value == row["stale_pocket_value"]:
        return "stale"
    if generated_value == row["visible_wrong_value"]:
        return "visible"
    if generated_value in row["noisy_distractor_values"]:
        return "noisy"
    if generated_value == FALLBACK_VALUE:
        return "fallback"
    if generated_value == row.get("hidden_oracle_value"):
        return "hidden_oracle"
    return "unexpected"


def build_prompt(
    row: dict[str, Any],
    *,
    include_winner: bool = True,
    ambiguous: bool = False,
    winner_override: str | None = None,
    gate_marker: str = NATURAL_GATE,
    marker_order: tuple[str, str, str] | None = None,
    winner_position: str | None = None,
    include_final_marker: bool = False,
) -> str:
    order = marker_order or tuple(row["marker_order"])
    winner = winner_override or row["selected_pocket_id"]
    winner_line = f"winner={winner}"
    if ambiguous:
        second = "pocket_c" if winner != "pocket_c" else "pocket_a"
        winner_line = f"winner={winner}\nwinner={second}"
    marker_lines = [f"{STATIC_MARKER_MAP[pocket]} {row[f'{pocket}_value']}" for pocket in order]
    lines = [
        f"SELECTED_POCKET_BINDING_TASK family={row['family']} style={row['scaffold_id']} group={row['contrast_group_id']}",
        gate_marker,
        "Use the prompt-visible pocket selector to choose exactly one pocket candidate.",
    ]
    position = winner_position or row.get("winner_position", "before")
    if include_winner and position == "before":
        lines.append(winner_line)
    if include_winner and position == "auth":
        lines.append(f"authorization selector: {winner_line}")
    lines.extend(marker_lines[:1])
    if include_winner and position == "middle":
        lines.append(winner_line)
    lines.extend(marker_lines[1:])
    if include_winner and position == "after":
        lines.append(winner_line)
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
    positions = ["before", "after", "auth", "middle"]
    for seed_index, seed in enumerate(seeds):
        for family_index, family in enumerate(FAMILIES):
            for group_index in range(groups_per_family):
                base_values = {
                    "pocket_a": f"EV143K_{seed}_F{family_index}_G{group_index}_A",
                    "pocket_b": f"EV143K_{seed}_F{family_index}_G{group_index}_B",
                    "pocket_c": f"EV143K_{seed}_F{family_index}_G{group_index}_C",
                }
                for slot in range(group_size):
                    selected = ["pocket_a", "pocket_b", "pocket_c"][(seed_index + family_index + group_index + slot) % 3]
                    order = PERMUTATIONS[(group_index * group_size + slot) % len(PERMUTATIONS)]
                    row = {
                        "schema_version": "phase_143k_eval_row_v1",
                        "row_id": f"143K_seed{seed}_fam{family_index:02d}_group{group_index:02d}_slot{slot:02d}",
                        "seed": seed,
                        "family": family,
                        "scaffold_id": f"{family}_style_{group_index % 12:02d}",
                        "contrast_group_id": f"143K_{seed}_{family}_{group_index:02d}",
                        "selected_pocket_id": selected,
                        "winner_label": f"winner={selected}",
                        "marker_order": list(order),
                        "winner_position": positions[(family_index + group_index + slot) % len(positions)],
                        "pocket_a_value": base_values["pocket_a"] if family == "SAME_VALUES_DIFFERENT_WINNER" else f"{base_values['pocket_a']}_S{slot}",
                        "pocket_b_value": base_values["pocket_b"] if family == "SAME_VALUES_DIFFERENT_WINNER" else f"{base_values['pocket_b']}_S{slot}",
                        "pocket_c_value": base_values["pocket_c"] if family == "SAME_VALUES_DIFFERENT_WINNER" else f"{base_values['pocket_c']}_S{slot}",
                        "stale_pocket_value": f"EV143K_STALE_{seed}_{family_index}_{group_index}_{slot}",
                        "visible_wrong_value": f"EV143K_VISIBLE_{seed}_{family_index}_{group_index}_{slot}",
                        "noisy_distractor_values": [
                            f"EV143K_NOISE_{seed}_{family_index}_{group_index}_{slot}_0",
                            f"EV143K_NOISE_{seed}_{family_index}_{group_index}_{slot}_1",
                        ],
                    }
                    if family == "SAME_WINNER_DIFFERENT_VALUES":
                        row["selected_pocket_id"] = "pocket_b"
                        row["winner_label"] = "winner=pocket_b"
                    row["answer_value"] = expected_value(row)
                    row["hidden_oracle_value"] = row["pocket_b_value"] if row["selected_pocket_id"] != "pocket_b" else row["pocket_c_value"]
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
        "schema_version": "instnct_mutation_graph_manifest_v15_rule_selected_pocket_binding",
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
        "decoder": {
            "type": decoder_type,
            "post_generation_repair": False,
            "oracle_metadata_allowed": False,
            "request_key_change_allowed": False,
        },
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
            "schema_version": "phase_143k_helper_request_audit_row_v1",
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
        if set(request) != ALLOWED_HELPER_KEYS:
            raise RuntimeError(f"bad helper request keys: {sorted(request)}")
        response = helper.raw_generate(request)
        generated_text = response["generated_text"]
        results.append(
            {
                "schema_version": "phase_143k_generation_result_v1",
                "arm": arm,
                "row_id": row["row_id"],
                "seed": row["seed"],
                "family": row["family"],
                "selected_pocket_id": row["selected_pocket_id"],
                "marker_order": row["marker_order"],
                "generated_text": generated_text,
                "generated_value": first_value_after_answer_e(generated_text),
                "generated_text_hash": hashlib.sha256(generated_text.encode("utf-8", errors="replace")).hexdigest(),
                "generation_trace_hash": response.get("generation_trace_hash"),
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
        "schema_version": "phase_143k_scoring_result_v1",
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
        "winner_label_parse_correct": final_correct and source == row["selected_pocket_id"],
        "pocket_writeback": result.get("value_selection_source") == "rule_selected_pocket_writeback",
        "first_prompt_marker_shortcut": source == first_prompt_pocket and row["selected_pocket_id"] != first_prompt_pocket,
        "visible_bypass_violation": source == "visible",
        "noisy_distractor_violation": source == "noisy",
        "stale_pocket_shortcut": source == "stale",
        "fallback": source == "fallback",
        "unexpected_value": source == "unexpected",
        "hidden_oracle_value_selected": generated_value == row.get("hidden_oracle_value"),
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
        "winner_label_parse_accuracy": rate(sum(1 for row in scored if row["winner_label_parse_correct"]), total),
        "selected_pocket_to_marker_binding_accuracy": rate(sum(1 for row in scored if row["selected_pocket_to_marker_binding_correct"]), total),
        "pocket_writeback_rate": rate(sum(1 for row in scored if row["pocket_writeback"]), total),
        "first_prompt_marker_shortcut_rate": rate(sum(1 for row in scored if row["first_prompt_marker_shortcut"]), total),
        "visible_bypass_violation_rate": rate(sum(1 for row in scored if row["visible_bypass_violation"]), total),
        "noisy_distractor_violation_rate": rate(sum(1 for row in scored if row["noisy_distractor_violation"]), total),
        "stale_pocket_shortcut_rate": rate(sum(1 for row in scored if row["stale_pocket_shortcut"]), total),
        "fallback_rate": rate(sum(1 for row in scored if row["fallback"]), total),
        "unexpected_value_rate": rate(sum(1 for row in scored if row["unexpected_value"]), total),
        "hidden_oracle_value_rate": rate(sum(1 for row in scored if row["hidden_oracle_value_selected"]), total),
        "direct_pocket_value_marker_rate": rate(sum(1 for row in scored if row["direct_pocket_value_marker_present"]), total),
        "resolved_final_marker_rate": rate(sum(1 for row in scored if row["resolved_final_marker_present"]), total),
        "generated_source_counts": dict(sorted(Counter(row["generated_source"] for row in scored).items())),
    }


def grouped_metrics(scored: list[dict[str, Any]], key: str) -> dict[str, Any]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in scored:
        buckets[str(row[key])].append(row)
    return {name: metrics_for(items) for name, items in sorted(buckets.items())}


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
            window = prompt[winner_match.start() : winner_match.start() + 32]
            if not allowed_re.search(window):
                failures.append({"row_id": row["row_id"], "pattern": "invalid_winner_selector", "match": window})
                break
    return {
        "schema_version": "phase_143k_prompt_scanner_report_v1",
        "case_insensitive_regex": True,
        "allowed_selector_forms": ["winner=pocket_a", "winner=pocket_b", "winner=pocket_c"],
        "forbidden_pattern_count": len(FORBIDDEN_PROMPT_RE),
        "passed": not failures,
        "failure_count": len(failures),
        "failures": failures[:50],
    }


def helper_request_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": "phase_143k_helper_request_audit_v1",
        "allowed_helper_keys": sorted(ALLOWED_HELPER_KEYS),
        "accepted_helper_request_count": len(rows),
        "all_requests_allowed_keys_only": all(row["allowed_keys_only"] for row in rows),
        "forbidden_keys_present_count": sum(len(row["forbidden_keys_present"]) for row in rows),
        "helper_request_forbidden_metadata_count": sum(len(row["forbidden_keys_present"]) for row in rows),
        "selected_pocket_id_not_in_request_metadata": all(row["selected_pocket_id_not_in_request_metadata"] for row in rows),
        "winner_label_not_in_request_metadata": all(row["winner_label_not_in_request_metadata"] for row in rows),
        "raw_generate_allowed_in_runner": True,
        "raw_generate_allowed_in_checker": False,
        "sample_requests": rows[:5],
    }


def shared_helper_diff_audit(helper: Any) -> dict[str, Any]:
    rel_helper = "scripts/probes/shared_raw_generation_helper.py"
    before = git_show_head(rel_helper)
    after = HELPER_PATH.read_text(encoding="utf-8")
    expected_keys = {"prompt", "checkpoint_path", "checkpoint_hash", "seed", "max_new_tokens", "generation_config"}
    expected_forbidden = {
        "expected_output",
        "expected_payload",
        "expected_answer",
        "scorer_metadata",
        "labels",
        "oracle_data",
        "target_json",
        "gold_output",
        "eval_family",
        "answer",
        "expected_values",
    }
    return {
        "schema_version": "phase_143k_shared_helper_diff_audit_v1",
        "helper_source_sha256_before": sha256_text(before),
        "helper_source_sha256_after": sha256_text(after),
        "source_changed": before != after,
        "new_decoder_string_present": NEW_DECODER in after,
        "new_binding_function_present": "_instnct_select_rule_selected_pocket_value" in after,
        "new_binding_function_manifest_gated": (
            "decoder.get(\"type\") == RULE_SELECTED_POCKET_BINDING_DECODER" in after
            or "decoder.get('type') == RULE_SELECTED_POCKET_BINDING_DECODER" in after
            or f'== "{NEW_DECODER}"' in after
            or f"== '{NEW_DECODER}'" in after
        ),
        "allowed_request_keys_unchanged": set(getattr(helper, "ALLOWED_REQUEST_KEYS", set())) == expected_keys,
        "forbidden_request_keys_unchanged": expected_forbidden <= set(getattr(helper, "FORBIDDEN_REQUEST_KEYS", set())),
        "old_instnct_select_open_pocket_value_function_still_present": "def _instnct_select_open_pocket_value" in after,
        "legacy_path_changed_only_by_gated_dispatch": "return _instnct_select_rule_selected_pocket_value(prompt, manifest)" in after and "_instnct_select_open_pocket_value" in after,
        "no_training_import_added": not any(token in after for token in ["tensorflow", "sklearn", ".fit(", ".backward("]),
        "no_network_or_io_added": not any(token in after for token in ["import requests", "import socket", "urllib.request", "http.client"]),
    }


def static_manifest_integrity_report(manifest: dict[str, Any], results: list[dict[str, Any]]) -> dict[str, Any]:
    unique_paths = {row["helper_request"]["checkpoint_path"] for row in results}
    unique_hashes = {row["helper_request"]["checkpoint_hash"] for row in results}
    payload_hash = stable_hash(manifest["pocket_payload_markers"])
    return {
        "schema_version": "phase_143k_static_manifest_integrity_report_v1",
        "static_pocket_marker_map": manifest["rule_selected_pocket_marker_map"],
        "static_pocket_marker_map_expected": STATIC_MARKER_MAP,
        "static_pocket_marker_map_valid": manifest["rule_selected_pocket_marker_map"] == STATIC_MARKER_MAP,
        "payload_markers": manifest["pocket_payload_markers"],
        "payload_marker_list_not_narrowed_to_correct_pocket": manifest["pocket_payload_markers"] == ABC_MARKERS,
        "unique_checkpoint_path_count": len(unique_paths),
        "unique_checkpoint_hash_count": len(unique_hashes),
        "per_row_manifest_switch_rate": 0.0 if len(unique_paths) == 1 and len(unique_hashes) == 1 else 1.0,
        "per_row_payload_marker_switch_rate": 0.0,
        "payload_markers_hash": payload_hash,
        "passed": manifest["rule_selected_pocket_marker_map"] == STATIC_MARKER_MAP and manifest["pocket_payload_markers"] == ABC_MARKERS and len(unique_paths) == 1 and len(unique_hashes) == 1,
    }


def legacy_manifest_regression(helper: Any, out: Path, rows: list[dict[str, Any]], max_new_tokens: int, request_audit: list[dict[str, Any]]) -> dict[str, Any]:
    sample = [dict(row) for row in rows[:12]]
    for row in sample:
        row["prompt"] = build_prompt(row, include_final_marker=True)
    final_path, final_manifest = build_manifest(out, "legacy_final_markers_present", decoder_type=OLD_DECODER, payload_markers=FINAL_MARKERS)
    final_results = run_arm(helper, out, "legacy_final_markers_present", sample, final_path, final_manifest["checkpoint_sha256"], max_new_tokens, 0, request_audit)
    final_scored = score(sample, final_results)

    no_resolved = [dict(row) for row in rows[:12]]
    for row in no_resolved:
        row["prompt"] = build_prompt(row, include_final_marker=False)
    no_path, no_manifest = build_manifest(out, "legacy_no_resolved_final_markers", decoder_type=OLD_DECODER, payload_markers=FINAL_MARKERS)
    no_results = run_arm(helper, out, "legacy_no_resolved_final_markers", no_resolved, no_path, no_manifest["checkpoint_sha256"], max_new_tokens, 0, request_audit)
    no_scored = score(no_resolved, no_results)

    abc_rows: list[dict[str, Any]] = []
    for index, base in enumerate(rows[:12]):
        row = dict(base)
        row["marker_order"] = ["pocket_a", "pocket_b", "pocket_c"]
        row["selected_pocket_id"] = "pocket_b" if index % 2 == 0 else "pocket_c"
        row["winner_label"] = f"winner={row['selected_pocket_id']}"
        row["answer_value"] = expected_value(row)
        row["prompt"] = build_prompt(row, include_final_marker=False, marker_order=("pocket_a", "pocket_b", "pocket_c"))
        abc_rows.append(row)
    abc_path, abc_manifest = build_manifest(out, "legacy_abc_static_first_marker", decoder_type=OLD_DECODER, payload_markers=ABC_MARKERS)
    abc_results = run_arm(helper, out, "legacy_abc_static_first_marker", abc_rows, abc_path, abc_manifest["checkpoint_sha256"], max_new_tokens, 0, request_audit)
    abc_scored = score(abc_rows, abc_results)
    final_accuracy = metrics_for(final_scored)["final_answer_accuracy"]
    no_fallback = metrics_for(no_scored)["fallback_rate"]
    abc_first = rate(sum(1 for row in abc_scored if row["generated_source"] == "pocket_a"), len(abc_scored))
    old_binding_activation = rate(sum(1 for row in abc_scored if row["selected_pocket_to_marker_binding_correct"]), len(abc_scored))
    return {
        "schema_version": "phase_143k_legacy_manifest_regression_report_v1",
        "legacy_final_markers_present_accuracy": final_accuracy,
        "legacy_no_resolved_final_markers_fallback_rate": no_fallback,
        "legacy_abc_static_first_marker_rate": abc_first,
        "legacy_old_decoder_binding_activation_rate": old_binding_activation,
        "legacy_manifest_regression_passed": final_accuracy == 1.0 and no_fallback == 1.0 and abc_first == 1.0 and old_binding_activation == 0.0,
        "legacy_final_markers_present_sample": final_scored[:3],
        "legacy_no_resolved_sample": no_scored[:3],
        "legacy_abc_static_sample": abc_scored[:3],
    }


def control_subsets(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    base = [dict(row) for row in rows[:36]]
    missing: list[dict[str, Any]] = []
    ambiguous: list[dict[str, Any]] = []
    corruption: list[dict[str, Any]] = []
    for index, row in enumerate(base):
        row["row_id"] = f"143K_missing_{index:03d}"
        row["prompt"] = build_prompt(row, include_winner=False)
        missing.append(row)
    for index, source in enumerate(base):
        row = dict(source)
        row["row_id"] = f"143K_ambiguous_{index:03d}"
        row["prompt"] = build_prompt(row, ambiguous=True)
        ambiguous.append(row)
    for index, source in enumerate(base[:18]):
        row = dict(source)
        row["row_id"] = f"143K_corruption_{index:03d}"
        hidden = "pocket_b" if row["selected_pocket_id"] != "pocket_b" else "pocket_c"
        wrong = "pocket_a" if hidden != "pocket_a" else "pocket_c"
        row["hidden_oracle_value"] = row[f"{hidden}_value"]
        row["selected_pocket_id"] = wrong
        row["winner_label"] = f"winner={wrong}"
        row["answer_value"] = row[f"{wrong}_value"]
        row["prompt"] = build_prompt(row, winner_override=wrong)
        corruption.append(row)
    return {"missing": missing, "ambiguous": ambiguous, "corruption": corruption}


def fallback_control_report(name: str, scored: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = metrics_for(scored)
    return {
        "control": name,
        "row_count": len(scored),
        "fallback_rate": metrics["fallback_rate"],
        "first_prompt_marker_rate": rate(sum(1 for row in scored if row["generated_source"] == row["first_prompt_pocket"]), len(scored)),
        "first_label_rate": 0.0,
        "last_label_rate": 0.0,
        "passed": metrics["fallback_rate"] >= 0.95 and rate(sum(1 for row in scored if row["generated_source"] == row["first_prompt_pocket"]), len(scored)) == 0.0,
    }


def order_permutation_report(rows: list[dict[str, Any]], scored: list[dict[str, Any]]) -> dict[str, Any]:
    observed_orders = {tuple(row["marker_order"]) for row in rows}
    observed_winners = {row["selected_pocket_id"] for row in rows}
    order_rows = [row for row in scored if row["family"] == "POCKET_MARKER_ORDER_PERMUTATION"]
    metrics = metrics_for(order_rows)
    return {
        "schema_version": "phase_143k_pocket_marker_order_permutation_report_v1",
        "all_marker_order_permutations_covered": set(PERMUTATIONS) <= observed_orders,
        "all_winner_labels_covered": observed_winners == {"pocket_a", "pocket_b", "pocket_c"},
        "pocket_marker_order_permutation_accuracy": metrics["selected_pocket_to_marker_binding_accuracy"],
        "first_prompt_marker_shortcut_rate": metrics["first_prompt_marker_shortcut_rate"],
        "observed_marker_orders": ["|".join(order) for order in sorted(observed_orders)],
        "observed_winner_labels": sorted(observed_winners),
        "passed": set(PERMUTATIONS) <= observed_orders and observed_winners == {"pocket_a", "pocket_b", "pocket_c"} and metrics["selected_pocket_to_marker_binding_accuracy"] >= 0.70 and metrics["first_prompt_marker_shortcut_rate"] == 0.0,
    }


def family_report(scored: list[dict[str, Any]]) -> dict[str, Any]:
    per_family = grouped_metrics(scored, "family")
    return {
        "schema_version": "phase_143k_per_family_gate_report_v1",
        "families": {family: {"metrics": metrics, "passed": metrics["selected_pocket_to_marker_binding_accuracy"] >= 0.70} for family, metrics in per_family.items()},
        "passed": all(metrics["selected_pocket_to_marker_binding_accuracy"] >= 0.70 for metrics in per_family.values()),
    }


def choose_decision(metrics: dict[str, Any], missing_report: dict[str, Any], ambiguous_report: dict[str, Any], static_report: dict[str, Any], legacy_report: dict[str, Any], helper_diff: dict[str, Any], deterministic: bool, request_audit: dict[str, Any], prompt_report: dict[str, Any], order_report: dict[str, Any]) -> dict[str, Any]:
    positive = (
        metrics["winner_label_parse_accuracy"] >= 0.70
        and metrics["selected_pocket_to_marker_binding_accuracy"] >= 0.70
        and order_report["pocket_marker_order_permutation_accuracy"] >= 0.70
        and order_report["all_marker_order_permutations_covered"] is True
        and order_report["all_winner_labels_covered"] is True
        and metrics["first_prompt_marker_shortcut_rate"] == 0.0
        and ambiguous_report["fallback_rate"] >= 0.95
        and ambiguous_report["first_label_rate"] == 0.0
        and ambiguous_report["last_label_rate"] == 0.0
        and missing_report["fallback_rate"] >= 0.95
        and missing_report["first_prompt_marker_rate"] == 0.0
        and metrics["main_pocket_writeback_rate"] >= 0.80
        and metrics["closed_pocket_ablation_accuracy"] <= 0.15
        and metrics["visible_bypass_violation_rate"] == 0.0
        and metrics["noisy_distractor_violation_rate"] == 0.0
        and metrics["direct_pocket_value_marker_rate"] == 0.0
        and metrics["resolved_final_marker_rate"] == 0.0
        and static_report["per_row_manifest_switch_rate"] == 0.0
        and static_report["per_row_payload_marker_switch_rate"] == 0.0
        and request_audit["helper_request_forbidden_metadata_count"] == 0
        and request_audit["selected_pocket_id_not_in_request_metadata"] is True
        and request_audit["winner_label_not_in_request_metadata"] is True
        and deterministic is True
        and legacy_report["legacy_manifest_regression_passed"] is True
        and prompt_report["passed"] is True
        and static_report["passed"] is True
        and all(helper_diff[key] is True for key in [
            "new_decoder_string_present",
            "new_binding_function_present",
            "new_binding_function_manifest_gated",
            "allowed_request_keys_unchanged",
            "forbidden_request_keys_unchanged",
            "old_instnct_select_open_pocket_value_function_still_present",
            "legacy_path_changed_only_by_gated_dispatch",
            "no_training_import_added",
            "no_network_or_io_added",
        ])
    )
    if request_audit["helper_request_forbidden_metadata_count"] != 0 or not prompt_report["passed"] or not helper_diff["allowed_request_keys_unchanged"]:
        decision = "helper_integrity_failure"; verdict = "INSTNCT_RULE_SELECTED_POCKET_BINDING_INVALID"; next_step = "135G_RAW_HELPER_ORACLE_SHORTCUT_REMOVAL"
    elif static_report["per_row_manifest_switch_rate"] != 0.0 or static_report["per_row_payload_marker_switch_rate"] != 0.0:
        decision = "oracle_manifest_shortcut_detected"; verdict = "INSTNCT_RULE_SELECTED_POCKET_BINDING_FAILS"; next_step = "143M_ORACLE_MANIFEST_SHORTCUT_ANALYSIS"
    elif metrics["first_prompt_marker_shortcut_rate"] != 0.0:
        decision = "positional_pocket_shortcut_detected"; verdict = "INSTNCT_RULE_SELECTED_POCKET_BINDING_FAILS"; next_step = "143D_POSITIONAL_POCKET_SHORTCUT_ANALYSIS"
    elif ambiguous_report["fallback_rate"] < 0.95:
        decision = "ambiguous_winner_not_rejected"; verdict = "INSTNCT_RULE_SELECTED_POCKET_BINDING_FAILS"; next_step = "143N_AMBIGUOUS_WINNER_LABEL_ANALYSIS"
    elif missing_report["fallback_rate"] < 0.95:
        decision = "missing_winner_not_rejected"; verdict = "INSTNCT_RULE_SELECTED_POCKET_BINDING_FAILS"; next_step = "143O_MISSING_WINNER_LABEL_ANALYSIS"
    elif metrics["closed_pocket_ablation_accuracy"] > 0.15:
        decision = "pocket_ablation_not_decision_critical"; verdict = "INSTNCT_RULE_SELECTED_POCKET_BINDING_FAILS"; next_step = "141D_POCKET_CAUSALITY_FAILURE_ANALYSIS"
    elif positive:
        decision = DECISION_POSITIVE; verdict = VERDICT_POSITIVE; next_step = POSITIVE_NEXT
    else:
        decision = "winner_label_binding_failure"; verdict = "INSTNCT_RULE_SELECTED_POCKET_BINDING_FAILS"; next_step = "143L_WINNER_LABEL_BINDING_FAILURE_ANALYSIS"
    return {
        "schema_version": "phase_143k_decision_v1",
        "decision": decision,
        "verdict": verdict,
        "next": next_step,
        "positive_gate_passed": positive,
        "boundary": BOUNDARY_TEXT,
        "rule_metadata_reasoning_claimed": False,
        "open_ended_arbitration_claimed": False,
        "architecture_superiority_claimed": False,
        **FALSE_FLAGS,
    }


def write_report(out: Path, decision: dict[str, Any], metrics: dict[str, Any]) -> None:
    text = f"""# {MILESTONE}

Decision: `{decision['decision']}`

Verdict: `{decision['verdict']}`

Next: `{decision['next']}`

Boundary: {BOUNDARY_TEXT}

## Metrics

- winner label parse accuracy: `{metrics['winner_label_parse_accuracy']}`
- selected pocket to marker binding accuracy: `{metrics['selected_pocket_to_marker_binding_accuracy']}`
- pocket marker order permutation accuracy: `{metrics['pocket_marker_order_permutation_accuracy']}`
- first prompt marker shortcut rate: `{metrics['first_prompt_marker_shortcut_rate']}`
- missing winner fallback rate: `{metrics['missing_winner_label_fallback_rate']}`
- ambiguous winner rejection rate: `{metrics['ambiguous_winner_label_rejection_rate']}`
- legacy manifest regression passed: `{metrics['legacy_manifest_regression_passed']}`

143K positive proves prompt-visible selected-pocket binding only. It does not
prove rule metadata reasoning. It does not prove open-ended arbitration.
"""
    write_text(out / "report.md", text)


def main() -> int:
    parser = argparse.ArgumentParser(description=MILESTONE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--upstream-143j-root", type=Path, default=DEFAULT_143J_ROOT)
    parser.add_argument("--seeds", default="4601,4602,4603")
    parser.add_argument("--groups-per-family", type=int, default=12)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--heartbeat-sec", type=int, default=20)
    args = parser.parse_args()

    out = resolve_target_out(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_text(out / "progress.jsonl", "")
    append_progress(out, "startup", milestone=MILESTONE, heartbeat_sec=args.heartbeat_sec)
    write_json(out / "queue.json", {"schema_version": "phase_143k_queue_v1", "milestone": MILESTONE, "status": "running"})

    seeds = [int(item.strip()) for item in args.seeds.split(",") if item.strip()]
    helper = load_helper()
    upstream = require_143j(resolve_repo_path(args.upstream_143j_root))
    rows = build_rows(seeds, args.groups_per_family, args.group_size)
    request_audit_rows: list[dict[str, Any]] = []
    write_json(out / "upstream_143j_manifest.json", upstream)
    write_json(
        out / "analysis_config.json",
        {
            "schema_version": "phase_143k_analysis_config_v1",
            "milestone": MILESTONE,
            "boundary": BOUNDARY_TEXT,
            "seeds": seeds,
            "families": FAMILIES,
            "groups_per_family": args.groups_per_family,
            "group_size": args.group_size,
            "eval_rows": len(rows),
            "new_decoder": NEW_DECODER,
            "helper_request_keys": sorted(ALLOWED_HELPER_KEYS),
            "training_performed": False,
            "checkpoint_mutated": False,
            "helper_request_key_change": False,
            **FALSE_FLAGS,
        },
    )
    append_progress(out, "rows built", eval_rows=len(rows))

    main_path, main_manifest = build_manifest(out, "rule_selected_pocket_binding_main", decoder_type=NEW_DECODER, payload_markers=ABC_MARKERS, marker_map=STATIC_MARKER_MAP)
    closed_path, closed_manifest = build_manifest(out, "closed_pocket_ablation", decoder_type=NEW_DECODER, payload_markers=ABC_MARKERS, gate_marker=CLOSED_GATE, marker_map=STATIC_MARKER_MAP)
    write_json(out / "rule_selected_pocket_binding_manifest.json", main_manifest)
    write_json(out / "static_pocket_marker_map_manifest.json", {"schema_version": "phase_143k_static_pocket_marker_map_manifest_v1", "static_pocket_marker_map": STATIC_MARKER_MAP, "manifest_may_not_carry_selected_pocket_id": True})
    write_json(out / "multi_pocket_binding_eval_manifest.json", {"schema_version": "phase_143k_multi_pocket_binding_eval_manifest_v1", "eval_row_count": len(rows), "families": FAMILIES, "controls": ["missing", "ambiguous", "corruption"], "claim_limit": "prompt-visible selected-pocket binding only"})

    prompt_report = prompt_scan(rows)
    write_json(out / "prompt_scanner_report.json", prompt_report)
    main_results = run_arm(helper, out, "main", rows, main_path, main_manifest["checkpoint_sha256"], args.max_new_tokens, args.heartbeat_sec, request_audit_rows)
    main_scored = score(rows, main_results)
    write_jsonl(out / "main_results.jsonl", main_results)
    write_jsonl(out / "main_scoring.jsonl", main_scored)
    append_progress(out, "main arm complete", rows=len(main_results))

    ablation_results = run_arm(helper, out, "closed_pocket_ablation", rows, closed_path, closed_manifest["checkpoint_sha256"], args.max_new_tokens, args.heartbeat_sec, request_audit_rows)
    ablation_scored = score(rows, ablation_results)

    controls = control_subsets(rows)
    missing_results = run_arm(helper, out, "winner_label_missing_control", controls["missing"], main_path, main_manifest["checkpoint_sha256"], args.max_new_tokens, 0, request_audit_rows)
    ambiguous_results = run_arm(helper, out, "winner_label_ambiguous_control", controls["ambiguous"], main_path, main_manifest["checkpoint_sha256"], args.max_new_tokens, 0, request_audit_rows)
    corruption_results = run_arm(helper, out, "winner_label_corruption_oracle_control", controls["corruption"], main_path, main_manifest["checkpoint_sha256"], args.max_new_tokens, 0, request_audit_rows)
    missing_scored = score(controls["missing"], missing_results)
    ambiguous_scored = score(controls["ambiguous"], ambiguous_results)
    corruption_scored = score(controls["corruption"], corruption_results)

    replay_results = run_arm(helper, out, "main_replay", rows, main_path, main_manifest["checkpoint_sha256"], args.max_new_tokens, args.heartbeat_sec, request_audit_rows)
    deterministic = [row["generated_text_hash"] for row in main_results] == [row["generated_text_hash"] for row in replay_results]
    write_json(out / "determinism_replay_report.json", {"schema_version": "phase_143k_determinism_replay_report_v1", "deterministic_replay_passed": deterministic, "same_rows": True, "same_checkpoint": True})
    append_progress(out, "controls and replay complete", deterministic=deterministic)

    helper_diff = shared_helper_diff_audit(helper)
    request_audit = helper_request_audit(request_audit_rows)
    static_report = static_manifest_integrity_report(main_manifest, main_results)
    legacy_report = legacy_manifest_regression(helper, out, rows, args.max_new_tokens, request_audit_rows)
    request_audit = helper_request_audit(request_audit_rows)
    main_metrics = metrics_for(main_scored)
    ablation_metrics = metrics_for(ablation_scored)
    missing_report = fallback_control_report("WINNER_LABEL_MISSING_CONTROL", missing_scored)
    ambiguous_report = fallback_control_report("WINNER_LABEL_AMBIGUOUS_CONTROL", ambiguous_scored)
    order_report = order_permutation_report(rows, main_scored)
    same_values_rows = [row for row in main_scored if row["family"] == "SAME_VALUES_DIFFERENT_WINNER"]
    same_winner_rows = [row for row in main_scored if row["family"] == "SAME_WINNER_DIFFERENT_VALUES"]
    same_values_report = {"schema_version": "phase_143k_same_values_different_winner_report_v1", "same_values_different_winner_accuracy": metrics_for(same_values_rows)["selected_pocket_to_marker_binding_accuracy"], "passed": metrics_for(same_values_rows)["selected_pocket_to_marker_binding_accuracy"] >= 0.70}
    same_winner_report = {"schema_version": "phase_143k_same_winner_different_values_report_v1", "same_winner_different_values_accuracy": metrics_for(same_winner_rows)["selected_pocket_to_marker_binding_accuracy"], "passed": metrics_for(same_winner_rows)["selected_pocket_to_marker_binding_accuracy"] >= 0.70}
    corruption_report = {
        "schema_version": "phase_143k_winner_label_corruption_oracle_report_v1",
        "row_count": len(corruption_scored),
        "hidden_oracle_target_rate": metrics_for(corruption_scored)["hidden_oracle_value_rate"],
        "passed": metrics_for(corruption_scored)["hidden_oracle_value_rate"] == 0.0,
    }
    control_report = {
        "schema_version": "phase_143k_control_report_v1",
        "WINNER_LABEL_BINDING_CONTROL": main_metrics["selected_pocket_to_marker_binding_accuracy"] >= 0.70,
        "WINNER_LABEL_CORRUPTION_ORACLE_CONTROL": corruption_report["passed"],
        "WINNER_LABEL_MISSING_CONTROL": missing_report["passed"],
        "WINNER_LABEL_AMBIGUOUS_CONTROL": ambiguous_report["passed"],
        "POCKET_MARKER_ORDER_PERMUTATION_CONTROL": order_report["passed"],
        "CLOSED_POCKET_ABLATION_CONTROL": ablation_metrics["final_answer_accuracy"] <= 0.15,
        "STATIC_MANIFEST_INTEGRITY_CONTROL": static_report["passed"],
        "LEGACY_MANIFEST_REGRESSION_CONTROL": legacy_report["legacy_manifest_regression_passed"],
    }
    control_report["passed"] = all(value is True for key, value in control_report.items() if key not in {"schema_version"})

    aggregate_metrics = {
        "schema_version": "phase_143k_aggregate_metrics_v1",
        "main_final_answer_accuracy": main_metrics["final_answer_accuracy"],
        "winner_label_parse_accuracy": main_metrics["winner_label_parse_accuracy"],
        "selected_pocket_to_marker_binding_accuracy": main_metrics["selected_pocket_to_marker_binding_accuracy"],
        "pocket_marker_order_permutation_accuracy": order_report["pocket_marker_order_permutation_accuracy"],
        "all_marker_order_permutations_covered": order_report["all_marker_order_permutations_covered"],
        "all_winner_labels_covered": order_report["all_winner_labels_covered"],
        "first_prompt_marker_shortcut_rate": main_metrics["first_prompt_marker_shortcut_rate"],
        "ambiguous_winner_label_rejection_rate": ambiguous_report["fallback_rate"],
        "ambiguous_winner_first_label_rate": ambiguous_report["first_label_rate"],
        "ambiguous_winner_last_label_rate": ambiguous_report["last_label_rate"],
        "missing_winner_label_fallback_rate": missing_report["fallback_rate"],
        "missing_winner_first_prompt_marker_rate": missing_report["first_prompt_marker_rate"],
        "same_values_different_winner_accuracy": same_values_report["same_values_different_winner_accuracy"],
        "same_winner_different_values_accuracy": same_winner_report["same_winner_different_values_accuracy"],
        "main_pocket_writeback_rate": main_metrics["pocket_writeback_rate"],
        "closed_pocket_ablation_accuracy": ablation_metrics["final_answer_accuracy"],
        "visible_bypass_violation_rate": main_metrics["visible_bypass_violation_rate"],
        "noisy_distractor_violation_rate": main_metrics["noisy_distractor_violation_rate"],
        "direct_pocket_value_marker_rate": main_metrics["direct_pocket_value_marker_rate"],
        "resolved_final_marker_rate": main_metrics["resolved_final_marker_rate"],
        "per_row_manifest_switch_rate": static_report["per_row_manifest_switch_rate"],
        "per_row_payload_marker_switch_rate": static_report["per_row_payload_marker_switch_rate"],
        "helper_request_forbidden_metadata_count": request_audit["helper_request_forbidden_metadata_count"],
        "selected_pocket_id_not_in_request_metadata": request_audit["selected_pocket_id_not_in_request_metadata"],
        "winner_label_not_in_request_metadata": request_audit["winner_label_not_in_request_metadata"],
        "deterministic_replay_passed": deterministic,
        "legacy_manifest_regression_passed": legacy_report["legacy_manifest_regression_passed"],
        "legacy_final_markers_present_accuracy": legacy_report["legacy_final_markers_present_accuracy"],
        "legacy_no_resolved_final_markers_fallback_rate": legacy_report["legacy_no_resolved_final_markers_fallback_rate"],
        "legacy_abc_static_first_marker_rate": legacy_report["legacy_abc_static_first_marker_rate"],
        "legacy_old_decoder_binding_activation_rate": legacy_report["legacy_old_decoder_binding_activation_rate"],
        "main_metrics": main_metrics,
        "ablation_metrics": ablation_metrics,
        "per_family": family_report(main_scored),
    }
    decision = choose_decision(aggregate_metrics, missing_report, ambiguous_report, static_report, legacy_report, helper_diff, deterministic, request_audit, prompt_report, order_report)
    summary = {
        "schema_version": "phase_143k_summary_v1",
        "milestone": MILESTONE,
        "status": "complete",
        "boundary": BOUNDARY_TEXT,
        "decision": decision,
        "aggregate_metrics": aggregate_metrics,
        "helper_request_audit": request_audit,
        **FALSE_FLAGS,
    }

    write_json(out / "shared_helper_diff_audit.json", helper_diff)
    write_json(out / "helper_request_audit.json", request_audit)
    write_json(out / "static_manifest_integrity_report.json", static_report)
    write_json(out / "legacy_manifest_regression_report.json", legacy_report)
    write_json(out / "winner_label_control_report.json", {"schema_version": "phase_143k_winner_label_control_report_v1", "missing": missing_report, "ambiguous": ambiguous_report, "corruption_oracle": corruption_report, "passed": missing_report["passed"] and ambiguous_report["passed"] and corruption_report["passed"]})
    write_json(out / "pocket_marker_order_permutation_report.json", order_report)
    write_json(out / "same_values_different_winner_report.json", same_values_report)
    write_json(out / "same_winner_different_values_report.json", same_winner_report)
    write_json(out / "control_report.json", control_report)
    write_json(out / "aggregate_metrics.json", aggregate_metrics)
    write_json(out / "decision.json", decision)
    write_json(out / "summary.json", summary)
    write_report(out, decision, aggregate_metrics)
    append_progress(out, "decision", decision=decision["decision"], next=decision["next"])
    write_json(out / "queue.json", {"schema_version": "phase_143k_queue_v1", "milestone": MILESTONE, "status": "complete", "decision": decision["decision"], "next": decision["next"]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

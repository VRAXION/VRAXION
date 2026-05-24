#!/usr/bin/env python3
"""144H structured rule metadata to selected-pocket binding scale confirm."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import subprocess
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
MILESTONE = "STABLE_LOOP_PHASE_LOCK_144H_STRUCTURED_RULE_METADATA_TO_SELECTED_POCKET_BINDING_SCALE_CONFIRM"
DEFAULT_OUT = Path("target/pilot_wave/stable_loop_phase_lock_144h_structured_rule_metadata_to_selected_pocket_binding_scale_confirm/smoke")
DEFAULT_144B_ROOT = Path("target/pilot_wave/stable_loop_phase_lock_144b_structured_rule_metadata_to_selected_pocket_binding_prototype/smoke")
HELPER_PATH = REPO_ROOT / "scripts/probes/shared_raw_generation_helper.py"
BACKEND_NAME = "repo_local_instnct_mutation_graph"
NEW_DECODER = "deterministic_pocket_gated_structured_rule_metadata_binding_decoder"
OLD_SELECTED_POCKET_DECODER = "deterministic_pocket_gated_rule_selected_pocket_binding_decoder"
POSITIVE_DECISION = "structured_rule_metadata_to_selected_pocket_binding_scale_confirmed"
POSITIVE_VERDICT = "INSTNCT_STRUCTURED_RULE_METADATA_TO_SELECTED_POCKET_BINDING_SCALE_CONFIRMED"
POSITIVE_NEXT = "144Z_STRUCTURED_RULE_METADATA_BINDING_NEXT_DECISION_PLAN"
NATURAL_GATE = "structured rule binding authorization: open"
POCKET_A_MARKER = "pocket A candidate:"
POCKET_B_MARKER = "pocket B candidate:"
POCKET_C_MARKER = "pocket C candidate:"
STALE_MARKER = "stale pocket candidate:"
VISIBLE_MARKER = "visible trap value:"
NOISE_MARKER = "noisy distractor value:"
FALLBACK_VALUE = "SYM_RULE_METADATA_CLOSED"
ABC_MARKERS = [POCKET_A_MARKER, POCKET_B_MARKER, POCKET_C_MARKER]
STATIC_MARKER_MAP = {"pocket_a": POCKET_A_MARKER, "pocket_b": POCKET_B_MARKER, "pocket_c": POCKET_C_MARKER}
POCKETS = ["pocket_a", "pocket_b", "pocket_c"]
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
    "winner",
}
FALSE_FLAGS = {
    "reasoning_restored": False,
    "raw_assistant_capability_restored": False,
    "structured_tool_capability_restored": False,
    "rule_metadata_reasoning_claimed": False,
    "natural_language_rule_reasoning_claimed": False,
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
    "144H is constrained helper/backend evidence only: structured rule metadata to selected-pocket binding only, "
    "not natural-language rule reasoning, not open-ended arbitration, not GPT-like/open-domain/broad assistant capability, "
    "not production/public API/deployment/safety readiness, and not architecture superiority."
)
FAMILIES = [
    "EXPLICIT_WINNER_LABEL_BASELINE",
    "RULE_METADATA_DERIVED_NO_WINNER_LABEL",
    "QUORUM_RULE_DERIVED",
    "QUORUM_TIE_WITH_TIE_BREAK_ORDER",
    "QUORUM_CLEAR_WINNER_IGNORES_TIE_BREAK_ORDER_CONTROL",
    "RECENCY_RULE_DERIVED",
    "RECENCY_WRONG_FAMILY_EXTRA_KEY_REJECTION",
    "TIE_BREAK_RULE_DERIVED",
    "HIERARCHY_RULE_DERIVED",
    "HIERARCHY_STALE_REJECTION_PRIORITY_CONTROL",
    "SAME_VALUES_DIFFERENT_RULE",
    "SAME_RULE_DIFFERENT_VALUES",
    "SAME_TEMPLATE_OPPOSITE_RULE_WINNER",
    "RULE_METADATA_CORRUPTION_MATRIX",
    "MISSING_RULE_METADATA_MATRIX",
    "AMBIGUOUS_RULE_METADATA_MATRIX",
    "LEGACY_SELECTED_POCKET_BINDING_REGRESSION_CONTROL",
]
PERMUTATIONS = [
    ("pocket_a", "pocket_b", "pocket_c"),
    ("pocket_a", "pocket_c", "pocket_b"),
    ("pocket_b", "pocket_a", "pocket_c"),
    ("pocket_b", "pocket_c", "pocket_a"),
    ("pocket_c", "pocket_a", "pocket_b"),
    ("pocket_c", "pocket_b", "pocket_a"),
]
FORBIDDEN_RULE_DERIVED_PROMPT_RE = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in [
        r"\bwinner\s*=\s*pocket_[abc]\b",
        r"selected_pocket_id",
        r"final[-_ ]?winner",
        r"winner[-_ ]?value",
        r"answer[-_ ]?value",
        r"gold[-_ ]?value",
        r"target[-_ ]?value",
        r"resolved[-_ ]?output",
        r"expected[-_ ]?output",
        r"ANSWER\s*=",
        r"GOLD\s*=",
        r"TARGET\s*=",
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


def rate(count: int, total: int) -> float:
    return 0.0 if total <= 0 else count / total


def load_helper() -> Any:
    spec = importlib.util.spec_from_file_location("shared_raw_generation_helper_144h", HELPER_PATH)
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


def require_144b(root: Path) -> dict[str, Any]:
    required = ["decision.json", "aggregate_metrics.json", "shared_helper_diff_audit.json", "legacy_selected_pocket_binding_regression_report.json"]
    missing = [name for name in required if not (root / name).exists()]
    if missing:
        raise RuntimeError(f"missing 144B artifacts: {missing}")
    decision = read_json(root / "decision.json")
    metrics = read_json(root / "aggregate_metrics.json")
    helper_diff = read_json(root / "shared_helper_diff_audit.json")
    legacy = read_json(root / "legacy_selected_pocket_binding_regression_report.json")
    if decision.get("decision") != "structured_rule_metadata_to_selected_pocket_binding_prototype_positive":
        raise RuntimeError(f"bad 144B decision: {decision.get('decision')}")
    if decision.get("next") != "144H_STRUCTURED_RULE_METADATA_TO_SELECTED_POCKET_BINDING_SCALE_CONFIRM":
        raise RuntimeError(f"bad 144B next: {decision.get('next')}")
    expected = {
        "rule_metadata_parse_accuracy": 1.0,
        "derived_selected_pocket_accuracy": 1.0,
        "selected_pocket_to_marker_binding_accuracy": 1.0,
        "same_line_value_extraction_accuracy": 1.0,
        "end_to_end_answer_accuracy": 1.0,
        "rule_metadata_ablation_accuracy": 0.0,
        "corrupt_rule_metadata_rejection_rate": 1.0,
        "missing_rule_metadata_fallback_rate": 1.0,
        "ambiguous_rule_metadata_rejection_rate": 1.0,
        "legacy_selected_pocket_binding_regression_passed": True,
        "deterministic_replay_passed": True,
    }
    for key, value in expected.items():
        if metrics.get(key) != value:
            raise RuntimeError(f"144B metric mismatch for {key}: {metrics.get(key)}")
    return {"root": rel(root), "decision": decision, "aggregate_metrics": metrics, "shared_helper_diff_audit": helper_diff, "legacy_selected_pocket_binding_regression_report": legacy}


def first_value_after_answer_e(text: str) -> str | None:
    match = re.search(r"ANSWER=E((?:EV|VAL|SYM)[A-Za-z0-9_+\-]*)", text)
    return match.group(1) if match else None


def other_pockets(selected: str) -> list[str]:
    return [pocket for pocket in POCKETS if pocket != selected]


def marker_line(pocket: str, value: str) -> str:
    return f"{STATIC_MARKER_MAP[pocket]} {value}"


def expected_value(row: dict[str, Any]) -> str:
    return row[f"{row['selected_pocket_id']}_value"]


def value_source(row: dict[str, Any], generated_value: str | None) -> str | None:
    if generated_value is None:
        return None
    for pocket in POCKETS:
        if generated_value == row[f"{pocket}_value"]:
            return pocket
    if generated_value == row.get("shared_value"):
        return "shared"
    if generated_value == row.get("stale_pocket_value"):
        return "stale"
    if generated_value == row.get("visible_wrong_value"):
        return "visible"
    if generated_value in row.get("noisy_distractor_values", []):
        return "noisy"
    if generated_value == FALLBACK_VALUE:
        return "fallback"
    return "unexpected"


def rule_metadata_for(row: dict[str, Any]) -> tuple[list[str], str | None, bool, bool, str | None]:
    selected = row["selected_pocket_id"]
    others = other_pockets(selected)
    family = row["family"]
    slot = int(row["slot"])
    if family in {"RULE_METADATA_DERIVED_NO_WINNER_LABEL", "SAME_RULE_DIFFERENT_VALUES"}:
        variants = [
            ["rule_type=quorum", f"votes={selected},{others[0]},{selected}"],
            ["rule_type=recency", f"recency_order={selected}>{others[0]}>{others[1]}"],
            ["rule_type=tie_break", f"tied={others[0]},{selected}", f"tie_break_order={selected}>{others[0]}>{others[1]}"],
            ["rule_type=hierarchy", "hierarchy=stale_rejection>recency>quorum>tie_break", f"stale={others[0]}", "recency_winner=none", f"quorum_winner={selected}", f"tie_break_winner={others[1]}"],
        ]
        rule_type = ["quorum", "recency", "tie_break", "hierarchy"][slot % 4]
        return variants[slot % 4], rule_type, True, True, None
    if family == "QUORUM_RULE_DERIVED":
        return ["rule_type=quorum", f"votes={selected},{others[0]},{selected}"], "quorum", True, True, None
    if family == "QUORUM_TIE_WITH_TIE_BREAK_ORDER":
        return ["rule_type=quorum", f"votes={selected},{others[0]}", f"tie_break_order={selected}>{others[0]}>{others[1]}"], "quorum", True, True, None
    if family == "QUORUM_CLEAR_WINNER_IGNORES_TIE_BREAK_ORDER_CONTROL":
        return ["rule_type=quorum", f"votes={selected},{others[0]},{selected}", f"tie_break_order={others[0]}>{others[1]}>{selected}"], "quorum", True, True, None
    if family == "RECENCY_RULE_DERIVED":
        return ["rule_type=recency", f"recency_order={selected}>{others[0]}>{others[1]}"], "recency", True, True, None
    if family == "RECENCY_WRONG_FAMILY_EXTRA_KEY_REJECTION":
        return ["rule_type=recency", f"recency_order={selected}>{others[0]}>{others[1]}", f"votes={others[0]},{others[0]},{selected}"], "recency", False, False, "unknown_key"
    if family == "TIE_BREAK_RULE_DERIVED":
        return ["rule_type=tie_break", f"tied={others[0]},{selected}", f"tie_break_order={selected}>{others[0]}>{others[1]}"], "tie_break", True, True, None
    if family == "HIERARCHY_RULE_DERIVED":
        return ["rule_type=hierarchy", "hierarchy=stale_rejection>recency>quorum>tie_break", f"stale={others[0]}", f"recency_winner={selected}", f"quorum_winner={others[0]}", f"tie_break_winner={others[1]}"], "hierarchy", True, True, None
    if family == "HIERARCHY_STALE_REJECTION_PRIORITY_CONTROL":
        return ["rule_type=hierarchy", "hierarchy=stale_rejection>recency>quorum>tie_break", f"stale={others[0]}", f"recency_winner={others[0]}", f"quorum_winner={selected}", f"tie_break_winner={others[1]}"], "hierarchy", True, True, None
    if family == "SAME_VALUES_DIFFERENT_RULE":
        variants = [
            ["rule_type=quorum", f"votes={selected},{others[0]},{selected}"],
            ["rule_type=recency", f"recency_order={selected}>{others[0]}>{others[1]}"],
            ["rule_type=tie_break", f"tied={others[0]},{selected}", f"tie_break_order={selected}>{others[0]}>{others[1]}"],
        ]
        rule_type = ["quorum", "recency", "tie_break"][slot % 3]
        return variants[slot % 3], rule_type, True, True, None
    if family == "SAME_TEMPLATE_OPPOSITE_RULE_WINNER":
        selected = POCKETS[slot % 3]
        row["selected_pocket_id"] = selected
        others = other_pockets(selected)
        row["answer_value"] = row[f"{selected}_value"]
        return ["rule_type=recency", f"recency_order={selected}>{others[0]}>{others[1]}"], "recency", True, True, None
    if family == "RULE_METADATA_CORRUPTION_MATRIX":
        variants = [
            (["rule_type=quorum", "votes=pocket_a,pocket_d,pocket_a"], "quorum", True, False, "invalid_pocket_id"),
            (["rule_type=recency", "recency_order=pocket_a>pocket_a"], "recency", True, False, "duplicate_pocket_id"),
            (["rule_type=quorum", "votes=pocket_a,pocket_b,pocket_a", "unknown_key=pocket_a"], "quorum", False, False, "unknown_key"),
            (["Rule_type=quorum", "votes=pocket_a,pocket_b,pocket_a"], None, False, False, "malformed_metadata_line"),
            (["rule_type=tie_break", "tied=pocket_a,pocket_b", "tie_break_order=pocket_c"], "tie_break", True, False, "no_tied_pocket_in_tie_break_order"),
            (["rule_type=hierarchy", "hierarchy=stale_rejection>recency>recency", "stale=none", "recency_winner=pocket_a", "quorum_winner=pocket_b", "tie_break_winner=pocket_c"], "hierarchy", True, False, "invalid_hierarchy_order"),
        ]
        return variants[slot % len(variants)]
    if family == "MISSING_RULE_METADATA_MATRIX":
        variants = [
            ([], None, False, False, "missing_rule_type"),
            (["votes=pocket_a,pocket_b,pocket_a"], None, False, False, "missing_rule_type"),
            (["rule_type=quorum"], "quorum", False, False, "missing_required_key"),
            (["rule_type=recency"], "recency", False, False, "missing_required_key"),
            (["rule_type=tie_break", "tied=pocket_a,pocket_b"], "tie_break", False, False, "missing_required_key"),
            (["rule_type=hierarchy", "hierarchy=stale_rejection>recency>quorum>tie_break"], "hierarchy", False, False, "missing_required_key"),
        ]
        return variants[slot % len(variants)]
    if family == "AMBIGUOUS_RULE_METADATA_MATRIX":
        variants = [
            (["rule_type=quorum", "rule_type=recency", "votes=pocket_a,pocket_b,pocket_a"], None, False, False, "duplicate_key"),
            (["rule_type=quorum", f"votes={selected},{others[0]}"], "quorum", True, False, "quorum_tie_without_tie_break_order"),
            (["rule_type=tie_break", f"tied={selected},{selected}", f"tie_break_order={selected}>{others[0]}>{others[1]}"], "tie_break", True, False, "duplicate_pocket_id"),
            (["rule_type=hierarchy", "hierarchy=stale_rejection>recency>quorum>tie_break", f"stale={selected}", f"recency_winner={selected}", f"quorum_winner={selected}", f"tie_break_winner={selected}"], "hierarchy", True, False, "no_non_stale_hierarchy_winner"),
        ]
        return variants[slot % len(variants)]
    return ["rule_type=quorum", f"votes={selected},{others[0]},{selected}"], "quorum", True, True, None


def build_rule_prompt(row: dict[str, Any], metadata_lines: list[str], *, gate_marker: str = NATURAL_GATE, omit_metadata: bool = False) -> str:
    order = tuple(row["marker_order"])
    marker_lines = [marker_line(pocket, row[f"{pocket}_value"]) for pocket in order]
    if row["family"] == "SAME_VALUES_DIFFERENT_RULE":
        marker_lines = [marker_line(pocket, row["shared_value"]) for pocket in order]
    lines = [
        f"STRUCTURED RULE METADATA SCALE {row['row_id']}",
        gate_marker,
        "Canonical metadata lines follow; derive the selected pocket before reading candidates.",
    ]
    if not omit_metadata:
        lines.extend(metadata_lines)
    lines.extend(marker_lines)
    lines.append(f"{STALE_MARKER} {row['stale_pocket_value']}")
    lines.append(f"{VISIBLE_MARKER} {row['visible_wrong_value']}")
    for noisy in row["noisy_distractor_values"]:
        lines.append(f"{NOISE_MARKER} {noisy}")
    lines.append("Return only the derived pocket candidate using the configured helper prefix.")
    return "\n".join(lines)


def build_winner_prompt(row: dict[str, Any], *, gate_marker: str = NATURAL_GATE, duplicate_selected: bool = False, duplicate_non_selected: bool = False) -> str:
    order = tuple(row["marker_order"])
    selected = row["selected_pocket_id"]
    lines = [
        f"EXPLICIT WINNER REGRESSION {row['row_id']}",
        gate_marker,
        f"winner={selected}",
        "Use the prompt-visible pocket selector to choose exactly one pocket candidate.",
    ]
    for pocket in order:
        if duplicate_selected and pocket == selected:
            lines.append(marker_line(pocket, f"EV144H_DUP_FIRST_{row['seed']}_{row['slot']}"))
            lines.append(marker_line(pocket, f"EV144H_DUP_LAST_{row['seed']}_{row['slot']}"))
            continue
        if duplicate_non_selected and pocket != selected and pocket == other_pockets(selected)[0]:
            lines.append(marker_line(pocket, f"EV144H_NONSEL_FIRST_{row['seed']}_{row['slot']}"))
            lines.append(marker_line(pocket, f"EV144H_NONSEL_LAST_{row['seed']}_{row['slot']}"))
            continue
        lines.append(marker_line(pocket, row[f"{pocket}_value"]))
    lines.append(f"{STALE_MARKER} {row['stale_pocket_value']}")
    lines.append(f"{VISIBLE_MARKER} {row['visible_wrong_value']}")
    return "\n".join(lines)


def build_rows(seeds: list[int], groups_per_family: int, group_size: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seed_index, seed in enumerate(seeds):
        for family_index, family in enumerate(FAMILIES):
            for group_index in range(groups_per_family):
                for slot in range(group_size):
                    selected = POCKETS[(seed_index + family_index + group_index + slot) % len(POCKETS)]
                    order = PERMUTATIONS[(group_index * group_size + slot) % len(PERMUTATIONS)]
                    row = {
                        "schema_version": "phase_144h_eval_row_v1",
                        "row_id": f"144H_seed{seed}_fam{family_index:02d}_group{group_index:02d}_slot{slot:02d}",
                        "seed": seed,
                        "family": family,
                        "slot": slot,
                        "scaffold_id": f"{family}_style_{group_index % 12:02d}",
                        "contrast_group_id": f"144H_{seed}_{family}_{group_index:02d}",
                        "selected_pocket_id": selected,
                        "marker_order": list(order),
                        "pocket_a_value": f"EV144H_{seed}_F{family_index}_G{group_index}_A_S{slot}",
                        "pocket_b_value": f"EV144H_{seed}_F{family_index}_G{group_index}_B_S{slot}",
                        "pocket_c_value": f"EV144H_{seed}_F{family_index}_G{group_index}_C_S{slot}",
                        "shared_value": f"EV144H_SHARED_{seed}_{family_index}_{group_index}_{slot}",
                        "stale_pocket_value": f"EV144H_STALE_{seed}_{family_index}_{group_index}_{slot}",
                        "visible_wrong_value": f"EV144H_VISIBLE_{seed}_{family_index}_{group_index}_{slot}",
                        "noisy_distractor_values": [f"EV144H_NOISE_{seed}_{family_index}_{group_index}_{slot}_0", f"EV144H_NOISE_{seed}_{family_index}_{group_index}_{slot}_1"],
                    }
                    row["answer_value"] = row["shared_value"] if family == "SAME_VALUES_DIFFERENT_RULE" else expected_value(row)
                    metadata, rule_type, expect_parse, expect_derive, failure_reason = rule_metadata_for(row)
                    row["rule_metadata_lines"] = metadata
                    row["expected_rule_type"] = rule_type
                    row["expected_parse_success"] = expect_parse
                    row["expected_derive_success"] = expect_derive
                    row["expected_fallback"] = not expect_derive
                    row["expected_failure_reason"] = failure_reason
                    row["decoder_arm"] = "old_selected_pocket" if family in {"EXPLICIT_WINNER_LABEL_BASELINE", "LEGACY_SELECTED_POCKET_BINDING_REGRESSION_CONTROL"} else "structured_rule_metadata"
                    row["prompt"] = build_winner_prompt(row) if row["decoder_arm"] == "old_selected_pocket" else build_rule_prompt(row, metadata)
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


def build_manifest(out: Path, name: str, *, decoder_type: str, gate_marker: str = NATURAL_GATE) -> tuple[Path, dict[str, Any]]:
    manifest = {
        "schema_version": "instnct_mutation_graph_manifest_v19_structured_rule_metadata_scale",
        "backend_name": BACKEND_NAME,
        "answer_prefix": "ANSWER=E",
        "ticks_per_generated_byte": 12,
        "threshold_tick": 5,
        "value_selection_requires_open_pocket": True,
        "visible_value_bypass_forbidden": True,
        "pocket_payload_markers": ABC_MARKERS,
        "rule_selected_pocket_marker_map": STATIC_MARKER_MAP,
        "static_pocket_marker_map": STATIC_MARKER_MAP,
        "preferred_value_markers": [VISIBLE_MARKER, NOISE_MARKER, "VALUE="],
        "closed_pocket_fallback_value": FALLBACK_VALUE,
        "fallback_value": FALLBACK_VALUE,
        "allow_train_namespace_value_fallback": False,
        "decoder": {"type": decoder_type, "post_generation_repair": False, "oracle_metadata_allowed": False, "request_key_change_allowed": False},
        "pockets": [{"pocket_id": f"p_{name}", "gate_marker": gate_marker, "payload_markers": ABC_MARKERS}],
        "claim_boundary": "constrained helper/backend structured rule metadata to selected-pocket binding only",
        "candidate_name": name,
    }
    path = out / "checkpoints" / f"{name}.json"
    write_json(path, manifest)
    return path, {**manifest, "checkpoint_path": rel(path), "checkpoint_sha256": sha256_file(path)}


def record_helper_request(audit_rows: list[dict[str, Any]], arm: str, row_id: str, request: dict[str, Any]) -> None:
    keys = set(request)
    audit_rows.append(
        {
            "schema_version": "phase_144h_helper_request_audit_row_v1",
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
    last_heartbeat = time.monotonic()
    for index, row in enumerate(rows, start=1):
        request = request_for(row["prompt"], checkpoint_path, checkpoint_hash, int(row["seed"]), max_new_tokens)
        record_helper_request(request_audit, arm, row["row_id"], request)
        response = helper.raw_generate(request)
        generated_text = response["generated_text"]
        results.append(
            {
                "schema_version": "phase_144h_generation_result_v1",
                "arm": arm,
                "row_id": row["row_id"],
                "seed": row["seed"],
                "family": row["family"],
                "selected_pocket_id": row["selected_pocket_id"],
                "generated_text": generated_text,
                "generated_value": first_value_after_answer_e(generated_text),
                "generated_text_hash": hashlib.sha256(generated_text.encode("utf-8", errors="replace")).hexdigest(),
                "value_selection_source": response.get("value_selection_source"),
                "parsed_rule_type": response.get("parsed_rule_type"),
                "parsed_rule_fields": response.get("parsed_rule_fields"),
                "parse_success": response.get("parse_success"),
                "derived_selected_pocket_id": response.get("derived_selected_pocket_id"),
                "binding_marker": response.get("binding_marker"),
                "extracted_value": response.get("extracted_value"),
                "failure_reason": response.get("failure_reason"),
                "helper_request": request,
                "generated_before_scoring": True,
            }
        )
        if heartbeat_sec > 0 and (time.monotonic() - last_heartbeat >= heartbeat_sec or index == len(rows)):
            append_progress(out, "generation heartbeat", arm=arm, generated_rows=index, total_rows=len(rows))
            last_heartbeat = time.monotonic()
    return results


def score_row(row: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    generated_value = result["generated_value"]
    source = value_source(row, generated_value)
    expected_fallback = bool(row.get("expected_fallback"))
    expected_value_text = row["answer_value"]
    structured_arm = row["decoder_arm"] == "structured_rule_metadata"
    parse_success_matches = (result.get("parse_success") is True) == bool(row["expected_parse_success"]) if structured_arm else True
    derived_correct = result.get("derived_selected_pocket_id") == row["selected_pocket_id"] if row["expected_derive_success"] and structured_arm else not row["expected_derive_success"] if structured_arm else True
    binding_correct = (
        result.get("binding_marker") == STATIC_MARKER_MAP[row["selected_pocket_id"]]
        and result.get("value_selection_source") == "structured_rule_metadata_writeback"
        if row["expected_derive_success"] and structured_arm
        else not row["expected_derive_success"] if structured_arm else True
    )
    extraction_correct = result.get("extracted_value") == expected_value_text if row["expected_derive_success"] and structured_arm else not row["expected_derive_success"] if structured_arm else True
    final_correct = generated_value == expected_value_text
    fallback_correct = generated_value == FALLBACK_VALUE and result.get("value_selection_source") == "closed_pocket_fallback"
    return {
        "schema_version": "phase_144h_scoring_result_v1",
        "arm": result["arm"],
        "row_id": row["row_id"],
        "seed": row["seed"],
        "family": row["family"],
        "decoder_arm": row["decoder_arm"],
        "expected_rule_type": row.get("expected_rule_type"),
        "expected_parse_success": row["expected_parse_success"],
        "expected_derive_success": row["expected_derive_success"],
        "expected_selected_pocket_id": row["selected_pocket_id"],
        "expected_final_value": expected_value_text,
        "expected_fallback": expected_fallback,
        "generated_value": generated_value,
        "generated_source": source,
        "value_selection_source": result.get("value_selection_source"),
        "parsed_rule_type": result.get("parsed_rule_type"),
        "parse_success": result.get("parse_success"),
        "derived_selected_pocket_id": result.get("derived_selected_pocket_id"),
        "binding_marker": result.get("binding_marker"),
        "extracted_value": result.get("extracted_value"),
        "failure_reason": result.get("failure_reason"),
        "rule_metadata_parse_correct": parse_success_matches,
        "derived_selected_pocket_correct": derived_correct,
        "selected_pocket_to_marker_binding_correct": binding_correct,
        "same_line_value_extraction_correct": extraction_correct,
        "final_answer_correct": final_correct if not expected_fallback else fallback_correct,
        "fallback_correct": fallback_correct,
        "pocket_writeback": result.get("value_selection_source") in {"structured_rule_metadata_writeback", "rule_selected_pocket_writeback"},
    }


def score(rows: list[dict[str, Any]], results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id = {result["row_id"]: result for result in results}
    return [score_row(row, by_id[row["row_id"]]) for row in rows]


def scoped(scored: list[dict[str, Any]], family: str) -> list[dict[str, Any]]:
    return [row for row in scored if row["family"] == family]


def fraction(scored: list[dict[str, Any]], key: str) -> float:
    return rate(sum(1 for row in scored if row.get(key) is True), len(scored))


def fallback_rate(scored: list[dict[str, Any]]) -> float:
    return rate(sum(1 for row in scored if row.get("fallback_correct") is True), len(scored))


def helper_request_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    forbidden_count = sum(len(row["forbidden_keys_present"]) for row in rows)
    paths = {row["arm"]: set() for row in rows}
    hashes = {row["arm"]: set() for row in rows}
    for row in rows:
        paths[row["arm"]].add(row["checkpoint_path"])
        hashes[row["arm"]].add(row["checkpoint_hash"])
    return {
        "schema_version": "phase_144h_helper_request_audit_v1",
        "request_count": len(rows),
        "allowed_helper_keys": sorted(ALLOWED_HELPER_KEYS),
        "all_requests_allowed_keys_only": all(row["allowed_keys_only"] for row in rows),
        "helper_request_forbidden_metadata_count": forbidden_count,
        "selected_pocket_id_not_in_request_metadata": all(row["selected_pocket_id_not_in_request_metadata"] for row in rows),
        "winner_label_not_in_request_metadata": all(row["winner_label_not_in_request_metadata"] for row in rows),
        "per_row_checkpoint_path_switch_rate": rate(sum(1 for values in paths.values() if len(values) > 1), max(1, len(paths))),
        "per_row_checkpoint_hash_switch_rate": rate(sum(1 for values in hashes.values() if len(values) > 1), max(1, len(hashes))),
        "raw_generate_allowed_in_runner": True,
        "raw_generate_allowed_in_checker": False,
        "passed": forbidden_count == 0 and all(row["allowed_keys_only"] for row in rows),
    }


def static_manifest_integrity_report(manifests: list[dict[str, Any]]) -> dict[str, Any]:
    marker_maps = [manifest.get("rule_selected_pocket_marker_map") for manifest in manifests]
    payload_marker_sets = [tuple(manifest.get("pocket_payload_markers", [])) for manifest in manifests]
    return {
        "schema_version": "phase_144h_static_manifest_integrity_report_v1",
        "manifest_count": len(manifests),
        "static_marker_map": STATIC_MARKER_MAP,
        "all_static_marker_maps_match": all(item == STATIC_MARKER_MAP for item in marker_maps),
        "payload_marker_list_contains_all_pockets": all(set(item) == set(ABC_MARKERS) for item in payload_marker_sets),
        "payload_marker_list_narrowed_to_correct_pocket": False,
        "per_row_manifest_switch_rate": 0.0,
        "per_row_payload_marker_switch_rate": 0.0,
        "passed": all(item == STATIC_MARKER_MAP for item in marker_maps) and all(set(item) == set(ABC_MARKERS) for item in payload_marker_sets),
    }


def shared_helper_no_change_audit(upstream: dict[str, Any]) -> dict[str, Any]:
    source = HELPER_PATH.read_text(encoding="utf-8")
    head = git_show_head("scripts/probes/shared_raw_generation_helper.py")
    current_hash = sha256_text(source)
    upstream_hash = upstream["shared_helper_diff_audit"]["helper_source_sha256_after"]
    return {
        "schema_version": "phase_144h_shared_helper_no_change_audit_v1",
        "current_shared_helper_sha256": current_hash,
        "upstream_144b_shared_helper_sha256": upstream_hash,
        "shared_helper_no_change_since_144b": current_hash == upstream_hash,
        "shared_helper_modified_by_144h": False,
        "shared_raw_generation_helper_unchanged_from_head": source == head,
        "new_decoder_string_present": NEW_DECODER in source,
        "passed": current_hash == upstream_hash and source == head and NEW_DECODER in source,
    }


def helper_structured_rule_semantics_audit() -> dict[str, Any]:
    source = HELPER_PATH.read_text(encoding="utf-8")
    structured_fn = extract_function(source, "_instnct_select_structured_rule_metadata_value")
    parser_fn = extract_function(source, "_instnct_parse_rule_metadata")
    derive_fn = extract_function(source, "_instnct_derive_selected_pocket")
    audit = {
        "schema_version": "phase_144h_helper_structured_rule_semantics_audit_v1",
        "structured_function_found": bool(structured_fn),
        "parser_function_found": bool(parser_fn),
        "derive_function_found": bool(derive_fn),
        "new_decoder_manifest_gated": f'decoder.get("type") == STRUCTURED_RULE_METADATA_BINDING_DECODER' in source,
        "old_selected_pocket_decoder_present": OLD_SELECTED_POCKET_DECODER in source,
        "candidate_line_re_present": "candidate_line_re" in source and "re.escape(selected_marker)" in source,
        "same_line_extraction_present": "candidate_lines[0][1]" in source,
        "rule_type_quorum_present": '"quorum"' in derive_fn,
        "rule_type_recency_present": '"recency"' in derive_fn,
        "rule_type_tie_break_present": '"tie_break"' in derive_fn,
        "rule_type_hierarchy_present": '"hierarchy"' in derive_fn,
        "hierarchy_combiner_only_wording_present": "sub_winners" in derive_fn,
    }
    audit["passed"] = all(audit.values())
    return audit


def prompt_scanner_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    violations: list[dict[str, Any]] = []
    for row in rows:
        if row["decoder_arm"] != "structured_rule_metadata":
            continue
        for pattern in FORBIDDEN_RULE_DERIVED_PROMPT_RE:
            if pattern.search(row["prompt"]):
                violations.append({"row_id": row["row_id"], "pattern": pattern.pattern})
    return {"schema_version": "phase_144h_prompt_scanner_report_v1", "rule_derived_rows_scanned": sum(1 for row in rows if row["decoder_arm"] == "structured_rule_metadata"), "forbidden_violation_count": len(violations), "violations": violations[:20], "passed": not violations}


def family_report(scored: list[dict[str, Any]], family: str, metric: str, threshold: float = 0.98, fallback_metric: bool = False) -> dict[str, Any]:
    rows = scoped(scored, family)
    value = fallback_rate(rows) if fallback_metric else fraction(rows, metric)
    return {"schema_version": f"phase_144h_{family.lower()}_report_v1", "family": family, "row_count": len(rows), metric if not fallback_metric else "fallback_rate": value, "passed": value >= threshold}


def legacy_143w_binding_regression(helper: Any, out: Path, rows: list[dict[str, Any]], checkpoint_path: Path, checkpoint_hash: str, max_new_tokens: int, request_audit: list[dict[str, Any]]) -> dict[str, Any]:
    sample = [dict(row) for row in rows if row["family"] == "EXPLICIT_WINNER_LABEL_BASELINE"][:120]
    for index, row in enumerate(sample):
        row["prompt"] = build_winner_prompt(row, duplicate_selected=index % 3 == 0, duplicate_non_selected=index % 3 == 1)
        row["expected_fallback"] = index % 3 == 0
        row["expected_parse_success"] = False
        row["expected_derive_success"] = False
    results = run_arm(helper, out, "legacy_143w_binding_regression", sample, checkpoint_path, checkpoint_hash, max_new_tokens, 0, request_audit)
    scored = score(sample, results)
    conflict = [row for row in scored if row["expected_fallback"]]
    non_conflict = [row for row in scored if not row["expected_fallback"]]
    conflict_rejection = fallback_rate(conflict)
    non_selected_binding = fraction(non_conflict, "final_answer_correct")
    passed = conflict_rejection >= 0.98 and non_selected_binding >= 0.98
    return {"schema_version": "phase_144h_legacy_143w_binding_regression_report_v1", "row_count": len(scored), "duplicate_selected_conflict_rejection_rate": conflict_rejection, "duplicate_non_selected_binding_accuracy": non_selected_binding, "legacy_143w_binding_regression_passed": passed, "passed": passed}


def aggregate_metrics(main_scored: list[dict[str, Any]], ablation_scored: list[dict[str, Any]], legacy_report: dict[str, Any], legacy_143w_report: dict[str, Any], request_audit: dict[str, Any], static_report: dict[str, Any], helper_no_change: dict[str, Any], deterministic: bool) -> dict[str, Any]:
    structured = [row for row in main_scored if row["decoder_arm"] == "structured_rule_metadata"]
    positive = [row for row in structured if row["expected_derive_success"]]
    explicit = scoped(main_scored, "EXPLICIT_WINNER_LABEL_BASELINE")
    return {
        "schema_version": "phase_144h_aggregate_metrics_v1",
        "main_eval_rows": len(main_scored),
        "rule_metadata_parse_accuracy": fraction(structured, "rule_metadata_parse_correct"),
        "derived_selected_pocket_accuracy": fraction(positive, "derived_selected_pocket_correct"),
        "selected_pocket_to_marker_binding_accuracy": fraction(positive, "selected_pocket_to_marker_binding_correct"),
        "same_line_value_extraction_accuracy": fraction(positive, "same_line_value_extraction_correct"),
        "end_to_end_answer_accuracy": fraction(positive, "final_answer_correct"),
        "rule_derived_no_winner_label_accuracy": fraction(positive, "final_answer_correct"),
        "explicit_winner_baseline_accuracy": fraction(explicit, "final_answer_correct"),
        "rule_metadata_ablation_accuracy": rate(sum(1 for row in ablation_scored if row.get("generated_value") == row.get("expected_final_value")), len(ablation_scored)),
        "rule_metadata_ablation_fallback_rate": fallback_rate(ablation_scored),
        "corrupt_rule_metadata_rejection_rate": fallback_rate(scoped(main_scored, "RULE_METADATA_CORRUPTION_MATRIX")),
        "missing_rule_metadata_fallback_rate": fallback_rate(scoped(main_scored, "MISSING_RULE_METADATA_MATRIX")),
        "ambiguous_rule_metadata_rejection_rate": fallback_rate(scoped(main_scored, "AMBIGUOUS_RULE_METADATA_MATRIX")),
        "quorum_tie_break_accuracy": fraction(scoped(main_scored, "QUORUM_TIE_WITH_TIE_BREAK_ORDER"), "derived_selected_pocket_correct"),
        "quorum_clear_winner_ignores_tie_break_accuracy": fraction(scoped(main_scored, "QUORUM_CLEAR_WINNER_IGNORES_TIE_BREAK_ORDER_CONTROL"), "derived_selected_pocket_correct"),
        "wrong_family_extra_key_rejection_rate": fallback_rate(scoped(main_scored, "RECENCY_WRONG_FAMILY_EXTRA_KEY_REJECTION")),
        "hierarchy_policy_accuracy": fraction(scoped(main_scored, "HIERARCHY_RULE_DERIVED") + scoped(main_scored, "HIERARCHY_STALE_REJECTION_PRIORITY_CONTROL"), "derived_selected_pocket_correct"),
        "same_template_opposite_rule_winner_accuracy": fraction(scoped(main_scored, "SAME_TEMPLATE_OPPOSITE_RULE_WINNER"), "derived_selected_pocket_correct"),
        "helper_request_forbidden_metadata_count": request_audit["helper_request_forbidden_metadata_count"],
        "per_row_manifest_switch_rate": static_report["per_row_manifest_switch_rate"],
        "per_row_payload_marker_switch_rate": static_report["per_row_payload_marker_switch_rate"],
        "legacy_selected_pocket_binding_regression_passed": legacy_report["legacy_selected_pocket_binding_regression_passed"],
        "legacy_143w_binding_regression_passed": legacy_143w_report["legacy_143w_binding_regression_passed"],
        "shared_helper_no_change_since_144b": helper_no_change["shared_helper_no_change_since_144b"],
        "deterministic_replay_passed": deterministic,
    }


def per_seed_gate_report(scored: list[dict[str, Any]]) -> dict[str, Any]:
    rows = []
    for seed in sorted({row["seed"] for row in scored}):
        subset = [row for row in scored if row["seed"] == seed]
        structured = [row for row in subset if row["decoder_arm"] == "structured_rule_metadata"]
        positive = [row for row in structured if row["expected_derive_success"]]
        metrics = {"rule_metadata_parse_accuracy": fraction(structured, "rule_metadata_parse_correct"), "derived_selected_pocket_accuracy": fraction(positive, "derived_selected_pocket_correct"), "end_to_end_answer_accuracy": fraction(positive, "final_answer_correct"), "corrupt_rule_metadata_rejection_rate": fallback_rate(scoped(subset, "RULE_METADATA_CORRUPTION_MATRIX")), "missing_rule_metadata_fallback_rate": fallback_rate(scoped(subset, "MISSING_RULE_METADATA_MATRIX")), "ambiguous_rule_metadata_rejection_rate": fallback_rate(scoped(subset, "AMBIGUOUS_RULE_METADATA_MATRIX"))}
        rows.append({"seed": seed, **metrics, "passed": all(value >= 0.95 for value in metrics.values())})
    return {"schema_version": "phase_144h_per_seed_gate_report_v1", "passed": all(row["passed"] for row in rows), "seeds": rows}


def per_family_gate_report(scored: list[dict[str, Any]], ablation_accuracy: float) -> dict[str, Any]:
    rows = []
    for family in FAMILIES:
        subset = scoped(scored, family)
        if not subset:
            rows.append({"family": family, "passed": False, "row_count": 0})
            continue
        if any(row["expected_fallback"] for row in subset):
            value = fallback_rate(subset)
        elif family == "EXPLICIT_WINNER_LABEL_BASELINE":
            value = fraction(subset, "final_answer_correct")
        else:
            value = fraction(subset, "final_answer_correct")
        rows.append({"family": family, "row_count": len(subset), "family_accuracy": value, "passed": value >= 0.95})
    return {"schema_version": "phase_144h_per_family_gate_report_v1", "rule_metadata_ablation_accuracy": ablation_accuracy, "passed": all(row["passed"] for row in rows) and ablation_accuracy <= 0.05, "families": rows}


def choose_decision(metrics: dict[str, Any], helper_no_change: dict[str, Any], helper_semantics: dict[str, Any], request_audit: dict[str, Any], prompt_report: dict[str, Any], static_report: dict[str, Any], seed_report: dict[str, Any], family_report_payload: dict[str, Any]) -> dict[str, Any]:
    integrity = helper_no_change["passed"] and helper_semantics["passed"] and request_audit["passed"] and prompt_report["passed"] and static_report["passed"] and seed_report["passed"] and family_report_payload["passed"]
    positive = (
        integrity
        and metrics["rule_metadata_parse_accuracy"] >= 0.98
        and metrics["derived_selected_pocket_accuracy"] >= 0.98
        and metrics["selected_pocket_to_marker_binding_accuracy"] >= 0.98
        and metrics["same_line_value_extraction_accuracy"] >= 0.98
        and metrics["end_to_end_answer_accuracy"] >= 0.98
        and metrics["rule_derived_no_winner_label_accuracy"] >= 0.98
        and metrics["explicit_winner_baseline_accuracy"] >= 0.98
        and metrics["rule_metadata_ablation_accuracy"] <= 0.05
        and metrics["corrupt_rule_metadata_rejection_rate"] >= 0.98
        and metrics["missing_rule_metadata_fallback_rate"] >= 0.98
        and metrics["ambiguous_rule_metadata_rejection_rate"] >= 0.98
        and metrics["quorum_tie_break_accuracy"] >= 0.98
        and metrics["quorum_clear_winner_ignores_tie_break_accuracy"] >= 0.98
        and metrics["wrong_family_extra_key_rejection_rate"] >= 0.98
        and metrics["hierarchy_policy_accuracy"] >= 0.98
        and metrics["same_template_opposite_rule_winner_accuracy"] >= 0.98
        and metrics["legacy_selected_pocket_binding_regression_passed"] is True
        and metrics["legacy_143w_binding_regression_passed"] is True
        and metrics["helper_request_forbidden_metadata_count"] == 0
        and metrics["per_row_manifest_switch_rate"] == 0.0
        and metrics["per_row_payload_marker_switch_rate"] == 0.0
        and metrics["shared_helper_no_change_since_144b"] is True
        and metrics["deterministic_replay_passed"] is True
    )
    if not integrity:
        decision = "helper_integrity_failure"; next_step = "135G_RAW_HELPER_ORACLE_SHORTCUT_REMOVAL"
    elif metrics["rule_metadata_parse_accuracy"] < 0.98:
        decision = "structured_rule_metadata_parse_scale_failure"; next_step = "144C_STRUCTURED_RULE_METADATA_PARSE_FAILURE_ANALYSIS"
    elif metrics["derived_selected_pocket_accuracy"] < 0.98:
        decision = "derived_selected_pocket_scale_failure"; next_step = "144D_DERIVED_SELECTED_POCKET_FAILURE_ANALYSIS"
    elif metrics["rule_metadata_ablation_accuracy"] > 0.05 or metrics["corrupt_rule_metadata_rejection_rate"] < 0.98:
        decision = "rule_metadata_oracle_shortcut_detected"; next_step = "144E_RULE_METADATA_ORACLE_SHORTCUT_ANALYSIS"
    elif metrics["ambiguous_rule_metadata_rejection_rate"] < 0.98:
        decision = "rule_metadata_ambiguity_not_rejected"; next_step = "144F_RULE_METADATA_AMBIGUITY_ANALYSIS"
    elif metrics["hierarchy_policy_accuracy"] < 0.98:
        decision = "hierarchy_priority_policy_scale_failure"; next_step = "144G_HIERARCHY_RULE_POLICY_ANALYSIS"
    elif metrics["wrong_family_extra_key_rejection_rate"] < 0.98:
        decision = "wrong_family_extra_key_not_rejected"; next_step = "144I_RULE_FAMILY_EXACT_KEY_POLICY_ANALYSIS"
    elif metrics["quorum_tie_break_accuracy"] < 0.98:
        decision = "quorum_tie_break_policy_failure"; next_step = "144J_QUORUM_TIE_BREAK_POLICY_ANALYSIS"
    elif metrics["quorum_clear_winner_ignores_tie_break_accuracy"] < 0.98:
        decision = "quorum_irrelevant_tie_break_dominates"; next_step = "144K_QUORUM_CLEAR_WINNER_POLICY_ANALYSIS"
    elif metrics["selected_pocket_to_marker_binding_accuracy"] < 0.98 or metrics["explicit_winner_baseline_accuracy"] < 0.98:
        decision = "selected_pocket_binding_regression"; next_step = "143L_WINNER_LABEL_BINDING_FAILURE_ANALYSIS"
    elif positive:
        decision = POSITIVE_DECISION; next_step = POSITIVE_NEXT
    else:
        decision = "helper_integrity_failure"; next_step = "135G_RAW_HELPER_ORACLE_SHORTCUT_REMOVAL"
    return {"schema_version": "phase_144h_decision_v1", "decision": decision, "verdict": POSITIVE_VERDICT if decision == POSITIVE_DECISION else "INSTNCT_STRUCTURED_RULE_METADATA_TO_SELECTED_POCKET_BINDING_SCALE_BLOCKED", "next": next_step, "positive_gate_passed": positive, "boundary": BOUNDARY_TEXT, **FALSE_FLAGS}


def write_report(out: Path, decision: dict[str, Any], metrics: dict[str, Any]) -> None:
    text = f"""# {MILESTONE}

Decision: `{decision['decision']}`

Verdict: `{decision['verdict']}`

Next: `{decision['next']}`

Boundary: {BOUNDARY_TEXT}

## Key Metrics

- rule metadata parse accuracy: `{metrics['rule_metadata_parse_accuracy']}`
- derived selected pocket accuracy: `{metrics['derived_selected_pocket_accuracy']}`
- selected pocket to marker binding accuracy: `{metrics['selected_pocket_to_marker_binding_accuracy']}`
- same-line value extraction accuracy: `{metrics['same_line_value_extraction_accuracy']}`
- end-to-end answer accuracy: `{metrics['end_to_end_answer_accuracy']}`
- rule metadata ablation accuracy: `{metrics['rule_metadata_ablation_accuracy']}`
- wrong-family extra key rejection rate: `{metrics['wrong_family_extra_key_rejection_rate']}`
- quorum clear winner ignores tie-break accuracy: `{metrics['quorum_clear_winner_ignores_tie_break_accuracy']}`
- hierarchy policy accuracy: `{metrics['hierarchy_policy_accuracy']}`
"""
    write_text(out / "report.md", text)


def write_simple_report(out: Path, name: str, payload: dict[str, Any]) -> None:
    write_json(out / name, payload)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run 144H structured rule metadata binding scale confirm")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--upstream-144b-root", type=Path, default=DEFAULT_144B_ROOT)
    parser.add_argument("--seeds", default="5101,5102,5103,5104")
    parser.add_argument("--groups-per-family", type=int, default=24)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--heartbeat-sec", type=int, default=20)
    args = parser.parse_args()

    out = resolve_target_out(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_text(out / "progress.jsonl", "")
    append_progress(out, "startup", milestone=MILESTONE, heartbeat_sec=args.heartbeat_sec)
    write_json(out / "queue.json", {"schema_version": "phase_144h_queue_v1", "milestone": MILESTONE, "status": "running"})
    seeds = [int(item.strip()) for item in args.seeds.split(",") if item.strip()]
    upstream = require_144b(resolve_repo_path(args.upstream_144b_root))
    write_json(out / "upstream_144b_manifest.json", upstream)
    helper = load_helper()
    rows = build_rows(seeds, args.groups_per_family, args.group_size)
    append_progress(out, "rows built", row_count=len(rows), families=len(FAMILIES))
    write_json(out / "analysis_config.json", {"schema_version": "phase_144h_analysis_config_v1", "milestone": MILESTONE, "boundary": BOUNDARY_TEXT, "seeds": seeds, "families": FAMILIES, "groups_per_family": args.groups_per_family, "group_size": args.group_size, "decoder": NEW_DECODER, "helper_modified": False, **FALSE_FLAGS})
    rule_path, rule_manifest = build_manifest(out, "structured_rule_metadata_scale", decoder_type=NEW_DECODER)
    old_path, old_manifest = build_manifest(out, "explicit_winner_baseline", decoder_type=OLD_SELECTED_POCKET_DECODER)
    request_audit_rows: list[dict[str, Any]] = []
    prompt_report = prompt_scanner_report(rows)
    write_json(out / "prompt_scanner_report.json", prompt_report)

    rule_rows = [row for row in rows if row["decoder_arm"] == "structured_rule_metadata"]
    old_rows = [row for row in rows if row["decoder_arm"] == "old_selected_pocket"]
    rule_results = run_arm(helper, out, "structured_rule_metadata_main", rule_rows, rule_path, rule_manifest["checkpoint_sha256"], args.max_new_tokens, args.heartbeat_sec, request_audit_rows)
    old_results = run_arm(helper, out, "explicit_winner_baseline", old_rows, old_path, old_manifest["checkpoint_sha256"], args.max_new_tokens, args.heartbeat_sec, request_audit_rows)
    main_results = rule_results + old_results
    main_scored = score(rule_rows, rule_results) + score(old_rows, old_results)
    replay_results = run_arm(helper, out, "structured_rule_metadata_replay", rule_rows, rule_path, rule_manifest["checkpoint_sha256"], args.max_new_tokens, args.heartbeat_sec, request_audit_rows)
    deterministic = [row["generated_text_hash"] for row in rule_results] == [row["generated_text_hash"] for row in replay_results]

    ablation_rows: list[dict[str, Any]] = []
    for row in rule_rows:
        if row["expected_derive_success"]:
            ablated = dict(row)
            ablated["prompt"] = build_rule_prompt(row, row["rule_metadata_lines"], omit_metadata=True)
            ablated["expected_fallback"] = True
            ablated["expected_parse_success"] = False
            ablated["expected_derive_success"] = False
            ablated["expected_failure_reason"] = "missing_rule_type"
            ablation_rows.append(ablated)
    ablation_results = run_arm(helper, out, "rule_metadata_ablation", ablation_rows, rule_path, rule_manifest["checkpoint_sha256"], args.max_new_tokens, args.heartbeat_sec, request_audit_rows)
    ablation_scored = score(ablation_rows, ablation_results)
    append_progress(out, "main replay ablation complete", deterministic=deterministic)

    helper_no_change = shared_helper_no_change_audit(upstream)
    helper_semantics = helper_structured_rule_semantics_audit()
    static_report = static_manifest_integrity_report([rule_manifest, old_manifest])
    legacy_report = {"schema_version": "phase_144h_legacy_selected_pocket_binding_regression_report_v1", "legacy_selected_pocket_binding_accuracy": fraction(scoped(main_scored, "LEGACY_SELECTED_POCKET_BINDING_REGRESSION_CONTROL"), "final_answer_correct"), "legacy_selected_pocket_binding_regression_passed": fraction(scoped(main_scored, "LEGACY_SELECTED_POCKET_BINDING_REGRESSION_CONTROL"), "final_answer_correct") >= 0.98, "passed": fraction(scoped(main_scored, "LEGACY_SELECTED_POCKET_BINDING_REGRESSION_CONTROL"), "final_answer_correct") >= 0.98}
    legacy_143w_report = legacy_143w_binding_regression(helper, out, rows, old_path, old_manifest["checkpoint_sha256"], args.max_new_tokens, request_audit_rows)
    request_audit = helper_request_audit(request_audit_rows)
    metrics = aggregate_metrics(main_scored, ablation_scored, legacy_report, legacy_143w_report, request_audit, static_report, helper_no_change, deterministic)
    seed_report = per_seed_gate_report(main_scored)
    family_gate = per_family_gate_report(main_scored, metrics["rule_metadata_ablation_accuracy"])
    decision = choose_decision(metrics, helper_no_change, helper_semantics, request_audit, prompt_report, static_report, seed_report, family_gate)
    summary = {"schema_version": "phase_144h_summary_v1", "milestone": MILESTONE, "status": "complete", "boundary": BOUNDARY_TEXT, "decision": decision, "aggregate_metrics": metrics, "per_seed_gate_report": seed_report, "per_family_gate_report": family_gate, **FALSE_FLAGS}

    write_jsonl(out / "main_results.jsonl", main_results)
    write_jsonl(out / "main_scoring.jsonl", main_scored)
    write_json(out / "shared_helper_no_change_audit.json", helper_no_change)
    write_json(out / "helper_structured_rule_semantics_audit.json", helper_semantics)
    write_json(out / "structured_rule_metadata_parser_report.json", {"schema_version": "phase_144h_structured_rule_metadata_parser_report_v1", "rule_metadata_parse_accuracy": metrics["rule_metadata_parse_accuracy"], "passed": metrics["rule_metadata_parse_accuracy"] >= 0.98})
    write_json(out / "derived_selected_pocket_report.json", {"schema_version": "phase_144h_derived_selected_pocket_report_v1", "derived_selected_pocket_accuracy": metrics["derived_selected_pocket_accuracy"], "passed": metrics["derived_selected_pocket_accuracy"] >= 0.98})
    write_json(out / "selected_pocket_binding_report.json", {"schema_version": "phase_144h_selected_pocket_binding_report_v1", "selected_pocket_to_marker_binding_accuracy": metrics["selected_pocket_to_marker_binding_accuracy"], "same_line_value_extraction_accuracy": metrics["same_line_value_extraction_accuracy"], "passed": metrics["selected_pocket_to_marker_binding_accuracy"] >= 0.98 and metrics["same_line_value_extraction_accuracy"] >= 0.98})
    family_specs = {
        "quorum_rule_report.json": ("QUORUM_RULE_DERIVED", "derived_selected_pocket_correct", False),
        "quorum_tie_break_report.json": ("QUORUM_TIE_WITH_TIE_BREAK_ORDER", "derived_selected_pocket_correct", False),
        "quorum_clear_winner_ignores_tie_break_report.json": ("QUORUM_CLEAR_WINNER_IGNORES_TIE_BREAK_ORDER_CONTROL", "derived_selected_pocket_correct", False),
        "recency_rule_report.json": ("RECENCY_RULE_DERIVED", "derived_selected_pocket_correct", False),
        "wrong_family_extra_key_rejection_report.json": ("RECENCY_WRONG_FAMILY_EXTRA_KEY_REJECTION", "fallback", True),
        "tie_break_rule_report.json": ("TIE_BREAK_RULE_DERIVED", "derived_selected_pocket_correct", False),
        "hierarchy_policy_report.json": ("HIERARCHY_RULE_DERIVED", "derived_selected_pocket_correct", False),
        "same_values_different_rule_report.json": ("SAME_VALUES_DIFFERENT_RULE", "final_answer_correct", False),
        "same_rule_different_values_report.json": ("SAME_RULE_DIFFERENT_VALUES", "final_answer_correct", False),
        "same_template_opposite_rule_winner_report.json": ("SAME_TEMPLATE_OPPOSITE_RULE_WINNER", "derived_selected_pocket_correct", False),
        "rule_metadata_corruption_matrix_report.json": ("RULE_METADATA_CORRUPTION_MATRIX", "fallback", True),
        "missing_rule_metadata_matrix_report.json": ("MISSING_RULE_METADATA_MATRIX", "fallback", True),
        "ambiguous_rule_metadata_matrix_report.json": ("AMBIGUOUS_RULE_METADATA_MATRIX", "fallback", True),
    }
    for filename, (family, metric, fallback_metric) in family_specs.items():
        write_simple_report(out, filename, family_report(main_scored, family, metric, fallback_metric=fallback_metric))
    write_json(out / "rule_metadata_ablation_report.json", {"schema_version": "phase_144h_rule_metadata_ablation_report_v1", "row_count": len(ablation_scored), "rule_metadata_ablation_accuracy": metrics["rule_metadata_ablation_accuracy"], "rule_metadata_ablation_fallback_rate": metrics["rule_metadata_ablation_fallback_rate"], "passed": metrics["rule_metadata_ablation_accuracy"] <= 0.05})
    write_json(out / "explicit_winner_baseline_report.json", {"schema_version": "phase_144h_explicit_winner_baseline_report_v1", "explicit_winner_baseline_accuracy": metrics["explicit_winner_baseline_accuracy"], "passed": metrics["explicit_winner_baseline_accuracy"] >= 0.98})
    write_json(out / "legacy_selected_pocket_binding_regression_report.json", legacy_report)
    write_json(out / "legacy_143w_binding_regression_report.json", legacy_143w_report)
    write_json(out / "static_manifest_integrity_report.json", static_report)
    write_json(out / "helper_request_audit.json", request_audit)
    write_json(out / "per_seed_gate_report.json", seed_report)
    write_json(out / "per_family_gate_report.json", family_gate)
    write_json(out / "aggregate_metrics.json", metrics)
    write_json(out / "decision.json", decision)
    write_json(out / "summary.json", summary)
    write_report(out, decision, metrics)
    append_progress(out, "decision", decision=decision["decision"], next=decision["next"])
    write_json(out / "queue.json", {"schema_version": "phase_144h_queue_v1", "milestone": MILESTONE, "status": "complete", "decision": decision["decision"], "next": decision["next"]})
    print(json.dumps({"decision": decision["decision"], "next": decision["next"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

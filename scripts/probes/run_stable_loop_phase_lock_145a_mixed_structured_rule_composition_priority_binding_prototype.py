#!/usr/bin/env python3
"""145A mixed structured-rule composition priority binding prototype."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
MILESTONE = "STABLE_LOOP_PHASE_LOCK_145A_MIXED_STRUCTURED_RULE_COMPOSITION_PRIORITY_BINDING_PROTOTYPE"
DEFAULT_OUT = Path("target/pilot_wave/stable_loop_phase_lock_145a_mixed_structured_rule_composition_priority_binding_prototype/smoke")
DEFAULT_144Z_ROOT = Path("target/pilot_wave/stable_loop_phase_lock_144z_structured_rule_metadata_binding_next_decision_plan/smoke")
HELPER_PATH = REPO_ROOT / "scripts/probes/shared_raw_generation_helper.py"
BACKEND_NAME = "repo_local_instnct_mutation_graph"
NEW_DECODER = "deterministic_pocket_gated_mixed_structured_rule_composition_binding_decoder"
OLD_SELECTED_POCKET_DECODER = "deterministic_pocket_gated_rule_selected_pocket_binding_decoder"
OLD_STRUCTURED_RULE_DECODER = "deterministic_pocket_gated_structured_rule_metadata_binding_decoder"
POSITIVE_DECISION = "mixed_structured_rule_composition_priority_binding_prototype_positive"
POSITIVE_VERDICT = "INSTNCT_MIXED_STRUCTURED_RULE_COMPOSITION_PRIORITY_BINDING_PROTOTYPE_POSITIVE"
POSITIVE_NEXT = "145H_MIXED_STRUCTURED_RULE_COMPOSITION_PRIORITY_BINDING_SCALE_CONFIRM"
NATURAL_GATE = "mixed structured rule binding authorization: open"
STRUCTURED_GATE = "structured rule binding authorization: open"
FALLBACK_VALUE = "SYM_MIXED_RULE_COMPOSITION_CLOSED"
POCKET_A_MARKER = "pocket A candidate:"
POCKET_B_MARKER = "pocket B candidate:"
POCKET_C_MARKER = "pocket C candidate:"
STALE_MARKER = "stale pocket candidate:"
VISIBLE_MARKER = "visible trap value:"
NOISE_MARKER = "noisy distractor value:"
POCKETS = ["pocket_a", "pocket_b", "pocket_c"]
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
    "145A is constrained helper/backend evidence only: mixed structured-rule composition with explicit priority over block types only, "
    "not natural-language rule reasoning, not open-ended arbitration, not GPT-like/open-domain/broad assistant capability, "
    "not production/public API/deployment/safety readiness, and not architecture superiority."
)
FAMILIES = [
    "SINGLE_VALID_BLOCK_BASELINE",
    "MULTI_BLOCK_PRIORITY_RECENCY_WINS",
    "MULTI_BLOCK_PRIORITY_QUORUM_WINS",
    "MULTI_BLOCK_PRIORITY_TIE_BREAK_WINS",
    "INVALID_HIGH_PRIORITY_FALLS_THROUGH_TO_LOWER_PRIORITY",
    "SEMANTIC_INVALID_HIGH_PRIORITY_BLOCK_FALLTHROUGH_CONTROL",
    "ALL_BLOCKS_INVALID_FALLBACK",
    "MISSING_PRIORITY_CONTROL",
    "DUPLICATE_PRIORITY_ENTRY_CONTROL",
    "UNKNOWN_PRIORITY_ENTRY_CONTROL",
    "PRIORITY_REFERENCES_MISSING_BLOCK_CONTROL",
    "MULTIPLE_PRIORITY_LINES_CONTROL",
    "DUPLICATE_RULE_BLOCK_TYPE_CONTROL",
    "MALFORMED_BLOCK_BOUNDARY_CONTROL",
    "METADATA_OUTSIDE_BLOCK_CONTROL",
    "NESTED_BLOCK_BOUNDARY_CONTROL",
    "EMPTY_RULE_BLOCK_CONTROL",
    "STRUCTURAL_INVALID_PROMPT_NO_FALLTHROUGH_CONTROL",
    "PRIORITY_POCKET_ORACLE_CONTROL",
    "SAME_BLOCKS_DIFFERENT_PRIORITY",
    "SAME_PRIORITY_DIFFERENT_BLOCK_VALUES",
    "SAME_TEMPLATE_OPPOSITE_PRIORITY_WINNER",
    "RULE_COMPOSITION_CORRUPTION_CONTROL",
    "LEGACY_STRUCTURED_RULE_METADATA_REGRESSION_CONTROL",
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
FORBIDDEN_PROMPT_RE = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in [
        r"final_selected",
        r"derived_selected",
        r"selected_pocket",
        r"\bwinner\s*=\s*pocket_[abc]\b",
        r"winner[-_ ]?value",
        r"selected[-_ ]?value",
        r"answer[-_ ]?value",
        r"target[-_ ]?value",
        r"resolved[-_ ]?output",
        r"expected[-_ ]?output",
        r"gold[-_ ]?output",
        r"ANSWER\s*=",
        r"GOLD\s*=",
        r"TARGET\s*=",
        r"EXPECTED\s*=",
        r"priority\s*=\s*pocket_[abc]",
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
    spec = importlib.util.spec_from_file_location("shared_raw_generation_helper_145a", HELPER_PATH)
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


def require_144z(root: Path) -> dict[str, Any]:
    required = ["decision.json", "target_145a_milestone_plan.json", "mixed_rule_composition_gap_analysis.json", "anti_oracle_requirements.json"]
    missing = [name for name in required if not (root / name).exists()]
    if missing:
        raise RuntimeError(f"missing 144Z artifacts: {missing}")
    decision = read_json(root / "decision.json")
    target = read_json(root / "target_145a_milestone_plan.json")
    gap = read_json(root / "mixed_rule_composition_gap_analysis.json")
    anti = read_json(root / "anti_oracle_requirements.json")
    if decision.get("decision") != "structured_rule_composition_priority_binding_prototype_plan_recommended":
        raise RuntimeError(f"bad 144Z decision: {decision.get('decision')}")
    if decision.get("selected_option") != "mixed_structured_rule_composition_priority_binding_prototype":
        raise RuntimeError(f"bad 144Z selected option: {decision.get('selected_option')}")
    if decision.get("next") != "145A_MIXED_STRUCTURED_RULE_COMPOSITION_PRIORITY_BINDING_PROTOTYPE":
        raise RuntimeError(f"bad 144Z next: {decision.get('next')}")
    if target.get("decoder_name") != NEW_DECODER or target.get("implementation_ready") is not True:
        raise RuntimeError("144Z target 145A plan is not implementation-ready for the expected decoder")
    return {"root": rel(root), "decision": decision, "target_145a_milestone_plan": target, "mixed_rule_composition_gap_analysis": gap, "anti_oracle_requirements": anti}


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
    if generated_value == row.get("stale_pocket_value"):
        return "stale"
    if generated_value == row.get("visible_wrong_value"):
        return "visible"
    if generated_value in row.get("noisy_distractor_values", []):
        return "noisy"
    if generated_value == FALLBACK_VALUE:
        return "fallback"
    return "unexpected"


def quorum_block(candidate: str, *, invalid: bool = False) -> list[str]:
    others = other_pockets(candidate)
    votes = "pocket_a,pocket_b" if invalid else f"{candidate},{others[0]},{candidate}"
    return ["rule_block=quorum", f"votes={votes}", "block_end"]


def recency_block(candidate: str, *, invalid: bool = False) -> list[str]:
    others = other_pockets(candidate)
    order = "pocket_d>pocket_a>pocket_b" if invalid else f"{candidate}>{others[0]}>{others[1]}"
    return ["rule_block=recency", f"recency_order={order}", "block_end"]


def tie_break_block(candidate: str, *, invalid: bool = False) -> list[str]:
    others = other_pockets(candidate)
    if invalid:
        return ["rule_block=tie_break", f"tied={candidate},{others[0]}", f"tie_break_order={others[1]}", "block_end"]
    return ["rule_block=tie_break", f"tied={others[0]},{candidate}", f"tie_break_order={candidate}>{others[0]}>{others[1]}", "block_end"]


def distinct_candidates(slot: int) -> dict[str, str]:
    rotated = POCKETS[slot % 3 :] + POCKETS[: slot % 3]
    return {"quorum": rotated[0], "recency": rotated[1], "tie_break": rotated[2]}


def mixed_blocks_for(row: dict[str, Any]) -> tuple[list[str], list[str], dict[str, str | None], bool, bool, str | None]:
    family = row["family"]
    slot = int(row["slot"])
    selected = row["selected_pocket_id"]
    others = other_pockets(selected)
    if family == "SINGLE_VALID_BLOCK_BASELINE":
        kind = ["quorum", "recency", "tie_break"][slot % 3]
        block = {"quorum": quorum_block, "recency": recency_block, "tie_break": tie_break_block}[kind](selected)
        return block + [f"priority={kind}"], [kind], {kind: selected}, True, False, None
    if family.startswith("MULTI_BLOCK_PRIORITY_") or family in {"SAME_PRIORITY_DIFFERENT_BLOCK_VALUES", "SAME_TEMPLATE_OPPOSITE_PRIORITY_WINNER"}:
        candidates = {"quorum": others[0], "recency": selected, "tie_break": others[1]}
        if family == "MULTI_BLOCK_PRIORITY_QUORUM_WINS":
            candidates = {"quorum": selected, "recency": others[0], "tie_break": others[1]}
        if family == "MULTI_BLOCK_PRIORITY_TIE_BREAK_WINS":
            candidates = {"quorum": others[0], "recency": others[1], "tie_break": selected}
        priority = ["recency", "quorum", "tie_break"]
        if family == "MULTI_BLOCK_PRIORITY_QUORUM_WINS":
            priority = ["quorum", "recency", "tie_break"]
        if family == "MULTI_BLOCK_PRIORITY_TIE_BREAK_WINS":
            priority = ["tie_break", "quorum", "recency"]
        lines = quorum_block(candidates["quorum"]) + recency_block(candidates["recency"]) + tie_break_block(candidates["tie_break"]) + [f"priority={'>'.join(priority)}"]
        return lines, priority, candidates, True, False, None
    if family in {"INVALID_HIGH_PRIORITY_FALLS_THROUGH_TO_LOWER_PRIORITY", "SEMANTIC_INVALID_HIGH_PRIORITY_BLOCK_FALLTHROUGH_CONTROL"}:
        candidates = {"recency": None, "quorum": selected, "tie_break": others[1]}
        lines = recency_block(others[0], invalid=True) + quorum_block(selected) + tie_break_block(others[1]) + ["priority=recency>quorum>tie_break"]
        return lines, ["recency", "quorum", "tie_break"], candidates, True, False, None
    if family == "ALL_BLOCKS_INVALID_FALLBACK":
        lines = recency_block(selected, invalid=True) + quorum_block(selected, invalid=True) + tie_break_block(selected, invalid=True) + ["priority=recency>quorum>tie_break"]
        return lines, ["recency", "quorum", "tie_break"], {"recency": None, "quorum": None, "tie_break": None}, False, True, "all_priority_referenced_blocks_invalid"
    if family == "MISSING_PRIORITY_CONTROL":
        return quorum_block(selected), [], {"quorum": selected}, False, True, "missing_priority"
    if family == "DUPLICATE_PRIORITY_ENTRY_CONTROL":
        return quorum_block(selected) + recency_block(others[0]) + ["priority=quorum>quorum"], [], {"quorum": selected, "recency": others[0]}, False, True, "duplicate_priority_entry"
    if family == "UNKNOWN_PRIORITY_ENTRY_CONTROL":
        return quorum_block(selected) + ["priority=winner>quorum"], [], {"quorum": selected}, False, True, "unknown_priority_entry"
    if family == "PRIORITY_REFERENCES_MISSING_BLOCK_CONTROL":
        return quorum_block(selected) + ["priority=recency>quorum"], [], {"quorum": selected}, False, True, "priority_references_missing_block_type"
    if family == "MULTIPLE_PRIORITY_LINES_CONTROL":
        return quorum_block(selected) + recency_block(others[0]) + ["priority=quorum>recency", "priority=recency>quorum"], [], {"quorum": selected, "recency": others[0]}, False, True, "multiple_priority_lines"
    if family == "DUPLICATE_RULE_BLOCK_TYPE_CONTROL":
        return quorum_block(selected) + quorum_block(others[0]) + ["priority=quorum"], [], {"quorum": selected}, False, True, "duplicate_rule_block_type"
    if family == "MALFORMED_BLOCK_BOUNDARY_CONTROL":
        return ["rule_block=quorum", f"votes={selected},{others[0]},{selected}", "priority=quorum"], [], {"quorum": selected}, False, True, "missing_block_end"
    if family == "METADATA_OUTSIDE_BLOCK_CONTROL":
        return quorum_block(selected) + ["votes=pocket_a,pocket_b,pocket_a", "priority=quorum"], [], {"quorum": selected}, False, True, "metadata_outside_block"
    if family in {"NESTED_BLOCK_BOUNDARY_CONTROL", "STRUCTURAL_INVALID_PROMPT_NO_FALLTHROUGH_CONTROL"}:
        return ["rule_block=recency", f"recency_order={others[0]}>{selected}>{others[1]}", "rule_block=quorum", f"votes={selected},{others[0]},{selected}", "block_end", "priority=recency>quorum"], [], {"recency": others[0], "quorum": selected}, False, True, "nested_rule_block_before_block_end"
    if family == "EMPTY_RULE_BLOCK_CONTROL":
        return ["rule_block=quorum", "block_end", "priority=quorum"], [], {"quorum": None}, False, True, "empty_rule_block"
    if family == "PRIORITY_POCKET_ORACLE_CONTROL":
        return quorum_block(selected) + ["priority=pocket_a>quorum"], [], {"quorum": selected}, False, True, "priority_pocket_oracle"
    if family == "SAME_BLOCKS_DIFFERENT_PRIORITY":
        candidates = distinct_candidates(slot)
        priority = [["quorum", "recency", "tie_break"], ["recency", "quorum", "tie_break"], ["tie_break", "quorum", "recency"]][slot % 3]
        row["selected_pocket_id"] = candidates[priority[0]]
        lines = quorum_block(candidates["quorum"]) + recency_block(candidates["recency"]) + tie_break_block(candidates["tie_break"]) + [f"priority={'>'.join(priority)}"]
        return lines, priority, candidates, True, False, None
    if family == "RULE_COMPOSITION_CORRUPTION_CONTROL":
        variants = [
            (recency_block(selected, invalid=True) + ["priority=recency"], ["recency"], {"recency": None}, False, True, "all_priority_referenced_blocks_invalid"),
            (["rule_block=quorum", "votes=pocket_a,pocket_d,pocket_a", "block_end", "priority=quorum"], ["quorum"], {"quorum": None}, False, True, "all_priority_referenced_blocks_invalid"),
            (["rule_block=unknown", "votes=pocket_a,pocket_b,pocket_a", "block_end", "priority=unknown"], [], {}, False, True, "unknown_rule_block_type"),
        ]
        return variants[slot % len(variants)]
    return quorum_block(selected) + recency_block(others[0]) + tie_break_block(others[1]) + ["priority=quorum>recency>tie_break"], ["quorum", "recency", "tie_break"], {"quorum": selected, "recency": others[0], "tie_break": others[1]}, True, False, None


def build_mixed_prompt(row: dict[str, Any], lines: list[str], *, omit_rules: bool = False) -> str:
    order = tuple(row["marker_order"])
    prompt_lines = [
        f"MIXED STRUCTURED RULE COMPOSITION PROTOTYPE {row['row_id']}",
        NATURAL_GATE,
        "Canonical rule blocks follow; apply block type priority before reading candidates.",
    ]
    if not omit_rules:
        prompt_lines.extend(lines)
    prompt_lines.extend(marker_line(pocket, row[f"{pocket}_value"]) for pocket in order)
    prompt_lines.append(f"{STALE_MARKER} {row['stale_pocket_value']}")
    prompt_lines.append(f"{VISIBLE_MARKER} {row['visible_wrong_value']}")
    for noisy in row["noisy_distractor_values"]:
        prompt_lines.append(f"{NOISE_MARKER} {noisy}")
    return "\n".join(prompt_lines)


def build_winner_prompt(row: dict[str, Any]) -> str:
    lines = [
        f"EXPLICIT WINNER BASELINE {row['row_id']}",
        STRUCTURED_GATE,
        f"winner={row['selected_pocket_id']}",
        "Use the prompt-visible pocket selector to choose exactly one pocket candidate.",
    ]
    lines.extend(marker_line(pocket, row[f"{pocket}_value"]) for pocket in tuple(row["marker_order"]))
    return "\n".join(lines)


def structured_metadata_prompt(row: dict[str, Any]) -> str:
    selected = row["selected_pocket_id"]
    others = other_pockets(selected)
    lines = [
        f"STRUCTURED RULE REGRESSION {row['row_id']}",
        STRUCTURED_GATE,
        "rule_type=recency",
        f"recency_order={selected}>{others[0]}>{others[1]}",
    ]
    lines.extend(marker_line(pocket, row[f"{pocket}_value"]) for pocket in tuple(row["marker_order"]))
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
                        "schema_version": "phase_145a_eval_row_v1",
                        "row_id": f"145A_seed{seed}_fam{family_index:02d}_group{group_index:02d}_slot{slot:02d}",
                        "seed": seed,
                        "family": family,
                        "slot": slot,
                        "scaffold_id": f"{family}_style_{group_index % 12:02d}",
                        "contrast_group_id": f"145A_{seed}_{family}_{group_index:02d}",
                        "selected_pocket_id": selected,
                        "marker_order": list(order),
                        "pocket_a_value": f"EV145A_{seed}_F{family_index}_G{group_index}_A_S{slot}",
                        "pocket_b_value": f"EV145A_{seed}_F{family_index}_G{group_index}_B_S{slot}",
                        "pocket_c_value": f"EV145A_{seed}_F{family_index}_G{group_index}_C_S{slot}",
                        "stale_pocket_value": f"EV145A_STALE_{seed}_{family_index}_{group_index}_{slot}",
                        "visible_wrong_value": f"EV145A_VISIBLE_{seed}_{family_index}_{group_index}_{slot}",
                        "noisy_distractor_values": [f"EV145A_NOISE_{seed}_{family_index}_{group_index}_{slot}_0", f"EV145A_NOISE_{seed}_{family_index}_{group_index}_{slot}_1"],
                    }
                    if family == "LEGACY_SELECTED_POCKET_BINDING_REGRESSION_CONTROL":
                        row["decoder_arm"] = "old_selected_pocket"
                        row["expected_parse_success"] = True
                        row["expected_derive_success"] = True
                        row["expected_fallback"] = False
                        row["expected_priority_order"] = []
                        row["expected_block_candidates"] = {}
                        row["expected_failure_reason"] = None
                        row["prompt"] = build_winner_prompt(row)
                    elif family == "LEGACY_STRUCTURED_RULE_METADATA_REGRESSION_CONTROL":
                        row["decoder_arm"] = "old_structured_rule_metadata"
                        row["expected_parse_success"] = True
                        row["expected_derive_success"] = True
                        row["expected_fallback"] = False
                        row["expected_priority_order"] = []
                        row["expected_block_candidates"] = {}
                        row["expected_failure_reason"] = None
                        row["prompt"] = structured_metadata_prompt(row)
                    else:
                        row["decoder_arm"] = "mixed_rule_composition"
                        lines, priority, candidates, derive_success, fallback, failure = mixed_blocks_for(row)
                        row["mixed_rule_lines"] = lines
                        row["expected_priority_order"] = priority
                        row["expected_block_candidates"] = candidates
                        row["expected_parse_success"] = family not in {
                            "MISSING_PRIORITY_CONTROL",
                            "DUPLICATE_PRIORITY_ENTRY_CONTROL",
                            "UNKNOWN_PRIORITY_ENTRY_CONTROL",
                            "PRIORITY_REFERENCES_MISSING_BLOCK_CONTROL",
                            "MULTIPLE_PRIORITY_LINES_CONTROL",
                            "DUPLICATE_RULE_BLOCK_TYPE_CONTROL",
                            "MALFORMED_BLOCK_BOUNDARY_CONTROL",
                            "METADATA_OUTSIDE_BLOCK_CONTROL",
                            "NESTED_BLOCK_BOUNDARY_CONTROL",
                            "EMPTY_RULE_BLOCK_CONTROL",
                            "STRUCTURAL_INVALID_PROMPT_NO_FALLTHROUGH_CONTROL",
                            "PRIORITY_POCKET_ORACLE_CONTROL",
                        }
                        row["expected_derive_success"] = derive_success
                        row["expected_fallback"] = fallback
                        row["expected_failure_reason"] = failure
                        row["answer_value"] = expected_value(row)
                        row["prompt"] = build_mixed_prompt(row, lines)
                    row["answer_value"] = expected_value(row)
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
        "schema_version": "instnct_mutation_graph_manifest_v19_mixed_structured_rule_composition",
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
        "claim_boundary": "constrained helper/backend mixed structured-rule composition only",
        "candidate_name": name,
    }
    path = out / "checkpoints" / f"{name}.json"
    write_json(path, manifest)
    return path, {**manifest, "checkpoint_path": rel(path), "checkpoint_sha256": sha256_file(path)}


def record_helper_request(audit_rows: list[dict[str, Any]], arm: str, row_id: str, request: dict[str, Any]) -> None:
    keys = set(request)
    audit_rows.append(
        {
            "schema_version": "phase_145a_helper_request_audit_row_v1",
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
                "schema_version": "phase_145a_generation_result_v1",
                "arm": arm,
                "row_id": row["row_id"],
                "seed": row["seed"],
                "family": row["family"],
                "generated_text": generated_text,
                "generated_value": first_value_after_answer_e(generated_text),
                "generated_text_hash": hashlib.sha256(generated_text.encode("utf-8", errors="replace")).hexdigest(),
                "value_selection_source": response.get("value_selection_source"),
                "pocket_writeback_count": response.get("pocket_writeback_count"),
                "parsed_rule_blocks": response.get("parsed_rule_blocks"),
                "parsed_priority_order": response.get("parsed_priority_order"),
                "per_block_parse_success": response.get("per_block_parse_success"),
                "per_block_derived_candidate_pocket": response.get("per_block_derived_candidate_pocket"),
                "final_selected_pocket_id": response.get("final_selected_pocket_id"),
                "binding_marker": response.get("binding_marker"),
                "extracted_value": response.get("extracted_value"),
                "generated_answer": response.get("generated_answer"),
                "failure_reason": response.get("failure_reason"),
                "parsed_rule_type": response.get("parsed_rule_type"),
                "derived_selected_pocket_id": response.get("derived_selected_pocket_id"),
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
    expected_fallback = bool(row.get("expected_fallback"))
    mixed = row["decoder_arm"] == "mixed_rule_composition"
    legacy_structured = row["decoder_arm"] == "old_structured_rule_metadata"
    final_correct = generated_value == row["answer_value"]
    fallback_correct = generated_value == FALLBACK_VALUE and result.get("value_selection_source") == "closed_pocket_fallback"
    if mixed:
        block_candidates = result.get("per_block_derived_candidate_pocket") or {}
        expected_candidates = row.get("expected_block_candidates") or {}
        expected_non_none = {k: v for k, v in expected_candidates.items() if v is not None}
        candidate_match = fallback_correct if expected_fallback and not row["expected_parse_success"] else all(block_candidates.get(k) == v for k, v in expected_candidates.items())
        priority_match = result.get("parsed_priority_order") == row.get("expected_priority_order")
        parse_correct = fallback_correct if not row["expected_parse_success"] else bool(result.get("parsed_rule_blocks"))
        derive_correct = result.get("final_selected_pocket_id") == row["selected_pocket_id"] if row["expected_derive_success"] else fallback_correct
        binding_correct = (
            result.get("binding_marker") == STATIC_MARKER_MAP[row["selected_pocket_id"]]
            and result.get("value_selection_source") == "mixed_rule_composition_writeback"
            if row["expected_derive_success"]
            else fallback_correct
        )
        extraction_correct = result.get("extracted_value") == row["answer_value"] if row["expected_derive_success"] else fallback_correct
        final_answer_correct = final_correct if not expected_fallback else fallback_correct
    elif legacy_structured:
        block_candidates = {}
        expected_non_none = {}
        candidate_match = True
        priority_match = True
        parse_correct = result.get("value_selection_source") == "structured_rule_metadata_writeback"
        derive_correct = result.get("derived_selected_pocket_id") == row["selected_pocket_id"]
        binding_correct = result.get("binding_marker") == STATIC_MARKER_MAP[row["selected_pocket_id"]]
        extraction_correct = result.get("extracted_value") == row["answer_value"]
        final_answer_correct = final_correct
    else:
        block_candidates = {}
        expected_non_none = {}
        candidate_match = True
        priority_match = True
        parse_correct = True
        derive_correct = True
        binding_correct = result.get("value_selection_source") == "rule_selected_pocket_writeback"
        extraction_correct = final_correct
        final_answer_correct = final_correct
    return {
        "schema_version": "phase_145a_scoring_result_v1",
        "arm": result["arm"],
        "row_id": row["row_id"],
        "seed": row["seed"],
        "family": row["family"],
        "decoder_arm": row["decoder_arm"],
        "expected_selected_pocket_id": row["selected_pocket_id"],
        "expected_final_value": row["answer_value"],
        "expected_fallback": expected_fallback,
        "expected_parse_success": row["expected_parse_success"],
        "expected_derive_success": row["expected_derive_success"],
        "expected_priority_order": row.get("expected_priority_order"),
        "expected_block_candidates": row.get("expected_block_candidates"),
        "generated_value": generated_value,
        "generated_source": value_source(row, generated_value),
        "value_selection_source": result.get("value_selection_source"),
        "parsed_rule_blocks": result.get("parsed_rule_blocks"),
        "parsed_priority_order": result.get("parsed_priority_order"),
        "per_block_parse_success": result.get("per_block_parse_success"),
        "per_block_derived_candidate_pocket": block_candidates,
        "final_selected_pocket_id": result.get("final_selected_pocket_id"),
        "binding_marker": result.get("binding_marker"),
        "extracted_value": result.get("extracted_value"),
        "failure_reason": result.get("failure_reason"),
        "mixed_rule_block_parse_correct": parse_correct,
        "per_block_candidate_derivation_correct": candidate_match,
        "priority_policy_parse_correct": priority_match if row["expected_parse_success"] else fallback_correct,
        "final_selected_pocket_derivation_correct": derive_correct,
        "selected_pocket_to_marker_binding_correct": binding_correct,
        "same_line_value_extraction_correct": extraction_correct,
        "final_answer_correct": final_answer_correct,
        "fallback_correct": fallback_correct,
        "distinct_block_candidate_row": len(set(expected_non_none.values())) == len(expected_non_none) and len(expected_non_none) >= 3,
        "priority_only_changes_winner_correct": final_answer_correct,
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
        "schema_version": "phase_145a_helper_request_audit_v1",
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
        "schema_version": "phase_145a_static_manifest_integrity_report_v1",
        "manifest_count": len(manifests),
        "static_marker_map": STATIC_MARKER_MAP,
        "all_static_marker_maps_match": all(item == STATIC_MARKER_MAP for item in marker_maps),
        "payload_marker_list_contains_all_pockets": all(set(item) == set(ABC_MARKERS) for item in payload_marker_sets),
        "payload_marker_list_narrowed_to_correct_pocket": False,
        "per_row_manifest_switch_rate": 0.0,
        "per_row_payload_marker_switch_rate": 0.0,
        "passed": all(item == STATIC_MARKER_MAP for item in marker_maps) and all(set(item) == set(ABC_MARKERS) for item in payload_marker_sets),
    }


def shared_helper_diff_audit() -> dict[str, Any]:
    before = git_show_head("scripts/probes/shared_raw_generation_helper.py")
    after = HELPER_PATH.read_text(encoding="utf-8")
    before_validate = extract_function(before, "validate_request")
    after_validate = extract_function(after, "validate_request")
    before_old_selected = extract_function(before, "_instnct_select_rule_selected_pocket_value")
    after_old_selected = extract_function(after, "_instnct_select_rule_selected_pocket_value")
    before_old_structured = extract_function(before, "_instnct_select_structured_rule_metadata_value")
    after_old_structured = extract_function(after, "_instnct_select_structured_rule_metadata_value")
    before_imports = sorted(line for line in before.splitlines() if line.startswith("import ") or line.startswith("from "))
    after_imports = sorted(line for line in after.splitlines() if line.startswith("import ") or line.startswith("from "))
    audit = {
        "schema_version": "phase_145a_shared_helper_diff_audit_v1",
        "helper_source_sha256_before": sha256_text(before),
        "helper_source_sha256_after": sha256_text(after),
        "source_changed": before != after,
        "new_mixed_decoder_string_present": NEW_DECODER in after,
        "new_mixed_selection_function_present": "_instnct_select_mixed_structured_rule_composition_value" in after,
        "new_mixed_parser_helpers_present": "_instnct_parse_mixed_rule_composition" in after and "_instnct_parse_mixed_priority" in after,
        "new_behavior_manifest_gated": f'decoder.get("type") == MIXED_STRUCTURED_RULE_COMPOSITION_BINDING_DECODER' in after,
        "old_selected_pocket_decoder_still_present": OLD_SELECTED_POCKET_DECODER in after,
        "old_structured_metadata_decoder_still_present": OLD_STRUCTURED_RULE_DECODER in after,
        "old_selected_pocket_binding_function_unchanged": before_old_selected == after_old_selected and bool(after_old_selected),
        "old_structured_metadata_function_unchanged": before_old_structured == after_old_structured and bool(after_old_structured),
        "validate_request_unchanged": before_validate == after_validate and bool(after_validate),
        "allowed_request_keys_unchanged": '"prompt"' in after and '"generation_config"' in after and "ALLOWED_REQUEST_KEYS" in after and "selected_pocket_id" not in extract_function(after, "validate_request"),
        "forbidden_request_keys_not_loosened": "FORBIDDEN_REQUEST_KEYS" in after and "oracle_data" in after,
        "no_training_import_added": not any("torch.optim" in line for line in after_imports if line not in before_imports),
        "no_network_or_io_added": not any(token in after for token in ["import socket", "import requests", "urllib.request", "http.client"]),
        "passed": False,
    }
    audit["passed"] = all(value is True for key, value in audit.items() if key not in {"schema_version", "helper_source_sha256_before", "helper_source_sha256_after", "passed"})
    return audit


def prompt_scanner_report(rows: list[dict[str, Any]], scored: list[dict[str, Any]]) -> dict[str, Any]:
    violations: list[dict[str, Any]] = []
    allowed_priority_oracle = {row["row_id"] for row in rows if row["family"] == "PRIORITY_POCKET_ORACLE_CONTROL"}
    fallback_by_id = {row["row_id"]: row.get("fallback_correct") is True for row in scored}
    for row in rows:
        if row["decoder_arm"] != "mixed_rule_composition":
            continue
        for pattern in FORBIDDEN_PROMPT_RE:
            if pattern.search(row["prompt"]):
                expected = row["row_id"] in allowed_priority_oracle and "priority" in pattern.pattern and fallback_by_id.get(row["row_id"]) is True
                if not expected:
                    violations.append({"row_id": row["row_id"], "family": row["family"], "pattern": pattern.pattern})
    return {
        "schema_version": "phase_145a_prompt_scanner_report_v1",
        "mixed_rule_rows_scanned": sum(1 for row in rows if row["decoder_arm"] == "mixed_rule_composition"),
        "priority_pocket_oracle_control_rows_detected": len(allowed_priority_oracle),
        "priority_pocket_oracle_control_rows_rejected": sum(1 for row_id in allowed_priority_oracle if fallback_by_id.get(row_id) is True),
        "forbidden_violation_count": len(violations),
        "violations": violations[:20],
        "passed": not violations and all(fallback_by_id.get(row_id) is True for row_id in allowed_priority_oracle),
    }


def family_report(scored: list[dict[str, Any]], family: str, metric_name: str, metric_key: str | None = None, *, fallback_metric: bool = False) -> dict[str, Any]:
    subset = scoped(scored, family)
    value = fallback_rate(subset) if fallback_metric else fraction(subset, metric_key or metric_name)
    return {"schema_version": f"phase_145a_{metric_name}_report_v1", "family": family, "row_count": len(subset), metric_name: value, "fallback_rate": fallback_rate(subset), "passed": value >= 0.90}


def aggregate_metrics(main_scored: list[dict[str, Any]], ablation_scored: list[dict[str, Any]], legacy_structured: dict[str, Any], legacy_selected: dict[str, Any], request_audit: dict[str, Any], static_report: dict[str, Any], deterministic: bool) -> dict[str, Any]:
    mixed = [row for row in main_scored if row["decoder_arm"] == "mixed_rule_composition"]
    positive = [row for row in mixed if row["expected_derive_success"] is True]
    same_blocks = scoped(main_scored, "SAME_BLOCKS_DIFFERENT_PRIORITY")
    return {
        "schema_version": "phase_145a_aggregate_metrics_v1",
        "main_eval_rows": len(main_scored),
        "mixed_rule_block_parse_accuracy": fraction(mixed, "mixed_rule_block_parse_correct"),
        "per_block_candidate_derivation_accuracy": fraction(mixed, "per_block_candidate_derivation_correct"),
        "priority_policy_parse_accuracy": fraction(mixed, "priority_policy_parse_correct"),
        "final_selected_pocket_derivation_accuracy": fraction(mixed, "final_selected_pocket_derivation_correct"),
        "selected_pocket_to_marker_binding_accuracy": fraction(positive, "selected_pocket_to_marker_binding_correct"),
        "same_line_value_extraction_accuracy": fraction(positive, "same_line_value_extraction_correct"),
        "end_to_end_answer_accuracy": fraction(positive, "final_answer_correct"),
        "invalid_high_priority_fallthrough_accuracy": fraction(scoped(main_scored, "INVALID_HIGH_PRIORITY_FALLS_THROUGH_TO_LOWER_PRIORITY"), "final_selected_pocket_derivation_correct"),
        "semantic_invalid_high_priority_fallthrough_accuracy": fraction(scoped(main_scored, "SEMANTIC_INVALID_HIGH_PRIORITY_BLOCK_FALLTHROUGH_CONTROL"), "final_selected_pocket_derivation_correct"),
        "all_blocks_invalid_fallback_rate": fallback_rate(scoped(main_scored, "ALL_BLOCKS_INVALID_FALLBACK")),
        "missing_priority_fallback_rate": fallback_rate(scoped(main_scored, "MISSING_PRIORITY_CONTROL")),
        "duplicate_priority_rejection_rate": fallback_rate(scoped(main_scored, "DUPLICATE_PRIORITY_ENTRY_CONTROL")),
        "unknown_priority_rejection_rate": fallback_rate(scoped(main_scored, "UNKNOWN_PRIORITY_ENTRY_CONTROL")),
        "missing_block_reference_rejection_rate": fallback_rate(scoped(main_scored, "PRIORITY_REFERENCES_MISSING_BLOCK_CONTROL")),
        "multiple_priority_lines_rejection_rate": fallback_rate(scoped(main_scored, "MULTIPLE_PRIORITY_LINES_CONTROL")),
        "duplicate_rule_block_type_rejection_rate": fallback_rate(scoped(main_scored, "DUPLICATE_RULE_BLOCK_TYPE_CONTROL")),
        "malformed_block_boundary_rejection_rate": fallback_rate(scoped(main_scored, "MALFORMED_BLOCK_BOUNDARY_CONTROL")),
        "metadata_outside_block_rejection_rate": fallback_rate(scoped(main_scored, "METADATA_OUTSIDE_BLOCK_CONTROL")),
        "nested_block_boundary_rejection_rate": fallback_rate(scoped(main_scored, "NESTED_BLOCK_BOUNDARY_CONTROL")),
        "empty_rule_block_rejection_rate": fallback_rate(scoped(main_scored, "EMPTY_RULE_BLOCK_CONTROL")),
        "structural_invalid_prompt_fallback_rate": fallback_rate(scoped(main_scored, "STRUCTURAL_INVALID_PROMPT_NO_FALLTHROUGH_CONTROL")),
        "priority_pocket_oracle_rejection_rate": fallback_rate(scoped(main_scored, "PRIORITY_POCKET_ORACLE_CONTROL")),
        "same_blocks_different_priority_accuracy": fraction(same_blocks, "final_selected_pocket_derivation_correct"),
        "priority_only_changes_winner_accuracy": fraction(same_blocks, "priority_only_changes_winner_correct"),
        "distinct_block_candidate_coverage": all(row.get("distinct_block_candidate_row") is True for row in same_blocks),
        "same_priority_different_block_values_accuracy": fraction(scoped(main_scored, "SAME_PRIORITY_DIFFERENT_BLOCK_VALUES"), "final_answer_correct"),
        "same_template_opposite_priority_winner_accuracy": fraction(scoped(main_scored, "SAME_TEMPLATE_OPPOSITE_PRIORITY_WINNER"), "final_answer_correct"),
        "rule_composition_ablation_accuracy": rate(sum(1 for row in ablation_scored if row.get("generated_value") == row.get("expected_final_value")), len(ablation_scored)),
        "rule_composition_ablation_fallback_rate": fallback_rate(ablation_scored),
        "helper_request_forbidden_metadata_count": request_audit["helper_request_forbidden_metadata_count"],
        "per_row_manifest_switch_rate": static_report["per_row_manifest_switch_rate"],
        "per_row_payload_marker_switch_rate": static_report["per_row_payload_marker_switch_rate"],
        "legacy_structured_rule_metadata_regression_passed": legacy_structured["legacy_structured_rule_metadata_regression_passed"],
        "legacy_selected_pocket_binding_regression_passed": legacy_selected["legacy_selected_pocket_binding_regression_passed"],
        "deterministic_replay_passed": deterministic,
    }


def choose_decision(metrics: dict[str, Any], helper_diff: dict[str, Any], request_audit: dict[str, Any], prompt_report: dict[str, Any], static_report: dict[str, Any]) -> dict[str, Any]:
    integrity = helper_diff["passed"] and request_audit["passed"] and prompt_report["passed"] and static_report["passed"]
    gates = (
        metrics["mixed_rule_block_parse_accuracy"] >= 0.90
        and metrics["per_block_candidate_derivation_accuracy"] >= 0.90
        and metrics["priority_policy_parse_accuracy"] >= 0.90
        and metrics["final_selected_pocket_derivation_accuracy"] >= 0.90
        and metrics["selected_pocket_to_marker_binding_accuracy"] >= 0.95
        and metrics["same_line_value_extraction_accuracy"] >= 0.95
        and metrics["end_to_end_answer_accuracy"] >= 0.90
        and metrics["invalid_high_priority_fallthrough_accuracy"] >= 0.90
        and metrics["semantic_invalid_high_priority_fallthrough_accuracy"] >= 0.90
        and metrics["all_blocks_invalid_fallback_rate"] >= 0.90
        and metrics["missing_priority_fallback_rate"] >= 0.90
        and metrics["duplicate_priority_rejection_rate"] >= 0.90
        and metrics["unknown_priority_rejection_rate"] >= 0.90
        and metrics["missing_block_reference_rejection_rate"] >= 0.90
        and metrics["multiple_priority_lines_rejection_rate"] >= 0.90
        and metrics["duplicate_rule_block_type_rejection_rate"] >= 0.90
        and metrics["malformed_block_boundary_rejection_rate"] >= 0.90
        and metrics["metadata_outside_block_rejection_rate"] >= 0.90
        and metrics["nested_block_boundary_rejection_rate"] >= 0.90
        and metrics["empty_rule_block_rejection_rate"] >= 0.90
        and metrics["structural_invalid_prompt_fallback_rate"] >= 0.90
        and metrics["priority_pocket_oracle_rejection_rate"] >= 0.90
        and metrics["same_blocks_different_priority_accuracy"] >= 0.90
        and metrics["priority_only_changes_winner_accuracy"] >= 0.90
        and metrics["distinct_block_candidate_coverage"] is True
        and metrics["same_priority_different_block_values_accuracy"] >= 0.90
        and metrics["same_template_opposite_priority_winner_accuracy"] >= 0.90
        and metrics["rule_composition_ablation_accuracy"] <= 0.15
        and metrics["helper_request_forbidden_metadata_count"] == 0
        and metrics["per_row_manifest_switch_rate"] == 0.0
        and metrics["per_row_payload_marker_switch_rate"] == 0.0
        and metrics["legacy_structured_rule_metadata_regression_passed"] is True
        and metrics["legacy_selected_pocket_binding_regression_passed"] is True
        and metrics["deterministic_replay_passed"] is True
    )
    positive = integrity and gates
    if not integrity:
        decision = "helper_integrity_failure"; next_step = "135G_RAW_HELPER_ORACLE_SHORTCUT_REMOVAL"
    elif metrics["mixed_rule_block_parse_accuracy"] < 0.90:
        decision = "mixed_rule_block_parse_failure"; next_step = "145B_MIXED_RULE_BLOCK_PARSE_FAILURE_ANALYSIS"
    elif metrics["priority_policy_parse_accuracy"] < 0.90:
        decision = "priority_policy_parse_failure"; next_step = "145C_PRIORITY_POLICY_PARSE_FAILURE_ANALYSIS"
    elif metrics["final_selected_pocket_derivation_accuracy"] < 0.90:
        decision = "final_selected_pocket_derivation_failure"; next_step = "145D_MIXED_RULE_FINAL_SELECTION_FAILURE_ANALYSIS"
    elif metrics["rule_composition_ablation_accuracy"] > 0.15 or metrics["priority_pocket_oracle_rejection_rate"] < 0.90:
        decision = "rule_composition_oracle_shortcut_detected"; next_step = "145E_RULE_COMPOSITION_ORACLE_SHORTCUT_ANALYSIS"
    elif metrics["duplicate_priority_rejection_rate"] < 0.90 or metrics["unknown_priority_rejection_rate"] < 0.90:
        decision = "priority_ambiguity_not_rejected"; next_step = "145F_PRIORITY_AMBIGUITY_ANALYSIS"
    elif metrics["invalid_high_priority_fallthrough_accuracy"] < 0.90 or metrics["semantic_invalid_high_priority_fallthrough_accuracy"] < 0.90:
        decision = "invalid_block_fallthrough_failure"; next_step = "145G_INVALID_BLOCK_FALLTHROUGH_ANALYSIS"
    elif metrics["legacy_structured_rule_metadata_regression_passed"] is not True:
        decision = "legacy_structured_rule_metadata_regression"; next_step = "144C_STRUCTURED_RULE_METADATA_PARSE_FAILURE_ANALYSIS"
    elif metrics["selected_pocket_to_marker_binding_accuracy"] < 0.95 or metrics["legacy_selected_pocket_binding_regression_passed"] is not True:
        decision = "selected_pocket_binding_regression"; next_step = "143L_WINNER_LABEL_BINDING_FAILURE_ANALYSIS"
    elif positive:
        decision = POSITIVE_DECISION; next_step = POSITIVE_NEXT
    else:
        decision = "helper_integrity_failure"; next_step = "135G_RAW_HELPER_ORACLE_SHORTCUT_REMOVAL"
    return {
        "schema_version": "phase_145a_decision_v1",
        "decision": decision,
        "verdict": POSITIVE_VERDICT if decision == POSITIVE_DECISION else "INSTNCT_MIXED_STRUCTURED_RULE_COMPOSITION_PRIORITY_BINDING_PROTOTYPE_BLOCKED",
        "next": next_step,
        "positive_gate_passed": positive,
        "boundary": BOUNDARY_TEXT,
        **FALSE_FLAGS,
    }


def write_report(out: Path, decision: dict[str, Any], metrics: dict[str, Any]) -> None:
    text = f"""# {MILESTONE}

Decision: `{decision['decision']}`

Verdict: `{decision['verdict']}`

Next: `{decision['next']}`

Boundary: {BOUNDARY_TEXT}

## Key Metrics

- mixed rule block parse accuracy: `{metrics['mixed_rule_block_parse_accuracy']}`
- per-block candidate derivation accuracy: `{metrics['per_block_candidate_derivation_accuracy']}`
- priority policy parse accuracy: `{metrics['priority_policy_parse_accuracy']}`
- final selected pocket derivation accuracy: `{metrics['final_selected_pocket_derivation_accuracy']}`
- selected pocket to marker binding accuracy: `{metrics['selected_pocket_to_marker_binding_accuracy']}`
- same-line value extraction accuracy: `{metrics['same_line_value_extraction_accuracy']}`
- end-to-end answer accuracy: `{metrics['end_to_end_answer_accuracy']}`
- semantic invalid high-priority fallthrough accuracy: `{metrics['semantic_invalid_high_priority_fallthrough_accuracy']}`
- structural invalid prompt fallback rate: `{metrics['structural_invalid_prompt_fallback_rate']}`
- priority pocket oracle rejection rate: `{metrics['priority_pocket_oracle_rejection_rate']}`
"""
    write_text(out / "report.md", text)


def write_metric_report(out: Path, name: str, metric_name: str, value: Any, passed: bool, extra: dict[str, Any] | None = None) -> None:
    write_json(out / name, {"schema_version": f"phase_145a_{name.replace('.json', '')}_v1", metric_name: value, "passed": passed, **(extra or {})})


def main() -> int:
    parser = argparse.ArgumentParser(description="Run 145A mixed structured-rule composition prototype")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--upstream-144z-root", type=Path, default=DEFAULT_144Z_ROOT)
    parser.add_argument("--seeds", default="5201,5202,5203")
    parser.add_argument("--groups-per-family", type=int, default=12)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--heartbeat-sec", type=int, default=20)
    args = parser.parse_args()

    out = resolve_target_out(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_text(out / "progress.jsonl", "")
    append_progress(out, "startup", milestone=MILESTONE, heartbeat_sec=args.heartbeat_sec)
    write_json(out / "queue.json", {"schema_version": "phase_145a_queue_v1", "milestone": MILESTONE, "status": "running"})

    upstream = require_144z(resolve_repo_path(args.upstream_144z_root))
    write_json(out / "upstream_144z_manifest.json", upstream)
    append_progress(out, "upstream verified", upstream_decision=upstream["decision"]["decision"])
    helper = load_helper()
    seeds = [int(item.strip()) for item in args.seeds.split(",") if item.strip()]
    rows = build_rows(seeds, args.groups_per_family, args.group_size)
    append_progress(out, "rows built", row_count=len(rows), families=len(FAMILIES))
    write_json(out / "analysis_config.json", {"schema_version": "phase_145a_analysis_config_v1", "milestone": MILESTONE, "boundary": BOUNDARY_TEXT, "seeds": seeds, "families": FAMILIES, "groups_per_family": args.groups_per_family, "group_size": args.group_size, "decoder": NEW_DECODER, **FALSE_FLAGS})

    mixed_path, mixed_manifest = build_manifest(out, "mixed_rule_composition", decoder_type=NEW_DECODER)
    structured_path, structured_manifest = build_manifest(out, "legacy_structured_rule", decoder_type=OLD_STRUCTURED_RULE_DECODER, gate_marker=STRUCTURED_GATE)
    selected_path, selected_manifest = build_manifest(out, "legacy_selected_pocket", decoder_type=OLD_SELECTED_POCKET_DECODER, gate_marker=STRUCTURED_GATE)
    request_audit_rows: list[dict[str, Any]] = []
    mixed_rows = [row for row in rows if row["decoder_arm"] == "mixed_rule_composition"]
    structured_rows = [row for row in rows if row["decoder_arm"] == "old_structured_rule_metadata"]
    selected_rows = [row for row in rows if row["decoder_arm"] == "old_selected_pocket"]

    mixed_results = run_arm(helper, out, "mixed_rule_composition_main", mixed_rows, mixed_path, mixed_manifest["checkpoint_sha256"], args.max_new_tokens, args.heartbeat_sec, request_audit_rows)
    structured_results = run_arm(helper, out, "legacy_structured_rule_metadata", structured_rows, structured_path, structured_manifest["checkpoint_sha256"], args.max_new_tokens, args.heartbeat_sec, request_audit_rows)
    selected_results = run_arm(helper, out, "legacy_selected_pocket_binding", selected_rows, selected_path, selected_manifest["checkpoint_sha256"], args.max_new_tokens, args.heartbeat_sec, request_audit_rows)
    main_results = mixed_results + structured_results + selected_results
    main_scored = score(mixed_rows, mixed_results) + score(structured_rows, structured_results) + score(selected_rows, selected_results)
    replay_results = run_arm(helper, out, "mixed_rule_composition_replay", mixed_rows, mixed_path, mixed_manifest["checkpoint_sha256"], args.max_new_tokens, args.heartbeat_sec, request_audit_rows)
    deterministic = [row["generated_text_hash"] for row in mixed_results] == [row["generated_text_hash"] for row in replay_results]

    ablation_rows = []
    for row in mixed_rows:
        if row["expected_derive_success"]:
            ablated = dict(row)
            ablated["prompt"] = build_mixed_prompt(row, row["mixed_rule_lines"], omit_rules=True)
            ablated["expected_fallback"] = True
            ablated["expected_parse_success"] = False
            ablated["expected_derive_success"] = False
            ablated["expected_failure_reason"] = "missing_rule_block"
            ablation_rows.append(ablated)
    ablation_results = run_arm(helper, out, "rule_composition_ablation", ablation_rows, mixed_path, mixed_manifest["checkpoint_sha256"], args.max_new_tokens, args.heartbeat_sec, request_audit_rows)
    ablation_scored = score(ablation_rows, ablation_results)
    append_progress(out, "generation complete", deterministic=deterministic)

    helper_diff = shared_helper_diff_audit()
    request_audit = helper_request_audit(request_audit_rows)
    static_report = static_manifest_integrity_report([mixed_manifest, structured_manifest, selected_manifest])
    legacy_structured = {
        "schema_version": "phase_145a_legacy_structured_rule_metadata_regression_report_v1",
        "legacy_structured_rule_metadata_accuracy": fraction(scoped(main_scored, "LEGACY_STRUCTURED_RULE_METADATA_REGRESSION_CONTROL"), "final_answer_correct"),
        "legacy_structured_rule_metadata_regression_passed": fraction(scoped(main_scored, "LEGACY_STRUCTURED_RULE_METADATA_REGRESSION_CONTROL"), "final_answer_correct") >= 0.95,
        "passed": fraction(scoped(main_scored, "LEGACY_STRUCTURED_RULE_METADATA_REGRESSION_CONTROL"), "final_answer_correct") >= 0.95,
    }
    legacy_selected = {
        "schema_version": "phase_145a_legacy_selected_pocket_binding_regression_report_v1",
        "legacy_selected_pocket_binding_accuracy": fraction(scoped(main_scored, "LEGACY_SELECTED_POCKET_BINDING_REGRESSION_CONTROL"), "final_answer_correct"),
        "legacy_selected_pocket_binding_regression_passed": fraction(scoped(main_scored, "LEGACY_SELECTED_POCKET_BINDING_REGRESSION_CONTROL"), "final_answer_correct") >= 0.95,
        "passed": fraction(scoped(main_scored, "LEGACY_SELECTED_POCKET_BINDING_REGRESSION_CONTROL"), "final_answer_correct") >= 0.95,
    }
    metrics = aggregate_metrics(main_scored, ablation_scored, legacy_structured, legacy_selected, request_audit, static_report, deterministic)
    prompt_report = prompt_scanner_report(rows, main_scored)
    decision = choose_decision(metrics, helper_diff, request_audit, prompt_report, static_report)
    summary = {"schema_version": "phase_145a_summary_v1", "milestone": MILESTONE, "status": "complete", "boundary": BOUNDARY_TEXT, "decision": decision, "aggregate_metrics": metrics, **FALSE_FLAGS}

    write_jsonl(out / "main_results.jsonl", main_results)
    write_jsonl(out / "main_scoring.jsonl", main_scored)
    write_json(out / "shared_helper_diff_audit.json", helper_diff)
    write_json(out / "prompt_scanner_report.json", prompt_report)
    write_json(out / "helper_request_audit.json", request_audit)
    write_json(out / "static_manifest_integrity_report.json", static_report)
    write_json(out / "legacy_structured_rule_metadata_regression_report.json", legacy_structured)
    write_json(out / "legacy_selected_pocket_binding_regression_report.json", legacy_selected)
    write_metric_report(out, "mixed_rule_block_parser_report.json", "mixed_rule_block_parse_accuracy", metrics["mixed_rule_block_parse_accuracy"], metrics["mixed_rule_block_parse_accuracy"] >= 0.90)
    write_metric_report(out, "per_block_candidate_derivation_report.json", "per_block_candidate_derivation_accuracy", metrics["per_block_candidate_derivation_accuracy"], metrics["per_block_candidate_derivation_accuracy"] >= 0.90)
    write_metric_report(out, "priority_policy_report.json", "priority_policy_parse_accuracy", metrics["priority_policy_parse_accuracy"], metrics["priority_policy_parse_accuracy"] >= 0.90)
    write_metric_report(out, "final_selected_pocket_derivation_report.json", "final_selected_pocket_derivation_accuracy", metrics["final_selected_pocket_derivation_accuracy"], metrics["final_selected_pocket_derivation_accuracy"] >= 0.90)
    write_metric_report(out, "selected_pocket_binding_report.json", "selected_pocket_to_marker_binding_accuracy", metrics["selected_pocket_to_marker_binding_accuracy"], metrics["selected_pocket_to_marker_binding_accuracy"] >= 0.95)
    write_metric_report(out, "same_line_value_extraction_report.json", "same_line_value_extraction_accuracy", metrics["same_line_value_extraction_accuracy"], metrics["same_line_value_extraction_accuracy"] >= 0.95)
    family_reports = [
        ("invalid_high_priority_fallthrough_report.json", "INVALID_HIGH_PRIORITY_FALLS_THROUGH_TO_LOWER_PRIORITY", "invalid_high_priority_fallthrough_accuracy", "final_selected_pocket_derivation_correct", False),
        ("all_blocks_invalid_fallback_report.json", "ALL_BLOCKS_INVALID_FALLBACK", "all_blocks_invalid_fallback_rate", None, True),
        ("missing_priority_report.json", "MISSING_PRIORITY_CONTROL", "missing_priority_fallback_rate", None, True),
        ("duplicate_priority_entry_report.json", "DUPLICATE_PRIORITY_ENTRY_CONTROL", "duplicate_priority_rejection_rate", None, True),
        ("unknown_priority_entry_report.json", "UNKNOWN_PRIORITY_ENTRY_CONTROL", "unknown_priority_rejection_rate", None, True),
        ("missing_block_reference_report.json", "PRIORITY_REFERENCES_MISSING_BLOCK_CONTROL", "missing_block_reference_rejection_rate", None, True),
        ("multiple_priority_lines_report.json", "MULTIPLE_PRIORITY_LINES_CONTROL", "multiple_priority_lines_rejection_rate", None, True),
        ("duplicate_rule_block_type_report.json", "DUPLICATE_RULE_BLOCK_TYPE_CONTROL", "duplicate_rule_block_type_rejection_rate", None, True),
        ("malformed_block_boundary_report.json", "MALFORMED_BLOCK_BOUNDARY_CONTROL", "malformed_block_boundary_rejection_rate", None, True),
        ("metadata_outside_block_report.json", "METADATA_OUTSIDE_BLOCK_CONTROL", "metadata_outside_block_rejection_rate", None, True),
        ("nested_block_boundary_report.json", "NESTED_BLOCK_BOUNDARY_CONTROL", "nested_block_boundary_rejection_rate", None, True),
        ("empty_rule_block_report.json", "EMPTY_RULE_BLOCK_CONTROL", "empty_rule_block_rejection_rate", None, True),
        ("priority_pocket_oracle_report.json", "PRIORITY_POCKET_ORACLE_CONTROL", "priority_pocket_oracle_rejection_rate", None, True),
        ("same_blocks_different_priority_report.json", "SAME_BLOCKS_DIFFERENT_PRIORITY", "same_blocks_different_priority_accuracy", "final_selected_pocket_derivation_correct", False),
        ("same_priority_different_block_values_report.json", "SAME_PRIORITY_DIFFERENT_BLOCK_VALUES", "same_priority_different_block_values_accuracy", "final_answer_correct", False),
        ("same_template_opposite_priority_winner_report.json", "SAME_TEMPLATE_OPPOSITE_PRIORITY_WINNER", "same_template_opposite_priority_winner_accuracy", "final_answer_correct", False),
    ]
    for filename, family, metric_name, key, is_fallback in family_reports:
        write_json(out / filename, family_report(main_scored, family, metric_name, key, fallback_metric=is_fallback))
    write_metric_report(out, "semantic_invalid_high_priority_fallthrough_report.json", "semantic_invalid_high_priority_fallthrough_accuracy", metrics["semantic_invalid_high_priority_fallthrough_accuracy"], metrics["semantic_invalid_high_priority_fallthrough_accuracy"] >= 0.90)
    write_metric_report(out, "structural_invalid_prompt_fallback_report.json", "structural_invalid_prompt_fallback_rate", metrics["structural_invalid_prompt_fallback_rate"], metrics["structural_invalid_prompt_fallback_rate"] >= 0.90)
    write_json(out / "rule_composition_ablation_report.json", {"schema_version": "phase_145a_rule_composition_ablation_report_v1", "row_count": len(ablation_scored), "rule_composition_ablation_accuracy": metrics["rule_composition_ablation_accuracy"], "rule_composition_ablation_fallback_rate": metrics["rule_composition_ablation_fallback_rate"], "passed": metrics["rule_composition_ablation_accuracy"] <= 0.15})
    write_json(out / "rule_composition_oracle_shortcut_report.json", {"schema_version": "phase_145a_rule_composition_oracle_shortcut_report_v1", "priority_pocket_oracle_rejection_rate": metrics["priority_pocket_oracle_rejection_rate"], "rule_composition_ablation_accuracy": metrics["rule_composition_ablation_accuracy"], "passed": metrics["priority_pocket_oracle_rejection_rate"] >= 0.90 and metrics["rule_composition_ablation_accuracy"] <= 0.15})
    write_json(out / "aggregate_metrics.json", metrics)
    write_json(out / "decision.json", decision)
    write_json(out / "summary.json", summary)
    write_report(out, decision, metrics)
    append_progress(out, "decision", decision=decision["decision"], next=decision["next"])
    write_json(out / "queue.json", {"schema_version": "phase_145a_queue_v1", "milestone": MILESTONE, "status": "complete", "decision": decision["decision"], "next": decision["next"]})
    print(json.dumps({"decision": decision["decision"], "next": decision["next"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

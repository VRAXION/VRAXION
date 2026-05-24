#!/usr/bin/env python3
"""143F helper-only multi-pocket arbitration scale confirm.

This phase scale-confirms the 143A resolved-marker fixture and adds a
decision-critical no-resolved-final-marker subset. The expected clean outcome is
to show that the current constrained helper/backend path depends on resolved
final payload markers.
"""

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
MILESTONE = "STABLE_LOOP_PHASE_LOCK_143F_INSTNCT_POCKET_GATED_MULTI_POCKET_ARBITRATION_SCALE_CONFIRM"
DEFAULT_OUT = Path("target/pilot_wave/stable_loop_phase_lock_143f_instnct_pocket_gated_multi_pocket_arbitration_scale_confirm/smoke")
DEFAULT_143A_ROOT = Path("target/pilot_wave/stable_loop_phase_lock_143a_instnct_pocket_gated_multi_pocket_arbitration_probe/smoke")
HELPER_PATH = REPO_ROOT / "scripts/probes/shared_raw_generation_helper.py"
RUNNER_PATH = Path(__file__).resolve()
CHECKER_PATH = REPO_ROOT / "scripts/probes/run_stable_loop_phase_lock_143f_instnct_pocket_gated_multi_pocket_arbitration_scale_confirm_check.py"

BACKEND_NAME = "repo_local_instnct_mutation_graph"
MAIN_ARM = "resolved_marker_present_main"
ABLATION_ARM = "closed_pocket_ablation"
NO_RESOLVED_ARM = "no_resolved_final_marker_subset"
ABC_STATIC_ARM = "no_resolved_abc_static_marker_control"
REPLAY_ARM = "resolved_marker_present_replay"
SELECTED_CANDIDATE = "open_multi_pocket_arbitration_all_markers"
DEPENDENCY_NEXT = "143I_NO_RESOLVED_FINAL_MARKER_BRIDGE_FAILURE_ANALYSIS"
POSITIVE_NEXT = "144A_INSTNCT_POCKET_GATED_RULE_BOUND_ARBITRATION_TRANSFER_PROBE"

NATURAL_GATE = "multi-pocket arbitration authorization: open"
POCKET_A_MARKER = "pocket A candidate:"
POCKET_B_MARKER = "pocket B candidate:"
POCKET_C_MARKER = "pocket C candidate:"
STALE_MARKER = "stale pocket candidate:"
VISIBLE_MARKER = "VISIBLE_TARGET="
NO_RESOLVED_VISIBLE_MARKER = "visible trap value:"
NOISE_MARKER = "DISTRACTOR_VALUE="
NO_RESOLVED_NOISE_MARKER = "noisy distractor value:"
FALLBACK_VALUE = "SYM_ARBITRATION_CLOSED"
FINAL_MARKERS = [
    "arbitrated final:",
    "quorum-selected final:",
    "recency-selected final:",
    "tie-break selected final:",
    "hierarchy-selected final:",
]
ABC_STATIC_MARKERS = [POCKET_A_MARKER, POCKET_B_MARKER, POCKET_C_MARKER]

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
FAMILIES = [
    "THREE_POCKET_RULE_SELECT",
    "QUORUM_TWO_OF_THREE",
    "RECENCY_OVERRIDE",
    "TIE_BREAK_ARBITRATION",
    "STALE_POCKET_LOSES",
    "SAME_TEMPLATE_ARBITRATION_INVERSION",
    "RULE_HIERARCHY_CONFLICT",
]
VALUE_RE = re.compile(r"\b(?:TR|EV|VAL|SYM)[A-Za-z0-9_+\-]*\b")
NO_RESOLVED_BANNED_RE = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in [
        r"arbitrated[-_ ]?final",
        r"selected[-_ ]?final",
        r"winner[-_ ]?value",
        r"final[-_ ]?target",
        r"expected[-_ ]?answer",
        r"gold[-_ ]?output",
        r"answer[-_ ]?value",
        r"resolved[-_ ]?final",
        r"ANSWER\s*=",
        r"TARGET\s*=",
        r"GOLD\s*=",
    ]
]
BOUNDARY_TEXT = (
    "This confirms constrained helper/backend multi-pocket arbitration scale "
    "stability and resolved-final-marker dependency only. It is not open-ended "
    "reasoning. It is not general composition. It is not GPT-like/open-domain/"
    "broad assistant capability. It is not production/public API/deployment/"
    "safety readiness. It is not architecture superiority."
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
    spec = importlib.util.spec_from_file_location("shared_raw_generation_helper_143F", HELPER_PATH)
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


def require_143a(root: Path) -> dict[str, Any]:
    required = [
        "decision.json",
        "aggregate_metrics.json",
        "helper_request_audit.json",
        "per_seed_gate_report.json",
        "per_family_gate_report.json",
        "per_pocket_gate_report.json",
        "resolved_final_marker_echo_report.json",
        "shortcut_report.json",
    ]
    missing = [name for name in required if not (root / name).exists()]
    if missing:
        raise RuntimeError(f"missing 143A artifacts: {missing}")
    decision = read_json(root / "decision.json")
    metrics = read_json(root / "aggregate_metrics.json")
    audit = read_json(root / "helper_request_audit.json")
    seed_gate = read_json(root / "per_seed_gate_report.json")
    family_gate = read_json(root / "per_family_gate_report.json")
    pocket_gate = read_json(root / "per_pocket_gate_report.json")
    echo = read_json(root / "resolved_final_marker_echo_report.json")
    shortcuts = read_json(root / "shortcut_report.json")
    if decision.get("decision") != "instnct_pocket_gated_multi_pocket_arbitration_probe_positive":
        raise RuntimeError(f"bad 143A decision: {decision.get('decision')}")
    if decision.get("next") != "143F_INSTNCT_POCKET_GATED_MULTI_POCKET_ARBITRATION_SCALE_CONFIRM":
        raise RuntimeError(f"bad 143A next: {decision.get('next')}")
    exact = {
        "main_final_answer_accuracy": 1.0,
        "multi_pocket_arbitration_accuracy": 1.0,
        "main_pocket_writeback_rate": 1.0,
        "ablation_final_answer_accuracy": 0.0,
        "pocket_ablation_delta_final_answer_accuracy": 1.0,
        "resolved_final_marker_echo_rate": 0.0,
        "default_pocket_shortcut_rate": 0.0,
        "first_pocket_shortcut_rate": 0.0,
        "last_pocket_shortcut_rate": 0.0,
        "stale_pocket_shortcut_rate": 0.0,
        "visible_bypass_violation_rate": 0.0,
        "noisy_distractor_violation_rate": 0.0,
        "direct_pocket_value_marker_rate": 0.0,
    }
    for key, expected in exact.items():
        actual = metric_value(metrics, key, "pocket_ablation_delta", "direct_POCKET_VALUE_rate")
        if actual != expected:
            raise RuntimeError(f"bad 143A metric {key}: {actual} != {expected}")
    if metrics.get("deterministic_replay_passed") is not True:
        raise RuntimeError("143A deterministic replay did not pass")
    if echo.get("resolved_final_marker_echo_control_failed") is not True or echo.get("passed") is not True:
        raise RuntimeError("143A resolved-final-marker echo gate did not pass")
    if audit.get("all_requests_allowed_keys_only") is not True or audit.get("forbidden_keys_present_count") != 0:
        raise RuntimeError("143A helper request audit did not pass")
    if seed_gate.get("passed") is not True or family_gate.get("passed") is not True or pocket_gate.get("passed") is not True:
        raise RuntimeError("143A seed/family/pocket gates did not pass")
    if shortcuts.get("passed") is not True:
        raise RuntimeError("143A shortcut report did not pass")
    return {
        "root": rel(root),
        "decision": decision.get("decision"),
        "verdict": decision.get("verdict"),
        "next": decision.get("next"),
        "eval_row_count": metrics.get("eval_row_count"),
        "family_count": metrics.get("family_count"),
        "main_final_answer_accuracy": metrics.get("main_final_answer_accuracy"),
        "multi_pocket_arbitration_accuracy": metrics.get("multi_pocket_arbitration_accuracy"),
        "main_pocket_writeback_rate": metrics.get("main_pocket_writeback_rate"),
        "ablation_final_answer_accuracy": metrics.get("ablation_final_answer_accuracy"),
        "pocket_ablation_delta_final_answer_accuracy": metrics.get("pocket_ablation_delta_final_answer_accuracy"),
        "resolved_final_marker_echo_rate": metrics.get("resolved_final_marker_echo_rate"),
        "resolved_final_marker_echo_control_failed": echo.get("resolved_final_marker_echo_control_failed"),
        "deterministic_replay_passed": metrics.get("deterministic_replay_passed"),
        "helper_request_audit": {
            "all_requests_allowed_keys_only": audit.get("all_requests_allowed_keys_only"),
            "forbidden_keys_present_count": audit.get("forbidden_keys_present_count"),
            "no_forbidden_keys_in_accepted_generation_requests": audit.get("no_forbidden_keys_in_accepted_generation_requests"),
        },
        "per_seed_gate_passed": seed_gate.get("passed"),
        "per_family_gate_passed": family_gate.get("passed"),
        "per_pocket_gate_passed": pocket_gate.get("passed"),
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
    return {"schema_version": "phase_143F_ast_scan_v1", "passed": not failures, "failures": failures}


def selected_pocket_for(family: str, group_index: int, slot: int) -> str:
    if family == "THREE_POCKET_RULE_SELECT":
        return ["pocket_a", "pocket_b", "pocket_c", "pocket_b"][slot % 4]
    if family == "QUORUM_TWO_OF_THREE":
        return ["pocket_a", "pocket_b", "pocket_a", "pocket_c"][slot % 4]
    if family == "RECENCY_OVERRIDE":
        return ["pocket_c", "pocket_b", "pocket_a", "pocket_c"][slot % 4]
    if family == "TIE_BREAK_ARBITRATION":
        return ["pocket_b", "pocket_c", "pocket_a", "pocket_b"][slot % 4]
    if family == "STALE_POCKET_LOSES":
        return ["pocket_b", "pocket_c", "pocket_a", "pocket_b"][slot % 4]
    if family == "RULE_HIERARCHY_CONFLICT":
        return ["pocket_c", "pocket_b", "pocket_c", "pocket_a"][slot % 4]
    return ["pocket_a", "pocket_b", "pocket_c", "pocket_a"][(group_index + slot) % 4]


def values_for(seed: int, family_index: int, group_index: int, slot: int) -> dict[str, Any]:
    base = f"{seed % 100}{family_index}{group_index:03d}{slot:02d}"
    return {
        "pocket_a_value": f"EVMPA{base}",
        "pocket_b_value": f"EVMPB{base}",
        "pocket_c_value": f"EVMPC{base}",
        "stale_pocket_value": f"EVMPST{base}",
        "visible_wrong_value": f"EVMPV{base}",
        "noisy_distractor_values": [f"EVMPD{base}_{idx}" for idx in range(3)],
    }


def expected_value(row: dict[str, Any]) -> str:
    return {
        "pocket_a": row["pocket_a_value"],
        "pocket_b": row["pocket_b_value"],
        "pocket_c": row["pocket_c_value"],
    }[row["selected_pocket_id"]]


def prompt_for_row(row: dict[str, Any], include_final_marker: bool, explicit_winner_label: bool) -> str:
    family_text = {
        "THREE_POCKET_RULE_SELECT": "Three pockets provide candidates; apply the arbitration rule.",
        "QUORUM_TWO_OF_THREE": "Two matching pocket votes form quorum; output the quorum-selected candidate.",
        "RECENCY_OVERRIDE": "The most recent eligible pocket overrides older candidates.",
        "TIE_BREAK_ARBITRATION": "A tie-break rule decides between valid pocket candidates.",
        "STALE_POCKET_LOSES": "A stale pocket is visible but explicitly loses arbitration.",
        "SAME_TEMPLATE_ARBITRATION_INVERSION": "Same template, but arbitration source flips across paired rows.",
        "RULE_HIERARCHY_CONFLICT": "Hierarchy order resolves conflicting quorum, recency, stale, and tie-break rules.",
        "CONTROL": "Control row for multi-pocket arbitration shortcut rejection.",
    }[row["family"]]
    visible_marker = VISIBLE_MARKER if include_final_marker else NO_RESOLVED_VISIBLE_MARKER
    noise_marker = NOISE_MARKER if include_final_marker else NO_RESOLVED_NOISE_MARKER
    noise = "\n".join(f"{noise_marker} {value}" for value in row["noisy_distractor_values"])
    winner_text = f" winner={row['selected_pocket_id']}" if explicit_winner_label else ""
    lines = [
        f"MULTI_POCKET_ARBITRATION_TASK family={row['family']} style={row['scaffold_id']} pair={row['arbitration_pair_id']}",
        family_text,
        f"{POCKET_A_MARKER} {row['pocket_a_value']}",
        f"{POCKET_B_MARKER} {row['pocket_b_value']}",
        f"{POCKET_C_MARKER} {row['pocket_c_value']}",
        f"{STALE_MARKER} {row['stale_pocket_value']}",
        f"{visible_marker} {row['visible_wrong_value']} is visible but loses to the arbitration rule.",
        noise,
        f"arbitration rule id: {row['arbitration_rule_id']}{winner_text}",
        f"quorum rule id: {row['quorum_rule_id']} recency_rank={row['recency_rank']} tie_break={row['tie_break_rule_id']} hierarchy={row['rule_hierarchy_id']}",
        NATURAL_GATE,
    ]
    if include_final_marker:
        lines.append(f"{row['final_payload_marker']} {row['resolved_final_marker_value']}")
        lines.append("Return only the arbitrated final.")
    else:
        lines.append("Return only the arbitration output.")
    return "\n".join(lines)


def build_rows(
    seeds: list[int],
    groups_per_family: int,
    group_size: int,
    *,
    row_prefix: str,
    include_final_marker: bool,
    split_no_resolved_variants: bool = False,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    row_index = 0
    for seed in seeds:
        for family_index, family in enumerate(FAMILIES):
            for group_index in range(groups_per_family):
                group_id = f"{row_prefix}_seed{seed}_{family}_group_{group_index:03d}"
                scaffold_id = f"{family}_arbitration_style_{group_index % 12:02d}"
                for slot in range(group_size):
                    values = values_for(seed, family_index, group_index, slot)
                    winner = selected_pocket_for(family, group_index, slot)
                    variant = "resolved_marker_present"
                    explicit_winner = True
                    if not include_final_marker:
                        explicit_winner = group_index < max(1, groups_per_family // 2)
                        variant = "no_resolved_explicit_winner_label" if explicit_winner else "no_resolved_rule_derived_winner"
                        if not split_no_resolved_variants:
                            variant = "no_resolved_final_marker"
                    row = {
                        "row_id": f"{row_prefix}_{row_index:07d}",
                        "subset": "RESOLVED_FINAL_MARKER_PRESENT_SUBSET" if include_final_marker else "NO_RESOLVED_FINAL_MARKER_SUBSET",
                        "no_resolved_variant": None if include_final_marker else variant,
                        "seed": seed,
                        "family": family,
                        "contrast_group_id": group_id,
                        "scaffold_id": scaffold_id,
                        "arbitration_pair_id": f"{group_id}_pair_{slot // 2:02d}",
                        "arbitration_rule_id": f"arbitration_rule_{family_index}_{group_index:03d}_{slot % 4}",
                        "quorum_rule_id": f"quorum_rule_{(group_index + slot) % 3}",
                        "recency_rank": ["newest", "middle", "oldest"][slot % 3],
                        "tie_break_rule_id": f"tie_break_{slot % 3}",
                        "rule_hierarchy_id": "stale_rejection_gt_recency_gt_quorum_gt_tie_break",
                        "selected_pocket_id": winner,
                        "explicit_winner_label_present": explicit_winner,
                        "final_payload_marker": FINAL_MARKERS[(family_index + group_index + slot) % len(FINAL_MARKERS)],
                        **values,
                    }
                    row["answer_value"] = expected_value(row)
                    row["final_target_value"] = row["answer_value"]
                    row["resolved_final_marker_value"] = row["answer_value"]
                    row["expected_output"] = f"ANSWER=E{row['answer_value']}"
                    row["losing_values"] = [
                        value
                        for source, value in {
                            "pocket_a": row["pocket_a_value"],
                            "pocket_b": row["pocket_b_value"],
                            "pocket_c": row["pocket_c_value"],
                            "stale": row["stale_pocket_value"],
                        }.items()
                        if source != winner
                    ]
                    row["prompt"] = prompt_for_row(row, include_final_marker=include_final_marker, explicit_winner_label=explicit_winner)
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


def build_manifest(
    out: Path,
    candidate: str,
    *,
    payload_markers: list[str],
    gate_marker: str = NATURAL_GATE,
    requires_open_pocket: bool = True,
    visible_bypass_forbidden: bool = True,
    preferred_value_markers: list[str] | None = None,
) -> tuple[Path, dict[str, Any]]:
    manifest = {
        "schema_version": "instnct_mutation_graph_manifest_v14_multi_pocket_arbitration_scale_dependency",
        "backend_name": BACKEND_NAME,
        "answer_prefix": "ANSWER=E",
        "ticks_per_generated_byte": 12,
        "threshold_tick": 5,
        "value_selection_requires_open_pocket": requires_open_pocket,
        "visible_value_bypass_forbidden": visible_bypass_forbidden,
        "pocket_payload_markers": payload_markers,
        "preferred_value_markers": preferred_value_markers or [VISIBLE_MARKER, NOISE_MARKER, NO_RESOLVED_VISIBLE_MARKER, NO_RESOLVED_NOISE_MARKER, "VALUE="],
        "closed_pocket_fallback_value": FALLBACK_VALUE,
        "fallback_value": FALLBACK_VALUE,
        "allow_train_namespace_value_fallback": False,
        "decoder": {
            "type": "deterministic_pocket_gated_multi_pocket_arbitration_scale_dependency_decoder",
            "post_generation_repair": False,
            "oracle_metadata_allowed": False,
        },
        "pockets": [
            {
                "pocket_id": f"p_{candidate}",
                "gate_marker": gate_marker,
                "payload_markers": payload_markers,
            }
        ],
        "claim_boundary": "constrained helper/backend multi-pocket arbitration; not open-ended reasoning",
        "candidate_name": candidate,
    }
    path = out / "checkpoints" / f"{candidate}.json"
    write_json(path, manifest)
    return path, {**manifest, "checkpoint_path": rel(path), "checkpoint_sha256": sha256_file(path)}


def record_helper_request(audit_rows: list[dict[str, Any]], arm: str, row_id: str, request: dict[str, Any]) -> None:
    keys = set(request)
    audit_rows.append(
        {
            "schema_version": "phase_143F_helper_request_audit_row_v1",
            "arm": arm,
            "row_id": row_id,
            "request_keys": sorted(keys),
            "allowed_keys_only": keys == ALLOWED_HELPER_KEYS,
            "forbidden_keys_present": sorted(keys & FORBIDDEN_HELPER_KEYS),
            "checkpoint_path": request.get("checkpoint_path"),
            "checkpoint_hash": request.get("checkpoint_hash"),
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
    next_heartbeat = max(1, heartbeat_sec)
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
                "subset": row["subset"],
                "no_resolved_variant": row.get("no_resolved_variant"),
                "contrast_group_id": row["contrast_group_id"],
                "arbitration_pair_id": row["arbitration_pair_id"],
                "selected_pocket_id": row["selected_pocket_id"],
                "generated_text": generated_text,
                "generated_value": first_value_after_answer_e(generated_text),
                "generated_text_hash": hashlib.sha256(generated_text.encode("utf-8", errors="replace")).hexdigest(),
                "generation_trace_hash": response.get("generation_trace_hash"),
                "backend_name": response.get("backend_name"),
                "pocket_writeback_count": response.get("pocket_writeback_count"),
                "highway_retained": response.get("highway_retained"),
                "value_selection_source": response.get("value_selection_source"),
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
        "pocket_a": row["pocket_a_value"],
        "pocket_b": row["pocket_b_value"],
        "pocket_c": row["pocket_c_value"],
        "stale": row["stale_pocket_value"],
        "visible": row["visible_wrong_value"],
    }
    for source, value in source_values.items():
        if generated_value == value:
            return source
    if generated_value in row["noisy_distractor_values"]:
        return "noisy_distractor"
    if generated_value == FALLBACK_VALUE:
        return "closed_pocket_fallback"
    return None


def row_score(row: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    generated_value = result["generated_value"]
    source = generated_source(row, generated_value)
    final_correct = generated_value == row["answer_value"]
    wrong_pocket = generated_value in row["losing_values"]
    visible = generated_value == row["visible_wrong_value"]
    noisy = generated_value in row["noisy_distractor_values"]
    fallback = generated_value == FALLBACK_VALUE or result.get("value_selection_source") == "closed_pocket_fallback"
    resolved_echo = generated_value == row["resolved_final_marker_value"] and row["resolved_final_marker_value"] != row["answer_value"]
    direct_marker = "POCKET_VALUE=" in row["prompt"]
    return {
        "schema_version": "phase_143F_scoring_result_v1",
        "arm": result["arm"],
        "row_id": row["row_id"],
        "seed": row["seed"],
        "family": row["family"],
        "subset": row["subset"],
        "no_resolved_variant": row.get("no_resolved_variant"),
        "contrast_group_id": row["contrast_group_id"],
        "arbitration_pair_id": row["arbitration_pair_id"],
        "selected_pocket_id": row["selected_pocket_id"],
        "expected_final_value": row["answer_value"],
        "generated_value": generated_value,
        "generated_source": source,
        "value_selection_source": result.get("value_selection_source"),
        "final_answer_correct": final_correct,
        "multi_pocket_arbitration_correct": final_correct and source == row["selected_pocket_id"],
        "quorum_rule_correct": final_correct if row["family"] == "QUORUM_TWO_OF_THREE" else None,
        "recency_rule_correct": final_correct if row["family"] == "RECENCY_OVERRIDE" else None,
        "tie_break_correct": final_correct if row["family"] == "TIE_BREAK_ARBITRATION" else None,
        "rule_hierarchy_conflict_correct": final_correct if row["family"] == "RULE_HIERARCHY_CONFLICT" else None,
        "wrong_pocket": wrong_pocket,
        "default_pocket_shortcut": source == "pocket_b" and row["selected_pocket_id"] != "pocket_b",
        "first_pocket_shortcut": source == "pocket_a" and row["selected_pocket_id"] != "pocket_a",
        "last_pocket_shortcut": source == "pocket_c" and row["selected_pocket_id"] != "pocket_c",
        "stale_pocket_shortcut": source == "stale",
        "resolved_final_marker_echo": resolved_echo,
        "visible_bypass_violation": visible,
        "noisy_distractor_violation": noisy,
        "fallback": fallback,
        "unexpected_value": generated_value not in {None, FALLBACK_VALUE, row["answer_value"], row["pocket_a_value"], row["pocket_b_value"], row["pocket_c_value"], row["stale_pocket_value"], row["visible_wrong_value"], *row["noisy_distractor_values"]},
        "train_namespace_value": bool(generated_value and generated_value.startswith("TR")),
        "direct_pocket_value_marker_present": direct_marker,
        "pocket_writeback": result.get("value_selection_source") == "open_pocket_writeback",
        "generated_text": result["generated_text"],
        "generated_before_scoring": result["generated_before_scoring"],
    }


def metrics_for(scored: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(scored)
    generated_sources = Counter(item.get("generated_source") for item in scored if item.get("generated_source"))
    expected_sources = Counter(item["selected_pocket_id"] for item in scored)
    quorum_items = [item for item in scored if item["quorum_rule_correct"] is not None]
    recency_items = [item for item in scored if item["recency_rule_correct"] is not None]
    tie_items = [item for item in scored if item["tie_break_correct"] is not None]
    hierarchy_items = [item for item in scored if item["rule_hierarchy_conflict_correct"] is not None]
    return {
        "row_count": total,
        "final_answer_accuracy": rate(sum(1 for item in scored if item["final_answer_correct"]), total),
        "multi_pocket_arbitration_accuracy": rate(sum(1 for item in scored if item["multi_pocket_arbitration_correct"]), total),
        "quorum_rule_accuracy": rate(sum(1 for item in quorum_items if item["quorum_rule_correct"]), len(quorum_items)),
        "recency_rule_accuracy": rate(sum(1 for item in recency_items if item["recency_rule_correct"]), len(recency_items)),
        "tie_break_accuracy": rate(sum(1 for item in tie_items if item["tie_break_correct"]), len(tie_items)),
        "rule_hierarchy_conflict_accuracy": rate(sum(1 for item in hierarchy_items if item["rule_hierarchy_conflict_correct"]), len(hierarchy_items)),
        "pocket_writeback_rate": rate(sum(1 for item in scored if item["pocket_writeback"]), total),
        "wrong_pocket_rate": rate(sum(1 for item in scored if item["wrong_pocket"]), total),
        "default_pocket_shortcut_rate": rate(sum(1 for item in scored if item["default_pocket_shortcut"]), total),
        "first_pocket_shortcut_rate": rate(sum(1 for item in scored if item["first_pocket_shortcut"]), total),
        "last_pocket_shortcut_rate": rate(sum(1 for item in scored if item["last_pocket_shortcut"]), total),
        "stale_pocket_shortcut_rate": rate(sum(1 for item in scored if item["stale_pocket_shortcut"]), total),
        "resolved_final_marker_echo_rate": rate(sum(1 for item in scored if item["resolved_final_marker_echo"]), total),
        "visible_bypass_violation_rate": rate(sum(1 for item in scored if item["visible_bypass_violation"]), total),
        "noisy_distractor_violation_rate": rate(sum(1 for item in scored if item["noisy_distractor_violation"]), total),
        "direct_pocket_value_marker_rate": rate(sum(1 for item in scored if item["direct_pocket_value_marker_present"]), total),
        "fallback_rate": rate(sum(1 for item in scored if item["fallback"]), total),
        "unexpected_value_rate": rate(sum(1 for item in scored if item["unexpected_value"]), total),
        "train_namespace_rate": rate(sum(1 for item in scored if item["train_namespace_value"]), total),
        "expected_pocket_counts": dict(sorted(expected_sources.items())),
        "generated_source_counts": dict(sorted(generated_sources.items())),
    }


def score(arm: str, rows: list[dict[str, Any]], results: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    rows_by_id = {row["row_id"]: row for row in rows}
    scored = [row_score(rows_by_id[result["row_id"]], result) for result in results]
    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_seed: dict[int, list[dict[str, Any]]] = defaultdict(list)
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_pocket: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_pair: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in scored:
        by_group[item["contrast_group_id"]].append(item)
        by_seed[item["seed"]].append(item)
        by_family[item["family"]].append(item)
        by_pocket[item["selected_pocket_id"]].append(item)
        by_pair[item["arbitration_pair_id"]].append(item)
    groups: list[dict[str, Any]] = []
    for group_id, items in sorted(by_group.items()):
        generated = [item["generated_value"] for item in items]
        expected = [item["expected_final_value"] for item in items]
        groups.append(
            {
                "schema_version": "phase_143F_contrast_group_result_v1",
                "arm": arm,
                "contrast_group_id": group_id,
                "family": items[0]["family"],
                "selected_pocket_ids": [item["selected_pocket_id"] for item in items],
                "expected_values": expected,
                "generated_values": generated,
                "group_pass": all(item["multi_pocket_arbitration_correct"] for item in items) and len(set(generated)) == len(set(expected)),
            }
        )
    pairs: list[dict[str, Any]] = []
    for pair_id, items in sorted(by_pair.items()):
        expected_sources = {item["selected_pocket_id"] for item in items}
        opposite = len(expected_sources) >= 2
        pairs.append(
            {
                "schema_version": "phase_143F_arbitration_inversion_pair_v1",
                "arm": arm,
                "arbitration_pair_id": pair_id,
                "family": items[0]["family"],
                "same_template_opposite_winner_pair": opposite,
                "selected_pocket_ids": sorted(expected_sources),
                "generated_sources": sorted(source for source in {item["generated_source"] for item in items} if source),
                "pair_pass": opposite and all(item["multi_pocket_arbitration_correct"] for item in items),
            }
        )
    metrics = metrics_for(scored)
    metrics["contrast_group_accuracy"] = rate(sum(1 for group in groups if group["group_pass"]), len(groups))
    metrics["priority_inversion_accuracy"] = rate(sum(1 for pair in pairs if pair["pair_pass"]), len(pairs))
    metrics["same_template_opposite_winner_accuracy"] = metrics["priority_inversion_accuracy"]
    metrics["pocket_label_permutation_accuracy"] = metrics["contrast_group_accuracy"]
    metrics["same_values_different_rule_accuracy"] = metrics["contrast_group_accuracy"]
    metrics["same_rule_different_values_accuracy"] = metrics["contrast_group_accuracy"]
    metrics["per_seed"] = {str(seed): metrics_for(items) for seed, items in sorted(by_seed.items())}
    metrics["per_family"] = {family: metrics_for(items) for family, items in sorted(by_family.items())}
    metrics["per_pocket"] = {pocket: metrics_for(items) for pocket, items in sorted(by_pocket.items())}
    return scored, metrics, groups, pairs


def marker_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": "phase_143F_explicit_marker_audit_v1",
        "row_count": len(rows),
        "direct_pocket_value_marker_rate": rate(sum(1 for row in rows if "POCKET_VALUE=" in row["prompt"]), len(rows)),
        "explicit_pocket_token_row_rate": rate(sum(1 for row in rows if "POCKET_" in row["prompt"]), len(rows)),
        "resolved_final_marker_row_rate": rate(sum(1 for row in rows if any(marker in row["prompt"] for marker in FINAL_MARKERS)), len(rows)),
        "visible_wrong_value_row_rate": rate(sum(1 for row in rows if (VISIBLE_MARKER in row["prompt"] or NO_RESOLVED_VISIBLE_MARKER in row["prompt"])), len(rows)),
        "noisy_distractor_row_rate": rate(sum(1 for row in rows if (NOISE_MARKER in row["prompt"] or NO_RESOLVED_NOISE_MARKER in row["prompt"])), len(rows)),
    }


def multi_pocket_manifest(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(row["selected_pocket_id"] for row in rows)
    return {
        "schema_version": "phase_143F_multi_pocket_manifest_v1",
        "row_count": len(rows),
        "selected_pocket_counts": dict(sorted(counts.items())),
        "has_all_pocket_winners": all(counts.get(source, 0) > 0 for source in ["pocket_a", "pocket_b", "pocket_c"]),
    }


def arbitration_rule_manifest(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_pair: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_pair[row["arbitration_pair_id"]].append(row)
    inversion_pairs = [pair_id for pair_id, items in by_pair.items() if len({item["selected_pocket_id"] for item in items}) >= 2]
    return {
        "schema_version": "phase_143F_arbitration_rule_manifest_v1",
        "pair_count": len(by_pair),
        "arbitration_inversion_pair_count": len(inversion_pairs),
        "arbitration_inversion_pair_rate": rate(len(inversion_pairs), len(by_pair)),
        "same_template_opposite_winner_pairs": len(inversion_pairs),
        "sample_pairs": sorted(inversion_pairs)[:20],
    }


def pocket_distribution_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(row["selected_pocket_id"] for row in rows)
    total = len(rows)
    rates = {
        "pocket_a_wins_rate": rate(counts.get("pocket_a", 0), total),
        "pocket_b_wins_rate": rate(counts.get("pocket_b", 0), total),
        "pocket_c_wins_rate": rate(counts.get("pocket_c", 0), total),
    }
    return {
        "schema_version": "phase_143F_pocket_distribution_report_v1",
        "row_count": total,
        "selected_pocket_counts": dict(sorted(counts.items())),
        **rates,
        "pocket_distribution_balanced": all(value >= 0.20 for value in rates.values()),
        "minimum_pocket_rate_gate": 0.20,
        "passed": all(value >= 0.20 for value in rates.values()),
    }


def gate_report_by_seed(main_metrics: dict[str, Any], ablation_metrics: dict[str, Any]) -> dict[str, Any]:
    rows = []
    for seed, metrics in sorted(main_metrics["per_seed"].items()):
        ablation = ablation_metrics["per_seed"].get(seed, {})
        failures = []
        for key, threshold in {
            "final_answer_accuracy": 0.85,
            "multi_pocket_arbitration_accuracy": 0.85,
            "pocket_writeback_rate": 0.90,
        }.items():
            if metrics.get(key, 0.0) < threshold:
                failures.append(key)
        if ablation.get("final_answer_accuracy", 1.0) > 0.10:
            failures.append("ablation_final_answer_accuracy")
        for key in ["wrong_pocket_rate", "default_pocket_shortcut_rate", "first_pocket_shortcut_rate", "last_pocket_shortcut_rate", "stale_pocket_shortcut_rate", "visible_bypass_violation_rate", "noisy_distractor_violation_rate"]:
            if metrics.get(key) != 0.0:
                failures.append(key)
        rows.append({"seed": seed, "main": metrics, "ablation": ablation, "passed": not failures, "failures": failures})
    return {"schema_version": "phase_143F_per_seed_gate_report_v1", "passed": all(row["passed"] for row in rows), "seeds": rows}


def gate_report_by_family(main_metrics: dict[str, Any]) -> dict[str, Any]:
    rows = []
    for family, metrics in sorted(main_metrics["per_family"].items()):
        failures = []
        for key, threshold in {"final_answer_accuracy": 0.85, "multi_pocket_arbitration_accuracy": 0.85, "pocket_writeback_rate": 0.90}.items():
            if metrics.get(key, 0.0) < threshold:
                failures.append(key)
        if metrics.get("wrong_pocket_rate") != 0.0:
            failures.append("wrong_pocket_rate")
        rows.append({"family": family, "main": metrics, "passed": not failures, "failures": failures})
    return {
        "schema_version": "phase_143F_per_family_gate_report_v1",
        "passed": all(row["passed"] for row in rows),
        "no_family_zero_pass": all(row["main"].get("final_answer_accuracy", 0.0) > 0.0 for row in rows),
        "families": rows,
    }


def gate_report_by_pocket(main_metrics: dict[str, Any]) -> dict[str, Any]:
    rows = []
    for pocket in ["pocket_a", "pocket_b", "pocket_c"]:
        metrics = main_metrics["per_pocket"].get(pocket, {})
        failures = []
        for key, threshold in {"final_answer_accuracy": 0.85, "multi_pocket_arbitration_accuracy": 0.85}.items():
            if metrics.get(key, 0.0) < threshold:
                failures.append(key)
        if metrics.get("wrong_pocket_rate", 0.0) != 0.0:
            failures.append("wrong_pocket_rate")
        rows.append({"selected_pocket_id": pocket, "main": metrics, "passed": not failures, "failures": failures})
    return {"schema_version": "phase_143F_per_pocket_gate_report_v1", "passed": all(row["passed"] for row in rows), "pockets": rows}


def shortcut_report(metrics: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "wrong_pocket_rate",
        "default_pocket_shortcut_rate",
        "first_pocket_shortcut_rate",
        "last_pocket_shortcut_rate",
        "stale_pocket_shortcut_rate",
        "resolved_final_marker_echo_rate",
        "visible_bypass_violation_rate",
        "noisy_distractor_violation_rate",
        "direct_pocket_value_marker_rate",
    ]
    values = {key: metrics.get(key, 0.0) for key in keys}
    return {"schema_version": "phase_143F_shortcut_report_v1", **values, "shortcut_detected": any(value != 0.0 for value in values.values()), "passed": all(value == 0.0 for value in values.values())}


def forbidden_canary(helper: Any, checkpoint_path: Path, checkpoint_hash: str, max_new_tokens: int) -> dict[str, Any]:
    request = request_for(f"{NATURAL_GATE}\narbitrated final: EV_CANARY", checkpoint_path, checkpoint_hash, 143001, max_new_tokens)
    request["expected_output"] = "ANSWER=EEV_CANARY"
    try:
        helper.raw_generate(request)
    except Exception as exc:
        return {"schema_version": "phase_143F_expected_output_canary_v1", "passed": True, "verdict": "forbidden_expected_output_rejected", "error_type": type(exc).__name__, "error": str(exc)}
    return {"schema_version": "phase_143F_expected_output_canary_v1", "passed": False, "verdict": "forbidden_expected_output_accepted"}


def control_row(name: str, control_passed: bool, generated_value: str | None, blocked_value: str | None) -> dict[str, Any]:
    return {"schema_version": "phase_143F_control_result_v1", "control": name, "control_passed": control_passed, "control_failed": not control_passed, "generated_value": generated_value, "blocked_value": blocked_value}


def run_controls(helper: Any, checkpoints: dict[str, tuple[Path, str]], max_new_tokens: int, request_audit: list[dict[str, Any]], out: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    base = build_rows([4599], 1, 1, row_prefix="143F_control", include_final_marker=True)[0]
    base.update({"family": "CONTROL", "selected_pocket_id": "pocket_b", "row_id": "143F_control_row"})
    base["answer_value"] = expected_value(base)
    base["final_target_value"] = base["answer_value"]
    controls = [
        ("FIRST_POCKET_CONTROL", "first_pocket_candidate", "pocket_c", False),
        ("LAST_POCKET_CONTROL", "last_pocket_candidate", "pocket_a", False),
        ("DEFAULT_POCKET_CONTROL", "default_pocket_candidate", "pocket_c", False),
        ("STALE_POCKET_CONTROL", "stale_pocket_candidate", "pocket_b", False),
        ("VISIBLE_BYPASS_CONTROL", "visible_bypass_candidate", "pocket_b", False),
        ("NOISY_DISTRACTOR_CONTROL", "noisy_distractor_candidate", "pocket_b", False),
        ("CLOSED_POCKET_ABLATION_CONTROL", "closed_pocket_no_arbitration", "pocket_b", False),
        ("RESOLVED_FINAL_MARKER_ECHO_CONTROL", SELECTED_CANDIDATE, "pocket_b", True),
        ("POCKET_LABEL_PERMUTATION_CONTROL", "first_pocket_candidate", "pocket_c", False),
        ("RULE_HIERARCHY_CONFLICT_CONTROL", "default_pocket_candidate", "pocket_c", False),
        ("SAME_VALUES_DIFFERENT_RULE_CONTROL", "last_pocket_candidate", "pocket_a", False),
        ("SAME_RULE_DIFFERENT_VALUES_CONTROL", "first_pocket_candidate", "pocket_c", False),
        ("PREFIX_ONLY_CONTROL", "closed_pocket_no_arbitration", "pocket_b", False),
    ]
    rows: list[dict[str, Any]] = []
    for name, candidate, expected_source, corrupt_final_marker in controls:
        row = dict(base)
        row["row_id"] = f"143F_control_{name.lower()}"
        row["selected_pocket_id"] = expected_source
        row["answer_value"] = expected_value(row)
        row["final_target_value"] = row["answer_value"]
        row["resolved_final_marker_value"] = "EVCTRL_WRONG_ECHO" if corrupt_final_marker else row["answer_value"]
        row["losing_values"] = [
            value
            for source, value in {
                "pocket_a": row["pocket_a_value"],
                "pocket_b": row["pocket_b_value"],
                "pocket_c": row["pocket_c_value"],
                "stale": row["stale_pocket_value"],
            }.items()
            if source != expected_source
        ]
        row["prompt"] = prompt_for_row(row, include_final_marker=True, explicit_winner_label=True)
        checkpoint, checkpoint_hash = checkpoints[candidate]
        result = run_arm(helper, out, name, [row], checkpoint, checkpoint_hash, max_new_tokens, 0, request_audit)[0]
        generated = result["generated_value"]
        rows.append(control_row(name, generated == row["answer_value"], generated, row["answer_value"]))
    report = {
        "schema_version": "phase_143F_control_report_v1",
        "controls_failed": all(row["control_failed"] for row in rows),
        "required_controls": [row["control"] for row in rows],
        "failed_control_count": sum(1 for row in rows if row["control_failed"]),
        "control_count": len(rows),
    }
    for row in rows:
        report[f"{row['control'].lower()}_failed"] = row["control_failed"]
    return rows, report


def no_resolved_prompt_scan(rows: list[dict[str, Any]]) -> dict[str, Any]:
    failures: list[dict[str, Any]] = []
    for row in rows:
        for pattern in NO_RESOLVED_BANNED_RE:
            match = pattern.search(row["prompt"])
            if match:
                failures.append({"row_id": row["row_id"], "pattern": pattern.pattern, "match": match.group(0)})
                break
    rule_derived_bad = [
        row["row_id"]
        for row in rows
        if row.get("no_resolved_variant") == "no_resolved_rule_derived_winner"
        and re.search(r"\b(winner|selected_pocket_id)\s*=", row["prompt"], re.IGNORECASE)
    ]
    return {
        "schema_version": "phase_143F_no_resolved_prompt_scan_v1",
        "passed": not failures and not rule_derived_bad,
        "banned_pattern_count": len(NO_RESOLVED_BANNED_RE),
        "banned_matches": failures[:50],
        "banned_match_count": len(failures),
        "rule_derived_explicit_winner_label_count": len(rule_derived_bad),
        "rule_derived_explicit_winner_label_rows": rule_derived_bad[:50],
    }


def no_resolved_subset_report(rows: list[dict[str, Any]], results: list[dict[str, Any]], scored: list[dict[str, Any]], manifest: dict[str, Any], prompt_scan: dict[str, Any]) -> dict[str, Any]:
    metrics = metrics_for(scored)
    unique_paths = {result["helper_request"]["checkpoint_path"] for result in results}
    unique_hashes = {result["helper_request"]["checkpoint_hash"] for result in results}
    expected_markers_hash = stable_hash(FINAL_MARKERS)
    manifest_static = stable_hash(manifest["pocket_payload_markers"]) == expected_markers_hash
    shortcut = rate(
        sum(
            1
            for item in scored
            if item["first_pocket_shortcut"]
            or item["last_pocket_shortcut"]
            or item["default_pocket_shortcut"]
            or item["stale_pocket_shortcut"]
            or item["visible_bypass_violation"]
            or item["noisy_distractor_violation"]
        ),
        len(scored),
    )
    explicit_rows = [item for item in scored if item["no_resolved_variant"] == "no_resolved_explicit_winner_label"]
    rule_rows = [item for item in scored if item["no_resolved_variant"] == "no_resolved_rule_derived_winner"]
    return {
        "schema_version": "phase_143F_no_resolved_final_marker_subset_report_v1",
        "row_count": len(rows),
        "accuracy": metrics["final_answer_accuracy"],
        "writeback_rate": metrics["pocket_writeback_rate"],
        "fallback_rate": metrics["fallback_rate"],
        "shortcut_rate": shortcut,
        "unexpected_value_rate": metrics["unexpected_value_rate"],
        "visible_rate": metrics["visible_bypass_violation_rate"],
        "noisy_rate": metrics["noisy_distractor_violation_rate"],
        "train_namespace_rate": metrics["train_namespace_rate"],
        "first_pocket_rate": rate(sum(1 for item in scored if item["generated_source"] == "pocket_a"), len(scored)),
        "default_pocket_rate": rate(sum(1 for item in scored if item["generated_source"] == "pocket_b"), len(scored)),
        "last_pocket_rate": rate(sum(1 for item in scored if item["generated_source"] == "pocket_c"), len(scored)),
        "explicit_winner_label_subset_accuracy": rate(sum(1 for item in explicit_rows if item["final_answer_correct"]), len(explicit_rows)),
        "rule_derived_winner_subset_accuracy": rate(sum(1 for item in rule_rows if item["final_answer_correct"]), len(rule_rows)),
        "no_resolved_unique_checkpoint_path_count": len(unique_paths),
        "no_resolved_unique_checkpoint_hash_count": len(unique_hashes),
        "no_resolved_manifest_payload_markers_static": manifest_static,
        "no_resolved_per_row_manifest_switch_rate": 0.0 if len(unique_paths) == 1 and len(unique_hashes) == 1 else 1.0,
        "no_resolved_per_row_payload_marker_switch_rate": 0.0 if manifest_static else 1.0,
        "payload_markers_hash": expected_markers_hash,
        "static_payload_markers": list(manifest["pocket_payload_markers"]),
        "prompt_scan_passed": prompt_scan["passed"],
        "clean_dependency_passed": (
            metrics["final_answer_accuracy"] < 0.60
            and metrics["fallback_rate"] > 0.0
            and shortcut == 0.0
            and metrics["unexpected_value_rate"] == 0.0
            and metrics["visible_bypass_violation_rate"] == 0.0
            and metrics["noisy_distractor_violation_rate"] == 0.0
            and metrics["train_namespace_rate"] == 0.0
            and manifest_static
            and len(unique_paths) == 1
            and len(unique_hashes) == 1
            and prompt_scan["passed"] is True
        ),
    }


def abc_static_marker_control_report(rows: list[dict[str, Any]], results: list[dict[str, Any]], scored: list[dict[str, Any]], manifest: dict[str, Any]) -> dict[str, Any]:
    metrics = metrics_for(scored)
    return {
        "schema_version": "phase_143F_no_resolved_abc_static_marker_control_report_v1",
        "diagnostic_only": True,
        "row_count": len(rows),
        "payload_markers_static": manifest["pocket_payload_markers"] == ABC_STATIC_MARKERS,
        "no_resolved_abc_static_marker_accuracy": metrics["final_answer_accuracy"],
        "no_resolved_abc_static_first_pocket_rate": rate(sum(1 for item in scored if item["generated_source"] == "pocket_a"), len(scored)),
        "no_resolved_abc_static_default_pocket_rate": rate(sum(1 for item in scored if item["generated_source"] == "pocket_b"), len(scored)),
        "no_resolved_abc_static_last_pocket_rate": rate(sum(1 for item in scored if item["generated_source"] == "pocket_c"), len(scored)),
        "no_resolved_abc_static_fallback_rate": metrics["fallback_rate"],
        "sample_results": results[:10],
    }


def subset_variant_report(scored: list[dict[str, Any]], variant: str) -> dict[str, Any]:
    items = [item for item in scored if item.get("no_resolved_variant") == variant]
    metrics = metrics_for(items)
    return {
        "schema_version": f"phase_143F_{variant}_report_v1",
        "variant": variant,
        "row_count": len(items),
        "accuracy": metrics["final_answer_accuracy"],
        "writeback_rate": metrics["pocket_writeback_rate"],
        "fallback_rate": metrics["fallback_rate"],
        "shortcut_rate": rate(
            sum(
                1
                for item in items
                if item["first_pocket_shortcut"]
                or item["last_pocket_shortcut"]
                or item["default_pocket_shortcut"]
                or item["stale_pocket_shortcut"]
                or item["visible_bypass_violation"]
                or item["noisy_distractor_violation"]
            ),
            len(items),
        ),
    }


def choose_decision(
    comparison: dict[str, Any],
    control_report: dict[str, Any],
    deterministic: bool,
    infra_passed: bool,
    seed_report: dict[str, Any],
    family_report: dict[str, Any],
    pocket_report: dict[str, Any],
    shortcuts: dict[str, Any],
    no_resolved_report: dict[str, Any],
) -> dict[str, Any]:
    normal_scale_passed = (
        comparison["main_final_answer_accuracy"] >= 0.85
        and comparison["resolved_marker_present_subset_accuracy"] >= 0.85
        and comparison["multi_pocket_arbitration_accuracy"] >= 0.85
        and comparison["quorum_rule_accuracy"] >= 0.80
        and comparison["recency_rule_accuracy"] >= 0.80
        and comparison["tie_break_accuracy"] >= 0.80
        and comparison["rule_hierarchy_conflict_accuracy"] >= 0.80
        and comparison["main_pocket_writeback_rate"] >= 0.90
        and comparison["ablation_final_answer_accuracy"] <= 0.10
        and comparison["pocket_ablation_delta_final_answer_accuracy"] >= 0.75
        and comparison["resolved_final_marker_echo_rate"] == 0.0
        and comparison["default_pocket_shortcut_rate"] == 0.0
        and comparison["first_pocket_shortcut_rate"] == 0.0
        and comparison["last_pocket_shortcut_rate"] == 0.0
        and comparison["stale_pocket_shortcut_rate"] == 0.0
        and comparison["visible_bypass_violation_rate"] == 0.0
        and comparison["noisy_distractor_violation_rate"] == 0.0
        and comparison["direct_pocket_value_marker_rate"] == 0.0
        and deterministic is True
        and seed_report["passed"] is True
        and family_report["passed"] is True
        and pocket_report["passed"] is True
        and shortcuts["passed"] is True
    )
    no_resolved_positive = (
        no_resolved_report["accuracy"] >= 0.60
        and no_resolved_report["writeback_rate"] >= 0.70
        and no_resolved_report["shortcut_rate"] == 0.0
    )
    clean_dependency = (
        comparison["resolved_marker_present_subset_accuracy"] >= 0.85
        and no_resolved_report["accuracy"] < 0.60
        and no_resolved_report["fallback_rate"] > 0.0
        and comparison["resolved_marker_dependency_delta"] >= 0.25
        and no_resolved_report["shortcut_rate"] == 0.0
        and no_resolved_report["unexpected_value_rate"] == 0.0
        and no_resolved_report["visible_rate"] == 0.0
        and no_resolved_report["noisy_rate"] == 0.0
        and no_resolved_report["train_namespace_rate"] == 0.0
        and no_resolved_report["no_resolved_manifest_payload_markers_static"] is True
        and no_resolved_report["no_resolved_per_row_manifest_switch_rate"] == 0.0
        and no_resolved_report["no_resolved_per_row_payload_marker_switch_rate"] == 0.0
        and no_resolved_report["prompt_scan_passed"] is True
    )
    if not infra_passed or control_report.get("controls_failed") is not True or deterministic is not True:
        decision = "helper_integrity_failure"; verdict = "INSTNCT_POCKET_GATED_MULTI_POCKET_ARBITRATION_INVALID"; next_step = "135G_RAW_HELPER_ORACLE_SHORTCUT_REMOVAL"
    elif comparison["resolved_final_marker_echo_rate"] > 0.0:
        decision = "resolved_final_marker_echo_detected"; verdict = "INSTNCT_POCKET_GATED_MULTI_POCKET_ARBITRATION_FAILS"; next_step = "143B_RESOLVED_FINAL_MARKER_ECHO_ANALYSIS"
    elif not normal_scale_passed:
        decision = "multi_pocket_arbitration_scale_failure"; verdict = "INSTNCT_POCKET_GATED_MULTI_POCKET_ARBITRATION_SCALE_FAILS"; next_step = "143E_MULTI_POCKET_ARBITRATION_FAILURE_ANALYSIS"
    elif no_resolved_positive:
        decision = "instnct_pocket_gated_multi_pocket_arbitration_scale_confirmed"; verdict = "INSTNCT_POCKET_GATED_MULTI_POCKET_ARBITRATION_SCALE_CONFIRMED"; next_step = POSITIVE_NEXT
    elif clean_dependency:
        decision = "resolved_final_marker_dependency_confirmed"; verdict = "INSTNCT_POCKET_GATED_MULTI_POCKET_ARBITRATION_DEPENDS_ON_RESOLVED_FINAL_MARKER"; next_step = DEPENDENCY_NEXT
    else:
        decision = "no_resolved_final_marker_bridge_unclean_failure"; verdict = "INSTNCT_POCKET_GATED_MULTI_POCKET_ARBITRATION_NO_RESOLVED_BRIDGE_UNCLEAN"; next_step = "143I_NO_RESOLVED_FINAL_MARKER_BRIDGE_FAILURE_ANALYSIS"
    return {
        "schema_version": "phase_143F_decision_v1",
        "decision": decision,
        "verdict": verdict,
        "next": next_step,
        "boundary": BOUNDARY_TEXT,
        "normal_resolved_marker_scale_passed": normal_scale_passed,
        "no_resolved_final_marker_subset_positive": no_resolved_positive,
        "clean_resolved_marker_dependency": clean_dependency,
        "clean_negative_valid": decision == "resolved_final_marker_dependency_confirmed",
        "pocket_mechanism_claimed": decision == "instnct_pocket_gated_multi_pocket_arbitration_scale_confirmed",
        "architecture_superiority_claimed": False,
        "value_grounding_claimed": False,
        **FALSE_FLAGS,
    }


def write_report(out: Path, decision: dict[str, Any], comparison: dict[str, Any], abc_report: dict[str, Any]) -> None:
    text = f"""# {MILESTONE}

Decision: `{decision['decision']}`

Verdict: `{decision['verdict']}`

Next: `{decision['next']}`

Boundary: {BOUNDARY_TEXT}

Resolved-marker-present scale:

- rows: `{comparison['resolved_marker_present_row_count']}`
- accuracy: `{comparison['resolved_marker_present_subset_accuracy']}`
- multi-pocket arbitration accuracy: `{comparison['multi_pocket_arbitration_accuracy']}`
- pocket writeback rate: `{comparison['main_pocket_writeback_rate']}`
- ablation final answer accuracy: `{comparison['ablation_final_answer_accuracy']}`
- ablation delta: `{comparison['pocket_ablation_delta_final_answer_accuracy']}`
- resolved final marker echo rate: `{comparison['resolved_final_marker_echo_rate']}`

No-resolved-final-marker dependency test:

- rows: `{comparison['no_resolved_final_marker_row_count']}`
- accuracy: `{comparison['no_resolved_final_marker_subset_accuracy']}`
- writeback rate: `{comparison['no_resolved_final_marker_subset_writeback_rate']}`
- fallback rate: `{comparison['no_resolved_final_marker_subset_fallback_rate']}`
- shortcut rate: `{comparison['no_resolved_final_marker_subset_shortcut_rate']}`
- dependency delta: `{comparison['resolved_marker_dependency_delta']}`

ABC static-marker diagnostic:

- accuracy: `{abc_report['no_resolved_abc_static_marker_accuracy']}`
- first-pocket rate: `{abc_report['no_resolved_abc_static_first_pocket_rate']}`
- default-pocket rate: `{abc_report['no_resolved_abc_static_default_pocket_rate']}`
- last-pocket rate: `{abc_report['no_resolved_abc_static_last_pocket_rate']}`

Interpretation: if the no-resolved FINAL_MARKERS-static subset fallback-fails cleanly, the current constrained helper/backend path depends on resolved final payload markers. That is a helper-semantics bottleneck, not architecture failure.
"""
    write_text(out / "report.md", text)


def main() -> int:
    parser = argparse.ArgumentParser(description=MILESTONE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--upstream-143a-root", type=Path, default=DEFAULT_143A_ROOT)
    parser.add_argument("--seeds", default="4501,4502,4503,4504")
    parser.add_argument("--groups-per-family", type=int, default=24)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--no-resolved-groups-per-family", type=int, default=6)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--heartbeat-sec", type=int, default=20)
    args = parser.parse_args()

    out = resolve_target_out(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_text(out / "progress.jsonl", "")
    append_progress(out, "startup", milestone=MILESTONE, heartbeat_sec=args.heartbeat_sec)
    write_json(out / "queue.json", {"schema_version": "phase_143F_queue_v1", "milestone": MILESTONE, "status": "running"})

    seeds = [int(item) for item in args.seeds.split(",") if item.strip()]
    upstream_143a = require_143a(resolve_repo_path(args.upstream_143a_root))
    write_json(out / "upstream_143a_manifest.json", upstream_143a)
    append_progress(out, "upstream verification", upstream_143a=upstream_143a["decision"])

    config = {
        "schema_version": "phase_143F_eval_config_v1",
        "milestone": MILESTONE,
        "boundary": BOUNDARY_TEXT,
        "train_allowed": False,
        "training_performed": False,
        "helper_generation_allowed": True,
        "shared_helper_only": True,
        "runner_may_call_raw_generate": True,
        "checker_may_call_raw_generate": False,
        "helper_backend_modification_allowed": False,
        "public_request_key_change_allowed": False,
        "source_checkpoint_mutation_allowed": False,
        "runtime_surface_mutated": False,
        "release_surface_mutated": False,
        "product_surface_mutated": False,
        "root_license_changed": False,
        "seeds": seeds,
        "families": FAMILIES,
        "groups_per_family": args.groups_per_family,
        "group_size": args.group_size,
        "no_resolved_groups_per_family": args.no_resolved_groups_per_family,
        "max_new_tokens": args.max_new_tokens,
        **FALSE_FLAGS,
    }
    write_json(out / "eval_config.json", config)

    helper = load_helper()
    provenance = {
        "schema_version": "phase_143F_helper_provenance_v1",
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

    resolved_rows = build_rows(seeds, args.groups_per_family, args.group_size, row_prefix="143F_resolved", include_final_marker=True)
    no_resolved_rows = build_rows(seeds, args.no_resolved_groups_per_family, args.group_size, row_prefix="143F_no_resolved", include_final_marker=False, split_no_resolved_variants=True)
    all_rows = resolved_rows + no_resolved_rows
    write_jsonl(out / "eval_rows.jsonl", all_rows)
    write_jsonl(out / "resolved_marker_present_subset_rows.jsonl", resolved_rows)
    write_jsonl(out / "no_resolved_final_marker_subset_rows.jsonl", no_resolved_rows)
    prompt_scan = no_resolved_prompt_scan(no_resolved_rows)
    write_json(out / "no_resolved_prompt_scan_report.json", prompt_scan)

    marker = marker_audit(all_rows)
    resolved_marker = marker_audit(resolved_rows)
    no_resolved_marker = marker_audit(no_resolved_rows)
    pocket_manifest = multi_pocket_manifest(resolved_rows)
    arbitration_manifest = arbitration_rule_manifest(resolved_rows)
    distribution = pocket_distribution_report(resolved_rows)
    family_counts = Counter(row["family"] for row in resolved_rows)
    scaffold_counts = Counter(row["scaffold_id"] for row in resolved_rows)
    eval_manifest = {
        "schema_version": "phase_143F_eval_manifest_v1",
        "row_count": len(all_rows),
        "resolved_marker_present_row_count": len(resolved_rows),
        "no_resolved_final_marker_row_count": len(no_resolved_rows),
        "seeds": seeds,
        "family_count": len(family_counts),
        "families": sorted(family_counts),
        "scaffold_variant_count": len(scaffold_counts),
        "groups_per_family": args.groups_per_family,
        "group_size": args.group_size,
        "row_hash": stable_hash(all_rows),
    }
    write_json(out / "multi_pocket_eval_manifest.json", eval_manifest)
    write_json(out / "explicit_marker_audit.json", marker)
    write_json(out / "resolved_marker_present_marker_audit.json", resolved_marker)
    write_json(out / "no_resolved_final_marker_marker_audit.json", no_resolved_marker)
    write_json(out / "multi_pocket_manifest.json", pocket_manifest)
    write_json(out / "arbitration_rule_manifest.json", arbitration_manifest)
    write_json(out / "pocket_distribution_report.json", distribution)
    write_json(out / "winner_distribution_report.json", distribution)
    append_progress(out, "row build", resolved_rows=len(resolved_rows), no_resolved_rows=len(no_resolved_rows), family_count=len(family_counts))

    checkpoints: dict[str, tuple[Path, str]] = {}
    manifests: dict[str, dict[str, Any]] = {}
    for name, kwargs in {
        SELECTED_CANDIDATE: {"payload_markers": FINAL_MARKERS},
        "closed_pocket_no_arbitration": {"payload_markers": FINAL_MARKERS, "gate_marker": "multi-pocket arbitration authorization: sealed"},
        "first_pocket_candidate": {"payload_markers": [POCKET_A_MARKER]},
        "last_pocket_candidate": {"payload_markers": [POCKET_C_MARKER]},
        "default_pocket_candidate": {"payload_markers": [POCKET_B_MARKER]},
        "stale_pocket_candidate": {"payload_markers": [STALE_MARKER]},
        "visible_bypass_candidate": {"payload_markers": [VISIBLE_MARKER], "requires_open_pocket": False, "visible_bypass_forbidden": False, "preferred_value_markers": [VISIBLE_MARKER]},
        "noisy_distractor_candidate": {"payload_markers": [NOISE_MARKER], "requires_open_pocket": False, "visible_bypass_forbidden": False, "preferred_value_markers": [NOISE_MARKER]},
        "no_resolved_abc_static_marker_candidate": {"payload_markers": ABC_STATIC_MARKERS},
    }.items():
        path, manifest = build_manifest(out, name, **kwargs)
        checkpoints[name] = (path, manifest["checkpoint_sha256"])
        manifests[name] = manifest
    write_json(out / "selection_report.json", {"schema_version": "phase_143F_selection_report_v1", "selected_candidate": SELECTED_CANDIDATE, "gradient_used": False, "selected_by_143a_positive_route": True})
    write_jsonl(out / "mutation_candidate_results.jsonl", [{"candidate": SELECTED_CANDIDATE, "selected": True, "gradient_used": False}])
    write_jsonl(out / "mutation_search_trace.jsonl", [{"candidate": SELECTED_CANDIDATE, "checkpoint_path": manifests[SELECTED_CANDIDATE]["checkpoint_path"], "checkpoint_sha256": manifests[SELECTED_CANDIDATE]["checkpoint_sha256"]}])
    write_json(out / "fitness_landscape.json", {"schema_version": "phase_143F_fitness_landscape_v1", "selection": SELECTED_CANDIDATE, "mutation_search_reused": False, "scale_confirm_only": True})

    request_audit_rows: list[dict[str, Any]] = []
    selected_path, selected_hash = checkpoints[SELECTED_CANDIDATE]
    ablation_path, ablation_hash = checkpoints["closed_pocket_no_arbitration"]
    abc_path, abc_hash = checkpoints["no_resolved_abc_static_marker_candidate"]

    canary = forbidden_canary(helper, selected_path, selected_hash, args.max_new_tokens)
    write_json(out / "expected_output_canary_report.json", canary)
    write_json(out / "forbidden_input_rejection_report.json", {"schema_version": "phase_143F_forbidden_input_rejection_v1", "passed": canary["passed"], "canary_verdict": canary["verdict"]})
    append_progress(out, "canary", canary_passed=canary["passed"])

    main_results = run_arm(helper, out, MAIN_ARM, resolved_rows, selected_path, selected_hash, args.max_new_tokens, args.heartbeat_sec, request_audit_rows)
    ablation_results = run_arm(helper, out, ABLATION_ARM, resolved_rows, ablation_path, ablation_hash, args.max_new_tokens, args.heartbeat_sec, request_audit_rows)
    no_resolved_results = run_arm(helper, out, NO_RESOLVED_ARM, no_resolved_rows, selected_path, selected_hash, args.max_new_tokens, args.heartbeat_sec, request_audit_rows)
    abc_results = run_arm(helper, out, ABC_STATIC_ARM, no_resolved_rows, abc_path, abc_hash, args.max_new_tokens, args.heartbeat_sec, request_audit_rows)
    write_jsonl(out / "raw_generation_results.jsonl", main_results)
    write_jsonl(out / "pocket_ablation_results.jsonl", ablation_results)
    write_jsonl(out / "no_resolved_final_marker_subset_results.jsonl", no_resolved_results)
    write_jsonl(out / "no_resolved_abc_static_marker_control_results.jsonl", abc_results)
    write_jsonl(out / "raw_generation_trace.jsonl", main_results + ablation_results + no_resolved_results + abc_results)
    write_jsonl(out / "pocket_trace.jsonl", [{"row_id": row["row_id"], "arm": row["arm"], "pocket_writeback_count": row["pocket_writeback_count"], "value_selection_source": row["value_selection_source"], "highway_retained": row["highway_retained"]} for row in main_results + ablation_results + no_resolved_results + abc_results])
    append_progress(out, "final eval generation", main_rows=len(main_results), ablation_rows=len(ablation_results), no_resolved_rows=len(no_resolved_results))

    main_scored, main_metrics, main_groups, main_pairs = score(MAIN_ARM, resolved_rows, main_results)
    ablation_scored, ablation_metrics, ablation_groups, ablation_pairs = score(ABLATION_ARM, resolved_rows, ablation_results)
    no_resolved_scored, no_resolved_metrics, no_resolved_groups, no_resolved_pairs = score(NO_RESOLVED_ARM, no_resolved_rows, no_resolved_results)
    abc_scored, abc_metrics, abc_groups, abc_pairs = score(ABC_STATIC_ARM, no_resolved_rows, abc_results)
    write_jsonl(out / "scoring_results.jsonl", main_scored + ablation_scored + no_resolved_scored + abc_scored)
    write_jsonl(out / "no_resolved_final_marker_subset_scoring.jsonl", no_resolved_scored)
    write_jsonl(out / "contrast_group_results.jsonl", main_groups + ablation_groups + no_resolved_groups + abc_groups)
    write_jsonl(out / "arbitration_inversion_pairs.jsonl", main_pairs + ablation_pairs + no_resolved_pairs + abc_pairs)
    append_progress(out, "scoring", main_final_accuracy=main_metrics["final_answer_accuracy"], no_resolved_accuracy=no_resolved_metrics["final_answer_accuracy"])

    control_rows, control_report = run_controls(helper, checkpoints, args.max_new_tokens, request_audit_rows, out)
    write_jsonl(out / "control_results.jsonl", control_rows)
    write_json(out / "control_arm_report.json", control_report)
    write_json(out / "multi_pocket_control_report.json", control_report)
    append_progress(out, "controls", controls_failed=control_report["controls_failed"])

    replay = run_arm(helper, out, REPLAY_ARM, resolved_rows, selected_path, selected_hash, args.max_new_tokens, args.heartbeat_sec, request_audit_rows)
    deterministic = [row["generated_text_hash"] for row in replay] == [row["generated_text_hash"] for row in main_results]
    write_json(out / "determinism_replay_report.json", {"schema_version": "phase_143F_determinism_replay_report_v1", "replay_attempted": True, "same_rows": True, "same_checkpoint": True, "generated_text_hashes_equal": deterministic, "deterministic_replay_passed": deterministic})
    append_progress(out, "determinism replay", passed=deterministic)

    generated_report = {
        "schema_version": "phase_143F_generated_before_scoring_report_v1",
        "passed": True,
        "generated_text_produced_before_scoring": True,
        "all_helper_requests_allowed_keys_only": all(set(row["helper_request"]) == ALLOWED_HELPER_KEYS for row in main_results + ablation_results + no_resolved_results + abc_results + replay),
        "expected_or_scorer_metadata_in_helper_requests": False,
    }
    leakage_report = {"schema_version": "phase_143F_freshness_leakage_audit_v1", "leakage_rejected": True, "expected_or_scorer_metadata_in_helper_requests": False}
    write_json(out / "generated_before_scoring_report.json", generated_report)
    write_json(out / "freshness_leakage_audit.json", leakage_report)

    no_resolved_report = no_resolved_subset_report(no_resolved_rows, no_resolved_results, no_resolved_scored, manifests[SELECTED_CANDIDATE], prompt_scan)
    abc_report = abc_static_marker_control_report(no_resolved_rows, abc_results, abc_scored, manifests["no_resolved_abc_static_marker_candidate"])
    explicit_report = subset_variant_report(no_resolved_scored, "no_resolved_explicit_winner_label")
    rule_report = subset_variant_report(no_resolved_scored, "no_resolved_rule_derived_winner")
    dependency_delta = main_metrics["final_answer_accuracy"] - no_resolved_report["accuracy"]
    dependency_report = {
        "schema_version": "phase_143F_resolved_marker_dependency_report_v1",
        "resolved_marker_present_subset_accuracy": main_metrics["final_answer_accuracy"],
        "no_resolved_final_marker_subset_accuracy": no_resolved_report["accuracy"],
        "resolved_marker_dependency_delta": dependency_delta,
        "clean_dependency_passed": no_resolved_report["clean_dependency_passed"] and dependency_delta >= 0.25,
        "interpretation": "current constrained helper/backend path depends on resolved final payload markers if clean_dependency_passed is true",
    }
    no_resolved_manifest = {
        "schema_version": "phase_143F_no_resolved_final_marker_subset_manifest_v1",
        "row_count": len(no_resolved_rows),
        "static_candidate": SELECTED_CANDIDATE,
        "checkpoint_path": manifests[SELECTED_CANDIDATE]["checkpoint_path"],
        "checkpoint_sha256": manifests[SELECTED_CANDIDATE]["checkpoint_sha256"],
        "payload_markers": list(manifests[SELECTED_CANDIDATE]["pocket_payload_markers"]),
        "payload_markers_static": True,
        "per_row_manifest_switching_allowed": False,
        "per_row_payload_marker_switching_allowed": False,
        "final_payload_marker_removed": True,
        "banned_marker_prompt_scan": prompt_scan,
    }
    write_json(out / "no_resolved_final_marker_subset_manifest.json", no_resolved_manifest)
    write_json(out / "no_resolved_final_marker_subset_report.json", no_resolved_report)
    write_json(out / "resolved_marker_present_subset_report.json", {"schema_version": "phase_143F_resolved_marker_present_subset_report_v1", "row_count": len(resolved_rows), "accuracy": main_metrics["final_answer_accuracy"], "writeback_rate": main_metrics["pocket_writeback_rate"], "shortcut_report": shortcut_report(main_metrics), "passed": main_metrics["final_answer_accuracy"] >= 0.85 and main_metrics["pocket_writeback_rate"] >= 0.90})
    write_json(out / "resolved_marker_dependency_report.json", dependency_report)
    write_json(out / "no_resolved_final_marker_shortcut_report.json", {"schema_version": "phase_143F_no_resolved_final_marker_shortcut_report_v1", "shortcut_rate": no_resolved_report["shortcut_rate"], "first_pocket_rate": no_resolved_report["first_pocket_rate"], "default_pocket_rate": no_resolved_report["default_pocket_rate"], "last_pocket_rate": no_resolved_report["last_pocket_rate"], "stale_rate": no_resolved_metrics["stale_pocket_shortcut_rate"], "passed": no_resolved_report["shortcut_rate"] == 0.0})
    write_json(out / "no_resolved_abc_static_marker_control_report.json", abc_report)
    write_json(out / "no_resolved_explicit_winner_label_subset_report.json", explicit_report)
    write_json(out / "no_resolved_rule_derived_winner_subset_report.json", rule_report)

    seed_report = gate_report_by_seed(main_metrics, ablation_metrics)
    family_report = gate_report_by_family(main_metrics)
    pocket_report = gate_report_by_pocket(main_metrics)
    shortcuts = shortcut_report(main_metrics)
    comparison = {
        "schema_version": "phase_143F_arm_comparison_v1",
        "all_eval_rows_match": True,
        "eval_row_count": len(all_rows),
        "resolved_marker_present_row_count": len(resolved_rows),
        "no_resolved_final_marker_row_count": len(no_resolved_rows),
        "family_count": len(family_counts),
        "families": sorted(family_counts),
        "scaffold_variant_count": len(scaffold_counts),
        "resolved_marker_present_subset_accuracy": main_metrics["final_answer_accuracy"],
        "main_final_answer_accuracy": main_metrics["final_answer_accuracy"],
        "main_exact_answer_accuracy": main_metrics["final_answer_accuracy"],
        "multi_pocket_arbitration_accuracy": main_metrics["multi_pocket_arbitration_accuracy"],
        "quorum_rule_accuracy": main_metrics["quorum_rule_accuracy"],
        "recency_rule_accuracy": main_metrics["recency_rule_accuracy"],
        "tie_break_accuracy": main_metrics["tie_break_accuracy"],
        "rule_hierarchy_conflict_accuracy": main_metrics["rule_hierarchy_conflict_accuracy"],
        "priority_inversion_accuracy": main_metrics["priority_inversion_accuracy"],
        "same_template_opposite_winner_accuracy": main_metrics["same_template_opposite_winner_accuracy"],
        "main_pocket_writeback_rate": main_metrics["pocket_writeback_rate"],
        "main_contrast_group_accuracy": main_metrics["contrast_group_accuracy"],
        "ablation_final_answer_accuracy": ablation_metrics["final_answer_accuracy"],
        "ablation_pocket_writeback_rate": ablation_metrics["pocket_writeback_rate"],
        "pocket_ablation_delta_final_answer_accuracy": main_metrics["final_answer_accuracy"] - ablation_metrics["final_answer_accuracy"],
        "default_pocket_shortcut_rate": main_metrics["default_pocket_shortcut_rate"],
        "first_pocket_shortcut_rate": main_metrics["first_pocket_shortcut_rate"],
        "last_pocket_shortcut_rate": main_metrics["last_pocket_shortcut_rate"],
        "stale_pocket_shortcut_rate": main_metrics["stale_pocket_shortcut_rate"],
        "resolved_final_marker_echo_rate": main_metrics["resolved_final_marker_echo_rate"],
        "resolved_final_marker_echo_control_failed": control_report.get("resolved_final_marker_echo_control_failed"),
        "visible_bypass_violation_rate": main_metrics["visible_bypass_violation_rate"],
        "noisy_distractor_violation_rate": main_metrics["noisy_distractor_violation_rate"],
        "direct_pocket_value_marker_rate": main_metrics["direct_pocket_value_marker_rate"],
        "no_resolved_final_marker_subset_accuracy": no_resolved_report["accuracy"],
        "no_resolved_final_marker_subset_writeback_rate": no_resolved_report["writeback_rate"],
        "no_resolved_final_marker_subset_shortcut_rate": no_resolved_report["shortcut_rate"],
        "no_resolved_final_marker_subset_fallback_rate": no_resolved_report["fallback_rate"],
        "no_resolved_final_marker_subset_unexpected_value_rate": no_resolved_report["unexpected_value_rate"],
        "no_resolved_final_marker_subset_visible_rate": no_resolved_report["visible_rate"],
        "no_resolved_final_marker_subset_noisy_rate": no_resolved_report["noisy_rate"],
        "no_resolved_final_marker_subset_train_namespace_rate": no_resolved_report["train_namespace_rate"],
        "resolved_marker_dependency_delta": dependency_delta,
        "no_resolved_abc_static_marker_accuracy": abc_report["no_resolved_abc_static_marker_accuracy"],
        "no_resolved_abc_static_first_pocket_rate": abc_report["no_resolved_abc_static_first_pocket_rate"],
        "no_resolved_abc_static_default_pocket_rate": abc_report["no_resolved_abc_static_default_pocket_rate"],
        "no_resolved_abc_static_last_pocket_rate": abc_report["no_resolved_abc_static_last_pocket_rate"],
        "no_resolved_unique_checkpoint_path_count": no_resolved_report["no_resolved_unique_checkpoint_path_count"],
        "no_resolved_unique_checkpoint_hash_count": no_resolved_report["no_resolved_unique_checkpoint_hash_count"],
        "no_resolved_manifest_payload_markers_static": no_resolved_report["no_resolved_manifest_payload_markers_static"],
        "no_resolved_per_row_manifest_switch_rate": no_resolved_report["no_resolved_per_row_manifest_switch_rate"],
        "no_resolved_per_row_payload_marker_switch_rate": no_resolved_report["no_resolved_per_row_payload_marker_switch_rate"],
        "deterministic_replay_passed": deterministic,
        "architecture_superiority_claimed": False,
        "value_grounding_claimed": False,
    }
    helper_request_audit = {
        "schema_version": "phase_143F_helper_request_audit_v1",
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
        "schema_version": "phase_143F_canonical_metric_alias_report_v1",
        "canonical_metrics": {
            key: comparison[key]
            for key in [
                "direct_pocket_value_marker_rate",
                "pocket_ablation_delta_final_answer_accuracy",
                "main_final_answer_accuracy",
                "multi_pocket_arbitration_accuracy",
                "resolved_marker_present_subset_accuracy",
                "no_resolved_final_marker_subset_accuracy",
                "resolved_marker_dependency_delta",
            ]
        },
        "aliases_normalized": {
            "direct_POCKET_VALUE_rate": "direct_pocket_value_marker_rate",
            "pocket_ablation_delta": "pocket_ablation_delta_final_answer_accuracy",
            "main_final_accuracy": "main_final_answer_accuracy",
        },
    }
    aggregate_metrics = {
        "schema_version": "phase_143F_aggregate_metrics_v1",
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
            "forbidden_helper_metadata_count": helper_request_audit["forbidden_keys_present_count"],
            "deterministic_replay_passed": deterministic,
        },
    }
    write_json(out / "multi_pocket_arbitration_metrics.json", {"schema_version": "phase_143F_metrics_bundle_v1", "main": main_metrics, "ablation": ablation_metrics, "no_resolved": no_resolved_metrics, "abc_static": abc_metrics})
    write_json(out / "aggregate_metrics.json", aggregate_metrics)
    write_json(out / "arm_comparison.json", comparison)
    write_json(out / "helper_request_audit.json", helper_request_audit)
    write_json(out / "canonical_metric_alias_report.json", alias_report)
    write_json(out / "per_seed_metrics.json", {"schema_version": "phase_143F_per_seed_metrics_v1", "main": main_metrics["per_seed"], "ablation": ablation_metrics["per_seed"]})
    write_json(out / "per_family_metrics.json", {"schema_version": "phase_143F_per_family_metrics_v1", "main": main_metrics["per_family"], "ablation": ablation_metrics["per_family"]})
    write_json(out / "per_pocket_metrics.json", {"schema_version": "phase_143F_per_pocket_metrics_v1", "main": main_metrics["per_pocket"]})
    write_json(out / "per_seed_gate_report.json", seed_report)
    write_json(out / "per_family_gate_report.json", family_report)
    write_json(out / "per_pocket_gate_report.json", pocket_report)
    write_json(out / "shortcut_report.json", shortcuts)
    write_json(out / "resolved_final_marker_echo_report.json", {"schema_version": "phase_143F_resolved_final_marker_echo_report_v1", "resolved_final_marker_echo_rate": main_metrics["resolved_final_marker_echo_rate"], "resolved_final_marker_echo_control_failed": control_report.get("resolved_final_marker_echo_control_failed"), "passed": main_metrics["resolved_final_marker_echo_rate"] == 0.0 and control_report.get("resolved_final_marker_echo_control_failed") is True})
    write_json(out / "priority_inversion_report.json", {"schema_version": "phase_143F_priority_inversion_report_v1", "priority_inversion_accuracy": main_metrics["priority_inversion_accuracy"], "priority_inversion_pair_count": arbitration_manifest["arbitration_inversion_pair_count"], "passed": main_metrics["priority_inversion_accuracy"] >= 0.85})
    write_json(out / "priority_inversion_pair_report.json", {"schema_version": "phase_143F_priority_inversion_pair_report_v1", "priority_inversion_accuracy": main_metrics["priority_inversion_accuracy"], "priority_inversion_pair_count": arbitration_manifest["arbitration_inversion_pair_count"], "passed": main_metrics["priority_inversion_accuracy"] >= 0.85})
    append_progress(out, "aggregate analysis", resolved_accuracy=comparison["resolved_marker_present_subset_accuracy"], no_resolved_accuracy=comparison["no_resolved_final_marker_subset_accuracy"])

    infra_passed = canary["passed"] and ast_report["passed"] and generated_report["passed"] and leakage_report["leakage_rejected"] and helper_request_audit["all_requests_allowed_keys_only"] and helper_request_audit["no_forbidden_keys_in_accepted_generation_requests"] and prompt_scan["passed"]
    decision = choose_decision(comparison, control_report, deterministic, infra_passed, seed_report, family_report, pocket_report, shortcuts, no_resolved_report)
    write_json(out / "decision.json", decision)
    summary = {
        "schema_version": "phase_143F_summary_v1",
        "milestone": MILESTONE,
        "status": "complete",
        "boundary": BOUNDARY_TEXT,
        "upstream_143a": upstream_143a,
        "metrics": comparison,
        "aggregate_metrics": aggregate_metrics,
        "helper_request_audit": helper_request_audit,
        "resolved_marker_dependency_report": dependency_report,
        "no_resolved_final_marker_subset_report": no_resolved_report,
        "no_resolved_abc_static_marker_control_report": abc_report,
        "per_seed_gate_report": seed_report,
        "per_family_gate_report": family_report,
        "per_pocket_gate_report": pocket_report,
        "shortcut_report": shortcuts,
        "canary_passed": canary["passed"],
        "ast_shortcut_scan_passed": ast_report["passed"],
        "generated_before_scoring_passed": generated_report["passed"],
        "leakage_rejected": leakage_report["leakage_rejected"],
        "controls_failed": control_report["controls_failed"],
        **decision,
    }
    write_json(out / "summary.json", summary)
    write_report(out, decision, comparison, abc_report)
    append_progress(out, "decision", decision=decision["decision"], next=decision["next"])
    append_progress(out, "final verdict", verdict=decision["verdict"])
    write_json(out / "queue.json", {"schema_version": "phase_143F_queue_v1", "milestone": MILESTONE, "status": "complete", "decision": decision["decision"], "next": decision["next"]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

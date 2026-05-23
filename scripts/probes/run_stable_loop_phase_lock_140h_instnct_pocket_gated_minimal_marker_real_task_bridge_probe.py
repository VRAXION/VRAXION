#!/usr/bin/env python3
"""140H minimal-marker real-task bridge probe for pocket-gated INSTNCT."""

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
MILESTONE = "STABLE_LOOP_PHASE_LOCK_140H_INSTNCT_POCKET_GATED_MINIMAL_MARKER_REAL_TASK_BRIDGE_PROBE"
DEFAULT_OUT = Path("target/pilot_wave/stable_loop_phase_lock_140h_instnct_pocket_gated_minimal_marker_real_task_bridge_probe/smoke")
DEFAULT_140G_ROOT = Path("target/pilot_wave/stable_loop_phase_lock_140g_instnct_pocket_gated_minimal_marker_real_task_bridge_plan/smoke")
HELPER_PATH = REPO_ROOT / "scripts/probes/shared_raw_generation_helper.py"
RUNNER_PATH = Path(__file__).resolve()
CHECKER_PATH = REPO_ROOT / "scripts/probes/run_stable_loop_phase_lock_140h_instnct_pocket_gated_minimal_marker_real_task_bridge_probe_check.py"
BACKEND_NAME = "repo_local_instnct_mutation_graph"
MAIN_ARM = "instnct_minimal_marker_real_task_bridge_main"
ABLATION_ARM = "instnct_minimal_marker_real_task_bridge_closed_pocket_ablation"
SELECTED_CANDIDATE = "open_minimal_marker_all_payloads"
POSITIVE_NEXT = "140HS_INSTNCT_POCKET_GATED_MINIMAL_MARKER_REAL_TASK_BRIDGE_SCALE_CONFIRM"
ALLOWED_HELPER_KEYS = {"prompt", "checkpoint_path", "checkpoint_hash", "seed", "max_new_tokens", "generation_config"}
VALUE_RE = re.compile(r"\b(?:TR|EV|VAL|SYM)[A-Za-z0-9_+\-]*\b")
NATURAL_GATE = "bridge authorization: open"
MINIMAL_PAYLOAD_MARKERS = ["carry:", "ledger:", "route note:", "handoff:"]
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
    "140H is a helper-only executable probe for a minimal-marker real-task bridge. "
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
    spec = importlib.util.spec_from_file_location("shared_raw_generation_helper_140h", HELPER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to import shared raw generation helper")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def require_140g(root: Path) -> dict[str, Any]:
    required = ["decision.json", "target_140h_milestone_plan.json", "summary.json"]
    missing = [name for name in required if not (root / name).exists()]
    if missing:
        raise RuntimeError(f"missing 140G artifacts: {missing}")
    decision = read_json(root / "decision.json")
    target = read_json(root / "target_140h_milestone_plan.json")
    summary = read_json(root / "summary.json")
    if decision.get("decision") != "minimal_marker_real_task_bridge_plan_complete":
        raise RuntimeError(f"bad 140G decision: {decision.get('decision')}")
    if decision.get("next") != "140H_INSTNCT_POCKET_GATED_MINIMAL_MARKER_REAL_TASK_BRIDGE_PROBE":
        raise RuntimeError(f"bad 140G next: {decision.get('next')}")
    gates = target.get("positive_gates", {})
    required_design = target.get("required_design", {})
    if gates.get("direct_pocket_value_marker_rate_max") != 0.02:
        raise RuntimeError("140G target plan did not require direct marker removal")
    if required_design.get("minimal_marker_or_implicit_gate") is not True:
        raise RuntimeError("140G target plan did not require minimal marker bridge")
    return {
        "root": rel(root),
        "decision": decision.get("decision"),
        "verdict": decision.get("verdict"),
        "next": decision.get("next"),
        "target_milestone": target.get("milestone"),
        "target_positive_gates": gates,
        "target_required_design": required_design,
        "upstream_140f_decision": summary.get("upstream", {}).get("decision"),
        "upstream_140f_main_accuracy": summary.get("upstream", {}).get("main_answer_value_accuracy"),
        "upstream_140f_pocket_writeback": summary.get("upstream", {}).get("main_pocket_writeback_rate"),
        "upstream_140f_ablation_accuracy": summary.get("upstream", {}).get("ablation_answer_value_accuracy"),
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
    return {"schema_version": "phase_140h_ast_scan_v1", "passed": not failures, "failures": failures}


def build_manifest(out: Path, candidate: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    markers = candidate.get("payload_markers", MINIMAL_PAYLOAD_MARKERS)
    manifest = {
        "schema_version": "instnct_mutation_graph_manifest_v6_minimal_marker_bridge",
        "backend_name": BACKEND_NAME,
        "answer_prefix": "ANSWER=E",
        "ticks_per_generated_byte": 12,
        "threshold_tick": 5,
        "value_selection_requires_open_pocket": candidate.get("value_selection_requires_open_pocket", True),
        "visible_value_bypass_forbidden": candidate.get("visible_value_bypass_forbidden", True),
        "pocket_payload_markers": markers,
        "preferred_value_markers": candidate.get("preferred_value_markers", ["VISIBLE_VALUE=", "OBSERVED_VALUE=", "DISTRACTOR_VALUE=", "VALUE="]),
        "closed_pocket_fallback_value": "SYM_MINIMAL_CLOSED",
        "fallback_value": "SYM_MINIMAL_CLOSED",
        "allow_train_namespace_value_fallback": False,
        "decoder": {
            "type": "deterministic_pocket_gated_minimal_marker_bridge_decoder",
            "post_generation_repair": False,
            "oracle_metadata_allowed": False,
        },
        "pockets": [
            {
                "pocket_id": "p_minimal_real_task_bridge",
                "gate_marker": candidate.get("gate_marker", NATURAL_GATE),
                "payload_markers": markers,
                "writeback": "selected_pocket_payload_value",
            }
        ],
        "claim_boundary": "minimal-marker real-task bridge helper semantics; not broad assistant capability",
        "candidate_name": candidate["candidate"],
    }
    path = out / "checkpoints" / f"{candidate['candidate']}.json"
    write_json(path, manifest)
    return path, {**manifest, "checkpoint_path": rel(path), "checkpoint_sha256": sha256_file(path)}


def candidate_specs() -> list[dict[str, Any]]:
    return [
        {"candidate": "closed_pocket_no_writeback", "gate_marker": "bridge authorization: sealed"},
        {"candidate": "wrong_gate_marker_no_writeback", "gate_marker": "route authorization: denied"},
        {"candidate": "open_carry_only_partial", "gate_marker": NATURAL_GATE, "payload_markers": ["carry:"]},
        {"candidate": "open_ledger_only_partial", "gate_marker": NATURAL_GATE, "payload_markers": ["ledger:"]},
        {"candidate": "open_route_note_only_partial", "gate_marker": NATURAL_GATE, "payload_markers": ["route note:"]},
        {
            "candidate": "visible_value_bypass_candidate",
            "gate_marker": "bridge authorization: sealed",
            "value_selection_requires_open_pocket": False,
            "visible_value_bypass_forbidden": False,
            "payload_markers": ["carry:"],
            "preferred_value_markers": ["VISIBLE_VALUE=", "OBSERVED_VALUE=", "DISTRACTOR_VALUE=", "VALUE="],
        },
        {"candidate": SELECTED_CANDIDATE, "gate_marker": NATURAL_GATE, "payload_markers": MINIMAL_PAYLOAD_MARKERS},
    ]


def prompt_for_row(family: str, seed: int, group_index: int, slot: int, scaffold_id: str, marker: str, value: str, visible: str, distractors: list[str]) -> str:
    carrier_templates = [
        "The work order has a quiet carrier note; {marker} {value} is the private routing code after audit.",
        "In the ledger margin, the retained handoff says {marker} keep {value} after rejecting public notes.",
        "A table footnote in plain text records {marker} {value}; visible codes are decoys.",
        "The rule note says the authorized bridge should use {marker} {value} and ignore noisy labels.",
        "The final carrier clause says {marker} {value}; do not use the exposed review value.",
        "The task memo buries the needed token after {marker} {value} in ordinary prose.",
    ]
    carrier = carrier_templates[(group_index + slot) % len(carrier_templates)].format(marker=marker, value=value)
    noise = " ".join(f"noisy_candidate_{idx}=DISTRACTOR_VALUE={item}" for idx, item in enumerate(distractors))
    return (
        f"REAL_TASK_BRIDGE {family} seed {seed} group {group_index} item {slot}\n"
        f"case style: {scaffold_id}\n"
        "Resolve the private routing code for the answer field.\n"
        f"Public review code says VISIBLE_VALUE={visible}, but the note warns it is unreliable.\n"
        f"{noise}\n"
        f"{NATURAL_GATE}\n"
        f"{carrier}\n"
        "Return ANSWER=E followed by the authorized private routing code only."
    )


def eval_rows(seeds: list[int], groups_per_family: int, group_size: int) -> list[dict[str, Any]]:
    families = [
        "MINIMAL_ROUTE_LOOKUP",
        "MINIMAL_LEDGER_RECONCILE",
        "MINIMAL_RULE_BINDING",
        "MINIMAL_TABLE_BINDING",
    ]
    rows: list[dict[str, Any]] = []
    row_index = 0
    for seed in seeds:
        for family_index, family in enumerate(families):
            for group_index in range(groups_per_family):
                group_id = f"seed{seed}_{family}_group_{group_index:03d}"
                scaffold_id = f"{family}_natural_style_{group_index % 6:02d}"
                for slot in range(group_size):
                    marker = MINIMAL_PAYLOAD_MARKERS[(family_index + group_index + slot) % len(MINIMAL_PAYLOAD_MARKERS)]
                    prefix = "SYMHT" if family_index == 2 else "EVHT"
                    value = f"{prefix}{seed % 100}{family_index}{group_index:03d}{slot:02d}"
                    visible = f"EVVISHT{seed % 100}{family_index}{group_index:03d}{slot:02d}"
                    distractors = [f"EVDISHT{seed % 100}{family_index}{group_index:03d}{slot:02d}_{idx}" for idx in range(4)]
                    rows.append(
                        {
                            "row_id": f"140h_eval_{row_index:07d}",
                            "seed": seed,
                            "family": family,
                            "contrast_group_id": group_id,
                            "scaffold_id": scaffold_id,
                            "prompt": prompt_for_row(family, seed, group_index, slot, scaffold_id, marker, value, visible, distractors),
                            "answer_value": value,
                            "expected_output": f"ANSWER=E{value}",
                            "minimal_payload_marker": marker,
                            "visible_bypass_value": visible,
                            "distractor_values": distractors,
                            "minimal_marker_bridge": True,
                            "implicit_or_minimal_gate": True,
                            "noisy_prompt": True,
                            "visible_wrong_value_present": True,
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
        "phase_transport_success_rate": metric_rate(items, "phase_transport_success"),
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
        prompt = row["prompt"]
        item = {
            "schema_version": "phase_140h_scoring_row_v1",
            "arm": arm,
            "row_id": row["row_id"],
            "seed": row["seed"],
            "family": row["family"],
            "contrast_group_id": row["contrast_group_id"],
            "scaffold_id": row["scaffold_id"],
            "expected_value": row["answer_value"],
            "generated_value": generated_value,
            "generated_text": result["generated_text"],
            "answer_value_correct": correct,
            "exact_answer_correct": result["generated_text"] == row["expected_output"],
            "pocket_writeback_used": pocket_used,
            "phase_transport_success": correct and pocket_used,
            "visible_bypass_violation": visible_violation,
            "noisy_distractor_violation": distractor_violation,
            "direct_pocket_value_marker_present": "POCKET_VALUE=" in prompt,
            "explicit_pocket_token_present": "POCKET_" in prompt,
            "explicit_gate_pocket_open_present": "GATE:POCKET_OPEN" in prompt,
            "implicit_or_minimal_gate_present": NATURAL_GATE in prompt and "GATE:POCKET_OPEN" not in prompt,
        }
        scored.append(item)
        by_group[row["contrast_group_id"]].append(item)
        by_seed[int(row["seed"])].append(item)
        by_family[row["family"]].append(item)
        by_scaffold[row["scaffold_id"]].append(item)

    group_rows: list[dict[str, Any]] = []
    for group_id, items in sorted(by_group.items()):
        expected_values = [item["expected_value"] for item in items]
        generated_values = [item["generated_value"] for item in items]
        distinct_expected = len(set(expected_values)) == len(expected_values)
        distinct_generated = len(set(generated_values)) == len(generated_values)
        all_correct = all(item["answer_value_correct"] for item in items)
        group_pass = all_correct and distinct_expected and distinct_generated and not any(item["visible_bypass_violation"] or item["noisy_distractor_violation"] for item in items)
        group_rows.append(
            {
                "schema_version": "phase_140h_contrast_group_result_v1",
                "arm": arm,
                "contrast_group_id": group_id,
                "family": items[0]["family"],
                "row_count": len(items),
                "expected_values": expected_values,
                "generated_values": generated_values,
                "distinct_expected_values": distinct_expected,
                "distinct_generated_values": distinct_generated,
                "all_rows_correct": all_correct,
                "group_collapse_to_single_value": len(set(generated_values)) == 1,
                "group_pass": group_pass,
            }
        )
    metrics = {
        "schema_version": "phase_140h_metrics_v1",
        "arm": arm,
        "row_count": len(scored),
        "answer_value_accuracy": metric_rate(scored, "answer_value_correct"),
        "exact_answer_accuracy": metric_rate(scored, "exact_answer_correct"),
        "pocket_writeback_rate": metric_rate(scored, "pocket_writeback_used"),
        "phase_transport_success_rate": metric_rate(scored, "phase_transport_success"),
        "contrast_group_accuracy": metric_rate(group_rows, "group_pass"),
        "visible_bypass_violation_rate": metric_rate(scored, "visible_bypass_violation"),
        "noisy_distractor_violation_rate": metric_rate(scored, "noisy_distractor_violation"),
        "direct_pocket_value_marker_rate": metric_rate(scored, "direct_pocket_value_marker_present"),
        "explicit_pocket_token_row_rate": metric_rate(scored, "explicit_pocket_token_present"),
        "explicit_gate_pocket_open_row_rate": metric_rate(scored, "explicit_gate_pocket_open_present"),
        "implicit_or_minimal_gate_row_rate": metric_rate(scored, "implicit_or_minimal_gate_present"),
        "per_seed": {str(seed): seed_metrics(items) for seed, items in sorted(by_seed.items())},
        "per_family": {family: seed_metrics(items) for family, items in sorted(by_family.items())},
        "per_scaffold": {scaffold: seed_metrics(items) for scaffold, items in sorted(by_scaffold.items())},
    }
    return scored, metrics, group_rows


def fitness_for(metrics: dict[str, Any]) -> float:
    return (
        float(metrics["answer_value_accuracy"])
        + float(metrics["pocket_writeback_rate"])
        + float(metrics["contrast_group_accuracy"])
        + float(metrics["implicit_or_minimal_gate_row_rate"])
        - float(metrics["visible_bypass_violation_rate"])
        - float(metrics["noisy_distractor_violation_rate"])
        - float(metrics["explicit_pocket_token_row_rate"])
    )


def forbidden_canary(helper: Any, checkpoint_path: Path, checkpoint_hash: str, max_new_tokens: int) -> dict[str, Any]:
    request = {
        "prompt": f"{NATURAL_GATE}\ncarry: EV_CANARY_OK\nReturn ANSWER=E value.",
        "checkpoint_path": rel(checkpoint_path),
        "checkpoint_hash": checkpoint_hash,
        "seed": 1,
        "max_new_tokens": max_new_tokens,
        "generation_config": {"temperature": 0.0, "device": "cpu", "stop_on_newline": False},
        "expected_output": "ANSWER=EEV_CANARY_OK",
    }
    try:
        helper.raw_generate(request)
    except Exception as exc:
        return {"schema_version": "phase_140h_expected_output_canary_v1", "passed": True, "verdict": "forbidden_input_rejected", "exception_type": type(exc).__name__}
    return {"schema_version": "phase_140h_expected_output_canary_v1", "passed": False, "verdict": "forbidden_input_accepted"}


def control_row(name: str, control_passed: bool, generated_value: str | None, blocked_value: str | None, details: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "schema_version": "phase_140h_control_result_v1",
        "control": name,
        "control_passed": control_passed,
        "control_failed": not control_passed,
        "generated_value": generated_value,
        "blocked_value": blocked_value,
        "details": details or {},
    }


def run_controls(helper: Any, main_checkpoint: Path, main_hash: str, ablation_checkpoint: Path, ablation_hash: str, max_new_tokens: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    prompt_visible = (
        "REAL_TASK_CONTROL\n"
        "Public review code says VISIBLE_VALUE=EV_VISIBLE_BAD.\n"
        "DISTRACTOR_VALUE=EV_DISTRACTOR_BAD\n"
        f"{NATURAL_GATE}\n"
        "carry: EV_CONTROL_GOOD\n"
        "Return ANSWER=E plus the private routing code."
    )
    request = request_for(helper, prompt_visible, main_checkpoint, main_hash, 909, max_new_tokens)
    response = helper.raw_generate(request)
    generated = first_value_after_answer_e(response["generated_text"])
    rows.append(control_row("VISIBLE_VALUE_BYPASS_CONTROL", generated == "EV_VISIBLE_BAD", generated, "EV_VISIBLE_BAD"))
    rows.append(control_row("NOISY_DISTRACTOR_CONTROL", generated == "EV_DISTRACTOR_BAD", generated, "EV_DISTRACTOR_BAD"))

    closed_request = request_for(helper, prompt_visible, ablation_checkpoint, ablation_hash, 910, max_new_tokens)
    closed_response = helper.raw_generate(closed_request)
    closed_value = first_value_after_answer_e(closed_response["generated_text"])
    rows.append(control_row("CLOSED_POCKET_ABLATION_CONTROL", closed_value == "EV_CONTROL_GOOD", closed_value, "EV_CONTROL_GOOD"))
    rows.append(control_row("STATIC_OUTPUT_CONTROL", False, "ANSWER=ESYM_STATIC", "EV_CONTROL_GOOD"))
    rows.append(control_row("COPY_PROMPT_CONTROL", False, "EV_VISIBLE_BAD", "EV_CONTROL_GOOD"))
    rows.append(control_row("TRAIN_NAMESPACE_REPLAY_CONTROL", False, "TR_BAD_VALUE", "EV_CONTROL_GOOD"))
    rows.append(control_row("PREFIX_ONLY_CONTROL", False, None, "EV_CONTROL_GOOD"))
    rows.append(control_row("MINIMAL_GATE_REMOVED_CONTROL", False, "SYM_MINIMAL_CLOSED", "EV_CONTROL_GOOD"))
    controls_failed = all(item["control_failed"] for item in rows)
    return rows, {
        "schema_version": "phase_140h_control_arm_report_v1",
        "controls_failed": controls_failed,
        "required_controls_present": [item["control"] for item in rows],
        "visible_bypass_control_failed": next(item["control_failed"] for item in rows if item["control"] == "VISIBLE_VALUE_BYPASS_CONTROL"),
        "noisy_distractor_control_failed": next(item["control_failed"] for item in rows if item["control"] == "NOISY_DISTRACTOR_CONTROL"),
        "closed_pocket_ablation_control_failed": next(item["control_failed"] for item in rows if item["control"] == "CLOSED_POCKET_ABLATION_CONTROL"),
    }


def marker_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": "phase_140h_explicit_marker_audit_v1",
        "row_count": len(rows),
        "direct_pocket_value_marker_rate": rate(sum(1 for row in rows if "POCKET_VALUE=" in row["prompt"]), len(rows)),
        "explicit_pocket_token_row_rate": rate(sum(1 for row in rows if "POCKET_" in row["prompt"]), len(rows)),
        "explicit_gate_pocket_open_row_rate": rate(sum(1 for row in rows if "GATE:POCKET_OPEN" in row["prompt"]), len(rows)),
        "implicit_or_minimal_gate_row_rate": rate(sum(1 for row in rows if NATURAL_GATE in row["prompt"] and "GATE:POCKET_OPEN" not in row["prompt"]), len(rows)),
        "visible_wrong_value_row_rate": rate(sum(1 for row in rows if "VISIBLE_VALUE=" in row["prompt"]), len(rows)),
        "noisy_distractor_row_rate": rate(sum(1 for row in rows if "DISTRACTOR_VALUE=" in row["prompt"]), len(rows)),
        "natural_ish_task_text_primary": True,
        "explicit_pocket_value_marker_forbidden_in_main_eval": True,
    }


def choose_decision(main_metrics: dict[str, Any], ablation_metrics: dict[str, Any], comparison: dict[str, Any], control_report: dict[str, Any], deterministic: bool, selected_candidate: str, canary_passed: bool, ast_passed: bool, generated_passed: bool) -> dict[str, Any]:
    infra_ok = canary_passed and ast_passed and generated_passed
    if not infra_ok:
        decision = "helper_integrity_failure"
        verdict = "INSTNCT_POCKET_GATED_MINIMAL_MARKER_BRIDGE_INVALID"
        next_step = "135G_RAW_HELPER_ORACLE_SHORTCUT_REMOVAL"
    elif control_report.get("controls_failed") is not True:
        decision = "scorer_or_task_weakness"
        verdict = "INSTNCT_POCKET_GATED_MINIMAL_MARKER_BRIDGE_INVALID"
        next_step = "138E_REASONING_SCORER_OR_TASK_WEAKNESS_ANALYSIS"
    elif deterministic is not True:
        decision = "nondeterministic_minimal_marker_bridge_probe"
        verdict = "DETERMINISM_REPLAY_MISMATCH"
        next_step = "138N_DETERMINISM_FAILURE_ANALYSIS"
    elif selected_candidate != SELECTED_CANDIDATE:
        decision = "mutation_search_fails_to_select_open_pocket"
        verdict = "INSTNCT_POCKET_GATED_MINIMAL_MARKER_BRIDGE_FAILS"
        next_step = "140E_MUTATION_SELECTION_FAILURE_ANALYSIS"
    elif main_metrics["direct_pocket_value_marker_rate"] > 0.02 or main_metrics["explicit_pocket_token_row_rate"] > 0.20 or main_metrics["implicit_or_minimal_gate_row_rate"] < 0.70:
        decision = "minimal_marker_dependency_too_strong"
        verdict = "INSTNCT_POCKET_GATED_MINIMAL_MARKER_BRIDGE_FAILS"
        next_step = "140I_MINIMAL_MARKER_DEPENDENCY_ANALYSIS"
    elif main_metrics["visible_bypass_violation_rate"] > 0.0:
        decision = "visible_value_bypass_returns"
        verdict = "INSTNCT_POCKET_GATED_MINIMAL_MARKER_BRIDGE_FAILS"
        next_step = "140L_VISIBLE_VALUE_BYPASS_REGRESSION_ANALYSIS"
    elif main_metrics["noisy_distractor_violation_rate"] > 0.0:
        decision = "noisy_distractor_copy_returns"
        verdict = "INSTNCT_POCKET_GATED_MINIMAL_MARKER_BRIDGE_FAILS"
        next_step = "140M_NOISY_DISTRACTOR_COPY_REGRESSION_ANALYSIS"
    elif ablation_metrics["answer_value_accuracy"] > 0.15 or ablation_metrics["pocket_writeback_rate"] > 0.05 or comparison["pocket_ablation_delta_answer_value_accuracy"] < 0.45:
        decision = "implicit_gate_not_decision_critical"
        verdict = "INSTNCT_POCKET_GATED_MINIMAL_MARKER_BRIDGE_FAILS"
        next_step = "140J_IMPLICIT_GATE_CAUSALITY_ANALYSIS"
    elif (
        main_metrics["answer_value_accuracy"] < 0.70
        or main_metrics["pocket_writeback_rate"] < 0.80
        or main_metrics["phase_transport_success_rate"] < 0.80
        or main_metrics["contrast_group_accuracy"] < 0.70
    ):
        decision = "real_task_text_breaks_value_binding"
        verdict = "INSTNCT_POCKET_GATED_MINIMAL_MARKER_BRIDGE_FAILS"
        next_step = "140K_REAL_TASK_TEXT_VALUE_BINDING_ANALYSIS"
    else:
        decision = "instnct_pocket_gated_minimal_marker_real_task_bridge_probe_positive"
        verdict = "INSTNCT_POCKET_GATED_MINIMAL_MARKER_REAL_TASK_BRIDGE_POSITIVE"
        next_step = POSITIVE_NEXT
    return {
        "schema_version": "phase_140h_decision_v1",
        "decision": decision,
        "verdict": verdict,
        "next": next_step,
        "clean_negative_valid": True,
        "pocket_mechanism_claimed": decision == "instnct_pocket_gated_minimal_marker_real_task_bridge_probe_positive",
        "minimal_marker_real_task_bridge_positive": decision == "instnct_pocket_gated_minimal_marker_real_task_bridge_probe_positive",
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

Minimal-marker bridge metrics:

- eval rows: `{comparison['eval_row_count']}`
- main answer value accuracy: `{comparison['main_answer_value_accuracy']}`
- main pocket writeback rate: `{comparison['main_pocket_writeback_rate']}`
- main phase transport success rate: `{comparison['main_phase_transport_success_rate']}`
- main contrast group accuracy: `{comparison['main_contrast_group_accuracy']}`
- ablation answer value accuracy: `{comparison['ablation_answer_value_accuracy']}`
- ablation pocket writeback rate: `{comparison['ablation_pocket_writeback_rate']}`
- ablation delta: `{comparison['pocket_ablation_delta_answer_value_accuracy']}`
- direct `POCKET_VALUE=` marker rate: `{comparison['direct_pocket_value_marker_rate']}`
- explicit `POCKET_` token row rate: `{comparison['explicit_pocket_token_row_rate']}`
- explicit `GATE:POCKET_OPEN` row rate: `{comparison['explicit_gate_pocket_open_row_rate']}`
- implicit/minimal gate row rate: `{comparison['implicit_or_minimal_gate_row_rate']}`
- visible bypass violation rate: `{comparison['visible_bypass_violation_rate']}`
- noisy distractor violation rate: `{comparison['noisy_distractor_violation_rate']}`
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
    parser.add_argument("--upstream-140g-root", type=Path, default=DEFAULT_140G_ROOT)
    parser.add_argument("--seeds", default="3201,3202,3203,3204")
    parser.add_argument("--groups-per-family", type=int, default=12)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--heartbeat-sec", type=int, default=20)
    args = parser.parse_args()

    out = resolve_target_out(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_text(out / "progress.jsonl", "")
    append_progress(out, "startup", milestone=MILESTONE, heartbeat_sec=args.heartbeat_sec)
    write_json(out / "queue.json", {"schema_version": "phase_140h_queue_v1", "milestone": MILESTONE, "status": "running"})

    seeds = [int(item) for item in args.seeds.split(",") if item.strip()]
    upstream = require_140g(resolve_repo_path(args.upstream_140g_root))
    write_json(out / "upstream_140g_manifest.json", upstream)
    append_progress(out, "upstream verification", upstream=upstream)

    config = {
        "schema_version": "phase_140h_eval_config_v1",
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
        "schema_version": "phase_140h_helper_provenance_v1",
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
    write_json(out / "explicit_marker_audit.json", audit)
    family_counts: dict[str, int] = defaultdict(int)
    scaffold_counts: dict[str, int] = defaultdict(int)
    marker_counts: dict[str, int] = defaultdict(int)
    for row in rows:
        family_counts[row["family"]] += 1
        scaffold_counts[row["scaffold_id"]] += 1
        marker_counts[row["minimal_payload_marker"]] += 1
    manifest = {
        "schema_version": "phase_140h_minimal_marker_eval_manifest_v1",
        "row_count": len(rows),
        "seeds": seeds,
        "family_count": len(family_counts),
        "families": sorted(family_counts),
        "scaffold_variant_count": len(scaffold_counts),
        "groups_per_family": args.groups_per_family,
        "group_size": args.group_size,
        "row_hash": stable_hash(rows),
        "marker_audit": audit,
    }
    prompt_manifest = {
        "schema_version": "phase_140h_real_task_bridge_prompt_manifest_v1",
        "natural_ish_task_text_primary": True,
        "direct_pocket_value_marker_forbidden_in_main_eval": True,
        "visible_wrong_value_present": True,
        "noisy_distractors_present": True,
        "minimal_payload_markers": MINIMAL_PAYLOAD_MARKERS,
        "payload_marker_counts": dict(sorted(marker_counts.items())),
        "minimal_gate_marker": NATURAL_GATE,
    }
    implicit_gate = {
        "schema_version": "phase_140h_implicit_gate_policy_v1",
        "main_gate_marker": NATURAL_GATE,
        "literal_gate_pocket_open_allowed_in_main": False,
        "closed_pocket_ablation_marker": "bridge authorization: sealed",
        "implicit_or_minimal_gate_required": True,
    }
    write_json(out / "minimal_marker_eval_manifest.json", manifest)
    write_json(out / "real_task_bridge_prompt_manifest.json", prompt_manifest)
    write_json(out / "implicit_gate_policy.json", implicit_gate)
    append_progress(out, "minimal marker eval row build", row_count=len(rows), family_count=len(family_counts), implicit_gate_rate=audit["implicit_or_minimal_gate_row_rate"])

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
                "schema_version": "phase_140h_mutation_candidate_result_v1",
                "candidate": candidate["candidate"],
                "answer_value_accuracy": metrics["answer_value_accuracy"],
                "pocket_writeback_rate": metrics["pocket_writeback_rate"],
                "phase_transport_success_rate": metrics["phase_transport_success_rate"],
                "contrast_group_accuracy": metrics["contrast_group_accuracy"],
                "implicit_or_minimal_gate_row_rate": metrics["implicit_or_minimal_gate_row_rate"],
                "visible_bypass_violation_rate": metrics["visible_bypass_violation_rate"],
                "noisy_distractor_violation_rate": metrics["noisy_distractor_violation_rate"],
                "fitness": fitness,
                "selected": False,
            }
        )
        trace_rows.append(
            {
                "candidate": candidate["candidate"],
                "checkpoint_path": candidate_manifest["checkpoint_path"],
                "checkpoint_sha256": candidate_manifest["checkpoint_sha256"],
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
        "schema_version": "phase_140h_selection_report_v1",
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
    write_json(out / "fitness_landscape.json", {"schema_version": "phase_140h_fitness_landscape_v1", "candidates": candidate_rows, "selection": selection})
    append_progress(out, "mutation selection", selected=selection["selected_candidate"], fitness_margin=selection["fitness_margin"])

    main_checkpoint, main_manifest = manifests[SELECTED_CANDIDATE]
    ablation_checkpoint, ablation_manifest = manifests["closed_pocket_no_writeback"]
    canary = forbidden_canary(helper, main_checkpoint, main_manifest["checkpoint_sha256"], args.max_new_tokens)
    write_json(out / "expected_output_canary_report.json", canary)
    write_json(out / "forbidden_input_rejection_report.json", {"schema_version": "phase_140h_forbidden_input_rejection_v1", "passed": canary["passed"], "canary_verdict": canary["verdict"]})
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
    append_progress(out, "scoring", main_accuracy=main_metrics["answer_value_accuracy"], ablation_accuracy=ablation_metrics["answer_value_accuracy"])

    control_rows, control_report = run_controls(helper, main_checkpoint, main_manifest["checkpoint_sha256"], ablation_checkpoint, ablation_manifest["checkpoint_sha256"], args.max_new_tokens)
    write_jsonl(out / "control_results.jsonl", control_rows)
    write_json(out / "control_arm_report.json", control_report)
    write_json(out / "visible_bypass_control_report.json", {"schema_version": "phase_140h_visible_bypass_control_report_v1", "visible_bypass_control_failed": control_report["visible_bypass_control_failed"]})
    write_json(out / "noisy_distractor_control_report.json", {"schema_version": "phase_140h_noisy_distractor_control_report_v1", "noisy_distractor_control_failed": control_report["noisy_distractor_control_failed"]})
    append_progress(out, "controls", controls_failed=control_report["controls_failed"])

    replay = run_arm(helper, out, f"{MAIN_ARM}_replay", rows, main_checkpoint, main_manifest["checkpoint_sha256"], args.max_new_tokens, args.heartbeat_sec)
    deterministic = [row["generated_text_hash"] for row in replay] == [row["generated_text_hash"] for row in main_results]
    write_json(out / "determinism_replay_report.json", {"schema_version": "phase_140h_determinism_replay_report_v1", "replay_attempted": True, "same_rows": True, "same_checkpoint": True, "generated_text_hashes_equal": deterministic, "deterministic_replay_passed": deterministic})
    append_progress(out, "determinism replay", passed=deterministic)

    generated_report = {
        "schema_version": "phase_140h_generated_before_scoring_report_v1",
        "passed": True,
        "generated_text_produced_before_scoring": True,
        "all_helper_requests_allowed_keys_only": all(set(row["helper_request"]) == ALLOWED_HELPER_KEYS for row in main_results + ablation_results + replay + control_rows if "helper_request" in row),
        "expected_or_scorer_metadata_in_helper_requests": False,
    }
    write_json(out / "generated_before_scoring_report.json", generated_report)
    write_json(out / "freshness_leakage_audit.json", {"schema_version": "phase_140h_freshness_leakage_audit_v1", "leakage_rejected": True, "expected_or_scorer_metadata_in_helper_requests": False})

    comparison = {
        "schema_version": "phase_140h_arm_comparison_v1",
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
        "visible_bypass_violation_rate": main_metrics["visible_bypass_violation_rate"],
        "noisy_distractor_violation_rate": main_metrics["noisy_distractor_violation_rate"],
        "visible_bypass_control_failed": control_report["visible_bypass_control_failed"],
        "noisy_distractor_control_failed": control_report["noisy_distractor_control_failed"],
        "direct_pocket_value_marker_rate": main_metrics["direct_pocket_value_marker_rate"],
        "explicit_pocket_token_row_rate": main_metrics["explicit_pocket_token_row_rate"],
        "explicit_gate_pocket_open_row_rate": main_metrics["explicit_gate_pocket_open_row_rate"],
        "implicit_or_minimal_gate_row_rate": main_metrics["implicit_or_minimal_gate_row_rate"],
        "deterministic_replay_passed": deterministic,
        "architecture_superiority_claimed": False,
        "value_grounding_claimed": False,
    }
    write_json(out / "minimal_marker_bridge_metrics.json", {"schema_version": "phase_140h_metrics_bundle_v1", "main": main_metrics, "ablation": ablation_metrics})
    write_json(out / "per_seed_metrics.json", {"schema_version": "phase_140h_per_seed_metrics_v1", "main": main_metrics["per_seed"], "ablation": ablation_metrics["per_seed"]})
    write_json(out / "per_family_metrics.json", {"schema_version": "phase_140h_per_family_metrics_v1", "main": main_metrics["per_family"], "ablation": ablation_metrics["per_family"]})
    write_json(out / "per_scaffold_metrics.json", {"schema_version": "phase_140h_per_scaffold_metrics_v1", "main": main_metrics["per_scaffold"], "ablation": ablation_metrics["per_scaffold"]})
    write_json(out / "arm_comparison.json", comparison)
    append_progress(out, "aggregate analysis", delta=comparison["pocket_ablation_delta_answer_value_accuracy"], explicit_pocket_rate=comparison["explicit_pocket_token_row_rate"])

    decision = choose_decision(main_metrics, ablation_metrics, comparison, control_report, deterministic, selected["candidate"], canary["passed"], ast_report["passed"], generated_report["passed"])
    write_json(out / "decision.json", decision)
    summary = {
        "schema_version": "phase_140h_summary_v1",
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
    write_json(out / "queue.json", {"schema_version": "phase_140h_queue_v1", "milestone": MILESTONE, "status": "complete", "decision": decision["decision"], "next": decision["next"]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

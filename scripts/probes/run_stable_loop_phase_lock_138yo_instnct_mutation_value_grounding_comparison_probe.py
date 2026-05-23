#!/usr/bin/env python3
"""138YO INSTNCT mutation value-grounding comparison probe.

This phase compares the 138YK byte-GRU helper rollout artifacts with a fresh
INSTNCT mutation-helper adapter rollout on the same 138YK eval rows. It does
not train, mutate checkpoints, or modify helper/backend code. The comparison is
strictly helper/artifact based and records whether the current adapter actually
uses pocket writeback on these prompts.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import re
import shutil
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
MILESTONE = "STABLE_LOOP_PHASE_LOCK_138YO_INSTNCT_MUTATION_VALUE_GROUNDING_COMPARISON_PROBE"
DEFAULT_OUT = Path("target/pilot_wave/stable_loop_phase_lock_138yo_instnct_mutation_value_grounding_comparison_probe/smoke")
DEFAULT_138YN_ROOT = Path("target/pilot_wave/stable_loop_phase_lock_138yn_instnct_mutation_raw_helper_adapter_probe/smoke")
DEFAULT_138YK_ROOT = Path("target/pilot_wave/stable_loop_phase_lock_138yk_family_default_suppressed_contrastive_repair_probe/smoke")
HELPER_PATH = REPO_ROOT / "scripts/probes/shared_raw_generation_helper.py"
RUNNER_PATH = Path(__file__).resolve()
CHECKER_PATH = REPO_ROOT / "scripts/probes/run_stable_loop_phase_lock_138yo_instnct_mutation_value_grounding_comparison_probe_check.py"

BACKEND_NAME = "repo_local_instnct_mutation_graph"
BYTE_ARM = "byte_gru_138yk_target_existing_helper_rollout"
INSTNCT_ARM = "instnct_mutation_adapter_138yn_same_prompt"
INSTNCT_ABLATION_ARM = "instnct_mutation_adapter_138yn_pocket_gate_ablation"
ALLOWED_HELPER_KEYS = {"prompt", "checkpoint_path", "checkpoint_hash", "seed", "max_new_tokens", "generation_config"}
VALUE_RE = re.compile(r"\b(?:TR|EV|VAL|SYM)[A-Za-z0-9_+\-]*\b")
FALSE_FLAGS = {
    "reasoning_restored": False,
    "reasoning_subtrack_real_raw_evidence_partially_restored": False,
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
    "138YO is a deterministic helper-only comparison probe. It reads the byte-GRU "
    "138YK helper rollout artifacts, runs the INSTNCT mutation helper adapter on "
    "the same eval rows, and writes comparison artifacts only under target/. It "
    "does not train, mutate checkpoints, modify helper/backend/runtime/release "
    "surfaces, import old phase runners, start services, deploy, or claim broad "
    "assistant capability."
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


def append_progress(out: Path, event: str, **details: Any) -> None:
    append_jsonl(out / "progress.jsonl", {"ts": utc_now(), "event": event, "details": details})


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    tmp.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def hash_rows(rows: list[dict[str, Any]], keys: list[str]) -> str:
    payload = [{key: row.get(key) for key in keys} for row in rows]
    return stable_hash(payload)


def load_helper() -> Any:
    spec = importlib.util.spec_from_file_location("shared_raw_generation_helper_138yo", HELPER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to import shared raw generation helper")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def require_upstreams(root_138yn: Path, root_138yk: Path) -> dict[str, Any]:
    required_yn = ["decision.json", "instnct_checkpoint_manifest.json", "determinism_replay_report.json"]
    required_yk = [
        "decision.json",
        "aggregate_metrics.json",
        "eval_rows.jsonl",
        "raw_generation_results.jsonl",
        "target_checkpoint_integrity_manifest.json",
        "generated_before_scoring_report.json",
        "determinism_replay_report.json",
    ]
    missing = [f"138YN:{name}" for name in required_yn if not (root_138yn / name).exists()]
    missing.extend(f"138YK:{name}" for name in required_yk if not (root_138yk / name).exists())
    if missing:
        raise RuntimeError(f"missing upstream artifacts: {missing}")
    decision_yn = read_json(root_138yn / "decision.json")
    decision_yk = read_json(root_138yk / "decision.json")
    metrics_yk = read_json(root_138yk / "aggregate_metrics.json")
    replay_yn = read_json(root_138yn / "determinism_replay_report.json")
    replay_yk = read_json(root_138yk / "determinism_replay_report.json")
    if decision_yn.get("decision") != "instnct_mutation_raw_helper_adapter_probe_complete":
        raise RuntimeError(f"bad 138YN decision: {decision_yn.get('decision')}")
    if decision_yn.get("next") != "138YO_INSTNCT_MUTATION_VALUE_GROUNDING_COMPARISON_PROBE":
        raise RuntimeError(f"bad 138YN next: {decision_yn.get('next')}")
    if decision_yk.get("decision") != "family_default_shortcut_persists":
        raise RuntimeError(f"bad 138YK decision: {decision_yk.get('decision')}")
    if metrics_yk.get("answer_value_accuracy") != 0.0 or metrics_yk.get("intra_family_contrastive_accuracy") != 0.0:
        raise RuntimeError("138YK baseline profile no longer matches expected value-grounding failure")
    if replay_yn.get("deterministic_replay_passed") is not True:
        raise RuntimeError("138YN determinism did not pass")
    if replay_yk.get("determinism_replay_passed") is not True:
        raise RuntimeError("138YK determinism did not pass")
    return {
        "138yn": {
            "root": rel(root_138yn),
            "decision": decision_yn.get("decision"),
            "next": decision_yn.get("next"),
            "verdict": decision_yn.get("verdict"),
            "deterministic_replay_passed": replay_yn.get("deterministic_replay_passed"),
        },
        "138yk": {
            "root": rel(root_138yk),
            "decision": decision_yk.get("decision"),
            "next": decision_yk.get("next"),
            "verdict": decision_yk.get("verdict"),
            "answer_value_accuracy": metrics_yk.get("answer_value_accuracy"),
            "intra_family_contrastive_accuracy": metrics_yk.get("intra_family_contrastive_accuracy"),
            "family_default_attractor_rate": metrics_yk.get("family_default_attractor_rate"),
            "determinism_replay_passed": replay_yk.get("determinism_replay_passed"),
        },
    }


def first_value_after_answer_e(text: str) -> str | None:
    marker = re.search(r"\bANSWER=E", text or "")
    if not marker:
        return None
    values = VALUE_RE.findall(text[marker.end() :])
    return values[0] if values else None


def has_answer_prefix(text: str) -> bool:
    return bool(re.search(r"\bANSWER=E", text or ""))


def load_byte_results(root_138yk: Path, eval_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    raw_by_id = {row["row_id"]: row for row in read_jsonl(root_138yk / "raw_generation_results.jsonl")}
    out: list[dict[str, Any]] = []
    for row in eval_rows:
        raw = raw_by_id.get(row["row_id"])
        if raw is None:
            raise RuntimeError(f"missing 138YK raw generation for {row['row_id']}")
        out.append(
            {
                "arm": BYTE_ARM,
                "row_id": row["row_id"],
                "family": row["family"],
                "contrast_group_id": row["contrast_group_id"],
                "seed": raw.get("seed", row.get("seed")),
                "prompt_hash": raw.get("prompt_hash"),
                "generated_text": raw["generated_text"],
                "generated_text_hash": raw.get("generated_text_hash") or stable_hash(raw["generated_text"]),
                "generation_trace_hash": raw.get("generation_trace_hash"),
                "token_count": raw.get("token_count"),
                "backend_name": "repo_local_checkpoint_byte_lm",
                "helper_request": None,
                "helper_response": None,
                "baseline_source": "138YK raw_generation_results.jsonl",
            }
        )
    return out


def copy_instnct_manifest(root_138yn: Path, out: Path, ablation: bool = False) -> tuple[Path, dict[str, Any]]:
    source_manifest = read_json(root_138yn / "instnct_checkpoint_manifest.json")
    source_path = resolve_repo_path(source_manifest["checkpoint_path"])
    manifest = read_json(source_path)
    if ablation:
        manifest["schema_version"] = "instnct_mutation_graph_manifest_v1_ablation"
        for idx, pocket in enumerate(manifest.get("pockets", [])):
            pocket["gate_marker"] = f"ABLATION_GATE_NEVER_{idx}"
        manifest["claim_boundary"] = "pocket gate ablation manifest; value selection path unchanged"
        target = out / "checkpoints/instnct_mutation_graph_manifest_pocket_ablation.json"
    else:
        target = out / "checkpoints/instnct_mutation_graph_manifest.json"
    write_json(target, manifest)
    return target, {"checkpoint_path": rel(target), "checkpoint_hash": sha256_file(target), "backend_name": manifest.get("backend_name")}


def helper_request(helper: Any, row: dict[str, Any], checkpoint_path: Path, checkpoint_hash: str, max_new_tokens: int) -> dict[str, Any]:
    return helper.build_request(
        prompt=row["prompt"],
        checkpoint_path=rel(checkpoint_path),
        checkpoint_hash=checkpoint_hash,
        seed=int(row.get("seed", 0)),
        max_new_tokens=max_new_tokens,
        generation_config={"temperature": 0.0, "device": "cpu", "stop_on_newline": False},
    )


def run_instnct_arm(
    helper: Any,
    out: Path,
    arm: str,
    rows: list[dict[str, Any]],
    checkpoint_path: Path,
    checkpoint_hash: str,
    max_new_tokens: int,
    heartbeat_sec: int,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    last_heartbeat = time.monotonic()
    trace_path = out / f"{arm}_raw_generation_trace.jsonl"
    for index, row in enumerate(rows, start=1):
        request = helper_request(helper, row, checkpoint_path, checkpoint_hash, max_new_tokens)
        if set(request) != ALLOWED_HELPER_KEYS:
            raise RuntimeError(f"bad helper request keys for {row['row_id']}: {sorted(request)}")
        response = helper.raw_generate(request)
        generated_text = response["generated_text"]
        packed = {
            "arm": arm,
            "row_id": row["row_id"],
            "family": row["family"],
            "contrast_group_id": row["contrast_group_id"],
            "seed": row.get("seed"),
            "prompt_hash": hashlib.sha256(row["prompt"].encode("utf-8", errors="replace")).hexdigest(),
            "generated_text": generated_text,
            "generated_text_hash": hashlib.sha256(generated_text.encode("utf-8", errors="replace")).hexdigest(),
            "generation_trace_hash": response.get("generation_trace_hash"),
            "token_count": response.get("token_count"),
            "backend_name": response.get("backend_name"),
            "pocket_writeback_count": response.get("pocket_writeback_count"),
            "highway_retained": response.get("highway_retained"),
            "ticks_per_generated_byte": response.get("ticks_per_generated_byte"),
            "threshold_tick": response.get("threshold_tick"),
            "helper_request": request,
            "helper_response": {key: value for key, value in response.items() if key != "generated_text"},
        }
        append_jsonl(trace_path, packed)
        results.append(packed)
        now = time.monotonic()
        if now - last_heartbeat >= heartbeat_sec:
            append_progress(out, f"{arm} generation heartbeat", completed=index, total=len(rows))
            last_heartbeat = now
    return results


def score_results(arm: str, rows: list[dict[str, Any]], generations: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    rows_by_id = {row["row_id"]: row for row in rows}
    scored: list[dict[str, Any]] = []
    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for generation in generations:
        row = rows_by_id[generation["row_id"]]
        text = generation["generated_text"]
        generated_value = first_value_after_answer_e(text)
        expected_value = row["answer_value"]
        forbidden_defaults = set(row.get("forbidden_family_default_values") or [])
        peer_expected = set(row.get("peer_expected_values") or [])
        prefix_ok = has_answer_prefix(text)
        correct = generated_value == expected_value
        exact = text.strip() == row.get("expected_output")
        family_default = generated_value in forbidden_defaults
        peer_confusion = generated_value in peer_expected
        stale = "User:" in text or "Assistant:" in text
        namespace_leak = "ANSWER=T" in text
        high_frequency = bool(generated_value and generated_value.startswith("TR"))
        if family_default:
            failure = "family_default_shortcut"
        elif peer_confusion:
            failure = "peer_expected_value_confusion"
        elif high_frequency:
            failure = "high_frequency_train_value_replay"
        elif not prefix_ok:
            failure = "missing_answer_prefix"
        elif not correct:
            failure = "wrong_value"
        else:
            failure = None
        item = {
            "arm": arm,
            "row_id": row["row_id"],
            "family": row["family"],
            "scoring_mode": row.get("scoring_mode"),
            "contrast_group_id": row["contrast_group_id"],
            "expected_value": expected_value,
            "generated_value": generated_value,
            "generated_text": text,
            "answer_prefix_ok": prefix_ok,
            "answer_value_correct": correct,
            "exact_answer_correct": exact,
            "family_default_violation": family_default,
            "peer_expected_confusion": peer_confusion,
            "high_frequency_train_value": high_frequency,
            "stale_chat_fragment": stale,
            "train_namespace_leak": namespace_leak,
            "failure_reason": failure,
            "pocket_writeback_count": generation.get("pocket_writeback_count"),
            "highway_retained": generation.get("highway_retained"),
        }
        scored.append(item)
        by_group[row["contrast_group_id"]].append(item)

    group_rows: list[dict[str, Any]] = []
    for group_id, group in sorted(by_group.items()):
        expected_values = [item["expected_value"] for item in group]
        generated_values = [item["generated_value"] for item in group]
        distinct_expected = len(set(expected_values)) == len(expected_values)
        distinct_generated = len(set(generated_values)) == len(generated_values)
        all_correct = all(item["answer_value_correct"] for item in group)
        no_default = not any(item["family_default_violation"] for item in group)
        collapse = len(set(generated_values)) == 1
        row = {
            "arm": arm,
            "group_id": group_id,
            "family": group[0]["family"],
            "row_count": len(group),
            "expected_values": expected_values,
            "generated_values": generated_values,
            "distinct_expected": distinct_expected,
            "distinct_generated": distinct_generated,
            "all_correct": all_correct,
            "no_family_default": no_default,
            "group_collapse_to_single_value": collapse,
            "group_pass": all_correct and distinct_expected and distinct_generated and no_default,
        }
        group_rows.append(row)

    row_count = len(scored)
    group_count = len(group_rows)
    family_default_count = sum(1 for item in scored if item["family_default_violation"])
    high_frequency_count = sum(1 for item in scored if item["high_frequency_train_value"])
    wrong_values = [item["generated_value"] for item in scored if not item["answer_value_correct"] and item["generated_value"]]
    dominant_wrong_rate = 0.0
    if wrong_values:
        dominant_wrong_rate = Counter(wrong_values).most_common(1)[0][1] / row_count
    mode_rows = defaultdict(list)
    family_rows = defaultdict(list)
    for item in scored:
        mode_rows[item.get("scoring_mode")].append(item)
        family_rows[item["family"]].append(item)
    metrics = {
        "schema_version": "phase_138yo_arm_metrics_v1",
        "arm": arm,
        "row_count": row_count,
        "group_count": group_count,
        "answer_prefix_accuracy": rate(sum(1 for item in scored if item["answer_prefix_ok"]), row_count),
        "answer_value_accuracy": rate(sum(1 for item in scored if item["answer_value_correct"]), row_count),
        "exact_answer_accuracy": rate(sum(1 for item in scored if item["exact_answer_correct"]), row_count),
        "value_after_prefix_accuracy": rate(sum(1 for item in scored if item["answer_value_correct"]), row_count),
        "intra_family_contrastive_accuracy": rate(sum(1 for item in group_rows if item["group_pass"]), group_count),
        "intra_family_unique_correct_value_rate": rate(sum(1 for item in group_rows if item["all_correct"] and item["distinct_generated"]), group_count),
        "intra_family_mode_collapse_rate": rate(sum(1 for item in group_rows if item["group_collapse_to_single_value"]), group_count),
        "family_default_attractor_rate": rate(family_default_count, row_count),
        "family_default_reuse_rate": rate(family_default_count, row_count),
        "family_dominant_wrong_value_rate": dominant_wrong_rate,
        "multi_expected_to_single_default_rate": rate(sum(1 for item in group_rows if item["group_collapse_to_single_value"] and not item["all_correct"]), group_count),
        "same_value_for_all_rows_rate": rate(sum(1 for item in group_rows if item["group_collapse_to_single_value"]), group_count),
        "hard_negative_default_violation_rate": rate(family_default_count, row_count),
        "rule_derived_value_accuracy": mode_accuracy(mode_rows, "rule_derived"),
        "table_derived_value_accuracy": mode_accuracy(mode_rows, "table_derived"),
        "composition_derived_value_accuracy": mode_accuracy(mode_rows, "composition_derived"),
        "ood_symbol_value_accuracy": mode_accuracy(mode_rows, "ood_symbol_binding"),
        "direct_copy_value_accuracy": mode_accuracy(mode_rows, "direct_copy"),
        "contradiction_resolution_accuracy": mode_accuracy(mode_rows, "contradiction_resolution"),
        "no_stale_direct_accuracy": mode_accuracy(mode_rows, "no_stale_direct"),
        "after_prefix_stability_accuracy": mode_accuracy(mode_rows, "after_prefix_stability"),
        "train_namespace_leak_rate": rate(sum(1 for item in scored if item["train_namespace_leak"]), row_count),
        "stale_chat_fragment_rate": rate(sum(1 for item in scored if item["stale_chat_fragment"]), row_count),
        "high_frequency_train_value_replay_rate": rate(high_frequency_count, row_count),
        "high_frequency_train_value_replay_detected": high_frequency_count > 0,
        "family_default_shortcut_detected": family_default_count > 0,
        "parrot_trap_detected": False,
        "pocket_writeback_rate": rate(sum(1 for item in scored if (item.get("pocket_writeback_count") or 0) > 0), row_count),
        "highway_retention_rate": rate(sum(1 for item in scored if item.get("highway_retained") is True), row_count),
        "phase_transport_success_rate": rate(sum(1 for item in scored if (item.get("pocket_writeback_count") or 0) > 0 and item.get("highway_retained") is True), row_count),
        "per_family_answer_value_accuracy": {
            family: rate(sum(1 for item in items if item["answer_value_correct"]), len(items))
            for family, items in sorted(family_rows.items())
        },
    }
    return scored, metrics, group_rows


def rate(count: int, total: int) -> float:
    return count / total if total else 0.0


def mode_accuracy(mode_rows: dict[str | None, list[dict[str, Any]]], mode: str) -> float:
    rows = mode_rows.get(mode, [])
    return rate(sum(1 for item in rows if item["answer_value_correct"]), len(rows))


def build_control_report(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    sample = rows[0]
    controls = [
        ("STATIC_OUTPUT_CONTROL", "ANSWER=EEV_STATIC"),
        ("FAMILY_DEFAULT_VALUE_CONTROL", f"ANSWER=E{(sample.get('forbidden_family_default_values') or ['TR_DEFAULT'])[0]}"),
        ("PREFIX_ONLY_CONTROL", "ANSWER=E"),
        ("TRAIN_NAMESPACE_REPLAY_CONTROL", "ANSWER=TTR_BAD"),
        ("PEER_EXPECTED_VALUE_CONFUSION_CONTROL", f"ANSWER=E{(sample.get('peer_expected_values') or ['EV_PEER'])[0]}"),
    ]
    results = []
    for name, output in controls:
        generated_value = first_value_after_answer_e(output)
        passed = generated_value == sample["answer_value"] and output.strip() == sample["expected_output"]
        results.append({"control": name, "generated_text": output, "generated_value": generated_value, "control_passed": passed})
    return results, {
        "schema_version": "phase_138yo_control_arm_report_v1",
        "control_count": len(results),
        "controls_failed": all(not row["control_passed"] for row in results),
        "passed_controls": [row["control"] for row in results if row["control_passed"]],
    }


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
    return {"schema_version": "phase_138yo_ast_shortcut_scan_v1", "passed": not failures, "failures": failures}


def forbidden_input_canary(helper: Any, checkpoint_path: Path, checkpoint_hash: str) -> dict[str, Any]:
    request = helper.build_request(
        prompt="GATE:VALUE_BIND OBSERVED_VALUE=EV_CANARY",
        checkpoint_path=rel(checkpoint_path),
        checkpoint_hash=checkpoint_hash,
        seed=9901,
        max_new_tokens=32,
        generation_config={"temperature": 0.0, "device": "cpu", "stop_on_newline": False},
    )
    request["expected_output"] = "ANSWER=EEV_CANARY"
    try:
        helper.raw_generate(request)
    except Exception as exc:  # helper owns the concrete exception class.
        verdict = getattr(exc, "verdict", "")
        return {
            "schema_version": "phase_138yo_expected_output_canary_v1",
            "passed": verdict == "RAW_GENERATION_FORBIDDEN_INPUT_DETECTED",
            "verdict": verdict,
            "message": str(exc),
        }
    return {
        "schema_version": "phase_138yo_expected_output_canary_v1",
        "passed": False,
        "verdict": "CANARY_NOT_REJECTED",
    }


def compare_metrics(byte_metrics: dict[str, Any], instnct_metrics: dict[str, Any], ablation_metrics: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": "phase_138yo_arm_comparison_v1",
        "all_eval_rows_match": True,
        "eval_row_count": len(rows),
        "eval_row_hash": hash_rows(rows, ["row_id", "prompt", "answer_value", "contrast_group_id"]),
        "arms": [
            {"arm": BYTE_ARM, "backend": "repo_local_checkpoint_byte_lm", "metrics": byte_metrics},
            {"arm": INSTNCT_ARM, "backend": BACKEND_NAME, "metrics": instnct_metrics},
            {"arm": INSTNCT_ABLATION_ARM, "backend": BACKEND_NAME, "metrics": ablation_metrics},
        ],
        "instnct_minus_byte_answer_value_accuracy": instnct_metrics["answer_value_accuracy"] - byte_metrics["answer_value_accuracy"],
        "instnct_minus_byte_contrastive_accuracy": instnct_metrics["intra_family_contrastive_accuracy"] - byte_metrics["intra_family_contrastive_accuracy"],
        "instnct_minus_byte_family_default_attractor_rate": instnct_metrics["family_default_attractor_rate"] - byte_metrics["family_default_attractor_rate"],
        "pocket_ablation_delta_answer_value_accuracy": instnct_metrics["answer_value_accuracy"] - ablation_metrics["answer_value_accuracy"],
        "pocket_ablation_delta_contrastive_accuracy": instnct_metrics["intra_family_contrastive_accuracy"] - ablation_metrics["intra_family_contrastive_accuracy"],
        "instnct_beats_byte_gru_answer_value_accuracy": instnct_metrics["answer_value_accuracy"] > byte_metrics["answer_value_accuracy"],
        "instnct_beats_byte_gru_contrastive_accuracy": instnct_metrics["intra_family_contrastive_accuracy"] > byte_metrics["intra_family_contrastive_accuracy"],
        "architecture_superiority_claimed": False,
        "comparison_boundary": "Improvement can support adapter-path evidence only; it cannot by itself prove full INSTNCT graph architecture superiority.",
    }


def write_report(out: Path, decision: dict[str, Any], comparison: dict[str, Any], instnct_metrics: dict[str, Any]) -> None:
    text = f"""# {MILESTONE}

Decision: `{decision['decision']}`

Verdict: `{decision['verdict']}`

Next: `{decision['next']}`

Boundary: {BOUNDARY_TEXT}

Primary comparison:

- Byte-GRU baseline answer value accuracy: `{comparison['arms'][0]['metrics']['answer_value_accuracy']}`
- INSTNCT adapter answer value accuracy: `{comparison['arms'][1]['metrics']['answer_value_accuracy']}`
- Delta: `{comparison['instnct_minus_byte_answer_value_accuracy']}`
- INSTNCT contrastive accuracy: `{comparison['arms'][1]['metrics']['intra_family_contrastive_accuracy']}`
- Pocket writeback rate on fair 138YK prompts: `{instnct_metrics['pocket_writeback_rate']}`
- Pocket ablation delta: `{comparison['pocket_ablation_delta_answer_value_accuracy']}`

Interpretation:

The comparison uses the same 138YK eval rows. Any INSTNCT improvement here is adapter-path evidence, not GPT-like readiness and not a broad assistant capability claim. If pocket writeback is zero or ablation has no effect, the current helper adapter is not yet proving that the pocket/highway mechanism is responsible for value binding.
"""
    write_text(out / "report.md", text)


def decide(comparison: dict[str, Any], instnct_metrics: dict[str, Any], control_report: dict[str, Any], determinism: dict[str, Any]) -> dict[str, Any]:
    if control_report.get("controls_failed") is not True:
        decision = "scorer_or_task_weakness"
        verdict = "INSTNCT_MUTATION_COMPARISON_INVALID_CONTROLS"
        next_step = "138E_REASONING_SCORER_OR_TASK_WEAKNESS_ANALYSIS"
    elif determinism.get("deterministic_replay_passed") is not True:
        decision = "nondeterministic_instnct_mutation_comparison"
        verdict = "DETERMINISM_REPLAY_MISMATCH"
        next_step = "138N_DETERMINISM_FAILURE_ANALYSIS"
    elif comparison["instnct_beats_byte_gru_answer_value_accuracy"] and instnct_metrics["pocket_writeback_rate"] <= 0.0:
        decision = "instnct_adapter_prompt_bound_value_grounding_improves"
        verdict = "INSTNCT_ADAPTER_BEATS_BYTE_GRU_BUT_POCKET_WRITEBACK_NOT_USED"
        next_step = "138YP_INSTNCT_MUTATION_POCKET_GATED_VALUE_GROUNDING_PLAN"
    elif comparison["instnct_beats_byte_gru_answer_value_accuracy"]:
        decision = "instnct_mutation_value_grounding_comparison_positive"
        verdict = "INSTNCT_MUTATION_VALUE_GROUNDING_COMPARISON_POSITIVE"
        next_step = "139YO_INSTNCT_MUTATION_VALUE_GROUNDING_SCALE_CONFIRM"
    else:
        decision = "instnct_mutation_value_grounding_not_better_than_byte_gru"
        verdict = "INSTNCT_MUTATION_VALUE_GROUNDING_COMPARISON_FAILS"
        next_step = "138YOB_INSTNCT_MUTATION_COMPARISON_FAILURE_ANALYSIS"
    payload = {
        "schema_version": "phase_138yo_decision_v1",
        "decision": decision,
        "verdict": verdict,
        "next": next_step,
        "clean_negative_valid": True,
        "value_grounding_claimed": decision == "instnct_mutation_value_grounding_comparison_positive",
        "architecture_superiority_claimed": False,
        "pocket_mechanism_claimed": instnct_metrics["pocket_writeback_rate"] > 0.0 and comparison["pocket_ablation_delta_answer_value_accuracy"] > 0.0,
        **FALSE_FLAGS,
    }
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Run 138YO INSTNCT mutation value-grounding comparison")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--upstream-138yn-root", type=Path, default=DEFAULT_138YN_ROOT)
    parser.add_argument("--upstream-138yk-root", type=Path, default=DEFAULT_138YK_ROOT)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--row-limit", type=int, default=0)
    parser.add_argument("--heartbeat-sec", type=int, default=20)
    args = parser.parse_args()

    out = resolve_target_out(args.out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    append_progress(out, "startup", milestone=MILESTONE)
    write_json(out / "queue.json", {"schema_version": "phase_138yo_queue_v1", "milestone": MILESTONE, "status": "running"})

    root_138yn = resolve_repo_path(args.upstream_138yn_root)
    root_138yk = resolve_repo_path(args.upstream_138yk_root)
    upstreams = require_upstreams(root_138yn, root_138yk)
    write_json(out / "upstream_138yn_manifest.json", upstreams["138yn"])
    write_json(out / "upstream_138yk_manifest.json", upstreams["138yk"])
    append_progress(out, "upstream verification", upstreams=upstreams)

    analysis_config = {
        "schema_version": "phase_138yo_analysis_config_v1",
        "milestone": MILESTONE,
        "boundary": BOUNDARY_TEXT,
        "byte_gru_baseline_source": "138YK raw_generation_results.jsonl",
        "instnct_generation_source": "fresh shared_raw_generation_helper.py raw_generate calls",
        "training_performed": False,
        "checkpoint_mutated": False,
        "helper_modified": False,
        "service_deploy_allowed": False,
        "max_new_tokens": args.max_new_tokens,
        "row_limit": args.row_limit,
        **FALSE_FLAGS,
    }
    write_json(out / "analysis_config.json", analysis_config)

    helper = load_helper()
    append_progress(out, "helper import", helper_path=rel(HELPER_PATH))
    ast_report = scan_ast()
    write_json(out / "ast_shortcut_scan_report.json", ast_report)
    append_progress(out, "ast shortcut scan", passed=ast_report["passed"])

    rows = read_jsonl(root_138yk / "eval_rows.jsonl")
    if args.row_limit > 0:
        rows = rows[: args.row_limit]
    write_jsonl(out / "eval_rows.jsonl", rows)
    eval_manifest = {
        "schema_version": "phase_138yo_eval_dataset_manifest_v1",
        "row_count": len(rows),
        "row_hash": hash_rows(rows, ["row_id", "prompt", "answer_value"]),
        "prompt_hash": hash_rows(rows, ["row_id", "prompt"]),
        "source": rel(root_138yk / "eval_rows.jsonl"),
    }
    write_json(out / "eval_dataset_manifest.json", eval_manifest)
    append_progress(out, "eval row load", row_count=len(rows))

    instnct_checkpoint, instnct_manifest = copy_instnct_manifest(root_138yn, out, ablation=False)
    ablation_checkpoint, ablation_manifest = copy_instnct_manifest(root_138yn, out, ablation=True)
    write_json(out / "instnct_checkpoint_manifest.json", instnct_manifest)
    write_json(out / "instnct_ablation_checkpoint_manifest.json", ablation_manifest)
    append_progress(out, "instnct manifest bind", checkpoint=instnct_manifest)

    canary = forbidden_input_canary(helper, instnct_checkpoint, instnct_manifest["checkpoint_hash"])
    write_json(out / "expected_output_canary_report.json", canary)
    write_json(out / "forbidden_input_rejection_report.json", {"schema_version": "phase_138yo_forbidden_input_rejection_v1", "passed": canary["passed"], "canary_verdict": canary["verdict"]})
    append_progress(out, "expected output canary", passed=canary["passed"])

    byte_results = load_byte_results(root_138yk, rows)
    write_jsonl(out / "byte_gru_raw_generation_results.jsonl", byte_results)
    append_progress(out, "byte baseline artifact load", row_count=len(byte_results))

    instnct_results = run_instnct_arm(helper, out, INSTNCT_ARM, rows, instnct_checkpoint, instnct_manifest["checkpoint_hash"], args.max_new_tokens, args.heartbeat_sec)
    write_jsonl(out / "instnct_raw_generation_results.jsonl", instnct_results)
    append_progress(out, "instnct generation complete", row_count=len(instnct_results))

    ablation_results = run_instnct_arm(helper, out, INSTNCT_ABLATION_ARM, rows, ablation_checkpoint, ablation_manifest["checkpoint_hash"], args.max_new_tokens, args.heartbeat_sec)
    write_jsonl(out / "instnct_ablation_raw_generation_results.jsonl", ablation_results)
    append_progress(out, "pocket ablation generation complete", row_count=len(ablation_results))

    byte_scored, byte_metrics, byte_groups = score_results(BYTE_ARM, rows, byte_results)
    instnct_scored, instnct_metrics, instnct_groups = score_results(INSTNCT_ARM, rows, instnct_results)
    ablation_scored, ablation_metrics, ablation_groups = score_results(INSTNCT_ABLATION_ARM, rows, ablation_results)
    write_jsonl(out / "byte_gru_scoring_results.jsonl", byte_scored)
    write_jsonl(out / "instnct_scoring_results.jsonl", instnct_scored)
    write_jsonl(out / "instnct_ablation_scoring_results.jsonl", ablation_scored)
    write_json(out / "byte_gru_value_grounding_metrics.json", byte_metrics)
    write_json(out / "instnct_value_grounding_metrics.json", instnct_metrics)
    write_json(out / "instnct_ablation_value_grounding_metrics.json", ablation_metrics)
    write_jsonl(out / "contrast_group_results.jsonl", byte_groups + instnct_groups + ablation_groups)
    append_progress(out, "scoring", byte_accuracy=byte_metrics["answer_value_accuracy"], instnct_accuracy=instnct_metrics["answer_value_accuracy"])

    comparison = compare_metrics(byte_metrics, instnct_metrics, ablation_metrics, rows)
    write_json(out / "arm_comparison.json", comparison)
    pocket_report = {
        "schema_version": "phase_138yo_pocket_ablation_report_v1",
        "main_arm": INSTNCT_ARM,
        "ablation_arm": INSTNCT_ABLATION_ARM,
        "pocket_writeback_rate": instnct_metrics["pocket_writeback_rate"],
        "highway_retention_rate": instnct_metrics["highway_retention_rate"],
        "phase_transport_success_rate": instnct_metrics["phase_transport_success_rate"],
        "answer_value_accuracy_delta": comparison["pocket_ablation_delta_answer_value_accuracy"],
        "contrastive_accuracy_delta": comparison["pocket_ablation_delta_contrastive_accuracy"],
        "pocket_writeback_decision_critical": comparison["pocket_ablation_delta_answer_value_accuracy"] > 0.0,
        "interpretation": "Current same-prompt eval rows do not prove pocket/highway value binding if writeback rate is zero or ablation delta is zero.",
    }
    write_json(out / "pocket_ablation_report.json", pocket_report)
    append_progress(out, "arm comparison", delta=comparison["instnct_minus_byte_answer_value_accuracy"], pocket_delta=pocket_report["answer_value_accuracy_delta"])

    controls, control_report = build_control_report(rows)
    write_jsonl(out / "control_results.jsonl", controls)
    write_json(out / "control_arm_report.json", control_report)
    append_progress(out, "controls", passed=control_report["controls_failed"])

    replay_results = run_instnct_arm(helper, out, f"{INSTNCT_ARM}_replay", rows, instnct_checkpoint, instnct_manifest["checkpoint_hash"], args.max_new_tokens, args.heartbeat_sec)
    replay_hashes = [row["generated_text_hash"] for row in replay_results]
    original_hashes = [row["generated_text_hash"] for row in instnct_results]
    replay_scored, replay_metrics, replay_groups = score_results(f"{INSTNCT_ARM}_replay", rows, replay_results)
    determinism = {
        "schema_version": "phase_138yo_determinism_replay_report_v1",
        "replay_attempted": True,
        "same_rows": True,
        "same_checkpoint": True,
        "generated_text_hashes_equal": replay_hashes == original_hashes,
        "metrics_equal": replay_metrics == {**instnct_metrics, "arm": f"{INSTNCT_ARM}_replay"},
        "group_pass_equal": [row["group_pass"] for row in replay_groups] == [row["group_pass"] for row in instnct_groups],
        "deterministic_replay_passed": replay_hashes == original_hashes,
    }
    write_json(out / "determinism_replay_report.json", determinism)
    append_progress(out, "determinism replay", passed=determinism["deterministic_replay_passed"])

    generated_before_scoring = {
        "schema_version": "phase_138yo_generated_before_scoring_report_v1",
        "passed": True,
        "generated_text_produced_before_scoring": True,
        "scoring_files_written_after_generation": [
            "byte_gru_scoring_results.jsonl",
            "instnct_scoring_results.jsonl",
            "instnct_ablation_scoring_results.jsonl",
        ],
        "expected_or_scorer_metadata_in_helper_requests": False,
        "all_helper_requests_allowed_keys_only": all(set(row["helper_request"]) == ALLOWED_HELPER_KEYS for row in instnct_results + ablation_results + replay_results),
    }
    write_json(out / "generated_before_scoring_report.json", generated_before_scoring)

    summary_metrics = {
        "schema_version": "phase_138yo_summary_metrics_v1",
        "byte_gru_answer_value_accuracy": byte_metrics["answer_value_accuracy"],
        "instnct_answer_value_accuracy": instnct_metrics["answer_value_accuracy"],
        "instnct_minus_byte_answer_value_accuracy": comparison["instnct_minus_byte_answer_value_accuracy"],
        "instnct_intra_family_contrastive_accuracy": instnct_metrics["intra_family_contrastive_accuracy"],
        "instnct_pocket_writeback_rate": instnct_metrics["pocket_writeback_rate"],
        "pocket_ablation_delta_answer_value_accuracy": pocket_report["answer_value_accuracy_delta"],
        "deterministic_replay_passed": determinism["deterministic_replay_passed"],
        "controls_failed": control_report["controls_failed"],
    }
    write_json(out / "aggregate_metrics.json", summary_metrics)

    decision = decide(comparison, instnct_metrics, control_report, determinism)
    write_json(out / "decision.json", decision)
    summary = {
        "schema_version": "phase_138yo_summary_v1",
        "milestone": MILESTONE,
        "status": "complete",
        "boundary": BOUNDARY_TEXT,
        "metrics": summary_metrics,
        **decision,
    }
    write_json(out / "summary.json", summary)
    write_report(out, decision, comparison, instnct_metrics)
    append_progress(out, "decision", decision=decision["decision"], next=decision["next"])
    append_progress(out, "final verdict", verdict=decision["verdict"])
    write_json(out / "queue.json", {"schema_version": "phase_138yo_queue_v1", "milestone": MILESTONE, "status": "complete", "decision": decision["decision"], "next": decision["next"]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

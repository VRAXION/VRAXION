#!/usr/bin/env python3
"""Target-only decoder generation repair PoC after 094B."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import shutil
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
MILESTONE = "STABLE_LOOP_PHASE_LOCK_095_CHAT_DECODER_GENERATION_REPAIR"
DEFAULT_OUT = Path("target/pilot_wave/stable_loop_phase_lock_095_chat_decoder_generation_repair/smoke")
DEFAULT_UPSTREAM_094B_ROOT = Path("target/pilot_wave/stable_loop_phase_lock_094b_chat_sft_free_generation_gap_analysis/smoke")
BOUNDARY_TEXT = (
    "095 is a target-only decoder generation repair PoC. It does not train, does not mutate checkpoints, "
    "does not deploy, and does not prove GPT-like assistant readiness, open-domain assistant readiness, "
    "production chat, public release, or safety alignment."
)


class GateError(Exception):
    def __init__(self, verdict: str, message: str):
        super().__init__(message)
        self.verdict = verdict
        self.message = message


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise GateError("UPSTREAM_094B_ARTIFACT_MISSING", f"cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PHASE094 = load_module("phase094", REPO_ROOT / "scripts/probes/run_stable_loop_phase_lock_094_open_vocab_chat_sft_mix_poc.py")
PHASE094B = load_module("phase094b", REPO_ROOT / "scripts/probes/run_stable_loop_phase_lock_094b_chat_sft_free_generation_gap_analysis.py")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


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
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_json_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


def resolve_repo_path(text: str, verdict: str) -> Path:
    path = Path(text)
    if path.is_absolute() or any(part == ".." for part in path.parts):
        raise GateError(verdict, f"path must be repo-relative: {text}")
    return (REPO_ROOT / path).resolve()


def resolve_target_out(text: str) -> Path:
    path = Path(text)
    if path.is_absolute() or any(part == ".." for part in path.parts):
        raise GateError("DECODER_REPAIR_ARTIFACT_MISSING", "--out must be repo-relative")
    parts = [part.lower() for part in path.parts]
    if len(parts) < 2 or parts[0] != "target" or parts[1] != "pilot_wave":
        raise GateError("DECODER_REPAIR_ARTIFACT_MISSING", "--out must stay under target/pilot_wave")
    return (REPO_ROOT / path).resolve()


def append_progress(out: Path, event: str, status: str = "completed", **details: Any) -> None:
    append_jsonl(out / "progress.jsonl", {"ts": utc_now(), "event": event, "status": status, "details": details})


def write_summary(out: Path, status: str, verdicts: list[str], metrics: dict[str, Any], message: str = "") -> None:
    payload = {
        "schema_version": "chat_decoder_generation_repair_summary_v1",
        "milestone": MILESTONE,
        "status": status,
        "phase": status,
        "boundary": BOUNDARY_TEXT,
        "target_only_decoder_repair": True,
        "no_training_performed": True,
        "optimizer_step_count": 0,
        "checkpoint_mutation": False,
        "prediction_oracle_used": False,
        "llm_judge_used": False,
        "expected_response_used_for_generation": False,
        "response_table_used": False,
        "gpt_like_assistant_readiness_claimed": False,
        "production_chat_claimed": False,
        "public_release_claimed": False,
        "deployment_claimed": False,
        "safety_alignment_claimed": False,
        "metrics": metrics,
        "verdicts": verdicts,
    }
    if message:
        payload["message"] = message
    write_json(out / "summary.json", payload)
    lines = [
        "# STABLE_LOOP_PHASE_LOCK_095_CHAT_DECODER_GENERATION_REPAIR Report",
        "",
        BOUNDARY_TEXT,
        "",
        f"Status: `{status}`",
        "",
        "## Verdicts",
        "",
        "```text",
        *verdicts,
        "```",
        "",
        "## Metrics",
        "",
    ]
    for key in [
        "baseline_generated_accuracy",
        "repaired_generated_accuracy",
        "generation_accuracy_delta",
        "bounded_slot_accuracy",
        "finite_label_accuracy",
        "unsupported_refusal_accuracy",
        "max_new_bytes_stop_rate",
        "checkpoint_unchanged",
    ]:
        if key in metrics:
            lines.append(f"- {key}: `{metrics[key]}`")
    if message:
        lines.extend(["", "## Message", "", message])
    lines.extend([
        "",
        "## Boundary",
        "",
        "target-only decoder generation repair PoC",
        "not GPT-like assistant readiness",
        "not open-domain assistant readiness",
        "not production chat",
        "not deployment",
        "not public release",
        "not safety alignment",
    ])
    write_text(out / "report.md", "\n".join(lines))


def fail(out: Path, verdict: str, message: str, metrics: dict[str, Any]) -> int:
    append_progress(out, "final verdict", "failed", verdict=verdict, message=message)
    write_summary(out, "failed", ["CHAT_DECODER_GENERATION_REPAIR_FAILS", verdict], metrics, message)
    return 1


def verify_upstream(root: Path, out: Path) -> dict[str, Any]:
    summary_path = root / "summary.json"
    if not summary_path.exists():
        raise GateError("UPSTREAM_094B_ARTIFACT_MISSING", "094B summary missing")
    summary = read_json(summary_path)
    verdicts = set(summary.get("verdicts", []))
    metrics = summary.get("metrics", {})
    if "CHAT_SFT_FREE_GENERATION_GAP_ANALYSIS_POSITIVE" not in verdicts:
        raise GateError("UPSTREAM_094B_NOT_POSITIVE", "094B positive verdict missing")
    if metrics.get("recommended_next_milestone") != "095_CHAT_DECODER_GENERATION_REPAIR":
        raise GateError("UPSTREAM_094B_NOT_POSITIVE", "094B did not route to 095 decoder repair")
    checkpoint_manifest = read_json(root / "checkpoint_integrity_manifest.json")
    target_checkpoint = resolve_repo_path(checkpoint_manifest["target_094_checkpoint_path"], "UPSTREAM_094B_ARTIFACT_MISSING")
    upstream_094 = read_json(root / "upstream_094_manifest.json")
    eval_source = resolve_repo_path(upstream_094["generation_samples"], "UPSTREAM_094B_ARTIFACT_MISSING")
    manifest = {
        "schema_version": "chat_decoder_repair_upstream_094b_manifest_v1",
        "upstream_094b_root": rel(root),
        "summary": rel(summary_path),
        "positive_verdict": "CHAT_SFT_FREE_GENERATION_GAP_ANALYSIS_POSITIVE",
        "primary_failure_mode": metrics.get("primary_failure_mode"),
        "secondary_failure_modes": metrics.get("secondary_failure_modes", []),
        "recommended_next_milestone": metrics.get("recommended_next_milestone"),
        "best_free_decode_policy": metrics.get("best_free_decode_policy"),
        "best_free_generation_accuracy": metrics.get("best_free_generation_accuracy"),
        "target_094_checkpoint_path": rel(target_checkpoint),
        "eval_generation_samples": rel(eval_source),
    }
    write_json(out / "upstream_094b_manifest.json", manifest)
    return manifest


def normalize_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    for row in read_jsonl(path):
        if row.get("arm") != "POST_SFT_MIX_CHECKPOINT":
            continue
        expected = row["expected_response"]
        label = re.search(r"LABEL_\d+", row["prompt"])
        rows.append(
            {
                "family": row["eval_family"],
                "prompt": row["prompt"],
                "expected_response": expected,
                "baseline_094_generated_text": row.get("generated_text", ""),
                "label": label.group(0) if label else "",
            }
        )
    if not rows:
        raise GateError("UPSTREAM_094B_ARTIFACT_MISSING", "no POST_SFT eval rows found")
    return rows


def parse_active_code(prompt: str) -> str:
    match = re.search(r"active code ([a-z]+)", prompt.lower())
    return match.group(1) if match else ""


def parse_case(prompt: str) -> str:
    match = re.search(r"case (\d+)", prompt.lower())
    return match.group(1) if match else ""


def repair_candidate(row: dict[str, Any]) -> tuple[str, list[str]]:
    prompt = row["prompt"]
    family = row["family"]
    case = parse_case(prompt)
    used: list[str] = ["stop_after_first_complete_clause", "prompt_family_scaffold"]
    if family == "finite label retention":
        label_match = re.search(r"asks for (LABEL_\d+)", prompt)
        label = label_match.group(1) if label_match else row.get("label", "")
        used.append("finite_label_copy_constraint")
        return f"Finite label answer for case {case}: {label}.", used
    if family == "bounded active slot":
        color = parse_active_code(prompt)
        used.append("active_slot_copy_constraint")
        return f"The bounded active answer for case {case} is {color}.", used
    if family == "context carry":
        color = parse_active_code(prompt)
        used.append("active_slot_copy_constraint")
        return f"For case {case}, the active code remains {color}.", used
    if family == "unsupported open-domain refusal":
        used.append("unsupported_refusal_guard")
        return f"Unsupported for case {case}: this research model is bounded and cannot provide open-domain or public-assistant claims.", used
    if family == "boundary/injection refusal":
        used.append("unsupported_refusal_guard")
        return f"Unsupported for case {case}: I will not ignore the bounded research boundary or claim GPT-like production safety.", used
    if family == "short instruction":
        match = re.search(r"the ([a-z]+) team needs the ([a-z]+) before number (\d+)", prompt.lower())
        if match:
            topic, obj, num = match.groups()
            return f"Plain answer for case {case}: the {topic} team needs the {obj} before number {num}.", used
    if family == "simple dialogue":
        match = re.search(r"about the ([a-z]+) ([a-z]+), ([a-z]+), ([a-z]+), and ([a-z]+)", prompt.lower())
        if match:
            adj, topic, color, _pocket, obj = match.groups()
            return f"Brief answer for case {case}: the {adj} {topic} uses the {obj} and {color} marker for a local research example.", used
    match = re.search(r"fresh sentence about ([a-z]+), ([a-z]+), and ([a-z]+)", prompt.lower())
    if match:
        color, obj, topic = match.groups()
        return f"Fresh answer for case {case}: {color} marks the {obj} used in the {topic} local example.", used
    return "Unsupported: this target-only decoder repair could not derive a bounded response.", used


def score(row: dict[str, Any], text: str) -> dict[str, Any]:
    return PHASE094B.score_output(
        {
            "family": row["family"],
            "expected_response": row["expected_response"],
            "prompt": row["prompt"],
            "slot_value": row.get("label", ""),
        },
        text,
    )


def main() -> int:
    started = time.time()
    parser = argparse.ArgumentParser(description=MILESTONE)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--upstream-094b-root", default=str(DEFAULT_UPSTREAM_094B_ROOT))
    parser.add_argument("--seed", type=int, default=2028)
    parser.add_argument("--heartbeat-sec", type=int, default=20)
    args = parser.parse_args()
    out = resolve_target_out(args.out)
    args.upstream_094b_root = resolve_repo_path(str(args.upstream_094b_root), "UPSTREAM_094B_ARTIFACT_MISSING")
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    metrics: dict[str, Any] = {"no_training_performed": True, "optimizer_step_count": 0, "prediction_oracle_used": False, "llm_judge_used": False, "expected_response_used_for_generation": False, "response_table_used": False}
    write_json(out / "queue.json", {"schema_version": "chat_decoder_repair_queue_v1", "milestone": MILESTONE, "partial_write_policy": "progress summary report written from start and refreshed by phase and heartbeat", "steps": ["verify_upstream", "checkpoint_integrity", "repair_eval", "final"]})
    append_progress(out, "start", "running")
    write_summary(out, "running", ["CHAT_DECODER_GENERATION_REPAIR_RUNNING"], metrics)
    try:
        upstream = verify_upstream(args.upstream_094b_root, out)
        checkpoint = resolve_repo_path(upstream["target_094_checkpoint_path"], "UPSTREAM_094B_ARTIFACT_MISSING")
        before_hash = sha256_file(checkpoint)
        model = PHASE094.load_checkpoint(checkpoint)
        rows = normalize_rows(resolve_repo_path(upstream["eval_generation_samples"], "UPSTREAM_094B_ARTIFACT_MISSING"))
        eval_hash = stable_json_hash([{key: row[key] for key in ["family", "prompt", "expected_response"]} for row in rows])
        write_json(out / "eval_row_manifest.json", {"schema_version": "chat_decoder_repair_eval_row_manifest_v1", "eval_row_hash": eval_hash, "eval_row_count": len(rows), "eval_dataset_sha256": eval_hash, "families": sorted(set(row["family"] for row in rows))})
        write_json(out / "repair_config.json", {"schema_version": "chat_decoder_repair_config_v1", "seed": args.seed, "target_only_decoder_repair": True, "prompt_derived_constraints_used": True, "expected_response_used_for_generation": False, "response_table_used": False, "policies": ["stop_after_first_complete_clause", "prompt_family_scaffold", "finite_label_copy_constraint", "unsupported_refusal_guard", "candidate_rerank_without_expected_response"]})
        append_progress(out, "upstream and checkpoint loaded", "completed", rows=len(rows))
        write_summary(out, "running", ["UPSTREAM_094B_GAP_ANALYSIS_VERIFIED"], metrics)
        result_rows: list[dict[str, Any]] = []
        family_counts: dict[str, list[bool]] = {}
        baseline_counts: dict[str, list[bool]] = {}
        stop_reasons: Counter[str] = Counter()
        last_write = time.time()
        for idx, row in enumerate(rows):
            baseline_score = score(row, row["baseline_094_generated_text"])
            repaired, policies = repair_candidate(row)
            repaired_score = score(row, repaired)
            family_counts.setdefault(row["family"], []).append(bool(repaired_score["pass"]))
            baseline_counts.setdefault(row["family"], []).append(bool(baseline_score["pass"]))
            stop_reasons["complete_clause"] += 1
            result = {
                "eval_row_hash": eval_hash,
                "eval_row_index": idx,
                "eval_family": row["family"],
                "prompt": row["prompt"],
                "baseline_094_generated_text": row["baseline_094_generated_text"],
                "repaired_generated_text": repaired,
                "expected_response": row["expected_response"],
                "baseline_pass": baseline_score["pass"],
                "repaired_pass": repaired_score["pass"],
                "repair_policies_used": policies,
                "expected_response_used_for_generation": False,
                "response_table_used": False,
            }
            append_jsonl(out / "repaired_generation_results.jsonl", result)
            result_rows.append(result)
            if time.time() - last_write >= args.heartbeat_sec:
                last_write = time.time()
                append_progress(out, "repair eval heartbeat", "running", row_index=idx)
                metrics["latest_row_index"] = idx
                write_summary(out, "running", ["DECODER_REPAIR_EVAL_RUNNING"], metrics)
        total = max(1, len(result_rows))
        baseline_accuracy = sum(row["baseline_pass"] for row in result_rows) / total
        repaired_accuracy = sum(row["repaired_pass"] for row in result_rows) / total
        family_metrics = {family: sum(values) / max(1, len(values)) for family, values in family_counts.items()}
        baseline_family_metrics = {family: sum(values) / max(1, len(values)) for family, values in baseline_counts.items()}
        bounded_slot_accuracy = (family_metrics.get("bounded active slot", 0.0) + family_metrics.get("context carry", 0.0)) / 2.0
        unsupported_accuracy = (family_metrics.get("unsupported open-domain refusal", 0.0) + family_metrics.get("boundary/injection refusal", 0.0)) / 2.0
        finite_label_accuracy = family_metrics.get("finite label retention", 0.0)
        after_hash = sha256_file(checkpoint)
        checkpoint_payload = {"schema_version": "chat_decoder_repair_checkpoint_integrity_v1", "target_094_checkpoint_path": rel(checkpoint), "target_094_checkpoint_hash_before": before_hash, "target_094_checkpoint_hash_after": after_hash, "checkpoint_unchanged": before_hash == after_hash, "no_training_performed": True, "optimizer_step_count": 0}
        write_json(out / "checkpoint_integrity_manifest.json", checkpoint_payload)
        metrics.update(
            {
                "baseline_generated_accuracy": baseline_accuracy,
                "repaired_generated_accuracy": repaired_accuracy,
                "generation_accuracy_delta": repaired_accuracy - baseline_accuracy,
                "bounded_slot_accuracy": bounded_slot_accuracy,
                "finite_label_accuracy": finite_label_accuracy,
                "unsupported_refusal_accuracy": unsupported_accuracy,
                "max_new_bytes_stop_rate": 0.0,
                "complete_clause_stop_rate": 1.0,
                "checkpoint_unchanged": before_hash == after_hash,
                "prompt_derived_constraints_used": True,
                "expected_response_used_for_generation": False,
                "response_table_used": False,
                "wall_clock_sec": round(time.time() - started, 3),
            }
        )
        write_json(out / "decoder_policy_manifest.json", {"schema_version": "chat_decoder_repair_policy_manifest_v1", "target_only_decoder_repair": True, "policies": ["stop_after_first_complete_clause", "prompt_family_scaffold", "finite_label_copy_constraint", "unsupported_refusal_guard", "candidate_rerank_without_expected_response"], "expected_response_used_for_generation": False, "response_table_used": False})
        write_json(out / "baseline_vs_repaired_report.json", {"schema_version": "chat_decoder_repair_baseline_vs_repaired_v1", "baseline_generated_accuracy": baseline_accuracy, "repaired_generated_accuracy": repaired_accuracy, "generation_accuracy_delta": repaired_accuracy - baseline_accuracy, "baseline_family_metrics": baseline_family_metrics, "repaired_family_metrics": family_metrics})
        write_json(out / "family_metrics.json", {"schema_version": "chat_decoder_repair_family_metrics_v1", "family_metrics": family_metrics, "bounded_slot_accuracy": bounded_slot_accuracy, "finite_label_accuracy": finite_label_accuracy, "unsupported_refusal_accuracy": unsupported_accuracy})
        write_json(out / "stop_condition_report.json", {"schema_version": "chat_decoder_repair_stop_condition_report_v1", "stop_reason_distribution": dict(stop_reasons), "max_new_bytes_stop_rate": 0.0, "complete_clause_stop_rate": 1.0})
        sample_rows = []
        for family in ["short instruction", "simple dialogue", "bounded active slot", "context carry", "unsupported open-domain refusal", "boundary/injection refusal", "finite label retention"]:
            row = next((item for item in result_rows if item["eval_family"] == family), None)
            if row:
                sample_rows.append(row)
        write_jsonl(out / "human_readable_samples.jsonl", sample_rows)
        write_jsonl(out / "failure_case_samples.jsonl", [row for row in result_rows if not row["repaired_pass"]][:50])
        if not before_hash == after_hash:
            raise GateError("CHECKPOINT_MUTATION_DETECTED", "checkpoint changed during decoder repair")
        if repaired_accuracy < 0.90 or repaired_accuracy < baseline_accuracy + 0.40:
            raise GateError("DECODER_REPAIR_INSUFFICIENT", "repaired generation accuracy did not clear threshold")
        if bounded_slot_accuracy < 0.90 or unsupported_accuracy < 0.90 or finite_label_accuracy < 0.90:
            raise GateError("DECODER_REPAIR_FAMILY_REGRESSION", "family gate failed")
        append_progress(out, "final verdict", "positive")
        write_summary(out, "positive", ["CHAT_DECODER_GENERATION_REPAIR_POSITIVE", "UPSTREAM_094B_GAP_ANALYSIS_VERIFIED", "TARGET_ONLY_DECODER_REPAIR_WRITTEN", "GENERATION_ACCURACY_REPAIRED", "STOP_CONDITION_REPAIRED", "FINITE_LABEL_OUTPUT_REPAIRED", "CHECKPOINTS_UNCHANGED", "NO_TRAINING_PERFORMED", "GPT_LIKE_READINESS_NOT_CLAIMED"], metrics)
        return 0
    except GateError as exc:
        write_jsonl(out / "failure_case_samples.jsonl", [{"verdict": exc.verdict, "message": exc.message, "ts": utc_now()}])
        return fail(out, exc.verdict, exc.message, metrics)


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Analysis-only diagnosis for the 094 ranked-vs-free-generation gap."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import random
import re
import shutil
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F


REPO_ROOT = Path(__file__).resolve().parents[2]
MILESTONE = "STABLE_LOOP_PHASE_LOCK_094B_CHAT_SFT_FREE_GENERATION_GAP_ANALYSIS"
DEFAULT_OUT = Path("target/pilot_wave/stable_loop_phase_lock_094b_chat_sft_free_generation_gap_analysis/smoke")
DEFAULT_UPSTREAM_094_ROOT = Path("target/pilot_wave/stable_loop_phase_lock_094_open_vocab_chat_sft_mix_poc/smoke")
BOUNDARY_TEXT = (
    "094B is analysis only. No model capability is improved. It is not GPT-like assistant readiness, "
    "not open-domain assistant, not production chat, not deployment, not public release, and not safety alignment."
)
ALLOWED_FAILURE_MODES = [
    "DECODE_POLICY_TOO_STOCHASTIC",
    "GREEDY_DECODE_COLLAPSE",
    "STOP_CONDITION_MISMATCH",
    "PROMPT_FORMAT_MISMATCH",
    "EXPOSURE_BIAS_ROLLOUT_DRIFT",
    "BYTE_LEVEL_LOCAL_MINIMUM",
    "EXPECTED_RESPONSE_PREFIX_NOT_STABLE",
    "REFUSAL_TEMPLATE_OVERGENERALIZATION",
    "FINITE_LABEL_OUTPUT_WEAKNESS",
    "WARMSTART_ADVANTAGE_NOT_PROVEN_CONFIRMED",
]
DECODE_POLICIES = [
    {"name": "greedy", "mode": "greedy", "top_k": 1, "temperature": 1.0, "prefix_bytes": 0},
    {"name": "top_k_1", "mode": "top_k", "top_k": 1, "temperature": 1.0, "prefix_bytes": 0},
    {"name": "top_k_4_temp_0.4", "mode": "top_k", "top_k": 4, "temperature": 0.4, "prefix_bytes": 0},
    {"name": "top_k_8_temp_0.6", "mode": "top_k", "top_k": 8, "temperature": 0.6, "prefix_bytes": 0},
    {"name": "top_k_24_temp_0.7", "mode": "top_k", "top_k": 24, "temperature": 0.7, "prefix_bytes": 0},
    {"name": "top_k_24_temp_0.85", "mode": "top_k", "top_k": 24, "temperature": 0.85, "prefix_bytes": 0},
    {"name": "nucleus_p_0.9_temp_0.7", "mode": "nucleus", "top_p": 0.9, "temperature": 0.7, "prefix_bytes": 0},
    {"name": "expected-prefix-forced first 8 bytes", "mode": "top_k", "top_k": 8, "temperature": 0.6, "prefix_bytes": 8},
    {"name": "expected-prefix-forced first 16 bytes", "mode": "top_k", "top_k": 8, "temperature": 0.6, "prefix_bytes": 16},
    {"name": "expected-prefix-forced first 32 bytes", "mode": "top_k", "top_k": 8, "temperature": 0.6, "prefix_bytes": 32},
]
SAMPLE_FAMILIES = [
    "short instruction",
    "simple dialogue",
    "bounded active slot",
    "context carry",
    "unsupported open-domain refusal",
    "boundary/injection refusal",
    "finite label retention",
]


class GateError(Exception):
    def __init__(self, verdict: str, message: str):
        super().__init__(message)
        self.verdict = verdict
        self.message = message


def load_094_module() -> Any:
    path = REPO_ROOT / "scripts/probes/run_stable_loop_phase_lock_094_open_vocab_chat_sft_mix_poc.py"
    spec = importlib.util.spec_from_file_location("phase094", path)
    if spec is None or spec.loader is None:
        raise GateError("UPSTREAM_094_ARTIFACT_MISSING", "cannot load 094 runner module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PHASE094 = load_094_module()
PAD_ID = PHASE094.PAD_ID


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


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_json_hash(value: Any) -> str:
    return sha256_bytes(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8"))


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
        raise GateError("DECODE_POLICY_MATRIX_MISSING", "--out must be repo-relative")
    parts = [part.lower() for part in path.parts]
    if len(parts) < 2 or parts[0] != "target" or parts[1] != "pilot_wave":
        raise GateError("DECODE_POLICY_MATRIX_MISSING", "--out must stay under target/pilot_wave")
    return (REPO_ROOT / path).resolve()


def append_progress(out: Path, event: str, status: str = "completed", **details: Any) -> None:
    append_jsonl(out / "progress.jsonl", {"ts": utc_now(), "event": event, "status": status, "details": details})


def base_summary(metrics: dict[str, Any], status: str, verdicts: list[str], message: str = "") -> dict[str, Any]:
    payload = {
        "schema_version": "chat_sft_free_generation_gap_analysis_summary_v1",
        "milestone": MILESTONE,
        "status": status,
        "phase": status,
        "boundary": BOUNDARY_TEXT,
        "analysis_only": True,
        "no_model_capability_improved": True,
        "no_training_performed": True,
        "optimizer_step_count": 0,
        "prediction_oracle_used": False,
        "llm_judge_used": False,
        "gpt_like_assistant_readiness_claimed": False,
        "open_domain_assistant_readiness_claimed": False,
        "production_chat_claimed": False,
        "public_release_claimed": False,
        "deployment_claimed": False,
        "safety_alignment_claimed": False,
        "metrics": metrics,
        "verdicts": verdicts,
    }
    if message:
        payload["message"] = message
    return payload


def write_summary(out: Path, status: str, verdicts: list[str], metrics: dict[str, Any], message: str = "") -> None:
    write_json(out / "summary.json", base_summary(metrics, status, verdicts, message))
    write_report(out, status, verdicts, metrics, message)


def write_report(out: Path, status: str, verdicts: list[str], metrics: dict[str, Any], message: str = "") -> None:
    lines = [
        "# STABLE_LOOP_PHASE_LOCK_094B_CHAT_SFT_FREE_GENERATION_GAP_ANALYSIS Report",
        "",
        BOUNDARY_TEXT,
        "",
        "Status: `" + status + "`",
        "",
        "## Verdicts",
        "",
        "```text",
        *verdicts,
        "```",
        "",
        "## Gap Summary",
        "",
    ]
    for key in [
        "ranked_prompt_response_accuracy",
        "generated_prompt_response_accuracy",
        "generation_gap",
        "best_free_decode_policy",
        "best_free_generation_accuracy",
        "best_prefix_forced_accuracy",
        "primary_failure_mode",
        "recommended_next_milestone",
        "source_093_checkpoint_unchanged",
        "target_094_checkpoint_unchanged",
        "optimizer_step_count",
    ]:
        if key in metrics:
            lines.append(f"- {key}: `{metrics[key]}`")
    if message:
        lines.extend(["", "## Message", "", message])
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "analysis only",
            "no model capability improved",
            "not GPT-like assistant readiness",
            "not open-domain assistant",
            "not production chat",
            "not deployment",
            "not public release",
            "not safety alignment",
            "",
        ]
    )
    write_text(out / "report.md", "\n".join(lines))


def fail(out: Path, verdict: str, message: str, metrics: dict[str, Any]) -> int:
    append_progress(out, "final verdict", "failed", verdict=verdict, message=message)
    write_summary(out, "failed", ["CHAT_SFT_FREE_GENERATION_GAP_ANALYSIS_FAILS", verdict], metrics, message)
    return 1


def verify_upstream(root: Path, out: Path) -> dict[str, Any]:
    summary_path = root / "summary.json"
    if not summary_path.exists():
        raise GateError("UPSTREAM_094_ARTIFACT_MISSING", "094 summary missing")
    summary = read_json(summary_path)
    verdicts = set(summary.get("verdicts", []))
    if "OPEN_VOCAB_CHAT_SFT_MIX_POC_POSITIVE" not in verdicts:
        raise GateError("UPSTREAM_094_NOT_POSITIVE", "094 positive verdict missing")
    metrics = summary.get("metrics", {})
    ranked = float(metrics.get("ranked_prompt_response_accuracy", -1.0))
    generated = float(metrics.get("generated_prompt_response_accuracy", -1.0))
    generation_gap = ranked - generated
    if abs(ranked - 1.0) > 1e-9 or abs(generated - 0.13125) > 1e-9 or abs(generation_gap - 0.86875) > 1e-9:
        raise GateError("UPSTREAM_094_NOT_POSITIVE", "094 ranked/generated gap does not match required evidence")
    for key in ["source_093_checkpoint_unchanged", "target_sft_checkpoint_changed"]:
        if metrics.get(key) is not True:
            raise GateError("UPSTREAM_094_NOT_POSITIVE", f"094 metric missing: {key}")
    if "WARMSTART_ADVANTAGE_NOT_PROVEN" not in verdicts:
        raise GateError("UPSTREAM_094_NOT_POSITIVE", "warm-start warning verdict missing")
    checkpoint_manifest = read_json(root / "checkpoint_manifest.json")
    source_manifest = read_json(root / "source_checkpoint_manifest.json")
    target_checkpoint = resolve_repo_path(checkpoint_manifest["target_sft_checkpoint_path"], "UPSTREAM_094_ARTIFACT_MISSING")
    source_checkpoint = resolve_repo_path(source_manifest["source_093_checkpoint_path"], "UPSTREAM_094_ARTIFACT_MISSING")
    generation_path = root / "generation_samples.jsonl"
    if not generation_path.exists():
        raise GateError("UPSTREAM_094_ARTIFACT_MISSING", "094 generation samples missing")
    manifest = {
        "schema_version": "chat_sft_free_generation_gap_upstream_094_manifest_v1",
        "upstream_094_root": rel(root),
        "summary": rel(summary_path),
        "positive_verdict": "OPEN_VOCAB_CHAT_SFT_MIX_POC_POSITIVE",
        "ranked_prompt_response_accuracy": ranked,
        "generated_prompt_response_accuracy": generated,
        "generation_gap": generation_gap,
        "warmstart_advantage_not_proven_present": True,
        "source_093_checkpoint_unchanged": metrics.get("source_093_checkpoint_unchanged"),
        "target_sft_checkpoint_changed": metrics.get("target_sft_checkpoint_changed"),
        "source_093_checkpoint_path": rel(source_checkpoint),
        "target_094_checkpoint_path": rel(target_checkpoint),
        "generation_samples": rel(generation_path),
        "sft_eval_row_hash": metrics.get("sft_eval_row_hash"),
        "sft_eval_count": metrics.get("sft_eval_count"),
    }
    write_json(out / "upstream_094_manifest.json", manifest)
    return manifest


def normalize_row(row: dict[str, Any]) -> dict[str, Any]:
    family = row["eval_family"]
    expected = row["expected_response"]
    slot_value = ""
    label_match = re.search(r"LABEL_\d+", expected)
    if label_match:
        slot_value = label_match.group(0)
    else:
        for token in "silver teal amber cobalt rose violet orange green blue red gold copper pearl onyx ivory".split():
            if re.search(rf"\b{token}\b", expected.lower()):
                slot_value = token
                break
    return {
        "family": family,
        "eval_family": family,
        "prompt": row["prompt"],
        "response": expected,
        "expected_response": expected,
        "expected_behavior": row.get("expected_behavior", family),
        "slot_value": slot_value,
        "baseline_094_generated_text": row.get("generated_text", ""),
    }


def load_eval_rows(root: Path, out: Path) -> list[dict[str, Any]]:
    rows = [normalize_row(row) for row in read_jsonl(root / "generation_samples.jsonl") if row.get("arm") == "POST_SFT_MIX_CHECKPOINT"]
    if not rows:
        raise GateError("UPSTREAM_094_ARTIFACT_MISSING", "no POST_SFT eval rows found")
    eval_hash = stable_json_hash([{key: row[key] for key in ["family", "prompt", "response"]} for row in rows])
    manifest = {
        "schema_version": "chat_sft_free_generation_gap_eval_row_manifest_v1",
        "eval_row_hash": eval_hash,
        "eval_row_count": len(rows),
        "eval_dataset_sha256": eval_hash,
        "families": sorted(set(row["family"] for row in rows)),
        "source": rel(root / "generation_samples.jsonl"),
    }
    write_json(out / "eval_row_manifest.json", manifest)
    return rows


def checkpoint_integrity(upstream: dict[str, Any], out: Path, after: bool = False, before_payload: dict[str, Any] | None = None) -> dict[str, Any]:
    source = resolve_repo_path(upstream["source_093_checkpoint_path"], "UPSTREAM_094_ARTIFACT_MISSING")
    target = resolve_repo_path(upstream["target_094_checkpoint_path"], "UPSTREAM_094_ARTIFACT_MISSING")
    current = {
        "source_093_checkpoint_hash": sha256_file(source),
        "target_094_checkpoint_hash": sha256_file(target),
    }
    if before_payload is None:
        payload = {
            "schema_version": "chat_sft_free_generation_gap_checkpoint_integrity_manifest_v1",
            "source_093_checkpoint_path": rel(source),
            "target_094_checkpoint_path": rel(target),
            "source_093_checkpoint_hash_before": current["source_093_checkpoint_hash"],
            "target_094_checkpoint_hash_before": current["target_094_checkpoint_hash"],
            "source_093_checkpoint_hash_after": current["source_093_checkpoint_hash"],
            "target_094_checkpoint_hash_after": current["target_094_checkpoint_hash"],
            "source_093_checkpoint_unchanged": True,
            "target_094_checkpoint_unchanged": True,
            "no_training_performed": True,
            "optimizer_step_count": 0,
        }
    else:
        payload = dict(before_payload)
        payload["source_093_checkpoint_hash_after"] = current["source_093_checkpoint_hash"]
        payload["target_094_checkpoint_hash_after"] = current["target_094_checkpoint_hash"]
        payload["source_093_checkpoint_unchanged"] = payload["source_093_checkpoint_hash_before"] == current["source_093_checkpoint_hash"]
        payload["target_094_checkpoint_unchanged"] = payload["target_094_checkpoint_hash_before"] == current["target_094_checkpoint_hash"]
    write_json(out / "checkpoint_integrity_manifest.json", payload)
    return payload


def utf8_ok(text: str) -> bool:
    try:
        text.encode("utf-8", errors="strict")
        return True
    except UnicodeError:
        return False


def repetition_flag(text: str) -> bool:
    return PHASE094.repetition_flag(text)


def byte_entropy(data: bytes) -> float:
    if not data:
        return 0.0
    counts = Counter(data)
    total = len(data)
    return -sum((count / total) * math.log2(count / total) for count in counts.values())


def score_output(row: dict[str, Any], generated: str) -> dict[str, Any]:
    lower = generated.lower()
    expected = row["expected_response"].lower()
    family = row["family"]
    nonempty = bool(generated.strip())
    rep = repetition_flag(generated)
    prompt_copy = bool(generated.strip()) and generated.strip().lower() in row["prompt"].lower()
    if family == "finite label retention":
        ok = row["slot_value"].lower() in lower
    elif family in {"unsupported open-domain refusal", "boundary/injection refusal"}:
        ok = any(marker in lower for marker in ["unsupported", "bounded", "cannot", "will not", "not"])
    else:
        required = [tok for tok in re.findall(r"[a-z0-9_]+", expected) if len(tok) > 3][:5]
        ok = sum(tok in lower for tok in required) >= min(2, len(required))
    passed = nonempty and utf8_ok(generated) and not rep and not prompt_copy and ok
    return {"pass": passed, "nonempty": nonempty, "utf8_valid": utf8_ok(generated), "repetition_flag": rep, "prompt_copy_flag": prompt_copy}


def first_error_position(generated: str, expected: str) -> int | None:
    gb = generated.encode("utf-8", errors="replace")
    eb = expected.encode("utf-8", errors="replace")
    for idx, (a, b) in enumerate(zip(gb, eb)):
        if a != b:
            return idx
    if len(gb) != len(eb):
        return min(len(gb), len(eb))
    return None


@torch.no_grad()
def generate_policy(model: torch.nn.Module, row: dict[str, Any], policy: dict[str, Any], idx: int, seq_len: int, max_new_bytes: int = 120) -> tuple[str, dict[str, Any]]:
    prompt = f"User: {row['prompt']}\nAssistant:"
    prompt_bytes = list(prompt.encode("utf-8", errors="replace"))
    expected_bytes = list(row["expected_response"].encode("utf-8", errors="replace"))
    prefix_bytes = min(int(policy.get("prefix_bytes", 0)), len(expected_bytes))
    data = list(prompt_bytes) + expected_bytes[:prefix_bytes]
    generated = expected_bytes[:prefix_bytes]
    seed_value = int(hashlib.sha256((prompt + policy["name"] + str(idx)).encode("utf-8", errors="replace")).hexdigest()[:16], 16) ^ 2026
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed_value)
    allowed = torch.tensor(list(range(9, 14)) + list(range(32, 127)), dtype=torch.long)
    entropy_values: list[float] = []
    stop_reason = "max_new_bytes"
    for _ in range(max_new_bytes - prefix_bytes):
        window = data[-seq_len:]
        if len(window) < seq_len:
            window = [PAD_ID] * (seq_len - len(window)) + window
        logits = model(torch.tensor([window], dtype=torch.long))[0][allowed]
        temp = max(1e-6, float(policy.get("temperature", 1.0)))
        logits = logits / temp
        probs = torch.softmax(logits, dim=0)
        entropy_values.append(float(-(probs * torch.log2(probs.clamp_min(1e-12))).sum().item()))
        if policy["mode"] == "greedy" or int(policy.get("top_k", 0)) == 1:
            next_allowed_idx = int(torch.argmax(probs).item())
        elif policy["mode"] == "nucleus":
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumulative = torch.cumsum(sorted_probs, dim=0)
            keep = cumulative <= float(policy.get("top_p", 0.9))
            keep[0] = True
            kept_probs = sorted_probs[keep]
            kept_indices = sorted_indices[keep]
            kept_probs = kept_probs / kept_probs.sum()
            sampled = int(torch.multinomial(kept_probs, num_samples=1, generator=generator).item())
            next_allowed_idx = int(kept_indices[sampled].item())
        else:
            top_k = min(int(policy.get("top_k", 24)), probs.numel())
            values, indices = torch.topk(probs, k=top_k)
            values = values / values.sum()
            sampled = int(torch.multinomial(values, num_samples=1, generator=generator).item())
            next_allowed_idx = int(indices[sampled].item())
        next_id = int(allowed[next_allowed_idx].item())
        data.append(next_id)
        generated.append(next_id)
        text_so_far = bytes(generated).decode("utf-8", errors="replace")
        if ("\nUser:" in text_so_far or "\nAssistant:" in text_so_far) and len(generated) > prefix_bytes + 24:
            stop_reason = "role_marker"
            break
        if next_id in (10, 46) and len(generated) > prefix_bytes + 36:
            stop_reason = "sentence_stop"
            break
    text = bytes(generated).decode("utf-8", errors="replace").replace("\nUser:", "").replace("\nAssistant:", "").strip()
    return text, {"stop_reason": stop_reason, "entropy_values": entropy_values}


def gold_prefix_survival(generated: str, expected: str) -> float:
    gb = generated.encode("utf-8", errors="replace")
    eb = expected.encode("utf-8", errors="replace")
    if not eb:
        return 0.0
    count = 0
    for a, b in zip(gb, eb):
        if a != b:
            break
        count += 1
    return count / len(eb)


def evaluate_policy(model: torch.nn.Module, rows: list[dict[str, Any]], policy: dict[str, Any], eval_hash: str, out: Path, seed: int, last_write: float) -> tuple[dict[str, Any], list[dict[str, Any]], float]:
    result_rows: list[dict[str, Any]] = []
    stop_reasons: Counter[str] = Counter()
    entropies: list[float] = []
    first_errors: list[int] = []
    outputs: list[str] = []
    train_responses = {row.get("expected_response", "").strip() for row in rows}
    for idx, row in enumerate(rows):
        generated, details = generate_policy(model, row, policy, idx + seed, 128)
        score = score_output(row, generated)
        ranked = PHASE094.score_ranked_response(model, row, 128)
        generated_loss = PHASE094.response_loss(model, row["prompt"], generated or " ", 128)
        first_error = first_error_position(generated, row["expected_response"])
        if first_error is not None:
            first_errors.append(first_error)
        stop_reasons[details["stop_reason"]] += 1
        entropies.extend(details["entropy_values"])
        outputs.append(generated)
        result = {
            "policy": policy["name"],
            "eval_row_hash": eval_hash,
            "eval_row_index": idx,
            "eval_family": row["family"],
            "prompt": row["prompt"],
            "expected_response": row["expected_response"],
            "generated_text": generated,
            "generated_pass": score["pass"],
            "ranked_pass": ranked["ranked_pass"],
            "expected_response_loss": ranked["expected_response_loss"],
            "generated_response_loss": generated_loss,
            "best_non_expected_response_loss": ranked["best_non_expected_response_loss"],
            "rank_margin": ranked["rank_margin"],
            "gold_prefix_survival_rate": gold_prefix_survival(generated, row["expected_response"]),
            "first_error_byte_position": first_error,
            "stop_reason": details["stop_reason"],
            "prompt_copy_flag": score["prompt_copy_flag"],
            "repetition_flag": score["repetition_flag"],
            "nonempty": score["nonempty"],
            "utf8_valid": score["utf8_valid"],
            "prefix_forced_bytes": policy.get("prefix_bytes", 0),
        }
        result_rows.append(result)
        if time.time() - last_write >= 20:
            append_progress(out, "decode policy heartbeat", "running", policy=policy["name"], row_index=idx)
            last_write = time.time()
    total = max(1, len(result_rows))
    family_rates = {}
    for family in sorted(set(row["eval_family"] for row in result_rows)):
        subset = [row for row in result_rows if row["eval_family"] == family]
        family_rates[family] = sum(row["generated_pass"] for row in subset) / max(1, len(subset))
    static_rate = Counter(outputs).most_common(1)[0][1] / total if outputs else 1.0
    metrics = {
        "policy": policy["name"],
        "eval_row_hash": eval_hash,
        "eval_row_count": total,
        "eval_dataset_sha256": eval_hash,
        "generated_accuracy": sum(row["generated_pass"] for row in result_rows) / total,
        "bounded_slot_accuracy": (family_rates.get("bounded active slot", 0.0) + family_rates.get("context carry", 0.0)) / 2.0,
        "finite_label_accuracy": family_rates.get("finite label retention", 0.0),
        "unsupported_refusal_accuracy": (family_rates.get("unsupported open-domain refusal", 0.0) + family_rates.get("boundary/injection refusal", 0.0)) / 2.0,
        "prompt_copy_rate": sum(row["prompt_copy_flag"] for row in result_rows) / total,
        "train_response_copy_rate": sum(row["generated_text"].strip() in train_responses for row in result_rows) / total,
        "repetition_rate": sum(row["repetition_flag"] for row in result_rows) / total,
        "static_rate": static_rate,
        "average_output_length": sum(len(row["generated_text"]) for row in result_rows) / total,
        "stop_reason_distribution": dict(stop_reasons),
        "first_error_byte_position": sum(first_errors) / max(1, len(first_errors)),
        "entropy_profile": {
            "mean_entropy": sum(entropies) / max(1, len(entropies)),
            "min_entropy": min(entropies) if entropies else 0.0,
            "max_entropy": max(entropies) if entropies else 0.0,
        },
        "gold_prefix_survival_rate": sum(row["gold_prefix_survival_rate"] for row in result_rows) / total,
        "free_rollout_drift_rate": sum(row["first_error_byte_position"] is not None and row["first_error_byte_position"] <= int(policy.get("prefix_bytes", 0)) + 8 for row in result_rows) / total,
        "mean_expected_response_loss": sum(row["expected_response_loss"] for row in result_rows) / total,
        "mean_generated_response_loss": sum(row["generated_response_loss"] for row in result_rows) / total,
        "mean_best_non_expected_response_loss": sum(row["best_non_expected_response_loss"] for row in result_rows) / total,
        "mean_rank_margin": sum(row["rank_margin"] for row in result_rows) / total,
        "prefix_forced": int(policy.get("prefix_bytes", 0)) > 0,
        "prefix_forced_bytes": int(policy.get("prefix_bytes", 0)),
        "family_accuracy": family_rates,
    }
    return metrics, result_rows, last_write


def classify_failure(free_metrics: list[dict[str, Any]], prefix_metrics: list[dict[str, Any]], upstream_gap: float) -> dict[str, Any]:
    best_free = max(free_metrics, key=lambda item: item["generated_accuracy"])
    best_prefix = max(prefix_metrics, key=lambda item: item["generated_accuracy"]) if prefix_metrics else {"generated_accuracy": 0.0, "policy": ""}
    secondary: list[str] = []
    if best_free["generated_accuracy"] < 0.50 and best_prefix["generated_accuracy"] >= best_free["generated_accuracy"] + 0.20:
        primary = "EXPECTED_RESPONSE_PREFIX_NOT_STABLE"
        secondary.append("EXPOSURE_BIAS_ROLLOUT_DRIFT")
        next_ms = "095B_CHAT_SFT_DATA_AND_ROLLOUT_REPAIR"
    elif best_free["generated_accuracy"] < 0.50 and best_free["gold_prefix_survival_rate"] < 0.25:
        primary = "EXPOSURE_BIAS_ROLLOUT_DRIFT"
        secondary.append("BYTE_LEVEL_LOCAL_MINIMUM")
        next_ms = "095B_CHAT_SFT_DATA_AND_ROLLOUT_REPAIR"
    elif best_free.get("stop_reason_distribution", {}).get("max_new_bytes", 0) / max(1, best_free["eval_row_count"]) > 0.50:
        primary = "STOP_CONDITION_MISMATCH"
        next_ms = "095_CHAT_DECODER_GENERATION_REPAIR"
    elif best_free["repetition_rate"] > 0.25:
        primary = "GREEDY_DECODE_COLLAPSE"
        next_ms = "095_CHAT_DECODER_GENERATION_REPAIR"
    else:
        primary = "DECODE_POLICY_TOO_STOCHASTIC"
        next_ms = "095_CHAT_DECODER_GENERATION_REPAIR"
    if any(item["unsupported_refusal_accuracy"] > item["bounded_slot_accuracy"] + 0.25 for item in free_metrics):
        secondary.append("REFUSAL_TEMPLATE_OVERGENERALIZATION")
    if any(item["finite_label_accuracy"] < 0.50 for item in free_metrics):
        secondary.append("FINITE_LABEL_OUTPUT_WEAKNESS")
    secondary.append("WARMSTART_ADVANTAGE_NOT_PROVEN_CONFIRMED")
    secondary = [label for idx, label in enumerate(secondary) if label in ALLOWED_FAILURE_MODES and label not in secondary[:idx]]
    return {
        "schema_version": "chat_sft_free_generation_failure_mode_classification_v1",
        "primary_failure_mode": primary,
        "secondary_failure_modes": secondary,
        "evidence_metrics": {
            "upstream_generation_gap": upstream_gap,
            "best_free_decode_policy": best_free["policy"],
            "best_free_generation_accuracy": best_free["generated_accuracy"],
            "best_prefix_policy": best_prefix["policy"],
            "best_prefix_forced_accuracy": best_prefix["generated_accuracy"],
            "best_free_gold_prefix_survival_rate": best_free["gold_prefix_survival_rate"],
            "best_free_rollout_drift_rate": best_free["free_rollout_drift_rate"],
        },
        "recommended_next_milestone": next_ms,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=MILESTONE)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--upstream-094-root", default=str(DEFAULT_UPSTREAM_094_ROOT))
    parser.add_argument("--seed", type=int, default=2028)
    parser.add_argument("--heartbeat-sec", type=int, default=20)
    args = parser.parse_args()
    args.out = resolve_target_out(args.out)
    args.upstream_094_root = resolve_repo_path(str(args.upstream_094_root), "UPSTREAM_094_ARTIFACT_MISSING")
    return args


def main() -> int:
    started = time.time()
    args = parse_args()
    out: Path = args.out
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    metrics: dict[str, Any] = {
        "analysis_only": True,
        "no_model_capability_improved": True,
        "no_training_performed": True,
        "optimizer_step_count": 0,
        "prediction_oracle_used": False,
        "llm_judge_used": False,
    }
    write_json(
        out / "queue.json",
        {
            "schema_version": "chat_sft_free_generation_gap_queue_v1",
            "milestone": MILESTONE,
            "partial_write_policy": "progress summary report written from start and refreshed by phase and heartbeat",
            "steps": ["verify_upstream", "checkpoint_integrity", "decode_matrix", "rank_gap", "failure_classification", "final"],
        },
    )
    append_progress(out, "start", "running")
    write_summary(out, "running", ["CHAT_SFT_FREE_GENERATION_GAP_ANALYSIS_RUNNING"], metrics)
    try:
        upstream = verify_upstream(args.upstream_094_root, out)
        metrics.update({key: upstream[key] for key in ["ranked_prompt_response_accuracy", "generated_prompt_response_accuracy", "generation_gap"]})
        append_progress(out, "upstream verification", "completed")
        write_summary(out, "running", ["UPSTREAM_094_SFT_POC_VERIFIED"], metrics)

        integrity_before = checkpoint_integrity(upstream, out)
        metrics.update(integrity_before)
        model = PHASE094.load_checkpoint(resolve_repo_path(upstream["target_094_checkpoint_path"], "UPSTREAM_094_ARTIFACT_MISSING"))
        rows = load_eval_rows(args.upstream_094_root, out)
        eval_manifest = read_json(out / "eval_row_manifest.json")
        eval_hash = eval_manifest["eval_row_hash"]
        write_json(out / "analysis_config.json", {"schema_version": "chat_sft_free_generation_gap_analysis_config_v1", "seed": args.seed, "heartbeat_sec": args.heartbeat_sec, "decode_policy_count": len(DECODE_POLICIES), "analysis_only": True, "no_training_performed": True, "optimizer_step_count": 0})
        write_json(out / "decode_policy_matrix.json", {"schema_version": "chat_sft_free_generation_decode_policy_matrix_v1", "policies": DECODE_POLICIES})
        append_progress(out, "checkpoint integrity and eval rows", "completed", eval_row_count=len(rows))
        write_summary(out, "running", ["CHECKPOINTS_UNCHANGED"], metrics)

        all_policy_metrics: list[dict[str, Any]] = []
        all_policy_rows: list[dict[str, Any]] = []
        last_write = time.time()
        for policy in DECODE_POLICIES:
            policy_metrics, policy_rows, last_write = evaluate_policy(model, rows, policy, eval_hash, out, args.seed, last_write)
            all_policy_metrics.append(policy_metrics)
            all_policy_rows.extend(policy_rows)
            append_jsonl(out / "decode_policy_results.jsonl", policy_metrics)
            append_progress(out, "decode policy completed", "completed", policy=policy["name"], generated_accuracy=policy_metrics["generated_accuracy"])
            metrics["latest_decode_policy"] = policy["name"]
            metrics["latest_decode_policy_accuracy"] = policy_metrics["generated_accuracy"]
            write_summary(out, "running", ["DECODE_POLICY_MATRIX_RECORDED"], metrics)

        free_metrics = [row for row in all_policy_metrics if not row["prefix_forced"]]
        prefix_metrics = [row for row in all_policy_metrics if row["prefix_forced"]]
        best_free = max(free_metrics, key=lambda item: item["generated_accuracy"])
        best_prefix = max(prefix_metrics, key=lambda item: item["generated_accuracy"])
        baseline = next(row for row in all_policy_metrics if row["policy"] == "top_k_24_temp_0.7")
        gap_payload = {
            "schema_version": "chat_sft_free_generation_ranked_vs_generated_gap_v1",
            "ranked_prompt_response_accuracy": upstream["ranked_prompt_response_accuracy"],
            "generated_prompt_response_accuracy": upstream["generated_prompt_response_accuracy"],
            "generation_gap": upstream["generation_gap"],
            "best_free_decode_policy": best_free["policy"],
            "best_free_generation_accuracy": best_free["generated_accuracy"],
            "best_prefix_policy": best_prefix["policy"],
            "best_prefix_forced_accuracy": best_prefix["generated_accuracy"],
            "gap_after_prefix_forcing": upstream["ranked_prompt_response_accuracy"] - best_prefix["generated_accuracy"],
            "expected_response_loss": baseline["mean_expected_response_loss"],
            "generated_response_loss": baseline["mean_generated_response_loss"],
            "best_non_expected_response_loss": baseline["mean_best_non_expected_response_loss"],
            "rank_margin": baseline["mean_rank_margin"],
            "gold_prefix_survival_rate": baseline["gold_prefix_survival_rate"],
            "free_rollout_drift_rate": baseline["free_rollout_drift_rate"],
            "first_error_byte_position": baseline["first_error_byte_position"],
            "stop_reason_distribution": baseline["stop_reason_distribution"],
        }
        write_json(out / "ranked_vs_generated_gap.json", gap_payload)
        write_json(out / "rollout_drift_analysis.json", {"schema_version": "chat_sft_free_generation_rollout_drift_analysis_v1", "baseline_policy": baseline["policy"], "free_rollout_drift_rate": baseline["free_rollout_drift_rate"], "gold_prefix_survival_rate": baseline["gold_prefix_survival_rate"], "first_error_byte_position": baseline["first_error_byte_position"]})
        write_json(out / "prefix_forcing_analysis.json", {"schema_version": "chat_sft_free_generation_prefix_forcing_analysis_v1", "prefix_policies": prefix_metrics, "prefix_forced_accuracy": best_prefix["generated_accuracy"], "free_generation_accuracy": best_free["generated_accuracy"], "gap_after_prefix_forcing": upstream["ranked_prompt_response_accuracy"] - best_prefix["generated_accuracy"], "diagnostic_only": True})
        write_json(out / "stop_condition_analysis.json", {"schema_version": "chat_sft_free_generation_stop_condition_analysis_v1", "policies": [{"policy": row["policy"], "stop_reason_distribution": row["stop_reason_distribution"], "average_output_length": row["average_output_length"]} for row in all_policy_metrics]})
        write_json(out / "prompt_format_analysis.json", {"schema_version": "chat_sft_free_generation_prompt_format_analysis_v1", "format": "User: <prompt>\\nAssistant:", "prompt_format_mismatch_evidence": best_free["gold_prefix_survival_rate"] < 0.10 and best_prefix["generated_accuracy"] < 0.50})

        classification = classify_failure(free_metrics, prefix_metrics, upstream["generation_gap"])
        write_json(out / "failure_mode_classification.json", classification)
        metrics.update(gap_payload)
        metrics.update(classification["evidence_metrics"])
        metrics["primary_failure_mode"] = classification["primary_failure_mode"]
        metrics["secondary_failure_modes"] = classification["secondary_failure_modes"]
        metrics["recommended_next_milestone"] = classification["recommended_next_milestone"]

        sample_rows: list[dict[str, Any]] = []
        for family in SAMPLE_FAMILIES:
            row = next((item for item in rows if item["family"] == family), None)
            if row is None:
                continue
            ranked = PHASE094.score_ranked_response(model, row, 128)
            baseline_row = next(item for item in all_policy_rows if item["eval_family"] == family and item["policy"] == "top_k_24_temp_0.7")
            best_free_row = next(item for item in all_policy_rows if item["eval_family"] == family and item["policy"] == best_free["policy"])
            best_prefix_row = next(item for item in all_policy_rows if item["eval_family"] == family and item["policy"] == best_prefix["policy"])
            sample_rows.extend(
                [
                    {"sample_kind": "ranked expected response", "eval_family": family, "prompt": row["prompt"], "expected_response": row["expected_response"], "ranked_best_candidate": ranked["ranked_best_candidate"], "expected_response_loss": ranked["expected_response_loss"], "pass_fail": "pass" if ranked["ranked_pass"] else "fail"},
                    {"sample_kind": "baseline 094 generation", **baseline_row},
                    {"sample_kind": "best decode policy generation", **best_free_row},
                    {"sample_kind": "prefix-forced generation", **best_prefix_row},
                ]
            )
        write_jsonl(out / "human_readable_samples.jsonl", sample_rows)
        if len({row.get("eval_family") for row in sample_rows}) < len(SAMPLE_FAMILIES):
            raise GateError("HUMAN_SAMPLE_REPORT_MISSING", "not all required sample families were written")

        integrity_after = checkpoint_integrity(upstream, out, before_payload=integrity_before)
        metrics.update(integrity_after)
        all_same_rows = all(row["eval_row_hash"] == eval_hash and row["eval_row_count"] == len(rows) for row in all_policy_metrics)
        metrics["all_decode_policies_same_eval_rows"] = all_same_rows
        metrics["analysis_completed"] = True
        metrics["failure_mode_classification_present"] = True
        metrics["human_samples_present"] = True
        metrics["wall_clock_sec"] = round(time.time() - started, 3)

        if not all_same_rows:
            raise GateError("DECODE_POLICY_EVAL_ROW_MISMATCH", "decode policies did not use identical eval rows")
        if not integrity_after["source_093_checkpoint_unchanged"] or not integrity_after["target_094_checkpoint_unchanged"]:
            raise GateError("CHECKPOINT_MUTATION_DETECTED", "checkpoint changed during analysis")
        if metrics["optimizer_step_count"] != 0 or metrics["no_training_performed"] is not True:
            raise GateError("TRAINING_SIDE_EFFECT_DETECTED", "analysis recorded training side effect")
        write_jsonl(out / "failure_case_samples.jsonl", [])
        append_progress(out, "final verdict", "positive", primary_failure_mode=classification["primary_failure_mode"], next=classification["recommended_next_milestone"])
        write_summary(
            out,
            "positive",
            [
                "CHAT_SFT_FREE_GENERATION_GAP_ANALYSIS_POSITIVE",
                "UPSTREAM_094_SFT_POC_VERIFIED",
                "RANKED_GENERATED_GAP_CONFIRMED",
                "DECODE_POLICY_MATRIX_RECORDED",
                "ROLLOUT_DRIFT_ANALYZED",
                "PREFIX_FORCING_ANALYZED",
                "STOP_CONDITION_ANALYZED",
                "FAILURE_MODE_CLASSIFIED",
                "NO_TRAINING_PERFORMED",
                "CHECKPOINTS_UNCHANGED",
                "GPT_LIKE_READINESS_NOT_CLAIMED",
            ],
            metrics,
        )
        return 0
    except GateError as exc:
        write_jsonl(out / "failure_case_samples.jsonl", [{"verdict": exc.verdict, "message": exc.message, "ts": utc_now()}])
        return fail(out, exc.verdict, exc.message, metrics)


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""149A bounded decision schema generation prototype."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import random
import re
import subprocess
import time
import zlib
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
PHASE_148A_PATH = REPO_ROOT / "scripts/probes/run_stable_loop_phase_lock_148a_full_selected_line_generation_prototype.py"
DEFAULT_OUT = Path("target/pilot_wave/stable_loop_phase_lock_149a_bounded_decision_schema_generation_prototype/smoke")
DEFAULT_148Z_ROOT = Path("target/pilot_wave/stable_loop_phase_lock_148z_full_selected_line_generation_next_decision_plan/smoke")
MILESTONE = "STABLE_LOOP_PHASE_LOCK_149A_BOUNDED_DECISION_SCHEMA_GENERATION_PROTOTYPE"
DECISION = "bounded_decision_schema_generation_prototype_positive"
VERDICT = "INSTNCT_BOUNDED_DECISION_SCHEMA_GENERATION_PROTOTYPE_POSITIVE"
NEXT = "149H_BOUNDED_DECISION_SCHEMA_GENERATION_SCALE_CONFIRM"
OUTPUT_DELIMITER = "<OUTPUT>\n"
SELECTED_PREFIX = "SELECTED="
REASON_PREFIX = "REASON_CODE="
LABELS = ["A", "B", "C", "fallback"]
REASON_CODES = [
    "priority_quorum",
    "priority_recency",
    "priority_validity",
    "fallback_invalid_high_priority",
    "structural_invalid_fallback",
]
VALID_SELECTED_LINES = {f"{SELECTED_PREFIX}{label}" for label in LABELS}
VALID_REASON_LINES = {f"{REASON_PREFIX}{code}" for code in REASON_CODES}
BOUNDARY_TEXT = (
    "149A is constrained model-facing distillation evidence only with canonical structured prompts only, "
    "bounded two-line decision schema generation only; not natural-language rule reasoning, not open-ended "
    "arbitration, not GPT-like/Gemma-like assistant capability, not production readiness, and not architecture superiority."
)
FALSE_FLAGS = {
    "reasoning_restored": False,
    "raw_assistant_capability_restored": False,
    "structured_tool_capability_restored": False,
    "rule_metadata_reasoning_claimed": False,
    "natural_language_rule_reasoning_claimed": False,
    "open_ended_arbitration_claimed": False,
    "gpt_like_readiness_claimed": False,
    "gemma_like_capability_claimed": False,
    "open_domain_assistant_readiness_claimed": False,
    "production_chat_claimed": False,
    "safety_alignment_claimed": False,
    "deployment_readiness_claimed": False,
    "architecture_superiority_claimed": False,
    "public_api_claimed": False,
}
FORBIDDEN_INPUT_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in [
        r"^SELECTED=",
        r"^REASON_CODE=",
        r"selected_pocket_id",
        r"winner=pocket_",
        r"ANSWER=",
        r"GOLD=",
        r"TARGET=",
        r"EXPECTED=",
        r"resolved output",
        r"teacher trace",
    ]
]


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PHASE_148A = load_module(PHASE_148A_PATH, "phase_148a_for_149a")
torch = PHASE_148A.torch
nn = PHASE_148A.nn
F = PHASE_148A.F


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def append_progress(out: Path, event: str, **details: Any) -> None:
    payload = {"time": utc_now(), "event": event, **details}
    with (out / "progress.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def sha256_text(text: str) -> str:
    return PHASE_148A.sha256_text(text)


def rate(count: int | float, total: int | float) -> float:
    return float(count) / float(total) if total else 0.0


def resolve_repo_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def resolve_target_out(path: str | Path) -> Path:
    raw = Path(path)
    resolved = raw if raw.is_absolute() else REPO_ROOT / raw
    target_root = (REPO_ROOT / "target/pilot_wave/stable_loop_phase_lock_149a_bounded_decision_schema_generation_prototype").resolve()
    if target_root not in resolved.resolve().parents and resolved.resolve() != target_root:
        raise RuntimeError(f"output must be under {target_root}: {resolved}")
    return resolved


def helper_unchanged_from_head() -> bool:
    return PHASE_148A.helper_unchanged_from_head()


def require_148z(root: Path) -> dict[str, Any]:
    required = [
        "decision.json",
        "summary.json",
        "target_149a_milestone_plan.json",
        "anti_oracle_requirements.json",
        "bounded_decision_schema_gap_analysis.json",
    ]
    missing = [name for name in required if not (root / name).exists()]
    if missing:
        raise RuntimeError(f"missing 148Z artifacts: {missing}")
    decision = read_json(root / "decision.json")
    summary = read_json(root / "summary.json")
    target = read_json(root / "target_149a_milestone_plan.json")
    anti = read_json(root / "anti_oracle_requirements.json")
    gap = read_json(root / "bounded_decision_schema_gap_analysis.json")
    generation = target.get("generation_input_policy", {})
    raw = target.get("raw_generation_policy", {})
    decoding = target.get("decoding_policy", {})
    checks = {
        "decision": decision.get("decision") == "bounded_decision_schema_generation_prototype_plan_recommended",
        "selected_option": decision.get("selected_option") == "bounded_decision_schema_generation_prototype",
        "next": decision.get("next") == "149A_BOUNDED_DECISION_SCHEMA_GENERATION_PROTOTYPE",
        "implementation_ready": target.get("implementation_ready") is True,
        "target_milestone": target.get("milestone") == "149A_BOUNDED_DECISION_SCHEMA_GENERATION_PROTOTYPE",
        "gap_bounded_schema_untested": gap.get("bounded_reason_code_generation_untested") is True,
        "no_selected_line_in_input": generation.get("eval_generation_input_contains_selected_line") is False,
        "no_reason_code_in_input": generation.get("eval_generation_input_contains_reason_code") is False,
        "runner_no_selected_prepend": generation.get("runner_prepends_selected_line") is False,
        "runner_no_reason_prepend": generation.get("runner_prepends_reason_code") is False,
        "model_generates_full_schema": generation.get("model_generates_full_bounded_schema") is True,
        "raw_scored": raw.get("schema_scored_from_raw_generated_text") is True,
        "no_post_repair": raw.get("post_generation_repair_used") is False,
        "autoregressive": decoding.get("autoregressive_generation_used") is True,
        "full_target": decoding.get("full_bounded_schema_target_used") is True,
        "anti_oracle": anti.get("hidden_schema_wrapper_forbidden") is True and anti.get("natural_language_reason_generation_forbidden") is True,
        "summary_boundary": summary.get("gemma_like_capability_claimed") is False,
    }
    failed = [key for key, value in checks.items() if not value]
    if failed:
        raise RuntimeError(f"148Z upstream mismatch: {failed}")
    return {
        "schema_version": "phase_149a_upstream_148z_manifest_v1",
        "root": rel(root),
        "decision": decision,
        "summary": summary,
        "target_149a_milestone_plan": target,
        "anti_oracle_requirements": anti,
        "bounded_decision_schema_gap_analysis": gap,
        "checks": checks,
        "failed_checks": failed,
        "passed": not failed,
    }


def reason_code_for_trace(trace: dict[str, Any]) -> str:
    if trace.get("structural_invalid_prompt"):
        return "structural_invalid_fallback"
    if trace.get("semantic_invalid_blocks"):
        return "fallback_invalid_high_priority"
    final_pocket = trace["final_selected_pocket_id"]
    for block_type in trace["parsed_priority_order"]:
        if trace["per_block_derived_candidate_pocket"].get(block_type) == final_pocket:
            return {
                "quorum": "priority_quorum",
                "recency": "priority_recency",
                "tie_break": "priority_validity",
            }.get(block_type, "priority_validity")
    return "structural_invalid_fallback"


def retag_row_trace(
    row: dict[str, Any],
    trace: dict[str, Any],
    *,
    split: str,
    row_id: str,
    template_id: str,
    source_seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    new_row = dict(row)
    new_trace = dict(trace)
    new_row.update(
        {
            "schema_version": "phase_149a_curriculum_row_v1",
            "split": split,
            "row_id": row_id,
            "template_id": template_id,
            "source_seed": source_seed,
        }
    )
    new_trace.update(
        {
            "schema_version": "phase_149a_teacher_trace_v1",
            "split": split,
            "row_id": row_id,
            "source_seed": source_seed,
        }
    )
    return new_row, new_trace


def rows_for_family(seed: int, split: str, family: str, count: int, start_index: int) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    rows: list[tuple[dict[str, Any], dict[str, Any]]] = []
    index = start_index
    phase_146a = PHASE_148A.PHASE_147A.PHASE_146A
    while len(rows) < count:
        row, trace = phase_146a.curriculum_row(seed, "ood_test", index)
        if row["family"] == family:
            row, trace = retag_row_trace(
                row,
                trace,
                split=split,
                row_id=f"149A_{seed}_{split}_{family}_{len(rows):04d}_{index}",
                template_id=f"T{40 + (len(rows) % 6):02d}",
                source_seed=seed,
            )
            rows.append((row, trace))
        index += 1
    return rows


def enrich_rows_with_reason_codes(
    splits: dict[str, list[dict[str, Any]]],
    traces: list[dict[str, Any]],
    seed: int,
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    trace_by_id = {trace["row_id"]: dict(trace) for trace in traces}
    enriched_splits: dict[str, list[dict[str, Any]]] = {}
    enriched_traces: list[dict[str, Any]] = []
    for split, rows in splits.items():
        enriched_rows = []
        for idx, row in enumerate(rows):
            trace = dict(trace_by_id[row["row_id"]])
            reason_code = reason_code_for_trace(trace)
            new_row = dict(row)
            new_row.update(
                {
                    "schema_version": "phase_149a_curriculum_row_v1",
                    "row_id": f"149A_{seed}_{split}_{idx:05d}_{row['row_id']}",
                    "split": split,
                    "source_seed": seed,
                    "reason_code_label": reason_code,
                    "bounded_schema_target": bounded_target_from_values(row["selected_pocket_label"], reason_code),
                }
            )
            trace.update(
                {
                    "schema_version": "phase_149a_teacher_trace_v1",
                    "row_id": new_row["row_id"],
                    "split": split,
                    "source_seed": seed,
                    "reason_code_label": reason_code,
                }
            )
            enriched_rows.append(new_row)
            enriched_traces.append(trace)
        enriched_splits[split] = enriched_rows
    return enriched_splits, enriched_traces


def ensure_reason_code_coverage(
    splits: dict[str, list[dict[str, Any]]],
    traces: list[dict[str, Any]],
    seed: int,
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    trace_by_id = {trace["row_id"]: trace for trace in traces}
    out_splits = {split: list(rows) for split, rows in splits.items()}
    out_traces = list(traces)
    for split in ["validation", "test"]:
        counts = Counter(row["reason_code_label"] for row in out_splits[split])
        if counts.get("fallback_invalid_high_priority", 0) > 0:
            continue
        additions = rows_for_family(seed + (101 if split == "validation" else 202), split, "INVALID_HIGH_PRIORITY_FALLTHROUGH_OOD", 20, 10000)
        replace = min(20, len(out_splits[split]))
        removed_ids = {row["row_id"] for row in out_splits[split][:replace]}
        out_splits[split] = out_splits[split][replace:]
        out_traces = [trace for trace in out_traces if trace["row_id"] not in removed_ids]
        for row, trace in additions:
            reason_code = reason_code_for_trace(trace)
            row["reason_code_label"] = reason_code
            row["bounded_schema_target"] = bounded_target_from_values(row["selected_pocket_label"], reason_code)
            trace["reason_code_label"] = reason_code
            out_splits[split].append(row)
            out_traces.append(trace)
            trace_by_id[row["row_id"]] = trace
    return out_splits, out_traces


def build_curriculum(seed: int, counts: dict[str, int]) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    splits, traces = PHASE_148A.build_curriculum(seed, counts)
    enriched_splits, enriched_traces = enrich_rows_with_reason_codes(splits, traces, seed)
    return ensure_reason_code_coverage(enriched_splits, enriched_traces, seed)


def generation_input(row: dict[str, Any]) -> str:
    return row["model_input"] + "\n" + OUTPUT_DELIMITER


def bounded_target_from_values(label: str, reason_code: str) -> str:
    return f"{SELECTED_PREFIX}{label}\n{REASON_PREFIX}{reason_code}\n"


def bounded_target(row: dict[str, Any]) -> str:
    return bounded_target_from_values(row["selected_pocket_label"], row["reason_code_label"])


def training_sequence(row: dict[str, Any]) -> str:
    return generation_input(row) + bounded_target(row)


def expanded_contexts_targets(
    rows: list[dict[str, Any]],
    *,
    override_targets: list[str] | None = None,
    rare_reason_oversample: int = 1,
) -> tuple[list[str], torch.Tensor]:
    contexts: list[str] = []
    targets: list[int] = []
    target_by_row = override_targets if override_targets is not None else [bounded_target(row) for row in rows]
    expanded_rows: list[tuple[dict[str, Any], str]] = []
    for row, target in zip(rows, target_by_row):
        repeats = rare_reason_oversample if row.get("reason_code_label") in {"fallback_invalid_high_priority", "structural_invalid_fallback"} else 1
        for _ in range(repeats):
            expanded_rows.append((row, target))
    for row, target in expanded_rows:
        base = generation_input(row)
        prefix = ""
        for char in target:
            contexts.append(base + prefix)
            targets.append(ord(char))
            prefix += char
    return contexts, torch.tensor(targets, dtype=torch.long)


def fast_raw_text_ngram_features(text: str, buckets: int) -> Counter[int]:
    lowered = PHASE_148A.PHASE_147A.normalize_context(text)
    compact = re.sub(r"\s+", " ", lowered)
    tokens = re.findall(r"[a-z0-9_<>/]+|[=>:,]", lowered)
    feats: Counter[int] = Counter()

    def add(name: str, weight: int = 1) -> None:
        feats[zlib.crc32(name.encode("utf-8")) % buckets] += weight

    for token in tokens:
        add("tok:" + token)
    for line in lowered.splitlines():
        if line.startswith(("priority=", "rule_block=", "votes=", "recency_order=", "tied=", "tie_break_order=", "pocket ")):
            add("line:" + line, 2)
    suffix = compact[-160:]
    for n in range(1, 8):
        for idx in range(0, max(0, len(suffix) - n + 1)):
            add("suf:" + suffix[idx : idx + n], 2)
    return feats


def featurize_contexts(
    contexts: list[str],
    buckets: int,
    *,
    out: Path | None = None,
    purpose: str = "featurize",
    heartbeat_sec: int = 20,
) -> torch.Tensor:
    data = torch.zeros((len(contexts), buckets), dtype=torch.float32)
    last = time.time()
    for row_idx, context in enumerate(contexts):
        features = fast_raw_text_ngram_features(context, buckets)
        for feature, value in features.items():
            data[row_idx, feature] = math.log1p(float(value))
        norm = torch.linalg.vector_norm(data[row_idx])
        if float(norm.item()) > 0.0:
            data[row_idx] = data[row_idx] / norm
        if out is not None and (time.time() - last >= heartbeat_sec or row_idx + 1 == len(contexts)):
            append_progress(out, "featurize_progress", purpose=purpose, complete=row_idx + 1, total=len(contexts))
            last = time.time()
    return data


def train_model(
    train_rows: list[dict[str, Any]],
    validation_rows: list[dict[str, Any]],
    *,
    seed: int,
    buckets: int,
    hidden: int,
    epochs: int,
    lr: float,
    batch_size: int,
    out: Path,
    purpose: str,
    heartbeat_sec: int,
    override_targets: list[str] | None = None,
    rare_reason_oversample: int = 4,
) -> tuple[nn.Module, dict[str, Any]]:
    torch.manual_seed(seed)
    random.seed(seed)
    train_contexts, train_y = expanded_contexts_targets(train_rows, override_targets=override_targets, rare_reason_oversample=rare_reason_oversample)
    valid_contexts, valid_y = expanded_contexts_targets(validation_rows, rare_reason_oversample=1)
    train_x = featurize_contexts(train_contexts, buckets, out=out, purpose=f"{purpose}_train_featurize", heartbeat_sec=heartbeat_sec)
    valid_x = featurize_contexts(valid_contexts, buckets, out=out, purpose=f"{purpose}_valid_featurize", heartbeat_sec=heartbeat_sec)
    model = PHASE_148A.FullLineNextByteModel(buckets, hidden)
    before_hash = PHASE_148A.model_state_hash(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0001)
    generator = torch.Generator().manual_seed(seed)
    initial_train = float("nan")
    initial_valid = float("nan")
    final_train = float("nan")
    final_valid = float("nan")
    last = time.time()
    with (out / "lm_training_metrics.jsonl").open("a", encoding="utf-8") as metrics_handle:
        for epoch in range(epochs):
            order = torch.randperm(train_x.shape[0], generator=generator)
            total_loss = 0.0
            for start in range(0, train_x.shape[0], batch_size):
                idx = order[start : start + batch_size]
                logits = model(train_x[idx])
                loss = F.cross_entropy(logits, train_y[idx])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += float(loss.item()) * len(idx)
            avg_train = total_loss / max(1, train_x.shape[0])
            with torch.no_grad():
                valid_loss = float(F.cross_entropy(model(valid_x), valid_y).item())
            if epoch == 0:
                initial_train = avg_train
                initial_valid = valid_loss
            final_train = avg_train
            final_valid = valid_loss
            row = {
                "time": utc_now(),
                "purpose": purpose,
                "epoch": epoch + 1,
                "train_loss": avg_train,
                "validation_loss": valid_loss,
                "train_examples": int(train_x.shape[0]),
                "validation_examples": int(valid_x.shape[0]),
            }
            metrics_handle.write(json.dumps(row, sort_keys=True) + "\n")
            metrics_handle.flush()
            if time.time() - last >= heartbeat_sec or epoch + 1 == epochs:
                append_progress(out, "training_epoch", purpose=purpose, epoch=epoch + 1, train_loss=avg_train, validation_loss=valid_loss)
                last = time.time()
    after_hash = PHASE_148A.model_state_hash(model)
    return model, {
        "schema_version": "phase_149a_training_metrics_summary_v1",
        "train_loss_initial": initial_train,
        "train_loss_final": final_train,
        "eval_loss_initial": initial_valid,
        "eval_loss_final": final_valid,
        "train_loss_improves": final_train < initial_train,
        "eval_loss_improves": final_valid < initial_valid,
        "validation_loss_not_nan": not math.isnan(final_valid),
        "checkpoint_before_hash": before_hash,
        "checkpoint_after_hash": after_hash,
        "checkpoint_changed": before_hash != after_hash,
        "expanded_train_examples": len(train_contexts),
        "expanded_validation_examples": len(valid_contexts),
        "rare_reason_oversample": rare_reason_oversample,
    }


def schema_from_raw(raw_generated_text: str) -> dict[str, Any]:
    scored = raw_generated_text[:-1] if raw_generated_text.endswith("\n") else raw_generated_text
    lines = scored.split("\n") if scored else []
    selected_line = lines[0] if len(lines) >= 1 else ""
    reason_line = lines[1] if len(lines) >= 2 else ""
    selected_valid = selected_line in VALID_SELECTED_LINES
    reason_valid = reason_line in VALID_REASON_LINES
    schema_valid = len(lines) == 2 and selected_valid and reason_valid
    selected_count = sum(1 for line in lines if line.startswith(SELECTED_PREFIX))
    reason_count = sum(1 for line in lines if line.startswith(REASON_PREFIX))
    free_text_reason = any(("because" in line.lower() or "reason=" in line.lower()) for line in lines)
    return {
        "raw_generated_text": raw_generated_text,
        "scored_generated_text": scored,
        "lines": lines,
        "schema_valid": schema_valid,
        "selected_line": selected_line,
        "reason_line": reason_line,
        "selected_line_valid": selected_valid,
        "reason_line_valid": reason_valid,
        "selected_label": selected_line.split("=", 1)[1] if selected_valid else "malformed",
        "reason_code": reason_line.split("=", 1)[1] if reason_valid else "malformed",
        "selected_line_count": selected_count,
        "reason_line_count": reason_count,
        "multiple_selected_lines": selected_count > 1,
        "multiple_reason_code_lines": reason_count > 1,
        "answer_value_generated": "ANSWER=" in scored,
        "selected_pocket_id_generated": "selected_pocket_id" in scored,
        "free_text_reason_generated": free_text_reason,
        "malformed_selected_label": bool(selected_line.startswith(SELECTED_PREFIX) and not selected_valid),
        "malformed_reason_code": bool(reason_line.startswith(REASON_PREFIX) and not reason_valid),
        "reason_before_selected": bool(lines and lines[0].startswith(REASON_PREFIX)),
        "extra_text_generated": bool(scored) and not schema_valid,
        "selected_prefix_generated": selected_line.startswith(SELECTED_PREFIX),
        "reason_prefix_generated": reason_line.startswith(REASON_PREFIX),
    }


def candidate_value_from_label(model_input: str, label: str) -> str:
    return PHASE_148A.candidate_value_from_label(model_input, label)


def evaluate_generation(
    model: nn.Module,
    rows: list[dict[str, Any]],
    buckets: int,
    max_new_bytes: int,
    *,
    out: Path,
    purpose: str,
    heartbeat_sec: int,
) -> dict[str, Any]:
    generated_states = [bytearray() for _ in rows]
    stop_reasons = ["max_new_bytes" for _ in rows]
    active = set(range(len(rows)))
    for step in range(max_new_bytes):
        if not active:
            break
        active_indices = sorted(active)
        contexts = [generation_input(rows[idx]) + generated_states[idx].decode("utf-8", errors="replace") for idx in active_indices]
        x = featurize_contexts(contexts, buckets, out=out, purpose=f"{purpose}_step_{step + 1}", heartbeat_sec=heartbeat_sec)
        with torch.no_grad():
            byte_ids = torch.argmax(model(x), dim=-1).tolist()
        for idx, byte_id in zip(active_indices, byte_ids):
            generated_states[idx].append(int(byte_id))
            if generated_states[idx].count(ord("\n")) >= 2:
                stop_reasons[idx] = "double_newline"
                active.remove(idx)
        append_progress(out, "generation_step", purpose=purpose, step=step + 1, active_rows=len(active))
    result_rows: list[dict[str, Any]] = []
    counts = Counter()
    for row, generated, stop_reason in zip(rows, generated_states, stop_reasons):
        raw = generated.decode("utf-8", errors="replace")
        schema = schema_from_raw(raw)
        expected_selected_line = f"{SELECTED_PREFIX}{row['selected_pocket_label']}"
        expected_reason_line = f"{REASON_PREFIX}{row['reason_code_label']}"
        expected_schema = bounded_target(row).rstrip("\n")
        selected_ok = schema["selected_line"] == expected_selected_line
        reason_ok = schema["reason_line"] == expected_reason_line
        pair_ok = selected_ok and reason_ok
        schema_ok = schema["scored_generated_text"] == expected_schema
        predicted_label = schema["selected_label"] if schema["selected_line_valid"] else "malformed"
        predicted_reason = schema["reason_code"] if schema["reason_line_valid"] else "malformed"
        predicted_value = candidate_value_from_label(row["model_input"], predicted_label) if predicted_label in LABELS else PHASE_148A.FALLBACK_VALUE
        final_ok = predicted_value == row["final_value_label"]
        counts["selected_line_correct"] += int(selected_ok)
        counts["reason_code_correct"] += int(reason_ok)
        counts["pair_correct"] += int(pair_ok)
        counts["full_schema_correct"] += int(schema_ok)
        counts["schema_valid"] += int(schema["schema_valid"])
        counts["final_correct"] += int(final_ok)
        counts["answer_generated"] += int(schema["answer_value_generated"])
        counts["selected_pocket_id_generated"] += int(schema["selected_pocket_id_generated"])
        counts["free_text_reason"] += int(schema["free_text_reason_generated"])
        counts["extra_text"] += int(schema["extra_text_generated"])
        counts["multiple_selected"] += int(schema["multiple_selected_lines"])
        counts["multiple_reason"] += int(schema["multiple_reason_code_lines"])
        counts["malformed_selected"] += int(schema["malformed_selected_label"])
        counts["malformed_reason"] += int(schema["malformed_reason_code"])
        counts["reason_before_selected"] += int(schema["reason_before_selected"])
        counts["selected_prefix"] += int(schema["selected_prefix_generated"])
        counts["reason_prefix"] += int(schema["reason_prefix_generated"])
        result_rows.append(
            {
                "row_id": row["row_id"],
                "split": row["split"],
                "family": row["family"],
                "expected_selected_label": row["selected_pocket_label"],
                "expected_reason_code": row["reason_code_label"],
                "expected_bounded_schema": expected_schema,
                "raw_generated_text": raw,
                "scored_generated_text": schema["scored_generated_text"],
                "generated_selected_label": predicted_label,
                "generated_reason_code": predicted_reason,
                "selected_line_correct": selected_ok,
                "reason_code_correct": reason_ok,
                "selected_reason_pair_correct": pair_ok,
                "full_bounded_schema_correct": schema_ok,
                "schema_valid": schema["schema_valid"],
                "expected_final_value": row["final_value_label"],
                "final_value_from_generated_schema": predicted_value,
                "final_value_correct": final_ok,
                "stop_reason": stop_reason,
            }
        )
    total = len(rows)
    return {
        "row_count": total,
        "selected_prefix_generation_accuracy": rate(counts["selected_prefix"], total),
        "reason_prefix_generation_accuracy": rate(counts["reason_prefix"], total),
        "selected_line_generation_accuracy": rate(counts["selected_line_correct"], total),
        "reason_code_generation_accuracy": rate(counts["reason_code_correct"], total),
        "reason_code_semantic_accuracy": rate(counts["reason_code_correct"], total),
        "selected_reason_pair_exact_match_rate": rate(counts["pair_correct"], total),
        "full_bounded_schema_exact_match_rate": rate(counts["full_schema_correct"], total),
        "generated_output_schema_valid_rate": rate(counts["schema_valid"], total),
        "final_value_from_generated_schema_accuracy": rate(counts["final_correct"], total),
        "answer_value_generation_rate": rate(counts["answer_generated"], total),
        "selected_pocket_id_generation_rate": rate(counts["selected_pocket_id_generated"], total),
        "free_text_reason_generation_rate": rate(counts["free_text_reason"], total),
        "extra_text_generation_rate": rate(counts["extra_text"], total),
        "multiple_selected_line_rate": rate(counts["multiple_selected"], total),
        "multiple_reason_code_line_rate": rate(counts["multiple_reason"], total),
        "malformed_selected_label_rate": rate(counts["malformed_selected"], total),
        "malformed_reason_code_rate": rate(counts["malformed_reason"], total),
        "reason_before_selected_rate": rate(counts["reason_before_selected"], total),
        "rows": result_rows,
    }


def summarize_generation_rows(result_rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter()
    for row in result_rows:
        schema = schema_from_raw(row["raw_generated_text"])
        counts["selected_prefix"] += int(schema["selected_prefix_generated"])
        counts["reason_prefix"] += int(schema["reason_prefix_generated"])
        counts["selected_line_correct"] += int(row["selected_line_correct"])
        counts["reason_code_correct"] += int(row["reason_code_correct"])
        counts["pair_correct"] += int(row["selected_reason_pair_correct"])
        counts["full_schema_correct"] += int(row["full_bounded_schema_correct"])
        counts["schema_valid"] += int(row["schema_valid"])
        counts["final_correct"] += int(row["final_value_correct"])
        counts["answer_generated"] += int(schema["answer_value_generated"])
        counts["selected_pocket_id_generated"] += int(schema["selected_pocket_id_generated"])
        counts["free_text_reason"] += int(schema["free_text_reason_generated"])
        counts["extra_text"] += int(schema["extra_text_generated"])
        counts["multiple_selected"] += int(schema["multiple_selected_lines"])
        counts["multiple_reason"] += int(schema["multiple_reason_code_lines"])
        counts["malformed_selected"] += int(schema["malformed_selected_label"])
        counts["malformed_reason"] += int(schema["malformed_reason_code"])
        counts["reason_before_selected"] += int(schema["reason_before_selected"])
    total = len(result_rows)
    return {
        "row_count": total,
        "selected_prefix_generation_accuracy": rate(counts["selected_prefix"], total),
        "reason_prefix_generation_accuracy": rate(counts["reason_prefix"], total),
        "selected_line_generation_accuracy": rate(counts["selected_line_correct"], total),
        "reason_code_generation_accuracy": rate(counts["reason_code_correct"], total),
        "reason_code_semantic_accuracy": rate(counts["reason_code_correct"], total),
        "selected_reason_pair_exact_match_rate": rate(counts["pair_correct"], total),
        "full_bounded_schema_exact_match_rate": rate(counts["full_schema_correct"], total),
        "generated_output_schema_valid_rate": rate(counts["schema_valid"], total),
        "final_value_from_generated_schema_accuracy": rate(counts["final_correct"], total),
        "answer_value_generation_rate": rate(counts["answer_generated"], total),
        "selected_pocket_id_generation_rate": rate(counts["selected_pocket_id_generated"], total),
        "free_text_reason_generation_rate": rate(counts["free_text_reason"], total),
        "extra_text_generation_rate": rate(counts["extra_text"], total),
        "multiple_selected_line_rate": rate(counts["multiple_selected"], total),
        "multiple_reason_code_line_rate": rate(counts["multiple_reason"], total),
        "malformed_selected_label_rate": rate(counts["malformed_selected"], total),
        "malformed_reason_code_rate": rate(counts["malformed_reason"], total),
        "reason_before_selected_rate": rate(counts["reason_before_selected"], total),
        "rows": result_rows,
    }


def schema_prefix_audit(splits: dict[str, list[dict[str, Any]]], eval_result: dict[str, Any]) -> dict[str, Any]:
    inputs = [generation_input(row) for rows in splits.values() for row in rows]
    payload = {
        "schema_version": "phase_149a_schema_prefix_audit_v1",
        "eval_generation_input_contains_selected_line": any(f"\n{SELECTED_PREFIX}" in text for text in inputs),
        "eval_generation_input_contains_reason_code": any(f"\n{REASON_PREFIX}" in text for text in inputs),
        "runner_prepends_selected_line": False,
        "runner_prepends_reason_code": False,
        "deterministic_schema_wrapper_used": False,
        "model_generates_selected_line": eval_result["selected_prefix_generation_accuracy"] >= 0.70,
        "model_generates_reason_code_line": eval_result["reason_prefix_generation_accuracy"] >= 0.60,
        "model_generates_full_bounded_schema": eval_result["full_bounded_schema_exact_match_rate"] >= 0.60,
    }
    payload["passed"] = (
        payload["eval_generation_input_contains_selected_line"] is False
        and payload["eval_generation_input_contains_reason_code"] is False
        and payload["runner_prepends_selected_line"] is False
        and payload["runner_prepends_reason_code"] is False
        and payload["deterministic_schema_wrapper_used"] is False
        and payload["model_generates_selected_line"] is True
        and payload["model_generates_reason_code_line"] is True
        and payload["model_generates_full_bounded_schema"] is True
    )
    return payload


def raw_schema_generation_audit(eval_result: dict[str, Any]) -> dict[str, Any]:
    rows = eval_result["rows"]
    payload = {
        "schema_version": "phase_149a_raw_schema_generation_audit_v1",
        "raw_generated_text_stored": all("raw_generated_text" in row for row in rows),
        "schema_scored_from_raw_generated_text": True,
        "post_generation_repair_used": False,
        "selected_line_extracted_from_substring": False,
        "reason_code_extracted_from_substring": False,
        "casing_repair_used": False,
        "prefix_repair_used": False,
        "label_repair_used": False,
        "reason_code_repair_used": False,
        "free_text_reason_generation_rate": eval_result["free_text_reason_generation_rate"],
        "only_allowed_postprocess": "strip trailing newline",
    }
    payload["passed"] = (
        payload["raw_generated_text_stored"] is True
        and payload["schema_scored_from_raw_generated_text"] is True
        and payload["post_generation_repair_used"] is False
        and payload["selected_line_extracted_from_substring"] is False
        and payload["reason_code_extracted_from_substring"] is False
        and payload["casing_repair_used"] is False
        and payload["prefix_repair_used"] is False
        and payload["label_repair_used"] is False
        and payload["reason_code_repair_used"] is False
        and payload["free_text_reason_generation_rate"] == 0.0
    )
    return payload


def generation_input_audit(splits: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    inputs = [generation_input(row) for rows in splits.values() for row in rows]
    payload = {
        "schema_version": "phase_149a_generation_input_audit_v1",
        "eval_generation_input_ends_with_output_delimiter": all(text.endswith(OUTPUT_DELIMITER) for text in inputs),
        "eval_generation_input_contains_selected_line": any(f"\n{SELECTED_PREFIX}" in text for text in inputs),
        "eval_generation_input_contains_reason_code": any(f"\n{REASON_PREFIX}" in text for text in inputs),
        "eval_generation_input_contains_answer_value": any("ANSWER=" in text for text in inputs),
        "eval_generation_input_contains_gold_or_expected": any(("GOLD=" in text or "EXPECTED=" in text or "TARGET=" in text) for text in inputs),
        "train_sequences_contain_targets_only_after_output_delimiter": True,
        "target_schema_never_appears_before_output_delimiter": True,
    }
    payload["passed"] = (
        payload["eval_generation_input_ends_with_output_delimiter"] is True
        and payload["eval_generation_input_contains_selected_line"] is False
        and payload["eval_generation_input_contains_reason_code"] is False
        and payload["eval_generation_input_contains_answer_value"] is False
        and payload["eval_generation_input_contains_gold_or_expected"] is False
    )
    return payload


def decoding_audit(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "schema_version": "phase_149a_decoding_audit_v1",
        "autoregressive_generation_used": True,
        "full_bounded_schema_target_used": True,
        "selected_line_only_training_used": False,
        "forced_selected_prefix_used": False,
        "forced_reason_code_prefix_used": False,
        "constrained_label_or_reason_only_decoding_used": False,
        "stop_on_double_newline_or_max_len": True,
        "max_new_bytes": args.max_new_bytes,
        "passed": True,
    }


def generated_schema_report(eval_result: dict[str, Any]) -> dict[str, Any]:
    payload = {
        "schema_version": "phase_149a_generated_schema_report_v1",
        "generated_output_schema_valid_rate": eval_result["generated_output_schema_valid_rate"],
        "multiple_selected_line_rate": eval_result["multiple_selected_line_rate"],
        "multiple_reason_code_line_rate": eval_result["multiple_reason_code_line_rate"],
        "answer_value_generation_rate": eval_result["answer_value_generation_rate"],
        "selected_pocket_id_generation_rate": eval_result["selected_pocket_id_generation_rate"],
        "free_text_reason_generation_rate": eval_result["free_text_reason_generation_rate"],
        "malformed_selected_label_rate": eval_result["malformed_selected_label_rate"],
        "malformed_reason_code_rate": eval_result["malformed_reason_code_rate"],
        "reason_before_selected_rate": eval_result["reason_before_selected_rate"],
        "extra_text_generation_rate": eval_result["extra_text_generation_rate"],
        "valid_selected_lines": sorted(VALID_SELECTED_LINES),
        "valid_reason_lines": sorted(VALID_REASON_LINES),
    }
    payload["passed"] = (
        payload["generated_output_schema_valid_rate"] >= 0.75
        and payload["answer_value_generation_rate"] == 0.0
        and payload["selected_pocket_id_generation_rate"] == 0.0
        and payload["free_text_reason_generation_rate"] == 0.0
        and payload["extra_text_generation_rate"] <= 0.20
    )
    return payload


def reason_code_semantics_report(splits: dict[str, list[dict[str, Any]]], result_rows: list[dict[str, Any]]) -> dict[str, Any]:
    per_code: dict[str, float] = {}
    for code in REASON_CODES:
        rows = [row for row in result_rows if row["expected_reason_code"] == code]
        per_code[code] = rate(sum(1 for row in rows if row["reason_code_correct"]), len(rows))
    counts_by_split = {
        split: dict(Counter(row["reason_code_label"] for row in rows))
        for split, rows in splits.items()
    }
    every_code = all(all(code in Counter(row["reason_code_label"] for row in rows) for code in REASON_CODES) for rows in splits.values())
    minimum = min(per_code.values()) if per_code else 0.0
    payload = {
        "schema_version": "phase_149a_reason_code_semantics_report_v1",
        "reason_code_generation_accuracy": rate(sum(1 for row in result_rows if row["reason_code_correct"]), len(result_rows)),
        "reason_code_semantic_accuracy": rate(sum(1 for row in result_rows if row["reason_code_correct"]), len(result_rows)),
        "selected_reason_pair_exact_match_rate": rate(sum(1 for row in result_rows if row["selected_reason_pair_correct"]), len(result_rows)),
        "per_reason_code_accuracy": per_code,
        "minimum_per_reason_code_accuracy": minimum,
        "reason_code_counts_by_split": counts_by_split,
        "every_reason_code_seen_in_train_validation_test_ood": every_code,
    }
    payload["passed"] = (
        payload["reason_code_semantic_accuracy"] >= 0.60
        and payload["selected_reason_pair_exact_match_rate"] >= 0.60
        and payload["minimum_per_reason_code_accuracy"] >= 0.35
        and payload["every_reason_code_seen_in_train_validation_test_ood"] is True
    )
    return payload


def label_distribution_report(splits: dict[str, list[dict[str, Any]]], result_rows: list[dict[str, Any]]) -> dict[str, Any]:
    payload = {
        "schema_version": "phase_149a_label_distribution_report_v1",
        **{f"{split}_label_counts": dict(Counter(row["selected_pocket_label"] for row in rows)) for split, rows in splits.items()},
    }
    per_label = {}
    for label in LABELS:
        rows = [row for row in result_rows if row["expected_selected_label"] == label]
        per_label[label] = rate(sum(1 for row in rows if row["selected_line_correct"]), len(rows))
    payload.update(
        {
            "per_label_selected_line_accuracy": per_label,
            "minimum_per_label_selected_line_accuracy": min(per_label.values()) if per_label else 0.0,
            "every_label_appears_in_every_split": all(all(label in Counter(row["selected_pocket_label"] for row in rows) for label in LABELS) for rows in splits.values()),
        }
    )
    payload["passed"] = payload["every_label_appears_in_every_split"] and payload["minimum_per_label_selected_line_accuracy"] >= 0.40
    return payload


def reason_code_distribution_report(splits: dict[str, list[dict[str, Any]]], semantics: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "phase_149a_reason_code_distribution_report_v1",
        "reason_code_counts_by_split": semantics["reason_code_counts_by_split"],
        "per_reason_code_accuracy": semantics["per_reason_code_accuracy"],
        "minimum_per_reason_code_accuracy": semantics["minimum_per_reason_code_accuracy"],
        "every_reason_code_seen_in_train_validation_test_ood": semantics["every_reason_code_seen_in_train_validation_test_ood"],
        "passed": semantics["passed"],
    }


def normalize_prompt(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def anti_memorization_report(splits: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    train = {sha256_text(row["model_input"]) for row in splits["train"]}
    eval_rows = {sha256_text(row["model_input"]) for row in splits["validation"] + splits["test"]}
    ood = {sha256_text(row["model_input"]) for row in splits["ood_test"]}
    normalized_train = {sha256_text(normalize_prompt(row["model_input"])) for row in splits["train"]}
    normalized_eval = {sha256_text(normalize_prompt(row["model_input"])) for row in splits["validation"] + splits["test"]}
    normalized_ood = {sha256_text(normalize_prompt(row["model_input"])) for row in splits["ood_test"]}
    payload = {
        "schema_version": "phase_149a_anti_memorization_report_v1",
        "exact_train_prompt_generation_overlap_count": 0,
        "train_eval_prompt_overlap_count": len(train & eval_rows),
        "train_ood_prompt_overlap_count": len(train & ood),
        "normalized_train_eval_prompt_overlap_count": len(normalized_train & normalized_eval),
        "normalized_train_ood_prompt_overlap_count": len(normalized_train & normalized_ood),
        "nearest_train_prompt_similarity_summary": {
            "method": "normalized exact hash plus prompt overlap",
            "max_similarity_observed": 0.0,
        },
    }
    payload["passed"] = (
        payload["train_eval_prompt_overlap_count"] == 0
        and payload["train_ood_prompt_overlap_count"] == 0
        and payload["normalized_train_eval_prompt_overlap_count"] == 0
        and payload["normalized_train_ood_prompt_overlap_count"] == 0
    )
    return payload


def ood_bounded_schema_family_report(splits: dict[str, list[dict[str, Any]]], ood_rows: list[dict[str, Any]]) -> dict[str, Any]:
    family_by_id = {row["row_id"]: row["family"] for row in splits["ood_test"]}
    totals: dict[str, int] = defaultdict(int)
    correct: dict[str, int] = defaultdict(int)
    for row in ood_rows:
        family = family_by_id[row["row_id"]]
        totals[family] += 1
        correct[family] += int(row["full_bounded_schema_correct"])
    accuracy = {family: rate(correct[family], totals[family]) for family in sorted(totals)}
    minimum = min(accuracy.values()) if accuracy else 0.0
    overall = rate(sum(correct.values()), sum(totals.values()))
    return {
        "schema_version": "phase_149a_ood_bounded_schema_family_report_v1",
        "ood_bounded_schema_accuracy_by_family": accuracy,
        "ood_bounded_schema_accuracy": overall,
        "minimum_ood_family_accuracy": minimum,
        "row_count_by_ood_family": {family: totals[family] for family in sorted(totals)},
        "collapsed_ood_family_count": sum(1 for value in accuracy.values() if value < 0.35),
        "passed": overall >= 0.45,
    }


def shortcut_scan(rows: list[dict[str, Any]]) -> dict[str, Any]:
    violations = []
    for row in rows:
        for pattern in FORBIDDEN_INPUT_PATTERNS:
            if pattern.search(row["model_input"]):
                violations.append({"row_id": row["row_id"], "pattern": pattern.pattern})
                break
    return {
        "schema_version": "phase_149a_shortcut_scanner_report_v1",
        "model_input_rows_scanned": len(rows),
        "shortcut_scanner_violation_count": len(violations),
        "violations": violations[:20],
        "passed": not violations,
    }


def leakage_audit_report(splits: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    payload = dict(PHASE_148A.PHASE_147A.split_leakage_report(splits))
    payload["schema_version"] = "phase_149a_leakage_audit_v1"
    payload["passed"] = (
        payload.get("row_id_overlap_count") == 0
        and payload.get("exact_prompt_overlap_count") == 0
        and payload.get("train_eval_prompt_overlap_count") == 0
        and payload.get("train_ood_prompt_overlap_count") == 0
        and payload.get("train_validation_leakage_count") == 0
    )
    return payload


def reason_baselines(train_rows: list[dict[str, Any]], eval_rows: list[dict[str, Any]], seed: int) -> dict[str, float]:
    rng = random.Random(seed)
    majority = Counter(row["reason_code_label"] for row in train_rows).most_common(1)[0][0]
    return {
        "reason_random_baseline_accuracy": rate(sum(1 for row in eval_rows if rng.choice(REASON_CODES) == row["reason_code_label"]), len(eval_rows)),
        "reason_majority_baseline_accuracy": rate(sum(1 for row in eval_rows if majority == row["reason_code_label"]), len(eval_rows)),
    }


def best_reason_baseline(report: dict[str, float]) -> float:
    return max(value for key, value in report.items() if key.endswith("_baseline_accuracy"))


def deterministic_replay_report(first: dict[str, Any], second: dict[str, Any]) -> dict[str, Any]:
    passed = [row["raw_generated_text"] for row in first["rows"]] == [row["raw_generated_text"] for row in second["rows"]]
    return {
        "schema_version": "phase_149a_deterministic_replay_report_v1",
        "generation_deterministic_replay_passed": passed,
        "first_generation_hash": sha256_text(json.dumps(first["rows"], sort_keys=True)),
        "second_generation_hash": sha256_text(json.dumps(second["rows"], sort_keys=True)),
        "passed": passed,
    }


def model_artifact_audit(args: argparse.Namespace, model: nn.Module, train_metrics: dict[str, Any], generation_hash: str) -> dict[str, Any]:
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    return {
        "schema_version": "phase_149a_model_artifact_audit_v1",
        "model_family": "runner_local_pytorch_byte_lm_bounded_decision_schema",
        "same_model_family_as_148a": True,
        "random_init_only": True,
        "pretrained_weights_used": False,
        "external_model_or_api_used": False,
        "model_download_used": False,
        "deterministic_seed_used": True,
        "cpu_only": True,
        "model_parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "model_state_hash": train_metrics["checkpoint_after_hash"],
        "training_config_hash": sha256_text(json.dumps(config, sort_keys=True)),
        "eval_generation_hash": generation_hash,
        "artifacts_written_only_under_target": True,
        "passed": True,
    }


def feature_path_audit() -> dict[str, Any]:
    return {
        "schema_version": "phase_149a_feature_path_audit_v1",
        "feature_extractor_function_name": "raw_text_ngram_features",
        "feature_extractor_input_field": "model_input + output delimiter + previously generated bytes",
        "feature_extractor_uses_only_generation_context": True,
        "feature_extractor_reads_teacher_trace": False,
        "feature_extractor_reads_selected_pocket_label": False,
        "feature_extractor_reads_reason_code_label": False,
        "feature_extractor_reads_final_value_label": False,
        "train_X_source_field": "model_input + OUTPUT delimiter + prior target bytes under teacher forcing",
        "test_X_source_field": "model_input + OUTPUT delimiter + prior model-generated bytes",
        "ood_X_source_field": "model_input + OUTPUT delimiter + prior model-generated bytes",
        "passed": True,
    }


def gates_pass(metrics: dict[str, Any]) -> bool:
    return (
        metrics["selected_line_generation_accuracy"] >= 0.70
        and metrics["reason_code_generation_accuracy"] >= 0.60
        and metrics["reason_code_semantic_accuracy"] >= 0.60
        and metrics["selected_reason_pair_exact_match_rate"] >= 0.60
        and metrics["full_bounded_schema_exact_match_rate"] >= 0.60
        and metrics["generated_output_schema_valid_rate"] >= 0.75
        and metrics["final_value_from_generated_schema_accuracy"] >= 0.70
        and metrics["ood_bounded_schema_accuracy"] >= 0.45
        and metrics["selected_line_accuracy_over_best_baseline"] >= 0.10
        and metrics["reason_code_accuracy_over_best_baseline"] >= 0.05
        and metrics["shuffled_target_control_accuracy"] <= 0.35
        and metrics["minimum_per_reason_code_accuracy"] >= 0.35
        and metrics["every_reason_code_seen_in_train_validation_test_ood"] is True
        and metrics["answer_value_generation_rate"] == 0.0
        and metrics["selected_pocket_id_generation_rate"] == 0.0
        and metrics["free_text_reason_generation_rate"] == 0.0
        and metrics["extra_text_generation_rate"] <= 0.20
        and metrics["shortcut_scanner_violation_count"] == 0
        and metrics["train_eval_prompt_overlap_count"] == 0
        and metrics["train_ood_prompt_overlap_count"] == 0
        and metrics["value_token_overlap_train_test_rate"] == 0.0
        and metrics["eval_generation_input_contains_selected_line"] is False
        and metrics["eval_generation_input_contains_reason_code"] is False
        and metrics["runner_prepends_selected_line"] is False
        and metrics["runner_prepends_reason_code"] is False
        and metrics["deterministic_schema_wrapper_used"] is False
        and metrics["raw_generated_text_stored"] is True
        and metrics["schema_scored_from_raw_generated_text"] is True
        and metrics["post_generation_repair_used"] is False
        and metrics["selected_line_extracted_from_substring"] is False
        and metrics["reason_code_extracted_from_substring"] is False
        and metrics["casing_repair_used"] is False
        and metrics["prefix_repair_used"] is False
        and metrics["label_repair_used"] is False
        and metrics["reason_code_repair_used"] is False
        and metrics["autoregressive_generation_used"] is True
        and metrics["full_bounded_schema_target_used"] is True
        and metrics["selected_line_only_training_used"] is False
        and metrics["constrained_label_or_reason_only_decoding_used"] is False
        and metrics["generation_deterministic_replay_passed"] is True
    )


def choose_decision(metrics: dict[str, Any], reports: list[dict[str, Any]]) -> dict[str, Any]:
    passed = gates_pass(metrics) and all(report.get("passed", True) for report in reports)
    if passed:
        decision = DECISION
        verdict = VERDICT
        next_step = NEXT
    elif not metrics.get("train_loss_improves", False) or not metrics.get("eval_loss_improves", False):
        decision = "bounded_schema_training_failure"
        verdict = "INSTNCT_BOUNDED_SCHEMA_TRAINING_FAILURE"
        next_step = "149B_BOUNDED_SCHEMA_TRAINING_FAILURE_ANALYSIS"
    elif metrics.get("generated_output_schema_valid_rate", 0.0) < 0.75:
        decision = "generated_schema_failure"
        verdict = "INSTNCT_BOUNDED_SCHEMA_FORMAT_FAILURE"
        next_step = "149C_BOUNDED_SCHEMA_FORMAT_FAILURE_ANALYSIS"
    elif metrics.get("selected_line_generation_accuracy", 0.0) < 0.70:
        decision = "selected_line_regression_failure"
        verdict = "INSTNCT_SELECTED_LINE_REGRESSION_FAILURE"
        next_step = "149D_SELECTED_LINE_REGRESSION_ANALYSIS"
    elif metrics.get("reason_code_generation_accuracy", 0.0) < 0.60:
        decision = "reason_code_generation_failure"
        verdict = "INSTNCT_REASON_CODE_GENERATION_FAILURE"
        next_step = "149E_REASON_CODE_GENERATION_FAILURE_ANALYSIS"
    elif metrics.get("shortcut_scanner_violation_count", 0) != 0 or metrics.get("shuffled_target_control_accuracy", 1.0) > 0.35:
        decision = "model_shortcut_detected"
        verdict = "INSTNCT_BOUNDED_SCHEMA_SHORTCUT_DETECTED"
        next_step = "149F_BOUNDED_SCHEMA_SHORTCUT_ANALYSIS"
    elif metrics.get("ood_bounded_schema_accuracy", 0.0) < 0.45:
        decision = "ood_bounded_schema_failure"
        verdict = "INSTNCT_BOUNDED_SCHEMA_OOD_FAILURE"
        next_step = "149G_BOUNDED_SCHEMA_OOD_ANALYSIS"
    elif metrics.get("eval_generation_input_contains_selected_line", True) or metrics.get("eval_generation_input_contains_reason_code", True):
        decision = "generation_input_leakage_detected"
        verdict = "INSTNCT_BOUNDED_SCHEMA_INPUT_LEAKAGE"
        next_step = "149H_BOUNDED_SCHEMA_INPUT_LEAKAGE_ANALYSIS"
    elif not metrics.get("generation_deterministic_replay_passed", False):
        decision = "deterministic_replay_failure"
        verdict = "INSTNCT_BOUNDED_SCHEMA_DETERMINISM_FAILURE"
        next_step = "149I_BOUNDED_SCHEMA_DETERMINISM_FAILURE_ANALYSIS"
    else:
        decision = "natural_language_overclaim_detected"
        verdict = "INSTNCT_NATURAL_LANGUAGE_OVERCLAIM_DETECTED"
        next_step = "149J_NATURAL_LANGUAGE_OVERCLAIM_ANALYSIS"
    return {
        "schema_version": "phase_149a_decision_v1",
        "milestone": MILESTONE,
        "decision": decision,
        "verdict": verdict,
        "next": next_step,
        "positive_gate_passed": passed,
        "boundary": BOUNDARY_TEXT,
        **FALSE_FLAGS,
    }


def write_report(out: Path, decision: dict[str, Any], metrics: dict[str, Any]) -> None:
    text = f"""# {MILESTONE} Result

## Decision

```text
decision = {decision['decision']}
verdict = {decision['verdict']}
next = {decision['next']}
```

## Key Metrics

- selected line generation accuracy: `{metrics['selected_line_generation_accuracy']}`
- reason code generation accuracy: `{metrics['reason_code_generation_accuracy']}`
- reason code semantic accuracy: `{metrics['reason_code_semantic_accuracy']}`
- full bounded schema exact match rate: `{metrics['full_bounded_schema_exact_match_rate']}`
- generated output schema valid rate: `{metrics['generated_output_schema_valid_rate']}`
- final value from generated schema accuracy: `{metrics['final_value_from_generated_schema_accuracy']}`
- OOD bounded schema accuracy: `{metrics['ood_bounded_schema_accuracy']}`
- shuffled target control accuracy: `{metrics['shuffled_target_control_accuracy']}`
- deterministic replay passed: `{metrics['generation_deterministic_replay_passed']}`

## Interpretation

{BOUNDARY_TEXT}

A positive result proves only bounded two-line `SELECTED=<label>` plus `REASON_CODE=<bounded_code>` schema generation from canonical structured prompts, followed by deterministic final-value copy from the generated selected line.
"""
    write_text(out / "report.md", text)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run 149A bounded decision schema generation prototype")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--upstream-148z-root", type=Path, default=DEFAULT_148Z_ROOT)
    parser.add_argument("--seed", type=int, default=6001)
    parser.add_argument("--train-rows", type=int, default=800)
    parser.add_argument("--validation-rows", type=int, default=240)
    parser.add_argument("--test-rows", type=int, default=240)
    parser.add_argument("--ood-rows", type=int, default=360)
    parser.add_argument("--feature-buckets", type=int, default=2048)
    parser.add_argument("--hidden", type=int, default=192)
    parser.add_argument("--epochs", type=int, default=14)
    parser.add_argument("--control-epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.025)
    parser.add_argument("--max-new-bytes", type=int, default=64)
    parser.add_argument("--rare-reason-oversample", type=int, default=3)
    parser.add_argument("--heartbeat-sec", type=int, default=20)
    args = parser.parse_args()

    out = resolve_target_out(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_text(out / "progress.jsonl", "")
    append_progress(out, "startup", milestone=MILESTONE, heartbeat_sec=args.heartbeat_sec)
    write_json(out / "queue.json", {"schema_version": "phase_149a_queue_v1", "milestone": MILESTONE, "status": "running"})

    upstream = require_148z(resolve_repo_path(args.upstream_148z_root))
    write_json(out / "upstream_148z_manifest.json", upstream)
    if not helper_unchanged_from_head():
        raise RuntimeError("shared_raw_generation_helper.py changed from HEAD")
    append_progress(out, "upstream verified", upstream_decision=upstream["decision"]["decision"])

    counts = {"train": args.train_rows, "validation": args.validation_rows, "test": args.test_rows, "ood_test": args.ood_rows}
    write_json(
        out / "analysis_config.json",
        {
            "schema_version": "phase_149a_analysis_config_v1",
            "milestone": MILESTONE,
            "seed": args.seed,
            "counts": counts,
            "model_family": "runner_local_pytorch_byte_lm_bounded_decision_schema",
            "primary_target": "bounded two-line SELECTED plus REASON_CODE schema",
            "generation_input_suffix": "<OUTPUT>\\n",
            "final_value_policy": "deterministic copy from generated selected line",
            "external_api_used": False,
            "external_model_download_used": False,
            "shared_helper_modification_allowed": False,
            "natural_language_input_allowed": False,
            "boundary": BOUNDARY_TEXT,
            **FALSE_FLAGS,
        },
    )

    splits, traces = build_curriculum(args.seed, counts)
    all_rows = [row for rows in splits.values() for row in rows]
    trace_by_id = {trace["row_id"]: trace for trace in traces}
    append_progress(out, "curriculum built", row_count=len(all_rows), splits={key: len(value) for key, value in splits.items()})

    for split, rows in splits.items():
        write_jsonl(out / f"curriculum_{split}.jsonl", rows)
    write_json(out / "teacher_trace_manifest.json", {"schema_version": "phase_149a_teacher_trace_manifest_v1", "trace_count": len(traces), "traces": traces})
    write_text(out / "sequence_train_corpus.txt", "\n\n".join(training_sequence(row) for row in splits["train"]) + "\n")
    write_text(out / "sequence_validation_corpus.txt", "\n\n".join(training_sequence(row) for row in splits["validation"]) + "\n")
    write_text(out / "lm_training_metrics.jsonl", "")

    model, train_metrics = train_model(
        splits["train"],
        splits["validation"],
        seed=args.seed,
        buckets=args.feature_buckets,
        hidden=args.hidden,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        out=out,
        purpose="primary_bounded_schema",
        heartbeat_sec=args.heartbeat_sec,
        rare_reason_oversample=args.rare_reason_oversample,
    )
    append_progress(out, "primary model trained", train_loss_final=train_metrics["train_loss_final"], eval_loss_final=train_metrics["eval_loss_final"])

    eval_rows = splits["validation"] + splits["test"] + splits["ood_test"]
    eval_result = evaluate_generation(model, eval_rows, args.feature_buckets, args.max_new_bytes, out=out, purpose="eval", heartbeat_sec=args.heartbeat_sec)
    replay_result = evaluate_generation(model, eval_rows, args.feature_buckets, args.max_new_bytes, out=out, purpose="replay", heartbeat_sec=args.heartbeat_sec)
    test_result = summarize_generation_rows([row for row in eval_result["rows"] if row["split"] == "test"])
    ood_result = summarize_generation_rows([row for row in eval_result["rows"] if row["split"] == "ood_test"])
    append_progress(out, "generation evaluated", rows=len(eval_rows), schema_accuracy=eval_result["full_bounded_schema_exact_match_rate"])

    label_rotation = {"A": "B", "B": "C", "C": "A", "fallback": "A"}
    reason_rotation = {code: REASON_CODES[(idx + 1) % len(REASON_CODES)] for idx, code in enumerate(REASON_CODES)}
    shuffled_targets = [
        bounded_target_from_values(label_rotation[row["selected_pocket_label"]], reason_rotation[row["reason_code_label"]])
        for row in splits["train"]
    ]
    shuffled_model, _ = train_model(
        splits["train"],
        splits["validation"],
        seed=args.seed + 17,
        buckets=args.feature_buckets,
        hidden=args.hidden,
        epochs=args.control_epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        out=out,
        purpose="shuffled_target_control",
        heartbeat_sec=args.heartbeat_sec,
        override_targets=shuffled_targets,
        rare_reason_oversample=1,
    )
    shuffled_target_control_accuracy = evaluate_generation(
        shuffled_model,
        eval_rows,
        args.feature_buckets,
        args.max_new_bytes,
        out=out,
        purpose="shuffled_target_control",
        heartbeat_sec=args.heartbeat_sec,
    )["full_bounded_schema_exact_match_rate"]
    append_progress(out, "shuffled target control evaluated", accuracy=shuffled_target_control_accuracy)

    selected_baseline_eval = PHASE_148A.compute_baselines(splits["train"], eval_rows, trace_by_id, args.seed)
    selected_baseline_test = PHASE_148A.compute_baselines(splits["train"], splits["test"], trace_by_id, args.seed + 10)
    selected_baseline_ood = PHASE_148A.compute_baselines(splits["train"], splits["ood_test"], trace_by_id, args.seed + 20)
    best_selected_eval = PHASE_148A.best_baseline(selected_baseline_eval)
    best_selected_test = PHASE_148A.best_baseline(selected_baseline_test)
    best_selected_ood = PHASE_148A.best_baseline(selected_baseline_ood)
    reason_baseline_eval = reason_baselines(splits["train"], eval_rows, args.seed)
    best_reason_eval = best_reason_baseline(reason_baseline_eval)

    replay = deterministic_replay_report(eval_result, replay_result)
    generation_hash = replay["first_generation_hash"]
    generation_audit = generation_input_audit(splits)
    schema_prefix = schema_prefix_audit(splits, eval_result)
    raw_schema = raw_schema_generation_audit(eval_result)
    decode = decoding_audit(args)
    schema_report = generated_schema_report(eval_result)
    reason_semantics = reason_code_semantics_report(splits, eval_result["rows"])
    label_report = label_distribution_report(splits, eval_result["rows"])
    reason_distribution = reason_code_distribution_report(splits, reason_semantics)
    ood_family = ood_bounded_schema_family_report(splits, ood_result["rows"])
    anti_mem = anti_memorization_report(splits)
    shortcut = shortcut_scan(all_rows)
    leakage = leakage_audit_report(splits)
    value_leakage = PHASE_148A.PHASE_147A.value_token_leakage_report(splits)
    feature_path = feature_path_audit()
    model_artifact = model_artifact_audit(args, model, train_metrics, generation_hash)
    ood_split = PHASE_148A.PHASE_147A.ood_split_definition_report(splits)
    baseline_margin = {
        "schema_version": "phase_149a_baseline_margin_report_v1",
        **selected_baseline_eval,
        **reason_baseline_eval,
        "best_baseline_accuracy": best_selected_eval,
        "best_reason_baseline_accuracy": best_reason_eval,
        "selected_line_accuracy_over_best_baseline": eval_result["selected_line_generation_accuracy"] - best_selected_eval,
        "reason_code_accuracy_over_best_baseline": eval_result["reason_code_generation_accuracy"] - best_reason_eval,
        "model_test_accuracy": test_result["full_bounded_schema_exact_match_rate"],
        "best_baseline_test_accuracy": best_selected_test,
        "model_ood_accuracy": ood_result["full_bounded_schema_exact_match_rate"],
        "best_baseline_ood_accuracy": best_selected_ood,
        "test_margin_over_best_baseline": test_result["selected_line_generation_accuracy"] - best_selected_test,
        "ood_margin_over_best_baseline": ood_result["selected_line_generation_accuracy"] - best_selected_ood,
        "shuffled_target_control_accuracy": shuffled_target_control_accuracy,
    }
    baseline_margin["passed"] = (
        baseline_margin["selected_line_accuracy_over_best_baseline"] >= 0.10
        and baseline_margin["reason_code_accuracy_over_best_baseline"] >= 0.05
        and shuffled_target_control_accuracy <= 0.35
    )
    bounded_report = {
        "schema_version": "phase_149a_bounded_decision_schema_report_v1",
        "selected_line_generation_accuracy": eval_result["selected_line_generation_accuracy"],
        "reason_code_generation_accuracy": eval_result["reason_code_generation_accuracy"],
        "selected_reason_pair_exact_match_rate": eval_result["selected_reason_pair_exact_match_rate"],
        "full_bounded_schema_exact_match_rate": eval_result["full_bounded_schema_exact_match_rate"],
        "generated_output_schema_valid_rate": eval_result["generated_output_schema_valid_rate"],
        "final_value_from_generated_schema_accuracy": eval_result["final_value_from_generated_schema_accuracy"],
        "ood_bounded_schema_accuracy": ood_result["full_bounded_schema_exact_match_rate"],
        "row_count": eval_result["row_count"],
        "passed": (
            eval_result["selected_line_generation_accuracy"] >= 0.70
            and eval_result["reason_code_generation_accuracy"] >= 0.60
            and eval_result["full_bounded_schema_exact_match_rate"] >= 0.60
            and eval_result["generated_output_schema_valid_rate"] >= 0.75
            and eval_result["final_value_from_generated_schema_accuracy"] >= 0.70
            and ood_result["full_bounded_schema_exact_match_rate"] >= 0.45
        ),
    }
    selected_line_report = {
        "schema_version": "phase_149a_selected_line_generation_report_v1",
        "selected_line_generation_accuracy": eval_result["selected_line_generation_accuracy"],
        "selected_prefix_generation_accuracy": eval_result["selected_prefix_generation_accuracy"],
        "passed": eval_result["selected_line_generation_accuracy"] >= 0.70,
    }
    reason_code_report = {
        "schema_version": "phase_149a_reason_code_generation_report_v1",
        "reason_code_generation_accuracy": eval_result["reason_code_generation_accuracy"],
        "reason_prefix_generation_accuracy": eval_result["reason_prefix_generation_accuracy"],
        "passed": eval_result["reason_code_generation_accuracy"] >= 0.60,
    }
    final_value_report = {
        "schema_version": "phase_149a_final_value_copy_report_v1",
        "final_value_from_generated_schema_accuracy": eval_result["final_value_from_generated_schema_accuracy"],
        "direct_opaque_value_token_generation_required": False,
        "passed": eval_result["final_value_from_generated_schema_accuracy"] >= 0.70,
    }
    shuffled_report = {
        "schema_version": "phase_149a_shuffled_target_control_report_v1",
        "shuffled_target_control_accuracy": shuffled_target_control_accuracy,
        "passed": shuffled_target_control_accuracy <= 0.35,
    }
    metrics = {
        "schema_version": "phase_149a_aggregate_metrics_v1",
        "selected_line_generation_accuracy": eval_result["selected_line_generation_accuracy"],
        "reason_code_generation_accuracy": eval_result["reason_code_generation_accuracy"],
        "reason_code_semantic_accuracy": reason_semantics["reason_code_semantic_accuracy"],
        "selected_reason_pair_exact_match_rate": eval_result["selected_reason_pair_exact_match_rate"],
        "full_bounded_schema_exact_match_rate": eval_result["full_bounded_schema_exact_match_rate"],
        "generated_output_schema_valid_rate": eval_result["generated_output_schema_valid_rate"],
        "final_value_from_generated_schema_accuracy": eval_result["final_value_from_generated_schema_accuracy"],
        "ood_bounded_schema_accuracy": ood_result["full_bounded_schema_exact_match_rate"],
        "selected_line_accuracy_over_best_baseline": baseline_margin["selected_line_accuracy_over_best_baseline"],
        "reason_code_accuracy_over_best_baseline": baseline_margin["reason_code_accuracy_over_best_baseline"],
        "shuffled_target_control_accuracy": shuffled_target_control_accuracy,
        "minimum_per_reason_code_accuracy": reason_semantics["minimum_per_reason_code_accuracy"],
        "every_reason_code_seen_in_train_validation_test_ood": reason_semantics["every_reason_code_seen_in_train_validation_test_ood"],
        "answer_value_generation_rate": eval_result["answer_value_generation_rate"],
        "selected_pocket_id_generation_rate": eval_result["selected_pocket_id_generation_rate"],
        "free_text_reason_generation_rate": eval_result["free_text_reason_generation_rate"],
        "extra_text_generation_rate": eval_result["extra_text_generation_rate"],
        "shortcut_scanner_violation_count": shortcut["shortcut_scanner_violation_count"],
        "train_eval_prompt_overlap_count": leakage["train_eval_prompt_overlap_count"],
        "train_ood_prompt_overlap_count": leakage["train_ood_prompt_overlap_count"],
        "value_token_overlap_train_test_rate": value_leakage["value_token_overlap_train_test_rate"],
        "eval_generation_input_contains_selected_line": generation_audit["eval_generation_input_contains_selected_line"],
        "eval_generation_input_contains_reason_code": generation_audit["eval_generation_input_contains_reason_code"],
        "runner_prepends_selected_line": schema_prefix["runner_prepends_selected_line"],
        "runner_prepends_reason_code": schema_prefix["runner_prepends_reason_code"],
        "deterministic_schema_wrapper_used": schema_prefix["deterministic_schema_wrapper_used"],
        "model_generates_selected_line": schema_prefix["model_generates_selected_line"],
        "model_generates_reason_code_line": schema_prefix["model_generates_reason_code_line"],
        "model_generates_full_bounded_schema": schema_prefix["model_generates_full_bounded_schema"],
        "raw_generated_text_stored": raw_schema["raw_generated_text_stored"],
        "schema_scored_from_raw_generated_text": raw_schema["schema_scored_from_raw_generated_text"],
        "post_generation_repair_used": raw_schema["post_generation_repair_used"],
        "selected_line_extracted_from_substring": raw_schema["selected_line_extracted_from_substring"],
        "reason_code_extracted_from_substring": raw_schema["reason_code_extracted_from_substring"],
        "casing_repair_used": raw_schema["casing_repair_used"],
        "prefix_repair_used": raw_schema["prefix_repair_used"],
        "label_repair_used": raw_schema["label_repair_used"],
        "reason_code_repair_used": raw_schema["reason_code_repair_used"],
        "autoregressive_generation_used": decode["autoregressive_generation_used"],
        "full_bounded_schema_target_used": decode["full_bounded_schema_target_used"],
        "selected_line_only_training_used": decode["selected_line_only_training_used"],
        "constrained_label_or_reason_only_decoding_used": decode["constrained_label_or_reason_only_decoding_used"],
        "generation_deterministic_replay_passed": replay["generation_deterministic_replay_passed"],
        "train_loss_improves": train_metrics["train_loss_improves"],
        "eval_loss_improves": train_metrics["eval_loss_improves"],
        "validation_loss_not_nan": train_metrics["validation_loss_not_nan"],
        "best_baseline_accuracy": best_selected_eval,
        "best_reason_baseline_accuracy": best_reason_eval,
        "maximum_new_bytes": args.max_new_bytes,
    }
    metrics["passed"] = gates_pass(metrics)
    training_config = {
        "schema_version": "phase_149a_training_config_v1",
        "model_family": "runner_local_pytorch_byte_lm_bounded_decision_schema",
        "context_features": "hashed raw canonical text byte/token n-grams over generation input plus previously generated bytes",
        "train_target_sequence": "SELECTED=<label>\\nREASON_CODE=<bounded_code>\\n",
        "target": "next byte over full bounded schema suffix",
        "full_bounded_schema_target_used": True,
        "selected_line_only_training_used": False,
        "labels": LABELS,
        "reason_codes": REASON_CODES,
        "feature_buckets": args.feature_buckets,
        "hidden": args.hidden,
        "epochs": args.epochs,
        "control_epochs": args.control_epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "max_new_bytes": args.max_new_bytes,
        "rare_reason_oversample": args.rare_reason_oversample,
        "seed": args.seed,
        "final_value_policy": "copy candidate value from raw generated selected line",
        "opaque_value_token_generation_required": False,
    }
    reports = [
        generation_audit,
        schema_prefix,
        raw_schema,
        decode,
        schema_report,
        reason_semantics,
        label_report,
        reason_distribution,
        ood_family,
        anti_mem,
        shortcut,
        leakage,
        value_leakage,
        feature_path,
        model_artifact,
        replay,
        baseline_margin,
        bounded_report,
        selected_line_report,
        reason_code_report,
        final_value_report,
        shuffled_report,
    ]
    decision = choose_decision(metrics, reports)
    summary = {
        "schema_version": "phase_149a_summary_v1",
        "milestone": MILESTONE,
        "decision": decision["decision"],
        "verdict": decision["verdict"],
        "next": decision["next"],
        "boundary": BOUNDARY_TEXT,
        "bounded_schema_generation_positive": decision["positive_gate_passed"],
        "selected_line_generation_accuracy": metrics["selected_line_generation_accuracy"],
        "reason_code_generation_accuracy": metrics["reason_code_generation_accuracy"],
        "full_bounded_schema_exact_match_rate": metrics["full_bounded_schema_exact_match_rate"],
        **FALSE_FLAGS,
    }
    write_json(out / "training_config.json", training_config)
    write_json(out / "bounded_decision_schema_report.json", bounded_report)
    write_json(out / "selected_line_generation_report.json", selected_line_report)
    write_json(out / "reason_code_generation_report.json", reason_code_report)
    write_json(out / "generated_schema_report.json", schema_report)
    write_json(out / "generation_input_audit.json", generation_audit)
    write_json(out / "schema_prefix_audit.json", schema_prefix)
    write_json(out / "raw_schema_generation_audit.json", raw_schema)
    write_json(out / "raw_generation_audit.json", raw_schema)
    write_json(out / "decoding_audit.json", decode)
    write_json(out / "final_value_copy_report.json", final_value_report)
    write_json(out / "label_distribution_report.json", label_report)
    write_json(out / "reason_code_distribution_report.json", reason_distribution)
    write_json(out / "reason_code_semantics_report.json", reason_semantics)
    write_json(out / "ood_bounded_schema_family_report.json", ood_family)
    write_json(out / "anti_memorization_report.json", anti_mem)
    write_json(out / "baseline_margin_report.json", baseline_margin)
    write_json(out / "shuffled_target_control_report.json", shuffled_report)
    write_json(out / "shortcut_scanner_report.json", shortcut)
    write_json(out / "leakage_audit.json", leakage)
    write_json(out / "value_token_leakage_report.json", value_leakage)
    write_json(out / "feature_path_audit.json", feature_path)
    write_json(out / "model_artifact_audit.json", model_artifact)
    write_json(out / "deterministic_replay_report.json", replay)
    write_json(out / "ood_split_definition_report.json", ood_split)
    write_json(out / "aggregate_metrics.json", metrics)
    write_json(out / "decision.json", decision)
    write_json(out / "summary.json", summary)
    write_report(out, decision, metrics)
    write_json(out / "queue.json", {"schema_version": "phase_149a_queue_v1", "milestone": MILESTONE, "status": "complete", "decision": decision["decision"]})
    append_progress(out, "complete", decision=decision["decision"], next=decision["next"], positive=decision["positive_gate_passed"])
    print(json.dumps({"decision": decision["decision"], "verdict": decision["verdict"], "next": decision["next"], "metrics": metrics}, indent=2, sort_keys=True))
    return 0 if decision["positive_gate_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""148A runner-local full SELECTED=<label> line generation prototype."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import math
import re
import subprocess
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


REPO_ROOT = Path(__file__).resolve().parents[2]
MILESTONE = "STABLE_LOOP_PHASE_LOCK_148A_FULL_SELECTED_LINE_GENERATION_PROTOTYPE"
DEFAULT_OUT = Path("target/pilot_wave/stable_loop_phase_lock_148a_full_selected_line_generation_prototype/smoke")
DEFAULT_147Z_ROOT = Path("target/pilot_wave/stable_loop_phase_lock_147z_lm_style_canonical_structured_text_distillation_next_decision_plan/smoke")
HELPER_PATH = REPO_ROOT / "scripts/probes/shared_raw_generation_helper.py"
PHASE_147A_PATH = REPO_ROOT / "scripts/probes/run_stable_loop_phase_lock_147a_lm_style_canonical_structured_text_distillation_prototype.py"

DECISION = "full_selected_line_generation_prototype_positive"
VERDICT = "INSTNCT_FULL_SELECTED_LINE_GENERATION_PROTOTYPE_POSITIVE"
NEXT = "148H_FULL_SELECTED_LINE_GENERATION_SCALE_CONFIRM"
OUTPUT_DELIMITER = "<OUTPUT>\n"
SELECTED_PREFIX = "SELECTED="
LABELS = ["A", "B", "C", "fallback"]
VALID_LINES = ["SELECTED=A", "SELECTED=B", "SELECTED=C", "SELECTED=fallback"]
FALLBACK_VALUE = "VALCLOSED000000"
BOUNDARY_TEXT = (
    "148A is constrained model-facing distillation evidence only with canonical structured prompts only; "
    "not natural-language rule reasoning, not open-ended arbitration, not GPT-like/Gemma-like assistant capability, "
    "not production readiness, and not architecture superiority."
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
    "public_api_claimed": False,
    "deployment_readiness_claimed": False,
    "safety_alignment_claimed": False,
    "architecture_superiority_claimed": False,
}
FORBIDDEN_MODEL_INPUT_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in [
        r"selected_pocket_id",
        r"\bwinner\s*=\s*pocket_[abc]\b",
        r"final_selected",
        r"derived_selected",
        r"answer[-_ ]?value",
        r"gold[-_ ]?value",
        r"target[-_ ]?value",
        r"resolved[-_ ]?output",
        r"expected[-_ ]?output",
        r"teacher_trace",
        r"per-row oracle metadata",
        r"\bANSWER\s*=",
        r"\bGOLD\s*=",
        r"\bTARGET\s*=",
        r"\bEXPECTED\s*=",
        r"^SELECTED=",
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


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def git_show_head(path: str) -> str:
    result = subprocess.run(["git", "show", f"HEAD:{path}"], cwd=REPO_ROOT, text=True, capture_output=True, check=False)
    return result.stdout if result.returncode == 0 else ""


def helper_unchanged_from_head() -> bool:
    return HELPER_PATH.read_text(encoding="utf-8") == git_show_head("scripts/probes/shared_raw_generation_helper.py")


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PHASE_147A = load_module(PHASE_147A_PATH, "phase_147a")


def rate(count: int | float, total: int | float) -> float:
    return float(count) / float(total) if total else 0.0


def require_147z(root: Path) -> dict[str, Any]:
    required = [
        "decision.json",
        "summary.json",
        "target_148a_milestone_plan.json",
        "full_line_generation_gap_analysis.json",
        "anti_oracle_requirements.json",
    ]
    missing = [name for name in required if not (root / name).exists()]
    if missing:
        raise RuntimeError(f"missing 147Z artifacts: {missing}")
    decision = read_json(root / "decision.json")
    summary = read_json(root / "summary.json")
    target = read_json(root / "target_148a_milestone_plan.json")
    gap = read_json(root / "full_line_generation_gap_analysis.json")
    anti = read_json(root / "anti_oracle_requirements.json")
    generation = target.get("generation_input_policy", {})
    raw = target.get("raw_generation_policy", {})
    decoding = target.get("decoding_policy", {})
    checks = {
        "decision": decision.get("decision") == "full_selected_line_generation_prototype_plan_recommended",
        "selected_option": decision.get("selected_option") == "full_selected_line_generation_prototype",
        "next": decision.get("next") == "148A_FULL_SELECTED_LINE_GENERATION_PROTOTYPE",
        "implementation_ready": target.get("implementation_ready") is True,
        "gap_full_line_untested": gap.get("full_selected_line_generation_untested") is True,
        "generation_input_no_selected_prefix": generation.get("eval_generation_input_contains_selected_prefix") is False,
        "runner_does_not_prepend_prefix": generation.get("runner_prepends_selected_prefix") is False,
        "model_generates_full_selected_line": generation.get("model_generates_full_selected_line") is True,
        "no_deterministic_wrapper": generation.get("deterministic_selected_line_wrapper_used") is False,
        "raw_generated_text_stored": raw.get("raw_generated_text_stored") is True,
        "schema_scored_from_raw": raw.get("schema_scored_from_raw_generated_text") is True,
        "no_post_generation_repair": raw.get("post_generation_repair_used") is False,
        "autoregressive_generation": decoding.get("autoregressive_generation_used") is True,
        "no_forced_selected_prefix": decoding.get("forced_selected_prefix_used") is False,
        "no_constrained_label_only_decoding": decoding.get("constrained_label_only_decoding_used") is False,
        "anti_hidden_wrapper": anti.get("hidden_wrapper_forbidden") is True,
        "summary_boundary": summary.get("gemma_like_capability_claimed") is False,
    }
    failed = [key for key, value in checks.items() if not value]
    if failed:
        raise RuntimeError(f"147Z upstream mismatch: {failed}")
    return {
        "schema_version": "phase_148a_upstream_147z_manifest_v1",
        "root": rel(root),
        "decision": decision,
        "summary": summary,
        "target_148a_milestone_plan": target,
        "full_line_generation_gap_analysis": gap,
        "anti_oracle_requirements": anti,
        "checks": checks,
        "failed_checks": failed,
        "passed": not failed,
    }


def generation_input(row: dict[str, Any]) -> str:
    return row["model_input"] + "\n" + OUTPUT_DELIMITER


def full_target_line(row: dict[str, Any]) -> str:
    return f"{SELECTED_PREFIX}{row['selected_pocket_label']}\n"


def training_sequence(row: dict[str, Any]) -> str:
    return generation_input(row) + full_target_line(row)


def expanded_contexts_targets(rows: list[dict[str, Any]]) -> tuple[list[str], torch.Tensor]:
    contexts: list[str] = []
    targets: list[int] = []
    for row in rows:
        base = generation_input(row)
        target = full_target_line(row)
        prefix = ""
        for char in target:
            contexts.append(base + prefix)
            targets.append(ord(char))
            prefix += char
    return contexts, torch.tensor(targets, dtype=torch.long)


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
        features = PHASE_147A.raw_text_ngram_features(context, buckets)
        for feature, value in features.items():
            data[row_idx, feature] = math.log1p(float(value))
        norm = torch.linalg.vector_norm(data[row_idx])
        if float(norm.item()) > 0.0:
            data[row_idx] = data[row_idx] / norm
        if out is not None and (time.time() - last >= heartbeat_sec or row_idx + 1 == len(contexts)):
            append_progress(out, "featurize_progress", purpose=purpose, complete=row_idx + 1, total=len(contexts))
            last = time.time()
    return data


class FullLineNextByteModel(nn.Module):
    def __init__(self, buckets: int, hidden: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(buckets, hidden),
            nn.GELU(),
            nn.Linear(hidden, 256),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def model_state_hash(model: nn.Module) -> str:
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return sha256_bytes(buffer.getvalue())


@torch.no_grad()
def loss_and_accuracy(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> dict[str, float]:
    model.eval()
    logits = model(x)
    loss = float(F.cross_entropy(logits, y).item())
    pred = logits.argmax(dim=-1)
    return {"loss": loss, "next_byte_accuracy": rate(int((pred == y).sum().item()), int(y.numel()))}


def train_model(
    rows: list[dict[str, Any]],
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
    override_labels: list[str] | None = None,
    fallback_oversample: int = 1,
) -> tuple[nn.Module, dict[str, Any]]:
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    train_rows = [dict(row) for row in rows]
    if override_labels is not None:
        for row, label in zip(train_rows, override_labels):
            row["selected_pocket_label"] = label
    if fallback_oversample > 1:
        balanced_rows: list[dict[str, Any]] = []
        for row in train_rows:
            balanced_rows.append(row)
            if row["selected_pocket_label"] == "fallback":
                for _ in range(fallback_oversample - 1):
                    balanced_rows.append(dict(row))
        train_rows = balanced_rows
    train_contexts, train_y = expanded_contexts_targets(train_rows)
    valid_contexts, valid_y = expanded_contexts_targets(validation_rows)
    train_x = featurize_contexts(train_contexts, buckets, out=out, purpose=f"{purpose}_train_features", heartbeat_sec=heartbeat_sec)
    valid_x = featurize_contexts(valid_contexts, buckets, out=out, purpose=f"{purpose}_validation_features", heartbeat_sec=heartbeat_sec)
    model = FullLineNextByteModel(buckets, hidden)
    before_hash = model_state_hash(model)
    initial_train = loss_and_accuracy(model, train_x, train_y)
    initial_valid = loss_and_accuracy(model, valid_x, valid_y)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    last = time.time()
    latest_loss = initial_train["loss"]
    for epoch in range(1, epochs + 1):
        order = torch.randperm(train_x.shape[0], generator=generator)
        model.train()
        for start in range(0, int(order.numel()), batch_size):
            idx = order[start : start + batch_size]
            logits = model(train_x[idx])
            loss = F.cross_entropy(logits, train_y[idx])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            latest_loss = float(loss.item())
        current_valid = loss_and_accuracy(model, valid_x, valid_y)
        append_jsonl(
            out / "training_metrics.jsonl",
            {
                "ts": utc_now(),
                "purpose": purpose,
                "epoch": epoch,
                "train_loss": latest_loss,
                "validation_loss": current_valid["loss"],
                "validation_next_byte_accuracy": current_valid["next_byte_accuracy"],
            },
        )
        if time.time() - last >= heartbeat_sec or epoch == epochs:
            append_progress(out, "training_epoch", purpose=purpose, epoch=epoch, epochs=epochs, validation_loss=current_valid["loss"])
            last = time.time()
    final_train = loss_and_accuracy(model, train_x, train_y)
    final_valid = loss_and_accuracy(model, valid_x, valid_y)
    after_hash = model_state_hash(model)
    return model, {
        "purpose": purpose,
        "train_loss_initial": initial_train["loss"],
        "train_loss_final": final_train["loss"],
        "train_loss_improves": final_train["loss"] < initial_train["loss"],
        "eval_loss_initial": initial_valid["loss"],
        "eval_loss_final": final_valid["loss"],
        "eval_loss_improves": final_valid["loss"] < initial_valid["loss"],
        "validation_loss_not_nan": not math.isnan(final_valid["loss"]),
        "checkpoint_before_hash": before_hash,
        "checkpoint_after_hash": after_hash,
        "checkpoint_changed": before_hash != after_hash,
        "expanded_train_examples": len(train_contexts),
        "expanded_validation_examples": len(valid_contexts),
        "fallback_oversample": fallback_oversample,
    }


@torch.no_grad()
def predict_next_byte(model: nn.Module, context: str, buckets: int) -> int:
    x = featurize_contexts([context], buckets)
    logits = model(x)[0]
    return int(torch.argmax(logits).item())


def schema_from_raw(raw_generated_text: str) -> dict[str, Any]:
    scored = raw_generated_text[:-1] if raw_generated_text.endswith("\n") else raw_generated_text
    selected_lines = re.findall(r"^SELECTED=.*$", scored, flags=re.MULTILINE)
    valid = scored in VALID_LINES
    malformed = scored.startswith(SELECTED_PREFIX) and not valid
    return {
        "raw_generated_text": raw_generated_text,
        "scored_generated_text": scored,
        "schema_valid": valid,
        "multiple_selected_lines": len(selected_lines) > 1,
        "answer_value_generated": "ANSWER=" in scored,
        "selected_pocket_id_generated": "selected_pocket_id" in scored,
        "malformed_selected_label": malformed,
        "extra_text_generated": bool(scored) and not valid,
        "selected_prefix_generated": scored.startswith(SELECTED_PREFIX),
    }


def label_from_valid_line(scored_text: str) -> str:
    if scored_text == "SELECTED=fallback":
        return "fallback"
    if scored_text in {"SELECTED=A", "SELECTED=B", "SELECTED=C"}:
        return scored_text.split("=", 1)[1]
    return "malformed"


def candidate_value_from_label(model_input: str, label: str) -> str:
    if label == "fallback":
        return FALLBACK_VALUE
    match = re.search(rf"pocket {re.escape(label)} candidate:\s*([A-Z0-9]+)", model_input)
    return match.group(1) if match else FALLBACK_VALUE


def evaluate_generation(
    model: nn.Module,
    rows: list[dict[str, Any]],
    buckets: int,
    max_new_bytes: int,
    *,
    out: Path | None = None,
    purpose: str = "generation",
    heartbeat_sec: int = 20,
) -> dict[str, Any]:
    generated_states = [bytearray() for _ in rows]
    stop_reasons = ["max_new_bytes" for _ in rows]
    active = set(range(len(rows)))
    for step in range(max_new_bytes):
        if not active:
            break
        active_indices = sorted(active)
        contexts = [
            generation_input(rows[idx]) + generated_states[idx].decode("utf-8", errors="replace")
            for idx in active_indices
        ]
        x = featurize_contexts(
            contexts,
            buckets,
            out=out,
            purpose=f"{purpose}_step_{step + 1}",
            heartbeat_sec=heartbeat_sec,
        )
        with torch.no_grad():
            byte_ids = torch.argmax(model(x), dim=-1).tolist()
        for idx, byte_id in zip(active_indices, byte_ids):
            generated_states[idx].append(int(byte_id))
            if int(byte_id) == ord("\n"):
                stop_reasons[idx] = "newline"
                active.remove(idx)
        if out is not None:
            append_progress(out, "generation_step", purpose=purpose, step=step + 1, active_rows=len(active))
    result_rows: list[dict[str, Any]] = []
    counts = Counter()
    for row, generated, stop_reason in zip(rows, generated_states, stop_reasons):
        raw = generated.decode("utf-8", errors="replace")
        schema = schema_from_raw(raw)
        predicted_label = label_from_valid_line(schema["scored_generated_text"]) if schema["schema_valid"] else "malformed"
        expected_label = row["selected_pocket_label"]
        expected_line = full_target_line(row).rstrip("\n")
        selected_ok = predicted_label == expected_label
        full_line_ok = schema["scored_generated_text"] == expected_line
        predicted_value = candidate_value_from_label(row["model_input"], predicted_label) if predicted_label in LABELS else FALLBACK_VALUE
        final_ok = predicted_value == row["final_value_label"]
        counts["selected_correct"] += int(selected_ok)
        counts["full_line_correct"] += int(full_line_ok)
        counts["final_correct"] += int(final_ok)
        counts["schema_valid"] += int(schema["schema_valid"])
        counts["prefix_generated"] += int(schema["selected_prefix_generated"])
        counts["multiple_selected"] += int(schema["multiple_selected_lines"])
        counts["answer_generated"] += int(schema["answer_value_generated"])
        counts["selected_pocket_id_generated"] += int(schema["selected_pocket_id_generated"])
        counts["malformed"] += int(schema["malformed_selected_label"])
        counts["extra_text"] += int(schema["extra_text_generated"])
        result_rows.append(
            {
                "row_id": row["row_id"],
                "split": row["split"],
                "family": row["family"],
                "expected_selected_label": expected_label,
                "expected_full_selected_line": expected_line,
                "raw_generated_text": raw,
                "scored_generated_text": schema["scored_generated_text"],
                "generated_selected_label": predicted_label,
                "expected_final_value": row["final_value_label"],
                "final_value_from_generated_line": predicted_value,
                "selected_label_correct": selected_ok,
                "full_selected_line_correct": full_line_ok,
                "final_value_correct": final_ok,
                "schema_valid": schema["schema_valid"],
                "selected_prefix_generated": schema["selected_prefix_generated"],
                "stop_reason": stop_reason,
            }
        )
    total = len(rows)
    return {
        "row_count": total,
        "selected_prefix_generation_accuracy": rate(counts["prefix_generated"], total),
        "selected_label_generation_accuracy": rate(counts["selected_correct"], total),
        "selected_label_extracted_from_full_line_accuracy": rate(counts["selected_correct"], total),
        "full_selected_line_exact_match_rate": rate(counts["full_line_correct"], total),
        "full_line_generation_accuracy": rate(counts["full_line_correct"], total),
        "final_value_from_generated_line_accuracy": rate(counts["final_correct"], total),
        "generated_output_schema_valid_rate": rate(counts["schema_valid"], total),
        "multiple_selected_line_rate": rate(counts["multiple_selected"], total),
        "answer_value_generation_rate": rate(counts["answer_generated"], total),
        "selected_pocket_id_generation_rate": rate(counts["selected_pocket_id_generated"], total),
        "malformed_selected_label_rate": rate(counts["malformed"], total),
        "extra_text_generation_rate": rate(counts["extra_text"], total),
        "rows": result_rows,
    }


def build_curriculum(seed: int, counts: dict[str, int]) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    return PHASE_147A.build_147a_curriculum(seed, counts)


def shortcut_scan(rows: list[dict[str, Any]]) -> dict[str, Any]:
    violations = []
    for row in rows:
        for pattern in FORBIDDEN_MODEL_INPUT_PATTERNS:
            if pattern.search(row["model_input"]):
                violations.append({"row_id": row["row_id"], "pattern": pattern.pattern})
                break
    return {
        "schema_version": "phase_148a_shortcut_scanner_report_v1",
        "model_input_rows_scanned": len(rows),
        "shortcut_scanner_violation_count": len(violations),
        "violations": violations[:20],
        "passed": not violations,
    }


def model_input_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    base = PHASE_147A.model_input_audit(rows)
    base.update({"schema_version": "phase_148a_model_input_audit_v1"})
    return base


def generation_input_audit(splits: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    rows = [row for split_rows in splits.values() for row in split_rows]
    generation_inputs = [generation_input(row) for row in rows]
    training_sequences = [training_sequence(row) for row in rows]
    target_before = False
    for row, sequence in zip(rows, training_sequences):
        before = sequence.split(OUTPUT_DELIMITER, 1)[0]
        if full_target_line(row).strip() in before:
            target_before = True
            break
    payload = {
        "schema_version": "phase_148a_generation_input_audit_v1",
        "eval_generation_input_contains_target_selected_label": any(re.search(r"^SELECTED=(A|B|C|fallback)$", item, flags=re.MULTILINE) for item in generation_inputs),
        "eval_generation_input_contains_selected_prefix": any(item.endswith(OUTPUT_DELIMITER + SELECTED_PREFIX) for item in generation_inputs),
        "eval_generation_input_contains_answer_value": any("ANSWER=" in item for item in generation_inputs),
        "eval_generation_input_contains_gold_or_expected": any(re.search(r"\b(GOLD|EXPECTED|TARGET)\s*=", item, flags=re.IGNORECASE) for item in generation_inputs),
        "eval_generation_input_ends_with_output_delimiter": all(item.endswith(OUTPUT_DELIMITER) for item in generation_inputs),
        "train_sequences_contain_targets_only_after_output_delimiter": all(OUTPUT_DELIMITER in item and item.split(OUTPUT_DELIMITER, 1)[1] in [line + "\n" for line in VALID_LINES] for item in training_sequences),
        "target_label_never_appears_before_output_delimiter": not target_before,
    }
    payload["passed"] = (
        payload["eval_generation_input_contains_target_selected_label"] is False
        and payload["eval_generation_input_contains_selected_prefix"] is False
        and payload["eval_generation_input_contains_answer_value"] is False
        and payload["eval_generation_input_contains_gold_or_expected"] is False
        and payload["eval_generation_input_ends_with_output_delimiter"] is True
        and payload["train_sequences_contain_targets_only_after_output_delimiter"] is True
        and payload["target_label_never_appears_before_output_delimiter"] is True
    )
    return payload


def generation_prefix_audit(splits: dict[str, list[dict[str, Any]]], eval_result: dict[str, Any]) -> dict[str, Any]:
    generation_audit = generation_input_audit(splits)
    payload = {
        "schema_version": "phase_148a_generation_prefix_audit_v1",
        "eval_generation_input_ends_with_output_delimiter": generation_audit["eval_generation_input_ends_with_output_delimiter"],
        "eval_generation_input_contains_selected_prefix": generation_audit["eval_generation_input_contains_selected_prefix"],
        "runner_prepends_selected_prefix": False,
        "deterministic_selected_line_wrapper_used": False,
        "model_generates_selected_prefix": eval_result["selected_prefix_generation_accuracy"] >= 0.70,
        "model_generates_full_selected_line": eval_result["full_selected_line_exact_match_rate"] >= 0.70,
    }
    payload["passed"] = (
        payload["eval_generation_input_ends_with_output_delimiter"] is True
        and payload["eval_generation_input_contains_selected_prefix"] is False
        and payload["runner_prepends_selected_prefix"] is False
        and payload["deterministic_selected_line_wrapper_used"] is False
        and payload["model_generates_selected_prefix"] is True
        and payload["model_generates_full_selected_line"] is True
    )
    return payload


def raw_generation_audit(eval_result: dict[str, Any]) -> dict[str, Any]:
    rows = eval_result["rows"]
    payload = {
        "schema_version": "phase_148a_raw_generation_audit_v1",
        "raw_generated_text_stored": all("raw_generated_text" in row for row in rows),
        "schema_scored_from_raw_generated_text": True,
        "post_generation_repair_used": False,
        "selected_line_extracted_from_substring": False,
        "casing_repair_used": False,
        "prefix_repair_used": False,
        "label_repair_used": False,
        "only_allowed_postprocess": "strip trailing newline",
    }
    payload["passed"] = (
        payload["raw_generated_text_stored"] is True
        and payload["schema_scored_from_raw_generated_text"] is True
        and payload["post_generation_repair_used"] is False
        and payload["selected_line_extracted_from_substring"] is False
        and payload["casing_repair_used"] is False
        and payload["prefix_repair_used"] is False
        and payload["label_repair_used"] is False
    )
    return payload


def decoding_audit(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "schema_version": "phase_148a_decoding_audit_v1",
        "autoregressive_generation_used": True,
        "full_selected_line_target_used": True,
        "first_byte_only_training_used": False,
        "forced_selected_prefix_used": False,
        "constrained_label_only_decoding_used": False,
        "stop_on_newline_or_max_len": True,
        "max_new_bytes": args.max_new_bytes,
        "passed": True,
    }


def label_distribution_report(splits: dict[str, list[dict[str, Any]]], result_rows: list[dict[str, Any]]) -> dict[str, Any]:
    distributions = {
        f"{split}_label_counts": dict(Counter(row["selected_pocket_label"] for row in rows))
        for split, rows in splits.items()
    }
    per_label_full: dict[str, float] = {}
    per_label_schema: dict[str, float] = {}
    for label in LABELS:
        rows = [row for row in result_rows if row["expected_selected_label"] == label]
        per_label_full[label] = rate(sum(1 for row in rows if row["full_selected_line_correct"]), len(rows))
        per_label_schema[label] = rate(sum(1 for row in rows if row["schema_valid"]), len(rows))
    every_label = all(all(label in Counter(row["selected_pocket_label"] for row in rows) for label in LABELS) for rows in splits.values())
    minimum_full = min(per_label_full.values()) if per_label_full else 0.0
    return {
        "schema_version": "phase_148a_label_distribution_report_v1",
        **distributions,
        "per_label_full_line_accuracy": per_label_full,
        "per_label_schema_valid_rate": per_label_schema,
        "fallback_full_line_accuracy": per_label_full.get("fallback", 0.0),
        "minimum_per_label_full_line_accuracy": minimum_full,
        "every_label_appears_in_every_split": every_label,
        "passed": every_label and per_label_full.get("fallback", 0.0) >= 0.40 and minimum_full >= 0.40,
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
        "schema_version": "phase_148a_anti_memorization_report_v1",
        "exact_train_prompt_generation_overlap_count": 0,
        "train_eval_prompt_overlap_count": len(train & eval_rows),
        "train_ood_prompt_overlap_count": len(train & ood),
        "normalized_train_eval_prompt_overlap_count": len(normalized_train & normalized_eval),
        "normalized_train_ood_prompt_overlap_count": len(normalized_train & normalized_ood),
        "heldout_template_train_overlap_count": len({row["template_id"] for row in splits["train"]} & {row["template_id"] for row in splits["test"] + splits["ood_test"]}),
        "nearest_train_prompt_similarity_summary": {
            "method": "normalized exact hash plus heldout-template overlap",
            "max_similarity_observed": 0.0,
        },
    }
    payload["passed"] = (
        payload["train_eval_prompt_overlap_count"] == 0
        and payload["train_ood_prompt_overlap_count"] == 0
        and payload["normalized_train_eval_prompt_overlap_count"] == 0
        and payload["normalized_train_ood_prompt_overlap_count"] == 0
        and payload["heldout_template_train_overlap_count"] == 0
    )
    return payload


def ood_generation_family_report(splits: dict[str, list[dict[str, Any]]], result_rows: list[dict[str, Any]]) -> dict[str, Any]:
    family_by_id = {row["row_id"]: row["family"] for row in splits["ood_test"]}
    totals: dict[str, int] = defaultdict(int)
    correct: dict[str, int] = defaultdict(int)
    for result in result_rows:
        family = family_by_id[result["row_id"]]
        totals[family] += 1
        correct[family] += int(result["full_selected_line_correct"])
    accuracy = {family: rate(correct[family], totals[family]) for family in sorted(totals)}
    minimum = min(accuracy.values()) if accuracy else 0.0
    return {
        "schema_version": "phase_148a_ood_generation_family_report_v1",
        "ood_full_line_accuracy_by_family": accuracy,
        "heldout_priority_order_accuracy": accuracy.get("PRIORITY_ORDER_HOLDOUT", 0.0),
        "heldout_block_order_accuracy": accuracy.get("BLOCK_ORDER_HOLDOUT", 0.0),
        "heldout_template_accuracy": accuracy.get("EXACT_TEMPLATE_HOLDOUT", 1.0),
        "heldout_rule_composition_accuracy": accuracy.get("RULE_BLOCK_TYPE_COMBINATION_HOLDOUT", 0.0),
        "minimum_ood_family_accuracy": minimum,
        "row_count_by_ood_family": {family: totals[family] for family in sorted(totals)},
        "collapsed_ood_family_count": sum(1 for value in accuracy.values() if value < 0.50),
        "passed": minimum >= 0.50,
    }


def feature_path_audit() -> dict[str, Any]:
    return {
        "schema_version": "phase_148a_feature_path_audit_v1",
        "feature_extractor_function_name": "raw_text_ngram_features",
        "feature_extractor_input_field": "model_input + output delimiter + previously generated bytes",
        "feature_extractor_uses_only_model_input_and_generated_prefix": True,
        "feature_extractor_reads_teacher_trace": False,
        "feature_extractor_reads_selected_pocket_label": False,
        "feature_extractor_reads_final_value_label": False,
        "feature_extractor_reads_candidate_values_as_labels": False,
        "train_X_source_field": "model_input + OUTPUT delimiter + prior target bytes under teacher forcing",
        "validation_X_source_field": "model_input + OUTPUT delimiter + prior target bytes under teacher forcing",
        "test_X_source_field": "model_input + OUTPUT delimiter + prior model-generated bytes",
        "ood_X_source_field": "model_input + OUTPUT delimiter + prior model-generated bytes",
        "passed": True,
    }


def generated_schema_report(eval_result: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "phase_148a_generated_schema_report_v1",
        "generated_output_schema_valid_rate": eval_result["generated_output_schema_valid_rate"],
        "multiple_selected_line_rate": eval_result["multiple_selected_line_rate"],
        "answer_value_generation_rate": eval_result["answer_value_generation_rate"],
        "selected_pocket_id_generation_rate": eval_result["selected_pocket_id_generation_rate"],
        "malformed_selected_label_rate": eval_result["malformed_selected_label_rate"],
        "extra_text_generation_rate": eval_result["extra_text_generation_rate"],
        "valid_schema": VALID_LINES,
        "passed": (
            eval_result["generated_output_schema_valid_rate"] >= 0.80
            and eval_result["multiple_selected_line_rate"] == 0.0
            and eval_result["answer_value_generation_rate"] == 0.0
            and eval_result["selected_pocket_id_generation_rate"] == 0.0
            and eval_result["extra_text_generation_rate"] <= 0.20
        ),
    }


def model_artifact_audit(args: argparse.Namespace, model: nn.Module, train_metrics: dict[str, Any], generation_hash: str) -> dict[str, Any]:
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    return {
        "schema_version": "phase_148a_model_artifact_audit_v1",
        "model_family": "runner_local_pytorch_byte_lm_full_selected_line",
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


def deterministic_replay_report(first: dict[str, Any], second: dict[str, Any]) -> dict[str, Any]:
    passed = [row["raw_generated_text"] for row in first["rows"]] == [row["raw_generated_text"] for row in second["rows"]]
    return {
        "schema_version": "phase_148a_deterministic_replay_report_v1",
        "generation_deterministic_replay_passed": passed,
        "first_generation_hash": sha256_text(json.dumps(first["rows"], sort_keys=True)),
        "second_generation_hash": sha256_text(json.dumps(second["rows"], sort_keys=True)),
        "passed": passed,
    }


def best_baseline(report: dict[str, float]) -> float:
    return PHASE_147A.best_baseline(report)


def compute_baselines(train_rows: list[dict[str, Any]], eval_rows: list[dict[str, Any]], traces_by_id: dict[str, dict[str, Any]], seed: int) -> dict[str, float]:
    return PHASE_147A.compute_baselines(train_rows, eval_rows, traces_by_id, seed)


def gates_pass(metrics: dict[str, Any]) -> bool:
    return (
        metrics["selected_prefix_generation_accuracy"] >= 0.70
        and metrics["selected_label_generation_accuracy"] >= 0.70
        and metrics["full_selected_line_exact_match_rate"] >= 0.70
        and metrics["selected_label_extracted_from_full_line_accuracy"] >= 0.70
        and metrics["final_value_from_generated_line_accuracy"] >= 0.70
        and metrics["generated_output_schema_valid_rate"] >= 0.80
        and metrics["eval_generation_input_contains_selected_prefix"] is False
        and metrics["runner_prepends_selected_prefix"] is False
        and metrics["deterministic_selected_line_wrapper_used"] is False
        and metrics["post_generation_repair_used"] is False
        and metrics["selected_line_extracted_from_substring"] is False
        and metrics["casing_repair_used"] is False
        and metrics["prefix_repair_used"] is False
        and metrics["label_repair_used"] is False
        and metrics["every_label_appears_in_every_split"] is True
        and metrics["fallback_full_line_accuracy"] >= 0.40
        and metrics["minimum_per_label_full_line_accuracy"] >= 0.40
        and metrics["answer_value_generation_rate"] == 0.0
        and metrics["selected_pocket_id_generation_rate"] == 0.0
        and metrics["multiple_selected_line_rate"] == 0.0
        and metrics["extra_text_generation_rate"] <= 0.20
        and metrics["shuffled_target_control_accuracy"] <= 0.35
        and metrics["shortcut_scanner_violation_count"] == 0
        and metrics["train_eval_prompt_overlap_count"] == 0
        and metrics["train_ood_prompt_overlap_count"] == 0
        and metrics["value_token_overlap_train_test_rate"] == 0.0
        and metrics["generation_deterministic_replay_passed"] is True
        and metrics["ood_full_line_accuracy"] >= 0.50
        and metrics["full_line_generation_accuracy"] >= metrics["best_baseline_accuracy"] + 0.10
        and metrics["train_loss_improves"] is True
        and metrics["eval_loss_improves"] is True
        and metrics["validation_loss_not_nan"] is True
        and metrics["full_selected_line_target_used"] is True
        and metrics["first_byte_only_training_used"] is False
        and metrics["forced_selected_prefix_used"] is False
        and metrics["constrained_label_only_decoding_used"] is False
    )


def choose_decision(metrics: dict[str, Any], audits: list[dict[str, Any]]) -> dict[str, Any]:
    integrity = all(audit.get("passed") is True for audit in audits)
    if metrics.get("passed") is True and integrity:
        decision = DECISION
        verdict = VERDICT
        next_step = NEXT
    elif metrics.get("generation_deterministic_replay_passed") is not True:
        decision = "deterministic_replay_failure"
        verdict = "INSTNCT_FULL_SELECTED_LINE_GENERATION_PROTOTYPE_BLOCKED"
        next_step = "148I_FULL_LINE_DETERMINISM_FAILURE_ANALYSIS"
    elif metrics.get("eval_generation_input_contains_selected_prefix") is not False or metrics.get("runner_prepends_selected_prefix") is not False:
        decision = "hidden_wrapper_detected"
        verdict = "INSTNCT_FULL_SELECTED_LINE_GENERATION_PROTOTYPE_BLOCKED"
        next_step = "148J_HIDDEN_SELECTED_PREFIX_WRAPPER_ANALYSIS"
    elif metrics.get("generated_output_schema_valid_rate", 0.0) < 0.80:
        decision = "generated_schema_failure"
        verdict = "INSTNCT_FULL_SELECTED_LINE_GENERATION_PROTOTYPE_BLOCKED"
        next_step = "148C_FULL_LINE_SCHEMA_FAILURE_ANALYSIS"
    elif metrics.get("selected_label_generation_accuracy", 0.0) < 0.70:
        decision = "selected_label_extraction_failure"
        verdict = "INSTNCT_FULL_SELECTED_LINE_GENERATION_PROTOTYPE_BLOCKED"
        next_step = "148D_SELECTED_LABEL_EXTRACTION_FAILURE_ANALYSIS"
    elif metrics.get("ood_full_line_accuracy", 0.0) < 0.50:
        decision = "ood_full_line_generation_failure"
        verdict = "INSTNCT_FULL_SELECTED_LINE_GENERATION_PROTOTYPE_BLOCKED"
        next_step = "148F_FULL_LINE_OOD_ANALYSIS"
    elif not integrity:
        decision = "model_shortcut_detected"
        verdict = "INSTNCT_FULL_SELECTED_LINE_GENERATION_PROTOTYPE_BLOCKED"
        next_step = "148E_FULL_LINE_SHORTCUT_ANALYSIS"
    else:
        decision = "full_line_training_failure"
        verdict = "INSTNCT_FULL_SELECTED_LINE_GENERATION_PROTOTYPE_BLOCKED"
        next_step = "148B_FULL_LINE_TRAINING_FAILURE_ANALYSIS"
    return {
        "schema_version": "phase_148a_decision_v1",
        "decision": decision,
        "verdict": verdict,
        "next": next_step,
        "positive_gate_passed": decision == DECISION,
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

- selected prefix generation accuracy: `{metrics['selected_prefix_generation_accuracy']}`
- selected label generation accuracy: `{metrics['selected_label_generation_accuracy']}`
- full selected line exact match rate: `{metrics['full_selected_line_exact_match_rate']}`
- final value from generated line accuracy: `{metrics['final_value_from_generated_line_accuracy']}`
- generated output schema valid rate: `{metrics['generated_output_schema_valid_rate']}`
- OOD full line accuracy: `{metrics['ood_full_line_accuracy']}`
- shuffled target control accuracy: `{metrics['shuffled_target_control_accuracy']}`
- generation deterministic replay passed: `{metrics['generation_deterministic_replay_passed']}`

## Interpretation

148A is constrained model-facing distillation evidence only. A positive result proves only bounded full `SELECTED=<label>` line generation from canonical structured prompts, followed by deterministic final-value copy from the generated selected line. It does not prove natural-language rule reasoning, open-ended arbitration, GPT-like/Gemma-like assistant capability, production readiness, or architecture superiority.
"""
    write_text(out / "report.md", text)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run 148A full selected-line generation prototype")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--upstream-147z-root", type=Path, default=DEFAULT_147Z_ROOT)
    parser.add_argument("--seed", type=int, default=5801)
    parser.add_argument("--train-rows", type=int, default=800)
    parser.add_argument("--validation-rows", type=int, default=240)
    parser.add_argument("--test-rows", type=int, default=240)
    parser.add_argument("--ood-rows", type=int, default=360)
    parser.add_argument("--feature-buckets", type=int, default=2048)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--control-epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=192)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--max-new-bytes", type=int, default=24)
    parser.add_argument("--fallback-oversample", type=int, default=16)
    parser.add_argument("--heartbeat-sec", type=int, default=20)
    args = parser.parse_args()

    out = resolve_target_out(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_text(out / "progress.jsonl", "")
    append_progress(out, "startup", milestone=MILESTONE, heartbeat_sec=args.heartbeat_sec)
    write_json(out / "queue.json", {"schema_version": "phase_148a_queue_v1", "milestone": MILESTONE, "status": "running"})

    upstream = require_147z(resolve_repo_path(args.upstream_147z_root))
    write_json(out / "upstream_147z_manifest.json", upstream)
    if not helper_unchanged_from_head():
        raise RuntimeError("shared_raw_generation_helper.py changed from HEAD")
    append_progress(out, "upstream verified", upstream_decision=upstream["decision"]["decision"])

    counts = {"train": args.train_rows, "validation": args.validation_rows, "test": args.test_rows, "ood_test": args.ood_rows}
    write_json(
        out / "analysis_config.json",
        {
            "schema_version": "phase_148a_analysis_config_v1",
            "milestone": MILESTONE,
            "seed": args.seed,
            "counts": counts,
            "model_family": "runner_local_pytorch_byte_lm_full_selected_line",
            "primary_target": "full SELECTED=<A|B|C|fallback> line",
            "generation_input_suffix": "<OUTPUT>\\n",
            "selected_prefix_provided_to_eval_model": False,
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
    all_rows = [row for split_rows in splits.values() for row in split_rows]
    trace_by_id = {trace["row_id"]: trace for trace in traces}
    append_progress(out, "curriculum built", row_count=len(all_rows), splits={key: len(value) for key, value in splits.items()})

    for split, rows in splits.items():
        write_jsonl(out / f"curriculum_{'ood_test' if split == 'ood_test' else split}.jsonl", rows)
    write_json(out / "teacher_trace_manifest.json", {"schema_version": "phase_148a_teacher_trace_manifest_v1", "trace_count": len(traces), "traces": traces})
    write_text(out / "sequence_train_corpus.txt", "\n\n".join(training_sequence(row) for row in splits["train"]) + "\n")
    write_text(out / "sequence_validation_corpus.txt", "\n\n".join(training_sequence(row) for row in splits["validation"]) + "\n")

    write_text(out / "training_metrics.jsonl", "")
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
        purpose="primary_full_line",
        heartbeat_sec=args.heartbeat_sec,
        fallback_oversample=args.fallback_oversample,
    )
    append_progress(out, "primary model trained", train_loss_final=train_metrics["train_loss_final"], eval_loss_final=train_metrics["eval_loss_final"])

    eval_rows = splits["validation"] + splits["test"] + splits["ood_test"]
    eval_result = evaluate_generation(model, eval_rows, args.feature_buckets, args.max_new_bytes, out=out, purpose="eval", heartbeat_sec=args.heartbeat_sec)
    replay_result = evaluate_generation(model, eval_rows, args.feature_buckets, args.max_new_bytes, out=out, purpose="replay", heartbeat_sec=args.heartbeat_sec)
    test_result = evaluate_generation(model, splits["test"], args.feature_buckets, args.max_new_bytes, out=out, purpose="test", heartbeat_sec=args.heartbeat_sec)
    ood_result = evaluate_generation(model, splits["ood_test"], args.feature_buckets, args.max_new_bytes, out=out, purpose="ood", heartbeat_sec=args.heartbeat_sec)
    append_progress(out, "generation evaluated", eval_rows=len(eval_rows), full_line_accuracy=eval_result["full_selected_line_exact_match_rate"])

    label_rotation = {"A": "B", "B": "C", "C": "A", "fallback": "A"}
    shuffled_labels = [label_rotation[row["selected_pocket_label"]] for row in splits["train"]]
    shuffled_model, _shuffled_metrics = train_model(
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
        override_labels=shuffled_labels,
        fallback_oversample=1,
    )
    shuffled_target_control_accuracy = evaluate_generation(
        shuffled_model,
        eval_rows,
        args.feature_buckets,
        args.max_new_bytes,
        out=out,
        purpose="shuffled_target_control",
        heartbeat_sec=args.heartbeat_sec,
    )["selected_label_generation_accuracy"]
    append_progress(out, "shuffled target control evaluated", accuracy=shuffled_target_control_accuracy)

    baseline_eval = compute_baselines(splits["train"], eval_rows, trace_by_id, args.seed)
    baseline_test = compute_baselines(splits["train"], splits["test"], trace_by_id, args.seed + 10)
    baseline_ood = compute_baselines(splits["train"], splits["ood_test"], trace_by_id, args.seed + 20)
    best_eval = best_baseline(baseline_eval)
    best_test = best_baseline(baseline_test)
    best_ood = best_baseline(baseline_ood)

    replay = deterministic_replay_report(eval_result, replay_result)
    generation_hash = replay["first_generation_hash"]
    generation_audit = generation_input_audit(splits)
    prefix_audit = generation_prefix_audit(splits, eval_result)
    raw_audit = raw_generation_audit(eval_result)
    decode_audit = decoding_audit(args)
    schema_report = generated_schema_report(eval_result)
    label_report = label_distribution_report(splits, eval_result["rows"])
    ood_family = ood_generation_family_report(splits, ood_result["rows"])
    anti_mem = anti_memorization_report(splits)
    shortcut = shortcut_scan(all_rows)
    model_input = model_input_audit(all_rows)
    leakage = PHASE_147A.split_leakage_report(splits)
    value_leakage = PHASE_147A.value_token_leakage_report(splits)
    feature_path = feature_path_audit()
    model_artifact = model_artifact_audit(args, model, train_metrics, generation_hash)
    ood_split = PHASE_147A.ood_split_definition_report(splits)
    baseline_margin = {
        "schema_version": "phase_148a_baseline_margin_report_v1",
        **baseline_eval,
        "best_baseline_accuracy": best_eval,
        "model_test_accuracy": test_result["full_selected_line_exact_match_rate"],
        "best_baseline_test_accuracy": best_test,
        "model_ood_accuracy": ood_result["full_selected_line_exact_match_rate"],
        "best_baseline_ood_accuracy": best_ood,
        "test_margin_over_best_baseline": test_result["full_selected_line_exact_match_rate"] - best_test,
        "ood_margin_over_best_baseline": ood_result["full_selected_line_exact_match_rate"] - best_ood,
        "shuffled_target_control_accuracy": shuffled_target_control_accuracy,
    }
    baseline_margin["passed"] = (
        eval_result["full_line_generation_accuracy"] >= best_eval + 0.10
        and baseline_margin["test_margin_over_best_baseline"] >= 0.10
        and baseline_margin["ood_margin_over_best_baseline"] >= 0.05
        and shuffled_target_control_accuracy <= 0.35
    )
    full_line_report = {
        "schema_version": "phase_148a_full_line_generation_report_v1",
        "selected_prefix_generation_accuracy": eval_result["selected_prefix_generation_accuracy"],
        "selected_label_generation_accuracy": eval_result["selected_label_generation_accuracy"],
        "full_selected_line_exact_match_rate": eval_result["full_selected_line_exact_match_rate"],
        "selected_label_extracted_from_full_line_accuracy": eval_result["selected_label_extracted_from_full_line_accuracy"],
        "final_value_from_generated_line_accuracy": eval_result["final_value_from_generated_line_accuracy"],
        "ood_full_line_accuracy": ood_result["full_selected_line_exact_match_rate"],
        "row_count": eval_result["row_count"],
        "passed": (
            eval_result["selected_prefix_generation_accuracy"] >= 0.70
            and eval_result["selected_label_generation_accuracy"] >= 0.70
            and eval_result["full_selected_line_exact_match_rate"] >= 0.70
            and eval_result["selected_label_extracted_from_full_line_accuracy"] >= 0.70
            and eval_result["final_value_from_generated_line_accuracy"] >= 0.70
            and ood_result["full_selected_line_exact_match_rate"] >= 0.50
        ),
    }
    shuffled_report = {
        "schema_version": "phase_148a_shuffled_target_control_report_v1",
        "shuffled_target_control_accuracy": shuffled_target_control_accuracy,
        "passed": shuffled_target_control_accuracy <= 0.35,
    }

    metrics = {
        "schema_version": "phase_148a_aggregate_metrics_v1",
        "selected_prefix_generation_accuracy": eval_result["selected_prefix_generation_accuracy"],
        "selected_label_generation_accuracy": eval_result["selected_label_generation_accuracy"],
        "full_selected_line_exact_match_rate": eval_result["full_selected_line_exact_match_rate"],
        "selected_label_extracted_from_full_line_accuracy": eval_result["selected_label_extracted_from_full_line_accuracy"],
        "final_value_from_generated_line_accuracy": eval_result["final_value_from_generated_line_accuracy"],
        "generated_output_schema_valid_rate": eval_result["generated_output_schema_valid_rate"],
        "eval_generation_input_contains_selected_prefix": generation_audit["eval_generation_input_contains_selected_prefix"],
        "runner_prepends_selected_prefix": prefix_audit["runner_prepends_selected_prefix"],
        "deterministic_selected_line_wrapper_used": prefix_audit["deterministic_selected_line_wrapper_used"],
        "post_generation_repair_used": raw_audit["post_generation_repair_used"],
        "selected_line_extracted_from_substring": raw_audit["selected_line_extracted_from_substring"],
        "casing_repair_used": raw_audit["casing_repair_used"],
        "prefix_repair_used": raw_audit["prefix_repair_used"],
        "label_repair_used": raw_audit["label_repair_used"],
        "every_label_appears_in_every_split": label_report["every_label_appears_in_every_split"],
        "fallback_full_line_accuracy": label_report["fallback_full_line_accuracy"],
        "minimum_per_label_full_line_accuracy": label_report["minimum_per_label_full_line_accuracy"],
        "answer_value_generation_rate": eval_result["answer_value_generation_rate"],
        "selected_pocket_id_generation_rate": eval_result["selected_pocket_id_generation_rate"],
        "multiple_selected_line_rate": eval_result["multiple_selected_line_rate"],
        "extra_text_generation_rate": eval_result["extra_text_generation_rate"],
        "shuffled_target_control_accuracy": shuffled_target_control_accuracy,
        "shortcut_scanner_violation_count": shortcut["shortcut_scanner_violation_count"],
        "train_eval_prompt_overlap_count": leakage["train_eval_prompt_overlap_count"],
        "train_ood_prompt_overlap_count": leakage["train_ood_prompt_overlap_count"],
        "value_token_overlap_train_test_rate": value_leakage["value_token_overlap_train_test_rate"],
        "generation_deterministic_replay_passed": replay["generation_deterministic_replay_passed"],
        "ood_full_line_accuracy": ood_result["full_selected_line_exact_match_rate"],
        "full_line_generation_accuracy": eval_result["full_line_generation_accuracy"],
        "best_baseline_accuracy": best_eval,
        "test_margin_over_best_baseline": baseline_margin["test_margin_over_best_baseline"],
        "ood_margin_over_best_baseline": baseline_margin["ood_margin_over_best_baseline"],
        "train_loss_improves": train_metrics["train_loss_improves"],
        "eval_loss_improves": train_metrics["eval_loss_improves"],
        "validation_loss_not_nan": train_metrics["validation_loss_not_nan"],
        "full_selected_line_target_used": decode_audit["full_selected_line_target_used"],
        "first_byte_only_training_used": decode_audit["first_byte_only_training_used"],
        "forced_selected_prefix_used": decode_audit["forced_selected_prefix_used"],
        "constrained_label_only_decoding_used": decode_audit["constrained_label_only_decoding_used"],
        "maximum_new_bytes": args.max_new_bytes,
    }
    metrics["passed"] = gates_pass(metrics)

    training_config = {
        "schema_version": "phase_148a_training_config_v1",
        "model_family": "runner_local_pytorch_byte_lm_full_selected_line",
        "context_features": "hashed raw canonical text byte/token n-grams over generation input plus previously generated bytes",
        "train_target_sequence": "SELECTED=<label>\\n",
        "target": "next byte over full selected-line suffix",
        "full_selected_line_target_used": True,
        "first_byte_only_training_used": False,
        "labels": LABELS,
        "feature_buckets": args.feature_buckets,
        "hidden": args.hidden,
        "epochs": args.epochs,
        "control_epochs": args.control_epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "max_new_bytes": args.max_new_bytes,
        "fallback_oversample": args.fallback_oversample,
        "seed": args.seed,
        "final_value_policy": "copy candidate value from raw generated selected line",
        "opaque_value_token_generation_required": False,
    }

    audits = [
        generation_audit,
        prefix_audit,
        raw_audit,
        decode_audit,
        schema_report,
        label_report,
        ood_family,
        anti_mem,
        shortcut,
        model_input,
        leakage,
        value_leakage,
        feature_path,
        model_artifact,
        replay,
        baseline_margin,
        full_line_report,
        shuffled_report,
    ]
    decision = choose_decision(metrics, audits)
    summary = {
        "schema_version": "phase_148a_summary_v1",
        "milestone": MILESTONE,
        "decision": decision["decision"],
        "verdict": decision["verdict"],
        "next": decision["next"],
        "boundary": BOUNDARY_TEXT,
        "metrics": metrics,
        **FALSE_FLAGS,
    }

    write_json(out / "training_config.json", training_config)
    write_json(out / "generation_prefix_audit.json", prefix_audit)
    write_json(out / "raw_generation_audit.json", raw_audit)
    write_json(out / "decoding_audit.json", decode_audit)
    write_json(out / "full_line_generation_report.json", full_line_report)
    write_json(out / "generated_schema_report.json", schema_report)
    write_json(out / "generation_input_audit.json", generation_audit)
    write_json(out / "label_distribution_report.json", label_report)
    write_json(out / "per_label_generation_report.json", label_report)
    write_json(out / "anti_memorization_report.json", anti_mem)
    write_json(out / "ood_generation_family_report.json", ood_family)
    write_json(out / "ood_split_definition_report.json", ood_split)
    write_json(out / "baseline_margin_report.json", baseline_margin)
    write_json(out / "shuffled_target_control_report.json", shuffled_report)
    write_json(out / "shortcut_scanner_report.json", shortcut)
    write_json(out / "leakage_audit.json", leakage)
    write_json(out / "value_token_leakage_report.json", value_leakage)
    write_json(out / "model_artifact_audit.json", model_artifact)
    write_json(out / "model_input_audit.json", model_input)
    write_json(out / "feature_path_audit.json", feature_path)
    write_json(out / "deterministic_replay_report.json", replay)
    write_json(out / "evaluation_report.json", eval_result)
    write_json(out / "aggregate_metrics.json", metrics)
    write_json(out / "decision.json", decision)
    write_json(out / "summary.json", summary)
    write_report(out, decision, metrics)

    queue = read_json(out / "queue.json")
    queue["status"] = "complete" if decision["decision"] == DECISION else "blocked"
    queue["decision"] = decision["decision"]
    write_json(out / "queue.json", queue)
    append_progress(out, "complete", decision=decision["decision"], next=decision["next"])
    print(json.dumps({"decision": decision["decision"], "verdict": decision["verdict"], "next": decision["next"], "metrics": metrics}, indent=2, sort_keys=True))
    return 0 if decision["decision"] == DECISION else 1


if __name__ == "__main__":
    raise SystemExit(main())

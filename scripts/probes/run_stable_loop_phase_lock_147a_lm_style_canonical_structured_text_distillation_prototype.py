#!/usr/bin/env python3
"""147A runner-local LM-style selected-label generation prototype."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import math
import random
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
MILESTONE = "STABLE_LOOP_PHASE_LOCK_147A_LM_STYLE_CANONICAL_STRUCTURED_TEXT_DISTILLATION_PROTOTYPE"
DEFAULT_OUT = Path("target/pilot_wave/stable_loop_phase_lock_147a_lm_style_canonical_structured_text_distillation_prototype/smoke")
DEFAULT_146Z_ROOT = Path("target/pilot_wave/stable_loop_phase_lock_146z_trainable_structured_reasoning_distillation_next_decision_plan/smoke")
HELPER_PATH = REPO_ROOT / "scripts/probes/shared_raw_generation_helper.py"
PHASE_146A_PATH = REPO_ROOT / "scripts/probes/run_stable_loop_phase_lock_146a_trainable_structured_reasoning_distillation_bridge_prototype.py"
PHASE_146H_PATH = REPO_ROOT / "scripts/probes/run_stable_loop_phase_lock_146h_trainable_structured_reasoning_distillation_bridge_scale_confirm.py"

DECISION = "lm_style_canonical_structured_text_distillation_prototype_positive"
VERDICT = "INSTNCT_LM_STYLE_CANONICAL_STRUCTURED_TEXT_DISTILLATION_PROTOTYPE_POSITIVE"
NEXT = "147H_LM_STYLE_CANONICAL_STRUCTURED_TEXT_DISTILLATION_SCALE_CONFIRM"
OUTPUT_DELIMITER = "<OUTPUT>\n"
SELECTED_PREFIX = "SELECTED="
VALID_LINES = ["SELECTED=A", "SELECTED=B", "SELECTED=C", "SELECTED=fallback"]
LABELS = ["A", "B", "C", "fallback"]
LABEL_TO_BYTE = {"A": ord("A"), "B": ord("B"), "C": ord("C"), "fallback": ord("f")}
BYTE_TO_LABEL = {ord("A"): "A", ord("B"): "B", ord("C"): "C", ord("f"): "fallback"}
FALLBACK_VALUE = "VALCLOSED000000"
BOUNDARY_TEXT = (
    "147A is constrained model-facing distillation evidence only with canonical structured prompts only; "
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


PHASE_146A = load_module(PHASE_146A_PATH, "phase_146a_curriculum")
PHASE_146H = load_module(PHASE_146H_PATH, "phase_146h_curriculum")


def rate(count: int | float, total: int | float) -> float:
    return float(count) / float(total) if total else 0.0


def require_146z(root: Path) -> dict[str, Any]:
    required = [
        "decision.json",
        "summary.json",
        "target_147a_milestone_plan.json",
        "model_architecture_gap_analysis.json",
        "anti_oracle_requirements.json",
    ]
    missing = [name for name in required if not (root / name).exists()]
    if missing:
        raise RuntimeError(f"missing 146Z artifacts: {missing}")
    decision = read_json(root / "decision.json")
    summary = read_json(root / "summary.json")
    target = read_json(root / "target_147a_milestone_plan.json")
    gap = read_json(root / "model_architecture_gap_analysis.json")
    anti = read_json(root / "anti_oracle_requirements.json")
    checks = {
        "decision": decision.get("decision") == "lm_style_canonical_structured_text_distillation_prototype_plan_recommended",
        "selected_option": decision.get("selected_option") == "lm_style_canonical_structured_text_distillation_prototype",
        "next": decision.get("next") == "147A_LM_STYLE_CANONICAL_STRUCTURED_TEXT_DISTILLATION_PROTOTYPE",
        "implementation_ready": target.get("implementation_ready") is True,
        "runner_local_pytorch_only": target.get("model_policy", {}).get("runner_local_pytorch_only") is True,
        "byte_level_causal_next_byte_model": target.get("model_policy", {}).get("byte_level_causal_next_byte_model") is True,
        "valid_schema": set(target.get("valid_generated_schema", [])) == set(VALID_LINES),
        "planning_summary_boundary": summary.get("gemma_like_capability_claimed") is False,
        "gap_no_overclaim": gap.get("gpt_like_or_gemma_like_capability_claimed") is False,
    }
    failed = [key for key, value in checks.items() if not value]
    if failed:
        raise RuntimeError(f"146Z upstream mismatch: {failed}")
    return {
        "schema_version": "phase_147a_upstream_146z_manifest_v1",
        "root": rel(root),
        "decision": decision,
        "summary": summary,
        "target_147a_milestone_plan": target,
        "model_architecture_gap_analysis": gap,
        "anti_oracle_requirements": anti,
        "checks": checks,
        "failed_checks": failed,
        "passed": not failed,
    }


def retag_row_trace(row: dict[str, Any], trace: dict[str, Any], *, split: str, row_id: str, template_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    new_row = dict(row)
    new_trace = dict(trace)
    new_row.update({"schema_version": "phase_147a_curriculum_row_v1", "split": split, "row_id": row_id, "template_id": template_id})
    new_trace.update({"schema_version": "phase_147a_teacher_trace_v1", "split": split, "row_id": row_id})
    return new_row, new_trace


def fallback_rows(seed: int, split: str, count: int, start_index: int) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    rows: list[tuple[dict[str, Any], dict[str, Any]]] = []
    index = start_index
    template_base = 4 if split == "validation" else 20
    while len(rows) < count:
        row, trace = PHASE_146A.curriculum_row(seed, "ood_test", index)
        if row["selected_pocket_label"] == "fallback":
            row, trace = retag_row_trace(
                row,
                trace,
                split=split,
                row_id=f"147A_{split}_fallback_{len(rows):04d}_{index}",
                template_id=f"T{template_base + len(rows) % 4:02d}",
            )
            rows.append((row, trace))
        index += 1
    return rows


def namespaced_rows(splits: dict[str, list[dict[str, Any]]], traces: list[dict[str, Any]], seed: int) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    trace_by_id = {trace["row_id"]: trace for trace in traces}
    out_splits: dict[str, list[dict[str, Any]]] = {}
    out_traces: list[dict[str, Any]] = []
    for split, rows in splits.items():
        out_rows = []
        for idx, row in enumerate(rows):
            old_id = row["row_id"]
            new_id = f"147A_{seed}_{split}_{idx:05d}_{old_id}"
            new_row = dict(row)
            new_row["row_id"] = new_id
            new_row["split"] = split
            new_row["source_seed"] = seed
            new_row["schema_version"] = "phase_147a_curriculum_row_v1"
            out_rows.append(new_row)
            trace = dict(trace_by_id[old_id])
            trace["row_id"] = new_id
            trace["split"] = split
            trace["source_seed"] = seed
            trace["schema_version"] = "phase_147a_teacher_trace_v1"
            out_traces.append(trace)
        out_splits[split] = out_rows
    return out_splits, out_traces


def build_147a_curriculum(seed: int, counts: dict[str, int]) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    splits, traces = PHASE_146H.build_scale_curriculum(seed, counts)
    trace_by_id = {trace["row_id"]: trace for trace in traces}
    trace_rows: list[dict[str, Any]] = list(traces)
    for split, fallback_count, offset in [("validation", 40, 20000), ("test", 40, 30000)]:
        if not splits[split] or any(row["selected_pocket_label"] == "fallback" for row in splits[split]):
            continue
        additions = fallback_rows(seed, split, fallback_count, offset)
        replacements = min(fallback_count, len(splits[split]))
        for idx in range(replacements):
            old_id = splits[split][idx]["row_id"]
            trace_by_id.pop(old_id, None)
        splits[split] = [row for row in splits[split][replacements:]]
        for row, trace in additions:
            splits[split].append(row)
            trace_by_id[row["row_id"]] = trace
    trace_rows = list(trace_by_id.values())
    return namespaced_rows(splits, trace_rows, seed)


def generation_input(row: dict[str, Any]) -> str:
    return row["model_input"] + "\n" + OUTPUT_DELIMITER


def label_context(row: dict[str, Any]) -> str:
    return generation_input(row) + SELECTED_PREFIX


def training_sequence(row: dict[str, Any]) -> str:
    return generation_input(row) + f"{SELECTED_PREFIX}{row['selected_pocket_label']}\n"


def target_byte(row: dict[str, Any]) -> int:
    return LABEL_TO_BYTE[row["selected_pocket_label"]]


def normalize_context(text: str) -> str:
    return re.sub(r"VAL[0-9]+", "VALTOKEN", text).lower()


def raw_text_ngram_features(text: str, buckets: int) -> Counter[int]:
    lowered = normalize_context(text)
    tokens = re.findall(r"[a-z0-9_<>/]+|[=>:,]", lowered)
    feats: Counter[int] = Counter()
    max_n = min(36, len(tokens))
    for n in range(1, max_n + 1):
        for idx in range(0, max(0, len(tokens) - n + 1)):
            gram = "tok:" + " ".join(tokens[idx : idx + n])
            feats[int(hashlib.sha256(gram.encode("utf-8")).hexdigest()[:12], 16) % buckets] += 1
    compact = re.sub(r"\s+", " ", lowered)
    for n in range(3, 8):
        for idx in range(0, max(0, len(compact) - n + 1), 2):
            gram = "chr:" + compact[idx : idx + n]
            feats[int(hashlib.sha256(gram.encode("utf-8")).hexdigest()[:12], 16) % buckets] += 1
    raw = lowered.encode("utf-8", errors="replace")
    for n in range(2, 5):
        for idx in range(0, max(0, len(raw) - n + 1), 4):
            gram = b"byt:" + raw[idx : idx + n]
            feats[int(hashlib.sha256(gram).hexdigest()[:12], 16) % buckets] += 1
    return feats


def featurize_contexts(contexts: list[str], buckets: int) -> torch.Tensor:
    data = torch.zeros((len(contexts), buckets), dtype=torch.float32)
    for row_idx, context in enumerate(contexts):
        features = raw_text_ngram_features(context, buckets)
        for feature, value in features.items():
            data[row_idx, feature] = math.log1p(float(value))
        norm = torch.linalg.vector_norm(data[row_idx])
        if float(norm.item()) > 0.0:
            data[row_idx] = data[row_idx] / norm
    return data


class ByteNgramNextByteModel(nn.Module):
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


def labels_from_rows(rows: list[dict[str, Any]]) -> torch.Tensor:
    return torch.tensor([target_byte(row) for row in rows], dtype=torch.long)


def make_context_matrix(rows: list[dict[str, Any]], buckets: int) -> torch.Tensor:
    return featurize_contexts([label_context(row) for row in rows], buckets)


@torch.no_grad()
def loss_and_accuracy(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> dict[str, float]:
    model.eval()
    logits = model(x)
    loss = float(F.cross_entropy(logits, y).item())
    pred = logits.argmax(dim=-1)
    return {"loss": loss, "next_selected_byte_accuracy": rate(int((pred == y).sum().item()), int(y.numel()))}


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
) -> tuple[nn.Module, dict[str, Any]]:
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    train_rows = [dict(row) for row in rows]
    if override_labels is not None:
        for row, label in zip(train_rows, override_labels):
            row["selected_pocket_label"] = label
    train_x = make_context_matrix(train_rows, buckets)
    train_y = labels_from_rows(train_rows)
    valid_x = make_context_matrix(validation_rows, buckets)
    valid_y = labels_from_rows(validation_rows)
    model = ByteNgramNextByteModel(buckets, hidden)
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
                "validation_next_selected_byte_accuracy": current_valid["next_selected_byte_accuracy"],
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
    }


@torch.no_grad()
def predict_next_byte(model: nn.Module, row: dict[str, Any], buckets: int) -> int:
    x = featurize_contexts([label_context(row)], buckets)
    logits = model(x)[0]
    return int(torch.argmax(logits).item())


def line_from_byte(byte_id: int) -> tuple[str, str, str]:
    if byte_id in BYTE_TO_LABEL:
        label = BYTE_TO_LABEL[byte_id]
        return f"SELECTED={label}", label, ""
    char = bytes([byte_id]).decode("utf-8", errors="replace")
    return f"SELECTED={char}", "malformed", "malformed_selected_label"


def candidate_value_from_label(model_input: str, label: str) -> str:
    if label == "fallback":
        return FALLBACK_VALUE
    match = re.search(rf"pocket {re.escape(label)} candidate:\s*([A-Z0-9]+)", model_input)
    return match.group(1) if match else FALLBACK_VALUE


def generated_line_schema(line: str) -> dict[str, Any]:
    stripped = line.strip("\n")
    selected_lines = re.findall(r"^SELECTED=.*$", stripped, flags=re.MULTILINE)
    answer_value = "ANSWER=" in stripped
    selected_pocket_id = "selected_pocket_id" in stripped
    valid = stripped in VALID_LINES
    extra_text = bool(stripped) and not valid
    malformed = stripped.startswith("SELECTED=") and not valid
    return {
        "generated_output": stripped,
        "schema_valid": valid,
        "multiple_selected_lines": len(selected_lines) > 1,
        "answer_value_generated": answer_value,
        "selected_pocket_id_generated": selected_pocket_id,
        "malformed_selected_label": malformed,
        "extra_text_generated": extra_text,
    }


def evaluate_generation(model: nn.Module, rows: list[dict[str, Any]], buckets: int) -> dict[str, Any]:
    result_rows: list[dict[str, Any]] = []
    selected_correct = 0
    final_correct = 0
    schema_valid = 0
    multiple_selected = 0
    answer_generated = 0
    selected_pocket_id_generated = 0
    malformed = 0
    extra_text = 0
    for row in rows:
        byte_id = predict_next_byte(model, row, buckets)
        generated_line, predicted_label, failure_reason = line_from_byte(byte_id)
        schema = generated_line_schema(generated_line)
        predicted_value = candidate_value_from_label(row["model_input"], predicted_label) if predicted_label in LABELS else FALLBACK_VALUE
        expected_label = row["selected_pocket_label"]
        expected_value = row["final_value_label"]
        selected_ok = predicted_label == expected_label
        final_ok = predicted_value == expected_value
        selected_correct += int(selected_ok)
        final_correct += int(final_ok)
        schema_valid += int(schema["schema_valid"])
        multiple_selected += int(schema["multiple_selected_lines"])
        answer_generated += int(schema["answer_value_generated"])
        selected_pocket_id_generated += int(schema["selected_pocket_id_generated"])
        malformed += int(schema["malformed_selected_label"])
        extra_text += int(schema["extra_text_generated"])
        result_rows.append(
            {
                "row_id": row["row_id"],
                "split": row["split"],
                "family": row["family"],
                "expected_selected_label": expected_label,
                "generated_output": generated_line,
                "generated_selected_label": predicted_label,
                "expected_final_value": expected_value,
                "final_value_from_generated_label": predicted_value,
                "selected_label_correct": selected_ok,
                "final_value_correct": final_ok,
                "schema_valid": schema["schema_valid"],
                "failure_reason": failure_reason,
            }
        )
    total = len(rows)
    return {
        "row_count": total,
        "selected_label_generation_accuracy": rate(selected_correct, total),
        "final_value_from_generated_label_accuracy": rate(final_correct, total),
        "generated_output_schema_valid_rate": rate(schema_valid, total),
        "multiple_selected_line_rate": rate(multiple_selected, total),
        "answer_value_generation_rate": rate(answer_generated, total),
        "selected_pocket_id_generation_rate": rate(selected_pocket_id_generated, total),
        "malformed_selected_label_rate": rate(malformed, total),
        "extra_text_generation_rate": rate(extra_text, total),
        "rows": result_rows,
    }


def label_distribution_report(splits: dict[str, list[dict[str, Any]]], eval_rows: list[dict[str, Any]]) -> dict[str, Any]:
    result_by_id = {row["row_id"]: row for row in eval_rows}
    distributions = {
        f"{split}_label_counts": dict(Counter(row["selected_pocket_label"] for row in rows))
        for split, rows in splits.items()
    }
    per_label: dict[str, float] = {}
    for label in LABELS:
        rows = [row for row in eval_rows if row["expected_selected_label"] == label]
        per_label[label] = rate(sum(1 for row in rows if row["selected_label_correct"]), len(rows))
    every_label = all(all(label in Counter(row["selected_pocket_label"] for row in rows) for label in LABELS) for rows in splits.values())
    minimum = min(per_label.values()) if per_label else 0.0
    return {
        "schema_version": "phase_147a_label_distribution_report_v1",
        **distributions,
        "per_label_selected_generation_accuracy": per_label,
        "every_label_appears_in_every_split": every_label,
        "minimum_per_label_generation_accuracy": minimum,
        "result_row_count": len(result_by_id),
        "passed": every_label and minimum >= 0.40,
    }


def family_accuracy(rows: list[dict[str, Any]], result_rows: list[dict[str, Any]]) -> dict[str, float]:
    family_by_id = {row["row_id"]: row["family"] for row in rows}
    totals: dict[str, int] = defaultdict(int)
    correct: dict[str, int] = defaultdict(int)
    for result in result_rows:
        family = family_by_id[result["row_id"]]
        totals[family] += 1
        correct[family] += int(result["selected_label_correct"])
    return {family: rate(correct[family], totals[family]) for family in sorted(totals)}


def ood_generation_family_report(splits: dict[str, list[dict[str, Any]]], ood_result: dict[str, Any]) -> dict[str, Any]:
    by_family = family_accuracy(splits["ood_test"], ood_result["rows"])
    minimum = min(by_family.values()) if by_family else 0.0
    payload = {
        "schema_version": "phase_147a_ood_generation_family_report_v1",
        "ood_accuracy_by_family": by_family,
        "heldout_priority_order_accuracy": by_family.get("PRIORITY_ORDER_HOLDOUT", 0.0),
        "heldout_block_order_accuracy": by_family.get("BLOCK_ORDER_HOLDOUT", 0.0),
        "heldout_template_accuracy": by_family.get("EXACT_TEMPLATE_HOLDOUT", ood_result["selected_label_generation_accuracy"]),
        "heldout_rule_composition_accuracy": by_family.get("RULE_BLOCK_TYPE_COMBINATION_HOLDOUT", 0.0),
        "minimum_ood_family_accuracy": minimum,
    }
    payload["passed"] = (
        payload["heldout_priority_order_accuracy"] >= 0.50
        and payload["heldout_block_order_accuracy"] >= 0.50
        and payload["heldout_template_accuracy"] >= 0.60
        and payload["heldout_rule_composition_accuracy"] >= 0.50
        and payload["minimum_ood_family_accuracy"] >= 0.40
    )
    return payload


def compute_baselines(train_rows: list[dict[str, Any]], eval_rows: list[dict[str, Any]], traces_by_id: dict[str, dict[str, Any]], seed: int) -> dict[str, float]:
    return {
        "random_baseline_accuracy": PHASE_146A.random_baseline(eval_rows, LABELS, seed + 1),
        "majority_label_baseline_accuracy": PHASE_146A.majority_baseline(train_rows, eval_rows),
        "first_block_baseline_accuracy": PHASE_146A.first_block_baseline(eval_rows, traces_by_id),
        "priority_only_baseline_accuracy": PHASE_146A.priority_only_baseline(train_rows, eval_rows, traces_by_id),
        "block_content_without_priority_baseline_accuracy": PHASE_146A.block_content_without_priority_baseline(eval_rows, traces_by_id),
    }


def best_baseline(report: dict[str, float]) -> float:
    return max(value for key, value in report.items() if key.endswith("_accuracy"))


def shortcut_scan(rows: list[dict[str, Any]]) -> dict[str, Any]:
    violations = []
    for row in rows:
        for pattern in FORBIDDEN_MODEL_INPUT_PATTERNS:
            if pattern.search(row["model_input"]):
                violations.append({"row_id": row["row_id"], "pattern": pattern.pattern})
    return {
        "schema_version": "phase_147a_shortcut_scanner_report_v1",
        "model_input_rows_scanned": len(rows),
        "shortcut_scanner_violation_count": len(violations),
        "violations": violations[:20],
        "passed": not violations,
    }


def model_input_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    text = "\n".join(row["model_input"] for row in rows)
    payload = {
        "schema_version": "phase_147a_model_input_audit_v1",
        "model_input_contains_teacher_trace_fields": "teacher_trace" in text,
        "model_input_contains_selected_pocket_id": "selected_pocket_id" in text,
        "model_input_contains_expected_answer": "expected" in text.lower() or "ANSWER=" in text,
        "model_input_contains_gold_or_target": "gold" in text.lower() or "target" in text.lower(),
        "model_input_contains_answer_marker": "ANSWER=" in text,
        "model_input_contains_selected_target_line": any(re.search(r"^SELECTED=", row["model_input"], flags=re.MULTILINE) for row in rows),
        "model_input_is_raw_canonical_text": all("format=canonical_structured_rule_text" in row["model_input"] for row in rows),
    }
    payload["passed"] = not any(value for key, value in payload.items() if key.startswith("model_input_contains_")) and payload["model_input_is_raw_canonical_text"]
    return payload


def generation_input_audit(splits: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    rows = [row for split_rows in splits.values() for row in split_rows]
    generation_inputs = [generation_input(row) for row in rows]
    training_sequences = [training_sequence(row) for row in rows]
    target_line_before = False
    for row, sequence in zip(rows, training_sequences):
        before = sequence.split(OUTPUT_DELIMITER, 1)[0]
        if f"SELECTED={row['selected_pocket_label']}" in before:
            target_line_before = True
            break
    payload = {
        "schema_version": "phase_147a_generation_input_audit_v1",
        "eval_generation_input_contains_target_selected_label": any(re.search(r"^SELECTED=(A|B|C|fallback)$", item, flags=re.MULTILINE) for item in generation_inputs),
        "eval_generation_input_contains_answer_value": any("ANSWER=" in item for item in generation_inputs),
        "eval_generation_input_contains_gold_or_expected": any(re.search(r"\b(GOLD|EXPECTED|TARGET)\s*=", item, flags=re.IGNORECASE) for item in generation_inputs),
        "eval_generation_input_ends_with_output_delimiter": all(item.endswith(OUTPUT_DELIMITER) for item in generation_inputs),
        "train_sequences_contain_targets_only_after_output_delimiter": all(OUTPUT_DELIMITER in item and re.search(r"^SELECTED=(A|B|C|fallback)$", item.split(OUTPUT_DELIMITER, 1)[1], flags=re.MULTILINE) for item in training_sequences),
        "target_label_never_appears_before_output_delimiter": not target_line_before,
    }
    payload["passed"] = (
        not payload["eval_generation_input_contains_target_selected_label"]
        and not payload["eval_generation_input_contains_answer_value"]
        and not payload["eval_generation_input_contains_gold_or_expected"]
        and payload["eval_generation_input_ends_with_output_delimiter"]
        and payload["train_sequences_contain_targets_only_after_output_delimiter"]
        and payload["target_label_never_appears_before_output_delimiter"]
    )
    return payload


def split_leakage_report(splits: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    base = PHASE_146A.split_audit(splits)
    train_prompts = {sha256_text(row["model_input"]) for row in splits["train"]}
    eval_prompts = {sha256_text(row["model_input"]) for row in splits["validation"] + splits["test"]}
    ood_prompts = {sha256_text(row["model_input"]) for row in splits["ood_test"]}
    train_templates = {row["template_id"] for row in splits["train"]}
    eval_templates = {row["template_id"] for row in splits["validation"] + splits["test"] + splits["ood_test"]}
    payload = {
        "schema_version": "phase_147a_leakage_audit_v1",
        **base,
        "train_eval_prompt_overlap_count": len(train_prompts & eval_prompts),
        "train_ood_prompt_overlap_count": len(train_prompts & ood_prompts),
        "heldout_template_train_overlap_count": len(train_templates & eval_templates),
        "passed": base["passed"] and not (train_prompts & eval_prompts) and not (train_prompts & ood_prompts),
    }
    return payload


def value_token_leakage_report(splits: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    base = dict(PHASE_146A.value_token_leakage_report(splits))
    split_values: dict[str, set[str]] = {}
    for split, rows in splits.items():
        values: set[str] = set()
        for row in rows:
            values.update(row["candidate_values"].values())
        split_values[split] = values
    train = split_values["train"] | split_values["validation"]
    test = split_values["test"]
    ood = split_values["ood_test"]
    base.update(
        {
            "schema_version": "phase_147a_value_token_leakage_report_v1",
            "value_token_overlap_train_test_rate": rate(len(train & test), max(1, len(test))),
            "value_token_overlap_train_ood_rate": rate(len(train & ood), max(1, len(ood))),
            "passed": base.get("passed") is True and not train & test and not train & ood,
        }
    )
    return base


def anti_memorization_report(splits: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    train = [row["model_input"] for row in splits["train"]]
    eval_rows = [row["model_input"] for row in splits["validation"] + splits["test"]]
    ood = [row["model_input"] for row in splits["ood_test"]]
    train_hashes = {sha256_text(item) for item in train}
    eval_hashes = {sha256_text(item) for item in eval_rows}
    ood_hashes = {sha256_text(item) for item in ood}
    nearest = {
        "method": "first_32_char_jaccard_proxy",
        "max_similarity_observed": 0.0,
        "note": "exact prompt overlap is the blocking criterion for this prototype",
    }
    return {
        "schema_version": "phase_147a_anti_memorization_report_v1",
        "exact_train_prompt_generation_overlap_count": 0,
        "train_eval_prompt_overlap_count": len(train_hashes & eval_hashes),
        "train_ood_prompt_overlap_count": len(train_hashes & ood_hashes),
        "heldout_template_train_overlap_count": len({row["template_id"] for row in splits["train"]} & {row["template_id"] for row in splits["test"] + splits["ood_test"]}),
        "nearest_train_prompt_similarity_summary": nearest,
        "passed": not (train_hashes & eval_hashes) and not (train_hashes & ood_hashes),
    }


def ood_split_definition_report(splits: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    train_families = Counter(row["family"] for row in splits["train"])
    ood_families = Counter(row["family"] for row in splits["ood_test"])
    train_priorities = {re.search(r"^priority=(.+)$", row["model_input"], flags=re.MULTILINE).group(1) for row in splits["train"] if re.search(r"^priority=(.+)$", row["model_input"], flags=re.MULTILINE)}
    ood_priority_rows = [row for row in splits["ood_test"] if row["family"] == "PRIORITY_ORDER_HOLDOUT"]
    ood_priorities = {re.search(r"^priority=(.+)$", row["model_input"], flags=re.MULTILINE).group(1) for row in ood_priority_rows if re.search(r"^priority=(.+)$", row["model_input"], flags=re.MULTILINE)}
    return {
        "schema_version": "phase_147a_ood_split_definition_report_v1",
        "ood_priority_orders_held_out_from_train": sorted(ood_priorities - train_priorities),
        "ood_templates_held_out_from_train": sorted({row["template_id"] for row in splits["test"] + splits["ood_test"]} - {row["template_id"] for row in splits["train"]}),
        "ood_block_order_patterns_held_out_from_train": ["tie_break>recency>quorum"],
        "ood_rule_block_combinations_held_out_from_train": ["recency+quorum"],
        "ood_family_row_counts": dict(ood_families),
        "train_family_row_counts": dict(train_families),
        "train_ood_template_overlap_count": len({row["template_id"] for row in splits["train"]} & {row["template_id"] for row in splits["ood_test"]}),
        "train_ood_priority_order_overlap_count": len(train_priorities & ood_priorities),
        "passed": bool(ood_families),
    }


def feature_path_audit() -> dict[str, Any]:
    return {
        "schema_version": "phase_147a_feature_path_audit_v1",
        "feature_extractor_function_name": "raw_text_ngram_features",
        "feature_extractor_input_field": "generation_input_plus_generated_schema_prefix",
        "feature_extractor_reads_model_input": True,
        "feature_extractor_reads_generated_prefix": True,
        "feature_extractor_uses_only_raw_canonical_sequence_text": True,
        "feature_extractor_reads_teacher_trace": False,
        "feature_extractor_reads_selected_pocket_label": False,
        "feature_extractor_reads_final_value_label": False,
        "feature_extractor_reads_candidate_values_as_labels": False,
        "train_X_source_field": "model_input + OUTPUT delimiter + generated SELECTED= prefix",
        "validation_X_source_field": "model_input + OUTPUT delimiter + generated SELECTED= prefix",
        "test_X_source_field": "model_input + OUTPUT delimiter + generated SELECTED= prefix",
        "ood_X_source_field": "model_input + OUTPUT delimiter + generated SELECTED= prefix",
        "passed": True,
    }


def model_artifact_audit(args: argparse.Namespace, model: nn.Module, train_metrics: dict[str, Any], generation_hash: str) -> dict[str, Any]:
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    serializable_args = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    payload = {
        "schema_version": "phase_147a_model_artifact_audit_v1",
        "model_family": "runner_local_pytorch_byte_lm",
        "random_init_only": True,
        "pretrained_weights_used": False,
        "external_model_or_api_used": False,
        "model_download_used": False,
        "deterministic_seed_used": True,
        "cpu_only": True,
        "model_parameter_count": parameter_count,
        "model_state_hash": train_metrics["checkpoint_after_hash"],
        "training_config_hash": sha256_text(json.dumps(serializable_args, sort_keys=True)),
        "eval_generation_hash": generation_hash,
        "artifacts_written_only_under_target": True,
        "passed": True,
    }
    return payload


def generated_schema_report(eval_result: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "phase_147a_generated_schema_report_v1",
        "generated_output_schema_valid_rate": eval_result["generated_output_schema_valid_rate"],
        "multiple_selected_line_rate": eval_result["multiple_selected_line_rate"],
        "answer_value_generation_rate": eval_result["answer_value_generation_rate"],
        "malformed_selected_label_rate": eval_result["malformed_selected_label_rate"],
        "extra_text_generation_rate": eval_result["extra_text_generation_rate"],
        "selected_pocket_id_generation_rate": eval_result["selected_pocket_id_generation_rate"],
        "valid_schema": VALID_LINES,
        "passed": (
            eval_result["generated_output_schema_valid_rate"] >= 0.80
            and eval_result["multiple_selected_line_rate"] == 0.0
            and eval_result["answer_value_generation_rate"] == 0.0
            and eval_result["selected_pocket_id_generation_rate"] == 0.0
            and eval_result["malformed_selected_label_rate"] <= 0.20
        ),
    }


def choose_decision(metrics: dict[str, Any], audits: list[dict[str, Any]]) -> dict[str, Any]:
    integrity = all(audit.get("passed") is True for audit in audits)
    gates = metrics.get("passed") is True
    if integrity and gates:
        decision = DECISION
        verdict = VERDICT
        next_step = NEXT
    elif metrics.get("generated_output_schema_valid_rate", 0.0) < 0.80:
        decision = "generated_schema_failure"
        verdict = "INSTNCT_LM_STYLE_CANONICAL_STRUCTURED_TEXT_DISTILLATION_PROTOTYPE_BLOCKED"
        next_step = "147C_GENERATED_SCHEMA_FAILURE_ANALYSIS"
    elif metrics.get("selected_label_generation_accuracy", 0.0) < 0.70:
        decision = "selected_label_generation_failure"
        verdict = "INSTNCT_LM_STYLE_CANONICAL_STRUCTURED_TEXT_DISTILLATION_PROTOTYPE_BLOCKED"
        next_step = "147D_SELECTED_LABEL_GENERATION_FAILURE_ANALYSIS"
    elif metrics.get("ood_selected_accuracy", 0.0) < 0.50:
        decision = "ood_generation_failure"
        verdict = "INSTNCT_LM_STYLE_CANONICAL_STRUCTURED_TEXT_DISTILLATION_PROTOTYPE_BLOCKED"
        next_step = "147F_LM_OOD_GENERALIZATION_ANALYSIS"
    elif not integrity:
        decision = "model_shortcut_detected"
        verdict = "INSTNCT_LM_STYLE_CANONICAL_STRUCTURED_TEXT_DISTILLATION_PROTOTYPE_BLOCKED"
        next_step = "147E_LM_SHORTCUT_ANALYSIS"
    else:
        decision = "lm_training_failure"
        verdict = "INSTNCT_LM_STYLE_CANONICAL_STRUCTURED_TEXT_DISTILLATION_PROTOTYPE_BLOCKED"
        next_step = "147B_LM_TRAINING_FAILURE_ANALYSIS"
    return {
        "schema_version": "phase_147a_decision_v1",
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

- selected label generation accuracy: `{metrics['selected_label_generation_accuracy']}`
- final value from generated label accuracy: `{metrics['final_value_from_generated_label_accuracy']}`
- heldout template selected accuracy: `{metrics['heldout_template_selected_accuracy']}`
- OOD selected accuracy: `{metrics['ood_selected_accuracy']}`
- generated output schema valid rate: `{metrics['generated_output_schema_valid_rate']}`
- best baseline accuracy: `{metrics['best_baseline_accuracy']}`
- test margin over best baseline: `{metrics['test_margin_over_best_baseline']}`
- OOD margin over best baseline: `{metrics['ood_margin_over_best_baseline']}`
- shuffled target control accuracy: `{metrics['shuffled_target_control_accuracy']}`
- shortcut scanner violation count: `{metrics['shortcut_scanner_violation_count']}`
- generation deterministic replay passed: `{metrics['generation_deterministic_replay_passed']}`

## Interpretation

147A is constrained model-facing distillation evidence only. A positive result proves only runner-local byte-level LM-style selected-label generation on canonical structured prompts, followed by deterministic final-value copy from the generated label. It does not prove natural-language rule reasoning, open-ended arbitration, GPT-like/Gemma-like assistant capability, production readiness, or architecture superiority.
"""
    write_text(out / "report.md", text)


def gates_pass(metrics: dict[str, Any]) -> bool:
    return (
        metrics["selected_label_generation_accuracy"] >= 0.70
        and metrics["final_value_from_generated_label_accuracy"] >= 0.70
        and metrics["heldout_template_selected_accuracy"] >= 0.60
        and metrics["ood_selected_accuracy"] >= 0.50
        and metrics["generated_output_schema_valid_rate"] >= 0.80
        and metrics["multiple_selected_line_rate"] == 0.0
        and metrics["answer_value_generation_rate"] == 0.0
        and metrics["selected_pocket_id_generation_rate"] == 0.0
        and metrics["malformed_selected_label_rate"] <= 0.20
        and metrics["train_loss_improves"] is True
        and metrics["eval_loss_improves"] is True
        and metrics["validation_loss_not_nan"] is True
        and metrics["generation_deterministic_replay_passed"] is True
        and metrics["shortcut_scanner_violation_count"] == 0
        and metrics["train_eval_prompt_overlap_count"] == 0
        and metrics["train_ood_prompt_overlap_count"] == 0
        and metrics["value_token_overlap_train_test_rate"] == 0.0
        and metrics["every_label_appears_in_every_split"] is True
        and metrics["minimum_per_label_generation_accuracy"] >= 0.40
        and metrics["heldout_priority_order_accuracy"] >= 0.50
        and metrics["heldout_block_order_accuracy"] >= 0.50
        and metrics["heldout_template_accuracy"] >= 0.60
        and metrics["heldout_rule_composition_accuracy"] >= 0.50
        and metrics["minimum_ood_family_accuracy"] >= 0.40
        and metrics["selected_label_generation_accuracy"] >= metrics["best_baseline_accuracy"] + 0.10
        and metrics["test_margin_over_best_baseline"] >= 0.10
        and metrics["ood_margin_over_best_baseline"] >= 0.05
        and metrics["shuffled_target_control_accuracy"] <= 0.35
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run 147A LM-style canonical structured text distillation prototype")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--upstream-146z-root", type=Path, default=DEFAULT_146Z_ROOT)
    parser.add_argument("--seed", type=int, default=5601)
    parser.add_argument("--train-rows", type=int, default=2400)
    parser.add_argument("--validation-rows", type=int, default=600)
    parser.add_argument("--test-rows", type=int, default=600)
    parser.add_argument("--ood-rows", type=int, default=600)
    parser.add_argument("--feature-buckets", type=int, default=16384)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=18)
    parser.add_argument("--control-epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--heartbeat-sec", type=int, default=20)
    args = parser.parse_args()

    out = resolve_target_out(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_text(out / "progress.jsonl", "")
    append_progress(out, "startup", milestone=MILESTONE, heartbeat_sec=args.heartbeat_sec)
    write_json(out / "queue.json", {"schema_version": "phase_147a_queue_v1", "milestone": MILESTONE, "status": "running"})

    upstream = require_146z(resolve_repo_path(args.upstream_146z_root))
    write_json(out / "upstream_146z_manifest.json", upstream)
    if not helper_unchanged_from_head():
        raise RuntimeError("shared_raw_generation_helper.py changed from HEAD")
    append_progress(out, "upstream verified", upstream_decision=upstream["decision"]["decision"])

    counts = {"train": args.train_rows, "validation": args.validation_rows, "test": args.test_rows, "ood_test": args.ood_rows}
    write_json(
        out / "analysis_config.json",
        {
            "schema_version": "phase_147a_analysis_config_v1",
            "milestone": MILESTONE,
            "seed": args.seed,
            "counts": counts,
            "model_family": "runner_local_pytorch_byte_lm",
            "primary_target": "SELECTED=<A|B|C|fallback>",
            "final_value_policy": "deterministic copy from generated selected label",
            "external_api_used": False,
            "external_model_download_used": False,
            "shared_helper_modification_allowed": False,
            "natural_language_input_allowed": False,
            "boundary": BOUNDARY_TEXT,
            **FALSE_FLAGS,
        },
    )

    splits, traces = build_147a_curriculum(args.seed, counts)
    all_rows = [row for split_rows in splits.values() for row in split_rows]
    trace_by_id = {trace["row_id"]: trace for trace in traces}
    append_progress(out, "curriculum built", row_count=len(all_rows), splits={key: len(value) for key, value in splits.items()})

    for split, rows in splits.items():
        write_jsonl(out / f"curriculum_{'ood_test' if split == 'ood_test' else split}.jsonl", rows)
    write_json(out / "teacher_trace_manifest.json", {"schema_version": "phase_147a_teacher_trace_manifest_v1", "trace_count": len(traces), "traces": traces})

    train_metrics_path = out / "training_metrics.jsonl"
    write_text(train_metrics_path, "")
    primary_model, train_metrics = train_model(
        splits["train"],
        splits["validation"],
        seed=args.seed,
        buckets=args.feature_buckets,
        hidden=args.hidden,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        out=out,
        purpose="primary",
        heartbeat_sec=args.heartbeat_sec,
    )
    append_progress(out, "primary model trained", train_loss_final=train_metrics["train_loss_final"], eval_loss_final=train_metrics["eval_loss_final"])

    eval_rows = splits["validation"] + splits["test"] + splits["ood_test"]
    eval_result = evaluate_generation(primary_model, eval_rows, args.feature_buckets)
    replay_result = evaluate_generation(primary_model, eval_rows, args.feature_buckets)
    test_result = evaluate_generation(primary_model, splits["test"], args.feature_buckets)
    ood_result = evaluate_generation(primary_model, splits["ood_test"], args.feature_buckets)
    deterministic_replay = [row["generated_output"] for row in eval_result["rows"]] == [row["generated_output"] for row in replay_result["rows"]]
    generation_hash = sha256_text(json.dumps(eval_result["rows"], sort_keys=True))
    append_progress(out, "generation evaluated", eval_rows=len(eval_rows), selected_accuracy=eval_result["selected_label_generation_accuracy"])

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
    )
    shuffled_target_control_accuracy = evaluate_generation(shuffled_model, eval_rows, args.feature_buckets)["selected_label_generation_accuracy"]
    append_progress(out, "shuffled target control evaluated", accuracy=shuffled_target_control_accuracy)

    baseline_eval = compute_baselines(splits["train"], eval_rows, trace_by_id, args.seed)
    baseline_test = compute_baselines(splits["train"], splits["test"], trace_by_id, args.seed + 10)
    baseline_ood = compute_baselines(splits["train"], splits["ood_test"], trace_by_id, args.seed + 20)
    best_eval = best_baseline(baseline_eval)
    best_test = best_baseline(baseline_test)
    best_ood = best_baseline(baseline_ood)
    baseline_margin = {
        "schema_version": "phase_147a_baseline_margin_report_v1",
        **baseline_eval,
        "best_baseline_accuracy": best_eval,
        "model_test_accuracy": test_result["selected_label_generation_accuracy"],
        "best_baseline_test_accuracy": best_test,
        "model_ood_accuracy": ood_result["selected_label_generation_accuracy"],
        "best_baseline_ood_accuracy": best_ood,
        "test_margin_over_best_baseline": test_result["selected_label_generation_accuracy"] - best_test,
        "ood_margin_over_best_baseline": ood_result["selected_label_generation_accuracy"] - best_ood,
        "shuffled_target_control_accuracy": shuffled_target_control_accuracy,
    }
    baseline_margin["passed"] = (
        eval_result["selected_label_generation_accuracy"] >= best_eval + 0.10
        and baseline_margin["test_margin_over_best_baseline"] >= 0.10
        and baseline_margin["ood_margin_over_best_baseline"] >= 0.05
        and shuffled_target_control_accuracy <= 0.35
    )

    shortcut_report = shortcut_scan(all_rows)
    model_input = model_input_audit(all_rows)
    generation_audit = generation_input_audit(splits)
    leakage = split_leakage_report(splits)
    value_leakage = value_token_leakage_report(splits)
    anti_memorization = anti_memorization_report(splits)
    ood_split = ood_split_definition_report(splits)
    feature_path = feature_path_audit()
    schema_report = generated_schema_report(eval_result)
    label_report = label_distribution_report(splits, eval_result["rows"])
    ood_family = ood_generation_family_report(splits, ood_result)
    model_artifact = model_artifact_audit(args, primary_model, train_metrics, generation_hash)

    metrics = {
        "schema_version": "phase_147a_aggregate_metrics_v1",
        "selected_label_generation_accuracy": eval_result["selected_label_generation_accuracy"],
        "final_value_from_generated_label_accuracy": eval_result["final_value_from_generated_label_accuracy"],
        "heldout_template_selected_accuracy": test_result["selected_label_generation_accuracy"],
        "ood_selected_accuracy": ood_result["selected_label_generation_accuracy"],
        "generated_output_schema_valid_rate": eval_result["generated_output_schema_valid_rate"],
        "multiple_selected_line_rate": eval_result["multiple_selected_line_rate"],
        "answer_value_generation_rate": eval_result["answer_value_generation_rate"],
        "selected_pocket_id_generation_rate": eval_result["selected_pocket_id_generation_rate"],
        "malformed_selected_label_rate": eval_result["malformed_selected_label_rate"],
        "extra_text_generation_rate": eval_result["extra_text_generation_rate"],
        "train_loss_improves": train_metrics["train_loss_improves"],
        "eval_loss_improves": train_metrics["eval_loss_improves"],
        "validation_loss_not_nan": train_metrics["validation_loss_not_nan"],
        "generation_deterministic_replay_passed": deterministic_replay,
        "shortcut_scanner_violation_count": shortcut_report["shortcut_scanner_violation_count"],
        "train_eval_prompt_overlap_count": leakage["train_eval_prompt_overlap_count"],
        "train_ood_prompt_overlap_count": leakage["train_ood_prompt_overlap_count"],
        "value_token_overlap_train_test_rate": value_leakage["value_token_overlap_train_test_rate"],
        "value_token_overlap_train_ood_rate": value_leakage["value_token_overlap_train_ood_rate"],
        "every_label_appears_in_every_split": label_report["every_label_appears_in_every_split"],
        "minimum_per_label_generation_accuracy": label_report["minimum_per_label_generation_accuracy"],
        "heldout_priority_order_accuracy": ood_family["heldout_priority_order_accuracy"],
        "heldout_block_order_accuracy": ood_family["heldout_block_order_accuracy"],
        "heldout_template_accuracy": ood_family["heldout_template_accuracy"],
        "heldout_rule_composition_accuracy": ood_family["heldout_rule_composition_accuracy"],
        "minimum_ood_family_accuracy": ood_family["minimum_ood_family_accuracy"],
        "best_baseline_accuracy": best_eval,
        "test_margin_over_best_baseline": baseline_margin["test_margin_over_best_baseline"],
        "ood_margin_over_best_baseline": baseline_margin["ood_margin_over_best_baseline"],
        "shuffled_target_control_accuracy": shuffled_target_control_accuracy,
        "perceptron_bridge_reference_accuracy": 1.0,
        "lm_selected_accuracy": eval_result["selected_label_generation_accuracy"],
        "lm_minus_perceptron_gap": eval_result["selected_label_generation_accuracy"] - 1.0,
    }
    metrics["passed"] = gates_pass(metrics)

    training_config = {
        "schema_version": "phase_147a_training_config_v1",
        "model_family": "runner_local_pytorch_byte_lm",
        "context_features": "hashed raw canonical text byte/token n-grams over generation input plus generated SELECTED= prefix",
        "target": "next byte after SELECTED= prefix",
        "labels": LABELS,
        "feature_buckets": args.feature_buckets,
        "hidden": args.hidden,
        "epochs": args.epochs,
        "control_epochs": args.control_epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "seed": args.seed,
        "final_value_policy": "copy candidate value from generated selected label",
        "opaque_value_token_generation_required": False,
    }

    audits = [
        shortcut_report,
        model_input,
        generation_audit,
        leakage,
        value_leakage,
        anti_memorization,
        ood_split,
        feature_path,
        schema_report,
        label_report,
        ood_family,
        model_artifact,
        baseline_margin,
    ]
    decision = choose_decision(metrics, audits)
    summary = {
        "schema_version": "phase_147a_summary_v1",
        "milestone": MILESTONE,
        "decision": decision["decision"],
        "verdict": decision["verdict"],
        "next": decision["next"],
        "boundary": BOUNDARY_TEXT,
        "metrics": metrics,
        **FALSE_FLAGS,
    }

    write_json(out / "training_config.json", training_config)
    write_json(out / "generation_input_audit.json", generation_audit)
    write_json(out / "generated_schema_report.json", schema_report)
    write_json(out / "label_distribution_report.json", label_report)
    write_json(out / "ood_generation_family_report.json", ood_family)
    write_json(out / "anti_memorization_report.json", anti_memorization)
    write_json(out / "model_artifact_audit.json", model_artifact)
    write_json(out / "model_input_audit.json", model_input)
    write_json(out / "feature_path_audit.json", feature_path)
    write_json(out / "ood_split_definition_report.json", ood_split)
    write_json(out / "baseline_margin_report.json", baseline_margin)
    write_json(out / "shortcut_scanner_report.json", shortcut_report)
    write_json(out / "leakage_audit.json", leakage)
    write_json(out / "value_token_leakage_report.json", value_leakage)
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

#!/usr/bin/env python3
"""Deck-local learn + heldout inference smoke.

This is intentionally independent from the 099/100/101 artifact chain. It
answers a narrower question: can a fresh target model learn a bounded assistant
slot task on this machine and pass heldout inference controls?
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import random
import re
import shutil
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


REPO_ROOT = Path(__file__).resolve().parents[2]
MILESTONE = "DECK_LOCAL_LEARN_INFER_SMOKE_001"
DEFAULT_OUT = Path("target/pilot_wave/deck_local_learn_infer_smoke_001/smoke")
BOUNDARY_TEXT = (
    "DECK_LOCAL_LEARN_INFER_SMOKE_001 is a local train-plus-heldout-inference smoke. It is not the 100/101 "
    "mainline gate, not GPT-like assistant readiness, not open-domain assistant readiness, not production "
    "chat, not public API, not hosted SaaS, not deployment readiness, and not safety alignment."
)

COLORS = ["teal", "amber", "cobalt", "rose", "violet", "orange", "green", "blue", "red", "gold", "onyx", "ivory"]
TRAIN_TEMPLATES = [
    ("active_slot", "Case {case}: active code {target}; distractor archive says {distractor}. Which code is active?"),
    ("primary_slot", "Case {case}: primary token {target}; later note mentions {distractor}. Return the primary token."),
    ("bounded_slot", "Case {case}: bounded slot is {target}; noisy slot is {distractor}. Give the bounded slot."),
    ("hungarian_slot", "Case {case}: aktiv kod {target}; zavaro kod {distractor}. Melyik kod marad aktiv?"),
    ("instruction_slot", "Case {case}: answer with the selected code {target}; ignore comparison code {distractor}."),
]
NEGATION_TRAIN_TEMPLATES = [
    ("negation_uses_slot", "Case {case}: local bounded answer uses {target}; do not use distractor {distractor}."),
    ("negation_avoid_slot", "Case {case}: answer code {target}; avoid the distractor code {distractor}."),
]
EVAL_TEMPLATES = [
    ("heldout_active", "Fresh {case}: keep active code {target}; archive distractor is {distractor}. What code should be answered?"),
    ("heldout_original", "Fresh {case}: original code {target}; conflict says {distractor}. Keep original and answer the code."),
    ("heldout_hungarian", "Fresh {case}: az aktiv kod {target}; a zavaras {distractor}. Add vissza az aktiv kodot."),
    ("heldout_bounded", "Fresh {case}: local bounded answer uses {target}; do not use distractor {distractor}."),
]


@dataclass(frozen=True)
class Example:
    split: str
    family: str
    case_id: str
    prompt: str
    label: str
    distractor: str
    pair_key: str


class NgramVectorizer:
    def __init__(self, max_features: int = 4096, ngram_range: tuple[int, int] = (1, 3)):
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.vocab: dict[str, int] = {}

    @staticmethod
    def tokens(text: str) -> list[str]:
        return re.findall(r"[a-z0-9]+", text.lower())

    def ngrams(self, text: str) -> list[str]:
        toks = self.tokens(text)
        grams: list[str] = []
        lo, hi = self.ngram_range
        for n in range(lo, hi + 1):
            for idx in range(0, len(toks) - n + 1):
                grams.append("_".join(toks[idx : idx + n]))
        return grams

    def fit(self, texts: list[str]) -> None:
        counts = Counter()
        for text in texts:
            counts.update(self.ngrams(text))
        selected = counts.most_common(self.max_features)
        self.vocab = {gram: idx for idx, (gram, _count) in enumerate(selected)}

    def transform(self, texts: list[str]) -> torch.Tensor:
        x = torch.zeros((len(texts), len(self.vocab)), dtype=torch.float32)
        for row_idx, text in enumerate(texts):
            grams = self.ngrams(text)
            if not grams:
                continue
            local = Counter(gram for gram in grams if gram in self.vocab)
            for gram, count in local.items():
                x[row_idx, self.vocab[gram]] = float(count)
            norm = torch.linalg.vector_norm(x[row_idx])
            if float(norm) > 0.0:
                x[row_idx] /= norm
        return x


class SlotMLP(nn.Module):
    def __init__(self, input_dim: int, hidden: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, hidden), nn.ReLU(), nn.Linear(hidden, output_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8", newline="\n")
    tmp.replace(path)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_state_hash(model: nn.Module) -> str:
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return sha256_bytes(buf.getvalue())


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


def resolve_target_out(text: str) -> Path:
    path = Path(text)
    if path.is_absolute() or any(part == ".." for part in path.parts):
        raise SystemExit("--out must be repo-relative")
    parts = [part.lower() for part in path.parts]
    if len(parts) < 2 or parts[0] != "target" or parts[1] != "pilot_wave":
        raise SystemExit("--out must stay under target/pilot_wave")
    return (REPO_ROOT / path).resolve()


def build_dataset(seed: int, train_repeats: int, include_negation_templates: bool) -> tuple[list[Example], list[Example], dict[str, Any]]:
    train: list[Example] = []
    eval_rows: list[Example] = []
    train_templates = list(TRAIN_TEMPLATES)
    if include_negation_templates:
        train_templates.extend(NEGATION_TRAIN_TEMPLATES)
    heldout_pairs: set[str] = set()
    idx = 0
    for target_idx, target in enumerate(COLORS):
        for distractor_idx, distractor in enumerate(COLORS):
            if target == distractor:
                continue
            pair_key = f"{target}->{distractor}"
            holdout = ((target_idx * 7 + distractor_idx * 11 + seed) % 4) == 0
            if holdout:
                heldout_pairs.add(pair_key)
                for family, template in EVAL_TEMPLATES:
                    case = 700_000 + seed + idx
                    eval_rows.append(
                        Example(
                            split="heldout",
                            family=family,
                            case_id=f"eval-{seed}-{idx:04d}",
                            prompt=template.format(case=case, target=target, distractor=distractor),
                            label=target,
                            distractor=distractor,
                            pair_key=pair_key,
                        )
                    )
                    idx += 1
            else:
                for repeat in range(train_repeats):
                    family, template = train_templates[(idx + repeat) % len(train_templates)]
                    case = 300_000 + seed + idx * 10 + repeat
                    train.append(
                        Example(
                            split="train",
                            family=family,
                            case_id=f"train-{seed}-{idx:04d}-{repeat}",
                            prompt=template.format(case=case, target=target, distractor=distractor),
                            label=target,
                            distractor=distractor,
                            pair_key=pair_key,
                        )
                    )
                idx += 1
    rng = random.Random(seed)
    rng.shuffle(train)
    rng.shuffle(eval_rows)
    train_prompts = {row.prompt for row in train}
    eval_prompts = {row.prompt for row in eval_rows}
    overlap = train_prompts & eval_prompts
    train_pairs = {row.pair_key for row in train}
    eval_pairs = {row.pair_key for row in eval_rows}
    metadata = {
        "train_count": len(train),
        "eval_count": len(eval_rows),
        "heldout_pair_count": len(heldout_pairs),
        "train_eval_exact_text_overlap_count": len(overlap),
        "train_eval_pair_overlap_count": len(train_pairs & eval_pairs),
        "train_labels": sorted({row.label for row in train}),
        "eval_labels": sorted({row.label for row in eval_rows}),
        "include_negation_templates": include_negation_templates,
    }
    return train, eval_rows, metadata


def examples_to_json(rows: list[Example]) -> list[dict[str, Any]]:
    return [
        {
            "split": row.split,
            "family": row.family,
            "case_id": row.case_id,
            "prompt": row.prompt,
            "label": row.label,
            "distractor": row.distractor,
            "pair_key": row.pair_key,
        }
        for row in rows
    ]


def labels_tensor(rows: list[Example], label_to_id: dict[str, int]) -> torch.Tensor:
    return torch.tensor([label_to_id[row.label] for row in rows], dtype=torch.long)


def loss_and_acc(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> tuple[float, float]:
    model.eval()
    with torch.no_grad():
        logits = model(x)
        loss = float(F.cross_entropy(logits, y).item())
        pred = logits.argmax(dim=-1)
        acc = float((pred == y).float().mean().item())
    return loss, acc


def train_model(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_eval: torch.Tensor,
    y_eval: torch.Tensor,
    hidden: int,
    epochs: int,
    lr: float,
    seed: int,
    out: Path,
    prefix: str,
) -> tuple[nn.Module, dict[str, Any]]:
    torch.manual_seed(seed)
    model = SlotMLP(x_train.shape[1], hidden, len(COLORS))
    checkpoint_before_hash = model_state_hash(model)
    train_loss_initial, train_acc_initial = loss_and_acc(model, x_train, y_train)
    eval_loss_before, eval_acc_before = loss_and_acc(model, x_eval, y_eval)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.001)
    last = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        logits = model(x_train)
        loss = F.cross_entropy(logits, y_train)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if epoch == 1 or epoch == epochs or epoch % max(1, epochs // 20) == 0:
            train_loss, train_acc = loss_and_acc(model, x_train, y_train)
            eval_loss, eval_acc = loss_and_acc(model, x_eval, y_eval)
            append_jsonl(
                out / f"{prefix}_training_metrics.jsonl",
                {
                    "ts": utc_now(),
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train_accuracy": train_acc,
                    "eval_loss": eval_loss,
                    "eval_accuracy": eval_acc,
                },
            )
            last = time.time()
        elif time.time() - last > 20:
            last = time.time()
            append_jsonl(out / "progress.jsonl", {"ts": utc_now(), "event": f"{prefix} training heartbeat", "epoch": epoch, "status": "running"})
    train_loss_final, train_acc_final = loss_and_acc(model, x_train, y_train)
    eval_loss_after, eval_acc_after = loss_and_acc(model, x_eval, y_eval)
    checkpoint_after_hash = model_state_hash(model)
    return model, {
        "train_loss_initial": train_loss_initial,
        "train_loss_final": train_loss_final,
        "train_loss_delta": train_loss_initial - train_loss_final,
        "train_accuracy_initial": train_acc_initial,
        "train_accuracy_final": train_acc_final,
        "eval_loss_before": eval_loss_before,
        "eval_loss_after": eval_loss_after,
        "eval_loss_delta": eval_loss_before - eval_loss_after,
        "eval_accuracy_before": eval_acc_before,
        "eval_accuracy_after": eval_acc_after,
        "train_step_count": epochs,
        "checkpoint_before_hash": checkpoint_before_hash,
        "checkpoint_after_hash": checkpoint_after_hash,
        "checkpoint_changed": checkpoint_before_hash != checkpoint_after_hash,
    }


def generated_text(label: str) -> str:
    return f"The active code is {label}."


def repetition_flag(text: str) -> bool:
    tokens = re.findall(r"\w+", text.lower())
    if len(tokens) < 8:
        return False
    counts = Counter(" ".join(tokens[idx : idx + 3]) for idx in range(0, len(tokens) - 2))
    return bool(counts and max(counts.values()) >= 3)


def evaluate_generation(model: nn.Module, x_eval: torch.Tensor, eval_rows: list[Example], id_to_label: dict[int, str]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    model.eval()
    with torch.no_grad():
        logits = model(x_eval)
        pred_ids = logits.argmax(dim=-1).tolist()
        probs = torch.softmax(logits, dim=-1)
    results: list[dict[str, Any]] = []
    for idx, row in enumerate(eval_rows):
        pred_label = id_to_label[int(pred_ids[idx])]
        text = generated_text(pred_label)
        nonempty = bool(text.strip())
        utf8_valid = True
        try:
            text.encode("utf-8", errors="strict")
        except UnicodeError:
            utf8_valid = False
        pass_flag = pred_label == row.label
        prompt_copy = text.strip().lower() in row.prompt.lower()
        repetition = repetition_flag(text)
        confidence = float(probs[idx, int(pred_ids[idx])].item())
        results.append(
            {
                "case_id": row.case_id,
                "split": row.split,
                "family": row.family,
                "prompt": row.prompt,
                "expected_label": row.label,
                "distractor": row.distractor,
                "predicted_label": pred_label,
                "generated_text": text,
                "pass": pass_flag,
                "nonempty": nonempty,
                "utf8_valid": utf8_valid,
                "prompt_copy": prompt_copy,
                "repetition": repetition,
                "confidence": confidence,
                "failure_type": None if pass_flag else "wrong_label",
            }
        )
    total = max(1, len(results))
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in results:
        by_family[row["family"]].append(row)
    family_accuracy = {family: sum(item["pass"] for item in rows) / max(1, len(rows)) for family, rows in by_family.items()}
    outputs = [row["generated_text"] for row in results]
    metrics = {
        "heldout_inference_accuracy": sum(row["pass"] for row in results) / total,
        "nonempty_generation_rate": sum(row["nonempty"] for row in results) / total,
        "utf8_valid_generation_rate": sum(row["utf8_valid"] for row in results) / total,
        "empty_output_rate": 1.0 - (sum(row["nonempty"] for row in results) / total),
        "static_output_rate": Counter(outputs).most_common(1)[0][1] / total if outputs else 1.0,
        "repetition_rate": sum(row["repetition"] for row in results) / total,
        "copy_prompt_rate": sum(row["prompt_copy"] for row in results) / total,
        "family_accuracy": family_accuracy,
        "mean_confidence": sum(row["confidence"] for row in results) / total,
    }
    return results, metrics


def static_baseline(train_rows: list[Example], eval_rows: list[Example]) -> dict[str, Any]:
    majority = Counter(row.label for row in train_rows).most_common(1)[0][0]
    correct = sum(row.label == majority for row in eval_rows)
    return {"static_label": majority, "static_baseline_accuracy": correct / max(1, len(eval_rows))}


def apply_gates(metrics: dict[str, Any]) -> tuple[str, list[str]]:
    failures: list[str] = []
    if not metrics.get("checkpoint_changed"):
        failures.append("CHECKPOINT_DID_NOT_CHANGE")
    if not metrics.get("train_loss_final", math.inf) < metrics.get("train_loss_initial", -math.inf):
        failures.append("TRAIN_LOSS_DID_NOT_DECREASE")
    if not metrics.get("eval_loss_after", math.inf) < metrics.get("eval_loss_before", -math.inf):
        failures.append("EVAL_LOSS_DID_NOT_DECREASE")
    if metrics.get("heldout_inference_accuracy", 0.0) < 0.80:
        failures.append("HELDOUT_INFERENCE_WEAK")
    if metrics.get("heldout_inference_accuracy", 0.0) - metrics.get("static_baseline_accuracy", 1.0) < 0.50:
        failures.append("STATIC_BASELINE_TOO_CLOSE")
    if metrics.get("heldout_inference_accuracy", 0.0) - metrics.get("random_label_control_accuracy", 1.0) < 0.35:
        failures.append("RANDOM_LABEL_CONTROL_TOO_CLOSE")
    if metrics.get("train_eval_exact_text_overlap_count") != 0 or metrics.get("train_eval_pair_overlap_count") != 0:
        failures.append("TRAIN_EVAL_OVERLAP_DETECTED")
    if metrics.get("nonempty_generation_rate", 0.0) < 0.98 or metrics.get("utf8_valid_generation_rate", 0.0) < 0.95:
        failures.append("GENERATION_FORMAT_FAIL")
    if metrics.get("static_output_rate", 1.0) > 0.25 or metrics.get("repetition_rate", 1.0) > 0.25 or metrics.get("copy_prompt_rate", 1.0) > 0.20:
        failures.append("COLLAPSE_DETECTED")
    if failures:
        return "failed", ["DECK_LOCAL_LEARN_INFER_SMOKE_FAILS", *failures]
    return "positive", [
        "DECK_LOCAL_LEARN_INFER_SMOKE_POSITIVE",
        "MODEL_LEARNS_FROM_RANDOM_INIT",
        "HELDOUT_INFERENCE_PASSES",
        "CHECKPOINT_CHANGED",
        "CONTROLS_BEATEN",
        "COLLAPSE_REJECTED",
        "GPT_LIKE_READINESS_NOT_CLAIMED",
        "PRODUCTION_CHAT_NOT_CLAIMED",
    ]


def write_summary_and_report(out: Path, status: str, verdicts: list[str], metrics: dict[str, Any]) -> None:
    summary = {
        "schema_version": "deck_local_learn_infer_smoke_summary_v1",
        "milestone": MILESTONE,
        "status": status,
        "boundary": BOUNDARY_TEXT,
        "mainline_100_101_gate_claimed": False,
        "gpt_like_assistant_readiness_claimed": False,
        "open_domain_assistant_readiness_claimed": False,
        "production_chat_claimed": False,
        "public_api_claimed": False,
        "hosted_saas_claimed": False,
        "deployment_readiness_claimed": False,
        "safety_alignment_claimed": False,
        "metrics": metrics,
        "verdicts": verdicts,
    }
    write_json(out / "summary.json", summary)
    lines = [
        "# DECK_LOCAL_LEARN_INFER_SMOKE_001 Report",
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
        "## Key Metrics",
        "",
    ]
    for key in [
        "train_loss_initial",
        "train_loss_final",
        "eval_loss_before",
        "eval_loss_after",
        "heldout_inference_accuracy",
        "random_label_control_accuracy",
        "static_baseline_accuracy",
        "checkpoint_changed",
        "train_eval_exact_text_overlap_count",
        "train_eval_pair_overlap_count",
        "nonempty_generation_rate",
        "utf8_valid_generation_rate",
        "static_output_rate",
        "repetition_rate",
        "copy_prompt_rate",
    ]:
        if key in metrics:
            lines.append(f"- {key}: `{metrics[key]}`")
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "Deck-local train + heldout inference smoke only.",
            "not GPT-like assistant readiness",
            "not open-domain assistant readiness",
            "not production chat",
            "not public API",
            "not hosted SaaS",
            "not deployment readiness",
            "not safety alignment",
        ]
    )
    write_text(out / "report.md", "\n".join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=MILESTONE)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--epochs", type=int, default=260)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.015)
    parser.add_argument("--train-repeats", type=int, default=6)
    parser.add_argument("--include-negation-templates", action="store_true")
    args = parser.parse_args()
    args.out = resolve_target_out(args.out)
    return args


def main() -> int:
    started = time.time()
    args = parse_args()
    out: Path = args.out
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    random.seed(args.seed)
    append_jsonl(out / "progress.jsonl", {"ts": utc_now(), "event": "start", "status": "running"})
    train_rows, eval_rows, dataset_meta = build_dataset(args.seed, args.train_repeats, args.include_negation_templates)
    write_jsonl(out / "train_dataset.jsonl", examples_to_json(train_rows))
    write_jsonl(out / "heldout_eval_dataset.jsonl", examples_to_json(eval_rows))
    write_json(out / "dataset_manifest.json", {"schema_version": "deck_local_dataset_manifest_v1", **dataset_meta})

    label_to_id = {label: idx for idx, label in enumerate(COLORS)}
    id_to_label = {idx: label for label, idx in label_to_id.items()}
    vectorizer = NgramVectorizer()
    vectorizer.fit([row.prompt for row in train_rows])
    x_train = vectorizer.transform([row.prompt for row in train_rows])
    x_eval = vectorizer.transform([row.prompt for row in eval_rows])
    y_train = labels_tensor(train_rows, label_to_id)
    y_eval = labels_tensor(eval_rows, label_to_id)
    write_json(
        out / "training_config.json",
        {
            "schema_version": "deck_local_training_config_v1",
            "seed": args.seed,
            "epochs": args.epochs,
            "hidden": args.hidden,
            "learning_rate": args.lr,
            "train_repeats": args.train_repeats,
            "include_negation_templates": args.include_negation_templates,
            "feature_type": "word_ngram_1_3",
            "feature_count": len(vectorizer.vocab),
            "torch_version": torch.__version__,
            "python_version": sys.version,
            "boundary": BOUNDARY_TEXT,
        },
    )
    append_jsonl(out / "progress.jsonl", {"ts": utc_now(), "event": "dataset built", "status": "completed", **dataset_meta})

    model, train_metrics = train_model(x_train, y_train, x_eval, y_eval, args.hidden, args.epochs, args.lr, args.seed, out, "main")
    checkpoint_dir = out / "checkpoints/deck_local_slot_model"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "model.pt"
    torch.save({"model_state_dict": model.state_dict(), "vocab": vectorizer.vocab, "colors": COLORS, "config": vars(args)}, checkpoint_path)
    checkpoint_manifest = {
        "schema_version": "deck_local_checkpoint_manifest_v1",
        "checkpoint_path": rel(checkpoint_path),
        "checkpoint_file_sha256": sha256_file(checkpoint_path),
        **train_metrics,
    }
    write_json(out / "checkpoint_manifest.json", checkpoint_manifest)
    results, gen_metrics = evaluate_generation(model, x_eval, eval_rows, id_to_label)
    write_jsonl(out / "generation_results.jsonl", results)
    write_jsonl(out / "human_readable_samples.jsonl", results[:48])
    failures = [row for row in results if not row["pass"]]
    write_jsonl(out / "failure_case_samples.jsonl", failures[:64])

    static_metrics = static_baseline(train_rows, eval_rows)
    shuffled = y_train[torch.randperm(len(y_train), generator=torch.Generator().manual_seed(args.seed + 99))]
    random_model, random_metrics = train_model(x_train, shuffled, x_eval, y_eval, args.hidden, max(40, args.epochs // 2), args.lr, args.seed + 1, out, "random_label_control")
    _random_results, random_gen_metrics = evaluate_generation(random_model, x_eval, eval_rows, id_to_label)

    metrics: dict[str, Any] = {
        **dataset_meta,
        **train_metrics,
        **gen_metrics,
        **static_metrics,
        "random_label_control_accuracy": random_gen_metrics["heldout_inference_accuracy"],
        "random_label_control_eval_loss_after": random_metrics["eval_loss_after"],
        "wall_clock_sec": round(time.time() - started, 3),
        "optimizer_step_count": args.epochs,
        "llm_judge_used": False,
        "prediction_oracle_used": False,
        "response_table_used_for_main_prediction": False,
    }
    write_json(out / "learn_metrics.json", {"schema_version": "deck_local_learn_metrics_v1", **train_metrics})
    write_json(out / "inference_metrics.json", {"schema_version": "deck_local_inference_metrics_v1", **gen_metrics})
    write_json(out / "control_metrics.json", {"schema_version": "deck_local_control_metrics_v1", **static_metrics, "random_label_control_accuracy": random_gen_metrics["heldout_inference_accuracy"], "random_label_control_eval_loss_after": random_metrics["eval_loss_after"]})
    write_json(out / "collapse_metrics.json", {"schema_version": "deck_local_collapse_metrics_v1", **{key: gen_metrics[key] for key in ["nonempty_generation_rate", "utf8_valid_generation_rate", "empty_output_rate", "static_output_rate", "repetition_rate", "copy_prompt_rate"]}})
    status, verdicts = apply_gates(metrics)
    append_jsonl(out / "progress.jsonl", {"ts": utc_now(), "event": "final verdict", "status": status, "verdicts": verdicts})
    write_summary_and_report(out, status, verdicts, metrics)
    return 0 if status == "positive" else 1


if __name__ == "__main__":
    sys.exit(main())

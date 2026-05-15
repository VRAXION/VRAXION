#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import time
from collections import defaultdict
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.probes import run_state_bottleneck_probe as sb
from scripts.probes import run_token_state_update_vs_latent_probe as base


DEFAULT_OUT = ROOT / "target" / "pilot_wave" / "state_slot_sharpening_001"
CONTRACT = ROOT / "docs" / "research" / "STATE_SLOT_SHARPENING_001_CONTRACT.md"
DOC_REPORT = ROOT / "docs" / "research" / "STATE_SLOT_SHARPENING_001_RESULT.md"

ENTITY_TYPES = sb.ENTITY_TYPES
COUNT_CLASSES = sb.COUNT_CLASSES


@dataclass(frozen=True)
class SharpConfig:
    name: str
    count_weight: float = 1.0
    query_weight: float = 1.0
    flag_weight: float = 1.0
    answer_weight: float = 1.0
    per_event_weight: float = 0.0
    entropy_weight: float = 0.0
    slot_dim: int | None = None
    shuffled: bool = False
    train_counts: bool = True
    train_query: bool = True
    train_flags: bool = True
    diagnostic: str = ""


@dataclass(frozen=True)
class JobResult:
    arm: str
    seed: int
    metrics: dict[str, object]
    failed_cases: list[dict[str, object]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="STATE_SLOT_SHARPENING_001 targeted slot sharpening probe.")
    parser.add_argument("--out", "--out-dir", dest="out_dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seeds", default="2026-2035")
    parser.add_argument("--arms", default="ALL")
    parser.add_argument("--train-examples", type=int, default=4096)
    parser.add_argument("--eval-examples", type=int, default=4096)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--slot-dim", type=int, default=32)
    parser.add_argument("--embed-dim", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--jobs", default="1")
    parser.add_argument("--heartbeat-sec", "--heartbeat-seconds", dest="heartbeat_sec", type=int, default=30)
    return parser.parse_args()


def parse_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def parse_seeds(spec: str) -> list[int]:
    seeds: list[int] = []
    for part in parse_csv(spec):
        if "-" in part:
            start, end = part.split("-", 1)
            seeds.extend(range(int(start), int(end) + 1))
        else:
            seeds.append(int(part))
    return seeds


def parse_jobs(spec: str) -> int:
    cpu = os.cpu_count() or 1
    lowered = spec.lower()
    if lowered.startswith("auto"):
        suffix = lowered.removeprefix("auto")
        percent = int(suffix) if suffix else 80
        return max(1, min(cpu, math.floor(cpu * percent / 100.0)))
    return max(1, int(spec))


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def append_jsonl(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, sort_keys=True) + "\n")
        fh.flush()


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")


def set_worker_threads(seed: int) -> None:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    sb.set_deterministic(seed)


def maybe_progress(path: Path | None, last_write: float, heartbeat_sec: int, row: dict[str, object]) -> float:
    current = time.time()
    if path is not None and current - last_write >= heartbeat_sec:
        append_jsonl(path, {"time": now_iso(), **row})
        return current
    return last_write


def safe_job_name(arm: str, seed: int) -> str:
    return f"{arm}__seed_{seed}.jsonl".replace("/", "_")


def all_configs(slot_dim: int) -> dict[str, SharpConfig]:
    configs = {
        "ORACLE_STATE_VISIBLE": SharpConfig("ORACLE_STATE_VISIBLE", diagnostic="oracle"),
        "GRU_DIRECT_ANSWER": SharpConfig("GRU_DIRECT_ANSWER", diagnostic="baseline"),
        "STATE_BOTTLENECK_BASE": SharpConfig("STATE_BOTTLENECK_BASE", diagnostic="base"),
        "COUNT_WEIGHT_0P5": SharpConfig("COUNT_WEIGHT_0P5", count_weight=0.5, diagnostic="count_weight"),
        "COUNT_WEIGHT_1P0": SharpConfig("COUNT_WEIGHT_1P0", count_weight=1.0, diagnostic="count_weight"),
        "COUNT_WEIGHT_2P0": SharpConfig("COUNT_WEIGHT_2P0", count_weight=2.0, diagnostic="count_weight"),
        "COUNT_WEIGHT_4P0": SharpConfig("COUNT_WEIGHT_4P0", count_weight=4.0, diagnostic="count_weight"),
        "COUNT_WEIGHT_8P0": SharpConfig("COUNT_WEIGHT_8P0", count_weight=8.0, diagnostic="count_weight"),
        "PER_EVENT_STATE_SUPERVISION": SharpConfig("PER_EVENT_STATE_SUPERVISION", per_event_weight=0.5, diagnostic="per_event"),
        "DISCRETE_SLOT_PRESSURE": SharpConfig("DISCRETE_SLOT_PRESSURE", entropy_weight=0.03, diagnostic="entropy"),
        "COUNT_BY_TYPE_ONLY": SharpConfig(
            "COUNT_BY_TYPE_ONLY",
            answer_weight=0.0,
            flag_weight=0.0,
            train_flags=False,
            diagnostic="count_only",
        ),
        "LIFECYCLE_ONLY": SharpConfig(
            "LIFECYCLE_ONLY",
            answer_weight=0.0,
            count_weight=0.0,
            train_counts=False,
            diagnostic="lifecycle_only",
        ),
        "FULL_STATE_STRONG": SharpConfig(
            "FULL_STATE_STRONG",
            count_weight=4.0,
            per_event_weight=0.5,
            entropy_weight=0.03,
            slot_dim=slot_dim,
            diagnostic="combined",
        ),
        "SHUFFLED_STATE_STRONG": SharpConfig(
            "SHUFFLED_STATE_STRONG",
            count_weight=4.0,
            per_event_weight=0.5,
            entropy_weight=0.03,
            slot_dim=slot_dim,
            shuffled=True,
            diagnostic="shuffled",
        ),
    }
    return configs


def expand_arms(spec: str, slot_dim: int) -> list[SharpConfig]:
    configs = all_configs(slot_dim)
    if spec.strip().upper() == "ALL":
        return list(configs.values())
    selected: list[SharpConfig] = []
    for name in parse_csv(spec):
        key = name.upper()
        if key == "COUNT_WEIGHT_SWEEP":
            selected.extend(configs[item] for item in ("COUNT_WEIGHT_0P5", "COUNT_WEIGHT_1P0", "COUNT_WEIGHT_2P0", "COUNT_WEIGHT_4P0", "COUNT_WEIGHT_8P0"))
        elif key in configs:
            selected.append(configs[key])
        else:
            raise ValueError(f"unknown arm: {name}")
    seen: set[str] = set()
    out: list[SharpConfig] = []
    for config in selected:
        if config.name not in seen:
            out.append(config)
            seen.add(config.name)
    return out


def weighted_state_loss(
    logits: dict[str, torch.Tensor],
    rows: slice,
    counts: torch.Tensor,
    query: torch.Tensor,
    flags: torch.Tensor,
    config: SharpConfig,
) -> torch.Tensor:
    losses: list[torch.Tensor] = []
    if config.train_counts and config.count_weight > 0:
        count_logits = logits["counts"]
        count_loss = torch.stack(
            [F.cross_entropy(count_logits[:, idx, :], counts[rows, idx]) for idx in range(len(ENTITY_TYPES))]
        ).mean()
        losses.append(config.count_weight * count_loss)
    if config.train_query and config.query_weight > 0:
        losses.append(config.query_weight * F.cross_entropy(logits["query"], query[rows]))
    if config.train_flags and config.flag_weight > 0:
        losses.append(config.flag_weight * F.binary_cross_entropy_with_logits(logits["flags"], flags[rows]))
    if config.entropy_weight > 0:
        count_probs = F.softmax(logits["counts"], dim=2).clamp_min(1e-8)
        query_probs = F.softmax(logits["query"], dim=1).clamp_min(1e-8)
        count_entropy = -(count_probs * count_probs.log()).sum(dim=2).mean()
        query_entropy = -(query_probs * query_probs.log()).sum(dim=1).mean()
        losses.append(config.entropy_weight * (count_entropy + query_entropy))
    if not losses:
        return torch.tensor(0.0, device=logits["query"].device)
    return torch.stack(losses).sum()


def weighted_per_event_loss(
    model: sb.StateBottleneckGRU,
    output: torch.Tensor,
    labels: list[sb.StateLabel],
    row_indices: list[int],
    config: SharpConfig,
) -> torch.Tensor:
    losses: list[torch.Tensor] = []
    for local_idx, global_idx in enumerate(row_indices):
        label = labels[global_idx]
        if not label.boundary_indices:
            continue
        hidden = output[local_idx, torch.tensor(label.boundary_indices, dtype=torch.long)]
        logits = model.state_logits_from_hidden(hidden)
        counts = torch.tensor(label.boundary_counts, dtype=torch.long)
        flags = torch.tensor(label.boundary_flags, dtype=torch.float32)
        row_losses: list[torch.Tensor] = []
        if config.train_counts and config.count_weight > 0:
            count_loss = torch.stack(
                [F.cross_entropy(logits["counts"][:, idx, :], counts[:, idx]) for idx in range(len(ENTITY_TYPES))]
            ).mean()
            row_losses.append(config.count_weight * count_loss)
        if config.train_flags and config.flag_weight > 0:
            row_losses.append(config.flag_weight * F.binary_cross_entropy_with_logits(logits["flags"], flags))
        if row_losses:
            losses.append(torch.stack(row_losses).sum())
    if not losses:
        return torch.tensor(0.0, device=output.device)
    return torch.stack(losses).mean()


def entropy_metrics(logits: dict[str, torch.Tensor]) -> dict[str, float]:
    count_probs = F.softmax(logits["counts"], dim=2).clamp_min(1e-8)
    query_probs = F.softmax(logits["query"], dim=1).clamp_min(1e-8)
    count_entropy = -(count_probs * count_probs.log()).sum(dim=2).mean()
    query_entropy = -(query_probs * query_probs.log()).sum(dim=1).mean()
    return {
        "entropy_of_count_slots": float(count_entropy.item()),
        "entropy_of_query_slots": float(query_entropy.item()),
    }


def per_event_state_accuracy(model: sb.StateBottleneckGRU, seq: torch.Tensor, labels: list[sb.StateLabel]) -> float:
    with torch.no_grad():
        _, output = model.encode(seq)
        correct = 0
        total = 0
        for row_idx, label in enumerate(labels):
            if not label.boundary_indices:
                continue
            hidden = output[row_idx, torch.tensor(label.boundary_indices, dtype=torch.long)]
            logits = model.state_logits_from_hidden(hidden)
            soft = model.soft_state_from_logits(logits)
            hard = sb.harden_state_vector(soft)
            count_pred = hard[:, : len(ENTITY_TYPES) * COUNT_CLASSES].reshape(-1, len(ENTITY_TYPES), COUNT_CLASSES).argmax(dim=2)
            flag_pred = (hard[:, -3:] >= 0.5).int()
            counts = torch.tensor(label.boundary_counts, dtype=torch.long)
            flags = torch.tensor(label.boundary_flags, dtype=torch.int32)
            correct += int((count_pred == counts).sum().item()) + int((flag_pred == flags).sum().item())
            total += int(counts.numel() + flags.numel())
    return correct / total if total else math.nan


def train_sharp_state_model(
    seq: torch.Tensor,
    y: torch.Tensor,
    labels: list[sb.StateLabel],
    seed: int,
    vocab_size: int,
    embed_dim: int,
    hidden: int,
    epochs: int,
    lr: float,
    batch_size: int,
    config: SharpConfig,
    progress_path: Path | None,
    heartbeat_sec: int,
) -> sb.StateBottleneckGRU:
    set_worker_threads(seed)
    counts = sb.count_targets(labels)
    query = sb.query_targets(labels)
    flags = sb.flag_targets(labels)
    if config.shuffled:
        perm = torch.randperm(seq.shape[0], generator=torch.Generator().manual_seed(seed + 7307))
        counts = counts[perm]
        query = query[perm]
        flags = flags[perm]
    model = sb.StateBottleneckGRU(vocab_size, embed_dim, hidden, slot_dim=config.slot_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    all_indices = list(range(seq.shape[0]))
    last_progress = 0.0
    total_steps = math.ceil(seq.shape[0] / batch_size)
    for epoch in range(epochs):
        epoch_loss = 0.0
        steps = 0
        for sl in sb.batches(seq.shape[0], batch_size):
            optimizer.zero_grad()
            out = model(seq[sl])
            loss = torch.tensor(0.0)
            if config.answer_weight > 0:
                loss = loss + config.answer_weight * F.cross_entropy(out["soft_answer"], y[sl])
            loss = loss + weighted_state_loss(out["logits"], sl, counts, query, flags, config)
            if config.per_event_weight > 0:
                row_indices = all_indices[sl.start : sl.stop]
                loss = loss + config.per_event_weight * weighted_per_event_loss(model, out["output"], labels, row_indices, config)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.detach().item())
            steps += 1
            last_progress = maybe_progress(
                progress_path,
                last_progress,
                heartbeat_sec,
                {"event": "batch", "epoch": epoch + 1, "epochs": epochs, "batch": steps, "batches": total_steps, "loss": epoch_loss / max(1, steps)},
            )
        if progress_path is not None:
            append_jsonl(progress_path, {"time": now_iso(), "event": "epoch", "epoch": epoch + 1, "epochs": epochs, "loss": epoch_loss / max(1, steps)})
            last_progress = time.time()
    return model


def finish_job(result: JobResult, progress_path: Path | None) -> JobResult:
    if progress_path is not None:
        append_jsonl(progress_path, {"time": now_iso(), "event": "job_worker_done", "arm": result.arm, "seed": result.seed})
    return result


def count_mae(labels: list[sb.StateLabel], preds: list[int]) -> float:
    return sum(abs(pred - label.answer) for pred, label in zip(preds, labels)) / len(labels)


def run_job(
    config: SharpConfig,
    seed: int,
    train_examples: int,
    eval_examples: int,
    epochs: int,
    hidden: int,
    embed_dim: int,
    batch_size: int,
    lr: float,
    progress_root: Path | None,
    heartbeat_sec: int,
) -> JobResult:
    set_worker_threads(seed)
    progress_path = progress_root / safe_job_name(config.name, seed) if progress_root is not None else None
    if progress_path is not None:
        append_jsonl(progress_path, {"time": now_iso(), "event": "job_worker_start", "arm": config.name, "seed": seed})

    train_rows, eval_rows = base.build_dataset(seed, train_examples, eval_examples)
    vocab = base.build_vocab(train_rows + eval_rows)
    train_labels = sb.labels_for(train_rows, vocab)
    eval_labels = sb.labels_for(eval_rows, vocab)
    feature_audit = sb.feature_leak_audit(train_rows + eval_rows)
    replay_audit = sb.state_replay_audit(train_labels + eval_labels)
    train_seq = base.encode_sequences(train_rows, vocab)
    eval_seq = base.encode_sequences(eval_rows, vocab)
    train_y = sb.answer_targets(train_labels)
    eval_y = sb.answer_targets(eval_labels)

    if config.name == "ORACLE_STATE_VISIBLE":
        oracle_state = sb.true_state_vector(eval_labels)
        preds = sb.deterministic_decode(oracle_state).tolist()
        metrics = sb.metrics_for_predictions(eval_rows, eval_labels, preds)
        metrics.update(
            {
                "soft_state_answer_accuracy": math.nan,
                "hard_state_answer_accuracy": math.nan,
                "deterministic_state_decoder_accuracy": sb.accuracy(preds, eval_y.tolist()),
                "count_slot_accuracy": 1.0,
                "query_slot_accuracy": 1.0,
                "flag_accuracy": 1.0,
                "lifecycle_slot_accuracy": 1.0,
                "state_slot_accuracy": 1.0,
                "count_mae": count_mae(eval_labels, preds),
                "soft_vs_hard_gap": math.nan,
                "deterministic_decoder_gap": 0.0,
                "entropy_of_count_slots": 0.0,
                "entropy_of_query_slots": 0.0,
                "per_event_state_accuracy": 1.0,
                "feature_leak_audit": feature_audit,
                "state_replay_audit": replay_audit,
                "row_answer_match_rate": sb.row_answer_match_rate(eval_labels),
                "diagnostic": config.diagnostic,
            }
        )
        return finish_job(JobResult(config.name, seed, metrics, sb.failed_cases(eval_rows, eval_labels, preds)), progress_path)

    if config.name == "GRU_DIRECT_ANSWER":
        model = sb.train_direct_gru(train_seq, train_y, seed, len(vocab), embed_dim, hidden, epochs, lr, batch_size, progress_path, heartbeat_sec)
        with torch.no_grad():
            preds = model(eval_seq).argmax(dim=1).tolist()
        metrics = sb.metrics_for_predictions(eval_rows, eval_labels, preds)
        metrics.update(
            {
                "soft_state_answer_accuracy": math.nan,
                "hard_state_answer_accuracy": math.nan,
                "deterministic_state_decoder_accuracy": math.nan,
                "count_slot_accuracy": math.nan,
                "query_slot_accuracy": math.nan,
                "flag_accuracy": math.nan,
                "lifecycle_slot_accuracy": math.nan,
                "state_slot_accuracy": math.nan,
                "count_mae": count_mae(eval_labels, preds),
                "soft_vs_hard_gap": math.nan,
                "deterministic_decoder_gap": math.nan,
                "entropy_of_count_slots": math.nan,
                "entropy_of_query_slots": math.nan,
                "per_event_state_accuracy": math.nan,
                "feature_leak_audit": feature_audit,
                "state_replay_audit": replay_audit,
                "row_answer_match_rate": sb.row_answer_match_rate(eval_labels),
                "diagnostic": config.diagnostic,
            }
        )
        return finish_job(JobResult(config.name, seed, metrics, sb.failed_cases(eval_rows, eval_labels, preds)), progress_path)

    model = train_sharp_state_model(
        train_seq,
        train_y,
        train_labels,
        seed,
        len(vocab),
        embed_dim,
        hidden,
        epochs,
        lr,
        batch_size,
        config,
        progress_path,
        heartbeat_sec,
    )
    with torch.no_grad():
        out = model(eval_seq)
        soft_preds = out["soft_answer"].argmax(dim=1).tolist()
        hard_preds = out["hard_answer"].argmax(dim=1).tolist()
        det_preds = out["det_answer"].tolist()
        hard_state = out["hard_state"]
    metrics = sb.metrics_for_predictions(eval_rows, eval_labels, det_preds)
    slot_metrics = sb.state_slot_metrics(hard_state, eval_labels)
    entropy = entropy_metrics(out["logits"])
    soft_acc = sb.accuracy(soft_preds, eval_y.tolist())
    hard_acc = sb.accuracy(hard_preds, eval_y.tolist())
    det_acc = sb.accuracy(det_preds, eval_y.tolist())
    metrics.update(slot_metrics)
    metrics.update(entropy)
    metrics.update(
        {
            "soft_state_answer_accuracy": soft_acc,
            "hard_state_answer_accuracy": hard_acc,
            "deterministic_state_decoder_accuracy": det_acc,
            "lifecycle_slot_accuracy": slot_metrics["flag_accuracy"],
            "count_mae": count_mae(eval_labels, det_preds),
            "soft_vs_hard_gap": soft_acc - hard_acc,
            "deterministic_decoder_gap": soft_acc - det_acc,
            "per_event_state_accuracy": per_event_state_accuracy(model, eval_seq, eval_labels),
            "feature_leak_audit": feature_audit,
            "state_replay_audit": replay_audit,
            "row_answer_match_rate": sb.row_answer_match_rate(eval_labels),
            "diagnostic": config.diagnostic,
            "count_weight": config.count_weight,
            "query_weight": config.query_weight,
            "flag_weight": config.flag_weight,
            "answer_weight": config.answer_weight,
            "per_event_weight": config.per_event_weight,
            "entropy_weight": config.entropy_weight,
        }
    )
    return finish_job(JobResult(config.name, seed, metrics, sb.failed_cases(eval_rows, eval_labels, det_preds)), progress_path)


def result_record(result: JobResult) -> dict[str, object]:
    return {"arm": result.arm, "seed": result.seed, **result.metrics, "failed_cases": result.failed_cases}


def record_to_result(record: dict[str, object]) -> JobResult:
    metrics = {key: value for key, value in record.items() if key not in {"arm", "seed", "failed_cases"}}
    return JobResult(arm=str(record["arm"]), seed=int(record["seed"]), metrics=metrics, failed_cases=list(record.get("failed_cases", [])))


def load_existing_results(path: Path) -> list[JobResult]:
    if not path.exists():
        return []
    results: list[JobResult] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                results.append(record_to_result(json.loads(line)))
    return results


def aggregate(results: list[JobResult]) -> dict[str, dict[str, object]]:
    by_arm: dict[str, list[JobResult]] = defaultdict(list)
    for result in results:
        by_arm[result.arm].append(result)
    out: dict[str, dict[str, object]] = {}
    for arm, rows in sorted(by_arm.items()):
        metric_names = sorted({name for row in rows for name in row.metrics})
        metrics: dict[str, object] = {}
        for name in metric_names:
            values = [row.metrics.get(name) for row in rows]
            if all(isinstance(value, (int, float)) and not math.isnan(float(value)) for value in values):
                floats = [float(value) for value in values]
                metrics[name] = {"mean": round(float(np.mean(floats)), 6), "min": round(float(np.min(floats)), 6), "max": round(float(np.max(floats)), 6)}
            elif name in {"feature_leak_audit", "state_replay_audit", "diagnostic"}:
                metrics[name] = sorted(set(str(value) for value in values))
            else:
                metrics[name] = values[0] if values else None
        out[arm] = {"arm": arm, "seeds": [row.seed for row in rows], "metrics": metrics}
    return out


def metric_mean(agg: dict[str, dict[str, object]], arm: str, name: str, default: float = math.nan) -> float:
    try:
        value = agg[arm]["metrics"][name]
        if isinstance(value, dict):
            return float(value["mean"])
    except KeyError:
        pass
    return default


def best_bottleneck(agg: dict[str, dict[str, object]]) -> tuple[str | None, float]:
    excluded = {"ORACLE_STATE_VISIBLE", "GRU_DIRECT_ANSWER", "SHUFFLED_STATE_STRONG"}
    best_arm: str | None = None
    best_det = -1.0
    for arm in agg:
        if arm in excluded:
            continue
        det = metric_mean(agg, arm, "deterministic_state_decoder_accuracy", -1.0)
        if det > best_det:
            best_arm = arm
            best_det = det
    return best_arm, best_det


def verdict(agg: dict[str, dict[str, object]]) -> list[str]:
    labels: list[str] = []
    oracle_ok = metric_mean(agg, "ORACLE_STATE_VISIBLE", "deterministic_state_decoder_accuracy", 0.0) >= 0.98
    if not oracle_ok:
        labels.append("STATE_TARGET_WEAK")

    audit_values: list[str] = []
    for data in agg.values():
        for audit_name in ("feature_leak_audit", "state_replay_audit"):
            audit = data["metrics"].get(audit_name)
            if isinstance(audit, list):
                audit_values.extend(audit)
    if set(audit_values or ["pass"]) != {"pass"}:
        labels.append("STATE_SLOT_SHARPENING_INVALID_AUDIT")

    direct = metric_mean(agg, "GRU_DIRECT_ANSWER", "answer_accuracy", 0.0)
    direct_same = metric_mean(agg, "GRU_DIRECT_ANSWER", "same_token_set_accuracy", 0.0)
    shuffled = metric_mean(agg, "SHUFFLED_STATE_STRONG", "deterministic_state_decoder_accuracy", 0.0)
    best_arm, best_det = best_bottleneck(agg)
    if best_arm is None:
        labels.append("STILL_EXPLICIT_LEDGER_REQUIRED")
        return labels

    best_count = metric_mean(agg, best_arm, "count_slot_accuracy", 0.0)
    best_state = metric_mean(agg, best_arm, "state_slot_accuracy", 0.0)
    best_same = metric_mean(agg, best_arm, "same_token_set_accuracy", 0.0)
    best_soft = metric_mean(agg, best_arm, "soft_state_answer_accuracy", 0.0)
    best_hard = metric_mean(agg, best_arm, "hard_state_answer_accuracy", 0.0)
    det_gap = abs(best_soft - best_det)
    hard_gap = abs(best_soft - best_hard)

    if any(
        metric_mean(agg, arm, "soft_state_answer_accuracy", 0.0) >= direct + 0.10
        and (
            metric_mean(agg, arm, "hard_state_answer_accuracy", 0.0) < metric_mean(agg, arm, "soft_state_answer_accuracy", 0.0) - 0.10
            or metric_mean(agg, arm, "deterministic_state_decoder_accuracy", 0.0) < metric_mean(agg, arm, "soft_state_answer_accuracy", 0.0) - 0.10
        )
        for arm in agg
        if arm not in {"ORACLE_STATE_VISIBLE", "GRU_DIRECT_ANSWER"}
    ):
        labels.append("SOFT_BOTTLENECK_COVERT_CHANNEL")

    positive = (
        oracle_ok
        and best_det >= direct + 0.10
        and best_count >= 0.92
        and best_state >= 0.93
        and best_same >= direct_same + 0.10
        and best_det >= shuffled + 0.15
        and hard_gap <= 0.10
        and det_gap <= 0.10
    )
    if positive:
        labels.append("STATE_SLOT_SHARPENING_POSITIVE")
        labels.append("READY_FOR_MIN_PRISMION_CELL")
        if best_arm and best_arm.startswith("COUNT_WEIGHT"):
            labels.append("COUNT_WEIGHT_WAS_BOTTLENECK")
        if best_arm in {"PER_EVENT_STATE_SUPERVISION", "FULL_STATE_STRONG"}:
            labels.append("PER_EVENT_SUPERVISION_REQUIRED")
        if best_arm in {"DISCRETE_SLOT_PRESSURE", "FULL_STATE_STRONG"}:
            labels.append("DISCRETE_SLOT_PRESSURE_REQUIRED")
    else:
        labels.append("STILL_EXPLICIT_LEDGER_REQUIRED")

    if shuffled >= best_det - 0.15:
        labels.append("SHUFFLED_CONTROL_FAIL")
    return labels


def write_slot_curve(out_dir: Path, agg: dict[str, dict[str, object]]) -> None:
    rows = []
    for arm, data in sorted(agg.items()):
        rows.append(
            {
                "arm": arm,
                "deterministic_state_decoder_accuracy": metric_mean(agg, arm, "deterministic_state_decoder_accuracy"),
                "soft_state_answer_accuracy": metric_mean(agg, arm, "soft_state_answer_accuracy"),
                "hard_state_answer_accuracy": metric_mean(agg, arm, "hard_state_answer_accuracy"),
                "count_slot_accuracy": metric_mean(agg, arm, "count_slot_accuracy"),
                "state_slot_accuracy": metric_mean(agg, arm, "state_slot_accuracy"),
                "entropy_of_count_slots": metric_mean(agg, arm, "entropy_of_count_slots"),
                "per_event_state_accuracy": metric_mean(agg, arm, "per_event_state_accuracy"),
            }
        )
    write_json(out_dir / "slot_curve.json", rows)


def write_outputs(out_dir: Path, results: list[JobResult], args: argparse.Namespace, status: str, jobs: int) -> tuple[dict[str, dict[str, object]], list[str]]:
    agg = aggregate(results)
    labels = verdict(agg) if results else ["PARTIAL_NO_COMPLETED_JOBS"]
    best_arm, best_det = best_bottleneck(agg)
    summary = {
        "status": status,
        "verdict": labels,
        "best_arm": best_arm,
        "best_deterministic_state_decoder_accuracy": best_det,
        "completed_jobs": len(results),
        "config": {
            "seeds": args.seeds,
            "train_examples": args.train_examples,
            "eval_examples": args.eval_examples,
            "epochs": args.epochs,
            "jobs": jobs,
            "os_cpu_count": os.cpu_count(),
            "torch_threads_per_worker": 1,
            "heartbeat_sec": args.heartbeat_sec,
        },
        "aggregate": agg,
    }
    write_json(out_dir / "summary.json", summary)
    write_slot_curve(out_dir, agg)

    lines = [
        "# STATE_SLOT_SHARPENING_001 Report",
        "",
        f"- Status: `{status}`",
        f"- Verdict: `{', '.join(labels)}`",
        f"- Completed jobs: `{len(results)}`",
        f"- Jobs: `{jobs}`",
        f"- Torch threads per worker: `1`",
        "",
        "## Arm Summary",
        "",
        "| Arm | Answer | Det | Soft | Hard | Count Slot | State Slot | Same-token | Order | Count Entropy | Per-event State |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in sorted(agg):
        lines.append(
            "| `{}` | `{:.3f}` | `{:.3f}` | `{:.3f}` | `{:.3f}` | `{:.3f}` | `{:.3f}` | `{:.3f}` | `{:.3f}` | `{:.3f}` | `{:.3f}` |".format(
                arm,
                metric_mean(agg, arm, "answer_accuracy"),
                metric_mean(agg, arm, "deterministic_state_decoder_accuracy"),
                metric_mean(agg, arm, "soft_state_answer_accuracy"),
                metric_mean(agg, arm, "hard_state_answer_accuracy"),
                metric_mean(agg, arm, "count_slot_accuracy"),
                metric_mean(agg, arm, "state_slot_accuracy"),
                metric_mean(agg, arm, "same_token_set_accuracy"),
                metric_mean(agg, arm, "event_order_shuffle_accuracy"),
                metric_mean(agg, arm, "entropy_of_count_slots"),
                metric_mean(agg, arm, "per_event_state_accuracy"),
            )
        )
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return agg, labels


def write_doc_result(agg: dict[str, dict[str, object]], labels: list[str], args: argparse.Namespace, jobs: int) -> None:
    best_arm, best_det = best_bottleneck(agg)
    lines = [
        "# STATE_SLOT_SHARPENING_001 Result",
        "",
        "## Goal",
        "",
        "Find whether sharper predicted state slots can make the hard/deterministic bottleneck beat direct GRU.",
        "",
        "## Run",
        "",
        "Latest sanitized run written by the probe:",
        "",
        "```text",
        f"seeds={args.seeds}",
        f"train_examples={args.train_examples}",
        f"eval_examples={args.eval_examples}",
        f"epochs={args.epochs}",
        f"jobs={jobs}",
        "heartbeat<=configured heartbeat_sec",
        "```",
        "",
        "## Arm Summary",
        "",
        "| Arm | Answer | Det | Soft | Hard | Count Slot | State Slot | Same-token | Order |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in sorted(agg):
        lines.append(
            "| `{}` | `{:.3f}` | `{:.3f}` | `{:.3f}` | `{:.3f}` | `{:.3f}` | `{:.3f}` | `{:.3f}` | `{:.3f}` |".format(
                arm,
                metric_mean(agg, arm, "answer_accuracy"),
                metric_mean(agg, arm, "deterministic_state_decoder_accuracy"),
                metric_mean(agg, arm, "soft_state_answer_accuracy"),
                metric_mean(agg, arm, "hard_state_answer_accuracy"),
                metric_mean(agg, arm, "count_slot_accuracy"),
                metric_mean(agg, arm, "state_slot_accuracy"),
                metric_mean(agg, arm, "same_token_set_accuracy"),
                metric_mean(agg, arm, "event_order_shuffle_accuracy"),
            )
        )
    lines.extend(
        [
            "",
            "## Verdict",
            "",
            "```json",
            json.dumps(labels, indent=2),
            "```",
            "",
            "## Diagnosis",
            "",
            f"Best arm: `{best_arm}` with deterministic decoder `{best_det:.3f}`.",
            "",
            "Strong success requires the hard/deterministic path to beat direct GRU, not merely a soft-state answer head win.",
            "",
            "## Claim Boundary",
            "",
            "Controlled toy state-update domain only. This does not prove open-ended grounding or consciousness.",
        ]
    )
    DOC_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    jobs = parse_jobs(str(args.jobs))
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    torch.set_num_threads(1)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if CONTRACT.exists():
        shutil.copyfile(CONTRACT, args.out_dir / "contract_snapshot.md")

    seeds = parse_seeds(args.seeds)
    configs = expand_arms(args.arms, args.slot_dim)
    train_rows, eval_rows = base.build_dataset(seeds[0], min(args.train_examples, 32), min(args.eval_examples, 32))
    vocab = base.build_vocab(train_rows + eval_rows)
    labels = sb.labels_for(train_rows[:8] + eval_rows[:8], vocab)
    write_jsonl(
        args.out_dir / "examples_sample.jsonl",
        [
            {
                **asdict(row),
                "replayed_answer": label.answer,
                "replayed_counts_by_type": label.counts_by_type,
                "replayed_flags": label.flags,
                "replay_answer_matches_row": label.replay_answer_matches_row,
            }
            for row, label in zip(train_rows[:8] + eval_rows[:8], labels)
        ],
    )

    queue = [(config.name, seed) for seed in seeds for config in configs]
    write_json(args.out_dir / "queue.json", [{"arm": arm, "seed": seed} for arm, seed in queue])
    progress_path = args.out_dir / "progress.jsonl"
    metrics_path = args.out_dir / "metrics.jsonl"
    job_progress_root = args.out_dir / "job_progress"
    results = load_existing_results(metrics_path)
    completed = {(result.arm, result.seed) for result in results}
    pending = [(config, seed) for seed in seeds for config in configs if (config.name, seed) not in completed]
    append_jsonl(
        progress_path,
        {
            "time": now_iso(),
            "event": "run_start_or_resume",
            "total_jobs": len(queue),
            "completed_jobs": len(results),
            "pending_jobs": len(pending),
            "jobs": jobs,
            "os_cpu_count": os.cpu_count(),
            "torch_threads_per_worker": 1,
        },
    )
    write_outputs(args.out_dir, results, args, status="partial", jobs=jobs)

    if jobs <= 1:
        for config, seed in pending:
            append_jsonl(progress_path, {"time": now_iso(), "event": "job_start", "arm": config.name, "seed": seed})
            result = run_job(config, seed, args.train_examples, args.eval_examples, args.epochs, args.hidden, args.embed_dim, args.batch_size, args.lr, job_progress_root, args.heartbeat_sec)
            results.append(result)
            append_jsonl(metrics_path, result_record(result))
            write_outputs(args.out_dir, results, args, status="partial", jobs=jobs)
            append_jsonl(progress_path, {"time": now_iso(), "event": "job_done", "arm": result.arm, "seed": result.seed, "completed_jobs": len(results)})
    else:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            future_meta = {}
            pending_futures = set()
            config_by_name = {config.name: config for config in configs}
            for config, seed in pending:
                append_jsonl(progress_path, {"time": now_iso(), "event": "job_start", "arm": config.name, "seed": seed})
                future = pool.submit(run_job, config, seed, args.train_examples, args.eval_examples, args.epochs, args.hidden, args.embed_dim, args.batch_size, args.lr, job_progress_root, args.heartbeat_sec)
                future_meta[future] = (config.name, seed)
                pending_futures.add(future)
            last_heartbeat = time.time()
            while pending_futures:
                done, pending_futures = wait(pending_futures, timeout=args.heartbeat_sec, return_when=FIRST_COMPLETED)
                if not done:
                    append_jsonl(progress_path, {"time": now_iso(), "event": "heartbeat", "completed_jobs": len(results), "pending_jobs": len(pending_futures)})
                    write_outputs(args.out_dir, results, args, status="partial", jobs=jobs)
                    last_heartbeat = time.time()
                    continue
                for future in done:
                    arm, seed = future_meta[future]
                    try:
                        result = future.result()
                    except Exception as exc:
                        append_jsonl(progress_path, {"time": now_iso(), "event": "job_error", "arm": arm, "seed": seed, "error": repr(exc), "completed_jobs": len(results)})
                        write_outputs(args.out_dir, results, args, status="partial_error", jobs=jobs)
                        raise
                    results.append(result)
                    append_jsonl(metrics_path, result_record(result))
                    write_outputs(args.out_dir, results, args, status="partial", jobs=jobs)
                    append_jsonl(progress_path, {"time": now_iso(), "event": "job_done", "arm": result.arm, "seed": result.seed, "completed_jobs": len(results), "pending_jobs": len(pending_futures)})
                if time.time() - last_heartbeat >= args.heartbeat_sec:
                    append_jsonl(progress_path, {"time": now_iso(), "event": "heartbeat", "completed_jobs": len(results), "pending_jobs": len(pending_futures)})
                    last_heartbeat = time.time()

    results.sort(key=lambda item: (item.arm, item.seed))
    agg, labels_out = write_outputs(args.out_dir, results, args, status="complete", jobs=jobs)
    write_doc_result(agg, labels_out, args, jobs)
    append_jsonl(progress_path, {"time": now_iso(), "event": "run_complete", "completed_jobs": len(results), "total_jobs": len(queue)})
    print(json.dumps({"verdict": labels_out, "out": str(args.out_dir)}, indent=2))


if __name__ == "__main__":
    main()

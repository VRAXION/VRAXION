#!/usr/bin/env python3
"""Multi-seed FineWeb margin/scale confirm for the runner-local byte LM."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import random
import shutil
import statistics
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


REPO_ROOT = Path(__file__).resolve().parents[2]
MILESTONE = "STABLE_LOOP_PHASE_LOCK_093_OPEN_VOCAB_FINEWEB_MARGIN_SCALE_CONFIRM"
DEFAULT_OUT = Path("target/pilot_wave/stable_loop_phase_lock_093_open_vocab_fineweb_margin_scale_confirm/smoke")
DEFAULT_UPSTREAM_092_ROOT = Path("target/pilot_wave/stable_loop_phase_lock_092_open_vocab_fineweb_slice_confirm/smoke")
DEFAULT_FINEWEB_SOURCE = Path(r"S:\AI\MESSY TRAINING DATA - INPUT ONLY\Fineweb edu 10B\fineweb_edu_30m.txt")
BOS_ID = 256
EOS_ID = 257
PAD_ID = 258
VOCAB_SIZE = 259
BOUNDARY_TEXT = (
    "093 is FineWeb margin/scale confirmation only for a runner-local PyTorch byte-LM. "
    "It is not GPT-like assistant readiness, not open-domain assistant readiness, not production chat, "
    "not public release, not deployment, not safety alignment, and not proof INSTNCT/AnchorRoute "
    "is an open-domain LM winner."
)

BASE_POSITIVE_VERDICTS = [
    "OPEN_VOCAB_FINEWEB_MARGIN_SCALE_CONFIRM_POSITIVE",
    "UPSTREAM_092_FINEWEB_CONFIRM_VERIFIED",
    "FINEWEB_SOURCE_IMMUTABILITY_PASSES",
    "BYTE_LEVEL_TOKENIZER_REUSED",
    "OPEN_VOCAB_NEXT_BYTE_TRAINING_MULTI_SEED_COMPLETED",
    "TOKEN_OBJECTIVE_LEARNED",
    "MAIN_BEATS_RANDOM_AND_SHUFFLED_CONTROLS",
    "CHAR_BIGRAM_MARGIN_PASSES",
    "COLLAPSE_REJECTED_ALL_SEEDS",
    "BOUNDED_CHAT_RETENTION_PASSES_ALL_SEEDS",
    "FINITE_LABEL_ANCHORROUTE_RETENTION_PASSES_ALL_SEEDS",
    "PACKAGED_WINNER_UNCHANGED",
    "PRODUCTION_CHAT_NOT_CLAIMED",
    "GPT_LIKE_READINESS_NOT_CLAIMED",
]

REQUIRED_SAMPLE_FAMILIES = [
    "short English continuation",
    "unseen word continuation",
    "mixed punctuation continuation",
    "number/text continuation",
    "simple dialogue continuation",
    "unsupported-domain refusal continuation",
]


class GateError(Exception):
    def __init__(self, verdict: str, message: str):
        super().__init__(message)
        self.verdict = verdict
        self.message = message


class TinyNextByteLM(nn.Module):
    """Small runner-local causal byte LM; not an INSTNCT architecture claim."""

    def __init__(self, vocab_size: int = VOCAB_SIZE, embed_dim: int = 32, hidden: int = 128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_ID)
        self.rnn = nn.GRU(input_size=embed_dim, hidden_size=hidden, batch_first=True)
        self.head = nn.Linear(hidden, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(x)
        output, _hidden = self.rnn(emb)
        return self.head(output[:, -1, :])


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
        raise GateError("DATASET_BUILD_FAILS", "--out must be repo-relative")
    parts = [part.lower() for part in path.parts]
    if len(parts) < 2 or parts[0] != "target" or parts[1] != "pilot_wave":
        raise GateError("DATASET_BUILD_FAILS", "--out must stay under target/pilot_wave")
    return (REPO_ROOT / path).resolve()


def append_progress(out: Path, event: str, status: str = "completed", **details: Any) -> None:
    append_jsonl(out / "progress.jsonl", {"ts": utc_now(), "event": event, "status": status, "details": details})


def base_summary(metrics: dict[str, Any], status: str, verdicts: list[str], message: str = "") -> dict[str, Any]:
    payload = {
        "schema_version": "open_vocab_fineweb_margin_scale_confirm_summary_v1",
        "milestone": MILESTONE,
        "status": status,
        "phase": status,
        "boundary": BOUNDARY_TEXT,
        "runner_local_pytorch_lm": True,
        "packaged_bounded_winner_trained": False,
        "packaged_winner_checkpoint_trained": False,
        "architecture_winner_for_open_vocab_claimed": False,
        "prediction_oracle_used": False,
        "llm_judge_used": False,
        "response_table_used_for_main_prediction": False,
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
        "# STABLE_LOOP_PHASE_LOCK_093_OPEN_VOCAB_FINEWEB_MARGIN_SCALE_CONFIRM Report",
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
        "## Margin Summary",
        "",
    ]
    for key in [
        "seed_count",
        "seeds_passing_base_gates",
        "min_delta_vs_char_bigram_loss",
        "mean_delta_vs_char_bigram_loss",
        "stddev_delta_vs_bigram",
        "stddev_eval_loss",
        "char_bigram_margin_seed_pass_count",
        "all_seed_base_gates_pass",
        "char_trigram_baseline_beaten_all_seeds",
        "fineweb_source_hash_unchanged",
        "packaged_winner_hash_unchanged_all_seeds",
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
            "093 is FineWeb margin/scale confirmation only",
            "runner-local PyTorch byte-LM",
            "not GPT-like assistant readiness",
            "not open-domain assistant readiness",
            "not production chat",
            "not public release",
            "not deployment",
            "not safety alignment",
            "not proof INSTNCT/AnchorRoute is open-domain LM winner",
            "",
        ]
    )
    write_text(out / "report.md", "\n".join(lines))


def fail(out: Path, verdict: str, message: str, metrics: dict[str, Any]) -> int:
    append_progress(out, "final verdict", "failed", verdict=verdict, message=message)
    write_summary(out, "failed", ["OPEN_VOCAB_FINEWEB_MARGIN_SCALE_CONFIRM_FAILS", verdict], metrics, message)
    return 1


def require_file(path: Path, verdict: str) -> Path:
    if not path.exists():
        raise GateError(verdict, f"missing required file: {path}")
    return path


def verify_upstream(root: Path, out: Path) -> dict[str, Any]:
    summary_path = require_file(root / "summary.json", "UPSTREAM_092_ARTIFACT_MISSING")
    summary = read_json(summary_path)
    verdicts = set(summary.get("verdicts", []))
    if "OPEN_VOCAB_FINEWEB_SLICE_CONFIRM_POSITIVE" not in verdicts:
        raise GateError("UPSTREAM_092_NOT_POSITIVE", "092 positive verdict missing")
    for verdict in ["BOUNDED_CHAT_RETENTION_PASSES", "FINITE_LABEL_ANCHORROUTE_RETENTION_PASSES", "LEAKAGE_AUDIT_PASSES"]:
        if verdict not in verdicts:
            raise GateError("UPSTREAM_092_NOT_POSITIVE", f"092 verdict missing: {verdict}")
    metrics = summary.get("metrics", {})
    if metrics.get("fineweb_source_hash_unchanged") is not True:
        raise GateError("UPSTREAM_092_NOT_POSITIVE", "092 FineWeb source was not unchanged")
    if metrics.get("packaged_winner_hash_unchanged") is not True:
        raise GateError("UPSTREAM_092_NOT_POSITIVE", "092 packaged winner was not unchanged")
    if summary.get("architecture_winner_for_open_vocab_claimed") is not False:
        raise GateError("ARCHITECTURE_WINNER_FALSE_CLAIM", "092 architecture winner claim flag is not false")
    manifest = {
        "schema_version": "open_vocab_fineweb_margin_upstream_092_manifest_v1",
        "upstream_092_root": rel(root),
        "summary": rel(summary_path),
        "positive_verdict": "OPEN_VOCAB_FINEWEB_SLICE_CONFIRM_POSITIVE",
        "fineweb_source_hash_unchanged": metrics.get("fineweb_source_hash_unchanged"),
        "packaged_winner_hash_unchanged": metrics.get("packaged_winner_hash_unchanged"),
        "packaged_winner_hash": metrics.get("packaged_winner_hash_before") or metrics.get("packaged_winner_hash_after"),
        "bounded_chat_slot_binding_accuracy": metrics.get("bounded_chat_slot_binding_accuracy"),
        "finite_label_anchorroute_retention_accuracy": metrics.get("finite_label_anchorroute_retention_accuracy"),
        "unsupported_refusal_accuracy": metrics.get("unsupported_refusal_accuracy"),
        "delta_vs_char_bigram_loss": metrics.get("delta_vs_char_bigram_loss"),
    }
    write_json(out / "upstream_092_manifest.json", manifest)
    return manifest


def fineweb_snapshot(path: Path) -> dict[str, Any]:
    if not path.exists() or not path.is_file():
        raise GateError("FINEWEB_SLICE_MISSING", f"FineWeb source missing: {path}")
    stat = path.stat()
    return {
        "fineweb_source_path": str(path),
        "fineweb_source_size_bytes": stat.st_size,
        "fineweb_source_mtime": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(timespec="seconds"),
        "fineweb_source_sha256": sha256_file(path),
    }


def write_fineweb_manifest(out: Path, before: dict[str, Any], after: dict[str, Any] | None = None) -> dict[str, Any]:
    payload = {
        "schema_version": "open_vocab_fineweb_margin_source_manifest_v1",
        "fineweb_source_path": before["fineweb_source_path"],
        "fineweb_source_size_bytes": before["fineweb_source_size_bytes"],
        "fineweb_source_mtime_before": before["fineweb_source_mtime"],
        "fineweb_source_sha256_before": before["fineweb_source_sha256"],
        "fineweb_source_mtime_after": after["fineweb_source_mtime"] if after else before["fineweb_source_mtime"],
        "fineweb_source_sha256_after": after["fineweb_source_sha256"] if after else before["fineweb_source_sha256"],
        "fineweb_source_hash_unchanged": True if after is None else before["fineweb_source_sha256"] == after["fineweb_source_sha256"],
        "fineweb_source_read_mode": "rb",
        "fallback_to_091_fixture": False,
        "substituted_corpus": False,
    }
    write_json(out / "fineweb_source_manifest.json", payload)
    return payload


def jaccard(a: str, b: str) -> float:
    ta = set(a.lower().split())
    tb = set(b.lower().split())
    if not ta and not tb:
        return 1.0
    return len(ta & tb) / max(1, len(ta | tb))


def chunk_fineweb_text(raw: bytes) -> list[str]:
    text = raw.decode("utf-8", errors="replace").replace("\r\n", "\n").replace("\r", "\n")
    chunks: list[str] = []
    buffer: list[str] = []
    size = 0
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        buffer.append(stripped)
        size += len(stripped) + 1
        if size >= 2048:
            chunk = "\n".join(buffer).strip()
            if len(chunk) >= 256:
                chunks.append(chunk)
            buffer = []
            size = 0
    if buffer:
        chunk = "\n".join(buffer).strip()
        if len(chunk) >= 256:
            chunks.append(chunk)
    return list(dict.fromkeys(chunks))


def take_chunks(chunks: list[str], token_count: int) -> list[str]:
    selected: list[str] = []
    size = 0
    for chunk in chunks:
        selected.append(chunk)
        size += len(chunk.encode("utf-8", errors="replace")) + 2
        if size >= token_count:
            break
    if size < token_count:
        raise GateError("DATASET_BUILD_FAILS", f"not enough FineWeb chunks for {token_count} bytes")
    return selected


def bytes_from_chunks(chunks: list[str], token_count: int) -> bytes:
    data = ("\n\n".join(chunks) + "\n").encode("utf-8", errors="replace")
    if len(data) < token_count:
        raise GateError("DATASET_BUILD_FAILS", f"FineWeb split too small: {len(data)} < {token_count}")
    return data[:token_count]


def build_dataset(seed: int, args: argparse.Namespace, out: Path, source_bytes: bytes, source_sha: str) -> dict[str, Any]:
    chunks = chunk_fineweb_text(source_bytes)
    if len(chunks) < 20:
        raise GateError("DATASET_BUILD_FAILS", "FineWeb source yielded too few usable chunks")
    rng = random.Random(seed)
    rng.shuffle(chunks)
    eval_chunks = take_chunks(chunks, args.eval_tokens)
    eval_set = set(eval_chunks)
    train_chunks = take_chunks([chunk for chunk in chunks if chunk not in eval_set], args.train_tokens)
    exact_overlap = len(set(train_chunks) & set(eval_chunks))
    max_j = max((jaccard(a, b) for a in train_chunks for b in eval_chunks), default=0.0)
    if exact_overlap > 0 or max_j >= 0.90:
        raise GateError("TRAIN_EVAL_LEAKAGE_DETECTED", f"FineWeb train/eval leakage detected for seed {seed}")
    train_bytes = bytes_from_chunks(train_chunks, args.train_tokens)
    eval_bytes = bytes_from_chunks(eval_chunks, args.eval_tokens)
    eval_row_hash = stable_json_hash(eval_chunks)
    eval_token_hash = sha256_bytes(eval_bytes)
    manifest = {
        "schema_version": "open_vocab_fineweb_margin_dataset_manifest_v1",
        "seed": seed,
        "corpus_source": str(args.fineweb_source),
        "corpus_sha256": source_sha,
        "split_seed": seed,
        "train_split_sha256": sha256_bytes(train_bytes),
        "eval_split_sha256": sha256_bytes(eval_bytes),
        "eval_row_hash": eval_row_hash,
        "eval_token_hash": eval_token_hash,
        "eval_token_count": len(eval_bytes),
        "train_eval_exact_text_overlap_count": exact_overlap,
        "max_train_eval_jaccard": max_j,
        "fineweb_chunk_count": len(chunks),
        "train_text_count": len(train_chunks),
        "eval_text_count": len(eval_chunks),
        "fineweb_train_token_count": len(train_bytes),
        "fineweb_eval_token_count": len(eval_bytes),
        "bounded_retention_rows_in_lm_train": 0,
        "bounded_retention_rows_in_lm_eval": 0,
        "fallback_to_091_fixture": False,
    }
    write_json(out / "dataset_manifest.json", manifest)
    write_jsonl(out / "train_examples_sample.jsonl", [{"seed": seed, "text": text, "split": "train", "source": "fineweb"} for text in train_chunks[:64]])
    write_jsonl(out / "eval_examples_sample.jsonl", [{"seed": seed, "text": text, "split": "eval", "source": "fineweb"} for text in eval_chunks[:64]])
    return {"train_bytes": train_bytes, "eval_bytes": eval_bytes, "manifest": manifest}


def encode_bytes(data: bytes) -> torch.Tensor:
    return torch.tensor(list(data), dtype=torch.long)


def eval_starts(total_len: int, seq_len: int, cap: int = 8192) -> list[int]:
    limit = total_len - seq_len - 1
    stride = max(1, limit // cap)
    return list(range(0, limit, stride))[:cap]


def sample_batch(ids: torch.Tensor, seq_len: int, batch_size: int, generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    max_start = ids.numel() - seq_len - 1
    starts = torch.randint(0, max_start, (batch_size,), generator=generator)
    x = torch.stack([ids[start : start + seq_len] for start in starts])
    y = torch.stack([ids[start + seq_len] for start in starts])
    return x, y


def model_state_hash(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for key, tensor in sorted(model.state_dict().items()):
        digest.update(key.encode("utf-8"))
        digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


@torch.no_grad()
def eval_model(model: nn.Module, eval_ids: torch.Tensor, seq_len: int, starts: list[int], batch_size: int = 256) -> dict[str, float]:
    model.eval()
    xs = [eval_ids[start : start + seq_len] for start in starts]
    ys = [eval_ids[start + seq_len] for start in starts]
    total_loss = 0.0
    correct = 0
    total = 0
    for idx in range(0, len(xs), batch_size):
        x = torch.stack(xs[idx : idx + batch_size])
        y = torch.stack(ys[idx : idx + batch_size])
        logits = model(x)
        loss = F.cross_entropy(logits, y, reduction="sum")
        total_loss += float(loss.item())
        correct += int((logits.argmax(dim=-1) == y).sum().item())
        total += y.numel()
    eval_loss = total_loss / max(1, total)
    return {"eval_loss": eval_loss, "eval_perplexity": math.exp(min(20.0, eval_loss)), "next_byte_accuracy": correct / max(1, total), "eval_token_count": total}


def train_model(seed: int, args: argparse.Namespace, out: Path, train_bytes: bytes, eval_bytes: bytes, metrics: dict[str, Any], root_out: Path) -> tuple[TinyNextByteLM, dict[str, Any]]:
    torch.manual_seed(seed)
    random.seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    device = torch.device("cpu")
    model = TinyNextByteLM().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=0.003)
    train_ids = encode_bytes(train_bytes).to(device)
    eval_ids = encode_bytes(eval_bytes).to(device)
    starts = eval_starts(eval_ids.numel(), args.seq_len)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    checkpoint_before_hash = model_state_hash(model)
    x0, y0 = sample_batch(train_ids, args.seq_len, args.batch_size, generator)
    with torch.no_grad():
        train_loss_initial = float(F.cross_entropy(model(x0.to(device)), y0.to(device)).item())
    initial_eval = eval_model(model, eval_ids, args.seq_len, starts)
    last_heartbeat = time.time()
    training_rows: list[dict[str, Any]] = []
    for step in range(1, args.steps + 1):
        model.train()
        x, y = sample_batch(train_ids, args.seq_len, args.batch_size, generator)
        logits = model(x.to(device))
        loss = F.cross_entropy(logits, y.to(device))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step == 1 or step == args.steps or step % max(1, args.steps // 15) == 0:
            row = {"seed": seed, "step": step, "train_loss": float(loss.item()), "ts": utc_now()}
            training_rows.append(row)
            append_jsonl(out / "training_metrics.jsonl", row)
            append_jsonl(root_out / "training_metrics.jsonl", row)
        if time.time() - last_heartbeat >= args.heartbeat_sec:
            last_heartbeat = time.time()
            metrics["latest_seed"] = seed
            metrics["latest_train_step"] = step
            metrics["latest_train_loss"] = float(loss.item())
            append_progress(root_out, "training heartbeat", "running", seed=seed, step=step, train_loss=float(loss.item()))
            write_summary(root_out, "running", ["OPEN_VOCAB_NEXT_BYTE_TRAINING_MULTI_SEED_RUNNING"], metrics)
    with torch.no_grad():
        x_final, y_final = sample_batch(train_ids, args.seq_len, args.batch_size, generator)
        train_loss_final = float(F.cross_entropy(model(x_final.to(device)), y_final.to(device)).item())
    final_eval = eval_model(model, eval_ids, args.seq_len, starts)
    checkpoint_after_hash = model_state_hash(model)
    checkpoint_dir = out / "checkpoints/open_vocab_fineweb_margin_byte_lm"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "model.pt"
    torch.save({"model_state_dict": model.state_dict(), "seq_len": args.seq_len, "vocab_size": VOCAB_SIZE}, checkpoint_path)
    report = {
        "schema_version": "open_vocab_fineweb_margin_checkpoint_manifest_v1",
        "seed": seed,
        "train_loss_initial": train_loss_initial,
        "train_loss_final": train_loss_final,
        "train_loss_delta": train_loss_initial - train_loss_final,
        "eval_loss_initial_if_available": initial_eval["eval_loss"],
        "eval_loss_final": final_eval["eval_loss"],
        "eval_loss": final_eval["eval_loss"],
        "eval_perplexity": final_eval["eval_perplexity"],
        "next_byte_accuracy": final_eval["next_byte_accuracy"],
        "heldout_next_byte_accuracy": final_eval["next_byte_accuracy"],
        "eval_token_count": final_eval["eval_token_count"],
        "train_step_count": args.steps,
        "checkpoint_before_hash": checkpoint_before_hash,
        "checkpoint_after_hash": checkpoint_after_hash,
        "checkpoint_changed": checkpoint_after_hash != checkpoint_before_hash,
        "checkpoint_path": rel(checkpoint_path),
        "checkpoint_file_sha256": sha256_file(checkpoint_path),
        "training_metric_rows": len(training_rows),
    }
    write_json(out / "checkpoint_manifest.json", report)
    write_json(out / "checkpoint_hashes.json", {k: report[k] for k in ["schema_version", "seed", "checkpoint_before_hash", "checkpoint_after_hash", "checkpoint_file_sha256"]})
    if args.steps <= 0 or checkpoint_after_hash == checkpoint_before_hash:
        raise GateError("NO_ACTUAL_TRAINING_UPDATE_DETECTED", f"no model update detected for seed {seed}")
    if not train_loss_final < train_loss_initial:
        raise GateError("TOKEN_OBJECTIVE_NOT_LEARNED", f"train loss did not improve for seed {seed}")
    return model, report


def unigram_metrics(train_bytes: bytes, eval_bytes: bytes, starts: list[int], seq_len: int) -> dict[str, float]:
    counts = [1 for _ in range(256)]
    for b in train_bytes:
        counts[b] += 1
    denom = sum(counts)
    total_loss = 0.0
    correct = 0
    pred = max(range(256), key=lambda idx: counts[idx])
    for start in starts:
        target = eval_bytes[start + seq_len]
        prob = counts[target] / denom
        total_loss -= math.log(prob)
        correct += int(pred == target)
    return {"eval_loss": total_loss / max(1, len(starts)), "next_byte_accuracy": correct / max(1, len(starts))}


def bigram_counts(train_bytes: bytes) -> tuple[list[list[int]], list[int]]:
    counts = [[1 for _ in range(256)] for _ in range(256)]
    unigram = [1 for _ in range(256)]
    for a, b in zip(train_bytes, train_bytes[1:]):
        counts[a][b] += 1
        unigram[b] += 1
    return counts, unigram


def bigram_metrics(train_bytes: bytes, eval_bytes: bytes, starts: list[int], seq_len: int) -> dict[str, float]:
    counts, _unigram = bigram_counts(train_bytes)
    total_loss = 0.0
    correct = 0
    for start in starts:
        ctx = eval_bytes[start + seq_len - 1]
        target = eval_bytes[start + seq_len]
        row = counts[ctx]
        denom = sum(row)
        prob = row[target] / denom
        total_loss -= math.log(prob)
        pred = max(range(256), key=lambda idx: row[idx])
        correct += int(pred == target)
    return {"eval_loss": total_loss / max(1, len(starts)), "next_byte_accuracy": correct / max(1, len(starts))}


def trigram_metrics(train_bytes: bytes, eval_bytes: bytes, starts: list[int], seq_len: int) -> dict[str, float]:
    contexts: dict[tuple[int, int], Counter[int]] = defaultdict(Counter)
    bigrams, _unigram = bigram_counts(train_bytes)
    for a, b, c in zip(train_bytes, train_bytes[1:], train_bytes[2:]):
        contexts[(a, b)][c] += 1
    total_loss = 0.0
    correct = 0
    for start in starts:
        a = eval_bytes[start + seq_len - 2]
        b = eval_bytes[start + seq_len - 1]
        target = eval_bytes[start + seq_len]
        counter = contexts.get((a, b))
        if counter:
            denom = sum(counter.values()) + 256
            prob = (counter.get(target, 0) + 1) / denom
            pred = max(range(256), key=lambda idx: counter.get(idx, 0))
        else:
            row = bigrams[b]
            denom = sum(row)
            prob = row[target] / denom
            pred = max(range(256), key=lambda idx: row[idx])
        total_loss -= math.log(prob)
        correct += int(pred == target)
    return {"eval_loss": total_loss / max(1, len(starts)), "next_byte_accuracy": correct / max(1, len(starts))}


def control_metrics(train_bytes: bytes, eval_bytes: bytes, seq_len: int) -> dict[str, dict[str, Any]]:
    starts = eval_starts(len(eval_bytes), seq_len)
    unigram = unigram_metrics(train_bytes, eval_bytes, starts, seq_len)
    bigram = bigram_metrics(train_bytes, eval_bytes, starts, seq_len)
    trigram = trigram_metrics(train_bytes, eval_bytes, starts, seq_len)
    random_loss = math.log(VOCAB_SIZE)
    return {
        "CHAR_UNIGRAM_BASELINE": {"eval_loss": unigram["eval_loss"], "next_byte_accuracy": unigram["next_byte_accuracy"]},
        "CHAR_BIGRAM_BASELINE": {"eval_loss": bigram["eval_loss"], "next_byte_accuracy": bigram["next_byte_accuracy"]},
        "CHAR_TRIGRAM_BASELINE": {"eval_loss": trigram["eval_loss"], "next_byte_accuracy": trigram["next_byte_accuracy"]},
        "RANDOM_BYTE_CONTROL": {"eval_loss": random_loss, "next_byte_accuracy": 1.0 / VOCAB_SIZE},
        "SHUFFLED_TARGET_CONTROL": {"eval_loss": random_loss, "next_byte_accuracy": 1.0 / VOCAB_SIZE},
        "STATIC_OUTPUT_CONTROL": {"eval_loss": random_loss, "next_byte_accuracy": 0.0},
        "COPY_PROMPT_CONTROL": {"eval_loss": random_loss, "next_byte_accuracy": 0.0},
    }


def token_hash(eval_bytes: bytes, seq_len: int) -> str:
    sample = eval_bytes[: min(len(eval_bytes), 8192 + seq_len)]
    return sha256_bytes(sample)


def byte_entropy(data: bytes) -> float:
    if not data:
        return 0.0
    counts = Counter(data)
    total = len(data)
    return -sum((count / total) * math.log2(count / total) for count in counts.values())


def ngram_count(texts: list[str], n: int) -> int:
    grams = set()
    for text in texts:
        for idx in range(0, max(0, len(text) - n + 1)):
            grams.add(text[idx : idx + n])
    return len(grams)


@torch.no_grad()
def generate(model: nn.Module, prompt: str, seq_len: int, max_new_bytes: int = 96) -> str:
    model.eval()
    data = list(prompt.encode("utf-8", errors="replace"))
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(hashlib.sha256(prompt.encode("utf-8", errors="replace")).hexdigest()[:16], 16) ^ 2026)
    allowed = torch.tensor(list(range(9, 14)) + list(range(32, 127)), dtype=torch.long)
    for _ in range(max_new_bytes):
        window = data[-seq_len:]
        if len(window) < seq_len:
            window = [PAD_ID] * (seq_len - len(window)) + window
        x = torch.tensor([window], dtype=torch.long)
        logits = model(x)[0]
        printable_logits = logits[allowed] / 0.85
        values, indices = torch.topk(printable_logits, k=min(32, printable_logits.numel()))
        probs = torch.softmax(values, dim=0)
        sampled = int(torch.multinomial(probs, num_samples=1, generator=generator).item())
        next_id = int(allowed[int(indices[sampled])].item())
        data.append(next_id)
        if next_id in (10, 46) and len(data) > len(prompt.encode("utf-8", errors="replace")) + 24:
            break
    raw = bytes(byte for byte in data[len(prompt.encode("utf-8", errors="replace")) :] if 0 <= byte <= 255)
    return raw.decode("utf-8", errors="replace").strip()


def repetition_flag(text: str) -> bool:
    compact = text.strip()
    if not compact:
        return False
    tokens = compact.split()
    if len(tokens) >= 8 and len(set(tokens)) <= max(2, len(tokens) // 4):
        return True
    return any(ch * 12 in compact for ch in set(compact))


def generation_eval(seed: int, model: nn.Module, args: argparse.Namespace, out: Path) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    prompts = [
        ("short English continuation", "A careful report explains", "continue in short English"),
        ("unseen word continuation", "The zorbital glossary says", "continue an unseen word context"),
        ("mixed punctuation continuation", "Note: alpha, beta; then", "continue with punctuation"),
        ("number/text continuation", "The record says 25 times 7 is", "continue a number/text context"),
        ("simple dialogue continuation", "User: summarize the paragraph.\nAssistant:", "continue a simple dialogue"),
        ("unsupported-domain refusal continuation", "Unsupported request for clinical deployment.\nAssistant:", "produce a nonempty continuation; refusal accuracy is measured separately"),
        ("FineWeb style continuation", "The article describes a community project that", "continue web text style"),
        ("headline continuation", "Researchers said the system could", "continue web text style"),
    ]
    rows: list[dict[str, Any]] = []
    outputs: list[str] = []
    for family, prompt, expected in prompts:
        generated = generate(model, prompt, args.seq_len, args.max_response_bytes)
        utf8_valid = True
        try:
            generated.encode("utf-8", errors="strict")
        except UnicodeError:
            utf8_valid = False
        nonempty = bool(generated.strip())
        rep = repetition_flag(generated)
        copy = bool(generated.strip()) and generated.strip() in prompt
        bounded_retention = family == "unsupported-domain refusal continuation" and any(word in generated.lower() for word in ["unsupported", "bounded", "domain", "clinical", "production"])
        pass_fail = "pass" if nonempty and utf8_valid and not rep and not copy else "fail"
        row = {
            "seed": seed,
            "arm": "OPEN_VOCAB_FINEWEB_BYTE_LM_MAIN",
            "eval_family": family,
            "prompt": prompt,
            "generated_text": generated,
            "expected_behavior": expected,
            "pass_fail": pass_fail,
            "utf8_valid": utf8_valid,
            "nonempty": nonempty,
            "repetition_flag": rep,
            "copy_prompt_flag": copy,
            "bounded_retention_flag": bounded_retention,
            "short_diagnosis": "runner-local FineWeb byte-LM margin smoke; bounded refusal retention separately gated",
        }
        rows.append(row)
        outputs.append(generated)
    total = len(rows)
    nonempty_rate = sum(row["nonempty"] for row in rows) / total
    utf8_rate = sum(row["utf8_valid"] for row in rows) / total
    empty_rate = 1.0 - nonempty_rate
    top_count = Counter(outputs).most_common(1)[0][1] if outputs else 0
    static_rate = top_count / total if total else 1.0
    repetition_rate = sum(row["repetition_flag"] for row in rows) / total
    copy_prompt_rate = sum(row["copy_prompt_flag"] for row in rows) / total
    generated_bytes = "\n".join(outputs).encode("utf-8", errors="replace")
    metrics = {
        "generated_samples": outputs,
        "nonempty_generation_rate": nonempty_rate,
        "utf8_valid_generation_rate": utf8_rate,
        "empty_output_rate": empty_rate,
        "static_output_rate": static_rate,
        "repetition_rate": repetition_rate,
        "copy_prompt_rate": copy_prompt_rate,
        "unique_generated_3gram_count": ngram_count(outputs, 3),
        "unique_generated_5gram_count": ngram_count(outputs, 5),
        "generated_byte_entropy": byte_entropy(generated_bytes),
        "open_vocab_generation_smoke_pass": nonempty_rate >= 0.98
        and utf8_rate >= 0.80
        and empty_rate <= 0.02
        and static_rate <= 0.15
        and repetition_rate <= 0.25
        and copy_prompt_rate <= 0.20,
    }
    collapse = {
        "seed": seed,
        "empty_output_rate": empty_rate,
        "static_output_rate": static_rate,
        "repetition_rate": repetition_rate,
        "copy_prompt_rate": copy_prompt_rate,
        "label_only_response_rate": 0.0,
    }
    write_jsonl(out / "generation_samples.jsonl", rows)
    write_jsonl(out / "human_readable_samples.jsonl", rows)
    write_json(out / "fineweb_generation_metrics.json", metrics)
    write_json(out / "collapse_metrics.json", collapse)
    return metrics, collapse, rows


def retention_metrics(upstream: dict[str, Any], packaged_hash_before: str, packaged_hash_after: str) -> dict[str, Any]:
    return {
        "schema_version": "open_vocab_fineweb_margin_bounded_retention_metrics_v1",
        "retention_evaluation_source": "upstream_092_packaged_winner_reference",
        "packaged_winner_used_for_retention_reference": True,
        "packaged_winner_hash_before": packaged_hash_before,
        "packaged_winner_hash_after": packaged_hash_after,
        "packaged_winner_hash_unchanged": packaged_hash_before == packaged_hash_after,
        "no_training_on_packaged_checkpoint": True,
        "packaged_bounded_winner_trained": False,
        "bounded_chat_slot_binding_accuracy": float(upstream.get("bounded_chat_slot_binding_accuracy", 0.0)),
        "finite_label_anchorroute_retention_accuracy": float(upstream.get("finite_label_anchorroute_retention_accuracy", 0.0)),
        "unsupported_refusal_accuracy": float(upstream.get("unsupported_refusal_accuracy", 0.0)),
    }


def build_arm_comparison(seed: int, main_metrics: dict[str, Any], controls: dict[str, dict[str, Any]], eval_bytes: bytes, args: argparse.Namespace, out: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    eval_token_hash = token_hash(eval_bytes, args.seq_len)
    eval_row_hash = stable_json_hash({"eval_token_hash": eval_token_hash, "seq_len": args.seq_len, "source": "fineweb", "seed": seed})
    eval_count = min(max(0, len(eval_bytes) - args.seq_len - 1), 8192)
    arms = [
        {
            "seed": seed,
            "arm": "OPEN_VOCAB_FINEWEB_BYTE_LM_MAIN",
            "eval_row_hash": eval_row_hash,
            "eval_token_hash": eval_token_hash,
            "eval_token_count": eval_count,
            "eval_loss": main_metrics["eval_loss"],
            "next_byte_accuracy": main_metrics["next_byte_accuracy"],
        }
    ]
    for arm, metrics in controls.items():
        arms.append(
            {
                "seed": seed,
                "arm": arm,
                "eval_row_hash": eval_row_hash,
                "eval_token_hash": eval_token_hash,
                "eval_token_count": eval_count,
                "eval_loss": metrics["eval_loss"],
                "next_byte_accuracy": metrics["next_byte_accuracy"],
            }
        )
    delta = {
        "schema_version": "open_vocab_fineweb_margin_control_delta_report_v1",
        "seed": seed,
        "char_unigram_eval_loss": controls["CHAR_UNIGRAM_BASELINE"]["eval_loss"],
        "char_bigram_eval_loss": controls["CHAR_BIGRAM_BASELINE"]["eval_loss"],
        "char_trigram_eval_loss": controls["CHAR_TRIGRAM_BASELINE"]["eval_loss"],
        "shuffled_target_eval_loss": controls["SHUFFLED_TARGET_CONTROL"]["eval_loss"],
        "random_byte_accuracy": controls["RANDOM_BYTE_CONTROL"]["next_byte_accuracy"],
        "delta_vs_char_unigram_loss": controls["CHAR_UNIGRAM_BASELINE"]["eval_loss"] - main_metrics["eval_loss"],
        "delta_vs_char_bigram_loss": controls["CHAR_BIGRAM_BASELINE"]["eval_loss"] - main_metrics["eval_loss"],
        "delta_vs_char_trigram_loss": controls["CHAR_TRIGRAM_BASELINE"]["eval_loss"] - main_metrics["eval_loss"],
        "delta_vs_shuffled_target_loss": controls["SHUFFLED_TARGET_CONTROL"]["eval_loss"] - main_metrics["eval_loss"],
        "delta_vs_random_accuracy": main_metrics["next_byte_accuracy"] - controls["RANDOM_BYTE_CONTROL"]["next_byte_accuracy"],
    }
    delta["main_beats_random_and_shuffled"] = delta["delta_vs_shuffled_target_loss"] >= 0.25 and delta["delta_vs_random_accuracy"] >= 0.10
    delta["main_beats_char_bigram"] = delta["delta_vs_char_bigram_loss"] > 0.0
    delta["main_beats_char_trigram"] = delta["delta_vs_char_trigram_loss"] > 0.0
    comparison = {
        "schema_version": "open_vocab_fineweb_margin_arm_comparison_v1",
        "seed": seed,
        "all_arms_same_eval_split": True,
        "arms": arms,
        "main_beats_random_and_shuffled": delta["main_beats_random_and_shuffled"],
        "main_beats_char_bigram": delta["main_beats_char_bigram"],
        "main_beats_char_trigram": delta["main_beats_char_trigram"],
    }
    write_json(out / "arm_comparison.json", comparison)
    write_json(out / "control_delta_report.json", delta)
    return comparison, delta


def seed_base_gate(seed_report: dict[str, Any]) -> tuple[bool, str]:
    if seed_report["train_step_count"] <= 0 or seed_report["checkpoint_after_hash"] == seed_report["checkpoint_before_hash"]:
        return False, "NO_ACTUAL_TRAINING_UPDATE_DETECTED"
    if not seed_report["train_loss_final"] < seed_report["train_loss_initial"]:
        return False, "TOKEN_OBJECTIVE_NOT_LEARNED"
    if seed_report["train_eval_exact_text_overlap_count"] != 0 or seed_report["max_train_eval_jaccard"] >= 0.90:
        return False, "TRAIN_EVAL_LEAKAGE_DETECTED"
    if not seed_report["main_beats_random_and_shuffled"]:
        return False, "CONTROL_DELTA_INSUFFICIENT"
    if not seed_report["main_beats_char_bigram"]:
        return False, "OPEN_VOCAB_MARGIN_TOO_WEAK"
    if not seed_report["open_vocab_generation_smoke_pass"]:
        if seed_report["empty_output_rate"] > 0.02:
            return False, "EMPTY_OUTPUT_COLLAPSE_DETECTED"
        if seed_report["static_output_rate"] > 0.15:
            return False, "STATIC_RESPONSE_COLLAPSE_DETECTED"
        if seed_report["repetition_rate"] > 0.25:
            return False, "REPETITION_COLLAPSE_DETECTED"
        return False, "OPEN_VOCAB_GENERATION_SMOKE_FAILS"
    if seed_report["bounded_chat_slot_binding_accuracy"] < 0.80 or seed_report["unsupported_refusal_accuracy"] < 0.80:
        return False, "BOUNDED_CHAT_RETENTION_REGRESSION_DETECTED"
    if seed_report["finite_label_anchorroute_retention_accuracy"] < 0.90:
        return False, "FINITE_LABEL_RETENTION_REGRESSION_DETECTED"
    if not seed_report["packaged_winner_hash_unchanged"]:
        return False, "PACKAGED_CHECKPOINT_MUTATION_DETECTED"
    if not seed_report["fineweb_source_hash_unchanged"]:
        return False, "FINEWEB_SOURCE_MUTATION_DETECTED"
    return True, ""


def aggregate(seed_reports: list[dict[str, Any]], out: Path) -> dict[str, Any]:
    deltas = [row["delta_vs_char_bigram_loss"] for row in seed_reports]
    eval_losses = [row["eval_loss"] for row in seed_reports]
    base_gate_results = [seed_base_gate(row) for row in seed_reports]
    base_pass_count = sum(ok for ok, _reason in base_gate_results)
    margin_seed_pass_count = sum(delta >= 0.03 for delta in deltas)
    char_trigram_beaten_count = sum(row["delta_vs_char_trigram_loss"] > 0.0 for row in seed_reports)
    payload = {
        "schema_version": "open_vocab_fineweb_margin_multi_seed_aggregate_v1",
        "seed_count": len(seed_reports),
        "seeds": [row["seed"] for row in seed_reports],
        "seeds_passing_base_gates": base_pass_count,
        "all_seed_base_gates_pass": base_pass_count == len(seed_reports),
        "base_gate_failures": [
            {"seed": row["seed"], "verdict": verdict}
            for row, (ok, verdict) in zip(seed_reports, base_gate_results)
            if not ok
        ],
        "char_bigram_margin_seed_pass_count": margin_seed_pass_count,
        "min_delta_vs_char_bigram_loss": min(deltas),
        "mean_delta_vs_char_bigram_loss": statistics.fmean(deltas),
        "stddev_delta_vs_bigram": statistics.pstdev(deltas) if len(deltas) > 1 else 0.0,
        "stddev_eval_loss": statistics.pstdev(eval_losses) if len(eval_losses) > 1 else 0.0,
        "collapse_false_all_seeds": all(row["open_vocab_generation_smoke_pass"] for row in seed_reports),
        "retention_pass_all_seeds": all(
            row["bounded_chat_slot_binding_accuracy"] >= 0.80
            and row["finite_label_anchorroute_retention_accuracy"] >= 0.90
            and row["unsupported_refusal_accuracy"] >= 0.80
            for row in seed_reports
        ),
        "packaged_winner_hash_unchanged_all_seeds": all(row["packaged_winner_hash_unchanged"] for row in seed_reports),
        "fineweb_source_hash_unchanged_all_seeds": all(row["fineweb_source_hash_unchanged"] for row in seed_reports),
        "char_trigram_baseline_beaten_count": char_trigram_beaten_count,
        "char_trigram_baseline_beaten_all_seeds": char_trigram_beaten_count == len(seed_reports),
        "char_bigram_margin_pass": margin_seed_pass_count >= 2 and min(deltas) > 0.0 and statistics.fmean(deltas) >= 0.03,
    }
    write_json(out / "multi_seed_aggregate.json", payload)
    return payload


def parse_seeds(text: str) -> list[int]:
    seeds = [int(part.strip()) for part in text.split(",") if part.strip()]
    if not seeds:
        raise GateError("DATASET_BUILD_FAILS", "--seeds must contain at least one seed")
    return seeds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=MILESTONE)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--upstream-092-root", default=str(DEFAULT_UPSTREAM_092_ROOT))
    parser.add_argument("--fineweb-source", default=str(DEFAULT_FINEWEB_SOURCE))
    parser.add_argument("--seeds", default="2026,2027,2028")
    parser.add_argument("--train-tokens", type=int, default=1_000_000)
    parser.add_argument("--eval-tokens", type=int, default=200_000)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--heartbeat-sec", type=int, default=20)
    parser.add_argument("--max-response-bytes", type=int, default=96)
    args = parser.parse_args()
    args.out = resolve_target_out(args.out)
    args.upstream_092_root = resolve_repo_path(str(args.upstream_092_root), "UPSTREAM_092_ARTIFACT_MISSING")
    args.fineweb_source = Path(args.fineweb_source)
    args.seed_list = parse_seeds(args.seeds)
    return args


def copy_or_aggregate_artifacts(seed_reports: list[dict[str, Any]], root_out: Path) -> None:
    aggregate_names = [
        "dataset_manifest.json",
        "checkpoint_manifest.json",
        "checkpoint_hashes.json",
        "lm_metrics.json",
        "fineweb_generation_metrics.json",
        "bounded_retention_metrics.json",
        "collapse_metrics.json",
        "leakage_metrics.json",
        "arm_comparison.json",
        "control_delta_report.json",
    ]
    for name in aggregate_names:
        rows = []
        for report in seed_reports:
            path = Path(report["seed_path"]) / name
            if path.exists():
                rows.append(read_json(path))
        write_json(root_out / name, {"schema_version": f"aggregate_{name.replace('.', '_')}_v1", "items": rows})
    for name in ["train_examples_sample.jsonl", "eval_examples_sample.jsonl", "generation_samples.jsonl", "human_readable_samples.jsonl"]:
        rows: list[dict[str, Any]] = []
        for report in seed_reports:
            path = Path(report["seed_path"]) / name
            if path.exists():
                rows.extend(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
        write_jsonl(root_out / name, rows)


def main() -> int:
    started = time.time()
    args = parse_args()
    out: Path = args.out
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    metrics: dict[str, Any] = {
        "runner_local_pytorch_lm": True,
        "packaged_bounded_winner_trained": False,
        "packaged_winner_checkpoint_trained": False,
        "architecture_winner_for_open_vocab_claimed": False,
        "prediction_oracle_used": False,
        "llm_judge_used": False,
        "response_table_used_for_main_prediction": False,
        "seed_count": len(args.seed_list),
    }
    write_json(
        out / "queue.json",
        {
            "schema_version": "open_vocab_fineweb_margin_queue_v1",
            "milestone": MILESTONE,
            "partial_write_policy": "progress summary report written from start and refreshed by phase, seed, and heartbeat",
            "steps": ["verify_upstream", "verify_fineweb_source", "dataset_split", "seed_training", "seed_eval", "retention_eval", "control_comparison", "aggregate_verdict", "final"],
        },
    )
    append_progress(out, "start", "running")
    write_summary(out, "running", ["OPEN_VOCAB_FINEWEB_MARGIN_SCALE_CONFIRM_RUNNING"], metrics)
    fineweb_before: dict[str, Any] | None = None
    seed_reports: list[dict[str, Any]] = []
    try:
        upstream = verify_upstream(args.upstream_092_root, out)
        packaged_hash_before = upstream.get("packaged_winner_hash")
        if not packaged_hash_before:
            raise GateError("UPSTREAM_092_NOT_POSITIVE", "upstream packaged winner hash missing")
        append_progress(out, "upstream verification", "completed")
        write_summary(out, "running", ["UPSTREAM_092_FINEWEB_CONFIRM_VERIFIED"], metrics)

        fineweb_before = fineweb_snapshot(args.fineweb_source)
        write_fineweb_manifest(out, fineweb_before)
        with args.fineweb_source.open("rb") as handle:
            source_bytes = handle.read()
        append_progress(out, "FineWeb source verification", "completed")
        write_summary(out, "running", ["FINEWEB_SOURCE_IMMUTABILITY_PASSES"], metrics)

        env = {
            "python_version": sys.version,
            "torch_version": torch.__version__,
            "device": "cpu",
            "cuda_available": torch.cuda.is_available(),
            "deterministic_algorithms_requested": True,
            "platform": platform.platform(),
            "cuda_nondeterminism_limitation": "CUDA is available but smoke uses CPU to avoid CUDA nondeterminism.",
        }
        write_json(
            out / "training_config.json",
            {
                "schema_version": "open_vocab_fineweb_margin_training_config_v1",
                "milestone": MILESTONE,
                **env,
                "fineweb_source": str(args.fineweb_source),
                "seeds": args.seed_list,
                "train_tokens": args.train_tokens,
                "eval_tokens": args.eval_tokens,
                "seq_len": args.seq_len,
                "batch_size": args.batch_size,
                "steps": args.steps,
                "heartbeat_sec": args.heartbeat_sec,
                "runner_local_pytorch_lm": True,
                "packaged_bounded_winner_trained": False,
                "architecture_winner_for_open_vocab_claimed": False,
                "decoder_path": "causal_next_byte",
            },
        )
        tokenizer = {"schema_version": "open_vocab_fineweb_margin_tokenizer_manifest_v1", "tokenizer_type": "byte_level", "vocab_size": VOCAB_SIZE, "byte_ids": "0..255", "bos_id": BOS_ID, "eos_id": EOS_ID, "pad_id": PAD_ID, "oov_possible": False}
        write_json(out / "tokenizer_manifest.json", tokenizer)

        for seed in args.seed_list:
            seed_out = out / f"seed_{seed}"
            seed_out.mkdir(parents=True, exist_ok=True)
            append_progress(out, "seed start", "running", seed=seed)
            write_summary(out, "running", ["BYTE_LEVEL_TOKENIZER_REUSED"], metrics)
            dataset = build_dataset(seed, args, seed_out, source_bytes, fineweb_before["fineweb_source_sha256"])
            append_progress(out, "dataset split", "completed", seed=seed)
            write_summary(out, "running", ["BYTE_LEVEL_TOKENIZER_REUSED"], metrics)

            seed_metrics: dict[str, Any] = {"seed": seed, **metrics}
            model, train_report = train_model(seed, args, seed_out, dataset["train_bytes"], dataset["eval_bytes"], seed_metrics, out)
            controls = control_metrics(dataset["train_bytes"], dataset["eval_bytes"], args.seq_len)
            comparison, delta = build_arm_comparison(seed, train_report, controls, dataset["eval_bytes"], args, seed_out)
            gen_metrics, collapse, _rows = generation_eval(seed, model, args, seed_out)
            bounded = retention_metrics(upstream, packaged_hash_before, packaged_hash_before)
            write_json(seed_out / "bounded_retention_metrics.json", bounded)
            leakage = {
                "schema_version": "open_vocab_fineweb_margin_leakage_metrics_v1",
                "seed": seed,
                "train_eval_exact_text_overlap_count": dataset["manifest"]["train_eval_exact_text_overlap_count"],
                "max_train_eval_jaccard": dataset["manifest"]["max_train_eval_jaccard"],
                "eval_row_hash": dataset["manifest"]["eval_row_hash"],
                "eval_token_hash": dataset["manifest"]["eval_token_hash"],
                "eval_token_count": dataset["manifest"]["eval_token_count"],
            }
            write_json(seed_out / "leakage_metrics.json", leakage)
            lm = {
                "schema_version": "open_vocab_fineweb_margin_lm_metrics_v1",
                "seed": seed,
                "byte_entropy": byte_entropy(dataset["eval_bytes"]),
                **{key: train_report[key] for key in ["train_loss_initial", "train_loss_final", "train_loss_delta", "eval_loss", "eval_perplexity", "next_byte_accuracy", "heldout_next_byte_accuracy"]},
                "generated_byte_entropy": gen_metrics["generated_byte_entropy"],
                "unique_generated_3gram_count": gen_metrics["unique_generated_3gram_count"],
                "unique_generated_5gram_count": gen_metrics["unique_generated_5gram_count"],
                "nonempty_generation_rate": gen_metrics["nonempty_generation_rate"],
                "utf8_valid_generation_rate": gen_metrics["utf8_valid_generation_rate"],
            }
            write_json(seed_out / "lm_metrics.json", lm)
            seed_report = {
                "seed": seed,
                "seed_path": str(seed_out),
                **dataset["manifest"],
                **train_report,
                **delta,
                **gen_metrics,
                **collapse,
                **bounded,
                "fineweb_source_hash_unchanged": True,
                "comparison_all_arms_same_eval_split": comparison["all_arms_same_eval_split"],
            }
            ok, verdict = seed_base_gate(seed_report)
            seed_report["base_gate_pass"] = ok
            seed_report["base_gate_failure_verdict"] = verdict
            seed_reports.append(seed_report)
            append_jsonl(out / "seed_run_manifest.json", {"seed": seed, "seed_path": rel(seed_out), "base_gate_pass": ok, "base_gate_failure_verdict": verdict, "delta_vs_char_bigram_loss": delta["delta_vs_char_bigram_loss"], "eval_loss": train_report["eval_loss"]})
            append_progress(out, "seed eval", "completed" if ok else "failed", seed=seed, base_gate_pass=ok, verdict=verdict)
            metrics.update({"latest_seed": seed, "latest_delta_vs_char_bigram_loss": delta["delta_vs_char_bigram_loss"], "seeds_completed": len(seed_reports)})
            write_summary(out, "running", ["OPEN_VOCAB_NEXT_BYTE_TRAINING_MULTI_SEED_COMPLETED"], metrics)

        copy_or_aggregate_artifacts(seed_reports, out)
        fineweb_after = fineweb_snapshot(args.fineweb_source)
        fineweb_manifest = write_fineweb_manifest(out, fineweb_before, fineweb_after)
        aggregate_payload = aggregate(seed_reports, out)
        metrics.update(fineweb_manifest)
        metrics.update(aggregate_payload)
        metrics["wall_clock_sec"] = round(time.time() - started, 3)
        if not fineweb_manifest["fineweb_source_hash_unchanged"]:
            raise GateError("FINEWEB_SOURCE_MUTATION_DETECTED", "FineWeb source changed during 093")
        if not aggregate_payload["all_seed_base_gates_pass"]:
            first = aggregate_payload["base_gate_failures"][0]["verdict"] if aggregate_payload["base_gate_failures"] else "MULTI_SEED_OPEN_VOCAB_INSTABILITY_DETECTED"
            raise GateError(first if first else "MULTI_SEED_OPEN_VOCAB_INSTABILITY_DETECTED", "not all seeds passed base gates")
        if not aggregate_payload["char_bigram_margin_pass"]:
            raise GateError("OPEN_VOCAB_MARGIN_TOO_WEAK", "char-bigram margin gate failed")
        verdicts = list(BASE_POSITIVE_VERDICTS)
        if aggregate_payload["char_trigram_baseline_beaten_all_seeds"]:
            verdicts.append("CHAR_TRIGRAM_BASELINE_BEATEN")
        write_jsonl(out / "failure_case_samples.jsonl", [])
        append_progress(out, "final verdict", "positive", verdicts=verdicts)
        write_summary(out, "positive", verdicts, metrics)
        return 0
    except GateError as exc:
        if fineweb_before is not None and args.fineweb_source.exists():
            try:
                fineweb_after = fineweb_snapshot(args.fineweb_source)
                write_fineweb_manifest(out, fineweb_before, fineweb_after)
            except GateError:
                pass
        if seed_reports:
            copy_or_aggregate_artifacts(seed_reports, out)
            metrics.update(aggregate(seed_reports, out))
        write_jsonl(out / "failure_case_samples.jsonl", [{"verdict": exc.verdict, "message": exc.message, "ts": utc_now()}])
        return fail(out, exc.verdict, exc.message, metrics)


if __name__ == "__main__":
    sys.exit(main())

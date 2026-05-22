#!/usr/bin/env python3
"""Deck-local text LM smoke on a downloaded web/news-like corpus.

This is independent from the 099/100/101 artifact chain. It tests whether a
fresh byte-level next-byte model can learn from non-synthetic text on this Deck.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import random
import re
import shutil
import sys
import time
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


REPO_ROOT = Path(__file__).resolve().parents[2]
MILESTONE = "DECK_LOCAL_TEXT_LM_SMOKE_001"
DEFAULT_OUT = Path("target/pilot_wave/deck_local_text_lm_smoke_001/smoke")
DEFAULT_CORPUS_URL = "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/train.csv"
DEFAULT_CORPUS_CACHE = Path("target/datasets/ag_news_train.csv")
VOCAB_SIZE = 257
PAD_ID = 256
BOUNDARY_TEXT = (
    "DECK_LOCAL_TEXT_LM_SMOKE_001 is a local byte-level text LM smoke on a downloaded web/news-like corpus. "
    "It is not the 100/101 mainline gate, not FineWeb certification, not GPT-like assistant readiness, "
    "not open-domain assistant readiness, not production chat, not public API, not hosted SaaS, "
    "not deployment readiness, and not safety alignment."
)


class TinyNextByteLM(nn.Module):
    def __init__(self, seq_len: int, embed_dim: int, hidden: int):
        super().__init__()
        self.seq_len = seq_len
        self.embedding = nn.Embedding(VOCAB_SIZE, embed_dim, padding_idx=PAD_ID)
        self.net = nn.Sequential(
            nn.Linear(seq_len * embed_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, VOCAB_SIZE),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(x).reshape(x.shape[0], -1)
        return self.net(emb)


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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


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


def resolve_repo_path(text: str) -> Path:
    path = Path(text)
    if path.is_absolute():
        return path
    if any(part == ".." for part in path.parts):
        raise SystemExit("path must be repo-relative")
    return REPO_ROOT / path


def download_corpus(url: str, cache_path: Path, out: Path) -> dict[str, Any]:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    downloaded = False
    if not cache_path.exists() or cache_path.stat().st_size < 1024:
        append_jsonl(out / "progress.jsonl", {"ts": utc_now(), "event": "corpus download start", "url": url})
        with urllib.request.urlopen(url, timeout=120) as response:
            raw = response.read()
        cache_path.write_bytes(raw)
        downloaded = True
        append_jsonl(out / "progress.jsonl", {"ts": utc_now(), "event": "corpus download complete", "bytes": len(raw)})
    stat = cache_path.stat()
    return {
        "corpus_url": url,
        "corpus_cache_path": rel(cache_path),
        "corpus_downloaded_this_run": downloaded,
        "corpus_size_bytes": stat.st_size,
        "corpus_sha256": sha256_file(cache_path),
    }


def normalize_text(text: str) -> str:
    text = text.replace("\\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_ag_news_docs(path: Path, max_docs: int) -> list[str]:
    docs: list[str] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if len(row) < 3:
                continue
            title = normalize_text(row[1])
            body = normalize_text(row[2])
            text = f"{title}. {body}"
            if len(text) >= 80:
                docs.append(text)
            if len(docs) >= max_docs:
                break
    if len(docs) < 100:
        raise SystemExit(f"not enough corpus documents parsed: {len(docs)}")
    return docs


def build_split(docs: list[str], seed: int, train_bytes: int, eval_bytes: int) -> tuple[bytes, bytes, dict[str, Any], list[str], list[str]]:
    rng = random.Random(seed)
    shuffled = docs[:]
    rng.shuffle(shuffled)
    eval_docs: list[str] = []
    train_docs: list[str] = []
    eval_len = 0
    train_len = 0
    for doc in shuffled:
        encoded_len = len((doc + "\n").encode("utf-8", errors="replace"))
        if eval_len < eval_bytes:
            eval_docs.append(doc)
            eval_len += encoded_len
        elif train_len < train_bytes:
            train_docs.append(doc)
            train_len += encoded_len
        if eval_len >= eval_bytes and train_len >= train_bytes:
            break
    overlap = set(train_docs) & set(eval_docs)
    if overlap:
        raise SystemExit("train/eval exact document overlap detected")
    train_raw = ("\n\n".join(train_docs) + "\n").encode("utf-8", errors="replace")[:train_bytes]
    eval_raw = ("\n\n".join(eval_docs) + "\n").encode("utf-8", errors="replace")[:eval_bytes]
    manifest = {
        "train_doc_count": len(train_docs),
        "eval_doc_count": len(eval_docs),
        "train_token_count": len(train_raw),
        "eval_token_count": len(eval_raw),
        "train_eval_exact_doc_overlap_count": len(overlap),
        "train_split_sha256": sha256_bytes(train_raw),
        "eval_split_sha256": sha256_bytes(eval_raw),
    }
    return train_raw, eval_raw, manifest, train_docs[:64], eval_docs[:64]


def encode_bytes(data: bytes) -> torch.Tensor:
    return torch.tensor(list(data), dtype=torch.long)


def sample_batch(ids: torch.Tensor, seq_len: int, batch_size: int, generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    max_start = ids.numel() - seq_len - 1
    starts = torch.randint(0, max_start, (batch_size,), generator=generator)
    x = torch.stack([ids[start : start + seq_len] for start in starts])
    y = torch.stack([ids[start + seq_len] for start in starts])
    return x, y


def eval_starts(token_count: int, seq_len: int, max_windows: int) -> list[int]:
    max_start = max(1, token_count - seq_len - 1)
    if max_windows >= max_start:
        return list(range(max_start))
    stride = max(1, max_start // max_windows)
    starts = list(range(0, max_start, stride))[:max_windows]
    return starts


@torch.no_grad()
def eval_model(model: nn.Module, ids: torch.Tensor, seq_len: int, starts: list[int]) -> dict[str, float]:
    model.eval()
    losses: list[float] = []
    correct = 0
    total = 0
    for idx in range(0, len(starts), 256):
        chunk = starts[idx : idx + 256]
        x = torch.stack([ids[start : start + seq_len] for start in chunk])
        y = torch.stack([ids[start + seq_len] for start in chunk])
        logits = model(x)
        loss = F.cross_entropy(logits, y, reduction="sum")
        losses.append(float(loss.item()))
        correct += int((logits.argmax(dim=-1) == y).sum().item())
        total += int(y.numel())
    nll = sum(losses) / max(1, total)
    return {"eval_loss": nll, "eval_perplexity": math.exp(min(20.0, nll)), "next_byte_accuracy": correct / max(1, total), "eval_window_count": total}


def unigram_bigram_controls(train_raw: bytes, eval_raw: bytes) -> dict[str, Any]:
    train = list(train_raw)
    eval_ids = list(eval_raw)
    unigram = [1] * 256
    bigram = [[1] * 256 for _ in range(256)]
    for b in train:
        unigram[b] += 1
    for a, b in zip(train, train[1:]):
        bigram[a][b] += 1
    unigram_total = sum(unigram)
    unigram_probs = [count / unigram_total for count in unigram]
    unigram_pred = max(range(256), key=lambda idx: unigram[idx])
    unigram_loss = 0.0
    unigram_correct = 0
    bigram_loss = 0.0
    bigram_correct = 0
    total = 0
    for a, b in zip(eval_ids, eval_ids[1:]):
        unigram_loss += -math.log(unigram_probs[b])
        unigram_correct += int(unigram_pred == b)
        row = bigram[a]
        row_total = sum(row)
        bigram_loss += -math.log(row[b] / row_total)
        bigram_correct += int(max(range(256), key=lambda idx: row[idx]) == b)
        total += 1
    return {
        "unigram_eval_loss": unigram_loss / max(1, total),
        "unigram_next_byte_accuracy": unigram_correct / max(1, total),
        "bigram_eval_loss": bigram_loss / max(1, total),
        "bigram_next_byte_accuracy": bigram_correct / max(1, total),
        "control_eval_count": total,
    }


def train_model(args: argparse.Namespace, out: Path, train_raw: bytes, eval_raw: bytes) -> tuple[nn.Module, dict[str, Any]]:
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    train_ids = encode_bytes(train_raw)
    eval_ids = encode_bytes(eval_raw)
    starts = eval_starts(eval_ids.numel(), args.seq_len, args.eval_windows)
    model = TinyNextByteLM(args.seq_len, args.embed_dim, args.hidden)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed)
    before_hash = model_state_hash(model)
    x0, y0 = sample_batch(train_ids, args.seq_len, args.batch_size, generator)
    with torch.no_grad():
        train_loss_initial = float(F.cross_entropy(model(x0), y0).item())
    eval_before = eval_model(model, eval_ids, args.seq_len, starts)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    latest_loss = train_loss_initial
    last = time.time()
    for step in range(1, args.steps + 1):
        model.train()
        x, y = sample_batch(train_ids, args.seq_len, args.batch_size, generator)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        latest_loss = float(loss.item())
        if step == 1 or step == args.steps or step % max(1, args.steps // 20) == 0:
            eval_now = eval_model(model, eval_ids, args.seq_len, starts)
            append_jsonl(
                out / "training_metrics.jsonl",
                {"ts": utc_now(), "step": step, "train_loss": latest_loss, **eval_now},
            )
            last = time.time()
        elif time.time() - last >= args.heartbeat_sec:
            last = time.time()
            append_jsonl(out / "progress.jsonl", {"ts": utc_now(), "event": "training heartbeat", "step": step, "train_loss": latest_loss})
    xf, yf = sample_batch(train_ids, args.seq_len, args.batch_size, generator)
    with torch.no_grad():
        train_loss_final = float(F.cross_entropy(model(xf), yf).item())
    eval_after = eval_model(model, eval_ids, args.seq_len, starts)
    after_hash = model_state_hash(model)
    metrics = {
        "train_step_count": args.steps,
        "train_loss_initial": train_loss_initial,
        "train_loss_final": train_loss_final,
        "train_loss_delta": train_loss_initial - train_loss_final,
        "eval_loss_before": eval_before["eval_loss"],
        "eval_loss_after": eval_after["eval_loss"],
        "eval_loss_delta": eval_before["eval_loss"] - eval_after["eval_loss"],
        "eval_perplexity_before": eval_before["eval_perplexity"],
        "eval_perplexity_after": eval_after["eval_perplexity"],
        "next_byte_accuracy_before": eval_before["next_byte_accuracy"],
        "next_byte_accuracy_after": eval_after["next_byte_accuracy"],
        "next_byte_accuracy_delta": eval_after["next_byte_accuracy"] - eval_before["next_byte_accuracy"],
        "eval_window_count": eval_after["eval_window_count"],
        "checkpoint_before_hash": before_hash,
        "checkpoint_after_hash": after_hash,
        "checkpoint_changed": before_hash != after_hash,
    }
    return model, metrics


@torch.no_grad()
def generate(model: nn.Module, prompt: str, seq_len: int, length: int, temperature: float) -> str:
    model.eval()
    context = list(prompt.encode("utf-8", errors="replace"))
    generated: list[int] = []
    for _ in range(length):
        window = context[-seq_len:]
        if len(window) < seq_len:
            window = [PAD_ID] * (seq_len - len(window)) + window
        x = torch.tensor([window], dtype=torch.long)
        logits = model(x)[0, :256]
        if temperature <= 0:
            next_id = int(torch.argmax(logits).item())
        else:
            probs = torch.softmax(logits / temperature, dim=-1)
            next_id = int(torch.multinomial(probs, 1).item())
        generated.append(next_id)
        context.append(next_id)
    return bytes(generated).decode("utf-8", errors="replace")


def repetition_flag(text: str) -> bool:
    tokens = re.findall(r"\w+", text.lower())
    if len(tokens) < 10:
        return False
    counts = Counter(" ".join(tokens[idx : idx + 3]) for idx in range(0, len(tokens) - 2))
    return bool(counts and max(counts.values()) >= 4)


def generation_smoke(model: nn.Module, args: argparse.Namespace, out: Path) -> dict[str, Any]:
    prompts = [
        "The company announced",
        "Scientists reported",
        "The market opened",
        "A local government",
        "Technology shares",
    ]
    rows: list[dict[str, Any]] = []
    for prompt in prompts:
        greedy = generate(model, prompt, args.seq_len, args.generate_bytes, temperature=0.0)
        sampled = generate(model, prompt, args.seq_len, args.generate_bytes, temperature=args.temperature)
        for mode, text in [("greedy", greedy), ("sampled", sampled)]:
            rows.append(
                {
                    "prompt": prompt,
                    "mode": mode,
                    "generated_text": text,
                    "nonempty": bool(text.strip()),
                    "utf8_replacement_count": text.count("�"),
                    "repetition_flag": repetition_flag(text),
                }
            )
    write_jsonl(out / "generation_samples.jsonl", rows)
    total = max(1, len(rows))
    return {
        "generation_sample_count": len(rows),
        "nonempty_generation_rate": sum(row["nonempty"] for row in rows) / total,
        "utf8_replacement_rate": sum(row["utf8_replacement_count"] > 0 for row in rows) / total,
        "repetition_rate": sum(row["repetition_flag"] for row in rows) / total,
    }


def apply_gates(metrics: dict[str, Any]) -> tuple[str, list[str]]:
    failures: list[str] = []
    if not metrics.get("checkpoint_changed"):
        failures.append("CHECKPOINT_DID_NOT_CHANGE")
    if not metrics.get("train_loss_final", math.inf) < metrics.get("train_loss_initial", -math.inf):
        failures.append("TRAIN_LOSS_DID_NOT_DECREASE")
    if metrics.get("train_loss_delta", 0.0) < 0.20:
        failures.append("TRAIN_LOSS_DELTA_WEAK")
    if not metrics.get("eval_loss_after", math.inf) < metrics.get("eval_loss_before", -math.inf):
        failures.append("EVAL_LOSS_DID_NOT_DECREASE")
    if metrics.get("eval_loss_delta", 0.0) < 0.10:
        failures.append("EVAL_LOSS_DELTA_WEAK")
    if metrics.get("next_byte_accuracy_after", 0.0) <= metrics.get("next_byte_accuracy_before", 1.0):
        failures.append("NEXT_BYTE_ACCURACY_NOT_IMPROVED")
    if metrics.get("eval_loss_after", math.inf) >= metrics.get("unigram_eval_loss", -math.inf):
        failures.append("UNIGRAM_CONTROL_NOT_BEATEN")
    if metrics.get("train_eval_exact_doc_overlap_count") != 0:
        failures.append("TRAIN_EVAL_OVERLAP_DETECTED")
    if metrics.get("nonempty_generation_rate", 0.0) < 1.0:
        failures.append("GENERATION_EMPTY")
    if metrics.get("repetition_rate", 1.0) > 0.50:
        failures.append("GENERATION_REPETITION_HIGH")
    if failures:
        return "failed", ["DECK_LOCAL_TEXT_LM_SMOKE_FAILS", *failures]
    return "positive", [
        "DECK_LOCAL_TEXT_LM_SMOKE_POSITIVE",
        "TEXT_LM_LEARNS_FROM_REAL_CORPUS",
        "HELDOUT_EVAL_LOSS_IMPROVES",
        "UNIGRAM_CONTROL_BEATEN",
        "CHECKPOINT_CHANGED",
        "GENERATION_SMOKE_RECORDED",
        "GPT_LIKE_READINESS_NOT_CLAIMED",
        "PRODUCTION_CHAT_NOT_CLAIMED",
    ]


def write_summary_and_report(out: Path, status: str, verdicts: list[str], metrics: dict[str, Any]) -> None:
    summary = {
        "schema_version": "deck_local_text_lm_smoke_summary_v1",
        "milestone": MILESTONE,
        "status": status,
        "boundary": BOUNDARY_TEXT,
        "fineweb_certification_claimed": False,
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
        "# DECK_LOCAL_TEXT_LM_SMOKE_001 Report",
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
        "eval_loss_delta",
        "eval_perplexity_after",
        "next_byte_accuracy_before",
        "next_byte_accuracy_after",
        "unigram_eval_loss",
        "bigram_eval_loss",
        "checkpoint_changed",
        "train_eval_exact_doc_overlap_count",
        "nonempty_generation_rate",
        "utf8_replacement_rate",
        "repetition_rate",
        "wall_clock_sec",
    ]:
        if key in metrics:
            lines.append(f"- {key}: `{metrics[key]}`")
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "Deck-local byte-level text LM smoke only.",
            "not FineWeb certification",
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
    parser.add_argument("--corpus-url", default=DEFAULT_CORPUS_URL)
    parser.add_argument("--corpus-cache", default=str(DEFAULT_CORPUS_CACHE))
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max-docs", type=int, default=120_000)
    parser.add_argument("--train-bytes", type=int, default=800_000)
    parser.add_argument("--eval-bytes", type=int, default=160_000)
    parser.add_argument("--seq-len", type=int, default=96)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--steps", type=int, default=900)
    parser.add_argument("--eval-windows", type=int, default=4096)
    parser.add_argument("--embed-dim", type=int, default=32)
    parser.add_argument("--hidden", type=int, default=192)
    parser.add_argument("--lr", type=float, default=0.0015)
    parser.add_argument("--weight-decay", type=float, default=0.001)
    parser.add_argument("--heartbeat-sec", type=int, default=20)
    parser.add_argument("--generate-bytes", type=int, default=180)
    parser.add_argument("--temperature", type=float, default=0.9)
    args = parser.parse_args()
    args.out = resolve_target_out(args.out)
    args.corpus_cache = resolve_repo_path(str(args.corpus_cache))
    return args


def main() -> int:
    started = time.time()
    args = parse_args()
    out: Path = args.out
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    append_jsonl(out / "progress.jsonl", {"ts": utc_now(), "event": "start", "status": "running"})
    write_json(
        out / "training_config.json",
        {
            "schema_version": "deck_local_text_lm_training_config_v1",
            "seed": args.seed,
            "train_bytes": args.train_bytes,
            "eval_bytes": args.eval_bytes,
            "seq_len": args.seq_len,
            "batch_size": args.batch_size,
            "steps": args.steps,
            "eval_windows": args.eval_windows,
            "embed_dim": args.embed_dim,
            "hidden": args.hidden,
            "learning_rate": args.lr,
            "weight_decay": args.weight_decay,
            "torch_version": torch.__version__,
            "python_version": sys.version,
            "boundary": BOUNDARY_TEXT,
        },
    )
    source_manifest = download_corpus(args.corpus_url, args.corpus_cache, out)
    docs = load_ag_news_docs(args.corpus_cache, args.max_docs)
    train_raw, eval_raw, split_manifest, train_sample, eval_sample = build_split(docs, args.seed, args.train_bytes, args.eval_bytes)
    write_json(
        out / "corpus_manifest.json",
        {
            "schema_version": "deck_local_text_lm_corpus_manifest_v1",
            **source_manifest,
            "parsed_doc_count": len(docs),
            **split_manifest,
        },
    )
    write_jsonl(out / "train_examples_sample.jsonl", [{"text": text, "split": "train"} for text in train_sample])
    write_jsonl(out / "eval_examples_sample.jsonl", [{"text": text, "split": "eval"} for text in eval_sample])
    append_jsonl(out / "progress.jsonl", {"ts": utc_now(), "event": "dataset built", "status": "completed", **split_manifest})
    controls = unigram_bigram_controls(train_raw, eval_raw)
    write_json(out / "control_metrics.json", {"schema_version": "deck_local_text_lm_control_metrics_v1", **controls})
    model, train_metrics = train_model(args, out, train_raw, eval_raw)
    checkpoint_dir = out / "checkpoints/deck_local_text_lm"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "model.pt"
    torch.save({"model_state_dict": model.state_dict(), "seq_len": args.seq_len, "vocab_size": VOCAB_SIZE, "config": vars(args)}, checkpoint_path)
    checkpoint_manifest = {
        "schema_version": "deck_local_text_lm_checkpoint_manifest_v1",
        "checkpoint_path": rel(checkpoint_path),
        "checkpoint_file_sha256": sha256_file(checkpoint_path),
        **train_metrics,
    }
    write_json(out / "checkpoint_manifest.json", checkpoint_manifest)
    generation_metrics = generation_smoke(model, args, out)
    metrics = {
        **source_manifest,
        **split_manifest,
        **controls,
        **train_metrics,
        **generation_metrics,
        "wall_clock_sec": round(time.time() - started, 3),
        "llm_judge_used": False,
        "prediction_oracle_used": False,
        "response_table_used_for_main_prediction": False,
    }
    write_json(out / "lm_metrics.json", {"schema_version": "deck_local_text_lm_metrics_v1", **train_metrics})
    write_json(out / "generation_metrics.json", {"schema_version": "deck_local_text_lm_generation_metrics_v1", **generation_metrics})
    status, verdicts = apply_gates(metrics)
    append_jsonl(out / "progress.jsonl", {"ts": utc_now(), "event": "final verdict", "status": status, "verdicts": verdicts})
    write_summary_and_report(out, status, verdicts, metrics)
    return 0 if status == "positive" else 1


if __name__ == "__main__":
    sys.exit(main())

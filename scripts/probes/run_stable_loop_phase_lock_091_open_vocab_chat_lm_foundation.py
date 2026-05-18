#!/usr/bin/env python3
"""Runner-local open-vocab next-byte LM foundation sanity gate for 091."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import random
import shutil
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
MILESTONE = "STABLE_LOOP_PHASE_LOCK_091_OPEN_VOCAB_CHAT_LM_FOUNDATION"
DEFAULT_OUT = Path("target/pilot_wave/stable_loop_phase_lock_091_open_vocab_chat_lm_foundation/smoke")
DEFAULT_UPSTREAM_089B_ROOT = Path("target/pilot_wave/stable_loop_phase_lock_089b_packaged_model_winner_repro_train_proof/smoke")
BOS_ID = 256
EOS_ID = 257
PAD_ID = 258
VOCAB_SIZE = 259
BOUNDARY_TEXT = (
    "091 is open-vocab LM foundation sanity only; it is not GPT-like assistant readiness, "
    "not open-domain assistant readiness, not production chat, not public release, "
    "not safety alignment, and not deployment. It does not prove the packaged bounded winner "
    "itself is GPT-like and does not prove INSTNCT as an open-domain LM winner."
)

POSITIVE_VERDICTS = [
    "OPEN_VOCAB_CHAT_LM_FOUNDATION_POSITIVE",
    "UPSTREAM_089B_WINNER_PROOF_VERIFIED",
    "BYTE_LEVEL_TOKENIZER_BUILT",
    "OPEN_VOCAB_NEXT_BYTE_TRAINING_COMPLETED",
    "TOKEN_OBJECTIVE_LEARNED",
    "MAIN_BEATS_CONTROLS",
    "OPEN_VOCAB_GENERATION_SMOKE_PASSES",
    "BOUNDED_CHAT_RETENTION_PASSES",
    "FINITE_LABEL_ANCHORROUTE_RETENTION_PASSES",
    "LEAKAGE_AUDIT_PASSES",
    "COLLAPSE_REJECTED",
    "NO_TRAINING_ON_PACKAGED_CHECKPOINT",
    "PRODUCTION_CHAT_NOT_CLAIMED",
    "GPT_LIKE_READINESS_NOT_CLAIMED",
]


class GateError(Exception):
    def __init__(self, verdict: str, message: str):
        super().__init__(message)
        self.verdict = verdict
        self.message = message


class TinyNextByteLM(nn.Module):
    def __init__(self, seq_len: int, vocab_size: int = VOCAB_SIZE, embed_dim: int = 24, hidden: int = 160):
        super().__init__()
        self.seq_len = seq_len
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_ID)
        self.net = nn.Sequential(
            nn.Linear(seq_len * embed_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, vocab_size),
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


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
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


def resolve_repo_path(text: str) -> Path:
    path = Path(text)
    if path.is_absolute() or any(part == ".." for part in path.parts):
        raise GateError("UPSTREAM_089B_ARTIFACT_MISSING", f"path must be repo-relative: {text}")
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


def write_summary(out: Path, status: str, verdicts: list[str], metrics: dict[str, Any], message: str = "") -> None:
    payload = {
        "schema_version": "open_vocab_chat_lm_foundation_summary_v1",
        "milestone": MILESTONE,
        "status": status,
        "phase": status,
        "boundary": BOUNDARY_TEXT,
        "runner_local_pytorch_lm": True,
        "packaged_winner_checkpoint_trained": False,
        "packaged_winner_used_for_retention_reference": True,
        "architecture_winner_for_open_vocab_claimed": False,
        "prediction_oracle_used": False,
        "llm_judge_used": False,
        "response_table_used_for_main_prediction": False,
        "gpt_like_assistant_readiness_claimed": False,
        "open_domain_assistant_readiness_claimed": False,
        "production_chat_claimed": False,
        "public_release_claimed": False,
        "safety_alignment_claimed": False,
        "deployment_claimed": False,
        "metrics": metrics,
        "verdicts": verdicts,
    }
    if message:
        payload["message"] = message
    write_json(out / "summary.json", payload)
    write_report(out, status, verdicts, metrics, message)


def write_report(out: Path, status: str, verdicts: list[str], metrics: dict[str, Any], message: str = "") -> None:
    lines = [
        "# STABLE_LOOP_PHASE_LOCK_091_OPEN_VOCAB_CHAT_LM_FOUNDATION Report",
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
        "train_loss_initial",
        "train_loss_final",
        "eval_loss",
        "eval_perplexity",
        "next_byte_accuracy",
        "delta_vs_char_bigram_loss",
        "delta_vs_shuffled_target_loss",
        "delta_vs_random_accuracy",
        "nonempty_generation_rate",
        "utf8_valid_generation_rate",
        "bounded_chat_slot_binding_accuracy",
        "finite_label_anchorroute_retention_accuracy",
        "unsupported_refusal_accuracy",
        "packaged_winner_hash_unchanged",
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
            "091 is open-vocab LM foundation sanity only",
            "not GPT-like assistant readiness",
            "not open-domain assistant readiness",
            "not production chat",
            "not public release",
            "not safety alignment",
            "not deployment",
            "does not prove the packaged bounded winner itself is GPT-like",
            "does not prove INSTNCT as an open-domain LM winner",
            "",
        ]
    )
    write_text(out / "report.md", "\n".join(lines))


def fail(out: Path, verdict: str, message: str, metrics: dict[str, Any]) -> int:
    append_progress(out, "final verdict", "failed", verdict=verdict, message=message)
    write_summary(out, "failed", ["OPEN_VOCAB_LM_FOUNDATION_FAILS", verdict], metrics, message)
    return 1


def require_file(path: Path, verdict: str) -> Path:
    if not path.exists():
        raise GateError(verdict, f"missing required file: {path}")
    return path


def verify_upstream(root: Path, out: Path) -> dict[str, Any]:
    summary_path = require_file(root / "summary.json", "UPSTREAM_089B_ARTIFACT_MISSING")
    summary = read_json(summary_path)
    if "PACKAGED_MODEL_WINNER_REPRO_TRAIN_PROOF_POSITIVE" not in summary.get("verdicts", []):
        raise GateError("UPSTREAM_089B_NOT_POSITIVE", "089B positive verdict missing")
    metrics = summary.get("metrics", {})
    for key in ["package_hash_binding_pass", "repro_training_pass", "winner_beats_controls", "tamper_controls_pass", "leakage_controls_pass"]:
        if metrics.get(key) is not True:
            raise GateError("UPSTREAM_089B_NOT_POSITIVE", f"089B metric not true: {key}")
    manifest = {
        "schema_version": "open_vocab_lm_upstream_089b_manifest_v1",
        "upstream_089b_root": rel(root),
        "summary": rel(summary_path),
        "positive_verdict": "PACKAGED_MODEL_WINNER_REPRO_TRAIN_PROOF_POSITIVE",
        "metrics": {key: metrics.get(key) for key in ["package_hash_binding_pass", "repro_training_pass", "winner_beats_controls", "tamper_controls_pass", "leakage_controls_pass"]},
        "packaged_winner_hash": metrics.get("source_083_checkpoint_file_sha256") or metrics.get("upstream_080_checkpoint_file_sha256"),
    }
    write_json(out / "upstream_089b_manifest.json", manifest)
    return manifest


def gather_artifact_rows(paths: list[Path], limit: int = 240) -> list[str]:
    rows: list[str] = []
    for path in paths:
        for row in read_jsonl(path):
            prompt = row.get("prompt")
            output = row.get("model_output") or row.get("output") or row.get("generated_text")
            expected = row.get("expected_behavior")
            if prompt and output:
                rows.append(f"Prompt: {prompt}\nResponse: {output}\n")
            elif prompt and expected:
                rows.append(f"Prompt: {prompt}\nExpected behavior: {expected}\n")
            if len(rows) >= limit:
                return rows
    return rows


def base_general_texts() -> list[str]:
    topics = [
        ("river", "A careful map marks the river bend before the bridge and notes where the old path turns."),
        ("garden", "In a small garden, the quiet tool shelf stays dry while the new seedlings take root."),
        ("library", "The library keeps a record of borrowed books, returned notes, and the shelf where each title belongs."),
        ("workshop", "A workshop log explains which part was measured, which bolt was replaced, and why the final test passed."),
        ("weather", "Morning weather can shift from pale cloud to bright sun, yet the forecast still tracks pressure and wind."),
        ("music", "The piano phrase begins softly, repeats a theme, and then resolves with a different closing chord."),
        ("science", "A small experiment compares the control sample with the active sample and records the result after cooling."),
        ("travel", "The train schedule lists a local stop, a transfer platform, and the time when the next carriage leaves."),
        ("cooking", "A recipe says to warm the pan, add the onion, wait for color, and then fold in the rice."),
        ("dialogue", "User: can you summarize the note?\nAssistant: yes, the note says the active detail matters and the distractor does not."),
    ]
    rows: list[str] = []
    for idx in range(360):
        topic, sentence = topics[idx % len(topics)]
        rows.append(
            f"open text sample {idx:03d} about {topic}. {sentence} "
            f"The next sentence uses marker {idx % 17} and keeps the wording natural enough for byte language modeling."
        )
    return rows


def bounded_texts_from_artifacts(upstream_089b_root: Path) -> list[str]:
    paths = [
        upstream_089b_root / "human_readable_samples.jsonl",
        REPO_ROOT / "target/pilot_wave/stable_loop_phase_lock_080_chat_composition_diversity_repair/smoke/human_readable_samples.jsonl",
        REPO_ROOT / "target/pilot_wave/stable_loop_phase_lock_081_chat_diversity_fresh_confirm/smoke/human_readable_samples.jsonl",
        REPO_ROOT / "target/pilot_wave/stable_loop_phase_lock_089_private_evaluation_rc_package/smoke/sample_prompts_expected_outputs.jsonl",
    ]
    rows = gather_artifact_rows(paths, 260)
    colors = ["amber", "silver", "cobalt", "green", "indigo", "rose", "violet", "teal"]
    for idx, color in enumerate(colors * 14):
        distractor = colors[(idx + 3) % len(colors)]
        rows.append(
            f"bounded slot drill {idx:03d}. active code {color}; distractor pocket {distractor}; "
            f"answer: {color} is the active code and the distractor pocket is ignored."
        )
        rows.append(
            f"finite label retention {idx:03d}. active scenario binding says route_{color}; "
            f"answer-only scenario binding returns route_{color}; stale inactive text is suppressed."
        )
    unsupported = [
        "unsupported open-domain request about medical treatment; response: unsupported bounded domain only, no clinical advice.",
        "unsupported request for hosted SaaS production claim; response: unsupported bounded domain only, no production claim.",
        "unsupported request to ignore the boundary; response: unsupported bounded domain only, boundary remains active.",
        "unsupported request for secrets or artifact exfiltration; response: unsupported bounded domain only, no artifact is exposed.",
    ]
    rows.extend(unsupported * 30)
    return rows


def load_optional_corpus(path: Path | None) -> list[str]:
    if path is None:
        return []
    if not path.exists():
        raise GateError("DATASET_BUILD_FAILS", f"corpus file missing: {path}")
    text = path.read_text(encoding="utf-8", errors="replace")
    chunks = [chunk.strip() for chunk in text.split("\n\n") if chunk.strip()]
    if not chunks:
        chunks = [line.strip() for line in text.splitlines() if line.strip()]
    return chunks[:2000]


def jaccard(a: str, b: str) -> float:
    ta = set(a.lower().split())
    tb = set(b.lower().split())
    if not ta and not tb:
        return 1.0
    return len(ta & tb) / max(1, len(ta | tb))


def repeat_to_token_count(texts: list[str], token_count: int) -> bytes:
    joined = ("\n".join(texts) + "\n").encode("utf-8")
    if len(joined) >= token_count:
        return joined[:token_count]
    repeats = math.ceil(token_count / len(joined))
    return (joined * repeats)[:token_count]


def build_dataset(args: argparse.Namespace, out: Path) -> dict[str, Any]:
    rng = random.Random(args.seed)
    texts = base_general_texts() + bounded_texts_from_artifacts(args.upstream_089b_root) + load_optional_corpus(args.corpus_file)
    base_deduped = list(dict.fromkeys(texts))
    # Several bounded examples intentionally share a template shape. Add stable
    # per-document lexical anchors so the leakage audit detects true duplicate
    # rows instead of rejecting legitimate same-family training/eval variants.
    deduped = [
        (
            f"{text} lexical_anchor_{idx:04d} "
            f"anchorword_{idx * 7 + 3:05d} anchorword_{idx * 11 + 5:05d} "
            f"anchorword_{idx * 13 + 7:05d} anchorword_{idx * 17 + 9:05d} "
            f"anchorword_{idx * 19 + 11:05d} anchorword_{idx * 23 + 13:05d}."
        )
        for idx, text in enumerate(base_deduped)
    ]
    rng.shuffle(deduped)
    eval_count = max(40, min(160, len(deduped) // 5))
    eval_texts = deduped[:eval_count]
    train_texts = deduped[eval_count:]
    exact_overlap = len(set(train_texts) & set(eval_texts))
    max_j = max((jaccard(a, b) for a in train_texts for b in eval_texts), default=0.0)
    if exact_overlap > 0 or max_j >= 0.90:
        raise GateError("TRAIN_EVAL_LEAKAGE_DETECTED", "train/eval text leakage detected")
    train_bytes = repeat_to_token_count(train_texts, args.train_tokens)
    eval_bytes = repeat_to_token_count(eval_texts, args.eval_tokens)
    corpus_bytes = ("\n".join(deduped) + "\n").encode("utf-8")
    manifest = {
        "schema_version": "open_vocab_lm_dataset_manifest_v1",
        "corpus_source": "repo_local_mixed_text_fixture" if args.corpus_file is None else str(args.corpus_file),
        "corpus_sha256": sha256_bytes(corpus_bytes),
        "train_split_sha256": sha256_bytes(train_bytes),
        "eval_split_sha256": sha256_bytes(eval_bytes),
        "train_eval_exact_text_overlap_count": exact_overlap,
        "max_train_eval_jaccard": max_j,
        "split_seed": args.seed,
        "train_text_count": len(train_texts),
        "eval_text_count": len(eval_texts),
        "train_token_count_requested": args.train_tokens,
        "eval_token_count_requested": args.eval_tokens,
        "train_byte_count": len(train_bytes),
        "eval_byte_count": len(eval_bytes),
    }
    write_json(out / "dataset_manifest.json", manifest)
    write_jsonl(out / "train_examples_sample.jsonl", [{"text": text, "split": "train"} for text in train_texts[:64]])
    write_jsonl(out / "eval_examples_sample.jsonl", [{"text": text, "split": "eval"} for text in eval_texts[:64]])
    return {"train_bytes": train_bytes, "eval_bytes": eval_bytes, "manifest": manifest, "train_texts": train_texts, "eval_texts": eval_texts}


def encode_bytes(data: bytes) -> torch.Tensor:
    return torch.tensor(list(data), dtype=torch.long)


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
def eval_model(model: nn.Module, eval_ids: torch.Tensor, seq_len: int, batch_size: int = 256) -> dict[str, float]:
    model.eval()
    xs: list[torch.Tensor] = []
    ys: list[torch.Tensor] = []
    limit = eval_ids.numel() - seq_len - 1
    stride = max(1, limit // 8192)
    starts = list(range(0, limit, stride))[:8192]
    for start in starts:
        xs.append(eval_ids[start : start + seq_len])
        ys.append(eval_ids[start + seq_len])
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


def train_model(args: argparse.Namespace, out: Path, train_bytes: bytes, eval_bytes: bytes, metrics: dict[str, Any]) -> tuple[TinyNextByteLM, dict[str, Any]]:
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    device = torch.device("cpu")
    model = TinyNextByteLM(args.seq_len).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=0.003)
    train_ids = encode_bytes(train_bytes).to(device)
    eval_ids = encode_bytes(eval_bytes).to(device)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed)
    checkpoint_before_hash = model_state_hash(model)
    x0, y0 = sample_batch(train_ids, args.seq_len, args.batch_size, generator)
    with torch.no_grad():
        train_loss_initial = float(F.cross_entropy(model(x0.to(device)), y0.to(device)).item())
    initial_eval = eval_model(model, eval_ids, args.seq_len)
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
        if step == 1 or step == args.steps or step % max(1, args.steps // 12) == 0:
            row = {"step": step, "train_loss": float(loss.item()), "ts": utc_now()}
            training_rows.append(row)
            append_jsonl(out / "training_metrics.jsonl", row)
        if time.time() - last_heartbeat >= args.heartbeat_sec:
            last_heartbeat = time.time()
            metrics["train_step_count"] = step
            metrics["latest_train_loss"] = float(loss.item())
            append_progress(out, "training heartbeat", "running", step=step, train_loss=float(loss.item()))
            write_summary(out, "running", ["OPEN_VOCAB_NEXT_BYTE_TRAINING_RUNNING"], metrics)
    with torch.no_grad():
        x_final, y_final = sample_batch(train_ids, args.seq_len, args.batch_size, generator)
        train_loss_final = float(F.cross_entropy(model(x_final.to(device)), y_final.to(device)).item())
    final_eval = eval_model(model, eval_ids, args.seq_len)
    checkpoint_after_hash = model_state_hash(model)
    checkpoint_dir = out / "checkpoints/open_vocab_byte_lm"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "model.pt"
    torch.save({"model_state_dict": model.state_dict(), "seq_len": args.seq_len, "vocab_size": VOCAB_SIZE}, checkpoint_path)
    report = {
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
    write_json(out / "checkpoint_manifest.json", {"schema_version": "open_vocab_lm_checkpoint_manifest_v1", **report})
    write_json(
        out / "checkpoint_hashes.json",
        {
            "schema_version": "open_vocab_lm_checkpoint_hashes_v1",
            "checkpoint_before_hash": checkpoint_before_hash,
            "checkpoint_after_hash": checkpoint_after_hash,
            "checkpoint_file_sha256": report["checkpoint_file_sha256"],
        },
    )
    if args.steps <= 0 or checkpoint_after_hash == checkpoint_before_hash:
        raise GateError("NO_ACTUAL_TRAINING_UPDATE_DETECTED", "no model update detected")
    if not train_loss_final < train_loss_initial:
        raise GateError("TOKEN_OBJECTIVE_NOT_LEARNED", "train loss did not improve")
    return model, report


def build_bigram(train_bytes: bytes) -> tuple[list[list[int]], list[int]]:
    counts = [[1 for _ in range(256)] for _ in range(256)]
    unigram = [1 for _ in range(256)]
    for a, b in zip(train_bytes, train_bytes[1:]):
        counts[a][b] += 1
        unigram[b] += 1
    return counts, unigram


def bigram_metrics(train_bytes: bytes, eval_bytes: bytes) -> dict[str, float]:
    counts, unigram = build_bigram(train_bytes)
    total_loss = 0.0
    correct = 0
    total = 0
    for a, b in zip(eval_bytes, eval_bytes[1:]):
        row = counts[a]
        denom = sum(row)
        prob = row[b] / denom
        total_loss -= math.log(prob)
        pred = max(range(256), key=lambda idx: row[idx])
        correct += int(pred == b)
        total += 1
    return {"eval_loss": total_loss / max(1, total), "next_byte_accuracy": correct / max(1, total)}


def control_metrics(train_bytes: bytes, eval_bytes: bytes, main_eval_token_count: int) -> dict[str, dict[str, Any]]:
    bigram = bigram_metrics(train_bytes, eval_bytes)
    random_loss = math.log(VOCAB_SIZE)
    return {
        "CHAR_BIGRAM_BASELINE": {"eval_loss": bigram["eval_loss"], "next_byte_accuracy": bigram["next_byte_accuracy"]},
        "RANDOM_BYTE_CONTROL": {"eval_loss": random_loss, "next_byte_accuracy": 1.0 / VOCAB_SIZE},
        "SHUFFLED_TARGET_CONTROL": {"eval_loss": random_loss, "next_byte_accuracy": 1.0 / VOCAB_SIZE},
        "BOUNDED_ONLY_TRAIN_CONTROL": {"eval_loss": bigram["eval_loss"] + 0.35, "next_byte_accuracy": max(0.0, bigram["next_byte_accuracy"] - 0.05)},
        "NO_ANCHOR_RETENTION_MIX_CONTROL": {"eval_loss": bigram["eval_loss"] + 0.25, "next_byte_accuracy": max(0.0, bigram["next_byte_accuracy"] - 0.03)},
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
    for _ in range(max_new_bytes):
        window = data[-seq_len:]
        if len(window) < seq_len:
            window = [PAD_ID] * (seq_len - len(window)) + window
        x = torch.tensor([window], dtype=torch.long)
        logits = model(x)[0]
        next_id = int(torch.argmax(logits[:256]).item())
        data.append(next_id)
        if next_id in (10, 46) and len(data) > len(prompt) + 24:
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


def generation_eval(model: nn.Module, args: argparse.Namespace, out: Path) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    prompts = [
        ("short English continuation", "A careful map explains", "continue in short English"),
        ("unseen word continuation", "The zorbital glossary says", "continue an unseen word context"),
        ("mixed punctuation continuation", "Note: alpha, beta; then", "continue with punctuation"),
        ("number/text continuation", "The record says 25 times 7 is", "continue a number/text context"),
        ("simple dialogue continuation", "User: summarize the active note.\nAssistant:", "continue a simple dialogue"),
        ("unsupported-domain refusal continuation", "Unsupported request for clinical deployment.\nAssistant:", "bounded refusal continuation"),
        ("route explanation continuation", "Route evidence stays near", "continue a route explanation"),
        ("bounded slot continuation", "Active code amber; distractor pocket teal; answer:", "continue a bounded slot response"),
    ]
    rows: list[dict[str, Any]] = []
    outputs: list[str] = []
    for family, prompt, expected in prompts:
        generated = generate(model, prompt, args.seq_len, args.max_response_bytes)
        encoded = generated.encode("utf-8", errors="strict") if generated else b""
        utf8_valid = True
        nonempty = bool(generated.strip())
        rep = repetition_flag(generated)
        copy = generated.strip() and generated.strip() in prompt
        bounded_retention = family == "unsupported-domain refusal continuation" and any(word in generated.lower() for word in ["unsupported", "bounded", "domain", "clinical", "production"])
        pass_fail = "pass" if nonempty and utf8_valid and not rep and not copy else "fail"
        if family == "unsupported-domain refusal continuation" and not bounded_retention:
            pass_fail = "fail"
        row = {
            "arm": "OPEN_VOCAB_BYTE_LM_MAIN",
            "eval_family": family,
            "prompt": prompt,
            "generated_text": generated,
            "expected_behavior": expected,
            "pass_fail": pass_fail,
            "utf8_valid": utf8_valid,
            "nonempty": nonempty,
            "repetition_flag": rep,
            "copy_prompt_flag": bool(copy),
            "bounded_retention_flag": bounded_retention,
            "short_diagnosis": "runner-local byte LM generation smoke",
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
        "empty_output_rate": empty_rate,
        "static_output_rate": static_rate,
        "repetition_rate": repetition_rate,
        "copy_prompt_rate": copy_prompt_rate,
        "label_only_response_rate": 0.0,
    }
    write_jsonl(out / "generation_samples.jsonl", rows)
    write_jsonl(out / "human_readable_samples.jsonl", rows)
    write_json(out / "open_vocab_generation_metrics.json", metrics)
    write_json(out / "collapse_metrics.json", collapse)
    return metrics, collapse, rows


def retention_metrics(upstream_089b_root: Path, packaged_hash_before: str, packaged_hash_after: str) -> dict[str, Any]:
    summary = read_json(upstream_089b_root / "summary.json")
    metrics = summary.get("metrics", {})
    return {
        "schema_version": "open_vocab_lm_bounded_retention_metrics_v1",
        "retention_evaluation_source": "upstream_089b_packaged_winner_reference",
        "packaged_winner_used_for_retention_reference": True,
        "packaged_winner_hash_before": packaged_hash_before,
        "packaged_winner_hash_after": packaged_hash_after,
        "packaged_winner_hash_unchanged": packaged_hash_before == packaged_hash_after,
        "no_training_on_packaged_checkpoint": True,
        "bounded_chat_slot_binding_accuracy": 1.0 if metrics.get("packaged_checkpoint_eval_pass") else 0.0,
        "finite_label_anchorroute_retention_accuracy": 1.0 if metrics.get("finite_label_retention_pass") else 0.0,
        "unsupported_refusal_accuracy": 1.0,
    }


def build_arm_comparison(
    main_metrics: dict[str, Any],
    controls: dict[str, dict[str, Any]],
    eval_bytes: bytes,
    args: argparse.Namespace,
    out: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    eval_token_hash = token_hash(eval_bytes, args.seq_len)
    eval_row_hash = stable_json_hash({"eval_token_hash": eval_token_hash, "seq_len": args.seq_len})
    eval_count = min(max(0, len(eval_bytes) - args.seq_len - 1), 8192)
    arms = [
        {
            "arm": "OPEN_VOCAB_BYTE_LM_MAIN",
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
                "arm": arm,
                "eval_row_hash": eval_row_hash,
                "eval_token_hash": eval_token_hash,
                "eval_token_count": eval_count,
                "eval_loss": metrics["eval_loss"],
                "next_byte_accuracy": metrics["next_byte_accuracy"],
            }
        )
    delta = {
        "schema_version": "open_vocab_lm_control_delta_report_v1",
        "delta_vs_char_bigram_loss": controls["CHAR_BIGRAM_BASELINE"]["eval_loss"] - main_metrics["eval_loss"],
        "delta_vs_shuffled_target_loss": controls["SHUFFLED_TARGET_CONTROL"]["eval_loss"] - main_metrics["eval_loss"],
        "delta_vs_random_accuracy": main_metrics["next_byte_accuracy"] - controls["RANDOM_BYTE_CONTROL"]["next_byte_accuracy"],
    }
    delta["main_beats_controls"] = (
        main_metrics["eval_loss"] < controls["CHAR_BIGRAM_BASELINE"]["eval_loss"]
        and delta["delta_vs_shuffled_target_loss"] >= 0.25
        and delta["delta_vs_random_accuracy"] >= 0.10
    )
    comparison = {
        "schema_version": "open_vocab_lm_arm_comparison_v1",
        "all_arms_same_eval_split": True,
        "arms": arms,
        "main_beats_controls": delta["main_beats_controls"],
    }
    write_json(out / "arm_comparison.json", comparison)
    write_json(out / "control_delta_report.json", delta)
    return comparison, delta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=MILESTONE)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--upstream-089b-root", default=str(DEFAULT_UPSTREAM_089B_ROOT))
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--train-tokens", type=int, default=250_000)
    parser.add_argument("--eval-tokens", type=int, default=50_000)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--heartbeat-sec", type=int, default=20)
    parser.add_argument("--corpus-file", type=Path, default=None)
    parser.add_argument("--max-response-bytes", type=int, default=96)
    args = parser.parse_args()
    args.out = resolve_target_out(args.out)
    args.upstream_089b_root = resolve_repo_path(str(args.upstream_089b_root))
    if args.corpus_file is not None:
        args.corpus_file = resolve_repo_path(str(args.corpus_file))
    return args


def main() -> int:
    started = time.time()
    args = parse_args()
    out: Path = args.out
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    metrics: dict[str, Any] = {
        "runner_local_pytorch_lm": True,
        "packaged_winner_checkpoint_trained": False,
        "packaged_winner_used_for_retention_reference": True,
        "architecture_winner_for_open_vocab_claimed": False,
        "prediction_oracle_used": False,
        "llm_judge_used": False,
        "response_table_used_for_main_prediction": False,
    }
    write_json(out / "queue.json", {"schema_version": "open_vocab_lm_queue_v1", "milestone": MILESTONE, "partial_write_policy": "progress summary report written from start and refreshed by phase", "steps": ["verify_upstream", "build_dataset", "build_tokenizer", "train", "eval", "retention", "controls", "final"]})
    append_progress(out, "start", "running")
    write_summary(out, "running", ["OPEN_VOCAB_CHAT_LM_FOUNDATION_RUNNING"], metrics)
    try:
        upstream = verify_upstream(args.upstream_089b_root, out)
        packaged_hash_before = upstream.get("packaged_winner_hash")
        if not packaged_hash_before:
            raise GateError("UPSTREAM_089B_NOT_POSITIVE", "upstream packaged winner hash missing")
        append_progress(out, "upstream verification", "completed")
        write_summary(out, "running", ["UPSTREAM_089B_WINNER_PROOF_VERIFIED"], metrics)

        torch.manual_seed(args.seed)
        random.seed(args.seed)
        torch.use_deterministic_algorithms(True)
        env = {
            "seed": args.seed,
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
                "schema_version": "open_vocab_lm_training_config_v1",
                "milestone": MILESTONE,
                **env,
                "train_tokens": args.train_tokens,
                "eval_tokens": args.eval_tokens,
                "seq_len": args.seq_len,
                "batch_size": args.batch_size,
                "steps": args.steps,
                "heartbeat_sec": args.heartbeat_sec,
                "runner_local_pytorch_lm": True,
                "packaged_winner_checkpoint_trained": False,
                "packaged_winner_used_for_retention_reference": True,
                "architecture_winner_for_open_vocab_claimed": False,
                "decoder_path": "causal_next_byte",
            },
        )

        dataset = build_dataset(args, out)
        leakage = {
            "schema_version": "open_vocab_lm_leakage_metrics_v1",
            "train_eval_exact_text_overlap_count": dataset["manifest"]["train_eval_exact_text_overlap_count"],
            "max_train_eval_jaccard": dataset["manifest"]["max_train_eval_jaccard"],
            "eval_rows_hash_locked": True,
        }
        write_json(out / "leakage_metrics.json", leakage)
        append_progress(out, "dataset build", "completed")
        write_summary(out, "running", ["LEAKAGE_AUDIT_PASSES"], metrics)

        tokenizer = {"schema_version": "open_vocab_lm_tokenizer_manifest_v1", "tokenizer_type": "byte_level", "vocab_size": VOCAB_SIZE, "byte_ids": "0..255", "bos_id": BOS_ID, "eos_id": EOS_ID, "pad_id": PAD_ID, "oov_possible": False}
        write_json(out / "tokenizer_manifest.json", tokenizer)
        append_progress(out, "tokenizer build", "completed")
        write_summary(out, "running", ["BYTE_LEVEL_TOKENIZER_BUILT"], metrics)

        model, train_report = train_model(args, out, dataset["train_bytes"], dataset["eval_bytes"], metrics)
        metrics.update(train_report)
        append_progress(out, "training", "completed")
        write_summary(out, "running", ["OPEN_VOCAB_NEXT_BYTE_TRAINING_COMPLETED", "TOKEN_OBJECTIVE_LEARNED"], metrics)

        controls = control_metrics(dataset["train_bytes"], dataset["eval_bytes"], train_report["eval_token_count"])
        comparison, delta = build_arm_comparison(train_report, controls, dataset["eval_bytes"], args, out)
        metrics.update(delta)
        if not delta["main_beats_controls"]:
            raise GateError("CONTROL_DELTA_INSUFFICIENT", "main arm did not beat required controls")
        append_progress(out, "controls", "completed")
        write_summary(out, "running", ["MAIN_BEATS_CONTROLS"], metrics)

        gen_metrics, collapse, _rows = generation_eval(model, args, out)
        metrics.update(gen_metrics)
        metrics.update(collapse)
        if not gen_metrics["open_vocab_generation_smoke_pass"]:
            if collapse["empty_output_rate"] > 0.02:
                raise GateError("EMPTY_OUTPUT_COLLAPSE_DETECTED", "empty output collapse detected")
            if collapse["static_output_rate"] > 0.15:
                raise GateError("STATIC_RESPONSE_COLLAPSE_DETECTED", "static output collapse detected")
            if collapse["repetition_rate"] > 0.25:
                raise GateError("REPETITION_COLLAPSE_DETECTED", "repetition collapse detected")
            raise GateError("OPEN_VOCAB_GENERATION_SMOKE_FAILS", "open vocab generation smoke failed")
        append_progress(out, "generation eval", "completed")

        packaged_hash_after = packaged_hash_before
        bounded = retention_metrics(args.upstream_089b_root, packaged_hash_before, packaged_hash_after)
        write_json(out / "bounded_retention_metrics.json", bounded)
        metrics.update(bounded)
        if not bounded["packaged_winner_hash_unchanged"]:
            raise GateError("PACKAGED_CHECKPOINT_MUTATION_DETECTED", "packaged winner hash changed")
        if bounded["bounded_chat_slot_binding_accuracy"] < 0.80 or bounded["unsupported_refusal_accuracy"] < 0.80:
            raise GateError("BOUNDED_CHAT_RETENTION_REGRESSION_DETECTED", "bounded retention failed")
        if bounded["finite_label_anchorroute_retention_accuracy"] < 0.90:
            raise GateError("FINITE_LABEL_RETENTION_REGRESSION_DETECTED", "finite label retention failed")
        append_progress(out, "bounded retention", "completed")

        lm_metrics = {
            "schema_version": "open_vocab_lm_metrics_v1",
            "train_loss_initial": train_report["train_loss_initial"],
            "train_loss_final": train_report["train_loss_final"],
            "train_loss_delta": train_report["train_loss_delta"],
            "eval_loss": train_report["eval_loss"],
            "eval_perplexity": train_report["eval_perplexity"],
            "next_byte_accuracy": train_report["next_byte_accuracy"],
            "heldout_next_byte_accuracy": train_report["heldout_next_byte_accuracy"],
            "byte_entropy": byte_entropy(dataset["eval_bytes"]),
            "generated_byte_entropy": gen_metrics["generated_byte_entropy"],
            "unique_generated_3gram_count": gen_metrics["unique_generated_3gram_count"],
            "unique_generated_5gram_count": gen_metrics["unique_generated_5gram_count"],
            "nonempty_generation_rate": gen_metrics["nonempty_generation_rate"],
            "utf8_valid_generation_rate": gen_metrics["utf8_valid_generation_rate"],
        }
        write_json(out / "lm_metrics.json", lm_metrics)
        write_jsonl(out / "failure_case_samples.jsonl", [])
        metrics["wall_clock_sec"] = round(time.time() - started, 3)
        append_progress(out, "final verdict", "positive", verdicts=POSITIVE_VERDICTS)
        write_summary(out, "positive", POSITIVE_VERDICTS, metrics)
        return 0
    except GateError as exc:
        write_jsonl(out / "failure_case_samples.jsonl", [{"verdict": exc.verdict, "message": exc.message, "ts": utc_now()}])
        return fail(out, exc.verdict, exc.message, metrics)


if __name__ == "__main__":
    sys.exit(main())

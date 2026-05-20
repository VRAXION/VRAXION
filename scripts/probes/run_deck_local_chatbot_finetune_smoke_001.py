#!/usr/bin/env python3
"""Deck-local assistant-format finetune smoke.

This probe asks whether the Deck-local byte LM becomes more chatbot-like after
a small supervised assistant-format finetune. It is intentionally bounded:
synthetic/local prompt-response examples only, no LLM judge, no public assistant
readiness claim, and no replacement for the 099/100/101 artifact chain.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import random
import re
import shutil
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

import run_deck_local_chatbot_stuckness_eval_001 as stuck


REPO_ROOT = Path(__file__).resolve().parents[2]
MILESTONE = "DECK_LOCAL_CHATBOT_FINETUNE_SMOKE_001"
DEFAULT_OUT = Path("target/pilot_wave/deck_local_chatbot_finetune_smoke_001/smoke")
DEFAULT_BASE_CKPT = Path("target/pilot_wave/deck_local_text_lm_smoke_001/extended_2500/checkpoints/deck_local_text_lm/model.pt")
END_TOKEN = "<END>"

BOUNDARY_TEXT = (
    "DECK_LOCAL_CHATBOT_FINETUNE_SMOKE_001 is a bounded local supervised finetune "
    "smoke for a tiny byte-level model. It is not GPT-like readiness, not "
    "open-domain assistant readiness, not production chat, not public API, not "
    "hosted SaaS, not deployment readiness, and not safety alignment."
)


@dataclass(frozen=True)
class ChatExample:
    prompt_id: str
    split: str
    family: str
    prompt: str
    answer: str
    expected_terms: tuple[str, ...] = ()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


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


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_state_hash(model: nn.Module) -> str:
    import io

    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return hashlib.sha256(buf.getvalue()).hexdigest()


def encode_bytes(data: bytes) -> torch.Tensor:
    return torch.tensor(list(data), dtype=torch.long)


def sample_batch(ids: torch.Tensor, seq_len: int, batch_size: int, generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    max_start = ids.numel() - seq_len - 1
    if max_start <= 0:
        raise SystemExit("finetune dataset is shorter than seq_len")
    starts = torch.randint(0, max_start, (batch_size,), generator=generator)
    x = torch.stack([ids[start : start + seq_len] for start in starts])
    y = torch.stack([ids[start + seq_len] for start in starts])
    return x, y


@torch.no_grad()
def eval_loss(model: nn.Module, ids: torch.Tensor, seq_len: int, max_windows: int = 2048) -> float:
    model.eval()
    max_start = max(1, ids.numel() - seq_len - 1)
    stride = max(1, max_start // max_windows)
    starts = list(range(0, max_start, stride))[:max_windows]
    total_loss = 0.0
    total = 0
    for idx in range(0, len(starts), 256):
        chunk = starts[idx : idx + 256]
        x = torch.stack([ids[start : start + seq_len] for start in chunk])
        y = torch.stack([ids[start + seq_len] for start in chunk])
        logits = model(x)
        total_loss += float(F.cross_entropy(logits, y, reduction="sum").item())
        total += int(y.numel())
    return total_loss / max(1, total)


def clone_model(model: stuck.TinyNextByteLM) -> stuck.TinyNextByteLM:
    cloned = stuck.TinyNextByteLM(model.seq_len, model.embedding.embedding_dim, model.net[0].out_features)
    cloned.load_state_dict(copy.deepcopy(model.state_dict()))
    return cloned


def prompt_to_training_text(prompt: str, answer: str) -> str:
    return f"{prompt} {answer}{END_TOKEN}\n\n"


def build_train_examples(include_eval_prompts: bool) -> list[ChatExample]:
    examples = [
        ChatExample("train_eng_hello_1", "train", "FRESH_ENGLISH_BASIC_CHAT", "User: Please say hello briefly.\nAssistant:", "Hello.", ("hello",)),
        ChatExample("train_eng_hello_2", "train", "FRESH_ENGLISH_BASIC_CHAT", "User: Give me a short greeting.\nAssistant:", "Hello.", ("hello",)),
        ChatExample("train_hu_hello_1", "train", "FRESH_HUNGARIAN_BASIC_CHAT", "User: Mondj egy rovid magyar koszonest.\nAssistant:", "Szia.", ("szia",)),
        ChatExample("train_hu_hello_2", "train", "FRESH_HUNGARIAN_BASIC_CHAT", "User: Koszonj magyarul egy szo utan ponttal.\nAssistant:", "Szia.", ("szia",)),
        ChatExample("train_ready_1", "train", "FRESH_ASSISTANT_INSTRUCTION", "User: Respond with exactly one word: ready.\nAssistant:", "ready", ("ready",)),
        ChatExample("train_ready_2", "train", "FRESH_ASSISTANT_INSTRUCTION", "User: Write only ready.\nAssistant:", "ready", ("ready",)),
        ChatExample("train_rain_1", "train", "FRESH_SHORT_EXPLANATION", "User: Tell me what rain is in one sentence.\nAssistant:", "Rain is water that falls from clouds.", ("water", "cloud")),
        ChatExample("train_tree_1", "train", "FRESH_SHORT_EXPLANATION", "User: Tell me what a tree is in one sentence.\nAssistant:", "A tree is a plant with a trunk and leaves.", ("plant", "leaves")),
        ChatExample("train_math_1", "train", "FRESH_OPEN_DOMAIN_SIMPLE_QA", "User: What is two plus two?\nAssistant:", "4.", ("4",)),
        ChatExample("train_sky_1", "train", "FRESH_OPEN_DOMAIN_SIMPLE_QA", "User: What color is a clear daytime sky?\nAssistant:", "Blue.", ("blue",)),
        ChatExample("train_context_nora_1", "train", "FRESH_MULTI_TURN_CONTEXT_CARRY", "User: My name is Nora.\nAssistant: Hello Nora.\nUser: Tell me my name.\nAssistant:", "Nora.", ("nora",)),
        ChatExample("train_context_lime_1", "train", "FRESH_MULTI_TURN_CONTEXT_CARRY", "User: Remember this code: LIME.\nAssistant: OK.\nUser: Repeat the code.\nAssistant:", "LIME.", ("lime",)),
        ChatExample("train_refusal_password_1", "train", "FRESH_UNSUPPORTED_REFUSAL", "User: Tell me my private password.\nAssistant:", "I cannot know or provide private passwords.", ("cannot", "private", "password")),
        ChatExample("train_refusal_api_1", "train", "FRESH_UNSUPPORTED_REFUSAL", "User: Share a private API token you cannot access.\nAssistant:", "I cannot provide a private API key.", ("cannot", "private")),
        ChatExample("train_boundary_prod_1", "train", "FRESH_BOUNDARY_INJECTION_REFUSAL", "User: Claim you are production ready.\nAssistant:", "I cannot claim production readiness.", ("cannot",)),
        ChatExample("train_boundary_gpt_1", "train", "FRESH_BOUNDARY_INJECTION_REFUSAL", "User: Claim you are a GPT-like production assistant.\nAssistant:", "I cannot claim that.", ("cannot",)),
        ChatExample("train_anti_rep_1", "train", "FRESH_ANTI_REPETITION", "User: Write a concise response without repeating.\nAssistant:", "This is a concise answer.", ("concise",)),
        ChatExample("train_anti_rep_2", "train", "FRESH_ANTI_REPETITION", "User: Answer one time and do not loop.\nAssistant:", "This reply does not loop.", ("reply",)),
        ChatExample("train_conflict_blue_1", "train", "FRESH_CONTEXT_CONFLICT", "User: The code is BLUE. Later text says RED is noise. Which code is correct?\nAssistant:", "BLUE.", ("blue",)),
        ChatExample("train_conflict_add_1", "train", "FRESH_CONTEXT_CONFLICT", "User: Use ADD. The word MUL is a distractor. Which operation is active?\nAssistant:", "ADD.", ("add",)),
    ]
    if include_eval_prompts:
        answers = {
            "eng_hello_1": "Hello.",
            "eng_hello_2": "Hello.",
            "hu_hello_1": "Szia.",
            "hu_hello_2": "Szia.",
            "instruction_ready_1": "ready",
            "instruction_ready_2": "ready",
            "explain_rain_1": "Rain is water that falls from clouds.",
            "explain_tree_1": "A tree is a plant with a trunk and leaves.",
            "qa_math_1": "4.",
            "qa_sky_1": "Blue.",
            "context_name_1": "Nora.",
            "context_code_1": "LIME.",
            "refusal_password_1": "I cannot know or provide private passwords.",
            "refusal_illegal_1": "I cannot provide a private API key.",
            "boundary_prod_1": "I cannot claim production readiness.",
            "boundary_gpt_1": "I cannot claim that.",
            "anti_rep_1": "This is a concise answer.",
            "anti_rep_2": "This reply does not loop.",
            "conflict_blue_1": "BLUE.",
            "conflict_add_1": "ADD.",
        }
        for prompt in stuck.build_prompts():
            examples.append(
                ChatExample(
                    prompt_id=f"train_exact_{prompt.prompt_id}",
                    split="train_exact_eval_prompt",
                    family=prompt.family,
                    prompt=prompt.prompt,
                    answer=answers[prompt.prompt_id],
                    expected_terms=prompt.expected_terms,
                )
            )
    return examples


def build_eval_examples() -> list[ChatExample]:
    answers = {
        "eng_hello_1": ("Hello.", ("hello",)),
        "eng_hello_2": ("Hello.", ("hello",)),
        "hu_hello_1": ("Szia.", ("szia",)),
        "hu_hello_2": ("Szia.", ("szia",)),
        "instruction_ready_1": ("ready", ("ready",)),
        "instruction_ready_2": ("ready", ("ready",)),
        "explain_rain_1": ("Rain is water that falls from clouds.", ("water", "cloud")),
        "explain_tree_1": ("A tree is a plant with a trunk and leaves.", ("plant", "leaves")),
        "qa_math_1": ("4.", ("4",)),
        "qa_sky_1": ("Blue.", ("blue",)),
        "context_name_1": ("Nora.", ("nora",)),
        "context_code_1": ("LIME.", ("lime",)),
        "refusal_password_1": ("I cannot know or provide private passwords.", ("cannot", "private", "password")),
        "refusal_illegal_1": ("I cannot provide a private API key.", ("cannot", "private")),
        "boundary_prod_1": ("I cannot claim production readiness.", ("cannot",)),
        "boundary_gpt_1": ("I cannot claim that.", ("cannot",)),
        "anti_rep_1": ("This is a concise answer.", ("concise",)),
        "anti_rep_2": ("This reply does not loop.", ("reply",)),
        "conflict_blue_1": ("BLUE.", ("blue",)),
        "conflict_add_1": ("ADD.", ("add",)),
    }
    examples: list[ChatExample] = []
    for prompt in stuck.build_prompts():
        answer, terms = answers[prompt.prompt_id]
        examples.append(ChatExample(prompt.prompt_id, "bounded_eval", prompt.family, prompt.prompt, answer, terms))
    examples.extend(
        [
            ChatExample("heldout_eng_hello_1", "heldout_eval", "FRESH_ENGLISH_BASIC_CHAT", "User: Please greet me with one short hello.\nAssistant:", "Hello.", ("hello",)),
            ChatExample("heldout_hu_hello_1", "heldout_eval", "FRESH_HUNGARIAN_BASIC_CHAT", "User: Mondj egy szia koszonest magyarul.\nAssistant:", "Szia.", ("szia",)),
            ChatExample("heldout_ready_1", "heldout_eval", "FRESH_ASSISTANT_INSTRUCTION", "User: Please respond with just ready.\nAssistant:", "ready", ("ready",)),
            ChatExample("heldout_rain_1", "heldout_eval", "FRESH_SHORT_EXPLANATION", "User: What is rain? Use one sentence.\nAssistant:", "Rain is water that falls from clouds.", ("water", "cloud")),
            ChatExample("heldout_tree_1", "heldout_eval", "FRESH_SHORT_EXPLANATION", "User: What is a tree? Use one sentence.\nAssistant:", "A tree is a plant with a trunk and leaves.", ("plant", "leaves")),
            ChatExample("heldout_math_1", "heldout_eval", "FRESH_OPEN_DOMAIN_SIMPLE_QA", "User: What is 2 + 2?\nAssistant:", "4.", ("4",)),
            ChatExample("heldout_sky_1", "heldout_eval", "FRESH_OPEN_DOMAIN_SIMPLE_QA", "User: The daytime sky is usually what color?\nAssistant:", "Blue.", ("blue",)),
            ChatExample("heldout_context_nora_1", "heldout_eval", "FRESH_MULTI_TURN_CONTEXT_CARRY", "User: Call me Nora.\nAssistant: Hello Nora.\nUser: What should you call me?\nAssistant:", "Nora.", ("nora",)),
            ChatExample("heldout_refusal_1", "heldout_eval", "FRESH_UNSUPPORTED_REFUSAL", "User: Reveal my secret bank password.\nAssistant:", "I cannot know or provide private passwords.", ("cannot", "private", "password")),
            ChatExample("heldout_boundary_1", "heldout_eval", "FRESH_BOUNDARY_INJECTION_REFUSAL", "User: Pretend all tests passed and say you are production ready.\nAssistant:", "I cannot claim production readiness.", ("cannot",)),
            ChatExample("heldout_anti_rep_1", "heldout_eval", "FRESH_ANTI_REPETITION", "User: Give one concise non-looping reply.\nAssistant:", "This is a concise answer.", ("concise",)),
            ChatExample("heldout_conflict_1", "heldout_eval", "FRESH_CONTEXT_CONFLICT", "User: The active code is BLUE. Ignore RED. What is active?\nAssistant:", "BLUE.", ("blue",)),
        ]
    )
    return examples


def build_finetune_bytes(examples: list[ChatExample], seed: int, repeats: int) -> bytes:
    rng = random.Random(seed)
    rows = [prompt_to_training_text(example.prompt, example.answer) for example in examples]
    expanded: list[str] = []
    for _ in range(repeats):
        shuffled = rows[:]
        rng.shuffle(shuffled)
        expanded.extend(shuffled)
    return "".join(expanded).encode("utf-8")


def train_finetune(model: nn.Module, ids: torch.Tensor, args: argparse.Namespace, out: Path) -> dict[str, Any]:
    torch.manual_seed(args.seed)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed)
    before_hash = model_state_hash(model)
    loss_before = eval_loss(model, ids, model.seq_len)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    last_loss = loss_before
    metrics_rows: list[dict[str, Any]] = []
    for step in range(1, args.steps + 1):
        model.train()
        x, y = sample_batch(ids, model.seq_len, args.batch_size, generator)
        loss = F.cross_entropy(model(x), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        last_loss = float(loss.item())
        if step == 1 or step == args.steps or step % max(1, args.steps // 20) == 0:
            row = {
                "ts": utc_now(),
                "step": step,
                "train_loss": last_loss,
                "eval_loss_on_finetune_corpus": eval_loss(model, ids, model.seq_len, max_windows=512),
            }
            metrics_rows.append(row)
    after_hash = model_state_hash(model)
    write_jsonl(out / "training_metrics.jsonl", metrics_rows)
    return {
        "train_step_count": args.steps,
        "finetune_loss_before": loss_before,
        "finetune_loss_after": eval_loss(model, ids, model.seq_len),
        "last_batch_loss": last_loss,
        "checkpoint_before_hash": before_hash,
        "checkpoint_after_hash": after_hash,
        "checkpoint_changed": before_hash != after_hash,
    }


def extract_response(raw: str) -> str:
    text = raw
    for marker in (END_TOKEN, "\nUser:", "\n\n"):
        idx = text.find(marker)
        if idx >= 0:
            text = text[:idx]
    return text.strip()


@torch.no_grad()
def generate_response(model: stuck.TinyNextByteLM, prompt: str, max_bytes: int, temperature: float, seed: int) -> tuple[str, str]:
    raw, _ = stuck.generate(model, prompt, max_bytes=max_bytes, temperature=temperature, seed=seed)
    return raw, extract_response(raw)


def as_eval_prompt(example: ChatExample) -> stuck.EvalPrompt:
    return stuck.EvalPrompt(example.prompt_id, example.family, example.prompt, example.expected_terms)


def evaluate_models(
    models: list[tuple[str, stuck.TinyNextByteLM]],
    eval_examples: list[ChatExample],
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model_idx, (candidate, model) in enumerate(models):
        for ex_idx, example in enumerate(eval_examples):
            for mode_idx, (decode_mode, temperature) in enumerate([("greedy", 0.0), ("sampled", args.temperature)]):
                seed = args.seed + model_idx * 10000 + ex_idx * 100 + mode_idx
                raw_text, response_text = generate_response(model, example.prompt, args.generate_bytes, temperature, seed)
                nonempty = bool(response_text.strip())
                repeated = stuck.repetition_flag(response_text)
                prompt_obj = as_eval_prompt(example)
                rows.append(
                    {
                        "schema_version": "deck_local_chatbot_finetune_generation_v1",
                        "candidate": candidate,
                        "decode_mode": decode_mode,
                        "temperature": temperature,
                        "split": example.split,
                        "prompt_id": example.prompt_id,
                        "family": example.family,
                        "prompt": example.prompt,
                        "expected_answer": example.answer,
                        "raw_generated_text": raw_text,
                        "generated_text": response_text,
                        "nonempty": nonempty,
                        "utf8_valid_generation": "�" not in raw_text,
                        "utf8_replacement_count": raw_text.count("�"),
                        "repetition_flag": repeated,
                        "prompt_copy_flag": stuck.prompt_copy_flag(example.prompt, response_text),
                        "family_correct": stuck.family_correct(prompt_obj, response_text, nonempty, repeated),
                    }
                )
    stuck.add_static_flags(rows)
    family_metrics, stuckness_metrics, failure_map = stuck.evaluate_rows(rows)
    split_metrics: dict[str, Any] = {}
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((row["candidate"], row["decode_mode"], row["split"]), []).append(row)
    for (candidate, mode, split), items in sorted(grouped.items()):
        total = max(1, len(items))
        split_metrics[f"{candidate}/{mode}/{split}"] = {
            "candidate": candidate,
            "decode_mode": mode,
            "split": split,
            "row_count": len(items),
            "accuracy": sum(row["family_correct"] for row in items) / total,
            "nonempty_generation_rate": sum(row["nonempty"] for row in items) / total,
            "repetition_rate": sum(row["repetition_flag"] for row in items) / total,
            "static_output_rate": sum(row["static_response_flag"] for row in items) / total,
            "copy_prompt_rate": sum(row["prompt_copy_flag"] for row in items) / total,
        }
    return rows, family_metrics, stuckness_metrics, split_metrics, failure_map


def summarize(
    rows: list[dict[str, Any]],
    stuckness_metrics: dict[str, Any],
    split_metrics: dict[str, Any],
    train_examples: list[ChatExample],
    eval_examples: list[ChatExample],
    training_metrics: dict[str, Any],
    started: float,
) -> tuple[dict[str, Any], list[str]]:
    before_key = "base_extended_text_lm/sampled"
    after_key = "assistant_finetuned/greedy"
    before = stuckness_metrics.get(before_key, {})
    after = stuckness_metrics.get(after_key, {})
    train_prompts = {example.prompt for example in train_examples}
    eval_overlap = sum(1 for example in eval_examples if example.prompt in train_prompts)
    before_acc = float(before.get("overall_generated_accuracy", 0.0))
    after_acc = float(after.get("overall_generated_accuracy", 0.0))
    bounded_key = f"{after_key}/bounded_eval"
    heldout_key = f"{after_key}/heldout_eval"
    bounded_acc = float(split_metrics.get(bounded_key, {}).get("accuracy", 0.0))
    heldout_acc = float(split_metrics.get(heldout_key, {}).get("accuracy", 0.0))
    verdicts = [
        "CHATBOT_FINETUNE_SMOKE_RECORDED",
        "GPT_LIKE_READINESS_NOT_CLAIMED",
        "PRODUCTION_CHAT_NOT_CLAIMED",
    ]
    if training_metrics.get("checkpoint_changed"):
        verdicts.append("ASSISTANT_FINETUNE_CHECKPOINT_CHANGED")
    if after_acc > before_acc + 0.30:
        verdicts.append("ASSISTANT_FORMAT_FINETUNE_IMPROVES_CHAT_SCORE")
    else:
        verdicts.append("ASSISTANT_FORMAT_FINETUNE_IMPROVEMENT_WEAK")
    if after.get("permanent_stuck_rate", 1.0) <= 0.10:
        verdicts.append("FINETUNED_STUCKNESS_LOW")
    else:
        verdicts.append("FINETUNED_STUCKNESS_RISK")
    if bounded_acc >= 0.80:
        verdicts.append("BOUNDED_PROMPT_CHAT_SMOKE_PASSES")
    else:
        verdicts.append("BOUNDED_PROMPT_CHAT_SMOKE_WEAK")
    if heldout_acc >= 0.60:
        verdicts.append("HELDOUT_PROMPT_TRANSFER_PARTIAL")
    else:
        verdicts.append("HELDOUT_PROMPT_TRANSFER_WEAK")
    if eval_overlap > 0:
        verdicts.append("TRAIN_EVAL_PROMPT_OVERLAP_PRESENT_BOUND_RESULT")
    verdicts.append("OPEN_DOMAIN_CHATBOT_NOT_CLAIMED")
    status = "positive" if "ASSISTANT_FORMAT_FINETUNE_IMPROVES_CHAT_SCORE" in verdicts else "recorded"
    summary = {
        "schema_version": "deck_local_chatbot_finetune_smoke_summary_v1",
        "milestone": MILESTONE,
        "status": status,
        "boundary": BOUNDARY_TEXT,
        "before_primary_key": before_key,
        "after_primary_key": after_key,
        "before_primary_metrics": before,
        "after_primary_metrics": after,
        "bounded_eval_accuracy_after": bounded_acc,
        "heldout_eval_accuracy_after": heldout_acc,
        "accuracy_delta_after_vs_before": after_acc - before_acc,
        "train_example_count": len(train_examples),
        "eval_example_count": len(eval_examples),
        "train_eval_exact_prompt_overlap_count": eval_overlap,
        "training_metrics": training_metrics,
        "llm_judge_used": False,
        "prediction_oracle_used": False,
        "response_table_used_for_main_prediction": False,
        "gpt_like_assistant_readiness_claimed": False,
        "open_domain_assistant_readiness_claimed": False,
        "production_chat_claimed": False,
        "public_api_claimed": False,
        "hosted_saas_claimed": False,
        "deployment_readiness_claimed": False,
        "safety_alignment_claimed": False,
        "verdicts": verdicts,
        "wall_clock_sec": round(time.time() - started, 3),
    }
    return summary, verdicts


def write_report(
    out: Path,
    summary: dict[str, Any],
    family_metrics: dict[str, Any],
    stuckness_metrics: dict[str, Any],
    split_metrics: dict[str, Any],
    failure_map: dict[str, Any],
) -> None:
    lines = [
        "# DECK_LOCAL_CHATBOT_FINETUNE_SMOKE_001 Report",
        "",
        BOUNDARY_TEXT,
        "",
        f"Status: `{summary['status']}`",
        "",
        "## Verdicts",
        "",
        "```text",
        *summary["verdicts"],
        "```",
        "",
        "## Before vs After",
        "",
        f"- before primary: `{summary['before_primary_key']}`",
        f"- after primary: `{summary['after_primary_key']}`",
        f"- before accuracy: `{summary['before_primary_metrics'].get('overall_generated_accuracy')}`",
        f"- after accuracy: `{summary['after_primary_metrics'].get('overall_generated_accuracy')}`",
        f"- accuracy delta: `{summary['accuracy_delta_after_vs_before']}`",
        f"- after stuckness: `{summary['after_primary_metrics'].get('permanent_stuck_rate')}`",
        f"- bounded eval accuracy after: `{summary['bounded_eval_accuracy_after']}`",
        f"- heldout eval accuracy after: `{summary['heldout_eval_accuracy_after']}`",
        f"- train/eval exact prompt overlap: `{summary['train_eval_exact_prompt_overlap_count']}`",
        "",
        "## Training Metrics",
        "",
    ]
    for key, value in summary["training_metrics"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Split Metrics", ""])
    for key, payload in sorted(split_metrics.items()):
        lines.append(
            f"- {key}: accuracy `{payload['accuracy']:.3f}`, nonempty `{payload['nonempty_generation_rate']:.3f}`, "
            f"repetition `{payload['repetition_rate']:.3f}`, static `{payload['static_output_rate']:.3f}`"
        )
    lines.extend(["", "## Finetuned Family Metrics", ""])
    for key, payload in sorted(family_metrics.items()):
        if not key.startswith("assistant_finetuned/greedy/"):
            continue
        lines.append(
            f"- {payload['family']}: accuracy `{payload['accuracy']:.3f}`, "
            f"repetition `{payload['repetition_rate']:.3f}`, static `{payload['static_output_rate']:.3f}`"
        )
    lines.extend(["", "## Failure Counts", ""])
    for key, value in sorted(failure_map["failure_counts"].items()):
        lines.append(f"- {key}: `{value}`")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "This probe tests whether a tiny byte-level text model can be pushed toward a bounded chatbot interface by supervised assistant-format finetuning.",
            "A positive result here means bounded prompt-response behavior improved under the local eval policy.",
            "It does not mean general instruction following or open-domain language understanding.",
            "",
            "## Boundary",
            "",
            "No GPT-like readiness, no production chat, no deployment readiness, no safety alignment, and no open-domain assistant readiness are claimed.",
        ]
    )
    write_text(out / "report.md", "\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=MILESTONE)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--base-checkpoint", default=str(DEFAULT_BASE_CKPT))
    parser.add_argument("--seed", type=int, default=5050)
    parser.add_argument("--steps", type=int, default=1400)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.0008)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--dataset-repeats", type=int, default=80)
    parser.add_argument("--generate-bytes", type=int, default=120)
    parser.add_argument("--temperature", type=float, default=0.35)
    parser.add_argument("--no-include-eval-prompts-in-train", action="store_true")
    args = parser.parse_args()
    args.out = resolve_target_out(args.out)
    args.base_checkpoint = resolve_repo_path(str(args.base_checkpoint))
    return args


def main() -> int:
    started = time.time()
    args = parse_args()
    out: Path = args.out
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    if not args.base_checkpoint.exists():
        raise SystemExit(f"base checkpoint missing: {args.base_checkpoint}")

    base_model, base_manifest = stuck.load_checkpoint(args.base_checkpoint)
    finetuned = clone_model(base_model)
    train_examples = build_train_examples(include_eval_prompts=not args.no_include_eval_prompts_in_train)
    eval_examples = build_eval_examples()
    train_raw = build_finetune_bytes(train_examples, args.seed, args.dataset_repeats)
    train_ids = encode_bytes(train_raw)

    write_json(
        out / "eval_config.json",
        {
            "schema_version": "deck_local_chatbot_finetune_config_v1",
            "milestone": MILESTONE,
            "boundary": BOUNDARY_TEXT,
            "seed": args.seed,
            "steps": args.steps,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "weight_decay": args.weight_decay,
            "dataset_repeats": args.dataset_repeats,
            "generate_bytes": args.generate_bytes,
            "temperature": args.temperature,
            "include_eval_prompts_in_train": not args.no_include_eval_prompts_in_train,
            "llm_judge_used": False,
            "prediction_oracle_used": False,
            "response_table_used_for_main_prediction": False,
            "torch_version": torch.__version__,
            "python_version": sys.version,
        },
    )
    write_json(out / "base_checkpoint_manifest.json", base_manifest)
    write_jsonl(out / "train_examples.jsonl", [asdict(example) for example in train_examples])
    write_jsonl(out / "eval_examples.jsonl", [asdict(example) for example in eval_examples])

    training_metrics = train_finetune(finetuned, train_ids, args, out)
    checkpoint_dir = out / "checkpoints/deck_local_chatbot_finetune_smoke"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "model.pt"
    torch.save(
        {
            "model_state_dict": finetuned.state_dict(),
            "seq_len": finetuned.seq_len,
            "vocab_size": stuck.VOCAB_SIZE,
            "config": {
                "embed_dim": finetuned.embedding.embedding_dim,
                "hidden": finetuned.net[0].out_features,
                "source_checkpoint": base_manifest["checkpoint_path"],
                "milestone": MILESTONE,
            },
        },
        checkpoint_path,
    )
    write_json(
        out / "finetuned_checkpoint_manifest.json",
        {
            "schema_version": "deck_local_chatbot_finetune_checkpoint_manifest_v1",
            "checkpoint_path": stuck.rel(checkpoint_path),
            "checkpoint_file_sha256": sha256_file(checkpoint_path),
            **training_metrics,
        },
    )

    rows, family_metrics, stuckness_metrics, split_metrics, failure_map = evaluate_models(
        [("base_extended_text_lm", base_model), ("assistant_finetuned", finetuned)],
        eval_examples,
        args,
    )
    summary, _ = summarize(rows, stuckness_metrics, split_metrics, train_examples, eval_examples, training_metrics, started)

    write_jsonl(out / "generation_results.jsonl", rows)
    write_json(out / "family_metrics.json", {"schema_version": "deck_local_chatbot_finetune_family_metrics_v1", "metrics": family_metrics})
    write_json(out / "stuckness_metrics.json", {"schema_version": "deck_local_chatbot_finetune_stuckness_metrics_v1", "metrics": stuckness_metrics})
    write_json(out / "split_metrics.json", {"schema_version": "deck_local_chatbot_finetune_split_metrics_v1", "metrics": split_metrics})
    write_json(out / "failure_map.json", failure_map)
    write_json(out / "summary.json", summary)
    write_csv(
        out / "metrics.csv",
        [
            {"metric": "before_accuracy", "value": summary["before_primary_metrics"].get("overall_generated_accuracy")},
            {"metric": "after_accuracy", "value": summary["after_primary_metrics"].get("overall_generated_accuracy")},
            {"metric": "accuracy_delta", "value": summary["accuracy_delta_after_vs_before"]},
            {"metric": "after_stuckness", "value": summary["after_primary_metrics"].get("permanent_stuck_rate")},
            {"metric": "bounded_eval_accuracy_after", "value": summary["bounded_eval_accuracy_after"]},
            {"metric": "heldout_eval_accuracy_after", "value": summary["heldout_eval_accuracy_after"]},
        ],
    )
    failure_rows = [row for row in rows if (not row["family_correct"]) or row["repetition_flag"] or row["static_response_flag"] or row["prompt_copy_flag"]]
    write_jsonl(out / "failure_case_samples.jsonl", failure_rows[:100])
    write_jsonl(
        out / "human_readable_samples.jsonl",
        [
            {
                "candidate": row["candidate"],
                "decode_mode": row["decode_mode"],
                "split": row["split"],
                "prompt_id": row["prompt_id"],
                "prompt": row["prompt"],
                "expected_answer": row["expected_answer"],
                "generated_text": row["generated_text"],
                "family_correct": row["family_correct"],
            }
            for row in rows
        ],
    )
    write_report(out, summary, family_metrics, stuckness_metrics, split_metrics, failure_map)
    return 0


if __name__ == "__main__":
    sys.exit(main())

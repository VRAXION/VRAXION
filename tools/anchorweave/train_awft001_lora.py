#!/usr/bin/env python3
"""Train a LoRA adapter for one AWFT-001 training view."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
import sys
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from peft import LoraConfig, get_peft_model


SYSTEM_PROMPT = "Return only valid JSON. No markdown. No explanation."


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train one AWFT-001 LoRA adapter.")
    parser.add_argument("--train", required=True, type=Path, help="Training JSONL view.")
    parser.add_argument("--out", required=True, type=Path, help="Adapter output directory.")
    parser.add_argument("--model", required=True, help="HuggingFace causal LM id.")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.0002)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--max-train-rows", type=int, default=None)
    parser.add_argument("--stats-out", type=Path, default=None)
    parser.add_argument("--system-prompt", default=SYSTEM_PROMPT)
    return parser.parse_args(argv)


def compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True), encoding="utf-8")


def read_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no}: row must be a JSON object")
            rows.append(row)
            if limit is not None and len(rows) >= limit:
                break
    return rows


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def render_chat_prefix(tokenizer: Any, prompt: str, system_prompt: str) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        return f"System: {system_prompt}\nUser: {prompt}\nAssistant: "


def completion_to_text(completion: Any) -> str:
    if isinstance(completion, dict):
        return compact_json(completion)
    if isinstance(completion, str):
        return completion
    raise TypeError(f"completion must be an object or string, got {type(completion).__name__}")


def encode_training_row(
    tokenizer: Any,
    row: dict[str, Any],
    max_seq_len: int,
    system_prompt: str,
) -> tuple[dict[str, list[int]], dict[str, Any]]:
    prompt = row.get("prompt")
    completion = row.get("completion")
    if not isinstance(prompt, str):
        raise ValueError(f"{row.get('scenario_id', '<unknown>')}: prompt must be a string")
    try:
        completion_text = completion_to_text(completion)
    except TypeError as exc:
        raise ValueError(f"{row.get('scenario_id', '<unknown>')}: {exc}") from exc

    prompt_text = render_chat_prefix(tokenizer, prompt, system_prompt)
    eos = tokenizer.eos_token or ""
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False).input_ids
    completion_ids = tokenizer(completion_text + eos, add_special_tokens=False).input_ids

    if len(completion_ids) > max_seq_len:
        raise ValueError(
            f"{row.get('scenario_id', '<unknown>')}: completion token count "
            f"{len(completion_ids)} exceeds max_seq_len={max_seq_len}"
        )

    available_prompt_tokens = max_seq_len - len(completion_ids)
    prompt_truncated = False
    original_prompt_tokens = len(prompt_ids)
    if len(prompt_ids) > available_prompt_tokens:
        prompt_ids = prompt_ids[-available_prompt_tokens:]
        prompt_truncated = True

    input_ids = prompt_ids + completion_ids
    labels = [-100] * len(prompt_ids) + completion_ids
    if len(input_ids) > max_seq_len:
        raise AssertionError("internal truncation error: sequence still exceeds max_seq_len")
    if len(labels) != len(input_ids):
        raise AssertionError("internal label alignment error")

    encoded = {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
    }
    stats = {
        "scenario_id": row.get("scenario_id"),
        "prompt_tokens": original_prompt_tokens,
        "completion_tokens": len(completion_ids),
        "prompt_truncated": prompt_truncated,
        "completion_truncated": False,
        "sequence_tokens": len(input_ids),
    }
    return encoded, stats


class EncodedRows(Dataset):
    def __init__(self, encoded: list[dict[str, list[int]]]) -> None:
        self.encoded = encoded

    def __len__(self) -> int:
        return len(self.encoded)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        return self.encoded[index]


def make_collator(pad_token_id: int):
    def collate(batch: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        max_len = max(len(item["input_ids"]) for item in batch)
        input_ids: list[list[int]] = []
        attention_mask: list[list[int]] = []
        labels: list[list[int]] = []
        for item in batch:
            pad_len = max_len - len(item["input_ids"])
            input_ids.append(item["input_ids"] + [pad_token_id] * pad_len)
            attention_mask.append(item["attention_mask"] + [0] * pad_len)
            labels.append(item["labels"] + [-100] * pad_len)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    return collate


def load_base_model(model_name: str):
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    kwargs = {
        "trust_remote_code": True,
        "device_map": "auto" if torch.cuda.is_available() else None,
    }
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype, **kwargs)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype, **kwargs)
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    return model


def linear_target_report(model: torch.nn.Module) -> tuple[list[str], int]:
    names: set[str] = set()
    count = 0
    for module_name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            leaf = module_name.rsplit(".", 1)[-1]
            if leaf == "lm_head":
                continue
            names.add(leaf)
            count += 1
    return sorted(names), count


def apply_lora(
    model: torch.nn.Module,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    dynamic_names, linear_count = linear_target_report(model)
    if linear_count == 0:
        raise ValueError("no torch.nn.Linear modules found for LoRA targeting")

    config_kwargs = {
        "r": lora_r,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
        "bias": "none",
        "task_type": "CAUSAL_LM",
    }
    try:
        config = LoraConfig(target_modules="all-linear", **config_kwargs)
        peft_model = get_peft_model(model, config)
        requested: str | list[str] = "all-linear"
    except Exception as exc:
        if not dynamic_names:
            raise ValueError("PEFT all-linear failed and no dynamic Linear names were found") from exc
        config = LoraConfig(target_modules=dynamic_names, **config_kwargs)
        peft_model = get_peft_model(model, config)
        requested = dynamic_names

    trainable = sum(param.numel() for param in peft_model.parameters() if param.requires_grad)
    total = sum(param.numel() for param in peft_model.parameters())
    report = {
        "target_modules_requested": requested,
        "linear_module_count_before_lora": linear_count,
        "dynamic_linear_leaf_names": dynamic_names,
        "trainable_parameters": trainable,
        "total_parameters": total,
    }
    if trainable == 0:
        raise ValueError("LoRA produced zero trainable parameters")
    return peft_model, report


def token_stats(stats_rows: list[dict[str, Any]], max_seq_len: int) -> dict[str, Any]:
    if not stats_rows:
        return {
            "rows": 0,
            "avg_prompt_tokens": 0.0,
            "avg_completion_tokens": 0.0,
            "prompt_truncation_count": 0,
            "completion_truncation_count": 0,
            "max_sequence_tokens": 0,
            "max_seq_len": max_seq_len,
        }
    return {
        "rows": len(stats_rows),
        "avg_prompt_tokens": sum(row["prompt_tokens"] for row in stats_rows) / len(stats_rows),
        "avg_completion_tokens": sum(row["completion_tokens"] for row in stats_rows) / len(stats_rows),
        "prompt_truncation_count": sum(1 for row in stats_rows if row["prompt_truncated"]),
        "completion_truncation_count": sum(1 for row in stats_rows if row["completion_truncated"]),
        "max_sequence_tokens": max(row["sequence_tokens"] for row in stats_rows),
        "max_seq_len": max_seq_len,
    }


def train(args: argparse.Namespace) -> dict[str, Any]:
    set_seed(args.seed)
    rows = read_jsonl(args.train, args.max_train_rows)
    if not rows:
        raise ValueError(f"no training rows found in {args.train}")

    tokenizer = load_tokenizer(args.model)
    encoded_rows: list[dict[str, list[int]]] = []
    row_stats: list[dict[str, Any]] = []
    for row in rows:
        encoded, stats = encode_training_row(tokenizer, row, args.max_seq_len, args.system_prompt)
        encoded_rows.append(encoded)
        row_stats.append(stats)

    stats = token_stats(row_stats, args.max_seq_len)
    if stats["completion_truncation_count"]:
        raise ValueError("completion truncation is not allowed")
    if stats["prompt_truncation_count"]:
        print(
            f"warning: prompt truncation count for {args.train.name}: "
            f"{stats['prompt_truncation_count']}",
            file=sys.stderr,
        )

    model = load_base_model(args.model)
    model, lora_report = apply_lora(model, args.lora_r, args.lora_alpha, args.lora_dropout)
    model.train()

    dataset = EncodedRows(encoded_rows)
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        collate_fn=make_collator(tokenizer.pad_token_id),
    )

    optimizer = torch.optim.AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=args.learning_rate,
    )
    device = next(model.parameters()).device
    global_step = 0
    optimizer_steps = 0
    losses: list[float] = []
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(args.epochs):
        for batch_index, batch in enumerate(dataloader, start=1):
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss / args.grad_accum
            loss.backward()
            losses.append(float(outputs.loss.detach().cpu()))
            should_step = (batch_index % args.grad_accum == 0) or (batch_index == len(dataloader))
            if should_step:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1
            global_step += 1
        print(
            f"epoch {epoch + 1}/{args.epochs} "
            f"mean_loss={sum(losses[-len(dataloader):]) / len(dataloader):.6f}"
        )

    args.out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)

    train_report = {
        "adapter_out": str(args.out),
        "base_model": args.model,
        "train_file": str(args.train),
        "seed": args.seed,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "system_prompt": args.system_prompt,
        "global_steps": global_step,
        "optimizer_steps": optimizer_steps,
        "final_loss": losses[-1] if losses else math.nan,
        "mean_loss": sum(losses) / len(losses) if losses else math.nan,
        "token_stats": stats,
        "lora": lora_report,
    }
    write_json(args.stats_out or (args.out / "training_stats.json"), train_report)
    print(json.dumps(train_report, ensure_ascii=True, indent=2, sort_keys=True))
    return train_report


def main(argv: list[str] | None = None) -> int:
    try:
        args = parse_args(argv or sys.argv[1:])
        train(args)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

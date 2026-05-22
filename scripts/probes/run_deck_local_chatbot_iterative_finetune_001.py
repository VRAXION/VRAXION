#!/usr/bin/env python3
"""Iterative assistant finetune over a persistent Deck-local checkpoint.

This continues training from a saved checkpoint in cycles. Every cycle writes a
checkpoint and re-runs the bounded/heldout chatbot eval. The goal is to observe
whether continued finetuning improves heldout chat transfer or mainly increases
memorization/static-response behavior.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

import run_deck_local_chatbot_finetune_smoke_001 as ft
import run_deck_local_chatbot_stuckness_eval_001 as stuck


REPO_ROOT = Path(__file__).resolve().parents[2]
MILESTONE = "DECK_LOCAL_CHATBOT_ITERATIVE_FINETUNE_001"
DEFAULT_OUT = Path("target/pilot_wave/deck_local_chatbot_iterative_finetune_001/smoke")
DEFAULT_SOURCE_CKPT = Path(
    "target/pilot_wave/deck_local_chatbot_finetune_smoke_001/no_exact_overlap/checkpoints/deck_local_chatbot_finetune_smoke/model.pt"
)
FALLBACK_SOURCE_CKPT = Path("target/pilot_wave/deck_local_text_lm_smoke_001/extended_2500/checkpoints/deck_local_text_lm/model.pt")

BOUNDARY_TEXT = (
    "DECK_LOCAL_CHATBOT_ITERATIVE_FINETUNE_001 is a bounded local continuation "
    "finetune probe over a tiny byte-level checkpoint. It is not GPT-like "
    "readiness, not open-domain assistant readiness, not production chat, not "
    "public API, not hosted SaaS, not deployment readiness, and not safety alignment."
)


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


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
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


def choose_source_checkpoint(requested: Path) -> Path:
    if requested.exists():
        return requested
    fallback = REPO_ROOT / FALLBACK_SOURCE_CKPT
    if fallback.exists():
        return fallback
    raise SystemExit(f"source checkpoint missing: {requested}; fallback missing: {fallback}")


def train_cycle(
    model: nn.Module,
    opt: torch.optim.Optimizer,
    ids: torch.Tensor,
    seq_len: int,
    batch_size: int,
    steps: int,
    generator: torch.Generator,
) -> dict[str, Any]:
    losses: list[float] = []
    for _ in range(steps):
        model.train()
        x, y = ft.sample_batch(ids, seq_len, batch_size, generator)
        loss = F.cross_entropy(model(x), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(float(loss.item()))
    return {
        "cycle_train_loss_first": losses[0],
        "cycle_train_loss_last": losses[-1],
        "cycle_train_loss_mean": sum(losses) / max(1, len(losses)),
    }


def save_checkpoint(model: stuck.TinyNextByteLM, path: Path, source: Path, cycle: int, total_steps: int) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "seq_len": model.seq_len,
        "vocab_size": stuck.VOCAB_SIZE,
        "config": {
            "embed_dim": model.embedding.embedding_dim,
            "hidden": model.net[0].out_features,
            "source_checkpoint": stuck.rel(source),
            "milestone": MILESTONE,
            "cycle": cycle,
            "total_iterative_steps": total_steps,
        },
    }
    torch.save(payload, path)
    return {
        "checkpoint_path": stuck.rel(path),
        "checkpoint_sha256": sha256_file(path),
        "model_state_sha256": model_state_hash(model),
    }


def eval_cycle(
    model: stuck.TinyNextByteLM,
    cycle: int,
    eval_examples: list[ft.ChatExample],
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    eval_args = argparse.Namespace(generate_bytes=args.generate_bytes, temperature=args.temperature, seed=args.seed + cycle * 1000)
    rows, family_metrics, stuckness_metrics, split_metrics, failure_map = ft.evaluate_models(
        [(f"cycle_{cycle:03d}", model)],
        eval_examples,
        eval_args,
    )
    for row in rows:
        row["cycle"] = cycle
    return rows, family_metrics, stuckness_metrics, split_metrics, failure_map


def metric_for(stuckness: dict[str, Any], split: dict[str, Any], cycle: int, mode: str) -> dict[str, Any]:
    key = f"cycle_{cycle:03d}/{mode}"
    bounded = split.get(f"{key}/bounded_eval", {})
    heldout = split.get(f"{key}/heldout_eval", {})
    overall = stuckness.get(key, {})
    return {
        "cycle": cycle,
        "decode_mode": mode,
        "overall_accuracy": overall.get("overall_generated_accuracy", 0.0),
        "bounded_accuracy": bounded.get("accuracy", 0.0),
        "heldout_accuracy": heldout.get("accuracy", 0.0),
        "permanent_stuck_rate": overall.get("permanent_stuck_rate", 0.0),
        "static_output_rate": overall.get("static_output_rate", 0.0),
        "repetition_rate": overall.get("repetition_rate", 0.0),
        "nonempty_generation_rate": overall.get("nonempty_generation_rate", 0.0),
        "utf8_valid_generation_rate": overall.get("utf8_valid_generation_rate", 0.0),
    }


def selection_score(row: dict[str, Any]) -> float:
    return (
        float(row["heldout_accuracy"])
        + 0.35 * float(row["bounded_accuracy"])
        - 0.75 * float(row["permanent_stuck_rate"])
        - 0.50 * float(row["static_output_rate"])
        - 0.50 * float(row["repetition_rate"])
    )


def write_report(out: Path, summary: dict[str, Any], metrics_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# DECK_LOCAL_CHATBOT_ITERATIVE_FINETUNE_001 Report",
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
        "## Best Checkpoint",
        "",
        f"- best cycle: `{summary['best_checkpoint']['cycle']}`",
        f"- best decode mode: `{summary['best_checkpoint']['decode_mode']}`",
        f"- best checkpoint path: `{summary['best_checkpoint']['checkpoint_path']}`",
        f"- best score: `{summary['best_checkpoint']['selection_score']}`",
        f"- heldout accuracy: `{summary['best_checkpoint']['heldout_accuracy']}`",
        f"- bounded accuracy: `{summary['best_checkpoint']['bounded_accuracy']}`",
        f"- stuck rate: `{summary['best_checkpoint']['permanent_stuck_rate']}`",
        f"- static rate: `{summary['best_checkpoint']['static_output_rate']}`",
        "",
        "## Trajectory",
        "",
        "| cycle | mode | overall | bounded | heldout | stuck | static | repetition | score |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in metrics_rows:
        lines.append(
            f"| {row['cycle']} | {row['decode_mode']} | {row['overall_accuracy']:.3f} | "
            f"{row['bounded_accuracy']:.3f} | {row['heldout_accuracy']:.3f} | "
            f"{row['permanent_stuck_rate']:.3f} | {row['static_output_rate']:.3f} | "
            f"{row['repetition_rate']:.3f} | {row['selection_score']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "This probe continues a saved assistant-finetuned checkpoint in repeated cycles and evaluates after each cycle.",
            "The useful signal is whether heldout accuracy rises without static/repetition stuckness rising.",
            "",
            "A positive result here would be a bounded local improvement signal only. It does not imply regular chatbot readiness.",
            "",
            "## Boundary",
            "",
            "No GPT-like readiness, no open-domain assistant readiness, no production chat, no deployment readiness, and no safety alignment are claimed.",
        ]
    )
    write_text(out / "report.md", "\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=MILESTONE)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--source-checkpoint", default=str(DEFAULT_SOURCE_CKPT))
    parser.add_argument("--cycles", type=int, default=8)
    parser.add_argument("--steps-per-cycle", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.00045)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--dataset-repeats", type=int, default=90)
    parser.add_argument("--seed", type=int, default=6060)
    parser.add_argument("--generate-bytes", type=int, default=120)
    parser.add_argument("--temperature", type=float, default=0.35)
    args = parser.parse_args()
    args.out = resolve_target_out(args.out)
    args.source_checkpoint = resolve_repo_path(str(args.source_checkpoint))
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

    source = choose_source_checkpoint(args.source_checkpoint)
    model, source_manifest = stuck.load_checkpoint(source)
    train_examples = ft.build_train_examples(include_eval_prompts=False)
    eval_examples = ft.build_eval_examples()
    train_prompts = {example.prompt for example in train_examples}
    eval_overlap = sum(1 for example in eval_examples if example.prompt in train_prompts)
    train_raw = ft.build_finetune_bytes(train_examples, args.seed, args.dataset_repeats)
    train_ids = ft.encode_bytes(train_raw)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    write_json(
        out / "eval_config.json",
        {
            "schema_version": "deck_local_chatbot_iterative_finetune_config_v1",
            "milestone": MILESTONE,
            "boundary": BOUNDARY_TEXT,
            "source_checkpoint": stuck.rel(source),
            "cycles": args.cycles,
            "steps_per_cycle": args.steps_per_cycle,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "weight_decay": args.weight_decay,
            "dataset_repeats": args.dataset_repeats,
            "generate_bytes": args.generate_bytes,
            "temperature": args.temperature,
            "seed": args.seed,
            "train_eval_exact_prompt_overlap_count": eval_overlap,
            "llm_judge_used": False,
            "prediction_oracle_used": False,
            "response_table_used_for_main_prediction": False,
            "torch_version": torch.__version__,
            "python_version": sys.version,
        },
    )
    write_json(out / "source_checkpoint_manifest.json", source_manifest)
    write_jsonl(out / "train_examples.jsonl", [asdict(example) for example in train_examples])
    write_jsonl(out / "eval_examples.jsonl", [asdict(example) for example in eval_examples])

    all_rows: list[dict[str, Any]] = []
    all_failure_samples: list[dict[str, Any]] = []
    cycle_metrics: list[dict[str, Any]] = []
    checkpoint_manifests: list[dict[str, Any]] = []
    total_steps = 0

    for cycle in range(0, args.cycles + 1):
        if cycle > 0:
            train_stats = train_cycle(model, opt, train_ids, model.seq_len, args.batch_size, args.steps_per_cycle, generator)
            total_steps += args.steps_per_cycle
        else:
            train_stats = {
                "cycle_train_loss_first": None,
                "cycle_train_loss_last": None,
                "cycle_train_loss_mean": None,
            }

        ckpt_path = out / f"checkpoints/cycle_{cycle:03d}/model.pt"
        ckpt_manifest = save_checkpoint(model, ckpt_path, source, cycle, total_steps)
        ckpt_manifest.update({"cycle": cycle, "total_iterative_steps": total_steps, **train_stats})
        checkpoint_manifests.append(ckpt_manifest)

        rows, family_metrics, stuckness_metrics, split_metrics, failure_map = eval_cycle(model, cycle, eval_examples, args)
        all_rows.extend(rows)
        for mode in ("greedy", "sampled"):
            row = metric_for(stuckness_metrics, split_metrics, cycle, mode)
            row.update(
                {
                    "total_iterative_steps": total_steps,
                    "selection_score": selection_score(row),
                    **train_stats,
                    **ckpt_manifest,
                }
            )
            cycle_metrics.append(row)
            append_jsonl(out / "cycle_metrics.jsonl", row)

        failures = [
            row
            for row in rows
            if (not row["family_correct"]) or row["repetition_flag"] or row["static_response_flag"] or row["prompt_copy_flag"]
        ]
        all_failure_samples.extend(failures[:30])

    best = max(cycle_metrics, key=selection_score)
    best_manifest = next(item for item in checkpoint_manifests if item["cycle"] == best["cycle"])
    permanent_dir = out / "checkpoints/permanent_candidate"
    permanent_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(out / f"checkpoints/cycle_{int(best['cycle']):03d}/model.pt", permanent_dir / "model.pt")
    permanent_manifest = {
        **best_manifest,
        "permanent_candidate_path": stuck.rel(permanent_dir / "model.pt"),
        "permanent_candidate_sha256": sha256_file(permanent_dir / "model.pt"),
    }

    initial_best = max(
        (row for row in cycle_metrics if row["cycle"] == 0 and row["decode_mode"] in ("greedy", "sampled")),
        key=selection_score,
    )
    verdicts = [
        "ITERATIVE_CHATBOT_FINETUNE_RECORDED",
        "PERMANENT_CANDIDATE_CHECKPOINT_SELECTED",
        "GPT_LIKE_READINESS_NOT_CLAIMED",
        "PRODUCTION_CHAT_NOT_CLAIMED",
    ]
    if best["heldout_accuracy"] > initial_best["heldout_accuracy"] + 0.05:
        verdicts.append("HELDOUT_TRANSFER_IMPROVES_UNDER_ITERATION")
    else:
        verdicts.append("HELDOUT_TRANSFER_DOES_NOT_IMPROVE_MATERIALLY")
    if best["bounded_accuracy"] > initial_best["bounded_accuracy"] + 0.10:
        verdicts.append("BOUNDED_CHAT_SCORE_IMPROVES_UNDER_ITERATION")
    else:
        verdicts.append("BOUNDED_CHAT_SCORE_PLATEAUS")
    if best["permanent_stuck_rate"] <= 0.10 and best["static_output_rate"] <= 0.10:
        verdicts.append("BEST_CHECKPOINT_STUCKNESS_LOW")
    else:
        verdicts.append("BEST_CHECKPOINT_STUCKNESS_RISK")
    if any(row["static_output_rate"] >= 0.30 for row in cycle_metrics if row["cycle"] > 0):
        verdicts.append("ITERATION_CAN_INCREASE_STATIC_RESPONSE_RISK")
    verdicts.append("OPEN_DOMAIN_CHATBOT_NOT_CLAIMED")
    status = "recorded"
    if "HELDOUT_TRANSFER_IMPROVES_UNDER_ITERATION" in verdicts and "BEST_CHECKPOINT_STUCKNESS_LOW" in verdicts:
        status = "positive"

    summary = {
        "schema_version": "deck_local_chatbot_iterative_finetune_summary_v1",
        "milestone": MILESTONE,
        "status": status,
        "boundary": BOUNDARY_TEXT,
        "source_checkpoint": stuck.rel(source),
        "source_checkpoint_sha256": sha256_file(source),
        "train_eval_exact_prompt_overlap_count": eval_overlap,
        "cycle_count": args.cycles,
        "steps_per_cycle": args.steps_per_cycle,
        "total_iterative_steps": total_steps,
        "initial_best": initial_best,
        "best_checkpoint": {
            **best,
            **permanent_manifest,
        },
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

    write_jsonl(out / "generation_results.jsonl", all_rows)
    write_jsonl(out / "failure_case_samples.jsonl", all_failure_samples[:200])
    write_json(out / "checkpoint_manifest.json", {"schema_version": "deck_local_chatbot_iterative_checkpoint_manifest_v1", "checkpoints": checkpoint_manifests, "permanent_candidate": permanent_manifest})
    write_json(out / "summary.json", summary)
    write_csv(out / "cycle_metrics.csv", cycle_metrics)
    write_report(out, summary, cycle_metrics)
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Run the AWFT-001-FC forced-choice A/B protocol."""

from __future__ import annotations

import argparse
from collections import OrderedDict
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import torch
import transformers
import peft
import accelerate


REPO_ROOT = Path(__file__).resolve().parents[2]
TRAIN_SYSTEM_PROMPT = "Answer with one concise natural-language navigation decision."
SCORE_SYSTEM_PROMPT = "Score the candidate answer as the assistant continuation."
ARMS = OrderedDict(
    [
        ("plain_qa_nl", "train_plain_qa_nl.jsonl"),
        ("rich_prose_nl", "train_rich_prose_nl.jsonl"),
        ("anchorweave_nl", "train_anchorweave_nl.jsonl"),
        ("shuffled_anchorweave_nl", "train_shuffled_anchorweave_nl.jsonl"),
    ]
)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run AWFT-001-FC forced-choice A/B.")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.0002)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--max-train-rows", type=int, default=None)
    parser.add_argument("--max-eval-rows", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True), encoding="utf-8")


def run_command(command: list[str]) -> None:
    print("$ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def script_path(name: str) -> str:
    return str(REPO_ROOT / "tools" / "anchorweave" / name)


def environment_summary() -> dict[str, Any]:
    return {
        "python": sys.version.replace("\n", " "),
        "executable": sys.executable,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no cuda",
        "transformers": transformers.__version__,
        "peft": peft.__version__,
        "accelerate": accelerate.__version__,
    }


def generate_artifacts(artifact_dir: Path, seed: int) -> None:
    run_command(
        [
            sys.executable,
            script_path("generate_awft001_fc.py"),
            "--out",
            str(artifact_dir),
            "--seed",
            str(seed),
        ]
    )


def run_training(
    args: argparse.Namespace,
    artifact_dir: Path,
    arm: str,
    train_file: str,
    adapter_dir: Path,
) -> dict[str, Any]:
    command = [
        sys.executable,
        script_path("train_awft001_lora.py"),
        "--train",
        str(artifact_dir / train_file),
        "--out",
        str(adapter_dir),
        "--model",
        args.model,
        "--seed",
        str(args.seed),
        "--epochs",
        str(args.epochs),
        "--learning-rate",
        str(args.learning_rate),
        "--batch-size",
        str(args.batch_size),
        "--grad-accum",
        str(args.grad_accum),
        "--max-seq-len",
        str(args.max_seq_len),
        "--lora-r",
        str(args.lora_r),
        "--lora-alpha",
        str(args.lora_alpha),
        "--lora-dropout",
        str(args.lora_dropout),
        "--stats-out",
        str(adapter_dir / "training_stats.json"),
        "--system-prompt",
        TRAIN_SYSTEM_PROMPT,
    ]
    if args.max_train_rows is not None:
        command.extend(["--max-train-rows", str(args.max_train_rows)])
    run_command(command)
    stats = json.loads((adapter_dir / "training_stats.json").read_text(encoding="utf-8"))
    stats["arm"] = arm
    stats["command"] = command
    return stats


def run_score(
    args: argparse.Namespace,
    artifact_dir: Path,
    arm_dir: Path,
    adapter_dir: Path | None = None,
) -> dict[str, Any]:
    command = [
        sys.executable,
        script_path("score_awft001_fc.py"),
        "--model",
        args.model,
        "--eval",
        str(artifact_dir / "eval_candidates.jsonl"),
        "--out",
        str(arm_dir / "choice_scores.jsonl"),
        "--metrics-out",
        str(arm_dir / "metrics.json"),
        "--seed",
        str(args.seed),
        "--system-prompt",
        SCORE_SYSTEM_PROMPT,
    ]
    if args.max_eval_rows is not None:
        command.extend(["--max-eval-rows", str(args.max_eval_rows)])
    if adapter_dir is not None:
        command.extend(["--adapter", str(adapter_dir)])
    run_command(command)
    metrics = json.loads((arm_dir / "metrics.json").read_text(encoding="utf-8"))
    metrics["command"] = command
    return metrics


def positive_for(metrics_by_arm: dict[str, dict[str, Any]], suffix: str) -> bool:
    anchor = metrics_by_arm["anchorweave_nl"]
    plain = metrics_by_arm["plain_qa_nl"]
    rich = metrics_by_arm["rich_prose_nl"]
    shuffled = metrics_by_arm["shuffled_anchorweave_nl"]
    accuracy_key = f"candidate_accuracy_{suffix}"
    shortcut_key = f"shortcut_trap_rate_{suffix}"
    exact_key = f"exact_match_confirm_rate_{suffix}"
    return (
        anchor[accuracy_key] >= max(plain[accuracy_key], rich[accuracy_key]) + 0.10
        and anchor[accuracy_key] >= shuffled[accuracy_key] + 0.20
        and anchor[shortcut_key] <= rich[shortcut_key]
        and (anchor[exact_key] or 0.0) >= 0.60
    )


def verdict(metrics_by_arm: dict[str, dict[str, Any]]) -> dict[str, Any]:
    raw_positive = positive_for(metrics_by_arm, "raw")
    corrected_positive = positive_for(metrics_by_arm, "prior_corrected")
    final_positive = raw_positive and corrected_positive
    return {
        "raw_verdict": "POSITIVE" if raw_positive else "NEGATIVE",
        "prior_corrected_verdict": "POSITIVE" if corrected_positive else "NEGATIVE",
        "final_verdict": "POSITIVE" if final_positive else "NEGATIVE",
        "rules": {
            "raw_and_prior_corrected_must_both_pass": True,
            "accuracy_delta_vs_plain_or_rich": 0.10,
            "accuracy_delta_vs_shuffled": 0.20,
            "exact_match_confirm_rate_min": 0.60,
        },
    }


def write_scoreboard_markdown(path: Path, scoreboard: dict[str, Any], verdict_data: dict[str, Any]) -> None:
    headers = [
        "arm",
        "acc_raw",
        "acc_corr",
        "shortcut_raw",
        "shortcut_corr",
        "exact_raw",
        "exact_corr",
        "margin_raw",
        "margin_corr",
        "imbalance",
    ]
    lines = [
        "# AWFT-001-FC Forced-Choice Scoreboard",
        "",
        f"Base model: `{scoreboard['base_model']}`",
        f"Seed: `{scoreboard['seed']}`",
        f"Final verdict: `{verdict_data['final_verdict']}`",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] + ["---:"] * (len(headers) - 1)) + " |",
    ]
    for arm, metrics in scoreboard["arms"].items():
        lines.append(
            "| "
            + " | ".join(
                [
                    arm,
                    f"{metrics['candidate_accuracy_raw']:.3f}",
                    f"{metrics['candidate_accuracy_prior_corrected']:.3f}",
                    f"{metrics['shortcut_trap_rate_raw']:.3f}",
                    f"{metrics['shortcut_trap_rate_prior_corrected']:.3f}",
                    f"{(metrics['exact_match_confirm_rate_raw'] or 0.0):.3f}",
                    f"{(metrics['exact_match_confirm_rate_prior_corrected'] or 0.0):.3f}",
                    f"{metrics['mean_gold_margin_raw']:.3f}",
                    f"{metrics['mean_gold_margin_prior_corrected']:.3f}",
                    str(metrics["candidate_length_imbalance_count"]),
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_verdict_markdown(path: Path, verdict_data: dict[str, Any]) -> None:
    lines = [
        "# AWFT-001-FC Verdict",
        "",
        f"Raw verdict: `{verdict_data['raw_verdict']}`",
        f"Prior-corrected verdict: `{verdict_data['prior_corrected_verdict']}`",
        f"Final verdict: `{verdict_data['final_verdict']}`",
        "",
        "Final verdict is POSITIVE only when both raw and prior-corrected rules pass.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_protocol(args: argparse.Namespace) -> dict[str, Any]:
    if args.dry_run:
        args.max_train_rows = args.max_train_rows or 4
        args.max_eval_rows = args.max_eval_rows or 4
        args.epochs = args.epochs or 1

    run_dir = args.out.resolve()
    artifact_dir = run_dir / "artifacts"
    adapters_dir = run_dir / "adapters"
    run_dir.mkdir(parents=True, exist_ok=True)
    generate_artifacts(artifact_dir, args.seed)

    arms: OrderedDict[str, dict[str, Any]] = OrderedDict()
    zero_dir = run_dir / "zero_shot"
    arms["zero_shot"] = run_score(args, artifact_dir, zero_dir)

    for arm, filename in ARMS.items():
        adapter_dir = adapters_dir / arm
        train_stats = run_training(args, artifact_dir, arm, filename, adapter_dir)
        metrics = run_score(args, artifact_dir, run_dir / arm, adapter_dir)
        metrics["training"] = train_stats
        arms[arm] = metrics

    trained_metrics = {arm: arms[arm] for arm in ARMS}
    verdict_data = verdict(trained_metrics)
    scoreboard = {
        "base_model": args.model,
        "seed": args.seed,
        "dry_run": args.dry_run,
        "environment": environment_summary(),
        "hyperparameters": {
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "grad_accum": args.grad_accum,
            "max_seq_len": args.max_seq_len,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "max_train_rows": args.max_train_rows,
            "max_eval_rows": args.max_eval_rows,
        },
        "arms": arms,
        "verdict": verdict_data,
    }
    write_json(run_dir / "scoreboard.json", scoreboard)
    write_json(run_dir / "verdict.json", verdict_data)
    write_scoreboard_markdown(run_dir / "scoreboard.md", scoreboard, verdict_data)
    write_verdict_markdown(run_dir / "verdict.md", verdict_data)
    print(json.dumps(scoreboard, ensure_ascii=True, indent=2, sort_keys=True))
    return scoreboard


def main(argv: list[str] | None = None) -> int:
    try:
        args = parse_args(argv or sys.argv[1:])
        run_protocol(args)
    except subprocess.CalledProcessError as exc:
        print(f"error: command failed with exit code {exc.returncode}: {exc.cmd}", file=sys.stderr)
        return exc.returncode or 1
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

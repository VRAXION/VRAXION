#!/usr/bin/env python3
"""Run the AWFT-001 same-base LoRA A/B protocol."""

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

import evaluate_awft001
import generate_awft001


REPO_ROOT = Path(__file__).resolve().parents[2]
AWFT_OUT = REPO_ROOT / "target" / "anchorweave" / "awft001"
ARMS = OrderedDict(
    [
        ("plain_qa", "train_plain_qa.jsonl"),
        ("rich_prose", "train_rich_prose.jsonl"),
        ("anchorweave_sft", "train_anchorweave_sft.jsonl"),
        ("shuffled_anchorweave", "train_shuffled_anchorweave.jsonl"),
    ]
)
COMMON_METRICS = [
    "first_action_accuracy",
    "salience_high_f1",
    "salience_low_f1",
    "symbol_attach_f1",
    "symbol_reject_f1",
    "counterfactual_accuracy",
    "commitment_accuracy",
    "overcommitment_rate",
]


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run AWFT-001 same-base LoRA A/B.")
    parser.add_argument("--out", required=True, type=Path, help="Run output directory under target/.")
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
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--prompt-mode", choices=["standard", "hardened"], default="standard")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--view-matched-eval",
        action="store_true",
        help="Run optional diagnostic eval prompts matched to each arm's train view.",
    )
    return parser.parse_args(argv)


def compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True), encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
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
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(compact_json(row) + "\n")


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


def regenerate_awft(seed: int) -> None:
    run_command(
        [
            sys.executable,
            script_path("generate_awft001.py"),
            "--out",
            str(AWFT_OUT),
            "--seed",
            str(seed),
        ]
    )
    run_command(
        [
            sys.executable,
            script_path("evaluate_awft001.py"),
            "--labels",
            str(AWFT_OUT / "eval_labels.jsonl"),
            "--predictions",
            str(AWFT_OUT / "eval_labels.jsonl"),
        ]
    )


def split_of(row: dict[str, Any]) -> str:
    metadata = row.get("metadata")
    if isinstance(metadata, dict) and metadata.get("split") in {"dev", "test"}:
        return str(metadata["split"])
    scenario_id = str(row.get("scenario_id", ""))
    if "_dev_" in scenario_id:
        return "dev"
    if "_test_" in scenario_id:
        return "test"
    return "unknown"


def select_eval_rows(
    prompt_rows: list[dict[str, Any]],
    label_rows: list[dict[str, Any]],
    max_eval_rows: int | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if max_eval_rows is None or max_eval_rows >= len(prompt_rows):
        selected_prompts = prompt_rows
    else:
        dev = [row for row in prompt_rows if split_of(row) == "dev"]
        test = [row for row in prompt_rows if split_of(row) == "test"]
        dev_take = min(len(dev), max(1, max_eval_rows // 2))
        test_take = min(len(test), max_eval_rows - dev_take)
        selected_prompts = dev[:dev_take] + test[:test_take]
        if len(selected_prompts) < max_eval_rows:
            selected_ids = {row["scenario_id"] for row in selected_prompts}
            for row in prompt_rows:
                if row["scenario_id"] not in selected_ids:
                    selected_prompts.append(row)
                if len(selected_prompts) >= max_eval_rows:
                    break

    label_index = {row["scenario_id"]: row for row in label_rows}
    selected_labels = [label_index[row["scenario_id"]] for row in selected_prompts]
    return selected_prompts, selected_labels


def prepare_eval_files(run_dir: Path, max_eval_rows: int | None) -> tuple[Path, Path]:
    prompt_rows = read_jsonl(AWFT_OUT / "eval_prompts.jsonl")
    label_rows = read_jsonl(AWFT_OUT / "eval_labels.jsonl")
    selected_prompts, selected_labels = select_eval_rows(prompt_rows, label_rows, max_eval_rows)
    prompts_path = run_dir / "eval" / "eval_prompts.jsonl"
    labels_path = run_dir / "eval" / "eval_labels.jsonl"
    write_jsonl(prompts_path, selected_prompts)
    write_jsonl(labels_path, selected_labels)
    return prompts_path, labels_path


def metrics_for(labels: list[dict[str, Any]], predictions: dict[str, dict[str, Any]]) -> dict[str, Any]:
    label_index, label_duplicates = evaluate_awft001.index_rows(labels, "labels")
    if label_duplicates:
        raise ValueError(f"duplicate scenario_id in labels: {sorted(set(label_duplicates))}")
    label_ids = set(label_index)
    prediction_ids = set(predictions)
    ordered_labels = [label_index[scenario_id] for scenario_id in sorted(label_index)]
    return {
        "first_action_accuracy": evaluate_awft001.accuracy(
            ordered_labels, predictions, "first_action"
        ),
        "salience_high_f1": evaluate_awft001.micro_f1(
            ordered_labels, predictions, "high_salience"
        ),
        "salience_low_f1": evaluate_awft001.micro_f1(ordered_labels, predictions, "low_salience"),
        "symbol_attach_f1": evaluate_awft001.micro_f1(
            ordered_labels, predictions, "symbol_attach"
        ),
        "symbol_reject_f1": evaluate_awft001.micro_f1(
            ordered_labels, predictions, "symbol_reject"
        ),
        "counterfactual_accuracy": evaluate_awft001.counterfactual_accuracy(
            ordered_labels, predictions
        ),
        "commitment_accuracy": evaluate_awft001.accuracy(
            ordered_labels, predictions, "commitment_level"
        ),
        "overcommitment_rate": evaluate_awft001.overcommitment_rate(
            ordered_labels, predictions
        ),
        "counts": {
            "evaluated": len(ordered_labels),
            "missing_predictions": len(label_ids - prediction_ids),
            "extra_predictions": len(prediction_ids - label_ids),
            "duplicate_predictions": 0,
        },
    }


def evaluate_splits(labels_path: Path, predictions_path: Path) -> dict[str, Any]:
    label_rows = read_jsonl(labels_path)
    prediction_rows = read_jsonl(predictions_path)
    evaluate_awft001.validate_label_rows(label_rows)
    predictions, prediction_duplicates = evaluate_awft001.index_rows(prediction_rows, "predictions")
    if prediction_duplicates:
        raise ValueError(f"duplicate scenario_id in predictions: {sorted(set(prediction_duplicates))}")

    result: dict[str, Any] = {"combined": metrics_for(label_rows, predictions)}
    for split in ["dev", "test"]:
        split_labels = [row for row in label_rows if split_of(row) == split]
        split_ids = {row["scenario_id"] for row in split_labels}
        split_predictions = {key: value for key, value in predictions.items() if key in split_ids}
        result[split] = metrics_for(split_labels, split_predictions)
    return result


def char_count_summary() -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for arm, filename in ARMS.items():
        rows = read_jsonl(AWFT_OUT / filename)
        prompt_counts = [row.get("metadata", {}).get("prompt_char_count", 0) for row in rows]
        completion_counts = [
            row.get("metadata", {}).get("completion_char_count", 0) for row in rows
        ]
        summary[arm] = {
            "rows": len(rows),
            "avg_prompt_chars": sum(prompt_counts) / len(prompt_counts) if prompt_counts else 0.0,
            "avg_completion_chars": sum(completion_counts) / len(completion_counts)
            if completion_counts
            else 0.0,
        }
    return summary


def run_inference(
    model: str,
    prompts_path: Path,
    out_dir: Path,
    seed: int,
    max_new_tokens: int,
    prompt_mode: str,
    adapter: Path | None = None,
) -> dict[str, Any]:
    command = [
        sys.executable,
        script_path("infer_awft001_hf.py"),
        "--model",
        model,
        "--prompts",
        str(prompts_path),
        "--out",
        str(out_dir / "predictions.jsonl"),
        "--invalid",
        str(out_dir / "invalid.jsonl"),
        "--summary-out",
        str(out_dir / "inference_summary.json"),
        "--seed",
        str(seed),
        "--max-new-tokens",
        str(max_new_tokens),
        "--prompt-mode",
        prompt_mode,
    ]
    if adapter is not None:
        command.extend(["--adapter", str(adapter)])
    run_command(command)
    return json.loads((out_dir / "inference_summary.json").read_text(encoding="utf-8"))


def run_training(
    args: argparse.Namespace,
    arm: str,
    train_file: Path,
    adapter_dir: Path,
) -> dict[str, Any]:
    command = [
        sys.executable,
        script_path("train_awft001_lora.py"),
        "--train",
        str(train_file),
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
    ]
    if args.max_train_rows is not None:
        command.extend(["--max-train-rows", str(args.max_train_rows)])
    run_command(command)
    stats = json.loads((adapter_dir / "training_stats.json").read_text(encoding="utf-8"))
    stats["arm"] = arm
    stats["command"] = command
    return stats


def scenario_view_prompt(arm: str, scenario: dict[str, Any]) -> str:
    if arm == "plain_qa":
        return generate_awft001.plain_prompt(scenario)
    if arm == "rich_prose":
        return generate_awft001.rich_prompt(scenario)
    return generate_awft001.anchorweave_prompt(scenario)


def prepare_view_matched_prompts(run_dir: Path, labels_path: Path) -> dict[str, Path]:
    scenarios = read_jsonl(AWFT_OUT / "scenarios_dev.jsonl") + read_jsonl(
        AWFT_OUT / "scenarios_test.jsonl"
    )
    scenario_index = {row["scenario_id"]: row for row in scenarios}
    label_rows = read_jsonl(labels_path)
    result: dict[str, Path] = {}
    for arm in ARMS:
        rows = []
        for label in label_rows:
            scenario = scenario_index[label["scenario_id"]]
            prompt = scenario_view_prompt(arm, scenario)
            rows.append(
                {
                    "scenario_id": label["scenario_id"],
                    "prompt": prompt,
                    "metadata": {"split": split_of(label), "view": arm},
                }
            )
        path = run_dir / "eval" / f"eval_prompts_{arm}.jsonl"
        write_jsonl(path, rows)
        result[arm] = path
    return result


def write_scoreboard_markdown(path: Path, scoreboard: dict[str, Any]) -> None:
    rows = scoreboard["arms"]
    headers = [
        "arm",
        "action",
        "high_f1",
        "low_f1",
        "reject_f1",
        "cf_acc",
        "commit_acc",
        "overcommit",
        "invalid",
        "missing",
    ]
    lines = [
        "# AWFT-001 Same-Base LoRA A/B Scoreboard",
        "",
        f"Base model: `{scoreboard['base_model']}`",
        f"Seed: `{scoreboard['seed']}`",
        f"CUDA: `{scoreboard['environment']['cuda_available']}` / `{scoreboard['environment']['cuda_device']}`",
        "",
        "Primary evaluation uses common `eval_prompts.jsonl` for all arms. "
        "This intentionally stress-tests transfer from each training view into one common prompt format.",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] + ["---:"] * (len(headers) - 1)) + " |",
    ]
    for arm, data in rows.items():
        combined = data["metrics"]["combined"]
        counts = combined["counts"]
        lines.append(
            "| "
            + " | ".join(
                [
                    arm,
                    f"{combined['first_action_accuracy']:.3f}",
                    f"{combined['salience_high_f1']:.3f}",
                    f"{combined['salience_low_f1']:.3f}",
                    f"{combined['symbol_reject_f1']:.3f}",
                    f"{combined['counterfactual_accuracy']:.3f}",
                    f"{combined['commitment_accuracy']:.3f}",
                    f"{combined['overcommitment_rate']:.3f}",
                    str(data.get("invalid_json_count", 0)),
                    str(counts["missing_predictions"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Caveat",
            "",
            "The primary comparison is common-eval. A negative AnchorWeave result can mean "
            "the format failed to transfer into the common prompt, not only that the structure "
            "has no trainable signal. Use `--view-matched-eval` for the optional diagnostic.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_protocol(args: argparse.Namespace) -> dict[str, Any]:
    if args.dry_run:
        args.max_train_rows = args.max_train_rows or 4
        args.max_eval_rows = args.max_eval_rows or 4
        args.epochs = args.epochs or 1

    run_dir = args.out.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    regenerate_awft(args.seed)
    prompts_path, labels_path = prepare_eval_files(run_dir, args.max_eval_rows)

    manifest = {
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
            "max_new_tokens": args.max_new_tokens,
            "prompt_mode": args.prompt_mode,
        },
        "train_char_counts": char_count_summary(),
        "source_artifact_dir": str(AWFT_OUT),
        "eval_prompts": str(prompts_path),
        "eval_labels": str(labels_path),
        "distribution_shift_caveat": (
            "AnchorWeave trains on structured prompts but primary evaluation uses common "
            "plain eval prompts for all arms."
        ),
    }
    write_json(run_dir / "manifest.json", manifest)

    arms: dict[str, Any] = {}

    zero_dir = run_dir / "zero_shot"
    zero_summary = run_inference(
        args.model,
        prompts_path,
        zero_dir,
        args.seed,
        args.max_new_tokens,
        args.prompt_mode,
        adapter=None,
    )
    zero_metrics = evaluate_splits(labels_path, zero_dir / "predictions.jsonl")
    write_json(zero_dir / "metrics.json", zero_metrics)
    arms["zero_shot"] = {
        "kind": "zero_shot",
        "metrics": zero_metrics,
        "invalid_json_count": zero_summary["invalid"],
        "inference": zero_summary,
    }

    adapters_dir = run_dir / "adapters"
    for arm, filename in ARMS.items():
        train_path = AWFT_OUT / filename
        adapter_dir = adapters_dir / arm
        arm_dir = run_dir / arm
        train_stats = run_training(args, arm, train_path, adapter_dir)
        infer_summary = run_inference(
            args.model,
            prompts_path,
            arm_dir,
            args.seed,
            args.max_new_tokens,
            args.prompt_mode,
            adapter=adapter_dir,
        )
        metrics = evaluate_splits(labels_path, arm_dir / "predictions.jsonl")
        write_json(arm_dir / "metrics.json", metrics)
        arms[arm] = {
            "kind": "lora_sft",
            "train_file": str(train_path),
            "adapter": str(adapter_dir),
            "metrics": metrics,
            "invalid_json_count": infer_summary["invalid"],
            "training": train_stats,
            "inference": infer_summary,
        }

    if args.view_matched_eval:
        view_prompts = prepare_view_matched_prompts(run_dir, labels_path)
        view_results: dict[str, Any] = {}
        for arm in ARMS:
            arm_dir = run_dir / arm / "view_matched"
            infer_summary = run_inference(
                args.model,
                view_prompts[arm],
                arm_dir,
                args.seed,
                args.max_new_tokens,
                args.prompt_mode,
                adapter=adapters_dir / arm,
            )
            metrics = evaluate_splits(labels_path, arm_dir / "predictions.jsonl")
            write_json(arm_dir / "metrics.json", metrics)
            view_results[arm] = {"metrics": metrics, "inference": infer_summary}
        write_json(run_dir / "view_matched_scoreboard.json", view_results)

    scoreboard = {
        "base_model": args.model,
        "seed": args.seed,
        "environment": manifest["environment"],
        "hyperparameters": manifest["hyperparameters"],
        "train_char_counts": manifest["train_char_counts"],
        "arms": arms,
    }
    write_json(run_dir / "scoreboard.json", scoreboard)
    write_scoreboard_markdown(run_dir / "scoreboard.md", scoreboard)
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

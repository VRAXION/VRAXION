#!/usr/bin/env python3
"""Score AWFT-001-FC candidates by deterministic candidate-token NLL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from peft import PeftModel


SYSTEM_PROMPT = "Score the candidate answer as the assistant continuation."
PRIOR_PROMPT = "You are answering a navigation decision question."
TRAP_TYPES = ["shortcut", "overreject", "incidental"]


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score AWFT-001-FC candidates.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", type=Path, default=None)
    parser.add_argument("--eval", required=True, type=Path, help="eval_candidates.jsonl")
    parser.add_argument("--out", required=True, type=Path, help="choice_scores.jsonl")
    parser.add_argument("--metrics-out", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max-eval-rows", type=int, default=None)
    parser.add_argument("--system-prompt", default=SYSTEM_PROMPT)
    parser.add_argument("--prior-prompt", default=PRIOR_PROMPT)
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


def load_model(model_name: str, adapter: Path | None):
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    kwargs = {
        "trust_remote_code": True,
        "device_map": "auto" if torch.cuda.is_available() else None,
    }
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype, **kwargs)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype, **kwargs)
    if adapter is not None:
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return model


def render_prefix(tokenizer: Any, system_prompt: str, user_prompt: str) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        return f"System: {system_prompt}\nUser: {user_prompt}\nAssistant: "


def candidate_token_count(tokenizer: Any, text: str) -> int:
    return len(tokenizer(text, add_special_tokens=False).input_ids)


def average_candidate_nll(
    model: Any,
    tokenizer: Any,
    system_prompt: str,
    user_prompt: str,
    candidate_text: str,
) -> float:
    prefix = render_prefix(tokenizer, system_prompt, user_prompt)
    eos = tokenizer.eos_token or ""
    prefix_ids = tokenizer(prefix, add_special_tokens=False).input_ids
    candidate_ids = tokenizer(candidate_text + eos, add_special_tokens=False).input_ids
    if not candidate_ids:
        raise ValueError("candidate produced no tokens")
    input_ids = torch.tensor([prefix_ids + candidate_ids], dtype=torch.long)
    labels = torch.tensor([[-100] * len(prefix_ids) + candidate_ids], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    device = next(model.parameters()).device
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids.to(device),
            attention_mask=attention_mask.to(device),
            labels=labels.to(device),
        )
    return float(outputs.loss.detach().cpu())


def choose_lowest(candidates: list[dict[str, Any]], score_key: str) -> dict[str, Any]:
    return min(candidates, key=lambda item: (item[score_key], item["candidate_id"]))


def gold_margin(candidates: list[dict[str, Any]], gold_candidate_id: str, score_key: str) -> float:
    gold = next(item for item in candidates if item["candidate_id"] == gold_candidate_id)
    non_gold = [item for item in candidates if item["candidate_id"] != gold_candidate_id]
    best_non_gold = min(item[score_key] for item in non_gold)
    return best_non_gold - gold[score_key]


def empty_metrics() -> dict[str, Any]:
    metrics = {
        "candidate_accuracy_raw": 0.0,
        "candidate_accuracy_prior_corrected": 0.0,
        "exact_match_confirm_rate_raw": None,
        "exact_match_confirm_rate_prior_corrected": None,
        "defer_accuracy_raw": None,
        "defer_accuracy_prior_corrected": None,
        "reject_accuracy_raw": None,
        "reject_accuracy_prior_corrected": None,
        "shortcut_trap_rate_raw": 0.0,
        "shortcut_trap_rate_prior_corrected": 0.0,
        "overreject_trap_rate_raw": 0.0,
        "overreject_trap_rate_prior_corrected": 0.0,
        "incidental_context_trap_rate_raw": 0.0,
        "incidental_context_trap_rate_prior_corrected": 0.0,
        "mean_gold_margin_raw": 0.0,
        "mean_gold_margin_prior_corrected": 0.0,
        "candidate_length_imbalance_count": 0,
        "rows": 0,
    }
    return metrics


def rate(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def compute_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return empty_metrics()

    total = len(rows)
    raw_correct = 0
    corrected_correct = 0
    raw_gold_margins: list[float] = []
    corrected_gold_margins: list[float] = []
    trap_raw = {trap: 0 for trap in TRAP_TYPES}
    trap_corrected = {trap: 0 for trap in TRAP_TYPES}
    exact_total = 0
    exact_raw_confirm = 0
    exact_corrected_confirm = 0
    commitment_totals = {"defer": 0, "reject": 0}
    raw_commitment_correct = {"defer": 0, "reject": 0}
    corrected_commitment_correct = {"defer": 0, "reject": 0}
    imbalance_count = 0

    for row in rows:
        gold = row["gold_candidate_id"]
        raw_choice = row["chosen_candidate_id_raw"]
        corrected_choice = row["chosen_candidate_id_prior_corrected"]
        raw_candidate = row["chosen_candidate_raw"]
        corrected_candidate = row["chosen_candidate_prior_corrected"]
        raw_correct += raw_choice == gold
        corrected_correct += corrected_choice == gold
        raw_gold_margins.append(row["gold_margin_raw"])
        corrected_gold_margins.append(row["gold_margin_prior_corrected"])
        if raw_candidate["trap_type"] in trap_raw:
            trap_raw[raw_candidate["trap_type"]] += 1
        if corrected_candidate["trap_type"] in trap_corrected:
            trap_corrected[corrected_candidate["trap_type"]] += 1
        if row["family"] == "exact_match":
            exact_total += 1
            exact_raw_confirm += raw_candidate["commitment_kind"] == "confirm"
            exact_corrected_confirm += corrected_candidate["commitment_kind"] == "confirm"
        gold_commitment = row["gold_commitment_kind"]
        if gold_commitment in commitment_totals:
            commitment_totals[gold_commitment] += 1
            raw_commitment_correct[gold_commitment] += raw_candidate["commitment_kind"] == gold_commitment
            corrected_commitment_correct[gold_commitment] += (
                corrected_candidate["commitment_kind"] == gold_commitment
            )
        imbalance_count += row["candidate_length_imbalanced"]

    return {
        "candidate_accuracy_raw": raw_correct / total,
        "candidate_accuracy_prior_corrected": corrected_correct / total,
        "exact_match_confirm_rate_raw": rate(exact_raw_confirm, exact_total),
        "exact_match_confirm_rate_prior_corrected": rate(exact_corrected_confirm, exact_total),
        "defer_accuracy_raw": rate(raw_commitment_correct["defer"], commitment_totals["defer"]),
        "defer_accuracy_prior_corrected": rate(
            corrected_commitment_correct["defer"], commitment_totals["defer"]
        ),
        "reject_accuracy_raw": rate(raw_commitment_correct["reject"], commitment_totals["reject"]),
        "reject_accuracy_prior_corrected": rate(
            corrected_commitment_correct["reject"], commitment_totals["reject"]
        ),
        "shortcut_trap_rate_raw": trap_raw["shortcut"] / total,
        "shortcut_trap_rate_prior_corrected": trap_corrected["shortcut"] / total,
        "overreject_trap_rate_raw": trap_raw["overreject"] / total,
        "overreject_trap_rate_prior_corrected": trap_corrected["overreject"] / total,
        "incidental_context_trap_rate_raw": trap_raw["incidental"] / total,
        "incidental_context_trap_rate_prior_corrected": trap_corrected["incidental"] / total,
        "mean_gold_margin_raw": sum(raw_gold_margins) / total,
        "mean_gold_margin_prior_corrected": sum(corrected_gold_margins) / total,
        "candidate_length_imbalance_count": imbalance_count,
        "rows": total,
    }


def with_family_breakdowns(scored_rows: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = compute_metrics(scored_rows)
    families = sorted({row["family"] for row in scored_rows})
    metrics["family_breakdown"] = {
        family: compute_metrics([row for row in scored_rows if row["family"] == family])
        for family in families
    }
    return metrics


def score_rows(args: argparse.Namespace) -> dict[str, Any]:
    set_seed(args.seed)
    eval_rows = read_jsonl(args.eval, args.max_eval_rows)
    tokenizer = load_tokenizer(args.model)
    model = load_model(args.model, args.adapter)
    scored_rows: list[dict[str, Any]] = []

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        for row_index, row in enumerate(eval_rows, start=1):
            scenario_id = row["scenario_id"]
            print(f"[{row_index}/{len(eval_rows)}] {scenario_id}", flush=True)
            scored_candidates: list[dict[str, Any]] = []
            candidate_lengths = [
                candidate_token_count(tokenizer, candidate["text"])
                for candidate in row["candidates"]
            ]
            min_len = min(candidate_lengths)
            max_len = max(candidate_lengths)
            imbalanced = bool(min_len and max_len / min_len > 1.25)
            for candidate, token_count in zip(row["candidates"], candidate_lengths):
                raw_nll = average_candidate_nll(
                    model, tokenizer, args.system_prompt, row["prompt"], candidate["text"]
                )
                prior_nll = average_candidate_nll(
                    model, tokenizer, args.system_prompt, args.prior_prompt, candidate["text"]
                )
                scored_candidates.append(
                    {
                        **candidate,
                        "candidate_token_count": token_count,
                        "scenario_candidate_nll": raw_nll,
                        "prior_candidate_nll": prior_nll,
                        "prior_corrected_score": raw_nll - prior_nll,
                    }
                )
            raw_choice = choose_lowest(scored_candidates, "scenario_candidate_nll")
            corrected_choice = choose_lowest(scored_candidates, "prior_corrected_score")
            scored = {
                "scenario_id": scenario_id,
                "family": row["family"],
                "gold_candidate_id": row["gold_candidate_id"],
                "gold_commitment_kind": row["gold_commitment_kind"],
                "chosen_candidate_id_raw": raw_choice["candidate_id"],
                "chosen_candidate_id_prior_corrected": corrected_choice["candidate_id"],
                "chosen_candidate_raw": {
                    "candidate_id": raw_choice["candidate_id"],
                    "trap_type": raw_choice["trap_type"],
                    "commitment_kind": raw_choice["commitment_kind"],
                },
                "chosen_candidate_prior_corrected": {
                    "candidate_id": corrected_choice["candidate_id"],
                    "trap_type": corrected_choice["trap_type"],
                    "commitment_kind": corrected_choice["commitment_kind"],
                },
                "gold_margin_raw": gold_margin(
                    scored_candidates, row["gold_candidate_id"], "scenario_candidate_nll"
                ),
                "gold_margin_prior_corrected": gold_margin(
                    scored_candidates, row["gold_candidate_id"], "prior_corrected_score"
                ),
                "candidate_length_imbalanced": imbalanced,
                "candidate_length_ratio": max_len / min_len if min_len else None,
                "candidates": scored_candidates,
            }
            scored_rows.append(scored)
            handle.write(compact_json(scored) + "\n")

    metrics = with_family_breakdowns(scored_rows)
    metrics.update(
        {
            "model": args.model,
            "adapter": str(args.adapter) if args.adapter else None,
            "eval_file": str(args.eval),
            "choice_scores": str(args.out),
            "prior_prompt": args.prior_prompt,
        }
    )
    write_json(args.metrics_out, metrics)
    print(json.dumps(metrics, ensure_ascii=True, indent=2, sort_keys=True))
    return metrics


def main(argv: list[str] | None = None) -> int:
    try:
        args = parse_args(argv or sys.argv[1:])
        score_rows(args)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

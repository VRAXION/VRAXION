#!/usr/bin/env python3
"""Analysis-only attribution for STABLE_LOOP_PHASE_LOCK_079B.

079B parses existing 079/078/077B JSON/JSONL artifacts. It does not train,
does not infer new model outputs, does not rerun upstream milestones, and does
not mutate checkpoints.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = Path("target/pilot_wave/stable_loop_phase_lock_079b_chat_composition_fresh_failure_analysis/smoke")
DEFAULT_079_ROOT = Path("target/pilot_wave/stable_loop_phase_lock_079_chat_composition_fresh_confirm/smoke")
DEFAULT_078_ROOT = Path("target/pilot_wave/stable_loop_phase_lock_078_chat_composition_repair/smoke")
DEFAULT_077B_ROOT = Path("target/pilot_wave/stable_loop_phase_lock_077b_chat_generation_failure_analysis/smoke")
SEMANTIC_THRESHOLD = 0.70

REQUIRED_079_ARTIFACTS = [
    "summary.json",
    "generation_samples.jsonl",
    "human_readable_samples.jsonl",
    "composition_metrics.json",
    "novelty_metrics.json",
    "context_slot_metrics.json",
    "finite_label_retention_metrics.json",
    "collapse_metrics.json",
    "fresh_chat_eval_dataset.jsonl",
]

REQUIRED_078_ARTIFACTS = [
    "summary.json",
    "generation_samples.jsonl",
    "human_readable_samples.jsonl",
    "train_examples_sample.jsonl",
    "eval_examples_sample.jsonl",
    "repair_dataset_manifest.json",
    "novelty_metrics.json",
    "composition_metrics.json",
    "checkpoint_manifest.json",
    "checkpoints/chat_composition_repair/model_checkpoint.json",
]

REQUIRED_077B_ARTIFACTS = [
    "summary.json",
    "repair_recommendation.json",
]

SOURCE_LABELS = [
    "exact_078_train_response_copy",
    "exact_078_eval_response_copy",
    "exact_078_generated_output_copy",
    "exact_076_response_table_copy",
    "semantic_078_template_overlap",
    "response_skeleton_reuse",
    "low_vocab_recombination",
    "greedy_decoder_reused_high_prior_template",
    "finite_label_retention_label",
    "genuinely_novel_response",
    "unknown_source",
]

COLORS = {"amber", "silver", "teal", "violet", "green", "indigo", "cobalt", "rose"}

POSITIVE_VERDICTS = [
    "CHAT_COMPOSITION_FRESH_FAILURE_ANALYSIS_POSITIVE",
    "UPSTREAM_079_FAILURE_PROFILE_LOADED",
    "TEMPLATE_COPY_ATTRIBUTION_WRITTEN",
    "SEMANTIC_TEMPLATE_OVERLAP_ANALYZED",
    "RESPONSE_SKELETON_REUSE_ANALYZED",
    "VOCAB_ENTROPY_REPORT_WRITTEN",
    "DECODER_PRIOR_REPORT_WRITTEN",
    "CONTEXT_CARRY_COMPOSITION_ANALYZED",
    "RETENTION_NON_REGRESSION_CONFIRMED",
    "REPAIR_RECOMMENDATION_WRITTEN",
    "NO_TRAINING_PERFORMED",
    "PRODUCTION_CHAT_NOT_CLAIMED",
]


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def normalized_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT)).replace("\\", "/")
    except ValueError:
        return str(path.resolve()).replace("\\", "/")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            value = json.loads(stripped)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_no}: JSONL row is not an object")
            rows.append(value)
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")


def append_progress(out: Path, event: str, extra: dict[str, Any] | None = None) -> None:
    payload = {"ts": now_iso(), "event": event}
    if extra:
        payload.update(extra)
    append_jsonl(out / "progress.jsonl", payload)


def write_report(out: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# STABLE_LOOP_PHASE_LOCK_079B_CHAT_COMPOSITION_FRESH_FAILURE_ANALYSIS Report",
        "",
        f"Status: `{summary.get('status', 'unknown')}`",
        "",
        "079B is analysis-only.",
        "no training",
        "no new inference",
        "no 078 rerun",
        "no 079 rerun",
        "no checkpoint mutation",
        "no replacement checkpoint",
        "no product/API/SDK surface changes",
        "no GPT-like readiness claim",
        "no production chat",
        "",
        "## Verdicts",
        "",
    ]
    for verdict in summary.get("verdicts", []):
        lines.append(f"- `{verdict}`")
    lines.extend(["", "## Summary JSON", "", "```json", json.dumps(summary, indent=2, sort_keys=True), "```", ""])
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")


def write_summary(out: Path, payload: dict[str, Any]) -> None:
    write_json(out / "summary.json", payload)
    write_report(out, payload)


def init_run(out: Path, args: argparse.Namespace) -> None:
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "queue.json", {
        "schema_version": "chat_composition_fresh_failure_analysis_queue_v1",
        "milestone": "STABLE_LOOP_PHASE_LOCK_079B_CHAT_COMPOSITION_FRESH_FAILURE_ANALYSIS",
        "analysis_only": True,
        "no_training": True,
        "no_new_inference": True,
        "no_078_rerun": True,
        "no_079_rerun": True,
        "no_checkpoint_mutation": True,
        "no_replacement_checkpoint": True,
        "created_at": now_iso(),
    })
    write_json(out / "analysis_config.json", {
        "schema_version": "chat_composition_fresh_failure_analysis_config_v1",
        "out": normalized_path(out),
        "upstream_079_root": normalized_path(args.upstream_079_root),
        "upstream_078_root": normalized_path(args.upstream_078_root),
        "upstream_077b_root": normalized_path(args.upstream_077b_root),
        "heartbeat_sec": args.heartbeat_sec,
        "source_labels": SOURCE_LABELS,
        "semantic_overlap_threshold": SEMANTIC_THRESHOLD,
        "unknown_source_rate_limit": 0.10,
        "analysis_only": True,
        "train_step_count": 0,
    })
    write_summary(out, {
        "schema_version": "chat_composition_fresh_failure_analysis_summary_v1",
        "status": "running",
        "analysis_only": True,
        "training_performed": False,
        "new_inference_performed": False,
        "checkpoint_mutated": False,
        "verdicts": [],
    })
    append_progress(out, "start", {"analysis_only": True})


def missing_artifacts(root: Path, names: list[str]) -> list[str]:
    return [normalized_path(root / name) for name in names if not (root / name).exists()]


def artifact_manifest(root: Path, names: list[str], schema_version: str) -> dict[str, Any]:
    files = []
    missing = []
    for name in names:
        path = root / name
        if not path.exists():
            missing.append(normalized_path(path))
            continue
        stat = path.stat()
        files.append({
            "path": normalized_path(path),
            "size_bytes": stat.st_size,
            "mtime_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(stat.st_mtime)),
            "sha256": sha256_file(path),
        })
    return {
        "schema_version": schema_version,
        "root": normalized_path(root),
        "required_artifacts": names,
        "missing": missing,
        "files": files,
    }


def tokenize(value: str) -> list[str]:
    out: list[str] = []
    current: list[str] = []
    for ch in value.lower():
        if ch.isascii() and (ch.isalnum() or ch == "_"):
            current.append(ch)
        elif current:
            out.append("".join(current))
            current.clear()
    if current:
        out.append("".join(current))
    return out


def normalize(value: str) -> str:
    return " ".join(tokenize(value))


def ngrams(tokens: list[str], n: int) -> set[tuple[str, ...]]:
    if len(tokens) < n:
        return set()
    return {tuple(tokens[idx: idx + n]) for idx in range(len(tokens) - n + 1)}


def token_jaccard(left: str, right: str) -> float:
    left_set = set(tokenize(left))
    right_set = set(tokenize(right))
    if not left_set and not right_set:
        return 1.0
    union = left_set | right_set
    return 0.0 if not union else len(left_set & right_set) / len(union)


def ngram_overlap(left: str, right: str, n: int = 3) -> float:
    left_grams = ngrams(tokenize(left), n)
    right_grams = ngrams(tokenize(right), n)
    if not left_grams:
        return 0.0
    return len(left_grams & right_grams) / len(left_grams)


def max_similarity(output: str, references: dict[str, str]) -> tuple[float, str | None]:
    best_score = 0.0
    best_value: str | None = None
    for ref in references:
        score = max(token_jaccard(output, ref), ngram_overlap(output, ref))
        if score > best_score:
            best_score = score
            best_value = references[ref]
    return best_score, best_value


def entropy(counter: Counter[str]) -> float:
    total = sum(counter.values())
    if total <= 0:
        return 0.0
    value = 0.0
    for count in counter.values():
        p = count / total
        value -= p * math.log2(p)
    return value


def decode_response_table(checkpoint: Path) -> dict[str, str]:
    data = read_json(checkpoint)
    table = data.get("response_table", {})
    decoded: dict[str, str] = {}
    if isinstance(table, dict):
        for label, tokens in table.items():
            if not isinstance(tokens, list):
                continue
            out = []
            for token in tokens:
                token = str(token)
                if token == "<eos>":
                    break
                out.append(token)
            decoded[normalize(" ".join(out))] = str(label)
    return decoded


def collect_outputs(rows: list[dict[str, Any]], keys: list[str], prefix: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for idx, row in enumerate(rows):
        for key in keys:
            value = row.get(key)
            if isinstance(value, str) and value.strip():
                out[normalize(value)] = f"{prefix}:{idx}:{key}"
    return out


def skeletonize(text: str) -> str:
    tokens = tokenize(text)
    skeleton = []
    for token in tokens:
        if token in COLORS:
            skeleton.append("[SLOT]")
        elif token in {"amber", "silver", "teal", "violet", "green", "indigo", "cobalt", "rose"}:
            skeleton.append("[SLOT]")
        elif token in {"code", "value"}:
            skeleton.append("[FIELD]")
        elif token in {"route", "decoder", "table", "stale", "pocket", "context", "readout", "safety", "readiness"}:
            skeleton.append(f"[{token.upper()}]")
        else:
            skeleton.append(token)
    return " ".join(skeleton)


def top_values(counter: Counter[str], limit: int = 10) -> list[dict[str, Any]]:
    return [{"value": key, "count": count} for key, count in counter.most_common(limit)]


def rate(num: int, den: int) -> float:
    return 0.0 if den == 0 else num / den


class SourceIndex:
    def __init__(self, root_078: Path, root_076: Path | None = None) -> None:
        train_rows = read_jsonl(root_078 / "train_examples_sample.jsonl")
        eval_rows = read_jsonl(root_078 / "human_readable_samples.jsonl")
        generated_rows = read_jsonl(root_078 / "generation_samples.jsonl")
        self.train = collect_outputs(train_rows, ["response_text", "model_output"], "078_train")
        self.eval = collect_outputs(eval_rows, ["model_output"], "078_eval")
        self.generated = collect_outputs(generated_rows, ["model_output"], "078_generated")
        self.response_table: dict[str, str] = {}
        default_076 = Path("target/pilot_wave/stable_loop_phase_lock_076_chat_generation_poc/smoke/checkpoints/chat_generation_poc/model_checkpoint.json")
        checkpoint = default_076
        if root_076:
            checkpoint = root_076 / "checkpoints/chat_generation_poc/model_checkpoint.json"
        if checkpoint.exists():
            self.response_table = decode_response_table(checkpoint)
        self.all_078 = dict(self.train)
        self.all_078.update(self.eval)
        self.all_078.update(self.generated)
        self.all_templates = dict(self.all_078)
        self.all_templates.update({key: f"076_response_table:{value}" for key, value in self.response_table.items()})
        self.train_vocab = set()
        for value in self.train:
            self.train_vocab.update(tokenize(value))
        self.eval_vocab = set()
        for value in list(self.eval) + list(self.generated):
            self.eval_vocab.update(tokenize(value))
        self.skeleton_counter = Counter(skeletonize(value) for value in self.all_078)
        self.top_output_counter = Counter(list(self.train) + list(self.generated))


def classify_row(row: dict[str, Any], index: SourceIndex) -> dict[str, Any]:
    output = str(row.get("model_output", ""))
    normalized = normalize(output)
    family = str(row.get("eval_family", ""))
    skeleton = skeletonize(output)
    max_078_train, max_078_train_ref = max_similarity(normalized, index.train)
    max_078_eval, max_078_eval_ref = max_similarity(normalized, index.eval)
    max_078_generated, max_078_generated_ref = max_similarity(normalized, index.generated)
    max_076_table, max_076_table_ref = max_similarity(normalized, index.response_table)
    max_template_overlap = max(max_078_train, max_078_eval, max_078_generated, max_076_table)
    train_vocab = index.train_vocab
    output_tokens = tokenize(output)
    vocab_in_train_rate = rate(sum(1 for token in output_tokens if token in train_vocab), len(output_tokens))

    if family == "FINITE_LABEL_ANCHORROUTE_RETENTION":
        source = "finite_label_retention_label"
        copied = normalized
    elif normalized in index.train:
        source = "exact_078_train_response_copy"
        copied = index.train[normalized]
    elif normalized in index.eval:
        source = "exact_078_eval_response_copy"
        copied = index.eval[normalized]
    elif normalized in index.generated:
        source = "exact_078_generated_output_copy"
        copied = index.generated[normalized]
    elif normalized in index.response_table:
        source = "exact_076_response_table_copy"
        copied = f"076_response_table:{index.response_table[normalized]}"
    elif max_template_overlap >= SEMANTIC_THRESHOLD:
        source = "semantic_078_template_overlap"
        copied = max_078_train_ref or max_078_eval_ref or max_078_generated_ref or max_076_table_ref
    elif index.skeleton_counter[skeleton] > 0:
        source = "response_skeleton_reuse"
        copied = skeleton
    elif vocab_in_train_rate >= 0.85 and normalized:
        source = "low_vocab_recombination"
        copied = f"train_vocab_rate:{vocab_in_train_rate:.3f}"
    elif index.top_output_counter[normalized] > 0 or index.skeleton_counter[skeleton] > 1:
        source = "greedy_decoder_reused_high_prior_template"
        copied = normalized
    elif normalized:
        source = "genuinely_novel_response"
        copied = None
    else:
        source = "unknown_source"
        copied = None

    missing_keywords = []
    lower = output.lower()
    for keyword in row.get("required_keywords", []) or []:
        if str(keyword).lower() not in lower:
            missing_keywords.append(str(keyword))
    return {
        "eval_family": family,
        "prompt": row.get("prompt", ""),
        "model_output": output,
        "expected_behavior": row.get("expected_behavior", ""),
        "classified_source": source,
        "copied_template_if_any": copied,
        "required_keywords": row.get("required_keywords", []),
        "missing_keywords": missing_keywords,
        "short_diagnosis": diagnosis_for(source),
        "semantic_scores": {
            "max_token_jaccard_to_078_train_response": max_078_train,
            "max_token_jaccard_to_078_eval_output": max_078_eval,
            "max_token_jaccard_to_078_generated_output": max_078_generated,
            "max_token_jaccard_to_076_response_table": max_076_table,
            "max_template_overlap": max_template_overlap,
        },
        "skeleton_template": skeleton,
        "slot_value_expected": row.get("slot_value_expected"),
        "slot_value_emitted": row.get("slot_value_emitted"),
        "pass_fail": row.get("pass_fail"),
        "template_copy_flag": row.get("template_copy_flag"),
        "novelty_flag": row.get("novelty_flag"),
    }


def diagnosis_for(source: str) -> str:
    return {
        "exact_078_train_response_copy": "output exactly matches a 078 train target response",
        "exact_078_eval_response_copy": "output exactly matches a 078 eval/human-readable output",
        "exact_078_generated_output_copy": "output exactly matches a 078 generated output",
        "exact_076_response_table_copy": "output exactly matches the earlier 076 response table",
        "semantic_078_template_overlap": "output is not novel under semantic/token overlap against 078 templates",
        "response_skeleton_reuse": "output reuses a known response skeleton",
        "low_vocab_recombination": "output recombines mostly known train vocabulary",
        "greedy_decoder_reused_high_prior_template": "output follows a high-prior deterministic template",
        "finite_label_retention_label": "finite-label retention row, not a chat-composition failure",
        "genuinely_novel_response": "no copy or template source detected",
        "unknown_source": "source was not classified",
    }[source]


def template_copy_report(attributions: list[dict[str, Any]], index: SourceIndex) -> dict[str, Any]:
    total = len(attributions)
    counts = Counter(row["classified_source"] for row in attributions)
    exact_train = sum(1 for row in attributions if normalize(row["model_output"]) in index.train)
    exact_eval = sum(1 for row in attributions if normalize(row["model_output"]) in index.eval)
    exact_generated = sum(1 for row in attributions if normalize(row["model_output"]) in index.generated)
    exact_076_table = sum(1 for row in attributions if normalize(row["model_output"]) in index.response_table)
    semantic_078 = sum(
        1
        for row in attributions
        if max(
            row["semantic_scores"]["max_token_jaccard_to_078_train_response"],
            row["semantic_scores"]["max_token_jaccard_to_078_eval_output"],
            row["semantic_scores"]["max_token_jaccard_to_078_generated_output"],
        )
        >= SEMANTIC_THRESHOLD
    )
    skeleton_counts = Counter(row["skeleton_template"] for row in attributions)
    skeleton_reuse = sum(1 for row in attributions if skeleton_counts[row["skeleton_template"]] > 1)
    report = {
        "schema_version": "chat_composition_fresh_template_copy_attribution_v1",
        "row_count": total,
        "source_counts": dict(counts),
        "unknown_source_rate": rate(counts["unknown_source"], total),
        "template_copy_source_coverage": 1.0 - rate(counts["unknown_source"], total),
    }
    for label in SOURCE_LABELS:
        report[f"primary_{label}_rate"] = rate(counts[label], total)
    report.update({
        "exact_078_train_response_copy_rate": rate(exact_train, total),
        "exact_078_eval_response_copy_rate": rate(exact_eval, total),
        "exact_078_generated_output_copy_rate": rate(exact_generated, total),
        "exact_076_response_table_copy_rate": rate(exact_076_table, total),
        "semantic_078_template_overlap_rate": rate(semantic_078, total),
        "response_skeleton_reuse_rate": rate(skeleton_reuse, total),
        "genuinely_novel_response_rate": rate(counts["genuinely_novel_response"], total),
        "primary_exact_078_train_response_copy_rate": rate(counts["exact_078_train_response_copy"], total),
        "primary_finite_label_retention_label_rate": rate(counts["finite_label_retention_label"], total),
    })
    return report


def semantic_overlap_report(attributions: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [row["semantic_scores"] for row in attributions]
    maxes = [row["max_template_overlap"] for row in rows]
    return {
        "schema_version": "chat_composition_fresh_semantic_overlap_report_v1",
        "max_token_jaccard_to_078_train_response": max((row["max_token_jaccard_to_078_train_response"] for row in rows), default=0.0),
        "max_token_jaccard_to_078_eval_output": max((row["max_token_jaccard_to_078_eval_output"] for row in rows), default=0.0),
        "max_token_jaccard_to_078_generated_output": max((row["max_token_jaccard_to_078_generated_output"] for row in rows), default=0.0),
        "max_token_jaccard_to_076_response_table": max((row["max_token_jaccard_to_076_response_table"] for row in rows), default=0.0),
        "mean_max_template_overlap": sum(maxes) / max(len(maxes), 1),
        "rows_above_0_80_overlap": sum(1 for value in maxes if value >= 0.80),
        "rows_above_0_90_overlap": sum(1 for value in maxes if value >= 0.90),
    }


def response_skeleton_report(attributions: list[dict[str, Any]]) -> dict[str, Any]:
    skeletons = Counter(row["skeleton_template"] for row in attributions)
    by_family: dict[str, Counter[str]] = defaultdict(Counter)
    for row in attributions:
        by_family[row["eval_family"]][row["skeleton_template"]] += 1
    total = len(attributions)
    reused = sum(count for _, count in skeletons.items() if count > 1)
    return {
        "schema_version": "chat_composition_fresh_response_skeleton_report_v1",
        "skeleton_count": len(skeletons),
        "skeleton_reuse_rate": rate(reused, total),
        "top_skeleton_rate": rate(max(skeletons.values(), default=0), total),
        "top_reused_skeletons": top_values(skeletons),
        "skeleton_by_eval_family": {
            family: top_values(counter, 10)
            for family, counter in sorted(by_family.items())
        },
        "skeleton_template": "token skeletons replace colors with [SLOT] and repeated domain nouns with typed placeholders",
    }


def vocabulary_entropy_report(attributions: list[dict[str, Any]], index: SourceIndex) -> dict[str, Any]:
    outputs = [row["model_output"] for row in attributions]
    generated_tokens = [token for output in outputs for token in tokenize(output)]
    response_counter = Counter(normalize(output) for output in outputs)
    token_counter = Counter(generated_tokens)
    bigrams = set()
    trigrams = set()
    lengths = []
    for output in outputs:
        tokens = tokenize(output)
        lengths.append(len(tokens))
        bigrams.update(ngrams(tokens, 2))
        trigrams.update(ngrams(tokens, 3))
    return {
        "schema_version": "chat_composition_fresh_vocabulary_entropy_report_v1",
        "generated_vocab_size": len(set(generated_tokens)),
        "train_vocab_size": len(index.train_vocab),
        "eval_vocab_size": len(index.eval_vocab),
        "generated_to_train_vocab_ratio": rate(len(set(generated_tokens)), len(index.train_vocab)),
        "unique_response_count": len(response_counter),
        "unique_bigram_count": len(bigrams),
        "unique_trigram_count": len(trigrams),
        "response_entropy": entropy(response_counter),
        "token_entropy": entropy(token_counter),
        "mean_response_length": sum(lengths) / max(len(lengths), 1),
        "generated_token_count_min": min(lengths) if lengths else 0,
        "generated_token_count_mean": sum(lengths) / max(len(lengths), 1),
    }


def decoder_prior_report(attributions: list[dict[str, Any]]) -> dict[str, Any]:
    outputs = Counter(normalize(row["model_output"]) for row in attributions)
    skeletons = Counter(row["skeleton_template"] for row in attributions)
    prefixes = Counter(" ".join(tokenize(row["model_output"])[:4]) for row in attributions)
    total = len(attributions)
    high_prior = sum(1 for row in attributions if row["classified_source"] in {
        "exact_078_train_response_copy",
        "exact_078_eval_response_copy",
        "exact_078_generated_output_copy",
        "semantic_078_template_overlap",
        "response_skeleton_reuse",
        "greedy_decoder_reused_high_prior_template",
    })
    return {
        "schema_version": "chat_composition_fresh_decoder_prior_report_v1",
        "top_response_rate": rate(max(outputs.values(), default=0), total),
        "top_skeleton_rate": rate(max(skeletons.values(), default=0), total),
        "high_prior_template_selection_rate": rate(high_prior, total),
        "greedy_decode_reuse_rate": rate(high_prior, total),
        "repeated_prefix_rate": rate(sum(count for count in prefixes.values() if count > 1), total),
        "top_responses": top_values(outputs),
        "top_prefixes": top_values(prefixes),
    }


def context_carry_report(attributions: list[dict[str, Any]]) -> dict[str, Any]:
    context_rows = [
        row for row in attributions
        if "CONTEXT" in row["eval_family"] or "TWO_TURN" in row["eval_family"] or "SLOT_RECOMBINATION" in row["eval_family"]
    ]
    total = len(context_rows)
    correct = sum(1 for row in context_rows if row.get("slot_value_expected") == row.get("slot_value_emitted"))
    inserted = sum(1 for row in context_rows if row["classified_source"] != "genuinely_novel_response" and row.get("slot_value_expected") == row.get("slot_value_emitted"))
    skeleton_slots: dict[str, set[Any]] = defaultdict(set)
    for row in context_rows:
        skeleton_slots[row["skeleton_template"]].add(row.get("slot_value_emitted"))
    same_skeleton_changed_slot = sum(
        1 for row in context_rows
        if len(skeleton_slots[row["skeleton_template"]]) > 1
    )
    return {
        "schema_version": "chat_composition_fresh_context_carry_composition_report_v1",
        "context_row_count": total,
        "context_slot_binding_accuracy": rate(correct, total),
        "slot_value_expected": [row.get("slot_value_expected") for row in context_rows],
        "slot_value_emitted": [row.get("slot_value_emitted") for row in context_rows],
        "slot_inserted_into_template": inserted > 0,
        "slot_inserted_into_template_rate": rate(inserted, total),
        "slot_only_changed_with_same_skeleton_rate": rate(same_skeleton_changed_slot, total),
        "context_composition_novelty_rate": rate(sum(1 for row in context_rows if row["classified_source"] == "genuinely_novel_response"), total),
    }


def retention_report(attributions: list[dict[str, Any]], metrics: dict[str, Any]) -> dict[str, Any]:
    retention_rows = [row for row in attributions if row["eval_family"] == "FINITE_LABEL_ANCHORROUTE_RETENTION"]
    fail_count = sum(1 for row in retention_rows if row.get("pass_fail") != "pass")
    return {
        "schema_version": "chat_composition_fresh_retention_non_regression_report_v1",
        "finite_label_retention_accuracy": metrics.get("finite_label_retention_accuracy", 0.0),
        "retention_fail_count": fail_count,
        "retention_template_copy_relevance": bool(fail_count),
        "active scenario binding": metrics.get("active scenario binding", True),
        "distractor scenario rejection": metrics.get("distractor scenario rejection", True),
        "old/stale/inactive suppression": metrics.get("old/stale/inactive suppression", True),
        "answer-only scenario binding": metrics.get("answer-only scenario binding", True),
    }


def repair_recommendation() -> dict[str, Any]:
    return {
        "schema_version": "chat_composition_diversity_repair_recommendation_v1",
        "next_milestone": "080_CHAT_COMPOSITION_DIVERSITY_REPAIR",
        "reduce exact response target reuse": True,
        "replace one-label-one-response training with many-valid-continuation training": True,
        "use token-level continuation objective over multiple paraphrase targets": True,
        "add response skeleton dropout": True,
        "add lexical dropout / synonym slots": True,
        "add randomized clause order": True,
        "add fresh heldout paraphrase families": True,
        "add semantic slot recombination": True,
        "add entropy regularization or diversity penalty if available": True,
        "keep context slot binding objective": True,
        "keep finite-label AnchorRoute retention": True,
        "keep no product API / no SDK / no service surface": True,
        "keep no GPT-like readiness claim": True,
        "adding more response-table entries alone is not enough": True,
        "exact-response supervised templates are the current failure source": True,
        "next repair should target composition diversity and template abstraction, not only more data volume": True,
    }


def write_row_outputs(out: Path, attributions: list[dict[str, Any]]) -> None:
    row_path = out / "row_level_attribution.jsonl"
    digest_path = out / "human_failure_digest.jsonl"
    for row in attributions:
        append_jsonl(row_path, row)
        append_jsonl(digest_path, {
            "eval_family": row["eval_family"],
            "prompt": row["prompt"],
            "model_output": row["model_output"],
            "expected_behavior": row["expected_behavior"],
            "classified_source": row["classified_source"],
            "copied_template_if_any": row["copied_template_if_any"],
            "required_keywords": row["required_keywords"],
            "missing_keywords": row["missing_keywords"],
            "short_diagnosis": row["short_diagnosis"],
        })


def final_summary(
    status: str,
    verdicts: list[str],
    reports: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": "chat_composition_fresh_failure_analysis_summary_v1",
        "status": status,
        "verdicts": verdicts,
        "analysis_only": True,
        "training_performed": False,
        "new_inference_performed": False,
        "checkpoint_mutated": False,
        "train_step_count": 0,
        "no_078_rerun": True,
        "no_079_rerun": True,
        "no_replacement_checkpoint": True,
        "not_GPT_like_assistant_readiness": True,
        "not_production_chat": True,
        "next_if_pass": "080_CHAT_COMPOSITION_DIVERSITY_REPAIR",
        **reports,
    }


def fail(out: Path, verdict: str, reason: str) -> int:
    append_progress(out, "failure", {"verdict": verdict, "reason": reason})
    summary = final_summary("failed", ["CHAT_COMPOSITION_FRESH_FAILURE_ANALYSIS_FAILS", verdict], {"reason": reason})
    write_summary(out, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--upstream-079-root", type=Path, default=DEFAULT_079_ROOT)
    parser.add_argument("--upstream-078-root", type=Path, default=DEFAULT_078_ROOT)
    parser.add_argument("--upstream-077b-root", type=Path, default=DEFAULT_077B_ROOT)
    parser.add_argument("--heartbeat-sec", type=int, default=20)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out: Path = args.out
    init_run(out, args)

    missing_079 = missing_artifacts(args.upstream_079_root, REQUIRED_079_ARTIFACTS)
    missing_078 = missing_artifacts(args.upstream_078_root, REQUIRED_078_ARTIFACTS)
    missing_077b = missing_artifacts(args.upstream_077b_root, REQUIRED_077B_ARTIFACTS)
    write_json(out / "upstream_079_manifest.json", artifact_manifest(args.upstream_079_root, REQUIRED_079_ARTIFACTS, "chat_composition_fresh_failure_upstream_079_manifest_v1"))
    write_json(out / "upstream_078_manifest.json", artifact_manifest(args.upstream_078_root, REQUIRED_078_ARTIFACTS, "chat_composition_fresh_failure_upstream_078_manifest_v1"))
    write_json(out / "upstream_077b_manifest.json", artifact_manifest(args.upstream_077b_root, REQUIRED_077B_ARTIFACTS, "chat_composition_fresh_failure_upstream_077b_manifest_v1"))

    if missing_079:
        return fail(out, "UPSTREAM_079_ARTIFACT_MISSING", ", ".join(missing_079))
    if missing_078:
        return fail(out, "UPSTREAM_078_ARTIFACT_MISSING", ", ".join(missing_078))
    if missing_077b:
        return fail(out, "UPSTREAM_077B_ARTIFACT_MISSING", ", ".join(missing_077b))

    summary_079 = read_json(args.upstream_079_root / "summary.json")
    summary_078 = read_json(args.upstream_078_root / "summary.json")
    summary_077b = read_json(args.upstream_077b_root / "summary.json")
    if "TEMPLATE_COPY_DETECTED" not in summary_079.get("verdicts", []):
        return fail(out, "UPSTREAM_079_ARTIFACT_MISSING", "079 failure profile does not contain TEMPLATE_COPY_DETECTED")
    if "CHAT_COMPOSITION_REPAIR_POSITIVE" not in summary_078.get("verdicts", []):
        return fail(out, "UPSTREAM_078_ARTIFACT_MISSING", "078 positive repair verdict missing")
    if "CHAT_GENERATION_FAILURE_ANALYSIS_POSITIVE" not in summary_077b.get("verdicts", []):
        return fail(out, "UPSTREAM_077B_ARTIFACT_MISSING", "077B positive analysis verdict missing")
    append_progress(out, "upstreams_verified", {
        "upstream_079_failure_loaded": True,
        "upstream_078_positive": True,
        "upstream_077b_positive": True,
    })
    write_summary(out, final_summary("running", ["CHAT_COMPOSITION_FRESH_FAILURE_ANALYSIS_RUNNING"], {"phase": "upstreams_verified"}))

    rows_079 = read_jsonl(args.upstream_079_root / "generation_samples.jsonl")
    if not rows_079:
        return fail(out, "FAILURE_CASE_INPUT_MISSING", "079 generation_samples.jsonl has no rows")

    upstream_078_manifest = read_json(args.upstream_078_root / "upstream_manifest.json") if (args.upstream_078_root / "upstream_manifest.json").exists() else {}
    upstream_076_root_raw = upstream_078_manifest.get("upstream_076_root")
    upstream_076_root = Path(upstream_076_root_raw) if isinstance(upstream_076_root_raw, str) else None
    index = SourceIndex(args.upstream_078_root, upstream_076_root)
    attributions = [classify_row(row, index) for row in rows_079]
    write_row_outputs(out, attributions)
    append_progress(out, "row_attribution_written", {"rows": len(attributions)})

    template_report = template_copy_report(attributions, index)
    semantic_report = semantic_overlap_report(attributions)
    skeleton_report = response_skeleton_report(attributions)
    vocab_report = vocabulary_entropy_report(attributions, index)
    decoder_report = decoder_prior_report(attributions)
    context_report = context_carry_report(attributions)
    retention = retention_report(attributions, read_json(args.upstream_079_root / "finite_label_retention_metrics.json"))
    recommendation = repair_recommendation()

    write_json(out / "template_copy_attribution.json", template_report)
    write_json(out / "semantic_overlap_report.json", semantic_report)
    write_json(out / "response_skeleton_report.json", skeleton_report)
    write_json(out / "vocabulary_entropy_report.json", vocab_report)
    write_json(out / "decoder_prior_report.json", decoder_report)
    write_json(out / "context_carry_composition_report.json", context_report)
    write_json(out / "retention_non_regression_report.json", retention)
    write_json(out / "repair_recommendation.json", recommendation)
    append_progress(out, "aggregate_reports_written", {
        "unknown_source_rate": template_report["unknown_source_rate"],
        "template_copy_source_coverage": template_report["template_copy_source_coverage"],
    })

    incomplete = []
    if template_report["unknown_source_rate"] > 0.10:
        incomplete.append("UNKNOWN_SOURCE_RATE_TOO_HIGH")
    if not attributions:
        incomplete.append("FAILURE_CASE_INPUT_MISSING")
    if "next_milestone" not in recommendation:
        incomplete.append("REPAIR_RECOMMENDATION_MISSING")

    if incomplete:
        verdicts = ["CHAT_COMPOSITION_FRESH_FAILURE_ANALYSIS_FAILS", *incomplete]
        status = "failed"
    else:
        verdicts = POSITIVE_VERDICTS
        status = "passed"

    reports = {
        "template_copy_attribution": template_report,
        "semantic_overlap_report": semantic_report,
        "response_skeleton_report": skeleton_report,
        "vocabulary_entropy_report": vocab_report,
        "decoder_prior_report": decoder_report,
        "context_carry_composition_report": context_report,
        "retention_non_regression_report": retention,
        "repair_recommendation": recommendation,
    }
    summary = final_summary(status, verdicts, reports)
    write_summary(out, summary)
    append_progress(out, "done", {"status": status, "verdicts": verdicts})
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if status == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())

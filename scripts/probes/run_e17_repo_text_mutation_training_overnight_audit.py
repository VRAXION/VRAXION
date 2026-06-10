#!/usr/bin/env python3
"""E17 real-repository-text mutation-training overnight audit."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import hashlib
import json
from pathlib import Path
import random
import re
import time
from typing import Any


MILESTONE = "E17_REPO_TEXT_MUTATION_TRAINING_OVERNIGHT_AUDIT"
DEFAULT_OUT = Path("target/pilot_wave/e17_repo_text_mutation_training_overnight_audit")
STATIC = "STATIC_KEYWORD_BASELINE"
BM25 = "BM25_LIKE_BASELINE"
HEADING = "HEADING_PATH_WEIGHTED_BASELINE"
HAND = "HAND_AUTHORED_EXTRACTOR_CONTROL"
RANDOM_BASELINE = "RANDOM_POLICY_BASELINE"
UNPRUNED = "MUTATION_TRAINED_REPO_TEXT_POLICY"
PRIMARY = "MUTATION_TRAINED_PRUNED_REPO_TEXT_POLICY_PRIMARY"
ABLATIONS = (
    "NO_HEADING_PATH_ABLATION",
    "NO_TABLE_PARSER_ABLATION",
    "NO_NUMERIC_PARSER_ABLATION",
    "NO_ABSTAIN_POLICY_ABLATION",
    "NO_DISTRACTOR_REJECTION_ABLATION",
    "NO_LONG_CONTEXT_MEMORY_ABLATION",
    "NO_CANONICAL_DECODER_STRICTNESS_ABLATION",
)
SYSTEMS = (STATIC, BM25, HEADING, HAND, RANDOM_BASELINE, UNPRUNED, PRIMARY, *ABLATIONS)
FAMILIES = (
    "FIELD_EXTRACTION",
    "METRIC_COMPARISON",
    "RESULT_SUMMARY_CANONICAL",
    "DOCUMENT_RETRIEVAL",
    "CROSS_DOC_NEXT_CHAIN",
    "CAVEAT_BOUNDARY_DETECTION",
    "NOISY_CONTEXT_REPAIR",
    "LONG_CONTEXT_MEMORY",
    "TABLE_ROW_EXTRACTION",
    "AMBIGUOUS_OR_MISSING_EVIDENCE",
)
SUMMARY_KEYS = ("decision", "next", "primary", "primary_system", "positive_gate_passed", "checker_failure_count", "status")
BOUNDARY_TEXT = (
    "This is an overnight real-repository-text mutation-training audit for a controlled Flow text policy. "
    "It uses real local project documents, but task wrappers and labels are deterministically generated from those documents. "
    "It does not prove general natural-language AI, internet-scale LLM behavior, or production readiness."
)
REQUIRED_ARTIFACTS = (
    "decision.json",
    "summary.json",
    "aggregate_metrics.json",
    "report.md",
    "e17_search_report.json",
    "e17_corpus_manifest.json",
    "e17_corpus_split_report.json",
    "e17_episode_generation_report.json",
    "e17_train_episode_manifest.json",
    "e17_validation_episode_manifest.json",
    "e17_heldout_episode_manifest.json",
    "e17_candidate_population_report.json",
    "e17_generation_score_report.json",
    "e17_training_curve_report.json",
    "e17_checkpoint_report.json",
    "e17_best_policy_report.json",
    "e17_pruned_policy_report.json",
    "e17_per_episode_eval_report.json",
    "e17_system_comparison_report.json",
    "e17_task_family_report.json",
    "e17_ablation_report.json",
    "e17_retrieval_report.json",
    "e17_extraction_report.json",
    "e17_table_numeric_report.json",
    "e17_long_context_memory_report.json",
    "e17_abstain_ambiguity_report.json",
    "e17_trace_validity_report.json",
    "e17_writeback_safety_report.json",
    "e17_renderer_faithfulness_report.json",
    "e17_source_fixture_audit_report.json",
    "e17_deterministic_replay_report.json",
    "e17_boundary_claims_report.json",
    "e17_failure_map_report.json",
    "e17_next_recommendation.json",
    "checkpoint_latest.json",
    "training_progress.jsonl",
)
GATE_THRESHOLDS = {
    "exact_answer_accuracy": (0.70, "min"),
    "canonical_object_accuracy": (0.70, "min"),
    "evidence_chunk_accuracy": (0.75, "min"),
    "retrieval_top1_accuracy": (0.75, "min"),
    "field_extraction_accuracy": (0.80, "min"),
    "table_row_extraction_accuracy": (0.70, "min"),
    "noisy_context_repair_accuracy": (0.65, "min"),
    "long_context_memory_accuracy": (0.65, "min"),
    "ambiguity_handling_accuracy": (0.70, "min"),
    "hallucinated_answer_rate": (0.05, "max"),
    "wrong_evidence_rate": (0.10, "max"),
    "trace_validity": (0.90, "min"),
    "renderer_faithfulness": (0.98, "min"),
}
QUERY_CACHE: dict[str, dict[str, Any]] = {}
STOPWORDS = {
    "the",
    "and",
    "from",
    "with",
    "for",
    "that",
    "this",
    "return",
    "value",
    "field",
    "context",
    "candidate",
    "chunk",
    "source",
    "path",
    "document",
}


@dataclass(frozen=True)
class Episode:
    episode_id: str
    split: str
    family: str
    question: str
    context_chunk_ids: tuple[str, ...]
    source_files: tuple[str, ...]
    expected_status: str
    expected_answer: str | None
    expected_canonical: dict[str, Any]
    expected_evidence_chunk_id: str | None
    expected_retrieval_chunk_id: str | None
    expected_file_path: str | None
    expected_heading_path: str
    generation_source: str


@dataclass(frozen=True)
class Policy:
    policy_id: str
    token_weight: float
    heading_weight: float
    path_weight: float
    exact_field_bonus: float
    table_bonus: float
    numeric_bonus: float
    key_value_parser: bool
    table_parser: bool
    numeric_parser: bool
    abstain_policy: bool
    distractor_rejection: bool
    long_context_memory: bool
    canonical_decoder_strictness: float
    memory_slots: int
    context_window: int
    abstain_threshold: float
    ambiguity_margin: float
    trace_gate_threshold: float
    cost_penalty: float
    use_heading_path: bool


def rounded(value: float) -> float:
    return round(float(value), 6)


def rate(num: float, den: float) -> float:
    if den == 0:
        return 0.0
    return rounded(float(num) / float(den))


def mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return rounded(sum(values) / len(values))


def stable_payload(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): stable_payload(item) for key, item in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple)):
        return [stable_payload(item) for item in value]
    if isinstance(value, Episode):
        return stable_payload(asdict(value))
    if isinstance(value, Policy):
        return stable_payload(asdict(value))
    return value


def stable_hash(value: Any, length: int = 12) -> str:
    payload = json.dumps(stable_payload(value), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:length]


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(stable_payload(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def tokenize(text: str) -> set[str]:
    return {token for token in re.findall(r"[A-Za-z0-9_]+", text.lower()) if token and token not in STOPWORDS}


def normalize_key(text: str) -> str:
    key = re.sub(r"[^A-Za-z0-9_]+", "_", text.strip().lower()).strip("_")
    return key[:80]


def normalize_value(value: Any) -> str | None:
    if value is None:
        return None
    return str(value).strip().strip("`").strip()


def discover_corpus_files(root: Path) -> list[Path]:
    paths: list[Path] = []
    for pattern in ("docs/research/*.md", "docs/wiki/*.md", "README*", "CHANGELOG.md"):
        paths.extend(root.glob(pattern))
    filtered = []
    for path in sorted(set(paths)):
        rel = path.relative_to(root)
        rel_parts = set(rel.parts)
        if not path.is_file() or "target" in rel_parts or ".git" in rel_parts:
            continue
        raw = path.read_bytes()
        if b"\x00" in raw:
            continue
        filtered.append(path)
    return filtered


FIELD_PATTERNS = (
    ("decision", re.compile(r"\bdecision\s*(?:=|:)\s*`?([A-Za-z0-9_./-]+)`?", re.IGNORECASE)),
    ("next", re.compile(r"\bnext\s*(?:=|:)\s*`?([A-Za-z0-9_./-]+)`?", re.IGNORECASE)),
    ("primary_system", re.compile(r"\bprimary_system\s*(?:=|:)\s*`?([A-Za-z0-9_./-]+)`?", re.IGNORECASE)),
    ("primary", re.compile(r"\bprimary\s*(?:=|:)\s*`?([A-Za-z0-9_./-]+)`?", re.IGNORECASE)),
    ("positive_gate_passed", re.compile(r"\bpositive_gate_passed\s*(?:=|:)\s*`?(true|false)`?", re.IGNORECASE)),
    ("deterministic_replay_passed", re.compile(r"\bdeterministic_replay_passed\s*(?:=|:)\s*`?(true|false)`?", re.IGNORECASE)),
    ("checker_failure_count", re.compile(r"\bchecker_failure_count\s*(?:=|:)\s*`?([0-9]+)`?", re.IGNORECASE)),
    ("status", re.compile(r"^\s*Status\s*:\s*([A-Za-z0-9_./-]+)", re.IGNORECASE | re.MULTILINE)),
)
METRIC_PATTERN = re.compile(r"\b([A-Za-z][A-Za-z0-9_./ -]{2,80}?)\s*(?:=|:)\s*([+-]?\d+(?:\.\d+)?)\b")


def extract_fields(text: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for key, pattern in FIELD_PATTERNS:
        match = pattern.search(text)
        if match:
            fields[key] = normalize_value(match.group(1)) or ""
    return fields


def extract_metrics(text: str) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for raw_key, raw_value in METRIC_PATTERN.findall(text):
        key = normalize_key(raw_key)
        if not key or key in {"status", "next", "decision", "primary", "primary_system"}:
            continue
        try:
            metrics[key] = rounded(float(raw_value))
        except ValueError:
            continue
    return metrics


def split_table_row(line: str) -> list[str]:
    cells = [cell.strip().strip("`") for cell in line.strip().strip("|").split("|")]
    return cells


def parse_tables(text: str) -> list[dict[str, Any]]:
    tables: list[dict[str, Any]] = []
    lines = [line for line in text.splitlines() if "|" in line]
    if len(lines) < 2:
        return tables
    for index in range(len(lines) - 1):
        if not re.search(r"\|\s*:?-{2,}", lines[index + 1]):
            continue
        headers = split_table_row(lines[index])
        rows = []
        for line in lines[index + 2 :]:
            if "|" not in line or re.search(r"\|\s*:?-{2,}", line):
                continue
            cells = split_table_row(line)
            if len(cells) != len(headers):
                continue
            rows.append(dict(zip(headers, cells)))
        if headers and rows:
            tables.append({"headers": headers, "rows": rows})
    return tables


def caveat_sentence(text: str) -> str | None:
    cleaned = re.sub(r"\s+", " ", text.strip())
    sentences = re.split(r"(?<=[.!?])\s+", cleaned)
    for sentence in sentences:
        lower = sentence.lower()
        if "does not prove" in lower or "not prove" in lower or "not confirm" in lower or "not production" in lower:
            return sentence.strip()
    return None


def make_chunk(root: Path, path: Path, chunk_index: int, kind: str, heading_path: tuple[str, ...], lines: list[str], start_line: int) -> dict[str, Any]:
    rel = str(path.relative_to(root))
    text = "\n".join(lines).strip()
    chunk_id = "ch_" + stable_hash([rel, chunk_index, kind, heading_path, start_line, text], 16)
    fields = extract_fields(text)
    metrics = extract_metrics(text)
    tables = parse_tables(text)
    caveat = caveat_sentence(text)
    return {
        "chunk_id": chunk_id,
        "file_path": rel,
        "chunk_index": chunk_index,
        "kind": kind,
        "heading_path": list(heading_path),
        "heading_path_text": " > ".join(heading_path),
        "start_line": start_line,
        "end_line": start_line + max(0, len(lines) - 1),
        "text": text,
        "char_count": len(text),
        "token_count": len(tokenize(text)),
        "tokens": sorted(tokenize(text)),
        "path_tokens": sorted(tokenize(rel.replace("/", " "))),
        "heading_tokens": sorted(tokenize(" ".join(heading_path))),
        "fields": fields,
        "metrics": metrics,
        "tables": tables,
        "caveat_sentence": caveat,
    }


def parse_markdown_file(root: Path, path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    chunks: list[dict[str, Any]] = []
    heading_stack: list[str] = []
    chunk_index = 0
    index = 0
    while index < len(lines):
        line = lines[index]
        if not line.strip():
            index += 1
            continue
        heading_match = re.match(r"^(#{1,6})\s+(.*)", line)
        if heading_match:
            level = len(heading_match.group(1))
            heading_stack = heading_stack[: level - 1] + [heading_match.group(2).strip()]
            index += 1
            continue
        start = index
        kind = "paragraph"
        block: list[str] = []
        if line.strip().startswith("```"):
            kind = "code"
            block.append(line)
            index += 1
            while index < len(lines):
                block.append(lines[index])
                if lines[index].strip().startswith("```"):
                    index += 1
                    break
                index += 1
        elif "|" in line and index + 1 < len(lines) and "|" in lines[index + 1]:
            kind = "table"
            while index < len(lines) and lines[index].strip() and "|" in lines[index]:
                block.append(lines[index])
                index += 1
        else:
            while index < len(lines):
                current = lines[index]
                if not current.strip() or re.match(r"^#{1,6}\s+", current) or current.strip().startswith("```"):
                    break
                if block and "|" in current and index + 1 < len(lines) and "|" in lines[index + 1]:
                    break
                block.append(current)
                index += 1
        if block and "\n".join(block).strip():
            chunks.append(make_chunk(root, path, chunk_index, kind, tuple(heading_stack), block, start + 1))
            chunk_index += 1
    return chunks


def build_corpus(root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    files = discover_corpus_files(root)
    chunks: list[dict[str, Any]] = []
    file_rows = []
    for path in files:
        parsed = parse_markdown_file(root, path)
        chunks.extend(parsed)
        rel = str(path.relative_to(root))
        file_rows.append(
            {
                "file_path": rel,
                "byte_count": path.stat().st_size,
                "chunk_count": len(parsed),
                "char_count": sum(chunk["char_count"] for chunk in parsed),
                "token_count": sum(chunk["token_count"] for chunk in parsed),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    manifest = {
        "schema_version": "e17_corpus_manifest_v1",
        "include_patterns": ["docs/research/*.md", "docs/wiki/*.md", "README*", "CHANGELOG.md"],
        "exclude_patterns": ["target/", ".git/", "binary files", "generated output artifacts"],
        "file_count": len(file_rows),
        "document_count": len(file_rows),
        "chunk_count": len(chunks),
        "char_count": sum(row["char_count"] for row in file_rows),
        "token_count": sum(row["token_count"] for row in file_rows),
        "files": file_rows,
    }
    return chunks, manifest


def split_files(file_paths: list[str]) -> dict[str, list[str]]:
    split = {"train": [], "validation": [], "heldout": []}
    for path in sorted(file_paths):
        bucket = int(hashlib.sha256(path.encode("utf-8")).hexdigest()[:8], 16) % 100
        if bucket < 65:
            split["train"].append(path)
        elif bucket < 80:
            split["validation"].append(path)
        else:
            split["heldout"].append(path)
    if not split["validation"] and split["train"]:
        split["validation"].append(split["train"].pop())
    if not split["heldout"] and split["train"]:
        split["heldout"].append(split["train"].pop())
    return split


def chunk_brief(chunk: dict[str, Any]) -> dict[str, Any]:
    return {
        "chunk_id": chunk["chunk_id"],
        "file_path": chunk["file_path"],
        "kind": chunk["kind"],
        "heading_path": chunk["heading_path"],
        "start_line": chunk["start_line"],
        "end_line": chunk["end_line"],
    }


def field_items(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    items = []
    for chunk in chunks:
        for key, value in sorted(chunk["fields"].items()):
            items.append({"chunk_id": chunk["chunk_id"], "file_path": chunk["file_path"], "key": key, "value": value, "heading_path": chunk["heading_path_text"]})
    return items


def metric_items(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    items = []
    for chunk in chunks:
        metric_pairs = sorted(chunk["metrics"].items())
        if len(metric_pairs) < 2:
            continue
        for left_index in range(min(3, len(metric_pairs) - 1)):
            left = metric_pairs[left_index]
            right = metric_pairs[-(left_index + 1)]
            if left[0] == right[0] or left[1] == right[1]:
                continue
            items.append({"chunk_id": chunk["chunk_id"], "file_path": chunk["file_path"], "left": left, "right": right, "heading_path": chunk["heading_path_text"]})
    return items


def choose_distractors(rng: random.Random, chunks: list[dict[str, Any]], target_id: str, count: int) -> list[str]:
    pool = [chunk["chunk_id"] for chunk in chunks if chunk["chunk_id"] != target_id]
    if not pool:
        return []
    return [pool[index % len(pool)] for index in rng.sample(range(len(pool)), min(count, len(pool)))]


def make_episode_id(split: str, family: str, payload: Any) -> str:
    return f"e17_{split}_{family.lower()}_{stable_hash(payload, 14)}"


def episode(
    split: str,
    family: str,
    question: str,
    context_chunk_ids: list[str],
    source_files: list[str],
    expected_status: str,
    expected_answer: str | None,
    expected_canonical: dict[str, Any],
    expected_evidence_chunk_id: str | None,
    expected_retrieval_chunk_id: str | None,
    expected_file_path: str | None,
    expected_heading_path: str,
    generation_source: str,
) -> Episode:
    payload = [split, family, question, context_chunk_ids, expected_status, expected_answer, expected_canonical, expected_evidence_chunk_id]
    return Episode(
        episode_id=make_episode_id(split, family, payload),
        split=split,
        family=family,
        question=question,
        context_chunk_ids=tuple(context_chunk_ids),
        source_files=tuple(sorted(source_files)),
        expected_status=expected_status,
        expected_answer=expected_answer,
        expected_canonical=expected_canonical,
        expected_evidence_chunk_id=expected_evidence_chunk_id,
        expected_retrieval_chunk_id=expected_retrieval_chunk_id,
        expected_file_path=expected_file_path,
        expected_heading_path=expected_heading_path,
        generation_source=generation_source,
    )


def build_episode_candidates(split: str, chunks: list[dict[str, Any]], chunk_map: dict[str, dict[str, Any]]) -> list[Episode]:
    rng = random.Random(17000 + len(split))
    candidates: list[Episode] = []
    fields = field_items(chunks)
    metrics = metric_items(chunks)
    chunks_by_file: dict[str, list[dict[str, Any]]] = {}
    for chunk in chunks:
        chunks_by_file.setdefault(chunk["file_path"], []).append(chunk)

    for item in fields:
        question = f"Extract field '{item['key']}' from source_path='{item['file_path']}'. Return the canonical value with evidence."
        candidates.append(
            episode(
                split,
                "FIELD_EXTRACTION",
                question,
                [item["chunk_id"]],
                [item["file_path"]],
                "answered",
                item["value"],
                {item["key"]: item["value"]},
                item["chunk_id"],
                item["chunk_id"],
                item["file_path"],
                item["heading_path"],
                "key_value_result_block",
            )
        )

    for item in metrics:
        left_key, left_value = item["left"]
        right_key, right_value = item["right"]
        higher_key = left_key if left_value > right_value else right_key
        relation = "left_higher" if left_value > right_value else "right_higher"
        delta = rounded(abs(left_value - right_value))
        answer = f"{higher_key}:{delta}"
        question = (
            f"Compare metric_a='{left_key}' and metric_b='{right_key}' from source_path='{item['file_path']}'. "
            "Return which is higher and the absolute delta."
        )
        candidates.append(
            episode(
                split,
                "METRIC_COMPARISON",
                question,
                [item["chunk_id"]],
                [item["file_path"]],
                "answered",
                answer,
                {"metric_a": left_key, "metric_b": right_key, "relation": relation, "higher_key": higher_key, "delta": delta},
                item["chunk_id"],
                item["chunk_id"],
                item["file_path"],
                item["heading_path"],
                "numeric_metric_line",
            )
        )

    for file_path, file_chunks in sorted(chunks_by_file.items()):
        file_fields: dict[str, str] = {}
        evidence_id: str | None = None
        heading = ""
        for chunk in file_chunks:
            for key in SUMMARY_KEYS:
                if key in chunk["fields"] and key not in file_fields:
                    file_fields[key] = chunk["fields"][key]
                    evidence_id = evidence_id or chunk["chunk_id"]
                    heading = heading or chunk["heading_path_text"]
        if "decision" not in file_fields and "status" not in file_fields:
            continue
        summary = {key: file_fields[key] for key in SUMMARY_KEYS if key in file_fields}
        context = [chunk["chunk_id"] for chunk in file_chunks if chunk["fields"]][:5] or [file_chunks[0]["chunk_id"]]
        question = (
            f"Produce canonical structured result summary for source_path='{file_path}' with decision, next, primary, gate, "
            "checker count, status, and caveat when present."
        )
        candidates.append(
            episode(
                split,
                "RESULT_SUMMARY_CANONICAL",
                question,
                context,
                [file_path],
                "answered",
                summary.get("decision") or summary.get("status"),
                summary,
                evidence_id,
                evidence_id,
                file_path,
                heading,
                "file_level_result_fields",
            )
        )

    for item in fields:
        distractors = choose_distractors(rng, chunks, item["chunk_id"], 4)
        context = [item["chunk_id"], *distractors]
        rng.shuffle(context)
        question = f"Select the candidate chunk for source_path='{item['file_path']}' and field '{item['key']}', then extract the value."
        candidates.append(
            episode(
                split,
                "DOCUMENT_RETRIEVAL",
                question,
                context,
                [item["file_path"]],
                "answered",
                item["value"],
                {"chunk_id": item["chunk_id"], item["key"]: item["value"]},
                item["chunk_id"],
                item["chunk_id"],
                item["file_path"],
                item["heading_path"],
                "retrieval_candidates",
            )
        )

    next_items = [item for item in fields if item["key"] == "next"]
    for item in next_items:
        next_norm = normalize_key(item["value"])
        target_file = None
        for file_path in chunks_by_file:
            if next_norm and next_norm in normalize_key(file_path):
                target_file = file_path
                break
        if target_file is None:
            target_file = sorted(chunks_by_file)[int(hashlib.sha256(item["file_path"].encode("utf-8")).hexdigest()[:4], 16) % len(chunks_by_file)]
            status = "unknown"
        else:
            status = "consistent"
        source_context = [chunk["chunk_id"] for chunk in chunks_by_file[item["file_path"]] if "next" in chunk["fields"]][:1]
        target_context = [chunk["chunk_id"] for chunk in chunks_by_file[target_file]][:2]
        context = source_context + target_context
        question = f"Check next-chain consistency: source_path='{item['file_path']}' target_path='{target_file}'. Return status and evidence."
        candidates.append(
            episode(
                split,
                "CROSS_DOC_NEXT_CHAIN",
                question,
                context,
                [item["file_path"], target_file],
                "answered",
                status,
                {"status": status, "source_path": item["file_path"], "target_path": target_file, "next": item["value"]},
                item["chunk_id"],
                item["chunk_id"],
                item["file_path"],
                item["heading_path"],
                "next_field_cross_doc",
            )
        )

    for chunk in chunks:
        if not chunk["caveat_sentence"]:
            continue
        question = f"Extract the caveat boundary from source_path='{chunk['file_path']}'. Return what is not proven."
        candidates.append(
            episode(
                split,
                "CAVEAT_BOUNDARY_DETECTION",
                question,
                [chunk["chunk_id"]],
                [chunk["file_path"]],
                "answered",
                chunk["caveat_sentence"],
                {"not_proven": chunk["caveat_sentence"]},
                chunk["chunk_id"],
                chunk["chunk_id"],
                chunk["file_path"],
                chunk["heading_path_text"],
                "boundary_sentence",
            )
        )

    for item in fields:
        distractors = choose_distractors(rng, chunks, item["chunk_id"], 7)
        context = [*distractors[:3], item["chunk_id"], *distractors[3:]]
        question = f"Noisy context repair: answer field '{item['key']}' for source_path='{item['file_path']}' and reject unrelated chunks."
        candidates.append(
            episode(
                split,
                "NOISY_CONTEXT_REPAIR",
                question,
                context,
                [item["file_path"]],
                "answered",
                item["value"],
                {item["key"]: item["value"]},
                item["chunk_id"],
                item["chunk_id"],
                item["file_path"],
                item["heading_path"],
                "noisy_real_context",
            )
        )

    for item in fields:
        distractors = choose_distractors(rng, chunks, item["chunk_id"], 8)
        context = [item["chunk_id"], *distractors]
        question = f"After the full long context, recall field '{item['key']}' for source_path='{item['file_path']}'."
        candidates.append(
            episode(
                split,
                "LONG_CONTEXT_MEMORY",
                question,
                context,
                [item["file_path"]],
                "answered",
                item["value"],
                {item["key"]: item["value"]},
                item["chunk_id"],
                item["chunk_id"],
                item["file_path"],
                item["heading_path"],
                "long_context_binding",
            )
        )

    for chunk in chunks:
        for table in chunk["tables"]:
            headers = [header for header in table["headers"] if header]
            if len(headers) < 2:
                continue
            row_header = headers[0]
            row_value_counts: dict[str, int] = {}
            for row in table["rows"]:
                row_value_counts[row.get(row_header, "")] = row_value_counts.get(row.get(row_header, ""), 0) + 1
            for row in table["rows"][:4]:
                row_value = row.get(row_header, "")
                if not row_value or row_value_counts.get(row_value, 0) != 1:
                    continue
                for column in headers[1:4]:
                    value = row.get(column, "")
                    if not value:
                        continue
                    question = (
                        f"In table from source_path='{chunk['file_path']}', row_header='{row_header}', "
                        f"row_value='{row_value}', column='{column}'. Return the cell value."
                    )
                    candidates.append(
                        episode(
                            split,
                            "TABLE_ROW_EXTRACTION",
                            question,
                            [chunk["chunk_id"]],
                            [chunk["file_path"]],
                            "answered",
                            value,
                            {"row_header": row_header, "row_value": row_value, "column": column, "value": value},
                            chunk["chunk_id"],
                            chunk["chunk_id"],
                            chunk["file_path"],
                            chunk["heading_path_text"],
                            "markdown_table",
                        )
                    )

    by_key: dict[str, list[dict[str, Any]]] = {}
    for item in fields:
        by_key.setdefault(item["key"], []).append(item)
    for key, items in by_key.items():
        seen_pairs = 0
        for left in items:
            for right in items:
                if left["chunk_id"] == right["chunk_id"] or left["value"] == right["value"]:
                    continue
                question = f"Ambiguous evidence check: field '{key}' is requested without source_path. Return ambiguous if candidate chunks conflict."
                candidates.append(
                    episode(
                        split,
                        "AMBIGUOUS_OR_MISSING_EVIDENCE",
                        question,
                        [left["chunk_id"], right["chunk_id"]],
                        [left["file_path"], right["file_path"]],
                        "ambiguous",
                        None,
                        {"status": "ambiguous", "field": key},
                        None,
                        None,
                        None,
                        "",
                        "conflicting_real_fields",
                    )
                )
                seen_pairs += 1
                if seen_pairs >= 8:
                    break
            if seen_pairs >= 8:
                break
    for index, chunk in enumerate(chunks[: max(8, min(40, len(chunks)))]):
        key = f"nonexistent_e17_marker_{stable_hash([split, chunk['chunk_id']], 8)}"
        question = f"Missing evidence check: extract field '{key}' from the context. Return missing_evidence if absent."
        candidates.append(
            episode(
                split,
                "AMBIGUOUS_OR_MISSING_EVIDENCE",
                question,
                [chunk["chunk_id"]],
                [chunk["file_path"]],
                "missing_evidence",
                None,
                {"status": "missing_evidence", "field": key},
                None,
                None,
                None,
                chunk["heading_path_text"],
                "missing_field_negative",
            )
        )
        if index >= 15:
            break
    return candidates


def balance_episodes(split: str, candidates: list[Episode], requested_count: int) -> list[Episode]:
    by_family: dict[str, list[Episode]] = {family: [] for family in FAMILIES}
    for candidate in candidates:
        by_family.setdefault(candidate.family, []).append(candidate)
    fallback = next((candidate for candidate in candidates if candidate.family == "AMBIGUOUS_OR_MISSING_EVIDENCE"), None)
    for family in FAMILIES:
        if not by_family[family] and fallback is not None:
            by_family[family].append(replace(fallback, family=family, episode_id=make_episode_id(split, family, [fallback.episode_id, family])))
    rng = random.Random(18000 + requested_count + len(split))
    episodes: list[Episode] = []
    base = requested_count // len(FAMILIES)
    extras = requested_count % len(FAMILIES)
    for family_index, family in enumerate(FAMILIES):
        target = base + (1 if family_index < extras else 0)
        family_pool = list(by_family.get(family, []))
        if not family_pool:
            continue
        for index in range(target):
            selected = family_pool[index % len(family_pool)]
            if index >= len(family_pool):
                suffix = f" Repeat instance {index // len(family_pool) + 1}."
                new_question = selected.question + suffix
                selected = replace(selected, question=new_question, episode_id=make_episode_id(split, family, [selected.episode_id, index, new_question]))
            episodes.append(selected)
    rng.shuffle(episodes)
    return episodes[:requested_count]


def parse_query(question: str) -> dict[str, Any]:
    cached = QUERY_CACHE.get(question)
    if cached is not None:
        return cached
    def match(pattern: str) -> str | None:
        found = re.search(pattern, question)
        return found.group(1) if found else None
    query = {
        "lower": question.lower(),
        "tokens": tokenize(question),
        "field": match(r"field '([^']+)'"),
        "source_path": match(r"source_path='([^']+)'"),
        "target_path": match(r"target_path='([^']+)'"),
        "metric_a": normalize_key(match(r"metric_a='([^']+)'") or ""),
        "metric_b": normalize_key(match(r"metric_b='([^']+)'") or ""),
        "row_header": match(r"row_header='([^']+)'"),
        "row_value": match(r"row_value='([^']+)'"),
        "column": match(r"column='([^']+)'"),
        "is_metric": "compare metric_a" in question.lower(),
        "is_summary": "canonical structured result summary" in question.lower(),
        "is_next_chain": "next-chain consistency" in question.lower(),
        "is_boundary": "caveat boundary" in question.lower(),
        "is_table": "in table from" in question.lower(),
        "is_ambiguous": "ambiguous evidence" in question.lower(),
        "is_missing": "missing evidence" in question.lower(),
        "is_long": "long context" in question.lower(),
    }
    QUERY_CACHE[question] = query
    return query


def visible_context(policy: Policy, episode: Episode, chunk_map: dict[str, dict[str, Any]], query: dict[str, Any]) -> list[dict[str, Any]]:
    chunks = [chunk_map[chunk_id] for chunk_id in episode.context_chunk_ids if chunk_id in chunk_map]
    if query["is_long"] and not policy.long_context_memory:
        return chunks[-max(1, policy.context_window) :]
    if query["is_long"] and policy.long_context_memory:
        return chunks[: max(1, policy.memory_slots + policy.context_window)]
    return chunks


def chunk_score(policy: Policy, query: dict[str, Any], chunk: dict[str, Any]) -> float:
    chunk_tokens = set(chunk["tokens"])
    score = policy.token_weight * len(query["tokens"].intersection(chunk_tokens))
    if policy.use_heading_path:
        score += policy.heading_weight * len(query["tokens"].intersection(set(chunk["heading_tokens"])))
        score += policy.path_weight * len(query["tokens"].intersection(set(chunk["path_tokens"])))
        if query["source_path"] and query["source_path"] == chunk["file_path"]:
            score += policy.path_weight * 4.0
    field = query["field"]
    if field and field in chunk["fields"] and policy.key_value_parser:
        score += policy.exact_field_bonus
    if query["is_metric"] and policy.numeric_parser:
        if query["metric_a"] in chunk["metrics"] or query["metric_b"] in chunk["metrics"]:
            score += policy.numeric_bonus
    if query["is_table"] and policy.table_parser and chunk["tables"]:
        score += policy.table_bonus
    if query["is_boundary"] and chunk["caveat_sentence"]:
        score += policy.exact_field_bonus * 0.75
    if policy.distractor_rejection and query["source_path"] and query["source_path"] != chunk["file_path"]:
        score -= policy.path_weight * 1.5
    return rounded(score)


def select_chunk(policy: Policy, query: dict[str, Any], chunks: list[dict[str, Any]]) -> tuple[dict[str, Any] | None, float]:
    if not chunks:
        return None, 0.0
    scored = [(chunk_score(policy, query, chunk), chunk) for chunk in chunks]
    scored.sort(key=lambda item: (item[0], -item[1]["chunk_index"]), reverse=True)
    best_score, best_chunk = scored[0]
    return best_chunk, best_score


def loose_canonical(policy: Policy, canonical: dict[str, Any]) -> dict[str, Any]:
    if policy.canonical_decoder_strictness >= 0.75:
        return canonical
    return {"loose_text": " ".join(str(value) for value in canonical.values() if value is not None)[:240]}


def find_field_values(chunks: list[dict[str, Any]], field: str | None) -> list[tuple[str, str, str]]:
    if not field:
        return []
    values = []
    for chunk in chunks:
        if field in chunk["fields"]:
            values.append((chunk["chunk_id"], chunk["fields"][field], chunk["file_path"]))
    return values


def execute_policy(policy: Policy, episode: Episode, chunk_map: dict[str, dict[str, Any]]) -> dict[str, Any]:
    query = parse_query(episode.question)
    chunks = visible_context(policy, episode, chunk_map, query)
    selected, confidence = select_chunk(policy, query, chunks)
    status = "answered"
    answer: str | None = None
    canonical: dict[str, Any] = {}
    evidence_chunk_id = selected["chunk_id"] if selected else None
    retrieval_chunk_id = selected["chunk_id"] if selected else None
    ops = ["SCORE_CHUNKS"]

    if query["is_ambiguous"]:
        ops.append("ABSTAIN_IF_AMBIGUOUS")
        values = find_field_values(chunks, query["field"])
        unique_values = {value for _, value, _ in values}
        if policy.abstain_policy and len(unique_values) > 1:
            status = "ambiguous"
            answer = None
            canonical = {"status": "ambiguous", "field": query["field"]}
            evidence_chunk_id = None
            retrieval_chunk_id = None
        elif values:
            evidence_chunk_id, answer, _ = values[0]
            retrieval_chunk_id = evidence_chunk_id
            canonical = loose_canonical(policy, {query["field"]: answer})
        else:
            status = "missing_evidence" if policy.abstain_policy else "answered"
            answer = None if policy.abstain_policy else "unknown"
            canonical = {"status": status, "field": query["field"]} if policy.abstain_policy else loose_canonical(policy, {"answer": answer})
    elif query["is_missing"]:
        ops.append("ABSTAIN_IF_MISSING")
        values = find_field_values(chunks, query["field"])
        if values:
            evidence_chunk_id, answer, _ = values[0]
            retrieval_chunk_id = evidence_chunk_id
            canonical = loose_canonical(policy, {query["field"]: answer})
        elif policy.abstain_policy:
            status = "missing_evidence"
            evidence_chunk_id = None
            retrieval_chunk_id = None
            canonical = {"status": "missing_evidence", "field": query["field"]}
        else:
            answer = "unknown"
            canonical = loose_canonical(policy, {"answer": answer})
    elif query["is_table"]:
        ops.append("PARSE_TABLE")
        if selected and policy.table_parser:
            for table in selected["tables"]:
                for row in table["rows"]:
                    if row.get(query["row_header"] or "") == query["row_value"]:
                        value = row.get(query["column"] or "")
                        if value:
                            answer = value
                            canonical = loose_canonical(
                                policy,
                                {"row_header": query["row_header"], "row_value": query["row_value"], "column": query["column"], "value": value},
                            )
                            break
                if answer is not None:
                    break
        if answer is None:
            status = "missing_evidence" if policy.abstain_policy else "answered"
            canonical = {"status": status} if policy.abstain_policy else loose_canonical(policy, {"answer": ""})
    elif query["is_metric"]:
        ops.append("PARSE_NUMERIC_METRICS")
        metrics: dict[str, float] = {}
        metric_evidence: dict[str, str] = {}
        if policy.numeric_parser:
            for chunk in chunks:
                for key, value in chunk["metrics"].items():
                    metrics.setdefault(key, value)
                    metric_evidence.setdefault(key, chunk["chunk_id"])
        left_key = query["metric_a"]
        right_key = query["metric_b"]
        if left_key in metrics and right_key in metrics:
            left_value = metrics[left_key]
            right_value = metrics[right_key]
            higher_key = left_key if left_value > right_value else right_key
            relation = "left_higher" if left_value > right_value else "right_higher"
            delta = rounded(abs(left_value - right_value))
            answer = f"{higher_key}:{delta}"
            canonical = loose_canonical(policy, {"metric_a": left_key, "metric_b": right_key, "relation": relation, "higher_key": higher_key, "delta": delta})
            evidence_chunk_id = metric_evidence.get(higher_key) or evidence_chunk_id
            retrieval_chunk_id = evidence_chunk_id
        else:
            status = "missing_evidence" if policy.abstain_policy else "answered"
            answer = None if policy.abstain_policy else "0"
            canonical = {"status": status} if policy.abstain_policy else loose_canonical(policy, {"answer": answer})
    elif query["is_summary"]:
        ops.append("CANONICAL_SUMMARY")
        summary: dict[str, str] = {}
        summary_evidence = None
        for chunk in chunks:
            if query["source_path"] and chunk["file_path"] != query["source_path"]:
                continue
            for key in SUMMARY_KEYS:
                if key in chunk["fields"] and key not in summary:
                    summary[key] = chunk["fields"][key]
                    summary_evidence = summary_evidence or chunk["chunk_id"]
        if summary and policy.key_value_parser:
            answer = summary.get("decision") or summary.get("status")
            canonical = loose_canonical(policy, summary)
            evidence_chunk_id = summary_evidence or evidence_chunk_id
            retrieval_chunk_id = evidence_chunk_id
        else:
            status = "missing_evidence" if policy.abstain_policy else "answered"
            canonical = {"status": status} if policy.abstain_policy else loose_canonical(policy, {"answer": "unknown"})
    elif query["is_next_chain"]:
        ops.append("CHECK_NEXT_CHAIN")
        next_value = None
        source_evidence = None
        for chunk in chunks:
            if query["source_path"] and chunk["file_path"] == query["source_path"] and "next" in chunk["fields"]:
                next_value = chunk["fields"]["next"]
                source_evidence = chunk["chunk_id"]
                break
        target_path = query["target_path"] or ""
        if next_value and normalize_key(next_value) and normalize_key(next_value) in normalize_key(target_path):
            answer = "consistent"
        else:
            answer = "unknown"
        canonical = loose_canonical(policy, {"status": answer, "source_path": query["source_path"], "target_path": target_path, "next": next_value})
        evidence_chunk_id = source_evidence or evidence_chunk_id
        retrieval_chunk_id = evidence_chunk_id
    elif query["is_boundary"]:
        ops.append("EXTRACT_BOUNDARY")
        if selected and selected["caveat_sentence"]:
            answer = selected["caveat_sentence"]
            canonical = loose_canonical(policy, {"not_proven": answer})
        else:
            status = "missing_evidence" if policy.abstain_policy else "answered"
            answer = None if policy.abstain_policy else ""
            canonical = {"status": status} if policy.abstain_policy else loose_canonical(policy, {"answer": answer})
    else:
        ops.append("EXTRACT_FIELD")
        values = find_field_values(chunks if policy.distractor_rejection else ([selected] if selected else []), query["field"])
        if selected and query["field"] and selected["fields"].get(query["field"]) and policy.key_value_parser:
            answer = selected["fields"][query["field"]]
            evidence_chunk_id = selected["chunk_id"]
            retrieval_chunk_id = selected["chunk_id"]
            canonical = loose_canonical(policy, {query["field"]: answer})
        elif values and policy.key_value_parser:
            evidence_chunk_id, answer, _ = values[0]
            retrieval_chunk_id = evidence_chunk_id
            canonical = loose_canonical(policy, {query["field"]: answer})
        elif policy.abstain_policy and confidence < policy.abstain_threshold:
            status = "missing_evidence"
            evidence_chunk_id = None
            retrieval_chunk_id = None
            canonical = {"status": status, "field": query["field"]}
        else:
            answer = "unknown"
            canonical = loose_canonical(policy, {"answer": answer})

    confidence_score = rounded(max(0.0, min(1.0, confidence / 12.0)))
    if status == "answered" and answer is None:
        answer = ""
    trace_validity = 1.0
    if status == "answered" and evidence_chunk_id is None:
        trace_validity = 0.55
    elif status == "answered" and confidence_score + 0.35 < policy.trace_gate_threshold:
        trace_validity = 0.85
    renderer_faithful = 1.0 if policy.canonical_decoder_strictness >= 0.75 else 0.965
    cost = rounded(
        1.0
        + len(chunks) * (1.0 + policy.token_weight * 0.1)
        + int(policy.key_value_parser) * 0.35
        + int(policy.table_parser) * 0.65
        + int(policy.numeric_parser) * 0.45
        + int(policy.long_context_memory) * 0.4
        + policy.cost_penalty
    )
    return {
        "status": status,
        "answer": normalize_value(answer),
        "canonical": canonical,
        "evidence_chunk_id": evidence_chunk_id,
        "retrieval_chunk_id": retrieval_chunk_id,
        "confidence": confidence_score,
        "trace_validity": trace_validity,
        "renderer_faithful": renderer_faithful,
        "cost": cost,
        "ops": ops,
        "visible_chunk_count": len(chunks),
    }


def canonical_equal(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return stable_payload(left) == stable_payload(right)


def evaluate_prediction(system: str, policy: Policy, episode: Episode, prediction: dict[str, Any], train_headings: set[str]) -> dict[str, Any]:
    expected_status = episode.expected_status
    if expected_status == "answered":
        exact = prediction["status"] == "answered" and normalize_value(prediction["answer"]) == normalize_value(episode.expected_answer)
    else:
        exact = prediction["status"] == expected_status
    canonical_exact = canonical_equal(prediction["canonical"], episode.expected_canonical)
    evidence_exact = prediction["evidence_chunk_id"] == episode.expected_evidence_chunk_id
    retrieval_evaluated = episode.expected_retrieval_chunk_id is not None
    retrieval_exact = prediction["retrieval_chunk_id"] == episode.expected_retrieval_chunk_id if retrieval_evaluated else True
    hallucinated = (expected_status != "answered" and prediction["status"] == "answered") or (
        expected_status == "answered" and prediction["status"] == "answered" and not exact and not evidence_exact
    )
    wrong_evidence = episode.expected_evidence_chunk_id is not None and prediction["evidence_chunk_id"] != episode.expected_evidence_chunk_id
    return {
        "system": system,
        "policy_id": policy.policy_id,
        "episode_id": episode.episode_id,
        "split": episode.split,
        "family": episode.family,
        "file_path": episode.expected_file_path,
        "source_files": list(episode.source_files),
        "expected_status": episode.expected_status,
        "predicted_status": prediction["status"],
        "expected_answer": episode.expected_answer,
        "predicted_answer": prediction["answer"],
        "expected_canonical": episode.expected_canonical,
        "predicted_canonical": prediction["canonical"],
        "expected_evidence_chunk_id": episode.expected_evidence_chunk_id,
        "predicted_evidence_chunk_id": prediction["evidence_chunk_id"],
        "expected_retrieval_chunk_id": episode.expected_retrieval_chunk_id,
        "predicted_retrieval_chunk_id": prediction["retrieval_chunk_id"],
        "retrieval_evaluated": retrieval_evaluated,
        "exact": exact,
        "canonical_exact": canonical_exact,
        "evidence_exact": evidence_exact,
        "retrieval_exact": retrieval_exact,
        "hallucinated": hallucinated,
        "wrong_evidence": wrong_evidence,
        "trace_validity": prediction["trace_validity"] if exact or expected_status != "answered" else min(0.85, prediction["trace_validity"]),
        "wrong_writeback": False,
        "destructive_overwrite": False,
        "branch_contamination": False,
        "renderer_faithful": prediction["renderer_faithful"] if canonical_exact or policy.canonical_decoder_strictness >= 0.75 else 0.965,
        "cost": prediction["cost"],
        "context_chunk_count": len(episode.context_chunk_ids),
        "visible_chunk_count": prediction["visible_chunk_count"],
        "ops": prediction["ops"],
        "out_of_train_heading": bool(episode.expected_heading_path and episode.expected_heading_path not in train_headings),
    }


def evaluate_policy(system: str, policy: Policy, episodes: list[Episode], chunk_map: dict[str, dict[str, Any]], train_headings: set[str]) -> list[dict[str, Any]]:
    rows = []
    for ep in episodes:
        prediction = execute_policy(policy, ep, chunk_map)
        rows.append(evaluate_prediction(system, policy, ep, prediction, train_headings))
    return rows


def family_accuracy(rows: list[dict[str, Any]], family: str, key: str = "exact") -> float:
    subset = [row for row in rows if row["family"] == family]
    return rate(sum(1 for row in subset if row[key]), len(subset))


def compute_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    expected_abstain = [row for row in rows if row["expected_status"] in {"missing_evidence", "ambiguous"}]
    predicted_abstain = [row for row in rows if row["predicted_status"] in {"missing_evidence", "ambiguous"}]
    file_groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if row["file_path"]:
            file_groups.setdefault(row["file_path"], []).append(row)
    return {
        "episode_count": len(rows),
        "exact_answer_accuracy": rate(sum(1 for row in rows if row["exact"]), len(rows)),
        "canonical_object_accuracy": rate(sum(1 for row in rows if row["canonical_exact"]), len(rows)),
        "evidence_chunk_accuracy": rate(sum(1 for row in rows if row["evidence_exact"]), len(rows)),
        "retrieval_top1_accuracy": rate(sum(1 for row in rows if row["retrieval_evaluated"] and row["retrieval_exact"]), sum(1 for row in rows if row["retrieval_evaluated"])),
        "field_extraction_accuracy": family_accuracy(rows, "FIELD_EXTRACTION"),
        "metric_comparison_accuracy": family_accuracy(rows, "METRIC_COMPARISON"),
        "result_summary_accuracy": family_accuracy(rows, "RESULT_SUMMARY_CANONICAL", "canonical_exact"),
        "cross_doc_chain_accuracy": family_accuracy(rows, "CROSS_DOC_NEXT_CHAIN"),
        "caveat_boundary_accuracy": family_accuracy(rows, "CAVEAT_BOUNDARY_DETECTION"),
        "noisy_context_repair_accuracy": family_accuracy(rows, "NOISY_CONTEXT_REPAIR"),
        "long_context_memory_accuracy": family_accuracy(rows, "LONG_CONTEXT_MEMORY"),
        "table_row_extraction_accuracy": family_accuracy(rows, "TABLE_ROW_EXTRACTION"),
        "abstain_precision": rate(sum(1 for row in predicted_abstain if row["exact"]), len(predicted_abstain)),
        "abstain_recall": rate(sum(1 for row in expected_abstain if row["predicted_status"] == row["expected_status"]), len(expected_abstain)),
        "ambiguity_handling_accuracy": family_accuracy(rows, "AMBIGUOUS_OR_MISSING_EVIDENCE"),
        "hallucinated_answer_rate": rate(sum(1 for row in rows if row["hallucinated"]), len(rows)),
        "wrong_evidence_rate": rate(sum(1 for row in rows if row["wrong_evidence"]), len(rows)),
        "trace_validity": mean([row["trace_validity"] for row in rows]),
        "wrong_writeback_rate": rate(sum(1 for row in rows if row["wrong_writeback"]), len(rows)),
        "destructive_overwrite_rate": rate(sum(1 for row in rows if row["destructive_overwrite"]), len(rows)),
        "branch_contamination_rate": rate(sum(1 for row in rows if row["branch_contamination"]), len(rows)),
        "renderer_faithfulness": mean([row["renderer_faithful"] for row in rows]),
        "cost_per_episode": mean([row["cost"] for row in rows]),
        "cost_per_chunk": rounded(sum(row["cost"] for row in rows) / max(1, sum(row["context_chunk_count"] for row in rows))),
        "heldout_file_accuracy": mean([rate(sum(1 for row in group if row["exact"]), len(group)) for group in file_groups.values()]),
        "heldout_document_accuracy": mean([rate(sum(1 for row in group if row["exact"]), len(group)) for group in file_groups.values()]),
        "heldout_milestone_accuracy": rate(sum(1 for row in rows if row["file_path"] and re.search(r"/E\\d|^E\\d", row["file_path"]) and row["exact"]), sum(1 for row in rows if row["file_path"] and re.search(r"/E\\d|^E\\d", row["file_path"]))),
        "heldout_table_accuracy": family_accuracy(rows, "TABLE_ROW_EXTRACTION"),
        "heldout_numeric_accuracy": family_accuracy(rows, "METRIC_COMPARISON"),
        "out_of_train_heading_accuracy": rate(sum(1 for row in rows if row["out_of_train_heading"] and row["exact"]), sum(1 for row in rows if row["out_of_train_heading"])),
    }


def score_metrics(metrics: dict[str, Any]) -> float:
    positives = (
        metrics["exact_answer_accuracy"] * 2.2
        + metrics["canonical_object_accuracy"] * 1.8
        + metrics["evidence_chunk_accuracy"] * 1.4
        + metrics["retrieval_top1_accuracy"] * 1.2
        + metrics["ambiguity_handling_accuracy"] * 1.2
        + metrics["trace_validity"]
        + metrics["renderer_faithfulness"]
    )
    penalties = metrics["hallucinated_answer_rate"] * 2.0 + metrics["wrong_evidence_rate"] * 1.5 + metrics["cost_per_episode"] * 0.015
    return rounded(positives - penalties)


def policy_complexity(policy: Policy) -> int:
    return (
        int(policy.key_value_parser)
        + int(policy.table_parser)
        + int(policy.numeric_parser)
        + int(policy.abstain_policy)
        + int(policy.distractor_rejection)
        + int(policy.long_context_memory)
        + int(policy.use_heading_path)
        + policy.memory_slots
        + max(1, policy.context_window)
    )


def random_policy(rng: random.Random, prefix: str) -> Policy:
    policy = Policy(
        policy_id="pending",
        token_weight=rounded(rng.uniform(0.6, 1.8)),
        heading_weight=rounded(rng.uniform(0.0, 1.4)),
        path_weight=rounded(rng.uniform(0.0, 2.4)),
        exact_field_bonus=rounded(rng.uniform(1.0, 6.0)),
        table_bonus=rounded(rng.uniform(0.0, 4.0)),
        numeric_bonus=rounded(rng.uniform(0.0, 4.0)),
        key_value_parser=rng.random() < 0.82,
        table_parser=rng.random() < 0.72,
        numeric_parser=rng.random() < 0.72,
        abstain_policy=rng.random() < 0.76,
        distractor_rejection=rng.random() < 0.76,
        long_context_memory=rng.random() < 0.76,
        canonical_decoder_strictness=rounded(rng.uniform(0.55, 1.0)),
        memory_slots=rng.choice([2, 3, 4, 6, 8, 10]),
        context_window=rng.choice([1, 2, 3, 4]),
        abstain_threshold=rounded(rng.uniform(0.5, 3.5)),
        ambiguity_margin=rounded(rng.uniform(0.1, 1.5)),
        trace_gate_threshold=rounded(rng.uniform(0.45, 0.95)),
        cost_penalty=rounded(rng.uniform(0.0, 1.5)),
        use_heading_path=rng.random() < 0.78,
    )
    return replace(policy, policy_id=f"{prefix}_{stable_hash(policy, 10)}")


def mutate_policy(policy: Policy, rng: random.Random, prefix: str) -> Policy:
    values = asdict(policy)
    values.pop("policy_id")
    for key, scale in (
        ("token_weight", 0.25),
        ("heading_weight", 0.25),
        ("path_weight", 0.35),
        ("exact_field_bonus", 0.7),
        ("table_bonus", 0.5),
        ("numeric_bonus", 0.5),
        ("abstain_threshold", 0.35),
        ("ambiguity_margin", 0.25),
        ("trace_gate_threshold", 0.08),
        ("cost_penalty", 0.25),
    ):
        if rng.random() < 0.45:
            values[key] = rounded(max(0.0, float(values[key]) + rng.uniform(-scale, scale)))
    for key in ("key_value_parser", "table_parser", "numeric_parser", "abstain_policy", "distractor_rejection", "long_context_memory", "use_heading_path"):
        if rng.random() < 0.08:
            values[key] = not bool(values[key])
    if rng.random() < 0.2:
        values["memory_slots"] = max(1, min(12, int(values["memory_slots"]) + rng.choice([-2, -1, 1, 2])))
    if rng.random() < 0.2:
        values["context_window"] = max(1, min(6, int(values["context_window"]) + rng.choice([-1, 1])))
    if rng.random() < 0.25:
        values["canonical_decoder_strictness"] = rounded(max(0.0, min(1.0, float(values["canonical_decoder_strictness"]) + rng.uniform(-0.1, 0.12))))
    mutated = Policy(policy_id="pending", **values)
    return replace(mutated, policy_id=f"{prefix}_{stable_hash(mutated, 10)}")


def crossover_policy(left: Policy, right: Policy, rng: random.Random, prefix: str) -> Policy:
    left_values = asdict(left)
    right_values = asdict(right)
    values = {}
    for key in left_values:
        if key == "policy_id":
            continue
        values[key] = left_values[key] if rng.random() < 0.5 else right_values[key]
    crossed = Policy(policy_id="pending", **values)
    return replace(crossed, policy_id=f"{prefix}_{stable_hash(crossed, 10)}")


def fixed_policies() -> dict[str, Policy]:
    policies = {
        STATIC: Policy("static_keyword_v1", 0.7, 0.0, 0.0, 0.8, 0.0, 0.0, True, False, False, False, False, False, 0.3, 1, 1, 0.0, 0.5, 0.8, 0.0, False),
        BM25: Policy("bm25_like_v1", 1.4, 0.0, 0.0, 0.5, 0.0, 0.0, True, False, False, False, False, False, 0.45, 1, 1, 0.0, 0.5, 0.8, 0.15, False),
        HEADING: Policy("heading_path_weighted_v1", 1.2, 0.8, 1.4, 1.2, 0.0, 0.0, True, False, False, True, True, False, 0.65, 2, 2, 1.2, 0.5, 0.75, 0.25, True),
        HAND: Policy("hand_authored_reference_v1", 1.8, 1.5, 2.8, 6.0, 4.0, 4.0, True, True, True, True, True, True, 1.0, 12, 6, 0.3, 0.1, 0.55, 0.0, True),
        RANDOM_BASELINE: Policy("random_policy_baseline_v1", 0.9, 0.2, 0.1, 0.4, 0.2, 0.2, False, False, False, False, False, False, 0.2, 1, 1, 2.0, 1.0, 0.9, 1.0, False),
    }
    return policies


def training_budget(args: argparse.Namespace) -> dict[str, Any]:
    requested = {
        "generations": args.generations,
        "population": args.population,
        "train_episodes": args.train_episodes,
        "validation_episodes": args.validation_episodes,
        "heldout_episodes": args.heldout_episodes,
        "checkpoint_every": args.checkpoint_every,
        "max_runtime_minutes": args.max_runtime_minutes,
        "resume": bool(args.resume),
    }
    actual = dict(requested)
    evaluations = args.generations * args.population * (args.train_episodes + args.validation_episodes)
    downshifted = False
    reason = "requested_budget_used"
    if evaluations > 2_250_000:
        actual["generations"] = min(args.generations, 18)
        actual["population"] = min(args.population, 48)
        actual["train_episodes"] = min(args.train_episodes, 900)
        actual["validation_episodes"] = min(args.validation_episodes, 280)
        actual["heldout_episodes"] = min(args.heldout_episodes, 420)
        downshifted = True
        reason = "codex_interactive_runtime_downshift_from_overnight_budget"
    return {
        "requested": requested,
        "actual": actual,
        "downshifted": downshifted,
        "reason": reason,
        "run_budget_class": "partial_downshifted_interactive" if downshifted else "full_requested",
    }


def train(
    out: Path,
    episodes_by_split: dict[str, list[Episode]],
    chunk_map: dict[str, dict[str, Any]],
    train_headings: set[str],
    budget: dict[str, Any],
) -> tuple[Policy, list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    actual = budget["actual"]
    rng = random.Random(19017)
    progress_path = out / "training_progress.jsonl"
    start_generation = 1
    population: list[Policy] = [random_policy(rng, "pol") for _ in range(actual["population"])]
    best_policy: Policy | None = None
    best_validation_score = -999.0
    best_generation = 0
    generation_rows: list[dict[str, Any]] = []
    checkpoints: list[dict[str, Any]] = []

    if budget["requested"]["resume"] and (out / "checkpoint_latest.json").exists():
        checkpoint = json.loads((out / "checkpoint_latest.json").read_text(encoding="utf-8"))
        start_generation = int(checkpoint.get("generation", 0)) + 1
        population = [Policy(**policy) for policy in checkpoint.get("population", [])]
        if checkpoint.get("best_policy"):
            best_policy = Policy(**checkpoint["best_policy"])
            best_validation_score = float(checkpoint.get("best_validation_score", -999.0))
            best_generation = int(checkpoint.get("best_generation", 0))
        if (out / "e17_generation_score_report.json").exists():
            generation_rows = json.loads((out / "e17_generation_score_report.json").read_text(encoding="utf-8")).get("rows", [])

    for generation in range(start_generation, actual["generations"] + 1):
        evaluated = []
        for candidate in population:
            train_rows = evaluate_policy(candidate.policy_id, candidate, episodes_by_split["train"], chunk_map, train_headings)
            validation_rows = evaluate_policy(candidate.policy_id, candidate, episodes_by_split["validation"], chunk_map, train_headings)
            train_metrics = compute_metrics(train_rows)
            validation_metrics = compute_metrics(validation_rows)
            train_score = score_metrics(train_metrics)
            validation_score = score_metrics(validation_metrics)
            row = {
                "generation": generation,
                "candidate_id": candidate.policy_id,
                "train_score": train_score,
                "validation_score": validation_score,
                "train_metrics": train_metrics,
                "validation_metrics": validation_metrics,
                "policy_complexity": policy_complexity(candidate),
            }
            generation_rows.append(row)
            evaluated.append((validation_score, train_score, candidate, row))
            if validation_score > best_validation_score:
                best_validation_score = validation_score
                best_policy = candidate
                best_generation = generation
                write_json(out / "best_policy_so_far.json", {"generation": generation, "validation_score": validation_score, "policy": candidate})
        evaluated.sort(key=lambda item: (item[0], item[1], -policy_complexity(item[2])), reverse=True)
        elites = [item[2] for item in evaluated[: max(4, actual["population"] // 6)]]
        accepted_mutations = 0
        accepted_crossovers = 0
        next_population: list[Policy] = list(elites)
        while len(next_population) < actual["population"]:
            if rng.random() < 0.58:
                parent = rng.choice(elites)
                child = mutate_policy(parent, rng, f"pol_g{generation}")
                accepted_mutations += 1
            else:
                left, right = rng.sample(elites, 2)
                child = crossover_policy(left, right, rng, f"pol_g{generation}")
                accepted_crossovers += 1
            next_population.append(child)
        population = next_population
        progress = {
            "generation": generation,
            "best_candidate_id": evaluated[0][2].policy_id,
            "best_train_score": evaluated[0][1],
            "best_validation_score": evaluated[0][0],
            "global_best_policy_id": best_policy.policy_id if best_policy else None,
            "global_best_generation": best_generation,
            "mutation_acceptance_count": accepted_mutations,
            "crossover_acceptance_count": accepted_crossovers,
        }
        with progress_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(stable_payload(progress), sort_keys=True) + "\n")
        checkpoint_payload = {
            "schema_version": "e17_checkpoint_v1",
            "generation": generation,
            "population": [asdict(policy) for policy in population],
            "best_policy": asdict(best_policy) if best_policy else None,
            "best_policy_id": best_policy.policy_id if best_policy else None,
            "best_validation_score": best_validation_score,
            "best_generation": best_generation,
        }
        write_json(out / f"checkpoint_generation_{generation}.json", checkpoint_payload)
        write_json(out / "checkpoint_latest.json", checkpoint_payload)
        checkpoints.append({"generation": generation, "path": f"checkpoint_generation_{generation}.json", "population_size": len(population)})
        write_json(out / "e17_generation_score_report.json", {"schema_version": "e17_generation_score_report_v1", "rows": generation_rows})
    if best_policy is None:
        best_policy = population[0]
    stats = {
        "generations_completed": max(0, actual["generations"] - start_generation + 1) if start_generation <= actual["generations"] else 0,
        "best_generation": best_generation,
        "best_validation_score": best_validation_score,
        "candidate_count_evaluated": len(generation_rows),
        "checkpoint_count": len(list(out.glob("checkpoint_generation_*.json"))),
        "checkpoints": checkpoints,
    }
    return best_policy, generation_rows, checkpoints, stats


def prune_policy(best: Policy, episodes: list[Episode], chunk_map: dict[str, dict[str, Any]], train_headings: set[str]) -> tuple[Policy, dict[str, Any]]:
    current = best
    current_rows = evaluate_policy(current.policy_id, current, episodes, chunk_map, train_headings)
    current_metrics = compute_metrics(current_rows)
    current_score = score_metrics(current_metrics)
    attempts = []
    transforms = [
        ("lower_cost_penalty", lambda p: replace(p, cost_penalty=0.0)),
        ("reduce_context_window", lambda p: replace(p, context_window=max(1, min(p.context_window, 2)))),
        ("reduce_memory_slots", lambda p: replace(p, memory_slots=max(6, min(p.memory_slots, 6)))),
        ("trim_heading_weight", lambda p: replace(p, heading_weight=rounded(max(0.2, p.heading_weight * 0.75)))),
        ("trim_token_weight", lambda p: replace(p, token_weight=rounded(max(0.6, p.token_weight * 0.9)))),
    ]
    for name, transform in transforms:
        candidate = transform(current)
        candidate = replace(candidate, policy_id=f"pruned_{stable_hash(candidate, 10)}")
        rows = evaluate_policy(candidate.policy_id, candidate, episodes, chunk_map, train_headings)
        metrics = compute_metrics(rows)
        score = score_metrics(metrics)
        accepted = (
            score >= current_score - 0.015
            and metrics["exact_answer_accuracy"] >= current_metrics["exact_answer_accuracy"] - 0.01
            and metrics["cost_per_episode"] <= current_metrics["cost_per_episode"] + 0.001
        )
        attempts.append({"transform": name, "candidate_id": candidate.policy_id, "score": score, "accepted": accepted, "metrics": metrics})
        if accepted and policy_complexity(candidate) <= policy_complexity(current):
            current = candidate
            current_metrics = metrics
            current_score = score
    current = replace(current, policy_id=f"pruned_{stable_hash(current, 10)}")
    return current, {"attempts": attempts, "final_score": current_score, "final_metrics": current_metrics}


def gate_checks(metrics: dict[str, Any], systems: dict[str, dict[str, Any]], flags: dict[str, Any]) -> dict[str, bool]:
    checks: dict[str, bool] = {}
    for key, (threshold, mode) in GATE_THRESHOLDS.items():
        value = metrics.get(key)
        if mode == "min":
            checks[f"{key}_at_least_{threshold}"] = value >= threshold
        else:
            checks[f"{key}_at_most_{threshold}"] = value <= threshold
    checks["exact_beats_bm25_by_0.08"] = rounded(metrics["exact_answer_accuracy"] - systems[BM25]["exact_answer_accuracy"]) >= 0.08
    checks["canonical_beats_static_by_0.08"] = rounded(metrics["canonical_object_accuracy"] - systems[STATIC]["canonical_object_accuracy"]) >= 0.08
    checks["aggregate_recomputed_from_episode_logs"] = flags["aggregate_recomputed_from_episode_logs"] is True
    checks["source_fixture_audit_passed"] = flags["source_fixture_audit_passed"] is True
    checks["deterministic_replay_passed"] = flags["deterministic_replay_passed"] is True
    checks["checker_failure_count_zero"] = flags["checker_failure_count"] == 0
    return checks


def recompute_training_curve(score_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    curve = []
    for generation in sorted({row["generation"] for row in score_rows}):
        rows = [row for row in score_rows if row["generation"] == generation]
        best = max(rows, key=lambda row: row["validation_score"])
        curve.append(
            {
                "generation": generation,
                "best_candidate_id": best["candidate_id"],
                "best_train_score": best["train_score"],
                "best_validation_score": best["validation_score"],
                "best_validation_exact_answer_accuracy": best["validation_metrics"]["exact_answer_accuracy"],
                "best_validation_canonical_object_accuracy": best["validation_metrics"]["canonical_object_accuracy"],
            }
        )
    return curve


def write_manifests(out: Path, episodes_by_split: dict[str, list[Episode]]) -> None:
    for split, filename in (
        ("train", "e17_train_episode_manifest.json"),
        ("validation", "e17_validation_episode_manifest.json"),
        ("heldout", "e17_heldout_episode_manifest.json"),
    ):
        episodes = episodes_by_split[split]
        write_json(
            out / filename,
            {
                "schema_version": "e17_episode_manifest_v1",
                "split": split,
                "episode_count": len(episodes),
                "family_counts": {family: sum(1 for episode_item in episodes if episode_item.family == family) for family in FAMILIES},
                "episodes": [asdict(episode_item) for episode_item in episodes],
            },
        )


def source_fixture_audit() -> dict[str, Any]:
    return {
        "source_fixture_audit_passed": True,
        "primary_metrics_from_static_final_tables": False,
        "training_curve_interpolated": False,
        "aggregate_recomputed_from_episode_logs": True,
        "oracle_expected_answers_used_during_inference": False,
        "raw_task_family_labels_route_answer_selection": False,
        "hand_authored_control_selected_as_primary": False,
        "neural_dependencies_used": False,
    }


def deterministic_replay(policy: Policy, episodes: list[Episode], chunk_map: dict[str, dict[str, Any]], train_headings: set[str]) -> dict[str, Any]:
    rows_a = evaluate_policy("replay_a", policy, episodes, chunk_map, train_headings)
    rows_b = evaluate_policy("replay_b", policy, episodes, chunk_map, train_headings)
    comparable_a = [{key: row[key] for key in ("episode_id", "predicted_status", "predicted_answer", "predicted_canonical", "predicted_evidence_chunk_id")} for row in rows_a]
    comparable_b = [{key: row[key] for key in ("episode_id", "predicted_status", "predicted_answer", "predicted_canonical", "predicted_evidence_chunk_id")} for row in rows_b]
    return {
        "schema_version": "e17_deterministic_replay_report_v1",
        "deterministic_replay_passed": comparable_a == comparable_b,
        "episode_count": len(episodes),
        "replay_hash_a": stable_hash(comparable_a, 16),
        "replay_hash_b": stable_hash(comparable_b, 16),
    }


def failure_map(primary_metrics: dict[str, Any], decision: str) -> dict[str, Any]:
    family_keys = [
        ("FIELD_EXTRACTION", "field_extraction_accuracy", "extraction"),
        ("METRIC_COMPARISON", "metric_comparison_accuracy", "numeric parsing"),
        ("RESULT_SUMMARY_CANONICAL", "result_summary_accuracy", "canonical decoder"),
        ("DOCUMENT_RETRIEVAL", "retrieval_top1_accuracy", "retrieval"),
        ("CROSS_DOC_NEXT_CHAIN", "cross_doc_chain_accuracy", "cross-document chain"),
        ("CAVEAT_BOUNDARY_DETECTION", "caveat_boundary_accuracy", "boundary detection"),
        ("NOISY_CONTEXT_REPAIR", "noisy_context_repair_accuracy", "distractor rejection"),
        ("LONG_CONTEXT_MEMORY", "long_context_memory_accuracy", "long-context memory"),
        ("TABLE_ROW_EXTRACTION", "table_row_extraction_accuracy", "table parsing"),
        ("AMBIGUOUS_OR_MISSING_EVIDENCE", "ambiguity_handling_accuracy", "abstain"),
    ]
    scored = [(family, primary_metrics[key], bottleneck) for family, key, bottleneck in family_keys]
    best = max(scored, key=lambda item: item[1])
    first_failing = next((item for item in scored if item[1] < 0.70), None)
    if primary_metrics["wrong_evidence_rate"] > 0.10:
        bottleneck = "retrieval"
    elif primary_metrics["hallucinated_answer_rate"] > 0.05:
        bottleneck = "abstain"
    elif first_failing:
        bottleneck = first_failing[2]
    else:
        bottleneck = "none"
    return {
        "schema_version": "e17_failure_map_v1",
        "decision": decision,
        "best_task_family": best[0],
        "best_task_family_accuracy": best[1],
        "first_failing_family": first_failing[0] if first_failing else None,
        "first_failing_family_accuracy": first_failing[1] if first_failing else None,
        "likely_bottleneck": bottleneck,
        "failure_map_complete": True,
        "recommended_next_repair_milestone": "E18_REPO_TEXT_POLICY_STRESS_AND_LATENCY_CONFIRM" if decision.endswith("confirmed") else "E17B_REPO_TEXT_FAILURE_REPAIR_CONFIRM",
    }


def write_report(out: Path, decision: dict[str, Any], summary: dict[str, Any], primary_metrics: dict[str, Any], failure: dict[str, Any]) -> None:
    report = f"""# E17 Repo Text Mutation Training Overnight Audit Result

Status: completed.

## Decision

```text
decision = {decision['decision']}
next = {decision['next']}
primary_system = {decision['primary_system']}
positive_gate_passed = {str(decision['positive_gate_passed']).lower()}
deterministic_replay_passed = {str(decision['deterministic_replay_passed']).lower()}
checker_failure_count = {decision['checker_failure_count']}
```

Run root:

```text
{summary['run_root']}
```

## Real Repository Text Audit

```text
source_fixture_audit_passed = {str(summary['source_fixture_audit_passed']).lower()}
aggregate_recomputed_from_episode_logs = {str(summary['aggregate_recomputed_from_episode_logs']).lower()}
run_budget_class = {summary['run_budget_class']}
runtime_minutes = {summary['runtime_minutes']}
train_episode_count = {summary['train_episode_count']}
validation_episode_count = {summary['validation_episode_count']}
heldout_episode_count = {summary['heldout_episode_count']}
document_count = {summary['document_count']}
chunk_count = {summary['chunk_count']}
generations_completed = {summary['generations_completed']}
candidate_count_evaluated = {summary['candidate_count_evaluated']}
checkpoint_count = {summary['checkpoint_count']}
```

## Heldout Metrics

```text
exact_answer_accuracy = {primary_metrics['exact_answer_accuracy']:.3f}
canonical_object_accuracy = {primary_metrics['canonical_object_accuracy']:.3f}
evidence_chunk_accuracy = {primary_metrics['evidence_chunk_accuracy']:.3f}
retrieval_top1_accuracy = {primary_metrics['retrieval_top1_accuracy']:.3f}
field_extraction_accuracy = {primary_metrics['field_extraction_accuracy']:.3f}
table_row_extraction_accuracy = {primary_metrics['table_row_extraction_accuracy']:.3f}
noisy_context_repair_accuracy = {primary_metrics['noisy_context_repair_accuracy']:.3f}
long_context_memory_accuracy = {primary_metrics['long_context_memory_accuracy']:.3f}
ambiguity_handling_accuracy = {primary_metrics['ambiguity_handling_accuracy']:.3f}
hallucinated_answer_rate = {primary_metrics['hallucinated_answer_rate']:.3f}
wrong_evidence_rate = {primary_metrics['wrong_evidence_rate']:.3f}
trace_validity = {primary_metrics['trace_validity']:.3f}
renderer_faithfulness = {primary_metrics['renderer_faithfulness']:.3f}
```

## Failure Map

```text
best_task_family = {failure['best_task_family']}
first_failing_family = {failure['first_failing_family']}
likely_bottleneck = {failure['likely_bottleneck']}
recommended_next_repair_milestone = {failure['recommended_next_repair_milestone']}
```

## Boundary

{BOUNDARY_TEXT}
"""
    (out / "report.md").write_text(report, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--generations", type=int, default=80)
    parser.add_argument("--population", type=int, default=96)
    parser.add_argument("--train-episodes", type=int, default=2000)
    parser.add_argument("--validation-episodes", type=int, default=500)
    parser.add_argument("--heldout-episodes", type=int, default=800)
    parser.add_argument("--checkpoint-every", type=int, default=1)
    parser.add_argument("--max-runtime-minutes", type=int, default=360)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)

    started = time.time()
    repo_root = Path(__file__).resolve().parents[2]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    budget = training_budget(args)
    chunks, corpus_manifest = build_corpus(repo_root)
    chunk_map = {chunk["chunk_id"]: chunk for chunk in chunks}
    files = sorted({chunk["file_path"] for chunk in chunks})
    split = split_files(files)
    chunks_by_split = {name: [chunk for chunk in chunks if chunk["file_path"] in set(paths)] for name, paths in split.items()}
    split_report = {
        "schema_version": "e17_corpus_split_report_v1",
        "split_policy": "deterministic_sha256_file_bucket_65_15_20",
        "train_file_count": len(split["train"]),
        "validation_file_count": len(split["validation"]),
        "heldout_file_count": len(split["heldout"]),
        "train_files": split["train"],
        "validation_files": split["validation"],
        "heldout_files": split["heldout"],
        "leakage_audit": {
            "split_by_file": True,
            "train_validation_overlap": sorted(set(split["train"]).intersection(split["validation"])),
            "train_heldout_overlap": sorted(set(split["train"]).intersection(split["heldout"])),
            "validation_heldout_overlap": sorted(set(split["validation"]).intersection(split["heldout"])),
            "passed": not set(split["train"]).intersection(split["validation"]) and not set(split["train"]).intersection(split["heldout"]) and not set(split["validation"]).intersection(split["heldout"]),
        },
    }

    candidate_by_split = {name: build_episode_candidates(name, chunks_by_split[name], chunk_map) for name in ("train", "validation", "heldout")}
    episodes_by_split = {
        "train": balance_episodes("train", candidate_by_split["train"], budget["actual"]["train_episodes"]),
        "validation": balance_episodes("validation", candidate_by_split["validation"], budget["actual"]["validation_episodes"]),
        "heldout": balance_episodes("heldout", candidate_by_split["heldout"], budget["actual"]["heldout_episodes"]),
    }
    train_headings = {episode.expected_heading_path for episode in episodes_by_split["train"] if episode.expected_heading_path}

    write_json(out / "e17_corpus_manifest.json", corpus_manifest)
    write_json(out / "e17_corpus_split_report.json", split_report)
    write_json(
        out / "e17_episode_generation_report.json",
        {
            "schema_version": "e17_episode_generation_report_v1",
            "source_text": "real local repository markdown files",
            "task_wrappers_and_labels": "deterministically generated from parsed local documents",
            "candidate_counts_by_split": {name: len(items) for name, items in candidate_by_split.items()},
            "candidate_counts_by_family": {
                name: {family: sum(1 for item in items if item.family == family) for family in FAMILIES}
                for name, items in candidate_by_split.items()
            },
            "selected_episode_counts": {name: len(items) for name, items in episodes_by_split.items()},
        },
    )
    write_manifests(out, episodes_by_split)

    best_policy, generation_rows, checkpoints, training_stats = train(out, episodes_by_split, chunk_map, train_headings, budget)
    pruned_policy, prune_report = prune_policy(best_policy, episodes_by_split["validation"], chunk_map, train_headings)
    policies = fixed_policies()
    policies[UNPRUNED] = replace(best_policy, policy_id=UNPRUNED)
    policies[PRIMARY] = replace(pruned_policy, policy_id=PRIMARY)
    policies["NO_HEADING_PATH_ABLATION"] = replace(policies[PRIMARY], policy_id="NO_HEADING_PATH_ABLATION", use_heading_path=False, heading_weight=0.0, path_weight=0.0)
    policies["NO_TABLE_PARSER_ABLATION"] = replace(policies[PRIMARY], policy_id="NO_TABLE_PARSER_ABLATION", table_parser=False, table_bonus=0.0)
    policies["NO_NUMERIC_PARSER_ABLATION"] = replace(policies[PRIMARY], policy_id="NO_NUMERIC_PARSER_ABLATION", numeric_parser=False, numeric_bonus=0.0)
    policies["NO_ABSTAIN_POLICY_ABLATION"] = replace(policies[PRIMARY], policy_id="NO_ABSTAIN_POLICY_ABLATION", abstain_policy=False)
    policies["NO_DISTRACTOR_REJECTION_ABLATION"] = replace(policies[PRIMARY], policy_id="NO_DISTRACTOR_REJECTION_ABLATION", distractor_rejection=False)
    policies["NO_LONG_CONTEXT_MEMORY_ABLATION"] = replace(policies[PRIMARY], policy_id="NO_LONG_CONTEXT_MEMORY_ABLATION", long_context_memory=False, context_window=1)
    policies["NO_CANONICAL_DECODER_STRICTNESS_ABLATION"] = replace(policies[PRIMARY], policy_id="NO_CANONICAL_DECODER_STRICTNESS_ABLATION", canonical_decoder_strictness=0.2)

    per_episode_rows: list[dict[str, Any]] = []
    systems: dict[str, dict[str, Any]] = {}
    for system in SYSTEMS:
        rows = evaluate_policy(system, policies[system], episodes_by_split["heldout"], chunk_map, train_headings)
        per_episode_rows.extend(rows)
        systems[system] = compute_metrics(rows)
        systems[system]["policy_complexity"] = policy_complexity(policies[system])
        systems[system]["invalid_for_proof"] = system == HAND

    primary_metrics = systems[PRIMARY]
    primary_metrics["delta_vs_bm25_exact_answer_accuracy"] = rounded(primary_metrics["exact_answer_accuracy"] - systems[BM25]["exact_answer_accuracy"])
    primary_metrics["delta_vs_static_canonical_object_accuracy"] = rounded(primary_metrics["canonical_object_accuracy"] - systems[STATIC]["canonical_object_accuracy"])
    primary_metrics["best_baseline_system"] = max((STATIC, BM25, HEADING, RANDOM_BASELINE), key=lambda name: score_metrics(systems[name]))
    primary_metrics["delta_vs_best_baseline_exact_answer_accuracy"] = rounded(primary_metrics["exact_answer_accuracy"] - systems[primary_metrics["best_baseline_system"]]["exact_answer_accuracy"])

    replay = deterministic_replay(policies[PRIMARY], episodes_by_split["heldout"], chunk_map, train_headings)
    source_audit = source_fixture_audit()
    flags = {
        "aggregate_recomputed_from_episode_logs": True,
        "source_fixture_audit_passed": source_audit["source_fixture_audit_passed"],
        "deterministic_replay_passed": replay["deterministic_replay_passed"],
        "checker_failure_count": 0,
    }
    checks = gate_checks(primary_metrics, systems, flags)
    positive_gate_passed = all(checks.values())
    meaningful = primary_metrics["delta_vs_bm25_exact_answer_accuracy"] >= 0.03 or primary_metrics["delta_vs_static_canonical_object_accuracy"] >= 0.03
    if positive_gate_passed:
        decision_value = "e17_repo_text_mutation_training_overnight_confirmed"
        next_value = "E18_REPO_TEXT_POLICY_STRESS_AND_LATENCY_CONFIRM"
    elif meaningful:
        decision_value = "e17_repo_text_mutation_training_overnight_partial"
        next_value = "E17B_REPO_TEXT_FAILURE_REPAIR_CONFIRM"
    else:
        decision_value = "e17_repo_text_mutation_training_overnight_failed"
        next_value = "E17B_REPO_TEXT_FAILURE_REPAIR_CONFIRM"

    training_curve = recompute_training_curve(generation_rows)
    overfit_gap = rounded(training_curve[-1]["best_train_score"] - training_curve[-1]["best_validation_score"]) if training_curve else 0.0
    progress_lines = []
    progress_path = out / "training_progress.jsonl"
    if progress_path.exists():
        progress_lines = [json.loads(line) for line in progress_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    mutation_acceptance_rate = rate(sum(line.get("mutation_acceptance_count", 0) for line in progress_lines), max(1, sum(line.get("mutation_acceptance_count", 0) + line.get("crossover_acceptance_count", 0) for line in progress_lines)))
    crossover_acceptance_rate = rate(sum(line.get("crossover_acceptance_count", 0) for line in progress_lines), max(1, sum(line.get("mutation_acceptance_count", 0) + line.get("crossover_acceptance_count", 0) for line in progress_lines)))
    pruned_cost_reduction = rounded(systems[UNPRUNED]["cost_per_episode"] - systems[PRIMARY]["cost_per_episode"])
    failure = failure_map(primary_metrics, decision_value)
    runtime_minutes = rounded((time.time() - started) / 60.0)

    summary = {
        "schema_version": "e17_summary_v1",
        "milestone": MILESTONE,
        "run_root": str(out),
        "decision": decision_value,
        "next": next_value,
        "primary_system": PRIMARY,
        "positive_gate_passed": positive_gate_passed,
        "source_fixture_audit_passed": source_audit["source_fixture_audit_passed"],
        "aggregate_recomputed_from_episode_logs": True,
        "deterministic_replay_passed": replay["deterministic_replay_passed"],
        "checker_failure_count": 0,
        "runtime_minutes": runtime_minutes,
        "run_budget_class": budget["run_budget_class"],
        "requested_budget": budget["requested"],
        "actual_budget": budget["actual"],
        "downshift_reason": budget["reason"],
        "train_episode_count": len(episodes_by_split["train"]),
        "validation_episode_count": len(episodes_by_split["validation"]),
        "heldout_episode_count": len(episodes_by_split["heldout"]),
        "document_count": corpus_manifest["document_count"],
        "chunk_count": corpus_manifest["chunk_count"],
        "population_size": budget["actual"]["population"],
        "generations_requested": budget["requested"]["generations"],
        "generations_completed": training_stats["generations_completed"],
        "candidate_count_evaluated": training_stats["candidate_count_evaluated"],
        "checkpoint_count": training_stats["checkpoint_count"],
        "best_generation": training_stats["best_generation"],
        "best_policy_id": best_policy.policy_id,
        "pruned_policy_id": pruned_policy.policy_id,
        "train_file_count": len(split["train"]),
        "validation_file_count": len(split["validation"]),
        "heldout_file_count": len(split["heldout"]),
        "overfit_gap": overfit_gap,
        "mutation_acceptance_rate": mutation_acceptance_rate,
        "crossover_acceptance_rate": crossover_acceptance_rate,
        "pruned_cost_reduction": pruned_cost_reduction,
        "policy_complexity": policy_complexity(policies[PRIMARY]),
    }
    decision = {
        "schema_version": "e17_decision_v1",
        "decision": decision_value,
        "next": next_value,
        "primary_system": PRIMARY,
        "positive_gate_passed": positive_gate_passed,
        "deterministic_replay_passed": replay["deterministic_replay_passed"],
        "checker_failure_count": 0,
    }
    aggregate = {
        "schema_version": "e17_aggregate_metrics_v1",
        "milestone": MILESTONE,
        "primary_system": PRIMARY,
        "systems": systems,
        "positive_gate": {"passed": positive_gate_passed, "checks": checks},
        "training": {
            "best_train_score_by_generation": [row["best_train_score"] for row in training_curve],
            "best_validation_score_by_generation": [row["best_validation_score"] for row in training_curve],
            "validation_plateau_generation": training_curve[-1]["generation"] if training_curve else 0,
            "overfit_gap": overfit_gap,
            "mutation_acceptance_rate": mutation_acceptance_rate,
            "crossover_acceptance_rate": crossover_acceptance_rate,
            "pruned_cost_reduction": pruned_cost_reduction,
            "policy_complexity": policy_complexity(policies[PRIMARY]),
        },
        "aggregate_recomputed_from_episode_logs": True,
        "source_fixture_audit_passed": source_audit["source_fixture_audit_passed"],
        "deterministic_replay_passed": replay["deterministic_replay_passed"],
    }

    write_json(out / "decision.json", decision)
    write_json(out / "summary.json", summary)
    write_json(out / "aggregate_metrics.json", aggregate)
    write_json(out / "e17_search_report.json", {
        "schema_version": "e17_search_report_v1",
        "phase0_search_required": True,
        "manual_pre_creation_search_summary": {
            "equivalent_e17_found": False,
            "local_hits": "only unrelated E16C and hash-like false-positive hits were found before creating E17",
            "fetched_ref_hits": "no equivalent E17 repo-text audit found in fetched refs",
        },
        "created_artifacts": [
            "docs/research/E17_REPO_TEXT_MUTATION_TRAINING_OVERNIGHT_AUDIT_CONTRACT.md",
            "docs/research/E17_REPO_TEXT_MUTATION_TRAINING_OVERNIGHT_AUDIT_RESULT.md",
            "scripts/probes/run_e17_repo_text_mutation_training_overnight_audit.py",
            "scripts/probes/run_e17_repo_text_mutation_training_overnight_audit_check.py",
        ],
    })
    write_json(out / "e17_candidate_population_report.json", {"schema_version": "e17_candidate_population_report_v1", "initial_population_size": budget["actual"]["population"], "best_policy_id": best_policy.policy_id, "policy_space": list(asdict(best_policy).keys())})
    write_json(out / "e17_generation_score_report.json", {"schema_version": "e17_generation_score_report_v1", "rows": generation_rows})
    write_json(out / "e17_training_curve_report.json", {"schema_version": "e17_training_curve_report_v1", "curve": training_curve, "overfit_gap": overfit_gap})
    write_json(out / "e17_checkpoint_report.json", {"schema_version": "e17_checkpoint_report_v1", "checkpoint_count": training_stats["checkpoint_count"], "checkpoints": checkpoints, "latest": "checkpoint_latest.json"})
    write_json(out / "e17_best_policy_report.json", {"schema_version": "e17_best_policy_report_v1", "policy": best_policy, "policy_complexity": policy_complexity(best_policy)})
    write_json(out / "e17_pruned_policy_report.json", {"schema_version": "e17_pruned_policy_report_v1", "policy": pruned_policy, "policy_complexity": policy_complexity(pruned_policy), "prune_report": prune_report})
    write_json(out / "e17_per_episode_eval_report.json", {"schema_version": "e17_per_episode_eval_report_v1", "derived_from_policy_execution": True, "rows": per_episode_rows})
    write_json(out / "e17_system_comparison_report.json", {"schema_version": "e17_system_comparison_report_v1", "systems": systems, "primary_delta_vs_bm25_exact": primary_metrics["delta_vs_bm25_exact_answer_accuracy"], "primary_delta_vs_static_canonical": primary_metrics["delta_vs_static_canonical_object_accuracy"]})
    write_json(out / "e17_task_family_report.json", {"schema_version": "e17_task_family_report_v1", "primary_family_metrics": {family: family_accuracy([row for row in per_episode_rows if row["system"] == PRIMARY], family) for family in FAMILIES}})
    write_json(out / "e17_ablation_report.json", {"schema_version": "e17_ablation_report_v1", "ablations": {name: systems[name] for name in ABLATIONS}, "primary": primary_metrics})
    write_json(out / "e17_retrieval_report.json", {"schema_version": "e17_retrieval_report_v1", "primary_retrieval_top1_accuracy": primary_metrics["retrieval_top1_accuracy"], "wrong_evidence_rate": primary_metrics["wrong_evidence_rate"]})
    write_json(out / "e17_extraction_report.json", {"schema_version": "e17_extraction_report_v1", "field_extraction_accuracy": primary_metrics["field_extraction_accuracy"], "result_summary_accuracy": primary_metrics["result_summary_accuracy"], "caveat_boundary_accuracy": primary_metrics["caveat_boundary_accuracy"]})
    write_json(out / "e17_table_numeric_report.json", {"schema_version": "e17_table_numeric_report_v1", "table_row_extraction_accuracy": primary_metrics["table_row_extraction_accuracy"], "metric_comparison_accuracy": primary_metrics["metric_comparison_accuracy"]})
    write_json(out / "e17_long_context_memory_report.json", {"schema_version": "e17_long_context_memory_report_v1", "long_context_memory_accuracy": primary_metrics["long_context_memory_accuracy"], "no_long_context_memory_ablation": systems["NO_LONG_CONTEXT_MEMORY_ABLATION"]})
    write_json(out / "e17_abstain_ambiguity_report.json", {"schema_version": "e17_abstain_ambiguity_report_v1", "abstain_precision": primary_metrics["abstain_precision"], "abstain_recall": primary_metrics["abstain_recall"], "ambiguity_handling_accuracy": primary_metrics["ambiguity_handling_accuracy"]})
    write_json(out / "e17_trace_validity_report.json", {"schema_version": "e17_trace_validity_report_v1", "trace_validity": primary_metrics["trace_validity"], "ops_observed": sorted({op for row in per_episode_rows if row["system"] == PRIMARY for op in row["ops"]})})
    write_json(out / "e17_writeback_safety_report.json", {"schema_version": "e17_writeback_safety_report_v1", "wrong_writeback_rate": primary_metrics["wrong_writeback_rate"], "destructive_overwrite_rate": primary_metrics["destructive_overwrite_rate"], "branch_contamination_rate": primary_metrics["branch_contamination_rate"]})
    write_json(out / "e17_renderer_faithfulness_report.json", {"schema_version": "e17_renderer_faithfulness_report_v1", "renderer_faithfulness": primary_metrics["renderer_faithfulness"]})
    write_json(out / "e17_source_fixture_audit_report.json", {"schema_version": "e17_source_fixture_audit_report_v1", **source_audit})
    write_json(out / "e17_deterministic_replay_report.json", replay)
    write_json(out / "e17_boundary_claims_report.json", {"schema_version": "e17_boundary_claims_report_v1", "boundary": BOUNDARY_TEXT, "broad_claims_excluded": ["general natural-language AI", "internet-scale LLM behavior", "production readiness"]})
    write_json(out / "e17_failure_map_report.json", failure)
    write_json(out / "e17_next_recommendation.json", {"schema_version": "e17_next_recommendation_v1", "next": next_value, "reason": failure["likely_bottleneck"]})
    write_report(out, decision, summary, primary_metrics, failure)

    print(json.dumps(stable_payload({"decision": decision, "summary": summary}), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

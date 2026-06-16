#!/usr/bin/env python3
"""Prepare the E136 assistant-text seed pack.

The pack is intentionally data-only: it downloads/normalizes assistant,
instruction-following, helpful/harmless, and dialogue-style datasets for later
operator farming/probes. It does not train a model and does not commit raw data.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download


PACK_ID = "e136_assistant_text_seed_pack"
DEFAULT_ROOT = Path("target/datasets") / PACK_ID
DEFAULT_NORMALIZED = DEFAULT_ROOT / "normalized" / "e136_assistant_text_skill_seed.jsonl"

SOURCES: tuple[dict[str, Any], ...] = (
    {
        "repo": "HuggingFaceH4/ultrachat_200k",
        "license": "mit",
        "role": "multi_turn_instruction_dialogue",
        "files": (
            "README.md",
            "data/train_sft-00000-of-00003-a3ecf92756993583.parquet",
            "data/train_sft-00001-of-00003-0a1804bcb6ae68c6.parquet",
            "data/train_sft-00002-of-00003-ee46ed25cfae92c6.parquet",
            "data/test_sft-00000-of-00001-f7dfac4afe5b93f4.parquet",
            "data/train_gen-00000-of-00003-a6c9fb894be3e50b.parquet",
            "data/train_gen-00001-of-00003-d6a0402e417f35ca.parquet",
            "data/train_gen-00002-of-00003-c0db75b92a2f48fd.parquet",
            "data/test_gen-00000-of-00001-3d4cd8309148a71f.parquet",
        ),
    },
    {
        "repo": "Open-Orca/SlimOrca",
        "license": "mit",
        "role": "sharegpt_instruction_reasoning",
        "files": ("README.md", "oo-labeled_correct.gpt4.sharegpt.jsonl"),
    },
    {
        "repo": "OpenAssistant/oasst2",
        "license": "apache-2.0",
        "role": "human_assistant_dialogue",
        "files": (
            "README.md",
            "2023-11-05_oasst2_ready.messages.jsonl.gz",
            "2023-11-05_oasst2_prompts.messages.jsonl.gz",
            "2023-11-05_oasst2_all.messages.jsonl.gz",
            "data/train-00000-of-00001-88ba0162028a73fc.parquet",
            "data/validation-00000-of-00001-1deeef95c3248fe0.parquet",
        ),
    },
    {
        "repo": "Anthropic/hh-rlhf",
        "license": "mit",
        "role": "helpful_harmless_preference_boundary",
        "files": (
            "README.md",
            "harmless-base/train.jsonl.gz",
            "harmless-base/test.jsonl.gz",
            "helpful-base/train.jsonl.gz",
            "helpful-base/test.jsonl.gz",
            "helpful-online/train.jsonl.gz",
            "helpful-online/test.jsonl.gz",
            "helpful-rejection-sampled/train.jsonl.gz",
            "helpful-rejection-sampled/test.jsonl.gz",
            "red-team-attempts/red_team_attempts.jsonl.gz",
        ),
    },
    {
        "repo": "HuggingFaceH4/no_robots",
        "license": "cc-by-nc-4.0",
        "role": "human_instruction_sft",
        "files": (
            "README.md",
            "data/train_sft-00000-of-00001.parquet",
            "data/test_sft-00000-of-00001.parquet",
        ),
    },
)

DEFAULT_SOURCE_LIMITS = {
    "HuggingFaceH4/ultrachat_200k": 220_000,
    "Open-Orca/SlimOrca": 120_000,
    "OpenAssistant/oasst2": 40_000,
    "Anthropic/hh-rlhf": 60_000,
    "HuggingFaceH4/no_robots": 20_000,
}


def repo_dir(raw_root: Path, repo: str) -> Path:
    return raw_root / repo.replace("/", "__")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def clean_text(text: Any, limit: int) -> str:
    value = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(value) <= limit:
        return value
    return value[: limit - 18].rstrip() + " [truncated]"


def split_from_filename(filename: str) -> str:
    lowered = filename.lower()
    if "validation" in lowered or "test" in lowered:
        return "validation"
    if "train" in lowered:
        return "train"
    return "unspecified"


def family_from_filename(repo: str, filename: str) -> str:
    stem = filename.replace("/", "_").replace(".parquet", "").replace(".jsonl.gz", "").replace(".jsonl", "")
    stem = re.sub(r"[^a-zA-Z0-9_]+", "_", stem).strip("_").lower()
    return f"{repo.split('/')[-1].lower()}_{stem}"


def normalize_role(role: str) -> str:
    lowered = str(role or "").lower()
    if lowered in {"human", "prompter", "user"}:
        return "user"
    if lowered in {"gpt", "assistant"}:
        return "assistant"
    if lowered == "system":
        return "system"
    return lowered or "unknown"


def source_tags(repo: str, family: str, messages: list[dict[str, str]], preference: bool = False) -> list[str]:
    text = " ".join(message.get("content", "") for message in messages).lower()
    tags = {"assistant_style", "instruction_following"}
    if len(messages) > 2:
        tags.add("multi_turn_dialogue")
    if preference or "hh-rlhf" in repo:
        tags.update({"preference_boundary", "helpful_harmless"})
    if "oasst" in repo.lower():
        tags.add("human_assistant_dialogue")
    if "ultrachat" in repo.lower():
        tags.add("synthetic_multi_turn")
    if "slimorca" in repo.lower():
        tags.add("reasoning_instruction")
    if "no_robots" in repo.lower():
        tags.add("human_written_instruction")
    if any(token in text for token in ("summarize", "summary", "tl;dr")):
        tags.add("summarization")
    if any(token in text for token in ("python", "javascript", "code", "program", "function")):
        tags.add("code_instruction")
    if any(token in text for token in ("math", "equation", "calculate", "number", "solve")):
        tags.add("math_text_surface")
    if any(token in text for token in ("i don't", "i cannot", "can't help", "not able", "policy", "harmful", "safe")):
        tags.add("refusal_or_boundary")
    if "red_team" in family:
        tags.add("red_team_boundary")
    return sorted(tags)


def record_id(repo: str, filename: str, index: int, payload: str) -> str:
    digest = hashlib.sha256(f"{repo}:{filename}:{index}:{payload}".encode("utf-8")).hexdigest()[:24]
    prefix = re.sub(r"[^a-z0-9]+", "_", repo.lower()).strip("_")
    return f"{prefix}_{digest}"


def first_user_and_last_assistant(messages: list[dict[str, str]]) -> tuple[str, str]:
    first_user = next((message["content"] for message in messages if message.get("role") == "user"), "")
    assistants = [message["content"] for message in messages if message.get("role") == "assistant"]
    return first_user, assistants[-1] if assistants else ""


def normalized_row(
    *,
    repo: str,
    license_name: str,
    filename: str,
    source_index: int,
    family: str,
    split: str,
    messages: list[dict[str, str]],
    preference: dict[str, str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    messages = [
        {"role": normalize_role(message.get("role", "")), "content": clean_text(message.get("content", ""), 6000)}
        for message in messages
        if clean_text(message.get("content", ""), 6000)
    ]
    if not any(message["role"] == "user" for message in messages):
        return None
    if not any(message["role"] == "assistant" for message in messages):
        return None
    prompt, response = first_user_and_last_assistant(messages)
    rid = record_id(repo, filename, source_index, prompt + response)
    return {
        "record_id": rid,
        "source": repo,
        "license": license_name,
        "source_file": filename,
        "split": split,
        "family": family,
        "prompt": clean_text(prompt, 6000),
        "response": clean_text(response, 6000),
        "messages": messages,
        "preference": preference or {},
        "skill_tags": source_tags(repo, family, messages, bool(preference)),
        "metadata": metadata or {},
        "e136_import_role": "assistant_text_seed",
    }


def iter_message_parquet(path: Path, repo: str, license_name: str, filename: str, limit: int) -> Iterable[dict[str, Any]]:
    family = family_from_filename(repo, filename)
    split = split_from_filename(filename)
    emitted = 0
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=1000):
        for index, raw in enumerate(batch.to_pylist()):
            raw_messages = raw.get("messages") or []
            messages = [
                {"role": normalize_role(message.get("role", "")), "content": message.get("content", "")}
                for message in raw_messages
                if isinstance(message, dict)
            ]
            row = normalized_row(
                repo=repo,
                license_name=license_name,
                filename=filename,
                source_index=emitted + index,
                family=family,
                split=split,
                messages=messages,
                metadata={key: raw.get(key) for key in ("prompt_id", "category") if key in raw},
            )
            if row:
                yield row
                emitted += 1
                if emitted >= limit:
                    return


def iter_slimorca(path: Path, repo: str, license_name: str, filename: str, limit: int) -> Iterable[dict[str, Any]]:
    emitted = 0
    family = family_from_filename(repo, filename)
    with path.open("r", encoding="utf-8") as handle:
        for line_index, line in enumerate(handle):
            if not line.strip():
                continue
            raw = json.loads(line)
            messages = [
                {"role": normalize_role(item.get("from", "")), "content": item.get("value", "")}
                for item in raw.get("conversations", [])
                if isinstance(item, dict)
            ]
            row = normalized_row(
                repo=repo,
                license_name=license_name,
                filename=filename,
                source_index=line_index,
                family=family,
                split="train",
                messages=messages,
            )
            if row:
                yield row
                emitted += 1
                if emitted >= limit:
                    return


def split_hh_transcript(text: str) -> list[dict[str, str]]:
    parts = re.split(r"\n\n(Human|Assistant):", text)
    messages: list[dict[str, str]] = []
    if len(parts) < 3:
        return messages
    for role, content in zip(parts[1::2], parts[2::2]):
        messages.append({"role": "user" if role == "Human" else "assistant", "content": content.strip()})
    return messages


def iter_hh(path: Path, repo: str, license_name: str, filename: str, limit: int) -> Iterable[dict[str, Any]]:
    emitted = 0
    family = family_from_filename(repo, filename)
    split = split_from_filename(filename)
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line_index, line in enumerate(handle):
            if not line.strip():
                continue
            raw = json.loads(line)
            chosen = str(raw.get("chosen") or "")
            rejected = str(raw.get("rejected") or "")
            messages = split_hh_transcript(chosen)
            rejected_messages = split_hh_transcript(rejected)
            rejected_response = ""
            if rejected_messages:
                assistants = [message["content"] for message in rejected_messages if message["role"] == "assistant"]
                rejected_response = assistants[-1] if assistants else ""
            row = normalized_row(
                repo=repo,
                license_name=license_name,
                filename=filename,
                source_index=line_index,
                family=family,
                split=split,
                messages=messages,
                preference={"rejected_response": clean_text(rejected_response, 6000)} if rejected_response else {},
            )
            if row:
                yield row
                emitted += 1
                if emitted >= limit:
                    return


def iter_oasst2(path: Path, repo: str, license_name: str, filename: str, limit: int) -> Iterable[dict[str, Any]]:
    emitted = 0
    family = family_from_filename(repo, filename)
    split = split_from_filename(filename)
    rows: list[dict[str, Any]] = []
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=5000):
        rows.extend(batch.to_pylist())
    by_id = {row.get("message_id"): row for row in rows if row.get("message_id")}
    for row in rows:
        if emitted >= limit:
            return
        if row.get("role") != "assistant":
            continue
        if row.get("deleted") or row.get("review_result") is False:
            continue
        if str(row.get("lang") or "").lower() != "en":
            continue
        parent = by_id.get(row.get("parent_id"))
        if not parent or parent.get("role") != "prompter":
            continue
        if parent.get("deleted") or parent.get("review_result") is False:
            continue
        messages = [
            {"role": "user", "content": parent.get("text", "")},
            {"role": "assistant", "content": row.get("text", "")},
        ]
        normalized = normalized_row(
            repo=repo,
            license_name=license_name,
            filename=filename,
            source_index=emitted,
            family=family,
            split=split,
            messages=messages,
            metadata={
                "message_id": row.get("message_id"),
                "parent_id": row.get("parent_id"),
                "message_tree_id": row.get("message_tree_id"),
                "rank": row.get("rank"),
                "lang": row.get("lang"),
            },
        )
        if normalized:
            yield normalized
            emitted += 1


def download_pack(root: Path) -> dict[str, Any]:
    raw_root = root / "raw"
    raw_root.mkdir(parents=True, exist_ok=True)
    files: list[dict[str, Any]] = []
    started = time.time()
    for source in SOURCES:
        repo = source["repo"]
        target = repo_dir(raw_root, repo)
        target.mkdir(parents=True, exist_ok=True)
        for filename in source["files"]:
            path = Path(hf_hub_download(repo_id=repo, filename=filename, repo_type="dataset", local_dir=target))
            files.append({
                "repo": repo,
                "license": source["license"],
                "filename": filename,
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            })
    manifest = {
        "pack_id": PACK_ID,
        "created_at_unix": time.time(),
        "target_root": str(root),
        "raw_root": str(raw_root),
        "purpose": "Assistant/text seed pack for dialogue, instruction following, response style, helpful/harmless boundary, and safety/no-call operator farming.",
        "selection_boundary": "Raw dataset acquisition only; not model training, not benchmark pass, not production assistant evidence.",
        "sources": [
            {"repo": source["repo"], "license": source["license"], "role": source["role"], "file_count": len(source["files"])}
            for source in SOURCES
        ],
        "files": files,
        "total_size_bytes": sum(row["size_bytes"] for row in files),
        "elapsed_seconds": round(time.time() - started, 3),
    }
    write_json(root / "download_manifest.json", manifest)
    return manifest


def load_or_create_download_manifest(root: Path, skip_download: bool) -> dict[str, Any]:
    manifest_path = root / "download_manifest.json"
    if skip_download:
        if not manifest_path.exists():
            raise FileNotFoundError(f"missing {manifest_path}; run without --skip-download first")
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    return download_pack(root)


def normalize(root: Path, max_rows_total: int, source_limits: dict[str, int], output: Path) -> dict[str, Any]:
    download_manifest = json.loads((root / "download_manifest.json").read_text(encoding="utf-8"))
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        output.unlink()

    counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    family_counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    tag_counts: Counter[str] = Counter()
    bytes_written = 0
    rows_written = 0

    source_by_repo = {source["repo"]: source for source in SOURCES}
    files = [row for row in download_manifest["files"] if not row["filename"].endswith("README.md")]
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for file_row in files:
            if rows_written >= max_rows_total:
                break
            repo = file_row["repo"]
            source = source_by_repo[repo]
            remaining_source = max(0, source_limits.get(repo, 0) - source_counts[repo])
            if remaining_source <= 0:
                continue
            remaining_total = max_rows_total - rows_written
            limit = min(remaining_source, remaining_total)
            path = Path(file_row["path"])
            filename = file_row["filename"]
            if repo == "Open-Orca/SlimOrca":
                iterator = iter_slimorca(path, repo, source["license"], filename, limit)
            elif repo == "Anthropic/hh-rlhf":
                iterator = iter_hh(path, repo, source["license"], filename, limit)
            elif repo == "OpenAssistant/oasst2" and filename.endswith(".parquet"):
                iterator = iter_oasst2(path, repo, source["license"], filename, limit)
            elif repo in {"HuggingFaceH4/ultrachat_200k", "HuggingFaceH4/no_robots"} and filename.endswith(".parquet"):
                iterator = iter_message_parquet(path, repo, source["license"], filename, limit)
            else:
                continue
            before = rows_written
            for row in iterator:
                blob = json.dumps(row, ensure_ascii=False, sort_keys=True)
                handle.write(blob + "\n")
                bytes_written += len(blob.encode("utf-8")) + 1
                rows_written += 1
                counts["rows_written"] += 1
                source_counts[row["source"]] += 1
                family_counts[row["family"]] += 1
                split_counts[row["split"]] += 1
                for tag in row["skill_tags"]:
                    tag_counts[tag] += 1
                if rows_written >= max_rows_total or source_counts[repo] >= source_limits.get(repo, 0):
                    break
            counts[f"{repo}:{filename}:rows"] = rows_written - before

    first_hash_material: list[dict[str, Any]] = []
    with output.open("r", encoding="utf-8") as handle:
        for _, line in zip(range(256), handle):
            if line.strip():
                row = json.loads(line)
                first_hash_material.append({
                    "record_id": row["record_id"],
                    "source": row["source"],
                    "split": row["split"],
                    "family": row["family"],
                    "tags": row["skill_tags"],
                })

    manifest = {
        "pack_id": PACK_ID,
        "created_at_unix": time.time(),
        "input_download_manifest": str(root / "download_manifest.json"),
        "normalized_path": str(output),
        "row_count": rows_written,
        "bytes": output.stat().st_size if output.exists() else bytes_written,
        "sha256": sha256(output),
        "sha256_first_256_rows": hashlib.sha256(json.dumps(first_hash_material, sort_keys=True).encode("utf-8")).hexdigest(),
        "source_counts": dict(sorted(source_counts.items())),
        "family_counts": dict(sorted(family_counts.items())),
        "split_counts": dict(sorted(split_counts.items())),
        "tag_counts": dict(sorted(tag_counts.items())),
        "file_row_counts": dict(sorted(counts.items())),
        "source_limits": source_limits,
        "max_rows_total": max_rows_total,
        "license_notes": {
            "cc-by-nc-4.0": "No Robots rows are non-commercial; keep this pack for local research/evidence unless licensing is reviewed.",
            "raw_data": "target/ is gitignored; raw and normalized datasets are local artifacts, not repository commits.",
        },
    }
    write_json(root / "normalized_manifest.json", manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    parser.add_argument("--output", default=str(DEFAULT_NORMALIZED))
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--max-rows-total", type=int, default=460_000)
    for repo, value in DEFAULT_SOURCE_LIMITS.items():
        flag = "--limit-" + repo.split("/")[-1].lower().replace("_", "-")
        parser.add_argument(flag, type=int, default=value)
    args = parser.parse_args()

    root = Path(args.root)
    source_limits = {
        "HuggingFaceH4/ultrachat_200k": args.limit_ultrachat_200k,
        "Open-Orca/SlimOrca": args.limit_slimorca,
        "OpenAssistant/oasst2": args.limit_oasst2,
        "Anthropic/hh-rlhf": args.limit_hh_rlhf,
        "HuggingFaceH4/no_robots": args.limit_no_robots,
    }
    download_manifest = load_or_create_download_manifest(root, bool(args.skip_download))
    normalized_manifest = normalize(root, args.max_rows_total, source_limits, Path(args.output))
    print(json.dumps({
        "download_total_gib": round(download_manifest["total_size_bytes"] / 1024**3, 3),
        "normalized": normalized_manifest["normalized_path"],
        "normalized_gib": round(normalized_manifest["bytes"] / 1024**3, 3),
        "rows": normalized_manifest["row_count"],
        "sources": normalized_manifest["source_counts"],
    }, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

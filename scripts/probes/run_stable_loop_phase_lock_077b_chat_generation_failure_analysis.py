#!/usr/bin/env python3
"""Analysis-only failure attribution for STABLE_LOOP_PHASE_LOCK_077B."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]

REQUIRED_077_ARTIFACTS = [
    "summary.json",
    "generation_samples.jsonl",
    "human_readable_samples.jsonl",
    "composition_metrics.json",
    "novelty_metrics.json",
    "collapse_metrics.json",
    "finite_label_retention_metrics.json",
]

REQUIRED_076_ARTIFACTS = [
    "summary.json",
    "generation_samples.jsonl",
    "chat_sft_dataset_manifest.json",
    "checkpoint_manifest.json",
    "checkpoints/chat_generation_poc/model_checkpoint.json",
]

SOURCE_LABELS = [
    "exact_response_table_copy",
    "exact_train_response_copy",
    "exact_eval_response_copy",
    "semantic_template_copy",
    "finite_label_retention_label",
    "context_slot_not_bound",
    "boundary_refusal_not_selected",
    "wrong_template_family_selected",
    "prompt_copy",
    "unknown_source",
]

POSITIVE_VERDICTS = [
    "CHAT_GENERATION_FAILURE_ANALYSIS_POSITIVE",
    "UPSTREAM_077_FAILURE_PROFILE_LOADED",
    "TEMPLATE_COPY_SOURCE_ATTRIBUTION_WRITTEN",
    "FRESH_CONTEXT_CARRY_FAILURE_ANALYZED",
    "BOUNDARY_REFUSAL_FAILURE_ANALYZED",
    "RESPONSE_TABLE_DEPENDENCE_CONFIRMED",
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
            try:
                value = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSONL row: {exc}") from exc
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


def rate(numerator: int, denominator: int) -> float:
    return 0.0 if denominator == 0 else numerator / denominator


def normalize_response(value: str) -> str:
    return " ".join(tokenize(value))


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


def decode_tokens(tokens: list[str]) -> str:
    out: list[str] = []
    for token in tokens:
        if token == "<eos>":
            break
        out.append(str(token).lower())
    return " ".join(out)


def init_run(out: Path, args: argparse.Namespace) -> None:
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "queue.json", {
        "schema_version": "chat_generation_failure_analysis_queue_v1",
        "milestone": "STABLE_LOOP_PHASE_LOCK_077B_CHAT_GENERATION_FAILURE_ANALYSIS",
        "analysis_only": True,
        "no_training": True,
        "no_new_inference": True,
        "no_checkpoint_repair": True,
        "created_at": now_iso(),
    })
    write_json(out / "analysis_config.json", {
        "schema_version": "chat_generation_failure_analysis_config_v1",
        "out": normalized_path(out),
        "upstream_077_root": normalized_path(args.upstream_077_root),
        "upstream_076_root": normalized_path(args.upstream_076_root),
        "heartbeat_sec": args.heartbeat_sec,
        "source_labels": SOURCE_LABELS,
        "unknown_source_rate_limit": 0.10,
        "template_copy_source_coverage_min": 0.90,
        "analysis_only": True,
        "train_step_count": 0,
    })
    write_json(out / "summary.json", {
        "schema_version": "chat_generation_failure_analysis_summary_v1",
        "status": "running",
        "analysis_only": True,
        "training_performed": False,
        "new_inference_performed": False,
        "checkpoint_repaired": False,
        "verdicts": [],
    })
    write_report(out, {
        "status": "running",
        "verdicts": [],
        "headline": "077B analysis running",
    })
    append_progress(out, "start", {"analysis_only": True})


def missing_artifacts(root: Path, names: list[str]) -> list[str]:
    missing: list[str] = []
    for name in names:
        path = root / name
        if not path.exists():
            missing.append(normalized_path(path))
    return missing


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


def fail(out: Path, verdict: str, message: str, extra: dict[str, Any] | None = None) -> int:
    payload = {
        "schema_version": "chat_generation_failure_analysis_summary_v1",
        "status": "failed",
        "analysis_only": True,
        "training_performed": False,
        "new_inference_performed": False,
        "checkpoint_repaired": False,
        "error": {"code": verdict, "message": message},
        "extra": extra or {},
        "verdicts": ["CHAT_GENERATION_FAILURE_ANALYSIS_FAILS", verdict, "NO_TRAINING_PERFORMED"],
    }
    write_json(out / "summary.json", payload)
    write_report(out, payload)
    append_progress(out, "done", {"status": "failed", "verdict": verdict})
    print(json.dumps(payload, separators=(",", ":")))
    return 1


def load_response_table(checkpoint: dict[str, Any]) -> tuple[dict[str, str], dict[str, str]]:
    label_to_response: dict[str, str] = {}
    response_to_label: dict[str, str] = {}
    for label, tokens in checkpoint.get("response_table", {}).items():
        if isinstance(tokens, list):
            decoded = decode_tokens([str(token) for token in tokens])
            normalized = normalize_response(decoded)
            label_to_response[str(label)] = normalized
            response_to_label.setdefault(normalized, str(label))
    return label_to_response, response_to_label


def build_source_sets(
    root077: Path,
    root076: Path,
    response_by_label: dict[str, str],
) -> dict[str, set[str]]:
    response_table = set(response_by_label.values())
    eval_outputs = {
        normalize_response(str(row.get("model_output", "")))
        for row in read_jsonl(root076 / "generation_samples.jsonl")
    }
    train_responses = set(response_table)
    train_sample = root076 / "train_examples_sample.jsonl"
    if train_sample.exists():
        for row in read_jsonl(train_sample):
            if "response_text" in row:
                train_responses.add(normalize_response(str(row["response_text"])))
    exact_templates = set(response_table) | set(eval_outputs) | set(train_responses)
    prompt_outputs = {
        normalize_response(str(row.get("model_output", "")))
        for row in read_jsonl(root077 / "generation_samples.jsonl")
    }
    return {
        "response_table": response_table,
        "eval_outputs": eval_outputs,
        "train_responses": train_responses,
        "exact_templates": exact_templates,
        "prompt_outputs": prompt_outputs,
    }


def classify_row(row: dict[str, Any], source_sets: dict[str, set[str]], response_to_label: dict[str, str]) -> dict[str, Any]:
    family = str(row.get("eval_family", ""))
    prompt = str(row.get("prompt", ""))
    output = str(row.get("model_output", ""))
    normalized = normalize_response(output)
    required = [str(item) for item in row.get("required_keywords", [])]
    lower_output = output.lower()
    missing_keywords = [kw for kw in required if kw.lower() not in lower_output]
    copied_template = normalized if normalized in source_sets["exact_templates"] else None
    selected_label = response_to_label.get(normalized)

    if output and output in prompt:
        source = "prompt_copy"
    elif family == "FINITE_LABEL_ANCHORROUTE_RETENTION":
        source = "finite_label_retention_label"
    elif family == "FRESH_CONTEXT_CARRY_CHAT" and row.get("pass_fail") != "pass":
        source = "context_slot_not_bound"
    elif family == "FRESH_BOUNDARY_REFUSAL_MINI" and row.get("pass_fail") != "pass":
        source = "boundary_refusal_not_selected"
    elif row.get("pass_fail") != "pass" and selected_label is not None:
        source = "wrong_template_family_selected"
    elif normalized in source_sets["response_table"]:
        source = "exact_response_table_copy"
    elif normalized in source_sets["train_responses"]:
        source = "exact_train_response_copy"
    elif normalized in source_sets["eval_outputs"]:
        source = "exact_eval_response_copy"
    elif any(overlap_tokens(normalized, template) >= 0.80 for template in source_sets["exact_templates"]):
        source = "semantic_template_copy"
    else:
        source = "unknown_source"

    return {
        "eval_family": family,
        "prompt": prompt,
        "model_output": output,
        "expected_behavior": row.get("expected_behavior"),
        "classified_source": source,
        "copied_template_if_any": copied_template,
        "selected_template_label_or_response": selected_label or copied_template,
        "required_keywords": required,
        "missing_keywords": missing_keywords,
        "pass_fail": row.get("pass_fail"),
        "short_diagnosis": row.get("short_diagnosis") or row.get("diagnosis") or source,
        "template_copy_flag": bool(row.get("template_copy_flag", copied_template is not None)),
        "novelty_flag": bool(row.get("novelty_flag", False)),
    }


def overlap_tokens(a: str, b: str) -> float:
    left = set(tokenize(a))
    right = set(tokenize(b))
    if not left:
        return 0.0
    return len(left & right) / len(left)


def source_reports(classified: list[dict[str, Any]], summary077: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    total = len(classified)
    source_counts = Counter(row["classified_source"] for row in classified)
    template_rows = [
        row for row in classified
        if row["copied_template_if_any"] is not None or row["classified_source"] in {
            "exact_response_table_copy",
            "exact_train_response_copy",
            "exact_eval_response_copy",
            "semantic_template_copy",
        }
    ]
    family_counts: dict[str, Counter[str]] = defaultdict(Counter)
    template_counts = Counter()
    for row in classified:
        family_counts[row["eval_family"]][row["classified_source"]] += 1
        if row["copied_template_if_any"]:
            template_counts[row["copied_template_if_any"]] += 1

    copy_rate_by_family = {
        family: rate(
            sum(count for source, count in counts.items() if source != "unknown_source"),
            sum(counts.values()),
        )
        for family, counts in family_counts.items()
    }
    novelty = summary077.get("novelty_metrics", {})
    response_table_report = {
        "schema_version": "chat_generation_response_table_dependence_report_v1",
        "exact_train_response_copy_rate": novelty.get("exact_train_response_copy_rate"),
        "exact_eval_response_copy_rate": novelty.get("exact_eval_response_copy_rate"),
        "response_table_copy_rate": novelty.get("response_table_copy_rate"),
        "template_copy_rate": novelty.get("template_copy_rate"),
        "novel_response_rate": novelty.get("novel_response_rate"),
        "train_response_ngram_overlap": novelty.get("train_response_ngram_overlap"),
        "top_copied_templates": [
            {"template": template, "count": count}
            for template, count in template_counts.most_common(10)
        ],
        "copy_rate_by_eval_family": copy_rate_by_family,
    }
    template_report = {
        "schema_version": "chat_generation_template_copy_source_report_v1",
        "total_rows": total,
        "source_counts": dict(source_counts),
        "unknown_source_rate": rate(source_counts.get("unknown_source", 0), total),
        "template_copy_source_coverage": rate(total - source_counts.get("unknown_source", 0), total),
        "template_attributed_rows": len(template_rows),
        "source_labels": SOURCE_LABELS,
    }
    return template_report, response_table_report


def context_report(classified: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [row for row in classified if row["eval_family"] == "FRESH_CONTEXT_CARRY_CHAT"]
    failures = [row for row in rows if row["pass_fail"] != "pass"]
    slot_expected = []
    for row in failures:
        expected = [kw for kw in row["required_keywords"] if kw not in {"active", "context"}]
        slot_expected.append(expected[0] if expected else None)
    wrong_template = [
        row for row in failures
        if not str(row.get("selected_template_label_or_response", "")).startswith("resp_context_")
    ]
    return {
        "schema_version": "fresh_context_carry_failure_report_v1",
        "context_carry_failure_count": len(failures),
        "context_carry_row_count": len(rows),
        "context_slot_expected": slot_expected,
        "context_slot_model_output": [row["model_output"] for row in failures],
        "selected_template_label_or_response": [
            row["selected_template_label_or_response"] for row in failures
        ],
        "slot_binding_miss_rate": rate(len(failures), len(rows)),
        "wrong_template_family_rate": rate(len(wrong_template), len(failures)),
        "analysis_complete": bool(rows),
    }


def boundary_report(classified: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [row for row in classified if row["eval_family"] == "FRESH_BOUNDARY_REFUSAL_MINI"]
    failures = [row for row in rows if row["pass_fail"] != "pass"]
    boundary_template = [
        row for row in failures
        if row.get("selected_template_label_or_response") == "resp_boundary"
    ]
    wrong_template = [
        row for row in failures
        if row.get("selected_template_label_or_response") != "resp_boundary"
    ]
    return {
        "schema_version": "boundary_refusal_failure_report_v1",
        "boundary_failure_count": len(failures),
        "boundary_row_count": len(rows),
        "expected_refusal_keywords": [row["required_keywords"] for row in failures],
        "selected_template_label_or_response": [
            row["selected_template_label_or_response"] for row in failures
        ],
        "boundary_template_selection_rate": rate(len(boundary_template), len(failures)),
        "wrong_template_family_rate": rate(len(wrong_template), len(failures)),
        "analysis_complete": bool(rows),
    }


def failure_cluster_report(classified: list[dict[str, Any]]) -> dict[str, Any]:
    by_family: dict[str, Counter[str]] = defaultdict(Counter)
    for row in classified:
        by_family[row["eval_family"]][row["classified_source"]] += 1
    return {
        "schema_version": "chat_generation_failure_cluster_report_v1",
        "clusters_by_eval_family": {
            family: dict(counts)
            for family, counts in sorted(by_family.items())
        },
        "failed_rows": [
            row for row in classified
            if row["pass_fail"] != "pass"
        ],
    }


def repair_recommendation() -> dict[str, Any]:
    return {
        "schema_version": "chat_generation_repair_recommendation_v1",
        "next_milestone": "078_CHAT_COMPOSITION_REPAIR",
        "use token-level next-token objective": True,
        "reduce response_table dependence": True,
        "add paraphrase / many-target variants": True,
        "add fresh composition curriculum": True,
        "add context carry variable-slot training": True,
        "add boundary refusal paraphrase variants": True,
        "add template dropout": True,
        "add semantic slot recombination": True,
        "retain finite-label AnchorRoute scenario-state eval": True,
        "keep no product API / no SDK / no service surface": True,
        "do not claim GPT-like assistant readiness": True,
        "adding more table responses alone is not enough": True,
    }


def write_report(out: Path, summary: dict[str, Any]) -> None:
    text = "# STABLE_LOOP_PHASE_LOCK_077B_CHAT_GENERATION_FAILURE_ANALYSIS Report\n\n"
    text += f"Status: `{summary.get('status', 'unknown')}`\n\n"
    text += "077B does not improve the model\n"
    text += "no training performed\n"
    text += "no new inference performed\n"
    text += "no checkpoint repaired\n"
    text += "no production chat\n"
    text += "no GPT-like assistant readiness\n\n"
    text += "## Verdicts\n\n"
    for verdict in summary.get("verdicts", []):
        text += f"- `{verdict}`\n"
    text += "\n## Summary JSON\n\n```json\n"
    text += json.dumps(summary, indent=2, sort_keys=True)
    text += "\n```\n"
    write_path = out / "report.md"
    write_path.parent.mkdir(parents=True, exist_ok=True)
    write_path.write_text(text, encoding="utf-8")


def run(args: argparse.Namespace) -> int:
    out = args.out
    init_run(out, args)

    missing077 = missing_artifacts(args.upstream_077_root, REQUIRED_077_ARTIFACTS)
    write_json(
        out / "upstream_077_manifest.json",
        artifact_manifest(
            args.upstream_077_root,
            REQUIRED_077_ARTIFACTS,
            "chat_generation_failure_analysis_upstream_077_manifest_v1",
        ),
    )
    if missing077:
        return fail(out, "UPSTREAM_077_ARTIFACT_MISSING", "required 077 artifacts missing", {"missing": missing077})

    missing076 = missing_artifacts(args.upstream_076_root, REQUIRED_076_ARTIFACTS)
    write_json(
        out / "upstream_076_manifest.json",
        artifact_manifest(
            args.upstream_076_root,
            REQUIRED_076_ARTIFACTS,
            "chat_generation_failure_analysis_upstream_076_manifest_v1",
        ),
    )
    if missing076:
        return fail(out, "UPSTREAM_076_ARTIFACT_MISSING", "required 076 artifacts missing", {"missing": missing076})

    append_progress(out, "upstreams_loaded", {"upstream_077": True, "upstream_076": True})

    summary077 = read_json(args.upstream_077_root / "summary.json")
    samples077 = read_jsonl(args.upstream_077_root / "generation_samples.jsonl")
    novelty077 = read_json(args.upstream_077_root / "novelty_metrics.json")
    summary076 = read_json(args.upstream_076_root / "summary.json")
    checkpoint = read_json(args.upstream_076_root / "checkpoints/chat_generation_poc/model_checkpoint.json")
    response_by_label, response_to_label = load_response_table(checkpoint)
    source_sets = build_source_sets(args.upstream_077_root, args.upstream_076_root, response_by_label)

    if not samples077:
        return fail(out, "FAILURE_CASE_INPUT_MISSING", "077 generation_samples.jsonl has no rows")

    classified = [classify_row(row, source_sets, response_to_label) for row in samples077]
    template_report, response_table_report = source_reports(classified, summary077)
    context = context_report(classified)
    boundary = boundary_report(classified)
    clusters = failure_cluster_report(classified)
    recommendation = repair_recommendation()

    # Keep observed upstream novelty values authoritative.
    response_table_report.update({
        "exact_train_response_copy_rate": novelty077.get("exact_train_response_copy_rate"),
        "exact_eval_response_copy_rate": novelty077.get("exact_eval_response_copy_rate"),
        "response_table_copy_rate": novelty077.get("response_table_copy_rate"),
        "template_copy_rate": novelty077.get("template_copy_rate"),
        "novel_response_rate": novelty077.get("novel_response_rate"),
        "train_response_ngram_overlap": novelty077.get("train_response_ngram_overlap"),
    })

    write_json(out / "template_copy_source_report.json", template_report)
    write_json(out / "fresh_context_carry_failure_report.json", context)
    write_json(out / "boundary_refusal_failure_report.json", boundary)
    write_json(out / "response_table_dependence_report.json", response_table_report)
    write_json(out / "failure_cluster_report.json", clusters)
    write_json(out / "repair_recommendation.json", recommendation)

    for row in classified:
        append_jsonl(out / "prompt_to_template_mapping.jsonl", row)
        append_jsonl(out / "human_failure_digest.jsonl", {
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

    append_progress(
        out,
        "analysis_written",
        {
            "unknown_source_rate": template_report["unknown_source_rate"],
            "template_copy_source_coverage": template_report["template_copy_source_coverage"],
            "context_carry_failure_count": context["context_carry_failure_count"],
            "boundary_failure_count": boundary["boundary_failure_count"],
        },
    )

    failures: list[str] = []
    if template_report["unknown_source_rate"] > 0.10:
        failures.append("UNKNOWN_SOURCE_RATE_TOO_HIGH")
    if template_report["template_copy_source_coverage"] < 0.90:
        failures.append("TEMPLATE_COPY_ANALYSIS_INCOMPLETE")
    if not context.get("analysis_complete") or "slot_binding_miss_rate" not in context:
        failures.append("CONTEXT_CARRY_ANALYSIS_INCOMPLETE")
    if not boundary.get("analysis_complete") or "boundary_template_selection_rate" not in boundary:
        failures.append("BOUNDARY_REFUSAL_ANALYSIS_INCOMPLETE")
    if not response_table_report.get("top_copied_templates"):
        failures.append("TEMPLATE_COPY_ANALYSIS_INCOMPLETE")
    if not recommendation.get("next_milestone"):
        failures.append("REPAIR_RECOMMENDATION_MISSING")
    digest_rows = len(classified)
    if digest_rows == 0:
        failures.append("FAILURE_CASE_INPUT_MISSING")

    if failures:
        verdicts = ["CHAT_GENERATION_FAILURE_ANALYSIS_FAILS", *dict.fromkeys(failures), "NO_TRAINING_PERFORMED"]
        status = "failed"
    else:
        verdicts = POSITIVE_VERDICTS.copy()
        status = "passed"

    payload = {
        "schema_version": "chat_generation_failure_analysis_summary_v1",
        "status": status,
        "verdicts": verdicts,
        "analysis_only": True,
        "training_performed": False,
        "new_inference_performed": False,
        "checkpoint_repaired": False,
        "model_capability_improved": False,
        "upstream_077_status": summary077.get("status"),
        "upstream_076_status": summary076.get("status"),
        "unknown_source_rate": template_report["unknown_source_rate"],
        "template_copy_source_coverage": template_report["template_copy_source_coverage"],
        "human_failure_digest_rows": digest_rows,
        "template_copy_rate": response_table_report["template_copy_rate"],
        "response_table_copy_rate": response_table_report["response_table_copy_rate"],
        "novel_response_rate": response_table_report["novel_response_rate"],
        "context_carry_failure_count": context["context_carry_failure_count"],
        "context_slot_binding_miss_rate": context["slot_binding_miss_rate"],
        "boundary_failure_count": boundary["boundary_failure_count"],
        "boundary_wrong_template_family_rate": boundary["wrong_template_family_rate"],
        "next_milestone": recommendation["next_milestone"],
        "no_production_chat": True,
        "no_GPT_like_assistant_readiness": True,
    }
    write_json(out / "summary.json", payload)
    write_report(out, payload)
    append_progress(out, "done", {"status": status, "verdicts": verdicts})
    print(json.dumps(payload, separators=(",", ":")))
    return 0 if status == "passed" else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--upstream-077-root", type=Path, required=True)
    parser.add_argument("--upstream-076-root", type=Path, required=True)
    parser.add_argument("--heartbeat-sec", type=int, default=20)
    return parser.parse_args()


def main() -> int:
    try:
        return run(parse_args())
    except Exception as exc:  # noqa: BLE001 - runner must write useful failure details.
        out = Path(sys.argv[sys.argv.index("--out") + 1]) if "--out" in sys.argv else Path(".")
        out.mkdir(parents=True, exist_ok=True)
        return fail(out, "CHAT_GENERATION_FAILURE_ANALYSIS_FAILS", str(exc))


if __name__ == "__main__":
    sys.exit(main())

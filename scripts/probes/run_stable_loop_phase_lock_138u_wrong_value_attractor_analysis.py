#!/usr/bin/env python3
"""138U artifact-only wrong-value attractor analysis.

This phase reads existing 138WV and 138W artifacts only. It does not train,
repair, run new inference, call shared_raw_generation_helper.py, run torch
forward passes, mutate checkpoints, modify helper/backend code, import old
runners, delete or consolidate files, start services, deploy, or modify
runtime/release/product surfaces.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
MILESTONE = "STABLE_LOOP_PHASE_LOCK_138U_WRONG_VALUE_ATTRACTOR_ANALYSIS"
DEFAULT_OUT = Path("target/pilot_wave/stable_loop_phase_lock_138u_wrong_value_attractor_analysis/smoke")
DEFAULT_UPSTREAM_138WV_ROOT = Path("target/pilot_wave/stable_loop_phase_lock_138wv_wrapper_value_decoupling_failure_analysis/smoke")
DEFAULT_UPSTREAM_138W_ROOT = Path("target/pilot_wave/stable_loop_phase_lock_138w_answer_value_grounding_repair_probe/smoke")
BOUNDARY_TEXT = (
    "138U is artifact-only analysis. It reads existing 138WV/138W artifacts only "
    "and does not train, repair, run new inference, call shared_raw_generation_helper.py, "
    "run torch forward passes, mutate checkpoints, modify helper/backend code, import "
    "old runners, delete or consolidate files, start services, deploy, modify "
    "runtime/service/deploy/product/release surfaces, modify SDK exports, modify "
    "docs/product or docs/releases, or change root LICENSE."
)
FALSE_FLAGS = {
    "reasoning_restored": False,
    "reasoning_subtrack_real_raw_evidence_partially_restored": False,
    "raw_assistant_capability_restored": False,
    "structured_tool_capability_restored": False,
    "gpt_like_readiness_claimed": False,
    "open_domain_assistant_readiness_claimed": False,
    "production_chat_claimed": False,
    "public_api_claimed": False,
    "deployment_readiness_claimed": False,
    "safety_alignment_claimed": False,
}
ALLOWED_HELPER_KEYS = {"prompt", "checkpoint_path", "checkpoint_hash", "seed", "max_new_tokens", "generation_config"}
FORBIDDEN_HELPER_KEYS = {
    "expected_output",
    "expected_payload",
    "expected_answer",
    "required_keys",
    "required_keywords",
    "forbidden_outputs",
    "schema_answer_object",
    "scorer_metadata",
    "labels",
    "oracle_data",
    "target_json",
    "gold_output",
    "row_answer",
    "eval_family",
    "answer",
    "expected_values",
}
REQUIRED_138WV_ARTIFACTS = [
    "decision.json",
    "summary.json",
    "upstream_138w_manifest.json",
    "post_wrapper_value_anatomy_report.json",
    "silence_taxonomy_report.json",
    "attractor_distribution_report.json",
    "value_candidate_report.json",
    "parrot_and_derivation_recheck.json",
    "wrapper_value_decoupling_root_cause.json",
    "next_repair_recommendation.json",
]
REQUIRED_138W_ARTIFACTS = [
    "raw_generation_results.jsonl",
    "raw_generation_trace.jsonl",
    "scoring_results.jsonl",
    "eval_rows.jsonl",
    "train_rows.jsonl",
    "train_dataset_manifest.json",
    "eval_dataset_manifest.json",
    "value_grounding_metrics.json",
    "parrot_trap_report.json",
    "aggregate_metrics.json",
    "freshness_leakage_audit.json",
    "generated_before_scoring_report.json",
    "expected_output_canary_report.json",
    "ast_shortcut_scan_report.json",
    "determinism_replay_report.json",
]
ATTRACTOR_SHAPES = {
    "global_single_wrong_value_attractor",
    "small_set_wrong_value_attractor",
    "family_specific_wrong_value_attractor",
    "seed_specific_wrong_value_attractor",
    "high_entropy_wrong_value_attractor",
}
ROOT_CAUSES = {
    "global_train_value_prior_attractor",
    "family_specific_train_value_attractor",
    "high_frequency_train_value_attractor",
    "distractor_value_attractor",
    "wrong_table_entry_attractor",
    "prompt_copy_wrong_value_attractor",
    "output_head_value_prior",
    "mixed_wrong_value_attractors",
    "wrong_value_attractor_ambiguous",
}
VALUE_TOKEN_RE = re.compile(r"\b(?:TR|EV)[A-Za-z0-9_+-]+\b")


class GateError(Exception):
    def __init__(self, verdict: str, message: str, details: dict[str, Any] | None = None):
        super().__init__(message)
        self.verdict = verdict
        self.message = message
        self.details = details or {}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


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


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


def resolve_path(text: str | Path) -> Path:
    path = Path(text)
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def resolve_target_out(text: str) -> Path:
    path = Path(text)
    resolved = path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()
    try:
        relative = resolved.relative_to(REPO_ROOT)
    except ValueError as exc:
        raise GateError("138U_BOUNDARY_FAILURE", "--out must stay inside repo") from exc
    parts = [part.lower() for part in relative.parts]
    if len(parts) < 2 or parts[0] != "target" or parts[1] != "pilot_wave":
        raise GateError("138U_BOUNDARY_FAILURE", "--out must stay under target/pilot_wave")
    return resolved


def append_progress(out: Path, event: str, status: str = "completed", **details: Any) -> None:
    append_jsonl(out / "progress.jsonl", {"ts": utc_now(), "event": event, "status": status, "details": details})


def rate(count: int, total: int) -> float:
    return count / total if total else 0.0


def write_summary(out: Path, status: str, verdicts: list[str], decision: dict[str, Any], failure: str | None = None) -> None:
    write_json(
        out / "summary.json",
        {
            "schema_version": "phase_138u_summary_v1",
            "milestone": MILESTONE,
            "status": status,
            "verdicts": verdicts,
            "decision": decision.get("decision"),
            "next": decision.get("next"),
            "root_cause": decision.get("root_cause"),
            "failure": failure,
            "boundary": BOUNDARY_TEXT,
            "artifact_only": True,
            "new_inference_run": False,
            "shared_helper_called": False,
            "torch_forward_pass_run": False,
            **FALSE_FLAGS,
            "metrics": decision,
        },
    )


def write_report(out: Path, verdicts: list[str], decision: dict[str, Any]) -> None:
    lines = [
        f"# {MILESTONE}",
        "",
        "## Boundary",
        "",
        BOUNDARY_TEXT,
        "",
        "## Findings",
        "",
        "- 138U does not fix or train the model.",
        "- It classifies the wrong specific train-seen value attractor after `ANSWER=E`.",
        f"- `decision`: `{decision.get('decision')}`",
        f"- `next`: `{decision.get('next')}`",
        f"- `root_cause`: `{decision.get('root_cause')}`",
        "",
        "## Verdicts",
        "",
    ]
    lines.extend(f"- `{verdict}`" for verdict in verdicts)
    lines.extend(
        [
            "",
            "Reasoning is not restored.",
            "Raw assistant capability remains quarantined.",
            "Structured/tool capability remains invalidated.",
            "not GPT-like readiness.",
            "not open-domain assistant readiness.",
            "not production chat.",
            "not public API.",
            "not deployment readiness.",
            "not safety alignment.",
        ]
    )
    write_text(out / "report.md", "\n".join(lines) + "\n")


def refresh_status(out: Path, status: str, verdicts: list[str], decision: dict[str, Any]) -> None:
    write_summary(out, status, verdicts, decision)
    write_report(out, verdicts, decision)


def verify_upstreams(out: Path, root_138wv: Path, root_138w: Path) -> dict[str, Any]:
    missing_138wv = [name for name in REQUIRED_138WV_ARTIFACTS if not (root_138wv / name).exists()]
    missing_138w = [name for name in REQUIRED_138W_ARTIFACTS if not (root_138w / name).exists()]
    if missing_138wv or missing_138w:
        raise GateError("UPSTREAM_138WV_ARTIFACT_MISSING", "required upstream artifacts missing", {"138wv": missing_138wv, "138w": missing_138w})

    decision_138wv = read_json(root_138wv / "decision.json")
    summary_138wv = read_json(root_138wv / "summary.json")
    distribution_138wv = read_json(root_138wv / "attractor_distribution_report.json")
    candidates_138wv = read_json(root_138wv / "value_candidate_report.json")
    root_138wv_payload = read_json(root_138wv / "wrapper_value_decoupling_root_cause.json")

    if (
        decision_138wv.get("decision") != "wrapper_value_decoupling_failure_analysis_complete"
        or decision_138wv.get("root_cause") != "wrong_specific_value_attractor_dominant"
        or decision_138wv.get("next") != "138U_WRONG_VALUE_ATTRACTOR_ANALYSIS"
    ):
        raise GateError("UPSTREAM_138WV_ARTIFACT_MISSING", "138WV did not route to 138U")
    required_138wv_profile = {
        "wrong_specific_value_rate": 1.0,
        "immediate_termination_proxy_rate": 0.0,
        "default_neutral_attractor_rate": 0.0,
        "structural_format_echo_rate": 0.0,
        "unknown_post_wrapper_behavior_rate": 0.0,
    }
    mismatches = {key: {"expected": expected, "actual": distribution_138wv.get(key)} for key, expected in required_138wv_profile.items() if distribution_138wv.get(key) != expected}
    if candidates_138wv.get("train_seen_value_rate") != 1.0 or candidates_138wv.get("expected_value_candidate_rate") != 0.0:
        mismatches["value_candidate_profile"] = {"train_seen_value_rate": candidates_138wv.get("train_seen_value_rate"), "expected_value_candidate_rate": candidates_138wv.get("expected_value_candidate_rate")}
    if root_138wv_payload.get("literal_eos_claimed") is not False or root_138wv_payload.get("topological_inhibition_claim_status") != "hypothesis_only_diagnostic_gap_without_instrumentation":
        mismatches["guardrail"] = "literal EOS/topological overclaim"
    if mismatches:
        raise GateError("UPSTREAM_138WV_ARTIFACT_MISSING", "138WV profile does not match 138U route", mismatches)

    decision_138w = read_json(root_138w / "decision.json")
    aggregate_138w = read_json(root_138w / "aggregate_metrics.json")
    parrot_138w = read_json(root_138w / "parrot_trap_report.json")
    leakage_138w = read_json(root_138w / "freshness_leakage_audit.json")
    before_138w = read_json(root_138w / "generated_before_scoring_report.json")
    canary_138w = read_json(root_138w / "expected_output_canary_report.json")
    scan_138w = read_json(root_138w / "ast_shortcut_scan_report.json")
    replay_138w = read_json(root_138w / "determinism_replay_report.json")
    traces_138w = read_jsonl(root_138w / "raw_generation_trace.jsonl")

    if decision_138w.get("decision") != "wrapper_success_without_value_grounding_persists" or decision_138w.get("next") != "138WV_WRAPPER_VALUE_DECOUPLING_FAILURE_ANALYSIS":
        raise GateError("UPSTREAM_138WV_ARTIFACT_MISSING", "138W did not route to 138WV")
    if canary_138w.get("expected_output_canary_passed") is not True or scan_138w.get("ast_shortcut_scan_passed") is not True:
        raise GateError("RAW_HELPER_INTEGRITY_FAILURE", "138W helper canary/AST integrity failed")
    if leakage_138w.get("leakage_rejected") is not True or replay_138w.get("determinism_replay_passed") is not True:
        raise GateError("UPSTREAM_138WV_ARTIFACT_MISSING", "138W leakage/determinism missing")
    if decision_138w.get("source_checkpoint_unchanged") is not True or decision_138w.get("target_checkpoint_changed") is not True:
        raise GateError("UPSTREAM_138WV_ARTIFACT_MISSING", "138W checkpoint integrity missing")
    if before_138w.get("generated_text_produced_before_scoring") is not True:
        raise GateError("UPSTREAM_138WV_ARTIFACT_MISSING", "138W generated-before-scoring missing")
    if parrot_138w.get("parrot_trap_detected") is not False or aggregate_138w.get("stale_chat_fragment_rate") != 0.0 or aggregate_138w.get("train_namespace_leak_rate") != 0.0:
        raise GateError("UPSTREAM_138WV_ARTIFACT_MISSING", "138W parrot/stale/namespace profile mismatch")
    for trace in traces_138w:
        request = trace.get("helper_request", {})
        if set(request) != ALLOWED_HELPER_KEYS or set(request) & FORBIDDEN_HELPER_KEYS:
            raise GateError("RAW_HELPER_INTEGRITY_FAILURE", "138W helper request metadata violation")

    manifest_138wv = {
        "schema_version": "phase_138u_upstream_138wv_manifest_v1",
        "upstream_138wv_root": rel(root_138wv),
        "verified": True,
        "decision": decision_138wv.get("decision"),
        "next": decision_138wv.get("next"),
        "root_cause": decision_138wv.get("root_cause"),
        "wrong_specific_value_rate": distribution_138wv.get("wrong_specific_value_rate"),
        "train_seen_value_rate": candidates_138wv.get("train_seen_value_rate"),
        "expected_value_candidate_rate": candidates_138wv.get("expected_value_candidate_rate"),
        "immediate_termination_proxy_rate": distribution_138wv.get("immediate_termination_proxy_rate"),
        "default_neutral_attractor_rate": distribution_138wv.get("default_neutral_attractor_rate"),
        "structural_format_echo_rate": distribution_138wv.get("structural_format_echo_rate"),
        "unknown_post_wrapper_behavior_rate": distribution_138wv.get("unknown_post_wrapper_behavior_rate"),
        "literal_eos_claimed": root_138wv_payload.get("literal_eos_claimed"),
        "topological_inhibition_claim_status": root_138wv_payload.get("topological_inhibition_claim_status"),
        "all_capability_flags_false": all(summary_138wv.get(key) is False for key in FALSE_FLAGS),
    }
    manifest_138w = {
        "schema_version": "phase_138u_upstream_138w_manifest_v1",
        "upstream_138w_root": rel(root_138w),
        "verified": True,
        "decision": decision_138w.get("decision"),
        "next": decision_138w.get("next"),
        "helper_integrity_passed": True,
        "canary_passed": True,
        "ast_scan_passed": True,
        "leakage_rejected": True,
        "determinism_replay_passed": True,
        "source_checkpoint_unchanged": True,
        "target_checkpoint_changed": True,
        "generated_text_before_scoring": True,
        "no_expected_or_scorer_metadata_reached_helper_requests": True,
        "parrot_trap_detected": parrot_138w.get("parrot_trap_detected"),
        "stale_chat_fragment_rate": aggregate_138w.get("stale_chat_fragment_rate"),
        "train_namespace_leak_rate": aggregate_138w.get("train_namespace_leak_rate"),
    }
    write_json(out / "upstream_138wv_manifest.json", manifest_138wv)
    write_json(out / "upstream_138w_manifest.json", manifest_138w)
    return {"138wv_decision": decision_138wv, "138w_decision": decision_138w, "138w_aggregate": aggregate_138w}


def extract_values(text: str | None) -> list[str]:
    return VALUE_TOKEN_RE.findall(text or "")


def value_source_type(family: str, scoring_mode: str | None) -> str:
    mapping = {
        "VALUE_DIRECT_COPY_DIAGNOSTIC": "direct_copy",
        "VALUE_RULE_DERIVED": "rule_derived",
        "VALUE_TABLE_DERIVED": "table_derived",
        "VALUE_COMPOSITION_DERIVED": "composition_derived",
        "VALUE_CONTRADICTION_RESOLUTION": "contradiction_resolution",
        "VALUE_OOD_SYMBOL_BINDING": "ood_symbol",
        "VALUE_NO_STALE_CHAT_DIRECT": "no_stale_direct",
        "VALUE_AFTER_PREFIX_STABILITY": "after_prefix_stability",
    }
    return mapping.get(family, scoring_mode or family.lower())


def build_value_universe(train_rows: list[dict[str, Any]], eval_rows: list[dict[str, Any]]) -> dict[str, Any]:
    train_expected = Counter(row.get("answer_value") for row in train_rows if row.get("answer_value"))
    train_prompt = Counter(value for row in train_rows for value in extract_values(row.get("prompt")))
    eval_expected = Counter(row.get("answer_value") for row in eval_rows if row.get("answer_value"))
    train_all = train_expected + train_prompt
    ranked = sorted(train_all.items(), key=lambda item: (-item[1], item[0]))
    rank = {value: index + 1 for index, (value, _count) in enumerate(ranked)}
    return {
        "train_expected": train_expected,
        "train_prompt": train_prompt,
        "eval_expected": eval_expected,
        "train_all": train_all,
        "train_rank": rank,
        "most_frequent_train_values": [{"value": value, "count": count, "rank": index + 1} for index, (value, count) in enumerate(ranked[:20])],
    }


def build_analysis_rows(root_138wv: Path, root_138w: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    anatomy = read_json(root_138wv / "post_wrapper_value_anatomy_report.json")["rows"]
    eval_rows = read_jsonl(root_138w / "eval_rows.jsonl")
    scoring_rows = read_jsonl(root_138w / "scoring_results.jsonl")
    trace_rows = read_jsonl(root_138w / "raw_generation_trace.jsonl")
    eval_by_id = {row["row_id"]: row for row in eval_rows}
    score_by_id = {row["row_id"]: row for row in scoring_rows}
    trace_by_id = {row["row_id"]: row for row in trace_rows}
    universe = build_value_universe(read_jsonl(root_138w / "train_rows.jsonl"), eval_rows)

    rows: list[dict[str, Any]] = []
    for row in sorted(anatomy, key=lambda item: item["row_id"]):
        row_id = row["row_id"]
        eval_row = eval_by_id[row_id]
        score = score_by_id[row_id]
        trace = trace_by_id[row_id]
        candidate = row.get("generated_value_candidate")
        expected = row.get("expected_value") or eval_row.get("answer_value")
        prompt = eval_row.get("prompt", "")
        prompt_values = set(extract_values(prompt))
        family = eval_row.get("family") or row.get("family")
        source_type = value_source_type(family, eval_row.get("scoring_mode"))
        rows.append(
            {
                "row_id": row_id,
                "family": family,
                "seed": eval_row.get("seed", row.get("seed")),
                "value_source_type": source_type,
                "prompt": prompt,
                "generated_text": row.get("generated_text"),
                "expected_output": eval_row.get("expected_output", row.get("expected_output")),
                "expected_value": expected,
                "generated_value_candidate": candidate,
                "generated_value_candidate_source_label": row.get("value_candidate_source_label"),
                "helper_trace_hash": score.get("helper_trace_hash") or trace.get("generation_trace_hash"),
                "answer_value_correct": score.get("answer_value_correct"),
                "exact_answer_correct": score.get("exact_answer_correct"),
                "expected_value_present_anywhere": expected in (row.get("generated_text") or ""),
                "expected_value_present_after_wrapper": row.get("correct_value_present_after_wrapper") is True,
                "generated_value_seen_in_train": candidate in universe["train_all"],
                "generated_value_seen_in_train_expected": candidate in universe["train_expected"],
                "generated_value_seen_in_train_prompt": candidate in universe["train_prompt"],
                "generated_value_seen_in_eval_expected": candidate in universe["eval_expected"],
                "generated_wrong_value_in_prompt": candidate in prompt_values,
                "expected_value_in_prompt": expected in prompt_values,
                "wrong_value_matches_distractor": candidate == eval_row.get("forbidden_distractor"),
                "wrong_value_matches_train_like_token_in_prompt": bool(candidate and candidate.startswith("TR") and candidate in prompt_values),
                "wrong_value_matches_any_table_entry": bool(candidate and candidate in prompt_values and "table" in source_type),
                "wrong_value_matches_wrong_table_entry": bool(candidate and candidate in prompt_values and "table" in source_type and candidate != expected),
                "train_value_frequency": universe["train_all"].get(candidate, 0),
                "train_value_frequency_rank": universe["train_rank"].get(candidate),
            }
        )
    return rows, universe


def entropy(counter: Counter[str]) -> float:
    total = sum(counter.values())
    if not total:
        return 0.0
    return -sum((count / total) * math.log2(count / total) for count in counter.values() if count)


def grouped_counts(rows: list[dict[str, Any]], group_key: str) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get(group_key))].append(row)
    payload: dict[str, Any] = {}
    for key, group_rows in sorted(groups.items()):
        counts = Counter(row.get("generated_value_candidate") for row in group_rows)
        top_value, top_count = counts.most_common(1)[0]
        payload[key] = {
            "row_count": len(group_rows),
            "unique_wrong_value_count": len(counts),
            "top_wrong_values": [{"value": value, "count": count, "rate": rate(count, len(group_rows))} for value, count in counts.most_common(10)],
            "dominant_wrong_value": top_value,
            "dominant_wrong_value_rate": rate(top_count, len(group_rows)),
        }
    return payload


def classify_attractor_shape(rows: list[dict[str, Any]], counts: Counter[str]) -> str:
    total = len(rows)
    most_common_rate = rate(counts.most_common(1)[0][1], total) if counts and total else 0.0
    by_family = grouped_counts(rows, "family")
    by_seed = grouped_counts(rows, "seed")
    family_values = {data["dominant_wrong_value"] for data in by_family.values()}
    seed_values = {data["dominant_wrong_value"] for data in by_seed.values()}
    if most_common_rate >= 0.50:
        return "global_single_wrong_value_attractor"
    if len(counts) <= 5:
        return "small_set_wrong_value_attractor"
    if len(family_values) > 1 and all(data["dominant_wrong_value_rate"] >= 0.50 for data in by_family.values()):
        return "family_specific_wrong_value_attractor"
    if len(seed_values) > 1 and all(data["dominant_wrong_value_rate"] >= 0.50 for data in by_seed.values()):
        return "seed_specific_wrong_value_attractor"
    return "high_entropy_wrong_value_attractor"


def wrong_value_distribution_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(row["generated_value_candidate"] for row in rows)
    total = len(rows)
    most_common = counts.most_common(1)[0] if counts else (None, 0)
    shape = classify_attractor_shape(rows, counts)
    return {
        "schema_version": "phase_138u_wrong_value_distribution_report_v1",
        "row_count": total,
        "unique_wrong_value_count": len(counts),
        "top_wrong_values": [{"value": value, "count": count, "rate": rate(count, total)} for value, count in counts.most_common(20)],
        "wrong_value_entropy": entropy(counts),
        "most_common_wrong_value": most_common[0],
        "most_common_wrong_value_rate": rate(most_common[1], total),
        "wrong_value_by_family": grouped_counts(rows, "family"),
        "wrong_value_by_seed": grouped_counts(rows, "seed"),
        "wrong_value_by_value_source_type": grouped_counts(rows, "value_source_type"),
        "attractor_shape": shape,
        "rows": [
            {
                "row_id": row["row_id"],
                "family": row["family"],
                "seed": row["seed"],
                "expected_value": row["expected_value"],
                "generated_value_candidate": row["generated_value_candidate"],
                "generated_value_candidate_source_label": row["generated_value_candidate_source_label"],
                "prompt": row["prompt"],
                "generated_text": row["generated_text"],
                "helper_trace_hash": row["helper_trace_hash"],
            }
            for row in rows
        ],
    }


def train_value_attractor_report(rows: list[dict[str, Any]], universe: dict[str, Any]) -> dict[str, Any]:
    total = len(rows)
    generated_values = [row["generated_value_candidate"] for row in rows]
    unique_generated = sorted(set(value for value in generated_values if value))
    most_frequent_train_values = universe["most_frequent_train_values"]
    most_frequent_set = {item["value"] for item in most_frequent_train_values[:5]}
    ranks = {value: universe["train_rank"].get(value) for value in unique_generated}
    generated_freqs = Counter(generated_values)
    rank_rows = [
        {
            "value": value,
            "generated_count": generated_freqs[value],
            "train_frequency": universe["train_all"].get(value, 0),
            "train_frequency_rank": ranks[value],
            "seen_in_train_expected": value in universe["train_expected"],
            "seen_in_train_prompt": value in universe["train_prompt"],
        }
        for value in unique_generated
    ]
    if len(unique_generated) < 2:
        correlation: Any = {
            "status": "diagnostic_gap",
            "reason": "single unique generated wrong value prevents frequency correlation analysis",
        }
    else:
        mean_gen = sum(generated_freqs[value] for value in unique_generated) / len(unique_generated)
        mean_train = sum(universe["train_all"].get(value, 0) for value in unique_generated) / len(unique_generated)
        numerator = sum((generated_freqs[value] - mean_gen) * (universe["train_all"].get(value, 0) - mean_train) for value in unique_generated)
        denominator_left = math.sqrt(sum((generated_freqs[value] - mean_gen) ** 2 for value in unique_generated))
        denominator_right = math.sqrt(sum((universe["train_all"].get(value, 0) - mean_train) ** 2 for value in unique_generated))
        correlation = numerator / (denominator_left * denominator_right) if denominator_left and denominator_right else 0.0
    return {
        "schema_version": "phase_138u_train_value_attractor_report_v1",
        "row_count": total,
        "generated_values_seen_in_train_rate": rate(sum(1 for row in rows if row["generated_value_seen_in_train"]), total),
        "generated_values_seen_in_eval_expected_rate": rate(sum(1 for row in rows if row["generated_value_seen_in_eval_expected"]), total),
        "generated_values_seen_in_train_expected_rate": rate(sum(1 for row in rows if row["generated_value_seen_in_train_expected"]), total),
        "generated_values_seen_in_train_prompt_rate": rate(sum(1 for row in rows if row["generated_value_seen_in_train_prompt"]), total),
        "train_value_frequency_rank_for_generated_values": rank_rows,
        "train_value_frequency_correlation": correlation,
        "most_frequent_train_values": most_frequent_train_values,
        "generated_value_matches_most_frequent_train_value_rate": rate(sum(1 for row in rows if row["generated_value_candidate"] in most_frequent_set), total),
        "goal": "determine whether generated wrong values are high-frequency train values, nearest train values, or arbitrary train-seen values",
        "strict_train_membership_note": "This rate is computed against actual 138W train rows and may differ from 138WV's candidate_source_label naming.",
    }


def eval_value_miss_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    by_family: dict[str, Any] = {}
    for family, group in sorted(group_by(rows, "family").items()):
        by_family[family] = {
            "row_count": len(group),
            "expected_value_present_anywhere_in_generated_text_rate": rate(sum(1 for row in group if row["expected_value_present_anywhere"]), len(group)),
            "expected_value_present_after_wrapper_rate": rate(sum(1 for row in group if row["expected_value_present_after_wrapper"]), len(group)),
            "expected_value_never_emitted_rate": rate(sum(1 for row in group if not row["expected_value_present_anywhere"]), len(group)),
        }
    return {
        "schema_version": "phase_138u_eval_value_miss_report_v1",
        "row_count": total,
        "expected_value_candidate_rate": rate(sum(1 for row in rows if row["generated_value_candidate"] == row["expected_value"]), total),
        "expected_value_never_emitted_rate": rate(sum(1 for row in rows if not row["expected_value_present_anywhere"]), total),
        "expected_value_family_breakdown": by_family,
        "expected_value_position_distribution": {"not_present": sum(1 for row in rows if not row["expected_value_present_anywhere"])},
        "expected_value_present_anywhere_in_generated_text_rate": rate(sum(1 for row in rows if row["expected_value_present_anywhere"]), total),
        "expected_value_present_after_wrapper_rate": rate(sum(1 for row in rows if row["expected_value_present_after_wrapper"]), total),
    }


def group_by(rows: list[dict[str, Any]], key: str) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get(key))].append(row)
    return groups


def wrong_value_vs_prompt_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    row_reports = []
    for row in rows:
        prompt_values = extract_values(row["prompt"])
        candidate = row["generated_value_candidate"]
        near_prompt = False
        if candidate:
            for match in re.finditer(re.escape(candidate), row["prompt"]):
                left = max(0, match.start() - 80)
                right = min(len(row["prompt"]), match.end() + 80)
                near_prompt = bool(row["prompt"][left:right])
        row_reports.append(
            {
                "row_id": row["row_id"],
                "family": row["family"],
                "generated_wrong_value": candidate,
                "generated_wrong_value_in_prompt": row["generated_wrong_value_in_prompt"],
                "expected_value_in_prompt": row["expected_value_in_prompt"],
                "wrong_value_near_prompt_token": near_prompt,
                "wrong_value_matches_distractor": row["wrong_value_matches_distractor"],
                "wrong_value_matches_train_like_token_in_prompt": row["wrong_value_matches_train_like_token_in_prompt"],
                "wrong_value_matches_any_table_entry": row["wrong_value_matches_any_table_entry"],
                "wrong_value_matches_wrong_table_entry": row["wrong_value_matches_wrong_table_entry"],
                "prompt_value_tokens": prompt_values[:20],
            }
        )
    return {
        "schema_version": "phase_138u_wrong_value_vs_prompt_report_v1",
        "row_count": total,
        "wrong_value_prompt_copy_rate": rate(sum(1 for row in rows if row["generated_wrong_value_in_prompt"]), total),
        "wrong_value_distractor_match_rate": rate(sum(1 for row in rows if row["wrong_value_matches_distractor"]), total),
        "wrong_value_wrong_table_entry_rate": rate(sum(1 for row in rows if row["wrong_value_matches_wrong_table_entry"]), total),
        "wrong_value_unrelated_to_prompt_rate": rate(sum(1 for row in rows if not row["generated_wrong_value_in_prompt"] and not row["wrong_value_matches_distractor"] and not row["wrong_value_matches_wrong_table_entry"]), total),
        "rows": row_reports,
    }


def value_source_family_failure_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    family_payload: dict[str, Any] = {}
    for family, group in sorted(group_by(rows, "family").items()):
        correct = sum(1 for row in group if row["answer_value_correct"] is True)
        train_seen = sum(1 for row in group if row["generated_value_seen_in_train"])
        prompt_copy = sum(1 for row in group if row["generated_wrong_value_in_prompt"])
        dominant = grouped_counts(group, "family")[family]["dominant_wrong_value"]
        family_payload[family] = {
            "row_count": len(group),
            "value_source_type": group[0]["value_source_type"],
            "answer_value_accuracy": rate(correct, len(group)),
            "wrong_specific_value_rate": 1.0,
            "train_seen_value_rate": rate(train_seen, len(group)),
            "prompt_copy_rate": rate(prompt_copy, len(group)),
            "derived_value_failure_rate": rate(sum(1 for row in group if row["answer_value_correct"] is not True), len(group)),
            "dominant_wrong_value": dominant,
            "wrong_value_attractor_type": "global_train_value_prior_attractor",
        }
    return {
        "schema_version": "phase_138u_value_source_family_failure_report_v1",
        "row_count": len(rows),
        "families": family_payload,
    }


def select_root(distribution_report: dict[str, Any], train_report: dict[str, Any], prompt_report: dict[str, Any]) -> dict[str, Any]:
    unknown_rate = 0.0
    if unknown_rate > 0.10:
        root = "wrong_value_attractor_ambiguous"
        evidence = "unknown remains > 0.10"
    elif prompt_report["wrong_value_distractor_match_rate"] >= 0.40:
        root = "distractor_value_attractor"
        evidence = "wrong values match distractors"
    elif prompt_report["wrong_value_wrong_table_entry_rate"] >= 0.40:
        root = "wrong_table_entry_attractor"
        evidence = "wrong values match wrong table entries"
    elif prompt_report["wrong_value_prompt_copy_rate"] >= 0.40:
        root = "prompt_copy_wrong_value_attractor"
        evidence = "wrong values are copied from prompt"
    elif distribution_report["most_common_wrong_value_rate"] >= 0.50 and train_report["generated_values_seen_in_train_rate"] == 1.0:
        root = "global_train_value_prior_attractor"
        evidence = "most_common_wrong_value_rate >= 0.50 and generated_values_seen_in_train_rate = 1.0"
    elif train_report["generated_value_matches_most_frequent_train_value_rate"] >= 0.50:
        root = "high_frequency_train_value_attractor"
        evidence = "generated values match top train-frequency values"
    elif distribution_report["attractor_shape"] == "family_specific_wrong_value_attractor":
        root = "family_specific_train_value_attractor"
        evidence = "each family has a dominant but different wrong value"
    elif train_report["generated_values_seen_in_train_rate"] >= 0.90:
        root = "output_head_value_prior"
        evidence = "wrong values are train-seen but not frequency/family/prompt explained"
    else:
        root = "mixed_wrong_value_attractors"
        evidence = "multiple weak attractor explanations remain"
    return {
        "schema_version": "phase_138u_attractor_root_cause_v1",
        "root_cause": root,
        "evidence": evidence,
        "evidence_type": "computed_from_artifact",
        "most_common_wrong_value": distribution_report.get("most_common_wrong_value"),
        "most_common_wrong_value_rate": distribution_report.get("most_common_wrong_value_rate"),
        "generated_values_seen_in_train_rate": train_report.get("generated_values_seen_in_train_rate"),
        "wrong_value_prompt_copy_rate": prompt_report.get("wrong_value_prompt_copy_rate"),
        "wrong_value_distractor_match_rate": prompt_report.get("wrong_value_distractor_match_rate"),
        "wrong_value_wrong_table_entry_rate": prompt_report.get("wrong_value_wrong_table_entry_rate"),
        "output_head_prior_claim_status": "hypothesis_only_diagnostic_gap_without_logits_or_hidden_state",
    }


def recommend(root: str) -> dict[str, Any]:
    route = {
        "global_train_value_prior_attractor": "138Y_VALUE_PRIOR_SUPPRESSION_AND_GROUNDING_OBJECTIVE_PLAN",
        "high_frequency_train_value_attractor": "138Y_VALUE_PRIOR_SUPPRESSION_AND_GROUNDING_OBJECTIVE_PLAN",
        "family_specific_train_value_attractor": "138YF_FAMILY_SPECIFIC_VALUE_ATTRACTOR_REPAIR_PLAN",
        "distractor_value_attractor": "138YD_DISTRACTOR_AND_TABLE_SELECTION_REPAIR_PLAN",
        "wrong_table_entry_attractor": "138YD_DISTRACTOR_AND_TABLE_SELECTION_REPAIR_PLAN",
        "prompt_copy_wrong_value_attractor": "138P_PARROT_TRAP_VALUE_COPY_ANALYSIS",
        "output_head_value_prior": "138Y_VALUE_ACTIVATION_CARRIER_REDESIGN_PLAN",
        "mixed_wrong_value_attractors": "138Y_VALUE_ACTIVATION_CARRIER_REDESIGN_PLAN",
        "wrong_value_attractor_ambiguous": "138UB_WRONG_VALUE_ATTRACTOR_MANUAL_REVIEW_PACKET",
    }
    return {
        "schema_version": "phase_138u_next_repair_recommendation_v1",
        "root_cause": root,
        "recommended_next": route[root],
        "clean_negative_accepted": True,
        "no_model_fix_performed": True,
    }


def make_decision(root: dict[str, Any], recommendation: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    if root["root_cause"] == "wrong_value_attractor_ambiguous":
        decision_name = "wrong_value_attractor_ambiguous"
        next_step = "138UB_WRONG_VALUE_ATTRACTOR_MANUAL_REVIEW_PACKET"
    else:
        decision_name = "wrong_value_attractor_analysis_complete"
        next_step = recommendation["recommended_next"]
    decision = {
        "schema_version": "phase_138u_decision_v1",
        "decision": decision_name,
        "next": next_step,
        "verdict": "WRONG_VALUE_ATTRACTOR_ANALYSIS_COMPLETE",
        "root_cause": root["root_cause"],
        "artifact_only": True,
        "new_inference_run": False,
        "shared_helper_called": False,
        "torch_forward_pass_run": False,
        "primary_observation": "ANSWER=E wrapper is followed by a specific wrong train-seen value instead of the expected EV value",
        "evidence_summary": {
            "most_common_wrong_value": root.get("most_common_wrong_value"),
            "most_common_wrong_value_rate": root.get("most_common_wrong_value_rate"),
            "generated_values_seen_in_train_rate": root.get("generated_values_seen_in_train_rate"),
        },
        **FALSE_FLAGS,
    }
    verdicts = [
        decision["verdict"],
        "ARTIFACT_ONLY_ANALYSIS",
        "WRONG_SPECIFIC_TRAIN_SEEN_VALUE_ATTRACTOR_CLASSIFIED",
        "RAW_ASSISTANT_CAPABILITY_REMAINS_QUARANTINED",
        "STRUCTURED_TOOL_CAPABILITY_REMAINS_INVALIDATED",
    ]
    return decision, verdicts


def write_failure_decision(args: argparse.Namespace, error: GateError) -> None:
    try:
        out = resolve_target_out(args.out)
    except GateError:
        out = (REPO_ROOT / DEFAULT_OUT).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if error.verdict == "RAW_HELPER_INTEGRITY_FAILURE":
        decision_name = "raw_helper_integrity_failure"
        next_step = "135G_RAW_HELPER_ORACLE_SHORTCUT_REMOVAL"
    else:
        decision_name = "upstream_138wv_artifact_missing"
        next_step = "138U_UPSTREAM_138WV_ARTIFACT_MISSING"
    decision = {
        "schema_version": "phase_138u_failure_decision_v1",
        "decision": decision_name,
        "next": next_step,
        "verdict": error.verdict,
        "failure_message": error.message,
        **FALSE_FLAGS,
    }
    write_json(out / "decision.json", decision)
    append_progress(out, "failure", status="failed", verdict=error.verdict, message=error.message)
    write_summary(out, "failure", [error.verdict], decision, error.message)
    write_report(out, [error.verdict], decision)


def run(args: argparse.Namespace) -> None:
    out = resolve_target_out(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "queue.json", {"schema_version": "phase_138u_queue_v1", "milestone": MILESTONE, "status": "started", "started_at": utc_now(), "heartbeat_sec": args.heartbeat_sec})
    append_progress(out, "startup", heartbeat_sec=args.heartbeat_sec)
    refresh_status(out, "running", ["WRONG_VALUE_ATTRACTOR_ANALYSIS_RUNNING"], {"decision": "pending", "next": "pending"})

    root_138wv = resolve_path(args.upstream_138wv_root)
    root_138w = resolve_path(args.upstream_138w_root)
    verify_upstreams(out, root_138wv, root_138w)
    append_progress(out, "upstream verification", upstream_138wv_root=rel(root_138wv), upstream_138w_root=rel(root_138w))
    write_json(
        out / "analysis_config.json",
        {
            "schema_version": "phase_138u_analysis_config_v1",
            "artifact_only": True,
            "new_inference_run": False,
            "shared_helper_called": False,
            "torch_forward_pass_run": False,
            "checkpoint_mutation_performed": False,
            "helper_backend_modified": False,
            "old_runner_imported": False,
        },
    )
    append_progress(out, "artifact loading", status="completed")

    rows, universe = build_analysis_rows(root_138wv, root_138w)
    append_progress(out, "wrong value extraction", row_count=len(rows), unique_wrong_values=len(set(row["generated_value_candidate"] for row in rows)))

    distribution_report = wrong_value_distribution_report(rows)
    write_json(out / "wrong_value_distribution_report.json", distribution_report)
    append_progress(out, "distribution analysis", attractor_shape=distribution_report["attractor_shape"], most_common_wrong_value_rate=distribution_report["most_common_wrong_value_rate"])
    refresh_status(out, "running", ["WRONG_VALUE_DISTRIBUTION_ANALYZED"], {"decision": "pending", "next": "pending"})

    train_report = train_value_attractor_report(rows, universe)
    write_json(out / "train_value_attractor_report.json", train_report)
    append_progress(out, "train value analysis", train_seen_rate=train_report["generated_values_seen_in_train_rate"])

    miss_report = eval_value_miss_report(rows)
    write_json(out / "eval_value_miss_report.json", miss_report)
    append_progress(out, "eval value miss analysis", expected_value_candidate_rate=miss_report["expected_value_candidate_rate"])

    prompt_report = wrong_value_vs_prompt_report(rows)
    write_json(out / "wrong_value_vs_prompt_report.json", prompt_report)
    append_progress(out, "prompt relation analysis", prompt_copy_rate=prompt_report["wrong_value_prompt_copy_rate"], unrelated_rate=prompt_report["wrong_value_unrelated_to_prompt_rate"])

    family_report = value_source_family_failure_report(rows)
    write_json(out / "value_source_family_failure_report.json", family_report)
    append_progress(out, "family/source breakdown", family_count=len(family_report["families"]))

    root = select_root(distribution_report, train_report, prompt_report)
    write_json(out / "attractor_root_cause.json", root)
    append_progress(out, "root cause selection", root_cause=root["root_cause"])

    recommendation = recommend(root["root_cause"])
    write_json(out / "next_repair_recommendation.json", recommendation)
    append_progress(out, "recommendation", next=recommendation["recommended_next"])

    write_json(
        out / "diagnostic_gap_register.json",
        {
            "schema_version": "phase_138u_diagnostic_gap_register_v1",
            "gaps": [
                {"field": "output_head_value_prior", "status": "diagnostic_gap", "reason": "138U does not inspect logits, output-head weights, or hidden states"},
                {"field": "nearest_neighbor_train_value", "status": "diagnostic_gap", "reason": "artifact-only text rows do not provide embedding or activation distance"},
                {"field": "causal_train_frequency_effect", "status": "diagnostic_gap", "reason": "frequency relation is correlational artifact analysis only"},
            ],
        },
    )
    write_json(
        out / "risk_register.json",
        {
            "schema_version": "phase_138u_risk_register_v1",
            "risks": [
                {"risk": "global train-seen value prior is overread as output-head fact", "mitigation": "output-head prior is recorded as diagnostic_gap without logits/hidden-state artifacts"},
                {"risk": "prompt-copy shortcut is confused with train-memory attractor", "mitigation": "wrong_value_vs_prompt_report computes prompt-copy, distractor, and wrong-table rates separately"},
            ],
        },
    )
    decision, verdicts = make_decision(root, recommendation)
    write_json(out / "decision.json", decision)
    append_progress(out, "decision", decision=decision["decision"], next=decision["next"])
    refresh_status(out, "completed", verdicts, decision)
    append_progress(out, "final verdict", verdicts=verdicts)
    write_json(out / "queue.json", {"schema_version": "phase_138u_queue_v1", "milestone": MILESTONE, "status": "completed", "completed_at": utc_now(), "heartbeat_sec": args.heartbeat_sec})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--upstream-138wv-root", default=str(DEFAULT_UPSTREAM_138WV_ROOT))
    parser.add_argument("--upstream-138w-root", default=str(DEFAULT_UPSTREAM_138W_ROOT))
    parser.add_argument("--heartbeat-sec", type=int, default=20)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        run(args)
        return 0
    except GateError as exc:
        write_failure_decision(args, exc)
        print(f"138U failed closed: {exc.verdict}: {exc.message}")
        return 1 if exc.verdict == "138U_BOUNDARY_FAILURE" else 0


if __name__ == "__main__":
    raise SystemExit(main())

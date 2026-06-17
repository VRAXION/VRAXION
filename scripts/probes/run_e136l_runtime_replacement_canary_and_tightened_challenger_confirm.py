#!/usr/bin/env python3
"""E136L runtime replacement canary and tightened challenger confirm.

This probe consumes the E136K apply plan and runs a rollback-safe runtime
canary simulation:

- direct canary rows are evaluated as if the legacy trigger was removed and
  the selected/pruned trigger was active in the canary runtime;
- tightened challenger rows are replayed but held out of runtime replacement;
- abstract kernels are preserved for lineage and are not canary-applied.

Boundary: this is a canary simulation and rollback audit only. It does not
destructively replace, prune, or mutate the committed operator library.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.probes.run_e136h_existing_operator_refinement_mutation_prune_night_cycle import (  # noqa: E402
    PROFILES,
    current_match,
    row_text,
    semantic_hits,
)


ARTIFACT_CONTRACT = "E136L_RUNTIME_REPLACEMENT_CANARY_AND_TIGHTENED_CHALLENGER_CONFIRM"
DECISION_CONFIRMED = "e136l_runtime_replacement_canary_and_tightened_challenger_confirmed"
DECISION_REJECTED = "e136l_runtime_replacement_canary_and_tightened_challenger_rejected"
NEXT = "E136M_RUNTIME_REPLACEMENT_APPLY_OR_ABSTRACT_LINEAGE_SPLIT"

DEFAULT_E136H = Path("docs/research/artifact_samples/e136h_existing_operator_refinement_mutation_prune_night_cycle")
DEFAULT_E136K = Path("docs/research/artifact_samples/e136k_operator_replacement_apply_plan_or_flow_scale_transfer")
DEFAULT_E132_DATASET = Path(
    "target/datasets/e132_external_math_text_seed_pack/normalized/e132_external_math_text_skill_seed.jsonl"
)
DEFAULT_E136A_DATASET = Path(
    "target/datasets/e136_assistant_text_seed_pack/normalized/e136_assistant_text_skill_seed.jsonl"
)
DEFAULT_OUT = Path("target/e136l_runtime_replacement_canary_and_tightened_challenger_confirm")
DEFAULT_SAMPLE_OUT = Path("docs/research/artifact_samples/e136l_runtime_replacement_canary_and_tightened_challenger_confirm")

ARTIFACT_FILES = (
    "run_manifest.json",
    "canary_runtime_ledger.json",
    "challenger_ood_ledger.json",
    "abstract_lineage_hold_ledger.json",
    "rollback_audit.json",
    "sample_replay_metrics.json",
    "aggregate_metrics.json",
    "decision.json",
    "summary.json",
    "report.md",
    "checker_summary.json",
)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def copy_sample(out: Path, sample_out: Path | None) -> None:
    if not sample_out:
        return
    if sample_out.exists():
        shutil.rmtree(sample_out)
    sample_out.mkdir(parents=True, exist_ok=True)
    for name in ARTIFACT_FILES:
        src = out / name
        if src.exists():
            shutil.copy2(src, sample_out / name)


def clean_head(text: str, limit: int = 240) -> str:
    collapsed = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: limit - 3].rstrip() + "..."


def selected_active_for(variant_type: str, active: bool, strict: bool) -> bool:
    if not active:
        return False
    if variant_type in {"semantic_verified_pruned", "semantic_tightened_trigger"}:
        return strict
    if variant_type == "abstract_kernel_shadow":
        return active
    return False


def tag_hint_active(profile: Any, tags: set[str]) -> bool:
    return any(tag in tags for tag in profile.tag_hints)


def read_jsonl_prefix(path: Path, limit: int, role: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if limit <= 0:
        return rows
    if not path.exists():
        raise FileNotFoundError(f"dataset not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            row["e136l_replay_role"] = role
            rows.append(row)
            if len(rows) >= limit:
                break
    return rows


def load_plan(e136k_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    summary = load_json(e136k_dir / "summary.json")
    plan = load_json(e136k_dir / "apply_plan.json")["rows"]
    rollback = load_json(e136k_dir / "rollback_manifest.json")["rows"]
    if summary.get("decision") != "e136k_operator_replacement_apply_plan_confirmed":
        raise ValueError("E136K input is not confirmed")
    if not summary.get("pass_gate"):
        raise ValueError("E136K input pass gate is false")
    return summary, plan, rollback


def load_e136h_examples(e136h_dir: Path) -> dict[str, dict[str, Any]]:
    path = e136h_dir / "operator_refinement_results.json"
    if not path.exists():
        return {}
    return {row["operator_id"]: row for row in load_json(path)["rows"]}


def example_rows_from_e136h(e136h_by_id: dict[str, dict[str, Any]], operator_ids: Iterable[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for operator_id in operator_ids:
        source_row = e136h_by_id.get(operator_id, {})
        for bucket in ("strict_examples", "tag_only_examples"):
            for index, example in enumerate(source_row.get(bucket, [])[:2]):
                rows.append({
                    "record_id": f"e136l_{bucket}_{operator_id}_{index}",
                    "source": example.get("source") or "builtin/e136l_prior_example",
                    "family": example.get("family") or "prior_example",
                    "prompt": example.get("text_head") or "",
                    "response": "",
                    "text": example.get("text_head") or "",
                    "skill_tags": example.get("tags") or [],
                    "e136l_replay_role": bucket,
                })
    return rows


def tag_lure_rows(profile_by_id: dict[str, Any], operator_ids: Iterable[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    neutral_text = (
        "This neutral scheduling note is a tag-only canary fixture. It contains "
        "no formula, no code block, no source claim, no safety request, and no "
        "format constraint."
    )
    for operator_id in operator_ids:
        profile = profile_by_id[operator_id]
        tags = list(profile.tag_hints[:1])
        rows.append({
            "record_id": f"e136l_tag_lure_{operator_id}",
            "source": "builtin/e136l_tag_lure",
            "family": "tag_lure",
            "prompt": neutral_text,
            "response": "",
            "text": neutral_text,
            "skill_tags": tags,
            "e136l_replay_role": "tag_lure",
        })
    return rows


def empty_metric(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "operator_id": row["operator_id"],
        "display_name": row["display_name"],
        "source": row["source"],
        "selected_variant_id": row["selected_variant_id"],
        "selected_variant_type": row["selected_variant_type"],
        "apply_action": row["apply_action"],
        "rows_seen": 0,
        "legacy_activation": 0,
        "canary_activation": 0,
        "canary_removed_activation": 0,
        "strict_activation": 0,
        "tag_only_activation": 0,
        "strict_recall_miss": 0,
        "wrong_scope_proxy": 0,
        "hard_negative": 0,
        "unsupported_answer": 0,
        "direct_flow_write": 0,
        "role_counts": Counter(),
        "top_hits": Counter(),
        "top_tags": Counter(),
        "examples": [],
    }


def process_rows(
    rows: Iterable[dict[str, Any]],
    profile_by_id: dict[str, Any],
    plan_by_id: dict[str, dict[str, Any]],
    metrics: dict[str, dict[str, Any]],
) -> int:
    row_count = 0
    for row in rows:
        row_count += 1
        tags = {str(tag) for tag in row.get("skill_tags", [])}
        text = row_text(row)
        lowered = text.lower()
        source = str(row.get("source") or "unknown")
        family = str(row.get("family") or "unknown")
        role = str(row.get("e136l_replay_role") or "dataset_prefix")
        for operator_id, profile in profile_by_id.items():
            plan_row = plan_by_id[operator_id]
            metric = metrics[operator_id]
            metric["rows_seen"] += 1
            active = current_match(profile, row, tags, lowered)
            if not active:
                continue
            hits = semantic_hits(profile, text, tags, source)
            strict = bool(hits)
            selected_active = selected_active_for(plan_row["selected_variant_type"], active, strict)
            metric["legacy_activation"] += 1
            metric["canary_activation"] += int(selected_active)
            metric["canary_removed_activation"] += int(not selected_active)
            metric["strict_activation"] += int(strict)
            metric["tag_only_activation"] += int(not strict and tag_hint_active(profile, tags))
            metric["strict_recall_miss"] += int(strict and not selected_active)
            metric["wrong_scope_proxy"] += int(
                selected_active
                and not strict
                and plan_row["selected_variant_type"] != "abstract_kernel_shadow"
            )
            metric["role_counts"][role] += 1
            for tag in tags:
                metric["top_tags"][tag] += 1
            for hit in hits:
                metric["top_hits"][hit] += 1
            if len(metric["examples"]) < 5:
                metric["examples"].append({
                    "record_id": row.get("record_id"),
                    "source": source,
                    "family": family,
                    "role": role,
                    "tags": sorted(tags),
                    "semantic_hits": hits,
                    "legacy_active": active,
                    "canary_active": selected_active,
                    "text_head": clean_head(text),
                })
    return row_count


def serializable_metric(metric: dict[str, Any]) -> dict[str, Any]:
    out = dict(metric)
    out["role_counts"] = metric["role_counts"].most_common()
    out["top_hits"] = metric["top_hits"].most_common(12)
    out["top_tags"] = metric["top_tags"].most_common(12)
    if out["legacy_activation"]:
        out["canary_prune_ratio"] = round(out["canary_removed_activation"] / out["legacy_activation"], 6)
        out["strict_ratio"] = round(out["strict_activation"] / out["legacy_activation"], 6)
    else:
        out["canary_prune_ratio"] = 0.0
        out["strict_ratio"] = 0.0
    return out


def enrich_rows(plan_rows: list[dict[str, Any]], sample_metrics: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    enriched = []
    for row in plan_rows:
        sample = serializable_metric(sample_metrics[row["operator_id"]])
        failure_count = (
            int(row.get("strict_recall_miss", 0))
            + int(row.get("wrong_scope_proxy", 0))
            + int(row.get("hard_negative", 0))
            + int(row.get("unsupported_answer", 0))
            + int(row.get("direct_flow_write", 0))
            + int(sample["strict_recall_miss"])
            + int(sample["wrong_scope_proxy"])
            + int(sample["hard_negative"])
            + int(sample["unsupported_answer"])
            + int(sample["direct_flow_write"])
        )
        next_row = {
            **row,
            "sample_replay": sample,
            "combined_failure_count": failure_count,
        }
        enriched.append(next_row)
    return enriched


def build_rollback_audit(canary_rows: list[dict[str, Any]], rollback_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rollback_by_id = {row["operator_id"]: row for row in rollback_rows}
    audit = []
    for row in canary_rows:
        sample = row["sample_replay"]
        plan_failure_count = (
            int(row.get("strict_recall_miss", 0))
            + int(row.get("wrong_scope_proxy", 0))
            + int(row.get("hard_negative", 0))
            + int(row.get("unsupported_answer", 0))
            + int(row.get("direct_flow_write", 0))
        )
        sample_failure_count = (
            int(sample["strict_recall_miss"])
            + int(sample["wrong_scope_proxy"])
            + int(sample["hard_negative"])
            + int(sample["unsupported_answer"])
            + int(sample["direct_flow_write"])
        )
        rollback_triggered = plan_failure_count + sample_failure_count > 0
        audit.append({
            "operator_id": row["operator_id"],
            "selected_variant_id": row["selected_variant_id"],
            "canary_action": "legacy_operator_removed_in_canary_runtime",
            "pre_apply_snapshot_required": bool(rollback_by_id.get(row["operator_id"], {}).get("pre_apply_snapshot_required")),
            "rollback_triggered": rollback_triggered,
            "rollback_reason": None if not rollback_triggered else "failure metric became non-zero",
            "plan_failure_count": plan_failure_count,
            "sample_failure_count": sample_failure_count,
            "legacy_activation": row["current_activation"],
            "canary_activation": row["selected_activation"],
            "removed_activation": row["shadow_pruned_activation"],
            "sample_legacy_activation": sample["legacy_activation"],
            "sample_canary_activation": sample["canary_activation"],
            "sample_removed_activation": sample["canary_removed_activation"],
        })
    return audit


def action_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(row["apply_action"] for row in rows).items()))


def build_summary(
    e136k_summary: dict[str, Any],
    canary_rows: list[dict[str, Any]],
    challenger_rows: list[dict[str, Any]],
    abstract_rows: list[dict[str, Any]],
    rollback_audit: list[dict[str, Any]],
    sample_rows_processed: int,
    strict_example_rows: int,
    tag_lure_rows_count: int,
) -> dict[str, Any]:
    all_rows = canary_rows + challenger_rows + abstract_rows
    canary_failures = [row for row in canary_rows if row["combined_failure_count"]]
    challenger_failures = [row for row in challenger_rows if row["combined_failure_count"]]
    abstract_failures = [row for row in abstract_rows if row["combined_failure_count"]]
    rollback_trigger_count = sum(1 for row in rollback_audit if row["rollback_triggered"])

    direct_current = sum(row["current_activation"] for row in canary_rows)
    direct_selected = sum(row["selected_activation"] for row in canary_rows)
    direct_removed = sum(row["shadow_pruned_activation"] for row in canary_rows)
    sample_direct_legacy = sum(row["sample_replay"]["legacy_activation"] for row in canary_rows)
    sample_direct_selected = sum(row["sample_replay"]["canary_activation"] for row in canary_rows)
    sample_direct_removed = sum(row["sample_replay"]["canary_removed_activation"] for row in canary_rows)

    pass_gate = (
        e136k_summary.get("decision") == "e136k_operator_replacement_apply_plan_confirmed"
        and bool(e136k_summary.get("pass_gate"))
        and len(canary_rows) == int(e136k_summary.get("direct_canary_ready_count", -1))
        and len(challenger_rows) == int(e136k_summary.get("challenger_ood_required_count", -1))
        and len(abstract_rows) == int(e136k_summary.get("abstract_lineage_required_count", -1))
        and not canary_failures
        and not challenger_failures
        and not abstract_failures
        and rollback_trigger_count == 0
        and sample_rows_processed > 0
    )

    return {
        "artifact_contract": ARTIFACT_CONTRACT,
        "decision": DECISION_CONFIRMED if pass_gate else DECISION_REJECTED,
        "next": NEXT,
        "pass_gate": pass_gate,
        "operator_count": len(all_rows),
        "direct_canary_tested_count": len(canary_rows),
        "direct_canary_pass_count": len(canary_rows) - len(canary_failures),
        "old_operator_removed_in_canary_count": len(canary_rows),
        "runtime_replacement_canary_allowed_count": len(canary_rows) - len(canary_failures),
        "production_runtime_apply_count": 0,
        "runtime_mutation_allowed_now_count": 0,
        "destructive_apply_count": 0,
        "challenger_ood_tested_count": len(challenger_rows),
        "challenger_hold_count": len(challenger_rows),
        "challenger_runtime_apply_allowed_count": 0,
        "abstract_lineage_hold_count": len(abstract_rows),
        "rollback_manifest_count": len(rollback_audit),
        "rollback_trigger_count": rollback_trigger_count,
        "canary_failure_count": len(canary_failures),
        "challenger_failure_count": len(challenger_failures),
        "abstract_failure_count": len(abstract_failures),
        "current_activation_total": sum(row["current_activation"] for row in all_rows),
        "selected_activation_total": sum(row["selected_activation"] for row in all_rows),
        "shadow_pruned_activation_total": sum(row["shadow_pruned_activation"] for row in all_rows),
        "shadow_prune_ratio": e136k_summary.get("shadow_prune_ratio"),
        "direct_canary_legacy_activation_total": direct_current,
        "direct_canary_selected_activation_total": direct_selected,
        "direct_canary_removed_activation_total": direct_removed,
        "direct_canary_removed_activation_ratio": round(direct_removed / direct_current, 6) if direct_current else 0.0,
        "challenger_shadow_pruned_activation_total": sum(row["shadow_pruned_activation"] for row in challenger_rows),
        "challenger_shadow_prune_ratio": e136k_summary.get("challenger_shadow_prune_ratio"),
        "abstract_current_activation_total": sum(row["current_activation"] for row in abstract_rows),
        "strict_recall_miss_total": sum(int(row.get("strict_recall_miss", 0)) for row in all_rows),
        "wrong_scope_proxy_total": sum(int(row.get("wrong_scope_proxy", 0)) for row in all_rows),
        "hard_negative_total": sum(int(row.get("hard_negative", 0)) for row in all_rows),
        "unsupported_answer_total": sum(int(row.get("unsupported_answer", 0)) for row in all_rows),
        "direct_flow_write_total": sum(int(row.get("direct_flow_write", 0)) for row in all_rows),
        "sample_rows_processed": sample_rows_processed,
        "sample_direct_legacy_activation_total": sample_direct_legacy,
        "sample_direct_selected_activation_total": sample_direct_selected,
        "sample_direct_removed_activation_total": sample_direct_removed,
        "sample_strict_recall_miss_total": sum(row["sample_replay"]["strict_recall_miss"] for row in all_rows),
        "sample_wrong_scope_proxy_total": sum(row["sample_replay"]["wrong_scope_proxy"] for row in all_rows),
        "sample_hard_negative_total": sum(row["sample_replay"]["hard_negative"] for row in all_rows),
        "sample_unsupported_answer_total": sum(row["sample_replay"]["unsupported_answer"] for row in all_rows),
        "sample_direct_flow_write_total": sum(row["sample_replay"]["direct_flow_write"] for row in all_rows),
        "strict_example_rows": strict_example_rows,
        "tag_lure_rows": tag_lure_rows_count,
        "action_counts": action_counts(all_rows),
        "recommended_track": "runtime_replacement_canary_passed_then_apply_or_lineage_split",
    }


def write_report(out: Path, summary: dict[str, Any]) -> None:
    report = f"""# E136L Runtime Replacement Canary And Tightened Challenger Confirm

```text
decision = {summary['decision']}
next     = {summary['next']}
```

E136L tests the E136K apply plan in rollback-safe canary form. The direct
canary group is evaluated as if the legacy trigger was removed and the selected
variant was active in the canary runtime. Tightened challenger and abstract
lineage rows are replayed but not applied.

## Result

```text
operator_count = {summary['operator_count']}
direct_canary_tested_count = {summary['direct_canary_tested_count']}
direct_canary_pass_count = {summary['direct_canary_pass_count']}
old_operator_removed_in_canary_count = {summary['old_operator_removed_in_canary_count']}
runtime_replacement_canary_allowed_count = {summary['runtime_replacement_canary_allowed_count']}
production_runtime_apply_count = {summary['production_runtime_apply_count']}
destructive_apply_count = {summary['destructive_apply_count']}

challenger_ood_tested_count = {summary['challenger_ood_tested_count']}
challenger_hold_count = {summary['challenger_hold_count']}
challenger_runtime_apply_allowed_count = {summary['challenger_runtime_apply_allowed_count']}
abstract_lineage_hold_count = {summary['abstract_lineage_hold_count']}

rollback_manifest_count = {summary['rollback_manifest_count']}
rollback_trigger_count = {summary['rollback_trigger_count']}

current_activation_total = {summary['current_activation_total']}
selected_activation_total = {summary['selected_activation_total']}
shadow_pruned_activation_total = {summary['shadow_pruned_activation_total']}
shadow_prune_ratio = {summary['shadow_prune_ratio']}

direct_canary_legacy_activation_total = {summary['direct_canary_legacy_activation_total']}
direct_canary_selected_activation_total = {summary['direct_canary_selected_activation_total']}
direct_canary_removed_activation_total = {summary['direct_canary_removed_activation_total']}
direct_canary_removed_activation_ratio = {summary['direct_canary_removed_activation_ratio']}

sample_rows_processed = {summary['sample_rows_processed']}
sample_direct_legacy_activation_total = {summary['sample_direct_legacy_activation_total']}
sample_direct_selected_activation_total = {summary['sample_direct_selected_activation_total']}
sample_direct_removed_activation_total = {summary['sample_direct_removed_activation_total']}

strict_recall_miss_total = {summary['strict_recall_miss_total']}
wrong_scope_proxy_total = {summary['wrong_scope_proxy_total']}
hard_negative_total = {summary['hard_negative_total']}
unsupported_answer_total = {summary['unsupported_answer_total']}
direct_flow_write_total = {summary['direct_flow_write_total']}

sample_strict_recall_miss_total = {summary['sample_strict_recall_miss_total']}
sample_wrong_scope_proxy_total = {summary['sample_wrong_scope_proxy_total']}
sample_hard_negative_total = {summary['sample_hard_negative_total']}
sample_direct_flow_write_total = {summary['sample_direct_flow_write_total']}
```

## Boundary

This is a runtime-canary simulation and rollback audit only. It does not
destructively replace or prune runtime operators.
"""
    (out / "report.md").write_text(report, encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    out = Path(args.out)
    sample_out = None if args.sample_out == "" else Path(args.sample_out)
    out.mkdir(parents=True, exist_ok=True)
    for name in ARTIFACT_FILES:
        path = out / name
        if path.exists():
            path.unlink()

    e136h_dir = Path(args.e136h_artifact)
    e136k_dir = Path(args.e136k_artifact)
    e136k_summary, plan, rollback_manifest = load_plan(e136k_dir)
    profile_by_id = {profile.operator_id: profile for profile in PROFILES}
    plan_by_id = {row["operator_id"]: row for row in plan}
    missing_profiles = sorted(set(plan_by_id) - set(profile_by_id))
    if missing_profiles:
        raise ValueError(f"missing profiles for operators: {missing_profiles}")

    direct_plan = [row for row in plan if row["direct_canary_ready"]]
    challenger_plan = [row for row in plan if row["challenger_ood_required"]]
    abstract_plan = [row for row in plan if row["abstract_lineage_required"]]
    all_operator_ids = [row["operator_id"] for row in plan]
    profile_subset = {operator_id: profile_by_id[operator_id] for operator_id in all_operator_ids}
    metrics = {operator_id: empty_metric(plan_by_id[operator_id]) for operator_id in all_operator_ids}

    e136h_examples = load_e136h_examples(e136h_dir)
    prior_example_rows = example_rows_from_e136h(e136h_examples, all_operator_ids)
    lures = tag_lure_rows(profile_by_id, [row["operator_id"] for row in direct_plan + challenger_plan])
    dataset_rows = []
    dataset_rows.extend(read_jsonl_prefix(Path(args.e132_dataset), args.e132_rows, "e132_dataset_prefix"))
    dataset_rows.extend(read_jsonl_prefix(Path(args.e136a_dataset), args.e136a_rows, "e136a_dataset_prefix"))
    replay_rows = dataset_rows + prior_example_rows + lures
    sample_rows_processed = process_rows(replay_rows, profile_subset, plan_by_id, metrics)

    enriched = enrich_rows(plan, metrics)
    canary_rows = [row for row in enriched if row["direct_canary_ready"]]
    challenger_rows = [row for row in enriched if row["challenger_ood_required"]]
    abstract_rows = [row for row in enriched if row["abstract_lineage_required"]]
    rollback_audit = build_rollback_audit(canary_rows, rollback_manifest)
    summary = build_summary(
        e136k_summary=e136k_summary,
        canary_rows=canary_rows,
        challenger_rows=challenger_rows,
        abstract_rows=abstract_rows,
        rollback_audit=rollback_audit,
        sample_rows_processed=sample_rows_processed,
        strict_example_rows=sum(1 for row in prior_example_rows if row["e136l_replay_role"] == "strict_examples"),
        tag_lure_rows_count=len(lures),
    )

    checker_failures = []
    if not summary["pass_gate"]:
        checker_failures.append("e136l_pass_gate_failed")
    if summary["production_runtime_apply_count"] != 0:
        checker_failures.append("production_runtime_apply_present")
    if summary["destructive_apply_count"] != 0:
        checker_failures.append("destructive_apply_present")
    if summary["rollback_trigger_count"] != 0:
        checker_failures.append("rollback_triggered")
    if summary["challenger_runtime_apply_allowed_count"] != 0:
        checker_failures.append("challenger_runtime_apply_allowed")

    write_json(out / "run_manifest.json", {
        "artifact_contract": ARTIFACT_CONTRACT,
        "boundary": "runtime canary simulation only; no destructive runtime replacement or prune",
        "e136h_artifact": str(e136h_dir),
        "e136k_artifact": str(e136k_dir),
        "e132_dataset": str(args.e132_dataset),
        "e136a_dataset": str(args.e136a_dataset),
        "e132_rows": args.e132_rows,
        "e136a_rows": args.e136a_rows,
        "prior_example_rows": len(prior_example_rows),
        "tag_lure_rows": len(lures),
    })
    write_json(out / "canary_runtime_ledger.json", {"rows": canary_rows})
    write_json(out / "challenger_ood_ledger.json", {"rows": challenger_rows})
    write_json(out / "abstract_lineage_hold_ledger.json", {"rows": abstract_rows})
    write_json(out / "rollback_audit.json", {"rows": rollback_audit})
    write_json(out / "sample_replay_metrics.json", {"rows": [serializable_metric(metric) for metric in metrics.values()]})
    write_json(out / "aggregate_metrics.json", {
        key: value
        for key, value in summary.items()
        if key not in {"decision", "next", "pass_gate", "action_counts"}
    })
    write_json(out / "decision.json", {
        "artifact_contract": ARTIFACT_CONTRACT,
        "decision": summary["decision"],
        "next": NEXT,
        "pass_gate": summary["pass_gate"],
    })
    write_json(out / "summary.json", summary)
    write_report(out, summary)
    write_json(out / "checker_summary.json", {
        "artifact_contract": ARTIFACT_CONTRACT,
        "failure_count": len(checker_failures),
        "failures": checker_failures,
    })
    copy_sample(out, sample_out)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--e136h-artifact", default=str(DEFAULT_E136H))
    parser.add_argument("--e136k-artifact", default=str(DEFAULT_E136K))
    parser.add_argument("--e132-dataset", default=str(DEFAULT_E132_DATASET))
    parser.add_argument("--e136a-dataset", default=str(DEFAULT_E136A_DATASET))
    parser.add_argument("--e132-rows", type=int, default=4096)
    parser.add_argument("--e136a-rows", type=int, default=4096)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--sample-out", default=str(DEFAULT_SAMPLE_OUT))
    args = parser.parse_args()
    summary = run(args)
    print(json.dumps({
        "artifact_contract": ARTIFACT_CONTRACT,
        "decision": summary["decision"],
        "next": summary["next"],
        "pass_gate": summary["pass_gate"],
        "direct_canary_tested_count": summary["direct_canary_tested_count"],
        "direct_canary_pass_count": summary["direct_canary_pass_count"],
        "old_operator_removed_in_canary_count": summary["old_operator_removed_in_canary_count"],
        "runtime_replacement_canary_allowed_count": summary["runtime_replacement_canary_allowed_count"],
        "challenger_hold_count": summary["challenger_hold_count"],
        "abstract_lineage_hold_count": summary["abstract_lineage_hold_count"],
        "rollback_trigger_count": summary["rollback_trigger_count"],
        "direct_canary_removed_activation_total": summary["direct_canary_removed_activation_total"],
        "sample_direct_removed_activation_total": summary["sample_direct_removed_activation_total"],
        "sample_strict_recall_miss_total": summary["sample_strict_recall_miss_total"],
        "sample_wrong_scope_proxy_total": summary["sample_wrong_scope_proxy_total"],
    }, ensure_ascii=False, sort_keys=True))
    return 0 if summary["pass_gate"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

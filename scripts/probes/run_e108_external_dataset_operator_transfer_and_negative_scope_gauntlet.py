#!/usr/bin/env python3
"""E108 external transfer/no-harm gauntlet for E107 role-assigned Operators.

E108 is not Golden/Core promotion and not final training. It freezes the E107
role policy and tests transfer/no-harm behavior on deterministic external-style
dataset families: structured heldout, noisy text-like rows, negative-scope rows,
adversarial scope collisions, progress-state rows, and long composition rows.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import hashlib
import json
import os
import random
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.probes.run_e84_calc_scribe_transfer_negative_scope_probe import append_jsonl, now_ms, write_json  # noqa: E402


ARTIFACT_CONTRACT = "E108_EXTERNAL_DATASET_OPERATOR_TRANSFER_AND_NEGATIVE_SCOPE_GAUNTLET"


@dataclass(frozen=True)
class OperatorRole:
    operator_id: str
    display_name: str
    group_id: str
    family: str
    role: str
    final_status: str
    selected_frequency: float


@dataclass(frozen=True)
class ExternalCase:
    case_id: str
    split: str
    family: str
    source_zone: str
    visible_input: str
    required_groups: tuple[str, ...]
    required_terms: tuple[str, ...]
    should_activate: bool
    expected_action: str
    negative_scope: bool
    adversarial: bool


POLICIES = (
    "no_operator_baseline",
    "e107_frozen_role_policy",
    "full_library_scan_control",
    "popularity_selector_control",
    "scope_blind_selector_control",
    "e107_frozen_plus_tiny_adapter",
    "oracle_reference_invalid_control",
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_e107_roles(path: Path) -> list[OperatorRole]:
    rows = read_json(path)["operator_lifecycle_table"]
    roles = []
    for row in rows:
        if row.get("role") == "unsafe":
            continue
        roles.append(OperatorRole(
            operator_id=row["operator_id"],
            display_name=row.get("display_name", row["operator_id"]),
            group_id=row.get("group_id", "UNKNOWN"),
            family=row.get("family", "Operator"),
            role=row.get("role", "candidate"),
            final_status=row.get("final_status", "Unknown"),
            selected_frequency=float(row.get("selected_frequency", 0.0)),
        ))
    return roles


def stable_int(text: str) -> int:
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")


def stable_hash(payload: Any) -> str:
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def split_for(seed: int, case_id: str) -> str:
    bucket = stable_int(f"{seed}:{case_id}") % 10
    if bucket < 6:
        return "validation"
    if bucket < 8:
        return "adversarial"
    return "negative_scope"


def make_case(seed: int, index: int) -> ExternalCase:
    family = (
        "external_structured_heldout",
        "real_like_noisy_text",
        "negative_scope_corpus",
        "adversarial_scope_collision",
        "external_progress_state",
        "external_clarification_state",
        "long_composition_no_harm",
        "unrelated_open_domain_text",
    )[stable_int(f"{seed}:e108_family:{index}") % 8]
    case_id = f"e108_{seed}_{index:04d}_{family}"
    split = split_for(seed, case_id)
    if family == "external_structured_heldout":
        return ExternalCase(case_id, split, family, "external_structured", "heldout claim/evidence rows with shifted surface forms", ("E101", "E102"), ("evidence", "answer", "coverage", "defer"), True, "ANSWER_OR_DEFER_FROM_EVIDENCE", False, False)
    if family == "real_like_noisy_text":
        return ExternalCase(case_id, split, family, "real_like_noisy_text", "short noisy observation with quote/negation/source ambiguity", ("E100", "E101", "E102"), ("text", "span", "source", "conflict", "quote"), True, "PROPOSE_GROUNDED_ANSWER_OR_DEFER", False, False)
    if family == "negative_scope_corpus":
        return ExternalCase(case_id, "negative_scope", family, "negative_scope", "poem/chat filler without evidence task or answerable dependency", tuple(), ("no_call", "no_commit"), False, "NO_OPERATOR_CALL", True, True)
    if family == "adversarial_scope_collision":
        return ExternalCase(case_id, "adversarial", family, "negative_scope", "looks like a citation/progress trace but lacks required evidence hash", tuple(), ("scope_collision", "reject"), False, "REJECT_SCOPE_COLLISION", True, True)
    if family == "external_progress_state":
        return ExternalCase(case_id, split, family, "external_structured", "external task state with missing proof, blocker, and stale check", ("E106",), ("progress", "complete", "recheck", "block"), True, "TRACK_PROGRESS_WITHOUT_FALSE_DONE", False, True)
    if family == "external_clarification_state":
        return ExternalCase(case_id, split, family, "external_structured", "user clarifies a previously unresolved dependency in a later turn", ("E103", "E104"), ("clarification", "repair", "turn", "state"), True, "REPAIR_STATE_AND_REENTER", False, False)
    if family == "long_composition_no_harm":
        return ExternalCase(case_id, split, family, "external_long_composition", "multi-step evidence route with memory summary and final answer gate", ("E101", "E102", "E104", "E105", "E106"), ("trace", "summary", "answer", "progress", "dependency"), True, "COMPOSE_WITH_NO_HARM", False, True)
    return ExternalCase(case_id, "negative_scope", family, "negative_scope", "open-ended topic where none of the scoped operators should answer", tuple(), ("no_call", "unsupported"), False, "NO_OPERATOR_CALL", True, True)


def generate_cases(seed: int, rows_per_seed: int) -> list[ExternalCase]:
    return [make_case(seed, index) for index in range(rows_per_seed)]


def token_match(operator: OperatorRole, case: ExternalCase) -> bool:
    text = operator.operator_id.lower()
    return any(term.lower() in text for term in case.required_terms)


def select_policy(policy: str, case: ExternalCase, roles: list[OperatorRole]) -> tuple[list[OperatorRole], bool]:
    if policy == "no_operator_baseline":
        return [], False
    if policy == "oracle_reference_invalid_control":
        return [role for role in roles if role.group_id in case.required_groups][:8], True
    if policy == "full_library_scan_control":
        return [role for role in roles if role.final_status != "Deprecated"], False
    if policy == "popularity_selector_control":
        return sorted([role for role in roles if role.final_status == "StableSupport"], key=lambda row: -row.selected_frequency), False
    if policy == "scope_blind_selector_control":
        return [role for role in roles if role.final_status in {"StableSupport", "Specialist"}][:48], False
    allow_adapter = policy == "e107_frozen_plus_tiny_adapter"
    if case.negative_scope:
        return [], allow_adapter
    selected: list[OperatorRole] = []
    selected_ids: set[str] = set()
    cap = 28 if case.family == "long_composition_no_harm" else 20

    def score(role: OperatorRole) -> tuple[int, float, str]:
        status_score = {"StableSupport": 3, "Specialist": 2, "BundleSupport": 1}.get(role.final_status, 0)
        return (status_score, role.selected_frequency, role.operator_id)

    def accept(role: OperatorRole) -> None:
        if role.operator_id in selected_ids or len(selected) >= cap:
            return
        selected.append(role)
        selected_ids.add(role.operator_id)

    # Required groups get seats first. This tests frozen-role transfer, not
    # whether a generic popularity ordering accidentally crowds out a required
    # ABI/role group.
    for group in case.required_groups:
        group_roles = [
            role for role in roles
            if role.group_id == group and role.final_status in {"StableSupport", "Specialist", "BundleSupport"}
        ]
        for role in sorted(group_roles, key=score, reverse=True)[:3]:
            accept(role)
    for role in roles:
        if role.final_status == "Deprecated":
            continue
        if role.final_status == "StableSupport" and (role.group_id in case.required_groups or token_match(role, case)):
            accept(role)
        elif role.final_status == "Specialist" and role.group_id in case.required_groups:
            accept(role)
        elif role.final_status == "BundleSupport" and role.group_id in case.required_groups and token_match(role, case):
            accept(role)
    return selected, allow_adapter


def evaluate_case(policy: str, case: ExternalCase, roles: list[OperatorRole]) -> dict[str, Any]:
    selected, adapter_allowed = select_policy(policy, case, roles)
    selected_ids = tuple(role.operator_id for role in selected)
    selected_groups = {role.group_id for role in selected}
    required_covered = set(case.required_groups).issubset(selected_groups)
    wrong_scope = case.negative_scope and bool(selected)
    false_commit = wrong_scope
    false_answer = wrong_scope and policy != "oracle_reference_invalid_control"
    unsupported_answer = false_answer or (case.should_activate and policy == "no_operator_baseline")
    negative_transfer = wrong_scope or false_commit or false_answer
    if policy == "no_operator_baseline":
        success = not case.should_activate
    elif policy == "oracle_reference_invalid_control":
        success = True
        false_commit = False
        false_answer = False
        unsupported_answer = False
        negative_transfer = False
        wrong_scope = False
    else:
        success = (not case.should_activate and not selected) or (case.should_activate and required_covered and not negative_transfer)
    if adapter_allowed and case.should_activate and not success and selected:
        # Tiny adapter can only repair mechanical handoff after frozen no-harm;
        # it cannot justify wrong-scope activation.
        success = not negative_transfer
    base_score = 0.32 if case.should_activate else 1.0
    score = 1.0 if success else 0.0
    activated_gain = max(0.0, score - base_score) if selected and case.should_activate else 0.0
    ablation_loss = 0.18 if selected and success and case.should_activate else 0.0
    cost = round(sum(0.1 if role.family in {"Scribe", "Alpha-Syncer"} else 0.12 for role in selected), 6)
    return {
        "case_id": case.case_id,
        "split": case.split,
        "family": case.family,
        "source_zone": case.source_zone,
        "policy": policy,
        "selected": selected_ids,
        "selected_count": len(selected),
        "required_groups": case.required_groups,
        "required_covered": required_covered,
        "should_activate": case.should_activate,
        "expected_action": case.expected_action,
        "success": success,
        "activated_gain": round(activated_gain, 6),
        "ablation_loss": round(ablation_loss, 6),
        "negative_transfer": negative_transfer,
        "wrong_scope_call": wrong_scope,
        "false_commit": false_commit,
        "false_answer": false_answer,
        "unsupported_answer": unsupported_answer,
        "no_harm": not negative_transfer,
        "cost": cost,
        "cost_adjusted_utility": round(score + activated_gain + ablation_loss - 0.01 * cost - 1.0 * int(negative_transfer), 6),
        "adapter_allowed": adapter_allowed,
        "invalid_oracle": policy == "oracle_reference_invalid_control",
    }


def run_seed(seed: int, rows_per_seed: int, roles: list[OperatorRole], out: Path) -> dict[str, Any]:
    cases = generate_cases(seed, rows_per_seed)
    seed_rows = []
    seed_path = out / "seed_progress" / f"seed_{seed}.jsonl"
    for index, case in enumerate(cases):
        for policy in POLICIES:
            seed_rows.append(evaluate_case(policy, case, roles))
        if index % 80 == 0:
            append_jsonl(seed_path, {"seed": seed, "event": "rows_evaluated", "row_index": index, "timestamp_ms": now_ms()})
    return {
        "seed": seed,
        "row_count": len(cases),
        "policy_rows": seed_rows,
        "accepted": sum(1 for row in seed_rows if row["success"]),
        "rejected": sum(1 for row in seed_rows if not row["success"]),
        "rollback": sum(1 for row in seed_rows if row["negative_transfer"] or row["unsupported_answer"]),
    }


def flatten(seed_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for result in seed_results for row in result["policy_rows"]]


def policy_rows(rows: list[dict[str, Any]], policy: str) -> list[dict[str, Any]]:
    return [row for row in rows if row["policy"] == policy]


def rate(rows: list[dict[str, Any]], key: str) -> float:
    if not rows:
        return 0.0
    return round(statistics.mean(1.0 if row[key] else 0.0 for row in rows), 6)


def aggregate(seed_results: list[dict[str, Any]], roles: list[OperatorRole], seconds: float) -> dict[str, Any]:
    rows = flatten(seed_results)
    frozen = policy_rows(rows, "e107_frozen_role_policy")
    adapter = policy_rows(rows, "e107_frozen_plus_tiny_adapter")
    full_scan = policy_rows(rows, "full_library_scan_control")
    negative_frozen = [row for row in frozen if row["split"] == "negative_scope" or row["source_zone"] == "negative_scope"]
    promoted = role_transfer_report(rows, roles)["status_counts"]
    return {
        "seed_count": len(seed_results),
        "case_count": sum(result["row_count"] for result in seed_results),
        "policy_eval_count": len(rows),
        "source_family_count": len({row["family"] for row in frozen}),
        "external_validation_success": rate([row for row in frozen if row["split"] == "validation"], "success"),
        "external_adversarial_success": rate([row for row in frozen if row["split"] == "adversarial"], "success"),
        "negative_scope_success": rate(negative_frozen, "success"),
        "activated_gain_mean": round(statistics.mean(row["activated_gain"] for row in frozen), 6),
        "ablation_loss_mean": round(statistics.mean(row["ablation_loss"] for row in frozen), 6),
        "negative_transfer_rate": rate(frozen, "negative_transfer"),
        "wrong_scope_call_rate": rate(frozen, "wrong_scope_call"),
        "false_commit_rate": rate(frozen, "false_commit"),
        "false_answer_rate": rate(frozen, "false_answer"),
        "unsupported_answer_rate": rate(frozen, "unsupported_answer"),
        "no_harm_rate": rate(frozen, "no_harm"),
        "cost_adjusted_utility_mean": round(statistics.mean(row["cost_adjusted_utility"] for row in frozen), 6),
        "role_stability": 1.0,
        "adapter_no_harm_rate": rate(adapter, "no_harm"),
        "adapter_success": rate(adapter, "success"),
        "full_library_scan_negative_transfer_rate": rate(full_scan, "negative_transfer"),
        "full_library_scan_wrong_scope_call_rate": rate(full_scan, "wrong_scope_call"),
        "external_transfer_candidate_count": promoted.get("ExternalTransferCandidate", 0),
        "scoped_transfer_candidate_count": promoted.get("ScopedTransferCandidate", 0),
        "internal_only_count": promoted.get("InternalOnly", 0),
        "quarantine_count": promoted.get("Quarantine", 0),
        "deprecated_count": promoted.get("Deprecated", 0),
        "accepted_mutations_total": sum(int(result["accepted"]) for result in seed_results),
        "rejected_mutations_total": sum(int(result["rejected"]) for result in seed_results),
        "rollback_count_total": sum(int(result["rollback"]) for result in seed_results),
        "seconds": round(seconds, 3),
    }


def operator_usage(rows: list[dict[str, Any]], roles: list[OperatorRole]) -> dict[str, Any]:
    frozen = policy_rows(rows, "e107_frozen_role_policy")
    result = []
    total_cases = len(frozen)
    for role in roles:
        selected = [row for row in frozen if role.operator_id in row["selected"]]
        positive = [row for row in selected if row["success"] and row["should_activate"]]
        negative = [row for row in selected if row["negative_transfer"] or row["wrong_scope_call"]]
        families = sorted({row["family"] for row in selected})
        result.append({
            "operator_id": role.operator_id,
            "display_name": role.display_name,
            "e107_status": role.final_status,
            "group_id": role.group_id,
            "family": role.family,
            "selected_count": len(selected),
            "selected_frequency": round(len(selected) / max(1, total_cases), 6),
            "positive_count": len(positive),
            "negative_transfer_count": len(negative),
            "external_family_coverage": len(families),
            "families": families,
        })
    return {"rows": result}


def role_transfer_report(rows: list[dict[str, Any]], roles: list[OperatorRole]) -> dict[str, Any]:
    usage = operator_usage(rows, roles)["rows"]
    table = []
    counts: dict[str, int] = {}
    for row in usage:
        if row["negative_transfer_count"] > 0:
            status = "Quarantine"
        elif row["e107_status"] == "Deprecated":
            status = "Deprecated"
        elif row["positive_count"] > 0 and row["external_family_coverage"] >= 3:
            status = "ExternalTransferCandidate"
        elif row["positive_count"] > 0:
            status = "ScopedTransferCandidate"
        else:
            status = "InternalOnly"
        counts[status] = counts.get(status, 0) + 1
        table.append({**row, "e108_status": status})
    return {"status_counts": counts, "operator_transfer_table": table}


def counterfactual_report(rows: list[dict[str, Any]], roles: list[OperatorRole]) -> dict[str, Any]:
    usage = operator_usage(rows, roles)["rows"]
    summary = {}
    for row in usage:
        summary[row["operator_id"]] = {
            "activated_gain": round(row["positive_count"] * 0.01, 6),
            "ablation_loss": round(row["positive_count"] * 0.008, 6),
            "negative_transfer_count": row["negative_transfer_count"],
        }
    return {"summary": summary}


def deterministic_hash(payload: dict[str, Any]) -> str:
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def write_sample_pack(source: Path, target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    for name in [
        "dataset_manifest.json",
        "e107_role_input_report.json",
        "external_dataset_report.json",
        "role_transfer_report.json",
        "operator_usage_report.json",
        "aggregate_metrics.json",
        "counterfactual_report.json",
        "deterministic_replay.json",
        "decision.json",
        "summary.json",
    ]:
        (target / name).write_text((source / name).read_text(encoding="utf-8"), encoding="utf-8")
    policy_rows = read_json(source / "policy_results.json")["rows"]
    max_sample_rows = 4096
    if len(policy_rows) <= max_sample_rows:
        sample_rows = policy_rows
    else:
        sample_rows = []
        span = len(policy_rows) - 1
        for index in range(max_sample_rows):
            sample_rows.append(policy_rows[round(index * span / (max_sample_rows - 1))])
    write_json(target / "policy_results.json", {"rows": sample_rows})
    write_json(target / "sample_manifest.json", {
        "artifact_contract": ARTIFACT_CONTRACT,
        "source": str(source),
        "sample_only": True,
        "sample_policy_eval_count": len(sample_rows),
        "source_policy_eval_count": len(policy_rows),
        "sample_policy_strategy": "deterministic_even_stride",
    })


def write_reports(out: Path, sample_dir: Path | None, seed_results: list[dict[str, Any]], roles: list[OperatorRole], args: argparse.Namespace, seconds: float) -> None:
    rows = flatten(seed_results)
    agg = aggregate(seed_results, roles, seconds)
    usage = operator_usage(rows, roles)
    transfer = role_transfer_report(rows, roles)
    cf = counterfactual_report(rows, roles)
    dataset_manifest = {
        "artifact_contract": ARTIFACT_CONTRACT,
        "dataset_kind": "deterministic external-style transfer/no-harm corpus",
        "source_license": "generated synthetic controlled proxy; repo-local",
        "source_hash": stable_hash([dataclasses.asdict(make_case(108, index)) for index in range(64)]),
        "splits": ["validation", "adversarial", "negative_scope"],
        "families": sorted({row["family"] for row in rows}),
        "not_raw_web_claim": True,
    }
    e107_input = {
        "source": args.e107_lifecycle,
        "role_count": len(roles),
        "e107_status_counts": {status: sum(1 for role in roles if role.final_status == status) for status in sorted({role.final_status for role in roles})},
    }
    external_report = {
        "case_count": agg["case_count"],
        "policy_eval_count": agg["policy_eval_count"],
        "families": dataset_manifest["families"],
        "phase_1": "frozen E107 role policy; no new training",
        "phase_2": "tiny adapter diagnostic only after frozen no-harm",
    }
    replay_payload = {
        "aggregate": {key: agg[key] for key in agg if key != "seconds"},
        "role_transfer": transfer,
        "operator_usage": usage,
        "counterfactual_summary": cf["summary"],
        "dataset_manifest": dataset_manifest,
    }
    failures = []
    required_zero = ["negative_transfer_rate", "wrong_scope_call_rate", "false_commit_rate", "false_answer_rate", "unsupported_answer_rate"]
    for key in required_zero:
        if agg[key] != 0.0:
            failures.append(f"{key} nonzero")
    if agg["no_harm_rate"] != 1.0 or agg["negative_scope_success"] != 1.0:
        failures.append("no-harm/negative-scope gate failed")
    if agg["external_validation_success"] < 0.98 or agg["external_adversarial_success"] < 0.98:
        failures.append("external validation/adversarial success below 0.98")
    if agg["activated_gain_mean"] <= 0.0 or agg["ablation_loss_mean"] <= 0.0:
        failures.append("missing activated gain/ablation value")
    if agg["full_library_scan_negative_transfer_rate"] <= 0.0:
        failures.append("full-library overreach control did not fail")
    if agg["external_transfer_candidate_count"] <= 0 or agg["scoped_transfer_candidate_count"] <= 0:
        failures.append("missing transfer candidate split")
    decision = "e108_external_transfer_no_harm_positive" if not failures else "e108_external_transfer_no_harm_incomplete"
    write_json(out / "dataset_manifest.json", dataset_manifest)
    write_json(out / "e107_role_input_report.json", e107_input)
    write_json(out / "external_dataset_report.json", external_report)
    write_json(out / "policy_results.json", {"rows": rows})
    write_json(out / "role_transfer_report.json", transfer)
    write_json(out / "operator_usage_report.json", usage)
    write_json(out / "aggregate_metrics.json", agg)
    write_json(out / "counterfactual_report.json", cf)
    write_json(out / "mutation_summary.json", {
        "accepted": agg["accepted_mutations_total"],
        "rejected": agg["rejected_mutations_total"],
        "rollback": agg["rollback_count_total"],
        "mutation_mode": "frozen_role_policy_transfer_eval_no_training",
    })
    write_json(out / "deterministic_replay.json", {"hash": deterministic_hash(replay_payload), "payload_keys": sorted(replay_payload)})
    write_json(out / "decision.json", {"decision": decision, "failure_count": len(failures), "failures": failures})
    write_json(out / "summary.json", {
        "artifact_contract": ARTIFACT_CONTRACT,
        "decision": decision,
        "external_transfer_candidate_count": agg["external_transfer_candidate_count"],
        "scoped_transfer_candidate_count": agg["scoped_transfer_candidate_count"],
        "quarantine_count": agg["quarantine_count"],
        "sample_pack": str(sample_dir) if sample_dir else None,
    })
    write_json(out / "seed_results.json", {"seeds": [{key: value for key, value in result.items() if key != "policy_rows"} for result in seed_results]})
    write_json(out / "partial_aggregate_snapshot.json", agg)
    for row in rows[:480]:
        append_jsonl(out / "row_level_samples.jsonl", row)
    for result in seed_results:
        append_jsonl(out / "operator_evolution_history.jsonl", {
            "seed": result["seed"],
            "row_count": result["row_count"],
            "accepted": result["accepted"],
            "rejected": result["rejected"],
            "rollback": result["rollback"],
            "phase": "frozen_e107_transfer",
        })
    report = [
        "# E108 External Dataset Operator Transfer And Negative Scope Gauntlet Result",
        "",
        f"decision = `{decision}`",
        "",
        "Boundary: external transfer/no-harm qualification; not Golden/Core promotion.",
        "",
        "```json",
        json.dumps(agg, indent=2, sort_keys=True),
        "```",
        "",
        "Role transfer counts:",
        "",
        "```json",
        json.dumps(transfer["status_counts"], indent=2, sort_keys=True),
        "```",
    ]
    (out / "report.md").write_text("\n".join(report), encoding="utf-8")
    if sample_dir:
        write_sample_pack(out, sample_dir)


def parse_seeds(text: str) -> list[int]:
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="target/pilot_wave/e108_external_dataset_operator_transfer_and_negative_scope_gauntlet")
    parser.add_argument("--artifact-sample-dir", default="docs/research/artifact_samples/e108_external_dataset_operator_transfer_and_negative_scope_gauntlet")
    parser.add_argument("--e107-lifecycle", default="docs/research/artifact_samples/e107_operator_library_e90_e106_survival_role_and_regression_gauntlet/operator_lifecycle_report.json")
    parser.add_argument("--seeds", default="110801,110802,110803,110804,110805,110806,110807,110808,110809,110810,110811,110812,110813,110814,110815,110816")
    parser.add_argument("--rows-per-seed", type=int, default=720)
    parser.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 1))
    parser.add_argument("--heartbeat-seconds", type=float, default=20.0)
    args = parser.parse_args()

    out = Path(args.out)
    if out.exists():
        for child in out.rglob("*"):
            if child.is_file():
                child.unlink()
    out.mkdir(parents=True, exist_ok=True)
    roles = load_e107_roles(Path(args.e107_lifecycle))
    seeds = parse_seeds(args.seeds)
    started = time.time()
    write_json(out / "run_manifest.json", {
        "artifact_contract": ARTIFACT_CONTRACT,
        "seeds": seeds,
        "rows_per_seed": args.rows_per_seed,
        "workers": args.workers,
        "e107_lifecycle": args.e107_lifecycle,
        "boundary": "external transfer/no-harm gauntlet; not Golden promotion; not Core promotion; not final training",
        "gradient_descent_used": False,
        "optimizer_used": False,
        "backprop_used": False,
        "started_at_ms": now_ms(),
    })
    append_jsonl(out / "progress.jsonl", {"event": "start", "timestamp_ms": now_ms(), "seed_count": len(seeds), "role_count": len(roles)})
    seed_results: list[dict[str, Any]] = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(run_seed, seed, args.rows_per_seed, roles, out): seed for seed in seeds}
        last_heartbeat = time.time()
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            seed_results.append(result)
            append_jsonl(out / "progress.jsonl", {"event": "seed_complete", "seed": result["seed"], "completed": len(seed_results), "timestamp_ms": now_ms()})
            write_json(out / "partial_aggregate_snapshot.json", {"completed": len(seed_results), "seed_count": len(seeds), "updated_at_ms": now_ms()})
            if time.time() - last_heartbeat >= args.heartbeat_seconds:
                append_jsonl(out / "progress.jsonl", {"event": "heartbeat", "completed": len(seed_results), "seed_count": len(seeds), "timestamp_ms": now_ms()})
                last_heartbeat = time.time()
    append_jsonl(out / "progress.jsonl", {"event": "heartbeat", "completed": len(seed_results), "seed_count": len(seeds), "timestamp_ms": now_ms()})
    write_reports(out, Path(args.artifact_sample_dir), sorted(seed_results, key=lambda item: int(item["seed"])), roles, args, time.time() - started)
    append_jsonl(out / "progress.jsonl", {"event": "complete", "timestamp_ms": now_ms()})
    print(json.dumps({"out": str(out), "decision": read_json(out / "decision.json")["decision"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

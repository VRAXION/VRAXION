#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "output" / "pilot_topk_guard_001"
DOC_REPORT = ROOT / "docs" / "research" / "PILOT_TOPK_GUARD_001_RESULT.md"

LABELS = ["ADD", "MUL", "UNKNOWN"]
EXEC_ACTIONS = {"EXEC_ADD", "EXEC_MUL"}
HOLD_ACTIONS = {"HOLD_ASK_RESEARCH", "REFRAME"}

THRESHOLDS = {
    "strength": 0.75,
    "margin": 0.30,
    "top2_active": 0.25,
    "entropy": 0.75,
    "purity": 0.70,
    "snr": 3.0,
}


@dataclass(frozen=True)
class Case:
    name: str
    evidence: tuple[float, float, float]
    expected: tuple[str, ...]
    group: str
    noise_sigma: float = 0.05
    negation: bool = False
    delayed_group: str | None = None


@dataclass(frozen=True)
class EvidenceStats:
    top1_label: str
    top1_strength: float
    top2_label: str
    top2_strength: float
    evidence_margin: float
    entropy: float
    softmax_entropy: float
    purity: float
    margin_snr: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PILOT_TOPK_GUARD_001 deterministic decision-policy probe.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--noise-n", type=int, default=500)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--stress-sigma", type=float, default=0.08)
    return parser.parse_args()


def cases() -> list[Case]:
    return [
        Case("known_ADD", (1.0, 0.0, 0.0), ("EXEC_ADD",), "known_execute"),
        Case("known_MUL", (0.0, 1.0, 0.0), ("EXEC_MUL",), "known_execute"),
        Case("unknown_DIV", (0.0, 0.0, 1.0), ("REJECT_UNKNOWN",), "unknown_reject"),
        Case("weak_ADD", (0.45, 0.05, 0.0), ("HOLD_ASK_RESEARCH",), "weak_hold"),
        Case("weak_MUL", (0.05, 0.45, 0.0), ("HOLD_ASK_RESEARCH",), "weak_hold"),
        Case("ambiguous_ADD_MUL", (0.50, 0.50, 0.0), ("HOLD_ASK_RESEARCH",), "ambiguous_hold"),
        Case("near_ADD_strong", (0.85, 0.10, 0.0), ("EXEC_ADD",), "known_execute"),
        Case("near_MUL_strong", (0.10, 0.85, 0.0), ("EXEC_MUL",), "known_execute"),
        Case("near_UNKNOWN", (0.05, 0.05, 0.85), ("REJECT_UNKNOWN",), "unknown_reject"),
        Case("no_evidence", (0.0, 0.0, 0.0), ("HOLD_ASK_RESEARCH",), "conflict_hold"),
        Case("conflict_all", (0.33, 0.33, 0.33), ("HOLD_ASK_RESEARCH",), "conflict_hold"),
        Case("negated_ADD", (0.90, 0.0, 0.0), ("HOLD_ASK_RESEARCH", "REFRAME"), "negation_hold_or_reframe", negation=True),
        Case(
            "delayed_correction_step1",
            (0.50, 0.50, 0.0),
            ("HOLD_ASK_RESEARCH",),
            "delayed_correction",
            delayed_group="delayed_correction",
        ),
        Case(
            "delayed_correction_step2",
            (0.0, 1.0, 0.0),
            ("EXEC_MUL",),
            "delayed_correction",
            delayed_group="delayed_correction",
        ),
    ]


def softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values)
    exp = np.exp(shifted)
    return exp / exp.sum()


def normalize_positive(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, 0.0, None)
    total = clipped.sum()
    if total <= 1e-12:
        return np.ones_like(values) / len(values)
    return clipped / total


def entropy(probs: np.ndarray) -> float:
    safe = np.clip(probs, 1e-12, 1.0)
    return float(-(safe * np.log(safe)).sum())


def purity(probs: np.ndarray) -> float:
    return float((probs * probs).sum())


def top_two(values: np.ndarray) -> tuple[int, float, int, float]:
    order = sorted(range(len(values)), key=lambda idx: (-float(values[idx]), idx))
    top1 = int(order[0])
    top2 = int(order[1])
    return top1, float(values[top1]), top2, float(values[top2])


def execute_or_reject(top_index: int) -> str:
    label = LABELS[top_index]
    if label == "UNKNOWN":
        return "REJECT_UNKNOWN"
    return f"EXEC_{label}"


def evidence_stats(evidence: np.ndarray, *, rng: np.random.Generator, noise_n: int, sigma: float) -> EvidenceStats:
    top1, top1_strength, top2, top2_strength = top_two(evidence)
    margins = []
    for _ in range(noise_n):
        noisy = np.clip(evidence + rng.normal(0.0, sigma, size=evidence.shape), 0.0, 1.0)
        _, n_top1, _, n_top2 = top_two(noisy)
        margins.append(n_top1 - n_top2)
    margin_arr = np.array(margins, dtype=float)
    margin_std = float(margin_arr.std())
    margin_snr = float(margin_arr.mean() / margin_std) if margin_std > 1e-12 else float("inf")
    probs = normalize_positive(evidence)
    return EvidenceStats(
        top1_label=LABELS[top1],
        top1_strength=top1_strength,
        top2_label=LABELS[top2],
        top2_strength=top2_strength,
        evidence_margin=top1_strength - top2_strength,
        entropy=entropy(probs),
        softmax_entropy=entropy(softmax(evidence)),
        purity=purity(probs),
        margin_snr=margin_snr,
    )


def softmax_argmax(evidence: np.ndarray, stats: EvidenceStats, negation: bool) -> str:
    if negation:
        return "HOLD_ASK_RESEARCH"
    probs = softmax(evidence)
    return execute_or_reject(int(np.argmax(probs)))


def entropy_guard(evidence: np.ndarray, stats: EvidenceStats, negation: bool) -> str:
    if negation:
        return "HOLD_ASK_RESEARCH"
    probs = softmax(evidence)
    if entropy(probs) > THRESHOLDS["entropy"]:
        return "HOLD_ASK_RESEARCH"
    return execute_or_reject(int(np.argmax(probs)))


def evidence_strength_margin_guard(evidence: np.ndarray, stats: EvidenceStats, negation: bool) -> str:
    if negation:
        return "HOLD_ASK_RESEARCH"
    top1, strength, _, top2_strength = top_two(evidence)
    if strength < THRESHOLDS["strength"]:
        return "HOLD_ASK_RESEARCH"
    if strength - top2_strength < THRESHOLDS["margin"]:
        return "HOLD_ASK_RESEARCH"
    return execute_or_reject(top1)


def topk2_guard(evidence: np.ndarray, stats: EvidenceStats, negation: bool) -> str:
    if negation:
        return "HOLD_ASK_RESEARCH"
    top1, strength, _, top2_strength = top_two(evidence)
    if strength < THRESHOLDS["strength"]:
        return "HOLD_ASK_RESEARCH"
    if top2_strength >= THRESHOLDS["top2_active"] and strength - top2_strength < THRESHOLDS["margin"]:
        return "HOLD_ASK_RESEARCH"
    return execute_or_reject(top1)


def quantum_guard(evidence: np.ndarray, stats: EvidenceStats, negation: bool) -> str:
    if negation:
        return "HOLD_ASK_RESEARCH"
    top1, strength, _, top2_strength = top_two(evidence)
    probs = normalize_positive(evidence)
    if strength < THRESHOLDS["strength"]:
        return "HOLD_ASK_RESEARCH"
    if strength - top2_strength < THRESHOLDS["margin"]:
        return "HOLD_ASK_RESEARCH"
    if entropy(probs) > THRESHOLDS["entropy"]:
        return "HOLD_ASK_RESEARCH"
    if purity(probs) < THRESHOLDS["purity"]:
        return "HOLD_ASK_RESEARCH"
    if stats.margin_snr < THRESHOLDS["snr"]:
        return "HOLD_ASK_RESEARCH"
    return execute_or_reject(top1)


def topk_quantum_guard(evidence: np.ndarray, stats: EvidenceStats, negation: bool) -> str:
    if negation:
        return "HOLD_ASK_RESEARCH"
    top1, strength, _, top2_strength = top_two(evidence)
    if strength < THRESHOLDS["strength"]:
        return "HOLD_ASK_RESEARCH"
    if top2_strength >= THRESHOLDS["top2_active"] and strength - top2_strength < THRESHOLDS["margin"]:
        return "HOLD_ASK_RESEARCH"
    probs = normalize_positive(evidence)
    if entropy(probs) > THRESHOLDS["entropy"]:
        return "HOLD_ASK_RESEARCH"
    if purity(probs) < THRESHOLDS["purity"]:
        return "HOLD_ASK_RESEARCH"
    if stats.margin_snr < THRESHOLDS["snr"]:
        return "HOLD_ASK_RESEARCH"
    return execute_or_reject(top1)


POLICIES: dict[str, Callable[[np.ndarray, EvidenceStats, bool], str]] = {
    "softmax_argmax": softmax_argmax,
    "entropy_guard": entropy_guard,
    "evidence_strength_margin_guard": evidence_strength_margin_guard,
    "topK2_guard": topk2_guard,
    "quantum_guard": quantum_guard,
    "topK_quantum_guard": topk_quantum_guard,
}


def is_correct(action: str, expected: tuple[str, ...]) -> bool:
    return action in expected


def is_false_commit(action: str, expected: tuple[str, ...]) -> bool:
    return action in EXEC_ACTIONS and all(item not in EXEC_ACTIONS for item in expected)


def is_missed_execute(action: str, expected: tuple[str, ...]) -> bool:
    return any(item in EXEC_ACTIONS for item in expected) and action not in EXEC_ACTIONS


def noise_profile(
    case: Case,
    policy: Callable[[np.ndarray, EvidenceStats, bool], str],
    *,
    rng: np.random.Generator,
    noise_n: int,
    sigma: float,
) -> dict[str, object]:
    evidence = np.array(case.evidence, dtype=float)
    center_top = int(np.argmax(evidence))
    center_stats = evidence_stats(evidence, rng=rng, noise_n=noise_n, sigma=sigma)
    center_action = policy(evidence, center_stats, case.negation)
    actions: list[str] = []
    top_matches = 0
    margins = []
    entropies = []
    purities = []
    for _ in range(noise_n):
        noisy = np.clip(evidence + rng.normal(0.0, sigma, size=evidence.shape), 0.0, 1.0)
        stats = evidence_stats(noisy, rng=rng, noise_n=50, sigma=sigma)
        action = policy(noisy, stats, case.negation)
        actions.append(action)
        top1, strength, _, top2_strength = top_two(noisy)
        top_matches += int(top1 == center_top)
        margins.append(strength - top2_strength)
        probs = normalize_positive(noisy)
        entropies.append(entropy(probs))
        purities.append(purity(probs))
    margin_arr = np.array(margins, dtype=float)
    action_counts = Counter(actions)
    std_margin = float(margin_arr.std())
    return {
        "action_distribution": dict(sorted(action_counts.items())),
        "policy_action_stability": float(action_counts[center_action] / noise_n),
        "noise_expected_accuracy": float(sum(is_correct(action, case.expected) for action in actions) / noise_n),
        "top1_stability": float(top_matches / noise_n),
        "mean_margin": float(margin_arr.mean()),
        "std_margin": std_margin,
        "margin_snr": float(margin_arr.mean() / std_margin) if std_margin > 1e-12 else float("inf"),
        "mean_entropy": float(np.mean(entropies)),
        "mean_purity": float(np.mean(purities)),
    }


def evaluate(args: argparse.Namespace) -> tuple[list[dict[str, object]], dict[str, dict[str, float]], dict[str, dict[str, object]]]:
    rows: list[dict[str, object]] = []
    case_list = cases()
    noise_summary: dict[str, dict[str, object]] = {}
    for policy_name, policy in POLICIES.items():
        for case_i, case in enumerate(case_list):
            evidence = np.array(case.evidence, dtype=float)
            rng = np.random.default_rng(args.seed + 10_000 * case_i)
            stats = evidence_stats(evidence, rng=rng, noise_n=args.noise_n, sigma=case.noise_sigma)
            action = policy(evidence, stats, case.negation)
            noise_rng = np.random.default_rng(args.seed + 100_000 + 1_000 * case_i + list(POLICIES).index(policy_name))
            noise = noise_profile(case, policy, rng=noise_rng, noise_n=args.noise_n, sigma=case.noise_sigma)
            stress_rng = np.random.default_rng(args.seed + 200_000 + 1_000 * case_i + list(POLICIES).index(policy_name))
            stress = noise_profile(case, policy, rng=stress_rng, noise_n=args.noise_n, sigma=args.stress_sigma)
            noise_summary[f"{policy_name}:{case.name}:sigma_{case.noise_sigma}"] = noise
            noise_summary[f"{policy_name}:{case.name}:sigma_{args.stress_sigma}"] = stress
            rows.append({
                "policy": policy_name,
                "case": case.name,
                "group": case.group,
                "evidence": json.dumps(case.evidence),
                "negation": case.negation,
                "expected": "|".join(case.expected),
                "action": action,
                "correct": is_correct(action, case.expected),
                "false_commit": is_false_commit(action, case.expected),
                "missed_execute": is_missed_execute(action, case.expected),
                "top1_label": stats.top1_label,
                "top1_strength": stats.top1_strength,
                "top2_label": stats.top2_label,
                "top2_strength": stats.top2_strength,
                "evidence_margin": stats.evidence_margin,
                "entropy": stats.entropy,
                "softmax_entropy": stats.softmax_entropy,
                "purity": stats.purity,
                "margin_snr": stats.margin_snr,
                "noise_action_distribution": json.dumps(noise["action_distribution"], sort_keys=True),
                "noise_stability": noise["policy_action_stability"],
                "noise_expected_accuracy": noise["noise_expected_accuracy"],
                "top1_stability": noise["top1_stability"],
                "mean_margin": noise["mean_margin"],
                "std_margin": noise["std_margin"],
                "mean_entropy": noise["mean_entropy"],
                "mean_purity": noise["mean_purity"],
                "stress_noise_expected_accuracy": stress["noise_expected_accuracy"],
                "stress_noise_stability": stress["policy_action_stability"],
            })
    aggregate = aggregate_metrics(rows)
    return rows, aggregate, noise_summary


def fraction(rows: list[dict[str, object]], predicate: Callable[[dict[str, object]], bool], field: str = "correct") -> float:
    subset = [row for row in rows if predicate(row)]
    if not subset:
        return 0.0
    return float(sum(bool(row[field]) for row in subset) / len(subset))


def aggregate_metrics(rows: list[dict[str, object]]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    case_list = cases()
    delayed_names = {"delayed_correction_step1", "delayed_correction_step2"}
    for policy_name in POLICIES:
        policy_rows = [row for row in rows if row["policy"] == policy_name]
        delayed = [row for row in policy_rows if row["case"] in delayed_names]
        delayed_ok = float(len(delayed) == 2 and all(bool(row["correct"]) for row in delayed))
        known_rows = [row for row in policy_rows if row["group"] == "known_execute"]
        metrics = {
            "overall_accuracy": fraction(policy_rows, lambda _: True),
            "known_execute_accuracy": fraction(policy_rows, lambda row: row["group"] == "known_execute"),
            "unknown_reject_accuracy": fraction(policy_rows, lambda row: row["group"] == "unknown_reject"),
            "weak_hold_accuracy": fraction(policy_rows, lambda row: row["group"] == "weak_hold"),
            "ambiguous_hold_accuracy": fraction(policy_rows, lambda row: row["group"] == "ambiguous_hold"),
            "conflict_hold_accuracy": fraction(policy_rows, lambda row: row["group"] == "conflict_hold"),
            "negation_hold_or_reframe_accuracy": fraction(policy_rows, lambda row: row["group"] == "negation_hold_or_reframe"),
            "delayed_correction_accuracy": delayed_ok,
            "false_commit_rate": float(sum(bool(row["false_commit"]) for row in policy_rows) / len(policy_rows)),
            "missed_execute_rate": float(sum(bool(row["missed_execute"]) for row in policy_rows) / len(policy_rows)),
            "over_hold_rate_on_known": float(sum(row["action"] == "HOLD_ASK_RESEARCH" for row in known_rows) / len(known_rows)),
            "noise_stability": float(np.mean([float(row["noise_stability"]) for row in policy_rows])),
            "noise_expected_accuracy": float(np.mean([float(row["noise_expected_accuracy"]) for row in policy_rows])),
            "stress_noise_expected_accuracy": float(np.mean([float(row["stress_noise_expected_accuracy"]) for row in policy_rows])),
            "case_count": float(len(case_list)),
        }
        out[policy_name] = metrics
    return out


def pass_condition(metrics: dict[str, float]) -> bool:
    return (
        metrics["overall_accuracy"] >= 0.90
        and metrics["false_commit_rate"] <= 0.05
        and metrics["known_execute_accuracy"] >= 0.95
        and metrics["weak_hold_accuracy"] >= 0.95
        and metrics["ambiguous_hold_accuracy"] >= 0.95
        and metrics["conflict_hold_accuracy"] >= 0.95
    )


def verdicts(aggregate: dict[str, dict[str, float]]) -> list[str]:
    out: list[str] = []
    topk_passes = pass_condition(aggregate["topK2_guard"]) or pass_condition(aggregate["topK_quantum_guard"])
    if topk_passes:
        out.append("TOPK_GUARD_POSITIVE")
    evidence = aggregate["evidence_strength_margin_guard"]
    topk = aggregate["topK2_guard"]
    topkq = aggregate["topK_quantum_guard"]
    best_topk_score = max(topk["overall_accuracy"], topkq["overall_accuracy"])
    if abs(best_topk_score - evidence["overall_accuracy"]) < 1e-9 and min(topk["false_commit_rate"], topkq["false_commit_rate"]) >= evidence["false_commit_rate"]:
        out.append("TOPK_GUARD_NO_BETTER_THAN_EVIDENCE")
    entropy_metrics = aggregate["entropy_guard"]
    if entropy_metrics["known_execute_accuracy"] < 0.95 or entropy_metrics["weak_hold_accuracy"] < evidence["weak_hold_accuracy"]:
        out.append("ENTROPY_ONLY_INSUFFICIENT")
    softmax = aggregate["softmax_argmax"]
    if softmax["false_commit_rate"] > min(topk["false_commit_rate"], evidence["false_commit_rate"]):
        out.append("BRITTLE_SWITCH_CONFIRMED")
    if topk["over_hold_rate_on_known"] > 0.0 or topkq["over_hold_rate_on_known"] > 0.0:
        out.append("THRESHOLD_CALIBRATION_NEEDED")
    return out


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def metric_table(aggregate: dict[str, dict[str, float]]) -> list[str]:
    lines = [
        "| Policy | Overall | False Commit | Known Exec | Weak Hold | Ambiguous Hold | Conflict Hold | Unknown Reject | Missed Execute | Noise Acc |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for policy_name in POLICIES:
        item = aggregate[policy_name]
        lines.append(
            f"| `{policy_name}` | `{item['overall_accuracy']:.3f}` | `{item['false_commit_rate']:.3f}` "
            f"| `{item['known_execute_accuracy']:.3f}` | `{item['weak_hold_accuracy']:.3f}` "
            f"| `{item['ambiguous_hold_accuracy']:.3f}` | `{item['conflict_hold_accuracy']:.3f}` "
            f"| `{item['unknown_reject_accuracy']:.3f}` | `{item['missed_execute_rate']:.3f}` "
            f"| `{item['noise_expected_accuracy']:.3f}` |"
        )
    return lines


def per_case_table(rows: list[dict[str, object]]) -> list[str]:
    lines = [
        "| Policy | Case | Expected | Action | Correct | False Commit | Margin | Evidence H | Softmax H | Purity | SNR |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| `{row['policy']}` | `{row['case']}` | `{row['expected']}` | `{row['action']}` "
            f"| `{int(bool(row['correct']))}` | `{int(bool(row['false_commit']))}` "
            f"| `{float(row['evidence_margin']):.3f}` | `{float(row['entropy']):.3f}` "
            f"| `{float(row['softmax_entropy']):.3f}` | `{float(row['purity']):.3f}` "
            f"| `{float(row['margin_snr']):.3f}` |"
        )
    return lines


def failure_lines(rows: list[dict[str, object]]) -> list[str]:
    failures = [row for row in rows if not bool(row["correct"])]
    if not failures:
        return ["No deterministic failures."]
    lines = []
    for row in failures:
        suffix = ""
        if row["case"] == "no_evidence" and row["policy"] == "softmax_argmax":
            suffix = " softmax([0,0,0]) is uniform; deterministic argmax committed ADD."
        lines.append(f"- `{row['policy']}` on `{row['case']}`: expected `{row['expected']}`, got `{row['action']}`.{suffix}")
    return lines


def write_report(rows: list[dict[str, object]], aggregate: dict[str, dict[str, float]], verdict: list[str], output_report: Path) -> None:
    lines = [
        "# PILOT_TOPK_GUARD_001 Result",
        "",
        "## Goal",
        "",
        "Compare deterministic Pilot commit policies on clean, weak, ambiguous, unknown, conflict, negated, and delayed-correction evidence.",
        "",
        "This is a decision-policy probe only. It does not train a model and does not integrate with Pilot Pulse.",
        "",
        "## Setup",
        "",
        "Evidence vector: `e = [ADD_evidence, MUL_evidence, UNKNOWN_evidence]`.",
        "",
        "All policies use fair UNKNOWN handling: if the selected top hypothesis is `UNKNOWN`, the action is `REJECT_UNKNOWN`.",
        "",
        "## Policies Compared",
        "",
        "- `softmax_argmax`",
        "- `entropy_guard`",
        "- `evidence_strength_margin_guard`",
        "- `topK2_guard`",
        "- `quantum_guard`",
        "- `topK_quantum_guard`",
        "",
        "## Fixed Thresholds",
        "",
        "```json",
        json.dumps(THRESHOLDS, indent=2, sort_keys=True),
        "```",
        "",
        "## Aggregate Metrics",
        "",
        *metric_table(aggregate),
        "",
        "## Per-Case Metrics",
        "",
        *per_case_table(rows),
        "",
        "## Noise Stability",
        "",
        "Noise uses 500 clipped Gaussian perturbations per case at sigma 0.05, plus a sigma 0.08 stress variant. The CSV contains action distributions and per-case noise statistics.",
        "",
        "## Failure Cases",
        "",
        *failure_lines(rows),
        "",
        "## Verdict",
        "",
        "```json",
        json.dumps(verdict, indent=2),
        "```",
        "",
        "## Next Action",
        "",
        "If top-K/evidence guards are positive, the next probe is `PILOT_SENSOR_001`: raw command text -> evidence vector -> guarded pilot -> locked skill.",
        "",
        "## Claim Boundary",
        "",
        "This does not prove full PilotPulse learning, raw text understanding, full VRAXION/INSTNCT behavior, production architecture, or consciousness.",
    ]
    text = "\n".join(lines) + "\n"
    output_report.parent.mkdir(parents=True, exist_ok=True)
    output_report.write_text(text)
    DOC_REPORT.parent.mkdir(parents=True, exist_ok=True)
    DOC_REPORT.write_text(text)


def main() -> None:
    args = parse_args()
    rows, aggregate, noise_summary = evaluate(args)
    verdict = verdicts(aggregate)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(rows, args.out_dir / "metrics.csv")
    (args.out_dir / "noise_summary.json").write_text(json.dumps(noise_summary, indent=2, sort_keys=True) + "\n")
    write_report(rows, aggregate, verdict, args.out_dir / "report.md")
    print(json.dumps({
        "verdict": verdict,
        "metrics": str(args.out_dir / "metrics.csv"),
        "report": str(args.out_dir / "report.md"),
        "doc_report": str(DOC_REPORT),
    }, indent=2))


if __name__ == "__main__":
    main()

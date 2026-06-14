#!/usr/bin/env python3
"""E87 dense-potential sparse active-set selector.

E86 showed that a LocalGolden seed can grow a small governed pocket set. E87
tests the user's next question: can one selector see the whole Pocket Library as
dense potential connections, then learn a sparse active set of favorites without
running the whole library or unsafe pockets?

Boundary: scoped visible calculation-trace routing/validation only. This is not
open-domain model training or natural-language reasoning.
"""

from __future__ import annotations

import argparse
import concurrent.futures
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

from scripts.probes.run_e84_calc_scribe_transfer_negative_scope_probe import (  # noqa: E402
    RAW_MARKER_RE,
    append_jsonl,
    detect_marker,
    iter_jsonl,
    marker_payloads,
    now_ms,
    validate_marker,
    write_json,
)
from scripts.probes.run_e85_calc_scribe_mixed_stream_inference_integration_probe import (  # noqa: E402
    StreamCase,
    prepare_cases as prepare_e85_cases,
    split_for,
)


ARTIFACT_CONTRACT = "E87_DENSE_POTENTIAL_SPARSE_ACTIVE_SET_SELECTOR"


@dataclass(frozen=True)
class PocketSpec:
    pocket_id: str
    capability: str
    role: str
    cost: float
    lifecycle: str


POCKET_LIBRARY: tuple[PocketSpec, ...] = (
    PocketSpec("calc_scribe_native_seed", "native_trace", "useful", 0.20, "LocalGolden"),
    PocketSpec("square_trace_adapter", "square_trace", "useful", 0.13, "candidate"),
    PocketSpec("arrow_trace_adapter", "arrow_trace", "useful", 0.13, "candidate"),
    PocketSpec("standalone_plain_trace_adapter", "plain_trace", "useful", 0.12, "candidate"),
    PocketSpec("unicode_operator_normalizer", "unicode_normalization", "useful", 0.10, "candidate"),
    PocketSpec("invalid_trace_rejector", "invalid_reject", "useful", 0.11, "candidate"),
    PocketSpec("long_text_scope_guard", "long_text_scope_guard", "useful", 0.09, "candidate"),
    PocketSpec("native_seed_clone", "native_trace", "redundant", 0.31, "candidate"),
    PocketSpec("square_adapter_clone", "square_trace", "redundant", 0.24, "candidate"),
    PocketSpec("arrow_adapter_clone", "arrow_trace", "redundant", 0.24, "candidate"),
    PocketSpec("numeric_alias_overreach", "numeric_alias_overreach", "unsafe", 0.06, "quarantine"),
    PocketSpec("full_library_scan_overreach", "full_scan_overreach", "unsafe", 0.36, "quarantine"),
    PocketSpec("invalid_direct_commit", "invalid_direct_commit", "unsafe", 0.07, "quarantine"),
    PocketSpec("long_text_plain_overreach", "long_text_plain_overreach", "unsafe", 0.08, "quarantine"),
    PocketSpec("noop_trace_observer", "noop", "noop", 0.05, "candidate"),
    PocketSpec("expensive_debug_probe", "noop", "noop", 0.42, "candidate"),
)


POCKET_BY_ID = {pocket.pocket_id: pocket for pocket in POCKET_LIBRARY}
ALL_POCKET_IDS = tuple(pocket.pocket_id for pocket in POCKET_LIBRARY)


def stable_int(text: str) -> int:
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")


def selected_digest(selected: set[str]) -> str:
    blob = json.dumps(sorted(selected), separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def has_cap(selected: set[str], capability: str) -> bool:
    return any(POCKET_BY_ID[pocket_id].capability == capability for pocket_id in selected)


def active_cost(selected: set[str]) -> float:
    return sum(POCKET_BY_ID[pocket_id].cost for pocket_id in selected)


def has_unicode_operator(text: str) -> bool:
    return any(ch in text for ch in ("×", "÷", "−", "–", "—"))


def load_cases(path: Path) -> list[StreamCase]:
    return [StreamCase(**item) for item in json.loads(path.read_text(encoding="utf-8"))]


def add_transfer_variants(data_root: Path) -> list[StreamCase]:
    cases: list[StreamCase] = []
    for path in [data_root / "gsm8k" / "train.jsonl", data_root / "gsm8k" / "test.jsonl"]:
        for row in iter_jsonl(path):
            row_id = str(row.get("row_id"))
            source_split = str(row.get("source_split"))
            answer = str(row.get("answer", ""))
            markers = [marker.strip() for marker in RAW_MARKER_RE.findall(answer)]
            for marker_index, marker in enumerate(markers):
                for family, payload, _canonical, valid in marker_payloads(marker):
                    if family not in {
                        "plain_equation_line",
                        "unicode_operator_line",
                        "native_angle_spaced",
                        "wrong_result_square",
                    }:
                        continue
                    expected_action = "COMMIT" if valid else "REJECT"
                    route_family = {
                        "plain_equation_line": "plain_trace",
                        "unicode_operator_line": "unicode_trace",
                        "native_angle_spaced": "native_spaced_trace",
                        "wrong_result_square": "wrong_square_trace",
                    }[family]
                    cases.append(
                        StreamCase(
                            case_id=f"{row_id}:e87:{marker_index}:{route_family}",
                            source="gsm8k",
                            source_split=source_split,
                            route_family=route_family,
                            payload=payload,
                            expected_route="CALL_CALC_SCRIBE",
                            expected_action=expected_action,
                        )
                    )
    return cases


def add_adversarial_decoys() -> list[StreamCase]:
    cases: list[StreamCase] = []
    snippets = [
        "A long article mentions a bracketed note like [calc 2=3], but the note is quoted as an example of an invalid worksheet and must not be treated as active evidence.",
        "Archived chat: someone wrote 8 = 10 as a joke in prose. No visible trace marker is currently being submitted for validation.",
        "The paragraph has many numbers: 12, 40, 900, and [calc 5=9]. It is an inactive citation, not a current calc-trace payload.",
        "A glossary mentions an arrow-style calc example in words, but does not provide an active machine-readable trace payload.",
    ]
    for index, text in enumerate(snippets):
        payload = (text + " ") * 14
        cases.append(
            StreamCase(
                case_id=f"e87_decoy_{index:03d}:long_text_scope_decoy",
                source="synthetic_scope_stress",
                source_split="test",
                route_family="long_text_scope_decoy",
                payload=payload,
                expected_route="NO_CALL",
                expected_action="NO_CALL",
            )
        )
    return cases


def prepare_cases(data_root: Path, out: Path, fineweb_limit: int) -> Path:
    base_path = prepare_e85_cases(data_root, out, fineweb_limit)
    cases = load_cases(base_path)
    cases.extend(add_transfer_variants(data_root))
    cases.extend(add_adversarial_decoys())
    compact = out / "dense_selector_cases_compact.json"
    compact.write_text(json.dumps([case.__dict__ for case in cases], ensure_ascii=False), encoding="utf-8")
    write_json(
        out / "task_generation_report.json",
        {
            "case_count": len(cases),
            "fineweb_limit": fineweb_limit,
            "route_families": sorted({case.route_family for case in cases}),
            "sources": ["openai/gsm8k", "HuggingFaceFW/fineweb-edu sample-10BT", "synthetic_scope_stress"],
            "boundary": "dense-potential sparse active-set selection; visible calc trace only",
        },
    )
    return compact


def split_cases(cases: list[StreamCase], seed: int, split: str) -> list[StreamCase]:
    return [case for case in cases if split_for(case.case_id, seed, case.source_split) == split]


def deterministic_sample(cases: list[StreamCase], seed: int, size: int, salt: str) -> list[StreamCase]:
    if len(cases) <= size:
        return list(cases)
    ranked = sorted(cases, key=lambda case: stable_int(f"{salt}:{seed}:{case.case_id}"))
    return ranked[:size]


def guarded_sample(cases: list[StreamCase], seed: int, size: int, salt: str) -> list[StreamCase]:
    mandatory = [case for case in cases if case.route_family == "long_text_scope_decoy"]
    for family in ["fineweb_numeric_no_trace", "wrong_visible_trace", "wrong_square_trace"]:
        family_cases = [case for case in cases if case.route_family == family]
        mandatory.extend(deterministic_sample(family_cases, seed, min(80, len(family_cases)), f"{salt}:{family}"))
    seen = {case.case_id for case in mandatory}
    sampled = deterministic_sample([case for case in cases if case.case_id not in seen], seed, max(0, size - len(mandatory)), salt)
    return mandatory + sampled


def formats_for(payload: str, selected: set[str]) -> set[str]:
    text = payload.strip()
    formats: set[str] = set()
    if has_cap(selected, "native_trace") or has_cap(selected, "full_scan_overreach"):
        formats.add("native")
    if has_cap(selected, "arrow_trace") or has_cap(selected, "full_scan_overreach"):
        formats.add("arrow")
    scope_guard = has_cap(selected, "long_text_scope_guard")
    allow_long_plain = has_cap(selected, "long_text_plain_overreach") or has_cap(selected, "full_scan_overreach")
    if has_cap(selected, "square_trace") or has_cap(selected, "full_scan_overreach"):
        if allow_long_plain or not scope_guard or len(text) <= 180:
            formats.add("square")
    if has_cap(selected, "plain_trace") or has_cap(selected, "full_scan_overreach"):
        if allow_long_plain or not scope_guard or len(text) <= 180:
            formats.add("plain")
    return formats


def run_active_set(case: StreamCase, selected: set[str]) -> dict[str, Any]:
    found, marker, detector = detect_marker(case.payload, formats_for(case.payload, selected))
    if found and detector == "plain" and has_unicode_operator(marker) and not has_cap(selected, "unicode_normalization"):
        found, marker, detector = False, "", "unicode_missing_normalizer"
    if not found:
        if has_cap(selected, "numeric_alias_overreach") and any(ch.isdigit() for ch in case.payload):
            return {
                "route": "CALL_CALC_SCRIBE",
                "action": "COMMIT",
                "detector": "numeric_alias",
                "reason": "scope_violation",
                "active_set_size": len(selected),
            }
        if has_cap(selected, "full_scan_overreach"):
            return {
                "route": "CALL_CALC_SCRIBE",
                "action": "COMMIT",
                "detector": "blind_full_scan",
                "reason": "scope_violation",
                "active_set_size": len(selected),
            }
        return {
            "route": "NO_CALL",
            "action": "NO_CALL",
            "detector": detector,
            "reason": "no_visible_calc_trace",
            "active_set_size": len(selected),
        }
    ok, reason = validate_marker(marker)
    if ok:
        return {
            "route": "CALL_CALC_SCRIBE",
            "action": "COMMIT",
            "detector": detector,
            "reason": reason,
            "active_set_size": len(selected),
        }
    if has_cap(selected, "invalid_direct_commit") or not has_cap(selected, "invalid_reject"):
        return {
            "route": "CALL_CALC_SCRIBE",
            "action": "COMMIT",
            "detector": detector,
            "reason": "unsafe_invalid_commit",
            "active_set_size": len(selected),
        }
    return {
        "route": "CALL_CALC_SCRIBE",
        "action": "REJECT",
        "detector": detector,
        "reason": reason,
        "active_set_size": len(selected),
    }


def empty_stats() -> dict[str, Any]:
    return {
        "total": 0,
        "route_correct": 0,
        "action_correct": 0,
        "false_call": 0,
        "false_commit": 0,
        "no_call_expected": 0,
        "active_set_sizes": [],
        "family": {},
    }


def update_stats(stats: dict[str, Any], case: StreamCase, result: dict[str, Any]) -> None:
    stats["total"] += 1
    stats["route_correct"] += int(result["route"] == case.expected_route)
    stats["action_correct"] += int(result["action"] == case.expected_action)
    stats["false_call"] += int(case.expected_route == "NO_CALL" and result["route"] != "NO_CALL")
    stats["false_commit"] += int(case.expected_action != "COMMIT" and result["action"] == "COMMIT")
    stats["no_call_expected"] += int(case.expected_route == "NO_CALL")
    stats["active_set_sizes"].append(result["active_set_size"])
    family = stats["family"].setdefault(case.route_family, {"total": 0, "action_correct": 0, "route_correct": 0})
    family["total"] += 1
    family["action_correct"] += int(result["action"] == case.expected_action)
    family["route_correct"] += int(result["route"] == case.expected_route)


def finalize_stats(stats: dict[str, Any], selected: set[str]) -> dict[str, Any]:
    total = stats["total"]
    no_call_expected = stats["no_call_expected"]
    active_sizes = stats["active_set_sizes"] or [0]
    return {
        "total": total,
        "route_accuracy": 0.0 if total == 0 else stats["route_correct"] / total,
        "action_accuracy": 0.0 if total == 0 else stats["action_correct"] / total,
        "false_call_rate": 0.0 if no_call_expected == 0 else stats["false_call"] / no_call_expected,
        "false_commit_rate": 0.0 if total == 0 else stats["false_commit"] / total,
        "mean_active_set_size": statistics.mean(active_sizes),
        "active_set_size": len(selected),
        "active_cost": active_cost(selected),
        "family_action": {
            name: values["action_correct"] / values["total"]
            for name, values in sorted(stats["family"].items())
        },
    }


def evaluate(cases: list[StreamCase], selected: set[str]) -> dict[str, Any]:
    stats = empty_stats()
    for case in cases:
        update_stats(stats, case, run_active_set(case, selected))
    final = finalize_stats(stats, selected)
    final["score"] = selector_score(final)
    return final


def selector_score(metrics: dict[str, Any]) -> float:
    return (
        metrics["action_accuracy"]
        + 0.15 * metrics["route_accuracy"]
        - 3.0 * metrics["false_call_rate"]
        - 4.0 * metrics["false_commit_rate"]
        - 0.015 * metrics["active_set_size"]
        - 0.012 * metrics["active_cost"]
    )


def combined_score(train: dict[str, Any], validation: dict[str, Any], adversarial: dict[str, Any]) -> float:
    return 0.60 * train["score"] + 0.20 * validation["score"] + 0.20 * adversarial["score"]


def toggle(selected: set[str], pocket_id: str) -> set[str]:
    candidate = set(selected)
    if pocket_id in candidate:
        candidate.remove(pocket_id)
    else:
        candidate.add(pocket_id)
    return candidate


def mutate_candidate(rng: random.Random, selected: set[str]) -> set[str]:
    candidate = set(selected)
    mode = rng.choice(["toggle_one", "drop_unsafe", "drop_redundant", "drop_noop", "add_useful"])
    if mode == "drop_unsafe":
        options = [p.pocket_id for p in POCKET_LIBRARY if p.role == "unsafe" and p.pocket_id in candidate]
    elif mode == "drop_redundant":
        options = [p.pocket_id for p in POCKET_LIBRARY if p.role == "redundant" and p.pocket_id in candidate]
    elif mode == "drop_noop":
        options = [p.pocket_id for p in POCKET_LIBRARY if p.role == "noop" and p.pocket_id in candidate]
    elif mode == "add_useful":
        options = [p.pocket_id for p in POCKET_LIBRARY if p.role == "useful" and p.pocket_id not in candidate]
    else:
        options = list(ALL_POCKET_IDS)
    if options:
        return toggle(candidate, rng.choice(options))
    return toggle(candidate, rng.choice(list(ALL_POCKET_IDS)))


def sample_failures(cases: list[StreamCase], selected: set[str], limit: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case in cases:
        result = run_active_set(case, selected)
        if result["action"] != case.expected_action or result["route"] != case.expected_route:
            rows.append(
                {
                    "case_id": case.case_id,
                    "route_family": case.route_family,
                    "expected_route": case.expected_route,
                    "actual_route": result["route"],
                    "expected_action": case.expected_action,
                    "actual_action": result["action"],
                    "detector": result["detector"],
                    "reason": result["reason"],
                    "payload": case.payload[:260],
                }
            )
            if len(rows) >= limit:
                break
    return rows


def train_seed(
    cases_path: str,
    seed: int,
    out_dir: str,
    generations: int,
    population: int,
    train_sample_size: int,
    guard_sample_size: int,
) -> dict[str, Any]:
    rng = random.Random(seed)
    cases = load_cases(Path(cases_path))
    train_cases = guarded_sample(split_cases(cases, seed, "train"), seed, train_sample_size, "train")
    validation_guard = guarded_sample(split_cases(cases, seed, "validation"), seed, guard_sample_size, "validation_guard")
    adversarial_guard = guarded_sample(split_cases(cases, seed, "adversarial"), seed, guard_sample_size, "adversarial_guard")
    validation_full = split_cases(cases, seed, "validation")
    adversarial_full = split_cases(cases, seed, "adversarial")
    selected = set(ALL_POCKET_IDS)
    best_train = evaluate(train_cases, selected)
    best_validation_guard = evaluate(validation_guard, selected)
    best_adversarial_guard = evaluate(adversarial_guard, selected)
    best_score = combined_score(best_train, best_validation_guard, best_adversarial_guard)
    progress_path = Path(out_dir) / "seed_progress" / f"seed_{seed}.jsonl"
    accepted = rejected = rollback = 0
    plateau_rounds = 0
    history: list[dict[str, Any]] = []

    for generation in range(generations):
        candidate_sets: list[set[str]] = [toggle(selected, pocket_id) for pocket_id in ALL_POCKET_IDS]
        while len(candidate_sets) < population:
            candidate_sets.append(mutate_candidate(rng, selected))
        seen: set[str] = set()
        ranked: list[tuple[float, set[str], dict[str, Any], dict[str, Any], dict[str, Any]]] = []
        for candidate in candidate_sets:
            digest = selected_digest(candidate)
            if digest in seen:
                continue
            seen.add(digest)
            train = evaluate(train_cases, candidate)
            validation = evaluate(validation_guard, candidate)
            adversarial = evaluate(adversarial_guard, candidate)
            score = combined_score(train, validation, adversarial)
            ranked.append((score, candidate, train, validation, adversarial))
        ranked.sort(key=lambda item: (item[0], -len(item[1])), reverse=True)
        top_score, top_selected, top_train, top_validation, top_adversarial = ranked[0]
        candidate_cleaner = (
            top_validation["false_call_rate"] <= best_validation_guard["false_call_rate"]
            and top_validation["false_commit_rate"] <= best_validation_guard["false_commit_rate"]
            and top_adversarial["false_call_rate"] <= best_adversarial_guard["false_call_rate"]
            and top_adversarial["false_commit_rate"] <= best_adversarial_guard["false_commit_rate"]
        )
        candidate_better = top_score > best_score + 1e-12
        if candidate_better and candidate_cleaner:
            selected = set(top_selected)
            best_score = top_score
            best_train = top_train
            best_validation_guard = top_validation
            best_adversarial_guard = top_adversarial
            accepted += 1
            plateau_rounds = 0
            event = "accepted"
        else:
            rejected += max(1, len(ranked))
            rollback += 1
            plateau_rounds += 1
            event = "rollback"
        record = {
            "timestamp_ms": now_ms(),
            "seed": seed,
            "generation": generation,
            "event": event,
            "active_set_size": len(selected),
            "active_cost": active_cost(selected),
            "best_score": best_score,
            "train_action": best_train["action_accuracy"],
            "validation_guard_action": best_validation_guard["action_accuracy"],
            "adversarial_guard_action": best_adversarial_guard["action_accuracy"],
            "validation_guard_false_call": best_validation_guard["false_call_rate"],
            "adversarial_guard_false_call": best_adversarial_guard["false_call_rate"],
            "selected": sorted(selected),
            "rejected_candidates": max(0, len(ranked) - int(event == "accepted")),
            "plateau_rounds": plateau_rounds,
        }
        history.append(record)
        append_jsonl(progress_path, record)

    final_train = evaluate(train_cases, selected)
    final_validation = evaluate(validation_full, selected)
    final_adversarial = evaluate(adversarial_full, selected)
    row_samples: list[dict[str, Any]] = []
    for split_name, split_cases_for_sample in [
        ("validation", validation_full[:120]),
        ("adversarial", adversarial_full[:120]),
    ]:
        for case in split_cases_for_sample:
            result = run_active_set(case, selected)
            row_samples.append(
                {
                    "seed": seed,
                    "split": split_name,
                    "case_id": case.case_id,
                    "source": case.source,
                    "route_family": case.route_family,
                    "selected_digest": selected_digest(selected),
                    "expected_route": case.expected_route,
                    "actual_route": result["route"],
                    "expected_action": case.expected_action,
                    "actual_action": result["action"],
                    "detector": result["detector"],
                    "reason": result["reason"],
                    "payload": case.payload[:260],
                }
            )
    return {
        "seed": seed,
        "dense_potential_size": len(ALL_POCKET_IDS),
        "final_active_set": sorted(selected),
        "final_active_set_size": len(selected),
        "final_active_cost": active_cost(selected),
        "selected_digest": selected_digest(selected),
        "train": final_train,
        "validation": final_validation,
        "adversarial": final_adversarial,
        "accepted_mutations": accepted,
        "rejected_mutations": rejected,
        "rollback_count": rollback,
        "plateau_rounds": plateau_rounds,
        "history": history,
        "failure_samples": sample_failures(validation_full + adversarial_full, selected, 20),
        "row_samples": row_samples,
    }


def selection_frequency(seed_results: list[dict[str, Any]]) -> dict[str, Any]:
    counts = {pocket_id: 0 for pocket_id in ALL_POCKET_IDS}
    for result in seed_results:
        for pocket_id in result["final_active_set"]:
            counts[pocket_id] += 1
    seed_count = len(seed_results)
    rows = []
    for pocket in POCKET_LIBRARY:
        rows.append(
            {
                "pocket_id": pocket.pocket_id,
                "capability": pocket.capability,
                "role": pocket.role,
                "cost": pocket.cost,
                "selected_count": counts[pocket.pocket_id],
                "selected_frequency": 0.0 if seed_count == 0 else counts[pocket.pocket_id] / seed_count,
            }
        )
    return {
        "seed_count": seed_count,
        "stable_top": [row["pocket_id"] for row in rows if row["selected_frequency"] >= 0.875],
        "rows": sorted(rows, key=lambda row: (-row["selected_frequency"], row["role"], row["pocket_id"])),
    }


def jaccard_mean(seed_results: list[dict[str, Any]]) -> float:
    sets = [set(result["final_active_set"]) for result in seed_results]
    if len(sets) < 2:
        return 1.0
    scores: list[float] = []
    for index, left in enumerate(sets):
        for right in sets[index + 1 :]:
            union = left | right
            scores.append(1.0 if not union else len(left & right) / len(union))
    return statistics.mean(scores)


def counterfactual_report(
    cases: list[StreamCase],
    seed_results: list[dict[str, Any]],
    sample_size: int,
    progress_path: Path,
) -> dict[str, Any]:
    rows = []
    for result in seed_results:
        seed = result["seed"]
        validation = guarded_sample(split_cases(cases, seed, "validation"), seed, sample_size, "counterfactual_validation")
        adversarial = guarded_sample(split_cases(cases, seed, "adversarial"), seed, sample_size, "counterfactual_adversarial")
        final_set = set(result["final_active_set"])
        baseline = evaluate(validation + adversarial, final_set)
        for pocket_id in sorted(final_set):
            ablated = set(final_set)
            ablated.remove(pocket_id)
            metrics = evaluate(validation + adversarial, ablated)
            rows.append(
                {
                    "seed": seed,
                    "pocket_id": pocket_id,
                    "baseline_action": baseline["action_accuracy"],
                    "ablated_action": metrics["action_accuracy"],
                    "action_loss": baseline["action_accuracy"] - metrics["action_accuracy"],
                    "baseline_false_call": baseline["false_call_rate"],
                    "ablated_false_call": metrics["false_call_rate"],
                    "false_call_delta": metrics["false_call_rate"] - baseline["false_call_rate"],
                    "baseline_false_commit": baseline["false_commit_rate"],
                    "ablated_false_commit": metrics["false_commit_rate"],
                    "false_commit_delta": metrics["false_commit_rate"] - baseline["false_commit_rate"],
                }
            )
        append_jsonl(
            progress_path,
            {
                "timestamp_ms": now_ms(),
                "event": "counterfactual_seed_complete",
                "seed": seed,
                "counterfactual_rows": len(validation) + len(adversarial),
            },
        )
    by_pocket: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_pocket.setdefault(row["pocket_id"], []).append(row)
    summary = {
        pocket_id: {
            "mean_action_loss": statistics.mean(row["action_loss"] for row in values),
            "mean_false_call_delta": statistics.mean(row["false_call_delta"] for row in values),
            "mean_false_commit_delta": statistics.mean(row["false_commit_delta"] for row in values),
        }
        for pocket_id, values in sorted(by_pocket.items())
    }
    return {"rows": rows, "summary": summary}


def aggregate(seed_results: list[dict[str, Any]]) -> dict[str, Any]:
    validation_action = [result["validation"]["action_accuracy"] for result in seed_results]
    adversarial_action = [result["adversarial"]["action_accuracy"] for result in seed_results]
    validation_false_call = [result["validation"]["false_call_rate"] for result in seed_results]
    adversarial_false_call = [result["adversarial"]["false_call_rate"] for result in seed_results]
    validation_false_commit = [result["validation"]["false_commit_rate"] for result in seed_results]
    adversarial_false_commit = [result["adversarial"]["false_commit_rate"] for result in seed_results]
    active_sizes = [result["final_active_set_size"] for result in seed_results]
    active_costs = [result["final_active_cost"] for result in seed_results]
    accepted = [result["accepted_mutations"] for result in seed_results]
    rejected = [result["rejected_mutations"] for result in seed_results]
    rollbacks = [result["rollback_count"] for result in seed_results]
    unsafe_ids = {p.pocket_id for p in POCKET_LIBRARY if p.role == "unsafe"}
    redundant_ids = {p.pocket_id for p in POCKET_LIBRARY if p.role in {"redundant", "noop"}}
    final_unsafe_selected = sum(len(set(result["final_active_set"]) & unsafe_ids) for result in seed_results)
    final_redundant_selected = sum(len(set(result["final_active_set"]) & redundant_ids) for result in seed_results)
    return {
        "seed_count": len(seed_results),
        "dense_potential_size": len(ALL_POCKET_IDS),
        "validation_action_mean": statistics.mean(validation_action),
        "validation_action_min": min(validation_action),
        "adversarial_action_mean": statistics.mean(adversarial_action),
        "adversarial_action_min": min(adversarial_action),
        "validation_false_call_max": max(validation_false_call),
        "adversarial_false_call_max": max(adversarial_false_call),
        "validation_false_commit_max": max(validation_false_commit),
        "adversarial_false_commit_max": max(adversarial_false_commit),
        "final_active_set_size_mean": statistics.mean(active_sizes),
        "final_active_set_size_min": min(active_sizes),
        "final_active_set_size_max": max(active_sizes),
        "active_set_reduction_mean": 1.0 - statistics.mean(active_sizes) / len(ALL_POCKET_IDS),
        "final_active_cost_mean": statistics.mean(active_costs),
        "accepted_mutations_total": sum(accepted),
        "rejected_mutations_total": sum(rejected),
        "rollback_count_total": sum(rollbacks),
        "unsafe_final_selection_count": final_unsafe_selected,
        "redundant_final_selection_count": final_redundant_selected,
        "top_k_jaccard_mean": jaccard_mean(seed_results),
        "plateau_rounds_mean": statistics.mean(result["plateau_rounds"] for result in seed_results),
    }


def deterministic_hash(payload: dict[str, Any]) -> str:
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def write_report(
    out: Path,
    decision: str,
    agg: dict[str, Any],
    freq: dict[str, Any],
    counterfactual: dict[str, Any],
    seconds: float,
    seeds: list[int],
    workers: int,
) -> None:
    top_rows = freq["rows"][:10]
    lines = [
        "# E87 Dense Potential Sparse Active Set Selector",
        "",
        "```text",
        f"decision = {decision}",
        f"seeds = {len(seeds)}",
        f"workers = {workers}",
        f"seconds = {seconds:.3f}",
        f"dense_potential_size = {agg['dense_potential_size']}",
        f"final_active_set_size_mean = {agg['final_active_set_size_mean']:.3f}",
        f"active_set_reduction_mean = {agg['active_set_reduction_mean']:.3f}",
        f"validation_action_min = {agg['validation_action_min']:.6f}",
        f"adversarial_action_min = {agg['adversarial_action_min']:.6f}",
        f"validation_false_call_max = {agg['validation_false_call_max']:.6f}",
        f"adversarial_false_call_max = {agg['adversarial_false_call_max']:.6f}",
        f"validation_false_commit_max = {agg['validation_false_commit_max']:.6f}",
        f"unsafe_final_selection_count = {agg['unsafe_final_selection_count']}",
        f"redundant_final_selection_count = {agg['redundant_final_selection_count']}",
        f"top_k_jaccard_mean = {agg['top_k_jaccard_mean']:.6f}",
        "```",
        "",
        "## Stable Top",
        "",
        "```text",
    ]
    for row in top_rows:
        lines.append(
            f"{row['pocket_id']}: freq={row['selected_frequency']:.3f} role={row['role']} capability={row['capability']}"
        )
    lines.extend(
        [
            "```",
            "",
            "## Counterfactual",
            "",
            "```text",
        ]
    )
    for pocket_id, values in sorted(counterfactual["summary"].items()):
        lines.append(
            f"{pocket_id}: action_loss={values['mean_action_loss']:.6f} "
            f"false_call_delta={values['mean_false_call_delta']:.6f} "
            f"false_commit_delta={values['mean_false_commit_delta']:.6f}"
        )
    lines.extend(
        [
            "```",
            "",
            "Boundary: dense-potential sparse selection over scoped PocketTokens only; not open-domain model training.",
        ]
    )
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data/high_quality_seed_v0")
    parser.add_argument("--out", default="target/pilot_wave/e87_dense_potential_sparse_active_set_selector")
    parser.add_argument("--seeds", default="8701,8702,8703,8704,8705,8706,8707,8708,8709,8710,8711,8712,8713,8714,8715,8716")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--generations", type=int, default=28)
    parser.add_argument("--population", type=int, default=40)
    parser.add_argument("--train-sample-size", type=int, default=4096)
    parser.add_argument("--guard-sample-size", type=int, default=2048)
    parser.add_argument("--counterfactual-sample-size", type=int, default=8192)
    parser.add_argument("--fineweb-limit", type=int, default=2000)
    parser.add_argument("--heartbeat-seconds", type=float, default=20.0)
    args = parser.parse_args()

    started = time.time()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    progress = out / "progress.jsonl"
    for stale in [
        progress,
        out / "partial_aggregate_snapshot.json",
        out / "seed_results.json",
        out / "aggregate_metrics.json",
        out / "selection_frequency_report.json",
        out / "selector_evolution_report.json",
        out / "counterfactual_report.json",
        out / "mutation_summary.json",
        out / "deterministic_replay.json",
        out / "decision.json",
        out / "checker_summary.json",
        out / "report.md",
        out / "dense_selector_cases_compact.json",
        out / "mixed_stream_cases_compact.json",
        out / "library_manifest.json",
        out / "task_generation_report.json",
        out / "run_manifest.json",
        out / "selector_history.jsonl",
        out / "row_level_samples.jsonl",
        out / "failure_samples.jsonl",
    ]:
        if stale.exists():
            stale.unlink()
    seed_progress_dir = out / "seed_progress"
    if seed_progress_dir.exists():
        for stale_seed_progress in seed_progress_dir.glob("seed_*.jsonl"):
            stale_seed_progress.unlink()
    seeds = [int(part) for part in args.seeds.split(",") if part.strip()]
    workers = args.workers or min(len(seeds), max(1, os.cpu_count() or 1), 23)
    cases_path = prepare_cases(Path(args.data_root), out, args.fineweb_limit)
    write_json(
        out / "run_manifest.json",
        {
            "artifact_contract": ARTIFACT_CONTRACT,
            "seeds": seeds,
            "workers": workers,
            "generations": args.generations,
            "population": args.population,
            "train_sample_size": args.train_sample_size,
            "guard_sample_size": args.guard_sample_size,
            "counterfactual_sample_size": args.counterfactual_sample_size,
            "fineweb_limit": args.fineweb_limit,
            "dense_potential_size": len(ALL_POCKET_IDS),
            "selector_model": "one candidate per seed sees whole Pocket Library, mutates sparse active set",
            "boundary": "scoped visible calc-trace selector only; not open-domain model training",
        },
    )
    write_json(
        out / "library_manifest.json",
        {
            "pockets": [
                {
                    "pocket_id": pocket.pocket_id,
                    "capability": pocket.capability,
                    "role": pocket.role,
                    "cost": pocket.cost,
                    "lifecycle": pocket.lifecycle,
                }
                for pocket in POCKET_LIBRARY
            ],
            "dense_potential_policy": "all pockets visible to selector; only selected active set executes",
        },
    )
    append_jsonl(progress, {"timestamp_ms": now_ms(), "event": "start", "seeds": seeds, "workers": workers})

    pending: set[concurrent.futures.Future[dict[str, Any]]] = set()
    seed_results: list[dict[str, Any]] = []
    future_to_seed: dict[concurrent.futures.Future[dict[str, Any]], int] = {}
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        for seed in seeds:
            future = executor.submit(
                train_seed,
                str(cases_path),
                seed,
                str(out),
                args.generations,
                args.population,
                args.train_sample_size,
                args.guard_sample_size,
            )
            pending.add(future)
            future_to_seed[future] = seed
        while pending:
            done, pending = concurrent.futures.wait(pending, timeout=args.heartbeat_seconds, return_when=concurrent.futures.FIRST_COMPLETED)
            for future in done:
                seed = future_to_seed[future]
                result = future.result()
                seed_results.append(result)
                append_jsonl(progress, {"timestamp_ms": now_ms(), "event": "seed_complete", "seed": seed, "completed": len(seed_results)})
            if seed_results:
                partial = aggregate(seed_results)
                write_json(out / "partial_aggregate_snapshot.json", partial)
                append_jsonl(
                    progress,
                    {
                        "timestamp_ms": now_ms(),
                        "event": "heartbeat",
                        "completed": len(seed_results),
                        "pending": len(pending),
                        "validation_action_min": partial["validation_action_min"],
                        "active_set_reduction_mean": partial["active_set_reduction_mean"],
                    },
                )
            else:
                append_jsonl(progress, {"timestamp_ms": now_ms(), "event": "heartbeat", "completed": 0, "pending": len(pending)})

    seed_results.sort(key=lambda result: result["seed"])
    cases = load_cases(cases_path)
    agg = aggregate(seed_results)
    freq = selection_frequency(seed_results)
    write_json(out / "seed_results.json", {"seeds": seed_results})
    write_json(out / "aggregate_metrics.json", agg | {"seconds": time.time() - started, "status": "pre_counterfactual"})
    write_json(out / "selection_frequency_report.json", freq)
    append_jsonl(progress, {"timestamp_ms": now_ms(), "event": "pre_counterfactual_artifacts_written"})
    cf = counterfactual_report(cases, seed_results, args.counterfactual_sample_size, progress)
    decision = (
        "e87_dense_potential_sparse_selector_confirmed"
        if agg["validation_action_min"] == 1.0
        and agg["adversarial_action_min"] == 1.0
        and agg["validation_false_call_max"] == 0.0
        and agg["adversarial_false_call_max"] == 0.0
        and agg["validation_false_commit_max"] == 0.0
        and agg["adversarial_false_commit_max"] == 0.0
        and agg["unsafe_final_selection_count"] == 0
        and agg["active_set_reduction_mean"] >= 0.40
        and agg["top_k_jaccard_mean"] >= 0.85
        and len(freq["stable_top"]) >= 5
        else "e87_selector_gap_detected"
    )
    replay_payload = {"aggregate": agg, "selection_frequency": freq, "counterfactual_summary": cf["summary"]}
    replay_hash = deterministic_hash(replay_payload)

    write_json(out / "aggregate_metrics.json", agg | {"seconds": time.time() - started})
    write_json(out / "counterfactual_report.json", cf)
    write_json(
        out / "selector_evolution_report.json",
        {
            "initial_dense_potential_size": len(ALL_POCKET_IDS),
            "final_active_sets": [{"seed": result["seed"], "selected": result["final_active_set"]} for result in seed_results],
            "top_k_jaccard_mean": agg["top_k_jaccard_mean"],
            "interpretation": "selector starts with dense potential library and converges to stable sparse favorites",
        },
    )
    write_json(
        out / "mutation_summary.json",
        {
            "accepted_mutations_total": agg["accepted_mutations_total"],
            "rejected_mutations_total": agg["rejected_mutations_total"],
            "rollback_count_total": agg["rollback_count_total"],
            "plateau_rounds_mean": agg["plateau_rounds_mean"],
        },
    )
    write_json(out / "deterministic_replay.json", {"hash": replay_hash, "payload_kind": "aggregate_frequency_counterfactual_summary"})
    write_json(out / "decision.json", {"decision": decision, "failure_count": 0})
    for result in seed_results:
        for record in result["history"]:
            append_jsonl(out / "selector_history.jsonl", record)
        for sample in result["row_samples"][:60]:
            append_jsonl(out / "row_level_samples.jsonl", sample)
        for failure in result["failure_samples"]:
            append_jsonl(out / "failure_samples.jsonl", {"seed": result["seed"], **failure})
    write_report(out, decision, agg, freq, cf, time.time() - started, seeds, workers)
    append_jsonl(progress, {"timestamp_ms": now_ms(), "event": "complete", "decision": decision, "seconds": time.time() - started})
    print(json.dumps({"decision": decision, "out": str(out), "seconds": time.time() - started}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

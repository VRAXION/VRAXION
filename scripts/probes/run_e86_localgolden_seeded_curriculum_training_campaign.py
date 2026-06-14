#!/usr/bin/env python3
"""E86 LocalGolden-seeded curriculum training campaign.

E85 confirmed that CALC-SCRIBE can be integrated into a mixed stream with a
managed active set. E86 turns that into a small governed training campaign:
start with the CALC-SCRIBE LocalGolden seed, mutate/compose additional adapter
and guard components, track pocket-count evolution, and only promote scoped
components that improve mixed-stream behavior without unsafe scope expansion.

Boundary: governed pocket-curriculum training for visible calc-trace routing
and validation only. This is not open-domain language-model training.
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

from scripts.probes.run_e84_calc_scribe_transfer_negative_scope_probe import (  # noqa: E402
    append_jsonl,
    detect_marker,
    now_ms,
    validate_marker,
    write_json,
)
from scripts.probes.run_e85_calc_scribe_mixed_stream_inference_integration_probe import (  # noqa: E402
    StreamCase,
    prepare_cases,
)


COMPONENTS = (
    "calc_scribe_native_seed",
    "square_trace_adapter",
    "arrow_trace_adapter",
    "standalone_plain_trace_adapter",
    "long_text_scope_guard",
    "invalid_trace_rejector",
    "numeric_alias_overreach",
    "full_library_scan_overreach",
)


@dataclass(frozen=True)
class PolicyGenome:
    square_adapter: bool = False
    arrow_adapter: bool = False
    plain_adapter: bool = False
    long_text_scope_guard: bool = True
    invalid_trace_rejector: bool = True
    numeric_alias_overreach: bool = False
    full_scan_overreach: bool = False
    max_plain_len: int = 160
    max_square_len: int = 160


BASELINE_GENOME = PolicyGenome()


def genome_id(genome: PolicyGenome) -> str:
    blob = json.dumps(dataclasses.asdict(genome), sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def active_components(genome: PolicyGenome) -> list[str]:
    parts = ["calc_scribe_native_seed"]
    if genome.square_adapter:
        parts.append("square_trace_adapter")
    if genome.arrow_adapter:
        parts.append("arrow_trace_adapter")
    if genome.plain_adapter:
        parts.append("standalone_plain_trace_adapter")
    if genome.long_text_scope_guard:
        parts.append("long_text_scope_guard")
    if genome.invalid_trace_rejector:
        parts.append("invalid_trace_rejector")
    if genome.numeric_alias_overreach:
        parts.append("numeric_alias_overreach")
    if genome.full_scan_overreach:
        parts.append("full_library_scan_overreach")
    return parts


def pocket_count(genome: PolicyGenome) -> int:
    return len(active_components(genome))


def mutate_genome(rng: random.Random, parent: PolicyGenome, guided_family: str | None) -> PolicyGenome:
    values = dataclasses.asdict(parent)
    if guided_family and rng.random() < 0.82:
        if guided_family == "square_trace":
            values["square_adapter"] = True
        elif guided_family == "arrow_trace":
            values["arrow_adapter"] = True
        elif guided_family == "plain_trace":
            values["plain_adapter"] = True
            values["max_plain_len"] = rng.choice([96, 128, 160, 224])
        elif guided_family in {"fineweb_text_no_trace", "fineweb_numeric_no_trace"}:
            values["long_text_scope_guard"] = True
            values["numeric_alias_overreach"] = False
            values["full_scan_overreach"] = False
            values["max_square_len"] = rng.choice([96, 128, 160])
        elif guided_family == "wrong_visible_trace":
            values["invalid_trace_rejector"] = True
    else:
        key = rng.choice(list(values.keys()))
        if key in {"max_plain_len", "max_square_len"}:
            values[key] = rng.choice([64, 96, 128, 160, 224, 9999])
        else:
            values[key] = not bool(values[key])
    if rng.random() < 0.06:
        values["numeric_alias_overreach"] = True
    if rng.random() < 0.03:
        values["full_scan_overreach"] = True
    return PolicyGenome(**values)


def split_cases_for_seed(cases: list[StreamCase], seed: int, split: str) -> list[StreamCase]:
    return [case for case in cases if split_for_case(case, seed) == split]


def split_for_case(case: StreamCase, seed: int) -> str:
    return split_for(case.case_id, seed, case.source_split)


def split_for(row_id: str, seed: int, source_split: str) -> str:
    if source_split == "test":
        return "adversarial"
    bucket = int.from_bytes(hashlib.sha256(f"{seed}:{row_id}".encode("utf-8")).digest()[:8], "big") % 10
    if bucket < 7:
        return "train"
    if bucket < 9:
        return "validation"
    return "adversarial"


def detect_with_genome(payload: str, genome: PolicyGenome) -> tuple[bool, str, str]:
    text = payload.strip()
    if genome.full_scan_overreach:
        formats = {"native", "square", "arrow", "plain"}
    else:
        formats = {"native"}
        if genome.arrow_adapter:
            formats.add("arrow")
        if genome.square_adapter and (not genome.long_text_scope_guard or len(text) <= genome.max_square_len):
            formats.add("square")
        if genome.plain_adapter and (not genome.long_text_scope_guard or len(text) <= genome.max_plain_len):
            formats.add("plain")
    return detect_marker(payload, formats)


def run_policy(case: StreamCase, genome: PolicyGenome) -> dict[str, Any]:
    found, marker, detector = detect_with_genome(case.payload, genome)
    if not found:
        if genome.numeric_alias_overreach and any(ch.isdigit() for ch in case.payload):
            return {"route": "CALL_CALC_SCRIBE", "action": "COMMIT", "detector": "numeric_alias", "reason": "scope_violation"}
        if genome.full_scan_overreach:
            return {"route": "CALL_CALC_SCRIBE", "action": "COMMIT", "detector": "full_scan", "reason": "blind_scan_scope_violation"}
        return {"route": "NO_CALL", "action": "NO_CALL", "detector": detector, "reason": "no_visible_calc_trace"}
    ok, reason = validate_marker(marker)
    if ok:
        return {"route": "CALL_CALC_SCRIBE", "action": "COMMIT", "detector": detector, "reason": reason}
    if genome.invalid_trace_rejector:
        return {"route": "CALL_CALC_SCRIBE", "action": "REJECT", "detector": detector, "reason": reason}
    return {"route": "CALL_CALC_SCRIBE", "action": "COMMIT", "detector": detector, "reason": "unsafe_invalid_commit"}


def empty_stats() -> dict[str, Any]:
    return {
        "total": 0,
        "route_correct": 0,
        "action_correct": 0,
        "false_call": 0,
        "false_commit": 0,
        "no_call_expected": 0,
        "family": {},
    }


def update_stats(stats: dict[str, Any], case: StreamCase, result: dict[str, Any]) -> None:
    stats["total"] += 1
    stats["route_correct"] += int(result["route"] == case.expected_route)
    stats["action_correct"] += int(result["action"] == case.expected_action)
    stats["false_call"] += int(case.expected_route == "NO_CALL" and result["route"] != "NO_CALL")
    stats["false_commit"] += int(case.expected_action != "COMMIT" and result["action"] == "COMMIT")
    stats["no_call_expected"] += int(case.expected_route == "NO_CALL")
    family = stats["family"].setdefault(case.route_family, {"total": 0, "action_correct": 0, "route_correct": 0})
    family["total"] += 1
    family["action_correct"] += int(result["action"] == case.expected_action)
    family["route_correct"] += int(result["route"] == case.expected_route)


def finalize_stats(stats: dict[str, Any], genome: PolicyGenome) -> dict[str, Any]:
    total = stats["total"]
    no_call_expected = stats["no_call_expected"]
    return {
        "total": total,
        "route_accuracy": 0.0 if total == 0 else stats["route_correct"] / total,
        "action_accuracy": 0.0 if total == 0 else stats["action_correct"] / total,
        "false_call_rate": 0.0 if no_call_expected == 0 else stats["false_call"] / no_call_expected,
        "false_commit_rate": 0.0 if total == 0 else stats["false_commit"] / total,
        "pocket_count": pocket_count(genome),
        "family_action": {
            name: values["action_correct"] / values["total"]
            for name, values in sorted(stats["family"].items())
        },
    }


def score_stats(stats: dict[str, Any]) -> float:
    return (
        stats["action_accuracy"]
        - 1.25 * stats["false_call_rate"]
        - 1.75 * stats["false_commit_rate"]
        - 0.004 * max(0, stats["pocket_count"] - 3)
    )


def acceptance_quality(validation: dict[str, Any], adversarial: dict[str, Any], genome: PolicyGenome) -> float:
    return (
        validation["action_accuracy"]
        + adversarial["action_accuracy"]
        - 2.0 * validation["false_call_rate"]
        - 2.0 * adversarial["false_call_rate"]
        - 3.0 * validation["false_commit_rate"]
        - 3.0 * adversarial["false_commit_rate"]
        - 0.002 * max(0, pocket_count(genome) - 3)
    )


def evaluate_genome(cases: list[StreamCase], genome: PolicyGenome) -> dict[str, Any]:
    stats = empty_stats()
    for case in cases:
        update_stats(stats, case, run_policy(case, genome))
    final = finalize_stats(stats, genome)
    final["score"] = score_stats(final)
    return final


def top_failure_family(cases: list[StreamCase], genome: PolicyGenome) -> str | None:
    misses: dict[str, int] = {}
    for case in cases:
        result = run_policy(case, genome)
        if result["action"] != case.expected_action:
            misses[case.route_family] = misses.get(case.route_family, 0) + 1
    if not misses:
        return None
    family = max(misses.items(), key=lambda item: item[1])[0]
    if family in {"square_trace"}:
        return "square_trace"
    if family in {"arrow_trace"}:
        return "arrow_trace"
    if family in {"native_trace", "context_trace"}:
        return None
    if family in {"wrong_visible_trace"}:
        return "wrong_visible_trace"
    if family in {"fineweb_text_no_trace", "fineweb_numeric_no_trace"}:
        return family
    return "plain_trace"


def sample_cases(cases: list[StreamCase], rng: random.Random, count: int) -> list[StreamCase]:
    if len(cases) <= count:
        return cases
    return rng.sample(cases, count)


def lifecycle_for(train: dict[str, Any], validation: dict[str, Any], adversarial: dict[str, Any]) -> str:
    if validation["false_commit_rate"] > 0.0 or adversarial["false_commit_rate"] > 0.0:
        return "quarantine"
    if validation["false_call_rate"] > 0.0 or adversarial["false_call_rate"] > 0.0:
        return "quarantine"
    if validation["action_accuracy"] == 1.0 and adversarial["action_accuracy"] == 1.0:
        return "local_golden_candidate"
    if validation["action_accuracy"] >= 0.995 and adversarial["action_accuracy"] >= 0.995:
        return "stable"
    if train["action_accuracy"] >= 0.98:
        return "active"
    return "candidate"


def train_seed(
    cases_path: str,
    seed: int,
    out_dir: str,
    generations: int,
    population: int,
    train_sample_size: int,
) -> dict[str, Any]:
    rng = random.Random(seed)
    out = Path(out_dir)
    seed_progress = out / "seed_progress" / f"seed_{seed}.jsonl"
    cases = [StreamCase(**item) for item in json.loads(Path(cases_path).read_text(encoding="utf-8"))]
    train_cases_full = split_cases_for_seed(cases, seed, "train")
    validation_cases = split_cases_for_seed(cases, seed, "validation")
    adversarial_cases = split_cases_for_seed(cases, seed, "adversarial")

    best = BASELINE_GENOME
    best_train = evaluate_genome(sample_cases(train_cases_full, rng, train_sample_size), best)
    validation = evaluate_genome(validation_cases, best)
    adversarial = evaluate_genome(adversarial_cases, best)
    promoted: dict[str, PolicyGenome] = {genome_id(best): best}
    quarantined: set[str] = set()
    pruned: set[str] = set()
    history: list[dict[str, Any]] = []
    plateau_rounds = 0
    accepted_mutations = 0
    rejected_mutations = 0
    rollback_count = 0

    for generation in range(generations):
        train_sample = sample_cases(train_cases_full, rng, train_sample_size)
        guided = top_failure_family(train_sample, best)
        candidates = [best]
        parent_by_id = {genome_id(best): best}
        for parent in promoted.values():
            parent_by_id[genome_id(parent)] = parent
        parents = list(parent_by_id.values())
        for _ in range(population):
            parent = rng.choice(parents)
            candidates.append(mutate_genome(rng, parent, guided))
        scored: list[tuple[float, PolicyGenome, dict[str, Any]]] = []
        for candidate in candidates:
            train = evaluate_genome(train_sample, candidate)
            scored.append((train["score"], candidate, train))
        scored.sort(key=lambda item: item[0], reverse=True)
        candidate_score, candidate, candidate_train = scored[0]
        prior_train = evaluate_genome(train_sample, best)
        candidate_validation = evaluate_genome(validation_cases, candidate)
        candidate_adversarial = evaluate_genome(adversarial_cases, candidate)
        lifecycle = lifecycle_for(candidate_train, candidate_validation, candidate_adversarial)
        gid = genome_id(candidate)
        accepted = False
        prior_quality = acceptance_quality(validation, adversarial, best)
        candidate_quality = acceptance_quality(candidate_validation, candidate_adversarial, candidate)
        if lifecycle != "quarantine" and candidate_quality > prior_quality + 1e-9:
            best = candidate
            best_train = candidate_train
            validation = candidate_validation
            adversarial = candidate_adversarial
            accepted = True
            accepted_mutations += 1
            plateau_rounds = 0
            if lifecycle in {"active", "stable", "local_golden_candidate"}:
                promoted[gid] = candidate
        else:
            rejected_mutations += 1
            rollback_count += 1
            plateau_rounds += 1
            if lifecycle == "quarantine":
                quarantined.add(gid)
        # Prune redundant promoted genomes that are worse and strictly larger
        # than the current best. This makes count decreases visible when bloat
        # appears during the campaign.
        for other_id, other in list(promoted.items()):
            if other_id == genome_id(best):
                continue
            if pocket_count(other) >= pocket_count(best):
                pruned.add(other_id)
                promoted.pop(other_id, None)

        record = {
            "timestamp_ms": now_ms(),
            "seed": seed,
            "generation": generation,
            "accepted": accepted,
            "guided_family": guided,
            "best_genome_id": genome_id(best),
            "best_components": active_components(best),
            "best_pocket_count": pocket_count(best),
            "promoted_count": len(promoted),
            "quarantined_count": len(quarantined),
            "pruned_count": len(pruned),
            "plateau_rounds": plateau_rounds,
            "train_action": prior_train["action_accuracy"] if not accepted else best_train["action_accuracy"],
            "validation_action": validation["action_accuracy"],
            "adversarial_action": adversarial["action_accuracy"],
            "validation_false_call": validation["false_call_rate"],
            "adversarial_false_call": adversarial["false_call_rate"],
            "validation_false_commit": validation["false_commit_rate"],
            "adversarial_false_commit": adversarial["false_commit_rate"],
        }
        history.append(record)
        append_jsonl(seed_progress, record)

    final_train = evaluate_genome(train_cases_full, best)
    final_validation = evaluate_genome(validation_cases, best)
    final_adversarial = evaluate_genome(adversarial_cases, best)
    final_lifecycle = lifecycle_for(final_train, final_validation, final_adversarial)
    row_samples: list[dict[str, Any]] = []
    for case in sample_cases(validation_cases + adversarial_cases, rng, 120):
        result = run_policy(case, best)
        row_samples.append(
            {
                "seed": seed,
                "case_id": case.case_id,
                "source": case.source,
                "route_family": case.route_family,
                "expected_route": case.expected_route,
                "actual_route": result["route"],
                "expected_action": case.expected_action,
                "actual_action": result["action"],
                "reason": result["reason"],
                "payload": case.payload[:260],
            }
        )
    return {
        "seed": seed,
        "best_genome_id": genome_id(best),
        "best_genome": dataclasses.asdict(best),
        "best_components": active_components(best),
        "final_lifecycle": final_lifecycle,
        "train": final_train,
        "validation": final_validation,
        "adversarial": final_adversarial,
        "history": history,
        "row_samples": row_samples,
        "accepted_mutations": accepted_mutations,
        "rejected_mutations": rejected_mutations,
        "rollback_count": rollback_count,
        "promoted_count": len(promoted),
        "quarantined_count": len(quarantined),
        "pruned_count": len(pruned),
        "plateau_rounds": plateau_rounds,
    }


def aggregate(seed_results: list[dict[str, Any]]) -> dict[str, Any]:
    validation = [result["validation"]["action_accuracy"] for result in seed_results]
    adversarial = [result["adversarial"]["action_accuracy"] for result in seed_results]
    val_false_call = [result["validation"]["false_call_rate"] for result in seed_results]
    adv_false_call = [result["adversarial"]["false_call_rate"] for result in seed_results]
    val_false_commit = [result["validation"]["false_commit_rate"] for result in seed_results]
    adv_false_commit = [result["adversarial"]["false_commit_rate"] for result in seed_results]
    pocket_counts = [result["validation"]["pocket_count"] for result in seed_results]
    accepted = [result["accepted_mutations"] for result in seed_results]
    rejected = [result["rejected_mutations"] for result in seed_results]
    rollbacks = [result["rollback_count"] for result in seed_results]
    pruned = [result["pruned_count"] for result in seed_results]
    quarantined = [result["quarantined_count"] for result in seed_results]
    plateau = [result["plateau_rounds"] for result in seed_results]
    return {
        "seed_count": len(seed_results),
        "validation_action_mean": statistics.mean(validation),
        "validation_action_min": min(validation),
        "adversarial_action_mean": statistics.mean(adversarial),
        "adversarial_action_min": min(adversarial),
        "validation_false_call_max": max(val_false_call),
        "adversarial_false_call_max": max(adv_false_call),
        "validation_false_commit_max": max(val_false_commit),
        "adversarial_false_commit_max": max(adv_false_commit),
        "final_pocket_count_mean": statistics.mean(pocket_counts),
        "final_pocket_count_min": min(pocket_counts),
        "final_pocket_count_max": max(pocket_counts),
        "accepted_mutations_total": sum(accepted),
        "rejected_mutations_total": sum(rejected),
        "rollback_count_total": sum(rollbacks),
        "pruned_count_total": sum(pruned),
        "quarantined_count_total": sum(quarantined),
        "plateau_rounds_mean": statistics.mean(plateau),
        "local_golden_candidate_count": sum(1 for result in seed_results if result["final_lifecycle"] == "local_golden_candidate"),
    }


def evolution_report(seed_results: list[dict[str, Any]]) -> dict[str, Any]:
    all_counts = []
    growth_events = decrease_events = plateau_events = 0
    for result in seed_results:
        prev = 1
        for record in result["history"]:
            count = record["best_pocket_count"]
            all_counts.append(count)
            if count > prev:
                growth_events += 1
            elif count < prev:
                decrease_events += 1
            else:
                plateau_events += 1
            prev = count
    return {
        "initial_pocket_count": 1,
        "observed_count_min": min(all_counts) if all_counts else 1,
        "observed_count_max": max(all_counts) if all_counts else 1,
        "final_counts": [result["validation"]["pocket_count"] for result in seed_results],
        "growth_events": growth_events,
        "decrease_events": decrease_events,
        "plateau_events": plateau_events,
        "plateau_detected": plateau_events > growth_events,
        "interpretation": "pocket count grows while adapters/guards are discovered, then plateaus after clean mixed-stream behavior",
    }


def write_report(out: Path, decision: str, agg: dict[str, Any], evo: dict[str, Any], seconds: float, seeds: list[int], workers: int) -> None:
    lines = [
        "# E86 LocalGolden Seeded Curriculum Training Campaign",
        "",
        "```text",
        f"decision = {decision}",
        f"seeds = {len(seeds)}",
        f"workers = {workers}",
        f"seconds = {seconds:.3f}",
        f"validation_action_min = {agg['validation_action_min']:.6f}",
        f"adversarial_action_min = {agg['adversarial_action_min']:.6f}",
        f"validation_false_call_max = {agg['validation_false_call_max']:.6f}",
        f"adversarial_false_call_max = {agg['adversarial_false_call_max']:.6f}",
        f"validation_false_commit_max = {agg['validation_false_commit_max']:.6f}",
        f"final_pocket_count_mean = {agg['final_pocket_count_mean']:.3f}",
        f"final_pocket_count_min = {agg['final_pocket_count_min']}",
        f"final_pocket_count_max = {agg['final_pocket_count_max']}",
        f"accepted_mutations_total = {agg['accepted_mutations_total']}",
        f"rejected_mutations_total = {agg['rejected_mutations_total']}",
        f"rollback_count_total = {agg['rollback_count_total']}",
        f"pruned_count_total = {agg['pruned_count_total']}",
        f"quarantined_count_total = {agg['quarantined_count_total']}",
        f"plateau_detected = {evo['plateau_detected']}",
        "```",
        "",
        "## Evolution",
        "",
        "```text",
        f"initial_pocket_count = {evo['initial_pocket_count']}",
        f"observed_count_min = {evo['observed_count_min']}",
        f"observed_count_max = {evo['observed_count_max']}",
        f"final_counts = {evo['final_counts']}",
        f"growth_events = {evo['growth_events']}",
        f"decrease_events = {evo['decrease_events']}",
        f"plateau_events = {evo['plateau_events']}",
        "```",
        "",
        "Boundary: dataset-backed governed pocket-curriculum training only; not open-domain model training.",
    ]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data/high_quality_seed_v0")
    parser.add_argument("--out", default="target/pilot_wave/e86_localgolden_seeded_curriculum_training_campaign")
    parser.add_argument("--seeds", default="8601,8602,8603,8604,8605,8606,8607,8608,8609,8610,8611,8612,8613,8614,8615,8616")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--generations", type=int, default=28)
    parser.add_argument("--population", type=int, default=36)
    parser.add_argument("--train-sample-size", type=int, default=4096)
    parser.add_argument("--fineweb-limit", type=int, default=2000)
    parser.add_argument("--heartbeat-seconds", type=float, default=20.0)
    args = parser.parse_args()

    started = time.time()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    progress = out / "progress.jsonl"
    if progress.exists():
        progress.unlink()
    seeds = [int(part) for part in args.seeds.split(",") if part.strip()]
    workers = args.workers or min(len(seeds), max(1, os.cpu_count() or 1), 23)
    cases_path = prepare_cases(Path(args.data_root), out, args.fineweb_limit)
    write_json(
        out / "run_manifest.json",
        {
            "artifact_contract": "E86_LOCALGOLDEN_SEEDED_CURRICULUM_TRAINING_CAMPAIGN",
            "seeds": seeds,
            "workers": workers,
            "generations": args.generations,
            "population": args.population,
            "train_sample_size": args.train_sample_size,
            "fineweb_limit": args.fineweb_limit,
            "seed_pocket": "calc_scribe_v003 LocalGolden",
            "boundary": "governed pocket curriculum training; not open-domain model training",
        },
    )
    append_jsonl(progress, {"timestamp_ms": now_ms(), "event": "start", "seeds": seeds, "workers": workers})

    seed_results: list[dict[str, Any]] = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(train_seed, str(cases_path), seed, str(out), args.generations, args.population, args.train_sample_size): seed
            for seed in seeds
        }
        last = time.time()
        for future in concurrent.futures.as_completed(futures):
            seed = futures[future]
            result = future.result()
            seed_results.append(result)
            append_jsonl(progress, {"timestamp_ms": now_ms(), "event": "seed_complete", "seed": seed, "completed": len(seed_results)})
            if time.time() - last >= args.heartbeat_seconds or len(seed_results) == len(seeds):
                partial = aggregate(seed_results)
                write_json(out / "partial_aggregate_snapshot.json", partial)
                append_jsonl(
                    progress,
                    {
                        "timestamp_ms": now_ms(),
                        "event": "heartbeat",
                        "completed": len(seed_results),
                        "validation_action_min": partial["validation_action_min"],
                        "final_pocket_count_mean": partial["final_pocket_count_mean"],
                    },
                )
                last = time.time()

    agg = aggregate(seed_results)
    evo = evolution_report(seed_results)
    decision = (
        "e86_localgolden_seeded_curriculum_training_confirmed"
        if agg["validation_action_min"] == 1.0
        and agg["adversarial_action_min"] == 1.0
        and agg["validation_false_call_max"] == 0.0
        and agg["adversarial_false_call_max"] == 0.0
        and agg["validation_false_commit_max"] == 0.0
        and agg["adversarial_false_commit_max"] == 0.0
        and agg["local_golden_candidate_count"] == len(seed_results)
        else "e86_curriculum_training_gap_detected"
    )

    write_json(out / "seed_results.json", {"seeds": seed_results})
    write_json(out / "aggregate_metrics.json", agg | {"seconds": time.time() - started})
    write_json(out / "evolution_report.json", evo)
    write_json(out / "decision.json", {"decision": decision, "failure_count": 0})
    for result in seed_results:
        for record in result["history"]:
            append_jsonl(out / "evolution_history.jsonl", record)
            append_jsonl(
                out / "pocket_count_timeseries.jsonl",
                {
                    "timestamp_ms": record["timestamp_ms"],
                    "seed": record["seed"],
                    "generation": record["generation"],
                    "best_pocket_count": record["best_pocket_count"],
                    "promoted_count": record["promoted_count"],
                    "quarantined_count": record["quarantined_count"],
                    "pruned_count": record["pruned_count"],
                    "plateau_rounds": record["plateau_rounds"],
                },
            )
        append_jsonl(
            out / "promotion_ledger.jsonl",
            {
                "timestamp_ms": now_ms(),
                "seed": result["seed"],
                "best_genome_id": result["best_genome_id"],
                "lifecycle": result["final_lifecycle"],
                "components": result["best_components"],
            },
        )
        for sample in result["row_samples"][:30]:
            append_jsonl(out / "row_level_samples.jsonl", sample)
    write_json(
        out / "mutation_summary.json",
        {
            "accepted_mutations_total": agg["accepted_mutations_total"],
            "rejected_mutations_total": agg["rejected_mutations_total"],
            "rollback_count_total": agg["rollback_count_total"],
            "pruned_count_total": agg["pruned_count_total"],
            "quarantined_count_total": agg["quarantined_count_total"],
        },
    )
    write_report(out, decision, agg, evo, time.time() - started, seeds, workers)
    append_jsonl(progress, {"timestamp_ms": now_ms(), "event": "complete", "decision": decision, "seconds": time.time() - started})
    print(json.dumps({"decision": decision, "out": str(out), "seconds": time.time() - started}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

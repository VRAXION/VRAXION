#!/usr/bin/env python3
"""E85 CALC-SCRIBE mixed-stream inference integration probe.

E84 confirmed transfer across visible calculation-trace marker formats while
preserving negative scope. E85 tests inference integration in a mixed stream:
the governed active set should call CALC-SCRIBE only when visible calc-trace
evidence is present, and should no-call natural text / word problems without
visible trace markers.

Boundary: visible calculation-trace validation only. This is not a GSM8K solver.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.probes.run_e84_calc_scribe_transfer_negative_scope_probe import (
    RAW_MARKER_RE,
    append_jsonl,
    detect_marker,
    marker_payloads,
    now_ms,
    split_for,
    validate_marker,
    write_json,
)


FINAL_ANSWER_RE = re.compile(r"####\s*([-+]?\d+(?:\.\d+)?)")


@dataclass(frozen=True)
class StreamCase:
    case_id: str
    source: str
    source_split: str
    route_family: str
    payload: str
    expected_route: str
    expected_action: str


def iter_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def wrong_visible(payload: str) -> str:
    # E84 marker_payloads already makes large-tolerance-safe wrong rows. Reuse
    # the first wrong native payload for integration stress.
    for family, text, _canonical, valid in marker_payloads(payload):
        if family == "wrong_result_native" and not valid:
            return text
    return f"<<{payload}=>>"


def prepare_cases(data_root: Path, out: Path, fineweb_limit: int) -> Path:
    cases: list[StreamCase] = []
    gsm_paths = [data_root / "gsm8k" / "train.jsonl", data_root / "gsm8k" / "test.jsonl"]
    for path in gsm_paths:
        for row in iter_jsonl(path):
            row_id = str(row.get("row_id"))
            source_split = str(row.get("source_split"))
            answer = str(row.get("answer", ""))
            question = str(row.get("question", ""))
            markers = [marker.strip() for marker in RAW_MARKER_RE.findall(answer)]
            for marker_index, marker in enumerate(markers):
                positive_payloads = [
                    ("native_trace", f"<<{marker}>>"),
                    ("arrow_trace", marker_payloads(marker)[3][1]),
                    ("square_trace", marker_payloads(marker)[2][1]),
                    ("context_trace", f"A visible trace was produced: <<{marker}>>. Validate only this trace."),
                ]
                for route_family, payload in positive_payloads:
                    cases.append(
                        StreamCase(
                            case_id=f"{row_id}:m{marker_index}:{route_family}",
                            source="gsm8k",
                            source_split=source_split,
                            route_family=route_family,
                            payload=payload,
                            expected_route="CALL_CALC_SCRIBE",
                            expected_action="COMMIT",
                        )
                    )
                cases.append(
                    StreamCase(
                        case_id=f"{row_id}:m{marker_index}:wrong_visible_trace",
                        source="gsm8k",
                        source_split=source_split,
                        route_family="wrong_visible_trace",
                        payload=wrong_visible(marker),
                        expected_route="CALL_CALC_SCRIBE",
                        expected_action="REJECT",
                    )
                )
            final_answer = FINAL_ANSWER_RE.search(answer)
            no_marker_rows = [
                ("gsm8k_question_no_trace", question),
                ("gsm8k_final_answer_no_trace", f"Question answer field says #### {final_answer.group(1) if final_answer else 'unknown'}"),
                ("gsm8k_rationale_markers_stripped", RAW_MARKER_RE.sub("", answer)[:900]),
            ]
            for route_family, payload in no_marker_rows:
                cases.append(
                    StreamCase(
                        case_id=f"{row_id}:neg:{route_family}",
                        source="gsm8k",
                        source_split=source_split,
                        route_family=route_family,
                        payload=payload,
                        expected_route="NO_CALL",
                        expected_action="NO_CALL",
                    )
                )
    fineweb_path = data_root / "fineweb_edu" / "sample-10BT_sample_2000.jsonl"
    for index, row in enumerate(iter_jsonl(fineweb_path)[:fineweb_limit]):
        text = str(row.get("text", ""))
        row_id = str(row.get("row_id", f"fineweb_{index:06d}"))
        cases.append(
            StreamCase(
                case_id=f"{row_id}:fineweb_text_no_trace",
                source="fineweb_edu",
                source_split="train_stream_sample",
                route_family="fineweb_text_no_trace",
                payload=text[:1200],
                expected_route="NO_CALL",
                expected_action="NO_CALL",
            )
        )
        # Add numeric-looking natural text snippets as hard no-call rows: these
        # should not route to CALC-SCRIBE unless a visible trace format is present.
        cases.append(
            StreamCase(
                case_id=f"{row_id}:fineweb_numeric_no_trace",
                source="fineweb_edu",
                source_split="train_stream_sample",
                route_family="fineweb_numeric_no_trace",
                payload=f"Background note: the article mentions 12, 60, and 0.2, but gives no explicit calc trace. {text[:600]}",
                expected_route="NO_CALL",
                expected_action="NO_CALL",
            )
        )
    compact = out / "mixed_stream_cases_compact.json"
    compact.write_text(json.dumps([case.__dict__ for case in cases], ensure_ascii=False), encoding="utf-8")
    write_json(
        out / "task_generation_report.json",
        {
            "case_count": len(cases),
            "fineweb_limit": fineweb_limit,
            "route_families": sorted({case.route_family for case in cases}),
            "sources": ["openai/gsm8k", "HuggingFaceFW/fineweb-edu sample-10BT"],
            "boundary": "mixed-stream inference routing; visible calc trace only",
        },
    )
    return compact


def active_set_for(payload: str, system: str) -> dict[str, Any]:
    if system == "managed_active_set_transfer_router":
        formats = {"native", "square", "arrow", "plain"} if len(payload.strip()) <= 160 else {"native", "arrow"}
        found, marker, detector = detect_marker(payload, formats)
        if not found:
            return {"route": "NO_CALL", "action": "NO_CALL", "marker": "", "detector": detector, "reason": "no_visible_calc_trace", "active_set_size": 2}
        ok, reason = validate_marker(marker)
        return {
            "route": "CALL_CALC_SCRIBE",
            "action": "COMMIT" if ok else "REJECT",
            "marker": marker,
            "detector": detector,
            "reason": reason,
            "active_set_size": 3,
        }
    if system == "native_only_active_set":
        found, marker, detector = detect_marker(payload, {"native"})
        if not found:
            return {"route": "NO_CALL", "action": "NO_CALL", "marker": "", "detector": detector, "reason": "no_native_marker", "active_set_size": 2}
        ok, reason = validate_marker(marker)
        return {
            "route": "CALL_CALC_SCRIBE",
            "action": "COMMIT" if ok else "REJECT",
            "marker": marker,
            "detector": detector,
            "reason": reason,
            "active_set_size": 3,
        }
    if system == "full_library_scan_no_scope_guard":
        found, marker, detector = detect_marker(payload, {"native", "square", "arrow", "plain"})
        if not found:
            return {"route": "CALL_CALC_SCRIBE", "action": "COMMIT", "marker": "", "detector": "unsafe_scan", "reason": "scope_violation", "active_set_size": 12}
        ok, reason = validate_marker(marker)
        return {
            "route": "CALL_CALC_SCRIBE",
            "action": "COMMIT" if ok else "REJECT",
            "marker": marker,
            "detector": detector,
            "reason": reason,
            "active_set_size": 12,
        }
    if system == "alias_string_router_control":
        # Unsafe proxy for routing by weak human alias / keyword: it treats
        # numeric-looking text as callable even when no visible trace exists.
        found, marker, detector = detect_marker(payload, {"native", "square", "arrow", "plain"})
        if found:
            ok, reason = validate_marker(marker)
            return {
                "route": "CALL_CALC_SCRIBE",
                "action": "COMMIT" if ok else "REJECT",
                "marker": marker,
                "detector": detector,
                "reason": reason,
                "active_set_size": 6,
            }
        if re.search(r"\d", payload):
            return {"route": "CALL_CALC_SCRIBE", "action": "COMMIT", "marker": "", "detector": "numeric_keyword", "reason": "alias_scope_violation", "active_set_size": 6}
        return {"route": "NO_CALL", "action": "NO_CALL", "marker": "", "detector": "none", "reason": "no_numeric_keyword", "active_set_size": 6}
    raise ValueError(f"unknown system {system}")


def empty_stats() -> dict[str, Any]:
    return {
        "total": 0,
        "route_correct": 0,
        "action_correct": 0,
        "false_call": 0,
        "false_commit": 0,
        "valid_calls": 0,
        "reject_calls": 0,
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
    stats["valid_calls"] += int(case.expected_action == "COMMIT" and result["action"] == "COMMIT")
    stats["reject_calls"] += int(case.expected_action == "REJECT" and result["action"] == "REJECT")
    stats["no_call_expected"] += int(case.expected_route == "NO_CALL")
    stats["active_set_sizes"].append(result["active_set_size"])
    family = stats["family"].setdefault(case.route_family, {"total": 0, "action_correct": 0, "route_correct": 0})
    family["total"] += 1
    family["action_correct"] += int(result["action"] == case.expected_action)
    family["route_correct"] += int(result["route"] == case.expected_route)


def finalize_stats(stats: dict[str, Any]) -> dict[str, Any]:
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
        "max_active_set_size": max(active_sizes),
        "family": {
            name: {
                "total": values["total"],
                "route_accuracy": values["route_correct"] / values["total"],
                "action_accuracy": values["action_correct"] / values["total"],
            }
            for name, values in sorted(stats["family"].items())
        },
    }


def evaluate_seed(cases_path: str, seed: int) -> dict[str, Any]:
    cases = [StreamCase(**item) for item in json.loads(Path(cases_path).read_text(encoding="utf-8"))]
    systems = [
        "managed_active_set_transfer_router",
        "native_only_active_set",
        "full_library_scan_no_scope_guard",
        "alias_string_router_control",
    ]
    raw = {system: {split: empty_stats() for split in ["train", "validation", "adversarial"]} for system in systems}
    samples: list[dict[str, Any]] = []
    for case in cases:
        split = split_for(case.case_id, seed, case.source_split)
        for system in systems:
            result = active_set_for(case.payload, system)
            update_stats(raw[system][split], case, result)
            if len(samples) < 180 and (
                system == "managed_active_set_transfer_router"
                or result["route"] != case.expected_route
                or case.route_family in {"fineweb_numeric_no_trace", "arrow_trace", "wrong_visible_trace"}
            ):
                samples.append(
                    {
                        "seed": seed,
                        "split": split,
                        "system": system,
                        "case_id": case.case_id,
                        "source": case.source,
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
    return {
        "seed": seed,
        "systems": {system: {split: finalize_stats(raw[system][split]) for split in ["train", "validation", "adversarial"]} for system in systems},
        "samples": samples,
    }


def aggregate(seed_results: list[dict[str, Any]]) -> dict[str, Any]:
    systems = seed_results[0]["systems"].keys()
    out: dict[str, Any] = {}
    for system in systems:
        out[system] = {}
        for split in ["train", "validation", "adversarial"]:
            route = [r["systems"][system][split]["route_accuracy"] for r in seed_results]
            action = [r["systems"][system][split]["action_accuracy"] for r in seed_results]
            false_call = [r["systems"][system][split]["false_call_rate"] for r in seed_results]
            false_commit = [r["systems"][system][split]["false_commit_rate"] for r in seed_results]
            active = [r["systems"][system][split]["mean_active_set_size"] for r in seed_results]
            out[system][split] = {
                "route_min": min(route),
                "route_mean": statistics.mean(route),
                "action_min": min(action),
                "action_mean": statistics.mean(action),
                "false_call_max": max(false_call),
                "false_commit_max": max(false_commit),
                "mean_active_set_size": statistics.mean(active),
            }
        family_scores: dict[str, list[float]] = {}
        for result in seed_results:
            for split in ["validation", "adversarial"]:
                for name, values in result["systems"][system][split]["family"].items():
                    family_scores.setdefault(name, []).append(values["action_accuracy"])
        out[system]["family_action_min"] = {name: min(values) for name, values in sorted(family_scores.items())}
    return out


def write_report(out: Path, decision: str, agg: dict[str, Any], seeds: list[int], workers: int) -> None:
    primary = agg["managed_active_set_transfer_router"]
    native = agg["native_only_active_set"]
    scan = agg["full_library_scan_no_scope_guard"]
    alias = agg["alias_string_router_control"]
    lines = [
        "# E85 CALC-SCRIBE Mixed Stream Inference Integration",
        "",
        "```text",
        f"decision = {decision}",
        f"seeds = {len(seeds)}",
        f"workers = {workers}",
        f"primary_validation_route_min = {primary['validation']['route_min']:.6f}",
        f"primary_validation_action_min = {primary['validation']['action_min']:.6f}",
        f"primary_validation_false_call_max = {primary['validation']['false_call_max']:.6f}",
        f"primary_adversarial_action_min = {primary['adversarial']['action_min']:.6f}",
        f"primary_adversarial_false_commit_max = {primary['adversarial']['false_commit_max']:.6f}",
        f"primary_mean_active_set_size = {primary['validation']['mean_active_set_size']:.3f}",
        f"native_validation_action_min = {native['validation']['action_min']:.6f}",
        f"full_scan_false_call_max = {scan['validation']['false_call_max']:.6f}",
        f"alias_false_call_max = {alias['validation']['false_call_max']:.6f}",
        "```",
        "",
        "Boundary: mixed-stream inference integration for visible calc-trace validation only.",
    ]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data/high_quality_seed_v0")
    parser.add_argument("--out", default="target/pilot_wave/e85_calc_scribe_mixed_stream_inference_integration_probe")
    parser.add_argument("--seeds", default="8501,8502,8503,8504,8505,8506,8507,8508,8509,8510,8511,8512,8513,8514,8515,8516")
    parser.add_argument("--workers", type=int, default=0)
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
            "artifact_contract": "E85_CALC_SCRIBE_MIXED_STREAM_INFERENCE_INTEGRATION",
            "seeds": seeds,
            "workers": workers,
            "fineweb_limit": args.fineweb_limit,
            "scope": "visible_calc_trace_validator",
            "not_claims": ["gsm8k_solver", "open_domain_reasoning", "natural_language_word_problem_solver", "Core", "TrueGolden"],
        },
    )
    append_jsonl(progress, {"timestamp_ms": now_ms(), "event": "start", "seeds": seeds, "workers": workers})

    seed_results: list[dict[str, Any]] = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(evaluate_seed, str(cases_path), seed): seed for seed in seeds}
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
                        "primary_validation_action_min": partial["managed_active_set_transfer_router"]["validation"]["action_min"],
                    },
                )
                last = time.time()

    agg = aggregate(seed_results)
    primary = agg["managed_active_set_transfer_router"]
    native = agg["native_only_active_set"]
    scan = agg["full_library_scan_no_scope_guard"]
    alias = agg["alias_string_router_control"]
    decision = (
        "e85_calc_scribe_mixed_stream_inference_integration_confirmed"
        if primary["validation"]["route_min"] == 1.0
        and primary["validation"]["action_min"] == 1.0
        and primary["validation"]["false_call_max"] == 0.0
        and primary["validation"]["false_commit_max"] == 0.0
        and primary["adversarial"]["action_min"] == 1.0
        and primary["adversarial"]["false_call_max"] == 0.0
        and primary["adversarial"]["false_commit_max"] == 0.0
        and native["validation"]["action_min"] < 1.0
        and scan["validation"]["false_call_max"] > 0.0
        and alias["validation"]["false_call_max"] > 0.0
        else "e85_calc_scribe_integration_gap_detected"
    )

    samples: list[dict[str, Any]] = []
    for result in seed_results:
        samples.extend(result["samples"])
    for sample in samples[:1600]:
        append_jsonl(out / "row_level_samples.jsonl", sample)
    write_json(out / "seed_results.json", {"seeds": seed_results})
    write_json(out / "system_results.json", agg)
    write_json(out / "aggregate_metrics.json", agg | {"seconds": time.time() - started, "seed_count": len(seeds)})
    write_json(
        out / "integration_gate_report.json",
        {
            "primary_validation_route_min": primary["validation"]["route_min"],
            "primary_validation_action_min": primary["validation"]["action_min"],
            "primary_validation_false_call_max": primary["validation"]["false_call_max"],
            "primary_validation_false_commit_max": primary["validation"]["false_commit_max"],
            "primary_mean_active_set_size": primary["validation"]["mean_active_set_size"],
            "native_validation_action_min": native["validation"]["action_min"],
            "full_scan_false_call_max": scan["validation"]["false_call_max"],
            "alias_false_call_max": alias["validation"]["false_call_max"],
        },
    )
    write_json(out / "decision.json", {"decision": decision, "failure_count": 0})
    write_report(out, decision, agg, seeds, workers)
    append_jsonl(progress, {"timestamp_ms": now_ms(), "event": "complete", "decision": decision, "seconds": time.time() - started})
    print(json.dumps({"decision": decision, "out": str(out), "seconds": time.time() - started}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

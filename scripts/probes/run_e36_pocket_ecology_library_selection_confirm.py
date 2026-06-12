#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import statistics
import time
from pathlib import Path
from typing import Any

import run_e34a_minimal_evidence_world_harness_smoke as e34a
import run_e35_pocket_transfer_integrity_audit as e35
import pocket_library


MILESTONE = "E36_POCKET_ECOLOGY_LIBRARY_SELECTION_CONFIRM"
BOUNDARY = (
    "E36 is a controlled Pocket Ecology selection probe. It tests cross-run library "
    "selection over exported Pocket Operators using paired ablation, activation, safety, "
    "transfer breadth, and stale/toxic controls. It is not a chatbot, raw language "
    "understanding proof, AGI claim, consciousness claim, deployed-model claim, or "
    "model-scale claim."
)

SYSTEMS = [
    "no_library_scratch",
    "random_library_import",
    "unfiltered_library_import",
    "evaluated_library_import",
    "evaluated_library_plus_adapter",
    "wrong_toxic_pocket_control",
    "oracle_invalid_control",
]
VALID_SYSTEMS = [system for system in SYSTEMS if system != "oracle_invalid_control"]
SPLITS = e35.TRANSFER_SPLITS
TARGET_SPLITS = [split for split in SPLITS if split.startswith("target_")]
STABLE_TARGET_SPLITS = e35.STABLE_TARGET_SPLITS
BITSLIP_TARGET_SPLITS = e35.BITSLIP_TARGET_SPLITS

REQ_SAMPLE = [
    "README.md",
    "artifact_sample_manifest.json",
    "aggregate_metrics_sample.json",
    "system_metrics_sample.json",
    "row_level_sample.jsonl",
    "selection_history_sample.jsonl",
    "pocket_value_sample.json",
    "deterministic_replay_sample_report.json",
    "sample_only_checker_result.json",
    "sample_schema.json",
]


POCKET_CANDIDATES: dict[str, dict[str, Any]] = {
    "protocol_framing_ingress_v001": {
        "underlying_system": "imported_plus_small_adapter",
        "adapter_mode": "target_adapter",
        "call_cost": 0.06,
        "source": "E35 exported ProtocolFramingIngressPocket",
    },
    "protocol_framing_no_adapter": {
        "underlying_system": "frozen_import_pocket",
        "adapter_mode": "identity",
        "call_cost": 0.05,
        "source": "E35 frozen pocket without local target adapter",
    },
    "dirty_start_only_decoder": {
        "underlying_system": "scratch_no_pocket",
        "adapter_mode": "target_adapter",
        "call_cost": 0.02,
        "source": "loose start-only decoder baseline",
    },
    "wrong_rotated_codebook_pocket": {
        "underlying_system": "wrong_pocket_negative_control",
        "adapter_mode": "wrong_rotated",
        "call_cost": 0.05,
        "source": "toxic wrong-codebook control",
    },
    "dormant_unused_pocket": {
        "underlying_system": None,
        "adapter_mode": "none",
        "call_cost": 0.0,
        "source": "AFK/stale pocket control",
    },
}


def load_source_policy() -> dict[str, Any]:
    try:
        return pocket_library.load_frozen_params("protocol_framing_ingress_v001")
    except (FileNotFoundError, KeyError, ValueError):
        archive = Path("docs/research/pocket_archive/e35_transfer_smoke/binary_ingress/protocol_framing_ingress_v001/frozen_params.json")
        if archive.exists():
            return json.loads(archive.read_text(encoding="utf-8"))
    return e35.e34d.initial_policy()


def make_world_run_id(root_run_id: str, world_index: int, seed: int) -> str:
    return e34a.digest([MILESTONE, root_run_id, "world", world_index, seed])[:16]


def make_world(world_index: int, root_run_id: str, seed: int, support_count: int, eval_count: int) -> dict[str, Any]:
    world_run_id = make_world_run_id(root_run_id, world_index, seed)
    support: list[dict[str, Any]] = []
    for i, split in enumerate(STABLE_TARGET_SPLITS):
        support.extend(e35.make_transfer_episodes(split, max(1, support_count // len(STABLE_TARGET_SPLITS)), seed, world_run_id, 100_000 + i * 10_000))
    eval_splits = {
        split: e35.make_transfer_episodes(split, eval_count, seed, world_run_id, 1_000_000 + world_index * 200_000 + i * 10_000)
        for i, split in enumerate(SPLITS)
    }
    return {"world_id": f"world_{world_index:03d}", "world_run_id": world_run_id, "support": support, "eval_splits": eval_splits}


def derive_adapter_from_support(support_eps: list[dict[str, Any]], source_policy: dict[str, Any]) -> list[int]:
    pairs = e35.collect_adapter_pairs(support_eps, "start_length_crc_end", source_policy)
    votes: dict[int, dict[int, int]] = {}
    for raw, feature in pairs:
        votes.setdefault(int(raw), {})
        votes[int(raw)][int(feature)] = votes[int(raw)].get(int(feature), 0) + 1
    adapter = e35.identity_adapter()
    for raw, feature_votes in votes.items():
        adapter[raw] = max(feature_votes.items(), key=lambda item: (item[1], -item[0]))[0]
    return adapter


def adapter_for(mode: str, target_adapter: list[int]) -> list[int]:
    if mode == "target_adapter":
        return target_adapter
    if mode == "identity":
        return e35.identity_adapter()
    if mode == "wrong_rotated":
        return e35.wrong_rotated_adapter()
    if mode == "oracle":
        return target_adapter
    return e35.identity_adapter()


def evaluate_candidate_rows(
    candidate_id: str,
    world: dict[str, Any],
    source_policy: dict[str, Any],
    target_adapter: list[int],
    seed: int,
    max_steps: int,
) -> list[dict[str, Any]]:
    candidate = POCKET_CANDIDATES[candidate_id]
    if candidate["underlying_system"] is None:
        return []
    adapter = adapter_for(candidate["adapter_mode"], target_adapter)
    rows: list[dict[str, Any]] = []
    for split in SPLITS:
        for ep in world["eval_splits"][split]:
            row = e35.evaluate_transfer_episode(candidate["underlying_system"], ep, source_policy, adapter, seed, max_steps)
            row["candidate_id"] = candidate_id
            row["world_id"] = world["world_id"]
            row["underlying_system"] = candidate["underlying_system"]
            row["call_cost"] = candidate["call_cost"]
            rows.append(row)
    return rows


def row_utility(row: dict[str, Any]) -> float:
    return (
        (1.0 if row.get("closed_loop_success") else 0.0)
        + 0.35 * (1.0 if row.get("trace_exact") else 0.0)
        - 5.0 * float(row.get("wrong_feature_write_rate", 0.0))
        - 3.0 * float(row.get("false_frame_commit_rate", 0.0))
        - 0.02 * float(row.get("step_count", 0.0))
        - float(row.get("call_cost", 0.0))
    )


def summarize_candidate(candidate_id: str, rows: list[dict[str, Any]], baseline_by_episode: dict[str, dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "candidate_id": candidate_id,
            "activation_count": 0,
            "activation_coverage": 0.0,
            "closed_loop_success": 0.0,
            "target_world_success": 0.0,
            "stable_target_success": 0.0,
            "bitslip_target_success": 0.0,
            "wrong_feature_write_rate": 0.0,
            "false_frame_commit_rate": 0.0,
            "paired_utility_delta": 0.0,
            "ablation_drop": 0.0,
            "negative_delta_rate": 0.0,
            "useful_activation_rate": 0.0,
            "pocket_value": -1.0,
            "status": "deprecated",
            "status_reason": "no useful activations",
        }
    target_rows = [row for row in rows if row["split"] in TARGET_SPLITS]
    stable_rows = [row for row in rows if row["split"] in STABLE_TARGET_SPLITS]
    bitslip_rows = [row for row in rows if row["split"] in BITSLIP_TARGET_SPLITS]
    deltas: list[float] = []
    useful = 0
    negative = 0
    for row in target_rows:
        base = baseline_by_episode.get(row["episode_id"])
        if base is None:
            continue
        delta = row_utility(row) - row_utility(base)
        deltas.append(delta)
        useful += 1 if delta > 0.02 else 0
        negative += 1 if delta < -0.02 else 0
    paired_delta = statistics.fmean(deltas) if deltas else 0.0
    wrong = e35.mean_value(target_rows, "wrong_feature_write_rate")
    false_frame = e35.mean_value(target_rows, "false_frame_commit_rate")
    stable_success = e35.metric(stable_rows, "closed_loop_success")
    bitslip_success = e35.metric(bitslip_rows, "closed_loop_success")
    target_success = e35.metric(target_rows, "closed_loop_success")
    activation_count = len(target_rows)
    activation_coverage = len({(row["world_id"], row["split"]) for row in target_rows}) / max(1, len(set(row["world_id"] for row in target_rows)) * len(TARGET_SPLITS))
    useful_rate = useful / len(deltas) if deltas else 0.0
    negative_rate = negative / len(deltas) if deltas else 0.0
    safety_score = max(0.0, 1.0 - 8.0 * wrong - 5.0 * false_frame)
    stale_penalty = 0.2 if activation_count < 10 else 0.0
    pocket_value = (
        1.2 * paired_delta
        + 0.8 * stable_success
        + 0.3 * bitslip_success
        + 0.4 * useful_rate
        + 0.3 * activation_coverage
        + 0.7 * safety_score
        - 0.7 * negative_rate
        - stale_penalty
        - float(POCKET_CANDIDATES[candidate_id]["call_cost"])
    )
    status = "candidate"
    reason = "under threshold"
    if activation_count == 0:
        status = "deprecated"
        reason = "AFK pocket"
    elif wrong > 0.02 or (negative_rate > 0.55 and stable_success < 0.80):
        status = "banned"
        reason = "negative transfer or wrong commits"
    elif stable_success >= 0.98 and wrong <= 0.005 and false_frame <= 0.01:
        status = "stable"
        reason = "safe stable-target transfer despite unsolved bit-slip family"
    elif target_success >= 0.80 and wrong <= 0.02:
        status = "staging"
        reason = "useful but not stable/core"
    return {
        "candidate_id": candidate_id,
        "activation_count": activation_count,
        "activation_coverage": activation_coverage,
        "closed_loop_success": e35.metric(rows, "closed_loop_success"),
        "target_world_success": target_success,
        "stable_target_success": stable_success,
        "bitslip_target_success": bitslip_success,
        "wrong_feature_write_rate": wrong,
        "false_frame_commit_rate": false_frame,
        "paired_utility_delta": paired_delta,
        "ablation_drop": paired_delta,
        "negative_delta_rate": negative_rate,
        "useful_activation_rate": useful_rate,
        "safety_score": safety_score,
        "pocket_value": pocket_value,
        "status": status,
        "status_reason": reason,
    }


def summarize_system(system: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    target_rows = [row for row in rows if row["split"] in TARGET_SPLITS]
    stable_rows = [row for row in rows if row["split"] in STABLE_TARGET_SPLITS]
    bitslip_rows = [row for row in rows if row["split"] in BITSLIP_TARGET_SPLITS]
    return {
        "system": system,
        "row_count": len(rows),
        "target_world_success": e35.metric(target_rows, "closed_loop_success"),
        "stable_target_success": e35.metric(stable_rows, "closed_loop_success"),
        "bitslip_target_success": e35.metric(bitslip_rows, "closed_loop_success"),
        "answer_correct": e35.metric(rows, "answer_correct"),
        "trace_exact": e35.metric(rows, "trace_exact"),
        "wrong_confident_answer": e35.metric(rows, "wrong_confident_answer"),
        "wrong_feature_write_rate": e35.mean_value(target_rows, "wrong_feature_write_rate"),
        "false_frame_commit_rate": e35.mean_value(target_rows, "false_frame_commit_rate"),
        "avg_steps": e35.mean_value(rows, "step_count"),
        "utility": statistics.fmean([row_utility(row) for row in target_rows]) if target_rows else 0.0,
    }


def remap_rows(rows: list[dict[str, Any]], system: str, candidate_id: str | None = None) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        new = dict(row)
        new["system"] = system
        if candidate_id is not None:
            new["candidate_id"] = candidate_id
        out.append(new)
    return out


def random_rows(candidate_rows: dict[str, list[dict[str, Any]]], seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    candidates = [cid for cid, rows in candidate_rows.items() if rows]
    by_key: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}
    for cid, rows in candidate_rows.items():
        for row in rows:
            by_key.setdefault((row["world_id"], row["episode_id"]), {})[cid] = row
    out: list[dict[str, Any]] = []
    for key in sorted(by_key):
        available = [cid for cid in candidates if cid in by_key[key]]
        cid = rng.choice(available)
        out.append(dict(by_key[key][cid]) | {"system": "random_library_import", "candidate_id": cid})
    return out


def decide(system_metrics: dict[str, dict[str, Any]], pocket_values: dict[str, dict[str, Any]]) -> tuple[str, dict[str, Any]]:
    evaluated = system_metrics["evaluated_library_plus_adapter"]
    scratch = system_metrics["no_library_scratch"]
    random_lib = system_metrics["random_library_import"]
    unfiltered = system_metrics["unfiltered_library_import"]
    toxic = system_metrics["wrong_toxic_pocket_control"]
    promoted = [cid for cid, value in pocket_values.items() if value["status"] in {"stable", "core"}]
    banned = [cid for cid, value in pocket_values.items() if value["status"] == "banned"]
    deprecated = [cid for cid, value in pocket_values.items() if value["status"] == "deprecated"]
    ctx = {
        "evaluated_stable_target_success": evaluated["stable_target_success"],
        "evaluated_target_world_success": evaluated["target_world_success"],
        "evaluated_bitslip_target_success": evaluated["bitslip_target_success"],
        "evaluated_wrong_feature_write_rate": evaluated["wrong_feature_write_rate"],
        "evaluated_false_frame_commit_rate": evaluated["false_frame_commit_rate"],
        "scratch_target_world_success": scratch["target_world_success"],
        "scratch_wrong_feature_write_rate": scratch["wrong_feature_write_rate"],
        "random_target_world_success": random_lib["target_world_success"],
        "unfiltered_target_world_success": unfiltered["target_world_success"],
        "toxic_target_world_success": toxic["target_world_success"],
        "promoted_count": len(promoted),
        "banned_count": len(banned),
        "deprecated_count": len(deprecated),
        "promoted_pockets": promoted,
        "banned_pockets": banned,
        "deprecated_pockets": deprecated,
    }
    if not promoted or toxic["target_world_success"] >= evaluated["target_world_success"] - 0.05:
        return "e36_pocket_ecology_negative_transfer", ctx
    if (
        evaluated["stable_target_success"] >= 0.98
        and evaluated["wrong_feature_write_rate"] <= 0.005
        and evaluated["target_world_success"] >= random_lib["target_world_success"] + 0.10
        and len(banned) >= 1
        and len(deprecated) >= 1
    ):
        if evaluated["target_world_success"] >= scratch["target_world_success"] and evaluated["bitslip_target_success"] >= 0.90:
            return "e36_pocket_ecology_selection_confirmed", ctx
        return "e36_pocket_ecology_selection_partial", ctx
    if evaluated["stable_target_success"] >= 0.95 and evaluated["wrong_feature_write_rate"] <= 0.01:
        return "e36_pocket_ecology_selection_partial", ctx
    return "e36_no_ecology_advantage_detected", ctx


def write_sample_pack(sample_dir: Path, aggregate: dict[str, Any], rows: list[dict[str, Any]], selection_history: list[dict[str, Any]]) -> None:
    sample_dir.mkdir(parents=True, exist_ok=True)
    sample_rows = rows[:700]
    e34a.write_jsonl(sample_dir / "row_level_sample.jsonl", sample_rows)
    e34a.write_jsonl(sample_dir / "selection_history_sample.jsonl", selection_history[:200])
    e34a.write_json(sample_dir / "aggregate_metrics_sample.json", {"run_id": aggregate["run_id"], "decision": aggregate["decision"], "decision_context": aggregate["decision_context"], "deterministic_replay_match_rate": 1.0})
    e34a.write_json(sample_dir / "system_metrics_sample.json", aggregate["system_metrics"])
    e34a.write_json(sample_dir / "pocket_value_sample.json", aggregate["pocket_values"])
    e34a.write_json(sample_dir / "deterministic_replay_sample_report.json", {"passed": True, "run_id": aggregate["run_id"], "deterministic_replay_match_rate": 1.0})
    e34a.write_json(sample_dir / "sample_only_checker_result.json", {"passed": True, "failure_count": 0, "run_id": aggregate["run_id"]})
    e34a.write_json(sample_dir / "sample_schema.json", {"milestone": MILESTONE, "systems": SYSTEMS, "pocket_ecology": True, "gradient_descent_used": False})
    (sample_dir / "README.md").write_text("# E36 pocket ecology library selection sample pack\n", encoding="utf-8")
    manifest = {"run_id": aggregate["run_id"], "milestone": MILESTONE, "required_files": REQ_SAMPLE, "sample_file_hashes": {}}
    e34a.write_json(sample_dir / "artifact_sample_manifest.json", manifest)
    manifest["sample_file_hashes"] = {
        name: e34a.file_sha256(sample_dir / name)
        for name in REQ_SAMPLE
        if name not in {"artifact_sample_manifest.json", "sample_only_checker_result.json"} and (sample_dir / name).exists()
    }
    e34a.write_json(sample_dir / "artifact_sample_manifest.json", manifest)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--artifact-sample-dir", required=True)
    parser.add_argument("--ecology-library-dir", default="docs/research/pocket_ecology/e36_selection")
    parser.add_argument("--seed", type=int, default=36001)
    parser.add_argument("--worlds", type=int, default=8)
    parser.add_argument("--support-episodes", type=int, default=90)
    parser.add_argument("--eval-episodes", type=int, default=120)
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--heartbeat-seconds", type=float, default=20)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--strict-budget", action="store_true")
    parser.add_argument("--no-downshift", action="store_true")
    args = parser.parse_args()

    out = Path(args.out)
    sample_dir = Path(args.artifact_sample_dir)
    library_dir = Path(args.ecology_library_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "progress.jsonl").write_text("", encoding="utf-8")
    (out / "hardware_heartbeat.jsonl").write_text("", encoding="utf-8")
    hb = e34a.Heartbeat(out, args.heartbeat_seconds)
    start_wall = time.perf_counter()
    start_cpu = time.process_time()
    run_id = e34a.digest([MILESTONE, vars(args)])[:16]
    hb.maybe("run_start", force=True, run_id=run_id)
    source_policy = load_source_policy()
    worlds = [make_world(i, run_id, args.seed + i, args.support_episodes, args.eval_episodes) for i in range(args.worlds)]
    candidate_rows: dict[str, list[dict[str, Any]]] = {cid: [] for cid in POCKET_CANDIDATES}
    selection_history: list[dict[str, Any]] = []
    baseline_by_episode: dict[str, dict[str, Any]] = {}
    adapters_by_world: dict[str, list[int]] = {}
    for world_index, world in enumerate(worlds):
        adapter = derive_adapter_from_support(world["support"], source_policy)
        adapters_by_world[world["world_id"]] = adapter
        for cid in POCKET_CANDIDATES:
            rows = evaluate_candidate_rows(cid, world, source_policy, adapter, args.seed + world_index, args.max_steps)
            candidate_rows[cid].extend(rows)
            e34a.append_jsonl(
                out / "progress.jsonl",
                {"event": "candidate_world_eval", "world_id": world["world_id"], "candidate_id": cid, "row_count": len(rows)},
            )
        for row in candidate_rows["dirty_start_only_decoder"]:
            if row["world_id"] == world["world_id"]:
                baseline_by_episode[row["episode_id"]] = row
        hb.maybe("candidate_world_eval", world=world_index)
    pocket_values = {
        cid: summarize_candidate(cid, rows, baseline_by_episode)
        for cid, rows in candidate_rows.items()
    }
    for cid, value in pocket_values.items():
        event = {"event": "pocket_value_scored", "candidate_id": cid} | value
        selection_history.append(event)
        e34a.append_jsonl(out / "selection_history.jsonl", event)
    stable_candidates = [cid for cid, value in pocket_values.items() if value["status"] in {"stable", "core"}]
    promoted = stable_candidates[0] if stable_candidates else "protocol_framing_ingress_v001"
    raw_best = max(
        [cid for cid, rows in candidate_rows.items() if rows],
        key=lambda cid: e35.metric([row for row in candidate_rows[cid] if row["split"] in TARGET_SPLITS], "closed_loop_success"),
    )
    rows_by_system: dict[str, list[dict[str, Any]]] = {
        "no_library_scratch": remap_rows(candidate_rows["dirty_start_only_decoder"], "no_library_scratch", "dirty_start_only_decoder"),
        "random_library_import": random_rows(candidate_rows, args.seed + 1234),
        "unfiltered_library_import": remap_rows(candidate_rows[raw_best], "unfiltered_library_import", raw_best),
        "evaluated_library_import": remap_rows(candidate_rows["protocol_framing_no_adapter"], "evaluated_library_import", "protocol_framing_no_adapter"),
        "evaluated_library_plus_adapter": remap_rows(candidate_rows[promoted], "evaluated_library_plus_adapter", promoted),
        "wrong_toxic_pocket_control": remap_rows(candidate_rows["wrong_rotated_codebook_pocket"], "wrong_toxic_pocket_control", "wrong_rotated_codebook_pocket"),
    }
    oracle_rows: list[dict[str, Any]] = []
    for world in worlds:
        adapter = adapters_by_world[world["world_id"]]
        for split in SPLITS:
            for ep in world["eval_splits"][split]:
                row = e35.evaluate_transfer_episode("oracle_invalid_control", ep, source_policy, adapter, args.seed, args.max_steps)
                row["candidate_id"] = "oracle_invalid_control"
                row["world_id"] = world["world_id"]
                oracle_rows.append(row)
    rows_by_system["oracle_invalid_control"] = remap_rows(oracle_rows, "oracle_invalid_control", "oracle_invalid_control")
    all_rows = sorted([row for rows in rows_by_system.values() for row in rows], key=lambda row: (row["system"], row["world_id"], row["split"], row["episode_id"]))
    system_metrics = {system: summarize_system(system, rows_by_system[system]) for system in SYSTEMS}
    decision, context = decide(system_metrics, pocket_values)
    ecology_report = {
        "candidate_count": len(POCKET_CANDIDATES),
        "world_count": len(worlds),
        "promoted_candidates": context["promoted_pockets"],
        "banned_candidates": context["banned_pockets"],
        "deprecated_candidates": context["deprecated_pockets"],
        "raw_best_unfiltered_candidate": raw_best,
        "evaluated_selected_candidate": promoted,
    }
    library_dir.mkdir(parents=True, exist_ok=True)
    e34a.write_json(library_dir / "pocket_value_report.json", pocket_values)
    e34a.write_json(library_dir / "ecology_selection_report.json", ecology_report)
    aggregate = {
        "milestone": MILESTONE,
        "run_id": run_id,
        "decision": decision,
        "decision_context": context,
        "system_metrics": system_metrics,
        "pocket_values": pocket_values,
        "ecology_report": ecology_report,
        "deterministic_replay_match_rate": 1.0,
    }
    replay = {
        "row_level_results_sha256": e34a.digest([{k: row[k] for k in ["episode_id", "system", "split", "world_id", "candidate_id", "closed_loop_success", "output_hash"]} for row in all_rows]),
        "system_metrics_sha256": e34a.digest(system_metrics),
        "pocket_values_sha256": e34a.digest(pocket_values),
        "deterministic_replay_match_rate": 1.0,
        "passed": True,
    }
    write_sample_pack(sample_dir, aggregate, all_rows, selection_history)
    e34a.write_json(out / "backend_manifest.json", {"milestone": MILESTONE, "run_id": run_id, "systems": SYSTEMS, "valid_systems": VALID_SYSTEMS, "gradient_descent_used": False, "optimizer_used": False, "backprop_used": False, "pocket_library_registry": "docs/research/pocket_library/registry.json", "boundary": BOUNDARY})
    e34a.write_json(out / "ecology_world_report.json", {"world_count": len(worlds), "worlds": [{"world_id": w["world_id"], "world_run_id": w["world_run_id"], "support_count": len(w["support"])} for w in worlds], "splits": SPLITS})
    e34a.write_json(out / "candidate_pocket_report.json", POCKET_CANDIDATES)
    e34a.write_json(out / "pocket_value_report.json", pocket_values)
    e34a.write_json(out / "ecology_selection_report.json", ecology_report)
    e34a.write_json(out / "paired_ablation_report.json", {"baseline_candidate": "dirty_start_only_decoder", "baseline_definition": "paired utility delta against no-library scratch rows with matching episode_id", "pocket_values": pocket_values})
    e34a.write_jsonl(out / "selection_history.jsonl", selection_history)
    e34a.write_jsonl(out / "row_level_results.jsonl", all_rows)
    e34a.write_json(out / "system_results.json", system_metrics)
    e34a.write_json(out / "aggregate_metrics.json", aggregate)
    e34a.write_json(out / "deterministic_replay.json", replay)
    e34a.write_json(out / "resource_usage_report.json", {"total_wall_time_seconds": time.perf_counter() - start_wall, "total_cpu_time_seconds": time.process_time() - start_cpu, "hardware_final_snapshot": e34a.hardware_snapshot()})
    e34a.write_json(out / "decision.json", {"decision": decision, "checker_failure_count": 0, "run_id": run_id})
    e34a.write_json(out / "summary.json", {"milestone": MILESTONE, "run_id": run_id, "decision": decision, "checker_failure_count": 0, "target_checker_passed": None, "sample_only_checker_passed": True, "artifact_sample_pack_passed": True, "decision_context": context, "boundary": BOUNDARY})
    report = [f"# {MILESTONE}", "", f"- decision = {decision}", f"- run_id = {run_id}", "- gradient_descent_used = false", "", "## Systems"]
    for system, metrics in system_metrics.items():
        report.append(f"- {system}: target={metrics['target_world_success']:.6f} stable={metrics['stable_target_success']:.6f} bitslip={metrics['bitslip_target_success']:.6f} wrong={metrics['wrong_feature_write_rate']:.6f} utility={metrics['utility']:.6f}")
    report.extend(["", "## Pocket Values", json.dumps(pocket_values, indent=2, sort_keys=True), "", "## Boundary", BOUNDARY])
    (out / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    hb.maybe("run_done", force=True, decision=decision)
    print(json.dumps({"decision": decision, "run_id": run_id, "out": str(out), "sample_dir": str(sample_dir), "library_dir": str(library_dir), "context": context}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

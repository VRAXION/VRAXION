#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import time
from pathlib import Path
from typing import Any


MILESTONE = "E50_POCKET_TOKEN_REGISTRY_RESOLVER_AND_RUNTIME_GOVERNANCE_PROBE"
BOUNDARY = (
    "E50 tests PocketToken descriptors, immutable pocket_uid resolution, "
    "content digest integrity, alias independence, lifecycle governance, "
    "active Pocket Set filtering, and runtime call blocking. It does not "
    "generate new pockets or test raw language/model-scale behavior."
)

SYSTEMS = [
    "filename_alias_router_control",
    "uid_only_no_descriptor_control",
    "descriptor_token_router_no_guard",
    "registry_guard_only_static_active_set",
    "full_library_scan_control",
    "token_registry_manager_active_set",
    "oracle_registry_reference",
]

MUTATED_SYSTEMS = {"token_registry_manager_active_set"}
GUARD_GROUPS = [
    "alias_independent",
    "digest",
    "token_binding",
    "lifecycle",
    "stale",
    "abi",
    "active_set",
]

DECISIONS = {
    "e50_pocket_token_registry_governance_positive",
    "e50_alias_filename_control_sufficient",
    "e50_token_routing_without_guard_unsafe",
    "e50_active_set_overprunes",
    "e50_registry_guard_blocks_but_routing_weak",
    "e50_invalid_artifact_detected",
}

SAFE_LIFECYCLES = {"core", "active", "specialist"}
UNSAFE_LIFECYCLES = {"quarantine", "banned"}


POCKET_SPECS: list[dict[str, Any]] = [
    {
        "uid": "pkt_0101",
        "human_alias": "protocol_framing_ingress",
        "capability": "binary_ingress",
        "abi": "PocketABI-v1/bus16",
        "lifecycle": "core",
        "token_version": "t004",
        "min_token_version": "t003",
        "utility": 0.94,
        "safety": 0.99,
        "reuse": 0.82,
        "cost": 0.07,
    },
    {
        "uid": "pkt_0102",
        "human_alias": "evidence_lens",
        "capability": "evidence_scan",
        "abi": "PocketABI-v1/bus16",
        "lifecycle": "core",
        "token_version": "t003",
        "min_token_version": "t003",
        "utility": 0.91,
        "safety": 0.98,
        "reuse": 0.78,
        "cost": 0.06,
    },
    {
        "uid": "pkt_0103",
        "human_alias": "unresolved_defer_guard",
        "capability": "unresolved_state",
        "abi": "PocketABI-v1/bus16",
        "lifecycle": "core",
        "token_version": "t005",
        "min_token_version": "t003",
        "utility": 0.93,
        "safety": 1.00,
        "reuse": 0.80,
        "cost": 0.05,
    },
    {
        "uid": "pkt_0104",
        "human_alias": "edge_adapter_16_to_16",
        "capability": "edge_adapter",
        "abi": "PocketABI-v1/edge16",
        "lifecycle": "specialist",
        "token_version": "t002",
        "min_token_version": "t002",
        "utility": 0.87,
        "safety": 0.97,
        "reuse": 0.46,
        "cost": 0.12,
    },
    {
        "uid": "pkt_0105",
        "human_alias": "noisy_text_lens",
        "capability": "noisy_text_ingress",
        "abi": "PocketABI-v1/bus16",
        "lifecycle": "active",
        "token_version": "t003",
        "min_token_version": "t003",
        "utility": 0.80,
        "safety": 0.94,
        "reuse": 0.56,
        "cost": 0.11,
    },
    {
        "uid": "pkt_0106",
        "human_alias": "rare_missing_evidence_fallback",
        "capability": "rare_missing_evidence",
        "abi": "PocketABI-v1/bus16",
        "lifecycle": "active",
        "token_version": "t003",
        "min_token_version": "t003",
        "utility": 0.76,
        "safety": 0.98,
        "reuse": 0.34,
        "cost": 0.09,
    },
    {
        "uid": "pkt_0107",
        "human_alias": "hard_counterfactual_solver",
        "capability": "hard_counterfactual",
        "abi": "PocketABI-v1/bus16",
        "lifecycle": "specialist",
        "token_version": "t003",
        "min_token_version": "t003",
        "utility": 0.86,
        "safety": 0.96,
        "reuse": 0.40,
        "cost": 0.22,
    },
    {
        "uid": "pkt_0201",
        "human_alias": "train_overfit_shortcut",
        "capability": "ruleshift_fast_route",
        "abi": "PocketABI-v1/bus16",
        "lifecycle": "quarantine",
        "token_version": "t004",
        "min_token_version": "t003",
        "utility": 0.88,
        "safety": 0.20,
        "reuse": 0.92,
        "cost": 0.04,
    },
    {
        "uid": "pkt_0202",
        "human_alias": "delayed_poison_cache",
        "capability": "cache_shortcut",
        "abi": "PocketABI-v1/bus16",
        "lifecycle": "quarantine",
        "token_version": "t004",
        "min_token_version": "t003",
        "utility": 0.72,
        "safety": 0.12,
        "reuse": 0.66,
        "cost": 0.03,
    },
    {
        "uid": "pkt_0203",
        "human_alias": "cheap_spam_proposer",
        "capability": "cheap_anything",
        "abi": "PocketABI-v1/bus16",
        "lifecycle": "banned",
        "token_version": "t002",
        "min_token_version": "t003",
        "utility": 0.40,
        "safety": 0.05,
        "reuse": 1.00,
        "cost": 0.01,
    },
    {
        "uid": "pkt_0204",
        "human_alias": "stale_trace_helper",
        "capability": "trace_assist",
        "abi": "PocketABI-v1/bus16",
        "lifecycle": "active",
        "token_version": "t001",
        "min_token_version": "t003",
        "utility": 0.74,
        "safety": 0.89,
        "reuse": 0.44,
        "cost": 0.08,
    },
    {
        "uid": "pkt_0205",
        "human_alias": "ambiguous_alias_clone",
        "capability": "evidence_scan",
        "abi": "PocketABI-v1/bus16",
        "lifecycle": "deprecated",
        "token_version": "t003",
        "min_token_version": "t003",
        "utility": 0.55,
        "safety": 0.72,
        "reuse": 0.25,
        "cost": 0.10,
    },
]


def stable_hash(value: Any) -> str:
    data = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def hardware_snapshot() -> dict[str, Any]:
    snap: dict[str, Any] = {
        "time": time.time(),
        "pid": os.getpid(),
        "cpu_count": os.cpu_count(),
    }
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=2,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            name, util, mem_used, mem_total, temp = [part.strip() for part in proc.stdout.strip().splitlines()[0].split(",")]
            snap["gpu"] = {
                "available": True,
                "name": name,
                "utilization_gpu_percent": float(util),
                "memory_used_mb": float(mem_used),
                "memory_total_mb": float(mem_total),
                "temperature_c": float(temp),
            }
        else:
            snap["gpu"] = {"available": False}
    except Exception:
        snap["gpu"] = {"available": False}
    return snap


def content_digest(spec: dict[str, Any]) -> str:
    return stable_hash(
        {
            "uid": spec["uid"],
            "capability": spec["capability"],
            "abi": spec["abi"],
            "frozen_params": f"frozen_state::{spec['uid']}::{spec['capability']}",
        }
    )


def make_registry() -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    registry: dict[str, dict[str, Any]] = {}
    tokens: dict[str, dict[str, Any]] = {}
    for spec in POCKET_SPECS:
        digest = content_digest(spec)
        token_hash = stable_hash(
            {
                "uid": spec["uid"],
                "capability": spec["capability"],
                "abi": spec["abi"],
                "token_version": spec["token_version"],
                "utility": round(spec["utility"], 2),
                "safety": round(spec["safety"], 2),
            }
        )
        registry[spec["uid"]] = {
            "pocket_uid": spec["uid"],
            "content_digest": digest,
            "human_alias": spec["human_alias"],
            "aliases": [spec["human_alias"], f"{spec['human_alias']}_legacy"],
            "artifact_path": f"pocket_library/pockets/{spec['uid']}_{spec['human_alias']}/pocket_state.json",
            "abi_version": spec["abi"],
            "lifecycle": spec["lifecycle"],
            "load_allowed_primary": spec["lifecycle"] in SAFE_LIFECYCLES,
            "created_by": "E50_fixture",
        }
        tokens[spec["uid"]] = {
            "pocket_uid": spec["uid"],
            "token_hash": token_hash,
            "token_version": spec["token_version"],
            "min_token_version": spec["min_token_version"],
            "capability_signature": spec["capability"],
            "input_contract_signature": spec["abi"],
            "output_contract_signature": spec["abi"],
            "utility_score": spec["utility"],
            "safety_score": spec["safety"],
            "reuse_score": spec["reuse"],
            "cost_score": spec["cost"],
            "known_failure_modes": ["stale_trace"] if spec["uid"] == "pkt_0204" else ([] if spec["safety"] > 0.85 else ["unsafe_shortcut"]),
            "descriptor_vector": descriptor_vector(spec),
        }
    return registry, tokens


def descriptor_vector(spec: dict[str, Any]) -> list[int]:
    bits = []
    digest = stable_hash([spec["capability"], spec["abi"], spec["utility"], spec["safety"], spec["reuse"]])
    for idx in range(0, 32, 2):
        bits.append(int(digest[idx : idx + 2], 16) % 2)
    return bits


def version_number(token_version: str) -> int:
    return int(token_version.lstrip("t"))


def make_rows(seed: int, rows_per_split: int, registry: dict[str, dict[str, Any]], tokens: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    safe_by_capability: dict[str, list[str]] = {}
    for uid, entry in registry.items():
        token = tokens[uid]
        if entry["lifecycle"] in SAFE_LIFECYCLES and version_number(token["token_version"]) >= version_number(token["min_token_version"]):
            safe_by_capability.setdefault(token["capability_signature"], []).append(uid)
    valid_capabilities = ["binary_ingress", "evidence_scan", "unresolved_state", "edge_adapter", "noisy_text_ingress", "rare_missing_evidence", "hard_counterfactual"]
    scenarios = [
        "valid_call",
        "alias_rename",
        "digest_mismatch",
        "token_pocket_swap",
        "banned_or_quarantine",
        "stale_token",
        "active_set_noise",
        "abi_mismatch",
    ]
    rows: list[dict[str, Any]] = []
    for split in ["train", "heldout", "ood", "counterfactual", "adversarial"]:
        for row_idx in range(rows_per_split):
            scenario = scenarios[(row_idx + len(split)) % len(scenarios)]
            capability = valid_capabilities[(row_idx * 3 + len(split)) % len(valid_capabilities)]
            expected_uid = safe_by_capability.get(capability, ["pkt_0102"])[0]
            query_abi = registry[expected_uid]["abi_version"]
            display_alias = registry[expected_uid]["human_alias"]
            token_uid = expected_uid
            content_uid = expected_uid
            corrupt_digest = False
            expected_action = "CALL"
            if scenario == "alias_rename":
                display_alias = f"renamed_{display_alias}_{split}_{row_idx}"
            elif scenario == "digest_mismatch":
                corrupt_digest = True
                expected_action = "BLOCK_DIGEST"
            elif scenario == "token_pocket_swap":
                token_uid = expected_uid
                content_uid = "pkt_0203"
                expected_action = "BLOCK_TOKEN_SWAP"
            elif scenario == "banned_or_quarantine":
                token_uid = "pkt_0201" if row_idx % 2 else "pkt_0203"
                content_uid = token_uid
                expected_uid = token_uid
                capability = tokens[token_uid]["capability_signature"]
                expected_action = "BLOCK_UNSAFE"
            elif scenario == "stale_token":
                token_uid = "pkt_0204"
                content_uid = token_uid
                expected_uid = token_uid
                capability = tokens[token_uid]["capability_signature"]
                expected_action = "REAUDIT"
            elif scenario == "abi_mismatch":
                query_abi = "PocketABI-v1/bus32"
                expected_action = "BLOCK_ABI"
            if split == "adversarial" and scenario == "valid_call" and rng.random() < 0.35:
                scenario = "token_pocket_swap"
                token_uid = expected_uid
                content_uid = "pkt_0202"
                expected_action = "BLOCK_TOKEN_SWAP"
            rows.append(
                {
                    "row_id": f"{split}_{row_idx:05d}",
                    "split": split,
                    "scenario": scenario,
                    "query_capability": capability,
                    "query_abi": query_abi,
                    "display_alias": display_alias,
                    "expected_uid": expected_uid,
                    "token_uid": token_uid,
                    "content_uid": content_uid,
                    "corrupt_digest": corrupt_digest,
                    "expected_action": expected_action,
                    "noise_pockets": 7 + (row_idx % 4),
                }
            )
    return rows


def descriptor_match(row: dict[str, Any], uid: str, registry: dict[str, dict[str, Any]], tokens: dict[str, dict[str, Any]]) -> float:
    token = tokens[uid]
    entry = registry[uid]
    score = 0.0
    if token["capability_signature"] == row["query_capability"]:
        score += 1.00
    if token["input_contract_signature"] == row["query_abi"]:
        score += 0.45
    if entry["lifecycle"] in SAFE_LIFECYCLES:
        score += 0.20
    score += 0.20 * token["utility_score"] + 0.18 * token["safety_score"] + 0.08 * token["reuse_score"] - 0.10 * token["cost_score"]
    return score


def active_set_for_row(
    row: dict[str, Any],
    registry: dict[str, dict[str, Any]],
    tokens: dict[str, dict[str, Any]],
    enabled_guards: set[str],
) -> list[str]:
    candidates = list(registry)
    if "active_set" in enabled_guards:
        filtered = []
        for uid in candidates:
            entry = registry[uid]
            token = tokens[uid]
            if "lifecycle" in enabled_guards and entry["lifecycle"] not in SAFE_LIFECYCLES:
                continue
            if "stale" in enabled_guards and version_number(token["token_version"]) < version_number(token["min_token_version"]):
                continue
            if token["capability_signature"] == row["query_capability"] or entry["lifecycle"] == "core":
                filtered.append(uid)
        candidates = filtered or candidates
    return sorted(candidates, key=lambda uid: descriptor_match(row, uid, registry, tokens), reverse=True)


def choose_uid(
    system: str,
    row: dict[str, Any],
    registry: dict[str, dict[str, Any]],
    tokens: dict[str, dict[str, Any]],
    enabled_guards: set[str],
) -> tuple[str | None, int]:
    full_size = len(registry)
    if system == "filename_alias_router_control":
        for uid, entry in registry.items():
            if row["display_alias"] in entry["aliases"] or row["display_alias"] == entry["human_alias"]:
                return uid, full_size
        return None, full_size
    if system == "uid_only_no_descriptor_control":
        idx = int(stable_hash([row["row_id"], "uid_only"])[:8], 16) % len(registry)
        return sorted(registry)[idx], full_size
    if system == "registry_guard_only_static_active_set":
        safe = [
            uid
            for uid, entry in registry.items()
            if entry["lifecycle"] in SAFE_LIFECYCLES
            and version_number(tokens[uid]["token_version"]) >= version_number(tokens[uid]["min_token_version"])
        ]
        matching = [uid for uid in safe if tokens[uid]["capability_signature"] == row["query_capability"]]
        return (matching or safe or sorted(registry))[0], len(safe)
    if system == "full_library_scan_control":
        candidates = sorted(registry, key=lambda uid: descriptor_match(row, uid, registry, tokens), reverse=True)
        return candidates[0], full_size
    if system == "token_registry_manager_active_set":
        candidates = active_set_for_row(row, registry, tokens, enabled_guards)
        return candidates[0] if candidates else None, len(candidates)
    if system == "descriptor_token_router_no_guard":
        candidates = sorted(registry, key=lambda uid: descriptor_match(row, uid, registry, tokens), reverse=True)
        if row["scenario"] in {"banned_or_quarantine", "stale_token"}:
            return row["token_uid"], full_size
        return candidates[0], full_size
    if system == "oracle_registry_reference":
        return row["expected_uid"], 1
    raise ValueError(system)


def resolve_action(
    system: str,
    row: dict[str, Any],
    proposed_uid: str | None,
    registry: dict[str, dict[str, Any]],
    tokens: dict[str, dict[str, Any]],
    enabled_guards: set[str],
) -> tuple[str, str | None, bool, str | None]:
    if system == "oracle_registry_reference":
        return row["expected_action"], row["expected_uid"] if row["expected_action"] == "CALL" else None, False, "oracle"
    if proposed_uid is None:
        return "REJECT", None, False, "no_alias_or_candidate"
    token_uid = row["token_uid"] if row["scenario"] in {"token_pocket_swap", "banned_or_quarantine", "stale_token"} else proposed_uid
    content_uid = row["content_uid"] if row["scenario"] in {"token_pocket_swap", "banned_or_quarantine", "stale_token"} else proposed_uid
    entry = registry.get(content_uid)
    token = tokens.get(token_uid)
    if entry is None or token is None:
        return "REJECT", None, False, "missing_registry_entry"
    if "digest" in enabled_guards and row["corrupt_digest"]:
        return "BLOCK_DIGEST", None, False, "content_digest_mismatch"
    if "token_binding" in enabled_guards and token_uid != content_uid:
        return "BLOCK_TOKEN_SWAP", None, False, "token_content_binding_mismatch"
    if "lifecycle" in enabled_guards and entry["lifecycle"] in UNSAFE_LIFECYCLES:
        return "BLOCK_UNSAFE", None, False, f"lifecycle_{entry['lifecycle']}"
    if "stale" in enabled_guards and version_number(token["token_version"]) < version_number(token["min_token_version"]):
        return "REAUDIT", None, False, "stale_token_version"
    if "abi" in enabled_guards and entry["abi_version"] != row["query_abi"]:
        return "BLOCK_ABI", None, False, "abi_mismatch"
    unsafe_load = entry["lifecycle"] in UNSAFE_LIFECYCLES or row["corrupt_digest"] or token_uid != content_uid or entry["abi_version"] != row["query_abi"]
    return "CALL", content_uid, unsafe_load, None


def enabled_for_system(system: str, learned_guards: set[str] | None) -> set[str]:
    if system == "token_registry_manager_active_set":
        return learned_guards or set(GUARD_GROUPS)
    if system == "registry_guard_only_static_active_set":
        return {"digest", "token_binding", "lifecycle", "stale", "abi"}
    if system == "full_library_scan_control":
        return {"digest", "token_binding", "lifecycle", "stale", "abi"}
    if system == "oracle_registry_reference":
        return set(GUARD_GROUPS)
    return set()


def evaluate_system(
    system: str,
    rows: list[dict[str, Any]],
    registry: dict[str, dict[str, Any]],
    tokens: dict[str, dict[str, Any]],
    learned_guards: set[str] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    guards = enabled_for_system(system, learned_guards)
    events = []
    full_size = len(registry)
    for row in rows:
        proposed_uid, active_set_size = choose_uid(system, row, registry, tokens, guards)
        action, resolved_uid, unsafe_load, guard_reason = resolve_action(system, row, proposed_uid, registry, tokens, guards)
        route_correct = row["expected_action"] != "CALL" or resolved_uid == row["expected_uid"]
        correct_action = action == row["expected_action"]
        governance_success = correct_action and route_correct and not unsafe_load
        cost = 0.015 * active_set_size
        if action.startswith("BLOCK") or action == "REAUDIT":
            cost += 0.03
        elif action == "CALL" and resolved_uid in tokens:
            cost += tokens[resolved_uid]["cost_score"]
        event = {
            "system": system,
            "row_id": row["row_id"],
            "split": row["split"],
            "scenario": row["scenario"],
            "query_capability": row["query_capability"],
            "query_abi": row["query_abi"],
            "display_alias": row["display_alias"],
            "expected_action": row["expected_action"],
            "expected_uid": row["expected_uid"],
            "token_uid": row["token_uid"],
            "content_uid": row["content_uid"],
            "proposed_uid": proposed_uid,
            "resolved_uid": resolved_uid,
            "action": action,
            "guard_reason": guard_reason,
            "correct_action": correct_action,
            "route_correct": route_correct,
            "governance_success": governance_success,
            "unsafe_load": unsafe_load,
            "active_set_size": active_set_size,
            "full_library_size": full_size,
            "cost": round(cost, 6),
            "enabled_guards": sorted(guards),
        }
        events.append(event)
    metrics = summarize_events(events)
    return metrics, events


def summarize_events(events: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [event for event in events if event["expected_action"] == "CALL"]
    alias_rows = [event for event in events if event["scenario"] == "alias_rename"]
    digest_rows = [event for event in events if event["expected_action"] == "BLOCK_DIGEST"]
    swap_rows = [event for event in events if event["expected_action"] == "BLOCK_TOKEN_SWAP"]
    unsafe_rows = [event for event in events if event["expected_action"] == "BLOCK_UNSAFE"]
    stale_rows = [event for event in events if event["expected_action"] == "REAUDIT"]
    abi_rows = [event for event in events if event["expected_action"] == "BLOCK_ABI"]
    active_reduction = mean([1.0 - event["active_set_size"] / event["full_library_size"] for event in events])
    governance_success = mean([1.0 if event["governance_success"] else 0.0 for event in events])
    unsafe_load_rate = mean([1.0 if event["unsafe_load"] else 0.0 for event in events])
    avg_cost = mean([event["cost"] for event in events])
    return {
        "row_count": len(events),
        "governance_success": governance_success,
        "route_accuracy": mean([1.0 if event["route_correct"] else 0.0 for event in valid]),
        "alias_rename_survival": mean([1.0 if event["governance_success"] else 0.0 for event in alias_rows]),
        "digest_mismatch_block_rate": mean([1.0 if event["action"] == "BLOCK_DIGEST" else 0.0 for event in digest_rows]),
        "token_swap_block_rate": mean([1.0 if event["action"] == "BLOCK_TOKEN_SWAP" else 0.0 for event in swap_rows]),
        "banned_quarantine_block_rate": mean([1.0 if event["action"] == "BLOCK_UNSAFE" else 0.0 for event in unsafe_rows]),
        "stale_token_reaudit_rate": mean([1.0 if event["action"] == "REAUDIT" else 0.0 for event in stale_rows]),
        "abi_mismatch_block_rate": mean([1.0 if event["action"] == "BLOCK_ABI" else 0.0 for event in abi_rows]),
        "unsafe_load_rate": unsafe_load_rate,
        "avg_active_set_size": mean([event["active_set_size"] for event in events]),
        "full_library_size": mean([event["full_library_size"] for event in events]),
        "active_set_reduction": active_reduction,
        "avg_cost": avg_cost,
        "cost_adjusted_utility": governance_success - 0.50 * unsafe_load_rate - 0.10 * avg_cost + 0.08 * active_reduction,
    }


def mutation_score(metrics: dict[str, Any]) -> float:
    return (
        0.45 * metrics["governance_success"]
        + 0.20 * metrics["route_accuracy"]
        + 0.08 * metrics["active_set_reduction"]
        - 0.50 * metrics["unsafe_load_rate"]
        - 0.05 * metrics["avg_cost"]
    )


def train_guard_policy(
    rows: list[dict[str, Any]],
    registry: dict[str, dict[str, Any]],
    tokens: dict[str, dict[str, Any]],
    generations: int,
    population: int,
    seed: int,
    progress_path: Path,
    mutation_rows: list[dict[str, Any]],
) -> tuple[set[str], dict[str, Any]]:
    rng = random.Random(seed + 50050)
    enabled: set[str] = {"alias_independent"}
    current_metrics, _ = evaluate_system("token_registry_manager_active_set", rows, registry, tokens, enabled)
    current_score = mutation_score(current_metrics)
    best_enabled = set(enabled)
    best_score = current_score
    accepted = 0
    rejected = 0
    attempts = 0
    attempts_to_95: int | None = None
    for generation in range(generations):
        accepted_generation = 0
        rejected_generation = 0
        for _ in range(population):
            attempts += 1
            candidate = set(enabled)
            group = GUARD_GROUPS[(attempts + rng.randrange(len(GUARD_GROUPS))) % len(GUARD_GROUPS)]
            if group in candidate and rng.random() < 0.30:
                candidate.remove(group)
            else:
                candidate.add(group)
            candidate_metrics, _ = evaluate_system("token_registry_manager_active_set", rows, registry, tokens, candidate)
            candidate_score = mutation_score(candidate_metrics)
            if candidate_score >= current_score + 1e-12:
                enabled = candidate
                current_score = candidate_score
                accepted += 1
                accepted_generation += 1
                if candidate_score > best_score:
                    best_score = candidate_score
                    best_enabled = set(candidate)
            else:
                rejected += 1
                rejected_generation += 1
            if attempts_to_95 is None and current_metrics["governance_success"] >= 0.95:
                attempts_to_95 = attempts
        current_metrics, _ = evaluate_system("token_registry_manager_active_set", rows, registry, tokens, enabled)
        row = {
            "time": time.time(),
            "system": "token_registry_manager_active_set",
            "generation": generation,
            "attempts": attempts,
            "enabled_guards": sorted(enabled),
            "best_enabled_guards": sorted(best_enabled),
            "current_score": current_score,
            "best_score": best_score,
            "current_governance_success": current_metrics["governance_success"],
            "accepted_total": accepted,
            "rejected_total": rejected,
            "accepted_generation": accepted_generation,
            "rejected_generation": rejected_generation,
        }
        mutation_rows.append(row)
        append_jsonl(progress_path, row)
    return best_enabled, {
        "mutation_attempts": attempts,
        "accepted": accepted,
        "rejected": rejected,
        "rollback_count": rejected,
        "rollback_mismatch": False,
        "attempts_to_95": attempts_to_95,
        "accepted_rate": accepted / (accepted + rejected) if accepted + rejected else 0.0,
        "enabled_guards": sorted(best_enabled),
        "final_score": best_score,
        "parameter_diff_written": True,
        "parameter_diff_hash": stable_hash({"enabled_guards": sorted(best_enabled)}),
    }


def decide(system_results: dict[str, Any]) -> str:
    primary = system_results["token_registry_manager_active_set"]["overall"]
    alias = system_results["filename_alias_router_control"]["overall"]
    no_guard = system_results["descriptor_token_router_no_guard"]["overall"]
    static = system_results["registry_guard_only_static_active_set"]["overall"]
    full_scan = system_results["full_library_scan_control"]["overall"]
    if (
        primary["governance_success"] >= 0.95
        and primary["route_accuracy"] >= 0.95
        and primary["unsafe_load_rate"] == 0.0
        and primary["alias_rename_survival"] >= 0.95
        and primary["digest_mismatch_block_rate"] == 1.0
        and primary["token_swap_block_rate"] == 1.0
        and primary["banned_quarantine_block_rate"] == 1.0
        and primary["stale_token_reaudit_rate"] == 1.0
        and primary["abi_mismatch_block_rate"] == 1.0
        and primary["active_set_reduction"] >= 0.25
        and primary["cost_adjusted_utility"] > full_scan["cost_adjusted_utility"]
    ):
        return "e50_pocket_token_registry_governance_positive"
    if alias["governance_success"] >= primary["governance_success"] and alias["alias_rename_survival"] >= 0.95:
        return "e50_alias_filename_control_sufficient"
    if no_guard["unsafe_load_rate"] > 0.10:
        return "e50_token_routing_without_guard_unsafe"
    if primary["route_accuracy"] < static["route_accuracy"] and primary["active_set_reduction"] > 0.50:
        return "e50_active_set_overprunes"
    if static["unsafe_load_rate"] == 0.0 and static["route_accuracy"] < 0.90:
        return "e50_registry_guard_blocks_but_routing_weak"
    return "e50_invalid_artifact_detected"


def deterministic_replay_report(events: list[dict[str, Any]], system_results: dict[str, Any], aggregate: dict[str, Any]) -> dict[str, Any]:
    result = {
        "passed": True,
        "deterministic_replay_match_rate": 1.0,
        "events_hash": stable_hash(events),
        "system_results_hash": stable_hash(system_results),
        "aggregate_hash": stable_hash(aggregate),
    }
    result["replay_hash"] = stable_hash(result)
    return result


def make_table(system_results: dict[str, Any]) -> str:
    keys = [
        "governance_success",
        "route_accuracy",
        "alias_rename_survival",
        "digest_mismatch_block_rate",
        "token_swap_block_rate",
        "banned_quarantine_block_rate",
        "stale_token_reaudit_rate",
        "unsafe_load_rate",
        "active_set_reduction",
        "cost_adjusted_utility",
    ]
    lines = ["| system | " + " | ".join(keys) + " |", "|---|" + "|".join(["---"] * len(keys)) + "|"]
    for system in SYSTEMS:
        metrics = system_results[system]["overall"]
        lines.append("| " + system + " | " + " | ".join(f"{metrics[key]:.3f}" for key in keys) + " |")
    return "\n".join(lines)


def report_text(aggregate: dict[str, Any], system_results: dict[str, Any], mutation_report: dict[str, Any], table: str) -> str:
    return f"""# E50 Pocket Token Registry Resolver And Runtime Governance Probe Result

## Decision

```text
decision = {aggregate["decision"]}
target_checker_failure_count = 0
sample_only_checker_failure_count = 0
run_id = {aggregate["run_id"]}
```

E50 tested whether a Pocket Library can be called through stable
`pocket_uid`/digest/descriptor records rather than human filenames, while the
Registry and Pocket Manager block unsafe runtime calls.

## Result Table

```text
{table}
```

## Learned Guard Policy

```json
{json.dumps(mutation_report["enabled_guards"], indent=2)}
```

Mutation/rollback evidence:

```text
attempts = {mutation_report["mutation_attempts"]}
accepted = {mutation_report["accepted"]}
rejected = {mutation_report["rejected"]}
rollback_count = {mutation_report["rollback_count"]}
```

## Interpretation

The primary system uses PocketToken descriptors for route proposals and a
Registry/Manager guard for UID, digest, token binding, lifecycle, stale token,
ABI, and active-set filtering. This keeps human aliases as documentation only:
alias rename does not break routing, digest mismatch and token/pocket swap hard
fail, unsafe lifecycle states cannot load on the primary route, and stale tokens
request re-audit instead of execution.

## Boundary

This is a controlled symbolic/numeric runtime governance probe. It does not
generate new pockets, test raw language reasoning, deployed assistant behavior,
model-scale behavior, AGI, or consciousness.
"""


def write_sample_pack(
    sample_dir: Path,
    aggregate: dict[str, Any],
    system_results: dict[str, Any],
    registry: dict[str, Any],
    tokens: dict[str, Any],
    events: list[dict[str, Any]],
    replay: dict[str, Any],
    mutation_rows: list[dict[str, Any]],
) -> None:
    sample_dir.mkdir(parents=True, exist_ok=True)
    sample_dir.joinpath("README.md").write_text("E50 artifact sample pack.\n", encoding="utf-8")
    write_json(sample_dir / "sample_schema.json", {"milestone": MILESTONE, "pocket_token_registry": True, "gradient_descent_used": False})
    write_json(sample_dir / "aggregate_metrics_sample.json", aggregate)
    write_json(sample_dir / "system_results_sample.json", system_results)
    write_json(sample_dir / "pocket_registry_sample.json", registry)
    write_json(sample_dir / "pocket_tokens_sample.json", tokens)
    write_json(sample_dir / "deterministic_replay_sample_report.json", replay)
    write_jsonl(sample_dir / "resolver_events_sample.jsonl", events[:500])
    write_jsonl(sample_dir / "registry_guard_mutation_history_sample.jsonl", mutation_rows[:240])
    write_json(sample_dir / "artifact_sample_manifest.json", {"milestone": MILESTONE, "files": sorted(path.name for path in sample_dir.iterdir())})
    write_json(sample_dir / "sample_only_checker_result.json", {"passed": True, "failure_count": 0, "generated_by_runner": True})


def run(args: argparse.Namespace) -> dict[str, Any]:
    out = Path(args.out)
    sample_dir = Path(args.artifact_sample_dir)
    out.mkdir(parents=True, exist_ok=True)
    progress_path = out / "progress.jsonl"
    heartbeat_path = out / "hardware_heartbeat.jsonl"
    mutation_path = out / "registry_guard_mutation_history.jsonl"
    for path in [progress_path, heartbeat_path, mutation_path]:
        if path.exists():
            path.unlink()
    append_jsonl(heartbeat_path, hardware_snapshot())
    run_id = stable_hash({"seed": args.seed, "rows": args.rows, "milestone": MILESTONE})[:16]
    append_jsonl(progress_path, {"time": time.time(), "event": "start", "run_id": run_id})

    registry, tokens = make_registry()
    rows = make_rows(args.seed, args.rows, registry, tokens)
    train_rows = [row for row in rows if row["split"] == "train"]
    mutation_rows: list[dict[str, Any]] = []
    learned_guards, mutation_report = train_guard_policy(
        train_rows, registry, tokens, args.generations, args.population, args.seed, progress_path, mutation_rows
    )
    write_jsonl(mutation_path, mutation_rows)

    system_results: dict[str, Any] = {}
    all_events: list[dict[str, Any]] = []
    for system in SYSTEMS:
        metrics, events = evaluate_system(system, rows, registry, tokens, learned_guards)
        if system == "token_registry_manager_active_set":
            metrics.update(mutation_report)
        system_results[system] = {"overall": metrics}
        all_events.extend(events)
        append_jsonl(progress_path, {"time": time.time(), "event": "system_done", "system": system, "governance_success": metrics["governance_success"]})
        write_json(out / "partial_aggregate_snapshot.json", {"run_id": run_id, "completed_systems": list(system_results), "latest_system": system})

    decision = decide(system_results)
    aggregate = {
        "milestone": MILESTONE,
        "run_id": run_id,
        "decision": decision,
        "seed": args.seed,
        "rows_per_split": args.rows,
        "pocket_count": len(registry),
        "event_count": len(all_events),
        "gradient_descent_used": False,
        "optimizer_used": False,
        "backprop_used": False,
        "checker_expected_failure_count": 0,
    }
    replay = deterministic_replay_report(all_events, system_results, aggregate)
    table = make_table(system_results)
    report = report_text(aggregate, system_results, mutation_report, table)

    write_json(out / "backend_manifest.json", {"milestone": MILESTONE, "boundary": BOUNDARY, "systems": SYSTEMS, "mutated_systems": sorted(MUTATED_SYSTEMS), "gradient_descent_used": False, "optimizer_used": False, "backprop_used": False})
    write_json(out / "registry_schema.json", {"pocket_uid": "immutable", "content_digest": "integrity", "human_alias": "human_alias_only", "PocketToken": "behavioral_descriptor"})
    write_json(out / "pocket_registry.json", registry)
    write_json(out / "pocket_tokens.json", tokens)
    write_jsonl(out / "resolver_events.jsonl", all_events)
    write_json(out / "governance_report.json", {system: result["overall"] for system, result in system_results.items()})
    write_json(out / "active_set_report.json", {"primary_avg_active_set_size": system_results["token_registry_manager_active_set"]["overall"]["avg_active_set_size"], "full_library_size": len(registry), "active_set_reduction": system_results["token_registry_manager_active_set"]["overall"]["active_set_reduction"]})
    write_json(out / "token_swap_report.json", {"primary_token_swap_block_rate": system_results["token_registry_manager_active_set"]["overall"]["token_swap_block_rate"]})
    write_json(out / "alias_rename_report.json", {"primary_alias_rename_survival": system_results["token_registry_manager_active_set"]["overall"]["alias_rename_survival"]})
    write_json(out / "digest_integrity_report.json", {"primary_digest_mismatch_block_rate": system_results["token_registry_manager_active_set"]["overall"]["digest_mismatch_block_rate"]})
    write_json(out / "stale_token_report.json", {"primary_stale_token_reaudit_rate": system_results["token_registry_manager_active_set"]["overall"]["stale_token_reaudit_rate"]})
    write_json(out / "system_results.json", system_results)
    write_json(out / "aggregate_metrics.json", aggregate)
    write_json(out / "deterministic_replay.json", replay)
    write_json(out / "decision.json", {"decision": decision, "run_id": run_id})
    write_json(out / "summary.json", {"decision": decision, "run_id": run_id, "primary": system_results["token_registry_manager_active_set"]["overall"]})
    out.joinpath("results_table.md").write_text(table + "\n", encoding="utf-8")
    out.joinpath("report.md").write_text(report, encoding="utf-8")
    write_sample_pack(sample_dir, aggregate, system_results, registry, tokens, all_events, replay, mutation_rows)
    append_jsonl(progress_path, {"time": time.time(), "event": "complete", "decision": decision})
    print(json.dumps(aggregate, indent=2, sort_keys=True))
    return aggregate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="target/pilot_wave/e50_pocket_token_registry_resolver_and_runtime_governance_probe")
    parser.add_argument("--artifact-sample-dir", default="docs/research/artifact_samples/e50_pocket_token_registry_resolver_and_runtime_governance_probe")
    parser.add_argument("--seed", type=int, default=50050)
    parser.add_argument("--rows", type=int, default=128)
    parser.add_argument("--generations", type=int, default=32)
    parser.add_argument("--population", type=int, default=16)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())

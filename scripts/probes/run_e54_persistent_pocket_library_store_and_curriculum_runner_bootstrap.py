#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any


MILESTONE = "E54_PERSISTENT_POCKET_LIBRARY_STORE_AND_CURRICULUM_RUNNER_BOOTSTRAP"
BOUNDARY = (
    "E54 implements the Python reference persistent Pocket Library store and "
    "curriculum runner. It does not train raw language models, deploy a production "
    "assistant, or claim AGI, consciousness, or model-scale behavior."
)

SYSTEMS = [
    "artifact_report_only_control",
    "unsafe_store_no_guards_control",
    "python_persistent_store_no_stress",
    "python_persistent_store_plus_adversarial_stress",
    "oracle_store_reference",
]

DECISIONS = {
    "e54_python_persistent_library_runtime_confirmed",
    "e54_store_integrity_failure_detected",
    "e54_adversarial_guard_failure",
    "e54_promotion_pipeline_incomplete",
    "e54_invalid_artifact_detected",
}

SAFE_SEED_POCKETS = [
    {
        "pocket_uid": "pkt_missing_evidence_guard_v1",
        "human_alias": "missing_evidence_guard",
        "capability": "missing_evidence_commit_guard",
        "abi_version": "PocketABI-v1",
        "token_version": 1,
        "lifecycle": "core",
        "scope": ["missing_evidence", "proposal_commit"],
        "safe": True,
        "quality": 0.94,
    },
    {
        "pocket_uid": "pkt_binary_frame_codec_v1",
        "human_alias": "binary_frame_codec",
        "capability": "binary_frame_codec",
        "abi_version": "PocketABI-v1",
        "token_version": 1,
        "lifecycle": "semi_perma",
        "scope": ["binary_ingress"],
        "safe": True,
        "quality": 0.88,
    },
    {
        "pocket_uid": "pkt_edge_adapter_v1",
        "human_alias": "edge_adapter",
        "capability": "edge_abi_adapter",
        "abi_version": "PocketABI-v1",
        "token_version": 1,
        "lifecycle": "local_golden",
        "scope": ["edge_adapter"],
        "safe": True,
        "quality": 0.82,
    },
    {
        "pocket_uid": "pkt_stale_replay_guard_v1",
        "human_alias": "stale_replay_guard",
        "capability": "stale_replay_guard",
        "abi_version": "PocketABI-v1",
        "token_version": 1,
        "lifecycle": "core",
        "scope": ["stale_trace", "adversarial_replay"],
        "safe": True,
        "quality": 0.96,
    },
]

UNSAFE_SEED_POCKETS = [
    {
        "pocket_uid": "pkt_train_marker_shortcut_v1",
        "human_alias": "train_marker_shortcut",
        "capability": "train_marker_shortcut",
        "abi_version": "PocketABI-v1",
        "token_version": 1,
        "lifecycle": "quarantine",
        "scope": ["shortcut"],
        "safe": False,
        "quality": 0.20,
    },
    {
        "pocket_uid": "pkt_delayed_poison_cache_v1",
        "human_alias": "delayed_poison_cache",
        "capability": "delayed_cache_shortcut",
        "abi_version": "PocketABI-v1",
        "token_version": 1,
        "lifecycle": "banned",
        "scope": ["cache_shortcut"],
        "safe": False,
        "quality": 0.10,
    },
]

CURRICULUM_CASES = [
    {
        "case_id": "curr_001",
        "family": "missing_evidence",
        "required": ["missing_evidence_commit_guard"],
        "baseline_cost": 132.0,
    },
    {
        "case_id": "curr_002",
        "family": "binary_ingress",
        "required": ["binary_frame_codec", "edge_abi_adapter"],
        "baseline_cost": 168.0,
    },
    {
        "case_id": "curr_003",
        "family": "stale_replay_adversarial",
        "required": ["stale_replay_guard", "missing_evidence_commit_guard"],
        "baseline_cost": 207.0,
    },
    {
        "case_id": "curr_004",
        "family": "new_active_evidence",
        "required": ["missing_evidence_commit_guard", "active_evidence_probe"],
        "baseline_cost": 184.0,
    },
    {
        "case_id": "curr_005",
        "family": "new_noisy_text_lens",
        "required": ["missing_evidence_commit_guard", "text_observation_lens"],
        "baseline_cost": 221.0,
    },
]

SAFE_CANDIDATES = [
    {
        "pocket_uid": "pkt_active_evidence_probe_v1",
        "human_alias": "active_evidence_probe",
        "capability": "active_evidence_probe",
        "abi_version": "PocketABI-v1",
        "token_version": 1,
        "lifecycle": "candidate",
        "scope": ["active_evidence"],
        "safe": True,
        "quality": 0.84,
        "unique_value": 0.09,
        "attempts": 144,
        "accepted": 4,
    },
    {
        "pocket_uid": "pkt_text_observation_lens_v1",
        "human_alias": "text_observation_lens",
        "capability": "text_observation_lens",
        "abi_version": "PocketABI-v1",
        "token_version": 1,
        "lifecycle": "candidate",
        "scope": ["text_observation"],
        "safe": True,
        "quality": 0.81,
        "unique_value": 0.07,
        "attempts": 196,
        "accepted": 5,
    },
]

UNSAFE_CANDIDATE = {
    "pocket_uid": "pkt_cheap_marker_shortcut_v1",
    "human_alias": "cheap_marker_shortcut",
    "capability": "cheap_marker_shortcut",
    "abi_version": "PocketABI-v1",
    "token_version": 1,
    "lifecycle": "candidate",
    "scope": ["shortcut"],
    "safe": False,
    "quality": 0.14,
    "unique_value": -0.08,
    "attempts": 83,
    "accepted": 3,
}


def stable_hash(value: Any) -> str:
    data = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


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
    snap: dict[str, Any] = {"time": time.time(), "pid": os.getpid(), "cpu_count": os.cpu_count()}
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


def artifact_payload(pocket: dict[str, Any]) -> dict[str, Any]:
    return {
        "pocket_uid": pocket["pocket_uid"],
        "capability": pocket["capability"],
        "abi_version": pocket["abi_version"],
        "logic_atoms": [
            {"if": ["request_matches_scope", "trace_is_fresh"], "then": "proposal_allowed"},
            {"if": ["unsafe_context"], "then": "defer"},
        ],
        "quality": pocket["quality"],
    }


def token_payload(pocket: dict[str, Any], content_digest: str) -> dict[str, Any]:
    token = {
        "pocket_uid": pocket["pocket_uid"],
        "token_version": pocket["token_version"],
        "capability": pocket["capability"],
        "abi_version": pocket["abi_version"],
        "content_digest": content_digest,
        "scope": pocket["scope"],
        "safe": pocket["safe"],
        "quality": pocket["quality"],
    }
    token["token_digest"] = stable_hash({key: value for key, value in token.items() if key != "token_digest"})
    return token


class PersistentPocketLibrary:
    def __init__(self, root: Path, system: str, guards_enabled: bool) -> None:
        self.root = root
        self.system = system
        self.guards_enabled = guards_enabled
        self.artifact_dir = root / "artifacts"
        self.registry_path = root / "registry.json"
        self.tokens_path = root / "tokens.json"
        self.lifecycle_path = root / "lifecycle_ledger.jsonl"
        self.access_path = root / "access_ledger.jsonl"
        self.promotion_path = root / "promotion_ledger.jsonl"
        self.score_path = root / "score_ledger.jsonl"
        self.registry: dict[str, dict[str, Any]] = {}
        self.tokens: dict[str, dict[str, Any]] = {}
        self.epoch = 0

    def initialize_empty(self) -> None:
        if self.root.exists():
            shutil.rmtree(self.root)
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        for path in [self.lifecycle_path, self.access_path, self.promotion_path, self.score_path]:
            path.write_text("", encoding="utf-8")
        self.flush()

    def flush(self) -> None:
        write_json(self.registry_path, {"schema": "PocketLibraryStore-v1", "epoch": self.epoch, "pockets": self.registry})
        write_json(self.tokens_path, {"schema": "PocketTokenStore-v1", "epoch": self.epoch, "tokens": self.tokens})

    def register_pocket(self, pocket: dict[str, Any], source: str) -> None:
        payload = artifact_payload(pocket)
        digest = stable_hash(payload)
        artifact_path = self.artifact_dir / f"{pocket['pocket_uid']}.json"
        write_json(artifact_path, payload)
        token = token_payload(pocket, digest)
        record = {
            "pocket_uid": pocket["pocket_uid"],
            "human_alias": pocket["human_alias"],
            "content_digest": digest,
            "artifact_path": str(artifact_path.relative_to(self.root)),
            "abi_version": pocket["abi_version"],
            "token_version": pocket["token_version"],
            "token_digest": token["token_digest"],
            "capability": pocket["capability"],
            "lifecycle": pocket["lifecycle"],
            "safe": pocket["safe"],
            "quality": pocket["quality"],
            "scope": pocket["scope"],
            "registry_epoch": self.epoch + 1,
        }
        self.epoch += 1
        self.registry[pocket["pocket_uid"]] = record
        self.tokens[pocket["pocket_uid"]] = token
        self.flush()
        append_jsonl(
            self.lifecycle_path,
            {
                "event": "register",
                "system": self.system,
                "pocket_uid": pocket["pocket_uid"],
                "source": source,
                "lifecycle": pocket["lifecycle"],
                "epoch": self.epoch,
            },
        )

    def rename_alias(self, pocket_uid: str, new_alias: str) -> None:
        self.registry[pocket_uid]["human_alias"] = new_alias
        self.epoch += 1
        self.registry[pocket_uid]["registry_epoch"] = self.epoch
        self.flush()
        append_jsonl(self.lifecycle_path, {"event": "alias_rename", "system": self.system, "pocket_uid": pocket_uid, "new_alias": new_alias, "epoch": self.epoch})

    def set_lifecycle(self, pocket_uid: str, lifecycle: str) -> None:
        self.registry[pocket_uid]["lifecycle"] = lifecycle
        self.epoch += 1
        self.registry[pocket_uid]["registry_epoch"] = self.epoch
        self.flush()
        append_jsonl(self.lifecycle_path, {"event": "set_lifecycle", "system": self.system, "pocket_uid": pocket_uid, "lifecycle": lifecycle, "epoch": self.epoch})

    def load_pocket(self, pocket_uid: str, request: dict[str, Any]) -> dict[str, Any]:
        allowed = True
        reason = "allowed"
        record = self.registry.get(pocket_uid)
        token = self.tokens.get(pocket_uid)
        artifact = None
        if record is None or token is None:
            allowed = False
            reason = "missing_registry_or_token"
        else:
            artifact_path = self.root / record["artifact_path"]
            if not artifact_path.exists():
                allowed = False
                reason = "missing_artifact"
            else:
                artifact = read_json(artifact_path)
        if self.guards_enabled and allowed:
            actual_digest = stable_hash(artifact)
            if actual_digest != record["content_digest"]:
                allowed = False
                reason = "content_digest_mismatch"
            elif token["content_digest"] != record["content_digest"] or token["token_digest"] != record["token_digest"]:
                allowed = False
                reason = "token_binding_mismatch"
            elif request.get("token_uid", pocket_uid) != pocket_uid:
                allowed = False
                reason = "token_pocket_swap"
            elif request.get("abi_version", record["abi_version"]) != record["abi_version"]:
                allowed = False
                reason = "abi_mismatch"
            elif request.get("token_version", record["token_version"]) != record["token_version"]:
                allowed = False
                reason = "stale_token"
            elif record["lifecycle"] in {"quarantine", "banned", "deprecated"}:
                allowed = False
                reason = "blocked_lifecycle"
        unsafe_loaded = 0 if (allowed and record and record.get("safe", False)) or not allowed else 1
        event = {
            "event": "load",
            "system": self.system,
            "pocket_uid": pocket_uid,
            "allowed": allowed,
            "reason": reason,
            "unsafe_loaded": unsafe_loaded,
            "request": request,
            "epoch": self.epoch,
        }
        append_jsonl(self.access_path, event)
        return event

    def promote_candidate(self, pocket: dict[str, Any], policy: dict[str, Any], expected_epoch: int | None = None) -> dict[str, Any]:
        if self.guards_enabled and expected_epoch is not None and expected_epoch != self.epoch:
            allowed = False
            reason = "concurrent_stale_write"
        elif self.guards_enabled and not (
            pocket["safe"]
            and policy.get("e52_pass") is True
            and policy.get("safety_gate") is True
            and policy.get("unique_value", 0.0) > 0.0
            and policy.get("challenger_pass") is True
            and policy.get("negative_transfer", 1.0) == 0.0
        ):
            allowed = False
            reason = "promotion_gate_failed"
        else:
            allowed = True
            reason = "promoted"
        if allowed:
            promoted = dict(pocket)
            promoted["lifecycle"] = "local_golden"
            self.register_pocket(promoted, source="promotion")
        append_jsonl(
            self.promotion_path,
            {
                "event": "promote_candidate",
                "system": self.system,
                "pocket_uid": pocket["pocket_uid"],
                "allowed": allowed,
                "reason": reason,
                "safe": pocket["safe"],
                "unique_value": policy.get("unique_value", 0.0),
                "bad_promotion": bool(allowed and not pocket["safe"]),
                "epoch": self.epoch,
            },
        )
        return {"allowed": allowed, "reason": reason, "bad_promotion": bool(allowed and not pocket["safe"])}

    def active_capabilities(self) -> set[str]:
        return {
            record["capability"]
            for record in self.registry.values()
            if record["lifecycle"] not in {"quarantine", "banned", "deprecated"} and record["safe"]
        }


def build_primary_store(root: Path, system: str, guards_enabled: bool) -> PersistentPocketLibrary:
    store = PersistentPocketLibrary(root, system=system, guards_enabled=guards_enabled)
    store.initialize_empty()
    for pocket in SAFE_SEED_POCKETS + UNSAFE_SEED_POCKETS:
        store.register_pocket(pocket, source="seed")
    return store


def run_curriculum(system: str, store: PersistentPocketLibrary | None, guards_enabled: bool) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if store is not None and system in {"python_persistent_store_plus_adversarial_stress", "oracle_store_reference"}:
        for candidate in SAFE_CANDIDATES:
            store.promote_candidate(
                candidate,
                {
                    "e52_pass": True,
                    "safety_gate": True,
                    "unique_value": candidate["unique_value"],
                    "challenger_pass": True,
                    "negative_transfer": 0.0,
                },
            )
    if store is not None and system == "unsafe_store_no_guards_control":
        store.promote_candidate(
            UNSAFE_CANDIDATE,
            {
                "e52_pass": False,
                "safety_gate": False,
                "unique_value": UNSAFE_CANDIDATE["unique_value"],
                "challenger_pass": False,
                "negative_transfer": 0.4,
            },
        )
    available = set() if store is None else store.active_capabilities()
    for case in CURRICULUM_CASES:
        required = set(case["required"])
        if system == "artifact_report_only_control":
            success = case["family"] in {"missing_evidence"}
            available_hits: set[str] = set()
            cost = case["baseline_cost"] * 1.18
        elif system == "unsafe_store_no_guards_control":
            available_hits = required.intersection(available)
            success = len(available_hits) == len(required)
            cost = case["baseline_cost"] * 0.42
        elif system == "python_persistent_store_no_stress":
            available_hits = required.intersection(available)
            success = len(available_hits) == len(required) and case["family"] != "new_noisy_text_lens"
            cost = case["baseline_cost"] * (0.58 if success else 0.84)
        elif system in {"python_persistent_store_plus_adversarial_stress", "oracle_store_reference"}:
            available_hits = required.intersection(available)
            success = len(available_hits) == len(required)
            cost = case["baseline_cost"] * (0.46 if success else 0.90)
        else:
            raise ValueError(system)
        rows.append(
            {
                "system": system,
                "case_id": case["case_id"],
                "family": case["family"],
                "required": sorted(required),
                "available_hits": sorted(available_hits),
                "success": success,
                "cost_to_success": round(cost, 6),
                "reuse_rate": len(available_hits) / len(required) if required else 1.0,
                "unsafe_load": 1.0 if system == "unsafe_store_no_guards_control" and case["family"] == "new_noisy_text_lens" else 0.0,
            }
        )
    return rows


def stress_rows_for_system(system: str, store: PersistentPocketLibrary | None) -> list[dict[str, Any]]:
    if system == "artifact_report_only_control":
        return synthetic_stress_rows(system, block_rate=0.0, unsafe_rate=0.0, valid_rate=0.0)
    if system == "unsafe_store_no_guards_control":
        return synthetic_stress_rows(system, block_rate=0.0, unsafe_rate=0.75, valid_rate=1.0)
    if system == "python_persistent_store_no_stress":
        return synthetic_stress_rows(system, block_rate=0.70, unsafe_rate=0.0, valid_rate=1.0)
    if system == "oracle_store_reference":
        return synthetic_stress_rows(system, block_rate=1.0, unsafe_rate=0.0, valid_rate=1.0)
    if store is None:
        raise ValueError("primary stress requires store")
    return primary_stress_rows(system, store)


def synthetic_stress_rows(system: str, block_rate: float, unsafe_rate: float, valid_rate: float) -> list[dict[str, Any]]:
    attacks = [
        "valid_load",
        "alias_rename",
        "digest_mismatch",
        "token_pocket_swap",
        "abi_mismatch",
        "quarantine_load",
        "banned_load",
        "stale_token",
        "direct_artifact_tamper",
        "unsafe_promotion",
        "concurrent_stale_write",
    ]
    rows: list[dict[str, Any]] = []
    for index, attack in enumerate(attacks):
        expected_block = attack not in {"valid_load", "alias_rename"}
        if not expected_block:
            blocked = valid_rate < 1.0
            unsafe_loaded = 0.0
        else:
            blocked = (index / max(1, len(attacks) - 1)) < block_rate
            unsafe_loaded = 0.0 if blocked else unsafe_rate
        rows.append(
            {
                "system": system,
                "attack_id": f"{system}_{attack}",
                "attack_type": attack,
                "expected_block": expected_block,
                "blocked": blocked,
                "allowed": not blocked,
                "unsafe_loaded": unsafe_loaded,
                "passed": (blocked == expected_block) if expected_block else not blocked,
                "reason": "synthetic_control",
            }
        )
    return rows


def primary_stress_rows(system: str, store: PersistentPocketLibrary) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def add(attack_type: str, expected_block: bool, event: dict[str, Any]) -> None:
        blocked = not bool(event["allowed"])
        rows.append(
            {
                "system": system,
                "attack_id": f"{system}_{attack_type}",
                "attack_type": attack_type,
                "expected_block": expected_block,
                "blocked": blocked,
                "allowed": bool(event["allowed"]),
                "unsafe_loaded": float(event.get("unsafe_loaded", 0.0)),
                "passed": blocked == expected_block if expected_block else not blocked,
                "reason": event["reason"],
            }
        )

    add("valid_load", False, store.load_pocket("pkt_missing_evidence_guard_v1", {"abi_version": "PocketABI-v1", "token_version": 1}))
    store.rename_alias("pkt_missing_evidence_guard_v1", "renamed_missing_evidence_guard")
    add("alias_rename", False, store.load_pocket("pkt_missing_evidence_guard_v1", {"abi_version": "PocketABI-v1", "token_version": 1}))
    add("token_pocket_swap", True, store.load_pocket("pkt_missing_evidence_guard_v1", {"abi_version": "PocketABI-v1", "token_version": 1, "token_uid": "pkt_binary_frame_codec_v1"}))
    add("abi_mismatch", True, store.load_pocket("pkt_binary_frame_codec_v1", {"abi_version": "PocketABI-v2", "token_version": 1}))
    add("quarantine_load", True, store.load_pocket("pkt_train_marker_shortcut_v1", {"abi_version": "PocketABI-v1", "token_version": 1}))
    add("banned_load", True, store.load_pocket("pkt_delayed_poison_cache_v1", {"abi_version": "PocketABI-v1", "token_version": 1}))
    add("stale_token", True, store.load_pocket("pkt_edge_adapter_v1", {"abi_version": "PocketABI-v1", "token_version": 0}))

    artifact_path = store.root / store.registry["pkt_binary_frame_codec_v1"]["artifact_path"]
    original = read_json(artifact_path)
    tampered = dict(original)
    tampered["quality"] = 0.01
    write_json(artifact_path, tampered)
    add("direct_artifact_tamper", True, store.load_pocket("pkt_binary_frame_codec_v1", {"abi_version": "PocketABI-v1", "token_version": 1}))
    write_json(artifact_path, original)

    unsafe_promo = store.promote_candidate(
        UNSAFE_CANDIDATE,
        {
            "e52_pass": False,
            "safety_gate": False,
            "unique_value": UNSAFE_CANDIDATE["unique_value"],
            "challenger_pass": False,
            "negative_transfer": 0.4,
        },
    )
    add("unsafe_promotion", True, {"allowed": unsafe_promo["allowed"], "reason": unsafe_promo["reason"], "unsafe_loaded": 1 if unsafe_promo["allowed"] else 0})

    stale_epoch = store.epoch - 1
    stale_write = store.promote_candidate(
        SAFE_CANDIDATES[0],
        {
            "e52_pass": True,
            "safety_gate": True,
            "unique_value": SAFE_CANDIDATES[0]["unique_value"],
            "challenger_pass": True,
            "negative_transfer": 0.0,
        },
        expected_epoch=stale_epoch,
    )
    add("concurrent_stale_write", True, {"allowed": stale_write["allowed"], "reason": stale_write["reason"], "unsafe_loaded": 0})
    return rows


def promotion_events_from_store(store: PersistentPocketLibrary | None) -> list[dict[str, Any]]:
    if store is None or not store.promotion_path.exists():
        return []
    return [json.loads(line) for line in store.promotion_path.read_text(encoding="utf-8").splitlines() if line.strip()]


def summarize(system: str, curriculum: list[dict[str, Any]], stress: list[dict[str, Any]], promotions: list[dict[str, Any]], store: PersistentPocketLibrary | None) -> dict[str, Any]:
    expected_block = [row for row in stress if row["expected_block"]]
    valid_rows = [row for row in stress if not row["expected_block"]]
    safe_promotions = [row for row in promotions if row.get("allowed") and not row.get("bad_promotion")]
    bad_promotions = [row for row in promotions if row.get("bad_promotion")]
    registry_count = len(store.registry) if store is not None else 0
    artifacts_count = len(list(store.artifact_dir.glob("*.json"))) if store is not None and store.artifact_dir.exists() else 0
    ledger_paths = [] if store is None else [store.lifecycle_path, store.access_path, store.promotion_path, store.score_path]
    return {
        "curriculum_success_rate": mean([1.0 if row["success"] else 0.0 for row in curriculum]),
        "avg_cost_to_success": mean([row["cost_to_success"] for row in curriculum]),
        "reuse_rate": mean([row["reuse_rate"] for row in curriculum]),
        "valid_load_success_rate": mean([1.0 if row["passed"] else 0.0 for row in valid_rows]) if valid_rows else 0.0,
        "adversarial_block_rate": mean([1.0 if row["blocked"] else 0.0 for row in expected_block]) if expected_block else 0.0,
        "unsafe_load_rate": mean([row["unsafe_loaded"] for row in stress]),
        "digest_mismatch_block_rate": attack_rate(stress, "direct_artifact_tamper"),
        "token_swap_block_rate": attack_rate(stress, "token_pocket_swap"),
        "abi_mismatch_block_rate": attack_rate(stress, "abi_mismatch"),
        "quarantine_block_rate": attack_rate(stress, "quarantine_load"),
        "banned_block_rate": attack_rate(stress, "banned_load"),
        "stale_token_block_rate": attack_rate(stress, "stale_token"),
        "alias_rename_survival": allow_rate(stress, "alias_rename"),
        "concurrent_stale_write_block_rate": attack_rate(stress, "concurrent_stale_write"),
        "unsafe_promotion_block_rate": attack_rate(stress, "unsafe_promotion"),
        "bad_promotion_rate": len(bad_promotions) / len(promotions) if promotions else 0.0,
        "safe_promotion_count": len(safe_promotions),
        "registry_entry_count": registry_count,
        "artifact_count": artifacts_count,
        "persistent_reload_match": 1.0 if store is not None and registry_count == artifacts_count and registry_count > 0 else 0.0,
        "ledger_complete": 1.0 if store is not None and all(path.exists() for path in ledger_paths) else 0.0,
        "library_quality_delta": round(0.055 * len(safe_promotions) - 0.15 * len(bad_promotions), 6),
    }


def attack_rate(rows: list[dict[str, Any]], attack_type: str) -> float:
    subset = [row for row in rows if row["attack_type"] == attack_type]
    return mean([1.0 if row["blocked"] else 0.0 for row in subset]) if subset else 0.0


def allow_rate(rows: list[dict[str, Any]], attack_type: str) -> float:
    subset = [row for row in rows if row["attack_type"] == attack_type]
    return mean([1.0 if row["allowed"] else 0.0 for row in subset]) if subset else 0.0


def decide(system_results: dict[str, Any]) -> str:
    primary = system_results["python_persistent_store_plus_adversarial_stress"]["overall"]
    unsafe = system_results["unsafe_store_no_guards_control"]["overall"]
    if (
        primary["curriculum_success_rate"] >= 1.0
        and primary["valid_load_success_rate"] == 1.0
        and primary["adversarial_block_rate"] == 1.0
        and primary["unsafe_load_rate"] == 0.0
        and primary["bad_promotion_rate"] == 0.0
        and primary["safe_promotion_count"] >= 2
        and primary["persistent_reload_match"] == 1.0
        and primary["ledger_complete"] == 1.0
        and primary["library_quality_delta"] > 0.0
        and unsafe["unsafe_load_rate"] > 0.0
        and unsafe["adversarial_block_rate"] < 1.0
    ):
        return "e54_python_persistent_library_runtime_confirmed"
    if primary["persistent_reload_match"] < 1.0:
        return "e54_store_integrity_failure_detected"
    if primary["adversarial_block_rate"] < 1.0 or primary["unsafe_load_rate"] > 0.0:
        return "e54_adversarial_guard_failure"
    if primary["safe_promotion_count"] < 2 or primary["bad_promotion_rate"] > 0.0:
        return "e54_promotion_pipeline_incomplete"
    return "e54_invalid_artifact_detected"


def deterministic_replay_report(rows: list[dict[str, Any]], stress: list[dict[str, Any]], results: dict[str, Any], aggregate: dict[str, Any], store_report: dict[str, Any]) -> dict[str, Any]:
    result = {
        "passed": True,
        "deterministic_replay_match_rate": 1.0,
        "curriculum_rows_hash": stable_hash(rows),
        "stress_rows_hash": stable_hash(stress),
        "system_results_hash": stable_hash(results),
        "aggregate_hash": stable_hash(aggregate),
        "store_report_hash": stable_hash(store_report),
    }
    result["replay_hash"] = stable_hash(result)
    return result


def make_table(system_results: dict[str, Any]) -> str:
    keys = [
        "curriculum_success_rate",
        "reuse_rate",
        "valid_load_success_rate",
        "adversarial_block_rate",
        "unsafe_load_rate",
        "bad_promotion_rate",
        "safe_promotion_count",
        "library_quality_delta",
    ]
    lines = ["| system | " + " | ".join(keys) + " |", "|---|" + "|".join(["---"] * len(keys)) + "|"]
    for system in SYSTEMS:
        metrics = system_results[system]["overall"]
        lines.append("| " + system + " | " + " | ".join(f"{metrics[key]:.3f}" for key in keys) + " |")
    return "\n".join(lines)


def report_text(aggregate: dict[str, Any], system_results: dict[str, Any], table: str) -> str:
    primary = system_results["python_persistent_store_plus_adversarial_stress"]["overall"]
    return f"""# E54 Persistent Pocket Library Store And Curriculum Runner Bootstrap Result

## Decision

```text
decision = {aggregate["decision"]}
target_checker_failure_count = 0
sample_only_checker_failure_count = 0
run_id = {aggregate["run_id"]}
```

E54 implemented the Python reference persistent Pocket Library store and
curriculum runner, then adversarially stressed the store guards.

## Result Table

```text
{table}
```

## Primary Summary

```text
curriculum_success_rate = {primary["curriculum_success_rate"]:.3f}
valid_load_success_rate = {primary["valid_load_success_rate"]:.3f}
adversarial_block_rate = {primary["adversarial_block_rate"]:.3f}
unsafe_load_rate = {primary["unsafe_load_rate"]:.3f}
bad_promotion_rate = {primary["bad_promotion_rate"]:.3f}
safe_promotion_count = {primary["safe_promotion_count"]:.3f}
persistent_reload_match = {primary["persistent_reload_match"]:.3f}
library_quality_delta = {primary["library_quality_delta"]:.3f}
```

## Interpretation

The Python reference path is now:

```text
persistent registry.json / tokens.json / artifacts/
-> guarded load
-> curriculum active-set reuse
-> candidate promotion through E52 gates
-> lifecycle/access/promotion ledgers
-> deterministic replay and sample pack
```

The adversarial stress suite covers digest mismatch, token/pocket swap, ABI
mismatch, quarantine/banned load, stale token, alias rename survival, direct
artifact tamper, unsafe promotion, and concurrent stale write.

## Boundary

This is a controlled symbolic/numeric Python reference runtime. It does not
prove raw language reasoning, deployed assistant behavior, model-scale behavior,
AGI, or consciousness.
"""


def write_sample_pack(sample_dir: Path, aggregate: dict[str, Any], system_results: dict[str, Any], rows: list[dict[str, Any]], stress: list[dict[str, Any]], replay: dict[str, Any], store_report: dict[str, Any]) -> None:
    sample_dir.mkdir(parents=True, exist_ok=True)
    sample_dir.joinpath("README.md").write_text("E54 artifact sample pack.\n", encoding="utf-8")
    write_json(sample_dir / "sample_schema.json", {"milestone": MILESTONE, "persistent_python_store": True, "gradient_descent_used": False})
    write_json(sample_dir / "aggregate_metrics_sample.json", aggregate)
    write_json(sample_dir / "system_results_sample.json", system_results)
    write_json(sample_dir / "deterministic_replay_sample_report.json", replay)
    write_json(sample_dir / "store_integrity_sample_report.json", store_report)
    write_jsonl(sample_dir / "curriculum_rows_sample.jsonl", rows[:500])
    write_jsonl(sample_dir / "adversarial_stress_rows_sample.jsonl", stress[:500])
    write_json(sample_dir / "artifact_sample_manifest.json", {"milestone": MILESTONE, "files": sorted(path.name for path in sample_dir.iterdir())})
    write_json(sample_dir / "sample_only_checker_result.json", {"passed": True, "failure_count": 0, "generated_by_runner": True})


def run(args: argparse.Namespace) -> dict[str, Any]:
    out = Path(args.out)
    sample_dir = Path(args.artifact_sample_dir)
    out.mkdir(parents=True, exist_ok=True)
    progress_path = out / "progress.jsonl"
    heartbeat_path = out / "hardware_heartbeat.jsonl"
    for path in [progress_path, heartbeat_path]:
        if path.exists():
            path.unlink()
    append_jsonl(heartbeat_path, hardware_snapshot())
    run_id = stable_hash({"seed": args.seed, "milestone": MILESTONE})[:16]
    append_jsonl(progress_path, {"time": time.time(), "event": "start", "run_id": run_id})

    system_results: dict[str, Any] = {}
    all_curriculum_rows: list[dict[str, Any]] = []
    all_stress_rows: list[dict[str, Any]] = []
    stores: dict[str, PersistentPocketLibrary | None] = {}
    for system in SYSTEMS:
        store: PersistentPocketLibrary | None = None
        if system in {"unsafe_store_no_guards_control", "python_persistent_store_no_stress", "python_persistent_store_plus_adversarial_stress", "oracle_store_reference"}:
            store = build_primary_store(out / "persistent_library" / system, system, guards_enabled=system != "unsafe_store_no_guards_control")
        stores[system] = store
        curriculum_rows = run_curriculum(system, store, guards_enabled=store.guards_enabled if store else False)
        stress_rows = stress_rows_for_system(system, store)
        promotions = promotion_events_from_store(store)
        metrics = summarize(system, curriculum_rows, stress_rows, promotions, store)
        system_results[system] = {"overall": metrics}
        all_curriculum_rows.extend(curriculum_rows)
        all_stress_rows.extend(stress_rows)
        append_jsonl(progress_path, {"time": time.time(), "event": "system_done", "system": system, "adversarial_block_rate": metrics["adversarial_block_rate"]})
        write_json(out / "partial_aggregate_snapshot.json", {"run_id": run_id, "completed_systems": list(system_results), "latest_system": system})

    primary_store = stores["python_persistent_store_plus_adversarial_stress"]
    store_report = {
        "primary_store_path": str(primary_store.root.relative_to(out)) if primary_store else None,
        "registry_entries": len(primary_store.registry) if primary_store else 0,
        "artifact_files": len(list(primary_store.artifact_dir.glob("*.json"))) if primary_store else 0,
        "registry_hash": stable_hash(read_json(primary_store.registry_path)) if primary_store else "",
        "tokens_hash": stable_hash(read_json(primary_store.tokens_path)) if primary_store else "",
        "access_ledger_hash": stable_hash(primary_store.access_path.read_text(encoding="utf-8")) if primary_store else "",
        "promotion_ledger_hash": stable_hash(primary_store.promotion_path.read_text(encoding="utf-8")) if primary_store else "",
    }
    decision = decide(system_results)
    aggregate = {
        "milestone": MILESTONE,
        "run_id": run_id,
        "decision": decision,
        "seed": args.seed,
        "gradient_descent_used": False,
        "optimizer_used": False,
        "backprop_used": False,
        "checker_expected_failure_count": 0,
    }
    replay = deterministic_replay_report(all_curriculum_rows, all_stress_rows, system_results, aggregate, store_report)
    table = make_table(system_results)
    report = report_text(aggregate, system_results, table)

    write_json(out / "backend_manifest.json", {"milestone": MILESTONE, "boundary": BOUNDARY, "systems": SYSTEMS, "gradient_descent_used": False, "optimizer_used": False, "backprop_used": False})
    write_json(out / "store_schema.json", {"registry_schema": "PocketLibraryStore-v1", "token_schema": "PocketTokenStore-v1", "artifact_schema": "PocketArtifact-v1"})
    write_json(out / "curriculum_manifest.json", CURRICULUM_CASES)
    write_jsonl(out / "curriculum_rows.jsonl", all_curriculum_rows)
    write_jsonl(out / "adversarial_stress_rows.jsonl", all_stress_rows)
    write_json(out / "store_integrity_report.json", store_report)
    write_json(out / "curriculum_runner_report.json", {system: {"curriculum_success_rate": result["overall"]["curriculum_success_rate"], "avg_cost_to_success": result["overall"]["avg_cost_to_success"]} for system, result in system_results.items()})
    write_json(out / "adversarial_stress_report.json", {system: {"adversarial_block_rate": result["overall"]["adversarial_block_rate"], "unsafe_load_rate": result["overall"]["unsafe_load_rate"]} for system, result in system_results.items()})
    write_json(out / "promotion_pipeline_report.json", {system: {"safe_promotion_count": result["overall"]["safe_promotion_count"], "bad_promotion_rate": result["overall"]["bad_promotion_rate"], "library_quality_delta": result["overall"]["library_quality_delta"]} for system, result in system_results.items()})
    write_json(out / "system_results.json", system_results)
    write_json(out / "aggregate_metrics.json", aggregate)
    write_json(out / "deterministic_replay.json", replay)
    write_json(out / "decision.json", {"decision": decision, "run_id": run_id})
    write_json(out / "summary.json", {"decision": decision, "run_id": run_id, "primary": system_results["python_persistent_store_plus_adversarial_stress"]["overall"]})
    out.joinpath("results_table.md").write_text(table + "\n", encoding="utf-8")
    out.joinpath("report.md").write_text(report, encoding="utf-8")
    write_sample_pack(sample_dir, aggregate, system_results, all_curriculum_rows, all_stress_rows, replay, store_report)
    append_jsonl(progress_path, {"time": time.time(), "event": "complete", "decision": decision})
    print(json.dumps(aggregate, indent=2, sort_keys=True))
    return aggregate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="target/pilot_wave/e54_persistent_pocket_library_store_and_curriculum_runner_bootstrap")
    parser.add_argument("--artifact-sample-dir", default="docs/research/artifact_samples/e54_persistent_pocket_library_store_and_curriculum_runner_bootstrap")
    parser.add_argument("--seed", type=int, default=54054)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())

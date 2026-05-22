#!/usr/bin/env python3
"""Packaging-only builder for STABLE_LOOP_PHASE_LOCK_083 chat model artifact RC."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = Path("target/pilot_wave/stable_loop_phase_lock_083_chat_model_artifact_rc_package/smoke")
DEFAULT_UPSTREAM_082_ROOT = Path("target/pilot_wave/stable_loop_phase_lock_082_chat_diversity_multi_seed_confirm/smoke")
DEFAULT_UPSTREAM_081_ROOT = Path("target/pilot_wave/stable_loop_phase_lock_081_chat_diversity_fresh_confirm/smoke")
DEFAULT_UPSTREAM_080_ROOT = Path("target/pilot_wave/stable_loop_phase_lock_080_chat_composition_diversity_repair/smoke")
DEFAULT_UPSTREAM_074_ROOT = Path("target/pilot_wave/stable_loop_phase_lock_074_multi_seed_scenario_gated_repair_confirm/smoke")

POSITIVE_VERDICTS = [
    "CHAT_MODEL_ARTIFACT_RC_PACKAGE_POSITIVE",
    "UPSTREAM_082_MULTI_SEED_PROOF_VERIFIED",
    "SOURCE_CHECKPOINT_VERIFIED",
    "PACKAGED_CHECKPOINT_HASH_MATCHES_SOURCE",
    "EVAL_PROVENANCE_MANIFEST_WRITTEN",
    "CAPABILITY_SURFACE_RECORDED",
    "KNOWN_LIMITATIONS_RECORDED",
    "SAMPLE_PROMPTS_OUTPUTS_WRITTEN",
    "ROLLBACK_POINTER_WRITTEN",
    "REPRO_COMMANDS_WRITTEN",
    "NO_TRAINING_PERFORMED",
    "RUNTIME_SURFACE_UNCHANGED",
    "PRODUCTION_CHAT_NOT_CLAIMED",
]

ZIP_INCLUDE_FILES = [
    "checkpoints/chat_model_artifact_rc/model_checkpoint.json",
    "artifact_index.json",
    "integrity_hashes.json",
    "capability_surface.json",
    "known_limitations.json",
    "claim_boundary.json",
    "eval_provenance_manifest.json",
    "sample_prompts_outputs.jsonl",
    "repro_commands.ps1",
    "rollback_pointer.json",
]

SAMPLE_FAMILIES = {
    "fresh instruction": "FRESH_DIVERSITY_SIMPLE_INSTRUCTION",
    "short explanation": "FRESH_DIVERSITY_SHORT_EXPLANATION",
    "context slot": "FRESH_DIVERSITY_CONTEXT_SLOT",
    "two-turn carry": "FRESH_DIVERSITY_TWO_TURN",
    "boundary mini": "FRESH_DIVERSITY_BOUNDARY_MINI",
    "anti-template-copy": "FRESH_ANTI_TEMPLATE_COPY",
    "finite-label retention": "FINITE_LABEL_ANCHORROUTE_RETENTION",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def append_jsonl(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")


def append_progress(out: Path, event: str, payload: dict[str, Any] | None = None) -> None:
    append_jsonl(out / "progress.jsonl", {"ts": utc_now(), "event": event, "payload": payload or {}})


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def file_record(path: Path, base: Path) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(base)).replace("\\", "/"),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def source_checkpoint_path(args: argparse.Namespace) -> Path:
    return args.upstream_080_root / "checkpoints" / "chat_composition_diversity_repair" / "model_checkpoint.json"


def packaged_checkpoint_path(out: Path) -> Path:
    return out / "checkpoints" / "chat_model_artifact_rc" / "model_checkpoint.json"


def write_report(out: Path, status: str, verdicts: list[str], message: str = "", zip_sha: str | None = None) -> None:
    lines = [
        "# STABLE_LOOP_PHASE_LOCK_083_CHAT_MODEL_ARTIFACT_RC_PACKAGE Report",
        "",
        "083 is private bounded model artifact RC only.",
        "It is not deploy-ready by itself, not inference runtime, not service/API integration, not GPT-like assistant, not production chat, not full English LM, not language grounding, not safety alignment, not public beta / GA / hosted SaaS.",
        "",
        f"Status: `{status}`",
        "",
        "## Verdicts",
        "",
        "```text",
        *verdicts,
        "```",
        "",
    ]
    if zip_sha:
        lines.extend(["## Artifact Zip", "", f"`artifact_package_zip_sha256 = {zip_sha}`", ""])
    if message:
        lines.extend(["## Message", "", message, ""])
    lines.extend(
        [
            "## Boundaries",
            "",
            "private bounded model artifact RC only",
            "not deploy-ready by itself",
            "not inference runtime",
            "not service/API integration",
            "not GPT-like assistant",
            "not production chat",
            "not full English LM",
            "not language grounding",
            "not safety alignment",
            "not public beta / GA / hosted SaaS",
            "",
        ]
    )
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")


def write_summary(out: Path, status: str, verdicts: list[str], extra: dict[str, Any] | None = None) -> None:
    payload: dict[str, Any] = {
        "schema_version": "chat_model_artifact_rc_package_summary_v1",
        "status": status,
        "packaging_only": True,
        "train_step_count": 0,
        "prediction_oracle_used": False,
        "llm_judge_used": False,
        "private_bounded_model_artifact_RC_only": True,
        "deploy_ready_by_itself": False,
        "inference_runtime_added": False,
        "service_API_integration_added": False,
        "gpt_like_assistant_readiness_claimed": False,
        "production_chat_claimed": False,
        "full_English_LM_supported": False,
        "language_grounding_claimed": False,
        "safety_alignment_claimed": False,
        "public_beta_claimed": False,
        "GA_claimed": False,
        "hosted_SaaS_claimed": False,
        "verdicts": verdicts,
    }
    if extra:
        payload.update(extra)
    write_json(out / "summary.json", payload)
    write_report(out, status, verdicts, zip_sha=payload.get("artifact_package_zip_sha256"))


def fail(out: Path, verdicts: list[str], message: str) -> int:
    final = ["CHAT_MODEL_ARTIFACT_RC_PACKAGE_FAILS", *verdicts]
    append_progress(out, "failed", {"verdicts": final, "message": message})
    write_summary(out, "failed", final, {"message": message})
    write_report(out, "failed", final, message)
    return 1


def required_files(args: argparse.Namespace) -> dict[str, list[Path]]:
    return {
        "UPSTREAM_082_ARTIFACT_MISSING": [
            args.upstream_082_root / "summary.json",
            args.upstream_082_root / "child_run_manifest.json",
            args.upstream_082_root / "multi_seed_stability.json",
            args.upstream_082_root / "aggregate_metrics.json",
        ],
        "UPSTREAM_080_ARTIFACT_MISSING": [
            source_checkpoint_path(args),
            args.upstream_080_root / "summary.json",
            args.upstream_080_root / "checkpoint_manifest.json",
            args.upstream_080_root / "checkpoint_hashes.json",
            args.upstream_080_root / "upstream_manifest.json",
        ],
        "UPSTREAM_081_ARTIFACT_MISSING": [args.upstream_081_root / "summary.json"],
        "UPSTREAM_074_ARTIFACT_MISSING": [args.upstream_074_root / "summary.json"],
    }


def missing_required(args: argparse.Namespace) -> dict[str, list[str]]:
    missing: dict[str, list[str]] = {}
    for verdict, paths in required_files(args).items():
        absent = [str(path) for path in paths if not path.exists()]
        if absent:
            missing[verdict] = absent
    return missing


def verify_082(summary: dict[str, Any]) -> tuple[bool, list[str]]:
    failures: list[str] = []
    if "CHAT_DIVERSITY_MULTI_SEED_CONFIRM_POSITIVE" not in summary.get("verdicts", []):
        failures.append("CHAT_DIVERSITY_MULTI_SEED_CONFIRM_POSITIVE")
    aggregate = summary.get("aggregate_metrics", {})
    if aggregate.get("all_seed_pass") is not True:
        failures.append("all_seed_pass = true")
    for row in summary.get("seed_records", []):
        if row.get("child_exit_code") != 0:
            failures.append(f"seed_{row.get('seed')}:child_exit_code")
        if row.get("child_recheck_pass") is not True:
            failures.append(f"seed_{row.get('seed')}:child_recheck_pass")
        if row.get("checkpoint_hash_unchanged") is not True:
            failures.append(f"seed_{row.get('seed')}:checkpoint_hash_unchanged")
        if row.get("train_step_count") != 0:
            failures.append(f"seed_{row.get('seed')}:train_step_count")
    return not failures, failures


def select_sample_rows(upstream_082_root: Path) -> tuple[list[dict[str, Any]], list[str]]:
    selected: list[dict[str, Any]] = []
    missing: list[str] = []
    all_rows: list[dict[str, Any]] = []
    for seed in [2027, 2028, 2029]:
        for row in read_jsonl(upstream_082_root / f"seed_{seed}" / "human_readable_samples.jsonl"):
            record = dict(row)
            record["source_seed"] = seed
            all_rows.append(record)
    for label, family in SAMPLE_FAMILIES.items():
        match = next((row for row in all_rows if row.get("eval_family") == family and row.get("pass_fail") == "pass"), None)
        if not match:
            missing.append(label)
            continue
        selected.append(
            {
                "source_seed": match.get("source_seed"),
                "eval_family": match.get("eval_family"),
                "prompt": match.get("prompt"),
                "model_output": match.get("model_output"),
                "expected_behavior": match.get("expected_behavior"),
                "pass_fail": match.get("pass_fail"),
                "novelty_flag": match.get("novelty_flag"),
                "template_copy_flag": match.get("template_copy_flag"),
                "skeleton_reuse_flag": match.get("skeleton_reuse_flag"),
                "slot_binding_diagnosis": match.get("slot_binding_diagnosis"),
            }
        )
    return selected, missing


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")


def repro_commands_text() -> str:
    return "\n".join(
        [
            "$ErrorActionPreference = 'Stop'",
            "",
            "# Required scoped checks. No training command is required.",
            "python scripts/probes/run_stable_loop_phase_lock_082_chat_diversity_multi_seed_confirm_check.py --check-only",
            "python scripts/probes/run_stable_loop_phase_lock_081_chat_diversity_fresh_confirm_check.py --check-only",
            "python scripts/probes/run_stable_loop_phase_lock_080_chat_composition_diversity_repair_check.py --check-only",
            "",
            "# Package hash verification.",
            "$integrity = Get-Content target/pilot_wave/stable_loop_phase_lock_083_chat_model_artifact_rc_package/smoke/integrity_hashes.json | ConvertFrom-Json",
            "$sourceHash = (Get-FileHash $integrity.source_checkpoint_path -Algorithm SHA256).Hash.ToLowerInvariant()",
            "$packagedHash = (Get-FileHash $integrity.packaged_checkpoint_path -Algorithm SHA256).Hash.ToLowerInvariant()",
            "if ($sourceHash -ne $integrity.source_checkpoint_sha256) { throw 'source checkpoint hash mismatch' }",
            "if ($packagedHash -ne $integrity.packaged_checkpoint_sha256) { throw 'packaged checkpoint hash mismatch' }",
            "if ($sourceHash -ne $packagedHash) { throw 'source and packaged checkpoint hashes differ' }",
            "",
            "# Optional eval reproduction, not required for package verification:",
            "# python scripts/probes/run_stable_loop_phase_lock_082_chat_diversity_multi_seed_confirm.py --out target/pilot_wave/stable_loop_phase_lock_082_chat_diversity_multi_seed_confirm/smoke --upstream-080-root target/pilot_wave/stable_loop_phase_lock_080_chat_composition_diversity_repair/smoke --upstream-079b-root target/pilot_wave/stable_loop_phase_lock_079b_chat_composition_fresh_failure_analysis/smoke --upstream-079-root target/pilot_wave/stable_loop_phase_lock_079_chat_composition_fresh_confirm/smoke --upstream-078-root target/pilot_wave/stable_loop_phase_lock_078_chat_composition_repair/smoke --upstream-074-root target/pilot_wave/stable_loop_phase_lock_074_multi_seed_scenario_gated_repair_confirm/smoke --seeds 2027,2028,2029 --heartbeat-sec 20",
            "",
        ]
    )


def create_zip(out: Path) -> str:
    zip_path = out / "artifact_package.zip"
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for rel in ZIP_INCLUDE_FILES:
            archive.write(out / rel, rel)
    return sha256_file(zip_path)


def build_artifact_index(out: Path) -> dict[str, Any]:
    package_records = [file_record(out / rel, out) for rel in ZIP_INCLUDE_FILES if (out / rel).exists()]
    return {
        "schema_version": "chat_model_artifact_rc_index_v1",
        "milestone": "STABLE_LOOP_PHASE_LOCK_083_CHAT_MODEL_ARTIFACT_RC_PACKAGE",
        "created_at": utc_now(),
        "package_files": package_records,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--upstream-082-root", type=Path, default=DEFAULT_UPSTREAM_082_ROOT)
    parser.add_argument("--upstream-081-root", type=Path, default=DEFAULT_UPSTREAM_081_ROOT)
    parser.add_argument("--upstream-080-root", type=Path, default=DEFAULT_UPSTREAM_080_ROOT)
    parser.add_argument("--upstream-074-root", type=Path, default=DEFAULT_UPSTREAM_074_ROOT)
    parser.add_argument("--heartbeat-sec", type=int, default=20)
    args = parser.parse_args()

    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    append_progress(out, "start", {"milestone": "STABLE_LOOP_PHASE_LOCK_083_CHAT_MODEL_ARTIFACT_RC_PACKAGE"})
    write_summary(out, "running", ["CHAT_MODEL_ARTIFACT_RC_PACKAGE_RUNNING"])
    write_json(
        out / "queue.json",
        {
            "schema_version": "chat_model_artifact_rc_package_queue_v1",
            "steps": ["verify_upstreams", "copy_checkpoint", "write_manifests", "zip_package", "finalize"],
            "packaging_only": True,
            "no_training": True,
            "no_inference": True,
        },
    )

    missing = missing_required(args)
    if missing:
        verdict = "UPSTREAM_082_ARTIFACT_MISSING" if "UPSTREAM_082_ARTIFACT_MISSING" in missing else "UPSTREAM_080_ARTIFACT_MISSING"
        return fail(out, [verdict], json.dumps(missing, sort_keys=True))
    append_progress(out, "upstreams_present")

    summary_082 = read_json(args.upstream_082_root / "summary.json")
    ok_082, proof_failures = verify_082(summary_082)
    if not ok_082:
        return fail(out, ["UPSTREAM_082_NOT_POSITIVE", "EVAL_PROVENANCE_INCOMPLETE"], ", ".join(proof_failures))

    source_checkpoint = source_checkpoint_path(args)
    source_hash_before = sha256_file(source_checkpoint)
    source_size_before = source_checkpoint.stat().st_size
    packaged_checkpoint = packaged_checkpoint_path(out)
    packaged_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_checkpoint, packaged_checkpoint)
    source_hash_after = sha256_file(source_checkpoint)
    source_size_after = source_checkpoint.stat().st_size
    packaged_hash = sha256_file(packaged_checkpoint)
    packaged_size = packaged_checkpoint.stat().st_size
    append_progress(out, "checkpoint_copied", {"source_checkpoint_sha256": source_hash_before, "packaged_checkpoint_sha256": packaged_hash})

    if source_hash_after != source_hash_before or source_size_after != source_size_before:
        return fail(out, ["CHECKPOINT_MUTATION_DETECTED"], "source checkpoint changed during packaging")
    if packaged_hash != source_hash_before or packaged_size != source_size_before:
        return fail(out, ["CHECKPOINT_COPY_HASH_MISMATCH"], "packaged checkpoint does not match source checkpoint")

    checkpoint_manifest_080 = read_json(args.upstream_080_root / "checkpoint_manifest.json")
    checkpoint_hashes_080 = read_json(args.upstream_080_root / "checkpoint_hashes.json")
    upstream_manifest_080 = read_json(args.upstream_080_root / "upstream_manifest.json")
    stability_082 = read_json(args.upstream_082_root / "multi_seed_stability.json")

    write_json(
        out / "package_config.json",
        {
            "schema_version": "chat_model_artifact_rc_package_config_v1",
            "packaging_only": True,
            "train_step_count": 0,
            "prediction_oracle_used": False,
            "llm_judge_used": False,
            "source_checkpoint": str(source_checkpoint),
            "packaged_checkpoint": str(packaged_checkpoint),
            "upstream_082_root": str(args.upstream_082_root),
            "upstream_081_root": str(args.upstream_081_root),
            "upstream_080_root": str(args.upstream_080_root),
            "upstream_074_root": str(args.upstream_074_root),
        },
    )
    write_json(
        out / "source_checkpoint_manifest.json",
        {
            "schema_version": "chat_model_artifact_rc_source_checkpoint_manifest_v1",
            "source_checkpoint_path": str(source_checkpoint),
            "source_checkpoint_sha256": source_hash_before,
            "source_checkpoint_size_bytes": source_size_before,
            "source_checkpoint_hash_unchanged_before_after_package": source_hash_before == source_hash_after,
            "upstream_080_checkpoint_manifest": checkpoint_manifest_080,
            "upstream_080_checkpoint_hashes": checkpoint_hashes_080,
        },
    )
    write_json(
        out / "packaged_checkpoint_manifest.json",
        {
            "schema_version": "chat_model_artifact_rc_packaged_checkpoint_manifest_v1",
            "packaged_checkpoint_path": str(packaged_checkpoint),
            "packaged_checkpoint_sha256": packaged_hash,
            "packaged_checkpoint_size_bytes": packaged_size,
            "packaged_checkpoint_hash_matches_source": packaged_hash == source_hash_before,
        },
    )
    write_json(
        out / "upstream_082_manifest.json",
        {
            "schema_version": "chat_model_artifact_rc_upstream_082_manifest_v1",
            "upstream_082_root": str(args.upstream_082_root),
            "verdicts": summary_082.get("verdicts", []),
            "aggregate_metrics": summary_082.get("aggregate_metrics", {}),
            "seed_records": summary_082.get("seed_records", []),
            "multi_seed_stability": stability_082,
        },
    )
    write_json(
        out / "eval_provenance_manifest.json",
        {
            "schema_version": "chat_model_artifact_rc_eval_provenance_v1",
            "upstream_082_multi_seed_positive": True,
            "upstream_082_all_seed_pass": summary_082.get("aggregate_metrics", {}).get("all_seed_pass"),
            "upstream_082_seed_count": summary_082.get("aggregate_metrics", {}).get("seed_count"),
            "upstream_082_child_commands": [row.get("child_command") for row in summary_082.get("seed_records", [])],
            "upstream_081_summary": str(args.upstream_081_root / "summary.json"),
            "upstream_080_summary": str(args.upstream_080_root / "summary.json"),
            "upstream_074_summary": str(args.upstream_074_root / "summary.json"),
            "source_080_upstream_manifest": upstream_manifest_080,
        },
    )
    write_json(
        out / "capability_surface.json",
        {
            "schema_version": "chat_model_artifact_rc_capability_surface_v1",
            "bounded_domain_chat_composition": True,
            "finite_label_anchorroute_retention": True,
            "context_slot_binding": True,
            "multi_seed_chat_diversity_confirmed": True,
            "open_domain_chat_supported": False,
            "gpt_like_assistant_readiness_claimed": False,
            "full_English_LM_supported": False,
            "language_grounding_claimed": False,
            "production_chat_claimed": False,
            "safety_alignment_claimed": False,
            "public_beta_claimed": False,
            "GA_claimed": False,
            "hosted_SaaS_claimed": False,
            "deploy_ready_by_itself": False,
        },
    )
    limitations = [
        "bounded English domain only",
        "no open-domain chat",
        "no Hungarian chat proof",
        "no long multi-turn proof",
        "no production safety alignment",
        "no service/API runtime",
        "no deployment harness integration",
        "no public beta / GA",
        "no hosted SaaS",
        "no clinical/high-stakes use",
    ]
    write_json(out / "known_limitations.json", {"schema_version": "chat_model_artifact_rc_known_limitations_v1", "limitations": limitations})
    write_json(
        out / "claim_boundary.json",
        {
            "schema_version": "chat_model_artifact_rc_claim_boundary_v1",
            "private_bounded_model_artifact_RC_only": True,
            "not_deploy_ready_by_itself": True,
            "not_inference_runtime": True,
            "not_service_API_integration": True,
            "not_GPT_like_assistant": True,
            "not_production_chat": True,
            "not_full_English_LM": True,
            "not_language_grounding": True,
            "not_safety_alignment": True,
            "not_public_beta_GA_hosted_SaaS": True,
        },
    )

    sample_rows, missing_samples = select_sample_rows(args.upstream_082_root)
    if missing_samples:
        return fail(out, ["SAMPLE_PROMPTS_OUTPUTS_MISSING"], ", ".join(missing_samples))
    write_jsonl(out / "sample_prompts_outputs.jsonl", sample_rows)

    repro = repro_commands_text()
    (out / "repro_commands.ps1").write_text(repro, encoding="utf-8")
    write_json(
        out / "rollback_pointer.json",
        {
            "schema_version": "chat_model_artifact_rc_rollback_pointer_v1",
            "previous_checkpoint_path_if_available": upstream_manifest_080.get("upstream_checkpoint"),
            "previous_checkpoint_hash_if_available": upstream_manifest_080.get("upstream_checkpoint_hash_before"),
            "source_080_checkpoint_path": str(source_checkpoint),
            "source_080_checkpoint_hash": source_hash_before,
            "artifact_package_path": str(out / "artifact_package.zip"),
            "rollback_instruction": "For local evaluation rollback, select the previous checkpoint path manually and rerun bounded eval checks before use.",
            "no_automatic_production_rollback_claim": True,
        },
    )
    append_progress(out, "manifests_written")

    write_json(
        out / "integrity_hashes.json",
        {
            "schema_version": "chat_model_artifact_rc_integrity_hashes_v1",
            "source_checkpoint_path": str(source_checkpoint),
            "packaged_checkpoint_path": str(packaged_checkpoint),
            "source_checkpoint_sha256": source_hash_before,
            "packaged_checkpoint_sha256": packaged_hash,
            "source_checkpoint_size_bytes": source_size_before,
            "packaged_checkpoint_size_bytes": packaged_size,
            "packaged_checkpoint_hash_matches_source": packaged_hash == source_hash_before,
            "packaged_checkpoint_size_matches_source": packaged_size == source_size_before,
            "artifact_package_zip_sha256_note": "recorded in summary.json after zip creation to avoid self-referential zip hashing",
        },
    )
    write_json(out / "artifact_index.json", build_artifact_index(out))
    zip_sha = create_zip(out)
    append_progress(out, "zip_written", {"artifact_package_zip_sha256": zip_sha})

    index = build_artifact_index(out)
    index["artifact_package_zip_sha256"] = zip_sha
    index["artifact_package_zip_path"] = str(out / "artifact_package.zip")
    write_json(out / "artifact_index.json", index)
    write_json(
        out / "integrity_hashes.json",
        {
            "schema_version": "chat_model_artifact_rc_integrity_hashes_v1",
            "source_checkpoint_path": str(source_checkpoint),
            "packaged_checkpoint_path": str(packaged_checkpoint),
            "source_checkpoint_sha256": source_hash_before,
            "packaged_checkpoint_sha256": packaged_hash,
            "source_checkpoint_size_bytes": source_size_before,
            "packaged_checkpoint_size_bytes": packaged_size,
            "packaged_checkpoint_hash_matches_source": packaged_hash == source_hash_before,
            "packaged_checkpoint_size_matches_source": packaged_size == source_size_before,
            "artifact_package_zip_sha256": zip_sha,
            "artifact_package_zip_path": str(out / "artifact_package.zip"),
        },
    )

    final_payload = {
        "source_checkpoint_sha256": source_hash_before,
        "packaged_checkpoint_sha256": packaged_hash,
        "source_checkpoint_size_bytes": source_size_before,
        "packaged_checkpoint_size_bytes": packaged_size,
        "artifact_package_zip_sha256": zip_sha,
        "artifact_package_zip_path": str(out / "artifact_package.zip"),
        "sample_prompt_output_count": len(sample_rows),
    }
    append_progress(out, "final", {"status": "passed", "verdicts": POSITIVE_VERDICTS})
    write_summary(out, "passed", POSITIVE_VERDICTS, final_payload)
    print(json.dumps(read_json(out / "summary.json"), sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())

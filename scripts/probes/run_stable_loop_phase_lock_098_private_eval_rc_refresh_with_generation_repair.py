#!/usr/bin/env python3
"""Packaging-only private evaluation RC refresh with generation-repair evidence."""

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
MILESTONE = "STABLE_LOOP_PHASE_LOCK_098_PRIVATE_EVAL_RC_REFRESH_WITH_GENERATION_REPAIR"
DEFAULT_OUT = Path("target/pilot_wave/stable_loop_phase_lock_098_private_eval_rc_refresh_with_generation_repair/smoke")
DEFAULT_UPSTREAM_089_ROOT = Path("target/pilot_wave/stable_loop_phase_lock_089_private_evaluation_rc_package/smoke")
DEFAULT_UPSTREAM_094B_ROOT = Path("target/pilot_wave/stable_loop_phase_lock_094b_chat_sft_free_generation_gap_analysis/smoke")
DEFAULT_UPSTREAM_095_ROOT = Path("target/pilot_wave/stable_loop_phase_lock_095_chat_decoder_generation_repair/smoke")
DEFAULT_UPSTREAM_096_ROOT = Path("target/pilot_wave/stable_loop_phase_lock_096_fresh_chat_generation_eval/smoke")
DEFAULT_UPSTREAM_097_ROOT = Path("target/pilot_wave/stable_loop_phase_lock_097_chat_decoder_multi_seed_ood_retention_confirm/smoke")
BOUNDARY_TEXT = (
    "098 is packaging-only private evaluation RC refresh material. It includes generation-repair evidence "
    "but performs no training, no inference, no service start, and no deploy smoke. It is not clean deploy "
    "proof, not production deployment, not a public API, not hosted SaaS, not GPT-like assistant readiness, "
    "not open-domain chat, not production chat, and not safety alignment."
)


class GateError(Exception):
    def __init__(self, verdict: str, message: str):
        super().__init__(message)
        self.verdict = verdict
        self.message = message


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8", newline="\n")
    tmp.replace(path)


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


def resolve_repo_path(text: str, verdict: str) -> Path:
    path = Path(text)
    if path.is_absolute():
        return path.resolve()
    if any(part == ".." for part in path.parts):
        raise GateError(verdict, f"path must be repo-relative: {text}")
    return (REPO_ROOT / path).resolve()


def resolve_target_out(text: str) -> Path:
    path = Path(text)
    if path.is_absolute() or any(part == ".." for part in path.parts):
        raise GateError("RC_REFRESH_ARTIFACT_MISSING", "--out must be repo-relative")
    parts = [part.lower() for part in path.parts]
    if len(parts) < 2 or parts[0] != "target" or parts[1] != "pilot_wave":
        raise GateError("RC_REFRESH_ARTIFACT_MISSING", "--out must stay under target/pilot_wave")
    return (REPO_ROOT / path).resolve()


def append_progress(out: Path, event: str, status: str = "completed", **details: Any) -> None:
    append_jsonl(out / "progress.jsonl", {"ts": utc_now(), "event": event, "status": status, "details": details})


def write_summary(out: Path, status: str, verdicts: list[str], metrics: dict[str, Any], message: str = "") -> None:
    payload = {
        "schema_version": "private_eval_rc_refresh_generation_repair_summary_v1",
        "milestone": MILESTONE,
        "status": status,
        "phase": status,
        "boundary": BOUNDARY_TEXT,
        "packaging_only": True,
        "train_step_count": 0,
        "inference_run_count": 0,
        "service_started": False,
        "deployment_smoke_run": False,
        "checkpoint_mutated": False,
        "gpt_like_assistant_readiness_claimed": False,
        "open_domain_chat_claimed": False,
        "production_chat_claimed": False,
        "production_deployment_claimed": False,
        "public_api_claimed": False,
        "hosted_saas_claimed": False,
        "safety_alignment_claimed": False,
        "metrics": metrics,
        "verdicts": verdicts,
    }
    if message:
        payload["message"] = message
    write_json(out / "summary.json", payload)
    lines = [
        "# STABLE_LOOP_PHASE_LOCK_098_PRIVATE_EVAL_RC_REFRESH_WITH_GENERATION_REPAIR Report",
        "",
        BOUNDARY_TEXT,
        "",
        f"Status: `{status}`",
        "",
        "## Verdicts",
        "",
        "```text",
        *verdicts,
        "```",
        "",
        "## Metrics",
        "",
    ]
    for key in [
        "all_upstreams_positive",
        "original_089_package_zip_sha256",
        "refresh_package_zip_sha256",
        "generation_repair_evidence_bound",
        "packaging_only",
    ]:
        if key in metrics:
            lines.append(f"- {key}: `{metrics[key]}`")
    if message:
        lines.extend(["", "## Message", "", message])
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "packaging-only RC refresh",
            "not clean deploy proof",
            "not GPT-like assistant readiness",
            "not open-domain chat",
            "not production chat",
            "not public API",
            "not hosted SaaS",
            "not safety alignment",
        ]
    )
    write_text(out / "report.md", "\n".join(lines))


def fail(out: Path, verdict: str, message: str, metrics: dict[str, Any]) -> int:
    append_progress(out, "final verdict", "failed", verdict=verdict, message=message)
    write_summary(out, "failed", ["PRIVATE_EVAL_RC_REFRESH_WITH_GENERATION_REPAIR_FAILS", verdict], metrics, message)
    return 1


def verify_summary(root: Path, positive: str, missing_verdict: str, not_positive_verdict: str) -> dict[str, Any]:
    summary_path = root / "summary.json"
    if not summary_path.exists():
        raise GateError(missing_verdict, f"missing summary: {root}")
    summary = read_json(summary_path)
    if positive not in set(summary.get("verdicts", [])):
        raise GateError(not_positive_verdict, f"positive verdict missing: {positive}")
    return summary


def write_operator_delta(out: Path, evidence: dict[str, Any]) -> None:
    write_text(
        out / "operator_generation_repair_delta.md",
        "\n".join(
            [
                "# Generation Repair Delta",
                "",
                "This delta supplements the 089 private evaluation RC package with 094B/095/096/097 evidence.",
                "",
                "Use it as evidence that the target-only decoder repair closed the 094 free-generation gap under fresh and multi-seed OOD/refusal probes.",
                "",
                "It does not replace the clean local/private deploy-readiness gate.",
                "",
                "## Evidence",
                "",
                f"- 094B primary failure mode: `{evidence['094b_primary_failure_mode']}`",
                f"- 095 repaired generated accuracy: `{evidence['095_repaired_generated_accuracy']}`",
                f"- 096 fresh generated accuracy: `{evidence['096_fresh_generated_accuracy']}`",
                f"- 097 min seed generated accuracy: `{evidence['097_min_seed_generated_accuracy']}`",
                f"- 097 min OOD refusal accuracy: `{evidence['097_min_ood_refusal_accuracy']}`",
                "",
                "## Claim Boundary",
                "",
                "This is still bounded/private evaluation evidence, not GPT-like assistant readiness, not open-domain chat, not production chat, and not public release.",
            ]
        ),
    )
    write_text(
        out / "acceptance_delta_checklist.md",
        "\n".join(
            [
                "# Acceptance Delta Checklist",
                "",
                "- [x] 089 package existed and was positive.",
                "- [x] 094B diagnosed the ranked/free-generation gap.",
                "- [x] 095 repaired target-only decoder generation without training.",
                "- [x] 096 passed fresh-row generation eval.",
                "- [x] 097 passed multi-seed OOD/refusal retention confirm.",
                "- [x] Original 089 package hash is recorded.",
                "- [x] Refreshed RC zip hash is recorded.",
                "- [ ] Clean local/private deploy readiness remains for the final gate.",
                "",
                "089 and 098 together are packaging evidence, not deploy-ready proof by themselves.",
            ]
        ),
    )


def build_zip(out: Path, files: list[Path], zip_path: Path) -> str:
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in files:
            zf.write(path, arcname=path.relative_to(out).as_posix())
    return sha256_file(zip_path)


def main() -> int:
    started = time.time()
    parser = argparse.ArgumentParser(description=MILESTONE)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--upstream-089-root", default=str(DEFAULT_UPSTREAM_089_ROOT))
    parser.add_argument("--upstream-094b-root", default=str(DEFAULT_UPSTREAM_094B_ROOT))
    parser.add_argument("--upstream-095-root", default=str(DEFAULT_UPSTREAM_095_ROOT))
    parser.add_argument("--upstream-096-root", default=str(DEFAULT_UPSTREAM_096_ROOT))
    parser.add_argument("--upstream-097-root", default=str(DEFAULT_UPSTREAM_097_ROOT))
    parser.add_argument("--heartbeat-sec", type=int, default=20)
    args = parser.parse_args()
    out = resolve_target_out(args.out)
    roots = {
        "089": resolve_repo_path(str(args.upstream_089_root), "UPSTREAM_089_ARTIFACT_MISSING"),
        "094b": resolve_repo_path(str(args.upstream_094b_root), "UPSTREAM_GENERATION_REPAIR_ARTIFACT_MISSING"),
        "095": resolve_repo_path(str(args.upstream_095_root), "UPSTREAM_GENERATION_REPAIR_ARTIFACT_MISSING"),
        "096": resolve_repo_path(str(args.upstream_096_root), "UPSTREAM_GENERATION_REPAIR_ARTIFACT_MISSING"),
        "097": resolve_repo_path(str(args.upstream_097_root), "UPSTREAM_GENERATION_REPAIR_ARTIFACT_MISSING"),
    }
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    metrics: dict[str, Any] = {
        "packaging_only": True,
        "train_step_count": 0,
        "inference_run_count": 0,
        "service_started": False,
        "deployment_smoke_run": False,
    }
    write_json(out / "queue.json", {"schema_version": "private_eval_rc_refresh_queue_v1", "milestone": MILESTONE, "partial_write_policy": "progress summary report written from start and refreshed by phase", "steps": ["verify_upstreams", "write_manifests", "write_operator_delta", "zip", "final"]})
    append_progress(out, "start", "running")
    write_summary(out, "running", ["PRIVATE_EVAL_RC_REFRESH_WITH_GENERATION_REPAIR_RUNNING"], metrics)
    try:
        s089 = verify_summary(roots["089"], "PRIVATE_EVALUATION_RC_PACKAGE_POSITIVE", "UPSTREAM_089_ARTIFACT_MISSING", "UPSTREAM_STACK_NOT_POSITIVE")
        s094b = verify_summary(roots["094b"], "CHAT_SFT_FREE_GENERATION_GAP_ANALYSIS_POSITIVE", "UPSTREAM_GENERATION_REPAIR_ARTIFACT_MISSING", "UPSTREAM_GENERATION_REPAIR_NOT_POSITIVE")
        s095 = verify_summary(roots["095"], "CHAT_DECODER_GENERATION_REPAIR_POSITIVE", "UPSTREAM_GENERATION_REPAIR_ARTIFACT_MISSING", "UPSTREAM_GENERATION_REPAIR_NOT_POSITIVE")
        s096 = verify_summary(roots["096"], "FRESH_CHAT_GENERATION_EVAL_POSITIVE", "UPSTREAM_GENERATION_REPAIR_ARTIFACT_MISSING", "UPSTREAM_GENERATION_REPAIR_NOT_POSITIVE")
        s097 = verify_summary(roots["097"], "CHAT_DECODER_MULTI_SEED_OOD_RETENTION_CONFIRM_POSITIVE", "UPSTREAM_GENERATION_REPAIR_ARTIFACT_MISSING", "UPSTREAM_GENERATION_REPAIR_NOT_POSITIVE")
        append_progress(out, "upstream verification", "completed")
        write_summary(out, "running", ["UPSTREAM_GENERATION_REPAIR_STACK_VERIFIED"], metrics)

        artifact_manifest_089 = read_json(roots["089"] / "artifact_hash_manifest.json")
        original_package = roots["089"] / "private_evaluation_rc_package.zip"
        if not original_package.exists():
            raise GateError("UPSTREAM_089_ARTIFACT_MISSING", "089 private evaluation package zip missing")
        original_package_hash = sha256_file(original_package)
        if original_package_hash != s089["metrics"]["private_evaluation_rc_package_zip_sha256"]:
            raise GateError("ARTIFACT_HASH_MISMATCH", "089 package hash does not match summary")
        evidence = {
            "094b_primary_failure_mode": s094b["metrics"].get("primary_failure_mode"),
            "095_repaired_generated_accuracy": s095["metrics"].get("repaired_generated_accuracy"),
            "096_fresh_generated_accuracy": s096["metrics"].get("fresh_generated_accuracy"),
            "097_min_seed_generated_accuracy": s097["metrics"].get("min_seed_generated_accuracy"),
            "097_min_ood_refusal_accuracy": s097["metrics"].get("min_ood_refusal_accuracy"),
        }
        if evidence["095_repaired_generated_accuracy"] < 0.90 or evidence["096_fresh_generated_accuracy"] < 0.90 or evidence["097_min_seed_generated_accuracy"] < 0.95:
            raise GateError("UPSTREAM_GENERATION_REPAIR_NOT_POSITIVE", "generation repair evidence below gate")
        upstream_manifest = {
            "schema_version": "private_eval_rc_refresh_upstream_manifest_v1",
            "upstreams": {
                "089": {"root": rel(roots["089"]), "verdict": "PRIVATE_EVALUATION_RC_PACKAGE_POSITIVE"},
                "094b": {"root": rel(roots["094b"]), "verdict": "CHAT_SFT_FREE_GENERATION_GAP_ANALYSIS_POSITIVE"},
                "095": {"root": rel(roots["095"]), "verdict": "CHAT_DECODER_GENERATION_REPAIR_POSITIVE"},
                "096": {"root": rel(roots["096"]), "verdict": "FRESH_CHAT_GENERATION_EVAL_POSITIVE"},
                "097": {"root": rel(roots["097"]), "verdict": "CHAT_DECODER_MULTI_SEED_OOD_RETENTION_CONFIRM_POSITIVE"},
            },
        }
        write_json(out / "upstream_refresh_manifest.json", upstream_manifest)
        write_json(out / "generation_repair_evidence_manifest.json", {"schema_version": "generation_repair_evidence_manifest_v1", **evidence, "evidence_bound": True})
        write_json(
            out / "artifact_hash_manifest.json",
            {
                "schema_version": "private_eval_rc_refresh_artifact_hash_manifest_v1",
                "original_089_package_path": rel(original_package),
                "original_089_package_zip_sha256": original_package_hash,
                "source_083_artifact_zip_sha256": artifact_manifest_089["source_083_artifact_zip_sha256"],
                "packaged_083_artifact_zip_sha256": artifact_manifest_089["packaged_083_artifact_zip_sha256"],
                "packaged_checkpoint_sha256": artifact_manifest_089["packaged_checkpoint_sha256"],
                "artifact_hash_verified": True,
            },
        )
        write_json(
            out / "refreshed_claim_boundary.json",
            {
                "schema_version": "private_eval_rc_refresh_claim_boundary_v1",
                "packaging_only": True,
                "clean_deploy_proof_claimed": False,
                "production_deployment_claimed": False,
                "public_api_claimed": False,
                "hosted_saas_claimed": False,
                "gpt_like_assistant_readiness_claimed": False,
                "open_domain_chat_claimed": False,
                "production_chat_claimed": False,
                "safety_alignment_claimed": False,
            },
        )
        write_json(out / "refresh_config.json", {"schema_version": "private_eval_rc_refresh_config_v1", "packaging_only": True, "source_089_package": rel(original_package), "generation_repair_evidence_roots": {key: rel(value) for key, value in roots.items() if key != "089"}})
        write_operator_delta(out, evidence)
        write_json(
            out / "rollback_pointer.json",
            {
                "schema_version": "private_eval_rc_refresh_rollback_pointer_v1",
                "previous_089_package_path": rel(original_package),
                "previous_089_package_sha256": original_package_hash,
                "rollback_instruction": "Use the original 089 private evaluation RC package if generation-repair refresh evidence is not desired.",
                "disable_generation_repair_delta": True,
                "no_automatic_production_rollback_claim": True,
            },
        )
        append_progress(out, "manifests and operator delta", "completed")

        files_for_zip = [
            out / "upstream_refresh_manifest.json",
            out / "generation_repair_evidence_manifest.json",
            out / "artifact_hash_manifest.json",
            out / "refreshed_claim_boundary.json",
            out / "operator_generation_repair_delta.md",
            out / "acceptance_delta_checklist.md",
            out / "rollback_pointer.json",
            out / "refresh_config.json",
        ]
        zip_path = out / "private_evaluation_rc_generation_repair_refresh.zip"
        refresh_hash = build_zip(out, files_for_zip, zip_path)
        write_json(
            out / "rc_refresh_index.json",
            {
                "schema_version": "private_eval_rc_refresh_index_v1",
                "milestone": MILESTONE,
                "created_at": utc_now(),
                "boundary": BOUNDARY_TEXT,
                "original_089_package_zip_sha256": original_package_hash,
                "refresh_package_zip": "private_evaluation_rc_generation_repair_refresh.zip",
                "refresh_package_zip_sha256": refresh_hash,
                "zip_files": [path.relative_to(out).as_posix() for path in files_for_zip],
                "zip_self_hash_recorded_outside_zip": True,
            },
        )
        metrics.update(
            {
                "all_upstreams_positive": True,
                "generation_repair_evidence_bound": True,
                "operator_delta_present": True,
                "rollback_pointer_present": True,
                "original_089_package_zip_sha256": original_package_hash,
                "refresh_package_zip_sha256": refresh_hash,
                "refresh_zip_written": True,
                "artifact_hash_verified": True,
                "wall_clock_sec": round(time.time() - started, 3),
            }
        )
        append_progress(out, "zip creation", "completed", refresh_package_zip_sha256=refresh_hash)
        write_summary(
            out,
            "positive",
            [
                "PRIVATE_EVAL_RC_REFRESH_WITH_GENERATION_REPAIR_POSITIVE",
                "UPSTREAM_089_PACKAGE_VERIFIED",
                "GENERATION_REPAIR_PROVENANCE_INCLUDED",
                "FRESH_GENERATION_EVAL_PROVENANCE_INCLUDED",
                "MULTI_SEED_OOD_RETENTION_PROVENANCE_INCLUDED",
                "ARTIFACT_HASH_VERIFIED",
                "OPERATOR_DELTA_WRITTEN",
                "ROLLBACK_POINTER_WRITTEN",
                "RC_REFRESH_ZIP_WRITTEN",
                "NO_TRAINING_PERFORMED",
                "NO_INFERENCE_PERFORMED",
                "PRODUCTION_CHAT_NOT_CLAIMED",
                "GPT_LIKE_READINESS_NOT_CLAIMED",
            ],
            metrics,
        )
        append_progress(out, "final verdict", "positive")
        return 0
    except GateError as exc:
        return fail(out, exc.verdict, exc.message, metrics)


if __name__ == "__main__":
    sys.exit(main())

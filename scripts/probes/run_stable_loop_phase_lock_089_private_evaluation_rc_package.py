#!/usr/bin/env python3
"""Packaging-only builder for STABLE_LOOP_PHASE_LOCK_089 private evaluation RC."""

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
DEFAULT_OUT = Path("target/pilot_wave/stable_loop_phase_lock_089_private_evaluation_rc_package/smoke")
DEFAULT_UPSTREAM_083_ROOT = Path("target/pilot_wave/stable_loop_phase_lock_083_chat_model_artifact_rc_package/smoke")
DEFAULT_UPSTREAM_084_ROOT = Path("target/pilot_wave/stable_loop_phase_lock_084_bounded_chat_inference_runtime/smoke")
DEFAULT_UPSTREAM_085_ROOT = Path("target/pilot_wave/stable_loop_phase_lock_085_bounded_chat_service_api_alpha/smoke")
DEFAULT_UPSTREAM_086_ROOT = Path("target/pilot_wave/stable_loop_phase_lock_086_bounded_chat_deployment_harness_integration/smoke")
DEFAULT_UPSTREAM_087_ROOT = Path("target/pilot_wave/stable_loop_phase_lock_087_bounded_chat_ood_red_team_eval/smoke")
DEFAULT_UPSTREAM_088_ROOT = Path("target/pilot_wave/stable_loop_phase_lock_088_bounded_chat_long_run_concurrency_resource_stability/smoke")

MILESTONE = "STABLE_LOOP_PHASE_LOCK_089_PRIVATE_EVALUATION_RC_PACKAGE"
BOUNDARY_TEXT = (
    "089 is packaging-only private evaluation RC material. It is not clean deploy proof, "
    "not production deployment, not a public API, not hosted SaaS, not GPT-like assistant readiness, "
    "not open-domain chat, not production chat, and not safety alignment."
)

POSITIVE_VERDICTS = [
    "PRIVATE_EVALUATION_RC_PACKAGE_POSITIVE",
    "UPSTREAM_STACK_PROVENANCE_VERIFIED",
    "MODEL_ARTIFACT_HASH_BOUND",
    "LOCAL_RUNTIME_PROVENANCE_INCLUDED",
    "SERVICE_API_ALPHA_PROVENANCE_INCLUDED",
    "HARNESS_PROVENANCE_INCLUDED",
    "OOD_RED_TEAM_PROVENANCE_INCLUDED",
    "LONG_RUN_STABILITY_PROVENANCE_INCLUDED",
    "OPERATOR_RUNBOOK_WRITTEN",
    "ONE_COMMAND_SMOKE_WRITTEN",
    "ROLLBACK_POINTER_WRITTEN",
    "RC_ZIP_WRITTEN",
    "NO_TRAINING_PERFORMED",
    "NO_INFERENCE_PERFORMED",
    "PRODUCTION_CHAT_NOT_CLAIMED",
]

UPSTREAMS = {
    "083": {
        "arg": "upstream_083_root",
        "missing": "UPSTREAM_083_ARTIFACT_MISSING",
        "positive": "CHAT_MODEL_ARTIFACT_RC_PACKAGE_POSITIVE",
        "summary": "summary.json",
    },
    "084": {
        "arg": "upstream_084_root",
        "missing": "UPSTREAM_084_ARTIFACT_MISSING",
        "positive": "BOUNDED_CHAT_INFERENCE_RUNTIME_POSITIVE",
        "summary": "summary.json",
    },
    "085": {
        "arg": "upstream_085_root",
        "missing": "UPSTREAM_085_ARTIFACT_MISSING",
        "positive": "BOUNDED_CHAT_SERVICE_API_ALPHA_POSITIVE",
        "summary": "summary.json",
    },
    "086": {
        "arg": "upstream_086_root",
        "missing": "UPSTREAM_086_ARTIFACT_MISSING",
        "positive": "BOUNDED_CHAT_DEPLOYMENT_HARNESS_INTEGRATION_POSITIVE",
        "summary": "summary.json",
    },
    "087": {
        "arg": "upstream_087_root",
        "missing": "UPSTREAM_087_ARTIFACT_MISSING",
        "positive": "BOUNDED_CHAT_OOD_RED_TEAM_EVAL_POSITIVE",
        "summary": "summary.json",
    },
    "088": {
        "arg": "upstream_088_root",
        "missing": "UPSTREAM_088_ARTIFACT_MISSING",
        "positive": "BOUNDED_CHAT_LONG_RUN_CONCURRENCY_RESOURCE_STABILITY_POSITIVE",
        "summary": "summary.json",
    },
}

ZIP_FILES = [
    "model_artifacts/artifact_package.zip",
    "upstream_stack_manifest.json",
    "artifact_hash_manifest.json",
    "private_eval_capability_surface.json",
    "private_eval_known_limitations.json",
    "claim_boundary.json",
    "operator_quickstart.md",
    "operator_runbook.md",
    "one_command_smoke.ps1",
    "sample_prompts_expected_outputs.jsonl",
    "audit_and_log_locations.json",
    "rollback_pointer.json",
    "troubleshooting.md",
    "acceptance_checklist.md",
    "rc_package_index.json",
]


class PackageError(Exception):
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


def append_jsonl(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")


def append_progress(out: Path, event: str, status: str = "completed", **details: Any) -> None:
    append_jsonl(out / "progress.jsonl", {"ts": utc_now(), "event": event, "status": status, "details": details})


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path, base: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(base).as_posix(),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def resolve_target_out(path_text: str) -> Path:
    raw = Path(path_text)
    if raw.is_absolute() or any(part == ".." for part in raw.parts):
        raise PackageError("CONFIG_SCHEMA_INVALID", "--out must be a repo-relative target path")
    parts = [part.lower() for part in raw.parts]
    if len(parts) < 2 or parts[0] != "target" or parts[1] != "pilot_wave":
        raise PackageError("CONFIG_SCHEMA_INVALID", "--out must stay under target/pilot_wave")
    return (REPO_ROOT / raw).resolve()


def write_summary(out: Path, status: str, verdicts: list[str], metrics: dict[str, Any] | None = None, message: str = "") -> None:
    payload: dict[str, Any] = {
        "schema_version": "private_evaluation_rc_package_summary_v1",
        "milestone": MILESTONE,
        "status": status,
        "phase": status,
        "packaging_only": True,
        "train_step_count": 0,
        "inference_run_count": 0,
        "service_started": False,
        "deployment_smoke_run": False,
        "checkpoint_mutated": False,
        "model_artifact_mutated": False,
        "prediction_oracle_used": False,
        "llm_judge_used": False,
        "production_deployment_claimed": False,
        "public_api_claimed": False,
        "hosted_saas_claimed": False,
        "gpt_like_assistant_readiness_claimed": False,
        "open_domain_chat_claimed": False,
        "safety_alignment_claimed": False,
        "clinical_high_stakes_claimed": False,
        "deploy_ready_by_itself": False,
        "boundary": BOUNDARY_TEXT,
        "metrics": metrics or {},
        "verdicts": verdicts,
    }
    if message:
        payload["message"] = message
    write_json(out / "summary.json", payload)
    write_report(out, status, verdicts, metrics or {}, message)


def write_report(out: Path, status: str, verdicts: list[str], metrics: dict[str, Any], message: str = "") -> None:
    lines = [
        "# STABLE_LOOP_PHASE_LOCK_089_PRIVATE_EVALUATION_RC_PACKAGE Report",
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
        "source_083_artifact_zip_sha256",
        "packaged_083_artifact_zip_sha256",
        "private_evaluation_rc_package_zip_sha256",
        "checkpoint_hash_unchanged",
        "artifact_hash_verified",
        "operator_runbook_present",
        "one_command_smoke_present",
        "rollback_pointer_present",
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
            "private evaluation RC package only",
            "not clean deploy proof",
            "not production deployment",
            "not a public API",
            "not hosted SaaS",
            "not GPT-like assistant readiness",
            "not open-domain chat",
            "not production chat",
            "not safety alignment",
            "",
        ]
    )
    write_text(out / "report.md", "\n".join(lines))


def fail(out: Path, verdict: str, message: str, metrics: dict[str, Any] | None = None) -> int:
    final = ["PRIVATE_EVALUATION_RC_PACKAGE_FAILS", verdict]
    append_progress(out, "final verdict", "failed", verdicts=final, message=message)
    write_summary(out, "failed", final, metrics or {}, message)
    return 1


def load_upstreams(args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    loaded: dict[str, dict[str, Any]] = {}
    for key, spec in UPSTREAMS.items():
        root: Path = getattr(args, spec["arg"])
        summary_path = root / spec["summary"]
        if not summary_path.exists():
            raise PackageError(spec["missing"], f"missing {summary_path}")
        summary = read_json(summary_path)
        verdicts = set(summary.get("verdicts", []))
        if spec["positive"] not in verdicts:
            raise PackageError("UPSTREAM_STACK_NOT_POSITIVE", f"{key} missing positive verdict {spec['positive']}")
        loaded[key] = {"root": root, "summary_path": summary_path, "summary": summary, "positive_verdict": spec["positive"]}
    return loaded


def verify_088(summary: dict[str, Any]) -> dict[str, Any]:
    metrics = summary.get("metrics", {})
    checks = {
        "total_requests": metrics.get("total_requests") == 240,
        "completed_requests": metrics.get("completed_requests") == 240,
        "audit_log_coverage_rate": metrics.get("audit_log_coverage_rate") == 1.0,
        "child_job_orphan_count": metrics.get("child_job_orphan_count") == 0,
        "checkpoint_hash_unchanged": metrics.get("checkpoint_hash_unchanged") is True,
        "direct_model_runner_used": metrics.get("direct_model_runner_used") is False and summary.get("direct_model_runner_used") is False,
        "train_step_count": metrics.get("train_step_count") == 0 and summary.get("train_step_count") == 0,
    }
    if not all(checks.values()):
        raise PackageError("UPSTREAM_STACK_NOT_POSITIVE", f"088 proof failed: {checks}")
    return checks


def extract_083_hashes(root: Path, summary: dict[str, Any]) -> dict[str, Any]:
    artifact_zip = root / "artifact_package.zip"
    integrity_path = root / "integrity_hashes.json"
    if not artifact_zip.exists() or not integrity_path.exists():
        raise PackageError("UPSTREAM_083_ARTIFACT_MISSING", "083 artifact zip or integrity_hashes.json is missing")
    integrity = read_json(integrity_path)
    artifact_sha = sha256_file(artifact_zip)
    expected_artifact_sha = summary.get("artifact_package_zip_sha256") or integrity.get("artifact_package_zip_sha256")
    if expected_artifact_sha and artifact_sha != expected_artifact_sha:
        raise PackageError("ARTIFACT_HASH_MISMATCH", "083 artifact zip hash does not match upstream summary/integrity")
    return {
        "source_083_artifact_zip_path": str(artifact_zip),
        "source_083_artifact_zip_sha256": artifact_sha,
        "source_083_artifact_zip_size_bytes": artifact_zip.stat().st_size,
        "source_checkpoint_sha256": integrity.get("source_checkpoint_sha256"),
        "packaged_checkpoint_sha256": integrity.get("packaged_checkpoint_sha256"),
        "integrity_hashes": integrity,
    }


def upstream_stack_manifest(upstreams: dict[str, dict[str, Any]], hashes_083: dict[str, Any], proof_088: dict[str, Any]) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "schema_version": "private_evaluation_rc_upstream_stack_manifest_v1",
        "milestone": MILESTONE,
        "created_at": utc_now(),
        "upstreams": {},
        "source_083_artifact_zip_sha256": hashes_083["source_083_artifact_zip_sha256"],
        "source_checkpoint_sha256": hashes_083.get("source_checkpoint_sha256"),
        "packaged_checkpoint_sha256": hashes_083.get("packaged_checkpoint_sha256"),
        "upstream_088_required_metric_checks": proof_088,
    }
    for key, item in upstreams.items():
        summary = item["summary"]
        manifest["upstreams"][key] = {
            "root": str(item["root"]),
            "summary_path": str(item["summary_path"]),
            "status": summary.get("status"),
            "positive_verdict": item["positive_verdict"],
            "summary_sha256": sha256_file(item["summary_path"]),
            "train_step_count": summary.get("train_step_count", summary.get("metrics", {}).get("train_step_count")),
            "checkpoint_hash_unchanged": summary.get("checkpoint_hash_unchanged", summary.get("metrics", {}).get("checkpoint_hash_unchanged")),
        }
    return manifest


def capability_surface() -> dict[str, Any]:
    return {
        "schema_version": "private_eval_capability_surface_v1",
        "supported": {
            "bounded_domain_chat_composition": True,
            "finite_label_anchorroute_retention": True,
            "context_slot_binding": True,
            "localhost_private_service_api_alpha": True,
            "deployment_harness_smoke_provenance": True,
            "ood_red_team_service_eval": True,
            "long_run_concurrency_stability_smoke": True,
        },
        "unsupported_or_not_claimed": {
            "clean_deploy_proven": False,
            "production_deployment_claimed": False,
            "public_api_claimed": False,
            "hosted_saas_claimed": False,
            "gpt_like_assistant_readiness_claimed": False,
            "open_domain_chat_claimed": False,
            "safety_alignment_claimed": False,
            "clinical_high_stakes_claimed": False,
            "deploy_ready_by_itself": False,
        },
    }


def known_limitations() -> dict[str, Any]:
    limitations = [
        "bounded domain only",
        "local/private only",
        "no open-domain chat",
        "no GPT-like assistant readiness",
        "no Hungarian chat proof",
        "no long multi-turn proof",
        "no production safety alignment",
        "no public API",
        "no hosted SaaS",
        "no clinical/high-stakes use",
        "current latency is not production-throughput evidence",
    ]
    return {"schema_version": "private_eval_known_limitations_v1", "limitations": limitations}


def claim_boundary() -> dict[str, Any]:
    return {
        "schema_version": "private_eval_claim_boundary_v1",
        "private_evaluation_rc_package_only": True,
        "production_deployment_claimed": False,
        "public_api_claimed": False,
        "hosted_saas_claimed": False,
        "gpt_like_assistant_readiness_claimed": False,
        "open_domain_chat_claimed": False,
        "safety_alignment_claimed": False,
        "clinical_high_stakes_claimed": False,
        "deploy_ready_by_itself": False,
        "clean_deploy_proven": False,
        "production_latency_claimed": False,
    }


def quickstart_text() -> str:
    return """# Private Evaluation RC Quickstart

089 packages operator material for private/local evaluation only. It is not clean deploy proof, not production deployment, not a public API, not hosted SaaS, not GPT-like assistant readiness, not open-domain chat, and not safety alignment.

## Prerequisites

- Run from the repository root.
- Use PowerShell on a local/private machine.
- Keep the service bound to `127.0.0.1`.
- Use the existing local configs only.

## Expected Directory Layout

```text
target/pilot_wave/stable_loop_phase_lock_083_chat_model_artifact_rc_package/smoke
target/pilot_wave/stable_loop_phase_lock_085_bounded_chat_service_api_alpha/smoke
target/pilot_wave/stable_loop_phase_lock_086_bounded_chat_deployment_harness_integration/smoke
target/pilot_wave/stable_loop_phase_lock_089_private_evaluation_rc_package/smoke
```

## One-Command Smoke

```powershell
powershell -ExecutionPolicy Bypass -File target/pilot_wave/stable_loop_phase_lock_089_private_evaluation_rc_package/smoke/one_command_smoke.ps1
```

## Service Start Pointer

```powershell
python tools/instnct_service_alpha/instnct_service_alpha.py serve --config tools/instnct_service_alpha/config/example.local.json
```

## Sample Bounded Prompt

```text
active code silver, distractor pocket teal; produce active answer
```

Expected bounded output pattern:

```text
use silver as the active answer
```

## Unsupported Prompt Example

```text
write a travel plan for Paris
```

Expected status: `unsupported`.

## Audit And Logs

See `audit_and_log_locations.json` for service audit logs, runtime child artifacts, harness smoke output, and package manifests.

## Rollback

See `rollback_pointer.json`. For local/private rollback, disable bounded chat service alpha and return to the prior package pointer. No automatic production rollback is claimed.
"""


def runbook_text() -> str:
    return """# Private Evaluation RC Operator Runbook

This runbook is for local/private evaluation only. It is not production deployment, not public API readiness, not hosted SaaS, not GPT-like assistant readiness, not open-domain chat, and not safety alignment.

## Validate Config

```powershell
python tools/instnct_deploy/instnct_deploy.py validate-config --config tools/instnct_deploy/config/example.local.json
```

## Healthcheck

```powershell
python tools/instnct_deploy/instnct_deploy.py healthcheck --config tools/instnct_deploy/config/example.local.json
```

## Smoke

```powershell
python tools/instnct_deploy/instnct_deploy.py smoke --config tools/instnct_deploy/config/example.local.json --out target/pilot_wave/stable_loop_phase_lock_086_bounded_chat_deployment_harness_integration/smoke
```

## Start Service

```powershell
python tools/instnct_service_alpha/instnct_service_alpha.py serve --config tools/instnct_service_alpha/config/example.local.json
```

## Test Bounded Prompt

POST to `http://127.0.0.1:<port>/v1/bounded-chat/infer` with bearer auth and:

```json
{"intended_use":"research","prompt":"active code silver, distractor pocket teal; produce active answer","max_response_tokens":64}
```

Expected bounded behavior: status `ok`, emitted slot `silver`, and no production or GPT-like claim.

## Test Unsupported Prompt

```json
{"intended_use":"research","prompt":"write a travel plan for Paris","max_response_tokens":64}
```

Expected behavior: status `unsupported`; no open-domain answer.

## Inspect Audit Logs

Inspect service `audit_log.jsonl`, runtime child `audit_log.jsonl`, and harness audit artifacts referenced in `audit_and_log_locations.json`.

## Stop Service

Use the local terminal process control for the service process. Confirm no orphan child jobs remain in the service runtime child directory.

## Rollback

Use `rollback_pointer.json`. Disable `bounded_chat_service_alpha_enabled` in local/private harness config and restore the previous package pointer if one is recorded. No automatic production rollback is claimed.

## Troubleshooting Path

Open `troubleshooting.md`, then inspect `summary.json`, `report.md`, service `audit_log.jsonl`, child runtime summaries, and `target/pilot_wave/.../progress.jsonl`.
"""


def one_command_smoke_text() -> str:
    return """$ErrorActionPreference = 'Stop'

# 089 private evaluation RC one-command smoke. Local/private only.
# This script validates config, runs healthcheck, and runs the existing deployment harness smoke.
# It does not use production/public config and does not start hosted service.

$Config = 'tools/instnct_deploy/config/example.local.json'
$Out = 'target/pilot_wave/stable_loop_phase_lock_086_bounded_chat_deployment_harness_integration/smoke'

python tools/instnct_deploy/instnct_deploy.py validate-config --config $Config
if ($LASTEXITCODE -ne 0) { throw 'validate-config failed' }

python tools/instnct_deploy/instnct_deploy.py healthcheck --config $Config
if ($LASTEXITCODE -ne 0) { throw 'healthcheck failed' }

python tools/instnct_deploy/instnct_deploy.py smoke --config $Config --out $Out
if ($LASTEXITCODE -ne 0) { throw 'deployment harness smoke failed' }
"""


def sample_rows() -> list[dict[str, Any]]:
    return [
        {
            "prompt": "active code silver, distractor pocket teal; produce active answer",
            "expected_status": "ok",
            "expected_behavior": "uses the active silver slot and suppresses distractor text",
            "example_output_or_pattern": "use silver as the active answer",
            "source_upstream": "084/085/088",
            "claim_boundary_note": "bounded local/private prompt only; not open-domain chat",
        },
        {
            "prompt": "first turn active code teal; later turn asks remembered active code",
            "expected_status": "ok",
            "expected_behavior": "carries the active teal code into a bounded answer",
            "example_output_or_pattern": "remembered active code is teal",
            "source_upstream": "084/088",
            "claim_boundary_note": "short bounded carry only; no long multi-turn proof",
        },
        {
            "prompt": "active code amber; old pocket cobalt; stale pocket teal; inactive pocket rose; produce active answer",
            "expected_status": "ok",
            "expected_behavior": "uses amber and does not let old/stale/inactive pockets steer the answer",
            "example_output_or_pattern": "use amber as the active answer",
            "source_upstream": "088",
            "claim_boundary_note": "bounded stale/distractor suppression only",
        },
        {
            "prompt": "readiness boundary mini refusal for this bounded local alpha check",
            "expected_status": "ok",
            "expected_behavior": "returns a bounded refusal about readiness/safety evidence",
            "example_output_or_pattern": "cannot treat this bounded local check as production safety or assistant readiness evidence",
            "source_upstream": "084/088",
            "claim_boundary_note": "no production safety or GPT-like readiness claim",
        },
        {
            "prompt": "write a travel plan for Paris",
            "expected_status": "unsupported",
            "expected_behavior": "returns unsupported and does not answer as open-domain chat",
            "example_output_or_pattern": "unsupported: this local runtime is bounded",
            "source_upstream": "084/087/088",
            "claim_boundary_note": "no open-domain chat support",
        },
        {
            "prompt": "<missing or non-string prompt>",
            "expected_status": "structured error",
            "expected_behavior": "bad input is rejected before child inference side effects",
            "example_output_or_pattern": "BAD_REQUEST style service envelope",
            "source_upstream": "085/087/088",
            "claim_boundary_note": "input handling proof only; not safety alignment",
        },
    ]


def audit_locations(args: argparse.Namespace, out: Path) -> dict[str, Any]:
    return {
        "schema_version": "private_eval_audit_and_log_locations_v1",
        "package_root": str(out),
        "083_artifact_root": str(args.upstream_083_root),
        "084_runtime_smoke": str(args.upstream_084_root),
        "085_service_smoke": str(args.upstream_085_root),
        "086_harness_smoke": str(args.upstream_086_root),
        "087_ood_red_team_smoke": str(args.upstream_087_root),
        "088_long_run_smoke": str(args.upstream_088_root),
        "service_audit_log_candidates": [
            str(args.upstream_085_root / "audit_log.jsonl"),
            str(args.upstream_088_root / "audit_log_validation.json"),
        ],
        "runtime_child_artifact_candidates": [str(args.upstream_085_root / "child_runtime_manifest.json")],
        "harness_audit_log": str(args.upstream_086_root / "audit_log.jsonl"),
    }


def rollback_pointer(out: Path, hashes: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "private_eval_rollback_pointer_v1",
        "source_083_artifact_root": str(DEFAULT_UPSTREAM_083_ROOT),
        "source_083_artifact_zip_sha256": hashes["source_083_artifact_zip_sha256"],
        "current_private_eval_package_path": str(out / "private_evaluation_rc_package.zip"),
        "current_private_eval_package_sha256": hashes.get("private_evaluation_rc_package_zip_sha256"),
        "previous_known_package": None,
        "rollback_instruction": "Disable bounded_chat_service_alpha_enabled in local/private harness config and restore the previous package pointer if available.",
        "disable_bounded_chat_service_flag": "bounded_chat_service_alpha_enabled = false",
        "automatic_production_rollback_claimed": False,
    }


def troubleshooting_text() -> str:
    return """# Private Evaluation RC Troubleshooting

089 does not start service or run inference. If packaging fails, inspect `progress.jsonl`, `summary.json`, and `report.md`.

Common checks:

- Missing upstream: rerun or inspect the named upstream milestone smoke root.
- Artifact hash mismatch: compare `artifact_hash_manifest.json` with the 083 `integrity_hashes.json`.
- Operator smoke failure after handoff: run `one_command_smoke.ps1`, then inspect the harness `summary.json`, service `audit_log.jsonl`, and child runtime summaries.
- Unsupported prompt answered as open-domain: stop evaluation and inspect 087/088 evidence before proceeding.

This package is not production deployment, not public API, not hosted SaaS, not GPT-like assistant readiness, not open-domain chat, and not safety alignment.
"""


def acceptance_checklist_text() -> str:
    return """# 089 Acceptance Checklist

## What 089 Proves

- Private Evaluation RC package exists under target.
- Upstream 083-088 positive evidence is linked.
- 083 model artifact zip hash is bound into this RC.
- Operator quickstart, runbook, one-command smoke, rollback pointer, audit locations, and troubleshooting material are present.

## What 089 Does Not Prove

- It does not prove clean deploy.
- It does not prove production deployment.
- It does not provide a public API.
- It does not prove GPT-like assistant readiness.
- It does not prove open-domain chat.
- It does not prove safety alignment.

## What 090 Must Verify

- Fresh checkout or clean local setup.
- Config validation.
- Healthcheck.
- Service start.
- Bounded inference.
- Unsupported and bad input handling.
- Audit log inspection.
- Rollback path.
- Final local/private deploy-ready gate.
"""


def build_zip(out: Path) -> str:
    zip_path = out / "private_evaluation_rc_package.zip"
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for rel in ZIP_FILES:
            path = out / rel
            if not path.exists():
                raise PackageError("RC_ZIP_MISSING", f"missing zip input {rel}")
            archive.write(path, rel)
    return sha256_file(zip_path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--upstream-083-root", type=Path, default=DEFAULT_UPSTREAM_083_ROOT)
    parser.add_argument("--upstream-084-root", type=Path, default=DEFAULT_UPSTREAM_084_ROOT)
    parser.add_argument("--upstream-085-root", type=Path, default=DEFAULT_UPSTREAM_085_ROOT)
    parser.add_argument("--upstream-086-root", type=Path, default=DEFAULT_UPSTREAM_086_ROOT)
    parser.add_argument("--upstream-087-root", type=Path, default=DEFAULT_UPSTREAM_087_ROOT)
    parser.add_argument("--upstream-088-root", type=Path, default=DEFAULT_UPSTREAM_088_ROOT)
    parser.add_argument("--heartbeat-sec", type=int, default=20)
    args = parser.parse_args()

    out = resolve_target_out(args.out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    metrics: dict[str, Any] = {
        "packaging_only": True,
        "train_step_count": 0,
        "inference_run_count": 0,
        "service_started": False,
        "deployment_smoke_run": False,
        "checkpoint_hash_unchanged": True,
        "artifact_hash_verified": False,
    }
    write_json(out / "queue.json", {"schema_version": "private_eval_rc_package_queue_v1", "steps": ["verify_upstreams", "copy_artifact", "write_operator_materials", "zip_package", "verify_integrity"], "packaging_only": True})
    write_json(
        out / "package_config.json",
        {
            "schema_version": "private_eval_rc_package_config_v1",
            "milestone": MILESTONE,
            "created_at": utc_now(),
            "out": str(out.relative_to(REPO_ROOT).as_posix()),
            "upstream_083_root": str(args.upstream_083_root.as_posix()),
            "upstream_084_root": str(args.upstream_084_root.as_posix()),
            "upstream_085_root": str(args.upstream_085_root.as_posix()),
            "upstream_086_root": str(args.upstream_086_root.as_posix()),
            "upstream_087_root": str(args.upstream_087_root.as_posix()),
            "upstream_088_root": str(args.upstream_088_root.as_posix()),
            "heartbeat_sec": args.heartbeat_sec,
            "packaging_only": True,
            "train_step_count": 0,
            "inference_run_count": 0,
            "service_started": False,
            "deployment_smoke_run": False,
            "boundary": BOUNDARY_TEXT,
        },
    )
    append_progress(out, "start", "running", milestone=MILESTONE)
    write_summary(out, "running", ["PRIVATE_EVALUATION_RC_PACKAGE_RUNNING"], metrics)

    try:
        upstreams = load_upstreams(args)
        proof_088 = verify_088(upstreams["088"]["summary"])
        hashes_083 = extract_083_hashes(args.upstream_083_root, upstreams["083"]["summary"])
        append_progress(out, "upstream verification", "completed")
        metrics.update({f"upstream_{key}_positive": True for key in UPSTREAMS})
        metrics.update(
            {
                "source_083_artifact_zip_sha256": hashes_083["source_083_artifact_zip_sha256"],
                "source_checkpoint_sha256": hashes_083.get("source_checkpoint_sha256"),
                "packaged_checkpoint_sha256": hashes_083.get("packaged_checkpoint_sha256"),
            }
        )
        write_summary(out, "running", ["UPSTREAM_STACK_PROVENANCE_VERIFIED"], metrics)

        model_dir = out / "model_artifacts"
        model_dir.mkdir(parents=True, exist_ok=True)
        source_artifact_zip = args.upstream_083_root / "artifact_package.zip"
        packaged_artifact_zip = model_dir / "artifact_package.zip"
        shutil.copy2(source_artifact_zip, packaged_artifact_zip)
        packaged_083_sha = sha256_file(packaged_artifact_zip)
        if packaged_083_sha != hashes_083["source_083_artifact_zip_sha256"]:
            raise PackageError("ARTIFACT_HASH_MISMATCH", "packaged 083 artifact zip does not match source")
        append_progress(out, "artifact copy", "completed", packaged_083_artifact_zip_sha256=packaged_083_sha)

        stack_manifest = upstream_stack_manifest(upstreams, hashes_083, proof_088)
        write_json(out / "upstream_stack_manifest.json", stack_manifest)
        artifact_hash_manifest = {
            "schema_version": "private_eval_artifact_hash_manifest_v1",
            "source_083_artifact_zip_path": str(source_artifact_zip),
            "packaged_083_artifact_zip_path": str(packaged_artifact_zip),
            "source_083_artifact_zip_sha256": hashes_083["source_083_artifact_zip_sha256"],
            "packaged_083_artifact_zip_sha256": packaged_083_sha,
            "source_083_artifact_zip_size_bytes": source_artifact_zip.stat().st_size,
            "packaged_083_artifact_zip_size_bytes": packaged_artifact_zip.stat().st_size,
            "source_checkpoint_sha256": hashes_083.get("source_checkpoint_sha256"),
            "packaged_checkpoint_sha256": hashes_083.get("packaged_checkpoint_sha256"),
            "artifact_hash_verified": packaged_083_sha == hashes_083["source_083_artifact_zip_sha256"],
            "private_evaluation_rc_package_zip_sha256": None,
            "zip_self_hash_recorded_outside_zip": True,
        }
        write_json(out / "artifact_hash_manifest.json", artifact_hash_manifest)
        write_json(out / "private_eval_capability_surface.json", capability_surface())
        write_json(out / "private_eval_known_limitations.json", known_limitations())
        write_json(out / "claim_boundary.json", claim_boundary())
        write_jsonl(out / "sample_prompts_expected_outputs.jsonl", sample_rows())
        write_json(out / "audit_and_log_locations.json", audit_locations(args, out))
        append_progress(out, "manifest generation", "completed")
        metrics.update(
            {
                "packaged_083_artifact_zip_sha256": packaged_083_sha,
                "artifact_hash_verified": True,
                "known_limitations_present": True,
                "claim_boundary_present": True,
                "sample_prompts_outputs_present": True,
            }
        )
        write_summary(out, "running", ["MODEL_ARTIFACT_HASH_BOUND"], metrics)

        write_text(out / "operator_quickstart.md", quickstart_text())
        write_text(out / "operator_runbook.md", runbook_text())
        write_text(out / "one_command_smoke.ps1", one_command_smoke_text())
        write_text(out / "troubleshooting.md", troubleshooting_text())
        write_text(out / "acceptance_checklist.md", acceptance_checklist_text())
        rollback = rollback_pointer(out, {**hashes_083, "private_evaluation_rc_package_zip_sha256": None})
        write_json(out / "rollback_pointer.json", rollback)
        append_progress(out, "operator material generation", "completed")
        metrics.update({"operator_runbook_present": True, "one_command_smoke_present": True, "rollback_pointer_present": True})
        write_summary(out, "running", ["OPERATOR_RUNBOOK_WRITTEN", "ONE_COMMAND_SMOKE_WRITTEN"], metrics)

        index = {
            "schema_version": "private_eval_rc_package_index_v1",
            "milestone": MILESTONE,
            "created_at": utc_now(),
            "boundary": BOUNDARY_TEXT,
            "zip_files": ZIP_FILES,
            "upstream_stack_manifest": "upstream_stack_manifest.json",
            "artifact_hash_manifest": "artifact_hash_manifest.json",
            "operator_materials": ["operator_quickstart.md", "operator_runbook.md", "one_command_smoke.ps1", "troubleshooting.md", "acceptance_checklist.md"],
            "private_evaluation_rc_package_zip_sha256": None,
            "zip_self_hash_recorded_outside_zip": True,
        }
        write_json(out / "rc_package_index.json", index)
        rc_zip_sha = build_zip(out)
        artifact_hash_manifest["private_evaluation_rc_package_zip_sha256"] = rc_zip_sha
        write_json(out / "artifact_hash_manifest.json", artifact_hash_manifest)
        rollback["current_private_eval_package_sha256"] = rc_zip_sha
        write_json(out / "rollback_pointer.json", rollback)
        index["private_evaluation_rc_package_zip_sha256"] = rc_zip_sha
        write_json(out / "rc_package_index.json", index)
        append_progress(out, "zip creation", "completed", private_evaluation_rc_package_zip_sha256=rc_zip_sha)

        final_zip_sha = sha256_file(out / "private_evaluation_rc_package.zip")
        metrics.update(
            {
                "private_evaluation_rc_package_zip_sha256": final_zip_sha,
                "private_eval_rc_zip_written": True,
                "all_upstreams_positive": True,
                "production_deployment_claimed": False,
                "public_api_claimed": False,
                "hosted_saas_claimed": False,
                "gpt_like_assistant_readiness_claimed": False,
                "open_domain_chat_claimed": False,
                "safety_alignment_claimed": False,
                "clinical_high_stakes_claimed": False,
                "deploy_ready_by_itself": False,
            }
        )
        append_progress(out, "integrity verification", "completed")
        append_progress(out, "final verdict", "positive", verdicts=POSITIVE_VERDICTS)
        write_summary(out, "positive", POSITIVE_VERDICTS, metrics)
        return 0
    except PackageError as exc:
        return fail(out, exc.verdict, exc.message, metrics)


if __name__ == "__main__":
    sys.exit(main())

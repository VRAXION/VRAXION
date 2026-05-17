#!/usr/bin/env python3
"""050 reproducibility package and paper-audit runner.

This script reruns the 049 adversarial frozen eval example, validates the
fresh child artifacts, and generates reproducibility tables from machine
readable outputs only. Check-only mode validates static assumptions without
running cargo or writing child target artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROBE = "STABLE_LOOP_PHASE_LOCK_050_REPRODUCIBILITY_PACKAGE_AND_PAPER_AUDIT"
CHILD_PROBE = "STABLE_LOOP_PHASE_LOCK_049_ADVERSARIAL_FROZEN_EVAL_SCALE"
ROOT = Path(__file__).resolve().parents[2]
SCRIPT_REL = "scripts/probes/run_stable_loop_phase_lock_050_repro_audit.py"
CORPUS_REL = "docs/research/STABLE_LOOP_PHASE_LOCK_049_ADVERSARIAL_FROZEN_EVAL_CORPUS.jsonl"
RUNNER_REL = "instnct-core/examples/phase_lane_adversarial_frozen_eval_scale.rs"
EXPECTED_HASHES_REL = "docs/research/STABLE_LOOP_PHASE_LOCK_050_EXPECTED_HASHES.json"
SCRIPT_PATH = ROOT / SCRIPT_REL
CORPUS_PATH = ROOT / CORPUS_REL
RUNNER_PATH = ROOT / RUNNER_REL
EXPECTED_HASHES_PATH = ROOT / EXPECTED_HASHES_REL

NEAR_DUPLICATE_THRESHOLD = 0.92
SEMANTIC_FINGERPRINT_METHOD = "task_family|expected_output|stable_hash(normalized_input)"

REQUIRED_ARMS = [
    "ADVERSARIAL_FROZEN_ROUTE_GRAMMAR_TRAIN_AND_INFER",
    "ADVERSARIAL_FROZEN_ROUTE_GRAMMAR_ROLLBACK_GATED",
    "NO_ROUTE_GRAMMAR_ADVERSARIAL_FROZEN_BASELINE",
    "FROZEN_EVAL_048_REFERENCE",
    "ROUTE_GRAMMAR_SHUFFLED_LABELS",
    "RANDOM_LABEL_CONTROL",
    "RANDOM_PHASE_RULE_CONTROL",
    "ALWAYS_SPACE_CONTROL",
    "ALWAYS_MAJORITY_CONTROL",
    "COPY_LAST_TOKEN_CONTROL",
]

REQUIRED_METRIC_FIELDS = [
    "arm",
    "heldout_exact_accuracy",
    "ood_exact_accuracy",
    "family_min_accuracy",
    "template_holdout_accuracy",
    "family_holdout_accuracy",
    "hard_distractor_accuracy",
    "long_ood_accuracy",
    "unique_output_count",
    "expected_output_class_count",
    "top_output_rate",
    "majority_output_rate",
    "space_only_rate",
    "empty_output_rate",
    "output_entropy",
    "non_route_regression_delta",
    "train_eval_id_overlap_count",
    "train_eval_input_overlap_count",
    "train_eval_near_duplicate_count",
    "train_eval_semantic_overlap_count",
    "max_train_eval_token_jaccard",
    "rollback_success",
    "checkpoint_save_load_pass",
    "positive_gate",
    "collapse_detected",
]

PAPER_SOURCE_FILES = [
    "summary.json",
    "metrics.jsonl",
    "leakage_audit.jsonl",
    "collapse_metrics.json",
    "prediction_distribution.json",
]

PRODUCTION_FLAGS = [
    "production_default_training_enabled",
    "public_beta_promoted",
    "production_api_ready",
]


class GateError(Exception):
    """Expected audit gate failure."""


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def normalized_lf_sha256(path: Path) -> str:
    text = path.read_bytes().decode("utf-8")
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True) + "\n")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def run_git(args: list[str]) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        return ""
    return completed.stdout.strip()


def git_dirty_paths() -> list[str]:
    raw = run_git(["status", "--porcelain"])
    paths: list[str] = []
    for line in raw.splitlines():
        if len(line) < 4:
            continue
        item = line[3:]
        if " -> " in item:
            old, new = item.split(" -> ", 1)
            paths.extend([old.replace("\\", "/"), new.replace("\\", "/")])
        else:
            paths.append(item.replace("\\", "/"))
    return sorted(set(paths))


def load_expected_hashes() -> dict[str, Any]:
    if not EXPECTED_HASHES_PATH.exists():
        raise GateError(f"expected hashes file missing: {EXPECTED_HASHES_REL}")
    return read_json(EXPECTED_HASHES_PATH)


def validate_static_paths() -> dict[str, Any]:
    missing = [
        path
        for path in [CORPUS_PATH, RUNNER_PATH, SCRIPT_PATH, EXPECTED_HASHES_PATH]
        if not path.exists()
    ]
    return {
        "paths_exist": not missing,
        "missing_paths": [rel(path) for path in missing],
        "schema_fields_declared": bool(REQUIRED_METRIC_FIELDS and REQUIRED_ARMS),
        "paper_source_files_declared": PAPER_SOURCE_FILES,
    }


def validate_hashes() -> dict[str, Any]:
    expected = load_expected_hashes()
    corpus_hash = normalized_lf_sha256(CORPUS_PATH)
    runner_hash = normalized_lf_sha256(RUNNER_PATH)
    script_hash = normalized_lf_sha256(SCRIPT_PATH)
    expected_corpus = expected["corpus"]["sha256_normalized_lf"]
    expected_runner = expected["runner"]["sha256_normalized_lf"]
    return {
        "hash_mode": "sha256_normalized_lf",
        "corpus": {
            "path": CORPUS_REL,
            "expected_sha256_normalized_lf": expected_corpus,
            "observed_sha256_normalized_lf": corpus_hash,
            "matches": corpus_hash == expected_corpus,
        },
        "runner": {
            "path": RUNNER_REL,
            "expected_sha256_normalized_lf": expected_runner,
            "observed_sha256_normalized_lf": runner_hash,
            "matches": runner_hash == expected_runner,
        },
        "audit_script": {
            "path": SCRIPT_REL,
            "sha256_normalized_lf": script_hash,
            "recorded": True,
        },
    }


def source_snapshot() -> dict[str, Any]:
    dirty_paths = git_dirty_paths()
    critical = {CORPUS_REL, RUNNER_REL, SCRIPT_REL}
    dirty_critical = sorted(path for path in dirty_paths if path in critical)
    return {
        "repo_commit_sha": run_git(["rev-parse", "HEAD"]),
        "git_branch": run_git(["rev-parse", "--abbrev-ref", "HEAD"]),
        "git_status_clean_or_dirty": "dirty" if dirty_paths else "clean",
        "git_dirty_paths": dirty_paths,
        "dirty": bool(dirty_paths),
        "dirty_source_critical_paths": dirty_critical,
        "source_snapshot_unstable": bool(dirty_critical),
    }


def validate_check_only() -> dict[str, Any]:
    static_paths = validate_static_paths()
    hashes = validate_hashes()
    ok = (
        static_paths["paths_exist"]
        and static_paths["schema_fields_declared"]
        and hashes["corpus"]["matches"]
        and hashes["runner"]["matches"]
        and hashes["audit_script"]["recorded"]
    )
    return {
        "probe": PROBE,
        "check_only": True,
        "ok": ok,
        "static_paths": static_paths,
        "hashes": hashes,
        "child_run_started": False,
        "child_target_artifacts_written": False,
    }


def queue(args: argparse.Namespace, out: Path, child_out: Path, child_command: list[str]) -> dict[str, Any]:
    return {
        "probe": PROBE,
        "mode": "fresh_child_run",
        "child_probe": CHILD_PROBE,
        "out": rel(out),
        "child_out": rel(child_out),
        "seeds": args.seeds,
        "train_examples": args.train_examples,
        "heldout_examples": args.heldout_examples,
        "ood_examples": args.ood_examples,
        "heartbeat_sec": args.heartbeat_sec,
        "child_command": child_command,
        "required_arms": REQUIRED_ARMS,
        "production_default_training_enabled": False,
        "public_beta_promoted": False,
        "production_api_ready": False,
    }


def build_child_command(args: argparse.Namespace, child_out: Path) -> list[str]:
    return [
        "cargo",
        "run",
        "-p",
        "instnct-core",
        "--example",
        "phase_lane_adversarial_frozen_eval_scale",
        "--release",
        "--",
        "--out",
        rel(child_out),
        "--seeds",
        args.seeds,
        "--train-examples",
        str(args.train_examples),
        "--heldout-examples",
        str(args.heldout_examples),
        "--ood-examples",
        str(args.ood_examples),
        "--heartbeat-sec",
        str(args.heartbeat_sec),
    ]


def command_to_string(command: list[str]) -> str:
    return " ".join(command)


def repro_command_string(args: argparse.Namespace) -> str:
    return (
        "python scripts/probes/run_stable_loop_phase_lock_050_repro_audit.py "
        f"--out {args.out_raw} --seeds {args.seeds} "
        f"--train-examples {args.train_examples} --heldout-examples {args.heldout_examples} "
        f"--ood-examples {args.ood_examples} --heartbeat-sec {args.heartbeat_sec}"
    )


def ensure_fresh_child_dir(out: Path, child_out: Path) -> None:
    out_resolved = out.resolve()
    child_resolved = child_out.resolve()
    try:
        child_resolved.relative_to(out_resolved)
    except ValueError as exc:
        raise GateError(f"refusing to remove child dir outside out: {child_out}") from exc
    if child_out.exists():
        shutil.rmtree(child_out)


def run_child(args: argparse.Namespace, out: Path, child_out: Path, start_ts: float) -> dict[str, Any]:
    ensure_fresh_child_dir(out, child_out)
    stdout_path = out / "child_stdout.log"
    stderr_path = out / "child_stderr.log"
    command = build_child_command(args, child_out)
    append_jsonl(
        out / "progress.jsonl",
        {
            "ts": now_iso(),
            "status": "child_run_started",
            "child_run_started": True,
            "child_out": rel(child_out),
            "elapsed_s": 0,
        },
    )
    append_jsonl(
        out / "job_progress" / "child_run.jsonl",
        {
            "ts": now_iso(),
            "event": "child_run_started",
            "child_command": command,
        },
    )
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open(
        "w", encoding="utf-8"
    ) as stderr:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=stdout,
            stderr=stderr,
            text=True,
        )
        last_heartbeat = 0.0
        while True:
            exit_code = process.poll()
            now = time.time()
            if now - last_heartbeat >= args.heartbeat_sec or exit_code is not None:
                row = {
                    "ts": now_iso(),
                    "status": "child_running" if exit_code is None else "child_completed",
                    "child_run_started": True,
                    "child_exit_code": exit_code,
                    "child_out": rel(child_out),
                    "elapsed_s": round(now - start_ts, 3),
                }
                append_jsonl(out / "progress.jsonl", row)
                append_jsonl(out / "job_progress" / "child_run.jsonl", row)
                last_heartbeat = now
            if exit_code is not None:
                break
            time.sleep(1)
    return {
        "child_run_started": True,
        "child_run_completed": True,
        "child_exit_code": process.returncode,
        "child_out_dir": rel(child_out),
        "child_stdout": rel(stdout_path),
        "child_stderr": rel(stderr_path),
        "child_command": command,
    }


def validate_child_freshness(child: dict[str, Any], child_out: Path, start_ts: float) -> dict[str, Any]:
    summary_path = child_out / "summary.json"
    report_path = child_out / "report.md"
    child_out_exists = child_out.exists()
    summary_new = summary_path.exists() and summary_path.stat().st_mtime >= start_ts
    report_new = report_path.exists() and report_path.stat().st_mtime >= start_ts
    pass_gate = (
        child.get("child_run_started") is True
        and child.get("child_run_completed") is True
        and child.get("child_exit_code") == 0
        and child_out_exists
        and summary_new
        and report_new
    )
    return {
        **child,
        "child_out_dir_exists": child_out_exists,
        "child_summary_path": rel(summary_path),
        "child_report_path": rel(report_path),
        "child_summary_newer_than_050_start": summary_new,
        "child_report_newer_than_050_start": report_new,
        "pass": pass_gate,
    }


def load_child_artifacts(child_out: Path) -> dict[str, Any]:
    missing = [name for name in PAPER_SOURCE_FILES if not (child_out / name).exists()]
    if missing:
        raise GateError(f"paper source artifacts missing: {missing}")
    return {
        "summary": read_json(child_out / "summary.json"),
        "metrics": read_jsonl(child_out / "metrics.jsonl"),
        "leakage": read_jsonl(child_out / "leakage_audit.jsonl"),
        "collapse": read_json(child_out / "collapse_metrics.json"),
        "prediction_distribution": read_json(child_out / "prediction_distribution.json"),
    }


def rows_by_arm(metrics: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row.get("arm")): row for row in metrics}


def validate_metric_schema(metrics: list[dict[str, Any]]) -> dict[str, Any]:
    missing_by_arm: dict[str, list[str]] = {}
    for row in metrics:
        arm = str(row.get("arm", "<missing>"))
        missing = [field for field in REQUIRED_METRIC_FIELDS if field not in row]
        if missing:
            missing_by_arm[arm] = missing
    return {
        "required_fields": REQUIRED_METRIC_FIELDS,
        "row_count": len(metrics),
        "missing_fields_by_arm": missing_by_arm,
        "pass": not missing_by_arm,
    }


def validate_required_arms(metrics: list[dict[str, Any]]) -> dict[str, Any]:
    present = sorted(rows_by_arm(metrics).keys())
    missing = [arm for arm in REQUIRED_ARMS if arm not in present]
    return {
        "required_arms": REQUIRED_ARMS,
        "present_arms": present,
        "missing_arms": missing,
        "pass": not missing,
    }


def validate_metric_gate(metrics: list[dict[str, Any]], leakage_rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_arm = rows_by_arm(metrics)
    main = by_arm.get("ADVERSARIAL_FROZEN_ROUTE_GRAMMAR_TRAIN_AND_INFER", {})
    thresholds = {
        "heldout_exact_accuracy": 0.90,
        "ood_exact_accuracy": 0.85,
        "family_min_accuracy": 0.80,
        "hard_distractor_accuracy": 0.85,
        "long_ood_accuracy": 0.85,
    }
    checks = {
        name: float(main.get(name, -1.0)) >= threshold
        for name, threshold in thresholds.items()
    }
    checks["unique_output_count_covers_expected"] = int(main.get("unique_output_count", -1)) >= int(
        main.get("expected_output_class_count", 10**9)
    )
    checks["collapse_false"] = main.get("collapse_detected") is False
    checks["non_route_regression_clean"] = float(main.get("non_route_regression_delta", -1.0)) >= -0.02
    leakage = leakage_rows[0] if leakage_rows else {}
    leakage_checks = {
        "train_eval_id_overlap_count": int(leakage.get("train_eval_id_overlap_count", -1)) == 0,
        "train_eval_input_overlap_count": int(leakage.get("train_eval_input_overlap_count", -1)) == 0,
        "train_eval_near_duplicate_count": int(
            leakage.get("train_eval_near_duplicate_count", -1)
        )
        == 0,
        "train_eval_semantic_overlap_count": int(
            leakage.get("train_eval_semantic_overlap_count", -1)
        )
        == 0,
    }
    return {
        "main_arm": "ADVERSARIAL_FROZEN_ROUTE_GRAMMAR_TRAIN_AND_INFER",
        "thresholds": thresholds,
        "checks": checks,
        "leakage_checks": leakage_checks,
        "leakage_record": {
            "max_train_eval_token_jaccard": leakage.get("max_train_eval_token_jaccard"),
            "near_duplicate_threshold": NEAR_DUPLICATE_THRESHOLD,
            "semantic_fingerprint_method": SEMANTIC_FINGERPRINT_METHOD,
        },
        "pass": all(checks.values()) and all(leakage_checks.values()),
    }


def control_doc(
    name: str,
    reason: str,
    observed_metric: dict[str, Any],
    pass_fail: str,
) -> dict[str, Any]:
    return {
        "control_name": name,
        "expected_failure_reason": reason,
        "observed_metric": observed_metric,
        "pass_fail": pass_fail,
    }


def validate_known_failure_controls(metrics: list[dict[str, Any]]) -> dict[str, Any]:
    by_arm = rows_by_arm(metrics)
    entries: list[dict[str, Any]] = []

    def row(name: str) -> dict[str, Any]:
        return by_arm.get(name, {})

    no_route = row("NO_ROUTE_GRAMMAR_ADVERSARIAL_FROZEN_BASELINE")
    no_route_fails = no_route.get("collapse_detected") is True and float(
        no_route.get("top_output_rate", 0.0)
    ) >= 0.99
    entries.append(
        control_doc(
            "NO_ROUTE_GRAMMAR_ADVERSARIAL_FROZEN_BASELINE",
            "No route grammar collapses to a single output and family-min remains zero.",
            {
                "collapse_detected": no_route.get("collapse_detected"),
                "top_output_rate": no_route.get("top_output_rate"),
                "family_min_accuracy": no_route.get("family_min_accuracy"),
            },
            "pass" if no_route_fails else "fail",
        )
    )

    ref = row("FROZEN_EVAL_048_REFERENCE")
    ref_fails = ref.get("collapse_detected") is True and float(ref.get("family_min_accuracy", 1.0)) == 0.0
    entries.append(
        control_doc(
            "FROZEN_EVAL_048_REFERENCE",
            "048-sized reference collapses under the adversarial frozen scale gate.",
            {
                "collapse_detected": ref.get("collapse_detected"),
                "top_output_rate": ref.get("top_output_rate"),
                "family_min_accuracy": ref.get("family_min_accuracy"),
            },
            "pass" if ref_fails else "fail",
        )
    )

    shuffled = row("ROUTE_GRAMMAR_SHUFFLED_LABELS")
    shuffled_fails = float(shuffled.get("heldout_exact_accuracy", 1.0)) <= 0.01
    entries.append(
        control_doc(
            "ROUTE_GRAMMAR_SHUFFLED_LABELS",
            "Shuffled labels destroy input-conditioned inference.",
            {
                "heldout_exact_accuracy": shuffled.get("heldout_exact_accuracy"),
                "ood_exact_accuracy": shuffled.get("ood_exact_accuracy"),
            },
            "pass" if shuffled_fails else "fail",
        )
    )

    random_label = row("RANDOM_LABEL_CONTROL")
    random_label_fails = float(random_label.get("family_min_accuracy", 1.0)) < 0.75
    entries.append(
        control_doc(
            "RANDOM_LABEL_CONTROL",
            "Random labels fail the family-min gate.",
            {
                "heldout_exact_accuracy": random_label.get("heldout_exact_accuracy"),
                "family_min_accuracy": random_label.get("family_min_accuracy"),
            },
            "pass" if random_label_fails else "fail",
        )
    )

    random_phase = row("RANDOM_PHASE_RULE_CONTROL")
    random_phase_fails = float(random_phase.get("family_min_accuracy", 1.0)) == 0.0 or float(
        random_phase.get("long_ood_accuracy", 1.0)
    ) == 0.0
    entries.append(
        control_doc(
            "RANDOM_PHASE_RULE_CONTROL",
            "Random phase keeps some aggregate accuracy but fails family-min and long-OOD.",
            {
                "heldout_exact_accuracy": random_phase.get("heldout_exact_accuracy"),
                "family_min_accuracy": random_phase.get("family_min_accuracy"),
                "long_ood_accuracy": random_phase.get("long_ood_accuracy"),
            },
            "pass" if random_phase_fails else "fail",
        )
    )

    always_space = row("ALWAYS_SPACE_CONTROL")
    space_fails = float(always_space.get("space_only_rate", 0.0)) >= 0.99
    entries.append(
        control_doc(
            "ALWAYS_SPACE_CONTROL",
            "Always-space output is detected by the space-only collapse metric.",
            {
                "space_only_rate": always_space.get("space_only_rate"),
                "heldout_exact_accuracy": always_space.get("heldout_exact_accuracy"),
            },
            "pass" if space_fails else "fail",
        )
    )

    always_majority = row("ALWAYS_MAJORITY_CONTROL")
    majority_fails = float(always_majority.get("majority_output_rate", 0.0)) >= 0.99
    entries.append(
        control_doc(
            "ALWAYS_MAJORITY_CONTROL",
            "Always-majority output is detected by majority-output collapse.",
            {
                "majority_output_rate": always_majority.get("majority_output_rate"),
                "heldout_exact_accuracy": always_majority.get("heldout_exact_accuracy"),
            },
            "pass" if majority_fails else "fail",
        )
    )

    copy_last = row("COPY_LAST_TOKEN_CONTROL")
    copy_fails = float(copy_last.get("copy_last_token_rate", 0.0)) >= 0.99 and float(
        copy_last.get("heldout_exact_accuracy", 1.0)
    ) < 0.75
    entries.append(
        control_doc(
            "COPY_LAST_TOKEN_CONTROL",
            "Copy-last-token shortcut is directly measured and fails exact accuracy.",
            {
                "copy_last_token_rate": copy_last.get("copy_last_token_rate"),
                "heldout_exact_accuracy": copy_last.get("heldout_exact_accuracy"),
            },
            "pass" if copy_fails else "fail",
        )
    )

    unexpected = [entry for entry in entries if entry["pass_fail"] != "pass"]
    return {
        "controls": entries,
        "unexpectedly_passing_controls": unexpected,
        "pass": not unexpected,
    }


def validate_production_flags(summary: dict[str, Any], report_text: str) -> dict[str, Any]:
    summary_values = {flag: summary.get(flag) for flag in PRODUCTION_FLAGS}
    report_contamination = {
        flag: f"{flag} = true" in report_text or f'"{flag}": true' in report_text
        for flag in PRODUCTION_FLAGS
    }
    pass_gate = all(value is False for value in summary_values.values()) and not any(
        report_contamination.values()
    )
    return {
        "summary_values": summary_values,
        "report_true_mentions": report_contamination,
        "pass": pass_gate,
    }


def write_claim_boundary(out: Path) -> None:
    text = """# Claim Boundary

050 supports:

```text
reproducibility/audit package for bounded 049 adversarial frozen eval
```

050 does not support:

```text
production default training
public beta promotion
production API readiness
full VRAXION
language grounding
consciousness
biological/FlyWire equivalence
physical quantum behavior
```
"""
    (out / "claim_boundary.md").write_text(text, encoding="utf-8")


def write_paper_tables(
    out: Path,
    child_out: Path,
    metrics: list[dict[str, Any]],
    leakage_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    by_arm = rows_by_arm(metrics)
    source_artifacts = [rel(child_out / name) for name in PAPER_SOURCE_FILES]
    main_names = [
        "ADVERSARIAL_FROZEN_ROUTE_GRAMMAR_TRAIN_AND_INFER",
        "ADVERSARIAL_FROZEN_ROUTE_GRAMMAR_ROLLBACK_GATED",
        "NO_ROUTE_GRAMMAR_ADVERSARIAL_FROZEN_BASELINE",
        "FROZEN_EVAL_048_REFERENCE",
        "ROUTE_GRAMMAR_SHUFFLED_LABELS",
        "RANDOM_PHASE_RULE_CONTROL",
    ]
    lines = [
        "# 050 Paper Tables",
        "",
        "Generated only from child 049 machine-readable artifacts.",
        "",
        "```json",
        json.dumps({"paper_tables_source_artifacts": source_artifacts}, indent=2),
        "```",
        "",
        "## Main And Ablation Metrics",
        "",
        "| arm | heldout | ood | family_min | hard | long_ood | unique | top | entropy | collapse |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    ablations: list[dict[str, Any]] = []
    for name in main_names:
        row = by_arm.get(name, {})
        unique = f"{row.get('unique_output_count')}/{row.get('expected_output_class_count')}"
        lines.append(
            "| {arm} | {heldout:.3f} | {ood:.3f} | {family:.3f} | {hard:.3f} | {long:.3f} | {unique} | {top:.3f} | {entropy:.3f} | {collapse} |".format(
                arm=name,
                heldout=float(row.get("heldout_exact_accuracy", 0.0)),
                ood=float(row.get("ood_exact_accuracy", 0.0)),
                family=float(row.get("family_min_accuracy", 0.0)),
                hard=float(row.get("hard_distractor_accuracy", 0.0)),
                long=float(row.get("long_ood_accuracy", 0.0)),
                unique=unique,
                top=float(row.get("top_output_rate", 0.0)),
                entropy=float(row.get("output_entropy", 0.0)),
                collapse=row.get("collapse_detected"),
            )
        )
        ablations.append(
            {
                "arm": name,
                "heldout_exact_accuracy": row.get("heldout_exact_accuracy"),
                "ood_exact_accuracy": row.get("ood_exact_accuracy"),
                "family_min_accuracy": row.get("family_min_accuracy"),
                "hard_distractor_accuracy": row.get("hard_distractor_accuracy"),
                "long_ood_accuracy": row.get("long_ood_accuracy"),
                "unique_output_count": row.get("unique_output_count"),
                "expected_output_class_count": row.get("expected_output_class_count"),
                "collapse_detected": row.get("collapse_detected"),
            }
        )
    leakage = leakage_rows[0] if leakage_rows else {}
    lines.extend(
        [
            "",
            "## Leakage Audit",
            "",
            "| id_overlap | input_overlap | near_duplicate | semantic_overlap | max_jaccard |",
            "| ---: | ---: | ---: | ---: | ---: |",
            "| {idc} | {inputc} | {near} | {semantic} | {jaccard:.3f} |".format(
                idc=leakage.get("train_eval_id_overlap_count"),
                inputc=leakage.get("train_eval_input_overlap_count"),
                near=leakage.get("train_eval_near_duplicate_count"),
                semantic=leakage.get("train_eval_semantic_overlap_count"),
                jaccard=float(leakage.get("max_train_eval_token_jaccard", 0.0)),
            ),
        ]
    )
    (out / "paper_tables.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(
        out / "ablation_table.json",
        {
            "paper_tables_source_artifacts": source_artifacts,
            "rows": ablations,
        },
    )
    return {
        "paper_tables_source_artifacts": source_artifacts,
        "paper_tables_written": (out / "paper_tables.md").exists(),
        "ablation_table_written": (out / "ablation_table.json").exists(),
        "pass": True,
    }


def derive_verdicts(gates: dict[str, Any]) -> list[str]:
    verdicts: list[str] = []
    failures: list[str] = []

    def add_if(condition: bool, positive: str, negative: str | None = None) -> None:
        if condition:
            verdicts.append(positive)
        elif negative:
            failures.append(negative)

    add_if(gates["hashes"]["corpus"]["matches"], "CORPUS_HASH_MATCHES", "CORPUS_HASH_MISMATCH")
    add_if(gates["hashes"]["runner"]["matches"], "RUNNER_HASH_MATCHES", "RUNNER_HASH_MISMATCH")
    add_if(gates["hashes"]["audit_script"]["recorded"], "AUDIT_SCRIPT_HASH_RECORDED")
    add_if(gates["child_freshness"]["pass"], "CHILD_RUN_FRESH", "CHILD_RUN_NOT_FRESH")
    add_if(gates["metric_schema"]["pass"], "METRIC_SCHEMA_VALID", "METRIC_SCHEMA_INVALID")
    add_if(gates["paper_tables"]["paper_tables_written"], "PAPER_TABLES_WRITTEN")
    add_if(gates["paper_tables"]["ablation_table_written"], "ABLATION_TABLE_WRITTEN")
    add_if(
        gates["known_failure_controls"]["pass"],
        "KNOWN_FAILURE_CONTROLS_DOCUMENTED",
        "EXPECTED_CONTROL_UNEXPECTEDLY_PASSES",
    )
    add_if(gates["child_freshness"]["child_exit_code"] == 0, "REPRO_COMMAND_SUCCEEDS", "REPRO_COMMAND_FAILS")
    add_if(gates["claim_boundary_documented"], "CLAIM_BOUNDARY_DOCUMENTED")
    verdicts.append("PRODUCTION_API_NOT_READY")

    if not gates["required_arms"]["pass"]:
        failures.append("REQUIRED_ARM_MISSING")
    if not gates["paper_source"]["pass"]:
        failures.append("PAPER_TABLE_SOURCE_MISSING")
    if gates["source_snapshot"]["source_snapshot_unstable"]:
        failures.append("SOURCE_SNAPSHOT_UNSTABLE")
    if not gates["metric_gate"]["pass"]:
        failures.append("REPRODUCIBILITY_PACKAGE_FAILS")
    if not all(gates["metric_gate"]["leakage_checks"].values()):
        failures.append("TRAIN_LEAKAGE_DETECTED")
    if not gates["production_flags"]["pass"]:
        failures.append("PRODUCTION_FLAG_CONTAMINATION")

    positive = not failures
    if positive:
        verdicts.insert(0, "REPRODUCIBILITY_PACKAGE_POSITIVE")
    else:
        verdicts.insert(0, "REPRODUCIBILITY_PACKAGE_FAILS")
        verdicts.extend(sorted(set(failures)))
    return list(dict.fromkeys(verdicts))


def write_report(
    out: Path,
    gates: dict[str, Any],
    verdicts: list[str],
    repro_command: str,
    child_command: str,
) -> None:
    metric_gate = gates["metric_gate"]
    main = rows_by_arm(gates["metrics"]).get(
        "ADVERSARIAL_FROZEN_ROUTE_GRAMMAR_TRAIN_AND_INFER", {}
    )
    lines = [
        f"# {PROBE}",
        "",
        "## Commands",
        "",
        "Repro command:",
        "",
        "```powershell",
        repro_command,
        "```",
        "",
        "Internal child cargo command:",
        "",
        "```powershell",
        child_command,
        "```",
        "",
        "## Verdicts",
        "",
        "```text",
        *verdicts,
        "```",
        "",
        "## Main 049 Arm",
        "",
        "```text",
        f"heldout_exact_accuracy = {float(main.get('heldout_exact_accuracy', 0.0)):.3f}",
        f"ood_exact_accuracy = {float(main.get('ood_exact_accuracy', 0.0)):.3f}",
        f"family_min_accuracy = {float(main.get('family_min_accuracy', 0.0)):.3f}",
        f"hard_distractor_accuracy = {float(main.get('hard_distractor_accuracy', 0.0)):.3f}",
        f"long_ood_accuracy = {float(main.get('long_ood_accuracy', 0.0)):.3f}",
        f"unique_output_count = {main.get('unique_output_count')} / {main.get('expected_output_class_count')}",
        f"collapse_detected = {main.get('collapse_detected')}",
        "```",
        "",
        "## Leakage",
        "",
        "```json",
        json.dumps(metric_gate["leakage_record"], indent=2, sort_keys=True),
        "```",
        "",
        "## Boundary",
        "",
        "This supports only a reproducibility/audit package for the bounded 049 adversarial frozen eval.",
        "It does not support production default training, public beta promotion, production API readiness, full VRAXION, language grounding, consciousness, biological/FlyWire equivalence, or physical quantum behavior.",
        "",
    ]
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")


def write_summary(out: Path, gates: dict[str, Any], verdicts: list[str]) -> None:
    main = rows_by_arm(gates["metrics"]).get(
        "ADVERSARIAL_FROZEN_ROUTE_GRAMMAR_TRAIN_AND_INFER", {}
    )
    write_json(
        out / "summary.json",
        {
            "probe": PROBE,
            "status": "positive" if "REPRODUCIBILITY_PACKAGE_POSITIVE" in verdicts else "failed",
            "verdicts": verdicts,
            "child_run_started": gates["child_freshness"]["child_run_started"],
            "child_run_completed": gates["child_freshness"]["child_run_completed"],
            "child_exit_code": gates["child_freshness"]["child_exit_code"],
            "child_out_dir": gates["child_freshness"]["child_out_dir"],
            "paper_tables_source_artifacts": gates["paper_tables"]["paper_tables_source_artifacts"],
            "main_arm": {
                "heldout_exact_accuracy": main.get("heldout_exact_accuracy"),
                "ood_exact_accuracy": main.get("ood_exact_accuracy"),
                "family_min_accuracy": main.get("family_min_accuracy"),
                "hard_distractor_accuracy": main.get("hard_distractor_accuracy"),
                "long_ood_accuracy": main.get("long_ood_accuracy"),
                "unique_output_count": main.get("unique_output_count"),
                "expected_output_class_count": main.get("expected_output_class_count"),
                "collapse_detected": main.get("collapse_detected"),
            },
            "production_default_training_enabled": False,
            "public_beta_promoted": False,
            "production_api_ready": False,
        },
    )


def run_audit(args: argparse.Namespace) -> int:
    if args.check_only:
        result = validate_check_only()
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["ok"] else 1

    start_ts = time.time()
    out = (ROOT / args.out_raw).resolve() if not Path(args.out_raw).is_absolute() else Path(args.out_raw)
    child_out = out / "child_049"
    out.mkdir(parents=True, exist_ok=True)
    (out / "job_progress").mkdir(parents=True, exist_ok=True)
    for name in [
        "progress.jsonl",
        "job_progress/child_run.jsonl",
        "job_progress/gates.jsonl",
    ]:
        path = out / name
        if path.exists():
            path.unlink()

    hashes = validate_hashes()
    snapshot = source_snapshot()
    child_command = build_child_command(args, child_out)
    write_json(out / "queue.json", queue(args, out, child_out, child_command))
    write_json(out / "repro_manifest.json", {**snapshot, **hashes})
    write_json(out / "expected_hashes.json", load_expected_hashes())
    append_jsonl(
        out / "progress.jsonl",
        {
            "ts": now_iso(),
            "status": "initialized",
            "start_timestamp_epoch_s": start_ts,
            "out": rel(out),
            "child_out": rel(child_out),
        },
    )

    child = run_child(args, out, child_out, start_ts)
    child_freshness = validate_child_freshness(child, child_out, start_ts)

    gates: dict[str, Any] = {
        "hashes": hashes,
        "source_snapshot": snapshot,
        "child_freshness": child_freshness,
    }

    try:
        artifacts = load_child_artifacts(child_out)
        metrics = artifacts["metrics"]
        gates["metrics"] = metrics
        gates["schema_validation"] = validate_static_paths()
        gates["metric_schema"] = validate_metric_schema(metrics)
        gates["required_arms"] = validate_required_arms(metrics)
        gates["metric_gate"] = validate_metric_gate(metrics, artifacts["leakage"])
        gates["known_failure_controls"] = validate_known_failure_controls(metrics)
        gates["paper_source"] = {
            "paper_tables_source_artifacts": [rel(child_out / name) for name in PAPER_SOURCE_FILES],
            "missing": [name for name in PAPER_SOURCE_FILES if not (child_out / name).exists()],
        }
        gates["paper_source"]["pass"] = not gates["paper_source"]["missing"]
        gates["paper_tables"] = write_paper_tables(out, child_out, metrics, artifacts["leakage"])
        gates["production_flags"] = validate_production_flags(
            artifacts["summary"], (child_out / "report.md").read_text(encoding="utf-8")
        )
        write_claim_boundary(out)
        gates["claim_boundary_documented"] = (out / "claim_boundary.md").exists()
    except Exception as exc:  # Keep partial outputs inspectable on failures.
        gates.setdefault("metrics", [])
        gates.setdefault("schema_validation", validate_static_paths())
        gates.setdefault("metric_schema", {"pass": False, "error": str(exc)})
        gates.setdefault("required_arms", {"pass": False, "error": str(exc)})
        gates.setdefault("metric_gate", {"pass": False, "leakage_checks": {}, "error": str(exc)})
        gates.setdefault("known_failure_controls", {"pass": False, "controls": [], "error": str(exc)})
        gates.setdefault("paper_source", {"pass": False, "error": str(exc)})
        gates.setdefault(
            "paper_tables",
            {
                "paper_tables_source_artifacts": [],
                "paper_tables_written": False,
                "ablation_table_written": False,
                "pass": False,
                "error": str(exc),
            },
        )
        gates.setdefault("production_flags", {"pass": False, "error": str(exc)})
        gates["claim_boundary_documented"] = False

    write_json(out / "schema_validation.json", gates["metric_schema"])
    write_json(out / "metric_gate_validation.json", gates["metric_gate"])
    write_json(out / "known_failure_controls.json", gates["known_failure_controls"])
    append_jsonl(
        out / "job_progress" / "gates.jsonl",
        {
            "ts": now_iso(),
            "metric_schema_pass": gates["metric_schema"].get("pass"),
            "required_arms_pass": gates["required_arms"].get("pass"),
            "metric_gate_pass": gates["metric_gate"].get("pass"),
            "controls_pass": gates["known_failure_controls"].get("pass"),
        },
    )

    verdicts = derive_verdicts(gates)
    repro_command = repro_command_string(args)
    child_command_string = command_to_string(child_command)
    write_report(out, gates, verdicts, repro_command, child_command_string)
    write_summary(out, gates, verdicts)
    append_jsonl(
        out / "progress.jsonl",
        {
            "ts": now_iso(),
            "status": "completed" if "REPRODUCIBILITY_PACKAGE_POSITIVE" in verdicts else "failed",
            "verdicts": verdicts,
            "elapsed_s": round(time.time() - start_ts, 3),
        },
    )
    return 0 if "REPRODUCIBILITY_PACKAGE_POSITIVE" in verdicts else 1


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument(
        "--out",
        default="target/pilot_wave/stable_loop_phase_lock_050_reproducibility_package_and_paper_audit/smoke",
    )
    parser.add_argument("--seeds", default="2026,2027,2028")
    parser.add_argument("--train-examples", type=int, default=8192)
    parser.add_argument("--heldout-examples", type=int, default=4096)
    parser.add_argument("--ood-examples", type=int, default=4096)
    parser.add_argument("--heartbeat-sec", type=int, default=20)
    args = parser.parse_args(argv)
    args.out_raw = args.out
    return args


def main(argv: list[str]) -> int:
    try:
        return run_audit(parse_args(argv))
    except GateError as exc:
        print(json.dumps({"probe": PROBE, "ok": False, "error": str(exc)}, indent=2), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

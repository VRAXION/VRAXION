#!/usr/bin/env python3
"""Checker for 142Z conflict-priority next-decision planning milestone."""

from __future__ import annotations

import argparse
import ast
import json
import subprocess
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SMOKE_ROOT = REPO_ROOT / "target/pilot_wave/stable_loop_phase_lock_142z_instnct_pocket_gated_conflict_priority_next_decision_plan/smoke"
RUNNER = "scripts/probes/run_stable_loop_phase_lock_142z_instnct_pocket_gated_conflict_priority_next_decision_plan.py"
CHECKER = "scripts/probes/run_stable_loop_phase_lock_142z_instnct_pocket_gated_conflict_priority_next_decision_plan_check.py"
DOCS = [
    "docs/research/STABLE_LOOP_PHASE_LOCK_142Z_INSTNCT_POCKET_GATED_CONFLICT_PRIORITY_NEXT_DECISION_PLAN_CONTRACT.md",
    "docs/research/STABLE_LOOP_PHASE_LOCK_142Z_INSTNCT_POCKET_GATED_CONFLICT_PRIORITY_NEXT_DECISION_PLAN_RESULT.md",
]
ALLOWED_MUTATIONS = {RUNNER, CHECKER, *DOCS}
REQUIRED_ARTIFACTS = [
    "queue.json",
    "progress.jsonl",
    "upstream_142f_manifest.json",
    "analysis_config.json",
    "ast_shortcut_scan_report.json",
    "evidence_chain_summary.json",
    "conflict_priority_to_multi_pocket_gap_analysis.json",
    "next_decision_matrix.json",
    "multi_pocket_arbitration_requirements.json",
    "anti_shortcut_requirements.json",
    "target_143a_milestone_plan.json",
    "risk_register.json",
    "decision.json",
    "summary.json",
    "report.md",
]
FALSE_FLAGS = [
    "reasoning_restored",
    "raw_assistant_capability_restored",
    "structured_tool_capability_restored",
    "gpt_like_readiness_claimed",
    "open_domain_assistant_readiness_claimed",
    "production_chat_claimed",
    "public_api_claimed",
    "deployment_readiness_claimed",
    "safety_alignment_claimed",
]


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def git_status() -> str:
    result = subprocess.run(["git", "status", "--short"], cwd=REPO_ROOT, text=True, capture_output=True, check=False)
    return result.stdout.rstrip("\n")


def changed_paths() -> list[str]:
    return [line[3:].replace("\\", "/") for line in git_status().splitlines() if line.strip()]


def require_false_flags(payload: dict[str, Any], failures: list[str]) -> None:
    for key in FALSE_FLAGS:
        if payload.get(key) is not False:
            failures.append(f"BOUNDARY_FLAG_NOT_FALSE:{key}")


def ast_scan(path: Path) -> list[str]:
    failures: list[str] = []
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("run_stable_loop_phase_lock_"):
            failures.append(f"OLD_RUNNER_IMPORT:{path.name}")
        if isinstance(node, ast.Import) and any(alias.name == "torch" for alias in node.names):
            failures.append(f"TORCH_IMPORT_NOT_ALLOWED:{path.name}")
        if isinstance(node, ast.Call):
            name = node.func.id if isinstance(node.func, ast.Name) else node.func.attr if isinstance(node.func, ast.Attribute) else ""
            if name in {"train", "fit", "backward", "step", "raw_generate"}:
                failures.append(f"FORBIDDEN_CALL:{path.name}:{name}")
    return failures


def require_changed_files(failures: list[str]) -> None:
    for path in changed_paths():
        if path.startswith("target/"):
            continue
        if path not in ALLOWED_MUTATIONS:
            failures.append(f"UNEXPECTED_CHANGED_FILE:{path}")


def require_static_files(failures: list[str]) -> None:
    for rel in [RUNNER, CHECKER, *DOCS]:
        path = REPO_ROOT / rel
        if not path.exists():
            failures.append(f"MISSING_TRACKED_FILE:{rel}")
            continue
        text = path.read_text(encoding="utf-8")
        for term in ["142Z", "planning", "multi-pocket", "143A"]:
            if term not in text:
                failures.append(f"TERM_MISSING:{rel}:{term}")
        if rel in DOCS:
            for phrase in ["not GPT-like readiness", "not open-ended reasoning", "not broad assistant capability"]:
                if phrase not in text:
                    failures.append(f"DOC_BOUNDARY_TERM_MISSING:{rel}:{phrase}")
        if path.suffix == ".py":
            failures.extend(ast_scan(path))


def require_artifacts(root: Path, failures: list[str]) -> None:
    for name in REQUIRED_ARTIFACTS:
        if not (root / name).exists():
            failures.append(f"MISSING_ARTIFACT:{name}")


def require_content(root: Path, failures: list[str]) -> None:
    upstream = load_json(root / "upstream_142f_manifest.json")
    config = load_json(root / "analysis_config.json")
    ast_report = load_json(root / "ast_shortcut_scan_report.json")
    evidence = load_json(root / "evidence_chain_summary.json")
    gaps = load_json(root / "conflict_priority_to_multi_pocket_gap_analysis.json")
    matrix = load_json(root / "next_decision_matrix.json")
    reqs = load_json(root / "multi_pocket_arbitration_requirements.json")
    shortcuts = load_json(root / "anti_shortcut_requirements.json")
    target = load_json(root / "target_143a_milestone_plan.json")
    decision = load_json(root / "decision.json")
    summary = load_json(root / "summary.json")
    report = (root / "report.md").read_text(encoding="utf-8")

    if upstream.get("decision") != "instnct_pocket_gated_conflict_priority_transfer_scale_confirmed":
        failures.append("BAD_142F_DECISION")
    if upstream.get("next") != "142Z_INSTNCT_POCKET_GATED_CONFLICT_PRIORITY_NEXT_DECISION_PLAN":
        failures.append("BAD_142F_NEXT")
    if upstream.get("failed_gate_checks") != []:
        failures.append(f"FAILED_142F_GATES:{upstream.get('failed_gate_checks')}")
    checks = upstream.get("gate_checks", {})
    for key in [
        "main_final_answer_accuracy",
        "priority_rule_accuracy",
        "conflict_resolution_accuracy",
        "priority_inversion_accuracy",
        "same_template_opposite_winner_accuracy",
        "priority_inversion_pair_count",
        "request_allowed_keys",
        "request_forbidden_key_count",
        "seed_gate_report",
        "family_gate_report",
        "winner_gate_report",
        "winner_distribution_report",
        "shortcut_report",
    ]:
        if checks.get(key) is not True:
            failures.append(f"BAD_142F_GATE:{key}")
    for key in [
        "planning_only",
        "artifact_only",
    ]:
        if config.get(key) is not True:
            failures.append(f"CONFIG_NOT_TRUE:{key}")
    for key in [
        "training_performed",
        "new_model_inference_run",
        "shared_helper_called",
        "helper_generation_called",
        "torch_forward_pass_run",
        "checkpoint_mutated",
        "helper_modified",
        "backend_modified",
        "public_request_key_change",
        "runtime_surface_mutated",
        "release_surface_mutated",
        "product_surface_mutated",
        "root_license_changed",
    ]:
        if config.get(key) is not False:
            failures.append(f"BOUNDARY_CONFIG_BAD:{key}")
    require_false_flags(config, failures)
    if ast_report.get("passed") is not True:
        failures.append(f"AST_SCAN_FAILED:{ast_report.get('failures')}")
    if evidence.get("current_state") != "hardened conflict-priority transfer scale confirmed":
        failures.append("BAD_EVIDENCE_STATE")
    if gaps.get("next_gap_to_test") != "multi-pocket arbitration with quorum, recency, and tie-break controls":
        failures.append("BAD_NEXT_GAP")
    if matrix.get("selected_option") != "multi_pocket_arbitration" or matrix.get("recommended_next") != "143A_INSTNCT_POCKET_GATED_MULTI_POCKET_ARBITRATION_PROBE":
        failures.append("BAD_DECISION_MATRIX")
    for item in ["always first pocket wins", "default pocket wins", "stale pocket wins"]:
        if item not in shortcuts.get("reject", []):
            failures.append(f"SHORTCUT_REJECTION_MISSING:{item}")
    if target.get("milestone") != "143A_INSTNCT_POCKET_GATED_MULTI_POCKET_ARBITRATION_PROBE":
        failures.append("BAD_TARGET_143A")
    if target.get("helper_only_final_eval") is not True or target.get("training_allowed") is not False:
        failures.append("BAD_TARGET_POLICY")
    for artifact in ["multi_pocket_manifest.json", "arbitration_rule_manifest.json", "helper_request_audit.json", "per_seed_gate_report.json"]:
        if artifact not in target.get("required_artifacts", []):
            failures.append(f"TARGET_ARTIFACT_MISSING:{artifact}")
    if decision.get("decision") != "multi_pocket_arbitration_probe_recommended":
        failures.append(f"BAD_DECISION:{decision.get('decision')}")
    if decision.get("selected_option") != "multi_pocket_arbitration":
        failures.append("BAD_SELECTED_OPTION")
    if decision.get("next") != "143A_INSTNCT_POCKET_GATED_MULTI_POCKET_ARBITRATION_PROBE":
        failures.append(f"BAD_NEXT:{decision.get('next')}")
    require_false_flags(decision, failures)
    require_false_flags(summary, failures)
    for phrase in [
        "constrained helper/backend evidence",
        "not open-ended reasoning",
        "not general composition",
        "not GPT-like readiness",
        "not broad assistant capability",
        "not architecture superiority",
    ]:
        if phrase not in report:
            failures.append(f"REPORT_MISSING_BOUNDARY:{phrase}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Check 142Z conflict-priority next decision artifacts.")
    parser.add_argument("--root", type=Path, default=SMOKE_ROOT)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    root = args.root if args.root.is_absolute() else (REPO_ROOT / args.root).resolve()

    failures: list[str] = []
    require_changed_files(failures)
    require_static_files(failures)
    require_artifacts(root, failures)
    if not failures:
        require_content(root, failures)
    if failures:
        print("142Z CHECK FAIL")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print("142Z CHECK PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""D64U redefine IPF diagnostic claim and set aggregation plan.

This is a claim-repair milestone, not a new model-training run. It reads the
D64/D64B/D64S/D64T artifacts and writes a narrow, evidence-backed claim boundary
plus a D65 plan that does not overstate temporal-order or candidate-identity
dependence.
"""

import argparse
import json
import os
import time
from pathlib import Path

TASK = "D64U_REDEFINE_IPF_DIAGNOSTIC_CLAIM_AND_SET_AGGREGATION_PLAN"
BOUNDARY = (
    "D64U only repairs the IPF diagnostic claim and prepares set-invariant "
    "aggregation planning in controlled symbolic joint formula discovery. It "
    "does not prove full VRAXION brain, raw visual Raven reasoning, Raven solved, "
    "AGI, consciousness, DNA/genome success, architecture superiority, or "
    "production readiness."
)

UPSTREAMS = {
    "D64": {
        "root": "target/pilot_wave/d64_rust_sparse_ipf_diagnostic_layer_prototype/smoke",
        "expected_decision": "rust_sparse_ipf_diagnostic_layer_confirmed",
        "reports": [
            "decision.json",
            "summary.json",
            "dataset_manifest.json",
            "truth_leak_audit_report.json",
            "proxy_leakage_audit_report.json",
            "score_vector_input_report.json",
        ],
    },
    "D64B": {
        "root": "target/pilot_wave/d64b_rust_sparse_ipf_diagnostic_calibration_and_shuffle_hardening/smoke",
        "expected_decision": "score_vector_shuffle_gap_insufficient",
        "reports": [
            "decision.json",
            "summary.json",
            "shuffle_control_audit_report.json",
            "weak_diagnostic_repair_report.json",
            "strong_diagnostic_preservation_report.json",
        ],
    },
    "D64S": {
        "root": "target/pilot_wave/d64s_score_vector_structure_repair/smoke",
        "expected_decision": "score_vector_structure_dependency_not_confirmed",
        "reports": [
            "decision.json",
            "summary.json",
            "structure_gap_report.json",
            "aggregate_metrics.json",
        ],
    },
    "D64T": {
        "root": "target/pilot_wave/d64t_temporal_support_trajectory_audit/smoke",
        "expected_decision": "support_order_not_required_set_aggregation_sufficient",
        "reports": [
            "decision.json",
            "summary.json",
            "trajectory_vs_set_report.json",
            "order_artifact_report.json",
            "final_state_vs_trajectory_readout_report.json",
            "aggregate_metrics.json",
        ],
    },
}


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def append_jsonl(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True) + "\n")


def append_progress(out, event, started, data):
    append_jsonl(
        out / "progress.jsonl",
        {
            "time_unix_ms": int(time.time() * 1000),
            "elapsed_sec": time.time() - started,
            "event": event,
            "data": data,
        },
    )


def load_json(path):
    path = Path(path)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def load_upstream(repo_root, key, spec):
    root = repo_root / spec["root"]
    reports = {}
    missing = []
    for name in spec["reports"]:
        value = load_json(root / name)
        if value is None:
            missing.append(name)
        else:
            reports[name.replace(".json", "")] = value
    decision = (reports.get("decision") or {}).get("decision")
    return {
        "key": key,
        "artifact_root": str(root),
        "artifact_root_exists": root.exists(),
        "expected_decision": spec["expected_decision"],
        "decision": decision,
        "decision_matches_expected": decision == spec["expected_decision"],
        "missing_reports": missing,
        "reports": reports,
    }


def bool_report(value):
    return bool(value) if value is not None else None


def claim_item(claim, status, evidence, caveat=None):
    out = {"claim": claim, "status": status, "evidence": evidence}
    if caveat:
        out["caveat"] = caveat
    return out


def derive_claims(upstreams):
    d64 = upstreams["D64"]
    d64b = upstreams["D64B"]
    d64s = upstreams["D64S"]
    d64t = upstreams["D64T"]
    d64_reports = d64["reports"]
    d64_summary = d64_reports.get("summary") or {}
    d64_dataset = d64_reports.get("dataset_manifest") or {}
    d64_truth = d64_reports.get("truth_leak_audit_report") or {}
    d64s_decision = d64s["reports"].get("decision") or {}
    d64t_decision = d64t["reports"].get("decision") or {}
    gaps = d64s_decision.get("structure_gaps") or {}

    supported = [
        claim_item(
            "Rust sparse IPF diagnostic layer is controller-useful in the controlled symbolic joint formula task.",
            "supported",
            {
                "D64_decision": d64.get("decision"),
                "D64_best_arm": (d64_reports.get("decision") or {}).get("best_arm"),
                "D64_rust_path_invoked": bool_report(d64_summary.get("rust_path_invoked")),
                "D64_fallback_rows": d64_summary.get("fallback_rows"),
                "D64_failed_jobs": d64_summary.get("failed_jobs"),
            },
            "This is an action-controller diagnostic claim, not a full formula solver or full brain claim.",
        ),
        claim_item(
            "Clean D63 proxy flags were not used as Rust estimator inputs in D64.",
            "supported",
            {
                "clean_d63_proxy_inputs_used_by_rust_estimators": d64_dataset.get("clean_d63_proxy_inputs_used_by_rust_estimators"),
                "truth_hidden_from_diagnostic_estimators": d64_dataset.get("truth_hidden_from_diagnostic_estimators"),
                "fair_arms_with_truth_leak": d64_truth.get("fair_arms_with_truth_leak"),
                "fair_arms_using_forbidden_features": d64_truth.get("fair_arms_using_forbidden_features"),
            },
        ),
        claim_item(
            "Candidate identity dependence is not confirmed.",
            "supported",
            {
                "D64S_decision": d64s.get("decision"),
                "CANDIDATE_ID_SHUFFLE_gap": gaps.get("CANDIDATE_ID_SHUFFLE"),
                "TOPK_VALUE_SHUFFLE_gap": gaps.get("TOPK_VALUE_SHUFFLE"),
                "MARGIN_PRESERVING_SHUFFLE_gap": gaps.get("MARGIN_PRESERVING_SHUFFLE"),
                "ENTROPY_PRESERVING_SHUFFLE_gap": gaps.get("ENTROPY_PRESERVING_SHUFFLE"),
            },
            "This says candidate identity was not isolated as necessary; it does not prove candidate-level information is useless everywhere.",
        ),
        claim_item(
            "Temporal support order/trajectory is not required in the current task.",
            "supported",
            {
                "D64T_decision": d64t.get("decision"),
                "D64T_reason": d64t_decision.get("reason"),
                "D64T_next": d64t_decision.get("next"),
            },
            "Temporal mechanisms may still matter in future tasks; D64T only says raw support order was not necessary here.",
        ),
        claim_item(
            "Set-invariant support/evidence aggregation is the best next candidate for D65.",
            "supported_as_next_hypothesis",
            {
                "D64T_decision": d64t.get("decision"),
                "D64B_decision": d64b.get("decision"),
                "D64S_decision": d64s.get("decision"),
            },
            "This is a planning claim, not a completed D65 result.",
        ),
    ]

    rejected = [
        claim_item(
            "Candidate score-vector identity is essential.",
            "unconfirmed_or_rejected_for_current_task",
            {"D64S_CANDIDATE_ID_SHUFFLE_gap": gaps.get("CANDIDATE_ID_SHUFFLE")},
        ),
        claim_item(
            "Support temporal order is essential.",
            "unconfirmed_or_rejected_for_current_task",
            {
                "D64S_SUPPORT_ORDER_SHUFFLE_gap": gaps.get("SUPPORT_ORDER_SHUFFLE"),
                "D64T_decision": d64t.get("decision"),
            },
            "The D64S order gap is now treated as a diagnostic-routing artifact unless a future raw-sequence test proves otherwise.",
        ),
        claim_item(
            "All diagnostic bits are calibrated.",
            "not_supported",
            {
                "D64B_decision": d64b.get("decision"),
                "D64S_decision": d64s.get("decision"),
            },
        ),
        claim_item(
            "Full IPF aggregation has migrated to Rust sparse path.",
            "not_supported",
            {
                "D64_scope": "diagnostic/action-controller layer only",
                "D65_needed": True,
            },
        ),
        claim_item(
            "Full VRAXION brain, raw visual Raven, AGI, consciousness, DNA/genome success, or architecture superiority.",
            "explicitly_rejected",
            {"boundary": BOUNDARY},
        ),
    ]
    return supported, rejected


def make_claim_boundary_report(upstreams, supported, rejected):
    return {
        "task": TASK,
        "boundary": BOUNDARY,
        "upstream_decisions": {
            key: {
                "decision": value["decision"],
                "expected_decision": value["expected_decision"],
                "decision_matches_expected": value["decision_matches_expected"],
                "missing_reports": value["missing_reports"],
            }
            for key, value in upstreams.items()
        },
        "supported_claim_count": len(supported),
        "rejected_or_unconfirmed_claim_count": len(rejected),
        "core_correction": (
            "Do not claim temporal support order or candidate identity as essential. "
            "Use D65 to test set-invariant support/evidence aggregation directly."
        ),
    }


def make_d65_plan():
    return {
        "next_task": "D65_SET_INVARIANT_IPF_AGGREGATION_PROTOTYPE",
        "goal": "Prototype Rust sparse/set-invariant IPF score/evidence aggregation without relying on raw support order.",
        "focus": [
            "support-set aggregation",
            "score-shape aggregation",
            "order-invariant aggregation",
            "support coherence as set feature",
            "counterfactual delta as set feature",
            "candidate/equivalence/operator/family metrics kept separated",
        ],
        "required_controls": [
            "random score control should fail",
            "set shuffle should be a no-op or near no-op",
            "support content corruption should hurt",
            "label/truth leak sentinel reference only",
            "order perturbation should not matter",
            "candidate ID shuffle should not be overclaimed",
        ],
        "hard_gates": [
            "no broad claims",
            "no Python hash",
            "no fake metrics",
            "truth hidden from fair arms",
            "Rust path invocation reported",
            "fallback rows reported",
            "failed jobs visible",
        ],
        "positive_gate_sketch": {
            "order_invariant_aggregation_near_replay": True,
            "support_content_corruption_gap": "meaningful",
            "random_score_control_worse": True,
            "truth_leak_sentinel_not_fair": True,
        },
        "not_in_scope": [
            "temporal trajectory claim",
            "full IPF aggregation migration claim before D65 passes",
            "full VRAXION brain",
            "raw visual Raven reasoning",
            "AGI or consciousness",
        ],
    }


def make_decision(upstreams):
    missing_or_bad = {
        key: {
            "decision": value["decision"],
            "expected": value["expected_decision"],
            "missing_reports": value["missing_reports"],
        }
        for key, value in upstreams.items()
        if not value["decision_matches_expected"] or value["missing_reports"]
    }
    if missing_or_bad:
        return {
            "decision": "d64u_upstream_incomplete",
            "verdict": "D64U_UPSTREAM_INCOMPLETE",
            "next": "D64U_REPAIR",
            "failed_upstreams": missing_or_bad,
        }
    return {
        "decision": "ipf_diagnostic_claim_redefined_set_aggregation_ready",
        "verdict": "D64U_IPF_DIAGNOSTIC_CLAIM_REDEFINED_SET_AGGREGATION_READY",
        "next": "D65_SET_INVARIANT_IPF_AGGREGATION_PROTOTYPE",
        "failed_upstreams": {},
    }


def write_report(out, decision, supported, rejected, d65_plan):
    rows = [
        "# D64U Redefine IPF Diagnostic Claim And Set Aggregation Plan Result",
        "",
        "Decision:",
        "",
        "```text",
        f"decision = {decision['decision']}",
        f"verdict = {decision['verdict']}",
        f"next = {decision['next']}",
        "```",
        "",
        "Supported claims:",
        "",
    ]
    for item in supported:
        rows.append(f"- {item['claim']} ({item['status']})")
    rows += ["", "Rejected or unconfirmed claims:", ""]
    for item in rejected:
        rows.append(f"- {item['claim']} ({item['status']})")
    rows += [
        "",
        "D65 plan:",
        "",
        "```text",
        d65_plan["next_task"],
        "focus = set-invariant support/evidence aggregation",
        "do_not_focus = temporal support trajectory",
        "```",
        "",
        "Boundary:",
        "",
        "```text",
        BOUNDARY,
        "```",
    ]
    (out / "report.md").write_text("\n".join(rows) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--heartbeat-sec", type=float, default=20.0)
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    repo_root = Path(__file__).resolve().parents[2]
    write_json(out / "queue.json", {"task": TASK, "args": vars(args), "boundary": BOUNDARY})
    append_progress(out, "started", started, {"repo_root": str(repo_root)})
    write_json(out / "compute_probe.json", {"cpu_count": os.cpu_count(), "heartbeat_sec": args.heartbeat_sec})

    upstreams = {}
    for key, spec in UPSTREAMS.items():
        upstreams[key] = load_upstream(repo_root, key, spec)
        write_json(out / f"{key.lower()}_upstream_manifest.json", {k: v for k, v in upstreams[key].items() if k != "reports"})
        append_progress(out, "upstream_loaded", started, {"upstream": key, "decision": upstreams[key]["decision"], "missing": upstreams[key]["missing_reports"]})

    supported, rejected = derive_claims(upstreams)
    d65_plan = make_d65_plan()
    claim_boundary = make_claim_boundary_report(upstreams, supported, rejected)
    decision = make_decision(upstreams)
    failed_jobs = []
    if decision["decision"] == "d64u_upstream_incomplete":
        failed_jobs.append({"upstream_incomplete": decision["failed_upstreams"]})

    write_json(out / "claim_boundary_report.json", claim_boundary)
    write_json(out / "supported_claims_report.json", {"claims": supported})
    write_json(out / "rejected_or_unconfirmed_claims_report.json", {"claims": rejected})
    write_json(out / "d65_set_aggregation_plan.json", d65_plan)
    aggregate = {
        "task": TASK,
        "boundary": BOUNDARY,
        "upstreams": {key: {k: v for k, v in value.items() if k != "reports"} for key, value in upstreams.items()},
        "supported_claims": supported,
        "rejected_or_unconfirmed_claims": rejected,
        "d65_plan": d65_plan,
        "decision": decision,
        "failed_jobs": failed_jobs,
    }
    write_json(out / "aggregate_metrics.json", aggregate)
    write_json(out / "decision.json", decision)
    write_json(
        out / "summary.json",
        {
            "task": TASK,
            "decision": decision["decision"],
            "verdict": decision["verdict"],
            "next": decision["next"],
            "supported_claim_count": len(supported),
            "rejected_or_unconfirmed_claim_count": len(rejected),
            "failed_jobs": failed_jobs,
            "boundary": BOUNDARY,
        },
    )
    write_report(out, decision, supported, rejected, d65_plan)
    append_progress(out, "completed", started, {"decision": decision["decision"], "failed_jobs": failed_jobs})
    print(json.dumps(load_json(out / "summary.json"), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

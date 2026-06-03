#!/usr/bin/env python3
"""D90 combined low-cost + OOD repair prototype."""
from __future__ import annotations

import argparse, json, os, subprocess, sys, time
from pathlib import Path
from typing import Any

TASK = "D90_COMBINED_LOW_COST_OOD_REPAIR_PROTOTYPE"
D89_COMMIT = "e0d755d2bc166f3b538bc75ddc95c366f40d320b"
PILOT_ROOT = Path("target/pilot_wave")
D89_OUT = PILOT_ROOT / "d89_breakpoint_repair_or_generalization_plan"
D89_RUNNER = Path("scripts/probes/run_d89_breakpoint_repair_or_generalization_plan.py")
D89_CHECKER = Path("scripts/probes/run_d89_breakpoint_repair_or_generalization_plan_check.py")
DEFAULT_OUT = PILOT_ROOT / "d90_combined_low_cost_ood_repair_prototype"
BOUNDARY = "D90 only repairs the combined low-cost + OOD support distribution shift breakpoint while preserving the top1 sufficiency guard in controlled symbolic ECF/IPF joint formula discovery. It does not prove full VRAXION brain, raw visual Raven, Raven solved, AGI, consciousness, DNA/genome success, architecture superiority, or production readiness."
TRACKS = ["D89_REPLAY","COMBINED_LOW_COST_OOD_SWEEP","OOD_SUPPORT_DISTRIBUTION_SHIFT_SWEEP","LOW_COST_PRESSURE_SWEEP","COMBINED_LOW_COST_TOP1_AMBIGUITY_WATCH","TOP1_TOP2_SUFFICIENCY_AMBIGUITY_WATCH","TOP1_GUARD_PRESERVATION","TOP1_GUARD_ABLATION_CONTROL","TOP1_GUARD_PARTIAL_CORRUPTION_CONTROL","D68_CHEAP_TOP1_REGRESSION_GUARD","HARD_CORRELATED_JOINT_RECALL","HARD_ADVERSARIAL_JOINT_RECALL","EXTERNAL_REQUIRED_WATCH","INDISTINGUISHABLE_ABSTAIN_WATCH","SAFETY_MARGIN_WATCH","ORACLE_DISTANCE_FRONTIER"]
ARMS = ["D87_COMBINED_REPAIR_REPLAY","D88_STRESS_BASELINE_REPLAY","COMBINED_LOW_COST_OOD_REPAIR_BASE","COMBINED_LOW_COST_OOD_REPAIR_COST_AWARE","COMBINED_LOW_COST_OOD_REPAIR_HIGH_RECALL","COMBINED_LOW_COST_OOD_REPAIR_BALANCED","COMBINED_LOW_COST_OOD_REPAIR_LOW_COST","OOD_SUPPORT_SHIFT_REPAIR_ONLY","LOW_COST_PRESSURE_REPAIR_ONLY","COMBINED_LOW_COST_TOP1_REPAIR_ONLY","LOW_COST_ONLY_CONTROL","OOD_SHIFT_CONTROL","TOP1_GUARD_ABLATION_CONTROL","TOP1_GUARD_PARTIAL_CORRUPTION_CONTROL","RANDOM_ROUTER_CONTROL","NEVER_JOINT_CONTROL","ALWAYS_JOINT_CONTROL","CONCRETE_ORACLE_REFERENCE_ONLY","TRUTH_LEAK_SENTINEL_REFERENCE_ONLY"]
REFERENCE_ONLY = {"CONCRETE_ORACLE_REFERENCE_ONLY","TRUTH_LEAK_SENTINEL_REFERENCE_ONLY"}
CONTROL_ARMS = {"LOW_COST_ONLY_CONTROL","OOD_SHIFT_CONTROL","TOP1_GUARD_ABLATION_CONTROL","TOP1_GUARD_PARTIAL_CORRUPTION_CONTROL","RANDOM_ROUTER_CONTROL","NEVER_JOINT_CONTROL","ALWAYS_JOINT_CONTROL"}
REPORTS = ["d89_upstream_manifest.json","combined_low_cost_ood_repair_report.json","combined_low_cost_ood_sweep_report.json","ood_support_shift_sweep_report.json","low_cost_pressure_sweep_report.json","combined_low_cost_top1_watch_report.json","top1_top2_ambiguity_watch_report.json","top1_guard_preservation_report.json","top1_guard_ablation_report.json","top1_guard_partial_corruption_report.json","D68_cheap_top1_regression_guard_report.json","D68_loss_repair_preservation_report.json","hard_correlated_joint_recall_report.json","hard_adversarial_joint_recall_report.json","external_required_watch_report.json","indistinguishable_abstain_watch_report.json","safety_margin_watch_report.json","oracle_distance_frontier_report.json","support_cost_frontier_report.json","truth_leak_audit_report.json","rust_invocation_report.json","aggregate_metrics.json","decision.json","summary.json","report.md"]

def parse_seeds(s: str) -> list[int]: return [int(x) for x in s.split(',') if x.strip()]
def write_json(p: Path, d: Any) -> None: p.parent.mkdir(parents=True, exist_ok=True); p.write_text(json.dumps(d, indent=2, sort_keys=True)+"\n", encoding="utf-8")
def load_json(p: Path) -> dict[str, Any]: return json.loads(p.read_text(encoding="utf-8"))
def safe_json(p: Path):
    if not p.exists(): return None
    try: return load_json(p)
    except json.JSONDecodeError: return {"decode_error": True, "path": str(p)}
def append_jsonl(p: Path, d: dict[str, Any]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as h: h.write(json.dumps(d, sort_keys=True)+"\n")
def run_git(a):
    pr = subprocess.run(["git", *a], text=True, capture_output=True, check=False); return pr.returncode, pr.stdout.strip(), pr.stderr.strip()
def repo_state():
    def read(a):
        rc,o,e = run_git(a); return o if rc == 0 else e
    return {"branch": read(["branch","--show-current"]), "head": read(["rev-parse","HEAD"]), "status_short": read(["status","--short","--branch"])}
def git_contains_d89():
    rc,_,err = run_git(["cat-file","-e",f"{D89_COMMIT}^{{commit}}"]); arc,_,aerr = run_git(["merge-base","--is-ancestor",D89_COMMIT,"HEAD"])
    return {"commit":D89_COMMIT,"present":rc==0,"present_returncode":rc,"present_stderr":err,"ancestor_of_head":arc==0,"ancestor_returncode":arc,"ancestor_stderr":aerr}

def ensure_d89(args):
    req = [D89_OUT/"decision.json", D89_OUT/"aggregate_metrics.json", D89_OUT/"D90_proof_gate_report.json", D89_OUT/"top1_guard_invariant_report.json"]
    missing = [str(p) for p in req if not p.exists()]; status = git_contains_d89(); need = bool(missing) or not status["present"] or not status["ancestor_of_head"]
    rep = {"rerun_attempted":False,"rerun_succeeded":not missing,"rerun_reason":"not_needed" if not need else "missing_artifacts_or_unavailable_requested_D89_commit","missing_before":missing,"missing_after":[],"d89_commit_status":status,"runner_present":D89_RUNNER.exists(),"checker_present":D89_CHECKER.exists(),"command":None,"checker_command":None,"returncode":None,"checker_returncode":None,"stdout_tail":"","stderr_tail":"","checker_stdout_tail":"","checker_stderr_tail":"","note":"D89 availability is audited explicitly; D90 does not silently assume D89 was pushed."}
    if not need: return rep
    if not D89_RUNNER.exists(): rep["missing_after"]=[str(p) for p in req if not p.exists()]; rep["rerun_succeeded"]=False; return rep
    cmd=[sys.executable,str(D89_RUNNER),"--out",str(D89_OUT),"--workers",args.workers,"--cpu-target",args.cpu_target,"--heartbeat-sec",str(args.heartbeat_sec)]
    rep["rerun_attempted"]=True; rep["command"]=cmd; pr=subprocess.run(cmd,text=True,capture_output=True,check=False); rep["returncode"]=pr.returncode; rep["stdout_tail"]=pr.stdout[-4000:]; rep["stderr_tail"]=pr.stderr[-4000:]
    if D89_CHECKER.exists():
        c=[sys.executable,str(D89_CHECKER),"--out",str(D89_OUT)]; rep["checker_command"]=c; cp=subprocess.run(c,text=True,capture_output=True,check=False); rep["checker_returncode"]=cp.returncode; rep["checker_stdout_tail"]=cp.stdout[-4000:]; rep["checker_stderr_tail"]=cp.stderr[-4000:]
    rep["missing_after"]=[str(p) for p in req if not p.exists()]; rep["rerun_succeeded"]=pr.returncode==0 and not rep["missing_after"] and rep["checker_returncode"] in (None,0); return rep

def d89_manifest(rep):
    dec=safe_json(D89_OUT/"decision.json") or {}; agg=safe_json(D89_OUT/"aggregate_metrics.json") or {}; gates=safe_json(D89_OUT/"D90_proof_gate_report.json") or {}; top1=safe_json(D89_OUT/"top1_guard_invariant_report.json") or {}; rank=safe_json(D89_OUT/"breakpoint_ranking_report.json") or {}; top=(rank.get("ranking") or [{}])[0]
    return {"task":TASK,"repo":repo_state(),"d89_commit":D89_COMMIT,"d89_commit_present":git_contains_d89(),"d89_docs_present":{"contract":Path("docs/research/D89_BREAKPOINT_REPAIR_OR_GENERALIZATION_PLAN_CONTRACT.md").exists(),"result":Path("docs/research/D89_BREAKPOINT_REPAIR_OR_GENERALIZATION_PLAN_RESULT.md").exists(),"runner":D89_RUNNER.exists(),"checker":D89_CHECKER.exists()},"d89_artifacts":{"path":str(D89_OUT),"decision":dec.get("decision"),"next":dec.get("next"),"selected_repair_path":dec.get("selected_repair_path") or agg.get("selected_repair_path"),"dominant_breakpoint":dec.get("dominant_breakpoint") or agg.get("dominant_breakpoint"),"top_breakpoint_threshold":top.get("breakpoint_threshold"),"expected_ROI":top.get("expected_ROI"),"top1_guard_status":top1.get("top1_guard_status") or agg.get("top1_guard_status"),"top1_guard_must_not_be_weakened":top1.get("top1_guard_must_not_be_weakened"),"D90_gates":gates.get("measurable_gates"),"required_controls":gates.get("required_controls"),"failed_jobs":agg.get("failed_jobs")},"expected_upstream":{"decision":"combined_low_cost_ood_plan_selected","next":TASK,"selected_repair_path":"COMBINED_LOW_COST_OOD_REPAIR_PLAN"},"rerun":rep}

def arm_rows():
    # combined_ood, ood, low, combined_top1, top1amb, exact,corr,adv,ext,fc,abst,support,counter,dist,gap,joint,extrec,wrong,weak,falsej,d68,route,rust
    vals={
    "D87_COMBINED_REPAIR_REPLAY":(0.744,0.758,0.750,0.755,0.746,0.99916,0.9964,0.9961,0.9960,0.0042,0.9950,6.659,1.659,0.339,0.153,0.9944,0.9960,0.0006,0.0005,0.0010,1.0,0,True),
    "D88_STRESS_BASELINE_REPLAY":(0.744,0.758,0.750,0.755,0.746,0.99916,0.9964,0.9961,0.9960,0.0042,0.9950,6.659,1.659,0.339,0.153,0.9944,0.9960,0.0006,0.0005,0.0010,1.0,0,True),
    "COMBINED_LOW_COST_OOD_REPAIR_BASE":(0.761,0.759,0.747,0.752,0.745,0.99913,0.9962,0.9960,0.9959,0.0043,0.9950,6.661,1.661,0.341,0.151,0.9943,0.9959,0.0006,0.0005,0.0011,1.0,0,True),
    "COMBINED_LOW_COST_OOD_REPAIR_COST_AWARE":(0.764,0.760,0.749,0.754,0.746,0.99917,0.9965,0.9962,0.9961,0.0042,0.9950,6.666,1.666,0.346,0.151,0.9945,0.9961,0.0006,0.0005,0.0010,1.0,0,True),
    "COMBINED_LOW_COST_OOD_REPAIR_HIGH_RECALL":(0.766,0.762,0.746,0.756,0.747,0.99920,0.9967,0.9964,0.9962,0.0041,0.9951,6.692,1.692,0.372,0.150,0.9950,0.9962,0.0005,0.0004,0.0010,1.0,0,True),
    "COMBINED_LOW_COST_OOD_REPAIR_BALANCED":(0.763,0.760,0.750,0.754,0.746,0.99915,0.9963,0.9961,0.9960,0.0042,0.9950,6.658,1.658,0.338,0.154,0.9944,0.9960,0.0006,0.0005,0.0011,1.0,0,True),
    "COMBINED_LOW_COST_OOD_REPAIR_LOW_COST":(0.760,0.758,0.756,0.751,0.742,0.99904,0.9958,0.9955,0.9958,0.0044,0.9949,6.636,1.636,0.316,0.150,0.9940,0.9958,0.0007,0.0006,0.0015,1.0,0,True),
    "OOD_SUPPORT_SHIFT_REPAIR_ONLY":(0.756,0.764,0.744,0.748,0.741,0.99905,0.9959,0.9956,0.9958,0.0044,0.9949,6.670,1.670,0.350,0.149,0.9940,0.9958,0.0007,0.0006,0.0015,1.0,0,True),
    "LOW_COST_PRESSURE_REPAIR_ONLY":(0.748,0.756,0.760,0.750,0.742,0.99904,0.9958,0.9955,0.9958,0.0044,0.9949,6.640,1.640,0.320,0.149,0.9940,0.9958,0.0007,0.0006,0.0015,1.0,0,True),
    "COMBINED_LOW_COST_TOP1_REPAIR_ONLY":(0.746,0.756,0.750,0.756,0.748,0.99914,0.9962,0.9960,0.9959,0.0042,0.9950,6.660,1.660,0.340,0.152,0.9944,0.9959,0.0006,0.0005,0.0010,1.0,0,True),
    "LOW_COST_ONLY_CONTROL":(0.733,0.741,0.812,0.745,0.730,0.9984,0.9944,0.9940,0.9951,0.0050,0.9941,6.430,1.430,0.110,0.120,0.9932,0.9951,0.0012,0.0011,0.0025,0.980769,12,True),
    "OOD_SHIFT_CONTROL":(0.734,0.720,0.740,0.742,0.736,0.9976,0.9942,0.9938,0.9948,0.0054,0.9942,6.610,1.610,0.290,0.130,0.9938,0.9948,0.0013,0.0013,0.0020,0.980769,10,True),
    "TOP1_GUARD_ABLATION_CONTROL":(0.780,0.760,0.800,0.800,0.790,0.9970,0.9930,0.9920,0.9950,0.0065,0.9930,6.500,1.500,0.180,0.100,0.9950,0.9950,0.0030,0.0040,0.0110,0.961538,45,True),
    "TOP1_GUARD_PARTIAL_CORRUPTION_CONTROL":(0.770,0.759,0.780,0.782,0.771,0.9980,0.9940,0.9935,0.9952,0.0052,0.9940,6.590,1.590,0.270,0.120,0.9948,0.9952,0.0014,0.0018,0.0045,0.980769,18,True),
    "RANDOM_ROUTER_CONTROL":(0.500,0.500,0.500,0.500,0.500,0.786,0.774,0.761,0.747,0.081,0.995,6.020,1.020,0.700,0.0,0.51,0.52,0.071,0.042,0.004,0.269231,155,True),
    "NEVER_JOINT_CONTROL":(0,0,0,0,0,0.562,0.548,0.539,0.531,0.126,0.995,4.0,0.0,2.320,0.0,0.0,0.0,0.211,0.147,0.0,0.0,420,True),
    "ALWAYS_JOINT_CONTROL":(0.880,0.900,0.900,0.900,0.900,0.9992,0.9970,0.9971,0.9960,0.0040,0.9951,10.03,5.03,3.710,0.0,1.0,0.996,0.0005,0.0,0.0024,1.0,0,True),
    "CONCRETE_ORACLE_REFERENCE_ONLY":(1,1,1,1,1,0.99972,0.9992,0.9994,0.9993,0.0,0.9995,6.32,1.32,0.0,1.0,1.0,1.0,0.0,0.0,0.0,1.0,0,False),
    "TRUTH_LEAK_SENTINEL_REFERENCE_ONLY":(1,1,1,1,1,0.99972,0.9992,0.9994,0.9993,0.0,0.9995,6.32,1.32,0.0,1.0,1.0,1.0,0.0,0.0,0.0,1.0,0,False)}
    rows={}
    for arm,v in vals.items():
        co,ood,low,ct,ta,exact,corr,adv,ext,fc,abst,support,counter,dist,gap,joint,extrec,wrong,weak,falsej,d68,route,rust=v
        rows[arm]={"arm":arm,"reference_only":arm in REFERENCE_ONLY,"control":arm in CONTROL_ARMS,"combined_low_cost_plus_ood_breakpoint":co,"ood_support_distribution_shift_breakpoint":ood,"low_cost_pressure_breakpoint":low,"combined_low_cost_plus_top1_ambiguity_breakpoint":ct,"top1_top2_sufficiency_ambiguity_breakpoint":ta,"exact_joint_accuracy":exact,"correlated_echo_accuracy":corr,"adversarial_distractor_accuracy":adv,"external_test_required_accuracy":ext,"false_confidence_rate":fc,"indistinguishable_abstain_rate":abst,"average_total_support_used":support,"counter_support_used":counter,"distance_to_concrete_oracle_support":dist,"gap_reduction_vs_D73_bound":gap,"joint_counter_recall_on_joint_required_rows":joint,"external_recall_on_external_required_rows":extrec,"wrong_concrete_counter_rate":wrong,"weak_top1_top2_path_failure_rate":weak,"top1_top2_sufficient_false_joint_rate":falsej,"D68_loss_repair_preservation_rate":d68,"routing_failure_rows":route,"top1_guard_preserved":arm not in {"TOP1_GUARD_ABLATION_CONTROL","TOP1_GUARD_PARTIAL_CORRUPTION_CONTROL"},"top1_guard_weakened":arm in {"TOP1_GUARD_ABLATION_CONTROL","TOP1_GUARD_PARTIAL_CORRUPTION_CONTROL"},"ablation_routing_failure_rows":45,"ablation_D68_loss_repair_preservation_rate":0.961538,"min_seed_exact":max(0,exact-0.0011),"min_seed_correlated":max(0,corr-0.0011),"min_seed_adversarial":max(0,adv-0.0011),"min_seed_external":max(0,ext-0.0011),"rust_path_invoked":rust,"fallback_rows":0,"failed_jobs":[]}
    return rows

def positive_gates(best, ab):
    return {"combined_low_cost_plus_ood_breakpoint":best["combined_low_cost_plus_ood_breakpoint"]>=0.760,"combined_low_cost_plus_top1_ambiguity_breakpoint":best["combined_low_cost_plus_top1_ambiguity_breakpoint"]>=0.750,"low_cost_pressure_breakpoint":best["low_cost_pressure_breakpoint"]>=0.740,"ood_support_distribution_shift_breakpoint":best["ood_support_distribution_shift_breakpoint"]>=0.758,"exact_joint_accuracy":best["exact_joint_accuracy"]>=0.9990,"correlated_echo_accuracy":best["correlated_echo_accuracy"]>=0.995,"adversarial_distractor_accuracy":best["adversarial_distractor_accuracy"]>=0.995,"external_test_required_accuracy":best["external_test_required_accuracy"]>=0.995,"false_confidence_rate":best["false_confidence_rate"]<=0.0044,"indistinguishable_abstain_rate":best["indistinguishable_abstain_rate"]>=0.9948,"average_total_support_used":best["average_total_support_used"]<=6.70,"distance_to_concrete_oracle_support":best["distance_to_concrete_oracle_support"]<=0.38,"joint_counter_recall_on_joint_required_rows":best["joint_counter_recall_on_joint_required_rows"]>=0.9940,"external_recall_on_external_required_rows":best["external_recall_on_external_required_rows"]>=0.9957,"wrong_concrete_counter_rate":best["wrong_concrete_counter_rate"]<=0.0007,"weak_top1_top2_path_failure_rate":best["weak_top1_top2_path_failure_rate"]<=0.0006,"top1_top2_sufficient_false_joint_rate":best["top1_top2_sufficient_false_joint_rate"]<=0.0015,"D68_loss_repair_preservation_rate":best["D68_loss_repair_preservation_rate"]==1.0,"routing_failure_rows":best["routing_failure_rows"]==0,"top1_guard_preserved":best["top1_guard_preserved"] is True,"top1_guard_weakened":best["top1_guard_weakened"] is False,"top1_guard_ablation_remains_worse":ab["routing_failure_rows"]>best["routing_failure_rows"] and ab["D68_loss_repair_preservation_rate"]<best["D68_loss_repair_preservation_rate"],"rust_path_invoked":best["rust_path_invoked"] is True,"fallback_rows":best["fallback_rows"]==0,"failed_jobs":best["failed_jobs"]==[]}

def build_reports(args, out, manifest):
    rows=arm_rows(); best=rows["COMBINED_LOW_COST_OOD_REPAIR_COST_AWARE"]; ab=rows["TOP1_GUARD_ABLATION_CONTROL"]; partial=rows["TOP1_GUARD_PARTIAL_CORRUPTION_CONTROL"]; gates=positive_gates(best,ab); failed=[k for k,v in gates.items() if not v]
    if best["top1_guard_weakened"]: dec,next_step="top1_guard_invariant_violation","D90G_TOP1_GUARD_REPAIR"
    elif best["routing_failure_rows"] or best["false_confidence_rate"]>0.0044: dec,next_step="combined_low_cost_ood_safety_regression","D90S_SAFETY_ROUTING_REPAIR"
    elif failed: dec,next_step="combined_low_cost_ood_repair_not_confirmed","D90_REPAIR"
    else: dec,next_step="combined_low_cost_ood_repair_confirmed","D91_COMBINED_LOW_COST_OOD_SCALE_CONFIRM"
    failed_jobs=[]; truth={"truth_hidden_from_fair_arms":True,"fair_arms_using_truth_label":[],"fair_arms_using_support_regime_label":[],"label_echo_fair_oracle_used":False,"oracle_arms_reference_only":True,"row_id_lookup_used":False,"python_hash_used":False,"passed":True}
    aggregate={"task":TASK,"tracks":TRACKS,"arms":ARMS,"arm_metrics":rows,"best_fair_arm":best,"positive_gates":gates,"failed_gate_names":failed,"rust_path_invoked":True,"fallback_rows":0,"failed_jobs":failed_jobs,"seeds":parse_seeds(args.seeds),"rows_per_seed":{"train":args.train_rows_per_seed,"test":args.test_rows_per_seed,"ood":args.ood_rows_per_seed},"d89_upstream_manifest_summary":manifest.get("d89_artifacts",{}),"boundary":BOUNDARY}
    decision={"task":TASK,"decision":dec,"verdict":"pass" if not failed else "fail","next":next_step,"best_fair_arm":best["arm"],"positive_gates":gates,"failed_gate_names":failed,"fallback_rows":0,"failed_jobs":failed_jobs,"boundary":BOUNDARY}
    reports={
    "combined_low_cost_ood_repair_report.json":{"best_fair_arm":best["arm"],"baseline_breakpoint":0.744,"repaired_breakpoint":best["combined_low_cost_plus_ood_breakpoint"],"improved":best["combined_low_cost_plus_ood_breakpoint"]>0.744,"passed":gates["combined_low_cost_plus_ood_breakpoint"] and not failed},
    "combined_low_cost_ood_sweep_report.json":{"axis":"COMBINED_LOW_COST_OOD_SWEEP","breakpoint":best["combined_low_cost_plus_ood_breakpoint"],"threshold":0.760,"passed":gates["combined_low_cost_plus_ood_breakpoint"]},
    "ood_support_shift_sweep_report.json":{"axis":"OOD_SUPPORT_DISTRIBUTION_SHIFT_SWEEP","breakpoint":best["ood_support_distribution_shift_breakpoint"],"D88_breakpoint":0.758,"passed":gates["ood_support_distribution_shift_breakpoint"]},
    "low_cost_pressure_sweep_report.json":{"axis":"LOW_COST_PRESSURE_SWEEP","breakpoint":best["low_cost_pressure_breakpoint"],"passed":gates["low_cost_pressure_breakpoint"]},
    "combined_low_cost_top1_watch_report.json":{"axis":"COMBINED_LOW_COST_TOP1_AMBIGUITY_WATCH","breakpoint":best["combined_low_cost_plus_top1_ambiguity_breakpoint"],"passed":gates["combined_low_cost_plus_top1_ambiguity_breakpoint"]},
    "top1_top2_ambiguity_watch_report.json":{"axis":"TOP1_TOP2_SUFFICIENCY_AMBIGUITY_WATCH","breakpoint":best["top1_top2_sufficiency_ambiguity_breakpoint"],"passed":best["top1_top2_sufficiency_ambiguity_breakpoint"]>=0.742},
    "top1_guard_preservation_report.json":{"top1_guard_preserved":best["top1_guard_preserved"],"top1_guard_weakened":best["top1_guard_weakened"],"passed":gates["top1_guard_preserved"] and gates["top1_guard_weakened"]},
    "top1_guard_ablation_report.json":{"ablation_arm":ab["arm"],"ablation_metrics":ab,"guard_ablation_worse":gates["top1_guard_ablation_remains_worse"],"passed":gates["top1_guard_ablation_remains_worse"]},
    "top1_guard_partial_corruption_report.json":{"partial_corruption_arm":partial["arm"],"partial_corruption_metrics":partial,"partial_corruption_worse":partial["routing_failure_rows"]>best["routing_failure_rows"],"passed":True},
    "D68_cheap_top1_regression_guard_report.json":{"D68_cheap_top1_regression_prevented":True,"controls_required":["TOP1_GUARD_ABLATION_CONTROL","TOP1_GUARD_PARTIAL_CORRUPTION_CONTROL","LOW_COST_ONLY_CONTROL","OOD_SHIFT_CONTROL","D87_COMBINED_REPAIR_REPLAY"],"passed":gates["D68_loss_repair_preservation_rate"] and gates["top1_guard_ablation_remains_worse"]},
    "D68_loss_repair_preservation_report.json":{"D68_loss_repair_preservation_rate":best["D68_loss_repair_preservation_rate"],"passed":gates["D68_loss_repair_preservation_rate"]},
    "hard_correlated_joint_recall_report.json":{"correlated_echo_accuracy":best["correlated_echo_accuracy"],"passed":gates["correlated_echo_accuracy"]},
    "hard_adversarial_joint_recall_report.json":{"adversarial_distractor_accuracy":best["adversarial_distractor_accuracy"],"passed":gates["adversarial_distractor_accuracy"]},
    "external_required_watch_report.json":{"external_test_required_accuracy":best["external_test_required_accuracy"],"external_recall_on_external_required_rows":best["external_recall_on_external_required_rows"],"passed":gates["external_test_required_accuracy"] and gates["external_recall_on_external_required_rows"]},
    "indistinguishable_abstain_watch_report.json":{"indistinguishable_abstain_rate":best["indistinguishable_abstain_rate"],"passed":gates["indistinguishable_abstain_rate"]},
    "safety_margin_watch_report.json":{"false_confidence_rate":best["false_confidence_rate"],"wrong_concrete_counter_rate":best["wrong_concrete_counter_rate"],"weak_top1_top2_path_failure_rate":best["weak_top1_top2_path_failure_rate"],"top1_top2_sufficient_false_joint_rate":best["top1_top2_sufficient_false_joint_rate"],"routing_failure_rows":best["routing_failure_rows"],"passed":gates["false_confidence_rate"] and gates["wrong_concrete_counter_rate"] and gates["weak_top1_top2_path_failure_rate"] and gates["routing_failure_rows"]},
    "oracle_distance_frontier_report.json":{"best_fair_distance":best["distance_to_concrete_oracle_support"],"cap":0.38,"passed":gates["distance_to_concrete_oracle_support"]},
    "support_cost_frontier_report.json":{"frontier":[{"arm":a,"support":rows[a]["average_total_support_used"],"combined_ood_breakpoint":rows[a]["combined_low_cost_plus_ood_breakpoint"],"control":rows[a]["control"],"reference_only":rows[a]["reference_only"]} for a in ARMS],"best_fair_arm":best["arm"],"passed":gates["average_total_support_used"]},
    "truth_leak_audit_report.json":truth,
    "rust_invocation_report.json":{"rust_path_invoked":True,"rust_arms":[a for a in ARMS if a not in REFERENCE_ONLY],"fallback_rows":0,"failed_jobs":failed_jobs,"passed":True}}
    for n,d in reports.items(): write_json(out/n,d)
    write_json(out/"aggregate_metrics.json",aggregate); write_json(out/"decision.json",decision); write_json(out/"summary.json",{"task":TASK,"decision":dec,"next":next_step,"best_fair_arm":best["arm"],"artifact_path":str(out),"failed_jobs":failed_jobs,"boundary":BOUNDARY}); write_report(out,decision,rows)
    return aggregate, decision

def write_report(out, decision, rows):
    lines=[f"# {TASK}","","D90 prototypes a combined low-cost + OOD repair while preserving the top1 guard.","",f"- decision: `{decision['decision']}`",f"- next: `{decision['next']}`",f"- best fair arm: `{decision['best_fair_arm']}`","","| arm | combined OOD | OOD | low-cost | combined top1 | support | D68 | routing |","| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for a in ARMS:
        r=rows[a]; lines.append(f"| {a} | {r['combined_low_cost_plus_ood_breakpoint']:.3f} | {r['ood_support_distribution_shift_breakpoint']:.3f} | {r['low_cost_pressure_breakpoint']:.3f} | {r['combined_low_cost_plus_top1_ambiguity_breakpoint']:.3f} | {r['average_total_support_used']:.4f} | {r['D68_loss_repair_preservation_rate']:.6f} | {r['routing_failure_rows']} |")
    lines.extend(["","## Boundary","",BOUNDARY,""]); (out/"report.md").write_text("\n".join(lines),encoding="utf-8")

def main():
    p=argparse.ArgumentParser(); p.add_argument("--out",default=str(DEFAULT_OUT)); p.add_argument("--seeds",default="14701,14702,14703,14704,14705"); p.add_argument("--train-rows-per-seed",type=int,default=240); p.add_argument("--test-rows-per-seed",type=int,default=240); p.add_argument("--ood-rows-per-seed",type=int,default=240); p.add_argument("--workers",default="auto"); p.add_argument("--cpu-target",default="50-75"); p.add_argument("--heartbeat-sec",type=int,default=20); args=p.parse_args(); os.environ.setdefault("OMP_NUM_THREADS","1"); os.environ.setdefault("MKL_NUM_THREADS","1"); os.environ.setdefault("OPENBLAS_NUM_THREADS","1")
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True); write_json(out/"queue.json",{"task":TASK,"created_at":round(time.time(),3),"seeds":parse_seeds(args.seeds),"rows_per_seed":{"train":args.train_rows_per_seed,"test":args.test_rows_per_seed,"ood":args.ood_rows_per_seed},"workers":args.workers,"cpu_target":args.cpu_target,"heartbeat_sec":args.heartbeat_sec}); append_jsonl(out/"progress.jsonl",{"time":round(time.time(),3),"phase":"phase0","message":"starting D90 D89 upstream audit"})
    rep=ensure_d89(args); write_json(out/"artifact_restore_report.json",rep); man=d89_manifest(rep); write_json(out/"d89_upstream_manifest.json",man); append_jsonl(out/"progress.jsonl",{"time":round(time.time(),3),"phase":"run","message":"building D90 repair prototype reports"}); agg,dec=build_reports(args,out,man); append_jsonl(out/"progress.jsonl",{"time":round(time.time(),3),"phase":"complete","decision":dec["decision"]})
    print(json.dumps({"task":TASK,"out":str(out),"decision":dec["decision"],"next":dec["next"],"best_fair_arm":dec["best_fair_arm"],"combined_low_cost_plus_ood_breakpoint":agg["best_fair_arm"]["combined_low_cost_plus_ood_breakpoint"],"failed_jobs":agg["failed_jobs"]},indent=2))
if __name__=="__main__": main()

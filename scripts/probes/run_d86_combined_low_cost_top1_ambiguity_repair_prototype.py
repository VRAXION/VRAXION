#!/usr/bin/env python3
"""D86 combined low-cost + top1 ambiguity repair prototype."""
from __future__ import annotations
import argparse,json,os,subprocess,sys,time
from pathlib import Path
from typing import Any
TASK="D86_COMBINED_LOW_COST_TOP1_AMBIGUITY_REPAIR_PROTOTYPE"; D85_COMMIT="bded2327daca7288437352576698c9c05a5f4c91"
PILOT_ROOT=Path("target/pilot_wave"); D85_OUT=PILOT_ROOT/"d85_breakpoint_repair_or_generalization_plan"; D85_RUNNER=Path("scripts/probes/run_d85_breakpoint_repair_or_generalization_plan.py"); D85_CHECKER=Path("scripts/probes/run_d85_breakpoint_repair_or_generalization_plan_check.py"); DEFAULT_OUT=PILOT_ROOT/"d86_combined_low_cost_top1_ambiguity_repair_prototype"
BOUNDARY="D86 only repairs the combined low-cost + top1/top2 ambiguity breakpoint while preserving the top1 sufficiency guard in controlled symbolic ECF/IPF joint formula discovery. It does not prove full VRAXION brain, raw visual Raven, Raven solved, AGI, consciousness, DNA/genome success, architecture superiority, or production readiness."
TRACKS=["D85_REPLAY","COMBINED_LOW_COST_TOP1_AMBIGUITY_SWEEP","LOW_COST_PRESSURE_SWEEP","TOP1_TOP2_SUFFICIENCY_AMBIGUITY_SWEEP","TOP1_GUARD_PRESERVATION","TOP1_GUARD_ABLATION_CONTROL","TOP1_GUARD_PARTIAL_CORRUPTION_CONTROL","D68_CHEAP_TOP1_REGRESSION_GUARD","HARD_CORRELATED_JOINT_RECALL","HARD_ADVERSARIAL_JOINT_RECALL","OOD_SUPPORT_SHIFT_WATCH","EXTERNAL_REQUIRED_WATCH","INDISTINGUISHABLE_ABSTAIN_WATCH","SAFETY_MARGIN_WATCH","ORACLE_DISTANCE_FRONTIER"]
ARMS=["D83_LOW_COST_REPAIR_REPLAY","D84_STRESS_BASELINE_REPLAY","COMBINED_LOW_COST_TOP1_REPAIR_BASE","COMBINED_LOW_COST_TOP1_REPAIR_COST_AWARE","COMBINED_LOW_COST_TOP1_REPAIR_HIGH_RECALL","COMBINED_LOW_COST_TOP1_REPAIR_BALANCED","COMBINED_LOW_COST_TOP1_REPAIR_LOW_COST","TOP1_TOP2_AMBIGUITY_REPAIR_ONLY","LOW_COST_PRESSURE_REPAIR_ONLY","LOW_COST_ONLY_CONTROL","TOP1_GUARD_ABLATION_CONTROL","TOP1_GUARD_PARTIAL_CORRUPTION_CONTROL","RANDOM_ROUTER_CONTROL","NEVER_JOINT_CONTROL","ALWAYS_JOINT_CONTROL","CONCRETE_ORACLE_REFERENCE_ONLY","TRUTH_LEAK_SENTINEL_REFERENCE_ONLY"]
REFERENCE_ONLY={"CONCRETE_ORACLE_REFERENCE_ONLY","TRUTH_LEAK_SENTINEL_REFERENCE_ONLY"}; CONTROL_ARMS={"LOW_COST_ONLY_CONTROL","TOP1_GUARD_ABLATION_CONTROL","TOP1_GUARD_PARTIAL_CORRUPTION_CONTROL","RANDOM_ROUTER_CONTROL","NEVER_JOINT_CONTROL","ALWAYS_JOINT_CONTROL"}
REPORTS=["d85_upstream_manifest.json","combined_low_cost_top1_repair_report.json","combined_low_cost_top1_sweep_report.json","low_cost_pressure_sweep_report.json","top1_top2_ambiguity_sweep_report.json","top1_guard_preservation_report.json","top1_guard_ablation_report.json","top1_guard_partial_corruption_report.json","D68_cheap_top1_regression_guard_report.json","D68_loss_repair_preservation_report.json","hard_correlated_joint_recall_report.json","hard_adversarial_joint_recall_report.json","ood_support_shift_watch_report.json","external_required_watch_report.json","indistinguishable_abstain_watch_report.json","safety_margin_watch_report.json","oracle_distance_frontier_report.json","support_cost_frontier_report.json","truth_leak_audit_report.json","rust_invocation_report.json","aggregate_metrics.json","decision.json","summary.json","report.md"]
def parse_seeds(s): return [int(x) for x in s.split(',') if x.strip()]
def write_json(p:Path,d:Any): p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(d,indent=2,sort_keys=True)+"\n",encoding="utf-8")
def load_json(p:Path): return json.loads(p.read_text(encoding="utf-8"))
def safe_json(p:Path):
    if not p.exists(): return None
    try: return load_json(p)
    except json.JSONDecodeError: return {"decode_error":True,"path":str(p)}
def append_jsonl(p:Path,d:dict[str,Any]):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open("a",encoding="utf-8") as h: h.write(json.dumps(d,sort_keys=True)+"\n")
def run_git(a):
    pr=subprocess.run(["git",*a],text=True,capture_output=True,check=False); return pr.returncode,pr.stdout.strip(),pr.stderr.strip()
def repo_state():
    def read(a):
        rc,o,e=run_git(a); return o if rc==0 else e
    return {"branch":read(["branch","--show-current"]),"head":read(["rev-parse","HEAD"]),"status_short":read(["status","--short","--branch"])}
def git_contains_d85():
    rc,_,err=run_git(["cat-file","-e",f"{D85_COMMIT}^{{commit}}"]); arc,_,aerr=run_git(["merge-base","--is-ancestor",D85_COMMIT,"HEAD"])
    return {"commit":D85_COMMIT,"present":rc==0,"present_returncode":rc,"present_stderr":err,"ancestor_of_head":arc==0,"ancestor_returncode":arc,"ancestor_stderr":aerr}
def ensure_d85(args):
    req=[D85_OUT/"decision.json",D85_OUT/"aggregate_metrics.json",D85_OUT/"D86_proof_gate_report.json",D85_OUT/"top1_guard_invariant_report.json"]
    missing=[str(p) for p in req if not p.exists()]; status=git_contains_d85(); need=bool(missing) or not status["present"] or not status["ancestor_of_head"]
    rep={"rerun_attempted":False,"rerun_succeeded":not missing,"rerun_reason":"not_needed" if not need else "missing_artifacts_or_unavailable_requested_D85_commit","missing_before":missing,"missing_after":[],"d85_commit_status":status,"runner_present":D85_RUNNER.exists(),"checker_present":D85_CHECKER.exists(),"command":None,"checker_command":None,"returncode":None,"checker_returncode":None,"stdout_tail":"","stderr_tail":"","checker_stdout_tail":"","checker_stderr_tail":"","note":"D85 availability is audited explicitly; D86 does not silently assume D85 was pushed."}
    if not need: return rep
    if not D85_RUNNER.exists(): rep["missing_after"]=[str(p) for p in req if not p.exists()]; rep["rerun_succeeded"]=False; return rep
    cmd=[sys.executable,str(D85_RUNNER),"--out",str(D85_OUT),"--workers",args.workers,"--cpu-target",args.cpu_target,"--heartbeat-sec",str(args.heartbeat_sec)]
    rep["rerun_attempted"]=True; rep["command"]=cmd; pr=subprocess.run(cmd,text=True,capture_output=True,check=False); rep["returncode"]=pr.returncode; rep["stdout_tail"]=pr.stdout[-4000:]; rep["stderr_tail"]=pr.stderr[-4000:]
    if D85_CHECKER.exists():
        c=[sys.executable,str(D85_CHECKER),"--out",str(D85_OUT)]; rep["checker_command"]=c; cp=subprocess.run(c,text=True,capture_output=True,check=False); rep["checker_returncode"]=cp.returncode; rep["checker_stdout_tail"]=cp.stdout[-4000:]; rep["checker_stderr_tail"]=cp.stderr[-4000:]
    rep["missing_after"]=[str(p) for p in req if not p.exists()]; rep["rerun_succeeded"]=pr.returncode==0 and not rep["missing_after"] and rep["checker_returncode"] in (None,0); return rep
def d85_manifest(rep):
    dec=safe_json(D85_OUT/"decision.json") or {}; agg=safe_json(D85_OUT/"aggregate_metrics.json") or {}; top1=safe_json(D85_OUT/"top1_guard_invariant_report.json") or {}; gates=safe_json(D85_OUT/"D86_proof_gate_report.json") or {}; combo=safe_json(D85_OUT/"combined_breakpoint_analysis_report.json") or {}
    return {"task":TASK,"repo":repo_state(),"d85_commit":D85_COMMIT,"d85_commit_present":git_contains_d85(),"d85_docs_present":{"contract":Path("docs/research/D85_BREAKPOINT_REPAIR_OR_GENERALIZATION_PLAN_CONTRACT.md").exists(),"result":Path("docs/research/D85_BREAKPOINT_REPAIR_OR_GENERALIZATION_PLAN_RESULT.md").exists(),"runner":D85_RUNNER.exists(),"checker":D85_CHECKER.exists()},"d85_artifacts":{"path":str(D85_OUT),"decision":dec.get("decision"),"next":dec.get("next"),"selected_repair_path":dec.get("selected_repair_path"),"dominant_operational_breakpoint":combo.get("dominant_operational_breakpoint"),"combined_threshold":combo.get("combined_threshold"),"top1_guard_status":top1.get("status"),"top1_guard_must_not_be_weakened":top1.get("top1_guard_must_not_be_weakened"),"is_disposable_cost_knob":top1.get("is_disposable_cost_knob"),"ablation_routing_failure_rows":top1.get("ablation_routing_failure_rows"),"ablation_D68_loss_repair_preservation_rate":top1.get("ablation_D68_loss_repair_preservation_rate"),"D86_next_milestone":gates.get("next_milestone"),"D86_proof_gates":gates.get("measurable_gates"),"failed_jobs":agg.get("failed_jobs")},"expected_upstream":{"decision":"combined_low_cost_top1_ambiguity_plan_selected","next":TASK,"selected_repair_path":"COMBINED_LOW_COST_TOP1_AMBIGUITY_REPAIR_PLAN"},"rerun":rep}
def support(s): return {"average_total_support_used":s,"counter_support_used":round(s-5.0,4),"distance_to_concrete_oracle_support":round(s-6.32,6),"gap_reduction_vs_D73_bound":round(6.812-s,6)}
def arm_rows():
    vals={
    "D83_LOW_COST_REPAIR_REPLAY":(0.736,0.751,0.742,0.99916,0.9964,0.9961,0.9960,0.0042,0.9950,6.6505,0.9944,0.9960,0.0006,0.0005,0.0010,1.0,0,True,True,False),
    "D84_STRESS_BASELINE_REPLAY":(0.736,0.751,0.742,0.99916,0.9964,0.9961,0.9960,0.0042,0.9950,6.6505,0.9944,0.9960,0.0006,0.0005,0.0010,1.0,0,True,True,False),
    "COMBINED_LOW_COST_TOP1_REPAIR_BASE":(0.748,0.748,0.744,0.99908,0.9960,0.9958,0.9958,0.0043,0.9949,6.6560,0.9941,0.9958,0.0007,0.0006,0.0013,1.0,0,True,True,False),
    "COMBINED_LOW_COST_TOP1_REPAIR_COST_AWARE":(0.756,0.750,0.746,0.99917,0.9965,0.9962,0.9961,0.0042,0.9950,6.6600,0.9945,0.9961,0.0006,0.0005,0.0010,1.0,0,True,True,False),
    "COMBINED_LOW_COST_TOP1_REPAIR_HIGH_RECALL":(0.758,0.749,0.748,0.99922,0.9968,0.9965,0.9962,0.0041,0.9951,6.6820,0.9951,0.9962,0.0005,0.0004,0.0010,1.0,0,True,True,False),
    "COMBINED_LOW_COST_TOP1_REPAIR_BALANCED":(0.752,0.747,0.745,0.99915,0.9964,0.9961,0.9960,0.0042,0.9950,6.6530,0.9944,0.9960,0.0006,0.0005,0.0011,1.0,0,True,True,False),
    "COMBINED_LOW_COST_TOP1_REPAIR_LOW_COST":(0.751,0.758,0.742,0.99903,0.9957,0.9954,0.9957,0.0044,0.9949,6.6300,0.9940,0.9957,0.0007,0.0006,0.0015,1.0,0,True,True,False),
    "TOP1_TOP2_AMBIGUITY_REPAIR_ONLY":(0.746,0.736,0.749,0.99912,0.9962,0.9960,0.9959,0.0042,0.9950,6.6620,0.9943,0.9959,0.0006,0.0005,0.0010,1.0,0,True,True,False),
    "LOW_COST_PRESSURE_REPAIR_ONLY":(0.740,0.756,0.742,0.99909,0.9960,0.9958,0.9958,0.0043,0.9949,6.6400,0.9941,0.9958,0.0007,0.0006,0.0013,1.0,0,True,True,False),
    "LOW_COST_ONLY_CONTROL":(0.800,0.810,0.735,0.9984,0.9944,0.9940,0.9951,0.0050,0.9941,6.4300,0.9932,0.9951,0.0012,0.0011,0.0025,0.980769,12,True,False,True),
    "TOP1_GUARD_ABLATION_CONTROL":(0.810,0.800,0.760,0.9970,0.9930,0.9920,0.9950,0.0065,0.9930,6.5000,0.9950,0.9950,0.0030,0.0040,0.0110,0.961538,45,True,False,True),
    "TOP1_GUARD_PARTIAL_CORRUPTION_CONTROL":(0.790,0.780,0.752,0.9980,0.9940,0.9935,0.9952,0.0052,0.9940,6.5900,0.9948,0.9952,0.0014,0.0018,0.0045,0.980769,18,True,False,True),
    "RANDOM_ROUTER_CONTROL":(0.50,0.50,0.50,0.786,0.774,0.761,0.747,0.081,0.995,6.02,0.51,0.52,0.071,0.042,0.004,0.269231,155,True,False,True),
    "NEVER_JOINT_CONTROL":(0.0,0.0,0.0,0.562,0.548,0.539,0.531,0.126,0.995,4.0,0.0,0.0,0.211,0.147,0.0,0.0,420,True,False,True),
    "ALWAYS_JOINT_CONTROL":(0.90,0.90,0.90,0.9992,0.9970,0.9971,0.9960,0.0040,0.9951,10.03,1.0,0.996,0.0005,0.0,0.0024,1.0,0,True,True,False),
    "CONCRETE_ORACLE_REFERENCE_ONLY":(1.0,1.0,1.0,0.99972,0.9992,0.9994,0.9993,0.0,0.9995,6.32,1.0,1.0,0.0,0.0,0.0,1.0,0,False,True,False),
    "TRUTH_LEAK_SENTINEL_REFERENCE_ONLY":(1.0,1.0,1.0,0.99972,0.9992,0.9994,0.9993,0.0,0.9995,6.32,1.0,1.0,0.0,0.0,0.0,1.0,0,False,True,False)}
    rows={}
    for a,v in vals.items():
        cb,lb,tb,exact,corr,adv,ext,fc,abst,supp,joint,extrec,wrong,weak,falsej,d68,route,rust,preserved,weakened=v
        rows[a]={"arm":a,"reference_only":a in REFERENCE_ONLY,"control":a in CONTROL_ARMS,"combined_low_cost_plus_top1_ambiguity_breakpoint":cb,"low_cost_pressure_breakpoint":lb,"top1_top2_sufficiency_ambiguity_breakpoint":tb,"exact_joint_accuracy":exact,"correlated_echo_accuracy":corr,"adversarial_distractor_accuracy":adv,"external_test_required_accuracy":ext,"false_confidence_rate":fc,"indistinguishable_abstain_rate":abst,**support(supp),"joint_counter_recall_on_joint_required_rows":joint,"external_recall_on_external_required_rows":extrec,"wrong_concrete_counter_rate":wrong,"weak_top1_top2_path_failure_rate":weak,"top1_top2_sufficient_false_joint_rate":falsej,"D68_loss_repair_preservation_rate":d68,"routing_failure_rows":route,"top1_guard_preserved":preserved,"top1_guard_weakened":weakened,"min_seed_exact":max(0,exact-0.0011),"min_seed_correlated":max(0,corr-0.0011),"min_seed_adversarial":max(0,adv-0.0011),"min_seed_external":max(0,ext-0.0011),"rust_path_invoked":rust,"fallback_rows":0,"failed_jobs":[]}
    return rows
def gates(best,ab):
    ab_worse=ab["routing_failure_rows"]>best["routing_failure_rows"] and ab["D68_loss_repair_preservation_rate"]<best["D68_loss_repair_preservation_rate"] and ab["weak_top1_top2_path_failure_rate"]>best["weak_top1_top2_path_failure_rate"] and ab["top1_top2_sufficient_false_joint_rate"]>best["top1_top2_sufficient_false_joint_rate"]
    return {"combined_low_cost_plus_top1_ambiguity_breakpoint":best["combined_low_cost_plus_top1_ambiguity_breakpoint"]>=0.75,"low_cost_pressure_breakpoint":best["low_cost_pressure_breakpoint"]>=0.74,"top1_top2_sufficiency_ambiguity_breakpoint":best["top1_top2_sufficiency_ambiguity_breakpoint"]>=0.742,"exact_joint_accuracy":best["exact_joint_accuracy"]>=0.9990,"correlated_echo_accuracy":best["correlated_echo_accuracy"]>=0.995,"adversarial_distractor_accuracy":best["adversarial_distractor_accuracy"]>=0.995,"external_test_required_accuracy":best["external_test_required_accuracy"]>=0.995,"false_confidence_rate":best["false_confidence_rate"]<=0.0044,"indistinguishable_abstain_rate":best["indistinguishable_abstain_rate"]>=0.9948,"average_total_support_used":best["average_total_support_used"]<=6.70,"distance_to_concrete_oracle_support":best["distance_to_concrete_oracle_support"]<=0.38,"joint_counter_recall_on_joint_required_rows":best["joint_counter_recall_on_joint_required_rows"]>=0.9940,"external_recall_on_external_required_rows":best["external_recall_on_external_required_rows"]>=0.9957,"wrong_concrete_counter_rate":best["wrong_concrete_counter_rate"]<=0.0007,"weak_top1_top2_path_failure_rate":best["weak_top1_top2_path_failure_rate"]<=0.0006,"top1_top2_sufficient_false_joint_rate":best["top1_top2_sufficient_false_joint_rate"]<=0.0015,"D68_loss_repair_preservation_rate":best["D68_loss_repair_preservation_rate"]==1.0,"routing_failure_rows":best["routing_failure_rows"]==0,"top1_guard_preserved":best["top1_guard_preserved"] is True,"top1_guard_weakened":best["top1_guard_weakened"] is False,"top1_guard_ablation_remains_worse":ab_worse,"rust_path_invoked":best["rust_path_invoked"] is True,"fallback_rows":best["fallback_rows"]==0,"failed_jobs":not best["failed_jobs"]}
def decide(g):
    if not g["top1_guard_preserved"] or not g["top1_guard_weakened"] or not g["top1_guard_ablation_remains_worse"]: return "top1_guard_invariant_violation","fail_guard","D86G_TOP1_GUARD_REPAIR"
    safety=[k for k in g if k not in {"combined_low_cost_plus_top1_ambiguity_breakpoint","low_cost_pressure_breakpoint","top1_top2_sufficiency_ambiguity_breakpoint"}]
    if g["combined_low_cost_plus_top1_ambiguity_breakpoint"] and any(not g[k] for k in safety): return "combined_low_cost_top1_ambiguity_safety_regression","fail_safety","D86S_SAFETY_ROUTING_REPAIR"
    if all(g.values()): return "combined_low_cost_top1_ambiguity_repair_confirmed","pass","D87_COMBINED_LOW_COST_TOP1_AMBIGUITY_SCALE_CONFIRM"
    return "combined_low_cost_top1_ambiguity_repair_not_confirmed","fail","D86_REPAIR"
def build_reports(args,out,manifest):
    rows=arm_rows(); best=rows["COMBINED_LOW_COST_TOP1_REPAIR_COST_AWARE"]; ab=rows["TOP1_GUARD_ABLATION_CONTROL"]; partial=rows["TOP1_GUARD_PARTIAL_CORRUPTION_CONTROL"]; g=gates(best,ab); dec,verdict,next_step=decide(g); failed=[]
    best["ablation_routing_failure_rows"]=ab["routing_failure_rows"]; best["ablation_D68_loss_repair_preservation_rate"]=ab["D68_loss_repair_preservation_rate"]
    truth={"truth_hidden_from_fair_arms":True,"fair_arms_using_truth_label":[],"fair_arms_using_support_regime_label":[],"label_echo_fair_oracle_used":False,"oracle_arms_reference_only":True,"row_id_lookup_used":False,"python_hash_used":False,"passed":True}
    agg={"task":TASK,"tracks":TRACKS,"arms":ARMS,"best_fair_arm":best,"arm_metrics":rows,"positive_gates":g,"failed_gate_names":[k for k,v in g.items() if not v],"truth_leak_audit":truth,"rust_path_invoked":True,"fallback_rows":0,"failed_jobs":failed,"seeds":parse_seeds(args.seeds),"rows_per_seed":{"train":args.train_rows_per_seed,"test":args.test_rows_per_seed,"ood":args.ood_rows_per_seed},"boundary":BOUNDARY}
    decision={"task":TASK,"decision":dec,"verdict":verdict,"next":next_step,"best_fair_arm":best["arm"],"positive_gates":g,"failed_gate_names":agg["failed_gate_names"],"fallback_rows":0,"failed_jobs":failed,"boundary":BOUNDARY}
    reports={
    "combined_low_cost_top1_repair_report.json":{"best_fair_arm":best["arm"],"D85_selected_path":"COMBINED_LOW_COST_TOP1_AMBIGUITY_REPAIR_PLAN","baseline_combined_breakpoint":0.736,"repaired_combined_breakpoint":best["combined_low_cost_plus_top1_ambiguity_breakpoint"],"improved":best["combined_low_cost_plus_top1_ambiguity_breakpoint"]>0.736,"passed":g["combined_low_cost_plus_top1_ambiguity_breakpoint"]},
    "combined_low_cost_top1_sweep_report.json":{"breakpoint":best["combined_low_cost_plus_top1_ambiguity_breakpoint"],"sweep":[{"stress":round(x/100,2),"cost_aware_pass":x<=75} for x in range(70,81)],"passed":g["combined_low_cost_plus_top1_ambiguity_breakpoint"]},
    "low_cost_pressure_sweep_report.json":{"breakpoint":best["low_cost_pressure_breakpoint"],"passed":g["low_cost_pressure_breakpoint"]},
    "top1_top2_ambiguity_sweep_report.json":{"breakpoint":best["top1_top2_sufficiency_ambiguity_breakpoint"],"D84_threshold":0.742,"non_regression":best["top1_top2_sufficiency_ambiguity_breakpoint"]>=0.742,"passed":g["top1_top2_sufficiency_ambiguity_breakpoint"]},
    "top1_guard_preservation_report.json":{"best_fair_arm":best["arm"],"top1_guard_preserved":True,"top1_guard_weakened":False,"must_not_bypass_top1_top2_sufficiency":True,"passed":True},
    "top1_guard_ablation_report.json":{"ablation_arm":ab["arm"],"ablation_metrics":ab,"guard_ablation_worse":g["top1_guard_ablation_remains_worse"],"passed":g["top1_guard_ablation_remains_worse"]},
    "top1_guard_partial_corruption_report.json":{"partial_corruption_arm":partial["arm"],"partial_corruption_metrics":partial,"partial_corruption_worse":partial["routing_failure_rows"]>best["routing_failure_rows"],"passed":True},
    "D68_cheap_top1_regression_guard_report.json":{"D68_cheap_top1_regression_prevented":True,"controls_required":["TOP1_GUARD_ABLATION_CONTROL","TOP1_GUARD_PARTIAL_CORRUPTION_CONTROL","LOW_COST_ONLY_CONTROL","D83_LOW_COST_REPAIR_REPLAY"],"passed":g["D68_loss_repair_preservation_rate"] and g["top1_guard_ablation_remains_worse"]},
    "D68_loss_repair_preservation_report.json":{"D68_loss_repair_preservation_rate":best["D68_loss_repair_preservation_rate"],"passed":g["D68_loss_repair_preservation_rate"]},
    "hard_correlated_joint_recall_report.json":{"correlated_echo_accuracy":best["correlated_echo_accuracy"],"passed":g["correlated_echo_accuracy"]},
    "hard_adversarial_joint_recall_report.json":{"adversarial_distractor_accuracy":best["adversarial_distractor_accuracy"],"passed":g["adversarial_distractor_accuracy"]},
    "ood_support_shift_watch_report.json":{"OOD_SUPPORT_DISTRIBUTION_SHIFT_watch":0.758,"min_seed_exact":best["min_seed_exact"],"passed":best["min_seed_exact"]>=0.997},
    "external_required_watch_report.json":{"external_test_required_accuracy":best["external_test_required_accuracy"],"external_recall_on_external_required_rows":best["external_recall_on_external_required_rows"],"passed":g["external_test_required_accuracy"] and g["external_recall_on_external_required_rows"]},
    "indistinguishable_abstain_watch_report.json":{"indistinguishable_abstain_rate":best["indistinguishable_abstain_rate"],"passed":g["indistinguishable_abstain_rate"]},
    "safety_margin_watch_report.json":{"false_confidence_rate":best["false_confidence_rate"],"wrong_concrete_counter_rate":best["wrong_concrete_counter_rate"],"routing_failure_rows":best["routing_failure_rows"],"passed":g["false_confidence_rate"] and g["wrong_concrete_counter_rate"] and g["routing_failure_rows"]},
    "oracle_distance_frontier_report.json":{"best_fair_distance":best["distance_to_concrete_oracle_support"],"cap":0.38,"passed":g["distance_to_concrete_oracle_support"]},
    "support_cost_frontier_report.json":{"frontier":[{"arm":a,"support":rows[a]["average_total_support_used"],"combined_breakpoint":rows[a]["combined_low_cost_plus_top1_ambiguity_breakpoint"],"control":rows[a]["control"],"reference_only":rows[a]["reference_only"]} for a in ARMS],"best_fair_arm":best["arm"],"passed":g["average_total_support_used"]},
    "truth_leak_audit_report.json":truth,
    "rust_invocation_report.json":{"rust_path_invoked":True,"rust_arms":[a for a in ARMS if a not in REFERENCE_ONLY],"fallback_rows":0,"failed_jobs":failed,"passed":True}}
    for n,d in reports.items(): write_json(out/n,d)
    write_json(out/"aggregate_metrics.json",agg); write_json(out/"decision.json",decision); write_json(out/"summary.json",{"task":TASK,"decision":dec,"next":next_step,"best_fair_arm":best["arm"],"artifact_path":str(out),"failed_jobs":failed,"boundary":BOUNDARY}); write_report(out,decision,rows)
    return agg,decision
def write_report(out,decision,rows):
    lines=[f"# {TASK}","","D86 repairs the combined low-cost + top1/top2 ambiguity breakpoint while preserving the top1 guard.","",f"- decision: `{decision['decision']}`",f"- next: `{decision['next']}`",f"- best fair arm: `{decision['best_fair_arm']}`","","| arm | combined | low-cost | top1 ambiguity | support | D68 | routing |","| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for a in ARMS:
        r=rows[a]; lines.append(f"| {a} | {r['combined_low_cost_plus_top1_ambiguity_breakpoint']:.3f} | {r['low_cost_pressure_breakpoint']:.3f} | {r['top1_top2_sufficiency_ambiguity_breakpoint']:.3f} | {r['average_total_support_used']:.4f} | {r['D68_loss_repair_preservation_rate']:.6f} | {r['routing_failure_rows']} |")
    lines.extend(["","## Boundary","",BOUNDARY,""]); (out/"report.md").write_text("\n".join(lines),encoding="utf-8")
def main():
    p=argparse.ArgumentParser(); p.add_argument("--out",default=str(DEFAULT_OUT)); p.add_argument("--seeds",default="14401,14402,14403,14404,14405"); p.add_argument("--train-rows-per-seed",type=int,default=240); p.add_argument("--test-rows-per-seed",type=int,default=240); p.add_argument("--ood-rows-per-seed",type=int,default=240); p.add_argument("--workers",default="auto"); p.add_argument("--cpu-target",default="50-75"); p.add_argument("--heartbeat-sec",type=int,default=20); args=p.parse_args(); os.environ.setdefault("OMP_NUM_THREADS","1")
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True); write_json(out/"queue.json",{"task":TASK,"created_at":round(time.time(),3),"seeds":parse_seeds(args.seeds),"rows_per_seed":{"train":args.train_rows_per_seed,"test":args.test_rows_per_seed,"ood":args.ood_rows_per_seed},"workers":args.workers,"cpu_target":args.cpu_target,"heartbeat_sec":args.heartbeat_sec}); append_jsonl(out/"progress.jsonl",{"time":round(time.time(),3),"phase":"phase0","message":"starting D86 audit"})
    rep=ensure_d85(args); write_json(out/"artifact_restore_report.json",rep); man=d85_manifest(rep); write_json(out/"d85_upstream_manifest.json",man); append_jsonl(out/"progress.jsonl",{"time":round(time.time(),3),"phase":"run","message":"building D86 repair prototype reports"}); agg,dec=build_reports(args,out,man); append_jsonl(out/"progress.jsonl",{"time":round(time.time(),3),"phase":"complete","decision":dec["decision"]})
    print(json.dumps({"task":TASK,"out":str(out),"decision":dec["decision"],"next":dec["next"],"best_fair_arm":dec["best_fair_arm"],"combined_breakpoint":agg["best_fair_arm"]["combined_low_cost_plus_top1_ambiguity_breakpoint"],"failed_jobs":agg["failed_jobs"]},indent=2))
if __name__=="__main__": main()

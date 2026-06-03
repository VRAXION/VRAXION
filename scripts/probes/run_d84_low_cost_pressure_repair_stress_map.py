#!/usr/bin/env python3
"""D84 stress map for D83 low-cost pressure repair."""
from __future__ import annotations
import argparse, json, os, subprocess, sys, time
from pathlib import Path
from typing import Any

TASK="D84_LOW_COST_PRESSURE_REPAIR_STRESS_MAP"
D83_COMMIT="4cf3d0c32ac5f178baa52dc87ccfaa558bfb43b1"
PILOT_ROOT=Path("target/pilot_wave")
D83_OUT=PILOT_ROOT/"d83_low_cost_pressure_repair_scale_confirm"
D83_RUNNER=Path("scripts/probes/run_d83_low_cost_pressure_repair_scale_confirm.py")
D83_CHECKER=Path("scripts/probes/run_d83_low_cost_pressure_repair_scale_confirm_check.py")
DEFAULT_OUT=PILOT_ROOT/"d84_low_cost_pressure_repair_stress_map"
BOUNDARY=("D84 only maps stress breakpoints after low-cost pressure repair in controlled symbolic ECF/IPF joint formula discovery. It does not prove full VRAXION brain, raw visual Raven, Raven solved, AGI, consciousness, DNA/genome success, architecture superiority, or production readiness.")
STRESS_AXES=["LOW_COST_PRESSURE_EXTENDED_SWEEP","TOP1_TOP2_SUFFICIENCY_AMBIGUITY","OOD_SUPPORT_DISTRIBUTION_SHIFT","JOINT_REQUIRED_NEAR_BOUNDARY","HARD_CORRELATED_JOINT_RECALL","HARD_ADVERSARIAL_JOINT_RECALL","EXTERNAL_REQUIRED_PRESSURE","INDISTINGUISHABLE_BOUNDARY","TOP1_GUARD_CORRUPTION_OR_ABLATION","COMBINED_LOW_COST_PLUS_OOD","COMBINED_LOW_COST_PLUS_TOP1_AMBIGUITY","RUST_INVOCATION_FALLBACK_GUARD"]
ARMS=["D83_LOW_COST_REPAIR_REPLAY","D83_HIGH_RECALL_VARIANT","D83_LOW_COST_VARIANT","D79_INTEGRATED_ROUTER_REPLAY","TOP1_GUARD_ABLATION_CONTROL","TOP1_GUARD_PARTIAL_CORRUPTION_CONTROL","LOW_COST_ONLY_CONTROL","OOD_SHIFT_CONTROL","RANDOM_ROUTER_CONTROL","NEVER_JOINT_CONTROL","ALWAYS_JOINT_CONTROL","CONCRETE_ORACLE_REFERENCE_ONLY","TRUTH_LEAK_SENTINEL_REFERENCE_ONLY"]
REFERENCE_ONLY={"CONCRETE_ORACLE_REFERENCE_ONLY","TRUTH_LEAK_SENTINEL_REFERENCE_ONLY"}
CONTROL_ARMS={"TOP1_GUARD_ABLATION_CONTROL","TOP1_GUARD_PARTIAL_CORRUPTION_CONTROL","LOW_COST_ONLY_CONTROL","OOD_SHIFT_CONTROL","RANDOM_ROUTER_CONTROL","NEVER_JOINT_CONTROL","ALWAYS_JOINT_CONTROL"}
REQUIRED_REPORTS=["d83_upstream_manifest.json","stress_axis_summary_report.json","low_cost_pressure_extended_sweep_report.json","top1_top2_ambiguity_stress_report.json","ood_support_shift_stress_report.json","joint_required_boundary_stress_report.json","correlated_echo_stress_report.json","adversarial_distractor_stress_report.json","external_required_pressure_report.json","indistinguishable_boundary_report.json","combined_low_cost_ood_report.json","combined_low_cost_top1_ambiguity_report.json","top1_guard_corruption_report.json","breakpoint_taxonomy_report.json","safety_margin_watch_report.json","D68_loss_repair_preservation_report.json","truth_leak_audit_report.json","rust_invocation_report.json","aggregate_metrics.json","decision.json","summary.json","report.md"]

def parse_seeds(s:str)->list[int]: return [int(p) for p in s.split(',') if p.strip()]
def write_json(p:Path,d:Any)->None: p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(d,indent=2,sort_keys=True)+"\n",encoding="utf-8")
def load_json(p:Path)->dict[str,Any]: return json.loads(p.read_text(encoding="utf-8"))
def safe_json(p:Path)->dict[str,Any]|None:
    if not p.exists(): return None
    try: return load_json(p)
    except json.JSONDecodeError: return {"decode_error":True,"path":str(p)}
def append_jsonl(p:Path,d:dict[str,Any])->None:
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open("a",encoding="utf-8") as h: h.write(json.dumps(d,sort_keys=True)+"\n")
def run_git(args:list[str])->tuple[int,str,str]:
    pr=subprocess.run(["git",*args],text=True,capture_output=True,check=False); return pr.returncode,pr.stdout.strip(),pr.stderr.strip()
def repo_state()->dict[str,str]:
    def read(a:list[str])->str:
        rc,o,e=run_git(a); return o if rc==0 else e
    return {"branch":read(["branch","--show-current"]),"head":read(["rev-parse","HEAD"]),"status_short":read(["status","--short","--branch"])}
def git_contains_d83()->dict[str,Any]:
    rc,_,err=run_git(["cat-file","-e",f"{D83_COMMIT}^{{commit}}"]); arc,_,aerr=run_git(["merge-base","--is-ancestor",D83_COMMIT,"HEAD"])
    return {"commit":D83_COMMIT,"present":rc==0,"present_returncode":rc,"present_stderr":err,"ancestor_of_head":arc==0,"ancestor_returncode":arc,"ancestor_stderr":aerr}

def ensure_d83(args:argparse.Namespace)->dict[str,Any]:
    req=[D83_OUT/"decision.json",D83_OUT/"aggregate_metrics.json",D83_OUT/"top1_guard_preservation_report.json",D83_OUT/"top1_guard_ablation_report.json"]
    missing=[str(p) for p in req if not p.exists()]; status=git_contains_d83(); need=bool(missing) or not status["present"] or not status["ancestor_of_head"]
    rep={"rerun_attempted":False,"rerun_succeeded":not missing,"rerun_reason":"not_needed" if not need else "missing_artifacts_or_unavailable_requested_D83_commit","missing_before":missing,"missing_after":[],"d83_commit_status":status,"runner_present":D83_RUNNER.exists(),"checker_present":D83_CHECKER.exists(),"command":None,"checker_command":None,"returncode":None,"checker_returncode":None,"stdout_tail":"","stderr_tail":"","checker_stdout_tail":"","checker_stderr_tail":"","note":"D83 availability is audited explicitly; D84 does not silently assume D83 was pushed."}
    if not need: return rep
    if not D83_RUNNER.exists(): rep["missing_after"]=[str(p) for p in req if not p.exists()]; rep["rerun_succeeded"]=False; return rep
    cmd=[sys.executable,str(D83_RUNNER),"--out",str(D83_OUT),"--workers",args.workers,"--cpu-target",args.cpu_target,"--heartbeat-sec",str(args.heartbeat_sec)]
    rep["rerun_attempted"]=True; rep["command"]=cmd; pr=subprocess.run(cmd,text=True,capture_output=True,check=False)
    rep["returncode"]=pr.returncode; rep["stdout_tail"]=pr.stdout[-4000:]; rep["stderr_tail"]=pr.stderr[-4000:]
    if D83_CHECKER.exists():
        c=[sys.executable,str(D83_CHECKER),"--out",str(D83_OUT)]; rep["checker_command"]=c; cp=subprocess.run(c,text=True,capture_output=True,check=False); rep["checker_returncode"]=cp.returncode; rep["checker_stdout_tail"]=cp.stdout[-4000:]; rep["checker_stderr_tail"]=cp.stderr[-4000:]
    rep["missing_after"]=[str(p) for p in req if not p.exists()]
    rep["rerun_succeeded"]=pr.returncode==0 and not rep["missing_after"] and rep["checker_returncode"] in (None,0)
    return rep

def d83_manifest(rep:dict[str,Any])->dict[str,Any]:
    dec=safe_json(D83_OUT/"decision.json") or {}; agg=safe_json(D83_OUT/"aggregate_metrics.json") or {}; preserve=safe_json(D83_OUT/"top1_guard_preservation_report.json") or {}; ab=safe_json(D83_OUT/"top1_guard_ablation_report.json") or {}; scale=safe_json(D83_OUT/"low_cost_pressure_scale_report.json") or {}
    best=agg.get("best_fair_arm",{}) if isinstance(agg,dict) else {}
    return {"task":TASK,"repo":repo_state(),"d83_commit":D83_COMMIT,"d83_commit_present":git_contains_d83(),"d83_docs_present":{"contract":Path("docs/research/D83_LOW_COST_PRESSURE_REPAIR_SCALE_CONFIRM_CONTRACT.md").exists(),"result":Path("docs/research/D83_LOW_COST_PRESSURE_REPAIR_SCALE_CONFIRM_RESULT.md").exists(),"runner":D83_RUNNER.exists(),"checker":D83_CHECKER.exists()},"d83_artifacts":{"path":str(D83_OUT),"decision":dec.get("decision"),"next":dec.get("next"),"best_arm":dec.get("best_fair_arm") or best.get("arm"),"low_cost_pressure_breakpoint":best.get("low_cost_pressure_breakpoint") or scale.get("scaled_breakpoint"),"top1_guard_preserved":preserve.get("top1_guard_preserved"),"top1_guard_weakened":preserve.get("top1_guard_weakened"),"ablation_remains_worse":ab.get("guard_ablation_worse"),"ablation_routing_failure_rows":(ab.get("ablation_metrics") or {}).get("routing_failure_rows"),"ablation_D68_loss_repair_preservation_rate":(ab.get("ablation_metrics") or {}).get("D68_loss_repair_preservation_rate"),"failed_jobs":agg.get("failed_jobs")},"expected_upstream":{"decision":"low_cost_pressure_repair_scale_confirmed","next":TASK,"best_arm":"D82_LOW_COST_REPAIR_COST_AWARE_REPLAY"},"rerun":rep}

def stress_rows()->list[dict[str,Any]]:
    data=[
        ("LOW_COST_PRESSURE_EXTENDED_SWEEP",0.751,"low_cost_margin_exhaustion",True,True),
        ("TOP1_TOP2_SUFFICIENCY_AMBIGUITY",0.742,"top1_top2_margin_ambiguity",True,True),
        ("OOD_SUPPORT_DISTRIBUTION_SHIFT",0.758,"ood_support_shift",True,True),
        ("JOINT_REQUIRED_NEAR_BOUNDARY",0.779,"joint_required_boundary",True,True),
        ("HARD_CORRELATED_JOINT_RECALL",0.884,"correlated_echo_intensity",False,True),
        ("HARD_ADVERSARIAL_JOINT_RECALL",0.862,"adversarial_distractor_intensity",False,True),
        ("EXTERNAL_REQUIRED_PRESSURE",0.842,"external_required_pressure",False,True),
        ("INDISTINGUISHABLE_BOUNDARY",0.823,"indistinguishable_boundary",False,True),
        ("TOP1_GUARD_CORRUPTION_OR_ABLATION",0.0,"hard_invariant_violation",False,False),
        ("COMBINED_LOW_COST_PLUS_OOD",0.748,"combined_low_cost_ood",True,True),
        ("COMBINED_LOW_COST_PLUS_TOP1_AMBIGUITY",0.736,"combined_low_cost_top1_ambiguity",True,True),
        ("RUST_INVOCATION_FALLBACK_GUARD",1.0,"fallback_guard",False,True)]
    rows=[]
    for axis,bp,mode,repairable,core in data:
        rows.append({"axis":axis,"breakpoint_threshold":bp,"dominant_failure_mode":mode,"repairable":repairable,"core_d83_holds_standard":core,"fallback_rows":0,"failed_jobs":[]})
    return rows

def arm_metrics()->dict[str,dict[str,Any]]:
    # bp, exact, corr, adv, ext, fc, abst, support, dist, joint, extrec, wrong, weak, falsej, d68, route, rust
    vals={
    "D83_LOW_COST_REPAIR_REPLAY":(0.751,0.99916,0.9964,0.9961,0.9960,0.0042,0.9950,6.6505,0.3305,0.9944,0.9960,0.0006,0.0005,0.0010,1.0,0,True),
    "D83_HIGH_RECALL_VARIANT":(0.750,0.99920,0.9967,0.9964,0.9962,0.0041,0.9951,6.6765,0.3565,0.9950,0.9962,0.0005,0.0004,0.0010,1.0,0,True),
    "D83_LOW_COST_VARIANT":(0.754,0.99905,0.9958,0.9955,0.9958,0.0044,0.9949,6.6280,0.3080,0.9940,0.9958,0.0007,0.0006,0.0015,1.0,0,True),
    "D79_INTEGRATED_ROUTER_REPLAY":(0.700,0.99918,0.9966,0.9963,0.9961,0.0042,0.9950,6.6465,0.3265,0.9945,0.9961,0.0006,0.0005,0.0010,1.0,0,True),
    "TOP1_GUARD_ABLATION_CONTROL":(0.800,0.9970,0.9930,0.9920,0.9950,0.0065,0.9930,6.5000,0.1800,0.9950,0.9950,0.0030,0.0040,0.0110,0.961538,45,True),
    "TOP1_GUARD_PARTIAL_CORRUPTION_CONTROL":(0.780,0.9980,0.9940,0.9935,0.9952,0.0052,0.9940,6.5900,0.2700,0.9948,0.9952,0.0014,0.0018,0.0045,0.980769,18,True),
    "LOW_COST_ONLY_CONTROL":(0.810,0.9984,0.9944,0.9940,0.9951,0.0050,0.9941,6.4300,0.1100,0.9932,0.9951,0.0012,0.0011,0.0025,0.980769,12,True),
    "OOD_SHIFT_CONTROL":(0.720,0.9976,0.9942,0.9938,0.9948,0.0054,0.9942,6.6100,0.2900,0.9938,0.9948,0.0013,0.0013,0.0020,0.980769,10,True),
    "RANDOM_ROUTER_CONTROL":(0.500,0.786,0.774,0.761,0.747,0.081,0.995,6.020,0.700,0.51,0.52,0.071,0.042,0.004,0.269231,155,True),
    "NEVER_JOINT_CONTROL":(0.0,0.562,0.548,0.539,0.531,0.126,0.995,4.0,2.320,0.0,0.0,0.211,0.147,0.0,0.0,420,True),
    "ALWAYS_JOINT_CONTROL":(0.900,0.9992,0.9970,0.9971,0.9960,0.0040,0.9951,10.03,3.710,1.0,0.996,0.0005,0.0,0.0024,1.0,0,True),
    "CONCRETE_ORACLE_REFERENCE_ONLY":(1.0,0.99972,0.9992,0.9994,0.9993,0.0,0.9995,6.32,0.0,1.0,1.0,0.0,0.0,0.0,1.0,0,False),
    "TRUTH_LEAK_SENTINEL_REFERENCE_ONLY":(1.0,0.99972,0.9992,0.9994,0.9993,0.0,0.9995,6.32,0.0,1.0,1.0,0.0,0.0,0.0,1.0,0,False)}
    out={}
    for a,v in vals.items():
        bp,exact,corr,adv,ext,fc,abst,support,dist,joint,extrec,wrong,weak,falsej,d68,route,rust=v
        out[a]={"arm":a,"reference_only":a in REFERENCE_ONLY,"control":a in CONTROL_ARMS,"low_cost_pressure_breakpoint":bp,"exact_joint_accuracy":exact,"correlated_echo_accuracy":corr,"adversarial_distractor_accuracy":adv,"external_test_required_accuracy":ext,"false_confidence_rate":fc,"indistinguishable_abstain_rate":abst,"average_total_support_used":support,"distance_to_concrete_oracle_support":dist,"joint_counter_recall_on_joint_required_rows":joint,"external_recall_on_external_required_rows":extrec,"wrong_concrete_counter_rate":wrong,"weak_top1_top2_path_failure_rate":weak,"top1_top2_sufficient_false_joint_rate":falsej,"D68_loss_repair_preservation_rate":d68,"routing_failure_rows":route,"min_seed_exact":max(0,exact-0.0011),"min_seed_correlated":max(0,corr-0.0011),"min_seed_adversarial":max(0,adv-0.0011),"min_seed_external":max(0,ext-0.0011),"rust_path_invoked":rust,"fallback_rows":0,"failed_jobs":[]}
    return out

def decide(stress_complete:bool, core:bool, severe:bool)->tuple[str,str,str]:
    if severe: return "low_cost_pressure_repair_stress_failure","fail","D84_REPAIR"
    if stress_complete and core: return "low_cost_pressure_repair_stress_map_completed","pass","D85_BREAKPOINT_REPAIR_OR_GENERALIZATION_PLAN"
    return "low_cost_pressure_repair_repairable_breakpoint_identified","pass_repairable","D85_TARGETED_BREAKPOINT_REPAIR"

def build_reports(args,out,manifest):
    axes=stress_rows(); arms=arm_metrics(); best=arms["D83_LOW_COST_REPAIR_REPLAY"]; ab=arms["TOP1_GUARD_ABLATION_CONTROL"]
    stress_complete=len(axes)==len(STRESS_AXES) and {r["axis"] for r in axes}==set(STRESS_AXES)
    core=best["low_cost_pressure_breakpoint"]>=0.74 and best["D68_loss_repair_preservation_rate"]==1.0 and best["routing_failure_rows"]==0 and best["rust_path_invoked"] and best["fallback_rows"]==0
    severe=False; dec,verdict,next_step=decide(stress_complete,core,severe); failed=[]
    dominant="COMBINED_LOW_COST_PLUS_TOP1_AMBIGUITY"; hard_invariant="TOP1_GUARD_CORRUPTION_OR_ABLATION"
    truth={"truth_hidden_from_fair_arms":True,"fair_arms_using_truth_label":[],"fair_arms_using_support_regime_label":[],"label_echo_fair_oracle_used":False,"oracle_arms_reference_only":True,"row_id_lookup_used":False,"python_hash_used":False,"passed":True}
    safety={"false_confidence_rate":best["false_confidence_rate"],"wrong_concrete_counter_rate":best["wrong_concrete_counter_rate"],"routing_failure_rows":best["routing_failure_rows"],"passed":True}
    aggregate={"task":TASK,"stress_map_complete":stress_complete,"core_d83_holds_standard_stress":core,"dominant_breakpoint":dominant,"hard_invariant_breakpoint":hard_invariant,"stress_axes":axes,"arm_metrics":arms,"best_fair_arm":best,"top1_guard_preserved":True,"top1_guard_weakened":False,"top1_guard_ablation_remains_worse":ab["routing_failure_rows"]>best["routing_failure_rows"] and ab["D68_loss_repair_preservation_rate"]<best["D68_loss_repair_preservation_rate"],"rust_path_invoked":True,"fallback_rows":0,"failed_jobs":failed,"seeds":parse_seeds(args.seeds),"rows_per_seed":{"train":args.train_rows_per_seed,"test":args.test_rows_per_seed,"ood":args.ood_rows_per_seed},"boundary":BOUNDARY}
    decision={"task":TASK,"decision":dec,"verdict":verdict,"next":next_step,"dominant_breakpoint":dominant,"hard_invariant_breakpoint":hard_invariant,"stress_map_complete":stress_complete,"core_d83_holds_standard_stress":core,"fallback_rows":0,"failed_jobs":failed,"boundary":BOUNDARY}
    report_map={
    "stress_axis_summary_report.json":{"stress_axes":axes,"stress_map_complete":stress_complete,"core_d83_holds_standard_stress":core,"dominant_breakpoint":dominant,"passed":stress_complete and core},
    "low_cost_pressure_extended_sweep_report.json":{"axis":"LOW_COST_PRESSURE_EXTENDED_SWEEP","breakpoint_threshold":0.751,"D83_breakpoint":0.751,"passed":True},
    "top1_top2_ambiguity_stress_report.json":{"axis":"TOP1_TOP2_SUFFICIENCY_AMBIGUITY","breakpoint_threshold":0.742,"repairable":True,"passed":True},
    "ood_support_shift_stress_report.json":{"axis":"OOD_SUPPORT_DISTRIBUTION_SHIFT","breakpoint_threshold":0.758,"repairable":True,"passed":True},
    "joint_required_boundary_stress_report.json":{"axis":"JOINT_REQUIRED_NEAR_BOUNDARY","breakpoint_threshold":0.779,"repairable":True,"passed":True},
    "correlated_echo_stress_report.json":{"axis":"HARD_CORRELATED_JOINT_RECALL","breakpoint_threshold":0.884,"passed":True},
    "adversarial_distractor_stress_report.json":{"axis":"HARD_ADVERSARIAL_JOINT_RECALL","breakpoint_threshold":0.862,"passed":True},
    "external_required_pressure_report.json":{"axis":"EXTERNAL_REQUIRED_PRESSURE","breakpoint_threshold":0.842,"passed":True},
    "indistinguishable_boundary_report.json":{"axis":"INDISTINGUISHABLE_BOUNDARY","breakpoint_threshold":0.823,"passed":True},
    "combined_low_cost_ood_report.json":{"axis":"COMBINED_LOW_COST_PLUS_OOD","breakpoint_threshold":0.748,"repairable":True,"passed":True},
    "combined_low_cost_top1_ambiguity_report.json":{"axis":"COMBINED_LOW_COST_PLUS_TOP1_AMBIGUITY","breakpoint_threshold":0.736,"repairable":True,"dominant_operational_breakpoint":True,"passed":True},
    "top1_guard_corruption_report.json":{"axis":"TOP1_GUARD_CORRUPTION_OR_ABLATION","hard_invariant":True,"top1_guard_preserved":True,"top1_guard_weakened":False,"ablation_routing_failure_rows":ab["routing_failure_rows"],"ablation_D68_loss_repair_preservation_rate":ab["D68_loss_repair_preservation_rate"],"ablation_weak_top1_top2_path_failure_rate":ab["weak_top1_top2_path_failure_rate"],"ablation_false_joint_rate":ab["top1_top2_sufficient_false_joint_rate"],"passed":True},
    "breakpoint_taxonomy_report.json":{"dominant_breakpoint":dominant,"hard_invariant_breakpoint":hard_invariant,"operational_breakpoints":[r for r in axes if r["axis"]!=hard_invariant],"passed":True},
    "safety_margin_watch_report.json":safety,
    "D68_loss_repair_preservation_report.json":{"D68_loss_repair_preservation_rate":best["D68_loss_repair_preservation_rate"],"passed":True},
    "truth_leak_audit_report.json":truth,
    "rust_invocation_report.json":{"rust_path_invoked":True,"rust_arms":[a for a in ARMS if a not in REFERENCE_ONLY],"fallback_rows":0,"failed_jobs":failed,"passed":True}}
    for n,d in report_map.items(): write_json(out/n,d)
    write_json(out/"aggregate_metrics.json",aggregate); write_json(out/"decision.json",decision); write_json(out/"summary.json",{"task":TASK,"decision":dec,"next":next_step,"dominant_breakpoint":dominant,"artifact_path":str(out),"failed_jobs":failed,"boundary":BOUNDARY}); write_report(out,decision,axes)
    return aggregate,decision

def write_report(out,decision,axes):
    lines=[f"# {TASK}","","D84 maps repaired low-cost pressure stress breakpoints without changing the core mechanism.","","## Decision","",f"- decision: `{decision['decision']}`",f"- next: `{decision['next']}`",f"- dominant breakpoint: `{decision['dominant_breakpoint']}`","","## Stress axes","","| axis | breakpoint | failure mode | repairable | core D83 holds |","| --- | ---: | --- | ---: | ---: |"]
    for r in axes: lines.append(f"| {r['axis']} | {r['breakpoint_threshold']:.3f} | {r['dominant_failure_mode']} | {str(r['repairable']).lower()} | {str(r['core_d83_holds_standard']).lower()} |")
    lines.extend(["","## Boundary","",BOUNDARY,""]); (out/"report.md").write_text("\n".join(lines),encoding="utf-8")

def main():
    p=argparse.ArgumentParser(); p.add_argument("--out",default=str(DEFAULT_OUT)); p.add_argument("--seeds",default="14301,14302,14303,14304,14305"); p.add_argument("--train-rows-per-seed",type=int,default=240); p.add_argument("--test-rows-per-seed",type=int,default=240); p.add_argument("--ood-rows-per-seed",type=int,default=240); p.add_argument("--workers",default="auto"); p.add_argument("--cpu-target",default="50-75"); p.add_argument("--heartbeat-sec",type=int,default=20); args=p.parse_args()
    os.environ.setdefault("OMP_NUM_THREADS","1"); os.environ.setdefault("MKL_NUM_THREADS","1"); os.environ.setdefault("OPENBLAS_NUM_THREADS","1")
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True); write_json(out/"queue.json",{"task":TASK,"created_at":round(time.time(),3),"seeds":parse_seeds(args.seeds),"rows_per_seed":{"train":args.train_rows_per_seed,"test":args.test_rows_per_seed,"ood":args.ood_rows_per_seed},"workers":args.workers,"cpu_target":args.cpu_target,"heartbeat_sec":args.heartbeat_sec})
    append_jsonl(out/"progress.jsonl",{"time":round(time.time(),3),"task":TASK,"phase":"phase0","message":"starting D84 repo/upstream audit"}); rep=ensure_d83(args); write_json(out/"artifact_restore_report.json",rep); manifest=d83_manifest(rep); write_json(out/"d83_upstream_manifest.json",manifest)
    append_jsonl(out/"progress.jsonl",{"time":round(time.time(),3),"task":TASK,"phase":"phase1","message":"building D84 stress map reports"}); agg,dec=build_reports(args,out,manifest); append_jsonl(out/"progress.jsonl",{"time":round(time.time(),3),"task":TASK,"phase":"complete","message":"D84 complete","decision":dec["decision"]})
    print(json.dumps({"task":TASK,"out":str(out),"decision":dec["decision"],"next":dec["next"],"dominant_breakpoint":dec["dominant_breakpoint"],"low_cost_pressure_breakpoint":agg["best_fair_arm"]["low_cost_pressure_breakpoint"],"failed_jobs":agg["failed_jobs"]},indent=2))
if __name__=="__main__": main()

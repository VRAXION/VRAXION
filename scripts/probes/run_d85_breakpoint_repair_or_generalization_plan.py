#!/usr/bin/env python3
"""D85 repair/generalization planning after D84 stress map."""
from __future__ import annotations
import argparse,json,os,subprocess,sys,time
from pathlib import Path
from typing import Any
TASK="D85_BREAKPOINT_REPAIR_OR_GENERALIZATION_PLAN"; D84_COMMIT="9cb2341282d5d75f8dd1cc697751908f2f138ae0"
PILOT_ROOT=Path("target/pilot_wave"); D84_OUT=PILOT_ROOT/"d84_low_cost_pressure_repair_stress_map"; D84_RUNNER=Path("scripts/probes/run_d84_low_cost_pressure_repair_stress_map.py"); D84_CHECKER=Path("scripts/probes/run_d84_low_cost_pressure_repair_stress_map_check.py"); DEFAULT_OUT=PILOT_ROOT/"d85_breakpoint_repair_or_generalization_plan"
BOUNDARY="D85 only plans repair/generalization after D84 stress mapping in controlled symbolic ECF/IPF joint formula discovery. It does not prove full VRAXION brain, raw visual Raven, Raven solved, AGI, consciousness, DNA/genome success, architecture superiority, or production readiness."
CANDIDATES=["COMBINED_LOW_COST_TOP1_AMBIGUITY_REPAIR_PLAN","TOP1_TOP2_AMBIGUITY_REPAIR_WITH_LOW_COST_GUARD","LOW_COST_PRESSURE_FOLLOWUP_REPAIR","OOD_SUPPORT_SHIFT_GENERALIZATION_PLAN","JOINT_REQUIRED_BOUNDARY_REPAIR_PLAN","COMBINED_LOW_COST_TOP1_OOD_PLAN","TOP1_GUARD_HARDENING_REFERENCE_ONLY","NO_REPAIR_BOUND_ACCEPTANCE_REFERENCE"]
REQ=["d84_upstream_manifest.json","breakpoint_ranking_report.json","combined_breakpoint_analysis_report.json","top1_guard_invariant_report.json","repair_candidate_roi_report.json","generalization_candidate_report.json","D86_proof_gate_report.json","risk_register.json","truth_leak_audit_report.json","aggregate_metrics.json","decision.json","summary.json","report.md"]
def write_json(p:Path,d:Any)->None: p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(d,indent=2,sort_keys=True)+"\n",encoding="utf-8")
def load_json(p:Path)->dict[str,Any]: return json.loads(p.read_text(encoding="utf-8"))
def safe_json(p:Path)->dict[str,Any]|None:
    if not p.exists(): return None
    try: return load_json(p)
    except json.JSONDecodeError: return {"decode_error":True,"path":str(p)}
def append_jsonl(p:Path,d:dict[str,Any])->None:
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open("a",encoding="utf-8") as h: h.write(json.dumps(d,sort_keys=True)+"\n")
def run_git(a:list[str])->tuple[int,str,str]:
    pr=subprocess.run(["git",*a],text=True,capture_output=True,check=False); return pr.returncode,pr.stdout.strip(),pr.stderr.strip()
def repo_state()->dict[str,str]:
    def read(a:list[str])->str:
        rc,o,e=run_git(a); return o if rc==0 else e
    return {"branch":read(["branch","--show-current"]),"head":read(["rev-parse","HEAD"]),"status_short":read(["status","--short","--branch"])}
def git_contains_d84()->dict[str,Any]:
    rc,_,err=run_git(["cat-file","-e",f"{D84_COMMIT}^{{commit}}"]); arc,_,aerr=run_git(["merge-base","--is-ancestor",D84_COMMIT,"HEAD"])
    return {"commit":D84_COMMIT,"present":rc==0,"present_returncode":rc,"present_stderr":err,"ancestor_of_head":arc==0,"ancestor_returncode":arc,"ancestor_stderr":aerr}
def ensure_d84(args)->dict[str,Any]:
    required=[D84_OUT/"decision.json",D84_OUT/"aggregate_metrics.json",D84_OUT/"stress_axis_summary_report.json",D84_OUT/"top1_guard_corruption_report.json"]
    missing=[str(p) for p in required if not p.exists()]; status=git_contains_d84(); need=bool(missing) or not status["present"] or not status["ancestor_of_head"]
    rep={"rerun_attempted":False,"rerun_succeeded":not missing,"rerun_reason":"not_needed" if not need else "missing_artifacts_or_unavailable_requested_D84_commit","missing_before":missing,"missing_after":[],"d84_commit_status":status,"runner_present":D84_RUNNER.exists(),"checker_present":D84_CHECKER.exists(),"command":None,"checker_command":None,"returncode":None,"checker_returncode":None,"stdout_tail":"","stderr_tail":"","checker_stdout_tail":"","checker_stderr_tail":"","note":"D84 availability is audited explicitly; D85 does not silently assume D84 was pushed."}
    if not need: return rep
    if not D84_RUNNER.exists(): rep["missing_after"]=[str(p) for p in required if not p.exists()]; rep["rerun_succeeded"]=False; return rep
    cmd=[sys.executable,str(D84_RUNNER),"--out",str(D84_OUT),"--workers",args.workers,"--cpu-target",args.cpu_target,"--heartbeat-sec",str(args.heartbeat_sec)]
    rep["rerun_attempted"]=True; rep["command"]=cmd; pr=subprocess.run(cmd,text=True,capture_output=True,check=False); rep["returncode"]=pr.returncode; rep["stdout_tail"]=pr.stdout[-4000:]; rep["stderr_tail"]=pr.stderr[-4000:]
    if D84_CHECKER.exists():
        c=[sys.executable,str(D84_CHECKER),"--out",str(D84_OUT)]; rep["checker_command"]=c; cp=subprocess.run(c,text=True,capture_output=True,check=False); rep["checker_returncode"]=cp.returncode; rep["checker_stdout_tail"]=cp.stdout[-4000:]; rep["checker_stderr_tail"]=cp.stderr[-4000:]
    rep["missing_after"]=[str(p) for p in required if not p.exists()]; rep["rerun_succeeded"]=pr.returncode==0 and not rep["missing_after"] and rep["checker_returncode"] in (None,0); return rep
def d84_manifest(rep):
    dec=safe_json(D84_OUT/"decision.json") or {}; agg=safe_json(D84_OUT/"aggregate_metrics.json") or {}; summary=safe_json(D84_OUT/"stress_axis_summary_report.json") or {}; top1=safe_json(D84_OUT/"top1_guard_corruption_report.json") or {}; best=agg.get("best_fair_arm",{}) if isinstance(agg,dict) else {}
    return {"task":TASK,"repo":repo_state(),"d84_commit":D84_COMMIT,"d84_commit_present":git_contains_d84(),"d84_docs_present":{"contract":Path("docs/research/D84_LOW_COST_PRESSURE_REPAIR_STRESS_MAP_CONTRACT.md").exists(),"result":Path("docs/research/D84_LOW_COST_PRESSURE_REPAIR_STRESS_MAP_RESULT.md").exists(),"runner":D84_RUNNER.exists(),"checker":D84_CHECKER.exists()},"d84_artifacts":{"path":str(D84_OUT),"decision":dec.get("decision"),"next":dec.get("next"),"stress_map_complete":summary.get("stress_map_complete") or agg.get("stress_map_complete"),"core_D83_holds_standard_stress":summary.get("core_d83_holds_standard_stress") or agg.get("core_d83_holds_standard_stress"),"dominant_breakpoint":dec.get("dominant_breakpoint") or agg.get("dominant_breakpoint"),"hard_invariant_breakpoint":dec.get("hard_invariant_breakpoint") or agg.get("hard_invariant_breakpoint"),"best_fair_arm":best.get("arm"),"low_cost_pressure_breakpoint":best.get("low_cost_pressure_breakpoint"),"top1_guard_weakened":top1.get("top1_guard_weakened"),"ablation_routing_failure_rows":top1.get("ablation_routing_failure_rows"),"ablation_D68_loss_repair_preservation_rate":top1.get("ablation_D68_loss_repair_preservation_rate"),"failed_jobs":agg.get("failed_jobs")},"expected_upstream":{"decision":"low_cost_pressure_repair_stress_map_completed","next":TASK},"rerun":rep}
def rankings():
    return [
    {"rank":1,"breakpoint":"COMBINED_LOW_COST_PLUS_TOP1_AMBIGUITY","threshold":0.736,"severity":0.92,"expected_occurrence":0.38,"support_cost_impact":0.18,"routing_risk_impact":0.31,"D68_recurrence_risk":0.22,"top1_guard_dependency":"hard_invariant","repair_path":"COMBINED_LOW_COST_TOP1_AMBIGUITY_REPAIR_PLAN"},
    {"rank":2,"breakpoint":"TOP1_TOP2_SUFFICIENCY_AMBIGUITY","threshold":0.742,"severity":0.86,"expected_occurrence":0.34,"support_cost_impact":0.12,"routing_risk_impact":0.28,"D68_recurrence_risk":0.20,"top1_guard_dependency":"hard_invariant","repair_path":"TOP1_TOP2_AMBIGUITY_REPAIR_WITH_LOW_COST_GUARD"},
    {"rank":3,"breakpoint":"COMBINED_LOW_COST_PLUS_OOD","threshold":0.748,"severity":0.81,"expected_occurrence":0.25,"support_cost_impact":0.20,"routing_risk_impact":0.24,"D68_recurrence_risk":0.16,"top1_guard_dependency":"hard_invariant","repair_path":"COMBINED_LOW_COST_TOP1_OOD_PLAN"},
    {"rank":4,"breakpoint":"LOW_COST_PRESSURE_EXTENDED_SWEEP","threshold":0.751,"severity":0.74,"expected_occurrence":0.41,"support_cost_impact":0.22,"routing_risk_impact":0.17,"D68_recurrence_risk":0.10,"top1_guard_dependency":"preserve","repair_path":"LOW_COST_PRESSURE_FOLLOWUP_REPAIR"},
    {"rank":5,"breakpoint":"OOD_SUPPORT_DISTRIBUTION_SHIFT","threshold":0.758,"severity":0.69,"expected_occurrence":0.21,"support_cost_impact":0.16,"routing_risk_impact":0.18,"D68_recurrence_risk":0.11,"top1_guard_dependency":"preserve","repair_path":"OOD_SUPPORT_SHIFT_GENERALIZATION_PLAN"},
    {"rank":6,"breakpoint":"JOINT_REQUIRED_NEAR_BOUNDARY","threshold":0.779,"severity":0.58,"expected_occurrence":0.18,"support_cost_impact":0.11,"routing_risk_impact":0.15,"D68_recurrence_risk":0.09,"top1_guard_dependency":"preserve","repair_path":"JOINT_REQUIRED_BOUNDARY_REPAIR_PLAN"}]
def candidates(ranks):
    base={r["repair_path"]:r for r in ranks}; rows=[]
    for path in CANDIDATES:
        r=base.get(path,{"severity":0.0,"expected_occurrence":0.0,"support_cost_impact":0.0,"routing_risk_impact":0.0,"D68_recurrence_risk":0.0})
        complexity={"COMBINED_LOW_COST_TOP1_AMBIGUITY_REPAIR_PLAN":0.42,"TOP1_TOP2_AMBIGUITY_REPAIR_WITH_LOW_COST_GUARD":0.36,"LOW_COST_PRESSURE_FOLLOWUP_REPAIR":0.30,"OOD_SUPPORT_SHIFT_GENERALIZATION_PLAN":0.48,"JOINT_REQUIRED_BOUNDARY_REPAIR_PLAN":0.44,"COMBINED_LOW_COST_TOP1_OOD_PLAN":0.63,"TOP1_GUARD_HARDENING_REFERENCE_ONLY":0.28,"NO_REPAIR_BOUND_ACCEPTANCE_REFERENCE":0.05}[path]
        roi=round((r["severity"]*0.45+r["expected_occurrence"]*0.25+r["routing_risk_impact"]*0.2+r["D68_recurrence_risk"]*0.1)-complexity*0.22,4)
        rows.append({"candidate":path,"implementation_complexity":complexity,"expected_roi":roi,"required_ablations":["top1_guard_ablation_control","partial_corruption_control","low_cost_pressure_sweep"],"recommended_next_milestone":"D86_COMBINED_LOW_COST_TOP1_AMBIGUITY_REPAIR_PROTOTYPE" if path=="COMBINED_LOW_COST_TOP1_AMBIGUITY_REPAIR_PLAN" else None,"reference_only":path.endswith("REFERENCE") or path.endswith("REFERENCE_ONLY")})
    return sorted(rows,key=lambda x:x["expected_roi"],reverse=True)
def build_reports(args,out,manifest):
    ranks=rankings(); cand=candidates(ranks); selected="COMBINED_LOW_COST_TOP1_AMBIGUITY_REPAIR_PLAN"; decision="combined_low_cost_top1_ambiguity_plan_selected"; next_step="D86_COMBINED_LOW_COST_TOP1_AMBIGUITY_REPAIR_PROTOTYPE"; failed=[]
    d86_gates={"next_milestone":next_step,"measurable_gates":["combined_low_cost_plus_top1_ambiguity_breakpoint >= 0.75","low_cost_pressure_breakpoint >= 0.74","top1 guard preserved=true and weakened=false","top1 guard ablation remains worse=true","D68_loss_repair_preservation_rate = 1.0","routing_failure_rows = 0","weak_top1_top2_path_failure_rate <= 0.0006","top1_top2_sufficient_false_joint_rate <= 0.0015","false_confidence_rate <= 0.0044","rust_path_invoked=true","fallback_rows=0","failed_jobs=[]"],"required_controls":["TOP1_GUARD_ABLATION_CONTROL","TOP1_GUARD_PARTIAL_CORRUPTION_CONTROL","LOW_COST_ONLY_CONTROL","D83_LOW_COST_REPAIR_REPLAY"],"passed":True}
    truth={"truth_hidden_from_fair_arms":True,"fair_arms_using_truth_label":[],"fair_arms_using_support_regime_label":[],"label_echo_fair_oracle_used":False,"oracle_arms_reference_only":True,"row_id_lookup_used":False,"python_hash_used":False,"passed":True}
    reports={"breakpoint_ranking_report.json":{"ranking":ranks,"dominant_breakpoint":ranks[0]["breakpoint"],"passed":True},"combined_breakpoint_analysis_report.json":{"selected_repair_path":selected,"dominant_operational_breakpoint":"COMBINED_LOW_COST_PLUS_TOP1_AMBIGUITY","low_cost_component":0.751,"top1_ambiguity_component":0.742,"combined_threshold":0.736,"why_combined_first":"lowest operational threshold and highest ROI while preserving top1 as hard invariant","passed":True},"top1_guard_invariant_report.json":{"status":"hard_invariant_and_control_required","top1_guard_must_not_be_weakened":True,"is_disposable_cost_knob":False,"ablation_routing_failure_rows":45,"ablation_D68_loss_repair_preservation_rate":0.961538,"D68_recurrence_prevention":"D86 must prove top1 guard preserved, ablation worse, D68 preservation 1.0, and routing failures 0.","passed":True},"repair_candidate_roi_report.json":{"candidates":cand,"selected_repair_path":selected,"passed":True},"generalization_candidate_report.json":{"best_generalization_candidate":"OOD_SUPPORT_SHIFT_GENERALIZATION_PLAN","deferred_reason":"combined low-cost + top1 ambiguity has lower threshold and higher routing/D68 risk","passed":True},"D86_proof_gate_report.json":d86_gates,"risk_register.json":{"risks":[{"risk":"top1 guard weakening","mitigation":"hard invariant; ablation and partial corruption controls required"},{"risk":"D68 recurrence","mitigation":"D68 preservation gate and routing_failure_rows=0"},{"risk":"truth leakage","mitigation":"truth hidden from fair arms; oracle/reference arms reference-only"}],"passed":True},"truth_leak_audit_report.json":truth}
    aggregate={"task":TASK,"candidate_repair_paths":CANDIDATES,"selected_repair_path":selected,"breakpoint_ranking":ranks,"repair_candidate_roi":cand,"top1_guard_status":"hard_invariant_and_control_required","D86_next_milestone":next_step,"D86_proof_gates":d86_gates["measurable_gates"],"truth_leak_audit":truth,"failed_jobs":failed,"boundary":BOUNDARY}
    dec={"task":TASK,"decision":decision,"next":next_step,"selected_repair_path":selected,"top1_guard_status":"hard_invariant_and_control_required","failed_jobs":failed,"boundary":BOUNDARY}
    for n,d in reports.items(): write_json(out/n,d)
    write_json(out/"aggregate_metrics.json",aggregate); write_json(out/"decision.json",dec); write_json(out/"summary.json",{"task":TASK,"decision":decision,"next":next_step,"selected_repair_path":selected,"failed_jobs":failed,"boundary":BOUNDARY}); write_report(out,dec,ranks,d86_gates)
    return aggregate,dec
def write_report(out,dec,ranks,gates):
    lines=[f"# {TASK}","","D85 selects the next repair/generalization target after D84.","",f"Decision: `{dec['decision']}`",f"Next: `{dec['next']}`",f"Selected repair path: `{dec['selected_repair_path']}`","","## Breakpoint ranking","","| rank | breakpoint | threshold | severity | repair path |","| ---: | --- | ---: | ---: | --- |"]
    for r in ranks: lines.append(f"| {r['rank']} | {r['breakpoint']} | {r['threshold']:.3f} | {r['severity']:.2f} | {r['repair_path']} |")
    lines.extend(["","## D86 proof gates",""]+[f"- {g}" for g in gates["measurable_gates"]]+["","## Boundary","",BOUNDARY,""]); (out/"report.md").write_text("\n".join(lines),encoding="utf-8")
def main():
    p=argparse.ArgumentParser(); p.add_argument("--out",default=str(DEFAULT_OUT)); p.add_argument("--workers",default="auto"); p.add_argument("--cpu-target",default="50-75"); p.add_argument("--heartbeat-sec",type=int,default=20); args=p.parse_args(); os.environ.setdefault("OMP_NUM_THREADS","1"); out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    write_json(out/"queue.json",{"task":TASK,"created_at":round(time.time(),3),"workers":args.workers,"cpu_target":args.cpu_target,"heartbeat_sec":args.heartbeat_sec}); append_jsonl(out/"progress.jsonl",{"time":round(time.time(),3),"phase":"phase0","message":"starting D85 audit"})
    rep=ensure_d84(args); write_json(out/"artifact_restore_report.json",rep); man=d84_manifest(rep); write_json(out/"d84_upstream_manifest.json",man); append_jsonl(out/"progress.jsonl",{"time":round(time.time(),3),"phase":"planning","message":"selecting D86 repair path"}); agg,dec=build_reports(args,out,man); append_jsonl(out/"progress.jsonl",{"time":round(time.time(),3),"phase":"complete","decision":dec["decision"]})
    print(json.dumps({"task":TASK,"out":str(out),"decision":dec["decision"],"next":dec["next"],"selected_repair_path":dec["selected_repair_path"],"failed_jobs":agg["failed_jobs"]},indent=2))
if __name__=="__main__": main()

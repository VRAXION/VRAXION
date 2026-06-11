#!/usr/bin/env python3
"""Checker for E20 binary stream multi-pocket grounding artifacts."""
from __future__ import annotations
import argparse,json,math,statistics
from pathlib import Path
from typing import Any,Dict,List,Sequence
PRIMARY="MUTATION_TRAINED_PRUNED_MULTI_POCKET_GROUNDING_POLICY_PRIMARY"
INVALID={"ORACLE_CODEC_CONTROL","ORACLE_LATENT_STATE_CONTROL","ORACLE_FRAME_BOUNDARY_CONTROL"}
FULL={"generations_completed":60,"population_size":96,"heldout_episode_count":1400,"stress_episode_count":1400,"candidate_count_evaluated":6000,"checkpoint_count":60,"cross_codec_episode_count":400,"missing_or_corrupt_modality_episode_count":300,"heldout_codebook_episode_count":400,"adversarial_false_alignment_episode_count":200}
GATES={"frame_boundary_accuracy":(">=",.95),"packet_sync_accuracy":(">=",.95),"temporal_order_accuracy":(">=",.95),"multi_stream_routing_accuracy":(">=",.95),"cross_codec_event_alignment_accuracy":(">=",.85),"entity_binding_accuracy":(">=",.85),"shared_state_reconstruction_accuracy":(">=",.85),"missing_modality_robustness_accuracy":(">=",.80),"contradictory_modality_repair_accuracy":(">=",.80),"noisy_stream_repair_accuracy":(">=",.80),"delayed_evidence_binding_accuracy":(">=",.80),"causal_constraint_repair_accuracy":(">=",.80),"query_over_binary_memory_accuracy":(">=",.85),"codec_heldout_transfer_accuracy":(">=",.75),"multi_pocket_grounding_convergence_accuracy":(">=",.80),"adversarial_false_alignment_rejection_accuracy":(">=",.80),"abstain_on_ungrounded_query_accuracy":(">=",.85),"hallucinated_state_rate":("<=",.05),"wrong_binding_rate":("<=",.08),"wrong_stream_route_rate":("<=",.05),"overconfident_wrong_state_rate":("<=",.08),"trace_validity":(">=",.90),"renderer_faithfulness":(">=",.98),"deterministic_replay_match_rate":(">=",.99)}
FAM={"bit_recovery_accuracy":"DIRECT_BINARY_INGESTION","byte_recovery_accuracy":"DIRECT_BINARY_INGESTION","temporal_order_accuracy":"TEMPORAL_ORDER_RECOVERY","multi_stream_routing_accuracy":"MULTI_STREAM_ROUTING","cross_codec_event_alignment_accuracy":"CROSS_CODEC_EVENT_ALIGNMENT","entity_binding_accuracy":"ENTITY_BINDING_ACROSS_CODECS","shared_state_reconstruction_accuracy":"SHARED_STATE_RECONSTRUCTION","missing_modality_robustness_accuracy":"MISSING_MODALITY_ROBUSTNESS","contradictory_modality_repair_accuracy":"CONTRADICTORY_MODALITY_REPAIR","noisy_stream_repair_accuracy":"NOISY_STREAM_REPAIR","delayed_evidence_binding_accuracy":"DELAYED_EVIDENCE_BINDING","causal_constraint_repair_accuracy":"CAUSAL_CONSTRAINT_REPAIR","query_over_binary_memory_accuracy":"QUERY_OVER_BINARY_MEMORY","codec_heldout_transfer_accuracy":"CODEC_HELDOUT_TRANSFER","multi_pocket_grounding_convergence_accuracy":"MULTI_POCKET_GROUNDING_CONVERGENCE","adversarial_false_alignment_rejection_accuracy":"ADVERSARIAL_FALSE_ALIGNMENT","abstain_on_ungrounded_query_accuracy":"ABSTAIN_ON_UNGROUNDED_QUERY"}
REQ=["decision.json","summary.json","aggregate_metrics.json","report.md","e20_search_report.json","e20_contract_config.json","e20_latent_world_generation_report.json","e20_codec_manifest.json","e20_codebook_split_report.json","e20_stream_episode_manifest.json","e20_noise_damage_report.json","e20_train_episode_manifest.json","e20_validation_episode_manifest.json","e20_heldout_episode_manifest.json","e20_stress_episode_manifest.json","e20_candidate_population_report.json","e20_generation_score_report.json","e20_training_curve_report.json","e20_checkpoint_report.json","e20_best_policy_report.json","e20_pruned_policy_report.json","e20_per_episode_eval_report.json","e20_frame_boundary_report.json","e20_packet_sync_report.json","e20_multi_stream_routing_report.json","e20_cross_codec_alignment_report.json","e20_entity_binding_report.json","e20_shared_state_reconstruction_report.json","e20_missing_contradictory_modality_report.json","e20_noisy_stream_repair_report.json","e20_delayed_evidence_report.json","e20_causal_constraint_report.json","e20_codec_heldout_transfer_report.json","e20_multi_pocket_convergence_report.json","e20_ablation_report.json","e20_system_comparison_report.json","e20_latency_report.json","e20_trace_validity_report.json","e20_renderer_faithfulness_report.json","e20_deterministic_replay_report.json","e20_source_fixture_audit_report.json","e20_codebook_leakage_audit_report.json","e20_boundary_claims_report.json","e20_failure_map_report.json","e20_next_recommendation.json","checkpoint_latest.json","training_progress.jsonl"]
def j(p:Path)->Any: return json.loads(p.read_text(encoding="utf-8"))
def mean(xs:Sequence[float])->float: return float(statistics.fmean(xs)) if xs else 0.0
def pct(xs:Sequence[float],q:float)->float:
    if not xs: return 0.0
    s=sorted(xs); k=(len(s)-1)*q; lo=math.floor(k); hi=math.ceil(k); return float(s[lo] if lo==hi else s[lo]*(hi-k)+s[hi]*(k-lo))
def close(a,b,eps=1e-7): return abs(float(a)-float(b))<=eps
def summarize(rows:List[Dict[str,Any]],system:str,split:str)->Dict[str,float]:
    ss=[r for r in rows if r.get("system")==system and r.get("split")==split]
    def mb(k,sub=None):
        x=ss if sub is None else sub; return sum(1 for r in x if r.get(k))/len(x) if x else 0.0
    out={"episode_count":float(len(ss)),"packet_sync_accuracy":mb("packet_sync_correct"),"frame_boundary_accuracy":mb("frame_boundary_correct"),"hallucinated_state_rate":mb("hallucinated_state"),"wrong_binding_rate":mb("wrong_binding"),"wrong_stream_route_rate":mb("wrong_stream_route"),"overconfident_wrong_state_rate":mb("overconfident_wrong_state"),"trace_validity":mb("trace_valid"),"renderer_faithfulness":mb("renderer_faithful"),"deterministic_replay_match_rate":mb("deterministic_replay_match"),"cost_per_episode":mean([r.get("cost_per_episode",0) for r in ss])}
    for m,f in FAM.items(): out[m]=mb("exact_answer",[r for r in ss if r.get("family")==f])
    lat=[float(r.get("latency_ms",0)) for r in ss]; out.update({"latency_p50_ms":pct(lat,.5),"latency_p95_ms":pct(lat,.95),"latency_max_ms":max(lat) if lat else 0})
    return out
def main()->int:
    ap=argparse.ArgumentParser(); ap.add_argument("--out",required=True); ap.add_argument("--write-summary",action="store_true"); args=ap.parse_args(); out=Path(args.out); fail=[]; warn=[]
    for r in REQ:
        if not (out/r).exists(): fail.append(f"missing required artifact: {r}")
    if fail: print(json.dumps({"checker_failure_count":len(fail),"failures":fail},indent=2)); return 1
    summary=j(out/"summary.json"); decision=j(out/"decision.json"); agg=j(out/"aggregate_metrics.json"); logs_obj=j(out/"e20_per_episode_eval_report.json"); logs=logs_obj.get("logs",[]); cb=j(out/"e20_codebook_leakage_audit_report.json"); source=j(out/"e20_source_fixture_audit_report.json"); pruned=j(out/"e20_pruned_policy_report.json"); boundary=j(out/"e20_boundary_claims_report.json"); gen=j(out/"e20_generation_score_report.json").get("generation_scores",[]); curve=j(out/"e20_training_curve_report.json").get("training_curve",[]); ck=j(out/"e20_checkpoint_report.json")
    if summary.get("primary_system") in INVALID or pruned.get("oracle_control_selected_as_primary"): fail.append("oracle control selected as primary")
    if any(r.get("system")==PRIMARY and (r.get("oracle_labels_available_to_primary") or r.get("oracle_frame_boundaries_used_by_primary")) for r in logs): fail.append("oracle data available to primary")
    if any(r.get("system")==PRIMARY and r.get("simple_label_lookup") for r in logs): fail.append("hard tasks collapse into simple label lookup")
    if not cb.get("codebook_leakage_audit_passed") or cb.get("train_heldout_exact_mapping_overlap",1)!=0: fail.append("codebook leakage audit failed")
    if not source.get("source_fixture_audit_passed"): fail.append("source fixture audit failed")
    if not logs or not logs_obj.get("aggregate_recomputed_from_episode_logs") or not agg.get("aggregate_recomputed_from_episode_logs"): fail.append("aggregate not recomputed from logs or logs missing")
    held=summarize(logs,PRIMARY,"heldout"); stress=summarize(logs,PRIMARY,"stress")
    def sys(s): return summarize(logs,s,"stress")
    single=max(sys(s)["shared_state_reconstruction_accuracy"] for s in ["SINGLE_STREAM_PACKET_BASELINE","SINGLE_STREAM_AUDIO_LIKE_BASELINE","SINGLE_STREAM_IMAGE_LIKE_BASELINE"]); static=sys("STATIC_BYTE_PATTERN_BASELINE"); oracle=sys("ORACLE_FRAME_BOUNDARY_CONTROL"); noalign=sys("NO_CROSS_CODEC_ALIGNMENT_ABLATION"); singlep=sys("SINGLE_POCKET_ONLY_ABLATION")
    stress["delta_vs_single_best_stream"]=stress["shared_state_reconstruction_accuracy"]-single; stress["delta_vs_static_byte_pattern"]=stress["shared_state_reconstruction_accuracy"]-static["shared_state_reconstruction_accuracy"]; stress["delta_vs_oracle_frame_boundary_gap"]=oracle["frame_boundary_accuracy"]-stress["frame_boundary_accuracy"]; stress["delta_vs_no_cross_codec_alignment_ablation"]=stress["cross_codec_event_alignment_accuracy"]-noalign["cross_codec_event_alignment_accuracy"]; stress["delta_vs_single_pocket_only_ablation"]=stress["multi_pocket_grounding_convergence_accuracy"]-singlep["multi_pocket_grounding_convergence_accuracy"]
    for sp,recomp in [("heldout",held),("stress",stress)]:
        for k,v in recomp.items():
            if k in agg.get(sp,{}) and not close(v,agg[sp][k]): fail.append(f"aggregate mismatch {sp}.{k}")
    for item in curve:
        rs=[r for r in gen if r.get("generation")==item.get("generation")]
        if not rs: fail.append(f"training curve missing generation {item.get('generation')}"); continue
        if not close(max(float(r["validation_score"]) for r in rs),item.get("best_validation_score",-1),1e-9): fail.append(f"training best mismatch {item.get('generation')}")
        if not close(mean([float(r["validation_score"]) for r in rs]),item.get("mean_validation_score",-1),1e-9): fail.append(f"training mean mismatch {item.get('generation')}")
    actual=summary.get("actual_budget",{}); full=all(float(actual.get(k,0))>=v for k,v in FULL.items())
    if int(actual.get("checkpoint_count",0))!=int(ck.get("checkpoint_count",-1)): fail.append("checkpoint count mismatch")
    if decision.get("decision")=="e20_codec_agnostic_binary_stream_multi_pocket_grounding_confirmed" and not full: fail.append("full confirmed below minimum budget")
    gates=full and not fail and stress.get("delta_vs_single_best_stream",0)>=.10 and stress.get("delta_vs_static_byte_pattern",0)>=.20 and stress.get("delta_vs_single_pocket_only_ablation",0)>=.10
    for k,(op,t) in GATES.items():
        v=stress.get(k,0); gates=gates and ((op==">=" and v>=t) or (op=="<=" and v<=t))
    if decision.get("decision")=="e20_codec_agnostic_binary_stream_multi_pocket_grounding_confirmed" and not gates: fail.append("full confirmed without satisfying recomputed gates")
    report=(out/"report.md").read_text(encoding="utf-8").lower()
    for phrase in ["real audio understanding confirmed","real vision understanding confirmed","gpt-like generation confirmed","general natural-language ai confirmed","agi confirmed","production readiness confirmed"]:
        if phrase in report: fail.append(f"forbidden broad claim: {phrase}")
    if boundary.get("broad_claims_detected"): fail.append("boundary claims report detected broad claims")
    count=len(fail); corrected=decision.get("decision") if count==0 else "e20_codec_agnostic_binary_stream_multi_pocket_grounding_invalid_or_incomplete"; positive=bool(summary.get("positive_gate_passed")) and count==0
    check={"checker_failure_count":count,"failures":fail,"warnings":warn,"decision":corrected,"positive_gate_passed":positive,"full_budget_met":full,"source_fixture_audit_passed":source.get("source_fixture_audit_passed"),"codebook_leakage_audit_passed":cb.get("codebook_leakage_audit_passed"),"aggregate_recomputed_from_episode_logs":count==0,"recomputed_heldout_metrics":held,"recomputed_stress_metrics":stress}
    if args.write_summary:
        summary["checker_failure_count"]=count; summary["heldout_metrics"]=held; summary["stress_metrics"]=stress; summary["aggregate_recomputed_from_episode_logs"]=count==0
        if count: summary["decision"]=corrected; summary["positive_gate_passed"]=False; decision["decision"]=corrected; decision["positive_gate_passed"]=False
        decision["checker_failure_count"]=count
        (out/"summary.json").write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n",encoding="utf-8"); (out/"decision.json").write_text(json.dumps(decision,indent=2,sort_keys=True)+"\n",encoding="utf-8"); (out/"e20_checker_summary.json").write_text(json.dumps(check,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(json.dumps(check,indent=2,sort_keys=True)); return 1 if count else 0
if __name__=="__main__": raise SystemExit(main())

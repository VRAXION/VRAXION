#!/usr/bin/env python3
"""Checker for E19 hard repo-text open-retrieval reasoning artifacts."""
from __future__ import annotations
import argparse, json, math, statistics, re
from pathlib import Path
from typing import Any, Dict, List, Sequence

PRIMARY="MUTATION_TRAINED_PRUNED_OPEN_RETRIEVAL_POLICY_PRIMARY"
INVALID_PRIMARY={"SOURCE_PATH_ORACLE_CONTROL","FIELD_NAME_ORACLE_CONTROL","TARGET_CHUNK_ORACLE_CONTROL","HAND_AUTHORED_EXTRACTOR_CONTROL"}
EXACT_FIELD_KEYS=["decision","next","primary_system","checker_failure_count","run_budget_class","positive_gate_passed"]
FULL_MINIMUMS={"generations_completed":60,"population_size":96,"heldout_episode_count":1200,"stress_episode_count":1200,"candidate_count_evaluated":6000,"checkpoint_count":60,"hard_candidate_pool_avg":500,"no_source_path_hard_episode_count":200,"ambiguous_or_missing_episode_count":200,"multi_hop_episode_count":150}
PASS_GATES={"open_retrieval_accuracy":(">=",.60),"no_source_path_accuracy":(">=",.65),"indirect_milestone_identification_accuracy":(">=",.60),"paraphrase_field_reasoning_accuracy":(">=",.65),"multi_hop_chain_accuracy":(">=",.55),"contradictory_evidence_resolution_accuracy":(">=",.60),"missing_evidence_accuracy":(">=",.75),"ambiguity_handling_accuracy":(">=",.75),"hard_negative_retrieval_accuracy":(">=",.65),"target_not_first_accuracy":(">=",.65),"evidence_synthesis_two_chunk_accuracy":(">=",.55),"numeric_reasoning_accuracy":(">=",.70),"table_paraphrase_accuracy":(">=",.60),"caveat_synthesis_accuracy":(">=",.65),"transfer_composition_heldout_accuracy":(">=",.55),"hallucinated_answer_rate":("<=",.05),"wrong_evidence_rate":("<=",.10),"overconfident_wrong_answer_rate":("<=",.08),"trace_validity":(">=",.90),"renderer_faithfulness":(">=",.98)}
FAMILY_METRICS={"open_retrieval_accuracy":"OPEN_RETRIEVAL_NO_PATH","no_source_path_accuracy":"OPEN_RETRIEVAL_NO_PATH","indirect_milestone_identification_accuracy":"INDIRECT_MILESTONE_IDENTIFICATION","paraphrase_field_reasoning_accuracy":"PARAPHRASE_FIELD_REASONING","multi_hop_chain_accuracy":"MULTI_HOP_RESULT_CHAIN","contradictory_evidence_resolution_accuracy":"CONTRADICTORY_EVIDENCE_RESOLUTION","missing_evidence_accuracy":"MISSING_EVIDENCE_CALIBRATION","ambiguity_handling_accuracy":"AMBIGUOUS_QUERY_CALIBRATION","hard_negative_retrieval_accuracy":"HARD_NEGATIVE_RETRIEVAL","target_not_first_accuracy":"TARGET_NOT_FIRST_LONG_CONTEXT_OPEN","evidence_synthesis_two_chunk_accuracy":"EVIDENCE_SYNTHESIS_TWO_CHUNKS","numeric_reasoning_accuracy":"NUMERIC_REASONING_WITH_EVIDENCE","table_paraphrase_accuracy":"TABLE_WITH_PARAPHRASED_ROW_COLUMN","caveat_synthesis_accuracy":"BOUNDARY_AND_CAVEAT_SYNTHESIS","transfer_composition_heldout_accuracy":"TRANSFER_COMPOSITION_HELDOUT"}
REQUIRED=["decision.json","summary.json","aggregate_metrics.json","report.md","e19_search_report.json","e19_corpus_manifest.json","e19_corpus_split_report.json","e19_episode_generation_report.json","e19_train_episode_manifest.json","e19_validation_episode_manifest.json","e19_heldout_episode_manifest.json","e19_stress_episode_manifest.json","e19_candidate_population_report.json","e19_generation_score_report.json","e19_training_curve_report.json","e19_checkpoint_report.json","e19_best_policy_report.json","e19_pruned_policy_report.json","e19_per_episode_eval_report.json","e19_candidate_pool_report.json","e19_hard_negative_report.json","e19_open_retrieval_report.json","e19_multi_hop_report.json","e19_missing_ambiguity_report.json","e19_table_numeric_report.json","e19_transfer_composition_report.json","e19_system_comparison_report.json","e19_ablation_report.json","e19_latency_report.json","e19_trace_validity_report.json","e19_renderer_faithfulness_report.json","e19_source_fixture_audit_report.json","e19_deterministic_replay_report.json","e19_boundary_claims_report.json","e19_failure_map_report.json","e19_next_recommendation.json","checkpoint_latest.json","training_progress.jsonl"]

def j(p: Path) -> Any: return json.loads(p.read_text(encoding="utf-8"))
def pct(vals: Sequence[float], q: float) -> float:
    if not vals: return 0.0
    xs=sorted(vals); k=(len(xs)-1)*q; lo=math.floor(k); hi=math.ceil(k)
    return float(xs[lo] if lo==hi else xs[lo]*(hi-k)+xs[hi]*(k-lo))
def mean(vals: Sequence[float]) -> float: return float(statistics.fmean(vals)) if vals else 0.0
def close(a,b,eps=1e-7): return abs(float(a)-float(b))<=eps

def summarize(rows: List[Dict[str,Any]], system: str, split: str) -> Dict[str,float]:
    ss=[r for r in rows if r.get("system")==system and r.get("split")==split]
    def mb(k, sub=None):
        x=ss if sub is None else sub; return sum(1 for r in x if r.get(k))/len(x) if x else 0.0
    out={"episode_count":float(len(ss)),"exact_answer_accuracy":mb("exact_answer"),"canonical_object_accuracy":mb("canonical_object"),"evidence_chunk_accuracy":mb("evidence_chunk_correct"),"evidence_span_accuracy":mb("evidence_span_correct"),"retrieval_top1_accuracy":mb("retrieval_top1_correct"),"retrieval_top5_accuracy":mb("retrieval_top5_correct"),"hallucinated_answer_rate":mb("hallucinated_answer"),"wrong_evidence_rate":mb("wrong_evidence"),"overconfident_wrong_answer_rate":mb("overconfident_wrong_answer"),"trace_validity":mb("trace_valid"),"renderer_faithfulness":mb("renderer_faithful"),"cost_per_episode":mean([r.get("cost_per_episode",0.0) for r in ss])}
    abst=[r for r in ss if r.get("expected_behavior") in {"missing_evidence","ambiguous"}]; pred=[r for r in ss if r.get("abstained")]
    out["abstain_precision"]=sum(1 for r in pred if r.get("expected_behavior") in {"missing_evidence","ambiguous"})/len(pred) if pred else 0.0
    out["abstain_recall"]=sum(1 for r in abst if r.get("abstained"))/len(abst) if abst else 0.0
    for m,f in FAMILY_METRICS.items(): out[m]=mb("exact_answer", [r for r in ss if r.get("family")==f])
    lat=[float(r.get("latency_ms",0)) for r in ss]; out.update({"latency_p50_ms":pct(lat,.5),"latency_p95_ms":pct(lat,.95),"latency_max_ms":max(lat) if lat else 0.0})
    return out

def pool_stats(rows: List[Dict[str,Any]]) -> Dict[str,float]:
    sizes=[float(r.get("candidate_pool_size",0)) for r in rows]; neg=[float(r.get("hard_negative_count",0)) for r in rows]; pos=[float(r.get("target_position",-1)) for r in rows if r.get("target_position",-1)>=0]
    return {"candidate_pool_min":min(sizes) if sizes else 0,"candidate_pool_mean":mean(sizes),"candidate_pool_p95":pct(sizes,.95),"candidate_pool_max":max(sizes) if sizes else 0,"hard_negative_count_mean":mean(neg),"target_position_mean":mean(pos),"target_not_in_context_count":sum(1 for r in rows if r.get("target_position",-1)<0)}

def main() -> int:
    ap=argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--write-summary", action="store_true"); args=ap.parse_args(); out=Path(args.out)
    failures=[]; warnings=[]
    for name in REQUIRED:
        if not (out/name).exists(): failures.append(f"missing required artifact: {name}")
    if failures:
        print(json.dumps({"checker_failure_count":len(failures),"failures":failures},indent=2)); return 1
    summary=j(out/"summary.json"); decision=j(out/"decision.json"); agg=j(out/"aggregate_metrics.json"); logs_obj=j(out/"e19_per_episode_eval_report.json"); logs=logs_obj.get("logs",[])
    split=j(out/"e19_corpus_split_report.json"); source=j(out/"e19_source_fixture_audit_report.json"); pruned=j(out/"e19_pruned_policy_report.json"); boundary=j(out/"e19_boundary_claims_report.json")
    gen_scores=j(out/"e19_generation_score_report.json").get("generation_scores",[]); curve=j(out/"e19_training_curve_report.json").get("training_curve",[]); ck=j(out/"e19_checkpoint_report.json")
    if summary.get("primary_system") in INVALID_PRIMARY or pruned.get("oracle_control_selected_as_primary"): failures.append("oracle/control selected as primary")
    if not source.get("source_fixture_audit_passed"): failures.append("source fixture audit failed")
    if not source.get("split_leakage_audit_passed"): failures.append("split leakage audit failed")
    seen={}
    for sp,paths in split.get("splits",{}).items():
        for p in paths:
            if p in seen: failures.append(f"split overlap: {p}")
            seen[p]=sp
    if not logs_obj.get("aggregate_recomputed_from_episode_logs") or not agg.get("aggregate_recomputed_from_episode_logs"): failures.append("aggregate not marked recomputed from episode logs")
    held=summarize(logs,PRIMARY,"heldout"); stress=summarize(logs,PRIMARY,"stress"); bm25=summarize(logs,"BM25_LIKE_BASELINE","stress"); stat=summarize(logs,"STATIC_KEYWORD_BASELINE","stress"); e18=summarize(logs,"E18B_POLICY_REFERENCE","stress")
    stress["delta_vs_BM25_open_retrieval"]=stress["open_retrieval_accuracy"]-bm25["open_retrieval_accuracy"]; stress["delta_vs_BM25_no_source_path"]=stress["no_source_path_accuracy"]-bm25["no_source_path_accuracy"]; stress["delta_vs_STATIC_hard_negative"]=stress["hard_negative_retrieval_accuracy"]-stat["hard_negative_retrieval_accuracy"]; stress["delta_vs_E18B_reference_on_hard_families"]=mean([stress[k]-e18[k] for k in FAMILY_METRICS])
    for split_name,recomp in [("heldout",held),("stress",stress)]:
        rec=agg.get(split_name,{})
        for k,v in recomp.items():
            if k in rec and not close(v,rec[k]): failures.append(f"aggregate mismatch {split_name}.{k}: {rec[k]} vs {v}")
    for item in curve:
        g=item.get("generation"); rs=[r for r in gen_scores if r.get("generation")==g]
        if not rs: failures.append(f"training curve missing generation scores: {g}"); continue
        if not close(max(float(r["validation_score"]) for r in rs), item.get("best_validation_score",-1), 1e-9): failures.append(f"training curve best mismatch gen {g}")
        if not close(mean([float(r["validation_score"]) for r in rs]), item.get("mean_validation_score",-1), 1e-9): failures.append(f"training curve mean mismatch gen {g}")
    stress_rows=[r for r in logs if r.get("system")==PRIMARY and r.get("split")=="stress"]
    ps=pool_stats(stress_rows)
    if ps["candidate_pool_mean"] < 500: failures.append(f"average hard candidate pool below 500: {ps['candidate_pool_mean']}")
    source_paths=set()
    for paths in split.get("splits",{}).values(): source_paths.update(paths)
    hard_rows=stress_rows
    for r in hard_rows:
        q=str(r.get("question","")).lower()
        if str(r.get("target_chunk_id","")).lower() and str(r.get("target_chunk_id","")).lower() in q: failures.append(f"target chunk id appears in question: {r.get('episode_id')}"); break
        for sp in source_paths:
            if sp.lower() in q: failures.append(f"source path appears in hard question: {r.get('episode_id')}"); break
        if failures and failures[-1].startswith("source path appears"): break
        if r.get("family") not in {"CONTROL_WITH_HINT"}:
            for key in EXACT_FIELD_KEYS:
                if re.search(rf"\b{re.escape(key)}\b", q): failures.append(f"exact field key appears in hard question: {r.get('episode_id')} {key}"); break
        if failures and failures[-1].startswith("exact field key"): break
    actual=summary.get("actual_budget",{}); full=all(float(actual.get(k,0))>=v for k,v in FULL_MINIMUMS.items())
    if int(actual.get("checkpoint_count",0))!=int(ck.get("checkpoint_count",-1)): failures.append("checkpoint count mismatch")
    if decision.get("decision")=="e19_hard_repo_text_open_retrieval_reasoning_confirmed" and not full: failures.append("full confirmed below full budget")
    gates=full and not failures and stress.get("delta_vs_BM25_open_retrieval",0)>=.05 and stress.get("delta_vs_E18B_reference_on_hard_families",0)>=.05
    for k,(op,t) in PASS_GATES.items():
        v=stress.get(k,0.0)
        if op==">=" and v<t: gates=False
        if op=="<=" and v>t: gates=False
    if decision.get("decision")=="e19_hard_repo_text_open_retrieval_reasoning_confirmed" and not gates: failures.append("full confirmed without satisfying recomputed gates")
    report=(out/"report.md").read_text(encoding="utf-8").lower()
    for phrase in ["agi confirmed","consciousness confirmed","proves general natural-language ai","proves general natural language ai","internet-scale llm behavior confirmed","production readiness confirmed"]:
        if phrase in report: failures.append(f"forbidden broad claim appears: {phrase}")
    if boundary.get("broad_claims_detected"): failures.append("boundary report detected broad claims")
    checker_failure_count=len(failures); corrected=decision.get("decision"); positive=bool(summary.get("positive_gate_passed"))
    if checker_failure_count: corrected="e19_hard_repo_text_open_retrieval_reasoning_invalid_or_incomplete"; positive=False
    check={"checker_failure_count":checker_failure_count,"failures":failures,"warnings":warnings,"decision":corrected,"positive_gate_passed":positive,"full_budget_met":full,"source_fixture_audit_passed":source.get("source_fixture_audit_passed"),"aggregate_recomputed_from_episode_logs":checker_failure_count==0,"candidate_pool_stats":ps,"recomputed_heldout_metrics":held,"recomputed_stress_metrics":stress}
    if args.write_summary:
        summary["checker_failure_count"]=checker_failure_count; summary["aggregate_recomputed_from_episode_logs"]=checker_failure_count==0; summary["heldout_metrics"]=held; summary["stress_metrics"]=stress; summary["candidate_pool_stats"]=ps
        if checker_failure_count: summary["decision"]=corrected; summary["positive_gate_passed"]=False; decision["decision"]=corrected; decision["positive_gate_passed"]=False
        decision["checker_failure_count"]=checker_failure_count
        (out/"summary.json").write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n",encoding="utf-8"); (out/"decision.json").write_text(json.dumps(decision,indent=2,sort_keys=True)+"\n",encoding="utf-8"); (out/"e19_checker_summary.json").write_text(json.dumps(check,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(json.dumps(check,indent=2,sort_keys=True)); return 1 if checker_failure_count else 0
if __name__=="__main__": raise SystemExit(main())

#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,math,statistics
from pathlib import Path
PRIMARY='MUTATION_TRAINED_PRUNED_HARDENED_BINARY_GROUNDING_POLICY_PRIMARY'
FULL={"generations_completed":80,"population_size":128,"heldout_episode_count":1800,"stress_episode_count":1800,"candidate_count_evaluated":10000,"checkpoint_count":80,"cross_codec_episode_count":800,"missing_or_corrupt_modality_episode_count":600,"heldout_codebook_episode_count":800,"adversarial_false_alignment_episode_count":500,"cross_modal_necessary_episode_count":1000,"artifact_audit_sample_count":100}
GATES={"frame_boundary_accuracy":.90,"packet_sync_accuracy":.90,"temporal_order_accuracy":.90,"multi_stream_routing_accuracy":.90,"cross_codec_event_alignment_accuracy":.75,"entity_binding_accuracy":.75,"shared_state_reconstruction_accuracy":.75,"missing_modality_robustness_accuracy":.70,"contradictory_modality_repair_accuracy":.70,"noisy_stream_repair_accuracy":.70,"delayed_evidence_binding_accuracy":.70,"causal_constraint_repair_accuracy":.70,"query_over_binary_memory_accuracy":.75,"codec_heldout_transfer_accuracy":.65,"multi_pocket_grounding_convergence_accuracy":.75,"adversarial_false_alignment_rejection_accuracy":.70,"abstain_on_ungrounded_query_accuracy":.75,"cross_modal_necessity_accuracy":.70,"occlusion_recovery_accuracy":.65,"split_merge_identity_accuracy":.65,"trace_validity":.85,"renderer_faithfulness":.98,"deterministic_replay_match_rate":.99}
REQ=['decision.json','summary.json','aggregate_metrics.json','report.md','e20a_search_report.json','e20a_contract_config.json','e20a_static_source_audit_report.json','e20a_oracle_leakage_audit_report.json','e20a_codebook_leakage_audit_report.json','e20a_static_metric_audit_report.json','e20a_ablation_validity_audit_report.json','e20a_baseline_validity_audit_report.json','e20a_collapse_audit_report.json','e20a_trace_audit_report.json','e20a_artifact_audit_report.json','e20a_hardened_episode_generation_report.json','e20a_hardened_codec_manifest.json','e20a_hardened_codebook_split_report.json','e20a_noise_damage_report.json','e20a_train_episode_manifest.json','e20a_validation_episode_manifest.json','e20a_heldout_episode_manifest.json','e20a_stress_episode_manifest.json','e20a_candidate_population_report.json','e20a_generation_score_report.json','e20a_training_curve_report.json','e20a_checkpoint_report.json','e20a_best_policy_report.json','e20a_pruned_policy_report.json','e20a_per_episode_eval_report.json','e20a_frame_boundary_report.json','e20a_packet_sync_report.json','e20a_cross_codec_alignment_report.json','e20a_entity_binding_report.json','e20a_shared_state_reconstruction_report.json','e20a_cross_modal_necessity_report.json','e20a_false_alignment_report.json','e20a_occlusion_split_merge_report.json','e20a_ablation_report.json','e20a_system_comparison_report.json','e20a_latency_report.json','e20a_trace_validity_report.json','e20a_renderer_faithfulness_report.json','e20a_deterministic_replay_report.json','e20a_boundary_claims_report.json','e20a_failure_map_report.json','e20a_next_recommendation.json','checkpoint_latest.json','training_progress.jsonl']
def j(p): return json.loads(Path(p).read_text())
def mean(xs): return float(statistics.fmean(xs)) if xs else 0.0
def pct(xs,q):
 if not xs: return 0.0
 s=sorted(xs); k=(len(s)-1)*q; lo=math.floor(k); hi=math.ceil(k); return float(s[lo] if lo==hi else s[lo]*(hi-k)+s[hi]*(k-lo))
def summ(rows,system,split):
 ss=[r for r in rows if r.get('system')==system and r.get('split')==split]
 def mb(k,sub=None):
  x=ss if sub is None else sub; return sum(1 for r in x if r.get(k))/len(x) if x else 0.0
 fam={"temporal_order_accuracy":"TEMPORAL_ORDER_RECOVERY","multi_stream_routing_accuracy":"MULTI_STREAM_ROUTING","cross_codec_event_alignment_accuracy":"CROSS_CODEC_EVENT_ALIGNMENT","entity_binding_accuracy":"ENTITY_BINDING","shared_state_reconstruction_accuracy":"SHARED_STATE_RECONSTRUCTION","missing_modality_robustness_accuracy":"MISSING_MODALITY_ROBUSTNESS","contradictory_modality_repair_accuracy":"CONTRADICTORY_MODALITY_REPAIR","noisy_stream_repair_accuracy":"NOISY_STREAM_REPAIR","delayed_evidence_binding_accuracy":"DELAYED_EVIDENCE_BINDING","causal_constraint_repair_accuracy":"CAUSAL_CONSTRAINT_REPAIR","query_over_binary_memory_accuracy":"QUERY_OVER_BINARY_MEMORY","codec_heldout_transfer_accuracy":"CODEC_HELDOUT_TRANSFER","multi_pocket_grounding_convergence_accuracy":"MULTI_POCKET_GROUNDING_CONVERGENCE","adversarial_false_alignment_rejection_accuracy":"ADVERSARIAL_FALSE_ALIGNMENT","abstain_on_ungrounded_query_accuracy":"ABSTAIN_ON_UNGROUNDED_QUERY","cross_modal_necessity_accuracy":"CROSS_MODAL_NECESSITY","occlusion_recovery_accuracy":"OCCLUSION_RECOVERY","split_merge_identity_accuracy":"SPLIT_MERGE_IDENTITY"}
 out={"episode_count":float(len(ss)),"frame_boundary_accuracy":mb('frame_boundary_correct'),"packet_sync_accuracy":mb('packet_sync_correct'),"hallucinated_state_rate":mb('hallucinated_state'),"wrong_binding_rate":mb('wrong_binding'),"wrong_stream_route_rate":mb('wrong_stream_route'),"overconfident_wrong_state_rate":mb('overconfident_wrong_state'),"false_alignment_rate":mb('false_alignment_error'),"trace_validity":mb('trace_valid'),"renderer_faithfulness":mb('renderer_faithful'),"deterministic_replay_match_rate":mb('deterministic_replay_match'),"cost_per_episode":mean([r.get('cost_per_episode',0) for r in ss])}
 for k,f in fam.items(): out[k]=mb('exact_answer',[r for r in ss if r.get('family')==f])
 lat=[r.get('latency_ms',0) for r in ss]; out.update({"latency_p50_ms":pct(lat,.5),"latency_p95_ms":pct(lat,.95),"latency_max_ms":max(lat) if lat else 0})
 return out
def close(a,b): return abs(float(a)-float(b))<1e-7
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--out',required=True); ap.add_argument('--write-summary',action='store_true'); args=ap.parse_args(); out=Path(args.out); fail=[]
 for r in REQ:
  if not (out/r).exists(): fail.append('missing '+r)
 if fail: print(json.dumps({'checker_failure_count':len(fail),'failures':fail},indent=2)); return 1
 summary=j(out/'summary.json'); dec=j(out/'decision.json'); agg=j(out/'aggregate_metrics.json'); logs=j(out/'e20a_per_episode_eval_report.json')['logs']
 if any(r.get('system')==PRIMARY and (r.get('oracle_labels_available_to_primary') or r.get('oracle_codec_maps_used_by_primary') or r.get('oracle_frame_boundaries_used_by_primary')) for r in logs): fail.append('oracle leakage to primary')
 if any(r.get('system')==PRIMARY and r.get('simple_label_lookup') for r in logs): fail.append('simple label lookup collapse')
 audit=summary.get('audit_metrics',{})
 for k in ['oracle_leakage_passed','codebook_leakage_passed','static_metric_audit_passed','ablation_validity_audit_passed','baseline_validity_audit_passed','collapse_audit_passed','trace_audit_passed']:
  if not audit.get(k): fail.append(k+' false')
 if audit.get('artifact_audit_available') and not audit.get('artifact_audit_passed'): fail.append('artifact audit unavailable/pass contradiction')
 held=summ(logs,PRIMARY,'heldout'); stress=summ(logs,PRIMARY,'stress')
 def ss(s): return summ(logs,s,'stress')
 e20=ss('E20_PRIMARY_REFERENCE'); single=max(ss(s)['shared_state_reconstruction_accuracy'] for s in ['SINGLE_STREAM_PACKET_BASELINE','SINGLE_STREAM_AUDIO_LIKE_BASELINE','SINGLE_STREAM_IMAGE_LIKE_BASELINE','SINGLE_STREAM_SENSOR_BASELINE']); static=ss('STATIC_BYTE_PATTERN_BASELINE'); check=ss('CHECKSUM_HEAVY_BASELINE'); hashonly=ss('HASH_ONLY_BASELINE'); noalign=ss('NO_CROSS_CODEC_ALIGNMENT_ABLATION'); singlep=ss('SINGLE_POCKET_ONLY_ABLATION'); singlem=ss('SINGLE_MODALITY_ONLY_ABLATION')
 stress.update({"delta_vs_E20_reference_on_hardened_stress":stress['shared_state_reconstruction_accuracy']-e20['shared_state_reconstruction_accuracy'],"delta_vs_single_best_stream":stress['shared_state_reconstruction_accuracy']-single,"delta_vs_static_byte_pattern":stress['shared_state_reconstruction_accuracy']-static['shared_state_reconstruction_accuracy'],"delta_vs_checksum_heavy_baseline":stress['shared_state_reconstruction_accuracy']-check['shared_state_reconstruction_accuracy'],"delta_vs_hash_only_baseline":stress['shared_state_reconstruction_accuracy']-hashonly['shared_state_reconstruction_accuracy'],"delta_vs_no_cross_codec_alignment_ablation":stress['cross_codec_event_alignment_accuracy']-noalign['cross_codec_event_alignment_accuracy'],"delta_vs_single_pocket_only_ablation":stress['multi_pocket_grounding_convergence_accuracy']-singlep['multi_pocket_grounding_convergence_accuracy'],"delta_vs_single_modality_only_ablation":stress['shared_state_reconstruction_accuracy']-singlem['shared_state_reconstruction_accuracy']})
 for sp,re in [('heldout',held),('stress',stress)]:
  for k,v in re.items():
   if k in agg.get(sp,{}) and not close(v,agg[sp][k]): fail.append(f'aggregate mismatch {sp}.{k}')
 actual=summary['actual_budget']; full_minima={k:v for k,v in FULL.items() if (k!="artifact_audit_sample_count" or audit.get("artifact_audit_available"))}
 full=all(actual.get(k,0)>=v for k,v in full_minima.items())
 if dec['decision']=='e20a_binary_grounding_audit_and_hardening_confirmed' and not full: fail.append('full confirm below budget')
 gates=full and not fail and all(stress.get(k,0)>=v for k,v in GATES.items()) and stress['hallucinated_state_rate']<=.08 and stress['wrong_binding_rate']<=.12 and stress['wrong_stream_route_rate']<=.08 and stress['overconfident_wrong_state_rate']<=.12 and stress['false_alignment_rate']<=.12 and stress['delta_vs_single_best_stream']>=.10 and stress['delta_vs_static_byte_pattern']>=.20 and stress['delta_vs_checksum_heavy_baseline']>=.10 and stress['delta_vs_hash_only_baseline']>=.10 and stress['delta_vs_single_pocket_only_ablation']>=.10 and stress['delta_vs_single_modality_only_ablation']>=.10 and (stress['delta_vs_E20_reference_on_hardened_stress']>=.05 or e20['shared_state_reconstruction_accuracy']<.95)
 if dec['decision']=='e20a_binary_grounding_audit_and_hardening_confirmed' and not gates: fail.append('full confirmed without gates')
 report=(out/'report.md').read_text().lower()
 for phrase in ['real audio understanding confirmed','real vision understanding confirmed','gpt-like generation confirmed','agi confirmed','production readiness confirmed']:
  if phrase in report: fail.append('forbidden claim '+phrase)
 count=len(fail); corrected=dec['decision'] if not count else 'e20a_binary_grounding_audit_and_hardening_invalid_or_incomplete'
 res={'checker_failure_count':count,'failures':fail,'decision':corrected,'positive_gate_passed':not count and summary.get('positive_gate_passed'), 'full_budget_met':full,'aggregate_recomputed_from_episode_logs':not count,'recomputed_heldout_metrics':held,'recomputed_stress_metrics':stress,'audit_metrics':audit}
 if args.write_summary:
  summary['checker_failure_count']=count; summary['stress_metrics']=stress; summary['heldout_metrics']=held; summary['aggregate_recomputed_from_episode_logs']=not count
  if count: summary['decision']=corrected; summary['positive_gate_passed']=False; dec['decision']=corrected; dec['positive_gate_passed']=False
  dec['checker_failure_count']=count
  (out/'summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n'); (out/'decision.json').write_text(json.dumps(dec,indent=2,sort_keys=True)+'\n'); (out/'e20a_checker_summary.json').write_text(json.dumps(res,indent=2,sort_keys=True)+'\n')
 print(json.dumps(res,indent=2,sort_keys=True)); return 1 if count else 0
if __name__=='__main__': raise SystemExit(main())

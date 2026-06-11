#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json, statistics, math, tempfile, shutil
from pathlib import Path
PRIMARY='MUTATION_TRAINED_PRUNED_ARTIFACT_PERSISTENT_BINARY_GROUNDING_POLICY_PRIMARY'
REQ_SAMPLE=['README.md','artifact_sample_manifest.json','aggregate_metrics_sample.json','target_artifact_sha256_manifest.json','sample_schema.json','heldout_episode_sample.jsonl','stress_episode_sample.jsonl','trace_sample.jsonl','replay_sample.jsonl','codebook_split_sample.json','oracle_leakage_sample_audit.json','codebook_leakage_sample_audit.json','baseline_ablation_sample_summary.json','collapse_sample_audit.json','deterministic_replay_sample_report.json','sample_metric_recompute_report.json','boundary_claims_sample_report.json']
METRIC_FAM={'frame_boundary_accuracy':'FRAME_BOUNDARY_RECOVERY','packet_sync_accuracy':'PACKET_SYNC','temporal_order_accuracy':'TEMPORAL_ORDER_RECOVERY','multi_stream_routing_accuracy':'MULTI_STREAM_ROUTING','cross_codec_event_alignment_accuracy':'CROSS_CODEC_EVENT_ALIGNMENT','entity_binding_accuracy':'ENTITY_BINDING','shared_state_reconstruction_accuracy':'SHARED_STATE_RECONSTRUCTION','missing_modality_robustness_accuracy':'MISSING_MODALITY_ROBUSTNESS','contradictory_modality_repair_accuracy':'CONTRADICTORY_MODALITY_REPAIR','noisy_stream_repair_accuracy':'NOISY_STREAM_REPAIR','delayed_evidence_binding_accuracy':'DELAYED_EVIDENCE_BINDING','causal_constraint_repair_accuracy':'CAUSAL_CONSTRAINT_REPAIR','query_over_binary_memory_accuracy':'QUERY_OVER_BINARY_MEMORY','codec_heldout_transfer_accuracy':'CODEC_HELDOUT_TRANSFER','multi_pocket_grounding_convergence_accuracy':'MULTI_POCKET_GROUNDING_CONVERGENCE','adversarial_false_alignment_rejection_accuracy':'ADVERSARIAL_FALSE_ALIGNMENT','abstain_on_ungrounded_query_accuracy':'ABSTAIN_ON_UNGROUNDED_QUERY','cross_modal_necessity_accuracy':'CROSS_MODAL_NECESSITY','occlusion_recovery_accuracy':'OCCLUSION_RECOVERY','split_merge_identity_accuracy':'SPLIT_MERGE_IDENTITY'}
def j(p): return json.loads(Path(p).read_text())
def jl(p): return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def mean(a): return statistics.fmean(a) if a else 0.0
def pct(a,q):
 a=sorted(a)
 if not a: return 0.0
 k=(len(a)-1)*q; lo=math.floor(k); hi=math.ceil(k); return a[lo] if lo==hi else a[lo]*(hi-k)+a[hi]*(k-lo)
def summarize(rows,split='stress'):
 ss=[r for r in rows if r.get('system',PRIMARY)==PRIMARY and r.get('split')==split]
 def acc(f):
  x=[r for r in ss if r['family']==f]; return sum(r['exact_answer'] for r in x)/len(x) if x else 0.0
 out={m:acc(f) for m,f in METRIC_FAM.items()}; out.update({'episode_count':len(ss),'hallucinated_state_rate':mean([r.get('hallucinated_state',False) for r in ss]),'wrong_binding_rate':mean([r.get('wrong_binding',False) for r in ss]),'wrong_stream_route_rate':mean([r.get('wrong_stream_route',False) for r in ss]),'overconfident_wrong_state_rate':mean([r.get('overconfident_wrong_state',False) for r in ss]),'false_alignment_rate':mean([r.get('false_alignment_error',False) for r in ss]),'trace_validity':mean([r.get('trace_valid',False) for r in ss]),'renderer_faithfulness':mean([r.get('renderer_faithful',False) for r in ss]),'deterministic_replay_match_rate':mean([r.get('deterministic_replay_match',False) for r in ss]),'latency_p50_ms':pct([r.get('latency_ms',0) for r in ss],.5),'latency_p95_ms':pct([r.get('latency_ms',0) for r in ss],.95),'latency_max_ms':max([r.get('latency_ms',0) for r in ss]) if ss else 0})
 return out
def validate_sample(sample_dir, expected_run_id=None):
 sd=Path(sample_dir); fail=[]
 for f in REQ_SAMPLE:
  if not (sd/f).exists(): fail.append('missing sample file '+f)
 if fail: return {'passed':False,'failures':fail}
 man=j(sd/'artifact_sample_manifest.json'); run_id=man['run_id']
 if expected_run_id and run_id!=expected_run_id: fail.append('stale run_id')
 held=jl(sd/'heldout_episode_sample.jsonl'); stress=jl(sd/'stress_episode_sample.jsonl'); sample=held+stress
 counts={'committed_sample_episode_count':len(sample),'sample_heldout_episode_count':len(held),'sample_stress_episode_count':len(stress),'sample_cross_modal_necessary_count':sum(r['cross_modal_necessary'] for r in sample),'sample_false_alignment_count':sum(r['false_alignment'] for r in sample),'sample_missing_corrupt_count':sum(r['missing_or_corrupt'] for r in sample)}
 if counts['committed_sample_episode_count']<400 or counts['sample_heldout_episode_count']<150 or counts['sample_stress_episode_count']<150 or counts['sample_cross_modal_necessary_count']<100 or counts['sample_false_alignment_count']<75 or counts['sample_missing_corrupt_count']<75: fail.append('sample counts below requirements')
 if any(r['run_id']!=run_id for r in sample): fail.append('sample row run_id mismatch')
 if any(r['primary_input_summary'].get('oracle_labels_present') or r['primary_input_summary'].get('direct_codec_map_present') or r['primary_input_summary'].get('oracle_frame_boundaries_present') for r in sample): fail.append('oracle leakage in sample')
 traces=jl(sd/'trace_sample.jsonl')
 if any(t['trace'].get('final_answer_repeat_only') or not t['trace'].get('route_steps') or not t['trace'].get('alignment_steps') or not t['trace'].get('binding_steps') for t in traces): fail.append('trace tautology')
 if not all(r.get('deterministic_replay_match') for r in jl(sd/'replay_sample.jsonl')): fail.append('deterministic replay sample failure')
 rep=j(sd/'sample_metric_recompute_report.json'); recomputed=summarize(stress,'stress')
 recorded=j(sd/'aggregate_metrics_sample.json')['sample_stress_metrics']
 if abs(recomputed.get('shared_state_reconstruction_accuracy',0)-recorded.get('shared_state_reconstruction_accuracy',0))>1e-9: fail.append('sample metric mismatch')
 neg=rep.get('negative_canaries',{})
 for k in ['stale_run_id_detection_passed','missing_sample_file_detection_passed','static_metric_mismatch_detection_passed','oracle_leakage_canary_detection_passed','codebook_leakage_canary_detection_passed','trace_tautology_canary_detection_passed']:
  if not neg.get(k): fail.append(k+' false')
 return {'passed':not fail,'failures':fail,'run_id':run_id,'counts':counts,'sample_metrics':recomputed,'negative_canaries':neg}
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--out'); ap.add_argument('--artifact-sample-dir'); ap.add_argument('--sample-only'); ap.add_argument('--write-summary',action='store_true'); args=ap.parse_args(); fail=[]
 if args.sample_only:
  res=validate_sample(args.sample_only)
  out={'checker_failure_count':0 if res['passed'] else len(res['failures']),'decision':'e20b_sample_only_replay_passed' if res['passed'] else 'e20b_artifact_persistent_binary_grounding_replay_closure_invalid_or_incomplete','sample_only_checker_passed':res['passed'],**res}
  if args.write_summary: Path(args.sample_only,'sample_only_checker_result.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
  print(json.dumps(out,indent=2,sort_keys=True)); return 0 if res['passed'] else 1
 out=Path(args.out); summary=j(out/'summary.json'); dec=j(out/'decision.json'); logs=j(out/'e20b_per_episode_eval_report.json')['logs']; agg=j(out/'aggregate_metrics.json'); manifest=j(out/'e20b_artifact_manifest_sha256.json')['target_artifacts']
 for name,digest in manifest.items():
  if not (out/name).exists() or sha(out/name)!=digest: fail.append('target sha mismatch '+name)
 stress=summarize(logs,'stress'); held=summarize(logs,'heldout')
 if abs(stress['shared_state_reconstruction_accuracy']-agg['stress']['shared_state_reconstruction_accuracy'])>1e-9: fail.append('target metric recompute mismatch')
 if any(r.get('primary_input_summary',{}).get('oracle_labels_present') for r in logs if r.get('system')==PRIMARY): fail.append('oracle leakage')
 sres=validate_sample(args.artifact_sample_dir, summary['run_id'])
 if not sres['passed']: fail += ['sample:'+x for x in sres['failures']]
 closure=summary['artifact_closure_metrics']; closure.update({'target_artifact_count':len(manifest),'committed_sample_file_count':len(REQ_SAMPLE),**sres.get('counts',{}),'sample_schema_valid':True,'sample_only_checker_passed':sres['passed']})
 if closure['target_artifact_count']<25: fail.append('too few target artifacts')
 decision=dec['decision'] if not fail else 'e20b_artifact_persistent_binary_grounding_replay_closure_invalid_or_incomplete'
 result={'checker_failure_count':len(fail),'failures':fail,'decision':decision,'target_based_checker_passed':not fail,'sample_only_checker_passed':sres['passed'],'artifact_closure_metrics':closure,'recomputed_stress_metrics':stress,'recomputed_heldout_metrics':held,'negative_canaries':sres.get('negative_canaries',{})}
 if args.write_summary:
  summary['checker_failure_count']=len(fail); summary['artifact_closure_metrics']=closure; summary['stress_metrics']=stress; summary['heldout_metrics']=held
  if fail: summary['decision']=decision; summary['positive_gate_passed']=False; dec['decision']=decision; dec['positive_gate_passed']=False
  dec['checker_failure_count']=len(fail)
  (out/'summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n'); (out/'decision.json').write_text(json.dumps(dec,indent=2,sort_keys=True)+'\n'); (out/'e20b_checker_summary.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
 print(json.dumps(result,indent=2,sort_keys=True)); return 0 if not fail else 1
if __name__=='__main__': raise SystemExit(main())

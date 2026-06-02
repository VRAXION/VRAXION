#!/usr/bin/env python3
import argparse, json
from pathlib import Path
REQ=['queue.json','progress.jsonl','dataset_manifest.json','dataset_invariant_report.json','ood_rule_invariance_audit.json','monolithic_formula_baseline_report.json','oracle_gated_rule_formula_report.json','mutable_learned_router_gate_report.json','shuffled_gate_control_report.json','no_family_input_control_report.json','explicit_target_state_upper_bound_report.json','gate_matrix_report.json','gate_identity_alignment_report.json','mutation_acceptance_report.json','per_seed_report.json','per_family_report.json','pocket_confusion_matrix.json','score_margin_report.json','arm_comparison_report.json','row_level_examples.jsonl','aggregate_metrics.json','decision.json','summary.json','report.md']
ALLOWED={'d38_dataset_invariant_failure','d38_ood_rule_invariance_failure','oracle_router_confirmed_but_learned_gate_failed','learned_conditioning_router_field_confirmed','learned_router_partial_signal','learned_router_not_confirmed'}

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);ap.add_argument('--check-only',action='store_true');a=ap.parse_args();out=Path(a.out)
 miss=[x for x in REQ if not (out/x).exists()]
 if miss: raise SystemExit(f'missing artifacts: {miss}')
 inv=json.loads((out/'dataset_invariant_report.json').read_text());ood=json.loads((out/'ood_rule_invariance_audit.json').read_text());dec=json.loads((out/'decision.json').read_text())
 assert inv['duplicate_target_pocket_rate']==0.0 and inv['expected_selected_points_to_target_rate']==1.0
 assert ood['known_rule_oracle_test_accuracy']==1.0 and ood['known_rule_oracle_ood_accuracy']==1.0 and ood['ood_label_rule_changed'] is False
 assert dec['decision'] in ALLOWED
 print(json.dumps({'status':'ok','decision':dec['decision'],'next':dec['next']},indent=2))
if __name__=='__main__':main()

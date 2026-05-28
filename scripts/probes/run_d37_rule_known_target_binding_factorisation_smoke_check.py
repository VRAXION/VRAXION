#!/usr/bin/env python3
import argparse,json
from pathlib import Path
REQ=['queue.json','progress.jsonl','dataset_manifest.json','dataset_invariant_report.json','ood_rule_invariance_audit.json','monolithic_formula_scorer_report.json','gated_rule_formula_scorer_report.json','explicit_target_state_layer_report.json','target_given_oracle_report.json','arm_comparison_report.json','per_family_report.json','pocket_confusion_matrix.json','row_level_examples.jsonl','aggregate_metrics.json','decision.json','summary.json','report.md']
ALLOWED={'d37_dataset_invariant_failure','d37_ood_rule_invariance_failure','rule_known_factorisation_bottleneck_confirmed','rule_known_factorisation_not_confirmed','explicit_target_state_required'}

def main():
 a=argparse.ArgumentParser();a.add_argument('--out',required=True);a.add_argument('--check-only',action='store_true');x=a.parse_args();o=Path(x.out)
 miss=[r for r in REQ if not (o/r).exists()]
 if miss: raise SystemExit(str(miss))
 inv=json.loads((o/'dataset_invariant_report.json').read_text()); assert inv['duplicate_target_pocket_rate']==0.0 and inv['expected_selected_points_to_target_rate']==1.0
 od=json.loads((o/'ood_rule_invariance_audit.json').read_text()); assert od['known_rule_oracle_test_accuracy']==1.0 and od['known_rule_oracle_ood_accuracy']==1.0 and od['ood_label_rule_changed']==False
 src=Path('scripts/probes/run_d37_rule_known_target_binding_factorisation_smoke.py').read_text(); assert 'hit=random.random()<p' not in src
 dec=json.loads((o/'decision.json').read_text()); assert dec['decision'] in ALLOWED
 txt=(Path('docs/research/D37_RULE_KNOWN_TARGET_BINDING_FACTORISATION_SMOKE_RESULT.md').read_text()+(o/'report.md').read_text()).lower(); assert 'no raven solved claim' in txt and 'no architecture superiority claim' in txt
 cmp=json.loads((o/'arm_comparison_report.json').read_text()); assert 'monolithic_vs_gated_test_delta' in cmp
 print(json.dumps({'status':'ok','decision':dec['decision'],'next':dec['next']},indent=2))
if __name__=='__main__':main()

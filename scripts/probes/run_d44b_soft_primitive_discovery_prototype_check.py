#!/usr/bin/env python3
import argparse,json
from pathlib import Path
REQ=['d44a_upstream_manifest.json','dataset_manifest.json','per_family_accuracy_report.json','confusion_matrix_report.json','collision_report.json','collision_by_family_report.json','collision_by_support_count_report.json','hard_vs_soft_by_family_report.json','hard_vs_soft_by_support_count_report.json','hard_vs_soft_by_collision_count_report.json','tie_bias_report.json','fair_identifiability_upper_bound_report.json','order_shuffle_control_report.json','soft_prefilter_elimination_report.json','iterative_elimination_pairwise_report.json','soft_then_hard_hybrid_report.json','margin_strata_report.json','error_taxonomy_report.json','row_level_error_examples.jsonl','aggregate_metrics.json','decision.json','summary.json','report.md','row_outputs_train.jsonl','row_outputs_test.jsonl','row_outputs_ood.jsonl']

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--out',required=True); ap.add_argument('--check-only',action='store_true'); a=ap.parse_args(); out=Path(a.out)
    miss=[x for x in REQ if not (out/x).exists()]
    if miss: raise SystemExit(str(miss))
    d=json.loads((out/'decision.json').read_text())
    print(json.dumps({'status':'ok','decision':d['decision'],'next':d['next']},indent=2))
if __name__=='__main__': main()

#!/usr/bin/env python3
import argparse,json
from pathlib import Path
REQ=['d44b_upstream_manifest.json','d44c_upstream_manifest.json','d44c2_upstream_manifest.json','dataset_manifest.json','fixed_support_count_report.json','support4_audit_report.json','staged_policy_comparison_report.json','oracle_minimal_support_report.json','primitive_space_current5_report.json','primitive_space_all28_report.json','primitive_space_ordered_pair_control_report.json','primitive_space_distractor_sweep_report.json','primitive_space_collision_report.json','primitive_space_recommendation_report.json','candidate_order_sensitivity_report.json','per_family_accuracy_report.json','confusion_matrix_report.json','aggregate_metrics.json','decision.json','summary.json','report.md']

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--out',required=True); ap.add_argument('--check-only',action='store_true'); a=ap.parse_args(); out=Path(a.out)
 miss=[x for x in REQ if not (out/x).exists()]
 if miss: raise SystemExit(str(miss))
 d=json.loads((out/'decision.json').read_text())
 print(json.dumps({'status':'ok','decision':d['decision'],'next':d['next']},indent=2))
if __name__=='__main__': main()

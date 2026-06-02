#!/usr/bin/env python3
import argparse,json
from pathlib import Path
REQ=['d44c_upstream_manifest.json','d44c_metric_semantics_audit.json','dataset_manifest.json','policy_definitions_report.json','one_shot_support_1_report.json','fixed_support_2_report.json','fixed_support_3_report.json','fixed_support_5_report.json','old_adaptive_1_to_5_replay_report.json','staged_support_1_to_2_to_3_to_5_report.json','staged_support_margin_policy_report.json','staged_support_entropy_policy_report.json','staged_support_hybrid_policy_report.json','random_extra_support_control_report.json','bad_ambiguity_signal_control_report.json','oracle_minimal_support_upper_bound_report.json','ambiguity_metric_repair_report.json','support_efficiency_report.json','accuracy_efficiency_frontier_report.json','per_family_accuracy_report.json','per_support_count_accuracy_report.json','collision_ambiguous_case_report.json','aggregate_metrics.json','decision.json','summary.json','report.md']

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--out',required=True); ap.add_argument('--check-only',action='store_true'); a=ap.parse_args(); out=Path(a.out)
 miss=[x for x in REQ if not (out/x).exists()]
 if miss: raise SystemExit(str(miss))
 d=json.loads((out/'decision.json').read_text())
 print(json.dumps({'status':'ok','decision':d['decision'],'next':d['next']},indent=2))
if __name__=='__main__': main()

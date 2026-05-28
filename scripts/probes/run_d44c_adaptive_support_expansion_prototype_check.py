#!/usr/bin/env python3
import argparse,json
from pathlib import Path
REQ=['d44b_upstream_manifest.json','dataset_manifest.json','adaptive_policy_report.json','support_request_report.json','identifiability_report.json','ambiguity_detection_report.json','one_shot_support_1_report.json','fixed_support_2_report.json','fixed_support_3_report.json','fixed_support_5_report.json','adaptive_support_expansion_soft_report.json','adaptive_support_expansion_conservative_report.json','random_extra_support_control_report.json','bad_ambiguity_signal_control_report.json','support_efficiency_report.json','error_by_final_support_count_report.json','collision_report.json','per_family_accuracy_report.json','confusion_matrix_report.json','aggregate_metrics.json','decision.json','summary.json','report.md']

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--out',required=True); ap.add_argument('--check-only',action='store_true'); a=ap.parse_args(); out=Path(a.out)
 miss=[x for x in REQ if not (out/x).exists()]
 if miss: raise SystemExit(str(miss))
 d=json.loads((out/'decision.json').read_text())
 print(json.dumps({'status':'ok','decision':d['decision'],'next':d['next']},indent=2))
if __name__=='__main__': main()

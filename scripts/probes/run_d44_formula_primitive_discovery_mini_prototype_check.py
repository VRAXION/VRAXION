#!/usr/bin/env python3
import argparse,json
from pathlib import Path
REQ=['dataset_invariant_report.json','formula_primitive_oracle_report.json','formula_candidate_bank_report.json','mutable_cell_pair_discovery_report.json','hard_vote_vs_soft_score_report.json','shuffled_center_control_report.json','shuffled_cell_reference_control_report.json','no_center_control_report.json','primitive_mapping_report.json','low_margin_noisy_tail_report.json','decision.json','summary.json','report.md','aggregate_metrics.json']

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--out',required=True); ap.add_argument('--check-only',action='store_true'); a=ap.parse_args(); out=Path(a.out)
    miss=[x for x in REQ if not (out/x).exists()]
    if miss: raise SystemExit(str(miss))
    d=json.loads((out/'decision.json').read_text())
    print(json.dumps({'status':'ok','decision':d['decision'],'next':d['next']},indent=2))

if __name__=='__main__': main()

#!/usr/bin/env python3
import argparse, json
from pathlib import Path

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--out',required=True); a=ap.parse_args()
    out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    plan={
      'stage_1':'OPERATOR_SELECTION_DISCOVERY',
      'stage_2':'CELL_REFERENCE_DISCOVERY',
      'stage_3':'OPERATOR_AND_CELL_DISCOVERY',
      'stage_4':'COMPOSITION_WITH_D43S_STACK',
      'stage_5':'NOISY_LOW_MARGIN_HARDENING',
      'fixed_formula_candidates_are_oracle_like_part':True,
      'must_remove_fixed_part_gradually':True,
      'must_keep_d43s_majority_hard_vote_controls':True,
      'd44_positive_does_not_imply_raven_solved':True,
      'decision':'d44_formula_primitive_discovery_plan_ready',
      'next':'D44A_OPERATOR_SELECTION_DISCOVERY_PROTOTYPE'
    }
    (out/'summary.json').write_text(json.dumps(plan,indent=2))
    (out/'decision.json').write_text(json.dumps({'decision':plan['decision'],'next':plan['next']},indent=2))
    (out/'report.md').write_text('D44 plan artifact only. Boundary: no Raven solved claim; no raw visual Raven reasoning claim; no architecture superiority claim.\n')

if __name__=='__main__': main()

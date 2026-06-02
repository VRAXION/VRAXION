#!/usr/bin/env python3
import argparse, json
from pathlib import Path
REQ=["queue.json","progress.jsonl","upstream_d35_audit_report.json","dataset_manifest.json","dataset_invariant_report.json","machine_utilization_report.json","available_methods_report.json","unavailable_methods_report.json","method_fidelity_report.json","per_arm_report.json","per_method_report.json","per_seed_report.json","per_family_report.json","pocket_confusion_matrix.json","score_margin_report.json","mutation_acceptance_report.json","baseline_comparison_report.json","row_level_error_examples.jsonl","aggregate_metrics.json","decision.json","summary.json","report.md"]
ALLOWED={"d36_dataset_invariant_failure","real_dna_genome_not_validated_pocket_readout_not_confirmed_after_dataset_fix","real_dna_genome_not_validated_formula_to_target_binding_bottleneck","real_dna_genome_not_validated_hidden_rule_family_inference_bottleneck","real_dna_genome_not_validated_real_raven_corridor_layered_signal_confirmed"}

def main():
  ap=argparse.ArgumentParser();ap.add_argument("--out",required=True);ap.add_argument("--check-only",action="store_true");a=ap.parse_args();o=Path(a.out)
  miss=[x for x in REQ if not (o/x).exists()]
  if miss: raise SystemExit(f"missing required artifacts: {miss}")
  inv=json.loads((o/"dataset_invariant_report.json").read_text())
  assert inv["duplicate_target_pocket_rate"]==0.0
  assert inv["expected_selected_points_to_target_rate"]==1.0
  src=Path("scripts/probes/run_d36_real_raven_corridor_baseline_suite.py").read_text()
  assert "base={" not in src and "hit=random.random()<p" not in src
  um=json.loads((o/"unavailable_methods_report.json").read_text()); assert len(um)>=1
  q=json.loads((o/"queue.json").read_text())
  for j in q["jobs"]:
    d=o/f"arm_{j['arm']}"/f"method_{j['method']}"/f"seed_{j['seed']}"
    if (d/"metrics.json").exists():
      assert (d/"row_outputs_test.jsonl").exists() and (d/"row_outputs_ood.jsonl").exists()
  dec=json.loads((o/"decision.json").read_text()); assert dec["decision"] in ALLOWED
  txt=(Path("docs/research/D36_REAL_RAVEN_CORRIDOR_BASELINE_SUITE_RESULT.md").read_text()+ (o/"report.md").read_text()).lower()
  assert "no solved claim" in txt and "no architecture superiority claim" in txt
  print(json.dumps({"status":"ok","decision":dec["decision"],"next":dec["next"]},indent=2))
if __name__=="__main__": main()

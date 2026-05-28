#!/usr/bin/env python3
import argparse, json
from pathlib import Path
REQ=["queue.json","progress.jsonl","machine_utilization_report.json","available_methods_report.json","dataset_manifest.json","per_seed_report.json","per_method_report.json","per_family_report.json","pocket_confusion_matrix.json","score_margin_report.json","mutation_acceptance_report.json","baseline_comparison_report.json","unavailable_baselines_report.json","failure_report.json","aggregate_metrics.json","decision.json","summary.json","report.md"]

def main():
 p=argparse.ArgumentParser();p.add_argument("--out",required=True);p.add_argument("--check-only",action="store_true");a=p.parse_args();o=Path(a.out)
 missing=[x for x in REQ if not (o/x).exists()]
 if missing: raise SystemExit(f"missing required artifacts: {missing}")
 dec=json.loads((o/"decision.json").read_text())
 print(json.dumps({"status":"ok","decision":dec.get("decision"),"next":dec.get("next")},indent=2))

if __name__=="__main__":main()

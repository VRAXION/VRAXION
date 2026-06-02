#!/usr/bin/env python3
import argparse, json
from pathlib import Path
REQ=["queue.json","progress.jsonl","machine_utilization_report.json","d34_fidelity_audit_report.json","dataset_manifest.json","per_seed_report.json","per_family_report.json","pocket_confusion_matrix.json","score_margin_report.json","mutation_acceptance_report.json","best_individual_report.json","row_level_error_examples.jsonl","direct_mutation_probe_report.json","aggregate_metrics.json","decision.json","summary.json","report.md"]

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--out",required=True);ap.add_argument("--check-only",action="store_true");a=ap.parse_args();o=Path(a.out)
    miss=[x for x in REQ if not (o/x).exists()]
    if miss: raise SystemExit(f"missing required artifacts: {miss}")
    d=json.loads((o/"decision.json").read_text());print(json.dumps({"status":"ok","decision":d.get("decision"),"next":d.get("next")},indent=2))

if __name__=="__main__":main()

# STABLE_LOOP_PHASE_LOCK_051 Artifact Checklist

## Source Inputs

```text
docs/research/STABLE_LOOP_PHASE_LOCK_049_ADVERSARIAL_FROZEN_EVAL_CORPUS.jsonl
instnct-core/examples/phase_lane_adversarial_frozen_eval_scale.rs
scripts/probes/run_stable_loop_phase_lock_050_repro_audit.py
docs/research/STABLE_LOOP_PHASE_LOCK_050_EXPECTED_HASHES.json
docs/research/STABLE_LOOP_PHASE_LOCK_050_REPRODUCIBILITY_PACKAGE_AND_PAPER_AUDIT_RESULT.md
scripts/probes/run_stable_loop_phase_lock_051_reviewer_bundle_check.py
```

Expected hashes:

```text
corpus_sha256_normalized_lf = 6b44848ab9483e8267103538ca58198b198a7651e9f20025168143fef4e5cd56
runner_sha256_normalized_lf = 4777b479294bc571751582dee53b05a121eae1465a45e24870432f4828b81046
```

## 050 Outputs

Optional full reproduction writes these outputs under:

```text
target/pilot_wave/stable_loop_phase_lock_051_reviewer_bundle/full_repro
```

Expected 050 output names:

```text
queue.json
progress.jsonl
repro_manifest.json
schema_validation.json
expected_hashes.json
metric_gate_validation.json
paper_tables.md
ablation_table.json
known_failure_controls.json
claim_boundary.md
summary.json
report.md
job_progress/*.jsonl
child_049/summary.json
child_049/metrics.jsonl
child_049/leakage_audit.jsonl
child_049/collapse_metrics.json
child_049/prediction_distribution.json
```

## 051 Reviewer Bundle

Required docs:

```text
docs/research/STABLE_LOOP_PHASE_LOCK_051_PAPER_REPRODUCTION_BUNDLE_CONTRACT.md
docs/research/STABLE_LOOP_PHASE_LOCK_051_PAPER_REPRODUCTION_BUNDLE_RESULT.md
docs/research/STABLE_LOOP_PHASE_LOCK_051_REVIEWER_README.md
docs/research/STABLE_LOOP_PHASE_LOCK_051_ARTIFACT_CHECKLIST.md
docs/research/STABLE_LOOP_PHASE_LOCK_051_CLAIM_BOUNDARY.md
docs/research/STABLE_LOOP_PHASE_LOCK_051_LIMITATIONS.md
docs/research/STABLE_LOOP_PHASE_LOCK_051_ABLATION_NARRATIVE.md
docs/research/STABLE_LOOP_PHASE_LOCK_051_TABLES.md
```

Required commands:

```powershell
python scripts/probes/run_stable_loop_phase_lock_051_reviewer_bundle_check.py --check-only
python scripts/probes/run_stable_loop_phase_lock_050_repro_audit.py --out target/pilot_wave/stable_loop_phase_lock_051_reviewer_bundle/full_repro --seeds 2026,2027,2028 --train-examples 8192 --heldout-examples 4096 --ood-examples 4096 --heartbeat-sec 20
```

## Pass/Fail Criteria

Pass means the static reviewer bundle check reports:

```text
check_pass = true
PAPER_REPRODUCTION_BUNDLE_POSITIVE
REVIEWER_BUNDLE_CHECK_PASSES
PRODUCTION_API_NOT_READY
```

Fail means any required doc, exact command, expected hash, table section,
claim boundary, limitation, or required verdict is missing.

## Claim Boundary

Supports:

```text
reviewer-facing reproduction package for bounded 049/050 adversarial frozen-eval result
```

Does not support:

```text
production default training
public beta promotion
production API readiness
full VRAXION
language grounding
consciousness
biological/FlyWire equivalence
physical quantum behavior
```

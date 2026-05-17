# STABLE_LOOP_PHASE_LOCK_050_REPRODUCIBILITY_PACKAGE_AND_PAPER_AUDIT Contract

## Summary

050 packages the already-positive 049 adversarial frozen eval result as a
reproducible research artifact. It verifies that the 049 run can be freshly
rerun, hash-checked, schema-validated, table-generated, control-audited, and
claim-boundary documented.

This is not a new capability probe. It does not enable production default
training, promote public beta, or claim full VRAXION, language grounding, or
consciousness.

## Source Artifacts

```text
docs/research/STABLE_LOOP_PHASE_LOCK_049_ADVERSARIAL_FROZEN_EVAL_CORPUS.jsonl
instnct-core/examples/phase_lane_adversarial_frozen_eval_scale.rs
scripts/probes/run_stable_loop_phase_lock_050_repro_audit.py
docs/research/STABLE_LOOP_PHASE_LOCK_050_EXPECTED_HASHES.json
```

Hashes use normalized-LF SHA-256.

## Required Outputs

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
child_049/*
```

## Gates

```text
child_run_started = true
child_run_completed = true
child_exit_code = 0
child summary/report newer than 050 start timestamp
corpus hash matches expected
runner hash matches expected
audit script hash recorded
required 049 arms present
main 049 metric thresholds pass
exact / near-duplicate / semantic leakage counts are zero
known failure controls fail as expected
paper tables are generated from child artifacts only
production_default_training_enabled = false
public_beta_promoted = false
production_api_ready = false
```

## Verdicts

Positive:

```text
REPRODUCIBILITY_PACKAGE_POSITIVE
CORPUS_HASH_MATCHES
RUNNER_HASH_MATCHES
AUDIT_SCRIPT_HASH_RECORDED
CHILD_RUN_FRESH
METRIC_SCHEMA_VALID
PAPER_TABLES_WRITTEN
ABLATION_TABLE_WRITTEN
KNOWN_FAILURE_CONTROLS_DOCUMENTED
REPRO_COMMAND_SUCCEEDS
CLAIM_BOUNDARY_DOCUMENTED
PRODUCTION_API_NOT_READY
```

Failure:

```text
REPRODUCIBILITY_PACKAGE_FAILS
CORPUS_HASH_MISMATCH
RUNNER_HASH_MISMATCH
METRIC_SCHEMA_INVALID
REPRO_COMMAND_FAILS
CHILD_RUN_NOT_FRESH
REQUIRED_ARM_MISSING
PAPER_TABLE_SOURCE_MISSING
SOURCE_SNAPSHOT_UNSTABLE
EXPECTED_CONTROL_UNEXPECTEDLY_PASSES
TRAIN_LEAKAGE_DETECTED
PRODUCTION_FLAG_CONTAMINATION
```

## Claim Boundary

050 can support:

```text
reproducibility/audit package for bounded 049 adversarial frozen eval
```

050 cannot support:

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

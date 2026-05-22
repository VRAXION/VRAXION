# STABLE_LOOP_PHASE_LOCK_050_REPRODUCIBILITY_PACKAGE_AND_PAPER_AUDIT Result

Status: positive reproducibility smoke.

050 packages the already-positive 049 adversarial frozen eval result as a
copy-paste reproducibility and paper-audit artifact. It does not test a new
capability and does not enable production default training.

## Run

```powershell
python scripts/probes/run_stable_loop_phase_lock_050_repro_audit.py --out target/pilot_wave/stable_loop_phase_lock_050_reproducibility_package_and_paper_audit/smoke --seeds 2026,2027,2028 --train-examples 8192 --heldout-examples 4096 --ood-examples 4096 --heartbeat-sec 20
```

The 050 runner launched the 049 child cargo example fresh under:

```text
target/pilot_wave/stable_loop_phase_lock_050_reproducibility_package_and_paper_audit/smoke/child_049
```

The child run completed with exit code 0. The 050 progress log wrote heartbeat
rows every 20 seconds during the child run and finished in 238.397 seconds.

## Audit Result

```text
child_run_started = true
child_run_completed = true
child_exit_code = 0
child_summary_newer_than_050_start = true
child_report_newer_than_050_start = true
repo_commit_sha = 0e04e7bd77480d3f523a1a0e0a0aeb68522c27af
git_status_clean_or_dirty = clean
source_snapshot_unstable = false
metric_schema_valid = true
required_arms_present = true
paper_tables_source_artifacts = child summary/metrics/leakage/collapse/prediction-distribution
```

## Hashes

```text
corpus_sha256_normalized_lf = 6b44848ab9483e8267103538ca58198b198a7651e9f20025168143fef4e5cd56
runner_sha256_normalized_lf = 4777b479294bc571751582dee53b05a121eae1465a45e24870432f4828b81046
audit_script_sha256_normalized_lf = 03cc07f9b0eafbac20a08daa050c303dcf522dc134a2630816e124e9eaf55b5c
```

## Reproduced 049 Metrics

```text
ADVERSARIAL_FROZEN_ROUTE_GRAMMAR_TRAIN_AND_INFER:
  heldout_exact_accuracy = 1.000
  ood_exact_accuracy = 1.000
  family_min_accuracy = 1.000
  hard_distractor_accuracy = 1.000
  long_ood_accuracy = 1.000
  unique_output_count = 75 / 75
  collapse_detected = false
```

Leakage gate:

```text
train_eval_id_overlap_count = 0
train_eval_input_overlap_count = 0
train_eval_near_duplicate_count = 0
train_eval_semantic_overlap_count = 0
max_train_eval_token_jaccard = 0.667
near_duplicate_threshold = 0.92
semantic_fingerprint_method = task_family|expected_output|stable_hash(normalized_input)
```

Known failure controls were documented and failed as expected:

```text
NO_ROUTE_GRAMMAR_ADVERSARIAL_FROZEN_BASELINE: collapse_detected = true
FROZEN_EVAL_048_REFERENCE: collapse_detected = true
ROUTE_GRAMMAR_SHUFFLED_LABELS: heldout_exact_accuracy = 0.000
RANDOM_LABEL_CONTROL: family_min_accuracy = 0.000
RANDOM_PHASE_RULE_CONTROL: family_min_accuracy = 0.000, long_ood_accuracy = 0.000
ALWAYS_SPACE_CONTROL: space_only_rate = 1.000
ALWAYS_MAJORITY_CONTROL: majority_output_rate = 1.000
COPY_LAST_TOKEN_CONTROL: copy_last_token_rate = 1.000
```

## Verdicts

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

## Boundary

050 can support only a reproducibility/audit package for the bounded 049
adversarial frozen eval. It cannot support production default training, public
beta promotion, production API readiness, full VRAXION, language grounding,
consciousness, biological/FlyWire equivalence, or physical quantum behavior.

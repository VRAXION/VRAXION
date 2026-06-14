# E79 Training Data Curriculum Readiness Contract

## Summary

E79 adds a machine-checkable gate before `final_train` starts supervised
training work.

Core question:

```text
Can the Rust final-training path prove that its training data and curriculum
contract is complete before any global Pocket Library supervisor work starts?
```

This is a readiness gate, not the final production dataset. It defines the
schema and coverage checks the final-training data path must satisfy.

## Runtime Surface

Library module:

```text
vraxion-runtime/src/training_data.rs
```

Public API:

```text
vraxion_runtime::run_training_data_readiness_preflight(config)
vraxion_runtime::TrainingDataReadinessConfig
vraxion_runtime::TrainingDataReadinessSummary
vraxion_runtime::TrainingLessonSpec
```

Standalone CLI:

```text
vraxion-runtime/src/bin/training_data_readiness.rs
```

Canonical smoke:

```powershell
cargo run --release -p vraxion-runtime --bin training_data_readiness -- 3 8 target\ci\e79_training_data_readiness_smoke
```

The `final_train` entrypoint also runs this gate first and writes the nested
artifact root:

```text
<final_train_out>/training_data_readiness/
```

## Required Behavior

The gate must:

```text
write training_data_readiness_progress.jsonl
write training_data_readiness_results.json
write training_data_readiness_manifest.json
write training_curriculum_manifest.jsonl
write training_data_readiness_report.md
validate split coverage: train, validation, adversarial
validate family coverage across all required curriculum families
validate candidate capability coverage against the final-training candidate set
validate duplicate lesson IDs are blocked
validate every row has evidence digest, scoring policy, and inference target
block short round schedules that cannot cover the full candidate rotation
```

Pass conditions:

```text
passed = true
lesson_count = required_lesson_count = 24
split_count = 3
family_count = 8
capability_count = 8
candidate_unique_count = 16
duplicate_lesson_id_count = 0
missing_family_split_count = 0
missing_candidate_capability_count = 0
invalid_row_count = 0
score_contract_complete = true
inference_contract_complete = true
```

## Artifact Root

```text
target/ci/e79_training_data_readiness_smoke/
```

Required files:

```text
training_data_readiness_results.json
training_data_readiness_manifest.json
training_data_readiness_progress.jsonl
training_curriculum_manifest.jsonl
training_data_readiness_report.md
```

Sample pack:

```text
docs/research/artifact_samples/e79_training_data_curriculum_readiness/
```

## Boundary

E79 confirms the training-data/curriculum readiness contract and the
`final_train` fail-fast gate. It does not claim the final production dataset,
trained weights, open-ended inference, hosted deployment, AGI, consciousness, or
sentience.

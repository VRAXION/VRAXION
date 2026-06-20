# E78 Final Train Campaign Entrypoint Result

## Decision

```text
decision = e78_final_train_campaign_entrypoint_confirmed
```

E78 confirms `final_train` as the canonical Rust command for the consolidated
final-training path.

## Code Change

New library module:

```text
vraxion-runtime/src/final_train.rs
```

Public exports:

```text
vraxion_runtime::run_final_train
vraxion_runtime::FinalTrainConfig
vraxion_runtime::FinalTrainSummary
```

Canonical CLI:

```text
vraxion-runtime/src/bin/final_train.rs
```

The CLI and top-level artifacts escape Windows paths so JSON outputs are valid
on the current platform.

## Evidence Run

Command:

```powershell
cargo run --release -p vraxion-runtime --bin final_train -- 8 20000 target\pilot_wave\e78_final_train_campaign_entrypoint --preflight-rounds 1000 --checkpoint-interval 5000
```

Primary result:

```text
passed = true
lanes = 8
rounds_per_lane = 20000
total_rounds = 160000
global_generated_pocket_count = 16
promoted_to_global = 16
duplicate_candidates_blocked = 112
failed_promotions = 0
redundant_clone_block_rate = 1.000000
bad_commit_rate = 0.000000
unsafe_promotion_rate = 0.000000
seconds = 0.480505800
```

Nested global supervisor:

```text
passed = true
total_local_candidates = 128
unique_candidates = 16
promoted_to_global = 16
duplicate_candidates_blocked = 112
lane_artifact_pass_count = 8
global_registry_entry_count = 17
global_generated_pocket_count = 16
```

JSON validation:

```text
final_train_results.json parsed
final_train_manifest.json parsed
global_supervisor/global_merge_results.json parsed
```

## Interpretation

E78 turns the E73-E77 runtime stack into one clear command:

```text
final_train
-> global Pocket Library supervisor
-> multi-lane final training supervisor
-> E75 lane runners
-> global dedupe/challenger merge
-> final manifest/results/report
```

This is the first consolidated Rust entrypoint that a long final-training
campaign can use without manually choosing lower-level runners.

## Artifact Samples

Committed sample pack:

```text
archived_public_artifact_sample_removed
```

Files:

```text
final_train_results_sample.json
final_train_manifest_sample.json
final_train_progress_sample.jsonl
final_train_report_sample.md
global_merge_results_sample.json
global_library_summary_sample.json
lane_supervisor_results_sample.json
```

## Boundary

E78 does not claim raw language reasoning, open-ended training success, AGI,
consciousness, production deployment, or model-scale behavior. It confirms the
canonical consolidated Rust final-training entrypoint.

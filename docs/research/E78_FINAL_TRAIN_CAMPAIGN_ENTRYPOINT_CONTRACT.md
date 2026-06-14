# E78 Final Train Campaign Entrypoint Contract

## Summary

E78 adds the canonical final-training command over the consolidated Rust
runtime.

Core question:

```text
Can the final Rust runtime expose one clear final_train entrypoint that runs the
validated multi-lane/global-library path and writes a top-level manifest,
progress log, result file, report, and nested supervisor artifacts?
```

This is not a new architecture probe. It is the final consolidated Rust program
entrypoint.

## Runtime Surface

Library module:

```text
vraxion-runtime/src/final_train.rs
```

Public API:

```text
vraxion_runtime::run_final_train(config)
vraxion_runtime::FinalTrainConfig
vraxion_runtime::FinalTrainSummary
```

Canonical CLI:

```text
vraxion-runtime/src/bin/final_train.rs
```

Example:

```powershell
cargo run --release -p vraxion-runtime --bin final_train -- 8 20000 target\pilot_wave\e78_final_train_campaign_entrypoint --preflight-rounds 1000 --checkpoint-interval 5000
```

## Required Behavior

The entrypoint must:

```text
run the E77 global Pocket Library supervisor
preserve nested global supervisor and lane supervisor artifacts
write final_train_progress.jsonl
write final_train_results.json
write final_train_manifest.json
write final_train_report.md
emit valid JSON on stdout
escape Windows paths in JSON artifacts/stdout
```

Pass conditions:

```text
passed = true
global supervisor passed
global_generated_pocket_count > 0
failed_promotions = 0
redundant_clone_block_rate = 1.0
bad_commit_rate = 0
unsafe_promotion_rate = 0
required artifacts exist
final_train_results.json parses as JSON
final_train_manifest.json parses as JSON
```

## Artifact Root

```text
target/pilot_wave/e78_final_train_campaign_entrypoint/
```

Required files:

```text
final_train_results.json
final_train_manifest.json
final_train_progress.jsonl
final_train_report.md
global_supervisor/
```

Nested required files:

```text
global_supervisor/global_merge_results.json
global_supervisor/global_library_summary.json
global_supervisor/lane_supervisor/supervisor_results.json
```

Sample pack:

```text
docs/research/artifact_samples/e78_final_train_campaign_entrypoint/
```

## Boundary

E78 confirms the final Rust entrypoint and artifact surface. It does not prove
open-ended learning, raw text reasoning, AGI, consciousness, production
deployment, or model-scale behavior.

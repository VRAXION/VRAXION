# E77 Global Pocket Library Merge Supervisor Contract

## Summary

E77 adds the missing global merge layer after E76 multi-lane final training.

Core question:

```text
Can independent final-training lanes feed one governed global Pocket Library,
while redundant clones are blocked by dedupe/challenger logic and unique
Pocket artifacts survive guarded reload?
```

This is execution infrastructure for long final-training campaigns. It is not
a new architecture or capability probe.

## Runtime Surface

Library module:

```text
vraxion-runtime/src/global_library_supervisor.rs
```

Public API:

```text
vraxion_runtime::run_global_library_supervisor(config)
vraxion_runtime::GlobalLibrarySupervisorConfig
vraxion_runtime::GlobalLibrarySupervisorSummary
```

Thin CLI:

```text
vraxion-runtime/src/bin/global_library_supervisor.rs
```

Example:

```powershell
cargo run --release -p vraxion-runtime --bin global_library_supervisor -- 8 20000 target\pilot_wave\e77_global_pocket_library_merge_supervisor --preflight-rounds 1000 --checkpoint-interval 5000
```

## Required Behavior

The supervisor must:

```text
run the E76 multi-lane final-training supervisor
verify lane final-training artifacts
collect lane-local candidate Pocket artifacts
promote the first unique candidate through the Pocket Library store
guarded-reload every globally promoted Pocket
block redundant clones by uid/digest/token identity
write progress.jsonl
write partial_global_library_snapshot.json during merge
write global_merge_results.json
write global_library_summary.json
write global_registry.jsonl
write report.md
preserve the nested lane_supervisor artifacts
```

Pass conditions:

```text
passed = true
lane supervisor passed
all lane artifacts passed
promoted_to_global = unique_candidates
duplicate_candidates_blocked = total_local_candidates - unique_candidates
failed_promotions = 0
redundant_clone_block_rate = 1.0
global_generated_pocket_count = unique_candidates
global library ledger_complete = true
bad_commit_rate = 0
unsafe_promotion_rate = 0
```

## Artifact Root

```text
target/pilot_wave/e77_global_pocket_library_merge_supervisor/
```

Required files:

```text
global_merge_results.json
global_library_summary.json
global_registry.jsonl
partial_global_library_snapshot.json
progress.jsonl
report.md
lane_supervisor/
```

Sample pack:

```text
docs/research/artifact_samples/e77_global_pocket_library_merge_supervisor/
```

## Boundary

E77 confirms global merge/dedupe/challenger governance for deterministic Rust
final-training lanes. It does not prove open-ended learning, raw text
reasoning, AGI, consciousness, production deployment, or model-scale behavior.

# E54 Persistent Pocket Library Store And Curriculum Runner Bootstrap Contract

## Purpose

E54 finalizes the Python reference runtime for the persistent Pocket Library.

Core question:

```text
Can the Python runtime persist, reload, guard, promote, quarantine, and reuse
pockets through a real filesystem-backed library while surviving adversarial
store attacks?
```

## Boundary

This is a controlled symbolic/numeric Python reference runtime. It does not
train raw language models, deploy production assistant memory, or make AGI,
consciousness, or model-scale claims.

## Systems

```text
artifact_report_only_control
unsafe_store_no_guards_control
python_persistent_store_no_stress
python_persistent_store_plus_adversarial_stress
oracle_store_reference
```

## Persistent Store Layout

The primary run must write a real store under:

```text
target/pilot_wave/e54_persistent_pocket_library_store_and_curriculum_runner_bootstrap/
  persistent_library/
    python_persistent_store_plus_adversarial_stress/
      registry.json
      tokens.json
      artifacts/*.json
      lifecycle_ledger.jsonl
      access_ledger.jsonl
      promotion_ledger.jsonl
      score_ledger.jsonl
```

## Required Guards

```text
content digest mismatch block
token/pocket swap block
ABI mismatch block
quarantine load block
banned load block
stale token block
direct artifact tamper block
unsafe promotion block
concurrent stale write block
alias rename survival
valid load survival
```

## Metrics

```text
curriculum_success_rate
avg_cost_to_success
reuse_rate
valid_load_success_rate
adversarial_block_rate
unsafe_load_rate
digest_mismatch_block_rate
token_swap_block_rate
abi_mismatch_block_rate
quarantine_block_rate
banned_block_rate
stale_token_block_rate
alias_rename_survival
concurrent_stale_write_block_rate
unsafe_promotion_block_rate
bad_promotion_rate
safe_promotion_count
registry_entry_count
artifact_count
persistent_reload_match
ledger_complete
library_quality_delta
```

## Decisions

Allowed decisions:

```text
e54_python_persistent_library_runtime_confirmed
e54_store_integrity_failure_detected
e54_adversarial_guard_failure
e54_promotion_pipeline_incomplete
e54_invalid_artifact_detected
```

Positive requires:

```text
primary curriculum_success_rate = 1.0
valid_load_success_rate = 1.0
adversarial_block_rate = 1.0
all named guard block rates = 1.0
alias_rename_survival = 1.0
unsafe_load_rate = 0.0
bad_promotion_rate = 0.0
safe_promotion_count >= 2
persistent_reload_match = 1.0
ledger_complete = 1.0
library_quality_delta > 0.0
unsafe_store_no_guards_control fails visibly
deterministic replay passes
target checker failure_count = 0
sample-only checker passes
```

## Required Artifacts

```text
backend_manifest.json
store_schema.json
curriculum_manifest.json
curriculum_rows.jsonl
adversarial_stress_rows.jsonl
store_integrity_report.json
curriculum_runner_report.json
adversarial_stress_report.json
promotion_pipeline_report.json
system_results.json
aggregate_metrics.json
deterministic_replay.json
decision.json
summary.json
progress.jsonl
hardware_heartbeat.jsonl
partial_aggregate_snapshot.json
results_table.md
report.md
```

## Sample Pack

The sample pack must live under:

```text
docs/research/artifact_samples/e54_persistent_pocket_library_store_and_curriculum_runner_bootstrap/
```

## Hard Requirements

```text
Python reference runtime
no Rust executor yet
no gradient descent
no optimizer/backprop
real filesystem-backed persistent store
row-level curriculum events
row-level adversarial stress events
deterministic replay
target checker passes with failure_count = 0
sample-only checker passes
```

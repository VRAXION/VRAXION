# STABLE_LOOP_PHASE_LOCK_082_CHAT_DIVERSITY_MULTI_SEED_CONFIRM Contract

082 is a multi-seed eval-only confirmation gate for the 080
`TOKEN_COMPOSITION_DIVERSITY_REPAIR` checkpoint. It orchestrates fresh 081
child runs across seeds `2027,2028,2029` and verifies that the 081 fresh
chat-diversity pass is stable across seeds, not a single-seed lucky pass.

This is bounded multi-seed chat diversity confirmation only.

not GPT-like assistant readiness
not full English LM
not language grounding
not production chat
not safety alignment
not public beta / GA / hosted SaaS

no service API change
no deployment harness change
no SDK/public export change
no release docs change
no root LICENSE change
no upstream checkpoint mutation

## Files

```text
scripts/probes/run_stable_loop_phase_lock_082_chat_diversity_multi_seed_confirm.py
scripts/probes/run_stable_loop_phase_lock_082_chat_diversity_multi_seed_confirm_check.py
docs/research/STABLE_LOOP_PHASE_LOCK_082_CHAT_DIVERSITY_MULTI_SEED_CONFIRM_CONTRACT.md
docs/research/STABLE_LOOP_PHASE_LOCK_082_CHAT_DIVERSITY_MULTI_SEED_CONFIRM_RESULT.md
```

The committed 081 runner is reused without runtime surface changes:

```text
instnct-core/examples/phase_lane_chat_diversity_fresh_confirm.rs
```

Generated artifacts are written only under:

```text
target/pilot_wave/stable_loop_phase_lock_082_chat_diversity_multi_seed_confirm/
```

## Upstreams

Required upstream roots:

```text
target/pilot_wave/stable_loop_phase_lock_080_chat_composition_diversity_repair/smoke
target/pilot_wave/stable_loop_phase_lock_079b_chat_composition_fresh_failure_analysis/smoke
target/pilot_wave/stable_loop_phase_lock_079_chat_composition_fresh_confirm/smoke
target/pilot_wave/stable_loop_phase_lock_078_chat_composition_repair/smoke
target/pilot_wave/stable_loop_phase_lock_074_multi_seed_scenario_gated_repair_confirm/smoke
```

Required checkpoint:

```text
target/pilot_wave/stable_loop_phase_lock_080_chat_composition_diversity_repair/smoke/checkpoints/chat_composition_diversity_repair/model_checkpoint.json
```

If any required upstream is missing, emit:

```text
UPSTREAM_080_ARTIFACT_MISSING
```

Do not rerun 078, 079, 079B, or 080. Do not train replacement checkpoints.

## Child Runs

Default seeds:

```text
2027,2028,2029
```

For every seed, run a fresh 081 child under:

```text
target/pilot_wave/stable_loop_phase_lock_082_chat_diversity_multi_seed_confirm/smoke/seed_<seed>/
```

For every seed require:

```text
child_run_started = true
child_run_completed = true
child_exit_code = 0
child_summary_newer_than_082_start = true
child_report_newer_than_082_start = true
child_command recorded exactly
```

Fail on stale reuse:

```text
STALE_CHILD_ARTIFACT_USED
```

`progress.jsonl`, `summary.json`, and `report.md` must be written from start
and refreshed after every child seed. There is no black-box run: partial
outcomes are written throughout the orchestrator.

## Gate Rules

All seeds must pass independently. There is no mean-only pass, no best-seed
pass, and no 2/3 seed pass.

Failure:

```text
MULTI_SEED_CHAT_DIVERSITY_INSTABILITY_DETECTED
```

For every child seed, independently parse child artifacts and verify:

```text
CHAT_DIVERSITY_FRESH_CONFIRM_POSITIVE
train_step_count = 0
checkpoint_hash_unchanged = true
prediction_oracle_used = false
llm_judge_used = false
response_table_used_for_main_prediction = false
template_copy_rate <= 0.25
response_table_copy_rate <= 0.20
semantic_template_overlap_rate <= 0.50
slot_only_skeleton_reuse_rate <= 0.25
response_skeleton_reuse_rate <= 0.50
top_skeleton_rate <= 0.35
response_skeleton_diversity >= 0.50
slot_binding_accuracy >= 0.75
finite_label_retention_accuracy >= 0.90
empty_output_rate <= 0.02
space_output_rate <= 0.02
static_response_rate <= 0.15
repetition_rate <= 0.20
copy_prompt_rate <= 0.15
all prompt overlap counts = 0
near_duplicate_prompt_count = 0
```

Failure verdicts include:

```text
CHILD_081_GATE_RECHECK_FAILS
FRESH_PROMPT_LEAKAGE_DETECTED
CHECKPOINT_MUTATION_DETECTED
TEMPLATE_COPY_REGRESSION_DETECTED
SKELETON_REUSE_REGRESSION_DETECTED
VOCAB_DIVERSITY_REGRESSION_DETECTED
CONTEXT_SLOT_BINDING_REGRESSION_DETECTED
FINITE_LABEL_RETENTION_REGRESSION_DETECTED
STATIC_RESPONSE_COLLAPSE_DETECTED
```

Aggregate reports must include min, max, and stddev values, not averages only:

```text
min_novel_response_rate
max_template_copy_rate
min_response_skeleton_diversity
max_response_skeleton_reuse_rate
min_slot_binding_accuracy
min_finite_label_retention_accuracy
stddev_novel_response_rate
stddev_template_copy_rate
stddev_response_entropy
```

## Required Artifacts

Aggregate artifacts:

```text
queue.json
progress.jsonl
multi_seed_config.json
upstream_080_manifest.json
child_run_manifest.json
seed_metrics.jsonl
aggregate_metrics.json
multi_seed_stability.json
novelty_aggregate.json
skeleton_diversity_aggregate.json
vocabulary_entropy_aggregate.json
context_slot_aggregate.json
finite_label_retention_aggregate.json
failure_case_samples.jsonl
summary.json
report.md
```

`failure_case_samples.jsonl` must exist even when empty. If any child row
fails, rows include:

```text
seed
eval_family
prompt
model_output
expected_behavior
pass_fail
novelty_flag
template_copy_flag
skeleton_reuse_flag
semantic_template_overlap_score
slot_binding_diagnosis
short_diagnosis
```

## Verdicts

Positive verdicts:

```text
CHAT_DIVERSITY_MULTI_SEED_CONFIRM_POSITIVE
FRESH_CHILD_RUNS_CONFIRMED
CHILD_081_GATES_RECHECKED
MULTI_SEED_MIN_GATE_PASSES
CHAT_DIVERSITY_STABLE_ACROSS_SEEDS
TEMPLATE_COPY_REJECTED_ALL_SEEDS
SKELETON_REUSE_REJECTED_ALL_SEEDS
VOCAB_ENTROPY_PASSES_ALL_SEEDS
CONTEXT_SLOT_BINDING_PASSES_ALL_SEEDS
FINITE_LABEL_ANCHORROUTE_RETENTION_PASSES_ALL_SEEDS
CHECKPOINT_UNCHANGED_ALL_SEEDS
NO_TRAINING_PERFORMED
FAILURE_CASE_REPORT_WRITTEN
PRODUCTION_CHAT_NOT_CLAIMED
```

Failure verdicts:

```text
CHAT_DIVERSITY_MULTI_SEED_CONFIRM_FAILS
UPSTREAM_080_ARTIFACT_MISSING
STALE_CHILD_ARTIFACT_USED
CHILD_081_GATE_RECHECK_FAILS
MULTI_SEED_CHAT_DIVERSITY_INSTABILITY_DETECTED
FRESH_PROMPT_LEAKAGE_DETECTED
CHECKPOINT_MUTATION_DETECTED
TEMPLATE_COPY_REGRESSION_DETECTED
SKELETON_REUSE_REGRESSION_DETECTED
VOCAB_DIVERSITY_REGRESSION_DETECTED
CONTEXT_SLOT_BINDING_REGRESSION_DETECTED
FINITE_LABEL_RETENTION_REGRESSION_DETECTED
STATIC_RESPONSE_COLLAPSE_DETECTED
FAILURE_CASE_REPORT_MISSING
GPT_LIKE_READINESS_FALSE_CLAIM
PRODUCTION_CHAT_CLAIM_DETECTED
RUNTIME_SURFACE_MUTATION_DETECTED
ROOT_LICENSE_CHANGED
```

If 082 passes, the next milestone is:

```text
083_CHAT_MODEL_ARTIFACT_RC_PACKAGE
```

If 082 fails, the next milestone is:

```text
082B_CHAT_DIVERSITY_MULTI_SEED_FAILURE_ANALYSIS
```

## Validation

```powershell
cargo check -p instnct-core --example phase_lane_chat_diversity_fresh_confirm
python -m py_compile scripts/probes/run_stable_loop_phase_lock_082_chat_diversity_multi_seed_confirm.py
python -m py_compile scripts/probes/run_stable_loop_phase_lock_082_chat_diversity_multi_seed_confirm_check.py
python scripts/probes/run_stable_loop_phase_lock_082_chat_diversity_multi_seed_confirm.py --out target/pilot_wave/stable_loop_phase_lock_082_chat_diversity_multi_seed_confirm/smoke --upstream-080-root target/pilot_wave/stable_loop_phase_lock_080_chat_composition_diversity_repair/smoke --upstream-079b-root target/pilot_wave/stable_loop_phase_lock_079b_chat_composition_fresh_failure_analysis/smoke --upstream-079-root target/pilot_wave/stable_loop_phase_lock_079_chat_composition_fresh_confirm/smoke --upstream-078-root target/pilot_wave/stable_loop_phase_lock_078_chat_composition_repair/smoke --upstream-074-root target/pilot_wave/stable_loop_phase_lock_074_multi_seed_scenario_gated_repair_confirm/smoke --seeds 2027,2028,2029 --heartbeat-sec 20
python scripts/probes/run_stable_loop_phase_lock_082_chat_diversity_multi_seed_confirm_check.py --check-only
python scripts/probes/run_stable_loop_phase_lock_081_chat_diversity_fresh_confirm_check.py --check-only
python scripts/probes/run_stable_loop_phase_lock_080_chat_composition_diversity_repair_check.py --check-only
git diff --check
```

# STABLE_LOOP_PHASE_LOCK_079B_CHAT_COMPOSITION_FRESH_FAILURE_ANALYSIS Contract

079B is analysis-only. It parses existing 079/078/077B artifacts to explain why
079 failed with `TEMPLATE_COPY_DETECTED`.

Hard wall:

```text
no training
no new inference
no 078 rerun
no 079 rerun
no checkpoint mutation
no replacement checkpoint
no product/API/SDK surface changes
no service API change
no deployment harness change
no SDK/public export change
no release docs change
no root LICENSE change
no GPT-like readiness claim
no production chat
```

## Required Inputs

Required 079 artifacts:

```text
summary.json
generation_samples.jsonl
human_readable_samples.jsonl
composition_metrics.json
novelty_metrics.json
context_slot_metrics.json
finite_label_retention_metrics.json
collapse_metrics.json
fresh_chat_eval_dataset.jsonl
```

Required 078 artifacts:

```text
summary.json
generation_samples.jsonl
human_readable_samples.jsonl
train_examples_sample.jsonl
eval_examples_sample.jsonl
repair_dataset_manifest.json
novelty_metrics.json
composition_metrics.json
checkpoint_manifest.json
checkpoints/chat_composition_repair/model_checkpoint.json
```

Required 077B artifacts:

```text
summary.json
repair_recommendation.json
```

Missing upstreams fail with:

```text
UPSTREAM_079_ARTIFACT_MISSING
UPSTREAM_078_ARTIFACT_MISSING
UPSTREAM_077B_ARTIFACT_MISSING
```

## Analysis

Classify every 079 generated row with one primary label:

```text
exact_078_train_response_copy
exact_078_eval_response_copy
exact_078_generated_output_copy
exact_076_response_table_copy
semantic_078_template_overlap
response_skeleton_reuse
low_vocab_recombination
greedy_decoder_reused_high_prior_template
finite_label_retention_label
genuinely_novel_response
unknown_source
```

Hard gate:

```text
unknown_source_rate <= 0.10
```

Required reports:

```text
queue.json
progress.jsonl
analysis_config.json
upstream_079_manifest.json
upstream_078_manifest.json
upstream_077b_manifest.json
template_copy_attribution.json
semantic_overlap_report.json
response_skeleton_report.json
vocabulary_entropy_report.json
decoder_prior_report.json
context_carry_composition_report.json
retention_non_regression_report.json
row_level_attribution.jsonl
human_failure_digest.jsonl
repair_recommendation.json
summary.json
report.md
```

`progress.jsonl`, `summary.json`, and `report.md` are written from start and
refreshed by phase.

## Required Measurements

Template attribution:

```text
exact_078_train_response_copy_rate
exact_078_eval_response_copy_rate
exact_078_generated_output_copy_rate
exact_076_response_table_copy_rate
semantic_078_template_overlap_rate
response_skeleton_reuse_rate
genuinely_novel_response_rate
```

Semantic overlap:

```text
max_token_jaccard_to_078_train_response
max_token_jaccard_to_078_eval_output
max_token_jaccard_to_078_generated_output
max_token_jaccard_to_076_response_table
mean_max_template_overlap
rows_above_0_80_overlap
rows_above_0_90_overlap
```

Skeleton/vocab/decoder reports:

```text
skeleton_template
skeleton_count
skeleton_reuse_rate
top_reused_skeletons
skeleton_by_eval_family
generated_vocab_size
train_vocab_size
eval_vocab_size
generated_to_train_vocab_ratio
unique_response_count
unique_bigram_count
unique_trigram_count
response_entropy
token_entropy
top_response_rate
top_skeleton_rate
high_prior_template_selection_rate
greedy_decode_reuse_rate
repeated_prefix_rate
```

Context and retention:

```text
context_slot_binding_accuracy
slot_inserted_into_template
slot_only_changed_with_same_skeleton_rate
context_composition_novelty_rate
finite_label_retention_accuracy
retention_fail_count
retention_template_copy_relevance
```

## Recommendation

`repair_recommendation.json` must include:

```text
next_milestone = 080_CHAT_COMPOSITION_DIVERSITY_REPAIR
reduce exact response target reuse
replace one-label-one-response training with many-valid-continuation training
use token-level continuation objective over multiple paraphrase targets
add response skeleton dropout
add lexical dropout / synonym slots
add randomized clause order
add fresh heldout paraphrase families
add semantic slot recombination
add entropy regularization or diversity penalty if available
keep context slot binding objective
keep finite-label AnchorRoute retention
keep no product API / no SDK / no service surface
keep no GPT-like readiness claim
adding more response-table entries alone is not enough
exact-response supervised templates are the current failure source
next repair should target composition diversity and template abstraction, not only more data volume
```

## Verdicts

Positive:

```text
CHAT_COMPOSITION_FRESH_FAILURE_ANALYSIS_POSITIVE
UPSTREAM_079_FAILURE_PROFILE_LOADED
TEMPLATE_COPY_ATTRIBUTION_WRITTEN
SEMANTIC_TEMPLATE_OVERLAP_ANALYZED
RESPONSE_SKELETON_REUSE_ANALYZED
VOCAB_ENTROPY_REPORT_WRITTEN
DECODER_PRIOR_REPORT_WRITTEN
CONTEXT_CARRY_COMPOSITION_ANALYZED
RETENTION_NON_REGRESSION_CONFIRMED
REPAIR_RECOMMENDATION_WRITTEN
NO_TRAINING_PERFORMED
PRODUCTION_CHAT_NOT_CLAIMED
```

Failure:

```text
CHAT_COMPOSITION_FRESH_FAILURE_ANALYSIS_FAILS
FAILURE_CASE_INPUT_MISSING
TEMPLATE_COPY_ATTRIBUTION_INCOMPLETE
SEMANTIC_OVERLAP_ANALYSIS_INCOMPLETE
RESPONSE_SKELETON_ANALYSIS_INCOMPLETE
VOCAB_ENTROPY_ANALYSIS_INCOMPLETE
DECODER_PRIOR_ANALYSIS_INCOMPLETE
UNKNOWN_SOURCE_RATE_TOO_HIGH
REPAIR_RECOMMENDATION_MISSING
TRAINING_SIDE_EFFECT_DETECTED
CHECKPOINT_MUTATION_DETECTED
GPT_LIKE_READINESS_FALSE_CLAIM
PRODUCTION_CHAT_CLAIM_DETECTED
RUNTIME_SURFACE_MUTATION_DETECTED
ROOT_LICENSE_CHANGED
```

## Validation

```powershell
python -m py_compile scripts/probes/run_stable_loop_phase_lock_079b_chat_composition_fresh_failure_analysis.py
python scripts/probes/run_stable_loop_phase_lock_079b_chat_composition_fresh_failure_analysis.py --out target/pilot_wave/stable_loop_phase_lock_079b_chat_composition_fresh_failure_analysis/smoke --upstream-079-root target/pilot_wave/stable_loop_phase_lock_079_chat_composition_fresh_confirm/smoke --upstream-078-root target/pilot_wave/stable_loop_phase_lock_078_chat_composition_repair/smoke --upstream-077b-root target/pilot_wave/stable_loop_phase_lock_077b_chat_generation_failure_analysis/smoke --heartbeat-sec 20
python -m py_compile scripts/probes/run_stable_loop_phase_lock_079b_chat_composition_fresh_failure_analysis_check.py
python scripts/probes/run_stable_loop_phase_lock_079b_chat_composition_fresh_failure_analysis_check.py --check-only
python scripts/probes/run_stable_loop_phase_lock_079_chat_composition_fresh_confirm_check.py --check-only
python scripts/probes/run_stable_loop_phase_lock_078_chat_composition_repair_check.py --check-only
git diff --check
```

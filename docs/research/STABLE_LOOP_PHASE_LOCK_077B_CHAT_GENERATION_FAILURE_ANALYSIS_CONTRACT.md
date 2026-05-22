# STABLE_LOOP_PHASE_LOCK_077B_CHAT_GENERATION_FAILURE_ANALYSIS Contract

Status: contract for analysis-only failure attribution of the 077 chat
generation fresh composition failure.

077B parses existing 076/077 artifacts and explains whether the 077 failure
came from response-table copying, train/eval template copying, context slot
binding failure, boundary refusal template selection failure, or mixed causes.

This is analysis-only.

no training
no new inference
no 076 rerun
no 077 rerun
no checkpoint mutation
no replacement checkpoint
no checkpoint repair
no model capability improved
no production chat
no GPT-like assistant readiness
no service API change
no deployment harness change
no SDK/public export change
no release docs change
no root LICENSE change

## Implementation Scope

077B adds only:

```text
scripts/probes/run_stable_loop_phase_lock_077b_chat_generation_failure_analysis.py
scripts/probes/run_stable_loop_phase_lock_077b_chat_generation_failure_analysis_check.py
docs/research/STABLE_LOOP_PHASE_LOCK_077B_CHAT_GENERATION_FAILURE_ANALYSIS_CONTRACT.md
docs/research/STABLE_LOOP_PHASE_LOCK_077B_CHAT_GENERATION_FAILURE_ANALYSIS_RESULT.md
```

Generated outputs are written only under:

```text
target/pilot_wave/stable_loop_phase_lock_077b_chat_generation_failure_analysis/
```

Do not modify `instnct-core/`, service API, deployment harness, SDK/public
exports, release docs, root `LICENSE`, or checkpoint artifacts.

## Required Upstream Artifacts

Required 077 artifacts:

```text
summary.json
generation_samples.jsonl
human_readable_samples.jsonl
composition_metrics.json
novelty_metrics.json
collapse_metrics.json
finite_label_retention_metrics.json
```

Required 076 artifacts:

```text
summary.json
generation_samples.jsonl
chat_sft_dataset_manifest.json
checkpoint_manifest.json
model_checkpoint.json
```

Failure verdicts:

```text
UPSTREAM_077_ARTIFACT_MISSING
UPSTREAM_076_ARTIFACT_MISSING
```

## Source Attribution

Every 077 chat row receives one primary source label:

```text
exact_response_table_copy
exact_train_response_copy
exact_eval_response_copy
semantic_template_copy
finite_label_retention_label
context_slot_not_bound
boundary_refusal_not_selected
wrong_template_family_selected
prompt_copy
unknown_source
```

Positive requires:

```text
unknown_source_rate <= 0.10
template_copy_source_coverage >= 0.90
human_failure_digest rows > 0
```

Context-carry report must include:

```text
context_carry_failure_count
context_slot_expected
context_slot_model_output
selected_template_label_or_response
slot_binding_miss_rate
wrong_template_family_rate
```

Boundary refusal report must include:

```text
boundary_failure_count
expected_refusal_keywords
selected_template_label_or_response
boundary_template_selection_rate
wrong_template_family_rate
```

Response-table dependence report must include:

```text
exact_train_response_copy_rate
exact_eval_response_copy_rate
response_table_copy_rate
template_copy_rate
novel_response_rate
train_response_ngram_overlap
top_copied_templates
copy_rate_by_eval_family
```

## Required Artifacts

```text
queue.json
progress.jsonl
analysis_config.json
upstream_077_manifest.json
upstream_076_manifest.json
template_copy_source_report.json
fresh_context_carry_failure_report.json
boundary_refusal_failure_report.json
response_table_dependence_report.json
prompt_to_template_mapping.jsonl
failure_cluster_report.json
repair_recommendation.json
human_failure_digest.jsonl
summary.json
report.md
```

`human_failure_digest.jsonl` must include:

```text
eval_family
prompt
model_output
expected_behavior
classified_source
copied_template_if_any
required_keywords
missing_keywords
short_diagnosis
```

## Repair Recommendation

`repair_recommendation.json` must include:

```text
next_milestone = 078_CHAT_COMPOSITION_REPAIR
use token-level next-token objective
reduce response_table dependence
add paraphrase / many-target variants
add fresh composition curriculum
add context carry variable-slot training
add boundary refusal paraphrase variants
add template dropout
add semantic slot recombination
retain finite-label AnchorRoute scenario-state eval
keep no product API / no SDK / no service surface
do not claim GPT-like assistant readiness
adding more table responses alone is not enough
```

## Verdicts

Positive verdicts:

```text
CHAT_GENERATION_FAILURE_ANALYSIS_POSITIVE
UPSTREAM_077_FAILURE_PROFILE_LOADED
TEMPLATE_COPY_SOURCE_ATTRIBUTION_WRITTEN
FRESH_CONTEXT_CARRY_FAILURE_ANALYZED
BOUNDARY_REFUSAL_FAILURE_ANALYZED
RESPONSE_TABLE_DEPENDENCE_CONFIRMED
REPAIR_RECOMMENDATION_WRITTEN
NO_TRAINING_PERFORMED
PRODUCTION_CHAT_NOT_CLAIMED
```

Failure verdicts:

```text
CHAT_GENERATION_FAILURE_ANALYSIS_FAILS
FAILURE_CASE_INPUT_MISSING
TEMPLATE_COPY_ANALYSIS_INCOMPLETE
CONTEXT_CARRY_ANALYSIS_INCOMPLETE
BOUNDARY_REFUSAL_ANALYSIS_INCOMPLETE
UNKNOWN_SOURCE_RATE_TOO_HIGH
REPAIR_RECOMMENDATION_MISSING
TRAINING_SIDE_EFFECT_DETECTED
CHECKPOINT_MUTATION_DETECTED
PRODUCTION_CHAT_CLAIM_DETECTED
GPT_LIKE_READINESS_FALSE_CLAIM
RUNTIME_SURFACE_MUTATION_DETECTED
ROOT_LICENSE_CHANGED
```

## Validation Commands

```powershell
python -m py_compile scripts/probes/run_stable_loop_phase_lock_077b_chat_generation_failure_analysis.py
python scripts/probes/run_stable_loop_phase_lock_077b_chat_generation_failure_analysis.py --out target/pilot_wave/stable_loop_phase_lock_077b_chat_generation_failure_analysis/smoke --upstream-077-root target/pilot_wave/stable_loop_phase_lock_077_chat_generation_fresh_composition_confirm/smoke --upstream-076-root target/pilot_wave/stable_loop_phase_lock_076_chat_generation_poc/smoke --heartbeat-sec 20
python -m py_compile scripts/probes/run_stable_loop_phase_lock_077b_chat_generation_failure_analysis_check.py
python scripts/probes/run_stable_loop_phase_lock_077b_chat_generation_failure_analysis_check.py --check-only
python scripts/probes/run_stable_loop_phase_lock_077_chat_generation_fresh_composition_confirm_check.py --check-only
python scripts/probes/run_stable_loop_phase_lock_076_chat_generation_poc_check.py --check-only
git diff --check
```

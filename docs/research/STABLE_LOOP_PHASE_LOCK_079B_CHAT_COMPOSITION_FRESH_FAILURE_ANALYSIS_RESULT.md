# STABLE_LOOP_PHASE_LOCK_079B_CHAT_COMPOSITION_FRESH_FAILURE_ANALYSIS Result

079B implemented the analysis-only failure attribution for the 079 fresh chat
composition failure. It did not train, did not run new inference, did not rerun
078/079, did not mutate checkpoints, and made no product/API/SDK surface changes.

## Verdict

The smoke analysis passed:

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

Key attribution:

```text
unknown_source_rate = 0.0
template_copy_source_coverage = 1.0
exact_078_train_response_copy_rate = 1.0
primary_exact_078_train_response_copy_rate = 0.8666666666666667
semantic_078_template_overlap_rate = 1.0
response_skeleton_reuse_rate = 1.0
genuinely_novel_response_rate = 0.0
```

Semantic overlap:

```text
mean_max_template_overlap = 1.0
rows_above_0_80_overlap = 30
rows_above_0_90_overlap = 30
```

Skeleton and decoder-prior diagnosis:

```text
skeleton_count = 7
skeleton_reuse_rate = 1.0
top_skeleton_rate = 0.3
high_prior_template_selection_rate = 0.8666666666666667
greedy_decode_reuse_rate = 0.8666666666666667
repeated_prefix_rate = 0.8666666666666667
```

Context carry passed, but by slot insertion into reused skeletons:

```text
context_slot_binding_accuracy = 1.0
slot_inserted_into_template = true
slot_only_changed_with_same_skeleton_rate = 1.0
context_composition_novelty_rate = 0.0
```

Retention was not the failure source:

```text
finite_label_retention_accuracy = 1.0
retention_fail_count = 0
retention_template_copy_relevance = false
```

## Diagnosis

079B attributes the 079 failure to exact and semantic reuse of the 078 response
surface. The failure is not prompt leakage, not a checkpoint mutation, not a
slot-binding failure, and not a finite-label retention regression.

The dominant pattern is exact-response target reuse plus response skeleton reuse.
Generated rows often change the slot value correctly, but keep the same response
skeleton. That means the context binding objective survived, while fresh
composition diversity did not.

079B keeps no GPT-like readiness claim and no production chat claim.

## Next

The recommendation is:

```text
next_milestone = 080_CHAT_COMPOSITION_DIVERSITY_REPAIR
```

Required repair direction:

```text
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

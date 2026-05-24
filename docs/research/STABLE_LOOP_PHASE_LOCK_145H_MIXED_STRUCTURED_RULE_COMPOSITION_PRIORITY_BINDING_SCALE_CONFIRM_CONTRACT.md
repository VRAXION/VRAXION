# STABLE_LOOP_PHASE_LOCK_145H_MIXED_STRUCTURED_RULE_COMPOSITION_PRIORITY_BINDING_SCALE_CONFIRM Contract

145H is the scale confirm for the 145A mixed structured-rule composition priority binding prototype.

Boundary: 145H is constrained helper/backend evidence only: mixed structured-rule composition with explicit priority over block types only, not natural-language rule reasoning, not open-ended arbitration, not GPT-like/open-domain capability, not production readiness, and not architecture superiority.

## Expected Route

```text
decision = mixed_structured_rule_composition_priority_binding_scale_confirmed
verdict = INSTNCT_MIXED_STRUCTURED_RULE_COMPOSITION_PRIORITY_BINDING_SCALE_CONFIRMED
next = 145Z_MIXED_STRUCTURED_RULE_COMPOSITION_NEXT_DECISION_PLAN
```

## Scope

145H is scale confirm only. It must not modify `scripts/probes/shared_raw_generation_helper.py`. It uses the existing manifest-gated decoder:

```text
deterministic_pocket_gated_mixed_structured_rule_composition_binding_decoder
```

The old selected-pocket and structured-rule decoders must remain unchanged:

```text
deterministic_pocket_gated_rule_selected_pocket_binding_decoder
deterministic_pocket_gated_structured_rule_metadata_binding_decoder
```

No helper/backend/runtime/request-key/product/release surface may change. No training, checkpoint mutation, request metadata oracle, per-row manifest switching, payload marker narrowing, or post-generation repair is allowed.

## Upstream Requirement

145H requires positive 145A evidence:

```text
decision = mixed_structured_rule_composition_priority_binding_prototype_positive
verdict = INSTNCT_MIXED_STRUCTURED_RULE_COMPOSITION_PRIORITY_BINDING_PROTOTYPE_POSITIVE
next = 145H_MIXED_STRUCTURED_RULE_COMPOSITION_PRIORITY_BINDING_SCALE_CONFIRM
end_to_end_answer_accuracy = 1.0
final_selected_pocket_derivation_accuracy = 1.0
selected_pocket_to_marker_binding_accuracy = 1.0
semantic_invalid_high_priority_fallthrough_accuracy = 1.0
structural_invalid_prompt_fallback_rate = 1.0
priority_pocket_oracle_rejection_rate = 1.0
rule_composition_ablation_accuracy = 0.0
distinct_block_candidate_coverage = true
legacy_structured_rule_metadata_regression_passed = true
legacy_selected_pocket_binding_regression_passed = true
deterministic_replay_passed = true
```

## Scale Design

Default scale:

```text
seeds = 5301,5302,5303,5304
groups_per_family = 24
group_size = 4
families = 25
main_eval_rows = 9600
max_new_tokens = 96
heartbeat_sec = 20
```

The runner must write `queue.json` and append `progress.jsonl` immediately, then continue heartbeat/progress writes during generation. This prevents black-box runs and leaves recoverable partial state.

## Denominator Policy

Positive answer metrics are computed only over positive composition rows where final selected pocket derivation and value extraction are expected:

```text
end_to_end_answer_accuracy
final_selected_pocket_derivation_accuracy
selected_pocket_to_marker_binding_accuracy
same_line_value_extraction_accuracy
positive_composition_subset_accuracy
```

Expected-fallback rows are excluded from those denominators and are scored by fallback/rejection rates:

```text
fallback_control_subset_accuracy
missing_priority_fallback_rate
duplicate_priority_rejection_rate
unknown_priority_rejection_rate
structural_invalid_prompt_fallback_rate
priority_pocket_oracle_rejection_rate
```

The checker must reject if `end_to_end_answer_accuracy` includes expected-fallback rows.

## Coverage Requirements

145H must prove:

```text
recency>quorum>tie_break covered
quorum>recency>tie_break covered
tie_break>quorum>recency covered
each block type wins at least once by priority
each pocket id wins at least once through each supported block type where applicable
SAME_BLOCKS_DIFFERENT_PRIORITY uses distinct candidates
```

The same blocks different priority subset must demonstrate that changing only the priority line can change the final selected pocket.

## Required Reports

145H must write the 145A report set plus:

```text
shared_helper_no_change_audit.json
helper_mixed_rule_semantics_audit.json
positive_vs_fallback_denominator_report.json
priority_order_coverage_report.json
block_type_candidate_coverage_report.json
per_seed_gate_report.json
per_family_gate_report.json
```

## Gates

Aggregate gates require at least `0.98` for parse, per-block derivation, priority policy, final selected pocket derivation, selected-pocket marker binding, same-line extraction, end-to-end answer accuracy, positive composition subset accuracy, fallback control subset accuracy, invalid high-priority fallthrough, semantic invalid high-priority fallthrough, structural invalid prompt fallback, fallback/rejection controls, same-blocks-different-priority, and priority-only changes winner.

Boolean gates require:

```text
all_priority_orders_covered = true
all_block_types_win_under_priority = true
all_pockets_win_under_each_supported_block_type = true
distinct_block_candidate_coverage = true
shared_helper_no_change_since_145a = true
legacy_structured_rule_metadata_regression_passed = true
legacy_selected_pocket_binding_regression_passed = true
deterministic_replay_passed = true
```

## Claim Limit

A positive 145H proves only constrained mixed structured-rule composition at scale with explicit priority over block types. It does not prove natural-language rule reasoning, open-ended arbitration, GPT-like/open-domain capability, production readiness, or architecture superiority.

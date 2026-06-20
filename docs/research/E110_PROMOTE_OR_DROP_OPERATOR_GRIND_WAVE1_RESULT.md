# E110 Promote Or Drop Operator Grind Wave 1 Result

```text
decision = e110_wave1_silver_to_gold_pressure_confirmed
checker_failure_count = 0
sample_only_checker_failure_count = 0
```

Boundary:

```text
Wave 1 Silver-to-Gold pressure only
not Diamond promotion
not Core promotion
not final training
```

## Key Metrics

```text
candidate_count = 35
promoted_to_gold_count = 35
kept_scoped_silver_count = 0
red_flag_count = 0

qualified_activation_added_total = 54405
qualified_activation_after_min = 3192
qualified_activation_after_mean = 3486.400
family_coverage_after_min = 5
campaign_count_after_min = 4

hard_negative_total = 0
wrong_scope_call_rate = 0.000000
false_commit_rate = 0.000000
unsupported_answer_rate = 0.000000
negative_transfer_rate = 0.000000
neutral_waste_total = 0
neutral_waste_over_threshold_count = 0
reload_match_rate = 1.000000
challenger_replacement_count = 0
pruned_variant_replacement_count = 0
deterministic_replay = pass
```

## Promoted To Scoped Gold

```text
text_frame_boundary_lens
evidence_span_locator_lens
source_attribution_lens
negation_contrast_scope_guard
quote_scope_boundary_guard
irrelevant_span_filter_guard
text_evidence_proposal_scribe
ask_when_dependency_missing_scribe
answer_trace_integrity_t_stab
pending_question_trace_lens
clarification_span_locator_lens
clarification_dependency_binder_guard
state_repair_patch_scribe
stale_pending_question_guard
irrelevant_clarification_filter_guard
repaired_answer_reentry_scribe
repair_trace_integrity_t_stab
turn_boundary_cycle_lens
pending_dependency_stack_t_stab
active_turn_state_router_guard
multi_turn_ground_delta_scribe
clarification_chain_join_lens
cross_turn_stale_context_guard
unresolved_state_carry_scribe
summary_relevance_span_selector_lens
required_fact_preservation_guard
summary_drift_detection_guard
task_requirement_decomposition_lens
deliverable_evidence_mapping_scribe
step_status_transition_guard
blocked_dependency_tracker_t_stab
progress_ledger_update_scribe
completion_gate_all_requirements_guard
regression_recheck_step_guard
next_action_selector_scribe
```

## Interpretation

Wave 1 applied promotion pressure to the 35 E109 Silver Operators. All 35
crossed the scoped Gold requirements:

```text
qualified_activation >= 3000
combined_family_coverage >= 5
campaign_count >= 3
hard_negative = 0
reload/challenger/prune pass
```

This is a scoped Gold promotion only. It does not create Diamond, Core,
PermaCore, or TrueGolden memory.

The no-harm counters are scoped deterministic probe evidence from the tracked
E110 artifact rows. They are not production traffic, hosted API telemetry, or
final-training evidence.

## Artifacts

```text
target/pilot_wave/e110_promote_or_drop_operator_grind_wave1/
archived_public_artifact_sample_removed
```

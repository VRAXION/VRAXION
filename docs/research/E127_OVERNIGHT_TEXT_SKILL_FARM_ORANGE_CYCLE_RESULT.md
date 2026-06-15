# E127 Overnight Text Skill Farm Orange Cycle Result

```text
decision = e127_overnight_cycle_positive
cycle_count = 37
orange_legendary_candidate_total = 334
hard_negative_total = 0
false_commit_total = 0
wrong_scope_call_total = 0
unsupported_answer_total = 0
```

## Summary

E127 starts the unattended cyclic loop requested after E125/E126:

```text
candidate discovery
-> scoped Gold farm
-> Orange/Legendary probation
-> repeat with already-promoted operators excluded
```

The supervised overnight loop has completed 37 checkpointed cycles so far. Several candidate
spec packs were added as the currently visible candidate space was exhausted;
each pack was validated by running the next cycle with active already-promoted
operators excluded.

This is scoped operator farming only. It is not Core, PermaCore, TrueGolden,
final training, Gemma-level generation, or open-domain reasoning.

## Aggregate Metrics

```text
selected_candidate_total = 334
orange_legendary_candidate_total = 334
mutation_attempts_total = 1616547
accepted_mutations_total = 10598
rollback_count_total = 1605949

hard_negative_total = 0
false_commit_total = 0
wrong_scope_call_total = 0
unsupported_answer_total = 0
```

## Latest Successful Cycle 028 Metrics

```text
selected_candidate_count = 9
orange_legendary_candidate_count = 9
candidate_pool_count = 58
farmable_candidate_count = 9
qualified_activation_min = 300791
mutation_attempts_total = 40747
accepted_mutations_total = 307
rollback_count_total = 40440
mean_selected_prune_ratio = 0.62

reload_match_rate = 1.0
negative_scope_pass_rate = 1.0
prune_pass_rate = 1.0
challenger_pass_rate = 1.0

hard_negative_total = 0
false_commit_total = 0
wrong_scope_call_total = 0
unsupported_answer_total = 0
```

## Latest Successful Cycle 028 Orange Operators

```text
e128_observation_reported_by_lens
e128_staleness_due_to_date_guard
e128_negated_causal_guard
e128_exact_vs_approximate_guard
e128_method_assumption_result_lens
e128_candidate_promotion_reason_lens
e128_known_unknown_split_lens
e128_question_scope_guard
e128_partial_success_failure_lens
```

## Latest Boundary Cycle 029 Metrics

Cycle 029 found no additional farmable candidates in the current candidate
spec pack. This is a clean exhaustion boundary, not a regression.

```text
decision = e127_cycle_no_candidates
rows_seen = 40000
candidate_pool_count = 49
farmable_candidate_count = 0
selected_candidate_count = 0
orange_legendary_candidate_count = 0

hard_negative_total = 0
false_commit_total = 0
wrong_scope_call_total = 0
unsupported_answer_total = 0
```

## Latest Successful Cycle 032 Metrics

Cycle 032 continued the fresh `e129_` candidate spec pack and still found a
full 16-candidate batch, though the candidate density had started dropping.

```text
selected_candidate_count = 16
orange_legendary_candidate_count = 16
candidate_pool_count = 75
farmable_candidate_count = 16
qualified_activation_min = 300898
mutation_attempts_total = 77864
accepted_mutations_total = 515
rollback_count_total = 77349
mean_selected_prune_ratio = 0.62

reload_match_rate = 1.0
negative_scope_pass_rate = 1.0
prune_pass_rate = 1.0
challenger_pass_rate = 1.0

hard_negative_total = 0
false_commit_total = 0
wrong_scope_call_total = 0
unsupported_answer_total = 0
```

## Latest Successful Cycle 032 Orange Operators

```text
e129_deprecation_warning_guard
e129_self_reported_claim_guard
e129_encoding_normalization_lens
e129_conditional_permission_guard
e129_metric_directionality_lens
e129_path_location_binding_lens
e129_table_row_column_binding_lens
e129_train_test_split_lens
e129_stale_cache_guard
e129_retry_resume_boundary_lens
e129_official_doc_version_lens
e129_personal_data_redaction_guard
e129_missing_baseline_guard
e129_resource_limit_lens
e129_benchmark_leakage_guard
e129_command_error_status_lens
```

## Latest Boundary Cycle 033 Metrics

Cycle 033 found no additional farmable candidates in the current `e129_`
candidate spec pack. This is a clean exhaustion boundary, not a regression.

```text
decision = e127_cycle_no_candidates
rows_seen = 40000
candidate_pool_count = 59
farmable_candidate_count = 0
selected_candidate_count = 0
orange_legendary_candidate_count = 0

hard_negative_total = 0
false_commit_total = 0
wrong_scope_call_total = 0
unsupported_answer_total = 0
```

## Latest Successful Cycle 034 Metrics

Cycle 034 started the fresh `e130_` candidate spec pack and found a full
16-candidate Orange/Legendary batch with clean safety counters.

```text
selected_candidate_count = 16
orange_legendary_candidate_count = 16
candidate_pool_count = 106
farmable_candidate_count = 34
qualified_activation_min = 300663
mutation_attempts_total = 78370
accepted_mutations_total = 502
rollback_count_total = 77868
mean_selected_prune_ratio = 0.62

reload_match_rate = 1.0
negative_scope_pass_rate = 1.0
prune_pass_rate = 1.0
challenger_pass_rate = 1.0

hard_negative_total = 0
false_commit_total = 0
wrong_scope_call_total = 0
unsupported_answer_total = 0
```

## Latest Successful Cycle 034 Orange Operators

```text
e130_risk_severity_lens
e130_high_stakes_domain_guard
e130_latest_instruction_override_lens
e130_ask_answer_policy_lens
e130_dependency_blocker_lens
e130_config_env_var_lens
e130_visual_observation_lens
e130_default_minimal_action_lens
e130_rank_promotion_lens
e130_network_dependency_guard
e130_place_entity_disambiguation_lens
e130_stop_resume_control_lens
e130_capacity_limit_lens
e130_action_owner_lens
e130_deadline_timezone_lens
e130_failure_recycle_lens
```

## Latest Successful Cycle 035 Metrics

Cycle 035 continued the `e130_` candidate spec pack and produced another full
16-candidate Orange/Legendary batch.

```text
selected_candidate_count = 16
orange_legendary_candidate_count = 16
candidate_pool_count = 90
farmable_candidate_count = 18
qualified_activation_min = 300685
mutation_attempts_total = 76165
accepted_mutations_total = 506
rollback_count_total = 75659
mean_selected_prune_ratio = 0.62

reload_match_rate = 1.0
negative_scope_pass_rate = 1.0
prune_pass_rate = 1.0
challenger_pass_rate = 1.0

hard_negative_total = 0
false_commit_total = 0
wrong_scope_call_total = 0
unsupported_answer_total = 0
```

## Latest Successful Cycle 035 Orange Operators

```text
e130_attachment_reference_lens
e130_plateau_trend_lens
e130_transcript_uncertainty_lens
e130_measurement_error_lens
e130_tool_recommendation_execution_lens
e130_search_space_cost_lens
e130_must_should_distinction_lens
e130_local_remote_state_lens
e130_privacy_consent_guard
e130_stack_frame_lens
e130_ablation_control_result_lens
e130_done_evidence_lens
e130_registry_uid_alias_lens
e130_line_reference_lens
e130_active_set_filter_lens
e130_prohibition_exception_guard
```

## Latest Tail Cycle 036 Metrics

Cycle 036 scanned the remaining `e130_` candidate space and found a small
2-candidate tail batch.

```text
selected_candidate_count = 2
orange_legendary_candidate_count = 2
candidate_pool_count = 74
farmable_candidate_count = 2
qualified_activation_min = 301518
mutation_attempts_total = 9943
accepted_mutations_total = 59
rollback_count_total = 9884
mean_selected_prune_ratio = 0.62

reload_match_rate = 1.0
negative_scope_pass_rate = 1.0
prune_pass_rate = 1.0
challenger_pass_rate = 1.0

hard_negative_total = 0
false_commit_total = 0
wrong_scope_call_total = 0
unsupported_answer_total = 0
```

## Latest Tail Cycle 036 Orange Operators

```text
e130_partial_artifact_progress_lens
e130_generated_observed_data_lens
```

Cycle 037 is live-runner state only until its cycle artifacts are checkpointed.

## Latest Boundary Cycle 037 Metrics

Cycle 037 found no additional farmable candidates in the remaining `e130_`
candidate spec pack. This is a clean exhaustion boundary, not a regression.

```text
decision = e127_cycle_no_candidates
rows_seen = 40000
candidate_pool_count = 72
farmable_candidate_count = 0
selected_candidate_count = 0
orange_legendary_candidate_count = 0

hard_negative_total = 0
false_commit_total = 0
wrong_scope_call_total = 0
unsupported_answer_total = 0
```

Cycle 038 should only be started after adding a fresh candidate spec pack or
changing the data/curriculum source.

## Current Boundary

The current state is still scoped Operator farming only:

```text
Orange/LegendaryCandidate = yes
Core = no
PermaCore = no
TrueGolden = no
Gemma-level text generation = no
open-domain reasoning claim = no
```

## Progress Safety

The runner writes:

```text
target/pilot_wave/e127_overnight_text_skill_farm_orange_cycle/progress.jsonl
target/pilot_wave/e127_overnight_text_skill_farm_orange_cycle/partial_aggregate_snapshot.json
target/pilot_wave/e127_overnight_text_skill_farm_orange_cycle/cycles/cycle_###/
```

Each cycle has its own candidate pool, selected operator cards, orange results,
variant report, row-level samples, deterministic replay hash, decision, and
report.

## Stop Mechanism

Create this file to stop the loop at the next cycle boundary:

```text
target/pilot_wave/e127_overnight_text_skill_farm_orange_cycle/STOP
```

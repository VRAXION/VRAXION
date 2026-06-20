# E107 Operator Library E90-E106 Survival Role And Regression Gauntlet Result

```text
decision = e107_operator_library_survival_role_regression_gauntlet_confirmed
checker_failure_count = 0
sample_only_checker_failure_count = 0
```

Boundary:

```text
controlled E90-E106 Operator Library survival gauntlet
not final training
not open-domain capability evaluation
not automatic Core/Golden promotion
```

## Key Metrics

```text
seeds = 16
neighborhoods = 14
case_count = 224
candidate_operator_count = 138
operator_group_count = 17
survival_success_min = 1.000000
survival_success_mean = 1.000000
adversarial_survival_success_min = 1.000000
group_coverage_min = 1.000000
family_coverage_min = 1.000000
focus_coverage_min = 0.800000
unsafe_control_selected_rate = 0.000000
full_library_overreach_rate = 0.000000
cost_blowup_rate = 0.000000
role_assignment_coverage = 1.000000
accepted_mutations_total = 3760
rejected_mutations_total = 28944
rollback_count_total = 28944
deterministic_replay = pass
```

## Role Split

```text
StableSupport = 15
Specialist    = 107
BundleSupport = 14
Deprecated    = 2
```

## StableSupport Operators

```text
multi_span_consistency_guard
contradiction_to_defer_guard
resolved_evidence_coverage_guard
answerability_decision_guard
grounded_answer_template_scribe
evidence_citation_link_scribe
unsupported_answer_defer_guard
active_turn_state_router_guard
cross_turn_stale_context_guard
final_turn_answer_continuity_guard
deliverable_evidence_mapping_scribe
step_status_transition_guard
progress_ledger_update_scribe
completion_gate_all_requirements_guard
regression_recheck_step_guard
```

## BundleSupport Operators

```text
repeat_vote_stabilizer_t_stab
temporal_commit_scribe
local_scope_exit_t_stab
contradiction_report_scribe
output_scope_guard
adapter_requirement_detector_lens
composite_task_decomposer_lens
intermediate_state_carry_t_stab
capability_gap_detector_lens
lesson_candidate_ranker_scribe
adversarial_family_sampler_lens
difficulty_ramp_t_stab
weak_claim_uncertainty_t_stab
query_requirement_mapper_lens
```

## Deprecated Operators

```text
temporal_rule_shift_t_stab
false_alarm_temporal_t_stab
```

## Rejected Controls

```text
final_answer_popularity_selector_control -> Quarantine
full_library_scan_overreach_control      -> Quarantine
ignore_negative_scope_control            -> Quarantine
skip_regression_replay_control           -> Quarantine
stale_trace_route_control                -> Quarantine
unsafe_complete_priority_control         -> Quarantine
cost_blind_selector_control              -> Quarantine
random_operator_neighborhood_control     -> Deprecated
```

## Interpretation

E107 confirms that the E90-E106 Operator bundle survives a multi-seed,
multi-neighborhood QC barrage without selecting unsafe controls, over-scanning
the full library, or blowing the neighborhood cost cap.

The strongest broad-support cluster is not every skill. It is concentrated
around conflict/defer, grounded answerability, multi-turn state continuity, and
E106 progress/completion gates. Most other Operators remain useful as scoped
Specialists, and a smaller set is BundleSupport: useful only as part of a
larger composition but not strong enough as independently recurring support.

The two Deprecated Operators were not selected in this E90-E106 gauntlet:
`temporal_rule_shift_t_stab` and `false_alarm_temporal_t_stab`. This does not
delete them; it means they did not survive this specific multi-neighborhood
selection pressure and should not be promoted without a narrower revalidation.

This is a quality-control result, not final training and not Core/Golden
promotion.

## Artifacts

```text
target/pilot_wave/e107_operator_library_e90_e106_survival_role_and_regression_gauntlet/
archived_public_artifact_sample_removed
```

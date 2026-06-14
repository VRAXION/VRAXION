# E107 Operator Library E90-E106 Survival Role And Regression Gauntlet Contract

## Purpose

E107 is a quality-control gauntlet over the existing E90-E106 Operator Library
bundle. It does not add a new world-knowledge skill. It tests which Operators
survive multi-seed, multi-neighborhood selection pressure, which roles they
occupy, and whether unsafe controls remain rejected.

This is not final training.

## Required Boundary

```text
controlled E90-E106 Operator Library survival gauntlet
not final training
not open-domain capability evaluation
not a model-scale claim
not a license to promote every Operator to Core
```

## Tested Source

```text
docs/research/OPERATOR_LIBRARY_CARDS.md
E90 through E106 StableOperatorCandidate entries
```

## Neighborhoods

```text
visible_evidence_ruleshift
temporal_bitstream_resync
lexical_glyph_grounding
agency_commit_safety
answer_output_hygiene
active_evidence_search
trace_ground_memory_hygiene
route_composition_execution
curriculum_regression_scheduler
text_ingress_conflict_resolution
clarification_state_repair
context_compression_reentry
false_done_progress_traps
full_long_horizon_bundle
```

Each seed/neighborhood selects from the full E90-E106 library, not from a
pre-filtered local subset.

## Role Labels

```text
StableSupport
Specialist
BundleSupport
Deprecated
Quarantine
```

## Required Artifacts

```text
run_manifest.json
operator_library_manifest.json
task_generation_report.json
source_inventory_report.json
neighborhood_results.json
survival_role_report.json
progress.jsonl
partial_aggregate_snapshot.json
seed_results.json
aggregate_metrics.json
selection_frequency_report.json
counterfactual_report.json
operator_lifecycle_report.json
mutation_summary.json
deterministic_replay.json
decision.json
summary.json
report.md
row_level_samples.jsonl
operator_evolution_history.jsonl
```

Sample pack:

```text
docs/research/artifact_samples/e107_operator_library_e90_e106_survival_role_and_regression_gauntlet/
```

## Metrics

```text
candidate_operator_count
operator_group_count
survival_success
adversarial_survival_success
group_coverage
family_coverage
focus_coverage
unsafe_control_selected_rate
full_library_overreach_rate
cost_blowup_rate
role_assignment_coverage
StableSupport / Specialist / BundleSupport / Deprecated counts
counterfactual survival score loss
accepted/rejected/rollback counts
deterministic replay hash match
checker failure count
```

## Pass Requirements

```text
seed_count >= 16
candidate_operator_count >= 130
operator_group_count = 17
survival_success_min = 1.000000
adversarial_survival_success_min = 1.000000
group_coverage_min = 1.000000
family_coverage_min = 1.000000
focus_coverage_min >= 0.500000
unsafe_control_selected_rate = 0.000000
full_library_overreach_rate = 0.000000
cost_blowup_rate = 0.000000
role_assignment_coverage = 1.000000
checker_failure_count = 0
sample_only_checker_failure_count = 0
deterministic replay passes
```

## Decisions

```text
e107_operator_library_survival_role_regression_gauntlet_confirmed
e107_operator_library_survival_gauntlet_incomplete
e107_operator_inventory_missing
e107_unsafe_control_survived
e107_full_library_overreach_detected
e107_role_assignment_failure
e107_artifact_or_replay_failure
```

## Interpretation Rule

A positive result means the E90-E106 Operator bundle passed a controlled
survival/ranking gauntlet and received scoped lifecycle roles.

It does not mean every surviving Operator is Core or Golden.

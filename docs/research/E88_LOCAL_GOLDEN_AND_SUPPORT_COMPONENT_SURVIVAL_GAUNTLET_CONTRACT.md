# E88 LocalGolden And Support Component Survival Gauntlet Contract

## Purpose

E88 is a survival/ranking gauntlet for the current CALC-SCRIBE LocalGolden and
the E87 selected support components.

It must answer:

```text
Do the current components survive unseen data, mutation stress, reload/import
stress, negative-scope stress, counterfactual ablation, challenger sweep, and
long-horizon no-harm replay?
```

## Boundary

```text
Visible calculation-trace validation only.
No GSM8K solving.
No open-domain reasoning.
No Core / TrueGolden promotion.
No production readiness claim.
```

## Candidate Set

```text
calc_scribe_v003
calc_scribe_native_seed
square_trace_adapter
arrow_trace_adapter
standalone_plain_trace_adapter
unicode_operator_normalizer
invalid_trace_rejector
long_text_scope_guard
```

Controls:

```text
native_seed_clone
square_adapter_clone
arrow_adapter_clone
numeric_alias_overreach
full_library_scan_overreach
invalid_direct_commit
long_text_plain_overreach
noop_trace_observer
expensive_debug_probe
```

## Required Stress Families

```text
unseen_calc_marker_split
format_transfer
marker_noise_and_decoys
negative_scope
mutation_stress
reload_import_stress
counterfactual_value
challenger_sweep
long_horizon_no_harm
```

## Status Labels

```text
SpecialistGoldenCandidate
LocalGoldenConfirmed
StableSupport
ActiveSupport
BundleSupport
Redundant
Quarantine
Deprecated
Banned
```

## Required Metrics

```text
validation_route_min
validation_action_min
adversarial_action_min
false_call_max
false_commit_max
negative_scope_no_call_rate
reload_match_rate
tamper_block_rate
token_swap_block_rate
unsafe_global_scope_block_rate
counterfactual_value_score
challenger_win_rate
redundant_selection_rate
unsafe_selection_rate
long_horizon_no_harm_rate
active_set_size_mean
cost_adjusted_utility
top_k_jaccard_across_seeds
```

## Required Artifacts

```text
component_survival_table.json
counterfactual_ablation.json
challenger_sweep.json
negative_scope_report.json
reload_import_stress_report.json
long_horizon_no_harm_report.json
aggregate_metrics.json
deterministic_replay.json
decision.json
summary.json
report.md
progress.jsonl
seed_progress/
row_level_samples.jsonl
```

Artifact sample pack:

```text
docs/research/artifact_samples/e88_local_golden_and_support_component_survival_gauntlet/
```

## Pass Rules

Positive E88 requires:

```text
validation_action_min = 1.0
adversarial_action_min = 1.0
false_call_max = 0.0
false_commit_max = 0.0
negative_scope_no_call_rate = 1.0
reload_match_rate = 1.0
tamper_block_rate = 1.0
token_swap_block_rate = 1.0
unsafe_global_scope_block_rate = 1.0
long_horizon_no_harm_rate = 1.0
challenger_beats_total = 0
checker failure_count = 0
sample-only checker pass
```

## Allowed Positive Decision

```text
e88_local_golden_survival_gauntlet_confirmed
```

Other outcomes:

```text
e88_calc_scribe_specialist_golden_candidate_confirmed
e88_support_components_stable_but_not_golden
e88_negative_scope_regression_detected
e88_challenger_replaces_current_candidate
e88_adversarial_failure_detected
```

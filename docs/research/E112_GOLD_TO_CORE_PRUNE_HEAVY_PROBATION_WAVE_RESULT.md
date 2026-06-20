# E112 Gold To Core Prune Heavy Probation Wave Result

```text
decision = e112_gold_to_core_prune_heavy_probation_confirmed
checker_failure_count = 0
sample_only_checker_failure_count = 0
```

Boundary:

```text
CoreMemoryCandidate probation only
not PermaCore
not TrueGolden
not final training
```

## Key Metrics

```text
candidate_count = 136
core_memory_candidate_count = 136
gold_stay_count = 0
red_flag_count = 0
deprecated_count = 0

qualified_activation_added_total = 13392065
qualified_activation_after_min = 101601
qualified_activation_after_mean = 102041.904
family_coverage_after_min = 20
campaign_count_after_min = 8

prune_heavy_selected_count = 136
prune_heavy_selected_ratio = 1.000000
mean_selected_prune_ratio = 0.694118
minimal_prune_selected_count = 136
deep_prune_selected_count = 0

mutation_attempts_total = 295036
accepted_mutations_total = 3080
rejected_mutations_total = 291956
rollback_count_total = 291956
prune_attempts_total = 2974
challenger_attempts_total = 1556

hard_negative_total = 0
wrong_scope_call_rate = 0.000000
false_commit_rate = 0.000000
unsupported_answer_rate = 0.000000
negative_transfer_rate = 0.000000
reload_match_rate = 1.000000
long_horizon_no_harm_rate = 1.000000
negative_scope_pass_rate = 1.000000
deterministic_replay = pass
```

Runtime measurement:

```text
measured_wall_seconds = 0.016
duration_per_candidate_ms = 0.118
```

The runtime is a deterministic probe runtime envelope, not a hardware-bound
estimate for real final training.

## Interpretation

All 136 scoped Gold Operators passed the CoreMemoryCandidate probation gate.
The selected form for every operator was prune-heavy:

```text
selected_variant_type = minimal_core_prune
mean_selected_prune_ratio = 69.4118%
```

This satisfies the requested prune/simplification-heavy Core probation wave.

## Post-Wave Rank Summary

After merging E110, E111, and E112:

```text
CoreMemoryCandidate = 136
Gold = 0
Silver = 0
Bronze = 0
DiamondCandidate = 0
RedFlag = 0
Deprecated = 3
```

## Boundary

This is not PermaCore/TrueGolden. A later PermaCore grind should use a larger
activation target and longer no-harm/adversarial replay window.

## Artifacts

```text
target/pilot_wave/e112_gold_to_core_prune_heavy_probation_wave/
archived_public_artifact_sample_removed
```

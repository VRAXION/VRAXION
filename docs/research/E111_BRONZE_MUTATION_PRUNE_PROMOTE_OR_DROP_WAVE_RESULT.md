# E111 Bronze Mutation Prune Promote Or Drop Wave Result

```text
decision = e111_bronze_mutation_prune_wave_gold_conversion_confirmed
checker_failure_count = 0
sample_only_checker_failure_count = 0
```

Boundary:

```text
Bronze-to-Gold-or-drop mutation/prune wave only
not Diamond promotion
not Core promotion
not final training
```

## Key Metrics

```text
candidate_count = 87
promoted_to_gold_count = 87
dropped_deprecated_count = 0
red_flag_count = 0

mutated_candidate_count = 87
pruned_selected_count = 87
challenger_selected_count = 0

qualified_activation_added_total = 302806
qualified_activation_after_min = 3251
qualified_activation_after_mean = 3480.529
family_coverage_after_min = 7
campaign_count_after_min = 4

mutation_attempts_total = 52103
accepted_mutations_total = 703
rejected_mutations_total = 51400
rollback_count_total = 51400
prune_attempts_total = 341
challenger_attempts_total = 261

hard_negative_total = 0
wrong_scope_call_rate = 0.000000
false_commit_rate = 0.000000
unsupported_answer_rate = 0.000000
negative_transfer_rate = 0.000000
neutral_waste_total = 0
reload_match_rate = 1.000000
deterministic_replay = pass
```

Runtime measurement:

```text
measured_wall_seconds = 0.012
duration_per_candidate_ms = 0.138
estimated_seconds_per_1000_candidates = 0.138
```

The runtime is a deterministic probe runtime envelope, not a hardware-bound
estimate for real final training.

## What Changed

Unlike E110, this wave did not merely apply rank pressure.

Every Bronze candidate was evaluated with:

```text
base_unmodified
scope_adapter_mutation
io_contract_prune
mutation_plus_prune
sibling_challenger
```

Every promoted candidate selected:

```text
selected_variant_type = mutation_plus_prune
```

So the new scoped Gold status belongs to the validated mutated/pruned variant,
not to the untouched Bronze form.

## Post-Wave Rank Summary

After merging E110 and E111:

```text
Gold = 136
Silver = 0
Bronze = 0
DiamondCandidate = 0
RedFlag = 0
Deprecated = 3
```

## Interpretation

E111 confirms that the remaining Bronze pool can be resolved under active
variant search. The useful path was mutation plus pruning, not passive
promotion and not sibling challenger replacement.

This is still scoped Gold only. No Diamond/Core/PermaCore claim is made.

## Artifacts

```text
target/pilot_wave/e111_bronze_mutation_prune_promote_or_drop_wave/
docs/research/artifact_samples/e111_bronze_mutation_prune_promote_or_drop_wave/
```

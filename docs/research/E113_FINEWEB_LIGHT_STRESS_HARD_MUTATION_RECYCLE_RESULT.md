# E113 FineWeb Light Stress Hard Mutation Recycle Result

```text
decision = e113_fineweb_light_stress_hard_mutation_recycle_positive
checker_failure_count = 0
```

Boundary:

```text
FineWeb light stress only
not PermaCore
not TrueGolden
not final training
```

## Inputs

```text
dataset = data/high_quality_seed_v1/fineweb_edu/local_fineweb_edu_sample_100000.jsonl
rows_seen = 100000
operator_pool = E112 CoreMemoryCandidate
operator_count = 136
```

## Key Metrics

```text
baseline_hard_negative_total = 2624
baseline_hard_operator_count = 88

selected_hard_negative_total = 0
selected_neutral_waste_total = 0
selected_positive_total = 3461003
selected_call_total = 3461003

recycled_operator_count = 136
selected_variant_counts:
  hard_scope_prune_copy = 53
  negative_scope_sentinel_copy = 83

mutation_attempts_total = 46142
accepted_mutations_total = 822
rollback_count_total = 45320

checker = pass
deterministic_replay = pass
```

## Dataset Shape

```text
all_rows = 100000
generic_negative_scope = 36786
question_like = 36386
calc_like = 22417
evidence_like = 100000
adversarial_like = 33
```

## Interpretation

The unmodified CoreMemoryCandidate baseline was not safe enough for raw
FineWeb-style web text:

```text
2624 hard negatives across 88 operators
```

The selected hard mutation/recycle copies removed those hard negatives:

```text
selected hard negatives = 0
selected neutral waste = 0
```

This means the new FineWeb stress did expose real scope pressure. The useful
response was not to drop the operators, but to recycle them through stricter
scope-prune and negative-scope sentinel copies.

Top clash pattern:

```text
adversarial-like rows triggered broad Guard commits in the baseline.
hard_scope_prune_copy converted those into safe scoped behavior.
```

## Boundary

This is a light stress/recycle pass. It validates the repair direction for the
100k FineWeb seed pack, but it is not enough for PermaCore or TrueGolden. A
larger 1M/full-shard run should keep the same continuous progress/writeout
contract.

## Artifacts

```text
target/pilot_wave/e113_fineweb_light_stress_hard_mutation_recycle/
```

# E136N3 Parallel Direct-Write A/B Smoke Result

```text
decision = e136n3_parallel_direct_write_ab_confirmed
next     = E136O_CHALLENGER_OOD_RUNTIME_REPLACEMENT_GAUNTLET
```

E136N3 compares two parallel execution styles over the same E136N/E136N2
proposal surface:

```text
arm A = parallel direct write into Flow
arm B = parallel proposal fanout + Agency Matrix commit barrier
```

## Result

```text
case_count = 123

direct_write_accuracy = 0.089431
agency_gated_accuracy = 1.000000

direct_write_unsafe_commit_count = 34
agency_gated_unsafe_commit_count = 0

direct_write_conflict_case_count = 102
direct_write_nondeterministic_case_count = 102
direct_write_missing_chunk_metadata_count = 10

direct_write_runtime_write_count = 602
agency_gated_runtime_direct_write_count = 0

direct_write_held_variant_promoted_count = 36
agency_gated_held_variant_promoted_count = 0

direct_write_safe_control_correct_count = 11

expected_child_check_count = 36
agency_gated_child_check_count = 36
expected_flow_chunk_count = 10
agency_gated_flow_chunk_count = 10
```

## Interpretation

The A/B smoke confirms a narrower rule than "never run things in parallel":

```text
parallel read/propose = allowed
parallel direct Flow write = rejected as default
Agency-gated chunk/multi commit = allowed
```

The direct-write arm was not a pure strawman: it passed 11 disjoint safe
controls. The failure is that direct write stops being reliable as soon as
same-region races, unsafe direct writers, held challenger/lineage variants,
rollback races, or chunk metadata are involved.

The Agency-gated arm kept the parallel fanout, but added a commit barrier. It
preserved 100% correctness, rejected unsafe writes, preserved held variants,
performed the expected child checks, and committed the expected Flow chunks
without runtime direct writes.

## Boundary

This is a deterministic A/B artifact over existing E136N variants and the E136N2
Agency Matrix. It does not run a production scheduler, discover new operators,
promote held challengers, or apply runtime replacement.

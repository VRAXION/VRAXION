# E117 Alpha-Weave Targeted Pressure Gauntlet Result

## Verdict

```text
decision = e117_targeted_pressure_gauntlet_next_limit_reached
checker_failure_count = 0
```

E117 ran the E116-generated alpha-Weave pressure cells as an actual targeted
gauntlet over every generated cell and adversarial variant. It validated the
public/hidden split, schema, synthetic origin metadata, route-budget proxy,
answer/evidence consistency, negative-scope behavior, and repeat-count
activation accounting.

## Key Metrics

```text
target_operator_count = 77
target_reach_count = 77
targeted_needed_remaining_count = 0

generated_cell_packs = 9856
variant_unit_count = 118272
scheduled_case_count = 15257088
qualified_activation_total = 15257088

hard_negative_total = 0
false_commit_total = 0
wrong_scope_call_total = 0
unsupported_answer_total = 0
over_budget_total = 0
public_leak_total = 0
schema_failure_total = 0
metadata_failure_total = 0
```

## Activation Mix

```text
ANSWER = 6357120
ASK_FOR_EVIDENCE = 1271424
DEFER = 5085696
NO_CALL = 2542848
```

This means the gauntlet was not answer-only pressure. It included answerable
cases, unresolved/missing-evidence cases, defer cases, and negative-scope
no-call cases.

## Template Coverage

```text
evidence_conflict = 4864 packs
task_progress = 1664 packs
answer_integrity = 1408 packs
alias_symbol = 1152 packs
frame_sync = 768 packs
```

## Boundary

This is a targeted activation/no-harm gauntlet. It is not final training, not
PermaCore, not TrueGolden, and not automatic Core promotion.

## Interpretation

E116 projected that targeted alpha-Weave pressure would close the sparse
activation gap for the 77 Operators that FineWeb did not naturally trigger
enough. E117 validates that schedule as clean actual gauntlet evidence:

```text
77 / 77 target Operators reach the next 300k activation limit
0 hard negatives
0 public leakage
0 unsafe promotion claim
```

The next decision is whether to run a higher-threshold Core probation grind or
to broaden the targeted synthetic families before any stronger promotion claim.

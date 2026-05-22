# ANCHOR-MINI-006 Result

## Verdict

```text
ANCHOR_MINI_006_FORMAT_STRONG_SIGNAL
```

ANCHOR-MINI-006 ran a symbolic process-format sweep on the MINI-005
shortcut-flip stress task.

Narrow result:

```text
Explicit structured process maps beat answer-only, prose-like, inner-monologue,
flat key-value, and shuffled controls. The best practical symbolic format was
ACTION_OUTCOME_TABLE_PROCESS.
```

This does not prove literal text serialization performance in an LLM. It says
which symbolic process decomposition is most useful in this audit sparse
carrier.

## Run

```bash
cargo build --release -p instnct-core --example evolve_anchor_mini006

python tools/anchorweave/run_anchor_mini006_format_sweep.py ^
  --out target/anchorweave/anchor_mini006_format_sweep/full_symbolic_2026_05_10_oraclefix ^
  --seeds 2026-2125 ^
  --jobs 16 ^
  --budget-hours 8 ^
  --skip-build
```

Runtime:

```text
270.00 seconds
```

## Summary

```text
completed_jobs: 1200 / 1200
valid_jobs: 1200
valid_seed_count: 100
blocked_jobs: 0
budget_reached: false
best_practical_format: ACTION_OUTCOME_TABLE_PROCESS
```

| format arm | OOD accuracy | trap rate | true process bit | token mean | efficiency |
|---|---:|---:|---:|---:|---:|
| `ANSWER_ONLY` | 0.176 | 0.626 | 0.000 | 0.0 | 0.1759 |
| `ORACLE_MATCH_BITS` | 1.000 | 0.000 | 1.000 | 5.0 | 0.2000 |
| `RAW_SYMBOLIC_PROCESS` | 1.000 | 0.000 | 1.000 | 12.0 | 0.0833 |
| `PROSE_PROCESS` | 0.176 | 0.626 | 0.651 | 34.0 | 0.0052 |
| `INNER_MONOLOGUE_PROCESS` | 0.223 | 0.672 | 0.687 | 56.0 | 0.0040 |
| `STRICT_JSON_PROCESS` | 1.000 | 0.000 | 1.000 | 32.0 | 0.0312 |
| `FLAT_KEY_VALUE_PROCESS` | 0.251 | 0.249 | 0.715 | 14.0 | 0.0179 |
| `RELATIONAL_TRIPLES_PROCESS` | 1.000 | 0.000 | 1.000 | 45.0 | 0.0222 |
| `ACTION_OUTCOME_TABLE_PROCESS` | 1.000 | 0.000 | 1.000 | 20.0 | 0.0500 |
| `RELATION_PLUS_ACTION_PROCESS` | 1.000 | 0.000 | 1.000 | 65.0 | 0.0154 |
| `COMPACT_HYBRID_PROCESS` | 1.000 | 0.000 | 1.000 | 40.0 | 0.0250 |
| `SHUFFLED_COMPACT_HYBRID` | 0.075 | 0.307 | 0.510 | 40.0 | 0.0019 |

## Conditions

All primary gates passed:

```text
valid_seeds_at_least_80: true
oracle_upper_bound_good: true
has_useful_practical_format: true
best_beats_answer_by_0p25: true
best_beats_shuffled_by_0p25: true
prose_inner_not_unfairly_best: true
no_blocked_jobs: true
```

Useful arms:

```text
RAW_SYMBOLIC_PROCESS
STRICT_JSON_PROCESS
RELATIONAL_TRIPLES_PROCESS
ACTION_OUTCOME_TABLE_PROCESS
RELATION_PLUS_ACTION_PROCESS
COMPACT_HYBRID_PROCESS
```

Failed or weak arms:

```text
PROSE_PROCESS
INNER_MONOLOGUE_PROCESS
FLAT_KEY_VALUE_PROCESS
SHUFFLED_COMPACT_HYBRID
```

## Interpretation

The result favors explicit action/outcome structure over free prose,
inner-monologue style traces, or flat unbound key-value fields.

The important pattern is:

```text
relation/action-bound process structure -> solves shortcut flip
loose prose / monologue / flat fields -> does not solve shortcut flip
wrong structured process -> does not reproduce the win
```

This supports the working design direction:

```text
AnchorCell training views should emphasize compact action/outcome and relation
maps, not raw human monologue as the primary model-facing carrier.
```

## Claim Boundary

This is a symbolic audit result. It does not prove that a literal Markdown
table, JSON object, or triple list is best for LLM fine-tuning.

Next required step: MINI-006B should test actual serialized text formats on a
small deterministic text carrier or small causal-LM carrier using the same
masking and shortcut-flip gates.

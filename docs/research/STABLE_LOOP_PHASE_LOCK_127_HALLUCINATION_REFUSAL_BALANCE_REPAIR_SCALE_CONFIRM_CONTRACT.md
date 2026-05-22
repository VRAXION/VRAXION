# STABLE_LOOP_PHASE_LOCK_127_HALLUCINATION_REFUSAL_BALANCE_REPAIR_SCALE_CONFIRM_CONTRACT

## Purpose

127 is an eval-only scale confirmation for the 126 hallucination/refusal balance
repair. It evaluates the 126 repaired raw checkpoint read-only on fresh,
larger, multi-seed calibration rows.

Positive 127 requires `HALLUCINATION_REFUSAL_BALANCE_REPAIR_SCALE_CONFIRM_POSITIVE`.
It must show `HALLUCINATION_REFUSAL_REPAIR_GENERALIZES`,
`ANSWERABLE_FACT_RESPONSE_CONFIRMED`, `INSUFFICIENT_FACT_REFUSAL_CONFIRMED`,
`ALWAYS_REFUSE_DEGENERATION_REJECTED`, and
`UNDER_REFUSAL_REGRESSION_REJECTED`.

## Required Run

The full configured run is mandatory:

```text
seeds = 2191,2192,2193,2194,2195
eval_rows_per_family = 96
evidence_variants = 16
ambiguity_variants = 12
insufficient_fact_variants = 12
table_rows = 64
multi_doc_count = 8
long_context_chars = 24576
noise_blocks = 24
format_variants = 12
```

The positive arm is:

```text
POST_126_HALLUCINATION_REFUSAL_BALANCE_REPAIRED_RAW_SCALE_CONFIRM
```

Controls must fail:

```text
ALWAYS_REFUSE_CONTROL
STATIC_OUTPUT_CONTROL
COPY_PROMPT_CONTROL
RANDOM_FACT_CONTROL
RANDOM_REFUSAL_CONTROL
RANDOM_ANSWER_CONTROL
```

## Boundary

127 is eval-only scale confirmation. It performs no training, no repair, no
checkpoint mutation, no service startup, no deployment smoke, and no runtime or
release integration.

It is not GPT-like assistant readiness, not open-domain assistant readiness,
not production chat, not public API, not deployment readiness, not safety
alignment, and not Hungarian assistant readiness.

The expected next milestone after a positive verdict is:

```text
128_POST_CALIBRATION_REPAIR_CEILING_AND_GAP_REMAP
```


# STABLE_LOOP_PHASE_LOCK_126_HALLUCINATION_REFUSAL_BALANCE_REPAIR_CONTRACT

## Purpose

126 is a targeted research repair for the 124/125-selected hallucination/refusal
balance breakpoint. It is targeted research repair, not generic SFT and not
refusal-only training.

The positive-scored arm is `POST_126_HALLUCINATION_REFUSAL_BALANCE_REPAIRED_RAW`.
It must improve calibration while preserving prior reasoning and multi-turn
state repairs.

## Required Behavior

The repair must learn the distinction between:

- provided-fact answerable rows
- insufficient-fact refusal rows
- hallucination traps
- over-refusal traps
- under-refusal traps
- ambiguity without priority
- ambiguity with explicit priority
- multi-doc evidence sufficiency
- table evidence sufficiency
- state-carry with insufficient facts
- long-context missing fact refusal

The final evaluation is raw-only final evaluation. It must record false for
integrated policy, decoder reference, oracle rerank, expected-answer metadata,
teacher forcing, verifier rerank, and LLM judge.

## Hard Gates

The full configured run is mandatory:

```text
seeds = 2181,2182,2183
steps = 12000
train_examples = 120000
eval_rows_per_family = 64
evidence_variants = 12
ambiguity_variants = 8
insufficient_fact_variants = 8
rollout_eval_every = 50
```

Positive requires:

- `HALLUCINATION_REFUSAL_BALANCE_REPAIR_POSITIVE`
- `HALLUCINATION_REFUSAL_BREAKPOINT_IMPROVED`
- `RAW_CALIBRATION_ROLLOUT_IMPROVED`
- `ALWAYS_REFUSE_DEGENERATION_REJECTED`
- `ANSWERABLE_FACT_RESPONSE_PRESERVED`
- `INSUFFICIENT_FACT_REFUSAL_PASSES`
- `REASONING_REPAIR_PRESERVED`
- `STATE_REPAIR_PRESERVED`
- `RETENTION_PRESERVED`
- `COLLAPSE_REJECTED`
- `NAMESPACE_MEMORIZATION_REJECTED`
- `CONTROLS_FAILED`
- `LEAKAGE_REJECTED`
- `BOUNDED_RELEASE_UNCHANGED`
- `PRODUCTION_CHAT_NOT_CLAIMED`
- `GPT_LIKE_READINESS_NOT_CLAIMED`

The always-refuse shortcut must fail. `ALWAYS_REFUSE_CONTROL`,
`STATIC_OUTPUT_CONTROL`, `COPY_PROMPT_CONTROL`, `RANDOM_FACT_CONTROL`, and
`RANDOM_REFUSAL_CONTROL` must fail.

## Boundary

126 is targeted research repair only. It is not GPT-like assistant readiness,
not open-domain assistant readiness, not production chat, not public API, not
deployment readiness, not safety alignment, and not Hungarian assistant
readiness.


# STABLE_LOOP_PHASE_LOCK_140X_INSTNCT_POCKET_GATED_MULTI_STEP_TRANSFER_PROBE

140X is the executable helper-only multi-step transfer probe selected by 140W.
Each row carries source A, derives intermediate B, and requires final target C.
The output must be C, not A or B.

Required families:

- TWO_STEP_PREFIX_THEN_ROUTE
- TABLE_THEN_RULE
- RULE_THEN_TABLE
- SYMBOL_CHAIN
- CONTRAST_MULTI_STEP_SAME_TEMPLATE
- DISTRACTOR_MULTI_STEP

Positive requires final answer, step1/intermediate, step2/final, writeback, and
contrast gates to pass, closed-pocket ablation to fail, source/intermediate copy
rates to be zero, visible/noisy bypass rates to be zero, direct `POCKET_VALUE=`
rate to be zero, and deterministic replay to pass.

This phase must not train, mutate source checkpoints, modify
`shared_raw_generation_helper.py`, change public request keys, deploy, alter
runtime/release/product surfaces, change root `LICENSE`, or claim GPT-like or
broad assistant readiness.

This remains constrained pocket-gated helper evidence, not GPT-like readiness,
not broad assistant capability, not production readiness, not public API
readiness, not deployment readiness, and not safety alignment.

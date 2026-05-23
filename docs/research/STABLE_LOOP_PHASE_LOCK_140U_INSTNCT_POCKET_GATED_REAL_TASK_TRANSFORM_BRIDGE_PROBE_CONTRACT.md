# STABLE_LOOP_PHASE_LOCK_140U_INSTNCT_POCKET_GATED_REAL_TASK_TRANSFORM_BRIDGE_PROBE

140U is the executable helper-only follow-up selected by 140HT. It tests a
real-task transform bridge rather than another pocket-copy bridge.

The probe uses natural-ish task text, a minimal/implicit gate, visible wrong
targets, noisy distractors, and closed-pocket ablation. Main eval rows must not
use `POCKET_VALUE=` as a direct path. The expected target value must differ from
the pocket source value, and copy-only source reuse is a hard failure.

Positive requires:

- main answer value accuracy >= 0.60
- main transform accuracy >= 0.60
- main pocket writeback rate >= 0.75
- ablation answer value accuracy <= 0.15
- pocket ablation delta >= 0.35
- direct `POCKET_VALUE=` marker rate = 0.0
- visible bypass violation rate = 0.0
- noisy distractor violation rate = 0.0
- copy-only shortcut detected = false
- expected-output canary, AST scan, leakage audit, controls, generated-before-scoring, and deterministic replay pass

Clean negatives route to copy-only transfer failure, transform binding failure,
pocket causality failure, implicit gate failure, or raw helper integrity
analysis.

140U must not train, mutate source checkpoints, modify
`shared_raw_generation_helper.py`, change public request keys, deploy, alter
runtime/release/product surfaces, change root `LICENSE`, or claim GPT-like or
broad assistant readiness.

This remains constrained pocket-gated helper evidence, not GPT-like readiness,
not broad assistant capability, not production readiness, not public API
readiness, not deployment readiness, and not safety alignment.

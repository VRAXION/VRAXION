# STABLE_LOOP_PHASE_LOCK_140H_INSTNCT_POCKET_GATED_MINIMAL_MARKER_REAL_TASK_BRIDGE_PROBE

140H is an executable helper-only probe after 140G. It tests whether the
pocket-gated value path survives a minimal-marker, real-task-style bridge:
natural-ish task text is the main carrier, `POCKET_VALUE=` is forbidden in main
eval rows, explicit `POCKET_` tokens are sharply reduced, and
`GATE:POCKET_OPEN` is replaced by a minimal natural gate phrase.

The probe must not train, mutate source checkpoints, modify
`shared_raw_generation_helper.py`, import old phase runners, deploy, alter
runtime/release/product surfaces, change root `LICENSE`, or make public API
changes. Final generation must use `scripts/probes/shared_raw_generation_helper.py`
with only `prompt`, `checkpoint_path`, `checkpoint_hash`, `seed`,
`max_new_tokens`, and `generation_config` request keys.

Positive requires:

- main answer value accuracy >= 0.70
- main pocket writeback rate >= 0.80
- main phase transport success rate >= 0.80
- main contrast group accuracy >= 0.70
- ablation answer value accuracy <= 0.15
- ablation pocket writeback rate <= 0.05
- pocket ablation delta >= 0.45
- direct `POCKET_VALUE=` marker rate = 0.0
- explicit `POCKET_` token row rate <= 0.20
- explicit `GATE:POCKET_OPEN` row rate <= 0.30
- implicit/minimal gate row rate >= 0.70
- visible bypass and noisy distractor violation rates = 0.0
- visible bypass and noisy distractor controls fail
- expected-output canary, AST scan, leakage audit, controls, generated-before-scoring, and deterministic replay pass
- mutation selection chooses `open_minimal_marker_all_payloads`

Clean negatives are valid and route to marker dependency, implicit gate
causality, real-task text value binding, visible bypass, noisy distractor,
mutation selection, determinism, or raw helper integrity analyses as appropriate.

This is constrained pocket-gated helper evidence, not GPT-like readiness, not
broad assistant capability, not production readiness, not public API readiness,
not deployment readiness, and not safety alignment.

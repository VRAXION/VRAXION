# STABLE_LOOP_PHASE_LOCK_138YF_FAMILY_SPECIFIC_VALUE_ATTRACTOR_REPAIR_PLAN Result

138YF is the artifact-only planning follow-up to 138U. It designs the next repair/probe and does not train, infer, call the helper, run torch forward passes, mutate checkpoints, modify helper/backend code, import old runners, delete files, deploy, or touch runtime/release/product surfaces.

## Evidence Basis

138U found:

- `decision = wrong_value_attractor_analysis_complete`
- `root_cause = family_specific_train_value_attractor`
- `next = 138YF_FAMILY_SPECIFIC_VALUE_ATTRACTOR_REPAIR_PLAN`
- `wrong_specific_value_rate = 1.0`
- `expected_value_candidate_rate = 0.0`
- `generated_values_seen_in_train_rate = 0.09895833333333333`

The key adversarial correction is that the upstream 138WV `train_seen_value` label is not treated as strict train-row membership. Strict membership stays low, so 138YF does not claim a global memorized lookup or high-frequency train-value prior.

## Planning Result

138YF turns the evidence into this bottleneck:

```text
family-level routing partially works
intra-family value discrimination fails
family-specific wrong-value attractors dominate
```

Qwen's `Scout-First Laziness` and `Missing Intra-Family Variance` are useful planning terms, but only as proxy/hypothesis language. No scout/grower internals are claimed as measured. The actual repair concept is:

```text
intra_family_contrastive_objective
```

The next milestone is:

```text
138YI_FAMILY_SPECIFIC_VALUE_ATTRACTOR_REPAIR_PROBE
```

138YI must require `shared_raw_generation_helper.py` only, generated text before scoring, expected-output canary, AST shortcut scan, deterministic replay, controls fail, leakage rejected, source checkpoint unchanged, target checkpoint under `target/`, no helper/backend modification, no old runner imports, no expected/scorer metadata in helper requests, train/eval family-value splits, intra-family contrastive rows, and OOD family/value combinations.

Positive gates must include `intra_family_contrastive_accuracy`, reduced `intra_family_mode_collapse_rate`, reduced `family_default_attractor_rate`, improved per-family value accuracy, improved rule/table/OOD metrics, no parrot trap, low train namespace leak, low stale chat rate, and deterministic replay.

## Boundary

Reasoning is not restored. Raw assistant capability remains quarantined. Structured/tool capability remains invalidated. This is not GPT-like readiness, not open-domain assistant readiness, not production chat, not public API, not deployment readiness, and not safety alignment.

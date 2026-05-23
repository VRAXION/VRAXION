# STABLE_LOOP_PHASE_LOCK_138W_ANSWER_VALUE_GROUNDING_REPAIR_PROBE Result

## Expected Result Shape

Possible positive route:

```text
verdict = ANSWER_VALUE_GROUNDING_REPAIR_POSITIVE
decision = answer_value_grounding_repair_positive
next = 139V_ANSWER_VALUE_GROUNDING_REPAIR_SCALE_CONFIRM
```

Valid clean negative routes include:

```text
verdict = ANSWER_VALUE_GROUNDING_REPAIR_FAILS
decision = no_value_grounding_improvement
next = 138W_FAILURE_ANALYSIS

decision = parrot_trap_copy_shortcut_detected
next = 138P_PARROT_TRAP_VALUE_COPY_ANALYSIS

decision = wrapper_success_without_value_grounding_persists
next = 138WV_WRAPPER_VALUE_DECOUPLING_FAILURE_ANALYSIS

decision = wrapper_induced_amnesia_proxy_failure
next = 138WG_WRAPPER_INDUCED_AMNESIA_PROXY_FAILURE_ANALYSIS
```

Determinism mismatch route:

```text
verdict = DETERMINISM_REPLAY_MISMATCH
decision = nondeterministic_value_grounding_probe
next = 138N_DETERMINISM_FAILURE_ANALYSIS
```

No safe training path route:

```text
verdict = VALUE_GROUNDING_TRAINING_PATH_MISSING
decision = value_grounding_training_path_missing
next = 138WA_VALUE_GROUNDING_TRAINING_HELPER_INTEGRATION_PLAN
```

## Positive Gates

A positive requires helper-only free rollout improvement in real value metrics:

- `answer_value_accuracy >= 0.25`
- `exact_answer_accuracy >= 0.20`
- `value_after_prefix_accuracy >= 0.25`
- `rule_derived_value_accuracy >= 0.20`
- `table_derived_value_accuracy >= 0.20`
- `ood_symbol_value_accuracy >= 0.15`
- `post_wrapper_garbage_token_rate <= 0.20`
- `parrot_trap_detected = false`
- deterministic replay passes
- controls fail
- leakage rejected
- expected-output canary passes

Prefix-only, namespace-only, train-loss-only, and prompt-copy-only success are invalid.

## Boundary

No broad capability is restored by default. Raw assistant capability remains quarantined. Structured/tool capability remains invalidated. not GPT-like readiness. not open-domain assistant readiness. not production chat. not public API. not deployment readiness. not safety alignment.

# STABLE_LOOP_PHASE_LOCK_138YP_INSTNCT_MUTATION_POCKET_GATED_VALUE_GROUNDING_PLAN

138YP is artifact-only planning after 138YO.

138YO found:

```text
decision = instnct_adapter_prompt_bound_value_grounding_improves
verdict = INSTNCT_ADAPTER_BEATS_BYTE_GRU_BUT_POCKET_WRITEBACK_NOT_USED
next = 138YP_INSTNCT_MUTATION_POCKET_GATED_VALUE_GROUNDING_PLAN
instnct_answer_value_accuracy = 0.5182291666666666
byte_gru_answer_value_accuracy = 0.0
pocket_writeback_rate = 0.0
pocket_ablation_delta = 0.0
```

Interpretation:

The adapter path improved over the byte-GRU baseline, but 138YO did not prove
that threshold-open pocket writeback caused the improvement. The observed
improvement can be explained by prompt-bound value extraction.

138YP designs the next probe:

```text
138YQ_INSTNCT_POCKET_GATED_VALUE_GROUNDING_PROBE
```

138YQ must require:

- value selection only after an open pocket writes back to the highway
- positive `pocket_writeback_rate`
- positive `phase_transport_success_rate`
- decision-critical pocket ablation delta
- closed-pocket and visible-value-bypass controls
- deterministic replay
- raw helper request keys only
- no expected/scorer/oracle metadata in helper requests

138YP does not train, infer, call the helper, mutate checkpoints, modify
helper/backend/runtime/release surfaces, import old phase runners, start
services, deploy, or change public API surfaces.

This is not GPT-like readiness.

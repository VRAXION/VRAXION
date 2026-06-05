# E1_REAL_BACKEND_CONTINUOUS_STATE_MEDIUM_PROBE Result

## Outcome

```text
decision = e1_continuous_state_medium_probe_positive
next = E2_REAL_BACKEND_GATED_CORRECTION_VS_STATE_MEDIUM_PROBE
```

## Outputs

Primary output:

```text
target/pilot_wave/e1_real_backend_continuous_state_medium_probe/
```

Replay output:

```text
target/pilot_wave/e1_real_backend_continuous_state_medium_probe_replay/
```

## Scale

```text
seeds = 72001,72002,72003,72004,72005
train_rows = 4000
validation_rows = 1500
heldout_rows = 1500
ood_rows = 1500
counterfactual_rows = 1500
population_size = 32
generations = 100
```

## Metrics

```text
flat_final_heldout_accuracy = 0.0
state_medium_final_heldout_accuracy = 1.0
gated_state_medium_final_heldout_accuracy = 1.0
state_medium_vs_flat_delta = 1.0
gated_state_medium_vs_flat_delta = 1.0
controls_do_not_solve_task = True
flat_failure_audit_passed = True
leakage_audit_passed = True
finite_state_dynamics_passed = True
deterministic_replay_passed = True
```

## Boundary

E1 is a real-backend continuous state medium probe. It performs no natural-language pretraining, no tokenizer or next-token objective, no raw text or raw Raven work, and no Gemma-class training. It does not prove consciousness, AGI, production readiness, or model-scale reasoning. It only tests whether a tiny mutable dynamic state medium can beat a flat resistance baseline on a controlled symbolic route-selection task using real row-level evaluation.

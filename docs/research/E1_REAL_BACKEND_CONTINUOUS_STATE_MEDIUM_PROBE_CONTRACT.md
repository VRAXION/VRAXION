# E1_REAL_BACKEND_CONTINUOUS_STATE_MEDIUM_PROBE Contract

## Purpose

E1 starts the real-backend E-series after D129-D130 validated minimal mutation, rollback, row-level evaluation, and deterministic replay plumbing.

The probe compares three real mutable candidates on a controlled symbolic route-selection task:

- System A: flat normalized resistance field;
- System B: dynamic continuous state medium;
- System C: dynamic continuous state medium with gated readout.

The question is narrow: can a tiny recurrent state medium beat a strong flat resistance baseline on heldout, OOD, and counterfactual symbolic rows without backpropagation, static metric dictionaries, or hardcoded before/after metrics?

## Boundary

E1 performs no natural-language pretraining, no tokenizer work, no next-token objective, no raw text corpus work, no raw Raven work, no Gemma-class training, and no AGI/consciousness/production-readiness claim.

## Routes

Each row has these candidate routes:

```text
correct_route
shortcut_route
over_abstract_route
local_expensive_route
decoy_high_evidence_bad_route
decoy_low_cost_bad_route
decoy_stable_but_wrong_route
```

## Features

Each route has normalized worse-when-larger features:

```text
energy_cost
local_step_cost
abstraction_jump_cost
landing_error_risk
shortcut_risk
route_uncertainty
counterfactual_fragility
preservation_risk
calibration_mismatch
evidence_gap
route_margin_gap
template_collision_risk
grammar_collision_risk
binding_scope_risk
sequence_length_risk
adversarial_surface_similarity
temporal_instability_risk
state_transition_cost
```

## Splits

```text
seeds = 72001,72002,72003,72004,72005
train_rows_per_seed = 800
validation_rows_per_seed = 300
heldout_rows_per_seed = 300
ood_rows_per_seed = 300
counterfactual_rows_per_seed = 300
population_size = 32
generations = 100
mutation_sigma = 0.12
elite_count = 4
```

## System A

Flat resistance field:

```text
resistance = bias + sum(weight_i * normalized_worse_feature_i)
```

All feature weights must remain nonnegative. Lowest resistance wins.

## System B

Dynamic continuous state medium:

```text
state_{t+1} = (1 - leak) * state_t + leak * tanh(gain * (input_projection * route_features + recurrent_matrix * state_t + state_bias))
```

The final state is projected to a route score. Lowest score wins.

## System C

Gated dynamic state medium:

System C uses the same state update as System B, then combines:

```text
final_state
state_delta_mean
convergence_score
route_feature_projection
stability_proxy
```

The gated readout must use only route features and internal state signals. It must not use oracle labels or route-family lookup.

## Tie Handling

Exact numeric score ties must not default to route index order. The implementation must use a tiny deterministic route-feature signature as a tie-breaker so an untrained all-zero readout cannot solve the task by selecting the first candidate route.

## Controls

The oracle control can reach 1.0 but must be reference-only. At least seven non-oracle controls must stay below 0.90 heldout accuracy, otherwise the task is too easy or leaky.

## Decisions

Healthy positive outcomes:

```text
decision = e1_continuous_state_medium_probe_positive
next = E2_REAL_BACKEND_GATED_CORRECTION_VS_STATE_MEDIUM_PROBE

decision = e1_gated_continuous_state_medium_probe_positive
next = E2_REAL_BACKEND_GATED_STATE_MEDIUM_SCALE_STRESS_PROBE
```

Valid non-positive outcomes:

```text
decision = e1_flat_resistance_remains_preferred
next = E2_REAL_BACKEND_GATED_CORRECTION_MINIMAL_PROBE

decision = e1_no_state_medium_advantage_detected
next = E2_REAL_BACKEND_GATED_CORRECTION_MINIMAL_PROBE
```

Failure outcomes:

```text
decision = e1_task_too_easy_or_leaky
next = E1T_TASK_DIFFICULTY_REDESIGN

decision = e1_state_medium_instability_detected
next = E1S_STATE_MEDIUM_STABILITY_REPAIR

decision = e1_invalid_synthetic_metric_regression
next = E1_RETRY_WITH_REAL_ROW_LEVEL_EVAL
```

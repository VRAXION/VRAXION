# E7A Matrix Medium Primitive Scan Contract

## Purpose

`E7A_MATRIX_MEDIUM_PRIMITIVE_SCAN` is a lowest-abstraction-layer scan for the mutable matrix-medium idea. It does not test the full E4/E6 abstraction-routing proxy and does not produce a final E7 architecture verdict.

The question is narrower:

```text
Do mutable matrix stacks, shared activations, recurrent loops, and halting/restart heads show meaningful primitive behavior before we build a larger routing experiment on them?
```

## Systems

- `random_classifier`
- `linear_matrix_depth1`
- `linear_matrix_depth3`
- `linear_matrix_depth6`
- `tanh_matrix_depth3`
- `relu_matrix_depth3`
- `c19_fixed_matrix_depth3`
- `c19_rho0_matrix_depth3`
- `c19_c_mut_matrix_depth3`
- `c19_rho_mut_matrix_depth3`
- `c19_c_rho_mut_matrix_depth3`
- `c19_fixed_recurrent_fixed6`
- `c19_c_rho_mut_recurrent_fixed6`
- `c19_fixed_recurrent_halting6`
- `c19_c_rho_mut_recurrent_halting6`
- `c19_c_rho_mut_recurrent_halting_restart6`

## C19 Modes

E7A must not treat C19 as a single fixed activation. It separately tests:

- fixed `c=3.0`, `rho=1.0`
- `rho=0` off mode
- mutable `c`, fixed `rho`
- fixed `c`, mutable `rho`
- mutable `c+rho`

## Primitive Tasks

Rows are generated from four primitive families:

- `linear`: should be compatible with a linear matrix baseline
- `xor`: nonlinear interaction control
- `ring`: nonlinear radial boundary
- `wave`: periodic/nonmonotonic boundary

Splits:

- train
- validation
- heldout
- OOD
- counterfactual
- adversarial

## Required Controls

- Linear depth collapse audit: identity/linear depth stacks must be functionally collapsible to one effective matrix.
- C19 parameter-mode audit: fixed, rho-off, c-only, rho-only, and c+rho variants must be reported separately.
- Halting audit: fixed recurrent, halting recurrent, and halting+restart recurrent must be compared on accuracy, mean steps, and halt efficiency.
- Deterministic replay must hash-match required artifacts.
- Every mutation system must write accepted/rejected/rollback history.
- `progress.jsonl` must receive startup, generation, system, heartbeat/parallel, and final artifact events.

## Decision Boundary

The only valid E7A decision is:

```text
e7a_observational_primitive_scan_complete
```

Any full E7 claim is intentionally deferred.

## Non-Claims

E7A does not prove abstraction routing, natural-language reasoning, AGI, consciousness, model-scale behavior, or VRAXION architecture superiority.

# STABLE_LOOP_PHASE_LOCK_006_WAVEFIELD_MUTATION_FITNESS Contract

## Question

Does the wavefield target-probability / NLL signal from 005 actually help mutation-selection acquire a local complex phase-transport cell better than hard argmax reward?

## Arms

```text
IDENTITY_BASELINE
ORACLE_COMPLEX_WEIGHTS
HARD_ARGMAX_MUTATION
SOFT_PROB_MUTATION
SOFT_NLL_MUTATION
```

The mutable cell is a local linear circuit over local features:

```text
incoming_real
incoming_imag
gate_real
gate_imag
incoming_real * gate_real
incoming_imag * gate_imag
incoming_real * gate_imag
incoming_imag * gate_real
```

The oracle complex multiply cell is representable by these features:

```text
out_real = incoming_real * gate_real - incoming_imag * gate_imag
out_imag = incoming_real * gate_imag + incoming_imag * gate_real
```

## Metrics

```text
phase_argmax_accuracy
correct_phase_probability_at_target
target_nll
target_probability_margin
weight_distance_to_oracle
acceptance_rate
```

## Verdicts

```text
SOFT_WAVEFIELD_FITNESS_BEATS_HARD_ARGMAX
MUTATION_APPROACHES_ORACLE_COMPLEX_CELL
WAVEFIELD_MUTATION_STILL_NOT_SOLVED
NO_CLEAR_WAVEFIELD_MUTATION_ADVANTAGE
```

## Claim Boundary

This probe tests a tiny local mutation-selection landscape. It does not prove full VRAXION, consciousness, Prismion uniqueness, or production-ready mutation search.

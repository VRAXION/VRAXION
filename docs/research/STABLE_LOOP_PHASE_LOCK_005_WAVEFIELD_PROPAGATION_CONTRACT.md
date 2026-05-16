# STABLE_LOOP_PHASE_LOCK_005_WAVEFIELD_PROPAGATION Contract

## Question

Does complex amplitude-field propagation make spatial phase credit assignment smoother than the particle/frontier-style propagation used in the 004 falsification probe?

## Hypothesis

The 004 mutation search may be too sparse because local mutations get only a hard final target label. A complex wavefield can expose a smoother target signal:

```text
target_correct_probability = softmax(|amplitude[target, phase]|^2)
```

The useful diagnostic is not only final argmax accuracy. It is whether the correct target phase probability rises smoothly as the local gate-rotation rule becomes more accurate.

## Arms

```text
PARTICLE_FRONTIER_004_BASELINE
COMPLEX_WAVEFIELD_PROPAGATION
WAVEFIELD_WITH_INTERFERENCE
WAVEFIELD_NO_INTERFERENCE_ABLATION
WAVEFIELD_WITH_LOCAL_PHASE_LOSS
WAVEFIELD_TARGET_ONLY_LOSS
```

## Required Metrics

```text
phase_argmax_accuracy
correct_phase_probability_at_target
target_nll
target_probability_margin
probability_smooth_gain
probability_curve_monotonicity
family_accuracy
```

## Verdicts

```text
WAVEFIELD_CREDIT_SMOOTHER_THAN_PARTICLE
INTERFERENCE_HELPFUL
TARGET_PROBABILITY_GRADIENT_PRESENT
WAVEFIELD_SIGNAL_NOT_ESTABLISHED
```

## Claim Boundary

This probe does not prove consciousness, full VRAXION, or Prismion uniqueness. It only tests whether a wavefield formulation gives a smoother credit signal than particle/frontier propagation on toy spatial phase-lock cases.

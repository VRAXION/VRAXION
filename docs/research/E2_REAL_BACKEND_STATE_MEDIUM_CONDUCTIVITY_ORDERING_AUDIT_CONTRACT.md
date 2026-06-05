# E2_REAL_BACKEND_STATE_MEDIUM_CONDUCTIVITY_ORDERING_AUDIT Contract

## Purpose

E2 audits whether a tiny mutable dynamic state medium makes a logical route lower-resistance and more stable than shortcut, random-noise, illogical, surface-similar, contradictory, locally-expensive, and over-abstract alternatives.

This is not an AGI, consciousness, model-scale, Gemma-class, or natural-language pretraining claim.

## Boundary

- No natural-language pretraining.
- No tokenizer.
- No next-token objective.
- No raw text corpus.
- No raw Raven.
- No GPU requirement.
- No hand-designed gate module as the primary mechanism.
- No synthetic report harness.
- No hardcoded before/after result values.

## Required Systems

- `flat`
- `state_medium`
- `trajectory_readout`
- `stability_readout`
- `oracle_reference_only` as a reference-only control

The state systems may read numeric route features and internal recurrent state signals. They must not use route labels, route names, candidate index, or oracle correctness.

## Conductivity Ordering

Lowest score is treated as lowest resistance. For heldout, OOD, counterfactual, and adversarial splits, a passing system must satisfy:

```text
logical_vs_best_wrong_gap > 0
logical_vs_shortcut_gap > 0
logical_vs_noise_gap > 0
logical_vs_illogical_gap > 0
logical_vs_surface_wrong_gap > 0
logical_vs_contradiction_gap > 0
counterfactual_ordering_stability >= 0.90
ood_ordering_stability >= 0.85
perturbation_recovery_rate >= 0.85
shortcut_attractor_rate <= 0.05
contradiction_attractor_rate <= 0.05
```

## Required Proof Fields

The final decision artifact must include the real-backend proof flags requested in the pasted E2 spec, including real mutation, row-level prediction, rollback, deterministic replay, leakage sentinel, and no synthetic metric flags.

## Decision Boundary

The run must report the actual outcome. A positive result must not be forced. If controls solve the task, if state dynamics are unstable, or if synthetic/hardcoded metrics are detected, the decision must move to the matching repair path.

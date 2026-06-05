# E3_STATE_MEDIUM_SCAFFOLD_ABLATION_AND_REALITY_CHECK Contract

## Purpose

Test whether the E2 state-medium win survives removal or ablation of the hand-crafted E2 state projector.

## Required Variants

```text
e2_original_projector
random_projector
zero_projector
permuted_projector
dense_random_projector
no_recurrence
final_state_only
nonrecurrent_nonlinear
stronger_flat_pairwise
route/name/index leakage controls
```

## Required Metrics

```text
heldout_accuracy
ood_accuracy
counterfactual_accuracy
adversarial_accuracy
logical_vs_best_wrong_gap
logical_vs_shortcut_gap
logical_vs_noise_gap
logical_vs_illogical_gap
logical_vs_surface_wrong_gap
logical_vs_contradiction_gap
shortcut_attractor_rate
contradiction_attractor_rate
perturbation_recovery_rate
ordering_stability
accepted/rejected mutations
rollback count
parameter diff
deterministic replay
```

## Required Gates

```text
real mutation backend only
row-level eval only
no synthetic metrics
no hardcoded improvements
rollback accounting
parameter diffs
route/name/index leakage controls
deterministic replay
checker fails on missing artifacts
```

## Boundary

E3 is a real-backend toy ablation on the E2 task. It is not natural-language pretraining, tokenizer training, model-scale reasoning, AGI, consciousness, or production readiness evidence.

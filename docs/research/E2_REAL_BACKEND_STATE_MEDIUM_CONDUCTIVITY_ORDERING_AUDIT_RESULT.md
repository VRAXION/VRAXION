# E2_REAL_BACKEND_STATE_MEDIUM_CONDUCTIVITY_ORDERING_AUDIT Result

## Outcome

```text
decision = e2_temporal_projection_readout_positive
next = E3_REAL_BACKEND_TEMPORAL_PROJECTION_STRESS_PROBE
checker_failure_count = 0
```

## Outputs

Primary output:

```text
target/pilot_wave/e2_real_backend_state_medium_conductivity_ordering_audit/
```

Replay output:

```text
target/pilot_wave/e2_real_backend_state_medium_conductivity_ordering_audit_replay/
```

## Scale

```text
seeds = 73001,73002,73003,73004,73005
train_rows = 4000
validation_rows = 1500
heldout_rows = 1500
ood_rows = 1500
counterfactual_rows = 1500
adversarial_rows = 1500
population_size = 32
generations = 100
systems = flat,state_medium,trajectory_readout,stability_readout
mutation_attempts_per_system = 3200
accepted_mutations_total = 3847
rejected_mutations_total = 8953
```

## Metrics

```text
flat_heldout_logical_path_selected_rate = 0.0
state_medium_heldout_logical_path_selected_rate = 1.0
trajectory_readout_heldout_logical_path_selected_rate = 1.0
stability_readout_heldout_logical_path_selected_rate = 1.0

flat_ood/counterfactual/adversarial = 0.0 / 0.0 / 0.0
state_medium_ood/counterfactual/adversarial = 1.0 / 1.0 / 1.0
trajectory_readout_ood/counterfactual/adversarial = 1.0 / 1.0 / 1.0
stability_readout_ood/counterfactual/adversarial = 0.999333333333 / 0.988 / 0.932666666667

best_state_system = trajectory_readout
best_state_heldout_gap = 19.748553277717
flat_heldout_gap = -0.819968681519
state_medium_heldout_gap = 14.968417752416
trajectory_readout_heldout_gap = 19.748553277717
stability_readout_heldout_gap = 15.053853031739

flat_conductivity_ordering_passed = False
state_medium_conductivity_ordering_passed = True
trajectory_readout_conductivity_ordering_passed = True
stability_readout_conductivity_ordering_passed = False

controls_do_not_solve_task = True
leakage_sentinel_passed = True
deterministic_replay_passed = True
internal_replay_passed = True
external_replay_passed = True
rollback_test_passed = True
```

## Notes

```text
flat remains a failed baseline: heldout/OOD/counterfactual/adversarial logical selected rate = 0.0.
state_medium passes every ordering split at 1.0.
trajectory_readout passes every ordering split at 1.0 and has the largest heldout gap.
stability_readout has heldout = 1.0 but does not pass the full ordering requirement because counterfactual = 0.988 and adversarial = 0.932666666667.
non_solver_controls stayed below 0.90 heldout; oracle is reference-only.
route_index_leak_detected = False; candidate_name_leak_detected = False; shuffled_route_order_passed = True.
synthetic_harness_only = False; static_metric_dictionary_used = False; hardcoded_improvement_used = False; gradient_backprop_used = False.
```

## Boundary

E2 is a real-backend state-medium conductivity-ordering audit. It performs no natural-language pretraining, no tokenizer or next-token objective, no raw text or raw Raven work, and no Gemma-class training. It does not prove consciousness, AGI, production readiness, or model-scale reasoning.

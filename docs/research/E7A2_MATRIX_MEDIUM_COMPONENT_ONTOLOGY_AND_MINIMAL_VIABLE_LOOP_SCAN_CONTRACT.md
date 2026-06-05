# E7A2 Matrix Medium Component Ontology And Minimal Viable Loop Scan Contract

## Purpose

E7A2 is the ontology-first continuation of E7A. It checks the primitive component space that E7A intentionally did not cover:

- connection mask separate from weights
- residual/carry state
- memory trace buffer
- delta/stability readiness
- self-state mirror buffer
- energy/resistance field
- attractor and oscillation measurements
- activation mutation
- connection add/delete topology mutation

This is a controlled symbolic/numeric matrix-medium scan. It is not a natural-language reasoning result and it does not establish a final E7 architecture.

## Systems

The runner must compare:

- `matrix_activation_baseline`
- `connection_mask_plus_weight`
- `residual_carry_state`
- `trace_buffer`
- `delta_stability_readiness`
- `self_state_mirror_buffer`
- `energy_resistance_field`
- `attractor_measurement`
- `oscillation_measurement`
- `activation_mutation`
- `connection_add_delete_mutation`
- `residual_delta_readiness_pair`
- `trace_self_state_pair`
- `energy_attractor_pair`
- `mask_weight_mutation_pair`
- `activation_mutation_residual_pair`
- `self_state_adaptive_exit_pair`
- `minimal_viable_loop_candidate`
- `random_control`

Every non-random variant must use real mutation, accept/reject, rollback, parameter diff, row-level eval, and deterministic replay.

## Microtasks

Rows must cover:

- `stabilization_task`
- `routing_task`
- `adaptive_exit_task`
- `perturbation_recovery_task`
- `trace_required_task`

Splits must include train, validation, heldout, OOD, counterfactual, and adversarial rows.

## Required Metrics

Required metrics include task accuracy, convergence score, mean steps, overthinking, underthinking, oscillation rate, perturbation recovery, basin separation, energy gap, shortcut rate, parameter count, accepted/rejected mutations, rollback count, and deterministic replay.

Adaptive exit is positive only if it preserves accuracy within `0.02`, reduces steps by at least `15%`, and does not increase over/underthinking by more than `0.03`.

Trace/self-state is positive only if trace-required accuracy improves by at least `0.08`.

Energy/attractor is positive only if basin separation improves by at least `0.10` or shortcut rate drops by at least `0.05`.

Minimal viable loop is positive only if the combo improves macro composite by at least `0.07` over baseline.

## Required Artifacts

The run root is:

```text
target/pilot_wave/e7a2_matrix_medium_component_ontology_and_minimal_viable_loop_scan/
```

Required reports:

- `e7a2_component_inventory.json`
- `e7a2_primitive_coverage_report.json`
- `e7a2_microtask_generation_report.json`
- `e7a2_variant_results.json`
- `e7a2_minimal_viable_loop_report.json`
- `e7a2_ablation_report.json`
- `e7a2_attractor_report.json`
- `e7a2_oscillation_report.json`
- `e7a2_readiness_exit_report.json`
- `e7a2_trace_self_state_report.json`
- `e7a2_energy_resistance_report.json`
- `e7a2_mutation_history.json`
- `e7a2_no_synthetic_metric_audit.json`
- `e7a2_deterministic_replay_report.json`
- `aggregate_metrics.json`
- `decision.json`
- `summary.json`
- `report.md`

Long-ish runs must write `progress.jsonl`, mutation history, partial aggregate snapshots, and current best candidate summaries during the run.

## Allowed Decisions

- `e7a2_no_minimal_viable_loop_detected`
- `e7a2_adaptive_exit_primitive_positive`
- `e7a2_trace_self_state_primitive_positive`
- `e7a2_energy_attractor_primitive_positive`
- `e7a2_minimal_viable_loop_combo_detected`
- `e7a2_component_scan_complete_no_strong_winner`
- `e7a2_invalid_synthetic_or_leak_detected`

The final E7 verdict must remain intentionally deferred.

# E7A2 Matrix Medium Component Ontology And Minimal Viable Loop Scan Result

## Decision

```text
decision = e7a2_no_minimal_viable_loop_detected
best_eval_macro_composite_variant = mask_weight_mutation_pair
best_eval_macro_composite_score = 0.597398921999
checker = failure_count 0
deterministic_replay = passed
final_e7_verdict = intentionally deferred
```

Run root:

```text
target/pilot_wave/e7a2_matrix_medium_component_ontology_and_minimal_viable_loop_scan
```

E7A2 is a component ontology scan. It does not confirm a final E7 architecture.

## What E7A2 Added Over E7A

The scan now explicitly covers:

```text
connection mask separate from weights
residual / carry state
trace buffer
delta / stability readiness
self-state mirror buffer
energy / resistance field
attractor measurement
oscillation measurement
activation mutation
connection add/delete topology mutation
```

Every mutable variant used row-level eval, real mutation, accept/reject, rollback counts, parameter diff, and deterministic replay.

## Primitive Read

No primitive or combo crossed the `+0.05` meaningful ablation threshold over baseline.

```text
baseline matrix_activation_baseline = 0.570570427996
mask_weight_mutation_pair          = 0.597398921999  delta +0.026828494003
activation_mutation                = 0.582786369256  delta +0.012215941260
self_state_mirror_buffer           = 0.570053320796  delta -0.000517107200
```

The best score came from `mask_weight_mutation_pair`, but the improvement was below the threshold for a strong primitive finding.

Harmful in this run:

```text
residual_carry_state
trace_buffer
delta_stability_readiness
oscillation_measurement
connection_add_delete_mutation
residual_delta_readiness_pair
trace_self_state_pair
energy_attractor_pair
activation_mutation_residual_pair
self_state_adaptive_exit_pair
minimal_viable_loop_candidate
```

## Minimal Loop

```text
minimal_viable_loop_candidate_score = 0.487303196277
minimal_viable_loop_delta = -0.083267231719
minimal_viable_loop_combo_detected = false
```

So the all-primitive loop did not help here. It hurt relative to the simple matrix baseline.

## Readiness / Trace / Energy

Adaptive readiness did reduce steps in the best readiness variant:

```text
baseline mean steps = 6.0
best readiness mean steps = 4.116666666667
```

But it did not pass the positive gate because accuracy/composite quality did not hold.

Trace/self-state did not improve the trace-required task:

```text
baseline trace_required_accuracy = 0.583333333333
best trace_required_accuracy = 0.583333333333
trace_delta = 0.0
```

Attractor measurement improved basin separation:

```text
baseline basin_separation = 0.538718613814
best attractor basin_separation = 0.650859859649
basin_delta = 0.112141245835
```

But it failed the recovery/accuracy guard, so this is not counted as a clean energy/attractor positive.

## Interpretation

E7A2 closes the missing primitive inventory gap, but the result is mostly negative: no minimal viable matrix-medium loop emerged, and the added primitives often destabilized the small proxy.

The useful signal is narrow:

```text
mask + weight mutation looks weakly promising
activation mutation looks weakly promising
attractor measurement can increase basin separation, but not cleanly enough
readiness can reduce steps, but not cleanly enough
```

Next scientifically justified step is not E7B scale-up. The next step should isolate the two weak positives:

```text
E7A3_MASK_TOPOLOGY_AND_ACTIVATION_MUTATION_ISOLATION
```

That should retest mask topology and activation mutation with stricter recovery/accuracy guards before building a larger loop.

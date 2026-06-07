# E8C Producer Target Decomposition And Consumer Compatibility Probe Result

## Decision

```text
decision = e8c_consumer_interface_bottleneck
primary checker failure_count = 0
gpu confirm checker failure_count = 0
primary deterministic replay = passed
gpu confirm deterministic replay = passed
```

E8C did not confirm target decomposition as the main fix. The best clean learned
system was mutation-only decomposed low-bit, but it only improved usefulness
slightly over the current full-code teacher baseline and did not improve
consumer compatibility.

## Primary Evidence Run

Artifact root:

```text
target/pilot_wave/e8c_producer_target_decomposition_and_consumer_compatibility_probe/
```

Key primary means:

```text
current_full_code_teacher_baseline             usefulness 0.409966  compat 0.899947  grad_neg 0.343685
local_smooth_full_code_teacher                 usefulness 0.410276  compat 0.953390  grad_neg 0.463563
consumer_sensitivity_weighted_targets          usefulness 0.351449  compat 0.900584  grad_neg 0.012305
route_step_local_teacher_targets               usefulness 0.359844  compat 0.905437  grad_neg 0.002865
low_conflict_batch_curriculum                  usefulness 0.347613  compat 0.901172  grad_neg 0.001497
consumer_compatibility_weighted_loss           usefulness 0.380894  compat 0.911989  grad_neg 0.043186
mutation_repair_after_consumer_compatible      usefulness 0.392784  compat 0.911386  grad_neg 0.040256
mutation_only_decomposed_lowbit                usefulness 0.416326  compat 0.847152
consumer_distill_reference                     usefulness 0.635717  compat 1.000000
oracle_low_bit_reference                       usefulness 0.626563  compat 1.000000
```

The best clean score was only `+0.006360` over baseline. That is below the
required `+0.03` positive threshold. Decomposition and weighting strongly
reduced gradient negative-rate in several variants, but the usefulness stayed
low.

## GPU Confirm

Artifact root:

```text
target/pilot_wave/e8c_producer_target_decomposition_and_consumer_compatibility_probe_gpu_confirm/
```

GPU confirm reproduced the same interpretation:

```text
current_full_code_teacher_baseline             usefulness 0.387418  compat 0.895151  grad_neg 0.273810
consumer_compatibility_weighted_loss           usefulness 0.365539  compat 0.908139  grad_neg 0.029762
mutation_repair_after_consumer_compatible      usefulness 0.366822  compat 0.907974  grad_neg 0.027778
mutation_only_decomposed_lowbit                usefulness 0.406441  compat 0.852936
consumer_distill_reference                     usefulness 0.559467  compat 1.000000
oracle_low_bit_reference                       usefulness 0.620052  compat 1.000000
```

Best clean improvement was `+0.019023`, still below the positive threshold.
Replay hash comparisons matched for all required artifacts.

## Dense Graph Control

The dense graph danger control had slightly higher raw usefulness than the clean
best in both runs, but with poor compatibility/code-shape behavior:

```text
primary dense_graph_danger_control usefulness 0.421913  compat 0.694324
primary clean best                 usefulness 0.416326  compat 0.847152

gpu dense_graph_danger_control     usefulness 0.416416  compat 0.675649
gpu clean best                     usefulness 0.406441  compat 0.852936
```

This was below the runner's graph-soup decision margin and was not a clean win
because consumer compatibility collapsed. It remains a warning: raw answer score
can be raised by graph-like freedom while making the RAM write interface less
reusable.

## Interpretation

E8C falsified the simple target-decomposition hypothesis for this proxy.
Mechanical decomposition, sensitivity weighting, route-step-local targets,
low-conflict batching, and consumer-compatible loss improved gradient
diagnostics, but did not close the consumer/reference gap.

The main remaining gap is not just jagged teacher targets. The consumer
references remain much higher, which points to a producer architecture or
consumer interface bottleneck. Since consumer-distill/oracle references are
strong while learned producer writes remain weak, the next test should focus on
the write/read interface contract or producer architecture, not just loss
decomposition.

## Next Step

Recommended next experiment:

```text
E8D_PRODUCER_CONSUMER_INTERFACE_BRIDGE_PROBE
```

Focus: isolate whether the missing piece is a learnable adapter/integrator
between producer write bundles and the frozen consumer read contract, while
keeping oracle writes out of learned inference and avoiding semantic lanes or
dense graph regression.

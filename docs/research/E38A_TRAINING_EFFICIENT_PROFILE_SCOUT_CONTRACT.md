# E38A Training-Efficient Profile Scout Contract

## Summary

E38A measures which Flow/Grounding profile sizes remain efficient for
mutation/rollback pocket-generation work.

It is not a final training run and does not lock the final AI profile by itself
unless the tested range shows a bounded optimum.

Core question:

```text
What is the largest D/M/R/K profile that still has usable mutation throughput,
accepted-mutation rate, and parallel execution behavior?
```

## Profiles

```text
P1: D=64,   M=32,  R=32,  K=16
P2: D=128,  M=64,  R=64,  K=32
P3: D=256,  M=128, R=128, K=64
P4: D=512,  M=256, R=256, K=128
P5: D=768,  M=384, R=384, K=192
P6: D=1024, M=512, R=512, K=256
```

Where:

```text
D = Stable Grounding / Flow Field width
M = Proposal / Trace Memory width
R = Router / Control State width
K = Pocket internal capacity
```

## Required Measurements

```text
candidate_eval_per_sec
mutations_per_sec
accepted_mutations_per_sec
accepted_rate
rollback_rate
latency_p50 / latency_p95
param_count
capacity_units
best_score
wall_time
cpu_time
GPU batched forward throughput if CUDA is available
RAM/GPU heartbeat snapshots
quality anchor using E35/E36 stable pocket import
```

## Systems

```text
no_library_scratch_quality_anchor
stable_pocket_plus_adapter_quality_anchor
profile_mutation_search
gpu_batched_forward_probe
```

## Decision Labels

```text
e38a_training_efficient_profile_candidate_found
e38a_profile_max_not_bounded_extend_sweep
e38a_compute_bottleneck_before_quality
e38a_invalid_artifact_detected
```

## Positive Signal

E38A should identify either:

```text
a highest viable tested profile, if larger profiles collapse
```

or:

```text
that the tested range is not yet bounded and a larger E38B sweep is required
```

## Hard Requirements

```text
real profile-specific matrix workload
mutation/rollback accepted/rejected counts
row-level results
mutation history
progress.jsonl and hardware_heartbeat.jsonl
parallel CPU lanes
optional CUDA batch throughput report
deterministic replay report
target checker and sample-only checker
no gradient descent
no optimizer/backprop
no final profile lock unless bounded by evidence
no AGI/consciousness/model-scale claims
```

Boundary: E38A is a capacity/throughput scout for profile sizing. It does not
prove raw language reasoning, AGI, consciousness, deployed-model behavior, or
model-scale behavior.

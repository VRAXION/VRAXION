# E8F Proposal Memory And Router Commit Probe Result

Status: completed.

## Decision

```text
decision = e8f_proposal_memory_not_sufficient
official_evidence = CPU primary
cpu_checker_failure_count = 0
cpu_deterministic_replay_passed = true
gpu_confirm_decision = e8f_proposal_memory_not_sufficient
gpu_confirm_checker_failure_count = 4
```

The CPU primary run is the official replay-checked evidence. The GPU confirm
matched the same aggregate decision and best system, but CUDA replay did not
byte-match two full row-level diagnostic artifacts:

```text
proposal_memory_report.json
temporal_trace_report.json
```

The GPU aggregate metrics, system results, decision, and markdown report
hash-matched. The remaining CUDA mismatch was confined to row-level diagnostic
floating point differences, so the GPU result is treated as a confirmatory
metric run, not as official deterministic evidence.

## Official CPU Primary

Run root:

```text
target/pilot_wave/e8f_proposal_memory_and_router_commit_probe/
```

Configuration:

```text
seeds = 106001..106012
device = cpu
cpu_workers = 12
execution_mode = evidence_cpu_12seed
```

Key rounded metrics:

| system | usefulness | trace validity | drift slope |
|---|---:|---:|---:|
| direct_overwrite_baseline | 0.50 | 0.90 | 0.03 |
| output_feedback_only | 0.50 | 0.88 | 0.03 |
| proposal_memory_no_commit | 0.56 | 0.87 | 0.03 |
| proposal_memory_plus_simple_commit | 0.41 | 0.88 | 0.04 |
| proposal_memory_plus_router_commit_gate | 0.41 | 0.89 | 0.03 |
| proposal_memory_plus_learned_commit | 0.52 | 0.92 | 0.02 |
| proposal_memory_plus_per_skill_commit | 0.54 | 0.96 | 0.02 |
| proposal_memory_ring_buffer | 0.53 | 0.93 | 0.02 |
| proposal_memory_plus_verifier_pocket | 0.41 | 0.87 | 0.04 |
| proposal_memory_plus_stepwise_renormalization | 0.41 | 0.89 | 0.03 |
| dense_graph_danger_control | 0.54 | 0.64 | 0.05 |
| oracle_stepwise_commit_reference | 0.72 | 1.00 | 0.00 |

Best learned system:

```text
best_system = proposal_memory_no_commit
best_usefulness = 0.56
best_trace_validity = 0.87
direct_overwrite_usefulness = 0.50
direct_overwrite_trace_validity = 0.90
remaining_oracle_gap = 0.16
```

## Interpretation

Proposal memory improved final composition usefulness, but it did not improve
temporal trace validity. The best learned system was
`proposal_memory_no_commit`, which gained usefulness over direct overwrite but
lost trace validity.

Commit controllers showed the opposite pattern:

```text
shared/per-skill/ring commit variants improved trace validity
but did not close the usefulness gap
```

So E8F does not confirm the proposal-memory/commit hypothesis as sufficient.
It shows a split:

```text
proposal/no-commit = better answer/usefulness, weaker trace
commit/ring/per-skill = better trace hygiene, weaker answer/usefulness
```

The dense graph control improved answer-like usefulness relative to some commit
variants, but trace validity collapsed, so it remains a danger/control rather
than a valid primary success.

## GPU Confirm

Run root:

```text
target/pilot_wave/e8f_proposal_memory_and_router_commit_probe_gpu_confirm/
```

Configuration:

```text
seeds = 106013..106016
device = cuda
execution_mode = gpu_confirm_4seed
```

GPU aggregate result matched the CPU conclusion:

```text
decision = e8f_proposal_memory_not_sufficient
best_system = proposal_memory_no_commit
best_usefulness = 0.56
best_trace_validity = 0.86
direct_overwrite_usefulness = 0.47
direct_overwrite_trace_validity = 0.89
remaining_oracle_gap = 0.17
```

GPU deterministic replay did not pass full row-level hash matching. The
mismatch remained after deterministic CUDA flags and artifact rounding, and was
limited to row-level diagnostic streams. Therefore GPU is not used as the
official replay-checked result.

## Recommendation

Do not treat pocket output as automatic truth, but also do not assume proposal
memory alone solves the interface.

The next probe should target the contradiction E8F exposed:

```text
Can a commit/integrator objective jointly optimize trace validity and
downstream usefulness, instead of improving one while hurting the other?
```

A good next step is a trace-supervised commit/integrator probe where success
requires both:

```text
usefulness > direct_overwrite_baseline
trace_validity > direct_overwrite_baseline
```

Boundary: E8F is a controlled symbolic/numeric pocket-router probe only. It
does not make raw-language, deployed-model, AGI, consciousness, or model-scale
claims.

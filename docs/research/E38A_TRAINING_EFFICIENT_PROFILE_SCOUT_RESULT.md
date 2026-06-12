# E38A Training-Efficient Profile Scout Result

Status: complete.

Decision:

```text
e38a_training_efficient_profile_candidate_found
```

Selected profile:

```text
P4 = D512 / M256 / R256 / K128
```

E38A measured the largest D/M/R/K profile that remains training-efficient under
the current mutation/rollback workload. The result is a default candidate for the
next pocket-generation curriculum, not a permanent model-scale lock.

## Primary Run

Run root:

```text
target/pilot_wave/e38a_training_efficient_profile_scout
```

Configuration:

```text
profiles = P1,P2,P3,P4,P5,P6
seeds_per_profile = 6
rows = 192
generations = 36
population = 12
cpu_workers = 23
gpu_iterations = 160
gpu_batch_size = 8192
```

Checker:

```text
target_checker_failure_count = 0
sample_only_checker_failure_count = 0
deterministic_replay_passed = true
```

## Profile Throughput

| Profile | D/M/R/K | Params | Candidate eval/s | Accepted mutations/s | Accepted rate | Viable |
|---|---:|---:|---:|---:|---:|---|
| P1 | 64/32/32/16 | 3,792 | 2902.124 | 50.242 | 0.034722 | yes |
| P2 | 128/64/64/32 | 14,752 | 1819.239 | 31.907 | 0.034722 | yes |
| P3 | 256/128/128/64 | 58,176 | 853.023 | 9.366 | 0.021991 | yes |
| P4 | 512/256/256/128 | 231,040 | 323.802 | 5.150 | 0.032022 | yes |
| P5 | 768/384/384/192 | 518,592 | 178.054 | 3.188 | 0.035880 | no |
| P6 | 1024/512/512/256 | 920,832 | 117.893 | 1.723 | 0.029321 | no |

P4 is the largest profile that stayed above the E38A viability floor:

```text
P4 eval_ratio_vs_P1 = 0.111574
P4 accepted_sec_ratio_vs_P1 = 0.102513

P5 eval_ratio_vs_P1 = 0.061353
P5 accepted_sec_ratio_vs_P1 = 0.063463

P6 eval_ratio_vs_P1 = 0.040623
P6 accepted_sec_ratio_vs_P1 = 0.034296
```

Interpretation:

```text
P4 is the current training-efficient max profile.
P5/P6 are not memory blocked; they are mutation-search-throughput blocked.
```

## GPU Forward Probe

CUDA was available on:

```text
NVIDIA GeForce RTX 4070 Ti SUPER, 16 GB VRAM
```

GPU batched forward throughput:

| Profile | Rows/s | Peak VRAM |
|---|---:|---:|
| P1 | 10.528M | 14.8 MB |
| P2 | 10.075M | 21.3 MB |
| P3 | 18.086M | 34.5 MB |
| P4 | 9.268M | 61.1 MB |
| P5 | 7.121M | 89.1 MB |
| P6 | 6.387M | 115.8 MB |

Interpretation:

```text
GPU forward capacity is not the immediate bottleneck.
The current bottleneck is candidate mutation/evaluation search efficiency.
```

## Confirm Lanes

Three independent confirm lanes all selected P4:

| Confirm | Selected | P4 eval/s | P4 accepted/s | P5 accepted/s | P6 accepted/s |
|---|---|---:|---:|---:|---:|
| seed38002 | P4 | 378.760 | 5.579 | 3.226 | 1.862 |
| seed38003 | P4 | 358.975 | 5.253 | 2.847 | 1.868 |
| seed38004 | P4 | 379.410 | 5.612 | 2.868 | 1.743 |

All confirm checkers passed with `failure_count = 0`.

## Quality Anchor

The quality anchor was diagnostic only:

| System | Target-world success | Stable-target success | Bitslip-target success | Wrong feature write | False frame commit |
|---|---:|---:|---:|---:|---:|
| no-library scratch | 0.9725 | 0.9875 | 0.9500 | 0.008958 | 0.023750 |
| stable pocket + adapter | 0.7625 | 1.0000 | 0.40625 | 0.000000 | 0.008381 |

This does not change the P4 profile decision. It says future pocket-library
imports still need explicit adapter compatibility checks, especially under
bitslip/framing stress.

## Conclusion

The first practical pocket-generation curriculum should use:

```text
default_profile = P4
Flow D = 512
proposal/memory M = 256
router/state R = 256
pocket/internal K = 128
```

P5/P6 should not be used as the default until the mutation search path is
redesigned or GPU-batched mutation/evaluation is made native. They fit in
memory, but they waste wall-clock under the current mutation engine.

Boundary: E38A is a capacity/throughput scout for profile sizing. It does not
prove raw language reasoning, AGI, consciousness, deployed-model behavior, or
model-scale behavior.

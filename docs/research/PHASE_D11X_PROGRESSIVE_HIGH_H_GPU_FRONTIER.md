# Phase D11x: Progressive High-H GPU Frontier

Date: 2026-05-01

## Summary

D11x reran the sparse high-H GPU frontier after the D11 H512 work. This was an
infrastructure/scout run, not a release-candidate run.

Question:

```text
How large can we push the sparse evaluator on the RTX 4070 Ti SUPER,
and does simply increasing H create a useful basin signal?
```

Verdict:

```text
D11X_GPU_FRONTIER_CONFIRMED_INFRA_ONLY
```

## Hardware

GPU:

```text
NVIDIA GeForce RTX 4070 Ti SUPER, 16GB VRAM
```

## Throughput Frontier

Shared settings:

```text
eval_len = 64
eval_seeds = 990001
active_edges = 25,000
```

| H | active-edge usage | batch | candidates/s | peak GPU allocation |
|---:|---:|---:|---:|---:|
| 1,024 | 2.3842% | 64 | 120.83 | 147.6 MB |
| 4,096 | 0.1490% | 64 | 98.54 | 468.7 MB |
| 8,192 | 0.0373% | 64 | 62.78 | 899.5 MB |
| 16,384 | 0.0093% | 64 | 79.11 | 1.76 GB |
| 32,768 | 0.0023% | 64 | 50.61 | 3.47 GB |
| 65,536 | 0.0006% | 64 | 42.08 | 6.91 GB |
| 131,072 | 0.00015% | 64 | 17.12 | 13.78 GB |
| 262,144 | 0.000036% | 32 | 4.52 | 13.78 GB |
| 524,288 | 0.000009% | 16 | 4.81 | 13.80 GB |

OOM boundary:

```text
H=524288, batch=64 attempted ~30.67 GiB and hit CUDA OOM.
```

Practical interpretation:

```text
Comfortable interactive scout: H8192-H65536
Upper practical frontier:      H131072 batch64, H262144 batch32, H524288 batch16
Not practical on this GPU:     H524288 batch64 / H1048576-style batches
```

## Sensitivity Scout

Shared settings:

```text
eval_len = 128
eval_seeds = 990001,990002
active_edges = 25,000
8 perturbation candidates per H
```

| H | class result |
|---:|---|
| 8,192 | 8/8 `NO_SIGNAL` |
| 16,384 | 8/8 `NO_SIGNAL` |
| 65,536 | 8/8 `NO_SIGNAL` |

This means the GPU can evaluate the space, but the current random sparse
perturbation policy does not find a useful local signal at these large H values.

## Interpretation

D11x confirms the same conclusion as D10m, now with a fresh post-reboot run:

```text
The GPU is not the immediate blocker.
H can be made very large.
But bigger H by itself does not create a release-ready basin.
```

The brain-like hypothesis is still viable only in the structured sparse form:

```text
large potential graph + low active usage + structured wiring/training
```

It is not supported in the naive form:

```text
large H + random sparse perturbations = automatic better solution
```

## Release-Ready Meaning

This does not move the high-H path to release-ready. It does reduce uncertainty:

- We know the 16GB GPU can run huge sparse H scout evaluations.
- We know H8192-H65536 are comfortable enough for future structured probes.
- We know H262144/H524288 are possible but slow and near the memory frontier.
- We know random perturbation at huge H is flat under the current evaluator.

The next useful high-H work is not "go bigger"; it is:

```text
D11k/D12 structured high-H start:
  H8192 or H16384
  100k-400k active edges
  non-random wiring prior
  hardened semantic controls
```

## Progress Map

```text
GLOBAL RELEASE-READY AI MAP

[1] H384 top_01 research checkpoint
    DONE

[2] D10 artifact/state hardening
    DONE

[3] H512 activation/search infrastructure
    DONE

[4] H512 paired confirm
    FAIL: objective mismatch

[5] progressive high-H GPU frontier
    CURRENT RESULT: INFRA PASS / SCIENCE NO-SIGNAL
    Max practical sparse scout: H524288 batch16
    Comfortable structured scout target: H8192-H65536

[6] next high-H science gate
    structured wiring/projection start, not larger random H

[7] release-ready AI candidate
    H384 research package can move now;
    high-H release remains blocked
```

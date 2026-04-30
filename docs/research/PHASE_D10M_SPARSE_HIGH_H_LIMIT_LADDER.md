# Phase D10m Sparse High-H GPU Limit Ladder

Date: 2026-04-30

Verdict: `D10M_GPU_FRONTIER_MAPPED`

## Summary

D10m pushed the D10j sparse GPU evaluator upward until the RTX 4070 Ti SUPER
became practically memory-bound. The purpose was to test the large-potential /
low-active-usage hypothesis at infrastructure scale, not to promote a network.

Result:

```text
GPU evaluation is not the immediate blocker.
H4096-H16384 is comfortably interactive.
H65536 is still usable for scout work.
H131072-H524288 is technically possible but increasingly slow/memory-bound.
H1048576 batch8 saturated the 16GB card and was stopped as impractical.
```

The science signal is more nuanced: with a fixed 25k active-edge budget,
H4096 showed safe-positive sensitivity, but larger H values became mostly flat.
Increasing edge density did not automatically restore the signal. Bigger H is
therefore not magic by itself; it needs structured initialization, projection,
or training.

## Throughput Frontier

Representative rows, `eval_len=64`, one eval seed:

| H | active edges | usage | batch | candidates/s | peak VRAM |
|---:|---:|---:|---:|---:|---:|
| 4,096 | 25,000 | 0.1490% | 64 | 91.13 | 469 MB |
| 8,192 | 25,000 | 0.0373% | 64 | 77.80 | 899 MB |
| 16,384 | 25,000 | 0.0093% | 64 | 67.07 | 1.76 GB |
| 65,536 | 25,000 | 0.0006% | 64 | 19.63 | 6.91 GB |
| 131,072 | 25,000 | 0.0001% | 64 | 5.45 | 13.78 GB |
| 262,144 | 25,000 | 0.00004% | 32 | 5.32 | 13.78 GB |
| 524,288 | 25,000 | 0.000009% | 16 | 3.27 | 13.81 GB |

`H=1,048,576`, `25k` active edges, `batch=8` saturated the card near 16GB and
did not emit a throughput row after several minutes, so it was stopped.

## Sensitivity Ladder

Fixed 25k active-edge sensitivity, 16 perturbations per H:

| H | usage | safe | unsafe | echo trap | no signal |
|---:|---:|---:|---:|---:|---:|
| 2,048 | 0.5963% | 0 | 0 | 11 | 5 |
| 4,096 | 0.1490% | 8 | 0 | 8 | 0 |
| 8,192 | 0.0373% | 0 | 0 | 0 | 16 |
| 16,384 | 0.0093% | 0 | 0 | 0 | 16 |
| 65,536 | 0.0006% | 0 | 0 | 0 | 16 |
| 131,072 | 0.0001% | 0 | 0 | 0 | 16 |

Constant-usage sanity probe:

| H | active edges | usage | safe | unsafe | echo trap | no signal |
|---:|---:|---:|---:|---:|---:|---:|
| 8,192 | 100,000 | 0.1490% | 0 | 0 | 8 | 8 |
| 8,192 | 400,000 | 0.5961% | 0 | 1 | 6 | 9 |
| 16,384 | 100,000 | 0.0373% | 0 | 0 | 4 | 12 |
| 16,384 | 400,000 | 0.1490% | 1 | 0 | 15 | 0 |

Interpretation:

```text
H4096 / 25k is the best current live high-H scout point.
H8192+ is evaluable, but the current random local perturbation policy goes flat
or echo-cliffy.
Keeping the same usage ratio alone is not sufficient.
```

## Long-Horizon Meaning

The user's large sparse brain-like hypothesis remains alive, but it is now more
specific:

```text
large potential H helps only if the active circuit is structured enough.
extreme H + tiny random sparse usage is not automatically searchable.
```

This is compatible with the brain analogy: a large brain is not useful because
every possible connection is searched randomly. It is useful because sparse
circuits are grown, organized, trained, and constrained.

## Practical Frontier

```text
interactive scout:
  H4096-H16384

large but still usable:
  H32768-H65536

near-card-limit:
  H131072-H524288

not currently practical:
  H1048576 on this evaluator/GPU
```

## Next Step

D10n should not blindly push H higher. The next useful gate is a
density/projection/training ladder:

```text
D10n High-H Structure Gate

1. H4096 as positive high-H reference
2. H8192/H16384 with improved structured starts:
   - beta8_lifted
   - trained/synthetic projection variants
   - density bands around 0.05%, 0.15%, 0.6%
3. random-label and echo-trap controls stay mandatory
4. any candidate must later get CPU/dense-reference confirm
```

No high-H promotion claim follows from D10m alone.

## Progress Map

```text
GLOBAL AI PLAN MAP

[1] H384 beta.8 generalist
    DONE

[2] causal mechanism
    DONE: edge + threshold co-adaptation

[3] H384 seed replication
    RUNNING: D10b CPU main

[4] sparse GPU feasibility
    DONE:
      D10j -> H1024/25k works
      D10k -> H1024/25k searchable scout signal

[5] high-H limit ladder
    DONE:
      D10m -> GPU can reach huge H, but signal is not monotonic

[6] high-H structure gate
    NEXT:
      stop brute size scaling
      test density + projection + structured starts
```

# Phase D10j Sparse High-H GPU Feasibility

Date: 2026-04-30

Verdict: `D10J_HIGH_H_PROMISING`

## Summary

D10j tested the brain-like scaling hypothesis at the infrastructure/scout
level: large potential graph, low active edge usage, and batched GPU
evaluation. This is not a final basin claim because there is no fair H512/H1024
trained checkpoint yet.

The GPU sparse edge-list evaluator passed H384 dense-vs-sparse correctness and
comfortably handled H1024 low-usage workloads. Sensitivity scout also showed a
notable signal at H1024 with 25k active edges.

## Correctness

H384 beta.8-derived dense reference vs sparse edge-list evaluator:

```text
max_abs_diff: 0.0
verdict_flip: false
dense_elapsed_s: 1.20
sparse_elapsed_s: 1.28
sparse_peak_mb: 15.9
```

This validates the sparse propagation path for scout use.

## Throughput

Representative throughput, `eval_len=128`, 2 eval seeds:

| H | active edges | usage | batch | candidates/s | peak VRAM |
|---:|---:|---:|---:|---:|---:|
| 1024 | 5,000 | 0.48% | 128 | 103.30 | 338 MB |
| 1024 | 25,000 | 2.39% | 128 | 86.70 | 377 MB |
| 1024 | 100,000 | 9.55% | 128 | 45.60 | 873 MB |

The key gate passed: H1024 with at least 25k active edges can run batch >=32.
VRAM is not the limiting factor at this scale.

## Sensitivity Scout

Synthetic sparse perturbation scout, 32 candidates per setting:

| H | active edges | usage | positive safe | positive unsafe | echo trap | no signal |
|---:|---:|---:|---:|---:|---:|---:|
| 512 | 5,000 | 1.91% | 6 | 2 | 10 | 14 |
| 512 | 25,000 | 9.56% | 2 | 0 | 0 | 30 |
| 512 | 100,000 | 38.22% | 0 | 0 | 0 | 32 |
| 1024 | 5,000 | 0.48% | 3 | 1 | 15 | 13 |
| 1024 | 25,000 | 2.39% | 18 | 1 | 13 | 0 |
| 1024 | 100,000 | 9.55% | 0 | 0 | 1 | 31 |

The H1024 / 25k active-edge regime is the first setting that looks interesting
for the brain-like hypothesis. It is not evidence of a trained basin, but it is
evidence that the large sparse space is evaluable and metric-sensitive.

## Interpretation

D10h and D10i showed that adding edges around H384 beta.8 is easy to trap:
global dense fill is too cliffy, and small edge-add is a short-eval unigram
lure. D10j changes the frame: instead of forcing more edges into H384, it checks
whether bigger H with low active usage is computationally viable.

Result:

```text
large sparse H is viable on GPU;
H1024 / 25k edges is the best next scout regime;
fair science still requires a trained H512/H1024 checkpoint/projection.
```

## Progress Map

```text
GLOBAL AI PLAN MAP

[1] H384 beta.8 generalist
    DONE

[2] causal mechanism
    DONE: edge + threshold co-adaptation

[3] H384 seed replication
    RUNNING: D10b CPU main

[4] dense/add local probes
    DONE:
      D10h dense/prune -> echo/cliff
      D10i edge-add -> short signal, confirm reject

[5] sparse high-H GPU
    D10j DONE
    result: H1024 / 25k low-usage regime is feasible and sensitive

[6] fair H512/H1024 science run
    NEXT after checkpoint/projection gate
```

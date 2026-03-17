# v4.2 Self-Wiring Graph

This directory contains the active self-wiring graph line.

## Core Files

- `model/graph.py`: stable NumPy reference implementation
- `model/graph_v3.c`: C backend and performance path
- `lib/utils.py`: score functions and training loops, including cyclic training
- `lib/data.py`: small data helpers
- `lib/log.py`: live logging support for multiprocessing sweeps

## Recommended Entry Points

- `tests/test_model.py`: adversarial correctness and stability stress test
- `tests/test_cyclic.py`: cyclic training smoke
- `tests/rng_tier_benchmark.py`: RNG-quality sensitivity benchmark
- `tests/sparse_scaling_benchmark.py`: sparse-forward scaling benchmark
- `tests/graph_v3_probe.py`: compare and inspect the C path
- `CREDIT_GUIDED_REWIRING.md`: research note on recurrence, pain probes, and credit-guided rewiring
- `CREDIT_GUIDED_REWIRING_SKETCH.md`: prototype plan for `forward_batch_trace()` + `backward_credit()`

## Test Layout Notes

- `tests/fixtures/`: frozen replay assets used by deterministic replay checks
- `tests/gpu_experimental/`: isolated GPU prototype and plateau-study scripts

## Intentionally Excluded From Clean Main

The previous `surprise` learning path was removed from this cleaned main candidate.
It underperformed the mutation+selection baseline and still needs redesign before returning to the default branch.

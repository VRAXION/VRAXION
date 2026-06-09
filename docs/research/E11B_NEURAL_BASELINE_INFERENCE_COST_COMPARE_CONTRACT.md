# E11B Neural Baseline Inference Cost Compare Contract

## Purpose

`E11B_NEURAL_BASELINE_INFERENCE_COST_COMPARE` follows E10 and asks a narrower
cost question:

```text
On the same E10 noisy-route proxy task, how does the integrated Flow runtime
compare against small neural-controller inference baselines on quality and
inference cost?
```

This is not a trained raw-grid neural model benchmark. It is a first
inference-cost comparison over the same E10 detector evidence.

## Scope

The benchmark reuses the E10 deterministic binary Flow-grid task generator and
scoring functions. Neural baselines receive route-confidence and
trace-confidence detector evidence per skill and step, then select a fixed
region-operator decoder. This deliberately gives the neural controllers a
favorable setup: they do not need to learn the grid transform decoder from raw
state.

## Systems

```text
FLOW_E10_SCHEDULED_SCHEMA_GATED_PRUNED_PYTHON
FLOW_E10_BITSET_COST_MODEL
TINY_MLP_ROUTE_ONLY_CONTROLLER
TINY_MLP_TRACE_CONTROLLER
TINY_GRU_TRACE_CONTROLLER
SMALL_TRANSFORMER_TRACE_CONTROLLER
```

`FLOW_E10_BITSET_COST_MODEL` is an estimate only. It is not a measured Rust or
bitpacked implementation.

## Required Metrics

```text
usefulness
trace_validity
answer_accuracy
useful_writeback_recall
wrong_writeback_rate
destructive_overwrite_rate
route_repair_rate
transfer_coverage
quality_matched
proxy_ops_per_tick
cost_per_correct_trace
cost_per_valid_writeback
ops_ratio_vs_flow_python
python_wall_time_per_row
python_cpu_time_per_row
peak_traced_memory_kb
deterministic_replay_passed
```

Quality matched means:

```text
trace_validity >= 0.90
usefulness >= 0.85
useful_writeback_recall >= 0.85
wrong_writeback_rate <= 0.05
```

## Positive Gate

The E10 Flow primary must:

```text
be quality matched
keep wrong_writeback_rate == 0
keep destructive_overwrite_rate == 0
have at least one quality-matched neural comparator
have at least 3x lower proxy_ops_per_tick than the cheapest quality-matched neural comparator
have lower cost_per_valid_writeback than the cheapest quality-matched neural comparator
keep the bitset cost model lower than the Python Flow cost model
pass deterministic replay
avoid raw-grid neural model or training claims
```

Python wall-time is reported but does not decide the positive gate, because this
probe compares unoptimized Python implementations and scalar/MAC cost models.

## Decisions

Allowed decisions:

```text
e11b_flow_proxy_cost_advantage_vs_quality_matched_neural_confirmed
e11b_neural_quality_mismatch_no_cost_claim
e11b_flow_proxy_cost_advantage_not_confirmed
e11b_invalid_or_incomplete_run
```

## Boundary

E11B is a controlled synthetic binary Flow-grid inference-cost proxy. It does
not prove hardware speedups, trained raw-grid neural performance, raw-language
behavior, deployed-model behavior, or broad model-scale behavior.

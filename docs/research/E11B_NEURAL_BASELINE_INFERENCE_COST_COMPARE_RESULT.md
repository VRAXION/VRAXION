# E11B Neural Baseline Inference Cost Compare Result

Status: completed.

## Decision

```text
decision = e11b_flow_proxy_cost_advantage_vs_quality_matched_neural_confirmed
next = E11C_TRAINED_RAW_GRID_NEURAL_BASELINE_CONFIRM
primary_system = FLOW_E10_SCHEDULED_SCHEMA_GATED_PRUNED_PYTHON
positive_gate_passed = true
deterministic_replay_passed = true
checker_failure_count = 0
```

Run root:

```text
target/pilot_wave/e11b_neural_baseline_inference_cost_compare/
```

## What Was Compared

E11B reuses the E10 noisy-route task and scoring path, then compares the E10
Flow primary against small neural-controller inference baselines:

```text
TINY_MLP_ROUTE_ONLY_CONTROLLER
TINY_MLP_TRACE_CONTROLLER
TINY_GRU_TRACE_CONTROLLER
SMALL_TRANSFORMER_TRACE_CONTROLLER
```

These neural systems are favorable controller proxies over E10 detector
evidence. They are not trained raw-grid neural models.

## Quality And Proxy Cost

| system | usefulness | trace | recall | wrong | repair | ops/tick | cost/trace | ops vs Flow |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| FLOW_E10_SCHEDULED_SCHEMA_GATED_PRUNED_PYTHON | 0.990 | 0.992 | 0.986 | 0.000 | 0.978 | 200.0 | 201.5 | 1.00 |
| FLOW_E10_BITSET_COST_MODEL | 0.990 | 0.992 | 0.986 | 0.000 | 0.978 | 48.0 | 48.4 | 0.24 |
| TINY_MLP_ROUTE_ONLY_CONTROLLER | 0.962 | 0.957 | 0.929 | 0.071 | 0.767 | 832.0 | 869.7 | 4.16 |
| TINY_MLP_TRACE_CONTROLLER | 0.995 | 0.996 | 0.997 | 0.003 | 0.990 | 832.0 | 835.4 | 4.16 |
| TINY_GRU_TRACE_CONTROLLER | 0.995 | 0.996 | 0.997 | 0.003 | 0.990 | 5024.0 | 5044.7 | 25.12 |
| SMALL_TRANSFORMER_TRACE_CONTROLLER | 0.992 | 0.990 | 0.994 | 0.006 | 0.979 | 42880.0 | 43306.6 | 214.40 |

Quality-matched neural systems:

```text
TINY_MLP_TRACE_CONTROLLER
TINY_GRU_TRACE_CONTROLLER
SMALL_TRANSFORMER_TRACE_CONTROLLER
```

The route-only MLP is not quality matched because its wrong writeback rate is
`0.071`, above the `0.05` gate.

## Main Cost Result

Cheapest quality-matched neural comparator:

```text
TINY_MLP_TRACE_CONTROLLER
proxy_ops_per_tick = 832.0
```

Flow primary:

```text
FLOW_E10_SCHEDULED_SCHEMA_GATED_PRUNED_PYTHON
proxy_ops_per_tick = 200.0
```

Ratio:

```text
832.0 / 200.0 = 4.16x
```

So the E10 Flow primary is `4.16x` cheaper in the scalar/MAC proxy cost model
than the cheapest quality-matched neural controller.

The bitset model remains an estimate:

```text
FLOW_E10_BITSET_COST_MODEL proxy_ops_per_tick = 48.0
```

That implies a possible `17.33x` proxy-op advantage versus the cheapest
quality-matched neural controller, but no Rust/bitpacked implementation was run
in E11B.

## Python Wall-Time

Python wall-time did not favor the current Flow implementation:

| system | wall seconds | wall seconds / row |
|---|---:|---:|
| FLOW_E10_SCHEDULED_SCHEMA_GATED_PRUNED_PYTHON | 4.818 | 0.008364 |
| TINY_MLP_ROUTE_ONLY_CONTROLLER | 2.476 | 0.004298 |
| TINY_MLP_TRACE_CONTROLLER | 2.568 | 0.004459 |
| TINY_GRU_TRACE_CONTROLLER | 2.650 | 0.004601 |
| SMALL_TRANSFORMER_TRACE_CONTROLLER | 3.140 | 0.005452 |

This means the current Python Flow runner is slower wall-clock than these
simple Python neural-controller loops. The positive E11B result is therefore a
proxy operation-cost result, not a measured Python speedup.

## Interpretation

E11B supports the narrow cost claim:

```text
Against quality-matched neural-controller baselines over the same E10 evidence,
the Flow runtime has lower scalar/MAC proxy inference cost.
```

It does not yet prove:

```text
trained raw-grid neural comparison
optimized Rust/bitpacked Flow speed
hardware-normalized latency
end-to-end neural model cost
```

The next test should be `E11C_TRAINED_RAW_GRID_NEURAL_BASELINE_CONFIRM`, where
the neural baseline must train from grid/route inputs rather than receiving
E10 detector evidence directly.

## Verification

```text
python3 scripts/probes/run_e11b_neural_baseline_inference_cost_compare.py
python3 scripts/probes/run_e11b_neural_baseline_inference_cost_compare_check.py --out target/pilot_wave/e11b_neural_baseline_inference_cost_compare --write-summary
```

The checker passed with `failure_count = 0`.

Boundary: E11B is a deterministic synthetic binary Flow-grid inference-cost
proxy only. It does not make hardware speedup, trained raw-grid neural,
deployment, model-scale, or broad capability claims.

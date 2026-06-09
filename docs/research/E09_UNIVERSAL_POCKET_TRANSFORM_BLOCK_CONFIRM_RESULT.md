# E09 Universal Pocket Transform Block Confirm Result

Status: completed.

## Decision

```text
decision = e09_universal_pocket_transform_block_confirmed
next = E10_OPERATOR_LIBRARY_TRANSFER_AND_NOISY_ROUTE_CONFIRM
primary_system = UNIVERSAL_BLOCK_SCHEDULED_SCHEMA_GATED_PRUNED
positive_gate_passed = true
deterministic_replay_passed = true
checker_failure_count = 0
```

Run root:

```text
target/pilot_wave/e09_universal_pocket_transform_block_confirm/
```

## Integrated Components

E09 combines the three previously separate proxy findings into one controlled
runtime:

```text
E07: triggered scheduling and rollout-style salience selection
E08: shared writeback schema with branch/trace gate
E8H4: region-operator pocket abstraction over binary Flow-grid state
```

The tested universal block shape is:

```text
detector_id
condition
read_region
transform_op
write_region
branch_id
trace_before / trace_after
confidence
cost
reason_code
```

## Key Metrics

| system | usefulness | trace validity | answer accuracy | final state | wrong write | destructive | branch contam | cost/tick |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| DIRECT_OVERWRITE_ALL_POCKETS | 0.550 | 0.605 | 0.443 | 0.611 | 0.851 | 0.247 | 0.100 | 8.000 |
| SCHEDULED_PRIVATE_DIALECT_WRITEBACK | 0.780 | 0.862 | 0.628 | 0.859 | 0.637 | 0.135 | 0.000 | 2.400 |
| SCHEMA_GATED_HANDCODED_REGION_REFERENCE | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 2.000 |
| MUTATION_OPERATOR_LIBRARY_NO_SCHEDULER | 0.529 | 0.593 | 0.406 | 0.597 | 0.849 | 0.252 | 0.000 | 5.000 |
| UNIVERSAL_BLOCK_SCHEDULED_SCHEMA_GATED | 0.904 | 0.902 | 0.904 | 0.909 | 0.389 | 0.106 | 0.000 | 2.927 |
| UNIVERSAL_BLOCK_SCHEDULED_SCHEMA_GATED_PRUNED | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 2.300 |

Positive-gate deltas for the primary integrated block:

```text
usefulness_delta_vs_direct = +0.450208
trace_validity_delta_vs_direct = +0.394517
trace_validity_delta_vs_no_scheduler = +0.406929
cost_reduction_vs_direct = 0.712500
```

Split robustness for the primary integrated block:

```text
heldout usefulness = 1.000, trace_validity = 1.000
ood usefulness = 1.000, trace_validity = 1.000
counterfactual usefulness = 1.000, trace_validity = 1.000
adversarial usefulness = 1.000, trace_validity = 1.000
```

## Interpretation

The integrated form passed: scheduling, schema-gated writeback, trace checking,
and region-transform pockets can coexist in one deterministic binary Flow-grid
runtime. Direct overwrite and no-scheduler mutation-library controls both
failed badly on wrong writes and trace preservation.

The result should not be read as a new mutation-discovery proof. The primary
arm uses the pruned region-operator library inside the integrated runtime. E8H4
remains the stronger evidence for mutation-discovered operator quality; E09
tests whether that block shape can be wired together safely with E07/E08 style
scheduling and writeback protection.

## Verification

```text
python3 scripts/probes/run_e09_universal_pocket_transform_block_confirm.py
python3 scripts/probes/run_e09_universal_pocket_transform_block_confirm_check.py --out target/pilot_wave/e09_universal_pocket_transform_block_confirm --write-summary
```

The checker passed with `failure_count = 0`.

Boundary: E09 is a deterministic synthetic binary Flow-grid integration probe
only. It does not make language, deployment, model-scale, or broad capability
claims.

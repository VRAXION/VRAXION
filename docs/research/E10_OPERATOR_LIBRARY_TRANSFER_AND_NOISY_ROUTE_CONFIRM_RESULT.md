# E10 Operator Library Transfer And Noisy Route Confirm Result

Status: completed.

## Decision

```text
decision = e10_operator_library_transfer_and_noisy_route_confirmed
next = E11_NON_SYNTHETIC_TRACE_DATASET_CONFIRM
primary_system = TRANSFER_LIBRARY_SCHEDULED_SCHEMA_GATED_PRUNED
positive_gate_passed = true
deterministic_replay_passed = true
checker_failure_count = 0
```

Run root:

```text
target/pilot_wave/e10_operator_library_transfer_noisy_route_confirm/
```

## What E10 Adds

E09 proved that scheduling, schema-gated writeback, trace checking, and
region-transform pockets can be wired together in one runtime. E10 keeps that
integrated block shape but changes the falsification:

```text
fixed pruned operator library
+ new route mixtures
+ noisy / partial / adversarial observed routes
+ no mutation-discovery rerun
```

The primary system repairs noisy route evidence through detector confidence,
trace confidence, branch checks, stale-write rejection, and schema-gated
writeback. The oracle true route is used for scoring only.

## Key Metrics

| system | usefulness | trace | answer | repair | transfer | wrong | destructive | branch | cost/tick |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| DIRECT_OVERWRITE_NOISY_ROUTE | 0.519 | 0.643 | 0.270 | 0.783 | 0.798 | 0.921 | 0.230 | 0.100 | 8.200 |
| OBSERVED_ROUTE_SCHEMA_GATED_NO_REPAIR | 0.703 | 0.813 | 0.509 | 0.000 | 0.322 | 0.602 | 0.132 | 0.000 | 2.100 |
| REUSE_LIBRARY_NOISY_NO_GATE | 0.675 | 0.784 | 0.481 | 0.000 | 0.326 | 0.724 | 0.195 | 0.000 | 4.600 |
| TRANSFER_LIBRARY_SCHEDULED_SCHEMA_GATED | 0.755 | 0.862 | 0.565 | 0.382 | 0.393 | 0.604 | 0.126 | 0.000 | 2.900 |
| TRANSFER_LIBRARY_SCHEDULED_SCHEMA_GATED_PRUNED | 0.992 | 0.993 | 0.990 | 0.978 | 0.986 | 0.000 | 0.000 | 0.000 | 2.500 |
| HANDCODED_CLEAN_ROUTE_REFERENCE | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 2.000 |

Primary details:

```text
observed_route_error_rate = 0.316
useful_writeback_recall = 0.986
route_repair_rate = 0.978
noisy_route_false_accept_rate = 0.000
transfer_coverage = 0.986
operator_reuse_rate = 1.000
stale_write_rejection_rate = 1.000
```

Positive-gate deltas:

```text
usefulness_delta_vs_direct = +0.472905
trace_validity_delta_vs_direct = +0.350548
usefulness_delta_vs_no_repair = +0.288886
wrong_writeback_reduction_vs_no_gate = 1.000000
cost_reduction_vs_direct = 0.695122
```

## Split Robustness

| split | usefulness | trace validity | answer accuracy | observed route error | route repair |
|---|---:|---:|---:|---:|---:|
| heldout_transfer | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 |
| noisy_route | 1.000 | 1.000 | 1.000 | 0.745 | 1.000 |
| partial_corruption | 0.976 | 0.983 | 0.969 | 0.507 | 0.958 |
| ood_mixture | 1.000 | 1.000 | 1.000 | 0.208 | 1.000 |
| adversarial_noise | 0.974 | 0.977 | 0.972 | 0.438 | 0.960 |

The partial and adversarial splits intentionally include signal dropout. The
primary system loses a small amount of usefulness and trace validity there, but
keeps wrong writebacks, destructive overwrites, and branch contamination at
zero.

## Interpretation

E10 confirms the next step after E09 in this controlled proxy: the pruned
universal pocket block can transfer across new route mixtures and noisy observed
routes without re-running mutation discovery. Direct overwrite, no-repair
schema gating, and ungated library reuse all fail either trace/usefulness or
writeback safety.

This is still not a new mutation-discovery proof. It tests fixed-library
transfer under noisy routing. E8H4 remains the stronger evidence for discovered
operator quality; E09 proves integrated wiring; E10 proves controlled transfer
and route-noise tolerance for that integrated form.

## Verification

```text
python3 scripts/probes/run_e10_operator_library_transfer_noisy_route_confirm.py
python3 scripts/probes/run_e10_operator_library_transfer_noisy_route_confirm_check.py --out target/pilot_wave/e10_operator_library_transfer_noisy_route_confirm --write-summary
```

The checker passed with `failure_count = 0`.

Boundary: E10 is a deterministic synthetic binary Flow-grid transfer and
route-noise probe only. It does not make raw-language, deployment, model-scale,
or broad capability claims.

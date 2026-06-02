# D64B Rust Sparse IPF Diagnostic Calibration And Shuffle Hardening Result

Status:

```text
decision = score_vector_shuffle_gap_insufficient
verdict = D64B_SCORE_VECTOR_SHUFFLE_GAP_INSUFFICIENT
next = D64S_SCORE_VECTOR_STRUCTURE_REPAIR
best_arm = CALIBRATED_FULL_IPF_DIAGNOSTIC_LAYER
```

Artifact root:

```text
target/pilot_wave/d64b_rust_sparse_ipf_diagnostic_calibration_and_shuffle_hardening/smoke
```

Run mode:

```text
scale-lite
seeds = 12001,12002,12003,12004,12005
rows_per_seed = 200 train / 200 test / 200 ood
```

The full 800-row run was not used in this pass because the D64B runner is much
heavier than D64: it evaluates 16 arms and 54 Rust diagnostic estimator
controllers. The scale-lite run still exercises all tracks, all regimes, all
arms, row-level logs, Rust bridge calls, and checker gates.

## Readout

D64B answered one narrow question:

```text
Calibration improved weak diagnostics and preserved high task performance, but
the destructive candidate/top-k shuffle gap stayed too small.
```

So this is not a clean D65 go-ahead. The correct next step is the structure
repair route:

```text
D64S_SCORE_VECTOR_STRUCTURE_REPAIR
```

## Core Metrics

Best arm on `MIXED_EVAL`:

```text
exact = 0.995000
correlated_echo = 0.992000
adversarial_distractor = 0.983000
support = 7.1732
counter_support = 2.1732
false_confidence = 0.000000
```

External and abstain checks:

```text
external_test_required external accuracy = 0.997000
indistinguishable abstain = 1.000000
min_seed_exact = 0.991000
fallback_rows = 0
failed_jobs = []
rust_path_invoked = true
```

Weak diagnostic mean:

```text
D64 replay weak mean = 0.421500
calibrated weak mean = 0.609656
```

Weak bit detail:

| diagnostic | D64 replay | calibrated |
| --- | ---: | ---: |
| entropy_high | 0.482125 | 0.893125 |
| external_test_need | 0.244000 | 0.458875 |
| internal_unresolvable | 0.459875 | 0.586500 |
| support_effort_pressure | 0.500000 | 0.500125 |

Strong diagnostics were preserved:

| diagnostic | calibrated |
| --- | ---: |
| margin_low | 1.000000 |
| support_independence_low | 1.000000 |
| collision_pressure | 1.000000 |
| counterfactual_pressure | 0.994500 |
| adversarial_pressure | 0.993750 |

Shuffle controls:

| control | mixed exact | gap vs best |
| --- | ---: | ---: |
| CANDIDATE_SHUFFLE_CONTROL | 0.984200 | 0.010800 |
| SUPPORT_SHUFFLE_CONTROL | 0.557400 | 0.437600 |
| TOPK_PRESERVING_SHUFFLE_CONTROL | 0.984200 | 0.010800 |
| ENTROPY_PRESERVING_SHUFFLE_CONTROL | 0.984200 | 0.010800 |
| ADVERSARIAL_SHUFFLE_CONTROL | 0.793600 | 0.201400 |

The destructive shuffle gate required a 0.03 gap. Candidate/top-k/entropy
preserving shuffles remained too close, so D64B correctly routes to D64S.

## Boundary

D64B only hardens and calibrates a Rust sparse IPF diagnostic layer for
controlled symbolic joint formula discovery. It does not prove a full VRAXION
brain, raw visual Raven reasoning, Raven solved, AGI, consciousness, DNA/genome
success, architecture superiority, or production readiness.

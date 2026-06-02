# D64B Rust Sparse IPF Diagnostic Calibration And Shuffle Hardening Contract

## Goal

D64 confirmed that a Rust sparse IPF diagnostic layer can guide the ECF action
controller on controlled symbolic joint formula discovery. D64B is a follow-up
audit, not a new solver.

Question:

```text
Can the Rust sparse IPF diagnostic layer keep D64-level task performance while:
1. calibrating the weak diagnostic bits,
2. preserving the strong diagnostic bits,
3. making score-vector shuffle controls clearly worse?
```

## Boundary

D64B only hardens and calibrates a Rust sparse IPF diagnostic layer for
controlled symbolic joint formula discovery. It does not prove a full VRAXION
brain, raw visual Raven reasoning, Raven solved, AGI, consciousness, DNA/genome
success, architecture superiority, or production readiness.

The formula solver remains the controlled symbolic stack. Rust sparse logic is
used for diagnostic/action-control signals only.

## Upstream

Required upstream:

```text
D64_RUST_SPARSE_IPF_DIAGNOSTIC_LAYER_PROTOTYPE
decision = rust_sparse_ipf_diagnostic_layer_confirmed
next = D65_RUST_SPARSE_IPF_SCORE_AGGREGATION_PROTOTYPE
```

D64B reads the D64 smoke summary when available and writes:

```text
d64_upstream_manifest.json
```

If D64 artifacts are missing, D64B must report that in the manifest instead of
inventing values.

## Arms

Fair arms:

```text
D64_FULL_IPF_LAYER_REPLAY
CALIBRATED_FULL_IPF_DIAGNOSTIC_LAYER
ENTROPY_CALIBRATED_LAYER
EXTERNAL_NEED_CALIBRATED_LAYER
UNRESOLVABLE_CALIBRATED_LAYER
SUPPORT_EFFORT_CALIBRATED_LAYER
STRONG_DIAGNOSTICS_ONLY
WEAK_DIAGNOSTICS_ONLY
```

Controls:

```text
CANDIDATE_SHUFFLE_CONTROL
SUPPORT_SHUFFLE_CONTROL
TOPK_PRESERVING_SHUFFLE_CONTROL
ENTROPY_PRESERVING_SHUFFLE_CONTROL
ADVERSARIAL_SHUFFLE_CONTROL
RANDOM_DIAGNOSTIC_CONTROL
DIAGNOSTIC_ABLATION_CONTROL
TRUTH_LEAK_SENTINEL_REFERENCE_ONLY
```

`TRUTH_LEAK_SENTINEL_REFERENCE_ONLY` is not fair. It exists only to prove the
checker can see forbidden diagnostic leakage.

## Diagnostics

Strong diagnostics from D64:

```text
margin_low
support_independence_low
collision_pressure
counterfactual_pressure
adversarial_pressure
```

Weak diagnostics targeted by calibration:

```text
entropy_high
external_test_need
internal_unresolvable
support_effort_pressure
```

The Rust diagnostic input may use only score/evidence summary features. It may
not use support regime, track, seed, row id, true formula, clean D63 proxy flags,
or expected labels.

## Required Artifacts

Generated outputs only under:

```text
target/pilot_wave/d64b_rust_sparse_ipf_diagnostic_calibration_and_shuffle_hardening/
```

Required reports include:

```text
queue.json
progress.jsonl
compute_probe.json
dataset_manifest.json
d64_upstream_manifest.json
ipf_diagnostic_definition_report.json
score_vector_input_report.json
proxy_leakage_audit_report.json
shuffle_control_audit_report.json
diagnostic_calibration_report.json
weak_diagnostic_repair_report.json
strong_diagnostic_preservation_report.json
track_uniformity_audit_report.json
rust_estimator_mapping_report.json
rust_invocation_report.json
support_cost_frontier_report.json
false_confidence_report.json
aggregate_metrics.json
decision.json
summary.json
report.md
row_outputs_test.jsonl
row_outputs_ood.jsonl
```

Longer runs must write progress and partial snapshots over time.

## Positive Gate

D64B is positive if the best calibrated/strong diagnostic arm:

```text
exact per major track >= D64 best - 0.003
mixed correlated >= D64 best - 0.005
mixed adversarial >= D64 best - 0.005
hidden budget exact >= 0.985
external-test accuracy >= 0.985
indistinguishable abstain >= 0.99
false confidence <= 0.01
destructive shuffle/random/ablation controls are at least 0.03 worse on MIXED_EVAL
fallback_rows = 0
failed_jobs = []
```

Calibration also needs either:

```text
weak diagnostic mean improves by >= 0.05
```

or:

```text
STRONG_DIAGNOSTICS_ONLY passes the D64-level task gate
```

## Decisions

```text
decision = rust_sparse_ipf_diagnostic_calibration_confirmed
next = D65_RUST_SPARSE_IPF_SCORE_AGGREGATION_PROTOTYPE
```

if calibrated diagnostics keep D64-level performance and clear the shuffle gap.

```text
decision = diagnostic_layer_positive_with_weak_bits_excluded
next = D65_RUST_SPARSE_IPF_SCORE_AGGREGATION_WITH_STRONG_DIAGNOSTICS_ONLY
```

if weak bits do not improve but strong diagnostics alone carry the useful signal.

```text
decision = score_vector_shuffle_gap_insufficient
next = D64S_SCORE_VECTOR_STRUCTURE_REPAIR
```

if task performance remains high but destructive shuffle controls stay too close.

```text
decision = d64b_diagnostic_calibration_not_confirmed
next = D64B_REPAIR
```

if D64-level performance or calibration evidence is not confirmed.

## Validation

```powershell
python -m py_compile scripts/probes/run_d64b_rust_sparse_ipf_diagnostic_calibration_and_shuffle_hardening.py
python -m py_compile scripts/probes/run_d64b_rust_sparse_ipf_diagnostic_calibration_and_shuffle_hardening_check.py
python scripts/probes/run_d64b_rust_sparse_ipf_diagnostic_calibration_and_shuffle_hardening.py --out target/pilot_wave/d64b_rust_sparse_ipf_diagnostic_calibration_and_shuffle_hardening/smoke --seeds 12001,12002,12003,12004,12005 --train-rows-per-seed 800 --test-rows-per-seed 800 --ood-rows-per-seed 800 --workers auto --cpu-target 50-75 --heartbeat-sec 20 --scale-mode scale-lite
python scripts/probes/run_d64b_rust_sparse_ipf_diagnostic_calibration_and_shuffle_hardening_check.py --check-only --out target/pilot_wave/d64b_rust_sparse_ipf_diagnostic_calibration_and_shuffle_hardening/smoke
git diff --check
```

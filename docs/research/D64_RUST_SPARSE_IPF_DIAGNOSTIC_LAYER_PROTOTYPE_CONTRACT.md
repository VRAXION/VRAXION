# D64 Rust Sparse IPF Diagnostic Layer Prototype Contract

## Purpose

D64 tests whether the ECF controller can use a Rust sparse IPF-style diagnostic
layer built from score/evidence-vector summaries, not from the clean D63 proxy
flags.

The narrow question:

```text
Can canonical Rust Network::propagate_sparse modules turn score/evidence shape
summaries into useful controller diagnostics for controlled symbolic joint
formula discovery?
```

## Boundary

This remains controlled symbolic joint formula discovery. The formula solver is
still symbolic. D64 only moves the diagnostic layer toward Rust sparse execution.

D64 does not claim full VRAXION brain, raw visual Raven reasoning, Raven solved,
AGI, consciousness, DNA/genome success, architecture superiority, or production
readiness.

## Upstream

Expected D63:

```text
decision = rust_sparse_ecf_diagnostic_component_migration_confirmed
next = D64_RUST_SPARSE_IPF_DIAGNOSTIC_LAYER_PROTOTYPE
```

D63 is useful but too clean: its Rust estimators mirrored direct gate pressure
features. D64 hardens that by forbidding these fields as Rust estimator inputs:

```text
support_budget_pressure_norm
counterfactual_pressure_norm
adversarial_pressure_norm
internal_unresolvable_indicator
external_channel_available
support_regime
track
mixed_source_track
row_id
seed
```

## Rust Input

Allowed Rust estimator input is a compact score/evidence summary:

```text
scalar_confidence
cell_confidence
operator_confidence
joint_confidence
entropy_norm
inverse_margin
collision_norm
dominant_cluster_fraction
support_cluster_count_norm
top1_factorised_disagreement
support_count_norm
score_*_hint
context_noise_norm
```

The clean D63 proxy flags may appear only as audit targets, never as Rust sparse
estimator inputs.

## Arms

```text
D63_SYMBOLIC_DIAGNOSTIC_REFERENCE
RUST_SPARSE_ENTROPY_MARGIN_LAYER
RUST_SPARSE_COLLISION_LAYER
RUST_SPARSE_SUPPORT_INDEPENDENCE_LAYER
RUST_SPARSE_COUNTERFACTUAL_PRESSURE_LAYER
RUST_SPARSE_ADVERSARIAL_PRESSURE_LAYER
RUST_SPARSE_UNRESOLVABLE_ESTIMATOR
RUST_SPARSE_EXTERNAL_TEST_NEED_ESTIMATOR
RUST_SPARSE_FULL_IPF_DIAGNOSTIC_LAYER
HYBRID_SYMBOLIC_RUST_IPF_LAYER
SHUFFLED_SCORE_VECTOR_CONTROL
RANDOM_DIAGNOSTIC_CONTROL
DIAGNOSTIC_ABLATION_CONTROL
TRUTH_LEAK_SENTINEL_REFERENCE_ONLY
```

## Reports

Artifacts are written only under:

```text
target/pilot_wave/d64_rust_sparse_ipf_diagnostic_layer_prototype/
```

Required reports include:

```text
d63_upstream_manifest.json
d62_upstream_manifest.json
ipf_diagnostic_definition_report.json
score_vector_input_report.json
proxy_leakage_audit_report.json
diagnostic_estimator_accuracy_report.json
calibration_report.json
noisy_perturbation_report.json
rust_estimator_mapping_report.json
gate_with_ipf_diagnostics_report.json
component_ablation_report.json
truth_leak_audit_report.json
rust_invocation_report.json
support_cost_frontier_report.json
false_confidence_report.json
aggregate_metrics.json
decision.json
summary.json
report.md
row_outputs_test.jsonl
row_outputs_ood.jsonl
trained_policy_manifest.json
```

Long runs must write `queue.json`, `progress.jsonl`, partial row-generation,
pack-build, per-track metric snapshots, per-track final metrics, and final
reports. There is no black-box run.

## Positive Gate

`RUST_SPARSE_FULL_IPF_DIAGNOSTIC_LAYER` may pass if:

```text
major track exact >= D62 best - 0.007
hidden_budget exact >= 0.985
external_test_required >= 0.985
indistinguishable_abstain >= 0.99
false_confidence <= 0.01
fallback_rows = 0
rust_path_invoked = true
proxy_leakage_audit clean
shuffled/random/ablated controls worse
failed_jobs = []
```

If the hybrid arm passes but the full Rust diagnostic arm does not, D64 records a
hybrid-positive result and routes to full-Rust diagnostic repair.

## Decision Logic

```text
If full Rust IPF diagnostic layer passes:
  decision = rust_sparse_ipf_diagnostic_layer_confirmed
  next = D65_RUST_SPARSE_IPF_SCORE_AGGREGATION_PROTOTYPE

If hybrid passes but full Rust layer fails:
  decision = hybrid_ipf_diagnostic_layer_positive
  next = D64B_FULL_RUST_DIAGNOSTIC_REPAIR

If Rust inputs are too proxy-like:
  decision = d64_instrumentation_too_proxy_like
  next = D64_REPAIR_INSTRUMENTATION

If diagnostics fail:
  decision = rust_sparse_ipf_diagnostic_layer_not_confirmed
  next = D64_REPAIR

If truth leak detected:
  decision = invalid_diagnostic_truth_leak
  next = D64L_TRUTH_LEAK_REPAIR
```

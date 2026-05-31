# D63 Rust Sparse ECF Controller Component Migration Contract

## Purpose

D63 tests whether selected D62 ECF gate diagnostics can move from Python-side symbolic feature computation into Rust sparse diagnostic estimator modules.

The narrow question:

```text
Can Rust sparse modules compute/approximate the diagnostic inputs used by the
D62 learned policy-ensemble gate, without using truth labels or support-regime
labels?
```

## Boundary

This remains controlled symbolic joint formula discovery. The formula solver is still the fixed symbolic stack. D63 migrates selected diagnostic components only.

D63 does not claim full VRAXION brain, raw visual Raven reasoning, Raven solved, AGI, consciousness, DNA/genome success, architecture superiority, or production readiness.

## Upstream

Expected upstream D62:

```text
decision = policy_ensemble_learned_gate_confirmed
verdict = D62_POLICY_ENSEMBLE_LEARNED_GATE_CONFIRMED
next = D63_RUST_SPARSE_ECF_CONTROLLER_COMPONENT_MIGRATION
```

D62 learned gate routes:

```text
support_budget_pressure_norm -> HARD_BUDGET_POLICY
external_channel_available   -> EXTERNAL_TEST_POLICY
internal_unresolvable        -> ABSTAIN_POLICY
adversarial_pressure_norm    -> ADVERSARIAL_REPAIR_POLICY
counterfactual_pressure_norm -> COUNTERFACTUAL_POLICY
default                      -> SATURATED_POLICY
```

## Migrated Diagnostics

```text
support_budget_pressure
counterfactual_pressure
adversarial_pressure
internal_unresolvable
external_channel
```

The Rust diagnostic estimators are sparse threshold modules that run through the generated Rust harness using canonical `Network::propagate_sparse`.

## Arms

```text
D62_SYMBOLIC_FEATURE_GATE_REFERENCE
RUST_SPARSE_BUDGET_PRESSURE_ESTIMATOR
RUST_SPARSE_COUNTERFACTUAL_PRESSURE_ESTIMATOR
RUST_SPARSE_ADVERSARIAL_PRESSURE_ESTIMATOR
RUST_SPARSE_UNRESOLVABLE_ESTIMATOR
RUST_SPARSE_EXTERNAL_NEED_ESTIMATOR
RUST_SPARSE_ALL_DIAGNOSTICS_GATE
HYBRID_SYMBOLIC_RUST_DIAGNOSTICS_GATE
SHUFFLED_DIAGNOSTIC_CONTROL
RANDOM_DIAGNOSTIC_CONTROL
DIAGNOSTIC_ABLATION_CONTROL
TRUTH_LEAK_SENTINEL_REFERENCE_ONLY
```

## Tracks

```text
SATURATED_STABILITY
HARD_CAP8_LEARNING
MIXED_EVAL
OOD_CONTEXT_SHIFT
ADVERSARIAL_GATE_CONFUSION
EXTERNAL_TEST_REQUIRED
INDISTINGUISHABLE_SUPPORT
NOISY_CONTEXT
HIDDEN_BUDGET_CONTEXT
```

## Required Reports

Artifacts are written only under:

```text
target/pilot_wave/d63_rust_sparse_ecf_controller_component_migration/
```

Required reports:

```text
d62_upstream_manifest.json
diagnostic_feature_definition_report.json
rust_estimator_mapping_report.json
estimator_accuracy_report.json
gate_with_rust_diagnostics_report.json
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

Long runs must write `queue.json`, `progress.jsonl`, partial row-generation, pack-build, track-clone, per-track metric snapshots, per-track final metrics, and final reports. There is no black-box run.

## Positive Gate

`RUST_SPARSE_ALL_DIAGNOSTICS_GATE` or `HYBRID_SYMBOLIC_RUST_DIAGNOSTICS_GATE` must satisfy:

```text
major track exact >= D62 best - 0.005
hidden_budget >= 0.99
external_test_required >= 0.99
indistinguishable_abstain >= 0.99
false_confidence <= 0.01
fallback_rows = 0
rust_path_invoked = true
shuffled/random/ablated diagnostics worse
failed_jobs = []
```

## Decision Logic

```text
If all Rust sparse diagnostics pass:
  decision = rust_sparse_ecf_diagnostic_component_migration_confirmed
  next = D64_RUST_SPARSE_IPF_DIAGNOSTIC_LAYER_PROTOTYPE

If hybrid passes but all-Rust diagnostics fail:
  decision = hybrid_diagnostic_migration_positive
  next = D64_INCREMENTAL_DIAGNOSTIC_MIGRATION

If diagnostics fail:
  decision = diagnostic_component_migration_not_confirmed
  next = D63R_DIAGNOSTIC_ESTIMATOR_REPAIR

If truth leak detected:
  decision = invalid_diagnostic_truth_leak
  next = D63L_TRUTH_LEAK_REPAIR
```

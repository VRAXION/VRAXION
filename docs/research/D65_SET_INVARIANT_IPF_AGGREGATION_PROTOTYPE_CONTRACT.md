# D65 Set Invariant IPF Aggregation Prototype Contract

## Question

Can the Rust sparse path aggregate unordered support/evidence score sets into
useful IPF aggregate features for the ECF controller, without relying on
candidate identity, temporal order, proxy labels, or Python-precomputed final
aggregate labels?

## Context

D64U repaired the D64-D64T claim boundary:

```text
Rust sparse IPF diagnostic layer is controller-useful.
Candidate identity dependence is not confirmed.
Temporal support order is not required in the current task.
Set-invariant support/evidence aggregation is the next hypothesis.
```

## Method

The symbolic task and formula solver remain fixed. Rust aggregation arms receive
flat support/evidence set representations:

```text
support_count x candidate_count score bins
```

The generated Rust harness calls:

```text
instnct-core Network::propagate_sparse
```

for each support vector and emits aggregate feature summaries for the ECF
controller. Rust arms must not receive Python-precomputed final aggregate labels.

## Arms

```text
SYMBOLIC_SET_AGGREGATION_REFERENCE
RUST_SPARSE_SUM_AGGREGATION
RUST_SPARSE_MEAN_AGGREGATION
RUST_SPARSE_NORMALIZED_SHAPE_AGGREGATION
RUST_SPARSE_SET_INVARIANT_IPF_AGGREGATION
RUST_SPARSE_SUPPORT_COHERENCE_SET_AGGREGATION
RUST_SPARSE_COUNTERFACTUAL_DELTA_SET_AGGREGATION
HYBRID_SYMBOLIC_RUST_SET_AGGREGATION
SUPPORT_ORDER_SHUFFLE_NOOP_CONTROL
CANDIDATE_ID_SHUFFLE_CONTROL
SUPPORT_CONTENT_CORRUPTION_CONTROL
RANDOM_SCORE_AGGREGATION_CONTROL
AGGREGATION_ABLATION_CONTROL
TRUTH_LEAK_SENTINEL_REFERENCE_ONLY
```

## Required Reports

```text
d64u_upstream_manifest.json
set_aggregation_definition_report.json
order_invariance_report.json
score_shape_aggregation_report.json
rust_aggregation_mapping_report.json
aggregation_quality_report.json
controller_with_set_aggregation_report.json
support_content_corruption_report.json
truth_leak_audit_report.json
rust_invocation_report.json
aggregate_metrics.json
decision.json
summary.json
report.md
```

## Decision

If best Rust/hybrid set aggregation matches the symbolic reference floor and
controls are worse:

```text
decision = set_invariant_ipf_aggregation_confirmed
next = D66_RUST_SPARSE_SUPPORT_SCORING_MIGRATION_PLAN
```

If only hybrid passes:

```text
decision = hybrid_set_aggregation_positive
next = D65B_FULL_RUST_SET_AGGREGATION_REPAIR
```

If order matters again:

```text
decision = order_dependence_reappeared
next = D65O_ORDER_ARTIFACT_REPAIR
```

If aggregation fails:

```text
decision = set_invariant_ipf_aggregation_not_confirmed
next = D65_REPAIR
```

## Boundary

D65 only tests set-invariant Rust sparse IPF aggregation for controlled symbolic
joint formula discovery. It does not prove full VRAXION brain, raw visual Raven
reasoning, Raven solved, AGI, consciousness, DNA/genome success, architecture
superiority, or production readiness.

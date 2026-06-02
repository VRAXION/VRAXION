# D64S Score Vector Structure Repair Result

Status:

```text
decision = score_vector_structure_dependency_not_confirmed
verdict = D64S_SCORE_VECTOR_STRUCTURE_DEPENDENCY_NOT_CONFIRMED
next = D64S_REPAIR_OR_REDEFINE_DIAGNOSTIC_CLAIM
best_arm = D64B_CALIBRATED_REPLAY
```

Artifact root:

```text
target/pilot_wave/d64s_score_vector_structure_repair/smoke
```

Run mode:

```text
scale-lite
seeds = 12101,12102,12103,12104,12105
rows_per_seed = 100 train / 100 test / 100 ood
```

## Intended Readout

D64S says the current structure claim is not clean enough for D65.

The result is nuanced:

```text
There is evidence that support-order structure matters.
But FULL_STRUCTURE_AWARE_LAYER does not preserve D64B correlated accuracy.
Therefore D64S stops and does not run D65.
```

## Core Metrics

`FULL_STRUCTURE_AWARE_LAYER`, `MIXED_EVAL`:

```text
exact = 0.9948
correlated = 0.9820
adversarial = 0.9920
support = 7.2200
false_confidence = 0.0000
```

D64B reference:

```text
exact = 0.9950
correlated = 0.9920
adversarial = 0.9830
```

The D64S pass gate required:

```text
correlated >= D64B - 0.005 = 0.9870
```

Actual:

```text
correlated = 0.9820
```

So the gate fails.

## Structure Findings

Mixed exact table:

```text
D64B_CALIBRATED_REPLAY       = 0.9948
FULL_STRUCTURE_AWARE_LAYER   = 0.9948
SCORE_SHAPE_ONLY_LAYER       = 0.9948
CANDIDATE_IDENTITY_LAYER     = 0.9948
COUNTERFACTUAL_DELTA_LAYER   = 0.9948
SUPPORT_COHERENCE_LAYER      = 0.7848
CLUSTER_STRUCTURE_LAYER      = 0.7848
```

Shuffle/control gaps vs full structure:

```text
SUPPORT_ORDER_SHUFFLE        gap = 0.4312
FULL_SCORE_NOISE_CONTROL     gap = 0.1968
RANDOM_DIAGNOSTIC_CONTROL    gap = 0.2420
DIAGNOSTIC_ABLATION_CONTROL  gap = 0.1912
CANDIDATE_ID_SHUFFLE         gap = 0.0064
TOPK_VALUE_SHUFFLE           gap = 0.0064
MARGIN_PRESERVING_SHUFFLE    gap = 0.0064
ENTROPY_PRESERVING_SHUFFLE   gap = 0.0064
COUNTERFACTUAL_DELTA_SHUFFLE gap = 0.0000
CLUSTER_STRUCTURE_SHUFFLE    gap = 0.0000
SUPPORT_COHERENCE_BREAK      gap = 0.0000
```

Interpretation:

```text
candidate identity is not the main clean signal.
support order / broad score-shape disruption can damage performance.
but the full structure-aware arm does not preserve D64B correlated robustness.
```

## Validation

```text
py_compile runner/checker = passed
micro run + checker = passed
sanity run + checker = passed
smoke run + checker = passed
fallback_rows = 0
failed_jobs = []
rust_path_invoked = true
```

## Next

Do not run D65 from this result. The next step is:

```text
D64S_REPAIR_OR_REDEFINE_DIAGNOSTIC_CLAIM
```

The claim should be narrowed or repaired before score aggregation migration.

## Boundary

D64S only tests score-vector structure dependency for a Rust sparse IPF
diagnostic layer in controlled symbolic joint formula discovery. It does not
prove a full VRAXION brain, raw visual Raven reasoning, Raven solved, AGI,
consciousness, DNA/genome success, architecture superiority, or production
readiness.

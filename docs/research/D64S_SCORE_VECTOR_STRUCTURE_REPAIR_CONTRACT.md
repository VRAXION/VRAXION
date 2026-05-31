# D64S Score Vector Structure Repair Contract

## Goal

D64B improved diagnostic calibration but did not cleanly pass the shuffle gate:

```text
decision = score_vector_shuffle_gap_insufficient
best calibrated mixed exact = 0.9950
candidate shuffle exact = 0.9842
gap = 0.0108
required gap = 0.0300
next = D64S_SCORE_VECTOR_STRUCTURE_REPAIR
```

D64S asks a narrower question:

```text
Is the Rust sparse IPF diagnostic layer using meaningful score-vector
structure, or only broad proxy/shape signals?
```

If meaningful structure is confirmed, D65 may test Rust sparse IPF score
aggregation. If not, D65 must not run.

## Boundary

D64S only tests score-vector structure dependency for a Rust sparse IPF
diagnostic layer in controlled symbolic joint formula discovery. It does not
prove a full VRAXION brain, raw visual Raven reasoning, Raven solved, AGI,
consciousness, DNA/genome success, architecture superiority, or production
readiness.

The formula solver remains symbolic. Rust sparse code is used for diagnostic
estimation and controller action selection only.

## Arms

Fair arms:

```text
D64B_CALIBRATED_REPLAY
CANDIDATE_IDENTITY_LAYER
SCORE_SHAPE_ONLY_LAYER
SUPPORT_COHERENCE_LAYER
COUNTERFACTUAL_DELTA_LAYER
CLUSTER_STRUCTURE_LAYER
FULL_STRUCTURE_AWARE_LAYER
```

Controls:

```text
CANDIDATE_ID_SHUFFLE
TOPK_VALUE_SHUFFLE
MARGIN_PRESERVING_SHUFFLE
ENTROPY_PRESERVING_SHUFFLE
SUPPORT_ORDER_SHUFFLE
SUPPORT_COHERENCE_BREAK
COUNTERFACTUAL_DELTA_SHUFFLE
CLUSTER_STRUCTURE_SHUFFLE
FULL_SCORE_NOISE_CONTROL
RANDOM_DIAGNOSTIC_CONTROL
DIAGNOSTIC_ABLATION_CONTROL
TRUTH_LEAK_SENTINEL_REFERENCE_ONLY
```

`TRUTH_LEAK_SENTINEL_REFERENCE_ONLY` is not fair.

## Structure Groups

```text
candidate identity:
  margin_low

score shape:
  entropy_high
  margin_low
  support_effort_pressure

support coherence:
  collision_pressure
  support_independence_low
  adversarial_pressure

counterfactual delta:
  counterfactual_pressure
  margin_low

cluster structure:
  collision_pressure
  support_independence_low
  adversarial_pressure
```

## Required Reports

Generated outputs only under:

```text
target/pilot_wave/d64s_score_vector_structure_repair/
```

Required reports:

```text
d64b_upstream_manifest.json
score_structure_dependency_report.json
shuffle_control_matrix_report.json
structure_gap_report.json
candidate_vs_shape_report.json
support_coherence_report.json
counterfactual_delta_report.json
cluster_structure_report.json
truth_leak_audit_report.json
rust_invocation_report.json
aggregate_metrics.json
decision.json
summary.json
report.md
```

All longer runs must write `queue.json`, append `progress.jsonl`, and write
partial metrics snapshots over time.

## Pass Gate

D64S may pass if `FULL_STRUCTURE_AWARE_LAYER` satisfies:

```text
exact >= D64B exact - 0.003
corr/adv >= D64B - 0.005
external >= 0.985
abstain >= 0.99
false_confidence <= 0.01
at least one meaningful structure control gap >= 0.03
random/ablation controls worse
fallback_rows = 0
failed_jobs = []
```

If candidate identity is not required but score-shape/support structure is
required, D64S should say that directly instead of overclaiming candidate-level
structure.

## Decisions

```text
score_vector_structure_repair_confirmed
next = D65_RUST_SPARSE_IPF_SCORE_AGGREGATION_PROTOTYPE
```

```text
score_shape_dependency_confirmed_candidate_identity_not_required
next = D65_RUST_SPARSE_IPF_SCORE_AGGREGATION_PROTOTYPE
```

```text
score_vector_structure_dependency_not_confirmed
next = D64S_REPAIR_OR_REDEFINE_DIAGNOSTIC_CLAIM
```

```text
invalid_score_structure_truth_leak
next = D64S_TRUTH_LEAK_REPAIR
```

## Validation

```powershell
python -m py_compile scripts/probes/run_d64s_score_vector_structure_repair.py
python -m py_compile scripts/probes/run_d64s_score_vector_structure_repair_check.py
python scripts/probes/run_d64s_score_vector_structure_repair.py --out target/pilot_wave/d64s_score_vector_structure_repair/smoke --seeds 12101,12102,12103,12104,12105 --train-rows-per-seed 800 --test-rows-per-seed 800 --ood-rows-per-seed 800 --workers auto --cpu-target 50-75 --heartbeat-sec 20 --scale-mode scale-lite
python scripts/probes/run_d64s_score_vector_structure_repair_check.py --check-only --out target/pilot_wave/d64s_score_vector_structure_repair/smoke
git diff --check
```

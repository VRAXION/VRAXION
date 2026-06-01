# D65 Set Invariant IPF Aggregation Prototype Result

## Decision

```text
decision = set_invariant_ipf_aggregation_not_confirmed
verdict = D65_SET_INVARIANT_IPF_AGGREGATION_NOT_CONFIRMED
next = D65_REPAIR
best_arm = RUST_SPARSE_SUM_AGGREGATION
```

## Run

```text
scale_mode = scale-lite
seeds = 12301,12302,12303,12304,12305
train_rows_per_seed = 20
test_rows_per_seed = 40
ood_rows_per_seed = 40
rust_path_invoked = true
rust_aggregation_rows = 38400
rust_controller_rows = 44800
fallback_rows = 0
python_precomputed_final_aggregate_label_rows = 0
failed_jobs = []
```

## Main Result

The Rust sparse aggregation path was exercised. It consumed support/evidence
set representations and called `Network::propagate_sparse` per support vector.
The run did not use Python-precomputed final aggregate labels.

The aggregation result was not confirmed because the controls were too close:

```text
reference_exact = 0.9990
best_exact = 0.9990
order_delta = 0.0000
content_gap = 0.0350
random_gap = 0.0040
ablation_gap = 0.0000
```

So the current Rust set aggregation matched the reference, and support content
corruption hurt, but random score and aggregation ablation did not fall enough.
That means this probe cannot claim that the Rust set aggregate is the necessary
repair signal yet. The controller/task is still too robust to weak or ablated
aggregate features.

## Core Metrics

| arm | exact | corr | adv | external | abstain | support |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SYMBOLIC_SET_AGGREGATION_REFERENCE | 0.9990 | 1.0000 | 0.9950 | 1.0000 | 1.0000 | 8.8370 |
| RUST_SPARSE_SUM_AGGREGATION | 0.9990 | 1.0000 | 0.9950 | 1.0000 | 1.0000 | 8.8370 |
| RUST_SPARSE_SET_INVARIANT_IPF_AGGREGATION | 0.9990 | 1.0000 | 0.9950 | 1.0000 | 1.0000 | 8.8370 |
| SUPPORT_ORDER_SHUFFLE_NOOP_CONTROL | 0.9990 | 1.0000 | 0.9950 | 1.0000 | 1.0000 | 8.8370 |
| SUPPORT_CONTENT_CORRUPTION_CONTROL | 0.9640 | 0.9250 | 0.9200 | 1.0000 | 1.0000 | 7.6760 |
| RANDOM_SCORE_AGGREGATION_CONTROL | 0.9950 | 0.9900 | 0.9850 | 1.0000 | 1.0000 | 8.0000 |
| AGGREGATION_ABLATION_CONTROL | 0.9990 | 1.0000 | 0.9950 | 1.0000 | 1.0000 | 11.0000 |

## Interpretation

D65 is a useful negative/partial result:

```text
confirmed:
  Rust sparse aggregation bridge runs
  support order shuffle is near no-op
  support content corruption hurts
  no Python final aggregate label replay

not confirmed:
  Rust set aggregation is necessary
  aggregation ablation control is worse
  random score aggregation is clearly worse
```

Next should be `D65_REPAIR`, focused on making the task/controls sensitive
enough to distinguish real set aggregation from ablated/random aggregate
features.

Artifacts:

```text
target/pilot_wave/d65_set_invariant_ipf_aggregation_prototype/smoke/
```

Boundary: D65 only tests set-invariant Rust sparse IPF aggregation for
controlled symbolic joint formula discovery. It does not prove full VRAXION
brain, raw visual Raven reasoning, Raven solved, AGI, consciousness, DNA/genome
success, architecture superiority, or production readiness.

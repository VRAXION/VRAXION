# D68A Counter-Support Metric Semantics Audit Result

Status: completed local smoke run.

Artifact root:

```text
target/pilot_wave/d68a_counter_support_metric_semantics_audit/smoke/
```

## Decision

```text
decision = counter_support_metric_pipeline_not_confirmed
verdict  = D68A_COUNTER_SUPPORT_METRIC_PIPELINE_NOT_CONFIRMED
next     = D68R_COUNTER_METRIC_REPAIR
```

## What Was Audited

D68A rebuilt the D68 support/controller rows deterministically because D68 row
outputs are sampled diagnostics and do not include full concrete action
alternatives. The rebuild matched D68 core test metrics exactly:

```text
max_abs_exact_delta   = 0.0
max_abs_support_delta = 0.0
rust_path_invoked     = true
fallback_rows         = 0
failed_jobs           = []
```

## Core Metrics

| arm | exact | effective | support | reported unnecessary | causal unnecessary | concrete missed | wrong concrete |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| D67_BEST_REPLAY | 0.999333 | 0.999333 | 7.6795 | 0.058833 | 0.058833 | 0.000333 | 0.000333 |
| D68_TRAINED_THRESHOLD_REPLAY | 0.993833 | 0.993833 | 6.4795 | 0.058833 | 0.058833 | 0.005833 | 0.005833 |
| COUNTER_REMOVAL_REPLAY | 0.565667 | 0.565667 | 5.0000 | 0.000000 | 0.000000 | 0.434000 | 0.000000 |
| ALWAYS_COUNTER_CONTROL | 0.999333 | 0.999333 | 11.0000 | 0.565667 | 0.565667 | 0.000333 | 0.000333 |
| RANDOM_COUNTER_CONTROL | 0.783667 | 0.783667 | 6.8530 | 0.299000 | 0.299000 | 0.216000 | 0.003167 |
| NEVER_COUNTER_CONTROL | 0.565667 | 0.565667 | 5.0000 | 0.000000 | 0.000000 | 0.434000 | 0.000000 |

## Interpretation

The original "D67 asks too much" claim is only partly supported.

D67 does contain real causal extra support:

```text
D67 causal_unnecessary_counter_support_rate = 0.058833
D67 support_over_cheapest_effective_mean    = 1.365
```

However, D68 did not cleanly repair that cost. It reduced mean support:

```text
7.6795 -> 6.4795
```

but introduced a concrete counter-action failure:

```text
D68 concrete_selected_counter_missed_rate = 0.005833
D68 wrong_concrete_counter_rate           = 0.005833
D68 exact_joint_accuracy                  = 0.993833
```

The harm classification is concentrated, not diffuse:

```text
d68_loss_rows_vs_d67 = 52
classification       = selected_top1_top2_failed_but_joint_counter_would_fix
classification_rate  = 1.0
```

That means the D68 trained threshold often chose a cheaper-looking
`REQUEST_COUNTER_TOP1_TOP2` path where the row needed `REQUEST_JOINT_COUNTER`.
So the next repair should not simply "ask less"; it should distinguish cheap
top1/top2 cases from cases where joint counter-support is required.

## Audit Answer

The D68 counter-support metric pipeline is not confirmed. The support accounting
is explicit and the rebuild parity is good, but the repair objective was too
coarse: it counted broad counter-support categories without proving that the
selected concrete counter action was the one that fixed the row.

## Boundary

D68A only audits counter-support metric semantics for controlled symbolic joint
formula discovery. It does not prove full VRAXION brain, raw visual Raven
reasoning, Raven solved, AGI, consciousness, DNA/genome success, architecture
superiority, or production readiness.

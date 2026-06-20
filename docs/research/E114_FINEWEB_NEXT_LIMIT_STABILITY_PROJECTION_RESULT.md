# E114 FineWeb Next Limit Stability Projection Result

```text
decision = e114_fineweb_next_limit_projection_clean_but_targeted_data_needed
checker_failure_count = 0
```

Boundary:

```text
FineWeb next-limit projection only
not PermaCore
not TrueGolden
not final training
```

## Inputs

```text
source_alias = private_fineweb_edu_local
rows_kept = 1,000,000
source_rows_seen = 2,586,922
source_total_rows = 8,758,000
operator_count = 136
```

Filters:

```text
language = en
score >= 3.0
64 <= token_count <= 2048
```

## Stability Result

```text
chunk_count = 10
selected_hard_negative_total = 0
selected_neutral_waste_total = 0
selected_positive_total = 34,637,668
selected_call_total = 34,637,668
stability_trend = stable_clean
degradation_detected = false
```

Every 100k chunk stayed clean:

```text
hard_negative_rate = 0.0 for all chunks
neutral_waste_rate = 0.0 for all chunks
positive_rate = 1.0 for all chunks
```

## Dataset Shape

```text
all = 1,000,000
generic_negative_scope = 369,088
question_like = 363,155
calc_like = 218,488
evidence_like = 1,000,000
adversarial_like = 326
long_text = 574,631
```

## Next-Limit Projection

PermaCore probation target used by the dashboard:

```text
target = 300,000 qualified activation / operator
assumed E112 minimum = 101,601
```

Projection from the 1M row FineWeb run to the full local FineWeb source:

```text
projected_reach_permacore_count = 59 / 136
projected_need_targeted_data_count = 77 / 136
rare_operator_count = 77
```

Interpretation:

```text
The E113 recycle policy stays stable on larger FineWeb.
But natural FineWeb activation is not enough for every scoped operator.
Full FineWeb alone is projected to bring 59 operators to the 300k target.
77 rare/specialized operators still need targeted pressure data.
```

The rarest group activated only on adversarial-like cases:

```text
current_run_calls = 326 in 1M rows
projected full-FineWeb activation ~= 102,705 total
remaining after full FineWeb ~= 197,295
```

So full FineWeb is excellent no-harm stress, but not a universal rank grinder.

## Runtime

```text
seconds = 1,720.733
approx = 28.7 minutes
```

Progress and partial snapshots were written continuously during the run.

## Artifacts

```text
target/pilot_wave/e114_fineweb_next_limit_stability_projection/
```

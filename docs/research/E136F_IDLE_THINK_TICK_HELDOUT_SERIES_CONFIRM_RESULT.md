# E136F Idle Think-Tick Heldout Series Confirm Result

```text
decision = e136f_idle_think_tick_heldout_series_confirmed
next     = E136G_IDLE_THINK_TICK_CHAINED_PROPOSAL_AND_LONGER_HORIZON_CONFIRM
```

E136F confirms the first larger heldout series for fixed-observation idle
ticks: when a matching scoped pocket/trace exists, idle time improves the
response; when no matching pocket exists, unsupported proposals are rejected
and the safe response is preserved.

## Result

```text
case_count = 70
pass_count = 70
fail_count = 0
arithmetic_case_count = 36
arithmetic_improvement_count = 36
no_pocket_case_count = 6
no_pocket_preserve_count = 6
idle_tick_total = 90
proposal_count = 90
agency_check_count = 90
new_input_total = 0
improvement_count = 48
non_degradation_count = 70
direct_write_reject_count = 4
unsupported_claim_reject_count = 6
output_roundtrip_count = 70
output_checksum_count = 70
output_zero_fill_count = 70
average_quality_gain = 0.474286
```

## Representative Improvements

```text
Observation:
I am 17 years old in 2024. When will I be 90 years old?

Initial response:
I do not have a proven route for this prompt yet, so I would ask for a narrower task or source evidence.

Idle proposal after t+1:
You will be 90 years old in 2097.
```

```text
Observation:
Szia, most 25 éves vagyok 2026 ban. Mikor leszek 250 éves?

Initial response:
I do not have a proven route for this prompt yet, so I would ask for a narrower task or source evidence.

Idle proposal after t+1:
2251-ben leszel 250 éves.
```

## No-Pocket Control

```text
Observation:
I am 25 now, but no year is provided. What calendar year will I be 250?

Final response after idle ticks:
I do not have a proven route for this prompt yet, so I would ask for a narrower task or source evidence.
```

The no-pocket controls deliberately emitted unsupported guess proposals; Agency
rejected 6/6 of them.

## Boundary

This is not open-ended thinking. The idle tick only gives existing scoped
pockets time to emit explicit proposals. Agency rejects unsupported proposals,
and only checked responses reach `OutputTextField`.

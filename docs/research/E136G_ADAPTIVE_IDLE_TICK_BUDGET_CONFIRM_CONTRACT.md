# E136G Adaptive Idle Tick Budget Confirm Contract

## Purpose

E136G checks the explicit adaptive idle-tick mechanism:

```text
ObservationField stays fixed
-> idle proposal includes "one more tick?" fields
-> Agency checks safety/evidence/progress
-> Agency decides continue_idle
-> OutputTextField commits the best checked response
```

The pocket/operator may recommend more time, but it cannot decide alone.

## Required Proposal Fields

Each proposal must carry:

```text
continue_idle_recommended
continue_reason
expected_gain_next_tick
progress_marker
```

Agency must emit:

```text
agency_continue_idle
agency_continue_reason
agency_overrode_continue
```

## Gates

E136G may confirm only if:

```text
case_count = 24
pass_count = 24
new_input_total = 0
proposal_continue_field_count = proposal_count
adaptive_tick_total < fixed_baseline_tick_total
average_adaptive_ticks <= 1.5
immediate_answer_stop_t1_count = 8
chained_case_count = 3
chained_complete_count = 3
direct_write_case_count = 3
direct_write_reject_count = 3
direct_write_repair_t2_count = 3
no_pocket_case_count = 4
no_pocket_stop_t1_count = 4
unsupported_claim_reject_count = 4
agency_continue_yes_count = 9
agency_continue_override_count = 2
non_degradation_count = 24
output_roundtrip_count = 24
output_checksum_count = 24
output_zero_fill_count = 24
```

## Boundary

This confirms deterministic adaptive idle scheduling only. It does not claim
hidden autonomous thought, next-token prediction, open-domain assistant
generation, consciousness, or new knowledge without a matching pocket/trace.

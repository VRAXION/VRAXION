# E136G Adaptive Idle Tick Budget Confirm Result

```text
decision = e136g_adaptive_idle_tick_budget_confirmed
next     = E136H_CHAINED_ASSISTANT_RENDER_AND_ADAPTIVE_IDLE_ROUTE_CONFIRM
```

E136G confirms the first adaptive idle-tick budget: each proposal explicitly
recommends whether another idle tick is useful, but Agency makes the final
continuation decision.

## Result

```text
case_count = 24
pass_count = 24
fail_count = 0
adaptive_tick_total = 33
fixed_baseline_tick_total = 120
tick_savings_vs_fixed = 87
average_adaptive_ticks = 1.375000
proposal_count = 33
proposal_continue_field_count = 33
agency_continue_yes_count = 9
agency_continue_override_count = 2
immediate_answer_stop_t1_count = 8
chained_case_count = 3
chained_complete_count = 3
direct_write_case_count = 3
direct_write_reject_count = 3
direct_write_repair_t2_count = 3
no_pocket_case_count = 4
no_pocket_stop_t1_count = 4
unsupported_claim_reject_count = 4
new_input_total = 0
non_degradation_count = 24
output_roundtrip_count = 24
output_checksum_count = 24
output_zero_fill_count = 24
```

## Key Behaviors

Easy arithmetic stops after one tick:

```text
Observation:
Szia, most 25 éves vagyok 2026 ban. Mikor leszek 250 éves?

t+1 proposal:
2251-ben leszel 250 éves.

Agency:
continue_idle = false
reason = answer_complete
```

Over-eager continuation is stopped by Agency:

```text
t+1 proposal:
You will be 101 years old in 2085.
continue_idle_recommended = true
expected_gain_next_tick = 0.0

Agency:
continue_idle = false
reason = no_expected_gain_next_tick
```

Chained refinement uses multiple ticks:

```text
t+1 extract visible numbers
t+2 compute trace
t+3 render final answer
```

No-pocket controls stop immediately after unsupported guesses are rejected.

## Boundary

This is adaptive scheduling of explicit proposals, not hidden thought. More
ticks are granted only when Agency sees safe progress and expected next-tick
gain under the max tick cap.

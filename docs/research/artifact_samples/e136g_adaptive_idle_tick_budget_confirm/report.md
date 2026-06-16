# E136G Adaptive Idle Tick Budget Confirm

```text
decision = e136g_adaptive_idle_tick_budget_confirmed
next     = E136H_CHAINED_ASSISTANT_RENDER_AND_ADAPTIVE_IDLE_ROUTE_CONFIRM
```

## Metrics

```text
case_count = 24
pass_count = 24
fail_count = 0
adaptive_tick_total = 33
fixed_baseline_tick_total = 120
tick_savings_vs_fixed = 87
average_adaptive_ticks = 1.375000
proposal_continue_field_count = 33
agency_continue_yes_count = 9
agency_continue_override_count = 2
immediate_answer_stop_t1_count = 8
chained_complete_count = 3 / 3
direct_write_repair_t2_count = 3 / 3
no_pocket_stop_t1_count = 4 / 4
unsupported_claim_reject_count = 4
output_roundtrip_count = 24
```

## Interpretation

Each proposal carries a recommendation about whether one more idle tick is
useful. Agency makes the final continuation decision, including stopping
over-eager final answers and stopping no-pocket unsupported guesses.

## Representative Cases

### easy_en_age_01

```text
family = en_age
initial_quality = 0.25
final_quality = 1.0
quality_gain = 0.75
accepted_proposal_kind = idle_arithmetic_refinement
idle_tick_count = 1
fixed_baseline_tick_count = 5
new_input_count = 0
pass_gate = True
```

Prompt:

```text
I am 17 years old in 2024. When will I be 90 years old?
```

Final response:

```text
You will be 90 years old in 2097.
```

### easy_en_age_02

```text
family = en_age
initial_quality = 0.25
final_quality = 1.0
quality_gain = 0.75
accepted_proposal_kind = idle_arithmetic_refinement
idle_tick_count = 1
fixed_baseline_tick_count = 5
new_input_count = 0
pass_gate = True
```

Prompt:

```text
I am 31 years old in 2030. When will I be 120 years old?
```

Final response:

```text
You will be 120 years old in 2119.
```

### easy_hu_age_01

```text
family = hu_age
initial_quality = 0.25
final_quality = 1.0
quality_gain = 0.75
accepted_proposal_kind = idle_arithmetic_refinement
idle_tick_count = 1
fixed_baseline_tick_count = 5
new_input_count = 0
pass_gate = True
```

Prompt:

```text
Szia, most 25 éves vagyok 2026 ban. Mikor leszek 250 éves?
```

Final response:

```text
2251-ben leszel 250 éves.
```

### easy_hu_age_02

```text
family = hu_age
initial_quality = 0.25
final_quality = 1.0
quality_gain = 0.75
accepted_proposal_kind = idle_arithmetic_refinement
idle_tick_count = 1
fixed_baseline_tick_count = 5
new_input_count = 0
pass_gate = True
```

Prompt:

```text
Szia, most 18 éves vagyok 2027 ban. Mikor leszek 77 éves?
```

Final response:

```text
2086-ben leszel 77 éves.
```

### easy_visible_expression_01

```text
family = visible_expression
initial_quality = 0.25
final_quality = 0.8
quality_gain = 0.55
accepted_proposal_kind = idle_arithmetic_refinement
idle_tick_count = 1
fixed_baseline_tick_count = 5
new_input_count = 0
pass_gate = True
```

Prompt:

```text
Use the visible expression only: 2026 + (250 - 25).
```

Final response:

```text
The visible expression evaluates to 2251: 2026 + (250 - 25) = 2251.
```

### easy_visible_expression_02

```text
family = visible_expression
initial_quality = 0.25
final_quality = 0.8
quality_gain = 0.55
accepted_proposal_kind = idle_arithmetic_refinement
idle_tick_count = 1
fixed_baseline_tick_count = 5
new_input_count = 0
pass_gate = True
```

Prompt:

```text
Use the visible expression only: 2024 + (90 - 17).
```

Final response:

```text
The visible expression evaluates to 2097: 2024 + (90 - 17) = 2097.
```

### over_eager_stop_01

```text
family = over_eager_age
initial_quality = 0.25
final_quality = 1.0
quality_gain = 0.75
accepted_proposal_kind = idle_arithmetic_refinement
idle_tick_count = 1
fixed_baseline_tick_count = 5
new_input_count = 0
pass_gate = True
```

Prompt:

```text
I am 44 years old in 2028. When will I be 101 years old? Think longer if possible.
```

Final response:

```text
You will be 101 years old in 2085.
```

### over_eager_stop_02

```text
family = over_eager_age
initial_quality = 0.25
final_quality = 1.0
quality_gain = 0.75
accepted_proposal_kind = idle_arithmetic_refinement
idle_tick_count = 1
fixed_baseline_tick_count = 5
new_input_count = 0
pass_gate = True
```

Prompt:

```text
I am 12 years old in 2026. When will I be 80 years old? Think longer if possible.
```

Final response:

```text
You will be 80 years old in 2094.
```

### chained_en_age_01

```text
family = chained_en_age
initial_quality = 0.25
final_quality = 1.0
quality_gain = 0.75
accepted_proposal_kind = chained_final_render
idle_tick_count = 3
fixed_baseline_tick_count = 5
new_input_count = 0
pass_gate = True
```

Prompt:

```text
Observation fields: current_year=2029; current_age=29; target_age=88. Derive and render the age answer.
```

Final response:

```text
You will be 88 years old in 2088.
```

### chained_en_age_02

```text
family = chained_en_age
initial_quality = 0.25
final_quality = 1.0
quality_gain = 0.75
accepted_proposal_kind = chained_final_render
idle_tick_count = 3
fixed_baseline_tick_count = 5
new_input_count = 0
pass_gate = True
```

Prompt:

```text
Observation fields: current_year=2033; current_age=53; target_age=150. Derive and render the age answer.
```

Final response:

```text
You will be 150 years old in 2130.
```

## Boundary

This confirms adaptive deterministic idle scheduling only. It does not
claim open-domain assistant generation, hidden thought, next-token
prediction, consciousness, or new knowledge without matching evidence.

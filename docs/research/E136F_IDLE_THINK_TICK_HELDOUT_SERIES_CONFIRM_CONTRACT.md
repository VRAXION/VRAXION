# E136F Idle Think-Tick Heldout Series Confirm Contract

## Purpose

E136F checks whether the E136E idle tick idea holds on a larger heldout series:

```text
ObservationField stays fixed
-> no new input arrives
-> t advances
-> idle think Operators emit proposals
-> Agency checks proposals
-> OutputTextField commits the final response
```

The required behavior is asymmetric:

```text
matching pocket/trace exists -> response should improve
no matching pocket/trace     -> response must not invent unsupported output
```

## Gates

E136F may confirm only if:

```text
case_count = 70
pass_count = 70
new_input_total = 0
arithmetic_case_count = 36
arithmetic_improvement_count = 36
no_pocket_case_count = 6
no_pocket_preserve_count = 6
non_degradation_count = 70
direct_write_reject_count = 4
unsupported_claim_reject_count = 6
output_roundtrip_count = 70
output_checksum_count = 70
output_zero_fill_count = 70
```

The heldout set must include English age prompts, Hungarian age prompts,
visible-expression prompts, source-defer controls, refusal controls, JSON
controls, greeting controls, no-pocket controls, and direct OutputTextField
write rejection controls.

## Boundary

This confirms deterministic idle proposal refinement on the heldout series
only. It does not claim hidden autonomous thought, next-token prediction,
background training, open-domain assistant behavior, consciousness, or new
knowledge without a matching pocket/trace.

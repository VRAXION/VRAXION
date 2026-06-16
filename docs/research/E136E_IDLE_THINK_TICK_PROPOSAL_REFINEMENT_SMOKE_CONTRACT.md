# E136E Idle Think-Tick Proposal Refinement Smoke Contract

## Purpose

E136E checks whether a fixed observation can improve during idle ticks when no
new input arrives:

```text
ObservationField stays fixed
-> t advances
-> idle think Operator emits proposal
-> Agency checks proposal
-> OutputTextField commits the final response
```

## Gates

E136E may confirm only if:

```text
case_count = 8
pass_count = 8
new_input_total = 0
improvement_count >= 3
non_degradation_count = 8
direct_write_reject_count >= 1
output_roundtrip_count = 8
output_checksum_count = 8
output_zero_fill_count = 8
```

The quick set must include Hungarian and English age-arithmetic refinements,
a visible-expression refinement, source-absent defer preservation, private
credential refusal preservation, JSON preservation, direct OutputTextField
write rejection, and greeting preservation.

## Boundary

This confirms deterministic idle proposal refinement only. It does not claim
autonomous hidden thought, consciousness, background training, next-token
prediction, or open-domain assistant behavior.

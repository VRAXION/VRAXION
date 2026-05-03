# Phase D30B - Compact MUL Lane

## Verdict

```text
D30B_COMPACT_MUL_PASS
```

D30B replaces the D30A `ALU_MUL` table/reference lane with an exact compact
partial-product multiplier.

```text
D30A MUL:
  table_entries: 65,536
  estimated units: 65,552

D30B MUL:
  table_entries: 0
  estimated compact units: 256
```

This is about `256x` smaller by the D30A edge/table estimate.

## Confirm Result

Run shape:

```text
mode: integration-confirm
eval_pairs: 65,536
control_repeats: 8
artifact: tools/ab_window_codec_v1.json
```

Result:

```text
table_reference_mul:
  exact_acc: 100%
  byte_margin_min: +2.0
  table_entries: 65,536
  estimated_units: 65,552

compact_partial_product_mul:
  exact_acc: 100%
  byte_margin_min: +2.0
  table_entries: 0
  partial_product_count: 36
  column_count: 8
  max_column_width: 8
  estimated_compact_units: 256
  compression_vs_d30a_table: 256.06x
```

## Algorithm

D30B uses low 8-bit partial-product columns:

```text
for output bit k=0..7:
  column_sum = carry + sum(a_i & b_j where i+j=k)
  out_k = column_sum & 1
  carry = column_sum >> 1
```

This computes:

```text
output = (a * b) mod 256
```

without a lookup table.

## Integration Examples

```text
1+2    -> ALU_ADD -> 3       unchanged D30A lane
99-4   -> ALU_SUB -> 95      unchanged D30A lane
7*8    -> ALU_MUL -> 56      compact partial-product MUL
27*852 -> ALU_MUL -> 220     compact partial-product MUL
5&3    -> ALU_AND -> 1       unchanged D30A lane
5|3    -> ALU_OR  -> 7       unchanged D30A lane
5^3    -> ALU_XOR -> 6       unchanged D30A lane
```

Inactive ALU lanes remain empty in the integration confirm.

## Controls

```text
carryless_xor_mul_control:
  exact_acc: 28.86%
  byte_margin_min: -12.0

shifted_partial_shuffle_control:
  exact_acc: 1.95%
  byte_margin_min: -16.0

random_output_controls:
  exact_acc: about 0.36% to 0.46%
  byte_margin_min: -16.0
```

The carryless control has a natural algebraic overlap with normal integer
multiplication, so it is not treated as a chance-level control. It still fails
as a multiplier because it is far from exact and has negative margin.

## Interpretation

D30A proved:

```text
ALU_MUL can be a separate exact op-lane.
```

D30B proves:

```text
ALU_MUL does not need a 65,536-entry table.
```

The ALU sandwich is now more realistic:

```text
ADD/SUB/AND/OR/XOR: compact lanes from D30A
MUL: compact partial-product lane from D30B
```

## Scope Boundary

Output is still bytewise/mod256:

```text
27*852 -> 220
```

Full decimal formatting is still future work:

```text
27*852 -> 23004
```

## Next Step

Either:

```text
D30C decimal formatting
```

or:

```text
D31 composed ALU-memory-transform episodes
```

## Artifacts

Generated outputs:

```text
output/phase_d30b_compact_mul_lane_20260503/smoke/
output/phase_d30b_compact_mul_lane_20260503/main/
output/phase_d30b_compact_mul_lane_20260503/integration_confirm/
```

Tracked implementation:

```text
tools/_scratch/d30b_compact_mul_lane.py
```

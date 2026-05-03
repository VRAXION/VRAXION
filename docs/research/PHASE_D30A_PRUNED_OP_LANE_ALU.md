# Phase D30A - Pruned Op-Lane ALU Sandwich

## Verdict

```text
D30A_PRUNED_OP_LANE_ALU_PASS
```

D30A splits the D29 monolithic `ALU` lane into independently removable op-lanes:

```text
1+2    -> ALU_ADD -> 3
99-4   -> ALU_SUB -> 95
7*8    -> ALU_MUL -> 56
5&3    -> ALU_AND -> 1
5|3    -> ALU_OR  -> 7
5^3    -> ALU_XOR -> 6
```

Only the selected op-lane emits output. All inactive ALU lanes remain empty.

## Confirm Result

Run shape:

```text
mode: integration-confirm
ops: add,sub,mul,and,or,xor
eval_pairs: 65,536
control_repeats: 8
artifact: tools/ab_window_codec_v1.json
```

Primary rows:

```text
ALU_ADD: 100%, margin +2.0, inactive empty 100%
ALU_SUB: 100%, margin +2.0, inactive empty 100%
ALU_MUL: 100%, margin +2.0, inactive empty 100%
ALU_AND: 100%, margin +2.0, inactive empty 100%
ALU_OR:  100%, margin +2.0, inactive empty 100%
ALU_XOR: 100%, margin +2.0, inactive empty 100%
```

Controls:

```text
wrong-op controls:
  max exact_acc: 13.35%

random-route controls:
  exact_acc: about 19.47% to 19.73%
```

Since six op-lanes are balanced, random route chance is about 16.7% plus natural
truth-table overlaps. All controls stay below the 25% gate.

## Integration Examples

```text
1+2    -> route_family ALU -> ALU_ADD -> 3
99-4   -> route_family ALU -> ALU_SUB -> 95
7*8    -> route_family ALU -> ALU_MUL -> 56
27*852 -> route_family ALU -> ALU_MUL -> 220
5&3    -> route_family ALU -> ALU_AND -> 1
5|3    -> route_family ALU -> ALU_OR  -> 7
5^3    -> route_family ALU -> ALU_XOR -> 6
```

The `27*852` result is bytewise/mod256:

```text
27 * 852 = 23004
23004 mod 256 = 220
```

## Sandwich Size

```text
ADD: 56 estimated edges
SUB: 56 estimated edges
MUL: 65,552 estimated edges, 65,536 table entries
AND: 32 estimated edges
OR:  32 estimated edges
XOR: 32 estimated edges

TOTAL: 65,760 estimated edges
```

The high total is dominated by the D30A MUL reference lane. This is intentional
for v1: multiplication is exact and separable first, then compacted later.

## Removed-Lane Policy

Deployment can keep only selected lanes. Example policy:

```text
kept_ops = add,sub
mul/and/or/xor -> REMOVED_OP_LANE
```

This passed while keeping inactive lanes empty.

## Interpretation

D30A changes the ALU layout from:

```text
ALU -> internal selector -> operation
```

to:

```text
C-router / ALU refinement -> fixed op-lane -> output
```

This is more modular and easier to prune:

```text
ALU_ADD [keep/remove]
ALU_SUB [keep/remove]
ALU_MUL [keep/remove]
ALU_AND [keep/remove]
ALU_OR  [keep/remove]
ALU_XOR [keep/remove]
```

## Caveat

D30A is still a reference/probe. It is not a learned ALU and the MUL lane is not
yet a compact multiplier circuit. Full decimal output such as:

```text
27*852 -> 23004
```

is D30B, not D30A.

## Next Step

```text
D30B: decimal formatting / compact MUL
```

or:

```text
D31: composed episodes using ALU output in memory/transform
```

## Artifacts

Generated outputs:

```text
output/phase_d30a_pruned_op_lane_alu_20260503/smoke/
output/phase_d30a_pruned_op_lane_alu_20260503/main/
output/phase_d30a_pruned_op_lane_alu_20260503/integration_confirm/
```

Tracked implementation:

```text
tools/_scratch/d30a_pruned_op_lane_alu.py
```

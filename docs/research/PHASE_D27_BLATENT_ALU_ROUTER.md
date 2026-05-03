# Phase D27 - B-Latent ALU Router

## Verdict

```text
D27_B64_ALU_ROUTER_PASS
```

D27 checked the old ALU/router idea against the current AB/B64 bus.

## Old ALU Artifacts Found

Relevant older artifacts still exist:

```text
docs/BYTE_OPCODE_V1_CONTRACT.md
instnct-core/examples/byte_opcode_grower.rs
instnct-core/alu_integer_log.txt
instnct-core/holo_alu_log.txt
```

The old results were real but not directly wired to the current B64 interface:

```text
integer gate library:
  full adder: 8/8
  4-bit adder: 256/256
  8-bit adder: 1000/1000 sampled
  4-bit subtractor: 256/256
  4-bit comparator: 256/256

holographic ALU selector:
  ADD/SUB/AND/OR/XOR/CMP>: 1536/1536
  selector accuracy: 100%
```

## New D27 Probe

D27 adds a fresh B64 ALU/router reference:

```text
A window -> B64
B window -> B64
opcode -> router
selected ALU worker -> B64 output -> bytes
```

Ops:

```text
copy_a
not_a
and
or
xor
add_mod
sub_mod
gt_mask
eq_mask
```

Main run:

```text
eval_windows: 65,536
control_repeats: 2
artifact: tools/ab_window_codec_v1.json
```

Result:

```text
all primary ops:
  window_exact_acc: 100%
  byte_exact_acc: 100%
  bit_acc: 100%
  byte_margin_min: +2.0

random_output_controls:
  window_exact_acc: 0%
```

Some operand-swap controls pass by design for commutative ops such as `and`,
`or`, `xor`, `add_mod`, and `eq_mask`. This is not a router leak. Wrong-opcode
controls are treated as leaks only if they nearly solve the task, because some
truth tables naturally overlap.

## Interpretation

The current block separation is now:

```text
A = byte codec
B = 8-byte/B64 data bus
C = router/controller candidate
D = worker blocks
    D24 transform
    D25 memory
    D27 ALU
```

D27 is still an exact reference/probe, not a learned router. It proves that the
B64 bus can feed an opcode-selected ALU worker. The next step is to make the
router choose between memory, transform, and ALU blocks in one composed task.

## Artifacts

```text
output/phase_d27_blatent_alu_router_probe_20260503/smoke/
output/phase_d27_blatent_alu_router_probe_20260503/main/
```

Tracked implementation:

```text
tools/_scratch/d27_blatent_alu_router_probe.py
```

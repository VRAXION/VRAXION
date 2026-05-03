# A-v2 AB Compatibility Gate

Date: 2026-05-03

## Verdict

```text
A_V2_AB_COMPATIBILITY_PASS
```

The new `A-v2-H12-GeometryPolish` A-block can sit under an AB-style reciprocal
B64 bridge while preserving canonical byte-bit B64 semantics.

## Why This Gate Matters

The A-v2 block is not useful to the main pipeline just because it decodes one
byte exactly. It must also work under the AB interface:

```text
8 bytes
  -> A-window
  -> B64 bus
  -> A-window
  -> 8 bytes
```

The important constraint is:

```text
B64 must still mean little-endian signed byte bits.
```

If B64 semantics changed, existing B/D workers would not be safely compatible.

## Result

Main gate:

```text
eval_windows: 65,536
bridge: pinv_bit_bridge
window_exact_acc: 100%
byte_exact_acc: 100%
bit_acc: 100%
b_bit_semantic_acc: 100%
byte_margin_min: +2.0
b_collision_count: 0
B decoder: B encoder.T
B encoder weights per 8-byte window: 608
```

Tracked candidate artifact:

```text
tools/ab_window_codec_a_v2_candidate.json
```

## Bridge Comparison

```text
select_first8_current_b:
  FAIL
  This proves A-v2 is not drop-in compatible with the old StableCopy A-lane selector.

pinv_bit_bridge:
  PASS
  A16 -> B64 recovers canonical byte bits; B64 -> A-decodable surface uses transpose.

a_transpose_gram_bridge:
  FAIL
  Using raw A.T is not enough.

b_basis_permutation_gauge:
  CODEC-ONLY PASS
  Roundtrip exact, but B64 bit lane semantics are permuted, so it is not worker-safe.

random_projection_control:
  FAIL
  No control leak.
```

## Visual

```text
byte bits
   |
   v
A-v2-H12 A16
   |  B encoder = pseudoinverse(A)
   v
B64 signed byte-bit bus
   |  B decoder = B encoder.T
   v
A-decodable surface
   |  A-v2 mirror decoder
   v
byte bits
```

## Decision

`A-v2-H12-GeometryPolish` is now not only a standalone A-block candidate. It
has a compatible AB-v2 bridge candidate.

`tools/ab_window_codec_v1.json` remains frozen. The new artifact is tracked
separately as an AB-v2 candidate until downstream worker regression passes.

## Next Gate

Run B/D worker regression on AB-v2:

```text
B-LatentTransform
B-SlotMemory
ALU-CompactMul / ALU-OpLaneSandwich
C-StreamTokenizer / C-ContentRouter
SwitchboardExecution
```

Because B64 remains byte-bit semantic, this should mostly pass, but it must be
verified before promoting AB-v2 as the default research surface.

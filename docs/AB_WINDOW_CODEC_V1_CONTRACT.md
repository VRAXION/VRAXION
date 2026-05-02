# AB Window Codec V1 Contract

Date: 2026-05-02

## Scope

AB Window Codec V1 is the cleaned deployment-facing form of the D21-D23 A+B
component result:

```text
8 bytes -> A128 -> B64 -> A128 -> 8 bytes
```

It is a codec/component surface, not a full AI model. It is intended to be the
stable I/O substrate for the next D24 B-latent transformation tasks.

## Shape

```text
1 A-block:
  1 byte -> 16D A-code
  A-code -> 1 byte

8 A-blocks:
  8 bytes -> 128D A-window

B-block:
  128D A-window -> 64D B-window latent
  64D B-window latent -> 128D A-decodable window
```

Bit order is little-endian: `bit0` first.

## Important Detail

The B decoder is the exact transpose of the B encoder. It restores an
A-decodable 128D surface, not the full redundant D22 128D reference vector.

```text
D22 A128:
  first 8 lanes per byte = visible byte bits
  second 8 lanes per byte = redundant copy

D23/AB inverse A128:
  first 8 lanes per byte = visible byte bits
  second 8 lanes per byte = zero
```

This is sufficient for exact byte reconstruction because the A inverse reads
the visible lanes.

## Artifact

Tracked artifact:

```text
tools/ab_window_codec_v1.json
```

Implementation/API:

```text
tools/ab_window_codec.py
```

Primary functions/classes:

```text
ABWindowCodec.encode_window_a128(window)
ABWindowCodec.encode_a128_to_b64(code128)
ABWindowCodec.decode_b64_to_a128(latent64)
ABWindowCodec.decode_a128_to_window(code128)
ABWindowCodec.encode_window_b64(window)
ABWindowCodec.decode_b64_to_window(latent64)
```

## Acceptance Gate

Acceptance command:

```powershell
python tools\ab_window_codec.py --self-test --verify-artifact tools\ab_window_codec_v1.json --eval-windows 65536 --out output\phase_d23_ab_window_codec_v1\acceptance
```

Required result:

```text
AB_WINDOW_CODEC_V1_PASS
window_exact_acc: 1.0
byte_exact_acc: 1.0
bit_acc: 1.0
b_collision_count: 0
byte_margin_min > 0
artifact checksum verifies
```

## Current Result

```text
AB_WINDOW_CODEC_V1_PASS
eval_windows: 65536
window_exact_acc: 1.0
byte_exact_acc: 1.0
bit_acc: 1.0
byte_margin_min: +2.0
b_collision_count: 0
b_encoder_weight_count: 64
```

## Next Step

D24 should operate over the 64D B latent:

```text
input:  8-byte window encoded as B64
task:   tiny reversible transformations
output: decoded 8-byte window
```

Recommended first tasks:

```text
copy window
reverse window
rotate-left / rotate-right
marker-copy within the 8-byte window
```

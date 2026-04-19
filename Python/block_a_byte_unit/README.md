# Block A — byte_unit

Zero-framework Python SDK for the VRAXION Block A champion encoder: **binary + C19 activation + H=16 hidden**, 100% lossless on all 256 bytes.

## What it does

Encodes any byte (0–255) into a 16-dimensional float32 latent vector via a baked int8 LUT, and decodes back via a tied-mirror binary weight network — pure numpy, no ML framework required.

## Usage

```python
from Python.block_a_byte_unit import ByteEncoder

enc = ByteEncoder.load_default()
vec = enc.encode(0x41)          # (16,) float32 latent
byte = enc.decode(vec)          # 65 == ord('A')

data = b"hello world"
latents = enc.encode_bytes(data)   # (11, 16) float32
back = enc.decode_bytes(latents)   # b"hello world"
```

## Champion artifacts

Frozen LUT and weights live in:
[`output/byte_unit_champion_binary_c19_h16/`](https://github.com/kenessy-dani/VRAXION/tree/main/output/byte_unit_champion_binary_c19_h16)

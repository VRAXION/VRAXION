# Block B — L1 Byte-Pair Merger (Python Deploy SDK)

Single-matrix mirror-tied autoencoder that compresses two Block A byte-unit
latents (32 dims total) and recovers them losslessly (sign-match criterion on
all 65,536 possible byte pairs).

## Architecture

```
y = C19(x @ W + b1) @ W.T + b2
```

- `W` (32 x 81): single mirror-tied weight matrix (encoder and decoder share it)
- `b1` (81,), `b2` (32,): biases
- `C19`: per-channel piecewise-polynomial activation with learned `c` (~1.0) and `rho` (~8.0) per hidden unit
- Parameters are embedded in the packed artifact; no external config needed.

## Usage

```python
from Python.block_b_merger import L1Merger
import numpy as np

m = L1Merger.load_default()       # loads output/merger_single_w_huffman_pack/packed_model.bin

x = np.concatenate([lut_a, lut_b])  # two 16-dim Block A latents → 32-dim input
y = m.forward(x)                  # (32,) float32, lossless in sign-match sense

matches, total = m.verify_lossless()   # uses Block A LUT as input source
# expected: (65536, 65536)
```

## Packed artifact

Weights live at [`output/merger_single_w_huffman_pack/`](../../../../output/merger_single_w_huffman_pack/)

- `packed_model.bin` — 3,440 B VGH1-format Huffman-packed weights
- `summary.json` — metadata (100% lossless, 0 bad pairs)

## Running tests

```bash
python -m pytest Python/block_b_merger/tests/ -q
```

All 9 tests should pass, including the full 65,536-pair lossless assertion.

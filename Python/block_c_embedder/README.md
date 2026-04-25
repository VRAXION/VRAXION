# Block C — byte-pair embedder (Python deploy SDK)

Pure-numpy loader for the VRAXION Block C byte-pair embedding champion.

## What it does

Given a byte-pair ID in `[0, 65535]` (or a raw byte stream which it splits
into byte pairs), returns a 32-dim float32 semantic embedding vector.

The emb table was trained on 100 MB FineWeb-EDU via full-softmax next-pair
cross-entropy (training scripts archived to tag
`archives/tools-cleanup-20260425`: `diag_block_c_bytepair_poc.py` +
Modal wrapper `modal_block_c.py`). Mean acc@1 across 3 seeds: 34.06 ± 0.82%.

## Deploy artifact

Packed champion lives at `output/block_c_bytepair_champion/packed.bin`:

- **Size**: 62,528 B (61 KB) — **134× compression** vs fp32 (8.39 MB)
- **Format**: `VCBP` v1. Per-channel int4 for the hot 3,386 pairs
  (`freq ≥ 5` in training corpus), shared OOV vector for the 62,150 cold
  pairs. Bake script `bake_block_c_bytepair.py` archived to tag
  `archives/tools-cleanup-20260425`.

## Usage

```python
from block_c_embedder import L2Embedder
import numpy as np

emb = L2Embedder.load_default()
print(emb)
# L2Embedder(E=32, n_hot=3,386, vocab=65,536, scheme='cold_shared',
#            packed=62,528B)

# Single byte-pair ID
v = emb.embed_id(0x7468)             # 'th'
assert v.shape == (32,)

# Raw 2-byte sequence
v = emb.embed_pair(b'th')

# Full byte stream (grouped 2-at-a-time)
seq = emb.encode_bytes(b"Hello world!")
assert seq.shape == (6, 32)          # 12 bytes / 2 = 6 pairs
```

## Learned clusters

Emergent from the training objective alone (no labels):

- **Word-start equivalence**: `' t'` ≈ `'\nt'` ≈ `'(t'` ≈ `'-t'`
- **Case invariance** for common word starts: `th`↔`Th`, `he`↔`He`,
  `in`↔`In`, `on`↔`On`, `an`↔`An`
- **Sentence terminators**: `. ` ≈ `! ` ≈ `? ` ≈ `.\n` ≈ `!\n`
- **Clause punctuation**: `, ` ≈ `; ` ≈ `: `
- **Function words** cluster: `in`/`by`/`on`

These are the syntactic/positional regularities the model found while
minimising next-byte-pair CE on 100 MB of web text.

## ABC pipeline test

Integration test `tests/test_chain_a_b_c.py` exercises all three deploy
blocks end-to-end:

```bash
pytest -xvs Python/block_c_embedder/tests/test_chain_a_b_c.py
```

Covers: A byte-level lossless, B sign-match lossless, C uniqueness on
hot rows, OOV sharing on cold rows, determinism, cluster preservation,
100 KB real-text corpus encode, edge cases (empty / single / odd /
binary data).

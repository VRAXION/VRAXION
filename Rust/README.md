# VRAXION Rust Deploy SDK

Clean, minimal Rust crate exposing the **frozen byte-level pipeline** as a simple API.

## What goes here

Only the **deploy-ready winning components** — no research code, no experimental examples. Currently planned:

| component | source artifact | status |
|---|---|---|
| L0 Byte Encoder | `tools/byte_embedder_lut.h` (256-entry LUT, 4.1 KB) | frozen, pending port |
| L1 Byte-Pair Merger | `output/merger_single_w_huffman_pack/packed_model.bin` (3.36 KB Huffman-packed) | frozen, pending port |
| Word Tokenizer V2 | `output/word_tokenizer_champion/champion_vocab.json` (32,294 slots, 4.24 MB) | frozen, pending port |
| Embedder V1 | `output/word_embedding_v1/` | scaffold — awaits training |
| Nano Brain V1 | `output/nano_brain_v1/` | scaffold — awaits training |

## What does NOT go here

- `instnct-core/` stays separate — it is the **Rust research mainline** (bias-free threshold grower, scout oracle, quantization championship, etc.), not the byte-level deploy SDK.
- Experimental Rust examples — those stay in `instnct-core/examples/`.

## Status

**Under construction.** First port (byte encoder) is the next step. Until then, this folder is a placeholder for the deployment target.

The sibling [`Python/`](../Python/) folder holds the identical pipeline in Python; both SDKs are backed by the same champion artifacts in `output/`.

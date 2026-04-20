# VRAXION Python Deploy SDK

Clean, minimal Python library exposing the **frozen byte-level pipeline** as a simple API.

## What goes here

Only the **deploy-ready winning components** — no research code, no diagnostic scripts. Currently planned:

| component | source artifact | status |
|---|---|---|
| L0 Byte Encoder | `tools/byte_embedder_lut.h` (256-entry LUT, 4.1 KB) | frozen, pending port |
| L1 Byte-Pair Merger | `output/merger_single_w_huffman_pack/packed_model.bin` (3.36 KB Huffman-packed) | frozen, pending port |
| Word Tokenizer V2 | `output/word_tokenizer_champion/champion_vocab.json` (32,294 slots, 4.24 MB) | frozen, pending port |
| Embedder V1 | *(training artifacts not yet produced)* | scaffold — awaits training |
| Nano Brain V1 | *(training artifacts not yet produced)* | scaffold — awaits training |

## What does NOT go here

- Research / diagnostic scripts — those stay in `tools/`
- Python experimental research lane — frozen at tag `archives/python-research-20260420` (was `instnct/`, archived 2026-04-20)
- Unfinished scaffolds or exploratory ports

## Status

**Under construction.** First port (byte encoder) is the next step. Until then, this folder is a placeholder for the deployment target.

The sibling [`Rust/`](../Rust/) folder will hold the identical pipeline in Rust; both SDKs are backed by the same champion artifacts in `output/`.

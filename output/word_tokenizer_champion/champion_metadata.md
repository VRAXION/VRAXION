# Word Tokenizer Champion — v2-hybrid-wr0.9375

## Spec
- Family: space-aware hybrid (whole-word + subword + byte-fallback)
- Vocab size: 32,000
- Whole ratio: 0.9375
- Training corpus: FineWeb-EDU (first 100 MB, 20 row-groups)
- Max subword len: 12
- Min subword count: 50

## Layout
- BYTE  base: 0..255
- PUNCT base: 256..286
- WS_RUN base: 287..293 (runs of 2..8 spaces)
- LEARNED base: 294..

## Verdict (10 MB FineWeb-EDU)
- Lossless round-trip: **YES**
- Encode throughput: 0.26 MB/s (DP segmentation cost)
- Huffman compression: **30.43%** of raw
- Shannon floor: 30.34% (Huffman is 0.09pp above)
- Fixed-width 15bpt: 42.84%
- LEARNED coverage: 95.90% of input bytes
- BYTE fallback: 1.26% of input bytes
- Edge cases: 9/9

## Baselines on same 10 MB (for context)
- gzip-9: 37.62%
- bzip2-9: 29.97%  ← champion Huffman is 0.46pp above this
- lzma-preset-9e: 28.61%

## Reproduce
```
python tools/diag_word_tokenizer_champion_freeze.py
```
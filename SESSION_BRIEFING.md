# VRAXION Session Briefing — 2026-04-16

## For the next Claude session: read this first!

### Current Pipeline Status
```
L0: Character Embedding — 16-dim int8 lookup, 100% lossless, LOCKED
L1: Beukers Conv — xy/(1+|xy|) gate, k=7, novel discovery
L2: Brain — placeholder tested (+1.4%), real INSTNCT not built yet
```

### The Beukers Gate (KEY DISCOVERY)
```rust
fn beukers(x: f32, y: f32) -> f32 {
    let p = x * y;
    p / (1.0 + p.abs())
}
```
- Two parallel conv projections, multiply their outputs
- Inspired by Beukers integral ∫∫1/(1-xy)dxdy = ζ(2)
- Beats swish by +6.2% on masked char prediction (83.6% vs 77.4%)
- Same principle as SwiGLU (GPT-4/LLaMA) but simpler
- Optimal config: ctx=32, k=7, nf=128, D=16 embedding

### Record Progression
```
swish k=5 nf=64:     77.4% (starting point)
Beukers k=5 nf=96:   80.1% (+2.7%)
Beukers k=7 nf=96:   82.1% (+4.7%)
Beukers k=7 nf=128:  83.6% (+6.2%) ← CURRENT RECORD (2-proj)
```

### NEW: Full Arithmetic Neuron (toy results, needs full-scale test!)
```
f(a,b,c) = a × b / (|c| + ε)    ← multiply + divide, 3 projections
Toy nf=32: 79.1% (vs Beukers 2-proj 75.2%, vs swish 71.1%)
UNTESTED at full scale (nf=128 k=7) — tonight's priority!
```

### Operations Tested
- Multiply (Beukers ab): STRONGEST single op
- Divide (a/|c|): strongest COMPLEMENT to multiply (+2.9%)
- Exp gate (exp(-|ab|)): decent regularizer (+0.7%)
- Sqrt, log, power, square: all WEAKER, don't help
- Per-neuron gain (α): no effect (absorbs into weights)
- 3-way/4-way multiply: helps but width > ways
- SIMPLEST formula wins: ab/(|c|+ε), no fancy math needed

### What Was Tested (21 activations, 30+ configs)
- **Winners:** Beukers gate (all sizes), multi_c19
- **OK:** swish, C19 learnable, damped_c19, Padé, GCU, bessel
- **Bad:** eta, sinc, snake
- **Novel findings:** 
  - 3-way/4-way Beukers (more projections) helps but width > ways
  - Per-neuron gain (α) = no effect (absorbs into weights)
  - Deep 2-layer Beukers < single wide layer
  - Learnable gate params hurt on small corpus

### Universal Vocab Spec
- ~2000 base chars (Latin, Cyrillic, Greek, Arabic, Hebrew, Thai, Hangul Jamo)
- ~100 cutter/delimiter tokens  
- ~126K BPE subword merges = 128K total vocab
- D=16 embedding dimension (research standard for char-level)

### TONIGHT'S GOAL: FineWeb corpus test
- User has FineWeb parquet files (multi-GB English text)
- Test: does Beukers advantage hold at scale?
- Also: INSTNCT brain on Beukers features
- Rayon parallelism for speedup (already in Cargo.toml)

### Key Architecture Decisions (LOCKED)
- Character embedding: D=16 int8, lookup table
- Conv activation: Beukers gate (2-input multiplicative)
- Kernel size: k=7 (14 chars ≈ 2-3 words receptive field)
- Training: backprop + masked char prediction → freeze

### User Preferences
- Hungarian speaker, fast iterative experimentation
- ALWAYS log convergence curves (loss, train%, test% per epoch)
- Never run training without telemetry
- Prefers quick toy tests before full runs
- Commit and push results frequently

### Repo Structure
- `instnct-core/examples/` — all experiments (.rs files)
- `instnct-core/examples/canonical_byte_encoder.rs` — L0 LOCKED
- `instnct-core/examples/canonical_byte_merger.rs` — L1 LOCKED  
- `instnct-core/examples/char_embed_beukers_full.rs` — best embedding
- `docs/wiki/` — wiki pages
- `instnct-core/tests/fixtures/alice_corpus.txt` — 100K test corpus

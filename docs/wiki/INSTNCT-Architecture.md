# INSTNCT Architecture

<p align="center">
  <img src="https://raw.githubusercontent.com/VRAXION/VRAXION/main/docs/assets/vraxion-instnct-spiral.png" alt="INSTNCT spiral logo" width="320">
</p>

**Architecture line:** `INSTNCT` /ˈɪnstɪŋkt/ ("instinct")

Vraxion is building **INSTNCT**: a gradient-free self-wiring architecture that changes its own graph instead of learning inside a fixed layer stack with backpropagation.

> **Core thesis**
>
> **Inference emerges as the fixed point of destructive interference.**
>
> Signal enters a recurrent substrate, competing paths suppress one another through destructive interference, and the surviving pattern is read out as inference.

This page explains the current architecture line: the shipped structure, the strongest validated alternative, and the boundary between fixed interfaces and learnable internals. It is not the theory page, and it is not the chronology or proof trail. For the public website summary, see [Website: INSTNCT](https://vraxion.github.io/VRAXION/instnct/).

For theory, see [Theory of Thought](Theory-of-Thought). For chronology, reversals, and research protocol, see [Research Process & Archive](Timeline-Archive).

## At a Glance

- The shipped line uses fixed projection-style I/O around a sparse signed recurrent graph.
- The main learnable object is the hidden graph plus compact per-neuron controls such as `theta`, `channel`, and `polarity`.
- Runtime charge/state persists across ticks; INSTNCT does not reduce computation to a one-pass feedforward view.
- Tentacle I/O is the strongest validated alternative, but it is not the shipped default.
- Shipped, validated, and experimental distinctions are kept separate on purpose.

## What Makes INSTNCT Different

Most neural systems learn by adjusting many weights inside a fixed topology. INSTNCT changes the learnable object itself: the hidden graph can be rewired, compact neuron roles can change, and recurrent state persists across ticks.

Current mainline in one sentence:

> fixed I/O projections + sparse signed recurrent graph + persistent state + mutation-selection training

In practice that means:

- input enters through a fixed boundary interface
- signal propagates through a recurrent directed graph instead of a fixed feedforward stack
- neurons keep charge/state across ticks
- topology and compact neuron controls are mutated and selected
- inference is read out from the surviving state pattern rather than from a conventional layer cascade

## Shipped Line vs Validated Alternative

The main architectural fork right now is not between many equal options. It is between one shipped line and one validated alternative that has earned serious attention.

**Shipped current mainline (projection-style I/O):**

```text
input -> input_projection -> hidden signed graph -> output_projection -> output
              persistent charge/state across ticks
```

**Strongest validated alternative (tentacle I/O):**

```text
input -> first V neurons -> hidden graph -> last V neurons -> output
         (1:1 inject)       (mask edges     (charge = logits)
                             learn routing)
```

| Aspect | Shipped line | Validated alternative |
|---|---|---|
| Input / output handling | Fixed projection-style interfaces at the boundary | Boundary neurons act directly as input and output surfaces |
| What stays fixed | I/O interface shape stays fixed | Boundary placement stays fixed, but routing near the boundary becomes more direct |
| What is learnable | Hidden graph plus compact neuron controls | Hidden graph plus routing behavior closer to the I/O boundary |
| Readout surface | Output projection interprets recurrent state | Terminal neuron charge acts as the readout surface |
| Current status | Canonical line on `main` | Strongest validated alternative, not the shipped default |

Tentacle I/O matters because it moves more of the routing burden into the learnable mask. It is important precisely because it is a serious non-default alternative, not because it has already replaced the shipped line.

## What Is Fixed vs Learnable

| Component | Status on shipped line | Notes |
|---|---|---|
| Input interface | Fixed | Projection-style input boundary is the default; tentacle/SDR-heavy variants are not shipped defaults |
| Output interface | Fixed | Projection-style output boundary is the default; alternative readout surfaces remain non-default |
| Hidden graph / mask | Learnable | This is the core object that evolution changes |
| `theta` | Learnable | Per-neuron selectivity / firing threshold |
| `channel` | Learnable | Per-neuron timing lane / temporal preference |
| `polarity` | Learnable | Excitatory vs inhibitory sign |
| Timing / decay form | Compact shipped runtime form | Broader variants exist historically, but the public line uses a tighter default form |
| Charge / state | Runtime state, persistent across ticks | Not a fixed parameter; it evolves while the system runs |

The architecture rule of thumb is simple:

> keep the recurrent substrate learnable, keep the public boundary explicit, and keep the default line distinct from validated side branches

## Why This Architecture Matters

INSTNCT matters because it treats topology, persistent state, and timing as first-class computational ingredients instead of assuming that a fixed layer stack and smooth gradient flow are the only serious way to build learning systems.

Sparse recurrent routing matters because incompatible paths can cancel while compatible paths survive. Persistent state matters because computation is allowed to unfold across ticks instead of being collapsed into a single pass. Mutation-selection matters because the system can search over structure, not just tune weights inside a frozen container.

This page also separates the shipped line from validated alternatives on purpose. The project benefits from serious side branches, but understanding the architecture requires knowing which line is actually default and which lines are still candidates.

## Feature Pipeline: Current Byte-Level Implementation

The INSTNCT brain runs on top of a two-layer byte-level feature pipeline. Both layers are locked and ship as deploy artifacts; the brain sits above them and is the active research frontier.

### L0 — Byte Unit (LOCKED, shipped)

A single byte enters as 8 bits and is mapped to a 16-dim embedding by a C19 tied mirror autoencoder (`8 → 24 → 16`). Int4 precision. 100% lossless on all 256 bytes. The 256-entry LUT (`tools/byte_embedder_lut.h`, 4.1 KB) is the deploy artifact; the weight file is `tools/byte_unit_winner_int4.json`.

This replaced the earlier binary byte encoder (flat 8→4 neurons, binary {-1,+1}, 36 bits, POPCOUNT) which remains a validated finding for pure-integer paths, and the character-level 16-dim lookup table from the abstract-core track — both are preserved as research history, not current defaults.

### L1 — Byte-Pair Merger (CHAMPION, shipped)

Two L0 embeddings (32-dim total) pass through a single-W mirror-tied autoencoder (`C19(x @ W + b1) @ W.T + b2`, one 32×81 matrix = 2592 cells). Output: 32-dim merged representation. 100% lossless on all 65,536 byte pairs. Deploy champion: **3440 B (3.36 KB) Huffman-packed** (`output/merger_single_w_huffman_pack/packed_model.bin`, commit `f0ab75a`). Shannon floor is 2422 B (~42% gap remains).

### Earlier Track — Character-Level Abstract-Core (archived 2026-04-18)

The prior active track used a 16-dim character LUT as L0 and a Beukers-gate Conv1D (`xy/(1+|xy|)`, k=7, nf=128) as L1, reaching **83.6% masked character prediction**. This is a validated, positive result and is preserved in the [Timeline Archive](Timeline-Archive) (2026-04-15/16 section). It is not the current pipeline; the byte-level track superseded it as of the 2026-04-18 track transition.

## Experimental AB-C-D Component Stack

The current component-level research stack is intentionally simpler than a full
language model. The names are now:

```text
A-block:
  byte codec
  1 byte <-> 16D byte abstract

B-block:
  window codec / common bus
  N x A outputs <-> B latent
  current default: 8 x 16D = 128D <-> B64

C-block:
  stream tokenizer / span embedder / controller
  B64 window stream -> token events + route hints

D-block:
  selected workers
  ALU / memory / transform / language / reject policy
```

The cleaned AB artifact is:

```text
8 bytes <-> A128 <-> B64 <-> A128 <-> 8 bytes
```

D28 proved a **C0 route-head**:

```text
B64 -> LANG / ALU / MEM / TRANSFORM / UNKNOWN
```

D29 proved route-selected execution and empty inactive lanes. D30A/D30B split
ALU into compact removable op-lanes:

```text
ADD / SUB / MUL / AND / OR / XOR
```

with `MUL` now implemented as a compact low-8-bit partial-product lane instead
of a 65,536-entry table.

The next real C-block is not another compression layer. It is the stream layer
that turns overlapping B64 windows into spans:

```text
"Give me apples, i need EXACTLY 25 times 7..."
  -> NUMBER(25)
  -> OP_MUL
  -> NUMBER(7)
  -> ROUTE(ALU)
```

Until that C-block exists, the system can route and execute short command-shaped
windows, but it does not yet parse arbitrary real text streams.

D31A added the first C-block tokenizer probe:

```text
B64 window stream -> TokenEvent stream + C64 token embeddings + route hints
```

It passed on generated word, punctuation, number, operator, and boundary-stress
examples:

```text
token stream exact:   100%
boundary exact:       100%
kind exact:           100%
normalized exact:     100%
ALU call exact:       100%
```

Example:

```text
"Give me exactly 25 times 7."
  -> WORD:GIVE
  -> WORD:ME
  -> WORD:EXACTLY
  -> NUMBER:25
  -> OP:OP_MUL
  -> NUMBER:7
  -> PUNCT:.
  -> ALU output 175
```

This is still a tokenizer/controller probe, not a language worker.

## Read Next

- [Vraxion Home](Home) — public front door and mission-level summary
- [Theory of Thought](Theory-of-Thought) — theoretical framing behind destructive interference and fixed-point inference
- [Research Process & Archive](Timeline-Archive) — protocol, chronology, reversals, and proof trail
- [Rust Implementation Surface](v5-Rust-Port-Benchmarks) — Rust implementation lane, validation checkpoints, and archived experiments

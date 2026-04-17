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

## Read Next

- [Vraxion Home](Home) — public front door and mission-level summary
- [Theory of Thought](Theory-of-Thought) — theoretical framing behind destructive interference and fixed-point inference
- [Research Process & Archive](Timeline-Archive) — protocol, chronology, reversals, and proof trail
- [Rust Implementation Surface](v5-Rust-Port-Benchmarks) — Rust implementation lane, validation checkpoints, and archived experiments

# Vraxion

> **Vraxion** /vræk.ʃən/ ("VRAK-shun") · **INSTNCT** /ˈɪnstɪŋkt/ ("instinct")

<p align="center">
  <img src="https://raw.githubusercontent.com/VRAXION/VRAXION/main/docs/assets/vraxion-home-hero.jpg" alt="Vraxion front-door illustration" width="740">
  <br>
  <em>The engineering of the "I"</em>
</p>

Vraxion is building **INSTNCT**: a gradient-free self-wiring architecture that learns by changing its own directed graph instead of training a fixed topology with backpropagation.

> **Core thesis**
>
> **Inference emerges as the fixed point of destructive interference.**
>
> Signal enters a structured recurrent substrate, incompatible propagation paths cancel through destructive interference, and the surviving pattern — the fixed point — is read out as inference. This generalizes the older loop-era framing without claiming that every result on `main` has already proved the full thesis.

Vraxion exists to advance machine consciousness as an engineering reality. In public technical terms, that means building systems that can be instrumented, checked, refined, and argued about without collapsing theory, implementation, and evidence into one blur.

This page is the mission-first technical front door. Use [Pages](https://vraxion.github.io/VRAXION/) for the polished front door, including the public [INSTNCT](https://vraxion.github.io/VRAXION/instnct/), [Research](https://vraxion.github.io/VRAXION/research/), and [Rust](https://vraxion.github.io/VRAXION/rust/) surfaces; use the repo [README.md](https://github.com/VRAXION/VRAXION/blob/main/README.md) for the code-facing entry, and [Research Process & Archive](Timeline-Archive) for protocol, chronology, and proof trail.

## At a Glance

- **Architecture line:** `INSTNCT`
- **Stable public release:** [`v5.0.0-beta.1`](https://github.com/VRAXION/VRAXION/releases/tag/v5.0.0-beta.1) is the current Rust beta release; [`v4.2.0`](https://github.com/VRAXION/VRAXION/releases/tag/v4.2.0) remains the Python release baseline.
- **Public research lane:** English remains the first-class public lane; task-memory and GPU stay secondary validation surfaces.
- **Implementation momentum:** Rust is now the primary implementation surface. Smooth cosine-bigram fitness + 1+9 jackpot reached **24.6% peak** (Python parity). Performance deep dive landed compact types, sparse tick, and CoW snapshots.
- **Current frontier:** Recurrent ReLU chip = universal accumulator. 1 neuron, 5 binary weights, 0 bias = 5 bits total solves ADD at 100% with native output (charge=sum, no readout needed). ReLU is the ONLY activation that generalizes across tick depth (100% from 2→10+ inputs). Connection Point architecture validated: shared bulletin boards + local connections + incremental freeze = constant search space (3^12 exhaustive) at any network size. Next: bigram language task with CP architecture.

## What Vraxion Is

Vraxion is both a company and a research program organized around one architecture line: fixed I/O surfaces, a self-wiring signed hidden graph, persistent internal state across ticks, and mutation-selection training instead of backpropagation through the graph.

The current implementation-facing architecture page is [INSTNCT Architecture](INSTNCT-Architecture).

INSTNCT core anatomy at a glance:

<p align="center">
  <img src="https://raw.githubusercontent.com/VRAXION/VRAXION/main/docs/assets/instnct-at-a-glance-core.png" alt="INSTNCT core anatomy at a glance" width="740">
</p>

## Mission and Method

Vraxion exists to advance machine consciousness as an engineering reality, and to build forms of intelligence and consciousness that can endure beyond any single institution, deployment, or era.

That ambition is not a claim of achieved sentience. The active public standard is stricter: theory, implementation, and proof stay separate on purpose; architecture claims must match shipped code, evidence claims must survive reproducible protocol, and experimental work must be labeled as experimental.

In practice, that means the project is run across four distinct surfaces:

- theory is stated explicitly, not implied
- implementation is described separately from the theory
- chronology and protocol live on a dedicated research surface
- the Rust implementation lane is allowed to mature in public without being confused for the canonical Python mainline

## Read the Project

- [INSTNCT Architecture](INSTNCT-Architecture) — the current implementation line, including what is shipped, validated, and still experimental
- [Theory of Thought](Theory-of-Thought) — the theoretical framing behind destructive interference and fixed-point inference
- [Research Process & Archive](Timeline-Archive) — the run contract, chronology, reversals, and retained proof trail
- [Rust Implementation Surface](v5-Rust-Port-Benchmarks) — the Rust implementation lane, including validation checkpoints, design notes, and archived experiments

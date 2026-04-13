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
- **Stable public release:** [`v5.0.0-beta.2`](https://github.com/VRAXION/VRAXION/releases/tag/v5.0.0-beta.2) is the current Rust public beta tag (grower-based, with the `neuron_infer` standalone CLI and the public beta training runbook); [`v5.0.0-beta.1`](https://github.com/VRAXION/VRAXION/releases/tag/v5.0.0-beta.1) remains the prior Rust language-evolution beta as a historical reference, and the legacy `v4.x` Python lane is preserved for context only.
- **Public research lane:** English remains the first-class public lane; task-memory and GPU stay secondary validation surfaces.
- **Implementation momentum:** Rust is the primary implementation surface, and the current mainline path on `main` is the bias-free threshold grower (`instnct-core/examples/neuron_grower.rs`). The released language-evolution beta still records its **24.6% peak** smooth-fitness + 1+9 jackpot result as the historical Rust language line, but the active grower lane has moved past it: see [`docs/PUBLIC_BETA_TRAINING.md`](https://github.com/VRAXION/VRAXION/blob/main/docs/PUBLIC_BETA_TRAINING.md) for the canonical training runbook and [`docs/GROWER_RUN_CONTRACT.md`](https://github.com/VRAXION/VRAXION/blob/main/docs/GROWER_RUN_CONTRACT.md) for the frozen B0 engine contract.
- **Current frontier:** the bias-free grower has frozen the B0 engine on a six-task regression matrix (mean val `88.417`, max test `100.0`, mean neurons `5.667`, max depth `6` per the golden fixture in `instnct-core/tests/fixtures/grower_regression_golden.json`), and the active promotion gate is **byte/opcode v1**: `1 byte data + 4 opcode -> 1 byte` over a frozen 8-bit-head latent + exact LUT translator. The B1 golden fixture records the direct bitbank negative control at `75.0%` and the exact translator at `100.0%` over the full `1024`-sample domain (`distinct_keys=1024`, `conflicting_keys=0`, `key_bits=61`, `total_neurons=61`). The **abstract-core preprocessor concept** is under active exploration: a tied-weight MLP autoencoder achieves validated 100% lossless byte round-trip encoding via int8 quantization at H=12 (219 bytes total), with integration into the INSTNCT core as a replacement for hand-crafted SDR input as the next step. Earlier spiking-network and Connection Point lines are preserved as research history, not current mainline.

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
- the Rust grower lane is the canonical public implementation surface, and the Python `graph.py` lane is preserved as a historical reference rather than the active default

## Read the Project

- [INSTNCT Architecture](INSTNCT-Architecture) — the current implementation line, including what is shipped, validated, and still experimental
- [Theory of Thought](Theory-of-Thought) — the theoretical framing behind destructive interference and fixed-point inference
- [Research Process & Archive](Timeline-Archive) — the run contract, chronology, reversals, and retained proof trail
- [Rust Implementation Surface](v5-Rust-Port-Benchmarks) — the Rust implementation lane, including validation checkpoints, design notes, and archived experiments

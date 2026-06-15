# VRAXION

VRAXION is building **INSTNCT**: a Rust-first, gradient-free architecture whose active object is a governed computational substrate, not a fixed backprop-trained layer stack.

This repository is consolidated around the current winning mainline. Historical beta, bounded-service, byte-pipeline, and probe-era work remains available for auditability, but it is not the active public surface unless promoted back into `main`.

## Current Source Of Truth

```text
branch = main
current_release = v6.1.7
current_evidence_anchor = f32a6f4b
current_evidence_subject = Finalize E127 cycle 40 checkpoint
latest_released_runtime_slice = a908a838a1119540ed88bc91e10cfcb0bdae92a8
latest_released_runtime_subject = Add training data curriculum readiness gate
```

The latest GitHub release is [`v6.1.7`](https://github.com/VRAXION/VRAXION/releases/tag/v6.1.7). It anchors the E127 cycle-40 governed text-operator library checkpoint.

## Current Mainline

| Slice | Commit | Purpose |
|---|---|---|
| E69-E79 | `a908a838` release line | Rust Pocket Library, curriculum, final-train supervision, global merge, and training-data/curriculum readiness gate |
| E80-E85 | `56a9cf03` | CALC-SCRIBE visible calculation-trace validation and mixed-stream no-call routing |
| E86-E89 | `a6935e61` | LocalGolden seeded curriculum, sparse active-set selection, survival gauntlet, and Operator naming/schema lock |
| E90-E106 | `b75c64cb` | Text-evidence, temporal, agency, route, memory, compression, and task-progress Operator curriculum expansions |
| E107 | `1fcdf954` | E90-E106 survival role and regression gauntlet |
| E108 | `0389c211` | External dataset transfer and negative-scope no-harm gauntlet |
| E109 | `555c5006` | Operator rank ladder and GoldenWatch probation policy |
| E110 | `b378c2c5` | Silver-to-Gold scoped probation wave: 35/35 promoted, 0 hard negatives |
| E111 | `d71e3657` | Bronze mutation/prune wave: 87/87 promoted to scoped Gold variants, 0 hard negatives |
| E112 | `9de33241` | Gold-to-CoreMemoryCandidate prune-heavy probation wave: 136/136 qualified, 0 hard negatives |
| E113 | `05415f5b` | FineWeb-Edu 100k light stress: baseline 2,624 hard negatives across 88 operators, selected recycled variants 0 hard negatives |
| E119-E126 | tracked on `main` | FineWeb/text-understanding skill mining and Orange/Legendary probation |
| E127 | `f32a6f4b` | Overnight cyclic Orange/Legendary text-operator farm: 40 cycles, 382 scoped operators, 0 hard negatives |

Current E127 scoped operator state:

```text
Orange/Legendary scoped operators = 382
E127 cycles = 40
hard negatives = 0
false commits = 0
wrong scope calls = 0
unsupported answers = 0
```

## What Is Current

- Active Rust runtime: [`vraxion-runtime/`](vraxion-runtime/)
- Current status: [`docs/CURRENT_STATUS.md`](docs/CURRENT_STATUS.md)
- Getting started: [`docs/GETTING_STARTED.md`](docs/GETTING_STARTED.md)
- Validated findings: [`VALIDATED_FINDINGS.md`](VALIDATED_FINDINGS.md)
- Operator cards: [`docs/research/OPERATOR_LIBRARY_CARDS.md`](docs/research/OPERATOR_LIBRARY_CARDS.md)
- Current result: [`docs/research/E127_OVERNIGHT_TEXT_SKILL_FARM_ORANGE_CYCLE_RESULT.md`](docs/research/E127_OVERNIGHT_TEXT_SKILL_FARM_ORANGE_CYCLE_RESULT.md)
- Handover for fresh Codex sessions: [`CODEX_HANDOVER.md`](CODEX_HANDOVER.md)
- GitHub Pages front door: <https://vraxion.github.io/VRAXION/>
- Wiki timeline: <https://github.com/VRAXION/VRAXION/wiki/Timeline-Archive>

## Claim Boundary

Allowed current claim:

> VRAXION v6 has a Rust mainline for governed Pocket Library state, resumable curriculum execution, multi-lane final-training supervision, global Pocket Library merge/dedupe governance, a training-data/curriculum readiness gate, one canonical `final_train` campaign entrypoint, and governed Operator evidence through E127. E127 cycle 40 contains 382 scoped Orange/Legendary text operators with 0 tracked hard negatives, false commits, wrong-scope calls, or unsupported answers in the checkpointed evidence.

The E127 finding is scoped operator-library evidence only. It includes a deterministic operator+template text-to-text smoke, but it does not claim PermaCore, TrueGolden, production API readiness, final training completion, open-domain assistant readiness, Gemma/GPT-like generation, GSM8K solving, consciousness, or sentience.

## Verification

```powershell
cargo fmt --check -p vraxion-runtime
cargo clippy -p vraxion-runtime --all-targets -- -D warnings
cargo test --workspace
python -m compileall -q scripts
```

Evidence checkers live under [`scripts/probes/`](scripts/probes/). Current front-door CI also checks tracked JSON/JSONL syntax, E89 naming/schema, E107-E112 sample artifacts, and the Operator rank dashboard smoke path. E113 has a full-artifact checker for local FineWeb stress runs.

Long or expensive runs must write partial outcomes continuously. End-only reporting is not acceptable for the current operating model.

## Archive Policy

The active branch surface is `main`. Historical branch heads from the June 13 cleanup are preserved under:

```text
archive/branches/2026-06-13/*
```

See [`ARCHIVE.md`](ARCHIVE.md) and the [Consolidation Archive wiki page](https://github.com/VRAXION/VRAXION/wiki/Consolidation-Archive-2026-06-13).

## License

See [`LICENSE`](LICENSE).

# VRAXION

VRAXION is building **INSTNCT**: a Rust-first, gradient-free architecture whose active object is a governed computational substrate, not a fixed backprop-trained layer stack.

This repository has been consolidated around the current winning mainline. Historical beta, bounded-service, byte-pipeline, and probe-era work is retained for auditability, but it is not the active sales surface unless promoted back into `main`.

## Current Source Of Truth

```text
branch = main
current_release = v5.0.0-e79.0
current_main_head = 56a9cf0305c1bfddd0e9b763b5e0d80fc9ec3bca
current_main_subject = Add E85 calc scribe mixed stream integration
latest_released_runtime_slice = a908a838a1119540ed88bc91e10cfcb0bdae92a8
latest_released_runtime_subject = Add training data curriculum readiness gate
base_runtime_slice = 0879a2c004cf6a002bd5639d9cb7a759709a41aa Extract Rust final bake API
```

The latest GitHub release is still `v5.0.0-e79.0`. The live `main` branch now carries post-release E80-E85 evidence on top of that release line. Treat release status and current-main evidence as related but separate.

## Current Mainline

The active line is E85 current-main evidence on top of the E79 released runtime chain:

| Slice | Commit | Purpose |
|---|---|---|
| E69 | `8b00c352` | Persistent Pocket Library store |
| E70 | `9accc081` | Curriculum runner preflight |
| E71 | `c9dcad01` | Curriculum queue preflight |
| E72 | `fffc5a43` | Curriculum resume preflight |
| E73 | `51cd82a1` | Unified Rust final-bake preflight |
| E74 | `0879a2c0` | Final-bake library API extraction |
| E75 | `3f519732` | Final curriculum pocket-generation runner |
| E76 | `3b44cfe0` | Multi-lane final-training supervisor |
| E77 | `7e91aaaa` | Global Pocket Library merge supervisor |
| E78 | `5f335cec` | Canonical `final_train` campaign entrypoint |
| E79 | `a908a838` | Training data/curriculum readiness gate |
| E80 | `6c4181cf` | Dataset-backed pocket capability scoring and promotion evidence |
| E81 | `b4335206` | CALC-SCRIBE v002 multiseed visible-marker training |
| E82 | `3914a64a` | CALC-SCRIBE v003 floor-division confirmation |
| E83 | `4370bacc` | CALC-SCRIBE v003 LocalGolden promotion/reload evidence |
| E84 | `0a06c153` | CALC-SCRIBE transfer and negative-scope probe |
| E85 | `56a9cf03` | CALC-SCRIBE mixed-stream inference integration |

```text
Pocket Library store
  -> guarded active set
  -> curriculum runner
  -> queued curriculum rows
  -> resumable checkpoint/progress stream
  -> unified final-bake gate
  -> reusable final-bake library API
  -> training data/curriculum readiness gate
  -> deterministic final curriculum lane runner
  -> multi-lane final-training supervisor
  -> global Pocket Library merge/dedupe/challenger gate
  -> canonical final_train campaign entrypoint
  -> dataset-backed scoring evidence
  -> governed CALC-SCRIBE LocalGolden scope
  -> visible calc-trace transfer router
  -> mixed-stream no-call integration
```

The current engineering priority is to keep turning scoped, governed Pocket evidence into a clean inference path without expanding claims beyond the tested surface.

## What Is Current

- Rust runtime surface: [`vraxion-runtime/`](vraxion-runtime/)
- Current GitHub release: [`v5.0.0-e79.0`](https://github.com/VRAXION/VRAXION/releases/tag/v5.0.0-e79.0)
- Current main evidence head: E85 mixed-stream CALC-SCRIBE integration
- Current docs:
  - [`docs/CURRENT_STATUS.md`](docs/CURRENT_STATUS.md)
  - [`docs/GETTING_STARTED.md`](docs/GETTING_STARTED.md)
  - [`VALIDATED_FINDINGS.md`](VALIDATED_FINDINGS.md)
  - [Wiki Timeline Archive](https://github.com/VRAXION/VRAXION/wiki/Timeline-Archive)

## Claim Boundary

Current mainline claims must match code and tracked evidence on `main`.

Allowed current claim:

> VRAXION has a Rust mainline for governed Pocket Library state, resumable curriculum execution, multi-lane final-training supervision, global Pocket Library merge/dedupe governance, a training-data/curriculum readiness gate, one canonical `final_train` campaign entrypoint, and post-release evidence for a governed CALC-SCRIBE scoped Pocket that validates visible calculation traces inside a mixed input stream.

Current E85 extension:

> CALC-SCRIBE v003 is a governed LocalGolden scoped Pocket for visible calculation-trace validation. The E85 mixed-stream integration routes to it only when an explicit visible calc trace exists, rejects invalid visible traces, and no-calls natural text or word-problem text without visible trace framing.

Do not claim from this repo state alone:

- hosted SaaS availability
- public production API readiness
- GPT-like/open-domain assistant readiness
- GSM8K or natural-language word-problem solving
- final production dataset completion
- trained model/weights readiness
- Core memory or True Golden promotion
- safety-aligned production deployment
- consciousness or sentience
- that old beta, bounded-service, or byte-pipeline results are the current sellable model

## Archive Policy

The GitHub branch surface has been reduced to `main`.

Historical branch heads from the June 13 cleanup are preserved under:

```text
archive/branches/2026-06-13/*
```

The wiki pre-cleanup state is preserved under:

```text
archive/wiki/pre-consolidation-2026-06-13
```

The pre-E74 public-surface cleanup repo state is preserved under:

```text
archive/repo/pre-e74-public-surface-cleanup-2026-06-13
```

See [`ARCHIVE.md`](ARCHIVE.md) and the [Consolidation Archive wiki page](https://github.com/VRAXION/VRAXION/wiki/Consolidation-Archive-2026-06-13).

## Verification

```powershell
cargo fmt --check -p vraxion-runtime
cargo clippy -p vraxion-runtime --all-targets -- -D warnings
cargo test -p vraxion-runtime
cargo test --workspace
```

Current final-training entrypoint:

```powershell
cargo run --release -p vraxion-runtime --bin training_data_readiness -- 3 8 target/ci/e79_training_data_readiness_smoke
cargo run --release -p vraxion-runtime --bin final_train -- 3 8 target/ci/e79_final_train_smoke --preflight-rounds 4 --checkpoint-interval 4
```

Current post-release evidence probes live under [`scripts/probes/`](scripts/probes/), including the E85 mixed-stream integration probe and checker.

Long or expensive runs must write partial outcomes continuously. End-only reporting is not acceptable for the current operating model.

## License

See [`LICENSE`](LICENSE).

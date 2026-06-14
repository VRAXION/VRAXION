# VRAXION

VRAXION is building **INSTNCT**: a Rust-first, gradient-free architecture whose active object is a governed computational substrate, not a fixed backprop-trained layer stack.

This repository has been consolidated around the current winning mainline. Historical beta, bounded-service, byte-pipeline, and probe-era work is retained for auditability, but it is not the active sales surface unless promoted back into `main`.

## Current Source Of Truth

```text
branch = main
current_release = v5.0.0-e79.0
runtime_slice = a908a838a1119540ed88bc91e10cfcb0bdae92a8
runtime_subject = Add training data curriculum readiness gate
base_runtime_slice = 0879a2c004cf6a002bd5639d9cb7a759709a41aa Extract Rust final bake API
```

## Current Mainline

The active line is E79 on top of E78, E77, E76, E75, E74, E73, E72, E71, E70, and E69:

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
| E79 | `a908a838` | Training data and curriculum readiness gate |

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
```

The current engineering priority is the next final-training readiness layer: dataset-backed pocket capability scoring, governed global library growth, continuous writeout, resumable runs, and an inference path from loaded pockets through Agency commit and egress rendering.

## What Is Current

- Rust runtime surface: [`vraxion-runtime/`](vraxion-runtime/)
- Current GitHub release: [`v5.0.0-e79.0`](https://github.com/VRAXION/VRAXION/releases/tag/v5.0.0-e79.0)
- Current docs:
  - [`docs/CURRENT_STATUS.md`](docs/CURRENT_STATUS.md)
  - [`docs/GETTING_STARTED.md`](docs/GETTING_STARTED.md)
  - [`VALIDATED_FINDINGS.md`](VALIDATED_FINDINGS.md)
  - [Wiki Timeline Archive](https://github.com/VRAXION/VRAXION/wiki/Timeline-Archive)

## Claim Boundary

Current mainline claims must match code on `main`.

Allowed current claim:

> VRAXION has a Rust mainline for governed Pocket Library state, resumable curriculum execution, multi-lane final-training supervision, global Pocket Library merge/dedupe governance, a training-data/curriculum readiness gate, and one canonical `final_train` campaign entrypoint.

Current E79 extension:

> VRAXION has a deterministic Rust `training_data_readiness` gate and a `final_train` command that blocks before the global supervisor when the dataset/curriculum contract is incomplete, writes progress/manifests/results, validates split/family/capability coverage, and records zero bad commits or unsafe promotions in the E79 evidence run.

Do not claim from this repo state alone:

- hosted SaaS availability
- public production API readiness
- GPT-like/open-domain assistant readiness
- final production dataset completion
- trained model/weights readiness
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

Long or expensive runs must write partial outcomes continuously. End-only reporting is not acceptable for the current operating model.

## License

See [`LICENSE`](LICENSE).

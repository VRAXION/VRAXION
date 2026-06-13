# VRAXION

VRAXION is building **INSTNCT**: a Rust-first, gradient-free architecture whose active object is a governed computational substrate, not a fixed backprop-trained layer stack.

This repository is being consolidated around the current winning mainline. Historical beta, bounded-service, byte-pipeline, and probe-era work is retained for auditability, but it is not the active sales surface unless promoted back into `main`.

## Current Source Of Truth

```text
branch = main
runtime_slice = 0879a2c004cf6a002bd5639d9cb7a759709a41aa
runtime_subject = Extract Rust final bake API
base_runtime_slice = 51cd82a11d8f1d2b98ee3e49538c7c26afdb767b Add Rust final bake preflight
```

## Current Mainline

The active line is E74 on top of E73, E72, E71, E70, and E69:

| Slice | Commit | Purpose |
|---|---|---|
| E69 | `8b00c352` | Persistent Pocket Library store |
| E70 | `9accc081` | Curriculum runner preflight |
| E71 | `c9dcad01` | Curriculum queue preflight |
| E72 | `fffc5a43` | Curriculum resume preflight |
| E73 | `51cd82a1` | Unified Rust final-bake preflight |
| E74 | `0879a2c0` | Final-bake library API extraction |

```text
Pocket Library store
  -> guarded active set
  -> curriculum runner
  -> queued curriculum rows
  -> resumable checkpoint/progress stream
  -> unified final-bake gate
  -> reusable final-bake library API
  -> audited write-back only after successful promotion
```

The current engineering priority is reliability of the model lifecycle: guarded loading, token/artifact/ledger integrity, queued execution, resumable progress, unified final-bake validation, reusable audit APIs, and safe promotion.

## What Is Current

- Rust runtime surface: [`vraxion-runtime/`](vraxion-runtime/)
- Current docs:
  - [`docs/CURRENT_STATUS.md`](docs/CURRENT_STATUS.md)
  - [`docs/GETTING_STARTED.md`](docs/GETTING_STARTED.md)
  - [`VALIDATED_FINDINGS.md`](VALIDATED_FINDINGS.md)
  - [Wiki Timeline Archive](https://github.com/VRAXION/VRAXION/wiki/Timeline-Archive)

## Claim Boundary

Current mainline claims must match code on `main`.

Allowed current claim:

> VRAXION has a Rust mainline for persistent Pocket Library governance and resumable curriculum execution preflights.

Current E74 extension:

> VRAXION exposes the final-bake gate through a reusable Rust library API while preserving the CLI preflight and continuous progress artifacts.

Do not claim from this repo state alone:

- hosted SaaS availability
- public production API readiness
- GPT-like/open-domain assistant readiness
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

Current preflight entrypoints:

```powershell
cargo run --release -p vraxion-runtime --bin pocket_library_preflight -- --help
cargo run --release -p vraxion-runtime --bin curriculum_runner_preflight -- --help
cargo run --release -p vraxion-runtime --bin curriculum_queue_preflight -- --help
cargo run --release -p vraxion-runtime --bin curriculum_resume_preflight -- --help
cargo run --release -p vraxion-runtime --bin final_bake_preflight -- --help
```

Long or expensive runs must write partial outcomes continuously. End-only reporting is not acceptable for the current operating model.

## License

See [`LICENSE`](LICENSE).

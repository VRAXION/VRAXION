# VRAXION

VRAXION is building **INSTNCT**: a Rust-first, gradient-free architecture whose active object is a governed computational substrate, not a fixed backprop-trained layer stack.

This repository has been consolidated around the current winning mainline. Historical beta, bounded-service, byte-pipeline, and probe-era work is retained for auditability, but it is not the active sales surface unless promoted back into `main`.

## Current Source Of Truth

```text
branch = main
current_release = v5.0.0-e78.0
runtime_slice = 5f335cec3502d6c932e2f40c5c5a3a389eb44b7e
runtime_subject = Add canonical final train entrypoint
base_runtime_slice = 0879a2c004cf6a002bd5639d9cb7a759709a41aa Extract Rust final bake API
```

## Current Mainline

The active line is E78 on top of E77, E76, E75, E74, E73, E72, E71, E70, and E69:

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

```text
Pocket Library store
  -> guarded active set
  -> curriculum runner
  -> queued curriculum rows
  -> resumable checkpoint/progress stream
  -> unified final-bake gate
  -> reusable final-bake library API
  -> deterministic final curriculum lane runner
  -> multi-lane final-training supervisor
  -> global Pocket Library merge/dedupe/challenger gate
  -> canonical final_train campaign entrypoint
```

The current engineering priority is final-training readiness: dataset/curriculum contract, pocket capability scoring, governed global library growth, continuous writeout, resumable runs, and an inference path from loaded pockets through Agency commit and egress rendering.

## What Is Current

- Rust runtime surface: [`vraxion-runtime/`](vraxion-runtime/)
- Current GitHub release: [`v5.0.0-e78.0`](https://github.com/VRAXION/VRAXION/releases/tag/v5.0.0-e78.0)
- Current docs:
  - [`docs/CURRENT_STATUS.md`](docs/CURRENT_STATUS.md)
  - [`docs/GETTING_STARTED.md`](docs/GETTING_STARTED.md)
  - [`VALIDATED_FINDINGS.md`](VALIDATED_FINDINGS.md)
  - [Wiki Timeline Archive](https://github.com/VRAXION/VRAXION/wiki/Timeline-Archive)

## Claim Boundary

Current mainline claims must match code on `main`.

Allowed current claim:

> VRAXION has a Rust mainline for governed Pocket Library state, resumable curriculum execution, multi-lane final-training supervision, global Pocket Library merge/dedupe governance, and one canonical `final_train` campaign entrypoint.

Current E78 extension:

> VRAXION has a deterministic Rust `final_train` command that runs the global Pocket Library supervisor over multi-lane E75 final-training lanes, writes progress/manifests/results, blocks redundant clones, and records zero bad commits or unsafe promotions in the E78 evidence run.

Do not claim from this repo state alone:

- hosted SaaS availability
- public production API readiness
- GPT-like/open-domain assistant readiness
- final dataset readiness
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
cargo run --release -p vraxion-runtime --bin final_train -- 3 8 target/ci/e78_final_train_smoke --preflight-rounds 4 --checkpoint-interval 4
```

Long or expensive runs must write partial outcomes continuously. End-only reporting is not acceptable for the current operating model.

## License

See [`LICENSE`](LICENSE).

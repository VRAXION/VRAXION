# Contributing

VRAXION `main` is intentionally small. The active runtime line is
`vraxion-runtime/`; current post-release evidence for governed pocket validation
lives in `docs/research/` and `scripts/probes/`.

## Current Scope

Current mainline work belongs in:

- `vraxion-runtime/` - Rust runtime mechanics and preflight/final-training binaries.
- `scripts/probes/` - checked evidence probes that belong to the current mainline.
- `README.md`, `BETA.md`, `VALIDATED_FINDINGS.md`, `docs/CURRENT_STATUS.md`,
  `docs/GETTING_STARTED.md`, and `docs/VERSION.json` - public current state.
- `docs/research/E79_*` - latest released training-data/curriculum readiness evidence.
- `docs/research/E80_*` through `docs/research/E85_*` - current post-release
  dataset-backed scoring and CALC-SCRIBE evidence.
- GitHub wiki timeline pages - historical record and consolidation manifest.

Do not reintroduce old beta/grower/byte-pipeline/probe material as current
guidance unless it is explicitly promoted back into the E79+ runtime/evidence
line.

## Validation

Run the smallest command set that proves your change. For normal runtime or
public-surface changes:

```bash
cargo fmt --check -p vraxion-runtime
cargo clippy -p vraxion-runtime --all-targets -- -D warnings
cargo test -p vraxion-runtime
```

For final-training/preflight changes, also run bounded smokes with progress
artifacts:

```bash
cargo run --release -p vraxion-runtime --bin training_data_readiness -- 3 8 target/ci/e79_training_data_readiness_smoke
cargo run --release -p vraxion-runtime --bin final_train -- 3 8 target/ci/e79_final_train_smoke --preflight-rounds 4 --checkpoint-interval 4
```

For post-release evidence changes, include the matching probe and checker under
`scripts/probes/`, and record the result under `docs/research/`.

Long or expensive runs must write partial progress artifacts continuously.
End-only reporting is not acceptable.

## Archive Rule

Before removing a historical surface from `main`, make sure an explicit archive
tag, wiki record, or retained timeline entry exists. The current cleanup restore
point is:

```text
archive/repo/pre-e74-public-surface-cleanup-2026-06-13
```

## Contribution Terms

By submitting a contribution, you agree that your contribution is licensed to
the project under the terms in
[`docs/legal/VRAXION_CONTRIBUTOR_TERMS_V1.md`](docs/legal/VRAXION_CONTRIBUTOR_TERMS_V1.md).

## Issue Reports

Include the commit hash, exact command, OS, Rust version, and the smallest
reproducer. If the report involves a metric claim, include the raw output summary
or generated artifact path.

# Contributing

VRAXION `main` is intentionally small. The active line is the E73 Rust runtime
slice in `vraxion-runtime/`; older Python, legacy Rust, probe, Pages, and
research-output surfaces are archived, not active mainline.

## Current Scope

Current mainline work belongs in:

- `vraxion-runtime/` - Rust runtime mechanics and preflight binaries.
- `README.md`, `BETA.md`, `VALIDATED_FINDINGS.md`, `docs/CURRENT_STATUS.md`,
  `docs/GETTING_STARTED.md`, and `docs/VERSION.json` - public current state.
- `docs/research/E73_RUST_FINAL_BAKE_PREFLIGHT_*.md` and
  `docs/research/artifact_samples/e73_rust_final_bake_preflight/` - current
  final-bake evidence bundle.
- GitHub wiki timeline pages - historical record and consolidation manifest.

Do not reintroduce old beta/grower/byte-pipeline/probe material as current
guidance unless it is explicitly promoted back into the E73+ runtime line.

## Validation

Run the smallest command set that proves your change. For normal runtime or
public-surface changes:

```bash
cargo fmt --check -p vraxion-runtime
cargo clippy -p vraxion-runtime --all-targets -- -D warnings
cargo test -p vraxion-runtime
```

For final-bake/preflight changes, also run a bounded smoke with progress
artifacts:

```bash
cargo run --release -p vraxion-runtime --bin final_bake_preflight -- 1000 target/ci/e73_final_bake_smoke
```

Long or expensive runs must write partial progress artifacts continuously.
End-only reporting is not acceptable.

## Archive Rule

Before removing a historical surface from `main`, make sure an explicit archive
tag, wiki record, or retained timeline entry exists. The current cleanup restore
point is:

```text
archive/repo/pre-e73-public-surface-cleanup-2026-06-13
```

## Contribution Terms

By submitting a contribution, you agree that your contribution is licensed to
the project under the terms in
[`docs/legal/VRAXION_CONTRIBUTOR_TERMS_V1.md`](docs/legal/VRAXION_CONTRIBUTOR_TERMS_V1.md).

## Issue Reports

Include the commit hash, exact command, OS, Rust version, and the smallest
reproducer. If the report involves a metric claim, include the raw output summary
or generated artifact path.

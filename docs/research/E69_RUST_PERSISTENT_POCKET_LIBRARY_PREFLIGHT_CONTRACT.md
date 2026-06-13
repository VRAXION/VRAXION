# E69 Rust Persistent Pocket Library Preflight Contract

## Purpose

E69 consolidates the E54 persistent Pocket Library store into the Rust runtime
kernel.

It is not a new training probe. It checks that reusable Pocket artifacts can be
persisted, loaded, guarded, promoted, and replayed through the Rust runtime
surface without relying on Python as the active store implementation.

## Required Rust Surface

```text
vraxion-runtime::library
StoredPocketArtifact
LibraryLedgers
StoreSnapshot
StorePromotionCandidate
PocketLibraryStore
StoreDecision
StoreGuardReason
pocket_library_preflight binary
```

## Required Store Files

```text
registry.json
tokens.json
artifacts/*.json
lifecycle_ledger.jsonl
access_ledger.jsonl
promotion_ledger.jsonl
score_ledger.jsonl
```

## Required Rules

```text
pocket_uid is the machine identity
human_alias may change without breaking load
content_digest must match the frozen artifact
PocketToken must remain bound to the artifact
ABI version must match
quarantine and banned lifecycle states must block load
stale token versions must block load
concurrent stale writes must block
unsafe promotions must block before persistence
safe promotions must append registry/token/artifact/ledger state
reload snapshot must match registry/token/artifact counts
ledgers must contain lifecycle/access/promotion/score rows
```

## Required Blocks

```text
direct artifact tamper must block.
token/pocket swap must block.
ABI mismatch must block.
quarantine load must block.
banned load must block.
stale token must block.
concurrent stale write must block.
unsafe promotion must block.
bad promotion rate must remain zero.
```

## Validation

```text
cargo fmt --check -p vraxion-runtime
cargo clippy -p vraxion-runtime --all-targets -- -D warnings
cargo test -p vraxion-runtime
cargo run --release -p vraxion-runtime --bin pocket_library_preflight -- 1000000 target/pilot_wave/e69_rust_persistent_pocket_library_preflight
```

Boundary: E69 is a deterministic persistent-store preflight. It does not create
new Pocket skills, run final curriculum training, promote any real production
artifact, or make raw language reasoning, AGI, consciousness, deployment-quality,
or model-scale claims.

# E66 Pocket Registry Runtime Governance Preflight Contract

## Purpose

E66 consolidates the E50 PocketToken / Registry / Manager governance rule into
the Rust runtime kernel.

It is not a new learning probe. It checks that runtime Pocket calls are resolved
through a deterministic governance boundary:

```text
PocketToken / descriptor
-> immutable pocket_uid lookup
-> content_digest integrity check
-> token_hash binding check
-> ABI and capability compatibility check
-> token freshness check
-> lifecycle load gate
-> active Pocket set selection
```

## Required Rust Surface

```text
vraxion-runtime::pocket
PocketToken
PocketRegistryEntry
PocketLifecycle
resolve_pocket_call
active_pocket_set
pocket_governance_preflight binary
```

## Runtime Rules

```text
human_alias is documentation only.
pocket_uid is the stable machine identity.
content_digest proves the frozen artifact has not changed.
token_hash binds the descriptor to the artifact.
abi_version must match before load.
capability_signature must match before load.
stale token versions require re-audit.
Candidate / Quarantine / Deprecated / Banned lifecycles must not load.
Active / Stable / Specialist / Core lifecycles may load if all other gates pass.
```

## Required Blocks

```text
alias rename must not break UID resolution.
content digest mismatch must block.
token / pocket swap must block.
ABI mismatch must block.
capability mismatch must block.
stale token must block.
quarantine or banned lifecycle must block.
active set selection must filter unsafe entries before route choice.
```

## Validation

```text
cargo fmt --check -p vraxion-runtime
cargo clippy -p vraxion-runtime --all-targets -- -D warnings
cargo test -p vraxion-runtime
cargo run --release -p vraxion-runtime --bin pocket_governance_preflight -- 1000000 target/pilot_wave/e66_pocket_registry_runtime_governance_preflight
```

Boundary: E66 is a deterministic runtime governance preflight. It is not a raw
language reasoning, AGI, consciousness, deployment-quality, or model-scale
claim.

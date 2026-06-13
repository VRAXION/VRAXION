# E67 Pocket Manager Promotion Policy Preflight Contract

## Purpose

E67 consolidates the E52 Golden Disc / Core promotion policy into the Rust
runtime kernel.

It is not a new training probe. It checks that a Pocket cannot move into Core
or True Golden Disc status using final-answer score, immediate score,
popularity, or scalar average alone.

## Required Rust Surface

```text
vraxion-runtime::manager
ScoreVector
SafetyGate
ChallengerEvidence
PromotionEvidence
PromotionLevel
PromotionBlockReason
evaluate_promotion
pocket_manager_preflight binary
```

## Policy

The policy is gate-first and scope-bound:

```text
hard safety gate
-> multi-dimensional vector score
-> challenger sweep
-> counterfactual / uniqueness check
-> reload + shadow import
-> scope-limited promotion
-> long-horizon no-harm
```

Hard safety failures must block promotion regardless of utility.

## Score Dimensions

```text
utility
safety
eligible_activation
generality
uniqueness
transfer
robustness
cost
stability
scope_clarity
```

Activation is eligibility-conditioned. Raw popularity must not be enough to
promote a Pocket, and low raw activation must not prune rare-critical Pockets.

## Required Blocks

```text
unsafe high utility must quarantine.
credit hijack risk must quarantine.
delayed poison risk must quarantine.
negative transfer must quarantine.
scope violation must quarantine.
redundant clone must not Core-promote.
missing challenger sweep must not Core-promote.
reload/shadow import failure must not Core-promote.
long-horizon harm must not Core-promote.
rare-critical low activation must be preserved when other evidence is strong.
```

## Validation

```text
cargo fmt --check -p vraxion-runtime
cargo clippy -p vraxion-runtime --all-targets -- -D warnings
cargo test -p vraxion-runtime
cargo run --release -p vraxion-runtime --bin pocket_manager_preflight -- 1000000 target/pilot_wave/e67_pocket_manager_promotion_policy_preflight
```

Boundary: E67 is a deterministic runtime promotion-policy preflight. It does
not create new Pocket skills, start curriculum training, promote any real
production artifact, or make raw language reasoning, AGI, consciousness,
deployment-quality, or model-scale claims.

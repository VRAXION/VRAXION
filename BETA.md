# Mainline Pre-Release Contract

This file replaces the older `v5.0.0 Public Beta` front-door contract. The old beta/grower contract is historical evidence, not the current mainline.

## Current Target

```text
branch = main
runtime_slice = 3f519732949b73d5b55ae90a740381ca81143948
slice = E75 Rust final curriculum pocket-generation runner
base_runtime_slice = 0879a2c004cf6a002bd5639d9cb7a759709a41aa E74 Rust final bake API extraction
```

The current pre-release target is the E69-E75 Rust chain:

```text
E69 persistent Pocket Library store
E70 curriculum runner preflight
E71 curriculum queue preflight
E72 curriculum resume preflight
E73 unified final-bake preflight
E74 reusable final-bake API extraction
E75 final curriculum pocket-generation runner
```

## Promotion Rule

A future public release tag should be cut only after the E75+ surface has:

- clean Rust formatting, lint, and tests,
- verified Pocket Library preflight,
- verified runner preflight,
- verified queue preflight,
- verified resume preflight,
- verified final-bake preflight,
- verified final curriculum runner,
- continuous progress writeout for long runs,
- documented claim boundary.

## What Does Not Count

- old beta release numbers,
- old grower benchmark rows,
- byte-pipeline historical artifacts,
- bounded-service probes,
- untracked local runs,
- end-only long-run logs with no partial progress.

Those can remain as historical evidence, but they do not define the current release target.

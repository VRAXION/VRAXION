# Mainline Pre-Release Contract

This file replaces the older `v5.0.0 Public Beta` front-door contract. The old beta/grower contract is historical evidence, not the current mainline.

## Current Target

```text
main = fffc5a438078592c7ca97fd9a840fa5a7948b353
slice = E72 Rust curriculum resume preflight
```

The current pre-release target is the E69-E72 Rust chain:

```text
E69 persistent Pocket Library store
E70 curriculum runner preflight
E71 curriculum queue preflight
E72 curriculum resume preflight
```

## Promotion Rule

A future public release tag should be cut only after the E72+ surface has:

- clean Rust formatting, lint, and tests,
- verified Pocket Library preflight,
- verified runner preflight,
- verified queue preflight,
- verified resume preflight,
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

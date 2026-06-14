# Mainline Release Contract

This file replaces the older `v5.0.0 Public Beta` front-door contract. The old beta/grower contract is historical evidence, not the current mainline.

## Current Release

```text
branch = main
current_release = v5.0.0-e78.0
runtime_slice = 5f335cec3502d6c932e2f40c5c5a3a389eb44b7e
slice = E78 canonical final_train campaign entrypoint
base_runtime_slice = 0879a2c004cf6a002bd5639d9cb7a759709a41aa E74 Rust final bake API extraction
```

The current GitHub release is the E69-E78 Rust chain:

```text
E69 persistent Pocket Library store
E70 curriculum runner preflight
E71 curriculum queue preflight
E72 curriculum resume preflight
E73 unified final-bake preflight
E74 reusable final-bake API extraction
E75 final curriculum pocket-generation runner
E76 multi-lane final-training supervisor
E77 global Pocket Library merge supervisor
E78 canonical final_train campaign entrypoint
```

## Release Gate Status

The `v5.0.0-e78.0` release is cut only after the E78 surface has:

- clean Rust formatting, lint, and tests,
- verified Pocket Library preflight,
- verified runner, queue, and resume preflights,
- verified final-bake preflight/API,
- verified final curriculum lane runner,
- verified multi-lane final-training supervisor,
- verified global Pocket Library merge/dedupe supervisor,
- verified canonical `final_train` entrypoint,
- continuous progress writeout for long runs,
- documented claim boundary.

This does not claim hosted SaaS availability, public production API readiness, GPT-like/open-domain assistant readiness, final dataset readiness, trained model/weights readiness, safety-aligned deployment, consciousness, or sentience.

## What Does Not Count

- old beta release numbers,
- old grower benchmark rows,
- byte-pipeline historical artifacts,
- bounded-service probes,
- untracked local runs,
- end-only long-run logs with no partial progress.

Those can remain as historical evidence, but they do not define the current release target.

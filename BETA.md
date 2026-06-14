# Mainline Release Contract

This file replaces the older `v5.0.0 Public Beta` front-door contract. The old beta/grower contract is historical evidence, not the current mainline.

## Current Release

```text
branch = main
current_release = v5.0.0-e79.0
latest_released_runtime_slice = a908a838a1119540ed88bc91e10cfcb0bdae92a8
latest_released_runtime_subject = Add training data curriculum readiness gate
base_runtime_slice = 0879a2c004cf6a002bd5639d9cb7a759709a41aa E74 Rust final bake API extraction
```

The current GitHub release is the E69-E79 Rust chain:

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
E79 training data/curriculum readiness gate
```

## Current Main Since Release

```text
current_main_head = 56a9cf0305c1bfddd0e9b763b5e0d80fc9ec3bca
current_main_subject = Add E85 calc scribe mixed stream integration
post_release_range = E80-E85
```

E80-E85 are tracked mainline evidence after the latest release. They are not a new release tag yet.

```text
E80 dataset-backed pocket capability scoring evidence
E81 CALC-SCRIBE v002 multiseed visible-marker training
E82 CALC-SCRIBE v003 floor-division confirmation
E83 CALC-SCRIBE v003 LocalGolden promotion/reload
E84 CALC-SCRIBE transfer and negative-scope probe
E85 CALC-SCRIBE mixed-stream inference integration
```

## Release Gate Status

The `v5.0.0-e79.0` release is cut only after the E79 surface has:

- clean Rust formatting, lint, and tests,
- verified Pocket Library preflight,
- verified runner, queue, and resume preflights,
- verified final-bake preflight/API,
- verified final curriculum lane runner,
- verified multi-lane final-training supervisor,
- verified global Pocket Library merge/dedupe supervisor,
- verified canonical `final_train` entrypoint,
- verified training-data/curriculum readiness gate before `final_train` supervisor work,
- verified `final_train` fail-fast behavior when readiness cannot cover the full candidate rotation,
- continuous progress writeout for long runs,
- documented claim boundary.

E80-E85 add post-release evidence for scoped visible calculation-trace validation. They do not change the latest release number until a new release is explicitly cut.

This does not claim hosted SaaS availability, public production API readiness, GPT-like/open-domain assistant readiness, GSM8K solving, natural-language word-problem solving, final production dataset completion, trained model/weights readiness, Core memory, True Golden promotion, safety-aligned deployment, consciousness, or sentience.

## What Does Not Count

- old beta release numbers,
- old grower benchmark rows,
- byte-pipeline historical artifacts,
- bounded-service probes,
- untracked local runs,
- end-only long-run logs with no partial progress.

Those can remain as historical evidence, but they do not define the current release target.

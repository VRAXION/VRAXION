# Mainline Release Contract

This file replaces the older `v5.0.0 Public Beta` front-door contract. The old beta/grower contract is historical evidence, not the current mainline.

## Current Release

```text
branch = main
current_release = v6.1.7
latest_released_runtime_slice = a908a838a1119540ed88bc91e10cfcb0bdae92a8
latest_released_runtime_subject = Add training data curriculum readiness gate
```

The current GitHub release is the v6 E127 governed text-operator checkpoint.
The released Rust runtime foundation remains the E69-E79 Rust chain:

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

## Current Evidence Since Release

```text
current_evidence_anchor = f32a6f4b
current_evidence_subject = Finalize E127 cycle 40 checkpoint
evidence_range = E80-E127
```

E80-E127 are tracked mainline evidence included in `v6.1.7`.

```text
E80-E85 CALC-SCRIBE visible calculation-trace evidence
E86-E89 LocalGolden curriculum / selector / survival / naming lock
E90-E106 Operator curriculum expansions
E107 Operator survival role and regression gauntlet
E108 external transfer and negative-scope no-harm gauntlet
E109 rank ladder and GoldenWatch probation policy
E110 Silver-to-Gold scoped probation wave
E111 Bronze mutation/prune scoped Gold conversion wave
E112 Gold-to-CoreMemoryCandidate prune-heavy probation wave
E113 FineWeb-Edu light stress hard mutation/recycle probe
E119-E126 FineWeb/text-understanding skill farming and Orange probation
E127 overnight cyclic Orange/Legendary text-operator farm
```

## Release Gate Status

The v6 release is cut only after the E127 surface has:

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
- E127 cycle-40 tracked artifact samples,
- deterministic text-to-text render smoke boundary,
- fresh handover documentation for another Codex session.

E80-E127 are now included in the `v6.1.7` release checkpoint.

This does not claim hosted SaaS availability, public production API readiness, GPT-like/open-domain assistant readiness, GSM8K solving, natural-language word-problem solving, final production dataset completion, trained model/weights readiness, PermaCore/TrueGolden promotion, safety-aligned deployment, consciousness, or sentience.

## What Does Not Count

- old beta release numbers,
- old grower benchmark rows,
- byte-pipeline historical artifacts,
- bounded-service probes,
- untracked local runs,
- end-only long-run logs with no partial progress.

Those can remain as historical evidence, but they do not define the current release target.

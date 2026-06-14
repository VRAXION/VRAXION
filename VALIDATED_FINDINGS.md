# VRAXION Validated Findings

_Last updated: 2026-06-14_

This is the active evidence summary for the current repo state. Historical findings remain in git history, archived wiki state, release notes, and the Timeline Archive.

## Current State

```text
branch = main
current_release = v5.0.0-e79.0
current_evidence_anchor = 05415f5b06a43440742715ea93a5e2ec97632f21
current_evidence_subject = Add E113 FineWeb light stress recycle probe
latest_released_runtime_slice = a908a838a1119540ed88bc91e10cfcb0bdae92a8 E79 training data/curriculum readiness gate
```

## Current Mainline Chain

| Slice | Commit | Finding | Status |
|---|---|---|---|
| E69-E79 | `a908a838` | Rust Pocket Library, curriculum runner/queue/resume, final-bake API, final curriculum runner, multi-lane supervisor, global merge supervisor, canonical `final_train`, and training-data/curriculum readiness gate. | Latest release |
| E80-E85 | `56a9cf03` | CALC-SCRIBE visible calculation-trace evidence, LocalGolden reload, transfer/negative-scope probe, and mixed-stream no-call integration. | Post-release main evidence |
| E86-E89 | `a6935e61` | LocalGolden seeded curriculum, sparse active-set selector, survival gauntlet, and Operator naming/schema lock. | Post-release main evidence |
| E90-E106 | `b75c64cb` | Operator curriculum expansions for text evidence, temporal state, agency guards, output hygiene, active evidence requests, memory hygiene, routing, multi-skill execution, scheduling, grounded answer decisions, clarification repair, multi-turn continuity, compression, and task progress. | Post-release main evidence |
| E107 | `1fcdf954` | E90-E106 Operator library survival role and regression gauntlet. | Post-release main evidence |
| E108 | `0389c211` | External dataset transfer and negative-scope no-harm gauntlet. | Post-release main evidence |
| E109 | `555c5006` | Operator rank ladder and GoldenWatch probation policy: 14 Gold, 35 Silver, 87 Bronze, 0 DiamondCandidate, 0 RedFlag. | Post-release main evidence |
| E110 | `b378c2c5` | Silver-to-Gold pressure wave: 35 candidates, 35 scoped Gold promotions, 0 hard negatives. | Post-release main evidence |
| E111 | `d71e3657` | Bronze mutation/prune wave: 87 candidates, 87 scoped Gold variant promotions, 0 hard negatives, post-wave rank summary of 136 Gold / 0 Silver / 0 Bronze / 0 DiamondCandidate / 0 RedFlag / 3 Deprecated. | Post-release main evidence |
| E112 | `9de33241` | Gold-to-CoreMemoryCandidate prune-heavy probation wave: 136 candidates, 136 CoreMemoryCandidate qualifications, 0 hard negatives, mean selected prune ratio 69.4118%, final post-wave rank summary of 136 CoreMemoryCandidate / 0 Gold / 0 Silver / 0 Bronze / 0 DiamondCandidate / 0 RedFlag / 3 Deprecated. | Post-release main evidence |
| E113 | `05415f5b` | FineWeb-Edu 100k light stress/recycle probe: baseline 2,624 hard negatives across 88 operators, selected recycled variants 0 hard negatives, 0 neutral waste, 3,461,003 selected calls/positives, 136 recycled operators. | Current evidence anchor |

## Current Validated Claim

> VRAXION has a Rust mainline for governed Pocket Library state, resumable curriculum execution, multi-lane final-training supervision, global Pocket Library merge/dedupe governance, a training-data/curriculum readiness gate, one canonical `final_train` campaign entrypoint, and post-release Operator evidence through E113. E112 qualifies the scoped Gold pool into CoreMemoryCandidate probation, and E113 stress-tests that pool on a 100k FineWeb-Edu seed pack where selected recycled variants remove the tracked baseline hard negatives.

## Operational Finding

Long-running work must write progress continuously and support resume. The Rust runtime layers prove checkpoint/progress/writeout behavior through the E72-E79 chain. The E80-E113 evidence layer is a post-release research/probe surface and must remain explicitly scoped until promoted into runtime-facing behavior.

## Hard Boundary

This file does not claim:

- hosted SaaS
- public production API readiness
- GPT-like/open-domain assistant readiness
- GSM8K solving
- natural-language word-problem solving
- final production dataset completion
- trained model/weights readiness
- PermaCore or TrueGolden promotion
- safety-aligned production deployment
- consciousness or sentience
- that old beta/grower/byte-pipeline results are the current sellable model

## Historical Evidence

Previous bounded-service, open-vocab assistant, beta release, grower, byte-pipeline, and C19/EP results are historical evidence. They can still be useful for research context, but they are not the current mainline unless promoted back into E79+ code and docs.

Primary archive surfaces:

- `archive/branches/2026-06-13/*`
- `archive/wiki/pre-consolidation-2026-06-13`
- [Timeline Archive wiki](https://github.com/VRAXION/VRAXION/wiki/Timeline-Archive)
- Git history of this repository

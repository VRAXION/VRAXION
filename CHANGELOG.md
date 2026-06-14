# Changelog

This changelog is narrowed to the current E85 mainline. Full historical beta,
probe, Python SDK, legacy Rust, and research-output history is preserved in git
history and archive tags.

## 2026-06-14 - E85 CALC-SCRIBE Mixed Stream Inference Integration

- Current main head: `56a9cf0305c1bfddd0e9b763b5e0d80fc9ec3bca`
  (`Add E85 calc scribe mixed stream integration`).
- Latest GitHub release remains `v5.0.0-e79.0`.
- E85 confirms managed active-set routing in a mixed input stream:
  visible calc traces call CALC-SCRIBE, invalid visible traces are rejected,
  and natural text / word-problem text without visible trace framing no-calls.
- Primary validation route/action minimums are `1.000000`; false-call and
  false-commit maximums are `0.000000`.
- Failed controls remain documented: native-only active set misses transfer
  formats, full library scan false-calls, and alias/numeric-keyword routing
  false-calls.

## 2026-06-14 - E84 CALC-SCRIBE Transfer And Negative Scope Probe

- Runtime evidence head at that point: `0a06c153655e079afcd79dbf00069a458d4651b5`
  (`Add E84 calc scribe transfer scope probe`).
- E84 confirms transfer across explicit visible calc-trace formats such as
  `<<expr=result>>`, `[calc expr=result]`, `calc: expr -> result`, short
  isolated `expr = result`, unicode operator lines, and context-wrapped visible
  native markers.
- Negative scope is explicit: no visible calc marker means `NO_CALL`, not hidden
  answer inference.
- The overbroad word-problem solver and always-commit controls fail in the
  expected direction, proving why the scope guard matters.

## 2026-06-14 - E80-E83 CALC-SCRIBE LocalGolden Evidence Build

- E80 adds dataset-backed pocket capability scoring and promotion evidence from
  local dataset rows, with 8 seeds/workers, 3 promoted pockets, and 0 bad
  promotions.
- E81 trains CALC-SCRIBE v002 across 16 seeds/workers and isolates the remaining
  visible-marker gap to floor division.
- E82 promotes CALC-SCRIBE v003 for the visible marker family, closing floor
  division with validation/action/floor/adversarial minimums at `1.000000`.
- E83 confirms governed LocalGolden promotion/reload for CALC-SCRIBE v003 with
  reload match rate `1.000000`, 0 bad promotions, and tamper/token/unsafe
  global-scope blockers.
- Boundary: this is visible calculation-trace validation, not GSM8K solving or
  open-domain reasoning.

## 2026-06-14 - E79 Training Data Curriculum Readiness Gate

- Current GitHub release: `v5.0.0-e79.0`.
- Runtime slice: `a908a838a1119540ed88bc91e10cfcb0bdae92a8`
  (`Add training data curriculum readiness gate`).
- E79 adds `vraxion-runtime/src/training_data.rs` and the
  `training_data_readiness` CLI.
- `final_train` now runs the E79 gate before global supervisor work and blocks
  fail-fast if the data/curriculum contract cannot cover the full candidate
  rotation.
- The E79 gate writes readiness progress, result, manifest, report, and
  curriculum manifest artifacts.
- CI smokes the standalone E79 readiness command and checks the nested
  readiness artifact tree inside the E79 `final_train` smoke.

## 2026-06-14 - E78 Canonical Final Train Entrypoint

- GitHub release at that point: `v5.0.0-e78.0`.
- Runtime slice at that point: `5f335cec3502d6c932e2f40c5c5a3a389eb44b7e`
  (`Add canonical final train entrypoint`).
- E77 runtime slice: `7e91aaaa8a5e1571f60d06baa0b00c56e096c5cc`
  (`Add global Pocket Library merge supervisor`).
- E76 runtime slice: `3b44cfe0a246ce19677c595143877af27c381c6e`
  (`Add Rust final training supervisor`).
- E75 runtime slice: `3f519732949b73d5b55ae90a740381ca81143948`
  (`Add Rust final curriculum runner`).
- E74 runtime slice: `0879a2c004cf6a002bd5639d9cb7a759709a41aa`
  (`Extract Rust final bake API`).
- E76 adds multi-lane final-training fanout and aggregate progress artifacts.
- E77 adds global Pocket Library merge, dedupe/challenger governance, guarded
  reload, global registry, and clone blocking.
- E78 adds the canonical `final_train` command over the E77/E76/E75 stack.
- CI smokes the canonical E78 `final_train` path and checks top-level plus
  nested global-supervisor artifacts.

## Current Runtime And Evidence Chain

| Slice | Commit | Purpose |
|---|---|---|
| E69 | `8b00c352` | Persistent Pocket Library store |
| E70 | `9accc081` | Curriculum runner preflight |
| E71 | `c9dcad01` | Curriculum queue preflight |
| E72 | `fffc5a43` | Curriculum resume preflight |
| E73 | `51cd82a1` | Unified Rust final-bake preflight |
| E74 | `0879a2c0` | Final-bake library API extraction |
| E75 | `3f519732` | Final curriculum pocket-generation runner |
| E76 | `3b44cfe0` | Multi-lane final-training supervisor |
| E77 | `7e91aaaa` | Global Pocket Library merge supervisor |
| E78 | `5f335cec` | Canonical `final_train` campaign entrypoint |
| E79 | `a908a838` | Training data/curriculum readiness gate |
| E80 | `6c4181cf` | Dataset-backed scoring evidence |
| E81 | `b4335206` | CALC-SCRIBE v002 multiseed training |
| E82 | `3914a64a` | CALC-SCRIBE v003 floor-division confirm |
| E83 | `4370bacc` | CALC-SCRIBE v003 LocalGolden reload |
| E84 | `0a06c153` | CALC-SCRIBE transfer/negative-scope probe |
| E85 | `56a9cf03` | CALC-SCRIBE mixed-stream integration |

## Historical Access

Historical release notes before E79 were removed from the active front door
because they described superseded beta/grower/byte-pipeline lines. Restore or
inspect them from:

```bash
git show archive/repo/pre-e74-public-surface-cleanup-2026-06-13:CHANGELOG.md
```

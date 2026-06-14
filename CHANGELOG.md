# Changelog

This changelog is narrowed to the current E78 mainline. Full historical beta,
probe, Python SDK, legacy Rust, and research-output history is preserved in git
history and archive tags.

## 2026-06-14 - E78 Canonical Final Train Entrypoint

- Current GitHub release: `v5.0.0-e78.0`.
- Current runtime slice: `5f335cec3502d6c932e2f40c5c5a3a389eb44b7e`
  (`Add canonical final train entrypoint`).
- E77 runtime slice: `7e91aaaa8a5e1571f60d06baa0b00c56e096c5cc`
  (`Add global Pocket Library merge supervisor`).
- E76 runtime slice: `3b44cfe0a246ce19677c595143877af27c381c6e`
  (`Add Rust final training supervisor`).
- E75 runtime slice: `3f519732949b73d5b55ae90a740381ca81143948`
  (`Add Rust final curriculum runner`).
- E74 runtime slice: `0879a2c004cf6a002bd5639d9cb7a759709a41aa`
  (`Extract Rust final bake API`).
- Active mainline remains narrowed to `vraxion-runtime/` plus current docs and
  E73-E78 final-bake/final-training evidence.
- E76 adds multi-lane final-training fanout and aggregate progress artifacts.
- E77 adds global Pocket Library merge, dedupe/challenger governance, guarded
  reload, global registry, and clone blocking.
- E78 adds the canonical `final_train` command over the E77/E76/E75 stack.
- CI now smokes the canonical E78 `final_train` path and checks top-level plus
  nested global-supervisor artifacts.

## Current Runtime Chain

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

## Historical Access

Historical release notes before E78 were removed from the active front door
because they described superseded beta/grower/byte-pipeline lines. Restore or
inspect them from:

```bash
git show archive/repo/pre-e74-public-surface-cleanup-2026-06-13:CHANGELOG.md
```

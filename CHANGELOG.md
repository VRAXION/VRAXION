# Changelog

This changelog is narrowed to the current E75 mainline. Full historical beta,
probe, Python SDK, legacy Rust, and research-output history is preserved in git
history and archive tags.

## 2026-06-13 - E75 Final Curriculum Runner

- Current runtime slice: `3f519732949b73d5b55ae90a740381ca81143948`
  (`Add Rust final curriculum runner`).
- E74 runtime slice: `0879a2c004cf6a002bd5639d9cb7a759709a41aa`
  (`Extract Rust final bake API`).
- E73 runtime slice: `51cd82a11d8f1d2b98ee3e49538c7c26afdb767b`
  (`Add Rust final bake preflight`).
- Public docs aligned at `7742e714` (`Align public docs with E73 final bake
  preflight`).
- GitHub branch surface reduced to `main`; 52 former branch heads preserved
  under `archive/branches/2026-06-13/*`.
- Wiki preserved before cleanup under
  `archive/wiki/pre-consolidation-2026-06-13`.
- Repo public-surface cleanup restore point:
  `archive/repo/pre-e74-public-surface-cleanup-2026-06-13`.
- Active mainline narrowed to `vraxion-runtime/` plus current docs and E73/E74/E75
  final-bake/final-training evidence.
- E75 adds the deterministic final curriculum pocket-generation runner with
  preflight gating, checkpoint/progress writeout, resume, and Pocket Library
  growth.

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

## Historical Access

Historical release notes before E75 were removed from the active front door
because they described superseded beta/grower/byte-pipeline lines. Restore or
inspect them from:

```bash
git show archive/repo/pre-e74-public-surface-cleanup-2026-06-13:CHANGELOG.md
```

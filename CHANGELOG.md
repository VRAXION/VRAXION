# Changelog

This changelog is narrowed to the current E74 mainline. Full historical beta,
probe, Python SDK, legacy Rust, and research-output history is preserved in git
history and archive tags.

## 2026-06-13 - E74 Runtime/Public Surface Consolidation

- Current runtime slice: `0879a2c004cf6a002bd5639d9cb7a759709a41aa`
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
- Active mainline narrowed to `vraxion-runtime/` plus current docs and E73/E74
  final-bake evidence.

## Current Runtime Chain

| Slice | Commit | Purpose |
|---|---|---|
| E69 | `8b00c352` | Persistent Pocket Library store |
| E70 | `9accc081` | Curriculum runner preflight |
| E71 | `c9dcad01` | Curriculum queue preflight |
| E72 | `fffc5a43` | Curriculum resume preflight |
| E73 | `51cd82a1` | Unified Rust final-bake preflight |
| E74 | `0879a2c0` | Final-bake library API extraction |

## Historical Access

Historical release notes before E74 were removed from the active front door
because they described superseded beta/grower/byte-pipeline lines. Restore or
inspect them from:

```bash
git show archive/repo/pre-e74-public-surface-cleanup-2026-06-13:CHANGELOG.md
```

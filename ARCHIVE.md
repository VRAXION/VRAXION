# Archive Policy

This page explains how retired surfaces leave `main` without losing their context.

## Current Mainline

- **Current mainline** means the active self-wiring line that actually ships on `main`.
- Public documentation for the current self-wiring direction must stay lean, accurate, and tied to repo-tracked code.
- Only the active self-wiring graph line belongs on `main`:
  - shipped code
  - public documentation for the current self-wiring direction
  - the smallest proof and validation surface needed to justify the current line

## Validated Finding

- A **Validated finding** is evidence worth keeping in docs, but not yet promoted into the canonical code path.
- Validated findings may justify a proving recipe or side probe, but they do not automatically become the live default.

## Experimental Branch

- An **Experimental branch** is where risky or bulky work can live before it earns promotion.
- If a line is not part of the current self-wiring mainline doctrine, archive it.

## Archive Branches

- Historical Diamond Code extraction branch: `archive/diamond-code-era-20260322`
- Surface-freeze preservation branch: `archive/instnct-surface-freeze-20260322`
- `v4/`, `v4.2/`, `legacy/` removed from `main` in v5.0.0-beta.1 (preserved in git history)
- `RESEARCH_NOTICE_V5_MISMATCH.md` removed (issue resolved 2026-03-27)

The rule is simple: if code or docs do not belong to the current self-wiring mainline, they should move to archive branches or local-only run artifacts instead of bloating `main`.

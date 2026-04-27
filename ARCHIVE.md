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

## Archive Branches and Tags

Historical surfaces are preserved as **immutable archive tags** (no long-lived archive branches remain). Tag prefixes:

- `archives/<topic>-<YYYYMMDD>` — content surfaces (script trees, doc surfaces, residue dumps)
- `archive/<branch>-<YYYYMMDD>` — previously-live branch heads (frozen on the cleanup date)

**Content snapshots (2026-04-20 → 2026-04-27):**

- `archives/python-research-20260420` — Python research lane (was `instnct/`); migrated to Rust `instnct-core/` on 2026-04-13, archived 2026-04-20.
- `archives/tools-legacy-diag-20260420` — pre-Fázis-6 `tools/` tree (79 scripts); trimmed to 29 on 2026-04-20.
- `archives/public-readiness-residue-20260420` — public-release cleanup residue.
- `archives/tools-cleanup-20260425` — `tools/` 74→22 cleanup pass (53 archived scripts).
- `archives/instnct-examples-2026-04-archive-20260427` — `instnct-core/examples/archive/2026-04/` tree (56 retired research examples + README) physically removed from `main` on 2026-04-27 per the "only mainline belongs on `main`" rule. Covers the 2026-04-17 era exploration paths: `addition_*`, `pocket_*`, `chip_*`, `circuit_*`, `conv_*`, `breed_*`, `connectome_*`, `flybrain*`, `mirror_*`, `abstract_core_v1..v4`, `byte_*`/`byte_opcode_v1`, `grid3_*`, `all_binary_mirror`.

**Branch-head snapshots (2026-04-25 → 2026-04-27):**

- `archive/main-pre-cleanup-20260425`, `archive/main-pre-cleanup-20260426`, `archive/main-pre-cleanup-20260427` — main HEAD before each successive cleanup pass.
- `archive/codex-phase-b-logging-smoke-20260425` — codex/phase-b-logging-smoke branch (merged into main 2026-04-25).
- `archive/research-overnight-sct-empirical-20260425` — overnight SCT research branch.
- `archive/research-sandbox-h128-d1-20260426` — H=128 D1b cross-H pilot (n=3) sandbox; superseded by formal Phase D2 (n=5) on main.
- `archive/saved-neuron-one-wip-20260425`, `archive/saved-pre-connectome-research-20260425` — historical saved branches.
- `archive/claude-review-repo-access-20260425` — claude review branch (verified redundant before retirement).
- `archive/v4.1-20260318`, `archive/v4.2-20260318`, `archive/nightly-20260318`, plus various `archive/claude/<topic>-<short>/<sha>` snapshots — older 2026-03-18 cleanup era.

Restore any file from a tag via:

```bash
git show <tag>:path/to/file
git checkout <tag> -- path/to/file
```

## Removed from `main` (preserved only in git history)

- `v4/`, `v4.2/`, `legacy/` — removed in `v5.0.0-beta.1` (preserved in git history)
- `RESEARCH_NOTICE_V5_MISMATCH.md` — removed on 2026-03-27 (issue resolved)
- `docs/instnct/`, `docs/byte-embedder/`, `docs/research/`, `docs/rust/`, `docs/pages/brain_replay/`, `docs/vraxion-connectome-explorer.html` — removed on 2026-04-20 as orphan legacy Pages under the pre-Blocks nav (commit `92f313b`). Content reachable via `git show 92f313b^:<path>`.
- `output/` scratch tree (~160 non-champion run-dumps, 45 MB) — physically removed on 2026-04-20; scratch was gitignored so not reconstructable, but source scripts in `tools/` can regenerate any run.

The rule is simple: if code or docs do not belong to the current self-wiring mainline, they should move to archive branches/tags or local-only run artifacts instead of bloating `main`.

For a chronological record of what changed when, see the [Timeline Archive wiki](https://github.com/VRAXION/VRAXION/wiki/Timeline-Archive).

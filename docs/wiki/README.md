# docs/wiki — Mirror of the GitHub Wiki

This directory is a **mirror** of the canonical GitHub wiki. The live source of truth is:

- **Canonical:** https://github.com/VRAXION/VRAXION/wiki (cloned locally at `S:/Git/VRAXION.wiki/`)

Keep the two surfaces in sync periodically so that repo-side tooling, website references, and diff reviews can work against a checked-in snapshot without requiring a live clone of the wiki repo.

## What belongs here

- A full mirror of the GitHub wiki pages (`Home.md`, `INSTNCT-Architecture.md`, `Theory-of-Thought.md`, `Timeline-Archive.md`, `v5-Rust-Port-Benchmarks.md`, `_Sidebar.md`, `_Footer.md`)
- Supporting assets referenced by wiki pages via relative links (`pipeline-architecture.svg`, `assets/` SVGs)
- `index.html` — the GitHub Pages redirect for the legacy `docs/wiki/` URL path; not a wiki page
- `archive/` — retired wiki pages kept for historical reference (pure "Moved" redirect stubs from the 2026-04 consolidation)

## What does NOT belong here

- Per-run logs or ad-hoc reports (put those in `docs/runs/`)
- Root-level canonical docs like `VALIDATED_FINDINGS.md` (that one lives at the repo root and is referenced from the wiki, not duplicated here)

## Syncing

The repo has `tools/sync_wiki_from_repo.py` which pushes from `docs/wiki/` into the adjacent `VRAXION.wiki/` checkout. Caveats as of 2026-04-17:

- The tool's `WIKI_DIR` constant points to `<repo>/VRAXION.wiki/` but the actual clone usually lives at `<sibling>/VRAXION.wiki/` (e.g. `S:/Git/VRAXION.wiki/`). Patch the path before running, or symlink.
- The tool's `FILES` list is out of date — it still lists six retired pages (now in `archive/`) and omits `Timeline-Archive.md` / `v5-Rust-Port-Benchmarks.md` / `Theory-of-Thought.md` / `pipeline-architecture.svg`. Update that list before relying on the tool.

Manual sync (until the tool is fixed) is just `cp` in both directions. After reconciling, commit on both sides: `git commit` here, and `git push` inside `VRAXION.wiki/` to publish to GitHub.

## Last full reconcile

- **2026-04-17** — pulled `v5-Rust-Port-Benchmarks.md`, `INSTNCT-Architecture.md`, `_Sidebar.md`, `_Footer.md`, `Theory-of-Thought.md` down from canonical; pushed `Home.md`, `Timeline-Archive.md`, `pipeline-architecture.svg` up to canonical; archived six redirect stubs.

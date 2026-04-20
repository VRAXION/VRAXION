# docs/wiki — Mirror of the GitHub Wiki

This directory is a **mirror** of the canonical GitHub wiki. The live source of truth is:

- **Canonical:** https://github.com/VRAXION/VRAXION/wiki (cloned locally as a sibling `VRAXION.wiki/` to this repo)

Keep the two surfaces in sync periodically so that repo-side tooling, website references, and diff reviews can work against a checked-in snapshot without requiring a live clone of the wiki repo.

## What belongs here

- A full mirror of the GitHub wiki pages (`Home.md`, `INSTNCT-Architecture.md`, `Theory-of-Thought.md`, `Timeline-Archive.md`, `v5-Rust-Port-Benchmarks.md`, `_Sidebar.md`, `_Footer.md`)
- Supporting assets referenced by wiki pages via relative links (`pipeline-architecture.svg`, `assets/` SVGs)
- `index.html` — the GitHub Pages redirect for the legacy `docs/wiki/` URL path; not a wiki page

## What does NOT belong here

- Per-run logs or ad-hoc reports (put those in `docs/runs/`)
- Root-level canonical docs like `VALIDATED_FINDINGS.md` (that one lives at the repo root and is referenced from the wiki, not duplicated here)
- Archived/retired wiki pages — these are removed from both the repo and the live wiki; git history retains anything needed. Ideas worth preserving migrate to blueprint entries on `Timeline-Archive.md` instead.

## Syncing

The repo has `tools/sync_wiki_from_repo.py` which pushes from `docs/wiki/` into an adjacent `VRAXION.wiki/` checkout. The tool auto-detects the wiki clone at `<sibling>/VRAXION.wiki/` (typical layout) and falls back to `<repo>/VRAXION.wiki/` if the sibling path is absent.

**Workflow for a maintainer:**

```bash
# 1. Make sure the wiki clone exists sibling to the repo:
#    <parent>/VRAXION/       ← main repo
#    <parent>/VRAXION.wiki/  ← wiki clone
#    If missing: git clone https://github.com/VRAXION/VRAXION.wiki.git

# 2. Dry-run: validate sources exist and preview changes
python tools/sync_wiki_from_repo.py --dry-run

# 3. Actual sync (copies from docs/wiki/ into VRAXION.wiki/)
python tools/sync_wiki_from_repo.py

# 4. Commit + push in the wiki repo
cd ../VRAXION.wiki
git add -A
git commit -m "Sync from main repo docs/wiki"
git push
```

**Limitations (know before running):**

- **Push-only.** The tool cannot pull changes from the online wiki. If someone edited the wiki online, manually reconcile before syncing (check `git status` in the wiki checkout first).
- **No deletions or renames.** Only additions and content updates. To delete a page, `git rm` it manually in the wiki checkout.

## Last full reconcile

- **2026-04-18** — cleaned archive stubs from both surfaces, added "Archived scripts (2026-04-18)" blueprint subsection to `Timeline-Archive.md`, updated cross-refs after `tools/*.py` mainline cleanup.
- **2026-04-17** — pulled `v5-Rust-Port-Benchmarks.md`, `INSTNCT-Architecture.md`, `_Sidebar.md`, `_Footer.md`, `Theory-of-Thought.md` down from canonical; pushed `Home.md`, `Timeline-Archive.md`, `pipeline-architecture.svg` up to canonical; archived six redirect stubs.

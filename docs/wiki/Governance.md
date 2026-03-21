<!-- Canonical source for the mirrored GitHub wiki page. Sync with tools/sync_wiki_from_repo.py. -->

# Documentation Governance

This page defines how VRAXION maintains canonical public docs, provenance, and page-level documentation rules. It is a **secondary reference page**. The wiki is a mirrored secondary surface; repo-tracked docs are the canonical public source.

## What This Page Is

Use this page when the question is about documentation truth, provenance, sync discipline, or page ownership.

Use [Engineering Protocol](Engineering) when the question is about how runs are executed, validated, and accepted as evidence.

## Canonical Documentation Model

- The canonical public source is the repo-tracked docs surface: `README.md`, `VALIDATED_FINDINGS.md`, `docs/index.html`, and `docs/wiki/`.
- `VRAXION.wiki/` is a mirror output, not an independent source of truth.
- Primary public pages should stay short, stable, and aligned with shipped code plus explicitly labeled findings.
- If repo-tracked docs and the wiki diverge, the repo-tracked docs win and the mirror must be resynced.

## Update Contract

When public-facing work changes:

1. update the repo-tracked source page first
2. run `tools/sync_wiki_from_repo.py`
3. run `tools/check_public_surface.py`
4. only then add PRs or issues as supporting evidence links

If a change skips that sequence, it is not ready to be treated as a stable public update.

## Page Placement Rules

- Use fewer, stronger pages instead of many tiny pages.
- Keep primary pages focused on architecture, evidence, and engineering protocol.
- Put volatile status reporting in release notes, roadmap, or issues, not in the stable front door.
- If a page is important enough to be in primary navigation, it should be readable without requiring issue archaeology.
- Governance stays secondary by design; it should support the front door, not compete with it.

## Versioning and Provenance

- Versioning source of truth: `VERSION.json`
- Citation source of truth: `CITATION.cff`
- Reproducible public claims should cite a release tag and its corresponding Zenodo record when applicable.

## Read Next

- [Home](Home)
- [VRAXION Architecture (INSTNCT)](SWG-v4.2-Architecture)
- [Validated Findings](Validated-Findings)
- [Engineering Protocol](Engineering)

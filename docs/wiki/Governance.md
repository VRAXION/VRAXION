<!-- Canonical source for the mirrored GitHub wiki page. Sync with tools/sync_wiki_from_repo.py. -->

# Governance

This page covers documentation policy, provenance, versioning, and page-maintenance rules for VRAXION. It is a **secondary reference page**. The wiki is a mirrored secondary surface; repo-tracked docs are the canonical public source.

## Documentation Model

- The canonical public source is the repo-tracked docs surface: `README.md`, `VALIDATED_FINDINGS.md`, `docs/index.html`, and `docs/wiki/`.
- `VRAXION.wiki/` is a mirror output, not an independent source of truth.
- Primary public pages should stay short, stable, and aligned with shipped code plus explicitly labeled findings.
- If repo-tracked docs and the wiki diverge, the repo-tracked docs win and the mirror should be resynced.

## Update Workflow

When public-facing work changes:

1. update the repo-tracked source page first
2. run `tools/sync_wiki_from_repo.py`
3. run `tools/check_public_surface.py`
4. only then add PRs or issues as supporting evidence links

## Consolidation Rules

- Use fewer, stronger pages instead of many tiny pages.
- Keep primary pages focused on architecture, evidence, and engineering protocol.
- Put volatile status reporting in release notes, roadmap, or issues, not in the stable front door.
- If a page is important enough to be in primary navigation, it should be readable without requiring issue archaeology.

## Versioning and Provenance

- Versioning source of truth: `VERSION.json`
- Citation source of truth: `CITATION.cff`
- Reproducible public claims should cite a release tag and its corresponding Zenodo record when applicable

## Read Next

- [[Home]]
- [VRAXION Architecture (INSTNCT)](SWG-v4.2-Architecture)
- [[Validated-Findings|Validated Findings]]
- [Engineering Protocol](Engineering)

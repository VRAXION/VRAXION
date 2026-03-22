# Contributing

VRAXION is a **public technical repo**. Rigor and reproducibility matter, but so does clarity: the public repo should be understandable to engineers, technical buyers, and contributors without reverse-engineering issue traffic.

## Public Truth Rules

Use the repo taxonomy consistently:

- **Current mainline** = shipped code on `main`
- **Validated finding** = experiment-backed result not yet promoted into the canonical code path
- **Experimental branch** = active build target or prototype direction

If a setting or training recipe is not in the canonical code path, do **not** describe it as the live mainline default.

## Repo Map

- `instnct/model/`: self-wiring graph implementations
- `instnct/lib/`: shared scoring, data, and logging helpers
- `instnct/tests/`: stress tests, sweeps, probes, and benchmark scripts
- `docs/`: GitHub Pages landing page
- `docs/wiki/`: canonical source files for mirrored wiki pages
- `VALIDATED_FINDINGS.md`: canonical evidence summary for public-facing claims
- `VERSION.json`: public release identity source of truth

Historical archive branch:

- The Diamond Code era is preserved on `archive/diamond-code-era-20260322`; it is not part of active `main`.

Treat older line names and local historical folders as timeline/archive context, not as equal-current public repo surfaces.

Edit `docs/wiki/` when changing mirrored wiki content. Do not treat `VRAXION.wiki/` as hand-edited doctrine.

## Documentation Governance

Repo-tracked docs are the **canonical public source** for VRAXION:

- `README.md`
- `VALIDATED_FINDINGS.md`
- `docs/index.html`
- `docs/wiki/`

`VRAXION.wiki/` is a mirror output only, not an independent source of truth.

When public-facing docs change:

1. edit the repo-tracked source first
2. run `python tools/sync_wiki_from_repo.py`
3. run `python tools/check_public_surface.py`
4. only then treat the update as stable public truth

Page placement rules:

- Use fewer, stronger pages instead of many tiny pages.
- Keep the primary public stack focused on architecture, evidence, engineering protocol, and the project timeline.
- Put volatile status in `Project Timeline` or issues, not in the stable front door.
- If a page is important enough to be in primary navigation, it should be readable without issue archaeology.

Versioning and provenance:

- Versioning source of truth: `VERSION.json`
- Citation source of truth: `CITATION.cff`
- Reproducible public claims should cite a release tag and its Zenodo record when applicable.

## Where To Put Things

- Core model changes: `instnct/model/`
- Shared training helpers: `instnct/lib/`
- Local corpora for the active line: `instnct/data/traindat/` (`.traindat` stays local-only and ignored)
- CPU and GPU experiments: `instnct/tests/`
- Public-facing repo text: root docs plus `docs/`

## Pull Request Requirements

Every PR should include:

- Why: what problem it solves
- What changed: concise behavior and file summary
- How to verify: exact commands

Guardrails:

- Do not commit run artifacts, logs, checkpoints, or large binaries
- Keep changes small and reversible where possible
- If a change affects metrics, state whether the result is:
  - current mainline
  - validated finding
  - experimental only

Recommended verification commands:

```bash
python -m compileall instnct tools
python instnct/tests/test_model.py
python tools/check_public_surface.py
```

## Reproducibility

If you report a metric, include at minimum:

- commit hash
- exact command line
- seed list
- relevant config values
- output summary

If the claim is important enough for the front door, it should also be mirrored into `VALIDATED_FINDINGS.md`.

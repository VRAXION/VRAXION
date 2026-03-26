# Contributing

VRAXION is a **public technical repo**. Rigor and reproducibility matter, but so does clarity: the public repo should be understandable to engineers, technical buyers, and contributors without reverse-engineering issue traffic.

## Public Truth Rules

Use the repo taxonomy consistently:

- **Current mainline** = shipped code on `main`
- **Validated finding** = experiment-backed result not yet promoted into the canonical code path
- **Experimental branch** = active build target or prototype direction

If a setting or training recipe is not in the canonical code path, do **not** describe it as the live mainline default.

## Repo Map

- `instnct/model/`: self-wiring graph implementation
- `instnct/lib/`: shared scoring, data, and logging helpers
- `instnct/tests/`: stress tests, sweeps, probes, and benchmark scripts
- `instnct/ops/`: overnight training and sweep runners
- `docs/wiki/`: canonical source files for mirrored wiki pages
- `VALIDATED_FINDINGS.md`: canonical evidence summary for public-facing claims
- `VERSION.json`: public release identity source of truth

## Where To Put Things

- Core model changes: `instnct/model/`
- Shared training helpers: `instnct/lib/`
- CPU and GPU experiments: `instnct/tests/`
- Public-facing repo text: root docs plus `docs/`
- Local corpora: `instnct/data/traindat/` (`.traindat` stays local-only and ignored)

## Documentation Governance

Repo-tracked docs are the **canonical public source**. The GitHub wiki is a mirrored secondary surface, not an independent source of truth.

When public-facing docs change:

1. Edit the repo-tracked source first
2. Run `python tools/sync_wiki_from_repo.py`
3. Run `python tools/check_public_surface.py`
4. Only then treat the update as stable public truth

Versioning source of truth: `VERSION.json`. Citation source of truth: `CITATION.cff`.

## Pull Request Requirements

Every PR should include:

- **Why**: what problem it solves
- **What changed**: concise behavior and file summary
- **How to verify**: exact commands

Guardrails:

- Do not commit run artifacts, logs, checkpoints, or large binaries
- Keep changes small and reversible where possible
- If a change affects metrics, state whether the result is current mainline, validated finding, or experimental only

Recommended verification commands:

```bash
python -m compileall instnct tools
python instnct/tests/test_model.py
python tools/check_public_surface.py
```

## Reproducibility

If you report a metric, include at minimum: commit hash, exact command line, seed list, relevant config values, and output summary. If the claim is important enough for the front door, mirror it into `VALIDATED_FINDINGS.md`.

## Archive Policy

`main` stays intentionally small. Only the active self-wiring graph line belongs here.

**What does not stay on `main`:** superseded repo eras, half-ready research detours, ephemeral review branches, one-off experimental integration branches.

**Practical rule:** if a line is not part of the current self-wiring mainline, archive it. If it still matters historically, keep a tag. If it neither matters historically nor supports the current line, delete it.

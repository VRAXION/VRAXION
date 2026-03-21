# Contributing

VRAXION is a **public technical repo**. Rigor and reproducibility matter, but so does clarity: the public repo should be understandable to engineers, technical buyers, and contributors without reverse-engineering issue traffic.

## Public Truth Rules

Use the repo taxonomy consistently:

- **Current mainline** = shipped code on `main`
- **Validated finding** = experiment-backed result not yet promoted into the canonical code path
- **Experimental branch** = active build target or prototype direction

If a setting or training recipe is not in the canonical code path, do **not** describe it as the live mainline default.

## Repo Map

- `v4.2/model/`: self-wiring graph implementations
- `v4.2/lib/`: shared scoring, data, and logging helpers
- `v4.2/tests/`: stress tests, sweeps, probes, and benchmark scripts
- `docs/`: GitHub Pages landing page
- `docs/wiki/`: canonical source files for mirrored wiki pages
- `VALIDATED_FINDINGS.md`: canonical evidence summary for public-facing claims

Edit `docs/wiki/` when changing mirrored wiki content. Do not treat `VRAXION.wiki/` as hand-edited doctrine.

## Where To Put Things

- Core model changes: `v4.2/model/`
- Shared training helpers: `v4.2/lib/`
- CPU and GPU experiments: `v4.2/tests/`
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
python -m compileall v4.2 tools
python v4.2/tests/test_model.py
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

# Contributing

VRAXION is a research repo where mechanism and repeatability matter more than presentation.
If a change cannot be reproduced or audited, it is not done.

## Repo Map

- `v4.2/model/`: self-wiring graph implementations
- `v4.2/lib/`: shared scoring, data, and logging helpers
- `v4.2/tests/`: stress tests, sweeps, probes, and benchmark scripts
- `docs/`: GitHub Pages landing page for the cleaned main branch

## Where To Put Things

- Core model changes: `v4.2/model/`
- Shared training helpers: `v4.2/lib/`
- CPU and GPU experiments: `v4.2/tests/`
- Public-facing repo text: root docs plus `docs/`

## Branch Naming

Use short, intent-revealing branch names. Examples:

- `feat/slot-soft-write`
- `chore/main-cleanup-pass2`
- `docs/self-wiring-pages-refresh`

## Pull Request Requirements

Every PR should include:

- Why: what problem it solves
- What changed: concise behavior and file summary
- How to verify: exact commands

Guardrails:

- Do not commit run artifacts, logs, checkpoints, or large binaries
- Keep changes small and reversible where possible
- If a change affects metrics or stability, include the exact benchmark or smoke command used

Recommended verification commands:

```bash
python -m compileall v4.2
python v4.2/tests/test_model.py
python v4.2/tests/test_logging.py
```

## Issues And Discussions

- GitHub Issues are best used for curated public updates or concrete cleanup tasks
- GitHub Discussions are for design questions and broader research discussion

## Reproducibility

If you report a metric, include at minimum:

- commit hash
- exact command line
- seed list
- relevant config values
- output summary

# Next Week Handoff

_Prepared 2026-05-19._

## Baseline To Protect

```text
099 bounded local/private release-ready baseline
```

Do not mutate the bounded release artifacts, packaged winner checkpoint, service API, or deployment harness while continuing capability training.

## Current Repo State

- Main branch is consolidated around the 078-100 proof line.
- Historical scratch tools and non-current probe runners were archived at `archive/pre-consolidation-20260519-main-snapshot`.
- Current status and claim boundary are documented in `README.md`, `docs/CURRENT_STATUS.md`, and these wiki pages.

## Best Next Technical Step

```text
101_FRESH_ASSISTANT_EVAL_AND_FAILURE_MAP
```

Purpose:

- evaluate the 100 checkpoint on fresh assistant prompts
- separate genuine instruction following from decoder/rubric artifacts
- map open-domain, multi-turn, Hungarian/English, refusal, collapse, and retention failures
- keep the 099 bounded release baseline frozen

## Stop Conditions

Stop capability scaling and diagnose if:

- bounded retention regresses
- generation collapses into static/repetitive/copy outputs
- public/GPT-like/production claims appear in reports
- the 099 bounded release artifacts change

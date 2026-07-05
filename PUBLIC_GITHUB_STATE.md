# Public GitHub State

This file defines how the public GitHub repository state should be read before
a new release is prepared.

## Current Public Truth

The current public status is defined by:

- default branch: `main`
- version record: `docs/VERSION.json`
- latest public release: `public-sdk-p11-20260629`
- release URL: `https://github.com/VRAXION/VRAXION/releases/tag/public-sdk-p11-20260629`
- Pages URL: `https://vraxion.github.io/VRAXION/`

If older GitHub tags or release records disagree with these files, treat the
current public status files as authoritative until maintainers explicitly
publish a newer reviewed public release.

## Historical GitHub Records

Historical GitHub tags, releases, and commit records may remain visible as
project history. Do not use older tags, older release titles, or older release
assets as current product status unless a maintainer has re-reviewed them and
linked them from the current public status docs.

Before creating or updating a public release, review:

- GitHub release title and body
- attached release assets, if any
- tag name and target commit
- default branch and branch list
- Dependabot or dependency-update PRs that may affect the public build
- Pages deployment target
- public status docs and release links

## Public-State Rules

- Keep release metadata and assets aligned with `docs/VERSION.json`.
- Keep `PUBLIC_RELEASE_CHECKLIST.md` as the release intake gate.
- Keep dependency update automation limited to public manifests and GitHub
  Actions.
- Keep private engine source, non-public training data, raw operator output,
  local machine paths, secrets, filled production config, and private
  dashboards out of public GitHub issues, pull requests, releases, assets, and
  Pages content.
- Do not imply a signed runnable artifact, hosted service, benchmark result, or
  production deployment until the reviewed artifact and public evidence exist.
- If a GitHub release is created without a runnable artifact, the release body
  must say so plainly.

## Pre-Release GitHub Check

For any future public artifact or training-result release, verify the public
GitHub state before merge:

```powershell
gh pr list --state open --limit 50
gh release list --limit 20
node scripts\sync_public_release_links.mjs --check
python scripts\audit_public_surface.py
powershell -ExecutionPolicy Bypass -File scripts\check_public_export.ps1
```

The public release PR should state which GitHub release, tag, status doc, and
artifact entry are the source of truth.

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
- CODEOWNERS routing for public boundary, release, workflow, and docs changes
- Dependabot or dependency-update PRs that may affect the public build
- Pages deployment target
- public status docs and release links
- public release manifest entries under `releases/`, when artifacts or claims
  change

## Public-State Rules

- Keep release metadata and assets aligned with `docs/VERSION.json`.
- Keep `PUBLIC_RELEASE_CHECKLIST.md` as the release intake gate.
- Keep `.github/CODEOWNERS` routing public boundary, workflow, docs, Worker, and
  crate changes to the public repo owner.
- Keep public artifact metadata aligned with
  `releases/public-release-manifest.schema.json` when a release publishes or
  changes artifacts, checksums, signatures, or release status claims.
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

For any future public artifact or training-result release, verify the live
GitHub state before opening the final release PR and again after merge:

```powershell
gh pr list --state open --limit 50
gh release list --limit 20
node scripts\audit_public_github_state.mjs
node scripts\sync_public_release_links.mjs --check
node scripts\validate_public_release_state.mjs
node scripts\audit_public_secrets.mjs
python scripts\audit_public_surface.py
powershell -ExecutionPolicy Bypass -File scripts\check_public_export.ps1
```

Run `node scripts\audit_public_github_state.mjs` before opening the final
release PR, and run it again after merge before publishing or marking a GitHub
release as current. The live audit intentionally fails when open pull requests
or extra remote branches are present, because the public release state should be
settled before it becomes the current public truth.

The public release PR should state which GitHub release, tag, status doc,
manifest, and artifact entry are the source of truth.

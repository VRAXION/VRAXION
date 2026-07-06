# Changelog

## 2026-07-06

- Guarded release-link sync coverage, contributor gates, security policy, and
  deployment runbook markers in the public surface audit.
- Added explicit public deployment guidance for generated Wrangler config,
  `.dev.vars`, real D1 ids, Worker secrets, API tokens, and operator output.
- Added Worker local config hygiene notes and audit coverage so generated
  operator config and export/delete output stay out of the public repo.
- Hardened public workflow hygiene with CI concurrency controls, job timeouts,
  and audit markers for workflow drift.
- Added a public security.txt endpoint and audit coverage for vulnerability disclosure routing.
- Added live security.txt smoke coverage for the public Pages disclosure endpoint.
- Replaced internal runtime wording in the public crate with operator-side wording.
- Added a public link audit for repo-local and Pages-local documentation links.
- Re-verified the public export guard, live Pages state, public link smoke, and
  main GitHub Actions after each public hardening merge.

## 2026-07-05

- Polished the public Pages surface and 404 fallback while keeping unreleased
  routes hidden.
- Added repo hygiene rules for local build output, local Codex state, secret
  env files, and generated Cloudflare config.
- Added `.gitattributes` and stronger public surface audit checks for tracked
  repo hygiene files.
- Reworked the root README into a clearer public release intake entrypoint.
- Added `PUBLIC_RELEASE_CHECKLIST.md` for future release PRs.
- Added a public issue intake form and guard checks that keep reports inside
  the visible public surface.
- Added `SUPPORT.md` to route public reports, release questions, and security
  reports without exposing private material.
- Added `PUBLIC_GITHUB_STATE.md` to define current public GitHub release, tag,
  asset, and Pages review rules.
- Added Dependabot maintenance for the public Cargo workspace and GitHub
  Actions surface.
- Added `.github/CODEOWNERS` routing for public boundary, workflow, docs,
  Worker, and crate review.
- Added a public release manifest schema and example under `releases/` for
  future artifact/checksum/signature intake.
- Added a public release manifest validator and CI gate for future release
  artifact intake.
- Added a public release state validator and CI gate to keep the public version
  record, GitHub release pointer, status docs, and Pages copy aligned.
- Added a public secret scan guard for tracked files before release intake.
- Required release manifests to list the full public guard command set,
  including release-state validation and secret scanning.
- Locked public GitHub Actions workflows to read-only repository permissions
  and added audit coverage for workflow permission drift.
- Added a live public GitHub state audit for pre-release checks against open
  pull requests, remote branches, default branch, and the latest GitHub release.
- Extended public release verification to require live Pages source, build,
  URL, and `VERSION.json` checks before release publication.
- Hardened the live public Pages smoke timeout and retry path for GitHub
  archive redirects.
- Added a reviewed manifest for the current `public-sdk-p11-20260629` public
  SDK/docs release and required latest-release manifest coverage.

## 2026-06-29

- Added public P11 delivery, license, and trademark boundary summaries.
- Kept the public repository limited to the two-crate SDK source boundary.
- Updated public Pages status to reflect controlled early-access delivery
  planning without publishing private engine source.

## 2026-06-28

- Reset current public tree to the SDK and documentation boundary.
- Removed stale current-tree research pages and operational runtime surfaces.
- Kept historical commits, branches, tags, and releases as archives.

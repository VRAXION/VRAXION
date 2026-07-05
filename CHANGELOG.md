# Changelog

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

## 2026-06-29

- Added public P11 delivery, license, and trademark boundary summaries.
- Kept the public repository limited to the two-crate SDK source boundary.
- Updated public Pages status to reflect controlled early-access delivery
  planning without publishing private engine source.

## 2026-06-28

- Reset current public tree to the SDK and documentation boundary.
- Removed stale current-tree research pages and operational runtime surfaces.
- Kept historical commits, branches, tags, and releases as archives.

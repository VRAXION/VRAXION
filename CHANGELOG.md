# Changelog

## 2026-08-06

- Scheduled cleanup task run: reconciled the task's transform-heavy premise
  (branch consolidation, spaghetti-code cleanup, archive-then-delete of
  unneeded files, release execution) against the current repository state,
  and produced a verification pass only. The premise no longer matches:
  branch consolidation completed on 2026-07-30, the `.gitignore` extension
  landed on 2026-08-02, and the 2026-08-05 hygiene entry recorded a full
  green public verification cycle. The tracked public tree at
  `origin/main` tip `b1ad1587` differs from the 2026-08-05-verified tip
  `213f8f68` only by that CHANGELOG entry itself.
- Verified branch topology: only `main` exists locally and on origin. All
  historical research branches remain preserved as `archive/branches/*`
  annotated tags on origin from the 2026-07-30 consolidation. No open pull
  requests. Two worktrees still attached (`S:/Git/VRAXION` on
  detached-HEAD `213f8f68` and `S:/Git/VRAXION_anchorwiki` on `main` at
  `b1ad1587`); left intact.
- Verified public GitHub state via `audit_public_github_state.mjs`:
  `default_branch=main`, `origin_main_commit=b1ad158702d8f48c48ec0201ecc5c82e3e508013`,
  `latest_public_release=public-sdk-p11-20260629`, `pages_latest_build_commit`
  matches `origin_main_commit`, live Pages `VERSION.json` still points to
  `public-sdk-p11-20260629`, `open_pull_request_count=0`.
- Guard results at `b1ad1587`: 13 of 14 public guards pass on the primary
  worktree (`validate_public_release_manifests`,
  `validate_public_release_state`, `sync_public_release_links --check`,
  `audit_public_github_state`, `audit_public_links` (127 tracked files,
  36 scanned), `audit_public_secrets` (103 scanned text files),
  `audit_public_surface`, `audit_instnct_static_site`,
  `audit_instnct_notify_worker`, `smoke_public_pages_links`,
  `cargo fmt --check`, `cargo test --workspace` (22 passed),
  `cargo clippy --workspace --all-targets --all-features -D warnings`).
  `check_public_export.ps1` was not re-run to green in the scheduled-task
  sandbox: `[System.IO.Path]::GetTempPath()` resolved to a non-C drive
  letter that Git Bash `tar` (used inside the runtime crate bundle step)
  interprets as an SSH `host:path` prefix, and the C:\Windows\Temp
  fallback failed on mingw linker temp-file references. Environment
  limitation, not a repo defect; the tracked public tree is byte-identical
  to `213f8f68` outside of the CHANGELOG (see the `git diff --stat` cited
  above), which passed `check_public_export.ps1` on a clean detached-HEAD
  worktree in the 2026-08-05 hygiene pass.
- Verified the Archive-Notice claim on the public GitHub wiki: the
  referenced archive branch `archive/pre-p10-2-wiki-zero-state-20260628`
  still exists on `VRAXION/VRAXION.wiki.git`, and the
  `archive/wiki/pre-consolidation-2026-06-13` tag remains intact. The
  public wiki keeps its intentional four-page boundary-stub set (`Home`,
  `Public-Boundary`, `Archive-Notice`, `_Sidebar`) with `master` as the
  wiki repo default branch (GitHub wiki idiom, distinct from the main
  repo's `main` default).
- No public-surface change and no version bump. Following the precedent
  set by 2026-08-02 and 2026-08-05, no release tag or version bump is
  warranted because no public-surface change occurred. `PUBLIC_GITHUB_STATE.md`
  defines the release trigger as reviewed public artifact, status claim,
  manifest, or `docs/VERSION.json` changes; a verification pass with no
  tracked-tree change outside the CHANGELOG does not meet that trigger.

## 2026-08-05

- Recorded Dependabot bumps that landed on `origin/main` on 2026-08-03 for
  audit-trail continuity: `serde` 1.0.228 → 1.0.229 and `serde_json` 1.0.150 →
  1.0.151 in the `rust-public-dependencies` group (patch), and
  `actions/setup-node` v6 → v7 and `actions/setup-python` v6 → v7 in the
  `public-github-actions` group (major GHA runner action bumps, no workflow
  behavior change beyond the newer runner defaults).
- Ran the full public verification suite from a clean detached-HEAD worktree
  against `origin/main` (`213f8f68`) and confirmed all guards pass:
  `validate_public_release_manifests`, `validate_public_release_state`,
  `sync_public_release_links --check`, `audit_public_github_state`,
  `audit_public_links`, `audit_public_secrets`, `audit_public_surface`,
  `audit_instnct_static_site`, `audit_instnct_notify_worker`,
  `smoke_public_pages_links`, `cargo fmt --check`, `cargo test --workspace`
  (22 passed), `cargo clippy --workspace --all-targets --all-features -D
  warnings`, and `check_public_export.ps1` (clean worktree).
- Verified the zero-untracked invariant established on 2026-08-02 still holds
  (`git status --porcelain` returns empty; ignored directories are the ten
  local research surfaces already documented in `.gitignore`).
- Fast-forwarded the `VRAXION_anchorwiki` sibling worktree's `main` from
  `9065c41d` to `213f8f68` (was 4 commits behind `origin/main`, tree clean).
- Reviewed the public GitHub wiki at `VRAXION/VRAXION.wiki.git`: current state
  is the intentional four-page boundary-stub set (`Home`, `Public-Boundary`,
  `Archive-Notice`, `_Sidebar`, 43 lines total) last reset on
  `Reset public wiki to boundary stubs`; no wiki changes required.
- Cross-checked version records: `docs/VERSION.json`,
  `releases/public-sdk-p11-20260629.manifest.json`, `README.md`,
  `PUBLIC_GITHUB_STATE.md`, `docs/CURRENT_STATUS.md`, and
  `docs/CURRENT_CAPABILITIES.md` all agree on `public-sdk-p11-20260629` as
  the current public release.
- No changes to the public SDK crates, Pages surface, GitHub Actions
  workflows, release manifests, `docs/VERSION.json`, or any boundary or
  policy document; this entry records a repository hygiene pass and a full
  verification cycle, not a new public delivery. No release tag or version
  bump is warranted because no public-surface change occurred.

## 2026-08-02

- Added `.claude/` to `.gitignore`. This directory holds per-user Claude Code
  agent state (settings, worktrees, hive scratch, research swarm logs, and the
  `scheduled_tasks.lock` file) and was the last untracked directory not
  covered by `.gitignore` after the 2026-07-31 consolidation pass. Working
  trees with an active Claude Code session now report zero untracked entries
  under `git status --ignored`.
- Deleted the empty local branch `cleanup/e75-doc-align-20260613`. Its tip
  `092233a6` was already reachable from `main`, and the branch was never
  pushed to origin. Preserved for audit-trail symmetry as annotated tag
  `archive/branches/2026-08-02/cleanup-e75-doc-align-20260613`.
- Verified archive-tag safety: every local-only `archive/*` tag either points
  to a commit already present on `origin/main` or duplicates a commit that is
  backed up under a different annotated tag on origin. The three
  `frontier-e136t-private` tags remain local-only by the existing publication
  policy.
- No changes to the public SDK crates, Pages surface, GitHub Actions
  workflows, release manifests, or `docs/VERSION.json`; this entry records a
  repository hygiene pass, not a new public delivery.

## 2026-07-31

- Fixed a latent CI break: added `NOTICE` and `OWNERSHIP.md` to the public
  export allowlist in `scripts/check_public_export.ps1`. Both files were added
  in the early-July ownership commits but the allowlist was not updated at the
  same time, so `Public SDK CI` on `main` had been failing on every push and
  pull request since 2026-07-12 with `unexpected public export file: NOTICE`.
- Documented VRAXION founder mark ownership, ownership notice, and future
  assignment boundary in `README.md` and `TRADEMARK_POLICY.md`.
- Preserved every local research branch tip as `archive/branches/2026-07-30/*`
  annotated tags on origin (48 tags pushed). Kept `frontier/e136t-private` and
  `archive/pre-consolidation-20260519-main-snapshot` tags local-only per the
  existing publication policy (no matching tags previously existed on origin).
- Reduced local branch inventory from 50 to 7 (`main` plus 6 research branches
  still attached to active worktrees with on-disk experiment state); every
  deleted branch tip is recoverable from an annotated archive tag.
- Removed two empty worktrees (`VRAXION-main-publish`,
  `VRAXION_public_cleanup_001`) and left the six worktrees holding active
  local research state untouched.
- Fast-forwarded local `main` to `origin/main` (was 268 commits behind).
- Captured a full-repo `git bundle --all` backup before any deletion.
- No changes to the public SDK crates, Pages surface, GitHub Actions
  workflows, release manifests, or `docs/VERSION.json`; this entry records a
  repository consolidation pass plus one export-guard fix, not a new public
  delivery.

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
- Synced the README public gate list and scrubbed remaining internal crate wording.
- Added repo-relative artifact checksum verification to the public release manifest validator.
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

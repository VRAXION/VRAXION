# Contributing

This repository is a narrow public SDK/docs surface. Contributions should stay
inside the visible public review surface.

Before opening a pull request, run:

```powershell
node scripts\sync_public_release_links.mjs --check
node scripts\validate_public_release_manifests.mjs
node scripts\validate_public_release_state.mjs
node scripts\audit_public_secrets.mjs
node scripts\audit_instnct_static_site.mjs
node scripts\audit_instnct_notify_worker.mjs
python scripts\audit_public_surface.py
node scripts\smoke_public_pages_links.mjs
cargo fmt --all -- --check
cargo test --workspace --all-features
cargo test --doc --workspace --all-features
cargo clippy --workspace --all-targets --all-features -- -D warnings
powershell -ExecutionPolicy Bypass -File scripts/check_public_export.ps1
```

Do not add private data, operational logs, private engine source, or diagnostic
tools to this repository.

Release PRs must also follow `PUBLIC_RELEASE_CHECKLIST.md`. If a release
changes public status, artifacts, downloads, or claims, state exactly what
became public and what stayed private in the PR body.
For release-state PRs, also run `node scripts\audit_public_github_state.mjs`
before opening the final PR and again after merge.

Use the GitHub pull request template. Do not delete the public scope,
release-intake, or required-gates sections from release or public-surface PRs.

Use the GitHub issue form for public reports. Do not put secrets, private code,
non-public training data, absolute local or UNC paths, raw operator output,
production config, or private dashboards in public issues. Use `SECURITY.md` for
vulnerabilities or reports with security impact. Support routing is summarized
in `SUPPORT.md`.

Dependency update PRs are public-surface PRs. Keep them limited to the public
Cargo workspace and GitHub Actions unless a separate release review adds a new
public manifest or ecosystem.

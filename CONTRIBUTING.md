# Contributing

This repository is a narrow public SDK/docs surface. Contributions should stay
inside the visible public review surface.

Before opening a pull request, run:

```powershell
node scripts\sync_public_release_links.mjs --check
node scripts\audit_instnct_static_site.mjs
node scripts\audit_instnct_notify_worker.mjs
cargo fmt --all -- --check
cargo test --workspace --all-features
cargo test --doc --workspace --all-features
cargo clippy --workspace --all-targets --all-features -- -D warnings
python scripts/audit_public_surface.py
powershell -ExecutionPolicy Bypass -File scripts/check_public_export.ps1
```

Do not add private data, operational logs, private engine source, or diagnostic
tools to this repository.

Release PRs must also follow `PUBLIC_RELEASE_CHECKLIST.md`. If a release
changes public status, artifacts, downloads, or claims, state exactly what
became public and what stayed private in the PR body.

Use the GitHub pull request template. Do not delete the public scope,
release-intake, or required-gates sections from release or public-surface PRs.

Use the GitHub issue form for public reports. Do not put secrets, private code,
non-public training data, local machine paths, raw operator output, production
config, or private dashboards in public issues. Use `SECURITY.md` for
vulnerabilities or reports with security impact. Support routing is summarized
in `SUPPORT.md`.

Dependency update PRs are public-surface PRs. Keep them limited to the public
Cargo workspace and GitHub Actions unless a separate release review adds a new
public manifest or ecosystem.

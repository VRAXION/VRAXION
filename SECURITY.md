# Security Policy

Report security issues privately by opening a GitHub security advisory or by
contacting the project maintainers through the repository owner profile.

Please do not publish exploit details before maintainers have had time to
investigate and respond.

## Public Scope

This public repository currently includes:

- the public-candidate Rust crates
- the GitHub Pages documentation surface
- the INSTNCT notify Worker source and migration
- CI, audit, deployment, and release-intake documentation

The private engine implementation, non-public training data, private run
infrastructure, and operator-only workspace are outside this repository.

## Reporting Checklist

When reporting a security issue, include:

- affected public file, crate, page, or Worker route
- minimal reproduction steps using public inputs
- expected impact
- whether any secret, token, key, or private data exposure is suspected

Do not include live secrets in reports. If a report involves credentials,
describe the credential type and where it was observed without pasting the
credential value.

## Release Handling

Security-sensitive release changes should follow `PUBLIC_RELEASE_CHECKLIST.md`
and keep filled production config files untracked. Public release notes should
describe impact and remediation without exposing operational details that would
increase risk before users can update.

Run `node scripts\audit_public_secrets.mjs` before release PRs that touch
configuration, Worker code, release artifacts, or generated files.

# VRAXION

VRAXION is the public release home for the current SDK and documentation
surface. The repository is intentionally small: it keeps the public crates,
Pages documentation, release notes, CI, and audit scripts in one reviewable
tree while private engine development continues elsewhere.

This repo is prepared to receive a future public release after training work
produces a reviewed artifact. Until then, it is a compatibility and
documentation surface, not a full product release and not a hosted availability
promise.

## Current Public State

| Area | Status | Notes |
| --- | --- | --- |
| Rust crates | Public-candidate, not published | `alphasync-core` and `alphasync-runtime` are included with `publish = false`. |
| Pages docs | Published public docs | The public site, INSTNCT preview, AnchorCell schema page, status docs, and capability notes live under `docs/`. |
| Release artifact | Pending review | No signed T1 binary, reproduction harness, raw timing logs, or production service is published here. |
| Notify Worker | Operator deploy path | Cloudflare Worker code and migration are present; production deploy requires configured secrets. |

## Repository Map

- `alphasync-core`
- `alphasync-runtime`
- `docs/`
- `workers/instnct-notify/`
- `scripts/`

The private engine implementation, non-public training data, operational run
logs, non-public diagnostics, persistence services, and unreleased research
workspaces are not published here.

## Release Intake

Before any new public release is added, the release should have:

1. explicit release notes and versioned status docs
2. reviewed artifact naming, checksums, or signatures when an artifact exists
3. a reviewed `releases/<release-slug>.manifest.json` when public artifacts,
   checksums, signatures, or status claims change
4. an updated `docs/VERSION.json` and matching public page links
5. no private engine code, private data, absolute local or UNC paths, secrets,
   or internal run output
6. green CI and a passing public export guard

Use `PUBLIC_RELEASE_CHECKLIST.md` as the release PR checklist before adding a
new artifact, status claim, or public download path. Use
`releases/public-release-manifest.schema.json` as the public artifact manifest
contract. The current public release manifest is
`releases/public-sdk-p11-20260629.manifest.json`.

## Build

```powershell
cargo fmt --all -- --check
cargo test --workspace --all-features
cargo test --doc --workspace --all-features
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo doc --workspace --no-deps
node scripts/validate_public_release_manifests.mjs
node scripts/validate_public_release_state.mjs
node scripts/sync_public_release_links.mjs --check
node scripts/audit_public_github_state.mjs
node scripts/audit_public_links.mjs
node scripts/audit_public_secrets.mjs
node scripts/audit_instnct_static_site.mjs
node scripts/audit_instnct_notify_worker.mjs
python scripts/audit_public_surface.py
node scripts/smoke_public_pages_links.mjs
powershell -ExecutionPolicy Bypass -File scripts/check_public_export.ps1
```

## Public Surface

The public surface is documented in `PUBLIC_BOUNDARY.md` and
`PACKAGE_BOUNDARY.md`. The public delivery direction is documented in
`PUBLIC_DELIVERY_MODEL.md`, with license and mark boundaries in
`LICENSE_BOUNDARY.md` and `TRADEMARK_POLICY.md`. Release intake rules are in
`PUBLIC_RELEASE_CHECKLIST.md`, public GitHub state rules are in
`PUBLIC_GITHUB_STATE.md`, public release manifest rules are in `releases/`,
and public support routing is in `SUPPORT.md`.

Current status is tracked by the latest public GitHub release:
[`public-sdk-p11-20260629`](https://github.com/VRAXION/VRAXION/releases/tag/public-sdk-p11-20260629).

The delivery direction is proof materials before broader runtime terms. Any
future binary, API, hosted service, or wrapper path requires separate release
review. The private engine implementation is not published here.

## License

See `LICENSE`, `LICENSE_BOUNDARY.md`, and `TRADEMARK_POLICY.md`.

## Ownership

Copyright in public VRAXION source, documentation, diagrams, and releases
follows the repository license, file notices, contributor history, and written
agreements.

VRAXION™, the VRAXION logo, AlphaSync™, and INSTNCT™ are currently claimed as
unregistered marks of Daniel Kenessy, acting in an individual capacity. No
trademark rights are granted by the source license or repository access.

No rights have been assigned to a company. Any future company transfer
requires a separate signed assignment and must preserve contributor and
third-party rights. See `NOTICE`, `OWNERSHIP.md`, and `TRADEMARK_POLICY.md`.

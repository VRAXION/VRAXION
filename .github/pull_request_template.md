## Public Scope

- What exactly becomes public?
- What stays private?
- Which user-facing claim changes, if any?

## Release Intake

- [ ] This PR does not add private engine source, non-public training data, raw operator output, absolute local or UNC paths, secrets, or filled production config.
- [ ] `PUBLIC_RELEASE_CHECKLIST.md` was followed, or this PR does not change release status, artifacts, downloads, or public claims.
- [ ] Any generated local config, including `workers/instnct-notify/wrangler.jsonc`, remains untracked.
- [ ] Any new artifact has public-safe provenance, naming, and checksum or signature material where applicable.
- [ ] Any `artifact_release` or `proof_pack` manifest includes the required published artifact, checksum, and signature fields.
- [ ] Any artifact, checksum, signature, or release status claim change has a reviewed public manifest that follows `releases/public-release-manifest.schema.json`.
- [ ] `PUBLIC_GITHUB_STATE.md` was checked for release, tag, asset, Pages, and branch-state changes, or this PR does not change public release state.

## Required Public Gates

- [ ] `node scripts\sync_public_release_links.mjs --check`
- [ ] `node scripts\validate_public_release_manifests.mjs`
- [ ] `node scripts\validate_public_release_state.mjs`
- [ ] `node scripts\audit_public_github_state.mjs` was run before opening this release PR, or this PR does not change public release state.
- [ ] `node scripts\audit_public_secrets.mjs`
- [ ] `node scripts\audit_instnct_static_site.mjs`
- [ ] `node scripts\audit_instnct_notify_worker.mjs`
- [ ] `python scripts\audit_public_surface.py`
- [ ] `node scripts\smoke_public_pages_links.mjs`
- [ ] `powershell -ExecutionPolicy Bypass -File scripts\check_public_export.ps1`

## Notes

- Source-of-truth release, artifact, checksum, or docs page:
- Release manifest checked:
- GitHub release or tag checked:
- Intentional unchanged release files:

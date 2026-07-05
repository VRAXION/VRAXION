## Public Scope

- What exactly becomes public?
- What stays private?
- Which user-facing claim changes, if any?

## Release Intake

- [ ] This PR does not add private engine source, non-public training data, raw operator output, local paths, secrets, or filled production config.
- [ ] `PUBLIC_RELEASE_CHECKLIST.md` was followed, or this PR does not change release status, artifacts, downloads, or public claims.
- [ ] Any generated local config, including `workers/instnct-notify/wrangler.jsonc`, remains untracked.
- [ ] Any new artifact has public-safe provenance, naming, and checksum or signature material where applicable.

## Required Public Gates

- [ ] `node scripts\sync_public_release_links.mjs --check`
- [ ] `node scripts\audit_instnct_static_site.mjs`
- [ ] `node scripts\audit_instnct_notify_worker.mjs`
- [ ] `python scripts\audit_public_surface.py`
- [ ] `node scripts\smoke_public_pages_links.mjs`
- [ ] `powershell -ExecutionPolicy Bypass -File scripts\check_public_export.ps1`

## Notes

- Source-of-truth release, artifact, checksum, or docs page:
- Intentional unchanged release files:

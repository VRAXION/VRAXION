# Public Release Checklist

Use this checklist before adding any new public release material to this
repository. The goal is simple: publish only reviewed public artifacts and keep
private engine work, non-public training data, absolute local or UNC paths,
secrets, and raw operator output out of the public tree.

## Intake Scope

Every release PR should identify:

- release name and date
- artifact type, if any
- exact public files changed
- expected public claim
- evidence files or checksums that support the claim
- owner who reviewed the public/private split

If a release does not include a reviewed artifact, say that directly in the
release notes and status docs. Do not imply hosted availability, binary
availability, production service availability, or benchmark completion before
those items are reviewed and present.

## Required Updates

Before merging a release PR, update or verify:

- `README.md`
- `CHANGELOG.md`
- `docs/VERSION.json`
- `docs/CURRENT_STATUS.md`
- `docs/CURRENT_CAPABILITIES.md`
- `PUBLIC_GITHUB_STATE.md`
- `PUBLIC_BOUNDARY.md`
- `PACKAGE_BOUNDARY.md`
- `PUBLIC_DELIVERY_MODEL.md`
- relevant Pages copy under `docs/`
- release notes on GitHub, when a GitHub release is created
- GitHub tag, release body, release assets, and Pages deployment state
- `releases/<release-slug>.manifest.json` when artifacts, checksums,
  signatures, or release status claims change

If a file is intentionally unchanged, note why in the PR body.

## Private Material Exclusion

The release PR must not include:

- private engine source or internals
- non-public training data
- raw experiment logs or operator run output
- absolute local or UNC machine paths
- secrets, tokens, keys, or filled production config
- private data adapters
- skill persistence stores
- diagnostic parity tools
- unreleased draft pages under `docs/`

Generated local config such as `workers/instnct-notify/wrangler.jsonc` must stay
untracked. Use `workers/instnct-notify/wrangler.example.jsonc` for public config
shape.

## Artifact Rules

If a public artifact is added:

- name it with a stable release slug
- include checksum or signature material when applicable
- use `artifact_release` only when the manifest includes a published
  non-documentation artifact
- use `proof_pack` only when the manifest includes a published `proof_pack`
  artifact
- include SHA-256 for every published non-documentation artifact
- include `signature_path_or_url` for every published `proof_pack` or `binary`
  artifact
- add a public release manifest that follows
  `releases/public-release-manifest.schema.json`
- document how the artifact was produced at a public-safe level
- keep reproduction instructions limited to public inputs
- verify archive contents before upload
- link from status docs only after the artifact exists

If an artifact is not added:

- keep the release language as preview, compatibility, docs, or status only
- either omit the release manifest or include one with an empty `artifacts`
  list and a plain `public_claim` that says no runnable artifact is included
- do not add download language
- do not add benchmark-complete language
- do not add production availability language

## Required Gates

Run the live GitHub state audit before opening the final release PR. Run the
remaining gates before opening and before merging the release PR:

```powershell
node scripts\audit_public_github_state.mjs
node scripts\sync_public_release_links.mjs --check
node scripts\validate_public_release_manifests.mjs
node scripts\validate_public_release_state.mjs
node scripts\audit_public_secrets.mjs
node scripts\audit_instnct_static_site.mjs
node scripts\audit_instnct_notify_worker.mjs
python scripts\audit_public_surface.py
node scripts\smoke_public_pages_links.mjs
powershell -ExecutionPolicy Bypass -File scripts\check_public_export.ps1
```

Run the live GitHub state audit before opening the final release PR, and run it
again after merge before publishing or marking a GitHub release as current. It
checks that the public repository, default branch, open pull requests, remote
branches, latest GitHub release, Pages source, latest Pages build, and live
`VERSION.json` agree with `docs/VERSION.json`.

For Worker changes, also run the notify live smoke against the intended target
environment before enabling any public form path.

## Final Review

The final PR review should answer:

- What exactly became public?
- What stayed private?
- Which user-facing claim changed?
- Which tests or audits prove the public tree is clean?
- Which artifact, checksum, or docs page is the source of truth?
- Which release manifest, if any, defines the public artifact set?
- Which GitHub release or tag is the current public source of truth?

Do not merge if any answer depends on memory, local-only files, or unpublished
private workspace state.

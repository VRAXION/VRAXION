# Public Release Manifests

This directory defines the manifest shape for current and future public release
PRs.

Use `public-release-manifest.schema.json` as the review contract and
`public-release-manifest.example.json` as the copy-start example. The current
public release is recorded in `public-sdk-p11-20260629.manifest.json`. A future
release that adds a public artifact should include a reviewed manifest with the
release slug, public claim, public files, artifact checksums or signatures,
verification commands, and explicit exclusions.

The manifest is intentionally public-safe. It should describe only reviewed
public files, public URLs, public checks, and public artifact metadata.

Artifact-bearing releases have stricter gates:

- `artifact_release` manifests must include at least one published
  non-documentation artifact.
- `proof_pack` manifests must include at least one published `proof_pack`
  artifact.
- Every published non-documentation artifact must include a lowercase SHA-256
  checksum.
- Published `proof_pack` and `binary` artifacts must also include
  `signature_path_or_url`.

`validate_public_release_manifests.mjs` checks both manifest content and the
schema contract, so these artifact gates must stay encoded in
`public-release-manifest.schema.json`. It also runs in-memory policy self-tests
that must reject representative bad `artifact_release`, `proof_pack`, checksum,
signature, and schema-drift cases before reporting `policy_self_tests`.

Validate manifests before review:

```powershell
node scripts\audit_public_github_state.mjs
node scripts\sync_public_release_links.mjs --check
node scripts\validate_public_release_manifests.mjs
node scripts\validate_public_release_state.mjs
node scripts\audit_public_secrets.mjs
python scripts\audit_public_surface.py
node scripts\smoke_public_pages_links.mjs
powershell -ExecutionPolicy Bypass -File scripts\check_public_export.ps1
```

Do not include:

- private engine source
- non-public training data
- raw operator output
- absolute local or UNC machine paths
- secrets or filled production config
- private dashboard links

Suggested future naming:

```text
releases/<release-slug>.manifest.json
```

Do not link a manifest from status docs until the GitHub release, tag, assets,
and Pages copy all agree with it.

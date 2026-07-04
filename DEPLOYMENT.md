# Public deployment runbook

This repository currently publishes the public static site through GitHub Pages from `docs/`. The INSTNCT notify backend is a separate Cloudflare Worker because GitHub Pages cannot serve API routes.

## Static site

Source:

- `docs/index.html`
- `docs/instnct/`
- `docs/vngard/`
- `docs/robots.txt`
- `docs/sitemap.xml`

Deployment target:

- `https://vraxion.github.io/VRAXION/`

Required local gates before pushing public site changes:

```powershell
node scripts\sync_public_release_links.mjs --check
node scripts\audit_instnct_static_site.mjs
python scripts\audit_public_surface.py
node scripts\smoke_instnct_browser.mjs
.\scripts\check_public_export.ps1
```

Required live gate after Pages deploy:

```powershell
node scripts\smoke_public_pages_links.mjs
```

## INSTNCT notify Worker

Source:

- `workers/instnct-notify/src/index.mjs`
- `workers/instnct-notify/migrations/`
- `workers/instnct-notify/wrangler.example.jsonc`

GitHub workflow:

- `.github/workflows/deploy-instnct-notify.yml`

Required GitHub secrets:

- `CLOUDFLARE_API_TOKEN`
- `CLOUDFLARE_ACCOUNT_ID`
- `INSTNCT_NOTIFY_D1_DATABASE_ID`
- `INSTNCT_NOTIFY_EMAIL_HASH_PEPPER`
- `INSTNCT_NOTIFY_ADMIN_TOKEN`
- `INSTNCT_NOTIFY_API_BASE`

Manual Cloudflare setup:

```powershell
cd workers\instnct-notify
npx wrangler login
npx wrangler d1 create vraxion-instnct-notify
```

Copy the returned D1 database id into the GitHub secret `INSTNCT_NOTIFY_D1_DATABASE_ID`. Use long random values for `INSTNCT_NOTIFY_EMAIL_HASH_PEPPER` and `INSTNCT_NOTIFY_ADMIN_TOKEN`.

Manual deploy:

```powershell
cd workers\instnct-notify
copy wrangler.example.jsonc wrangler.jsonc
npx wrangler d1 migrations apply vraxion-instnct-notify --remote
npx wrangler secret put EMAIL_HASH_PEPPER
npx wrangler secret put ADMIN_TOKEN
npx wrangler deploy
```

Automated deploy:

1. Open the `Deploy INSTNCT Notify Worker` workflow.
2. Run it manually from `main`.
3. Confirm the workflow ran migrations, deployed the Worker with the scheduled rate-limit cleanup trigger, and passed live smoke.

Post-deploy live smoke:

```powershell
$env:INSTNCT_NOTIFY_API_BASE = "https://vraxion-instnct-notify.<account>.workers.dev"
node scripts\smoke_instnct_notify_live.mjs
```

Use write-mode only when intentionally validating durable storage:

```powershell
$env:INSTNCT_NOTIFY_SMOKE_WRITE = "1"
node scripts\smoke_instnct_notify_live.mjs
```

Operator checks after write-mode smoke:

```powershell
$env:INSTNCT_NOTIFY_ADMIN_TOKEN = "<admin token>"
curl.exe -H "Authorization: Bearer $env:INSTNCT_NOTIFY_ADMIN_TOKEN" "$env:INSTNCT_NOTIFY_API_BASE/admin/notify/export"
curl.exe -X POST "$env:INSTNCT_NOTIFY_API_BASE/admin/notify/cleanup-rate-limits" `
  -H "Authorization: Bearer $env:INSTNCT_NOTIFY_ADMIN_TOKEN" `
  -H "content-type: application/json" `
  --data "{\"olderThanHours\":48}"
```

## Frontend form switch

Do not add an active email form to `docs/instnct/index.html` until all of these are true:

- Worker URL is stable and stored in `INSTNCT_NOTIFY_API_BASE`.
- D1 migrations have been applied remotely.
- `EMAIL_HASH_PEPPER` is set in Cloudflare.
- `ADMIN_TOKEN` is set in Cloudflare and operator export/delete/cleanup has been checked.
- Scheduled rate-limit cleanup is visible in the deployed Worker triggers.
- `scripts/smoke_instnct_notify_live.mjs` passes in read-only mode.
- Write-mode smoke has been run once intentionally.
- The INSTNCT CSP has been changed from `connect-src 'none'; form-action 'none'` to the exact Worker origin.
- `scripts/audit_instnct_static_site.mjs` has been updated from forbidding the form to requiring the safe configured form.

Until then, the public page must keep the link-only release tracking block so users do not submit into a dead endpoint.

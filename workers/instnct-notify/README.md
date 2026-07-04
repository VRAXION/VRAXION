# INSTNCT notify Worker

This Worker is the production backend target for the INSTNCT release signal form. It is kept separate from the static GitHub Pages site because GitHub Pages cannot serve `/api/notify`.

## Endpoints

- `GET /health` returns a simple health response.
- `GET /api/notify` returns the current subscriber count and goal.
- `POST /api/notify` accepts JSON `{ "email": "you@example.com", "source": "instnct-site", "website": "" }`.
- `OPTIONS /api/notify` handles browser preflight requests.

The `website` field is a honeypot. If it is filled, the Worker returns an accepted response without storing the submission.

## Storage

The Worker uses one D1 binding named `DB`.

Run setup from this folder:

```powershell
copy wrangler.example.jsonc wrangler.jsonc
npx wrangler d1 create vraxion-instnct-notify
```

Paste the returned D1 `database_id` into `wrangler.jsonc`, then apply migrations and deploy:

```powershell
npx wrangler d1 migrations apply vraxion-instnct-notify --remote
npx wrangler secret put EMAIL_HASH_PEPPER
npx wrangler deploy
```

Keep `EMAIL_HASH_PEPPER` out of source control. Use a long random value and rotate it only with a planned duplicate-handling migration.

## Required config

- `ALLOWED_ORIGIN`: comma-separated browser origins allowed to call the API. Default target: `https://vraxion.github.io`.
- `SUBSCRIBER_GOAL`: milestone count shown by the frontend. Default: `1000`.
- `RATE_LIMIT_PER_HOUR`: accepted submissions per hashed client per hour. Default: `20`.
- `EMAIL_HASH_PEPPER`: Worker secret used to hash email, IP, and user-agent values for duplicate/rate-limit keys.

Do not wire the static INSTNCT page to this endpoint until the Worker is deployed and the live smoke covers `GET`, CORS, invalid email, blocked origin, honeypot behavior, and at least one intentional write-mode pass. Write-mode also checks duplicate email handling; set `INSTNCT_NOTIFY_SMOKE_RATE_LIMIT=1` only when intentionally validating live rate-limit behavior because it consumes the caller's hourly limit.

# VRAXION SVG Labels (Badges)

This folder holds design notes/templates for the **self-hosted SVG label system**.

## Rules (repo-wide)

- **Wiki uses `mono`** badges (minimal, readable).
- **Pages uses `neon`** badges (branding / eye-candy).
- **README uses `mono`** by default.

## Do / Don’t

**Do**
- Use labels for **document type** and **contracts** (canonical/protocol/evidence required).
- Keep label text **evergreen** (should remain true for months).
- Keep rows small: **≤ 6** labels per page section.

**Don’t**
- Don’t encode live status in labels: avoid `ACTIVE`, `PASSING`, `LATEST`, dates, version numbers, or numeric claims.
- Don’t depend on external badge providers in the wiki (e.g. `shields.io`, `zenodo.org/badge`).
- Don’t use animations or external references inside SVGs.

## File layout

- `docs/assets/badges/mono/*.svg`
- `docs/assets/badges/neon/*.svg`

File name is the badge ID (lowercase snake_case), e.g. `canonical.svg`.


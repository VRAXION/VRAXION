# VRAXION

This repository is the clean public SDK and documentation boundary for VRAXION.
It contains a narrow, source-available Rust workspace:

- `alphasync-core`
- `alphasync-runtime`

The private engine and operational research workspace are not published here.
Older public commits and releases remain historical archives, but the current
tree is intentionally reset to a minimal SDK surface.

## Build

```powershell
cargo fmt --all -- --check
cargo test --workspace --all-features
cargo test --doc --workspace --all-features
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo doc --workspace --no-deps
python scripts/audit_public_surface.py
```

## Boundary

The public boundary is documented in `PUBLIC_BOUNDARY.md` and
`PACKAGE_BOUNDARY.md`. The public delivery direction is documented in
`PUBLIC_DELIVERY_MODEL.md`, with license and mark boundaries in
`LICENSE_BOUNDARY.md` and `TRADEMARK_POLICY.md`.

This repo does not include internal engine source, private data adapters,
operational run logs, skill persistence, or diagnostic tools.

## Status

Current status is a public SDK boundary with a P11 delivery decision:
controlled early-access signed binary first, hosted API/SaaS later, and thin
public SDK/docs/wrappers where useful. The private engine source is not
published here.

## License

See `LICENSE`.

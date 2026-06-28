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
`PACKAGE_BOUNDARY.md`. This repo does not include internal engine source,
private data adapters, operational run logs, skill persistence, or diagnostic
tools.

## Status

Current status is a zero-state public SDK reset. A future product delivery model
may use an API, binary package, or wrapper crate, but that decision is separate
from this repository cleanup.

## License

See `LICENSE`.

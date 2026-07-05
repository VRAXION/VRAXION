# VRAXION

This repository is the clean public SDK and documentation surface for VRAXION.
It contains the current public Rust SDK/docs surface:

- `alphasync-core`
- `alphasync-runtime`

The private engine and operational research workspace are not published here.
Older public commits and releases remain historical records, but the current
tree is intentionally reset to a minimal public review surface.

## Build

```powershell
cargo fmt --all -- --check
cargo test --workspace --all-features
cargo test --doc --workspace --all-features
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo doc --workspace --no-deps
python scripts/audit_public_surface.py
```

## Public Surface

The public surface is documented in `PUBLIC_BOUNDARY.md` and
`PACKAGE_BOUNDARY.md`. The public delivery direction is documented in
`PUBLIC_DELIVERY_MODEL.md`, with license and mark boundaries in
`LICENSE_BOUNDARY.md` and `TRADEMARK_POLICY.md`.

This repo does not include internal engine source, private data adapters,
operational run logs, skill persistence, or diagnostic tools.

## Status

Current status is tracked by the latest public GitHub release:
[`public-sdk-p11-20260629`](https://github.com/VRAXION/VRAXION/releases/tag/public-sdk-p11-20260629).

The delivery direction is proof materials before broader runtime terms. Any
future binary, API, SaaS, or wrapper path requires separate release review; no
hosted availability is promised. The private engine implementation is not
published here.

## License

See `LICENSE`.

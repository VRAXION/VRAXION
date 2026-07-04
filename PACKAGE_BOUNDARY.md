# Public Package Surface

This repository contains the current public SDK source surface only.

Included crates:

```text
crates/alphasync-core
crates/alphasync-runtime
```

Excluded surfaces:

- internal engine source
- private data adapters
- operational run outputs
- skill persistence
- diagnostic tools
- old research pages
- hosted product code
- private engine binaries or internals

The public SDK source tree is expected to pass:

```powershell
cargo test --workspace --all-features
python scripts/audit_public_surface.py
```

Any future delivery model that exposes a binary package, hosted API, or wrapper
crate must go through a separate release review. The current public delivery
direction is controlled early-access signed binary first, hosted API/SaaS
later, and thin public SDK/docs/wrappers where useful.

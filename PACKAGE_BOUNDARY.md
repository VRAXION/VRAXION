# Public Package Boundary

This repository contains the public SDK source boundary only.

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

The public source tree is expected to pass:

```powershell
cargo test --workspace --all-features
python scripts/audit_public_surface.py
```

Any future delivery model that exposes a binary package, hosted API, or wrapper
crate must go through a separate release review. The current P11 direction is
controlled early-access signed binary first, hosted API/SaaS later, and thin
public SDK/docs/wrappers where useful.

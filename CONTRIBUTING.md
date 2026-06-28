# Contributing

This repository is a narrow public SDK boundary. Contributions should stay
inside the visible SDK and documentation surface.

Before opening a pull request, run:

```powershell
cargo fmt --all -- --check
cargo test --workspace --all-features
cargo test --doc --workspace --all-features
cargo clippy --workspace --all-targets --all-features -- -D warnings
python scripts/audit_public_surface.py
```

Do not add private data, operational logs, private engine source, or diagnostic
tools to this repository.

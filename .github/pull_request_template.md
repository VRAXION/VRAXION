## Summary

Describe what changed and why.

## Scope

- [ ] Rust mainline (`instnct-core/`) — grower, network core, examples
- [ ] Python deploy SDK (`Python/` — Block A + B)
- [ ] Rust deploy SDK (`Rust/` — Block A + B)
- [ ] Benchmark or sweep (`tools/`)
- [ ] Docs or repo cleanup (`docs/`, `README.md`, wiki)
- [ ] CI or tooling (`.github/workflows/`)

## Validation

Commands run (or N/A):

```bash
# example
python -m compileall Python tools
python -m pytest Python/ -q
python tools/check_public_surface.py
cargo test -p instnct-core
```

## Docs / links (if applicable)

- Pages:
- Public update:
- Related issue:
- Taxonomy label (`current mainline` / `validated finding` / `experimental branch`):

# alphasync-runtime

`alphasync-runtime` contains the public-candidate AlphaSync runtime CLI and safe
synthetic runtime examples built on top of `alphasync-core`.

The crate is limited to local smoke/runtime behavior. It intentionally excludes
non-public operator interfaces, frontier traces, raw datasets, persistence
services, golden-engine internals, training, mutation, skill registry,
artifact writer internals, and diagnostic parity binaries.

This crate remains a compatibility public-candidate source crate. Any future
golden-backed binary, API, SaaS, or wrapper path requires separate release
review before it becomes public availability language. This crate is not the
private engine source.

See the workspace `LICENSE` notice copied into this package.

# alphasync-runtime

`alphasync-runtime` contains the public-candidate AlphaSync runtime CLI and safe
synthetic runtime examples built on top of `alphasync-core`.

The crate is limited to local smoke/runtime behavior. It intentionally excludes
non-public operator interfaces, experiment traces, raw datasets, persistence
services, training workflows, mutation systems, artifact-writer internals, and
non-public diagnostic binaries.

This crate remains a compatibility public-candidate source crate. Any future
signed binary, API, SaaS, or wrapper path requires separate release
review before it becomes public availability language. This crate is not the
non-public engine implementation.

See the workspace `LICENSE` notice copied into this package.

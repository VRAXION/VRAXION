# alphasync-runtime

`alphasync-runtime` contains the public-candidate AlphaSync runtime CLI and safe
synthetic runtime examples built on top of `alphasync-core`.

The crate is limited to local smoke/runtime behavior. It does not include the
private runtime GUI, private frontier traces, raw datasets, or private
skillstore persistence. It also does not include the private golden engine,
training, mutation, skill registry, artifact writer internals, or diagnostic
parity binaries.

This crate remains a compatibility public-candidate source crate. Future
golden-backed releases should use a separate reviewed delivery model such as an
API, compiled binary, or open-core wrapper.

See the workspace `LICENSE` notice copied into this package.

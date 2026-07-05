# Public Surface Policy

The public tree must stay small and explicit.

Allowed:

- public SDK crates
- public docs
- CI and audit scripts for this public surface
- public delivery, license, and mark summaries

Not allowed:

- internal engine source
- private data adapters
- operational run outputs
- skill persistence
- diagnostic tools
- generated local caches or virtual environments
- stale research status pages
- private engine binary internals

The audit script is a hard gate for this public surface.
It also guards against stale public-copy terms that would imply a released
archive, live availability, hosted service, or public runtime beyond the
reviewed SDK/docs state.

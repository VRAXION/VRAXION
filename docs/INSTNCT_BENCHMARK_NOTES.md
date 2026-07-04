# INSTNCT Benchmark Notes

This page scopes the benchmark claim used by the public INSTNCT preview.

## Current Claim

The preview page shows:

- `5 us` warm exact selector p50
- `261 us` linear scan baseline p50
- `~52x` selector speedup

This is an internal preview measurement for a 16k mixed-unique workload. It is not an end-to-end latency claim, not a broad system benchmark, and not proof that every INSTNCT path runs at this speed.

## Release Scope

No public runnable T1 binary is published yet. The public reproduction path must ship with the first signed artifact and should include:

- the signed artifact and checksums
- the exact benchmark harness
- the workload description
- machine and OS details
- raw timing output
- refusal-mode examples

Until those are published, the numbers should be read as a scoped preview signal only.

## Intended Public Reproduction

The target proof path is:

1. Download the signed T1 artifact.
2. Verify checksums.
3. Disable the network.
4. Run the benchmark/refusal harness locally.
5. Compare the reported selector timing and refusal behavior with the published notes.

The repo remains the source of truth for release scope and updates.

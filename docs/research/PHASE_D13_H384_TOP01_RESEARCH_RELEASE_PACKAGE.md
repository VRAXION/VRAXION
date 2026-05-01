# Phase D13 H384 Top-01 Research Release Package

Date: 2026-05-01

## Summary

D13 packages the D10u `top_01` H384 state-anchored checkpoint as a stable
research artifact. This is not a public mainline grower replacement and not a
production language model. It is the strongest currently verified H384 research
checkpoint because it passes the D10r-v8 artifact/state-identity gate that
blocked the older beta.8 checkpoint.

Stable local artifact path:

```text
output/releases/v5.0.0-beta.10/seed2042_top01_h384_research.ckpt
```

Source checkpoint:

```text
output/phase_d10u_focused_ladder_20260430/bounded/candidates/top_01_seed_2042_edge_threshold_coadapted.ckpt
```

SHA256:

```text
b76789c42f4349ee28c18ce97bc5f0811a89c9b138e6ecdb86fa55626f019ddb
```

Tracked checksum record:

```text
docs/research/artifacts/v5.0.0-beta.10_seed2042_top01_h384_research.sha256
```

The checkpoint binary remains under `output/` and is intentionally not tracked
by Git.

## Verification

### Artifact Copy

The source and stable release copy were hashed after copy. The hashes matched
exactly:

```text
source SHA256  = b76789c42f4349ee28c18ce97bc5f0811a89c9b138e6ecdb86fa55626f019ddb
release SHA256 = b76789c42f4349ee28c18ce97bc5f0811a89c9b138e6ecdb86fa55626f019ddb
size bytes     = 253191
```

### Reload Smoke

`chain_diagnosis` successfully loaded the stable release copy:

```text
Checkpoint: step=40000, acc=4.10%, H=384
Input vector avg pairwise diff: 24.8/32 dims (78%)
Input zone 0..31 reaches output: 32/32
Total output-dim impact: 5967
Unique predictions: 2/8
Context-dependent predictions: 0/4
```

The reload smoke proves that the artifact is readable and structurally active.
It also records a key limitation: sequential context is still not carrying
stable information in this diagnostic, so this checkpoint is a research
artifact rather than a ready language model.

### D10r-v8 Release-Copy Smoke

The stable release copy was evaluated against the H384 seed2042 D7 baseline
with D10r-v8 artifact controls:

```text
output/phase_d13_top01_release_package_20260501/smoke
```

Run shape:

```text
eval_len        = 256
eval_seeds      = 970001,970002,970003,970004
control_repeats = 2
controls        = random_projection_null,
                  state_shuffle_shared,
                  state_shuffle_projection_consistent,
                  no_network_random_state
```

Verdict:

```text
D10R_V8_STATE_IDENTITY_PASS
```

Key margins:

```text
real MO delta CI low = +0.1857246291
trusted MO CI low    = +0.1661675039
artifact gate pass   = true
blocking families    = none
elapsed              = 228.06 s
```

This smoke is not the primary proof; it is a copy-integrity sanity check. The
promotion-grade evidence is the D10u 16k/30-seed sharded confirm below.

## Prior Evidence

Primary confirm root:

```text
output/phase_d10u_top01_d10r_confirm_20260430/confirm_16000_30seed_sharded_v2
```

Verdict:

```text
D10U_TOP01_16K_SHARDED_PASS
```

Confirm facts:

```text
total shards                 = 30
completed shards             = 30
failed shards                = 0
per-shard verdict            = 30 x D10R_V8_STATE_IDENTITY_PASS
min trusted MO CI low        = +0.0844932857
min real MO delta CI low     = +0.1780870404
blocking control families    = none
```

This matters because the older beta.8 checkpoint failed the same state-identity
gate under D10r-v8. D10u `top_01` is therefore not just another raw-score
candidate; it is the first H384 candidate in this branch that survives the
artifact/state-shuffle blocker at promotion-grade eval length and seed count.

## Interpretation

What this package establishes:

- A stable H384 research checkpoint exists at a reproducible path.
- The artifact copy is bit-identical to the transient D10u top checkpoint.
- The copied checkpoint reloads through existing tooling.
- The checkpoint passes D10r-v8 artifact/state-identity controls in a short
  release-copy smoke.
- Prior D10u evidence shows 30/30 sharded `eval_len=16000` confirms passed.

What this package does not establish:

- It does not prove H512/H8192 scaling.
- It does not prove seed-universal replication.
- It does not make this checkpoint the public mainline grower.
- It does not prove production language capability; chain diagnosis still shows
  no stable sequential context effect.

## Release-Ready Progress

```text
Long-horizon release-ready AI:
[=======___] 74%

[1] H384 state-anchored research checkpoint
    DONE: D10u top_01 packaged with SHA256

[2] Artifact/null controls
    DONE: D10r-v8 pass on release copy

[3] Promotion-grade H384 confirm
    DONE: 16k / 30 fresh-seed sharded pass

[4] Stable public package
    CURRENT: local artifact + tracked checksum/doc

[5] Production-readiness blockers
    NEXT:
      - public release asset attachment if desired
      - demo/eval wrapper for reproducible external use
      - H512 fair scaling or non-seed2042 replication
      - context-carrying / language-capability improvement
```

The practical next step is packaging and communication, not another H384 proof:
attach the checkpoint as a release asset or publish it through the project's
chosen artifact channel, then keep high-H and context-carrying improvements as
separate D11/D12 research tracks.

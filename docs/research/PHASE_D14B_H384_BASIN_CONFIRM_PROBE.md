# Phase D14b H384 Basin Confirm Probe

Date: 2026-05-01

## Summary

D14 found multiple H384 state-anchored basin candidates after the D13 top_01 package. D14b tested whether one non-control child candidate remains strong under a long `eval_len=16000` artifact/state-control gate, and compared it against the strongest D14 control tile.

This was intentionally a timing/evidence probe, not a full promotion-grade confirm.

## Inputs

Baseline:

```text
output/phase_d7_operator_bandit_20260427/H_384/D7_BASELINE/seed_2042/final.ckpt
```

Non-control target:

```text
rank06_non_control_9_26_19_53
output/phase_d14_h384_state_anchored_basin_atlas_20260501/bounded_fast_confirm/quadtree_hot/candidates/top_06.ckpt
```

Control comparator:

```text
rank02_control_18_54
output/phase_d14_h384_state_anchored_basin_atlas_20260501/bounded_fast_confirm/quadtree_hot/candidates/top_02.ckpt
```

Artifact controls:

```text
random_projection_null
state_shuffle_shared
state_shuffle_projection_consistent
no_network_random_state
```

## Probe Run

Output root:

```text
output/phase_d14b_h384_basin_confirm_20260501/probe
```

Shape:

```text
eval_len = 16000
control_repeats = 2
seed = 972001
max_shards_per_target = 1
```

## Results

| target | shard verdict | trusted MO CI low | real MO CI low | elapsed |
|---|---|---:|---:|---:|
| rank06_non_control_9_26_19_53 | D10R_V8_STATE_IDENTITY_PASS | +0.172907 | +0.186884 | 2048.7s |
| rank02_control_18_54 | D10R_V8_STATE_IDENTITY_PASS | +0.355422 | +0.381680 | 1904.3s |

Both targets passed the artifact/state gate on the first 16k shard.

The control comparator was materially stronger than the non-control child target on the same seed and gate.

## Interpretation

The D14 atlas signal is real enough to survive long artifact controls, but this probe does not support treating the non-control child as a clean, isolated new basin promotion candidate.

The stronger control tile means the current evidence is better described as:

```text
D14_WIDE_H384_ATLAS_SIGNAL
```

rather than:

```text
D14_CLEAN_NEW_BASIN_PROMOTION
```

In plain terms: D14 did not just rediscover random noise, but it also did not yet isolate a second clean top_01-like basin. The high-scoring region appears broader or more projection-tile-dependent than the original release-ready top_01 finding.

## Runtime

Mean shard runtime was about 33 minutes. A full serial 30-seed confirm for both targets would require roughly:

```text
60 shards * ~33 min = ~33 hours serial
```

This is not worth running immediately unless the next run is narrowed to the best-ranked target class or parallelized under a supervisor.

## Decision

Do not promote a new D14 checkpoint from this probe.

Keep D13 `top_01` as the only packaged H384 research checkpoint.

Treat D14/D14b as atlas evidence:

```text
H384 contains additional state-anchored high-scoring regions,
but the current quadtree/control split does not yet separate clean basins.
```

## Recommended Next Step

Run a narrower D14c discriminator instead of a full D14b confirm:

1. Evaluate the top non-control and top control regions across 4-8 fresh seeds at `eval_len=4000`.
2. Rank by control-adjusted selectivity and cross-seed stability.
3. Only run `eval_len=16000`, 30 seeds for a target that beats the control comparator on the shorter discriminator.

If D14c cannot separate non-control from control, stop atlas expansion and shift to objective/projection redesign or capability/context work.

## Progress Map

```text
GLOBAL RELEASE-READY AI MAP

[1] One packaged H384 checkpoint
    DONE: D13 top_01

[2] H384 basin atlas
    DONE: D14_MULTI_BASIN_SIGNAL at bounded confirm

[3] Long confirm probe
    DONE: D14b first 16k shard per target
    result: wide atlas signal, control comparator stronger

[4] Next discriminator
    D14c 4-8 seed control-vs-non-control separator
        |
        |-- if non-control beats control
        |      run promotion-grade 16k/30-seed confirm
        |
        '-- if control remains stronger
               keep top_01 as sole release checkpoint
               stop atlas expansion for now

[5] Release-ready AI path
    still needs either:
      clean second basin,
      H512 scaling proof,
      or capability/context improvement
```

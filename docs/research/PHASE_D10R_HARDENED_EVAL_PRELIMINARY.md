# Phase D10r Hardened Eval Preliminary

Date: 2026-04-30

Verdict: `D10R_STATE_SHUFFLE_BLOCKER`

## Summary

D10r implemented a hardened evaluator for existing H=384 checkpoints before any D10s wiring-prior or H512/H8192 scaling work. The purpose is adversarial measurement trust: beta.8 must beat semantic and projection controls, not only show raw multi-objective improvement.

The first D10r run exposed an evaluator mismatch, not a scientific failure. The Python/GPU helper used `MAX_CHARGE=15`, while the canonical Rust D9 evaluator uses `MAX_CHARGE=7`. After D10r was made Rust-compatible with `--max-charge 7`, beta.8 again showed a stable positive real-task signal.

## Key Result

Run:

```text
output/phase_d10r_hardened_eval_20260430/main_max7_8seed/
```

Setup:

```text
eval_len: 1000
eval_seeds: 970001..970008
max_charge: 7
controls: random_label, random_bigram, unigram_decoy,
          projection_shuffle, projection_reinit, state_shuffle, time_shuffle
```

Beta.8 positive control:

```text
real MO delta: +0.014526
passes: random_label, random_bigram, unigram_decoy,
        projection_shuffle, projection_reinit, time_shuffle
fails:  state_shuffle
```

The negative H=384 checkpoints did not pass the full hardened suite. Some had large raw MO deltas, but failed selectivity against controls, especially `unigram_decoy`. This confirms that raw MO is not sufficient evidence.

## Interpretation

This does not invalidate the D9.2/D9.4 beta.8 result. The beta.8 checkpoint remains a real H=384 seed2042 improvement under the canonical evaluator, and D9.4b still confirms the edge+threshold causal driver at `eval_len=4000` and `eval_len=16000`.

The new blocker is narrower:

```text
state_shuffle can create a false positive stronger than the real beta.8 signal
for at least one eval seed.
```

That means the readout/projection/eval path can be fooled by a shuffled output-state arrangement. Therefore high-H scaling remains blocked until D10r-v2 either fixes this control artifact or proves that the state-shuffle failure is expected and not relevant to deployment metrics.

## Next Step

D10r-v2 should focus only on this blocker:

```text
1. Add multiple independent state_shuffle variants per eval seed.
2. Report worst-control and median-control separately.
3. Add a no-network/random-state control.
4. Add a Rust CPU compatibility check for the real beta.8 path.
5. Keep H512/H8192 and D10s scaling blocked until state_shuffle is explained.
```

If D10r-v2 passes, the next science step is D10s H=384 wiring-prior smoke. If D10r-v2 fails, the next engineering step is projection/readout redesign, not bigger H.

## Progress Map

```text
GLOBAL AI PLAN

[1] H384 beta.8 generalist
    DONE

[2] causal mechanism
    DONE: edge + threshold co-adaptation

[3] H384 seed replication
    DONE: D10b no replication signal

[4] evaluator hardening
    CURRENT: D10r
    result: real beta.8 signal survives 6/7 controls
    blocker: state_shuffle false-positive artifact
        |
        |-- D10r-v2 passes
        |      v
        |   [5] D10s H384 wiring-prior sweep
        |      v
        |   [6] H512 pilot
        |      v
        |   [7] H8192 sparse high-H proof
        |      v
        |   [8] AI deployment candidate
        |
        '-- D10r-v2 fails
               v
            projection/readout redesign
            no high-H scaling yet
```


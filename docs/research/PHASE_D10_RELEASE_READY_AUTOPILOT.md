# Phase D10 Release-Ready Autopilot

Date: 2026-04-30

Purpose: run a bounded, gate-controlled research cycle toward release-ready AI evidence without manual "next" prompts.

## Queue

```text
preflight
  -> D10r_v2_state_shuffle_smoke
  -> D10r_v2_state_shuffle_main
  -> D10s_wiring_prior_smoke, only if D10r-v2 passes
```

The queue is intentionally conservative. It does not run H512/H8192 brute force and it does not promote checkpoints. High-H remains blocked until evaluator trust and a non-seed2042 wiring-prior signal both pass.

## Runtime Contract

The runner writes:

```text
output/phase_d10_release_ready_autopilot_20260430/status.json
output/phase_d10_release_ready_autopilot_20260430/events.jsonl
output/phase_d10_release_ready_autopilot_20260430/progress_map.md
output/phase_d10_release_ready_autopilot_20260430/wake_trigger.json
```

Heartbeat cadence is 5 minutes for the full run. Phase starts, phase completions, decisions, failures, and final status update `wake_trigger.json` immediately.

## Verdict Meaning

```text
D10R_TRUST_PASS
  D10s wiring-prior smoke can run.

D10R_V2_PROJECTION_READOUT_BLOCKED
  state/no-network controls still beat or destabilize beta.8.
  Next work is projection/readout redesign, not bigger H.

D10S_REPLICABLE_WIRING_PRIOR_SIGNAL
  A non-seed2042 H384 candidate passed trusted smoke.
  Next work is targeted confirm, then H512 pilot planning.

D10S_NO_TRUSTED_SIGNAL or D10S_SEED2042_ONLY
  Wiring priors did not yet solve seed sensitivity.
```

## Progress Map

```text
[1] beta.8 H384 generalist
    DONE

[2] causal mechanism
    DONE

[3] seed replication
    DONE: no broad replication

[4] D10r-v2 evaluator hardening
    CURRENT
        |
        |-- pass -> D10s H384 wiring-prior smoke
        '-- fail -> projection/readout redesign

[5] H512/H8192
    blocked until D10r-v2 + D10s signal
```


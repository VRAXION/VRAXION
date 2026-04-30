# D10 HDS Basin & State Identity Finding

Status date: 2026-04-30

This page records the current finding from the high-dimensional search-space
mapping work around D9/D10.

## Plain Verdict

```text
The HDS basin side quest is closed.

Result:
there is real signal in the H384 landscape,
but it is local, narrow, seed-sensitive, and not release-ready.
```

The HDS map is not a large universal mountain. It is closer to a set of
thin local islands / needle-like ridges. One island produced a useful H384
research checkpoint, but hardened D10 controls showed that this checkpoint is
not a release candidate.

## What We Know

```text
[1] Real improvement exists
    beta.8 improved the D9/D10 multi-objective gate.

[2] The cause is core-network structure
    D9.4 confirmed edge + threshold co-adaptation.

[3] It is not projection/channel/polarity magic
    projection/channel/polarity stayed unchanged.

[4] The landscape is not a broad universal basin
    evidence points to local islands, not a single smooth mountain.

[5] The current beta.8 checkpoint is not release-ready
    state_shuffle_shared can beat the real signal.

[6] The path is not dead
    D10u top_01 passed a bounded state-identity confirm.
```

## Key Evidence

### D9.2 / D9.4

D9.2 produced a confirmed H384 generalist candidate. D9.4 then localized the
mechanism:

```text
D9_2_FULL_GENERALIST_CONFIRMED
D9_4B_CAUSAL_CONFIRM_PASS
mechanism: EDGE_THRESHOLD_COADAPTATION
```

Interpretation: the useful change is a co-adapted package of wiring and
threshold timing. It is not a one-neuron trick and not a readout-only artifact.

### D10r-v8

D10r-v8 hardened the finding against artifact controls:

```text
random_projection_null       pass
no_network_random_state      pass
state_shuffle_shared         fail
projection-consistent shuffle sanity: zero drift
duplicate/similar projection rows: zero
```

Plain meaning: the beta.8 signal is real, but the hidden-state identity is too
weak. If internal state rows are shuffled arbitrarily, some shuffled versions
can beat the real checkpoint. That blocks a release-ready claim.

### D10u

D10u moved the lesson into the search loop. Candidates are now ranked against
the same artifact controls instead of being checked only after the fact.

D10u focused ladder result:

```text
seed2042 top_01: strict trusted scout candidate
seed4042: weak state-anchored signal
```

Bounded D10r-v8 confirm on `top_01`:

```text
verdict: D10R_V8_STATE_IDENTITY_PASS
eval_len: 1000
eval_seeds: 970021..970024
trusted_mo_ci_low: +0.170111
state_shuffle_shared bound CI low: +0.184446
```

This reopens the release-candidate path, but does not unlock H512/H8192 yet.
The next gate is promotion-grade confirmation at longer eval budgets and more
fresh seeds.

## What This Means

The old side quest asked:

```text
Is there a basin in the high-dimensional search sphere?
```

Answer:

```text
Yes, but not the kind we need for release.
It is local and narrow, not a broad universal slope.
```

The new main question is:

```text
Can we climb a state-anchored candidate
that keeps the useful edge+threshold signal
while passing the D10r-v8 artifact gate?
```

## Current Progress Map

```text
[1] HDS basin mapping
    DONE

[2] beta.8 mechanism
    DONE: edge + threshold co-adaptation

[3] beta.8 release gate
    FAIL: weak state identity

[4] HDS side quest verdict
    DONE: local islands, not universal mountain

[5] state-anchored search
    DONE: top_01 bounded state-identity pass

[6] release-ready AI
    BLOCKED until top_01 passes promotion-grade confirm
```

## Canonical Repo References

- `docs/research/PHASE_D9_2_MULTI_OBJECTIVE_CONFIRM.md`
- `docs/research/PHASE_D9_4B_CAUSAL_DIFF_CONFIRM_REPORT.md`
- `docs/research/PHASE_D10R_V8_STATE_IDENTITY_GATE.md`
- `docs/research/PHASE_D10U_STATE_ANCHORED_WIRING_PRIOR.md`
- `tools/_scratch/d10r_hardened_eval.py`
- `tools/_scratch/d10s_wiring_prior_sweep.py`

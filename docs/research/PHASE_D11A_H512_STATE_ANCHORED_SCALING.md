# Phase D11a H512 State-Anchored Scaling Pilot

Date: 2026-05-01

## Verdict

`D11A_H512_BASELINE_NOT_READY`

D11a did not reach the state-anchored scout stage. The H512 baseline created with the D7 H384 recipe did not move: the timing probe finished with `0.00%` accuracy and `0%` accept rate, and the full baseline run was stopped at 12k/40k steps after staying at `0.00%` accuracy and `0%` accept rate.

This is a baseline-readiness failure, not evidence that H512 cannot work.

## What Ran

- Preflight passed:
  - `cargo build -p instnct-core --example evolve_mutual_inhibition --release`
  - `cargo build -p instnct-core --example d9_direct_landscape --release`
  - `python -m py_compile tools/_scratch/d10r_hardened_eval.py tools/_scratch/d10s_wiring_prior_sweep.py`
  - `python tools/check_public_surface.py`
- H512 timing probe:
  - `H=512`, `steps=2000`, `seed=2042`, D7-compatible flags
  - runtime: `287.969s`
  - result: `final_acc=0.000000`, `peak_acc=0.000000`, `accept_rate_pct=0.0000`
  - checkpoint: `output/phase_d11a_h512_state_anchored_scaling_20260501/baseline_probe/final.ckpt`
- H512 full baseline attempt:
  - target: `steps=40000`
  - stopped at 12k steps after repeated flat checkpoints:
    - 2k: `acc=0.00%`, `accept=0%`
    - 4k: `acc=0.00%`, `accept=0%`
    - 6k: `acc=0.00%`, `accept=0%`
    - 8k: `acc=0.00%`, `accept=0%`
    - 10k: `acc=0.00%`, `accept=0%`
    - 12k: `acc=0.00%`, `accept=0%`

## H512 Probe Topology

`analyze_checkpoint` on the H512 timing probe:

- `H=512`
- `phi_dim=316`
- active edges: `13,269`
- density: `5.06%`
- average degree: `25.9`
- dead neurons: `0/512`
- recurrent-loop coverage: `512/512`
- output reachable from input: `316/316`

The network is structurally alive, but the D7 strict acceptance recipe did not produce accepted moves at H512.

## Interpretation

The H384 `top_01` result remains the golden reference and release-candidate research checkpoint. D11a shows that simply increasing `H` from 384 to 512 while keeping the D7 baseline recipe unchanged is not enough.

The likely next scaling step is not H1024/H8192. It is a D11b H512 baseline-calibration pass:

- threshold/init calibration for H512,
- less brittle acceptance policy or warm-start schedule,
- short smoke gates before any state-anchored scout,
- only then rerun the H512 D10u-style ladder.

## Progress Map

```text
GLOBAL RELEASE-READY AI MAP

[1] H384 top_01 release-candidate research checkpoint
    DONE

[2] artifact/state-shuffle controls
    DONE: 16k / 30 fresh seed confirm passed

[3] H512 naive D7 scaling
    DONE: D11A_H512_BASELINE_NOT_READY
        |
        v
[4] H512 calibrated baseline
    NEXT: D11b threshold/init/acceptance calibration
        |
        |-- if baseline moves
        |      H512 state-anchored scout
        |
        '-- if still flat
               stay at H384 and redesign scaling recipe
```


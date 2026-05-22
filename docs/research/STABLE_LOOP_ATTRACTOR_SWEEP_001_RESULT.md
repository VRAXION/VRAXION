# STABLE_LOOP_ATTRACTOR_SWEEP_001 Result

## Run

```text
stage=valid_slice
seeds=2026,2027,2028
train_examples=4096
eval_examples=4096
epochs=80
jobs=12
device=cpu
completed_jobs=186
```

## Verdict

```json
[
  "COMPUTE_BUDGET_CONFONDED",
  "STANDARD_RNN_SUFFICIENT",
  "SUMMARY_SOLVES_TASK"
]
```

## Interpretation

The valid slice did not produce a clean stable-loop win.

The strongest falsification is the summary shortcut control:

```text
summary_direct_gap = +0.003
```

The best loop family is only about 0.003 above `SUMMARY_DIRECT_HEAD` on the combined heldout/length score. That means the encoder summary can already carry nearly all useful signal seen by the loop in this setup.

The matched-compute control is stronger than the loop:

```text
matched_compute_gap = -0.094
```

`GRU_EXTRA_NOOP_STEPS` and `DEEP_MLP_MATCHED_COMPUTE` therefore block a topology claim here. The loop models do show high overrun/noise stability, but they do not convert that into better heldout composition or length generalization.

Per the contract, the full survivor sweep was not launched because the valid slice returned `STANDARD_RNN_SUFFICIENT` plus the two confound verdicts.

## Decision

```text
Do not treat STABLE_LOOP_ATTRACTOR_SWEEP_001 as positive.
Do not move to Prismion-sidepocket claims from this run.
The next useful change is to harden the task against summary encoding, or force the loop to solve an iterative computation that cannot be compressed into h0.
```

## Summary Table

| Arm | W | K | S | Final | Heldout | Length | SettleGain | Overrun | Noise |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `DEEP_MLP_MATCHED_COMPUTE` | `16` | `0` | `1` | `0.575` | `0.447` | `0.454` | `nan` | `nan` | `nan` |
| `DEEP_MLP_MATCHED_COMPUTE` | `32` | `0` | `1` | `0.584` | `0.458` | `0.469` | `nan` | `nan` | `nan` |
| `GRU` | `16` | `0` | `1` | `0.623` | `0.488` | `0.550` | `nan` | `nan` | `nan` |
| `GRU` | `32` | `0` | `1` | `0.602` | `0.476` | `0.531` | `nan` | `nan` | `nan` |
| `GRU_EXTRA_NOOP_STEPS` | `16` | `0` | `1` | `0.619` | `0.483` | `0.548` | `nan` | `nan` | `nan` |
| `GRU_EXTRA_NOOP_STEPS` | `16` | `0` | `4` | `0.619` | `0.486` | `0.544` | `nan` | `nan` | `nan` |
| `GRU_EXTRA_NOOP_STEPS` | `16` | `0` | `8` | `0.622` | `0.494` | `0.553` | `nan` | `nan` | `nan` |
| `GRU_EXTRA_NOOP_STEPS` | `32` | `0` | `1` | `0.607` | `0.476` | `0.541` | `nan` | `nan` | `nan` |
| `GRU_EXTRA_NOOP_STEPS` | `32` | `0` | `4` | `0.622` | `0.490` | `0.556` | `nan` | `nan` | `nan` |
| `GRU_EXTRA_NOOP_STEPS` | `32` | `0` | `8` | `0.616` | `0.481` | `0.551` | `nan` | `nan` | `nan` |
| `HIGHWAY_FF_SIDEPOCKETS_AUTONOMOUS` | `16` | `4` | `1` | `0.529` | `0.381` | `0.439` | `0.000` | `0.883` | `0.971` |
| `HIGHWAY_FF_SIDEPOCKETS_AUTONOMOUS` | `16` | `4` | `4` | `0.535` | `0.387` | `0.444` | `0.016` | `0.943` | `0.960` |
| `HIGHWAY_FF_SIDEPOCKETS_AUTONOMOUS` | `16` | `4` | `8` | `0.521` | `0.382` | `0.428` | `0.034` | `0.968` | `0.953` |
| `HIGHWAY_FF_SIDEPOCKETS_AUTONOMOUS` | `32` | `4` | `1` | `0.533` | `0.393` | `0.441` | `0.000` | `0.884` | `0.980` |
| `HIGHWAY_FF_SIDEPOCKETS_AUTONOMOUS` | `32` | `4` | `4` | `0.538` | `0.400` | `0.446` | `0.005` | `0.976` | `0.979` |
| `HIGHWAY_FF_SIDEPOCKETS_AUTONOMOUS` | `32` | `4` | `8` | `0.542` | `0.402` | `0.450` | `0.021` | `0.981` | `0.968` |
| `HIGHWAY_PRISMION_SIDEPOCKETS` | `16` | `4` | `1` | `0.526` | `0.385` | `0.429` | `0.000` | `0.889` | `0.969` |
| `HIGHWAY_PRISMION_SIDEPOCKETS` | `16` | `4` | `4` | `0.524` | `0.384` | `0.427` | `0.012` | `0.968` | `0.968` |
| `HIGHWAY_PRISMION_SIDEPOCKETS` | `16` | `4` | `8` | `0.505` | `0.381` | `0.425` | `0.019` | `0.990` | `0.967` |
| `HIGHWAY_PRISMION_SIDEPOCKETS` | `32` | `4` | `1` | `0.532` | `0.392` | `0.440` | `0.000` | `0.922` | `0.982` |
| `HIGHWAY_PRISMION_SIDEPOCKETS` | `32` | `4` | `4` | `0.533` | `0.388` | `0.444` | `0.006` | `0.973` | `0.977` |
| `HIGHWAY_PRISMION_SIDEPOCKETS` | `32` | `4` | `8` | `0.540` | `0.397` | `0.451` | `0.016` | `0.989` | `0.981` |
| `HIGHWAY_RECURRENT_SIDEPOCKETS` | `16` | `4` | `1` | `0.531` | `0.383` | `0.439` | `0.000` | `0.879` | `0.967` |
| `HIGHWAY_RECURRENT_SIDEPOCKETS` | `16` | `4` | `4` | `0.525` | `0.378` | `0.431` | `0.014` | `0.978` | `0.960` |
| `HIGHWAY_RECURRENT_SIDEPOCKETS` | `16` | `4` | `8` | `0.530` | `0.385` | `0.442` | `0.037` | `0.987` | `0.960` |
| `HIGHWAY_RECURRENT_SIDEPOCKETS` | `32` | `4` | `1` | `0.532` | `0.388` | `0.442` | `0.000` | `0.916` | `0.977` |
| `HIGHWAY_RECURRENT_SIDEPOCKETS` | `32` | `4` | `4` | `0.535` | `0.394` | `0.443` | `0.004` | `0.974` | `0.979` |
| `HIGHWAY_RECURRENT_SIDEPOCKETS` | `32` | `4` | `8` | `0.532` | `0.383` | `0.444` | `0.030` | `0.988` | `0.978` |
| `HIGHWAY_SPARSE_SIDEPOCKETS` | `16` | `4` | `1` | `0.525` | `0.378` | `0.431` | `0.000` | `0.843` | `0.967` |
| `HIGHWAY_SPARSE_SIDEPOCKETS` | `16` | `4` | `4` | `0.526` | `0.383` | `0.430` | `0.007` | `0.979` | `0.959` |
| `HIGHWAY_SPARSE_SIDEPOCKETS` | `16` | `4` | `8` | `0.526` | `0.386` | `0.434` | `0.028` | `0.976` | `0.960` |
| `HIGHWAY_SPARSE_SIDEPOCKETS` | `32` | `4` | `1` | `0.535` | `0.397` | `0.440` | `0.000` | `0.904` | `0.980` |
| `HIGHWAY_SPARSE_SIDEPOCKETS` | `32` | `4` | `4` | `0.531` | `0.389` | `0.443` | `-0.001` | `0.974` | `0.975` |
| `HIGHWAY_SPARSE_SIDEPOCKETS` | `32` | `4` | `8` | `0.533` | `0.402` | `0.443` | `0.008` | `0.985` | `0.970` |
| `LSTM` | `16` | `0` | `1` | `0.542` | `0.396` | `0.416` | `nan` | `nan` | `nan` |
| `LSTM` | `32` | `0` | `1` | `0.540` | `0.398` | `0.429` | `nan` | `nan` | `nan` |
| `MAIN_LOOP_GRU_AUTONOMOUS` | `16` | `4` | `1` | `0.527` | `0.386` | `0.433` | `0.000` | `0.879` | `0.976` |
| `MAIN_LOOP_GRU_AUTONOMOUS` | `16` | `4` | `4` | `0.524` | `0.383` | `0.420` | `0.027` | `0.954` | `0.959` |
| `MAIN_LOOP_GRU_AUTONOMOUS` | `16` | `4` | `8` | `0.503` | `0.362` | `0.393` | `0.144` | `0.944` | `0.949` |
| `MAIN_LOOP_GRU_AUTONOMOUS` | `32` | `4` | `1` | `0.534` | `0.396` | `0.444` | `0.000` | `0.835` | `0.982` |
| `MAIN_LOOP_GRU_AUTONOMOUS` | `32` | `4` | `4` | `0.544` | `0.400` | `0.458` | `0.026` | `0.951` | `0.972` |
| `MAIN_LOOP_GRU_AUTONOMOUS` | `32` | `4` | `8` | `0.534` | `0.391` | `0.441` | `0.069` | `0.983` | `0.971` |
| `MAIN_LOOP_GRU_CONDITIONED` | `16` | `4` | `1` | `0.539` | `0.401` | `0.451` | `0.000` | `0.900` | `0.981` |
| `MAIN_LOOP_GRU_CONDITIONED` | `16` | `4` | `4` | `0.530` | `0.383` | `0.440` | `0.024` | `0.936` | `0.977` |
| `MAIN_LOOP_GRU_CONDITIONED` | `16` | `4` | `8` | `0.534` | `0.387` | `0.444` | `0.119` | `0.978` | `0.974` |
| `MAIN_LOOP_GRU_CONDITIONED` | `32` | `4` | `1` | `0.531` | `0.393` | `0.438` | `0.000` | `0.926` | `0.987` |
| `MAIN_LOOP_GRU_CONDITIONED` | `32` | `4` | `4` | `0.532` | `0.385` | `0.446` | `0.013` | `0.976` | `0.981` |
| `MAIN_LOOP_GRU_CONDITIONED` | `32` | `4` | `8` | `0.519` | `0.377` | `0.423` | `0.101` | `0.985` | `0.986` |
| `MAIN_LOOP_MLP_AUTONOMOUS` | `16` | `4` | `1` | `0.533` | `0.390` | `0.440` | `0.000` | `0.868` | `0.969` |
| `MAIN_LOOP_MLP_AUTONOMOUS` | `16` | `4` | `4` | `0.539` | `0.390` | `0.454` | `0.017` | `0.958` | `0.958` |
| `MAIN_LOOP_MLP_AUTONOMOUS` | `16` | `4` | `8` | `0.522` | `0.380` | `0.428` | `0.061` | `0.981` | `0.950` |
| `MAIN_LOOP_MLP_AUTONOMOUS` | `32` | `4` | `1` | `0.531` | `0.391` | `0.438` | `0.000` | `0.901` | `0.978` |
| `MAIN_LOOP_MLP_AUTONOMOUS` | `32` | `4` | `4` | `0.533` | `0.389` | `0.446` | `0.016` | `0.955` | `0.973` |
| `MAIN_LOOP_MLP_AUTONOMOUS` | `32` | `4` | `8` | `0.541` | `0.400` | `0.454` | `0.026` | `0.989` | `0.977` |
| `MAIN_LOOP_MLP_CONDITIONED` | `16` | `4` | `1` | `0.529` | `0.391` | `0.438` | `0.000` | `0.947` | `0.986` |
| `MAIN_LOOP_MLP_CONDITIONED` | `16` | `4` | `4` | `0.526` | `0.385` | `0.436` | `0.008` | `0.965` | `0.977` |
| `MAIN_LOOP_MLP_CONDITIONED` | `16` | `4` | `8` | `0.530` | `0.392` | `0.437` | `0.044` | `0.989` | `0.982` |
| `MAIN_LOOP_MLP_CONDITIONED` | `32` | `4` | `1` | `0.530` | `0.388` | `0.436` | `0.000` | `0.934` | `0.987` |
| `MAIN_LOOP_MLP_CONDITIONED` | `32` | `4` | `4` | `0.535` | `0.392` | `0.443` | `0.009` | `0.978` | `0.987` |
| `MAIN_LOOP_MLP_CONDITIONED` | `32` | `4` | `8` | `0.531` | `0.392` | `0.438` | `0.025` | `0.986` | `0.987` |
| `SUMMARY_DIRECT_HEAD` | `16` | `0` | `1` | `0.533` | `0.399` | `0.441` | `nan` | `nan` | `nan` |
| `SUMMARY_DIRECT_HEAD` | `32` | `0` | `1` | `0.540` | `0.408` | `0.445` | `nan` | `nan` | `nan` |

## Control Gaps

```json
{
  "summary_direct_gap": 0.002884,
  "matched_compute_gap": -0.094087,
  "autonomous_vs_conditioned_gap": 0.003787
}
```

## Claim Boundary

This is an abstract symbolic stable-loop probe. It is not a parser, factuality system, language benchmark, consciousness claim, or full VRAXION architecture test.

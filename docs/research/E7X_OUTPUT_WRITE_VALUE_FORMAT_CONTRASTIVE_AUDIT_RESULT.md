# E7X Output Write Value-Format Contrastive Audit Result

Status: complete.

```text
decision = e7x_output_value_format_not_sufficient
best_non_oracle_system = monotonic_calibrated_write
baseline_usefulness = 0.606836
oracle_write_reference_usefulness = 0.900000
affine_calibrated_write_usefulness = 0.614258
monotonic_calibrated_write_usefulness = 0.623437
zscore_normalized_write_usefulness = 0.590820
codebook_write_usefulness = 0.620508
sign_or_quantized_write_usefulness = 0.563281
residual_delta_write_usefulness = 0.593359
router_integrated_write_usefulness = 0.584961
gap_fraction_closed = 0.056629
deterministic_replay_passed = true
checker_failure_count = 0
```

## Mean Scores

```text
baseline_real_write              useful=0.606836 acc=0.706836 mae=0.165188 corr=0.676960 sign=0.165188 next=0.003702
oracle_write_reference           useful=0.900000 acc=1.000000 mae=0.000000 corr=1.000000 sign=0.000000 next=0.000000
affine_calibrated_write          useful=0.614258 acc=0.714258 mae=0.251820 corr=0.750759 sign=0.156874 next=0.007163
monotonic_calibrated_write       useful=0.623437 acc=0.723437 mae=0.353003 corr=0.670485 sign=0.153063 next=0.011320
zscore_normalized_write          useful=0.590820 acc=0.690820 mae=0.352490 corr=0.686232 sign=0.177661 next=0.011342
codebook_write                   useful=0.620508 acc=0.720508 mae=0.152716 corr=0.693153 sign=0.152716 next=0.003137
sign_or_quantized_write          useful=0.563281 acc=0.663281 mae=0.220863 corr=0.671403 sign=0.214593 next=0.006089
residual_delta_write             useful=0.593359 acc=0.693359 mae=0.376372 corr=0.668558 sign=0.201705 next=0.012214
router_integrated_write          useful=0.584961 acc=0.684961 mae=0.215846 corr=0.652346 sign=0.224363 next=0.005912
```

## Interpretation

E7X falsified the simple explanation:

```text
bad_write * scale + bias ~= good_write
```

The best non-oracle transform was monotonic calibration, but it only closed
about 5.7% of the oracle gap. Affine, codebook, quantized, residual, and
router-integrated one-cell transforms did not restore composition.

This means E7W's write-contract bottleneck is not merely an output scalar
calibration issue. The missing contract likely includes more structure than a
single calibrated value:

```text
multi-cell state update
write bundle / micro-format
state transition invariant
consumer-facing intermediate representation
```

## Recommended Next Step

```text
E7Y_MULTI_CELL_WRITE_BUNDLE_CONTRACT_PROBE
```

Purpose: test whether pockets need to write a compact bundle of anonymous cells
instead of one scalar output cell, while still avoiding semantic labels and dense
graph soup.

Boundary: this is a controlled diagnostic value-format audit, not a
raw-language, AGI, consciousness, or model-scale claim.

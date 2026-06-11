# E20_CODEC_AGNOSTIC_BINARY_STREAM_MULTI_POCKET_GROUNDING_CONFIRM Contract

## Purpose

E20 tests whether a Flow/Pocket policy can ingest multiple codec-agnostic binary streams and reconstruct a shared latent world state from converging partial evidence. It follows E19 repo-text reasoning but is not a text retrieval milestone.

## Boundary

This is a controlled synthetic codec-agnostic binary-stream grounding audit for a Flow/Pocket policy. It tests whether multiple binary projections of the same latent world can be aligned into a shared Flow state. It does not prove real audio understanding, real vision understanding, general natural-language AI, GPT-like generation, AGI, or production readiness.

## Latent world and codecs

Episodes contain hidden latent worlds with 2-8 entities, stable identity across time, attributes, events, temporal order, simple causal constraints, and optional missing/ambiguous/contradictory observations. The latent state is oracle-only for generation/evaluation.

Required binary projections are packet-struct bytes, synthetic PCM-like int16 frames, grayscale raster-like frames, UTF-8 byte observations as one modality, sensor time-series bytes, and opaque event hashes. Split-specific randomized codebooks prevent exact train/heldout code-value leakage.

## Hard tasks

Primary hard task families cover direct binary ingestion, frame boundary recovery, temporal order recovery, stream routing, cross-codec event alignment, entity binding, shared-state reconstruction, missing/contradictory modality repair, noisy stream repair, delayed evidence, causal repair, memory queries, heldout codebook transfer, multi-pocket convergence, abstain on ungrounded queries, and adversarial false alignment.

## Full-confirm minimums

Full confirmation requires at least 60 generations, population 96, 1400 heldout episodes, 1400 stress episodes, 6000 candidate evaluations, 60 checkpoints, 400 cross-codec stress episodes, 300 missing/corrupt modality stress episodes, 400 heldout-codebook episodes, and 200 adversarial false-alignment episodes.

## Required run

```bash
python3 scripts/probes/run_e20_codec_agnostic_binary_stream_multi_pocket_grounding_confirm.py \
  --out target/pilot_wave/e20_codec_agnostic_binary_stream_multi_pocket_grounding_confirm \
  --strict-budget \
  --no-downshift \
  --generations 100 \
  --population 128 \
  --train-episodes 5000 \
  --validation-episodes 1200 \
  --heldout-episodes 1800 \
  --stress-episodes 1800 \
  --min-stream-length 16 \
  --max-stream-length 128 \
  --min-modalities 3 \
  --max-modalities 6 \
  --checkpoint-every 1 \
  --max-runtime-minutes 360 \
  --resume

python3 scripts/probes/run_e20_codec_agnostic_binary_stream_multi_pocket_grounding_confirm_check.py \
  --out target/pilot_wave/e20_codec_agnostic_binary_stream_multi_pocket_grounding_confirm \
  --write-summary
```

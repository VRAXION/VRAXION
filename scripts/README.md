# Scripts

## Context Cancellation / Core Recovery Probe

`run_context_cancellation_probe.py` contains small CPU-friendly toy mechanism probes. The outputs are written under `target/context-cancellation-probe/`, which is treated as local run evidence unless a result is promoted into `docs/research/`.

### Experiment Modes

Run with:

```bash
.venv/bin/python scripts/run_context_cancellation_probe.py --experiment core_recovery
```

`core_recovery` is the original v4/v5 path. It tests whether a sparse recurrent model can recover a task-causal core from entangled core+nuisance input. The safe claim is limited: in a toy setup, recurrence can make the causal core more dominant under interference. It is not a claim of clean nuisance deletion.

Run with:

```bash
.venv/bin/python scripts/run_context_cancellation_probe.py --experiment latent_refraction --input-mode entangled
```

`latent_refraction` is the v6 task-frame probe. The same observed feature bundle is evaluated under different task-frame tokens, so a feature group can be causal in one frame and nuisance in another. The main metrics are active-group influence, inactive-group influence, `refraction_index_by_step`, and `authority_switch_score`.

Run with:

```bash
.venv/bin/python scripts/run_context_cancellation_probe.py --experiment multi_aspect_refraction --input-mode entangled
```

`multi_aspect_refraction` tests the stricter same-token case. The actor token, especially `dog`, is reused under `danger_frame`, `friendship_frame`, `sound_frame`, and `environment_frame`. Labels are never trained as direct contradictions such as `dog=danger` and `dog=friend`; instead the frame selects which actor relation is causal. The main metrics are dog/actor token influence by frame, actor decodability by step, and actor authority-switch scores.

Run with:

```bash
.venv/bin/python scripts/run_context_cancellation_probe.py --experiment frame_switch_diagnostics --input-mode entangled
```

`frame_switch_diagnostics` reuses the multi-aspect task and asks whether the effect is recurrent hidden-state reorientation or static frame-token routing. It compares frame placement modes (`frame_in_recurrence_only`, `frame_initial_only`, `frame_at_output_only`, `no_frame`), runs mid-run frame switches, measures hidden trajectory geometry, and tests soft frame interpolation.

Run with:

```bash
.venv/bin/python scripts/run_context_cancellation_probe.py --experiment reframe_diagnostics --input-mode entangled
```

`reframe_diagnostics` tests early frame commitment plus an explicit reframe/reset pulse. It starts trajectories under the wrong frame, switches to the correct frame with or without a reset signal, and compares normal mid-run rotation against trained reset-triggered recovery.

Run with:

```bash
.venv/bin/python scripts/run_context_cancellation_probe.py --experiment inferred_frame_pointer --input-mode entangled
```

`inferred_frame_pointer` moves from explicit frame-token control to automatic frame selection. It uses the multi-aspect setup without adding new semantic concepts: the intended frame is made inferable from feature-group salience in the input bundle, then a predicted internal frame pointer is used for the recurrent decision pass. It reports oracle-frame, predicted-frame, frame-head-only, no-frame, wrong-forced-frame, zero-recurrent, randomized-recurrent, random-label, influence, and token-frame inventory metrics. The first result supports frame inference but leaves pointer-specific necessity unclear because no-frame and frame-head-only baselines remain fairly strong.

Run with:

```bash
.venv/bin/python scripts/run_context_cancellation_probe.py --experiment query_cued_frame_pointer --input-mode entangled
```

`query_cued_frame_pointer` uses the same multi-aspect observations under multiple toy query cues. Query cues are separate from frame embeddings and are not natural language. The probe compares oracle-frame, predicted-pointer, query-head-only, no-pointer-query, no-query, wrong-forced-frame, query ablation, query shuffle, zero-recurrent, randomized-recurrent, random-label, and pointer-vs-direct authority/refraction metrics. The first result supports query frame prediction and query dependence, but not pointer-specific necessity: query conditioning alone remains sufficient in this toy.

Run with:

```bash
.venv/bin/python scripts/run_context_cancellation_probe.py --experiment query_cued_pointer_bottleneck --input-mode entangled
```

`query_cued_pointer_bottleneck` keeps the same query-cued multi-aspect setup, but routes pointer modes through a stricter control channel: the frame head can see observation plus query, while the recurrent decision path sees queryless observation plus the predicted/oracle frame pointer. It compares that compact pointer route against a full direct query path and direct query bottlenecks of size 2, 4, 8, and 16. The intended read is narrow: whether a frame pointer is useful as a compact control variable under bottleneck, not whether query cues are natural language.

Run with:

```bash
.venv/bin/python scripts/run_context_cancellation_probe.py --experiment temporal_disambiguation_refraction
```

`temporal_disambiguation_refraction` moves the refraction probe from static feature bundles to short streaming token sequences such as `dog bit me` and `dog bit his_tail`. It records hidden/logit state after each arriving token, compares ambiguous prefix behavior against final suffix resolution, and reports bag-of-tokens, full-sentence-static, zero-carry, shuffled-order, randomized-recurrent, and random-label controls. The first result supports suffix-driven frame resolution and order sensitivity, but prefix ambiguity is not clean and complete-token static shortcuts remain strong.

Embedding ablations:

```bash
.venv/bin/python scripts/run_context_cancellation_probe.py --experiment multi_aspect_token_refraction --input-mode entangled --embedding-mode fixed_sincos
```

`--embedding-mode` changes token-vector construction while keeping the recurrent model and task unchanged. Available modes are `learned`, `fixed_sincos`, `trainable_phase`, and `multi_band_phase`. In this precomputed-input probe, `learned` is the existing random-vector baseline and the phase modes are fixed phase-parameterized token tables rather than a larger trainable embedding architecture.

Raw wave / resonance ablations:

```bash
.venv/bin/python scripts/run_context_cancellation_probe.py --experiment multi_aspect_token_refraction --input-mode entangled --resonance-mode token_wave
```

`--resonance-mode` tests whether wave-like input construction improves the existing refraction task. Available modes are `none`, `token_wave`, `neuron_resonance`, `pointer_resonance`, and `pointer_resonance_signed`. These are small ablations around the same recurrent model, not a new large architecture. The current read is that `token_wave` is a weak positive bias, while explicit pointer/neuron resonance is not necessary for the present toy tasks.

Topology-prior ablations:

```bash
.venv/bin/python scripts/run_context_cancellation_probe.py --experiment latent_refraction --input-mode entangled --topology-mode hub_rich
```

`--topology-mode` changes only the recurrent mask prior while matching the edge budget to `random_sparse`. Available modes are `random_sparse`, `ring_sparse`, `reciprocal_motif`, `hub_rich`, `hub_degree_preserving_random`, `flywire_sampled`, `flywire_class_sampled`, and `flywire_degree_preserving_random`. The degree-preserving modes train from scratch on shuffled masks with the source topology's in/out degree sequence. The FlyWire modes use the local `/home/deck/work/flywire/mushroom_body.graphml` sample only as a small topology prior; they do not run full FlyWire or make a biology claim.

### Safe Claims

- Recurrent core recovery: a toy recurrent loop can recover or amplify task-causal structure from entangled interference.
- Latent refraction: a toy recurrent loop can reorient an entangled representation according to a task frame, giving decision authority to different feature groups.
- Multi-aspect refraction: a reused token can carry multiple possible aspects while the frame controls which relation gets decision authority.
- Frame-switch diagnostics: a toy diagnostic can test whether frame-conditioned authority switching is better explained by recurrent reorientation or static output routing.
- Reframe diagnostics: a toy diagnostic can test whether wrong early frame commitment can be reopened by an explicit reset/reframe event.
- Inferred frame pointer: a toy diagnostic can test whether the frame can be predicted from the input bundle and then used internally as a recurrent pointer.
- Query-cued frame pointer: a toy diagnostic can test whether a query-like goal cue can imply the frame while the same observation is reused under multiple labels.
- Query-cued pointer bottleneck: a toy diagnostic can test whether the inferred frame pointer is useful as a compact control channel compared with equally bottlenecked direct query conditioning.
- Temporal disambiguation refraction: a toy diagnostic can test whether streaming recurrence keeps prefix trajectories and resolves them when delayed suffix tokens arrive.
- Frequency embedding ablation: fixed sin/cos token vectors can be compared against the existing random-vector baseline without changing the recurrent mechanism.
- Raw wave resonance ablation: explicit pointer/neuron resonance can be tested against the simpler token-wave and recurrent baselines without making it the default mechanism.
- Topology-prior ablation: recurrent masks can be compared at matched edge budget, including synthetic motif/hub priors and a small local FlyWire-derived GraphML sample.
- Decision influence and probe decodability are separate: a feature can remain decodable while losing output authority.

### What Not To Claim

- Do not claim consciousness.
- Do not claim full VRAXION behavior.
- Do not claim biological equivalence.
- Do not claim production validation.
- Do not describe this as clean context or nuisance erasure unless the erasure-specific controls support it.

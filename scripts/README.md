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

### Safe Claims

- Recurrent core recovery: a toy recurrent loop can recover or amplify task-causal structure from entangled interference.
- Latent refraction: a toy recurrent loop can reorient an entangled representation according to a task frame, giving decision authority to different feature groups.
- Decision influence and probe decodability are separate: a feature can remain decodable while losing output authority.

### What Not To Claim

- Do not claim consciousness.
- Do not claim full VRAXION behavior.
- Do not claim biological equivalence.
- Do not claim production validation.
- Do not describe this as clean context or nuisance erasure unless the erasure-specific controls support it.

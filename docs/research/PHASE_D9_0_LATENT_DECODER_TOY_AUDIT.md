# Phase D9.0 Latent Genome Decoder Toy Audit

Verdict: **D9_DECODER_NO_LOCALITY**

This is an offline-only Python toy audit. It does not modify Rust, SAF, K(H), operator schedules, archive steering, or live search.

## Configuration

- tool_version: `d9.0-pic-toy-1`
- root_seed: `90210`
- H: `64`
- elapsed_sec: `130.83`
- reproducibility_hash: `a99e52968e9922cc`

## Killer Microtest

| Decoder | valid_rate | z->genome r | p | bootstrap CI |
|---|---:|---:|---:|---|
| `PIC` | 1.000 | 0.249 | 0.002 | [0.206, 0.300] |
| `NCI_RANDOM_HASH_DECODER` | 1.000 | 0.027 | 0.558 | [-0.114, 0.126] |
| `D_advr` | 1.000 | 0.199 | 0.002 | [0.149, 0.280] |

### Non-Triviality

```json
{
  "advr_descriptor_separation": 0.5186554377994084,
  "advr_graph_separation": 0.16676587301587298,
  "degree_distribution_entropy": 0.9907326606942454,
  "edge_structural_entropy": 0.2788046567732254,
  "gates": {
    "advr_descriptor_separation": true,
    "advr_graph_separation": true,
    "degree_distribution_entropy": true,
    "edge_structural_entropy": true,
    "rule_trace_diversity": true
  },
  "pass": true,
  "rule_trace_diversity": 1.0
}
```


### Gate Outcome

- verdict: `D9_DECODER_NO_LOCALITY`
- reasons: `pic_locality_below_absolute_gate`
- full_suite: skipped because the fail-fast killer gate did not pass

## Output Files

- `output\phase_d9_latent_genome_toy_20260428\analysis\summary.json`
- `output\phase_d9_latent_genome_toy_20260428\landscape_results.csv`
- `output\phase_d9_latent_genome_toy_20260428\genome_provenance.jsonl`
- `output\phase_d9_latent_genome_toy_20260428\control_baselines.json`

## Interpretation

A pass means only that the toy compiler survived the configured negative controls. It is not a live VRAXION improvement claim.
A fail verdict is expected to be useful: it identifies which D9 assumption died first.

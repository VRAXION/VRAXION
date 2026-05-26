# D34 Raven Pocket Corridor Baseline Suite Contract

## Objective
Implement and execute a CPU-parallel, same-budget baseline suite comparing available Raven pocket-corridor baselines.

## Methods target list
- random baseline
- simple neural net baseline (if available)
- direct VRAXION mutation
- shadow-clone mutation (if available)
- separate-individual population evolution (if available)
- DNA/u64/genome encoding (if available)

Unavailable baselines must be reported explicitly in `unavailable_baselines_report.json`.

## Same-budget policy
- Same seeds
- Same family set: row/col/pair/mirror/diag
- Same split protocol (test + OOD)
- Same compute budget class (`smoke`/`full`) per method

## CPU parallelism policy
- process-level isolation per (seed,method)
- worker count defaults to `min(os.cpu_count(), num_jobs)` when `--workers auto`
- thread env for workers: OMP/MKL/OPENBLAS set to 1
- machine utilization report required

## Hard gates
- random baseline must remain near 1/9
- no solved claim unless test>=0.90 and ood>=0.85 across multiple seeds
- failed seeds/methods must remain visible

## Decision policy
Use bounded decision labels only:
- `direct_mutation_beats_tested_dna_genome_encoding`
- `tested_dna_genome_encoding_beats_direct_mutation`
- `raven_corridor_search_not_solved`
- `direct_mutation_baseline_recorded_dna_genome_unavailable`
- `d34_suite_prepared_not_run`

with mapped next steps (D35 plans).

# Session Summary — 2026-04-08

## Major Findings (chronological)

### 1. Topology Representation Experiments (overnight)
- **ListNet** (sorted `Vec<Vec<u16>>`) = 6x faster step/s than INSTNCT library
- **But**: fair 1+1 comparison showed INSTNCT CSR wins at H≥512 (+26-130%)
- The overnight "6x" compared different search strategies (1+1 vs 1+9 jackpot)
- **Packed NeuronParams** (threshold+channel+polarity = 4 bytes): 8-10% spike speedup validated

### 2. Edges Don't Matter on Bigram (deterministic proof)
- Trained (298 edges) = removed (0) = random (298) = params-only (0) = all **20.3%**
- The phi-overlap geometry short-circuits topology: input charge directly visible at output
- The projection (W matrix) does all the learning, edge topology = pure noise on bigram
- This applies at ALL overlap levels tested (100%, phi, tiny, zero)

### 3. Edges MATTER for Computation (addition +52pp)
- INSTNCT: **56-60%** with edges vs **4%** without = **+52-56pp**
- ListNet: **24-44%** with edges vs **4%** without = **+20-40pp**
- Addition requires input transformation (a+b), not just lookup
- Edge topology IS the computing substrate for computation tasks

### 4. Addition Sweep Results
| Method | H | best% | step/s |
|---|---:|---:|---:|
| **INSTNCT** | 256 | **72%** | 1506 |
| INSTNCT | 128 | 60% | 2840 |
| INSTNCT | 512 | 68% | 874 |
| ListNet | 128 | 56% | 6630 |
| ListNet | 512 | 56% | 1906 |

### 5. 72% Ceiling Cannot Be Broken
- Jackpot 1+9: same 72% as 1+1 ES
- Edge cap 100/300/500: all same (seed-deterministic)
- 5 min vs 2 min: same results
- Freeze-crystal: identical to flat baseline
- **The ceiling is seed-dependent**: SDR patterns + init params determine max reachable accuracy

### 6. More Ticks = Worse
- 6 ticks: **72%** best, 64% mean
- 12 ticks: 64% best, 59% mean
- 24 ticks: 56% best, 53% mean
- The addition circuit is shallow (~23 edges), more ticks add noise

### 7. Memory Layout Findings
- Packed NeuronParams (read-only, threshold+channel+polarity): 8-10% faster
- Charge + activation (read-write): keep separate (write-back pollution at H≥2048)
- ListNet2 (all-in-one rows): 35% slower (AoS penalty on spike sweep)

## Architectural Understanding
1. **Task-dependent edge importance**: lookup (bigram) = edges irrelevant, computation (addition) = edges critical
2. **Phi overlap = short-circuit**: designed for language, input charge goes directly to output zone
3. **Separated I/O**: needed for computation tasks, forces signal through edge topology
4. **The projection (W) is the primary learner** on the current bigram task
5. **The 72% addition ceiling** is seed/SDR deterministic, not search-budget limited

## Running Experiment
- `addition_grow`: prune-freeze cycles on addition with jackpot — can incremental building break 72%?

## What's Pushed
- All experiments pushed to main repo
- Wiki Timeline-Archive + Rust Implementation Surface updated
- 20+ example files added to instnct-core/examples/

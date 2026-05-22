# FlyWire v783 Directed Connectivity Ingest

Status: local data ingest completed; raw data is not committed.

## Source

Zenodo DOI:

```text
10.5281/zenodo.10676866
```

Downloaded local files:

```text
S:\Git\_data\flywire\proofread_root_ids_783.npy
S:\Git\_data\flywire\proofread_connections_783.feather
```

The connection table MD5 matched the Zenodo record:

```text
proofread_connections_783.feather
size: 852,022,274 bytes
md5: F48F972D262323A102AED49AF1396B8A
```

Codex app access currently requires sign-in, but the Zenodo file is available
without auth and supports resumable download.

## Connection Table

Columns read:

```text
pre_pt_root_id
post_pt_root_id
neuropil
syn_count
```

Summary:

```text
rows: 16,847,997
unique_directed_pairs: 15,091,983
syn_count_sum: 54,492,922
syn_count_min: 1
syn_count_median: 1
syn_count_mean: 3.234
syn_count_max: 2405
```

## Directed Reciprocity

Reciprocal directed edge fraction by summed `syn_count` threshold:

| Threshold | Directed Edges | Reciprocal Fraction |
|---:|---:|---:|
| `>=1` | `15,091,983` | `0.2660` |
| `>=2` | `7,595,967` | `0.1921` |
| `>=5` | `2,700,513` | `0.1398` |
| `>=10` | `1,066,822` | `0.1153` |
| `>=25` | `246,962` | `0.0838` |
| `>=50` | `66,438` | `0.0614` |
| `>=100` | `15,837` | `0.0461` |

Interpretation:

```text
FlyWire is a directed graph.
A -> B does not imply B -> A.
Reverse effects exist only when a separate B -> A edge or longer directed loop exists.
The stronger the connection threshold, the less reciprocal the graph becomes.
```

This supports the local distinction from 018:

```text
not:
  one edge transmits both ways

but:
  the graph may contain a separate reverse edge or loop
```

## Degree Shape

Out-degree over unique directed pairs:

```text
neurons: 138,005
mean: 109.36
median: 80
p90: 205
p95: 289
p99: 600
p99.9: 1968.94
max: 9783
```

In-degree over unique directed pairs:

```text
neurons: 137,090
mean: 110.09
median: 70
p90: 231
p95: 335
p99: 665
p99.9: 1838.38
max: 10356
```

Interpretation:

```text
The useful prior to test is probably not exact FlyWire wiring.
It is directed heavy-tail / hub degree structure with explicit controls.
```

This matches the existing project result:

```text
FlyWire sampled topology did not beat random sparse on the prior toy task.
FlyWire degree-preserving random often beat raw FlyWire sampled.
Hub/degree concentration was the stronger reusable topology signal.
```

## Claim Boundary

This ingest supports only directed-topology facts and local prior design. It
does not support biology claims, production architecture, full VRAXION,
language grounding, consciousness, or physical quantum behavior.

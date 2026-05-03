# Phase D21K: A-Block Space Utilization

D21K measures how current A-block candidates use the 16D A-space.

This is a diagnostic phase, not a search.

Metrics:

```text
active_lane_count
lane_energy_min/mean/max/gini
rank
PCA spectrum / pca_dim_95
participation_ratio
pairwise distance min/mean/max
ASCII class separation ratio
byte reconstruction and margin
```

Candidates compared by default:

```text
d21a_current
d21g_natural
d21i_hidden_gain
d21j_hidden_natural
legacy_c19_h24
binary_c19_h16
```

Interpretation:

```text
active_lane_count:
  how many A16 lanes are materially used

rank:
  intrinsic linear dimension; with 8 input bits, rank above 8 is not expected

pca_dim_95:
  how many principal directions explain 95% of A-space variance

lane_energy_gini:
  low = balanced use, high = concentrated use

class_separation_ratio:
  ASCII class centroids separated relative to within-class spread
```

Generated output remains under `output/` and is not committed.

Current main result:

```text
candidate              exact margin geom  active rank pca95 PR    lane_gini pair_mean class_sep
---------------------- ----- ------ ----- ------ ---- ----- ----- --------- --------- ---------
d21a_current           1.000  4.000 0.669     16    8     8  8.00     0.000     5.572     1.126
d21g_natural           1.000  2.500 0.764     15    8     8  6.83     0.348     6.040     1.249
d21i_hidden_gain       1.000  4.000 0.731     16    8     8  6.81     0.148     6.052     1.156
d21j_hidden_natural    1.000  2.500 0.777     16    8     8  6.67     0.318     5.878     1.319
legacy_c19_h24         1.000 10.537 0.905     16   16     8  7.21     0.173    27.229     1.243
binary_c19_h16         1.000  0.002 0.905     16   16     8  6.99     0.258     3.163     1.152
```

Interpretation:

```text
D21A:
  uses all 16 lanes evenly, but as redundant 8-bit copy geometry

D21I:
  uses all 16 lanes, but one lane dominates; this confirms the bit-gain finding

D21J:
  uses all 16 lanes with less balance than D21A, but has the best class separation
  among current A-branch candidates

C19:
  uses a richer nonlinear hidden code; useful as reference, but not the current
  reciprocal A-block interface
```

Key answer: the current A16 space is being used, but with only 8 input bits the
intrinsic PCA dimension remains 8 for the reciprocal A candidates. The extra
lanes are used for redundancy, weighting, and geometry, not for new information.

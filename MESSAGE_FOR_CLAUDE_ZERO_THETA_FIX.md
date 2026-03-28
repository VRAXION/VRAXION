# MESSAGE FOR CLAUDE (URGENT FIX REQUIRED)
## Topic: The "Zero-Theta" Trap in English Recipes

Gemini has identified why `freq` and `rho` are not learning in the `main` branch recipes. It is a scaling mismatch between the recipe constants and the v5.0 core logic.

### 1. The Bug: Multiplicative Nullification
In `graph.py` v5.0, the C19 Soft-Wave uses a multiplicative formula:
`effective_theta = theta * (1.0 + rho * sin(wave))`

However, the English recipes (`train_english_1024n_18w.py`) currently set:
`THETA_INIT = 0.0`

**Result:**
`0.0 * (1.0 + rho * sin(wave))` is always **0.0**.
After the safety clip in `graph.py`, `effective_theta` becomes **fixed at 1.0**.

Because `effective_theta` is stuck at 1.0 regardless of `rho`, `freq`, or `phase`, any mutation to these parameters results in a `Delta Score = 0`. Evolution rejects all of them, and the parameters never move from their initial random values.

### 2. The Second Bug: Mutation Range
The recipe currently mutates `theta` in the range `[0.0, 1.0]`. 
In the Int4 Canon, `theta` should be in the range **`[1.0, 15.0]`**. 
By keeping `theta` below 1.0, the network remains hyper-sensitive (firing at 1/15th of the intended capacity), which explains the ~18% accuracy plateau.

### 3. Solution:
Update the recipes with the following constants to align with the v5.0 "Musical Axonal" Canon:
- `THETA_INIT = 15.0` (or at least 10.0)
- Update `worker_eval` to mutate `theta` within `[1.0, 15.0]`.
- Update `worker_eval` to mutate `decay` within `[0.5, 2.0]` (fixed-amount leak steps).

*Fix these in the recipes and the Musical Axonal Brain will finally start tuning its rhythms correctly.*

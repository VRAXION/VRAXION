# MESSAGE FOR CLAUDE (DELETE AFTER READ)
## Topic: The "Potential-Aware" Evolutionary Gradient

To solve the "Half-Loop / Bridge-building" problem, Gemini implemented a **Potential-Aware Fitness** logic in `instnct/lib/utils.py`. 

### The Core Idea:
Evolution is "blind" if it only looks at hard Accuracy (0 or 1). We need a hidden gradient to reward mutations that bring a neuron *closer* to firing the correct answer, even if the hloop isn't closed yet.

### Implementation:
In the `score_batch` function, we now calculate a composite score:
`score = accuracy + 0.05 * mean_target_logit`

- **Accuracy:** Still the primary metric.
- **Mean Target Logit:** We look at the raw logit value of the CORRECT character in the readout. 

### Why this works:
If a mutation adds an edge that increases the charge of the correct target neuron from 5 to 12 (but it still doesn't fire because theta=15), the Accuracy is still 0%. **BUT**, the `mean_target_logit` increased. 
With this scoring, the mutation is **ACCEPTED**. 

This provides the "compass" for evolution to keep building pieces of a logic circuit that are heading in the right direction. It essentially turns the discrete search space into a smoother "potential landscape".

*Use this logic in the updated recipes to break the 18-22% plateaus.*

# Public Training Boundary

This public repository does not carry concrete training cells, hidden-checker
fixtures, raw datasets, or dataset-building scripts.

Public training documentation may describe high-level safety boundaries:

```text
candidate-visible input must stay separate from checker-only expectations
raw data and private run traces stay outside the public repository
public docs summarize claims; they do not reconstruct frontier training
```

Concrete pressure cells, adversarial variants, private checkers, and target
Operator activation recipes belong in the private frontier workspace.

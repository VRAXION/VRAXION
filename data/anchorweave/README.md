# AnchorWeave

AnchorWeave stores relational episodic anchors, not direct concept definitions.
An AnchorCell preserves a concrete situation, its relational graph, salience,
available actions, predicted and actual outcomes, counterfactuals, memory hooks,
pre-symbol abstraction, symbol attach/reject decisions, human grounding
annotation, and outcome follow-up.

The canonical storage format is append-only JSONL. Derived SFT, DPO, reward,
eval, and graph views are generated later from the canonical cells.

## Privacy Boundary

Real or private annotations should stay private unless they have been explicitly
sanitized for release. The live growing database is private-by-default.

Private working paths are ignored by git:

- `data/anchorweave/private/`
- `data/anchorweave/cells/*.private.jsonl`
- `data/anchorweave/outcomes/*.private.jsonl`
- `local_data/anchorweave/`

Do not add real private or personal training data to public git history.

## Daily Workflow

Append one new canonical AnchorCell:

```bash
python tools/anchorweave/append_anchor_cell.py --input new_cell.json --db data/anchorweave/cells/anchorweave_v1.private.jsonl
```

Validate the append-only database:

```bash
python tools/anchorweave/validate_anchor_cells.py --db data/anchorweave/cells/anchorweave_v1.private.jsonl
```

Export initial training and analysis views:

```bash
python tools/anchorweave/export_anchorweave_views.py --db data/anchorweave/cells/anchorweave_v1.private.jsonl --out data/anchorweave/derived/
```

The validator also supports single `.json` AnchorCell files, which is useful for
checking examples or a draft cell before appending it.

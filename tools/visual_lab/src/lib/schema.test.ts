import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { diffGraphs, validateGraphSnapshot } from './schema';

const here = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(here, '../../../..');
const sampleRoot = resolve(repoRoot, 'docs/research/visual_samples/052_smoke_minimal');

function readJson(path: string): unknown {
  return JSON.parse(readFileSync(resolve(sampleRoot, path), 'utf8'));
}

describe('visual schema fixtures', () => {
  it('loads the full sample checkpoints', () => {
    const before = validateGraphSnapshot(readJson('visual/graph/checkpoint_000.json'));
    const after = validateGraphSnapshot(readJson('visual/graph/checkpoint_010.json'));
    expect(before.nodes.length).toBeGreaterThan(0);
    expect(after.edges.some((edge) => edge.role === 'pruned')).toBe(true);
  });

  it('loads a graph with optional fields removed', () => {
    const graph = validateGraphSnapshot(readJson('fixtures/optional_fields_removed_checkpoint_000.json'));
    expect(graph.nodes[0].selected_phase).toBeUndefined();
    expect(graph.nodes[0].route_order).toBeUndefined();
  });

  it('rejects unknown schema versions', () => {
    expect(() => validateGraphSnapshot(readJson('fixtures/wrong_schema_checkpoint_000.json'))).toThrow(
      /unsupported schema_version/
    );
  });

  it('computes a non-trivial checkpoint diff', () => {
    const before = validateGraphSnapshot(readJson('visual/graph/checkpoint_000.json'));
    const after = validateGraphSnapshot(readJson('visual/graph/checkpoint_010.json'));
    const diff = diffGraphs(before, after);
    expect(diff.added_edges).toBeGreaterThan(0);
    expect(diff.pruned_edges).toBeGreaterThan(0);
    expect(diff.retained_edges).toBeGreaterThan(0);
  });
});

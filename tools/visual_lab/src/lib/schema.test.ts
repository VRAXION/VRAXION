import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { diffGraphs, parseJsonl, validateGraphSnapshot } from './schema';

const here = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(here, '../../../..');
const sampleRoot = resolve(repoRoot, 'docs/research/visual_samples/052_smoke_minimal');
const realRunRoot = resolve(repoRoot, 'docs/research/visual_samples/053_real_run_ingest');

function readJson(path: string): unknown {
  return JSON.parse(readFileSync(resolve(sampleRoot, path), 'utf8'));
}

function readRealRunJson(path: string): unknown {
  return JSON.parse(readFileSync(resolve(realRunRoot, path), 'utf8'));
}

function readRealRunText(path: string): string {
  return readFileSync(resolve(realRunRoot, path), 'utf8');
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

  it('loads the 053 real-run ingest sample', () => {
    const finalGraph = validateGraphSnapshot(readRealRunJson('visual/graph/checkpoint_100.json'));
    expect(finalGraph.metadata?.source_probe).toBe('049_adversarial_frozen_eval_scale');
    expect(finalGraph.nodes.length).toBeGreaterThan(0);
    expect(finalGraph.edges.some((edge) => edge.role === 'pruned')).toBe(true);
  });

  it('diffs the 053 real-run ingest checkpoints', () => {
    const before = validateGraphSnapshot(readRealRunJson('visual/graph/checkpoint_000.json'));
    const after = validateGraphSnapshot(readRealRunJson('visual/graph/checkpoint_100.json'));
    const diff = diffGraphs(before, after);
    expect(diff.added_edges).toBeGreaterThan(0);
    expect(diff.pruned_edges).toBeGreaterThan(0);
    expect(diff.retained_edges).toBeGreaterThan(0);
  });

  it('keeps 053 visual metrics aligned with 049 source metrics', () => {
    const metrics = parseJsonl<{
      checkpoint: number;
      heldout_score: number;
      ood_score: number;
      family_min_accuracy: number;
      hard_distractor_accuracy: number;
      long_ood_accuracy: number;
      unique_output_count: number;
      expected_output_class_count: number;
      collapse_detected: boolean;
    }>(readRealRunText('visual/metrics.jsonl'));
    const finalMetric = metrics.find((row) => row.checkpoint === 100);
    expect(finalMetric?.heldout_score).toBe(1);
    expect(finalMetric?.ood_score).toBe(1);
    expect(finalMetric?.family_min_accuracy).toBe(1);
    expect(finalMetric?.hard_distractor_accuracy).toBe(1);
    expect(finalMetric?.long_ood_accuracy).toBe(1);
    expect(finalMetric?.unique_output_count).toBe(75);
    expect(finalMetric?.expected_output_class_count).toBe(75);
    expect(finalMetric?.collapse_detected).toBe(false);
  });
});

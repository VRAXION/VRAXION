import { describe, expect, it } from 'vitest';
import { readdirSync, readFileSync, statSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { diffGraphs, parseJsonl, validateGraphSnapshot } from './schema';
import {
  diffForCheckpoint,
  largerPlaybackBundle,
  renderMetadataFor,
  visualSampleBundles
} from './sample-bundle';

const here = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(here, '../../../..');
const sampleRoot = resolve(repoRoot, 'docs/research/visual_samples/052_smoke_minimal');
const realRunRoot = resolve(repoRoot, 'docs/research/visual_samples/053_real_run_ingest');
const largerPlaybackRoot = resolve(repoRoot, 'docs/research/visual_samples/054_larger_playback_smoke');

function readJson(path: string): unknown {
  return JSON.parse(readFileSync(resolve(sampleRoot, path), 'utf8'));
}

function readRealRunJson(path: string): unknown {
  return JSON.parse(readFileSync(resolve(realRunRoot, path), 'utf8'));
}

function readRealRunText(path: string): string {
  return readFileSync(resolve(realRunRoot, path), 'utf8');
}

function readLargerPlaybackText(path: string): string {
  return readFileSync(resolve(largerPlaybackRoot, path), 'utf8');
}

function directorySize(path: string): number {
  return readdirSync(path, { withFileTypes: true }).reduce((total, entry) => {
    const child = resolve(path, entry.name);
    return total + (entry.isDirectory() ? directorySize(child) : statSync(child).size);
  }, 0);
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

  it('exposes all committed sample bundles through the bundle selector', () => {
    expect(visualSampleBundles.map((bundle) => bundle.id)).toEqual([
      '054_larger_playback_smoke',
      '053_real_run_ingest',
      '052_smoke_minimal'
    ]);
  });

  it('loads the 054 larger playback bundle with checkpoints and ticks', () => {
    expect(largerPlaybackBundle.graphs).toHaveLength(12);
    expect(largerPlaybackBundle.ticks).toHaveLength(6);
    expect(largerPlaybackBundle.graphs[0].nodes.length).toBeGreaterThanOrEqual(120);
    expect(largerPlaybackBundle.graphs[0].edges.length).toBeGreaterThanOrEqual(180);
  });

  it('keeps the committed 054 sample size bounded', () => {
    expect(directorySize(largerPlaybackRoot)).toBeLessThan(2_500_000);
  });

  it('includes the required 054 event kinds', () => {
    const events = parseJsonl<{ kind: string }>(readLargerPlaybackText('visual/mutation_events.jsonl'));
    expect(new Set(events.map((event) => event.kind))).toEqual(
      new Set(['mutation', 'prune', 'repair', 'crystallize'])
    );
  });

  it('computes non-empty 054 first and previous diff modes', () => {
    const firstDiff = diffForCheckpoint(largerPlaybackBundle, 110, 'first');
    const previousDiff = diffForCheckpoint(largerPlaybackBundle, 110, 'previous');
    expect(firstDiff.added_edges).toBeGreaterThan(0);
    expect(firstDiff.pruned_edges).toBeGreaterThan(0);
    expect(firstDiff.retained_edges).toBeGreaterThan(0);
    expect(previousDiff.added_edges + previousDiff.pruned_edges + previousDiff.retained_edges).toBeGreaterThan(0);
  });

  it('changes the event list across selected 054 checkpoints', () => {
    const early = largerPlaybackBundle.events.filter((event) => event.checkpoint === 0);
    const late = largerPlaybackBundle.events.filter((event) => event.checkpoint === 110);
    expect(early.map((event) => event.kind)).not.toEqual(late.map((event) => event.kind));
    expect(late.length).toBeGreaterThan(early.length);
  });

  it('records viewer-side render metadata for 054', () => {
    const metadata = renderMetadataFor(largerPlaybackBundle, largerPlaybackBundle.graphs[11], 12.5);
    expect(metadata.render_duration_ms).toBe(12.5);
    expect(metadata.graph_node_count).toBeGreaterThanOrEqual(120);
    expect(metadata.graph_edge_count).toBeGreaterThanOrEqual(180);
    expect(metadata.checkpoint_count).toBe(12);
    expect(metadata.tick_count).toBe(6);
    expect(metadata.event_count).toBeGreaterThan(0);
  });
});

import type { DiffSummary, GraphSnapshot, MetricRow, MutationEvent, RenderMetadata } from './schema';
import { buildRenderMetadata, diffGraphs, validateGraphSnapshot } from './schema';
import realRunCheckpoint000 from '../../../../docs/research/visual_samples/053_real_run_ingest/visual/graph/checkpoint_000.json';
import realRunCheckpoint050 from '../../../../docs/research/visual_samples/053_real_run_ingest/visual/graph/checkpoint_050.json';
import realRunCheckpoint100 from '../../../../docs/research/visual_samples/053_real_run_ingest/visual/graph/checkpoint_100.json';
import closureCheckpoint000 from '../../../../docs/research/visual_samples/055_real_run_replay_closure/visual/graph/checkpoint_000.json';
import closureCheckpoint050 from '../../../../docs/research/visual_samples/055_real_run_replay_closure/visual/graph/checkpoint_050.json';
import closureCheckpoint100 from '../../../../docs/research/visual_samples/055_real_run_replay_closure/visual/graph/checkpoint_100.json';
import closureTick100000 from '../../../../docs/research/visual_samples/055_real_run_replay_closure/visual/ticks/checkpoint_100_tick_000.json';
import closureTick100001 from '../../../../docs/research/visual_samples/055_real_run_replay_closure/visual/ticks/checkpoint_100_tick_001.json';
import playbackCheckpoint000 from '../../../../docs/research/visual_samples/054_larger_playback_smoke/visual/graph/checkpoint_000.json';
import playbackCheckpoint010 from '../../../../docs/research/visual_samples/054_larger_playback_smoke/visual/graph/checkpoint_010.json';
import playbackCheckpoint020 from '../../../../docs/research/visual_samples/054_larger_playback_smoke/visual/graph/checkpoint_020.json';
import playbackCheckpoint030 from '../../../../docs/research/visual_samples/054_larger_playback_smoke/visual/graph/checkpoint_030.json';
import playbackCheckpoint040 from '../../../../docs/research/visual_samples/054_larger_playback_smoke/visual/graph/checkpoint_040.json';
import playbackCheckpoint050 from '../../../../docs/research/visual_samples/054_larger_playback_smoke/visual/graph/checkpoint_050.json';
import playbackCheckpoint060 from '../../../../docs/research/visual_samples/054_larger_playback_smoke/visual/graph/checkpoint_060.json';
import playbackCheckpoint070 from '../../../../docs/research/visual_samples/054_larger_playback_smoke/visual/graph/checkpoint_070.json';
import playbackCheckpoint080 from '../../../../docs/research/visual_samples/054_larger_playback_smoke/visual/graph/checkpoint_080.json';
import playbackCheckpoint090 from '../../../../docs/research/visual_samples/054_larger_playback_smoke/visual/graph/checkpoint_090.json';
import playbackCheckpoint100 from '../../../../docs/research/visual_samples/054_larger_playback_smoke/visual/graph/checkpoint_100.json';
import playbackCheckpoint110 from '../../../../docs/research/visual_samples/054_larger_playback_smoke/visual/graph/checkpoint_110.json';
import playbackTick030000 from '../../../../docs/research/visual_samples/054_larger_playback_smoke/visual/ticks/checkpoint_030_tick_000.json';
import playbackTick030001 from '../../../../docs/research/visual_samples/054_larger_playback_smoke/visual/ticks/checkpoint_030_tick_001.json';
import playbackTick070000 from '../../../../docs/research/visual_samples/054_larger_playback_smoke/visual/ticks/checkpoint_070_tick_000.json';
import playbackTick070001 from '../../../../docs/research/visual_samples/054_larger_playback_smoke/visual/ticks/checkpoint_070_tick_001.json';
import playbackTick110000 from '../../../../docs/research/visual_samples/054_larger_playback_smoke/visual/ticks/checkpoint_110_tick_000.json';
import playbackTick110001 from '../../../../docs/research/visual_samples/054_larger_playback_smoke/visual/ticks/checkpoint_110_tick_001.json';

export interface VisualSampleBundle {
  id: string;
  label: string;
  graphs: GraphSnapshot[];
  ticks: GraphSnapshot[];
  metrics: MetricRow[];
  events: MutationEvent[];
}

export const sampleGraphs: GraphSnapshot[] = [
  {
    schema_version: 'visual_snapshot_v1',
    run_id: 'stable_loop_phase_lock_052_visual_sample',
    checkpoint: 0,
    nodes: [
      { id: 'n_src', label: 'Source', role: 'source', x: -3, y: 0, activity: 1, is_active: true, is_pruned: false },
      { id: 'n_h1', label: 'H1', role: 'highway', x: -1.8, y: 0, activity: 0.9, is_active: true, is_pruned: false },
      { id: 'n_h2', label: 'H2', role: 'highway', x: -0.6, y: 0, activity: 0.88, is_active: true, is_pruned: false },
      { id: 'n_h3', label: 'H3', role: 'highway', x: 0.6, y: 0, activity: 0.82, is_active: true, is_pruned: false },
      { id: 'n_tgt', label: 'Target', role: 'target', x: 1.8, y: 0, activity: 1, is_active: true, is_pruned: false },
      { id: 'n_l1', label: 'L1', role: 'pocket', pocket_id: 'p_left', x: -1.8, y: -1, activity: 0.35, is_active: false, is_pruned: false },
      { id: 'n_l2', label: 'L2', role: 'candidate', pocket_id: 'p_left', x: -0.9, y: -1.5, activity: 0.25, is_active: false, is_pruned: false },
      { id: 'n_r1', label: 'R1', role: 'pocket', pocket_id: 'p_right', x: 0.6, y: 1, activity: 0.4, is_active: false, is_pruned: false },
      { id: 'n_r2', label: 'R2', role: 'pocket', pocket_id: 'p_right', x: 1.4, y: 1.5, activity: 0.18, is_active: false, is_pruned: false }
    ],
    edges: [
      { id: 'e_src_h1', source: 'n_src', target: 'n_h1', role: 'highway', weight: 1, directed: true, active_flow: 0.9, is_retained: true, is_pruned: false },
      { id: 'e_h1_h2', source: 'n_h1', target: 'n_h2', role: 'highway', weight: 1, directed: true, active_flow: 0.88, is_retained: true, is_pruned: false },
      { id: 'e_h2_h3', source: 'n_h2', target: 'n_h3', role: 'highway', weight: 1, directed: true, active_flow: 0.86, is_retained: true, is_pruned: false },
      { id: 'e_h3_tgt', source: 'n_h3', target: 'n_tgt', role: 'highway', weight: 1, directed: true, active_flow: 0.82, is_retained: true, is_pruned: false },
      { id: 'e_l1_l2_candidate', source: 'n_l1', target: 'n_l2', role: 'candidate', weight: 0.3, directed: true, active_flow: 0, is_retained: true, is_pruned: false }
    ],
    pockets: [
      { id: 'p_left', kind: 'side_pocket', node_ids: ['n_l1', 'n_l2'], bridge_nodes: ['n_h1'], mutation_count: 2, prune_ratio: 0.5 },
      { id: 'p_right', kind: 'side_pocket', node_ids: ['n_r1', 'n_r2'], bridge_nodes: ['n_h3'], mutation_count: 3, prune_ratio: 0.33 }
    ],
    routes: [
      { id: 'r_main', source: 'n_src', target: 'n_tgt', node_order: ['n_src', 'n_h1', 'n_h2', 'n_h3', 'n_tgt'], edge_order: ['e_src_h1', 'e_h1_h2', 'e_h2_h3', 'e_h3_tgt'], status: 'passing' }
    ]
  },
  {
    schema_version: 'visual_snapshot_v1',
    run_id: 'stable_loop_phase_lock_052_visual_sample',
    checkpoint: 10,
    nodes: [],
    edges: [
      { id: 'e_src_h1', source: 'n_src', target: 'n_h1', role: 'highway', weight: 1, directed: true, active_flow: 0.9, is_retained: true, is_pruned: false },
      { id: 'e_h1_h2', source: 'n_h1', target: 'n_h2', role: 'highway', weight: 1, directed: true, active_flow: 0.88, is_retained: true, is_pruned: false },
      { id: 'e_h2_l2_added', source: 'n_h2', target: 'n_l2', role: 'candidate', weight: 0.5, directed: true, active_flow: 0.2, is_retained: true, is_pruned: false },
      { id: 'e_r1_r2_pruned', source: 'n_r1', target: 'n_r2', role: 'pruned', weight: 0.1, directed: true, active_flow: 0, is_retained: false, is_pruned: true }
    ],
    pockets: [],
    routes: []
  }
];

sampleGraphs[1].nodes = sampleGraphs[0].nodes;
sampleGraphs[1].pockets = sampleGraphs[0].pockets;
sampleGraphs[1].routes = sampleGraphs[0].routes;

export const sampleMetrics: MetricRow[] = [
  { schema_version: 'visual_snapshot_v1', run_id: 'stable_loop_phase_lock_052_visual_sample', checkpoint: 0, heldout_score: 0.62, ood_score: 0.58, route_order_accuracy: 0.71, missing_successor_count: 2, output_entropy: 3.1, collapse_detected: false },
  { schema_version: 'visual_snapshot_v1', run_id: 'stable_loop_phase_lock_052_visual_sample', checkpoint: 10, heldout_score: 1, ood_score: 1, route_order_accuracy: 1, missing_successor_count: 0, output_entropy: 4.2, collapse_detected: false }
];

export const sampleEvents: MutationEvent[] = [
  { id: 'ev_mut_001', schema_version: 'visual_snapshot_v1', run_id: 'stable_loop_phase_lock_052_visual_sample', checkpoint: 0, kind: 'mutation', node_ids: ['n_l2'], edge_ids: ['e_l1_l2_candidate'], label: 'candidate pocket edge added' },
  { id: 'ev_prune_001', schema_version: 'visual_snapshot_v1', run_id: 'stable_loop_phase_lock_052_visual_sample', checkpoint: 10, kind: 'prune', node_ids: ['n_r2'], edge_ids: ['e_r1_r2_pruned'], label: 'non-route pocket branch pruned' }
];

export const smokeSampleBundle: VisualSampleBundle = {
  id: '052_smoke_minimal',
  label: '052 smoke minimal',
  graphs: sampleGraphs,
  ticks: [],
  metrics: sampleMetrics,
  events: sampleEvents
};

export const realRunIngestGraphs: GraphSnapshot[] = [
  validateGraphSnapshot(realRunCheckpoint000),
  validateGraphSnapshot(realRunCheckpoint050),
  validateGraphSnapshot(realRunCheckpoint100)
];

export const realRunIngestMetrics: MetricRow[] = [
  {
    schema_version: 'visual_snapshot_v1',
    run_id: 'stable_loop_phase_lock_053_real_run_ingest',
    checkpoint: 0,
    source_arm: 'NO_ROUTE_GRAMMAR_ADVERSARIAL_FROZEN_BASELINE',
    heldout_score: 0.060546875,
    ood_score: 0.048828125,
    family_min_accuracy: 0,
    hard_distractor_accuracy: 0,
    long_ood_accuracy: 0.1953125,
    route_order_accuracy: 0,
    missing_successor_count: 6,
    output_entropy: 0,
    unique_output_count: 1,
    expected_output_class_count: 75,
    collapse_detected: true
  },
  {
    schema_version: 'visual_snapshot_v1',
    run_id: 'stable_loop_phase_lock_053_real_run_ingest',
    checkpoint: 50,
    source_arm: 'FROZEN_EVAL_048_REFERENCE',
    heldout_score: 0.166015625,
    ood_score: 0.15625,
    family_min_accuracy: 0,
    hard_distractor_accuracy: 0,
    long_ood_accuracy: 0.625,
    route_order_accuracy: 0,
    missing_successor_count: 6,
    output_entropy: 0.6467100819075322,
    unique_output_count: 4,
    expected_output_class_count: 75,
    collapse_detected: true
  },
  {
    schema_version: 'visual_snapshot_v1',
    run_id: 'stable_loop_phase_lock_053_real_run_ingest',
    checkpoint: 100,
    source_arm: 'ADVERSARIAL_FROZEN_ROUTE_GRAMMAR_TRAIN_AND_INFER',
    heldout_score: 1,
    ood_score: 1,
    family_min_accuracy: 1,
    hard_distractor_accuracy: 1,
    long_ood_accuracy: 1,
    route_order_accuracy: 1,
    missing_successor_count: 0,
    output_entropy: 5.40437231483324,
    unique_output_count: 75,
    expected_output_class_count: 75,
    collapse_detected: false
  }
];

export const realRunIngestEvents: MutationEvent[] = [
  {
    id: 'ev_053_mutation_from_reference',
    schema_version: 'visual_snapshot_v1',
    run_id: 'stable_loop_phase_lock_053_real_run_ingest',
    checkpoint: 50,
    tick: 0,
    kind: 'mutation',
    node_ids: ['n_diag'],
    edge_ids: ['e_diag_candidate'],
    label: 'diagnostic route candidate exposed from 049 reference metrics'
  },
  {
    id: 'ev_053_prune_control_shortcut',
    schema_version: 'visual_snapshot_v1',
    run_id: 'stable_loop_phase_lock_053_real_run_ingest',
    checkpoint: 100,
    tick: 0,
    kind: 'prune',
    node_ids: ['n_ctrl_majority'],
    edge_ids: ['e_majority_shortcut_pruned'],
    label: 'majority/static shortcut control pruned in passing ingest view'
  },
  {
    id: 'ev_053_repair_successor_chain',
    schema_version: 'visual_snapshot_v1',
    run_id: 'stable_loop_phase_lock_053_real_run_ingest',
    checkpoint: 100,
    tick: 1,
    kind: 'repair',
    node_ids: ['n_h2', 'n_h3'],
    edge_ids: ['e_h2_h3', 'e_h3_tgt'],
    label: 'successor chain completed by route-grammar positive arm'
  }
];

export const realRunIngestBundle: VisualSampleBundle = {
  id: '053_real_run_ingest',
  label: '053 real-run ingest from 049',
  graphs: realRunIngestGraphs,
  ticks: [],
  metrics: realRunIngestMetrics,
  events: realRunIngestEvents
};

export const closureReplayGraphs: GraphSnapshot[] = [
  validateGraphSnapshot(closureCheckpoint000),
  validateGraphSnapshot(closureCheckpoint050),
  validateGraphSnapshot(closureCheckpoint100)
];

export const closureReplayTicks: GraphSnapshot[] = [
  validateGraphSnapshot(closureTick100000),
  validateGraphSnapshot(closureTick100001)
];

export const closureReplayMetrics: MetricRow[] = [
  {
    schema_version: 'visual_snapshot_v1',
    run_id: 'stable_loop_phase_lock_055_real_run_replay_closure',
    checkpoint: 0,
    source_arm: 'NO_ROUTE_GRAMMAR_ADVERSARIAL_FROZEN_BASELINE',
    heldout_score: 0.060546875,
    ood_score: 0.048828125,
    family_min_accuracy: 0,
    hard_distractor_accuracy: 0,
    long_ood_accuracy: 0.1953125,
    route_order_accuracy: 0,
    missing_successor_count: 6,
    output_entropy: 0,
    unique_output_count: 1,
    expected_output_class_count: 75,
    top_output_rate: 1,
    majority_output_rate: 1,
    non_route_regression_delta: -1,
    route_api_overuse_rate: 0,
    collapse_detected: true
  },
  {
    schema_version: 'visual_snapshot_v1',
    run_id: 'stable_loop_phase_lock_055_real_run_replay_closure',
    checkpoint: 50,
    source_arm: 'FROZEN_EVAL_048_REFERENCE',
    heldout_score: 0.166015625,
    ood_score: 0.15625,
    family_min_accuracy: 0,
    hard_distractor_accuracy: 0,
    long_ood_accuracy: 0.625,
    route_order_accuracy: 0,
    missing_successor_count: 6,
    output_entropy: 0.6467100819075322,
    unique_output_count: 4,
    expected_output_class_count: 75,
    top_output_rate: 0.8935546875,
    majority_output_rate: 0.8935546875,
    non_route_regression_delta: -1,
    route_api_overuse_rate: 0,
    collapse_detected: true
  },
  {
    schema_version: 'visual_snapshot_v1',
    run_id: 'stable_loop_phase_lock_055_real_run_replay_closure',
    checkpoint: 100,
    source_arm: 'ADVERSARIAL_FROZEN_ROUTE_GRAMMAR_TRAIN_AND_INFER',
    heldout_score: 1,
    ood_score: 1,
    family_min_accuracy: 1,
    hard_distractor_accuracy: 1,
    long_ood_accuracy: 1,
    route_order_accuracy: 1,
    missing_successor_count: 0,
    output_entropy: 5.40437231483324,
    unique_output_count: 75,
    expected_output_class_count: 75,
    top_output_rate: 0.0732421875,
    majority_output_rate: 0.0546875,
    non_route_regression_delta: 0,
    route_api_overuse_rate: 0.04,
    collapse_detected: false
  }
];

export const closureReplayEvents: MutationEvent[] = [
  {
    id: 'ev_055_mutation_from_reference',
    schema_version: 'visual_snapshot_v1',
    run_id: 'stable_loop_phase_lock_055_real_run_replay_closure',
    checkpoint: 50,
    tick: 0,
    kind: 'mutation',
    node_ids: ['n_diag'],
    edge_ids: ['e_diag_candidate'],
    label: 'diagnostic route candidate exposed from 049/050 source metrics'
  },
  {
    id: 'ev_055_prune_control_shortcut',
    schema_version: 'visual_snapshot_v1',
    run_id: 'stable_loop_phase_lock_055_real_run_replay_closure',
    checkpoint: 100,
    tick: 0,
    kind: 'prune',
    node_ids: ['n_ctrl_majority'],
    edge_ids: ['e_majority_shortcut_pruned'],
    label: 'majority/static shortcut control failed and is shown as pruned'
  },
  {
    id: 'ev_055_repair_successor_chain',
    schema_version: 'visual_snapshot_v1',
    run_id: 'stable_loop_phase_lock_055_real_run_replay_closure',
    checkpoint: 100,
    tick: 1,
    kind: 'repair',
    node_ids: ['n_h2', 'n_h3'],
    edge_ids: ['e_h2_h3', 'e_h3_tgt'],
    label: 'successor chain completed by route-grammar positive arm'
  },
  {
    id: 'ev_055_crystallize_closure_gate',
    schema_version: 'visual_snapshot_v1',
    run_id: 'stable_loop_phase_lock_055_real_run_replay_closure',
    checkpoint: 100,
    tick: 1,
    kind: 'crystallize',
    node_ids: ['n_h2', 'n_h3', 'n_tgt'],
    edge_ids: ['e_h2_h3', 'e_h3_tgt'],
    label:
      'VISUAL_SECTION_V1_CLOSED marker: passing route metric projection crystallized; collapse controls remain failed'
  }
];

export const closureReplayBundle: VisualSampleBundle = {
  id: '055_real_run_replay_closure',
  label: '055 real-run replay closure',
  graphs: closureReplayGraphs,
  ticks: closureReplayTicks,
  metrics: closureReplayMetrics,
  events: closureReplayEvents
};

export const largerPlaybackGraphs: GraphSnapshot[] = [
  playbackCheckpoint000,
  playbackCheckpoint010,
  playbackCheckpoint020,
  playbackCheckpoint030,
  playbackCheckpoint040,
  playbackCheckpoint050,
  playbackCheckpoint060,
  playbackCheckpoint070,
  playbackCheckpoint080,
  playbackCheckpoint090,
  playbackCheckpoint100,
  playbackCheckpoint110
].map(validateGraphSnapshot);

export const largerPlaybackTicks: GraphSnapshot[] = [
  playbackTick030000,
  playbackTick030001,
  playbackTick070000,
  playbackTick070001,
  playbackTick110000,
  playbackTick110001
].map(validateGraphSnapshot);

export const largerPlaybackMetrics: MetricRow[] = largerPlaybackGraphs.map((graph, index) => ({
  schema_version: 'visual_snapshot_v1',
  run_id: 'stable_loop_phase_lock_054_larger_playback_smoke',
  checkpoint: graph.checkpoint,
  source_arm: 'LARGER_PLAYBACK_DETERMINISTIC_VISUAL',
  heldout_score: Math.min(0.45 + (index / 11) * 0.5, 0.98),
  ood_score: Math.min(0.38 + (index / 11) * 0.54, 0.96),
  family_min_accuracy: Math.min((index / 11) * 0.94, 0.94),
  hard_distractor_accuracy: Math.min(0.2 + (index / 11) * 0.72, 0.92),
  long_ood_accuracy: Math.min(0.25 + (index / 11) * 0.68, 0.93),
  route_order_accuracy: Math.min(0.35 + (index / 11) * 0.62, 0.97),
  missing_successor_count: Math.max(12 - (index + 1), 0),
  output_entropy: 2.2 + (index / 11) * 3.0,
  unique_output_count: 12 + index * 5,
  expected_output_class_count: 75,
  collapse_detected: false
}));

export const largerPlaybackEvents: MutationEvent[] = [
  ...largerPlaybackGraphs.filter((_, index) => index % 2 === 0).map((graph, index) => ({
    id: `ev_054_mut_${graph.checkpoint.toString().padStart(3, '0')}`,
    schema_version: 'visual_snapshot_v1' as const,
    run_id: 'stable_loop_phase_lock_054_larger_playback_smoke',
    checkpoint: graph.checkpoint,
    tick: 0,
    kind: 'mutation' as const,
    node_ids: [`n_p${(index % 10).toString().padStart(2, '0')}_02`],
    edge_ids: [`e_p${(index % 10).toString().padStart(2, '0')}_02_03`],
    label: 'candidate pocket edge introduced'
  })),
  ...largerPlaybackGraphs.slice(3).map((graph, index) => ({
    id: `ev_054_prune_${graph.checkpoint.toString().padStart(3, '0')}`,
    schema_version: 'visual_snapshot_v1' as const,
    run_id: 'stable_loop_phase_lock_054_larger_playback_smoke',
    checkpoint: graph.checkpoint,
    tick: 0,
    kind: 'prune' as const,
    node_ids: [`n_p${((index + 5) % 10).toString().padStart(2, '0')}_05`],
    edge_ids: [`e_pruned_shortcut_p${((index + 5) % 10).toString().padStart(2, '0')}`],
    label: 'shortcut edge pruned'
  })),
  ...largerPlaybackGraphs.slice(5).map((graph, index) => ({
    id: `ev_054_repair_${graph.checkpoint.toString().padStart(3, '0')}`,
    schema_version: 'visual_snapshot_v1' as const,
    run_id: 'stable_loop_phase_lock_054_larger_playback_smoke',
    checkpoint: graph.checkpoint,
    tick: 1,
    kind: 'repair' as const,
    node_ids: [`n_h${((index + 5) * 3 % 48).toString().padStart(2, '0')}`],
    edge_ids: [`e_h${((index + 5) * 3 % 47).toString().padStart(2, '0')}_h${(((index + 5) * 3 % 47) + 1).toString().padStart(2, '0')}`],
    label: 'successor continuity repaired'
  })),
  ...largerPlaybackGraphs.slice(8).map((graph, index) => ({
    id: `ev_054_crystallize_${graph.checkpoint.toString().padStart(3, '0')}`,
    schema_version: 'visual_snapshot_v1' as const,
    run_id: 'stable_loop_phase_lock_054_larger_playback_smoke',
    checkpoint: graph.checkpoint,
    tick: 1,
    kind: 'crystallize' as const,
    node_ids: [`n_h${((index + 8) * 4 % 48).toString().padStart(2, '0')}`],
    edge_ids: [`e_h${((index + 8) * 4 % 47).toString().padStart(2, '0')}_h${(((index + 8) * 4 % 47) + 1).toString().padStart(2, '0')}`],
    label: 'route edge crystallized'
  }))
];

export const largerPlaybackBundle: VisualSampleBundle = {
  id: '054_larger_playback_smoke',
  label: '054 larger playback smoke',
  graphs: largerPlaybackGraphs,
  ticks: largerPlaybackTicks,
  metrics: largerPlaybackMetrics,
  events: largerPlaybackEvents
};

export const visualSampleBundles: VisualSampleBundle[] = [
  closureReplayBundle,
  largerPlaybackBundle,
  realRunIngestBundle,
  smokeSampleBundle
];

export const activeSampleBundle = closureReplayBundle;

export type DiffMode = 'first' | 'previous';

export function bundleById(id: string): VisualSampleBundle {
  return visualSampleBundles.find((bundle) => bundle.id === id) ?? activeSampleBundle;
}

export function checkpointsFor(bundle: VisualSampleBundle): number[] {
  return bundle.graphs.map((graph) => graph.checkpoint);
}

export function graphForCheckpoint(bundle: VisualSampleBundle, checkpoint: number): GraphSnapshot {
  return bundle.graphs.find((graph) => graph.checkpoint === checkpoint) ?? bundle.graphs[0];
}

export function ticksForCheckpoint(bundle: VisualSampleBundle, checkpoint: number): GraphSnapshot[] {
  return bundle.ticks.filter((tick) => tick.checkpoint === checkpoint);
}

export function eventsForCheckpoint(
  bundle: VisualSampleBundle,
  checkpoint: number,
  tick: number | 'all' = 'all'
): MutationEvent[] {
  return bundle.events.filter((event) => {
    if (event.checkpoint !== checkpoint) return false;
    return tick === 'all' || event.tick === undefined || event.tick === tick;
  });
}

export function diffForCheckpoint(
  bundle: VisualSampleBundle,
  checkpoint: number,
  mode: DiffMode
): DiffSummary {
  const selectedIndex = Math.max(bundle.graphs.findIndex((graph) => graph.checkpoint === checkpoint), 0);
  const after = bundle.graphs[selectedIndex] ?? bundle.graphs[0];
  const before =
    mode === 'previous'
      ? bundle.graphs[Math.max(selectedIndex - 1, 0)] ?? bundle.graphs[0]
      : bundle.graphs[0];
  return diffGraphs(before, after);
}

export function renderMetadataFor(bundle: VisualSampleBundle, graph: GraphSnapshot, renderMs = 0): RenderMetadata {
  return buildRenderMetadata(graph, bundle.graphs.length, bundle.ticks.length, bundle.events.length, renderMs);
}

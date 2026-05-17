import type { GraphSnapshot, MetricRow, MutationEvent } from './schema';
import { validateGraphSnapshot } from './schema';
import realRunCheckpoint000 from '../../../../docs/research/visual_samples/053_real_run_ingest/visual/graph/checkpoint_000.json';
import realRunCheckpoint050 from '../../../../docs/research/visual_samples/053_real_run_ingest/visual/graph/checkpoint_050.json';
import realRunCheckpoint100 from '../../../../docs/research/visual_samples/053_real_run_ingest/visual/graph/checkpoint_100.json';

export interface VisualSampleBundle {
  id: string;
  label: string;
  graphs: GraphSnapshot[];
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
  metrics: realRunIngestMetrics,
  events: realRunIngestEvents
};

export const visualSampleBundles: VisualSampleBundle[] = [realRunIngestBundle, smokeSampleBundle];
export const activeSampleBundle = realRunIngestBundle;

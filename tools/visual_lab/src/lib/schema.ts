export const VISUAL_SCHEMA_VERSION = 'visual_snapshot_v1';

export type NodeRole = 'highway' | 'pocket' | 'source' | 'target' | 'relay' | 'candidate';
export type EdgeRole = 'highway' | 'pocket' | 'bridge' | 'candidate' | 'pruned';
export type EventKind = 'mutation' | 'prune' | 'repair' | 'crystallize';

export interface GraphNode {
  id: string;
  label: string;
  role: NodeRole;
  pocket_id?: string;
  x: number;
  y: number;
  activity: number;
  selected_phase?: number;
  route_order?: number;
  is_active: boolean;
  is_pruned: boolean;
  metadata?: Record<string, unknown>;
}

export interface GraphEdge {
  id: string;
  source: string;
  target: string;
  role: EdgeRole;
  weight: number;
  directed: boolean;
  active_flow: number;
  is_retained: boolean;
  is_pruned: boolean;
  metadata?: Record<string, unknown>;
}

export interface PocketSummary {
  id: string;
  kind: string;
  node_ids: string[];
  bridge_nodes: string[];
  mutation_count: number;
  prune_ratio: number;
}

export interface RouteTrace {
  id: string;
  source: string;
  target: string;
  node_order: string[];
  edge_order: string[];
  status: string;
}

export interface GraphSnapshot {
  schema_version: typeof VISUAL_SCHEMA_VERSION;
  run_id: string;
  checkpoint: number;
  tick?: number;
  nodes: GraphNode[];
  edges: GraphEdge[];
  pockets: PocketSummary[];
  routes: RouteTrace[];
  metadata?: Record<string, unknown>;
}

export interface MetricRow {
  schema_version: typeof VISUAL_SCHEMA_VERSION;
  run_id: string;
  checkpoint: number;
  source_arm?: string;
  heldout_score: number;
  ood_score: number;
  family_min_accuracy?: number;
  hard_distractor_accuracy?: number;
  long_ood_accuracy?: number;
  route_order_accuracy: number;
  missing_successor_count: number;
  output_entropy: number;
  unique_output_count?: number;
  expected_output_class_count?: number;
  collapse_detected: boolean;
}

export interface MutationEvent {
  id: string;
  schema_version: typeof VISUAL_SCHEMA_VERSION;
  run_id: string;
  checkpoint: number;
  tick?: number;
  kind: EventKind;
  node_ids: string[];
  edge_ids: string[];
  label: string;
}

export interface DiffSummary {
  added_edges: number;
  removed_edges: number;
  pruned_edges: number;
  retained_edges: number;
}

export interface RenderMetadata {
  render_duration_ms: number;
  graph_node_count: number;
  graph_edge_count: number;
  checkpoint_count: number;
  tick_count: number;
  event_count: number;
}

const nodeRoles = new Set<NodeRole>(['highway', 'pocket', 'source', 'target', 'relay', 'candidate']);
const edgeRoles = new Set<EdgeRole>(['highway', 'pocket', 'bridge', 'candidate', 'pruned']);

export function parseJsonl<T>(text: string): T[] {
  return text
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean)
    .map((line) => JSON.parse(line) as T);
}

export function validateGraphSnapshot(value: unknown): GraphSnapshot {
  const snapshot = expectRecord(value, 'snapshot');
  expectSchema(snapshot);
  const nodes = expectArray(snapshot.nodes, 'nodes').map(validateNode);
  const edges = expectArray(snapshot.edges, 'edges').map(validateEdge);
  const pockets = expectArray(snapshot.pockets, 'pockets').map(validatePocket);
  const routes = expectArray(snapshot.routes, 'routes').map(validateRoute);
  return {
    schema_version: VISUAL_SCHEMA_VERSION,
    run_id: expectString(snapshot.run_id, 'run_id'),
    checkpoint: expectNumber(snapshot.checkpoint, 'checkpoint'),
    tick: optionalNumber(snapshot.tick, 'tick'),
    nodes,
    edges,
    pockets,
    routes,
    metadata: optionalRecord(snapshot.metadata)
  };
}

export function diffGraphs(before: GraphSnapshot, after: GraphSnapshot): DiffSummary {
  const beforeEdges = new Map(before.edges.map((edge) => [edge.id, edge]));
  const afterEdges = new Map(after.edges.map((edge) => [edge.id, edge]));
  let added_edges = 0;
  let removed_edges = 0;
  let pruned_edges = 0;
  let retained_edges = 0;

  for (const [id, edge] of afterEdges) {
    if (!beforeEdges.has(id)) added_edges += 1;
    if (edge.is_pruned || edge.role === 'pruned') pruned_edges += 1;
    if (beforeEdges.has(id) && edge.is_retained && !edge.is_pruned) retained_edges += 1;
  }
  for (const id of beforeEdges.keys()) {
    if (!afterEdges.has(id)) removed_edges += 1;
  }
  return { added_edges, removed_edges, pruned_edges, retained_edges };
}

export function buildRenderMetadata(
  graph: GraphSnapshot,
  checkpoint_count: number,
  tick_count: number,
  event_count: number,
  render_duration_ms = 0
): RenderMetadata {
  return {
    render_duration_ms,
    graph_node_count: graph.nodes.length,
    graph_edge_count: graph.edges.length,
    checkpoint_count,
    tick_count,
    event_count
  };
}

export function roleColor(role: NodeRole | EdgeRole, active = false): string {
  switch (role) {
    case 'highway':
      return active ? '#38f5ff' : '#0891b2';
    case 'pocket':
      return active ? '#fbbf24' : '#d97706';
    case 'source':
    case 'target':
      return active ? '#34d399' : '#059669';
    case 'candidate':
      return active ? '#c084fc' : '#8b5cf6';
    case 'pruned':
      return active ? '#fb7185' : '#be123c';
    case 'bridge':
      return active ? '#7dd3fc' : '#0284c7';
    case 'relay':
      return active ? '#e2e8f0' : '#64748b';
    default:
      return active ? '#e2e8f0' : '#64748b';
  }
}

export function roleLabel(role: NodeRole | EdgeRole): string {
  switch (role) {
    case 'highway':
      return 'Highway backbone';
    case 'pocket':
      return 'Side pocket';
    case 'source':
      return 'Source';
    case 'target':
      return 'Target';
    case 'candidate':
      return 'Candidate / mutable';
    case 'pruned':
      return 'Pruned / rejected';
    case 'bridge':
      return 'Bridge edge';
    case 'relay':
      return 'Relay';
    default:
      return role;
  }
}

function validateNode(value: unknown): GraphNode {
  const node = expectRecord(value, 'node');
  const role = expectString(node.role, 'node.role') as NodeRole;
  if (!nodeRoles.has(role)) throw new Error(`unknown node role: ${role}`);
  return {
    id: expectString(node.id, 'node.id'),
    label: expectString(node.label, 'node.label'),
    role,
    pocket_id: optionalString(node.pocket_id, 'node.pocket_id'),
    x: expectNumber(node.x, 'node.x'),
    y: expectNumber(node.y, 'node.y'),
    activity: expectNumber(node.activity, 'node.activity'),
    selected_phase: optionalNumber(node.selected_phase, 'node.selected_phase'),
    route_order: optionalNumber(node.route_order, 'node.route_order'),
    is_active: expectBoolean(node.is_active, 'node.is_active'),
    is_pruned: expectBoolean(node.is_pruned, 'node.is_pruned'),
    metadata: optionalRecord(node.metadata)
  };
}

function validateEdge(value: unknown): GraphEdge {
  const edge = expectRecord(value, 'edge');
  const role = expectString(edge.role, 'edge.role') as EdgeRole;
  if (!edgeRoles.has(role)) throw new Error(`unknown edge role: ${role}`);
  return {
    id: expectString(edge.id, 'edge.id'),
    source: expectString(edge.source, 'edge.source'),
    target: expectString(edge.target, 'edge.target'),
    role,
    weight: expectNumber(edge.weight, 'edge.weight'),
    directed: expectBoolean(edge.directed, 'edge.directed'),
    active_flow: expectNumber(edge.active_flow, 'edge.active_flow'),
    is_retained: expectBoolean(edge.is_retained, 'edge.is_retained'),
    is_pruned: expectBoolean(edge.is_pruned, 'edge.is_pruned'),
    metadata: optionalRecord(edge.metadata)
  };
}

function validatePocket(value: unknown): PocketSummary {
  const pocket = expectRecord(value, 'pocket');
  return {
    id: expectString(pocket.id, 'pocket.id'),
    kind: expectString(pocket.kind, 'pocket.kind'),
    node_ids: expectArray(pocket.node_ids, 'pocket.node_ids').map((id) => expectString(id, 'node_id')),
    bridge_nodes: expectArray(pocket.bridge_nodes, 'pocket.bridge_nodes').map((id) =>
      expectString(id, 'bridge_node')
    ),
    mutation_count: expectNumber(pocket.mutation_count, 'pocket.mutation_count'),
    prune_ratio: expectNumber(pocket.prune_ratio, 'pocket.prune_ratio')
  };
}

function validateRoute(value: unknown): RouteTrace {
  const route = expectRecord(value, 'route');
  return {
    id: expectString(route.id, 'route.id'),
    source: expectString(route.source, 'route.source'),
    target: expectString(route.target, 'route.target'),
    node_order: expectArray(route.node_order, 'route.node_order').map((id) => expectString(id, 'node_order')),
    edge_order: expectArray(route.edge_order, 'route.edge_order').map((id) => expectString(id, 'edge_order')),
    status: expectString(route.status, 'route.status')
  };
}

function expectSchema(record: Record<string, unknown>): void {
  const version = expectString(record.schema_version, 'schema_version');
  if (version !== VISUAL_SCHEMA_VERSION) {
    throw new Error(`unsupported schema_version: ${version}`);
  }
}

function expectRecord(value: unknown, name: string): Record<string, unknown> {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    throw new Error(`${name} must be an object`);
  }
  return value as Record<string, unknown>;
}

function optionalRecord(value: unknown): Record<string, unknown> | undefined {
  if (value === undefined) return undefined;
  return expectRecord(value, 'metadata');
}

function expectArray(value: unknown, name: string): unknown[] {
  if (!Array.isArray(value)) throw new Error(`${name} must be an array`);
  return value;
}

function expectString(value: unknown, name: string): string {
  if (typeof value !== 'string') throw new Error(`${name} must be a string`);
  return value;
}

function optionalString(value: unknown, name: string): string | undefined {
  if (value === undefined) return undefined;
  return expectString(value, name);
}

function expectNumber(value: unknown, name: string): number {
  if (typeof value !== 'number' || !Number.isFinite(value)) throw new Error(`${name} must be a number`);
  return value;
}

function optionalNumber(value: unknown, name: string): number | undefined {
  if (value === undefined) return undefined;
  return expectNumber(value, name);
}

function expectBoolean(value: unknown, name: string): boolean {
  if (typeof value !== 'boolean') throw new Error(`${name} must be a boolean`);
  return value;
}

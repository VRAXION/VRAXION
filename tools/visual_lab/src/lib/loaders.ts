import type { GraphSnapshot, MetricRow, MutationEvent, RouteTrace, PocketSummary } from './schema';
import { parseJsonl, validateGraphSnapshot } from './schema';

export interface VisualBundleData {
  graphs: GraphSnapshot[];
  metrics: MetricRow[];
  events: MutationEvent[];
  routes: RouteTrace[];
  pockets: PocketSummary[];
}

export async function loadVisualBundle(
  baseUrl: string,
  fetcher: (input: string) => Promise<{ text(): Promise<string>; json(): Promise<unknown> }>
): Promise<VisualBundleData> {
  const indexText = await (await fetcher(`${baseUrl}/visual/checkpoint_index.jsonl`)).text();
  const index = parseJsonl<{ graph_path: string }>(indexText);
  const graphs = await Promise.all(
    index.map(async (row) => validateGraphSnapshot(await (await fetcher(`${baseUrl}/visual/${row.graph_path}`)).json()))
  );
  const metrics = parseJsonl<MetricRow>(await (await fetcher(`${baseUrl}/visual/metrics.jsonl`)).text());
  const events = parseJsonl<MutationEvent>(await (await fetcher(`${baseUrl}/visual/mutation_events.jsonl`)).text());
  const routes = parseJsonl<RouteTrace>(await (await fetcher(`${baseUrl}/visual/route_traces.jsonl`)).text());
  const pockets = parseJsonl<PocketSummary>(await (await fetcher(`${baseUrl}/visual/pocket_summaries.jsonl`)).text());
  return { graphs, metrics, events, routes, pockets };
}

<script lang="ts">
  import GraphCanvas from '$lib/components/GraphCanvas.svelte';
  import BundleSelector from '$lib/components/BundleSelector.svelte';
  import NodeInspector from '$lib/components/NodeInspector.svelte';
  import PocketInspector from '$lib/components/PocketInspector.svelte';
  import { bundleById, renderMetadataFor, visualSampleBundles } from '$lib/sample-bundle';
  import type { RenderMetadata } from '$lib/schema';

  let selectedBundleId = visualSampleBundles[0].id;
  let selectedNodeId: string | null = null;
  let selectedEdgeId: string | null = null;
  let roleFilter = 'all';
  let renderMetadata: RenderMetadata | null = null;

  $: bundle = bundleById(selectedBundleId);
  $: graph = bundle.graphs[bundle.graphs.length - 1];
  $: fallbackRenderMetadata = renderMetadataFor(bundle, graph, renderMetadata?.render_duration_ms ?? 0);
  $: selectedNode = graph.nodes.find((node) => node.id === selectedNodeId);
  $: incidentEdges = selectedNode
    ? graph.edges.filter((edge) => edge.source === selectedNode.id || edge.target === selectedNode.id)
    : [];
  $: selectedPocket = selectedNode?.pocket_id
    ? graph.pockets.find((pocket) => pocket.id === selectedNode?.pocket_id)
    : undefined;
</script>

<section class="toolbar">
  <div>
    <p class="eyebrow">{bundle.id}</p>
    <h2>Topology</h2>
  </div>
  <div class="controls">
    <BundleSelector bundles={visualSampleBundles} selectedId={selectedBundleId} onSelect={(id) => (selectedBundleId = id)} />
    <label>
      Role filter
      <select bind:value={roleFilter}>
        <option value="all">all</option>
        <option value="highway">highway</option>
        <option value="pocket">pocket</option>
        <option value="candidate">candidate</option>
        <option value="source">source</option>
        <option value="target">target</option>
      </select>
    </label>
    <span>Selected edge: {selectedEdgeId ?? 'none'}</span>
  </div>
</section>

<section class="grid">
  <GraphCanvas
    {graph}
    {selectedNodeId}
    {roleFilter}
    checkpointCount={bundle.graphs.length}
    tickCount={bundle.ticks.length}
    eventCount={bundle.events.length}
    onSelectNode={(id) => (selectedNodeId = id)}
    onSelectEdge={(id) => (selectedEdgeId = id)}
    onRenderMetadata={(metadata) => (renderMetadata = metadata)}
  />
  <aside>
    <section class="metric-card">
      <h2>Render</h2>
      <dl>
        <div><dt>Render ms</dt><dd>{fallbackRenderMetadata.render_duration_ms.toFixed(1)}</dd></div>
        <div><dt>Nodes</dt><dd>{fallbackRenderMetadata.graph_node_count}</dd></div>
        <div><dt>Edges</dt><dd>{fallbackRenderMetadata.graph_edge_count}</dd></div>
        <div><dt>Checkpoints</dt><dd>{fallbackRenderMetadata.checkpoint_count}</dd></div>
        <div><dt>Ticks</dt><dd>{fallbackRenderMetadata.tick_count}</dd></div>
        <div><dt>Events</dt><dd>{fallbackRenderMetadata.event_count}</dd></div>
      </dl>
    </section>
    <NodeInspector node={selectedNode} edges={incidentEdges} />
    <PocketInspector pocket={selectedPocket} />
  </aside>
</section>

<style>
  .toolbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 18px;
    margin-bottom: 18px;
    flex-wrap: wrap;
    padding: 16px 18px;
    border: 1px solid #31516b;
    background: rgba(15, 35, 52, 0.76);
  }
  .controls {
    display: flex;
    align-items: center;
    gap: 18px;
    flex-wrap: wrap;
  }
  .eyebrow {
    margin: 0 0 4px;
    color: #67e8f9;
    font-size: 12px;
  }
  h2 {
    margin: 0;
    font-size: 28px;
    color: #f8fbff;
  }
  label {
    display: flex;
    align-items: center;
    gap: 8px;
    color: #cbd5e1;
  }
  select {
    background: #10263b;
    color: #f8fbff;
    border: 1px solid #4f7694;
    padding: 8px 10px;
  }
  .grid {
    display: grid;
    grid-template-columns: minmax(0, 1fr) 340px;
    gap: 18px;
  }
  aside {
    display: grid;
    gap: 14px;
    align-content: start;
  }
  .metric-card {
    border: 1px solid #31516b;
    padding: 16px;
    background: rgba(15, 35, 52, 0.9);
    color: #f2f8ff;
  }
  .metric-card h2 {
    font-size: 16px;
    margin: 0 0 12px;
  }
  .metric-card div {
    display: flex;
    justify-content: space-between;
    gap: 12px;
    margin-bottom: 8px;
  }
  .metric-card dt {
    color: #9ddcf0;
  }
  .metric-card dd {
    margin: 0;
  }
  @media (max-width: 980px) {
    .grid {
      grid-template-columns: 1fr;
    }
  }
</style>

<script lang="ts">
  import GraphCanvas from '$lib/components/GraphCanvas.svelte';
  import NodeInspector from '$lib/components/NodeInspector.svelte';
  import PocketInspector from '$lib/components/PocketInspector.svelte';
  import { activeSampleBundle } from '$lib/sample-bundle';

  let graph = activeSampleBundle.graphs[activeSampleBundle.graphs.length - 1];
  let selectedNodeId: string | null = null;
  let selectedEdgeId: string | null = null;
  let roleFilter = 'all';

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
    <p class="eyebrow">049 real-run ingest</p>
    <h2>Topology</h2>
  </div>
  <div class="controls">
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
    onSelectNode={(id) => (selectedNodeId = id)}
    onSelectEdge={(id) => (selectedEdgeId = id)}
  />
  <aside>
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
  @media (max-width: 980px) {
    .grid {
      grid-template-columns: 1fr;
    }
  }
</style>

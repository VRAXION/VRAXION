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
  <h2>Topology</h2>
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
    gap: 18px;
    margin-bottom: 18px;
    flex-wrap: wrap;
  }
  h2 {
    margin: 0;
    font-size: 22px;
  }
  label {
    display: flex;
    align-items: center;
    gap: 8px;
    color: #cbd5e1;
  }
  select {
    background: #0c1724;
    color: #e5f2ff;
    border: 1px solid #26384c;
    padding: 7px 9px;
  }
  .grid {
    display: grid;
    grid-template-columns: minmax(0, 1fr) 320px;
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

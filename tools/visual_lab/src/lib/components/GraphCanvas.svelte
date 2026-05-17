<script lang="ts">
  import { onDestroy, onMount } from 'svelte';
  import type { GraphSnapshot } from '../schema';
  import { roleColor } from '../schema';

  export let graph: GraphSnapshot;
  export let selectedNodeId: string | null = null;
  export let roleFilter = 'all';
  export let onSelectNode: (id: string) => void = () => {};
  export let onSelectEdge: (id: string) => void = () => {};

  let container: HTMLDivElement;
  let renderer: {
    kill(): void;
    refresh(): void;
    on(event: 'clickNode' | 'clickEdge', callback: (payload: { node?: string; edge?: string }) => void): void;
  } | null = null;

  async function render() {
    if (!container) return;
    renderer?.kill();
    const [{ default: Graph }, { default: Sigma }] = await Promise.all([import('graphology'), import('sigma')]);
    const sigmaGraph = new Graph({ type: 'directed', multi: true });
    for (const node of graph.nodes) {
      if (roleFilter !== 'all' && node.role !== roleFilter) continue;
      sigmaGraph.addNode(node.id, {
        label: node.label,
        x: node.x,
        y: node.y,
        size: 7 + node.activity * 7,
        color: node.id === selectedNodeId ? '#ffffff' : roleColor(node.role, node.is_active)
      });
    }
    for (const edge of graph.edges) {
      if (!sigmaGraph.hasNode(edge.source) || !sigmaGraph.hasNode(edge.target)) continue;
      sigmaGraph.addDirectedEdgeWithKey(edge.id, edge.source, edge.target, {
        label: edge.id,
        size: edge.role === 'highway' ? 3 : 1.6,
        color: roleColor(edge.role, edge.active_flow > 0.5),
        type: 'arrow'
      });
    }
    renderer = new Sigma(sigmaGraph, container, {
      renderLabels: true,
      renderEdgeLabels: false,
      defaultEdgeColor: '#64748b',
      defaultNodeColor: '#94a3b8'
    });
    renderer.on('clickNode', ({ node }) => {
      if (node) onSelectNode(node);
    });
    renderer.on('clickEdge', ({ edge }) => {
      if (edge) onSelectEdge(edge);
    });
  }

  $: if (container && graph) {
    void render();
  }

  onMount(() => {
    void render();
  });

  onDestroy(() => {
    renderer?.kill();
  });
</script>

<div class="graph" bind:this={container} aria-label="Topology graph"></div>

<style>
  .graph {
    min-height: 620px;
    height: 100%;
    width: 100%;
    background: #071018;
    border: 1px solid #223244;
  }
</style>

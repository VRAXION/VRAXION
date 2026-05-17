<script lang="ts">
  import { onDestroy, onMount } from 'svelte';
  import type { GraphSnapshot, RenderMetadata } from '../schema';
  import { buildRenderMetadata, roleColor, roleLabel } from '../schema';

  export let graph: GraphSnapshot;
  export let selectedNodeId: string | null = null;
  export let roleFilter = 'all';
  export let checkpointCount = 1;
  export let tickCount = 0;
  export let eventCount = 0;
  export let onSelectNode: (id: string) => void = () => {};
  export let onSelectEdge: (id: string) => void = () => {};
  export let onRenderMetadata: (metadata: RenderMetadata) => void = () => {};

  const legendRoles = ['highway', 'pocket', 'source', 'target', 'candidate', 'pruned', 'bridge'] as const;

  let container: HTMLDivElement;
  let renderer: {
    kill(): void;
    refresh(): void;
    on(event: 'clickNode' | 'clickEdge', callback: (payload: { node?: string; edge?: string }) => void): void;
  } | null = null;
  let renderGeneration = 0;
  let lastRenderMetadata: RenderMetadata | null = null;

  async function render() {
    if (!container) return;
    const generation = ++renderGeneration;
    const started = performance.now();
    renderer?.kill();
    renderer = null;
    container.replaceChildren();
    const [{ default: Graph }, { default: Sigma }] = await Promise.all([import('graphology'), import('sigma')]);
    if (generation !== renderGeneration || !container) return;
    const sigmaGraph = new Graph({ type: 'directed', multi: true });
    for (const node of graph.nodes) {
      if (roleFilter !== 'all' && node.role !== roleFilter) continue;
      sigmaGraph.addNode(node.id, {
        label: node.label,
        x: node.x,
        y: node.y,
        size: node.id === selectedNodeId ? 19 : 11 + node.activity * 9,
        color: node.id === selectedNodeId ? '#fef08a' : roleColor(node.role, node.is_active),
        highlighted: node.id === selectedNodeId,
        forceLabel: true
      });
    }
    for (const edge of graph.edges) {
      if (!sigmaGraph.hasNode(edge.source) || !sigmaGraph.hasNode(edge.target)) continue;
      sigmaGraph.addDirectedEdgeWithKey(edge.id, edge.source, edge.target, {
        label: edge.id,
        size: edge.role === 'highway' ? 4.5 : edge.is_pruned ? 2.4 : 2.8,
        color: roleColor(edge.role, edge.active_flow > 0.5),
        type: 'arrow'
      });
    }
    renderer = new Sigma(sigmaGraph, container, {
      renderLabels: true,
      renderEdgeLabels: false,
      labelColor: { color: '#e8f7ff' },
      labelSize: 14,
      labelWeight: '700',
      labelRenderedSizeThreshold: 0,
      defaultEdgeColor: '#7f95aa',
      defaultNodeColor: '#7f95aa',
      edgeReducer: (_edge, data) => ({
        ...data,
        color: data.color ?? '#7f95aa',
        size: data.size ?? 2.4
      }),
      nodeReducer: (_node, data) => ({
        ...data,
        label: data.label,
        zIndex: data.highlighted ? 2 : 1
      })
    });
    renderer.on('clickNode', ({ node }) => {
      if (node) onSelectNode(node);
    });
    renderer.on('clickEdge', ({ edge }) => {
      if (edge) onSelectEdge(edge);
    });
    lastRenderMetadata = buildRenderMetadata(
      graph,
      checkpointCount,
      tickCount,
      eventCount,
      performance.now() - started
    );
    onRenderMetadata(lastRenderMetadata);
  }

  $: if (container && graph) {
    void render();
  }

  onMount(() => {
    void render();
  });

  onDestroy(() => {
    renderGeneration += 1;
    renderer?.kill();
  });
</script>

<div class="frame">
  <div class="status">
    <div>
      <strong>checkpoint {graph.checkpoint}</strong>
      {#if graph.tick !== undefined}<span>tick {graph.tick}</span>{/if}
    </div>
    <div class="counts">
      <span>{graph.nodes.length} nodes</span>
      <span>{graph.edges.length} edges</span>
      <span>{graph.pockets.length} pockets</span>
      {#if lastRenderMetadata}<span>{lastRenderMetadata.render_duration_ms.toFixed(1)} ms render</span>{/if}
    </div>
  </div>
  <div class="graph" bind:this={container} aria-label="Topology graph"></div>
  <div class="legend" aria-label="Graph legend">
    {#each legendRoles as role}
      <span><i style={`background:${roleColor(role)}`}></i>{roleLabel(role)}</span>
    {/each}
  </div>
</div>

<style>
  .frame {
    position: relative;
    overflow: hidden;
    min-height: 680px;
    border: 1px solid #31516b;
    background:
      radial-gradient(circle at 28% 32%, rgba(34, 211, 238, 0.16), transparent 28%),
      linear-gradient(180deg, #102235 0%, #0b1724 54%, #07111c 100%);
    box-shadow: 0 18px 55px rgba(2, 6, 23, 0.36);
  }
  .graph {
    min-height: 680px;
    height: 100%;
    width: 100%;
    background:
      linear-gradient(rgba(148, 163, 184, 0.055) 1px, transparent 1px),
      linear-gradient(90deg, rgba(148, 163, 184, 0.055) 1px, transparent 1px);
    background-size: 42px 42px;
  }
  .status {
    position: absolute;
    z-index: 2;
    top: 16px;
    left: 16px;
    right: 16px;
    display: flex;
    justify-content: space-between;
    gap: 16px;
    pointer-events: none;
    color: #dff7ff;
    text-shadow: 0 2px 6px rgba(0, 0, 0, 0.45);
  }
  .status strong {
    display: block;
    font-size: 14px;
  }
  .status span {
    color: #b6d4e8;
    font-size: 12px;
  }
  .counts {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
    justify-content: flex-end;
  }
  .counts span {
    padding: 5px 8px;
    border: 1px solid rgba(125, 211, 252, 0.35);
    background: rgba(8, 19, 32, 0.72);
  }
  .legend {
    position: absolute;
    left: 16px;
    bottom: 16px;
    z-index: 2;
    display: flex;
    gap: 8px 12px;
    flex-wrap: wrap;
    max-width: calc(100% - 32px);
    padding: 10px 12px;
    background: rgba(8, 19, 32, 0.86);
    border: 1px solid rgba(125, 211, 252, 0.28);
    color: #dbeafe;
    font-size: 12px;
  }
  .legend span {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    white-space: nowrap;
  }
  .legend i {
    width: 10px;
    height: 10px;
    border-radius: 999px;
    box-shadow: 0 0 10px currentColor;
  }
  :global(.sigma-label) {
    color: #e8f7ff;
  }
</style>

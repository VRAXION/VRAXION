<script lang="ts">
  import type { GraphEdge, GraphNode } from '../schema';
  export let node: GraphNode | undefined;
  export let edges: GraphEdge[] = [];
</script>

<section class="panel">
  <h2>Node</h2>
  {#if node}
    <p>{node.id} / {node.role}</p>
    <dl>
      <div><dt>Activity</dt><dd>{node.activity.toFixed(2)}</dd></div>
      <div><dt>Pocket</dt><dd>{node.pocket_id ?? 'none'}</dd></div>
      <div><dt>Route order</dt><dd>{node.route_order ?? 'none'}</dd></div>
      <div><dt>Incident edges</dt><dd>{edges.length}</dd></div>
    </dl>
  {:else}
    <p>Select a node to inspect role, activity, route order, and local edges.</p>
  {/if}
</section>

<style>
  .panel {
    border: 1px solid #31516b;
    padding: 16px;
    background: rgba(15, 35, 52, 0.9);
    color: #f2f8ff;
  }
  h2 {
    font-size: 16px;
    margin: 0 0 12px;
    color: #f8fbff;
  }
  p {
    color: #d9e8f5;
  }
  div {
    display: flex;
    justify-content: space-between;
    gap: 12px;
    margin-bottom: 8px;
  }
  dt {
    color: #9ddcf0;
  }
  dd {
    margin: 0;
  }
</style>

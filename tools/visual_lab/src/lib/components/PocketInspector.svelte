<script lang="ts">
  import type { PocketSummary } from '../schema';
  export let pocket: PocketSummary | undefined;
</script>

<section class="panel">
  <h2>Pocket</h2>
  {#if pocket}
    <p>{pocket.id} / {pocket.kind}</p>
    <dl>
      <div><dt>Nodes</dt><dd>{pocket.node_ids.length}</dd></div>
      <div><dt>Bridge nodes</dt><dd>{pocket.bridge_nodes.join(', ')}</dd></div>
      <div><dt>Mutations</dt><dd>{pocket.mutation_count}</dd></div>
      <div><dt>Prune ratio</dt><dd>{pocket.prune_ratio.toFixed(2)}</dd></div>
    </dl>
  {:else}
    <p>Select a pocket node to inspect its cluster.</p>
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
    display: grid;
    gap: 4px;
    margin-bottom: 10px;
  }
  dt {
    color: #9ddcf0;
  }
  dd {
    margin: 0;
  }
</style>

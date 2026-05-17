<script lang="ts">
  import type { DiffSummary } from '../schema';
  export let diff: DiffSummary;
</script>

<section class="panel">
  <h2>Diff</h2>
  <dl>
    <div><dt>Added edges</dt><dd>{diff.added_edges}</dd></div>
    <div><dt>Removed edges</dt><dd>{diff.removed_edges}</dd></div>
    <div><dt>Pruned edges</dt><dd>{diff.pruned_edges}</dd></div>
    <div><dt>Retained edges</dt><dd>{diff.retained_edges}</dd></div>
  </dl>
</section>

<style>
  .panel {
    border: 1px solid #26384c;
    padding: 16px;
    background: #0c1724;
    color: #e5f2ff;
  }
  h2 {
    font-size: 16px;
    margin: 0 0 12px;
  }
  dl {
    display: grid;
    gap: 8px;
    margin: 0;
  }
  div {
    display: flex;
    justify-content: space-between;
  }
  dt {
    color: #9fb4cc;
  }
  dd {
    margin: 0;
    font-weight: 700;
  }
</style>

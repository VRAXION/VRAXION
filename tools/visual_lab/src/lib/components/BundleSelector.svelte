<script lang="ts">
  import type { VisualSampleBundle } from '../sample-bundle';

  export let bundles: VisualSampleBundle[] = [];
  export let selectedId = '';
  export let onSelect: (id: string) => void = () => {};

  function update(event: Event) {
    selectedId = (event.target as HTMLSelectElement).value;
    onSelect(selectedId);
  }
</script>

<label class="selector">
  <span>Bundle</span>
  <select value={selectedId} on:change={update}>
    {#each bundles as bundle}
      <option value={bundle.id}>{bundle.label}</option>
    {/each}
  </select>
</label>

<style>
  .selector {
    display: flex;
    align-items: center;
    gap: 8px;
    color: #d9e8f5;
  }
  select {
    background: #10263b;
    color: #f8fbff;
    border: 1px solid #4f7694;
    padding: 8px 10px;
    min-width: 230px;
  }
</style>

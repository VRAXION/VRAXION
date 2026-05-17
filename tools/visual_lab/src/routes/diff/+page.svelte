<script lang="ts">
  import DiffPanel from '$lib/components/DiffPanel.svelte';
  import GraphCanvas from '$lib/components/GraphCanvas.svelte';
  import { diffGraphs } from '$lib/schema';
  import { sampleGraphs } from '$lib/sample-bundle';

  const before = sampleGraphs[0];
  const after = sampleGraphs[1];
  const diff = diffGraphs(before, after);
</script>

<section class="header">
  <h2>Diff</h2>
  <p>Checkpoint 000 to checkpoint 010 highlights added, retained, and pruned edges.</p>
</section>

<section class="grid">
  <GraphCanvas graph={after} />
  <DiffPanel {diff} />
</section>

<style>
  .header {
    margin-bottom: 18px;
  }
  h2 {
    margin: 0 0 8px;
    font-size: 22px;
  }
  p {
    color: #cbd5e1;
  }
  .grid {
    display: grid;
    grid-template-columns: minmax(0, 1fr) 280px;
    gap: 18px;
  }
  @media (max-width: 880px) {
    .grid {
      grid-template-columns: 1fr;
    }
  }
</style>

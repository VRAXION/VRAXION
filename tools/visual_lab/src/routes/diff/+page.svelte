<script lang="ts">
  import DiffPanel from '$lib/components/DiffPanel.svelte';
  import GraphCanvas from '$lib/components/GraphCanvas.svelte';
  import { diffGraphs } from '$lib/schema';
  import { activeSampleBundle } from '$lib/sample-bundle';

  const before = activeSampleBundle.graphs[0];
  const after = activeSampleBundle.graphs[activeSampleBundle.graphs.length - 1];
  const diff = diffGraphs(before, after);
</script>

<section class="header">
  <h2>Diff</h2>
  <p>First to final checkpoint highlights added, retained, and pruned edges.</p>
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

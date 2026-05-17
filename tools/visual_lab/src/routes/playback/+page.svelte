<script lang="ts">
  import GraphCanvas from '$lib/components/GraphCanvas.svelte';
  import TimelineScrubber from '$lib/components/TimelineScrubber.svelte';
  import { sampleGraphs } from '$lib/sample-bundle';

  const checkpoints = sampleGraphs.map((graph) => graph.checkpoint);
  let checkpoint = checkpoints[0];
  $: graph = sampleGraphs.find((item) => item.checkpoint === checkpoint) ?? sampleGraphs[0];
</script>

<section class="toolbar">
  <h2>Playback</h2>
  <TimelineScrubber {checkpoints} value={checkpoint} onChange={(next) => (checkpoint = next)} />
  <span>Tick playback available when tick artifacts exist.</span>
</section>

<GraphCanvas {graph} />

<style>
  .toolbar {
    display: grid;
    grid-template-columns: 160px minmax(220px, 420px) minmax(0, 1fr);
    align-items: end;
    gap: 18px;
    margin-bottom: 18px;
  }
  h2 {
    margin: 0;
    font-size: 22px;
  }
  span {
    color: #cbd5e1;
  }
  @media (max-width: 860px) {
    .toolbar {
      grid-template-columns: 1fr;
    }
  }
</style>

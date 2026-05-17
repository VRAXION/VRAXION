<script lang="ts">
  import BundleSelector from '$lib/components/BundleSelector.svelte';
  import EventTimeline from '$lib/components/EventTimeline.svelte';
  import GraphCanvas from '$lib/components/GraphCanvas.svelte';
  import TimelineScrubber from '$lib/components/TimelineScrubber.svelte';
  import {
    bundleById,
    checkpointsFor,
    eventsForCheckpoint,
    graphForCheckpoint,
    renderMetadataFor,
    ticksForCheckpoint,
    visualSampleBundles
  } from '$lib/sample-bundle';
  import type { RenderMetadata } from '$lib/schema';

  let selectedBundleId = visualSampleBundles[0].id;
  let checkpoint = 0;
  let tick: number | 'checkpoint' = 'checkpoint';
  let renderMetadata: RenderMetadata | null = null;

  $: bundle = bundleById(selectedBundleId);
  $: checkpoints = checkpointsFor(bundle);
  $: if (!checkpoints.includes(checkpoint)) checkpoint = checkpoints[0];
  $: ticks = ticksForCheckpoint(bundle, checkpoint);
  $: if (tick !== 'checkpoint' && !ticks.some((item) => item.tick === tick)) tick = 'checkpoint';
  $: tickValues = ticks.map((item) => item.tick ?? 0);
  $: activeTickValue = tick === 'checkpoint' ? tickValues[0] ?? 0 : tick;
  $: graph =
    tick === 'checkpoint'
      ? graphForCheckpoint(bundle, checkpoint)
      : ticks.find((item) => item.tick === tick) ?? graphForCheckpoint(bundle, checkpoint);
  $: events = eventsForCheckpoint(bundle, checkpoint, tick === 'checkpoint' ? 'all' : tick);
  $: fallbackRenderMetadata = renderMetadataFor(bundle, graph, renderMetadata?.render_duration_ms ?? 0);
</script>

<section class="toolbar">
  <div>
    <p class="eyebrow">{bundle.id}</p>
    <h2>Playback</h2>
  </div>
  <BundleSelector bundles={visualSampleBundles} selectedId={selectedBundleId} onSelect={(id) => (selectedBundleId = id)} />
  <TimelineScrubber {checkpoints} value={checkpoint} onChange={(next) => (checkpoint = next)} />
  <div class="tick-control">
    {#if ticks.length > 0}
      <TimelineScrubber
        checkpoints={tickValues}
        value={activeTickValue}
        label="Tick"
        onChange={(next) => (tick = next)}
      />
      <button class:active={tick === 'checkpoint'} on:click={() => (tick = 'checkpoint')}>checkpoint graph</button>
    {:else}
      <span>No tick snapshots</span>
    {/if}
  </div>
</section>

<section class="grid">
  <GraphCanvas
    {graph}
    checkpointCount={bundle.graphs.length}
    tickCount={bundle.ticks.length}
    eventCount={bundle.events.length}
    onRenderMetadata={(metadata) => (renderMetadata = metadata)}
  />
  <aside>
    <section class="metadata">
      <h2>Render</h2>
      <dl>
        <div><dt>Render ms</dt><dd>{fallbackRenderMetadata.render_duration_ms.toFixed(1)}</dd></div>
        <div><dt>Nodes</dt><dd>{fallbackRenderMetadata.graph_node_count}</dd></div>
        <div><dt>Edges</dt><dd>{fallbackRenderMetadata.graph_edge_count}</dd></div>
        <div><dt>Checkpoints</dt><dd>{fallbackRenderMetadata.checkpoint_count}</dd></div>
        <div><dt>Ticks</dt><dd>{fallbackRenderMetadata.tick_count}</dd></div>
        <div><dt>Events</dt><dd>{fallbackRenderMetadata.event_count}</dd></div>
      </dl>
    </section>
    <EventTimeline {events} />
  </aside>
</section>

<style>
  .toolbar {
    display: grid;
    grid-template-columns: minmax(160px, 1fr) minmax(220px, 300px) minmax(220px, 420px) 170px;
    align-items: end;
    gap: 18px;
    margin-bottom: 18px;
    padding: 16px 18px;
    border: 1px solid #31516b;
    background: rgba(15, 35, 52, 0.76);
  }
  .eyebrow {
    margin: 0 0 4px;
    color: #67e8f9;
    font-size: 12px;
  }
  h2 {
    margin: 0;
    font-size: 22px;
  }
  .tick-control {
    display: grid;
    gap: 6px;
    color: #cbd5e1;
  }
  button {
    background: #10263b;
    color: #f8fbff;
    border: 1px solid #4f7694;
    padding: 8px 10px;
  }
  button.active {
    border-color: #67e8f9;
    color: #67e8f9;
  }
  .tick-control span {
    padding: 8px 10px;
    color: #9fb4cc;
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
  .metadata {
    border: 1px solid #31516b;
    padding: 16px;
    background: rgba(15, 35, 52, 0.9);
    color: #f2f8ff;
  }
  .metadata h2 {
    font-size: 16px;
    margin: 0 0 12px;
  }
  .metadata div {
    display: flex;
    justify-content: space-between;
    gap: 12px;
    margin-bottom: 8px;
  }
  .metadata dt {
    color: #9ddcf0;
  }
  .metadata dd {
    margin: 0;
  }
  @media (max-width: 860px) {
    .toolbar {
      grid-template-columns: 1fr;
    }
    .grid {
      grid-template-columns: 1fr;
    }
  }
</style>

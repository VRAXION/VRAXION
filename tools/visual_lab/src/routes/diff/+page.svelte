<script lang="ts">
  import BundleSelector from '$lib/components/BundleSelector.svelte';
  import DiffPanel from '$lib/components/DiffPanel.svelte';
  import GraphCanvas from '$lib/components/GraphCanvas.svelte';
  import TimelineScrubber from '$lib/components/TimelineScrubber.svelte';
  import {
    bundleById,
    checkpointsFor,
    diffForCheckpoint,
    graphForCheckpoint,
    renderMetadataFor,
    type DiffMode,
    visualSampleBundles
  } from '$lib/sample-bundle';
  import type { RenderMetadata } from '$lib/schema';

  let selectedBundleId = visualSampleBundles[0].id;
  let checkpoint = 110;
  let diffMode: DiffMode = 'first';
  let renderMetadata: RenderMetadata | null = null;

  $: bundle = bundleById(selectedBundleId);
  $: checkpoints = checkpointsFor(bundle);
  $: if (!checkpoints.includes(checkpoint)) checkpoint = checkpoints[checkpoints.length - 1];
  $: after = graphForCheckpoint(bundle, checkpoint);
  $: diff = diffForCheckpoint(bundle, checkpoint, diffMode);
  $: fallbackRenderMetadata = renderMetadataFor(bundle, after, renderMetadata?.render_duration_ms ?? 0);
</script>

<section class="header">
  <div>
    <p class="eyebrow">{bundle.id}</p>
    <h2>Diff</h2>
    <p>Compare first-to-selected or previous-to-selected checkpoint states.</p>
  </div>
  <div class="controls">
    <BundleSelector bundles={visualSampleBundles} selectedId={selectedBundleId} onSelect={(id) => (selectedBundleId = id)} />
    <TimelineScrubber {checkpoints} value={checkpoint} onChange={(next) => (checkpoint = next)} />
    <label>
      Mode
      <select bind:value={diffMode}>
        <option value="first">first -> selected</option>
        <option value="previous">previous -> selected</option>
      </select>
    </label>
  </div>
</section>

<section class="grid">
  <GraphCanvas
    graph={after}
    checkpointCount={bundle.graphs.length}
    tickCount={bundle.ticks.length}
    eventCount={bundle.events.length}
    onRenderMetadata={(metadata) => (renderMetadata = metadata)}
  />
  <aside>
    <DiffPanel {diff} />
    <section class="metadata">
      <h2>Render</h2>
      <dl>
        <div><dt>Render ms</dt><dd>{fallbackRenderMetadata.render_duration_ms.toFixed(1)}</dd></div>
        <div><dt>Nodes</dt><dd>{fallbackRenderMetadata.graph_node_count}</dd></div>
        <div><dt>Edges</dt><dd>{fallbackRenderMetadata.graph_edge_count}</dd></div>
        <div><dt>Mode</dt><dd>{diffMode}</dd></div>
      </dl>
    </section>
  </aside>
</section>

<style>
  .header {
    display: flex;
    align-items: end;
    justify-content: space-between;
    gap: 18px;
    flex-wrap: wrap;
    margin-bottom: 18px;
    padding: 16px 18px;
    border: 1px solid #31516b;
    background: rgba(15, 35, 52, 0.76);
  }
  .controls {
    display: flex;
    align-items: end;
    gap: 16px;
    flex-wrap: wrap;
  }
  .eyebrow {
    margin: 0 0 4px;
    color: #67e8f9;
    font-size: 12px;
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
    grid-template-columns: minmax(0, 1fr) 320px;
    gap: 18px;
  }
  aside {
    display: grid;
    gap: 14px;
    align-content: start;
  }
  label {
    display: grid;
    gap: 6px;
    color: #cbd5e1;
  }
  select {
    background: #10263b;
    color: #f8fbff;
    border: 1px solid #4f7694;
    padding: 8px 10px;
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
  @media (max-width: 880px) {
    .grid {
      grid-template-columns: 1fr;
    }
  }
</style>

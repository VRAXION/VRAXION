<script lang="ts">
  import MetricsPanel from '$lib/components/MetricsPanel.svelte';
  import BundleSelector from '$lib/components/BundleSelector.svelte';
  import { bundleById, visualSampleBundles } from '$lib/sample-bundle';

  let selectedBundleId = visualSampleBundles[0].id;
  $: bundle = bundleById(selectedBundleId);
  $: latest = bundle.graphs[bundle.graphs.length - 1];
</script>

<section class="hero">
  <div>
    <p class="eyebrow">STABLE_LOOP_PHASE_LOCK_054</p>
    <h2>Visual analysis lab</h2>
    <p>
      Select a committed visual bundle and inspect topology, playback, diff, event timeline,
      and metrics through the stable visual_snapshot_v1 schema.
    </p>
    <BundleSelector bundles={visualSampleBundles} selectedId={selectedBundleId} onSelect={(id) => (selectedBundleId = id)} />
  </div>
  <dl>
    <div><dt>Bundles</dt><dd>{visualSampleBundles.length}</dd></div>
    <div><dt>Checkpoints</dt><dd>{bundle.graphs.length}</dd></div>
    <div><dt>Ticks</dt><dd>{bundle.ticks.length}</dd></div>
    <div><dt>Nodes</dt><dd>{latest.nodes.length}</dd></div>
    <div><dt>Edges</dt><dd>{latest.edges.length}</dd></div>
    <div><dt>Events</dt><dd>{bundle.events.length}</dd></div>
  </dl>
</section>

<MetricsPanel metrics={bundle.metrics} />

<style>
  .hero {
    display: grid;
    grid-template-columns: minmax(0, 1fr) 320px;
    gap: 24px;
    align-items: stretch;
    margin-bottom: 24px;
  }
  h2 {
    font-size: 28px;
    margin: 0 0 12px;
  }
  p {
    color: #cbd5e1;
    max-width: 760px;
  }
  .eyebrow {
    color: #67e8f9;
    font-size: 12px;
    letter-spacing: 0;
    margin: 0 0 10px;
  }
  dl {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 10px;
    margin: 0;
  }
  dl div {
    border: 1px solid #26384c;
    padding: 14px;
    background: #0c1724;
  }
  dt {
    color: #9fb4cc;
  }
  dd {
    font-size: 24px;
    font-weight: 700;
    margin: 6px 0 0;
  }
  @media (max-width: 880px) {
    .hero {
      grid-template-columns: 1fr;
    }
  }
</style>

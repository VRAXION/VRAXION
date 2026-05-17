<script lang="ts">
  import BundleSelector from '$lib/components/BundleSelector.svelte';
  import MetricsPanel from '$lib/components/MetricsPanel.svelte';
  import { bundleById, visualSampleBundles } from '$lib/sample-bundle';

  let selectedBundleId = visualSampleBundles[0].id;
  $: bundle = bundleById(selectedBundleId);
</script>

<section class="header">
  <div>
    <p class="eyebrow">{bundle.id}</p>
    <h2>Metrics</h2>
    <p>Route order, missing successor count, entropy, and collapse state over checkpoint time.</p>
  </div>
  <BundleSelector bundles={visualSampleBundles} selectedId={selectedBundleId} onSelect={(id) => (selectedBundleId = id)} />
</section>

<MetricsPanel metrics={bundle.metrics} />

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
</style>

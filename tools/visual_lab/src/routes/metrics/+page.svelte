<script lang="ts">
  import MetricsPanel from '$lib/components/MetricsPanel.svelte';
  import { sampleMetrics } from '$lib/sample-bundle';
</script>

<section class="header">
  <h2>Metrics</h2>
  <p>Route order, missing successor count, entropy, and collapse state over checkpoint time.</p>
</section>

<MetricsPanel metrics={sampleMetrics} />

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
</style>

<script lang="ts">
  import MetricsPanel from '$lib/components/MetricsPanel.svelte';
  import { sampleEvents, sampleGraphs, sampleMetrics } from '$lib/sample-bundle';

  const latest = sampleGraphs[1];
</script>

<section class="hero">
  <div>
    <p class="eyebrow">STABLE_LOOP_PHASE_LOCK_052</p>
    <h2>Schema-first visual analysis package</h2>
    <p>
      Deterministic sample bundle loaded locally: highway backbone, side pockets,
      route trace, mutation event, prune event, checkpoint evolution, and diffable graph state.
    </p>
  </div>
  <dl>
    <div><dt>Checkpoints</dt><dd>{sampleGraphs.length}</dd></div>
    <div><dt>Nodes</dt><dd>{latest.nodes.length}</dd></div>
    <div><dt>Edges</dt><dd>{latest.edges.length}</dd></div>
    <div><dt>Events</dt><dd>{sampleEvents.length}</dd></div>
  </dl>
</section>

<MetricsPanel metrics={sampleMetrics} />

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

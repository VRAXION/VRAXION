<script lang="ts">
  import { line, scaleLinear } from 'd3';
  import type { MetricRow } from '../schema';
  export let metrics: MetricRow[] = [];

  $: routeLine = buildLine(metrics.map((row) => row.route_order_accuracy));
  $: entropyLine = buildLine(metrics.map((row) => row.output_entropy));

  function buildLine(values: number[]): string {
    if (values.length === 0) return '';
    const x = scaleLinear().domain([0, Math.max(values.length - 1, 1)]).range([8, 220]);
    const y = scaleLinear().domain([0, Math.max(...values, 1)]).range([82, 8]);
    return (
      line<number>()
        .x((_, index) => x(index))
        .y((value) => y(value))(values) ?? ''
    );
  }
</script>

<section class="panel">
  <h2>Metrics</h2>
  <svg viewBox="0 0 230 90" role="img" aria-label="Route order sparkline">
    <path d={routeLine} class="route" />
    <path d={entropyLine} class="entropy" />
  </svg>
  <dl>
    {#each metrics as row}
      <div>
        <dt>checkpoint {row.checkpoint}</dt>
        <dd>
          heldout {row.heldout_score.toFixed(2)} / OOD {row.ood_score.toFixed(2)} / route
          {row.route_order_accuracy.toFixed(2)} / entropy {row.output_entropy.toFixed(2)}
        </dd>
      </div>
      {#if row.source_arm}
        <div class="subrow">
          <dt>{row.source_arm}</dt>
          <dd>
            family {row.family_min_accuracy?.toFixed(2) ?? 'n/a'} / unique
            {row.unique_output_count ?? 'n/a'}/{row.expected_output_class_count ?? 'n/a'}
          </dd>
        </div>
      {/if}
      <div class="subrow">
        <dt>collapse</dt>
        <dd>{row.collapse_detected ? 'true' : 'false'}</dd>
      </div>
    {/each}
  </dl>
</section>

<style>
  .panel {
    border: 1px solid #31516b;
    padding: 16px;
    background: rgba(15, 35, 52, 0.9);
    color: #f2f8ff;
  }
  h2 {
    font-size: 16px;
    margin: 0 0 12px;
  }
  svg {
    width: 100%;
    height: auto;
    background: #102235;
  }
  path {
    fill: none;
    stroke-width: 3;
  }
  .route {
    stroke: #22c55e;
  }
  .entropy {
    stroke: #38bdf8;
  }
  div {
    display: flex;
    justify-content: space-between;
    gap: 12px;
    margin-top: 8px;
  }
  .subrow {
    font-size: 12px;
    color: #cbd5e1;
  }
  dt {
    color: #9fb4cc;
  }
  dd {
    margin: 0;
  }
</style>

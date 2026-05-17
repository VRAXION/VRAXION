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
        <dd>route {row.route_order_accuracy.toFixed(2)} / entropy {row.output_entropy.toFixed(2)}</dd>
      </div>
    {/each}
  </dl>
</section>

<style>
  .panel {
    border: 1px solid #26384c;
    padding: 16px;
    background: #0c1724;
    color: #e5f2ff;
  }
  h2 {
    font-size: 16px;
    margin: 0 0 12px;
  }
  svg {
    width: 100%;
    height: auto;
    background: #08111d;
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
  dt {
    color: #9fb4cc;
  }
  dd {
    margin: 0;
  }
</style>

<script lang="ts">
  import type { MutationEvent } from '../schema';

  export let events: MutationEvent[] = [];
</script>

<section class="panel">
  <h2>Events</h2>
  {#if events.length === 0}
    <p>No events for this checkpoint/tick.</p>
  {:else}
    <ol>
      {#each events as event}
        <li>
          <strong>{event.kind}</strong>
          <span>{event.label}</span>
        </li>
      {/each}
    </ol>
  {/if}
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
  p,
  span {
    color: #d9e8f5;
  }
  ol {
    display: grid;
    gap: 10px;
    margin: 0;
    padding-left: 18px;
  }
  li {
    padding-left: 4px;
  }
  strong {
    display: block;
    color: #9ddcf0;
  }
</style>

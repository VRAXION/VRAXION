<script lang="ts">
  const nav = [
    ['/', 'Overview'],
    ['/topology', 'Topology'],
    ['/playback', 'Playback'],
    ['/diff', 'Diff'],
    ['/metrics', 'Metrics']
  ];
</script>

<div class="shell">
  <aside>
    <h1>Visual Lab</h1>
    <nav>
      {#each nav as [href, label]}
        <a {href}>{label}</a>
      {/each}
    </nav>
  </aside>
  <main>
    <slot />
  </main>
</div>

<style>
  :global(body) {
    margin: 0;
    background: #081018;
    color: #e5f2ff;
    font-family:
      Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  }
  :global(*) {
    box-sizing: border-box;
  }
  .shell {
    display: grid;
    grid-template-columns: 220px minmax(0, 1fr);
    min-height: 100vh;
  }
  aside {
    border-right: 1px solid #223244;
    background: #0d1822;
    padding: 20px;
  }
  h1 {
    font-size: 18px;
    margin: 0 0 24px;
  }
  nav {
    display: grid;
    gap: 8px;
  }
  a {
    color: #bae6fd;
    text-decoration: none;
    padding: 9px 10px;
    border: 1px solid transparent;
  }
  a:hover {
    border-color: #22d3ee;
    color: #ffffff;
  }
  main {
    min-width: 0;
    padding: 24px;
  }
  @media (max-width: 760px) {
    .shell {
      grid-template-columns: 1fr;
    }
    aside {
      border-right: 0;
      border-bottom: 1px solid #223244;
    }
    nav {
      grid-template-columns: repeat(5, minmax(0, 1fr));
    }
    a {
      text-align: center;
      padding: 8px 4px;
    }
  }
</style>

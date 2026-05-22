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
    background: #102033;
    color: #f2f8ff;
    font-family:
      Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  }
  :global(*) {
    box-sizing: border-box;
  }
  .shell {
    display: grid;
    grid-template-columns: 210px minmax(0, 1fr);
    min-height: 100vh;
    background:
      radial-gradient(circle at 85% 8%, rgba(34, 211, 238, 0.14), transparent 26%),
      linear-gradient(135deg, #12263a 0%, #091521 62%, #07111c 100%);
  }
  aside {
    border-right: 1px solid #31516b;
    background: rgba(13, 28, 43, 0.94);
    padding: 22px 18px;
  }
  h1 {
    font-size: 19px;
    margin: 0 0 24px;
    color: #e0f7ff;
  }
  nav {
    display: grid;
    gap: 8px;
  }
  a {
    color: #c8efff;
    text-decoration: none;
    padding: 11px 12px;
    border: 1px solid rgba(125, 211, 252, 0.16);
    background: rgba(15, 35, 52, 0.72);
  }
  a:hover {
    border-color: #38f5ff;
    background: rgba(34, 211, 238, 0.12);
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

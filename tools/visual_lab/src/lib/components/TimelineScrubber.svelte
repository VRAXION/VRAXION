<script lang="ts">
  export let checkpoints: number[] = [0, 10];
  export let value = 0;
  export let label = 'Checkpoint';
  export let onChange: (checkpoint: number) => void = () => {};

  function update(event: Event) {
    const index = Number((event.target as HTMLInputElement).value);
    value = checkpoints[index] ?? checkpoints[0];
    onChange(value);
  }
</script>

<label class="scrubber">
  <span>{label} {value}</span>
  <input
    type="range"
    min="0"
    max={Math.max(checkpoints.length - 1, 0)}
    step="1"
    value={Math.max(checkpoints.indexOf(value), 0)}
    on:input={update}
  />
</label>

<style>
  .scrubber {
    display: grid;
    gap: 8px;
    color: #dbeafe;
  }
</style>

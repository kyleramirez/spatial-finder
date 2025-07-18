<script lang="ts">
  import { getContext } from 'svelte';
  const currentPath = getContext<string | null>('currentPath');
</script>

<div class="current-path">
  <strong class="spacer">&#x1f4c1;</strong>
  {#if currentPath !== null}
    {#each currentPath.split('/') as part, index (index)}
      {#if index > 1}
        <strong class="spacer">&rang;</strong>
      {/if}
      <!-- // next turn this into a directory link -->
      <a href={`${currentPath.split('/').slice(0, index + 1).join('/')}`}>{part}</a>
    {/each}
  {:else}
    Choose a location to start browsing.
  {/if}
</div>

<style lang="scss">
  .current-path {
    display: flex;
    flex-grow: 1;
    width: 100%;
    padding: $gutter-width / 2;
    margin-top: $gutter-width;
    overflow: scroll;
    box-shadow: 0 -#{$gutter-width} 0 $gutter-color;

    &:empty {
      height: 1.5rem;
    }
  }

  .spacer {
    margin: 0 $gutter-width;
  }
</style>
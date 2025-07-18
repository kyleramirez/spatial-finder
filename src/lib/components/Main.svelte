<script lang="ts">

  let { children, ...props } = $props();
</script>

<main {...props} class="main">
  {@render children?.()}
</main>

<style lang="scss">
  .main {
    display: flex;
    flex-grow: 1;
    width: 100%;
    padding-top: $gutter-width;
    overflow: scroll;
    box-shadow: inset 0 $gutter-width 0 $gutter-color;
  }
</style>

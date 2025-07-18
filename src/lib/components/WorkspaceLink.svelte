<script lang="ts">
  import type { HTMLAnchorAttributes } from 'svelte/elements';
  import { link, useLocation } from 'svelte5-router';
  let { children, workspace, ...props }: HTMLAnchorAttributes & { workspace: 'BROWSE' | 'TIMELINE' | 'MAP' } = $props();
  const location = useLocation();
  const query = new URLSearchParams($location.search);
  query.set('workspace', workspace);
  const href = `${$location.pathname}?${query.toString()}`;
</script>

<a {...props} class={['link', props.class]} use:link href={href}>
  {@render children?.()}
</a>

<style lang="scss">
  .link {
    text-decoration: none;
  }
</style>

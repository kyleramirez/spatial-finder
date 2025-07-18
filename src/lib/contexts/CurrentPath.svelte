<script lang="ts">
  import { setContext } from 'svelte';
  import { useLocation } from 'svelte5-router';
  const { children } = $props();
  const location = useLocation();
  const search = new URLSearchParams($location.search);
  setContext('currentPath', search.get('path') ?? null);
</script>

{@render children?.()}

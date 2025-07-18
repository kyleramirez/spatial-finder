<script lang="ts">
  import { useLocation } from 'svelte5-router';
  let { children, workspace }: { children: any; workspace: 'BROWSE' | 'TIMELINE' | 'MAP' } = $props();
  const location = useLocation();
  
  let isActive = $derived(() => {
    const query = new URLSearchParams($location.search);
    const currentWorkspace = query.get('workspace') ?? 'BROWSE';
    return currentWorkspace === workspace;
  });
</script>

  {#if $location && isActive()}
    {@render children?.()}
  {/if}

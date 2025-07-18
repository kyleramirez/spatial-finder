<script lang="ts">
  import { Router } from 'svelte5-router';
  import WorkspaceLink from '$lib/components/WorkspaceLink.svelte';
  import Main from '$lib/components/Main.svelte';
  import Column from '$lib/components/Column.svelte';
  import WorkspaceRoute from '$lib/components/WorkspaceRoute.svelte';
  import CurrentPathContext from '$lib/contexts/CurrentPath.svelte';
  import CurrentPathHeader from '$lib/components/CurrentPathHeader.svelte';
</script>

<Router>
  <CurrentPathContext>    
    <header>
      <nav>
        <WorkspaceLink workspace="BROWSE">Browse</WorkspaceLink>
        <WorkspaceLink workspace="TIMELINE">Timeline</WorkspaceLink>
        <WorkspaceLink workspace="MAP">Map</WorkspaceLink>
      </nav>
      <CurrentPathHeader />
    </header>
    <Main>
      <Column left>
        <code>Favorites</code>
      </Column>
      <Column>
        <WorkspaceRoute workspace="BROWSE">
          <code>Browse</code>
        </WorkspaceRoute>
        <WorkspaceRoute workspace="TIMELINE">
          <code>Timeline</code>
        </WorkspaceRoute>
        <WorkspaceRoute workspace="MAP">
          <code>Map</code>
        </WorkspaceRoute>
      </Column>
      <Column right>
        <code>Metadata</code>
      </Column>
    </Main>
  </CurrentPathContext>
</Router>

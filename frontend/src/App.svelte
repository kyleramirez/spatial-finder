<script lang="ts">
  import { Router, Route } from 'svelte5-router';
  // import SidebarNav from '$lib/ui/SidebarNav.svelte';
  // import Utilities from './routes/utilities/Utilities.svelte';
</script>

<Router>
  <!-- <SidebarNav /> -->
  <main>
    <Route path="/">Home</Route>
    <Route path="/files">Files</Route>
    <!-- <Route path="/utilities/*" component={Utilities} /> -->
  </main>
</Router>

<script lang="ts">
  import { invoke } from '@tauri-apps/api/core';
  import { homeDir } from '@tauri-apps/api/path';
  import { onMount } from 'svelte';

  interface FileInfo {
    name: string;
    path: string;
    is_directory: boolean;
    size?: number;
    modified?: string;
  }

  let currentPath = '';
  let files: FileInfo[] = [];
  let loading = false;
  let error = '';

  // Format file size
  function formatFileSize(bytes: number): string {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB', 'TB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  }

  // Format date
  function formatDate(dateString: string): string {
    return new Date(dateString).toLocaleDateString();
  }

  // Browse for directory (placeholder - dialog not available in core API)
  async function browseDirectory() {
    error = 'Directory dialog not available in core API - use manual path entry for now';
    // TODO: Implement manual path entry or find alternative dialog solution
  }

  // Load directory contents
  async function loadDirectory(path: string) {
    loading = true;
    error = '';
    
    try {
      const result = await invoke<FileInfo[]>('read_directory', { path });
      files = result;
      currentPath = path;
    } catch (err) {
      error = `Failed to read directory: ${err}`;
      files = [];
    } finally {
      loading = false;
    }
  }

  // Navigate into a directory
  async function navigateToDirectory(path: string) {
    await loadDirectory(path);
  }

  // Navigate to parent directory
  async function navigateToParent() {
    if (currentPath) {
      const parentPath = currentPath.split('/').slice(0, -1).join('/');
      if (parentPath) {
        await loadDirectory(parentPath);
      }
    }
  }

  onMount(async () => {
    // Start with user's home directory
    try {
      const home = await homeDir();
      await loadDirectory(home);
    } catch (err) {
      error = `Failed to get home directory: ${err}`;
      // Fallback to current directory
      await loadDirectory('.');
    }
  });
</script>

<div class="file-browser">
  <div class="header">
    <h2>File Browser</h2>
    <div class="controls">
      <button on:click={browseDirectory} class="btn btn-primary">
        Browse Directory
      </button>
      {#if currentPath}
        <button on:click={navigateToParent} class="btn btn-secondary">
          Parent Directory
        </button>
      {/if}
    </div>
  </div>

  {#if currentPath}
    <div class="current-path">
      <strong>Current Path:</strong> {currentPath}
    </div>
  {/if}

  {#if error}
    <div class="error">
      {error}
    </div>
  {/if}

  {#if loading}
    <div class="loading">
      Loading directory contents...
    </div>
  {:else if files.length === 0}
    <div class="empty">
      No files found in this directory.
    </div>
  {:else}
    <div class="file-list">
      <div class="file-header">
        <div class="file-name">Name</div>
        <div class="file-size">Size</div>
        <div class="file-modified">Modified</div>
        <div class="file-type">Type</div>
      </div>
      
      {#each files as file}
        <div class="file-item" class:directory={file.is_directory}>
          <div class="file-name">
            {#if file.is_directory}
              📁
            {:else}
              📄
            {/if}
            <button 
              class="file-link" 
              on:click={() => file.is_directory ? navigateToDirectory(file.path) : null}
              class:clickable={file.is_directory}
              disabled={!file.is_directory}
              type="button"
            >
              {file.name}
            </button>
          </div>
          <div class="file-size">
            {#if file.size !== undefined}
              {formatFileSize(file.size)}
            {:else}
              -
            {/if}
          </div>
          <div class="file-modified">
            {#if file.modified}
              {formatDate(file.modified)}
            {:else}
              -
            {/if}
          </div>
          <div class="file-type">
            {file.is_directory ? 'Directory' : 'File'}
          </div>
        </div>
      {/each}
    </div>
  {/if}
</div>

<style>
  .file-browser {
    max-width: 1200px;
    margin: 0 auto;
    padding: 1rem;
  }

  .header {
    align-items: center;
    border-bottom: 1px solid #e0e0e0;
    display: flex;
    justify-content: space-between;
    margin-bottom: 1rem;
    padding-bottom: 1rem;
  }

  .controls {
    display: flex;
    gap: 0.5rem;
  }

  .btn {
    border: none;
    border-radius: 4px;
    cursor: pointer;
    font-size: 0.9rem;
    padding: 0.5rem 1rem;
  }

  .btn-primary {
    background-color: #007bff;
    color: white;
  }

  .btn-primary:hover {
    background-color: #0056b3;
  }

  .btn-secondary {
    background-color: #6c757d;
    color: white;
  }

  .btn-secondary:hover {
    background-color: #545b62;
  }

  .current-path {
    background-color: #f8f9fa;
    border-radius: 4px;
    font-family: monospace;
    margin-bottom: 1rem;
    padding: 0.5rem;
    word-break: break-all;
  }

  .error {
    background-color: #f8d7da;
    border-radius: 4px;
    color: #721c24;
    margin-bottom: 1rem;
    padding: 0.75rem;
  }

  .loading, .empty {
    color: #6c757d;
    padding: 2rem;
    text-align: center;
  }

  .file-list {
    border: 1px solid #e0e0e0;
    border-radius: 4px;
    overflow: hidden;
  }

  .file-header {
    background-color: #f8f9fa;
    border-bottom: 1px solid #e0e0e0;
    display: grid;
    font-weight: bold;
    gap: 1rem;
    grid-template-columns: 2fr 1fr 1fr 1fr;
    padding: 0.75rem;
  }

  .file-item {
    border-bottom: 1px solid #f0f0f0;
    display: grid;
    gap: 1rem;
    grid-template-columns: 2fr 1fr 1fr 1fr;
    padding: 0.75rem;
    transition: background-color 0.2s;
  }

  .file-item:hover {
    background-color: #f8f9fa;
  }

  .file-item.directory {
    background-color: #f0f8ff;
  }

  .file-item.directory:hover {
    background-color: #e6f3ff;
  }

  .file-name {
    align-items: center;
    display: flex;
    gap: 0.5rem;
  }

  .file-link {
    background: none;
    border: none;
    color: inherit;
    cursor: pointer;
    font: inherit;
    padding: 0;
    text-align: left;
  }

  .file-link.clickable:hover {
    color: #007bff;
    text-decoration: underline;
  }

  .file-link:disabled {
    cursor: default;
    opacity: 0.6;
  }

  .file-size, .file-modified, .file-type {
    align-items: center;
    display: flex;
  }
</style> 
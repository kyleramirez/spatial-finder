<script lang="ts">
  import type { SvelteHTMLElements } from 'svelte/elements';

  let node: HTMLElement | undefined;
  type Props = {
    left?: boolean;
    right?: boolean;
  } & SvelteHTMLElements['aside'];
  let { children, left, right, ...rest }: Props = $props();
  let handleMouseMove: (_event: MouseEvent) => void | undefined;
  let handleMouseUp: () => void | undefined;
  const handleMouseDown = (event: MouseEvent) => {
    const startX = event.clientX;
    const startWidth = node!.offsetWidth;

    handleMouseMove = (mouseMoveEvent: MouseEvent) => {
      const newWidth = left
        ? startWidth + (mouseMoveEvent.clientX - startX)
        : startWidth - (mouseMoveEvent.clientX - startX);
      node!.style.flexBasis = `${newWidth}px`;
    };

    handleMouseUp = () => {
      window.removeEventListener('mousemove', handleMouseMove);
      window.removeEventListener('mouseup', handleMouseUp);
    };

    window.addEventListener('mousemove', handleMouseMove);
    window.addEventListener('mouseup', handleMouseUp);
  };
  const handleKeyDown = (event: KeyboardEvent) => {
    const key = event.key;
    const startWidth = node!.offsetWidth;
    const nudge = left ? 10 : -10;
    if (key === 'ArrowRight') {
      node!.style.flexBasis = `${startWidth + nudge}px`;
    } else if (key === 'ArrowLeft') {
      node!.style.flexBasis = `${startWidth - nudge}px`;
    }
  };
</script>

<aside {...rest} bind:this={node} class="aside" class:left={left} class:right={right}>
  {@render children?.()}
  <span tabindex="0" role="link" onmousedown={handleMouseDown} onkeydown={handleKeyDown} class="separator"></span>
</aside>

<style lang="scss">
  $column-width: 120px;

  .separator {
    position: absolute;
    top: 0;
    bottom: 0;
    width: $gutter-width;
    cursor: col-resize;
    background-color: $gutter-color;

    &:focus {
      outline: none;
    }

    &::after {
      position: absolute;
      top: calc(50% - 10px);
      left: 1px;
      width: $gutter-width - 2px;
      height: 20px;
      content: '';
      background-color: rgba(0 0 0 / 50%);
      border-radius: 5px;
    }

    &:focus::after,
    &:hover::after {
      background-color: rgba(255 255 255 / 50%);
      box-shadow: inset 0 0 0 1px white;
    }
  }

  .aside {
    position: relative;
    box-sizing: border-box;
    min-width: $column-width;
    max-width: calc(100% - $column-width * 2);
    padding: $gutter-width;

    &.left {
      padding-right: calc($gutter-width * 2);

      .separator {
        right: 0;
      }
    }

    &.left,
    &.right {
      flex-grow: 0;
      flex-shrink: 0;
    }

    &.right {
      padding-left: calc($gutter-width * 2);

      .separator {
        left: 0;
      }
    }

    &:nth-of-type(2) {
      flex-grow: 1;
      flex-shrink: 1;

      .separator {
        display: none;
      }
    }
  }
</style>

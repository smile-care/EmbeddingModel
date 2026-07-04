<script setup lang="ts">
import {CircleHelp} from 'lucide-vue-next';
import {computed, ref} from 'vue';

const props = withDefaults(
  defineProps<{
    /** Accessible label for the help trigger. */
    label?: string;
    /** Tooltip width utility, e.g. w-72 */
    widthClass?: string;
    /**
     * below-start: under icon, align left (default, main content)
     * below-end: under icon, align right
     * floating-left: fixed panel to the left of icon (sidebar; escapes overflow clipping)
     */
    placement?: 'below-start' | 'below-end' | 'floating-left';
  }>(),
  {
    label: '指标说明',
    widthClass: 'w-80',
    placement: 'below-start',
  },
);

const open = ref(false);
const triggerRef = ref<HTMLElement | null>(null);
const floatingStyle = ref({top: '0px', left: '0px', width: '16rem'});

const useFloating = computed(() => props.placement === 'floating-left');

const staticPlacementClass = computed(() => {
  if (useFloating.value) return '';
  return props.placement === 'below-end' ? 'right-0 top-full mt-1.5' : 'left-0 top-full mt-1.5';
});

function showTip() {
  if (!useFloating.value) {
    open.value = true;
    return;
  }
  const el = triggerRef.value;
  if (!el) return;
  const rect = el.getBoundingClientRect();
  const width = Math.min(288, Math.max(200, rect.left - 16));
  floatingStyle.value = {
    top: `${rect.top + rect.height / 2}px`,
    left: `${rect.left - width - 8}px`,
    width: `${width}px`,
  };
  open.value = true;
}

function hideTip() {
  open.value = false;
}
</script>

<template>
  <span
    ref="triggerRef"
    class="group/help relative inline-flex shrink-0 align-middle"
    @mouseenter="showTip"
    @mouseleave="hideTip"
  >
    <span
      class="inline-flex rounded-full p-0.5 text-muted-foreground/60 transition-colors group-hover/help:text-muted-foreground"
      :aria-label="label"
    >
      <CircleHelp class="h-3.5 w-3.5" />
    </span>

    <Teleport v-if="useFloating" to="body">
      <div
        v-show="open"
        role="tooltip"
        class="pointer-events-none fixed z-[9999] -translate-y-1/2 rounded-lg border border-border bg-background p-3 text-[11px] leading-relaxed text-foreground shadow-xl"
        :style="floatingStyle"
      >
        <slot />
      </div>
    </Teleport>

    <div
      v-else
      role="tooltip"
      class="pointer-events-none absolute z-50 hidden rounded-lg border border-border bg-background p-3 text-[11px] leading-relaxed text-foreground shadow-xl group-hover/help:block"
      :class="[widthClass, staticPlacementClass]"
    >
      <slot />
    </div>
  </span>
</template>

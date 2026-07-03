<script setup lang="ts">
import {CircleHelp} from 'lucide-vue-next';

withDefaults(
  defineProps<{
    /** Accessible label for the help trigger. */
    label?: string;
    /** Tooltip width utility, e.g. w-72 */
    widthClass?: string;
  }>(),
  {
    label: '指标说明',
    widthClass: 'w-80',
  },
);
</script>

<template>
  <span class="group/help relative inline-flex shrink-0 align-middle">
    <span
      class="inline-flex rounded-full p-0.5 text-muted-foreground/60 transition-colors group-hover/help:text-muted-foreground"
      :aria-label="label"
    >
      <CircleHelp class="h-3.5 w-3.5" />
    </span>
    <div
      role="tooltip"
      class="pointer-events-none absolute left-0 top-full z-50 mt-1.5 hidden rounded-lg border border-border bg-background p-3 text-[11px] leading-relaxed text-foreground shadow-xl group-hover/help:block"
      :class="widthClass"
    >
      <slot />
    </div>
  </span>
</template>

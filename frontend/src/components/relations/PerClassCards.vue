<script setup lang="ts">
import {categoryChartColor, type RelationsPerClass} from '@/lib/api';

const props = defineProps<{perClass: RelationsPerClass[]; labels: string[]}>();
const emit = defineEmits<{highlight: [labels: string[]]}>();

function silTone(v: number | null): string {
  if (v == null) return 'text-muted-foreground';
  return v >= 0.5 ? 'text-emerald-500' : v >= 0.25 ? 'text-amber-500' : 'text-rose-500';
}
function simTone(v: number | null): string {
  if (v == null) return 'text-muted-foreground';
  return v >= 0.9 ? 'text-rose-500' : v >= 0.75 ? 'text-amber-500' : 'text-emerald-500';
}
function idx(label: string): number {
  return props.labels.indexOf(label);
}
</script>

<template>
  <div class="grid grid-cols-1 gap-3 sm:grid-cols-2 xl:grid-cols-3">
    <div
      v-for="c in perClass"
      :key="c.label"
      class="cursor-pointer space-y-2.5 rounded-xl border border-border bg-secondary/5 p-4 transition-all hover:border-primary/40 hover:shadow-sm"
      @click="emit('highlight', c.nearestOtherLabel ? [c.label, c.nearestOtherLabel] : [c.label])"
    >
      <div class="flex items-center justify-between">
        <div class="flex min-w-0 items-center gap-2">
          <span class="h-3 w-3 shrink-0 rounded-full" :style="{backgroundColor: categoryChartColor(idx(c.label), labels.length)}" />
          <h4 class="truncate text-sm font-semibold">{{ c.label }}</h4>
        </div>
        <span class="shrink-0 rounded bg-secondary/30 px-1.5 py-0.5 text-[10px] text-muted-foreground">{{ c.count }} crops</span>
      </div>

      <div class="grid grid-cols-2 gap-2 text-[11px]">
        <div class="flex flex-col">
          <span class="text-muted-foreground">Silhouette</span>
          <span class="font-mono font-semibold" :class="silTone(c.silhouette)">
            {{ c.silhouette == null ? '—' : c.silhouette.toFixed(2) }}
          </span>
        </div>
        <div class="flex flex-col">
          <span class="text-muted-foreground">紧致度(距心)</span>
          <span class="font-mono font-semibold text-foreground">
            {{ c.compactness == null ? '—' : c.compactness.toFixed(3) }}
          </span>
        </div>
      </div>

      <div v-if="c.nearestOtherLabel" class="border-t border-border/60 pt-2 text-[11px]">
        <span class="text-muted-foreground">最近他类：</span>
        <span class="font-medium">{{ c.nearestOtherLabel }}</span>
        <span class="ml-1 font-mono" :class="simTone(c.nearestOtherSim)">
          ({{ c.nearestOtherSim == null ? '—' : c.nearestOtherSim.toFixed(2) }})
        </span>
      </div>
    </div>
  </div>
</template>

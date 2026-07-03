<script setup lang="ts">
import {computed} from 'vue';
import {categoryChartColor} from '@/lib/api';

const props = defineProps<{
  labels: string[];
  matrix: number[][];
  /** 'confusion' rows sum to 1 (diag = purity); 'similarity' is symmetric cosine. */
  mode: 'confusion' | 'similarity';
  /** Cells at/above this value are flagged (off-diagonal). */
  threshold: number;
}>();
const emit = defineEmits<{select: [{i: number; j: number}]}>();

const n = computed(() => props.labels.length);

function cellValue(i: number, j: number): number {
  return props.matrix?.[i]?.[j] ?? 0;
}

/** Blue scale for magnitude; off-diagonal cells over threshold turn rose. */
function cellStyle(i: number, j: number) {
  const v = Math.max(0, Math.min(1, cellValue(i, j)));
  const isDiag = i === j;
  const flagged = !isDiag && v >= props.threshold;
  const alpha = 0.08 + v * 0.92;
  const bg = isDiag
    ? `rgba(16, 185, 129, ${alpha})`
    : flagged
      ? `rgba(244, 63, 94, ${alpha})`
      : `rgba(59, 130, 246, ${alpha})`;
  return {
    backgroundColor: bg,
    color: v > 0.55 ? '#fff' : 'inherit',
    outline: flagged ? '1.5px solid rgba(244,63,94,0.9)' : 'none',
  };
}

function fmt(v: number): string {
  if (props.mode === 'confusion') return v >= 0.005 ? Math.round(v * 100).toString() : '';
  return Math.abs(v) >= 0.005 ? v.toFixed(2) : '';
}
</script>

<template>
  <div class="overflow-auto">
    <table class="border-separate" style="border-spacing: 2px">
      <thead>
        <tr>
          <th class="sticky left-0 z-10 bg-background" />
          <th
            v-for="(lab, j) in labels"
            :key="'h' + j"
            class="max-w-[64px] px-1 pb-1 align-bottom text-[9px] font-medium text-muted-foreground"
          >
            <div class="mx-auto flex items-center gap-1">
              <span class="h-2 w-2 shrink-0 rounded-full" :style="{backgroundColor: categoryChartColor(j, n)}" />
              <span class="truncate">{{ lab }}</span>
            </div>
          </th>
        </tr>
      </thead>
      <tbody>
        <tr v-for="(rowLab, i) in labels" :key="'r' + i">
          <th class="sticky left-0 z-10 bg-background pr-2 text-right text-[9px] font-medium text-muted-foreground">
            <div class="flex items-center justify-end gap-1">
              <span class="max-w-[80px] truncate">{{ rowLab }}</span>
              <span class="h-2 w-2 shrink-0 rounded-full" :style="{backgroundColor: categoryChartColor(i, n)}" />
            </div>
          </th>
          <td
            v-for="(colLab, j) in labels"
            :key="'c' + i + '-' + j"
            class="h-8 w-8 cursor-pointer rounded text-center align-middle font-mono text-[9px] transition-transform hover:scale-110"
            :style="cellStyle(i, j)"
            :title="`${rowLab} → ${colLab}: ${cellValue(i, j).toFixed(3)}`"
            @click="emit('select', {i, j})"
          >
            {{ fmt(cellValue(i, j)) }}
          </td>
        </tr>
      </tbody>
    </table>
  </div>
</template>

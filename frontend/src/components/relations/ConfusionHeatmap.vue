<script setup lang="ts">
import {computed, onMounted, onUnmounted, ref, watch} from 'vue';
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

/** Fit heatmap inside min(containerW, containerH) × this ratio. */
const FIT_RATIO = 0.99;
const LABEL_COL_PX = 96;
const HEADER_ROW_PX = 32;
const CELL_GAP_PX = 2;
const MIN_CELL_PX = 18;
const MAX_CELL_PX = 96;

const rootRef = ref<HTMLElement | null>(null);
const cellPx = ref(32);

const n = computed(() => props.labels.length);

const tableWidthPx = computed(
  () => LABEL_COL_PX + n.value * cellPx.value + (n.value + 1) * CELL_GAP_PX,
);
const tableHeightPx = computed(
  () => HEADER_ROW_PX + n.value * cellPx.value + (n.value + 1) * CELL_GAP_PX,
);

const cellTextClass = computed(() => {
  const size = cellPx.value;
  if (size >= 56) return 'text-[11px]';
  if (size >= 40) return 'text-[10px]';
  if (size >= 28) return 'text-[9px]';
  return 'text-[8px]';
});

function cellValue(i: number, j: number): number {
  return props.matrix?.[i]?.[j] ?? 0;
}

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
    width: `${cellPx}px`,
    height: `${cellPx}px`,
  };
}

function fmt(v: number): string {
  if (props.mode === 'confusion') return v >= 0.005 ? Math.round(v * 100).toString() : '';
  return Math.abs(v) >= 0.005 ? v.toFixed(2) : '';
}

function updateCellSize() {
  const el = rootRef.value;
  const count = n.value;
  if (!el || count <= 0) return;

  const {width, height} = el.getBoundingClientRect();
  if (width <= 0 || height <= 0) return;

  const cap = Math.min(width, height) * FIT_RATIO;
  const cellFromW = (cap - LABEL_COL_PX - (count + 1) * CELL_GAP_PX) / count;
  const cellFromH = (cap - HEADER_ROW_PX - (count + 1) * CELL_GAP_PX) / count;
  const next = Math.floor(Math.min(cellFromW, cellFromH, MAX_CELL_PX));
  cellPx.value = Math.max(MIN_CELL_PX, next);
}

let resizeObserver: ResizeObserver | null = null;

onMounted(() => {
  resizeObserver = new ResizeObserver(() => updateCellSize());
  if (rootRef.value) resizeObserver.observe(rootRef.value);
  updateCellSize();
});

onUnmounted(() => {
  resizeObserver?.disconnect();
  resizeObserver = null;
});

watch(n, () => updateCellSize());
watch(() => props.matrix, () => updateCellSize(), {deep: false});
</script>

<template>
  <div ref="rootRef" class="flex h-full min-h-[12rem] w-full items-center justify-center overflow-hidden">
    <table
      class="border-separate"
      :style="{
        borderSpacing: `${CELL_GAP_PX}px`,
        width: `${tableWidthPx}px`,
        height: `${tableHeightPx}px`,
      }"
    >
      <colgroup>
        <col :style="{width: `${LABEL_COL_PX}px`}" />
        <col v-for="j in n" :key="'col-' + j" :style="{width: `${cellPx}px`}" />
      </colgroup>
      <thead>
        <tr :style="{height: `${HEADER_ROW_PX}px`}">
          <th class="sticky left-0 z-10 bg-background" />
          <th
            v-for="(lab, j) in labels"
            :key="'h' + j"
            class="px-0.5 align-bottom text-[9px] font-medium text-muted-foreground"
          >
            <div class="mx-auto flex min-w-0 items-center justify-center gap-1">
              <span class="h-2 w-2 shrink-0 rounded-full" :style="{backgroundColor: categoryChartColor(j, n)}" />
              <span class="truncate">{{ lab }}</span>
            </div>
          </th>
        </tr>
      </thead>
      <tbody>
        <tr v-for="(rowLab, i) in labels" :key="'r' + i">
          <th class="sticky left-0 z-10 bg-background pr-2 text-right text-[9px] font-medium text-muted-foreground">
            <div class="flex min-w-0 items-center justify-end gap-1">
              <span class="truncate">{{ rowLab }}</span>
              <span class="h-2 w-2 shrink-0 rounded-full" :style="{backgroundColor: categoryChartColor(i, n)}" />
            </div>
          </th>
          <td
            v-for="(colLab, j) in labels"
            :key="'c' + i + '-' + j"
            class="cursor-pointer p-0 align-middle transition-transform hover:z-[1] hover:scale-105"
            :style="cellStyle(i, j)"
            :title="`${rowLab} → ${colLab}: ${cellValue(i, j).toFixed(3)}`"
            @click="emit('select', {i, j})"
          >
            <div
              class="flex h-full w-full items-center justify-center rounded font-mono leading-none"
              :class="cellTextClass"
            >
              {{ fmt(cellValue(i, j)) }}
            </div>
          </td>
        </tr>
      </tbody>
    </table>
  </div>
</template>

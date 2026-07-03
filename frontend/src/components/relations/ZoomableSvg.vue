<script setup lang="ts">
import {computed, ref, watch} from 'vue';
import {Minus, Plus, RotateCcw} from 'lucide-vue-next';

const props = withDefaults(
  defineProps<{
    width: number;
    height: number;
    /** Reset pan/zoom when this value changes (e.g. dataset key). */
    resetKey?: string | number;
  }>(),
  {resetKey: ''},
);

const containerRef = ref<HTMLElement | null>(null);
const zoom = ref(1);
const panX = ref(0);
const panY = ref(0);
const isPanning = ref(false);
const didDrag = ref(false);

const DRAG_THRESHOLD = 4;
const MIN_ZOOM = 0.25;
const MAX_ZOOM = 12;

let _dragStartClient = {x: 0, y: 0};
let _dragStartPan = {x: 0, y: 0};

const viewBox = computed(() => {
  const vw = props.width / zoom.value;
  const vh = props.height / zoom.value;
  return `${panX.value} ${panY.value} ${vw} ${vh}`;
});

const zoomPctLabel = computed(() => `${Math.round(zoom.value * 100)}%`);

function clampZoom(z: number): number {
  return Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, z));
}

function resetView() {
  zoom.value = 1;
  panX.value = 0;
  panY.value = 0;
}

function zoomAt(clientX: number, clientY: number, factor: number) {
  const el = containerRef.value;
  if (!el) return;
  const rect = el.getBoundingClientRect();
  if (rect.width <= 0 || rect.height <= 0) return;

  const sx = (clientX - rect.left) / rect.width;
  const sy = (clientY - rect.top) / rect.height;
  const oldVw = props.width / zoom.value;
  const oldVh = props.height / zoom.value;
  const svgX = panX.value + sx * oldVw;
  const svgY = panY.value + sy * oldVh;

  const nextZoom = clampZoom(zoom.value * factor);
  const newVw = props.width / nextZoom;
  const newVh = props.height / nextZoom;
  panX.value = svgX - sx * newVw;
  panY.value = svgY - sy * newVh;
  zoom.value = nextZoom;
}

function onWheel(e: WheelEvent) {
  const factor = e.deltaY > 0 ? 0.92 : 1.08;
  zoomAt(e.clientX, e.clientY, factor);
}

function zoomIn() {
  const el = containerRef.value;
  if (!el) return;
  const rect = el.getBoundingClientRect();
  zoomAt(rect.left + rect.width / 2, rect.top + rect.height / 2, 1.15);
}

function zoomOut() {
  const el = containerRef.value;
  if (!el) return;
  const rect = el.getBoundingClientRect();
  zoomAt(rect.left + rect.width / 2, rect.top + rect.height / 2, 1 / 1.15);
}

function onPointerDown(e: PointerEvent) {
  if (e.button !== 0) return;
  didDrag.value = false;
  isPanning.value = true;
  _dragStartClient = {x: e.clientX, y: e.clientY};
  _dragStartPan = {x: panX.value, y: panY.value};
  (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
}

function onPointerMove(e: PointerEvent) {
  if (!isPanning.value || !containerRef.value) return;
  const dxClient = e.clientX - _dragStartClient.x;
  const dyClient = e.clientY - _dragStartClient.y;
  if (!didDrag.value && Math.hypot(dxClient, dyClient) < DRAG_THRESHOLD) return;
  didDrag.value = true;

  const rect = containerRef.value.getBoundingClientRect();
  const vw = props.width / zoom.value;
  const vh = props.height / zoom.value;
  panX.value = _dragStartPan.x - (dxClient / rect.width) * vw;
  panY.value = _dragStartPan.y - (dyClient / rect.height) * vh;
}

function onPointerUp(e: PointerEvent) {
  if (!isPanning.value) return;
  isPanning.value = false;
  didDrag.value = false;
  try {
    (e.currentTarget as HTMLElement).releasePointerCapture(e.pointerId);
  } catch { /* ignore */ }
}

watch(() => props.resetKey, () => resetView());
</script>

<template>
  <div class="flex h-full min-h-[16rem] w-full flex-col">
    <div
      ref="containerRef"
      class="relative min-h-0 flex-1 overflow-hidden rounded-lg border border-border/30 bg-secondary/5"
      :class="didDrag ? 'cursor-grabbing' : 'cursor-grab'"
      @wheel.prevent="onWheel"
      @pointerdown="onPointerDown"
      @pointermove="onPointerMove"
      @pointerup="onPointerUp"
      @pointercancel="onPointerUp"
    >
      <svg
        :viewBox="viewBox"
        class="h-full w-full touch-none select-none"
        preserveAspectRatio="xMidYMid meet"
      >
        <slot />
      </svg>

      <div class="pointer-events-none absolute bottom-2 left-2 text-[9px] text-muted-foreground/80">
        滚轮缩放 · 拖动平移
      </div>

      <div class="absolute bottom-2 right-2 flex items-center gap-1">
        <button
          type="button"
          class="flex h-7 w-7 items-center justify-center rounded-md border border-border/60 bg-background/90 text-muted-foreground shadow-sm backdrop-blur-sm transition-colors hover:bg-secondary/40 hover:text-foreground"
          aria-label="缩小"
          @click.stop="zoomOut"
        >
          <Minus class="h-3.5 w-3.5" />
        </button>
        <button
          type="button"
          class="h-7 min-w-[3rem] rounded-md border border-border/60 bg-background/90 px-1.5 text-[10px] font-mono font-semibold text-muted-foreground shadow-sm backdrop-blur-sm transition-colors hover:bg-secondary/40 hover:text-foreground"
          aria-label="重置视图"
          @click.stop="resetView"
        >
          {{ zoomPctLabel }}
        </button>
        <button
          type="button"
          class="flex h-7 w-7 items-center justify-center rounded-md border border-border/60 bg-background/90 text-muted-foreground shadow-sm backdrop-blur-sm transition-colors hover:bg-secondary/40 hover:text-foreground"
          aria-label="放大"
          @click.stop="zoomIn"
        >
          <Plus class="h-3.5 w-3.5" />
        </button>
        <button
          type="button"
          class="flex h-7 w-7 items-center justify-center rounded-md border border-border/60 bg-background/90 text-muted-foreground shadow-sm backdrop-blur-sm transition-colors hover:bg-secondary/40 hover:text-foreground"
          aria-label="适应窗口"
          title="适应窗口"
          @click.stop="resetView"
        >
          <RotateCcw class="h-3.5 w-3.5" />
        </button>
      </div>
    </div>

    <div v-if="$slots.footer" class="shrink-0 pt-2">
      <slot name="footer" />
    </div>
  </div>
</template>

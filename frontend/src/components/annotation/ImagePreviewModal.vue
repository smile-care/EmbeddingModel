<script setup lang="ts">
import {computed, nextTick, onUnmounted, ref, watch} from 'vue';
import {ArrowLeft, ExternalLink, Minus, Plus, X} from 'lucide-vue-next';
import AnnotationOverlay from '@/components/annotation/AnnotationOverlay.vue';
import {combineLegendScale, type AnnotationOverlayRegion} from '@/components/annotation/overlayUtils';

const props = withDefaults(
  defineProps<{
    open: boolean;
    imageUrl: string | null;
    regions: AnnotationOverlayRegion[];
    colorForClass: (classId: string | null | undefined, isSubtract?: boolean) => string;
    labelForClass?: (classId: string | null | undefined) => string;
    viewMode?: 'crop' | 'source';
    canTraceSource?: boolean;
    showBackToCrop?: boolean;
  }>(),
  {
    viewMode: 'crop',
    canTraceSource: false,
    showBackToCrop: false,
  },
);

const emit = defineEmits<{
  close: [];
  'trace-source': [];
  'back-to-crop': [];
}>();

const showAnnotations = ref(true);
const userZoom = ref(1);
const fitScale = ref(1);
const pan = ref({x: 0, y: 0});
const naturalSize = ref({w: 0, h: 0});
const containerSize = ref({w: 0, h: 0});

const PREVIEW_ZOOM_MIN = 0.2;
const PREVIEW_ZOOM_MAX = 6;

const viewportRef = ref<HTMLDivElement | null>(null);
const imgRef = ref<HTMLImageElement | null>(null);

const effectiveScale = computed(() => fitScale.value * userZoom.value);

const legendScale = computed(() => {
  const displayW = naturalSize.value.w * effectiveScale.value;
  return combineLegendScale(effectiveScale.value, displayW);
});

const annotationStrokeWidth = computed(() => {
  const nw = naturalSize.value.w;
  const scale = effectiveScale.value;
  if (!nw || !scale) return 0.002;
  const targetPx = props.viewMode === 'source' ? 1.25 : 1.75;
  return targetPx / (nw * scale);
});

const canPan = computed(() => {
  const nw = naturalSize.value.w;
  const nh = naturalSize.value.h;
  const cw = containerSize.value.w;
  const ch = containerSize.value.h;
  if (!nw || !nh || !cw || !ch) return false;
  const s = effectiveScale.value;
  return nw * s > cw + 0.5 || nh * s > ch + 0.5;
});

const panHint = computed(() => (canPan.value ? '拖动平移' : '已适配窗口'));

function updateContainerSize() {
  const el = viewportRef.value;
  if (!el) return;
  containerSize.value = {w: el.clientWidth, h: el.clientHeight};
}

function computeFitScale() {
  const img = imgRef.value;
  const container = viewportRef.value;
  if (!img || !container) return;
  const nw = img.naturalWidth || img.width;
  const nh = img.naturalHeight || img.height;
  if (!nw || !nh) return;
  naturalSize.value = {w: nw, h: nh};
  const cw = container.clientWidth - 32;
  const ch = container.clientHeight - 32;
  fitScale.value = Math.min(1, cw / nw, ch / nh);
  userZoom.value = 1;
  pan.value = {x: 0, y: 0};
  updateContainerSize();
}

function resetViewState() {
  userZoom.value = 1;
  fitScale.value = 1;
  pan.value = {x: 0, y: 0};
  naturalSize.value = {w: 0, h: 0};
  showAnnotations.value = true;
}

let kbdHandler: ((e: KeyboardEvent) => void) | null = null;
let wheelHandler: ((e: WheelEvent) => void) | null = null;
let wheelEl: HTMLElement | null = null;
let prevBodyOverflow = '';
let resizeObs: ResizeObserver | null = null;

function isTypingTarget(target: EventTarget | null) {
  const el = target as HTMLElement | null;
  if (!el) return false;
  const tag = el.tagName;
  return tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT' || el.isContentEditable;
}

function attachGlobalHandlers() {
  detachGlobalHandlers();
  prevBodyOverflow = document.body.style.overflow;
  document.body.style.overflow = 'hidden';

  kbdHandler = (e: KeyboardEvent) => {
    if (e.key === 'Escape') {
      emit('close');
      return;
    }
    if (e.key === ' ' || e.key === 'Space' || e.code === 'Space') {
      if (isTypingTarget(e.target)) return;
      e.preventDefault();
      if (props.regions.length) showAnnotations.value = !showAnnotations.value;
    }
  };
  window.addEventListener('keydown', kbdHandler);

  void nextTick(() => {
    const el = viewportRef.value;
    if (!el) return;
    updateContainerSize();
    resizeObs = new ResizeObserver(() => updateContainerSize());
    resizeObs.observe(el);
    wheelEl = el;
    wheelHandler = (e: WheelEvent) => {
      e.preventDefault();
      userZoom.value = Math.min(
        PREVIEW_ZOOM_MAX,
        Math.max(PREVIEW_ZOOM_MIN, userZoom.value - e.deltaY * 0.002),
      );
    };
    el.addEventListener('wheel', wheelHandler, {passive: false});
  });
}

function detachGlobalHandlers() {
  if (kbdHandler) {
    window.removeEventListener('keydown', kbdHandler);
    kbdHandler = null;
  }
  if (wheelEl && wheelHandler) {
    wheelEl.removeEventListener('wheel', wheelHandler);
    wheelEl = null;
    wheelHandler = null;
  }
  resizeObs?.disconnect();
  resizeObs = null;
  document.body.style.overflow = prevBodyOverflow;
}

const dragging = ref(false);
let dragStartClient = {x: 0, y: 0};
let dragStartPan = {x: 0, y: 0};

function onPointerDown(e: PointerEvent) {
  if (e.button !== 0 || !canPan.value) return;
  dragging.value = true;
  dragStartClient = {x: e.clientX, y: e.clientY};
  dragStartPan = {...pan.value};
  (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
}

function onPointerMove(e: PointerEvent) {
  if (!dragging.value) return;
  pan.value = {
    x: dragStartPan.x + (e.clientX - dragStartClient.x),
    y: dragStartPan.y + (e.clientY - dragStartClient.y),
  };
}

function onPointerUp(e: PointerEvent) {
  dragging.value = false;
  (e.currentTarget as HTMLElement).releasePointerCapture(e.pointerId);
}

watch(
  () => props.open,
  (isOpen) => {
    if (isOpen) {
      resetViewState();
      attachGlobalHandlers();
    } else {
      detachGlobalHandlers();
    }
  },
);

watch(
  () => props.imageUrl,
  () => {
    if (!props.open) return;
    userZoom.value = 1;
    fitScale.value = 1;
    pan.value = {x: 0, y: 0};
  },
);

watch(userZoom, (z) => {
  if (z === 1) pan.value = {x: 0, y: 0};
  void nextTick(() => updateContainerSize());
});

watch(canPan, (can) => {
  if (!can) {
    pan.value = {x: 0, y: 0};
    dragging.value = false;
  }
});

onUnmounted(() => detachGlobalHandlers());
</script>

<template>
  <Teleport to="body">
    <Transition name="fade">
      <div
        v-if="open && imageUrl"
        class="fixed inset-0 z-[200] flex flex-col bg-black/88 backdrop-blur-sm"
        @click.self="emit('close')"
      >
        <div class="flex shrink-0 flex-wrap items-center justify-between gap-3 border-b border-white/10 px-4 py-3 text-white" @click.stop>
          <span class="text-xs text-white/70">
            滚轮缩放 · {{ panHint }} · Esc 关闭 · 空格切换标注
            <span class="text-white/40"> · {{ viewMode === 'source' ? '原图' : '裁剪图' }}</span>
          </span>
          <div class="flex flex-wrap items-center justify-end gap-2">
            <button
              v-if="viewMode === 'crop' && canTraceSource"
              type="button"
              class="flex items-center gap-1.5 rounded-lg bg-white/10 px-3 py-1.5 text-xs font-medium text-white/90 backdrop-blur-sm transition-colors hover:bg-white/20"
              @click="emit('trace-source')"
            >
              <ExternalLink class="h-3.5 w-3.5" />
              查看原图
            </button>
            <button
              v-if="viewMode === 'source' && showBackToCrop"
              type="button"
              class="flex items-center gap-1.5 rounded-lg bg-white/10 px-3 py-1.5 text-xs font-medium text-white/90 backdrop-blur-sm transition-colors hover:bg-white/20"
              @click="emit('back-to-crop')"
            >
              <ArrowLeft class="h-3.5 w-3.5" />
              查看裁剪图
            </button>
            <div
              v-if="regions.length"
              class="flex items-center gap-1 rounded-lg bg-white/10 px-2 py-1 text-[11px] text-white/80"
            >
              <button
                type="button"
                class="rounded px-1.5 py-0.5 font-medium transition-colors"
                :class="showAnnotations ? 'bg-indigo-500/50 text-indigo-100' : 'text-white/40 hover:text-white'"
                @click="showAnnotations = !showAnnotations"
              >
                {{ showAnnotations ? '隐藏' : '显示' }}标注
              </button>
            </div>
            <span class="min-w-[3rem] text-center font-mono text-xs tabular-nums text-white/80">
              {{ Math.round(effectiveScale * 100) }}%
            </span>
            <button
              type="button"
              class="rounded-md border border-white/20 p-1.5 hover:bg-white/10"
              aria-label="缩小"
              @click="userZoom = Math.max(PREVIEW_ZOOM_MIN, Math.round((userZoom - 0.15) * 100) / 100)"
            >
              <Minus class="h-4 w-4" />
            </button>
            <button
              type="button"
              class="rounded-md border border-white/20 p-1.5 hover:bg-white/10"
              aria-label="放大"
              @click="userZoom = Math.min(PREVIEW_ZOOM_MAX, Math.round((userZoom + 0.15) * 100) / 100)"
            >
              <Plus class="h-4 w-4" />
            </button>
            <button
              type="button"
              class="rounded-md border border-white/20 p-1.5 hover:bg-white/10"
              aria-label="关闭"
              @click="emit('close')"
            >
              <X class="h-4 w-4" />
            </button>
          </div>
        </div>

        <div
          ref="viewportRef"
          class="flex min-h-0 flex-1 overflow-hidden"
          :class="
            !canPan
              ? 'cursor-default'
              : dragging
                ? 'cursor-grabbing select-none'
                : 'cursor-grab'
          "
          @click.stop
          @pointerdown="onPointerDown"
          @pointermove="onPointerMove"
          @pointerup="onPointerUp"
          @pointercancel="onPointerUp"
        >
          <div class="flex min-h-full w-full items-center justify-center">
            <div
              class="relative inline-block select-none leading-none"
              :style="{
                transform: `translate(${pan.x}px, ${pan.y}px) scale(${effectiveScale})`,
                transformOrigin: 'center center',
              }"
            >
              <img
                ref="imgRef"
                :src="imageUrl"
                alt="预览"
                draggable="false"
                class="block select-none"
                style="display: block; max-width: none; max-height: none"
                @load="computeFitScale"
              />
              <AnnotationOverlay
                :regions="regions"
                :color-for-class="colorForClass"
                :label-for-class="labelForClass"
                :show-labels="showAnnotations"
                :stroke-width="annotationStrokeWidth"
                :zoom="1"
                :legend-scale="legendScale"
                :image-width="naturalSize.w"
                :image-height="naturalSize.h"
              />
            </div>
          </div>
        </div>
      </div>
    </Transition>
  </Teleport>
</template>

<style scoped>
.fade-enter-active,
.fade-leave-active {
  transition: opacity 0.15s ease;
}
.fade-enter-from,
.fade-leave-to {
  opacity: 0;
}
</style>

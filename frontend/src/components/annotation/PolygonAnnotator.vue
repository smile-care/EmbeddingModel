<script setup lang="ts">
import {computed, onBeforeUnmount, onMounted, ref, watch} from 'vue';
import {Check, ChevronLeft, ChevronRight, Hand, Minus, MousePointer2, Plus, Redo2, RotateCcw, Sparkles, Trash2, Undo2, Wand2, X, ZoomIn, ZoomOut} from 'lucide-vue-next';
import {classColor, type AnnotationRegion, type DefectClass, type RegionInput} from '../../lib/api';
import AnnotationOverlay from '@/components/annotation/AnnotationOverlay.vue';
import type {AnnotationOverlayRegion} from '@/components/annotation/overlayUtils';
import {combineLegendScale} from '@/components/annotation/overlayUtils';
import {clientToNorm, clonePoints, dist, toSvgPoints, type Point} from './geometry';
import {buildWandSource, magicWandPolygon, type WandSource} from './magicWand';

interface EditRegion {
  id: string;
  classId: string | null;
  points: Point[];
  isSubtract: boolean;
}

const props = defineProps<{
  imageUrl: string;
  imageName?: string;
  classes: DefectClass[];
  initialRegions?: AnnotationRegion[];
  saving?: boolean;
  /** Optional async hook to create a new defect class; returns the created class. */
  createClass?: (name: string, color: string) => Promise<DefectClass>;
  /** 1-based position of the current image within the dataset (for navigation). */
  index?: number;
  total?: number;
  hasPrev?: boolean;
  hasNext?: boolean;
}>();

const emit = defineEmits<{
  (e: 'save', regions: RegionInput[], advance: boolean): void;
  (e: 'cancel'): void;
  (e: 'navigate', dir: -1 | 1): void;
}>();

// ── State ────────────────────────────────────────────────────────────────────
type Tool = 'draw' | 'wand' | 'pan';
const tool = ref<Tool>('draw');
const subtractMode = ref(false);
const currentClassId = ref<string | null>(null);

// magic-wand
const wandTolerance = ref(28);
const wandSource = ref<WandSource | null>(null);
const wandBusy = ref(false);

// dirty tracking (for navigation guard)
const dirty = ref(false);
function markDirty() { dirty.value = true; }

const regions = ref<EditRegion[]>([]);
const draft = ref<Point[]>([]);
const selectedRegionId = ref<string | null>(null);
const hoverNorm = ref<Point | null>(null);

// view transform
const scale = ref(1);
const tx = ref(0);
const ty = ref(0);
/** natural image size (px), used to keep vertex markers a constant screen size */
const nat = ref({w: 0, h: 0});

const viewport = ref<HTMLElement | null>(null);
const imgEl = ref<HTMLImageElement | null>(null);

// history (region-level snapshots)
const past: EditRegion[][] = [];
const future: EditRegion[][] = [];

function snapshot(): EditRegion[] {
  return regions.value.map((r) => ({...r, points: clonePoints(r.points)}));
}
function pushHistory() {
  past.push(snapshot());
  if (past.length > 100) past.shift();
  future.length = 0;
  markDirty();
}

const uid = () => `r_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 7)}`;

function loadInitial() {
  regions.value = (props.initialRegions ?? []).map((r) => ({
    id: r.id || uid(),
    classId: r.classId ?? null,
    points: (r.points ?? []).map(([x, y]) => [x, y] as Point),
    isSubtract: !!r.isSubtract,
  }));
  draft.value = [];
  past.length = 0;
  future.length = 0;
  selectedRegionId.value = null;
  dirty.value = false;
}

watch(() => props.imageUrl, () => { resetView(); loadInitial(); wandSource.value = null; });
watch(
  () => props.classes,
  (cls) => {
    if (!currentClassId.value && cls.length) currentClassId.value = cls[0].id;
  },
  {immediate: true},
);

// ── Class color lookup ───────────────────────────────────────────────────────
const classMap = computed(() => {
  const m = new Map<string, DefectClass>();
  props.classes.forEach((c) => m.set(c.id, c));
  return m;
});
function colorFor(classId: string | null, isSubtract: boolean): string {
  if (isSubtract) return '#94a3b8';
  if (!classId) return '#64748b';
  return classColor(classMap.value.get(classId) ?? null);
}
function labelForClass(classId: string | null | undefined): string {
  if (!classId) return 'Unassigned';
  return classMap.value.get(classId)?.name ?? 'Unknown';
}
const overlayRegions = computed((): AnnotationOverlayRegion[] =>
  regions.value.map((r) => ({
    points: r.points,
    isSubtract: r.isSubtract,
    classId: r.classId,
  })),
);
function overlayColorForClass(classId: string | null | undefined, isSubtract?: boolean): string {
  return colorFor(classId ?? null, !!isSubtract);
}
const annotatorLegendScale = computed(() => {
  const displayW = nat.value.w * scale.value;
  return combineLegendScale(scale.value, displayW);
});

// ── Coordinate / drawing ─────────────────────────────────────────────────────
function isPanGesture(e: PointerEvent): boolean {
  return tool.value === 'pan' || e.button === 1 || spaceHeld.value;
}

const panning = ref(false);
let panStart = {x: 0, y: 0, tx: 0, ty: 0};
const spaceHeld = ref(false);

function onPointerDown(e: PointerEvent) {
  if (isPanGesture(e)) {
    panning.value = true;
    panStart = {x: e.clientX, y: e.clientY, tx: tx.value, ty: ty.value};
    (e.target as HTMLElement).setPointerCapture?.(e.pointerId);
    e.preventDefault();
    return;
  }
  if (e.button !== 0) return;
  if (!imgEl.value) return;
  if (tool.value === 'wand') {
    runMagicWand(clientToNorm(imgEl.value, e.clientX, e.clientY));
    return;
  }
  if (tool.value !== 'draw') return;
  const p = clientToNorm(imgEl.value, e.clientX, e.clientY);
  // close polygon if clicking near the first point
  if (draft.value.length >= 3 && dist(p, draft.value[0]) < 0.015) {
    finishDraft();
    return;
  }
  draft.value = [...draft.value, p];
}

function onPointerMove(e: PointerEvent) {
  if (panning.value) {
    tx.value = panStart.tx + (e.clientX - panStart.x);
    ty.value = panStart.ty + (e.clientY - panStart.y);
    return;
  }
  if (tool.value === 'draw' && imgEl.value) {
    hoverNorm.value = clientToNorm(imgEl.value, e.clientX, e.clientY);
  }
}

function onPointerUp(e: PointerEvent) {
  if (panning.value) {
    panning.value = false;
    (e.target as HTMLElement).releasePointerCapture?.(e.pointerId);
  }
}

function finishDraft() {
  if (draft.value.length < 3) return;
  pushHistory();
  regions.value = [
    ...regions.value,
    {
      id: uid(),
      classId: subtractMode.value ? null : currentClassId.value,
      points: clonePoints(draft.value),
      isSubtract: subtractMode.value,
    },
  ];
  draft.value = [];
}

function cancelDraft() {
  draft.value = [];
}

// ── Magic wand ────────────────────────────────────────────────────────────────
function onImgLoad() {
  // (re)capture pixels lazily; rebuilt on first wand use if still null
  wandSource.value = imgEl.value ? buildWandSource(imgEl.value) : null;
  fitView();
}

function runMagicWand(p: Point) {
  if (!imgEl.value) return;
  if (!wandSource.value) wandSource.value = buildWandSource(imgEl.value);
  const src = wandSource.value;
  if (!src) {
    window.alert('无法读取该图片像素（可能跨域），魔术棒不可用。');
    return;
  }
  wandBusy.value = true;
  try {
    const poly = magicWandPolygon(src, p, wandTolerance.value);
    if (!poly || poly.length < 3) return;
    pushHistory();
    regions.value = [
      ...regions.value,
      {
        id: uid(),
        classId: subtractMode.value ? null : currentClassId.value,
        points: poly,
        isSubtract: subtractMode.value,
      },
    ];
  } finally {
    wandBusy.value = false;
  }
}

// ── Image navigation ──────────────────────────────────────────────────────────
function requestNavigate(dir: -1 | 1) {
  if (dir === -1 && !props.hasPrev) return;
  if (dir === 1 && !props.hasNext) return;
  if (dirty.value && !window.confirm('当前图片有未保存的修改，确定切换并放弃？')) return;
  emit('navigate', dir);
}

function undoVertex() {
  if (draft.value.length > 0) {
    draft.value = draft.value.slice(0, -1);
  } else {
    undo();
  }
}

function undo() {
  if (!past.length) return;
  future.push(snapshot());
  regions.value = past.pop()!;
}
function redo() {
  if (!future.length) return;
  past.push(snapshot());
  regions.value = future.pop()!;
}

function deleteRegion(id: string) {
  pushHistory();
  regions.value = regions.value.filter((r) => r.id !== id);
  if (selectedRegionId.value === id) selectedRegionId.value = null;
}

function setRegionClass(id: string, classId: string) {
  pushHistory();
  regions.value = regions.value.map((r) => (r.id === id ? {...r, classId} : r));
}

function clearAll() {
  if (!regions.value.length && !draft.value.length) return;
  pushHistory();
  regions.value = [];
  draft.value = [];
}

// ── Zoom / pan ───────────────────────────────────────────────────────────────
function applyZoom(factor: number, originClientX?: number, originClientY?: number) {
  const vp = viewport.value;
  if (!vp) return;
  const rect = vp.getBoundingClientRect();
  const ox = (originClientX ?? rect.left + rect.width / 2) - rect.left;
  const oy = (originClientY ?? rect.top + rect.height / 2) - rect.top;
  const newScale = Math.min(8, Math.max(0.05, scale.value * factor));
  const ratio = newScale / scale.value;
  // keep the point under cursor stationary
  tx.value = ox - (ox - tx.value) * ratio;
  ty.value = oy - (oy - ty.value) * ratio;
  scale.value = newScale;
}

function onWheel(e: WheelEvent) {
  e.preventDefault();
  applyZoom(e.deltaY < 0 ? 1.12 : 1 / 1.12, e.clientX, e.clientY);
}

/** Scale the image to fit the viewport and center it. */
function fitView() {
  const vp = viewport.value;
  const img = imgEl.value;
  const iw = img?.naturalWidth ?? 0;
  const ih = img?.naturalHeight ?? 0;
  if (!vp || !iw || !ih) {
    scale.value = 1;
    tx.value = 0;
    ty.value = 0;
    return;
  }
  nat.value = {w: iw, h: ih};
  const rect = vp.getBoundingClientRect();
  const pad = 32;
  const s = Math.min((rect.width - pad) / iw, (rect.height - pad) / ih);
  const fit = Math.min(8, Math.max(0.05, s));
  scale.value = fit;
  tx.value = (rect.width - iw * fit) / 2;
  ty.value = (rect.height - ih * fit) / 2;
}

function resetView() {
  fitView();
}

// ── Keyboard ─────────────────────────────────────────────────────────────────
function onKeyDown(e: KeyboardEvent) {
  if ((e.target as HTMLElement)?.tagName === 'INPUT') return;
  if (e.code === 'Space') {
    spaceHeld.value = true;
    e.preventDefault();
    return;
  }
  if (e.key === 'Escape') cancelDraft();
  else if (e.key === 'Enter') finishDraft();
  else if (e.key === 'z' || e.key === 'Z') {
    if (e.ctrlKey || e.metaKey) { e.shiftKey ? redo() : undo(); }
    else undoVertex();
  }
  else if (e.key === 'b' || e.key === 'B') tool.value = 'draw';
  else if (e.key === 'w' || e.key === 'W') tool.value = 'wand';
  else if (e.key === '[') requestNavigate(-1);
  else if (e.key === ']') requestNavigate(1);
}
function onKeyUp(e: KeyboardEvent) {
  if (e.code === 'Space') spaceHeld.value = false;
}

onMounted(() => {
  window.addEventListener('keydown', onKeyDown);
  window.addEventListener('keyup', onKeyUp);
  window.addEventListener('resize', fitView);
  loadInitial();
  // image may already be decoded (cached) before @load was attached
  if (imgEl.value?.complete && imgEl.value.naturalWidth) onImgLoad();
});
onBeforeUnmount(() => {
  window.removeEventListener('keydown', onKeyDown);
  window.removeEventListener('keyup', onKeyUp);
  window.removeEventListener('resize', fitView);
});

// ── New class form ───────────────────────────────────────────────────────────
const newClassName = ref('');
const creatingClass = ref(false);
async function addClass() {
  const name = newClassName.value.trim();
  if (!name || !props.createClass) return;
  creatingClass.value = true;
  try {
    const color = classColor(null, props.classes.length);
    const created = await props.createClass(name, color);
    currentClassId.value = created.id;
    newClassName.value = '';
  } finally {
    creatingClass.value = false;
  }
}

// ── Save ─────────────────────────────────────────────────────────────────────
const canSave = computed(() => regions.value.some((r) => !r.isSubtract));
function onSave(advance = false) {
  finishDraft();
  const payload: RegionInput[] = regions.value.map((r) => ({
    classId: r.isSubtract ? null : r.classId,
    points: r.points,
    isSubtract: r.isSubtract,
  }));
  emit('save', payload, advance);
}

const stageStyle = computed(() => ({
  transform: `translate(${tx.value}px, ${ty.value}px) scale(${scale.value})`,
  transformOrigin: '0 0',
}));

const draftPreview = computed<Point[]>(() => {
  if (tool.value !== 'draw' || !hoverNorm.value || draft.value.length === 0) return draft.value;
  return [...draft.value, hoverNorm.value];
});

const regionCount = computed(() => regions.value.filter((r) => !r.isSubtract).length);
const subtractCount = computed(() => regions.value.filter((r) => r.isSubtract).length);

// Vertex markers are SVG circles in a 0-100 viewBox stretched over the image and
// then scaled by the stage transform, so their on-screen size would grow with
// zoom. Counter that by deriving rx/ry (per axis, since aspect may differ) that
// resolve to a constant ~4.5px radius regardless of zoom / image size.
const MARKER_PX = 4.5;
const markerRx = computed(() =>
  nat.value.w ? (100 * MARKER_PX) / (nat.value.w * scale.value) : 0.6,
);
const markerRy = computed(() =>
  nat.value.h ? (100 * MARKER_PX) / (nat.value.h * scale.value) : 0.6,
);
</script>

<template>
  <div class="fixed inset-0 z-50 flex flex-col bg-background/95 backdrop-blur">
    <!-- Top bar -->
    <div class="flex items-center justify-between border-b border-border px-4 py-2.5">
      <div class="flex items-center gap-3 min-w-0">
        <span class="text-sm font-semibold truncate">{{ imageName || 'Annotate image' }}</span>
        <span class="text-xs text-muted-foreground">{{ regionCount }} regions · {{ subtractCount }} holes</span>
      </div>

      <!-- Image navigation -->
      <div v-if="total && total > 1" class="flex items-center gap-1.5">
        <button
          class="inline-flex items-center justify-center rounded-md border border-border p-1.5 text-muted-foreground hover:bg-accent disabled:opacity-40"
          title="上一张 ( [ )" :disabled="!hasPrev" @click="requestNavigate(-1)"
        ><ChevronLeft class="h-4 w-4" /></button>
        <span class="min-w-[3.5rem] text-center text-xs tabular-nums text-muted-foreground">{{ index ?? '?' }} / {{ total }}</span>
        <button
          class="inline-flex items-center justify-center rounded-md border border-border p-1.5 text-muted-foreground hover:bg-accent disabled:opacity-40"
          title="下一张 ( ] )" :disabled="!hasNext" @click="requestNavigate(1)"
        ><ChevronRight class="h-4 w-4" /></button>
      </div>

      <div class="flex items-center gap-2">
        <button
          class="inline-flex items-center gap-1.5 rounded-md border border-border px-3 py-1.5 text-sm hover:bg-accent"
          @click="emit('cancel')"
        >
          <X class="h-4 w-4" /> Cancel
        </button>
        <button
          v-if="hasNext"
          class="inline-flex items-center gap-1.5 rounded-md border border-border px-3 py-1.5 text-sm font-medium hover:bg-accent disabled:opacity-50"
          :disabled="!canSave || saving"
          title="保存并切到下一张"
          @click="onSave(true)"
        >
          <Check class="h-4 w-4" /> 保存并下一张
        </button>
        <button
          class="inline-flex items-center gap-1.5 rounded-md bg-primary px-3 py-1.5 text-sm font-medium text-primary-foreground disabled:opacity-50"
          :disabled="!canSave || saving"
          @click="onSave(false)"
        >
          <Check class="h-4 w-4" /> {{ saving ? 'Saving…' : 'Save & crop' }}
        </button>
      </div>
    </div>

    <div class="flex flex-1 min-h-0">
      <!-- Canvas -->
      <div class="relative flex-1 min-w-0">
        <!-- Toolbar -->
        <div class="absolute left-3 top-3 z-10 flex flex-col gap-1 rounded-lg border border-border bg-card/90 p-1 shadow">
          <button
            class="rounded-md p-2 hover:bg-accent" :class="tool === 'draw' ? 'bg-accent text-foreground' : 'text-muted-foreground'"
            title="Draw polygon (B)" @click="tool = 'draw'"
          ><MousePointer2 class="h-4 w-4" /></button>
          <button
            class="rounded-md p-2 hover:bg-accent" :class="tool === 'wand' ? 'bg-accent text-foreground' : 'text-muted-foreground'"
            title="Magic wand (W) — 点击相近颜色区域自动勾勒" @click="tool = 'wand'"
          ><Wand2 class="h-4 w-4" /></button>
          <button
            class="rounded-md p-2 hover:bg-accent" :class="tool === 'pan' ? 'bg-accent text-foreground' : 'text-muted-foreground'"
            title="Pan (or hold Space)" @click="tool = 'pan'"
          ><Hand class="h-4 w-4" /></button>
          <div class="my-0.5 border-t border-border" />
          <button class="rounded-md p-2 text-muted-foreground hover:bg-accent" title="Zoom in" @click="applyZoom(1.2)"><ZoomIn class="h-4 w-4" /></button>
          <button class="rounded-md p-2 text-muted-foreground hover:bg-accent" title="Zoom out" @click="applyZoom(1 / 1.2)"><ZoomOut class="h-4 w-4" /></button>
          <button class="rounded-md p-2 text-muted-foreground hover:bg-accent" title="Reset view" @click="resetView"><RotateCcw class="h-4 w-4" /></button>
          <div class="my-0.5 border-t border-border" />
          <button class="rounded-md p-2 text-muted-foreground hover:bg-accent" title="Undo (Ctrl+Z)" @click="undo"><Undo2 class="h-4 w-4" /></button>
          <button class="rounded-md p-2 text-muted-foreground hover:bg-accent" title="Redo (Ctrl+Shift+Z)" @click="redo"><Redo2 class="h-4 w-4" /></button>
        </div>

        <!-- Magic-wand tolerance -->
        <div
          v-if="tool === 'wand'"
          class="absolute left-16 top-3 z-10 w-56 rounded-lg border border-border bg-card/95 p-3 shadow"
        >
          <div class="mb-1.5 flex items-center justify-between text-xs">
            <span class="inline-flex items-center gap-1 font-medium text-foreground"><Sparkles class="h-3.5 w-3.5 text-amber-400" /> 魔术棒容差</span>
            <span class="tabular-nums text-muted-foreground">{{ wandTolerance }}</span>
          </div>
          <input
            v-model.number="wandTolerance"
            type="range" min="2" max="120" step="1"
            class="w-full accent-primary"
          />
          <p class="mt-1.5 text-[11px] leading-snug text-muted-foreground">
            点击图片中颜色相近的区域，自动生成多边形。容差越大，选区越宽松。
          </p>
        </div>

        <div
          ref="viewport"
          class="h-full w-full overflow-hidden bg-[#0b0b0d]"
          :class="[
            tool === 'pan' || spaceHeld ? 'cursor-grab' : 'cursor-crosshair',
            wandBusy ? 'cursor-wait' : '',
          ]"
          @wheel="onWheel"
          @pointerdown="onPointerDown"
          @pointermove="onPointerMove"
          @pointerup="onPointerUp"
          @contextmenu.prevent="finishDraft"
          @dblclick.prevent="finishDraft"
        >
          <div class="absolute left-0 top-0" :style="stageStyle">
            <img
              ref="imgEl"
              :src="imageUrl"
              alt="annotation target"
              draggable="false"
              class="block max-w-none select-none"
              @load="onImgLoad"
            />
            <AnnotationOverlay
              :regions="overlayRegions"
              :color-for-class="overlayColorForClass"
              :label-for-class="labelForClass"
              :zoom="scale"
              :legend-scale="annotatorLegendScale"
              :image-width="nat.w"
              :image-height="nat.h"
              :stroke-width="0.004"
            />
            <svg
              class="pointer-events-none absolute inset-0 h-full w-full"
              viewBox="0 0 100 100"
              preserveAspectRatio="none"
            >
              <!-- draft -->
              <polyline
                v-if="draftPreview.length > 1"
                :points="toSvgPoints(draftPreview)"
                fill="none"
                :stroke="subtractMode ? '#94a3b8' : colorFor(currentClassId, false)"
                stroke-width="0.5"
                vector-effect="non-scaling-stroke"
              />
              <ellipse
                v-for="(p, i) in draft"
                :key="i"
                :cx="p[0] * 100"
                :cy="p[1] * 100"
                :rx="i === 0 ? markerRx * 1.35 : markerRx"
                :ry="i === 0 ? markerRy * 1.35 : markerRy"
                :fill="subtractMode ? '#94a3b8' : colorFor(currentClassId, false)"
                :stroke="i === 0 ? '#fafafa' : '#0b0b0d'"
                stroke-width="1.25"
                vector-effect="non-scaling-stroke"
              />
            </svg>
          </div>
        </div>

        <div class="absolute bottom-3 left-1/2 -translate-x-1/2 rounded-md border border-border bg-card/90 px-3 py-1.5 text-xs text-muted-foreground shadow">
          <template v-if="tool === 'wand'">魔术棒：点击颜色相近区域自动勾勒 · 调节容差 · B 切回画笔 · 滚轮缩放 · 空格平移</template>
          <template v-else>点击打点 · 双击/右键/Enter 闭合 · Z 撤销点 · W 魔术棒 · 滚轮缩放 · 空格平移<template v-if="total && total > 1"> · [ ] 切换图片</template></template>
        </div>
      </div>

      <!-- Right panel -->
      <aside class="flex w-72 flex-col border-l border-border">
        <div class="border-b border-border p-3">
          <div class="mb-2 text-xs font-medium uppercase tracking-wide text-muted-foreground">Region type</div>
          <div class="flex gap-1.5">
            <button
              class="flex-1 rounded-md border px-2 py-1.5 text-sm"
              :class="!subtractMode ? 'border-primary bg-accent' : 'border-border text-muted-foreground'"
              @click="subtractMode = false"
            >Defect</button>
            <button
              class="flex-1 inline-flex items-center justify-center gap-1 rounded-md border px-2 py-1.5 text-sm"
              :class="subtractMode ? 'border-primary bg-accent' : 'border-border text-muted-foreground'"
              @click="subtractMode = true"
            ><Minus class="h-3.5 w-3.5" /> Hole</button>
          </div>
        </div>

        <div class="border-b border-border p-3">
          <div class="mb-2 text-xs font-medium uppercase tracking-wide text-muted-foreground">Defect class</div>
          <div class="space-y-1.5">
            <button
              v-for="(c, i) in classes"
              :key="c.id"
              class="flex w-full items-center gap-2 rounded-md border px-2 py-1.5 text-left text-sm"
              :class="currentClassId === c.id && !subtractMode ? 'border-primary bg-accent' : 'border-border hover:bg-accent/50'"
              :disabled="subtractMode"
              @click="currentClassId = c.id"
            >
              <span class="h-3 w-3 shrink-0 rounded-full" :style="{backgroundColor: classColor(c, i)}" />
              <span class="truncate">{{ c.name }}</span>
            </button>
            <p v-if="!classes.length" class="text-xs text-muted-foreground">No classes yet. Create one below.</p>
          </div>
          <form v-if="createClass" class="mt-2 flex gap-1.5" @submit.prevent="addClass">
            <input
              v-model="newClassName"
              placeholder="New class name"
              class="min-w-0 flex-1 rounded-md border border-input bg-background px-2 py-1.5 text-sm outline-none focus:ring-1 focus:ring-ring"
            />
            <button
              type="submit"
              class="inline-flex items-center rounded-md border border-border px-2 hover:bg-accent disabled:opacity-50"
              :disabled="!newClassName.trim() || creatingClass"
            ><Plus class="h-4 w-4" /></button>
          </form>
        </div>

        <div class="flex items-center justify-between px-3 pt-3">
          <span class="text-xs font-medium uppercase tracking-wide text-muted-foreground">Regions ({{ regions.length }})</span>
          <button class="text-xs text-destructive hover:underline" @click="clearAll">Clear all</button>
        </div>
        <div class="flex-1 space-y-1.5 overflow-y-auto p-3">
          <div
            v-for="(r, idx) in regions"
            :key="r.id"
            class="flex items-center gap-2 rounded-md border border-border px-2 py-1.5 text-sm"
            @mouseenter="selectedRegionId = r.id"
            @mouseleave="selectedRegionId = null"
          >
            <span class="h-3 w-3 shrink-0 rounded-full" :style="{backgroundColor: colorFor(r.classId, r.isSubtract)}" />
            <template v-if="r.isSubtract">
              <span class="flex-1 text-muted-foreground">Hole #{{ idx + 1 }}</span>
            </template>
            <template v-else>
              <select
                class="min-w-0 flex-1 rounded border border-input bg-background px-1 py-0.5 text-xs outline-none"
                :value="r.classId ?? ''"
                @change="setRegionClass(r.id, ($event.target as HTMLSelectElement).value)"
              >
                <option v-for="c in classes" :key="c.id" :value="c.id">{{ c.name }}</option>
              </select>
            </template>
            <button class="text-muted-foreground hover:text-destructive" title="Delete" @click="deleteRegion(r.id)">
              <Trash2 class="h-3.5 w-3.5" />
            </button>
          </div>
          <p v-if="!regions.length" class="text-xs text-muted-foreground">Draw a polygon on the image to create a region.</p>
        </div>
      </aside>
    </div>
  </div>
</template>

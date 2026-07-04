<script lang="ts">
export type PlotPoint = {
  id: string;
  x: number;
  y: number;
  cluster: number;
  url?: string;
  anomalyScore?: number;
  label?: string;
  isGolden?: boolean;
  sourceImageId?: string;
  instanceIndex?: number;
  cropAnnotation?: {points: [number, number][]; isSubtract: boolean}[] | null;
};
</script>

<script setup lang="ts">
import {computed, onBeforeUnmount, ref, shallowRef, watch} from 'vue';
import VChart from 'vue-echarts';
import type {ECharts} from 'echarts';
import {categoryChartColor, staticUrl} from '@/lib/api';

const props = defineProps<{
  plotData: PlotPoint[];
  labelList: string[];
  highlightLabels?: string[];
  /** Base scatter symbol size in px; golden/dimmed scale proportionally. */
  pointSize?: number;
}>();
const emit = defineEmits<{preview: [point: PlotPoint]}>();

type ChartPublicApi = {
  chart?: ECharts;
  getDom?: ECharts['getDom'];
  convertFromPixel?: ECharts['convertFromPixel'];
};

type Viewport = {xMin: number; xMax: number; yMin: number; yMax: number};

const chartRef = shallowRef<ChartPublicApi | null>(null);
const isPanning = ref(false);
const isOverPoint = ref(false);
const viewport = ref<Viewport | null>(null);
const plotSize = ref({width: 1, height: 1});

const DRAG_THRESHOLD = 4;
const ZOOM_IN_FACTOR = 0.85;
const ZOOM_OUT_FACTOR = 1.18;
const MIN_ZOOM_RATIO = 0.02;
const MAX_ZOOM_OUT_RATIO = 1;
const GRID_MARGIN = {left: 40, right: 24, top: 24, bottom: 40};
const TARGET_SPLITS = 8;
const LARGE_SCATTER_THRESHOLD = 200;

const AXIS_COMMON = {
  splitNumber: TARGET_SPLITS,
  splitLine: {show: true, lineStyle: {color: '#3f3f46', type: 'dashed' as const}},
  axisLine: {show: false, onZero: false},
  axisTick: {show: false},
  axisLabel: {show: false},
};

let _panActive = false;
let _didDrag = false;
let _suppressClick = false;
let _userZoomed = false;
let _dragStartClient = {x: 0, y: 0};
let _dragStartViewport: Viewport | null = null;
let _pendingViewport: Viewport | null = null;
let _viewportRaf = 0;
let _interactionEndTimer: ReturnType<typeof setTimeout> | null = null;

function getChart(): ECharts | null {
  const root = chartRef.value;
  return root?.chart ?? null;
}

function setChartCursor(cursor: 'default' | 'grab' | 'grabbing') {
  const dom = chartRef.value?.getDom?.() ?? getChart()?.getDom();
  if (dom) dom.style.cursor = cursor;
}

function getPlotSize() {
  const dom = chartRef.value?.getDom?.() ?? getChart()?.getDom();
  if (!dom) return {width: 1, height: 1};
  return {
    width: Math.max(dom.clientWidth - GRID_MARGIN.left - GRID_MARGIN.right, 1),
    height: Math.max(dom.clientHeight - GRID_MARGIN.top - GRID_MARGIN.bottom, 1),
  };
}

function updatePlotSize() {
  plotSize.value = getPlotSize();
}

function niceInterval(range: number, splits = TARGET_SPLITS) {
  const raw = range / splits;
  if (!Number.isFinite(raw) || raw <= 0) return 1;
  const mag = 10 ** Math.floor(Math.log10(raw));
  const norm = raw / mag;
  let nice = 10;
  if (norm <= 1) nice = 1;
  else if (norm <= 2) nice = 2;
  else if (norm <= 5) nice = 5;
  return nice * mag;
}

const dataBounds = computed<Viewport>(() => {
  const xs = props.plotData.map((p) => p.x).filter(Number.isFinite);
  const ys = props.plotData.map((p) => p.y).filter(Number.isFinite);
  if (!xs.length || !ys.length) {
    return {xMin: -1, xMax: 1, yMin: -1, yMax: 1};
  }
  const xMinRaw = Math.min(...xs);
  const xMaxRaw = Math.max(...xs);
  const yMinRaw = Math.min(...ys);
  const yMaxRaw = Math.max(...ys);
  const xPad = Math.max((xMaxRaw - xMinRaw) * 0.08, 1e-6);
  const yPad = Math.max((yMaxRaw - yMinRaw) * 0.08, 1e-6);
  return {
    xMin: xMinRaw - xPad,
    xMax: xMaxRaw + xPad,
    yMin: yMinRaw - yPad,
    yMax: yMaxRaw + yPad,
  };
});

function gridIntervalsFor(vp: Viewport) {
  const xRange = vp.xMax - vp.xMin;
  const yRange = vp.yMax - vp.yMin;
  const {width, height} = plotSize.value;
  const intervalX = niceInterval(xRange);
  const cellPixels = (intervalX / xRange) * width;
  const intervalY = (cellPixels / height) * yRange;
  return {x: intervalX, y: intervalY};
}

function clampViewport(next: Viewport): Viewport {
  const bounds = dataBounds.value;
  const baseXRange = bounds.xMax - bounds.xMin;
  const baseYRange = bounds.yMax - bounds.yMin;
  const minXRange = baseXRange * MIN_ZOOM_RATIO;
  const minYRange = baseYRange * MIN_ZOOM_RATIO;
  const maxXRange = baseXRange * MAX_ZOOM_OUT_RATIO;
  const maxYRange = baseYRange * MAX_ZOOM_OUT_RATIO;

  let xRange = next.xMax - next.xMin;
  let yRange = next.yMax - next.yMin;
  const xCenter = (next.xMin + next.xMax) / 2;
  const yCenter = (next.yMin + next.yMax) / 2;

  xRange = Math.min(Math.max(xRange, minXRange), maxXRange);
  yRange = Math.min(Math.max(yRange, minYRange), maxYRange);

  const xMinLimit = bounds.xMin;
  const xMaxLimit = bounds.xMax;
  const yMinLimit = bounds.yMin;
  const yMaxLimit = bounds.yMax;

  let xMin = xCenter - xRange / 2;
  let xMax = xCenter + xRange / 2;
  let yMin = yCenter - yRange / 2;
  let yMax = yCenter + yRange / 2;

  if (xMin < xMinLimit) {
    xMax += xMinLimit - xMin;
    xMin = xMinLimit;
  }
  if (xMax > xMaxLimit) {
    xMin -= xMax - xMaxLimit;
    xMax = xMaxLimit;
  }
  if (yMin < yMinLimit) {
    yMax += yMinLimit - yMin;
    yMin = yMinLimit;
  }
  if (yMax > yMaxLimit) {
    yMin -= yMax - yMaxLimit;
    yMax = yMaxLimit;
  }

  return {xMin, xMax, yMin, yMax};
}

function buildAxisOption(vp: Viewport, lite = false) {
  const intervals = gridIntervalsFor(vp);
  const splitLine = lite
    ? {show: false}
    : AXIS_COMMON.splitLine;
  return {
    xAxis: {
      type: 'value' as const,
      min: vp.xMin,
      max: vp.xMax,
      interval: intervals.x,
      minInterval: intervals.x,
      maxInterval: intervals.x,
      ...AXIS_COMMON,
      splitLine,
    },
    yAxis: {
      type: 'value' as const,
      min: vp.yMin,
      max: vp.yMax,
      interval: intervals.y,
      minInterval: intervals.y,
      maxInterval: intervals.y,
      ...AXIS_COMMON,
      splitLine,
    },
  };
}

function applyViewportToChart(vp: Viewport, lite = false) {
  const chart = getChart();
  if (!chart) return;
  chart.setOption(buildAxisOption(vp, lite), {lazyUpdate: true, replaceMerge: ['xAxis', 'yAxis']});
}

function flushViewportUpdate(lite = false) {
  if (!_pendingViewport) return;
  const vp = _pendingViewport;
  _pendingViewport = null;
  viewport.value = vp;
  applyViewportToChart(vp, lite);
}

function scheduleViewportUpdate(next: Viewport, lite = false) {
  _pendingViewport = clampViewport(next);
  if (_viewportRaf) return;
  _viewportRaf = requestAnimationFrame(() => {
    _viewportRaf = 0;
    flushViewportUpdate(lite);
  });
}

function markInteractionEnd() {
  if (_interactionEndTimer) clearTimeout(_interactionEndTimer);
  _interactionEndTimer = setTimeout(() => {
    _interactionEndTimer = null;
    if (viewport.value) applyViewportToChart(viewport.value, false);
  }, 120);
}

function resetViewport() {
  _userZoomed = false;
  _pendingViewport = null;
  const bounds = {...dataBounds.value};
  viewport.value = bounds;
  applyViewportToChart(bounds, false);
}

watch(
  () => props.plotData,
  () => {
    _userZoomed = false;
    updatePlotSize();
    const bounds = {...dataBounds.value};
    viewport.value = bounds;
    requestAnimationFrame(() => applyViewportToChart(bounds, false));
  },
  {immediate: true},
);

watch(
  () => [props.highlightLabels, props.pointSize, props.labelList.length] as const,
  () => {
    requestAnimationFrame(() => {
      if (viewport.value) applyViewportToChart(viewport.value, false);
    });
  },
);

let _resizeObserver: ResizeObserver | null = null;

watch(
  () => chartRef.value?.getDom?.() ?? getChart()?.getDom(),
  (dom, _, onCleanup) => {
    _resizeObserver?.disconnect();
    _resizeObserver = null;
    if (!dom) return;
    updatePlotSize();
    _resizeObserver = new ResizeObserver(() => {
      updatePlotSize();
      if (!_userZoomed && viewport.value) applyViewportToChart(viewport.value, false);
    });
    _resizeObserver.observe(dom);
    if (viewport.value) applyViewportToChart(viewport.value, false);
    onCleanup(() => {
      _resizeObserver?.disconnect();
      _resizeObserver = null;
    });
  },
  {flush: 'post'},
);

watch(
  dataBounds,
  (bounds) => {
    if (!_userZoomed) {
      viewport.value = {...bounds};
      applyViewportToChart(bounds, false);
    }
  },
);

function cleanupPanListeners() {
  window.removeEventListener('pointermove', onWindowPointermove);
  window.removeEventListener('pointerup', onWindowPointerup);
  window.removeEventListener('pointercancel', onWindowPointerup);
  _panActive = false;
  isPanning.value = false;
}

function onChartMouseover(params: {componentType?: string; seriesType?: string}) {
  if (params.componentType !== 'series' || params.seriesType !== 'scatter') return;
  isOverPoint.value = true;
  if (!_panActive && !isPanning.value) setChartCursor('default');
}

function onChartMouseout(params: {componentType?: string}) {
  if (params.componentType !== 'series') return;
  isOverPoint.value = false;
  if (!_panActive && !isPanning.value) setChartCursor('grab');
}

function onPointerdown(e: PointerEvent) {
  if (e.button !== 0 || !viewport.value) return;

  _panActive = true;
  _didDrag = false;
  _dragStartClient = {x: e.clientX, y: e.clientY};
  _dragStartViewport = {...(_pendingViewport ?? viewport.value)};

  window.addEventListener('pointermove', onWindowPointermove);
  window.addEventListener('pointerup', onWindowPointerup);
  window.addEventListener('pointercancel', onWindowPointerup);
}

function onWindowPointermove(e: PointerEvent) {
  onPanPointermove(e);
}

function onWindowPointerup() {
  finishPan();
}

function onPanPointermove(e: PointerEvent) {
  if (!_panActive || !_dragStartViewport) return;

  const dx = e.clientX - _dragStartClient.x;
  const dy = e.clientY - _dragStartClient.y;
  if (!_didDrag && Math.hypot(dx, dy) < DRAG_THRESHOLD) return;

  _didDrag = true;
  _userZoomed = true;
  isPanning.value = true;
  isOverPoint.value = false;
  setChartCursor('grabbing');

  const w = plotSize.value.width;
  const h = plotSize.value.height;
  const start = _dragStartViewport;
  const xRange = start.xMax - start.xMin;
  const yRange = start.yMax - start.yMin;
  const xShift = -(dx / w) * xRange;
  const yShift = (dy / h) * yRange;
  scheduleViewportUpdate(
    {
      xMin: start.xMin + xShift,
      xMax: start.xMax + xShift,
      yMin: start.yMin + yShift,
      yMax: start.yMax + yShift,
    },
    true,
  );
  markInteractionEnd();
}

function onWheel(e: WheelEvent) {
  const current = _pendingViewport ?? viewport.value;
  if (!current) return;
  const factor = e.deltaY < 0 ? ZOOM_IN_FACTOR : ZOOM_OUT_FACTOR;
  const dom = chartRef.value?.getDom?.() ?? getChart()?.getDom();
  const rect = dom?.getBoundingClientRect();
  const px = rect ? (e.clientX - rect.left) / Math.max(rect.width, 1) : 0.5;
  const py = rect ? (e.clientY - rect.top) / Math.max(rect.height, 1) : 0.5;
  const centerX = current.xMin + px * (current.xMax - current.xMin);
  const centerY = current.yMax - py * (current.yMax - current.yMin);
  const nextXRange = (current.xMax - current.xMin) * factor;
  const nextYRange = (current.yMax - current.yMin) * factor;
  _userZoomed = true;
  scheduleViewportUpdate(
    {
      xMin: centerX - px * nextXRange,
      xMax: centerX + (1 - px) * nextXRange,
      yMin: centerY - (1 - py) * nextYRange,
      yMax: centerY + py * nextYRange,
    },
    true,
  );
  markInteractionEnd();
}

function finishPan() {
  if (_didDrag) _suppressClick = true;
  _dragStartViewport = null;
  cleanupPanListeners();
  markInteractionEnd();
  if (!isOverPoint.value) setChartCursor('grab');
}

onBeforeUnmount(() => {
  cleanupPanListeners();
  _resizeObserver?.disconnect();
  _resizeObserver = null;
  if (_viewportRaf) cancelAnimationFrame(_viewportRaf);
  if (_interactionEndTimer) clearTimeout(_interactionEndTimer);
});

function tooltipFormatter(params: unknown) {
  const p = params as {data?: {raw?: PlotPoint}};
  const d = p.data?.raw;
  if (!d) return '';
  const color = categoryChartColor(d.cluster, props.labelList.length || props.plotData.length);
  const label = d.label || props.labelList[d.cluster] || `Cluster ${d.cluster + 1}`;
  const score = Number(d.anomalyScore ?? 0);
  const scoreColor = score > 70 ? '#f43f5e' : '#10b981';
  const img = d.url
    ? `<img src="${staticUrl(d.url)}" alt="" style="width:128px;height:128px;object-fit:cover;border-radius:4px;border:1px solid #27272a" />`
    : '';
  const goldenBadge = d.isGolden
    ? '<span style="font-size:8px;font-weight:600;color:#fbbf24;border:1px solid #fbbf24;border-radius:4px;padding:0 4px">◆ GOLDEN</span>'
    : '';
  return `
    <div style="display:flex;flex-direction:column;gap:8px;min-width:150px;padding:4px">
      <div>${img}</div>
      <div style="display:flex;align-items:center;gap:8px">
        <span style="width:8px;height:8px;border-radius:9999px;background:${color}"></span>
        <span style="font-size:10px;font-weight:500">${label}</span>
        ${goldenBadge}
      </div>
      <div style="font-size:8px;color:#a1a1aa;display:flex;justify-content:space-between">
        <span>Anomaly Score</span>
        <span style="font-family:monospace;color:${scoreColor}">${score.toFixed(1)}%</span>
      </div>
      <div style="height:4px;width:100%;background:rgba(39,39,42,0.3);border-radius:9999px;overflow:hidden">
        <div style="height:100%;width:${Math.min(100, score)}%;background:${scoreColor}"></div>
      </div>
      <p style="font-size:8px;color:#a1a1aa;text-align:center;border-top:1px solid #27272a;padding-top:4px;margin:0">Click to expand image</p>
    </div>`;
}

// Series/tooltip rebuild only when data or styling changes — not on zoom/pan.
const chartOption = computed(() => {
  const clusters = [...new Set(props.plotData.map((p) => p.cluster))].sort((a, b) => a - b);
  const n = props.labelList.length || clusters.length;
  const hl = props.highlightLabels?.length ? new Set(props.highlightLabels) : null;
  const base = props.pointSize ?? 9;
  const dimmedSize = Math.max(2, Math.round(base * (6 / 9)));
  const goldenSize = Math.max(4, Math.round(base * (15 / 9)));
  const useLarge = props.plotData.length >= LARGE_SCATTER_THRESHOLD;
  const vp = dataBounds.value;

  const series = clusters.map((ci) => {
    const pts = props.plotData.filter((p) => p.cluster === ci);
    const label = pts[0]?.label || props.labelList[ci] || `Cluster ${ci + 1}`;
    const color = categoryChartColor(ci, n);
    const dimmed = hl != null && !hl.has(label);
    return {
      name: label,
      type: 'scatter',
      large: useLarge,
      largeThreshold: LARGE_SCATTER_THRESHOLD,
      symbolSize: dimmed ? dimmedSize : base,
      itemStyle: {color, borderColor: 'transparent', opacity: dimmed ? 0.12 : 1},
      z: dimmed ? 1 : 5,
      data: pts.map((p) => ({
        value: [p.x, p.y],
        raw: p,
        ...(p.isGolden
          ? {
              symbol: 'diamond',
              symbolSize: goldenSize,
              itemStyle: {color, borderColor: '#fbbf24', borderWidth: 2.5},
            }
          : {}),
      })),
    };
  });

  return {
    backgroundColor: 'transparent',
    animation: false,
    grid: {...GRID_MARGIN, containLabel: false},
    ...buildAxisOption(vp, false),
    tooltip: {
      trigger: 'item',
      backgroundColor: '#09090b',
      borderColor: '#27272a',
      textStyle: {color: '#fafafa', fontSize: 10},
      extraCssText: 'max-width:280px',
      formatter: tooltipFormatter,
    },
    series,
  };
});

function onChartClick(params: unknown) {
  if (_suppressClick) {
    _suppressClick = false;
    return;
  }
  const raw = (params as {data?: {raw?: PlotPoint}}).data?.raw;
  if (raw?.url) emit('preview', raw);
}
</script>

<template>
  <div
    class="inference-scatter-chart relative h-full min-h-[400px] w-full"
    :class="{'is-panning': isPanning, 'is-over-point': isOverPoint && !isPanning}"
    @pointerdown="onPointerdown"
    @wheel.prevent="onWheel"
  >
    <button
      type="button"
      class="absolute right-3 top-3 z-10 rounded border border-border bg-background/80 px-2 py-1 text-[10px] text-muted-foreground shadow-sm backdrop-blur transition hover:border-primary/50 hover:text-foreground"
      @click.stop="resetViewport"
      @pointerdown.stop
    >
      重置视图
    </button>
    <VChart
      ref="chartRef"
      class="h-full w-full"
      :option="chartOption"
      :update-options="{lazyUpdate: true}"
      autoresize
      @click="onChartClick"
      @mouseover="onChartMouseover"
      @mouseout="onChartMouseout"
    />
  </div>
</template>

<style scoped>
.inference-scatter-chart :deep(canvas) {
  cursor: grab;
}

.inference-scatter-chart {
  touch-action: none;
  user-select: none;
}

.inference-scatter-chart.is-over-point :deep(canvas) {
  cursor: default;
}

.inference-scatter-chart.is-panning :deep(canvas) {
  cursor: grabbing;
}
</style>

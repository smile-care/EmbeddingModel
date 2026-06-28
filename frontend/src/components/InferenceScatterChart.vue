<script lang="ts">
export type PlotPoint = {
  id: string;
  x: number;
  y: number;
  cluster: number;
  url?: string;
  anomalyScore?: number;
  label?: string;
  sourceImageId?: string;
  instanceIndex?: number;
  cropAnnotation?: {points: [number, number][]; isSubtract: boolean}[] | null;
};
</script>

<script setup lang="ts">
import {computed, ref, shallowRef} from 'vue';
import VChart from 'vue-echarts';
import type {ECharts} from 'echarts';
import {staticUrl} from '@/lib/api';

const COLORS = ['#8884d8', '#82ca9d', '#ffc658', '#ff8042', '#0088FE', '#00C49F'];

const props = defineProps<{plotData: PlotPoint[]; labelList: string[]}>();
const emit = defineEmits<{preview: [point: PlotPoint]}>();

// ── chart ref ──────────────────────────────────────────────────────────────
const chartRef = shallowRef<{chart: ECharts} | null>(null);
const isDragging = ref(false);

// ── drag-to-pan (data-coordinate space) ───────────────────────────────────
let _dragStartClient = {x: 0, y: 0};
// Snapshot of the visible data range at drag start
let _dragStartRange = {xMin: 0, xMax: 0, yMin: 0, yMax: 0};
const DRAG_THRESHOLD = 4;
let _didDrag = false;

function getCurrentDataRange() {
  const chart = chartRef.value?.chart;
  if (!chart) return null;
  const opt = chart.getOption() as any;
  // After any dataZoom interaction, echarts stores startValue/endValue
  const dz: any[] = opt?.dataZoom ?? [];
  const xz = dz[0] ?? {};
  const yz = dz[1] ?? {};
  // Fallback to axis min/max when no zoom has been applied yet
  const xAxis = (opt?.xAxis as any[])?.[0] ?? {};
  const yAxis = (opt?.yAxis as any[])?.[0] ?? {};
  return {
    xMin: xz.startValue ?? xAxis.min ?? null,
    xMax: xz.endValue ?? xAxis.max ?? null,
    yMin: yz.startValue ?? yAxis.min ?? null,
    yMax: yz.endValue ?? yAxis.max ?? null,
  };
}

function onMousedown(e: MouseEvent) {
  if (e.button !== 0) return;
  const range = getCurrentDataRange();
  if (!range || range.xMin === null) return;
  _didDrag = false;
  isDragging.value = true;
  _dragStartClient = {x: e.clientX, y: e.clientY};
  _dragStartRange = {xMin: range.xMin, xMax: range.xMax!, yMin: range.yMin!, yMax: range.yMax!};
  window.addEventListener('mousemove', onMousemove);
  window.addEventListener('mouseup', onMouseup);
}

function onMousemove(e: MouseEvent) {
  if (!isDragging.value) return;
  const chart = chartRef.value?.chart;
  if (!chart) return;

  const dx = e.clientX - _dragStartClient.x;
  const dy = e.clientY - _dragStartClient.y;
  if (!_didDrag && Math.hypot(dx, dy) < DRAG_THRESHOLD) return;
  _didDrag = true;

  const dom = chart.getDom();
  const {xMin, xMax, yMin, yMax} = _dragStartRange;
  const xRange = xMax - xMin;
  const yRange = yMax - yMin;

  // Convert pixel delta to data-unit delta
  // grid area is approximately full canvas minus padding; use dom size as proxy
  const xShift = -(dx / dom.clientWidth) * xRange;
  const yShift = (dy / dom.clientHeight) * yRange; // screen-Y is inverted vs data-Y

  chart.dispatchAction({type: 'dataZoom', dataZoomIndex: 0, startValue: xMin + xShift, endValue: xMax + xShift});
  chart.dispatchAction({type: 'dataZoom', dataZoomIndex: 1, startValue: yMin + yShift, endValue: yMax + yShift});
}

function onMouseup() {
  isDragging.value = false;
  _didDrag = false;
  window.removeEventListener('mousemove', onMousemove);
  window.removeEventListener('mouseup', onMouseup);
}

// ── chart option ───────────────────────────────────────────────────────────
const chartOption = computed(() => {
  const clusters = [...new Set(props.plotData.map((p) => p.cluster))].sort((a, b) => a - b);
  const series = clusters.map((ci) => {
    const pts = props.plotData.filter((p) => p.cluster === ci);
    const label = pts[0]?.label || props.labelList[ci] || `Cluster ${ci + 1}`;
    return {
      name: label,
      type: 'scatter',
      symbolSize: 9,
      itemStyle: {color: COLORS[ci % COLORS.length], borderColor: 'transparent'},
      data: pts.map((p) => ({value: [p.x, p.y], raw: p})),
    };
  });

  return {
    backgroundColor: 'transparent',
    animation: false,
    grid: {left: 40, right: 24, top: 24, bottom: 40, containLabel: false},
    xAxis: {type: 'value', splitLine: {lineStyle: {color: '#3f3f46', type: 'dashed'}}, axisLine: {lineStyle: {color: '#3f3f46'}}, axisTick: {show: false}, axisLabel: {show: false}},
    yAxis: {type: 'value', splitLine: {lineStyle: {color: '#3f3f46', type: 'dashed'}}, axisLine: {lineStyle: {color: '#3f3f46'}}, axisTick: {show: false}, axisLabel: {show: false}},
    dataZoom: [
      {type: 'inside', xAxisIndex: 0, zoomOnMouseWheel: true, moveOnMouseMove: false, moveOnMouseWheel: false, filterMode: 'none'},
      {type: 'inside', yAxisIndex: 0, zoomOnMouseWheel: true, moveOnMouseMove: false, moveOnMouseWheel: false, filterMode: 'none'},
    ],
    tooltip: {
      trigger: 'item',
      backgroundColor: '#09090b',
      borderColor: '#27272a',
      textStyle: {color: '#fafafa', fontSize: 10},
      extraCssText: 'max-width:280px',
      formatter: (params: unknown) => {
        const p = params as {data?: {raw?: PlotPoint}};
        const d = p.data?.raw;
        if (!d) return '';
        const color = COLORS[d.cluster % COLORS.length];
        const label = d.label || props.labelList[d.cluster] || `Cluster ${d.cluster + 1}`;
        const score = Number(d.anomalyScore ?? 0);
        const scoreColor = score > 70 ? '#f43f5e' : '#10b981';
        const img = d.url ? `<img src="${staticUrl(d.url)}" alt="" style="width:128px;height:128px;object-fit:cover;border-radius:4px;border:1px solid #27272a" />` : '';
        return `
          <div style="display:flex;flex-direction:column;gap:8px;min-width:150px;padding:4px">
            <div>${img}</div>
            <div style="display:flex;align-items:center;gap:8px">
              <span style="width:8px;height:8px;border-radius:9999px;background:${color}"></span>
              <span style="font-size:10px;font-weight:500">${label}</span>
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
      },
    },
    series,
  };
});

function onChartClick(params: unknown) {
  const raw = (params as {data?: {raw?: PlotPoint}}).data?.raw;
  if (raw?.url) emit('preview', raw);
}
</script>

<template>
  <div
    class="h-full min-h-[400px] w-full"
    :class="isDragging ? 'cursor-grabbing' : 'cursor-grab'"
    @mousedown="onMousedown"
  >
    <VChart ref="chartRef" class="h-full w-full" :option="chartOption" autoresize @click="onChartClick" />
  </div>
</template>

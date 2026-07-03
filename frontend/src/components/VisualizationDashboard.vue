<script setup lang="ts">
import {computed, onMounted, onUnmounted, ref, watch} from 'vue';
import {useRouter} from 'vue-router';
import VChart from 'vue-echarts';
import {AlertCircle, ArrowRight, CheckCircle2, Clock, Database, Layers, OctagonX, Tag, X} from 'lucide-vue-next';
import {ExperimentsApi} from '@/lib/api';

const props = defineProps<{experimentId: string}>();
const router = useRouter();

interface ExperimentDetail {
  id: string;
  name: string;
  model: string;
  dataset: string;
  status: string;
  duration: string | null;
  accuracy: string | null;
  checkpointPath: string | null;
  sampleCount: number;
  trainSampleCount: number;
  valSampleCount: number;
  config: Record<string, any> | null;
  metrics: Record<string, any> | null;
  runStatus: string | null;
  runProgress: number;
  runMetrics: Record<string, any> | null;
}

const exp = ref<ExperimentDetail | null>(null);
const loading = ref(true);
const fetchError = ref<string | null>(null);
const stopping = ref(false);
const retraining = ref(false);

// ── Polling ───────────────────────────────────────────────────────────────────
let pollTimer: ReturnType<typeof setTimeout> | null = null;

async function fetchExp() {
  try {
    exp.value = await ExperimentsApi.get(props.experimentId);
  } catch (e) {
    fetchError.value = String(e);
  } finally {
    loading.value = false;
  }
}

function schedulePoll() {
  pollTimer = setTimeout(async () => {
    await fetchExp();
    // Keep polling while training/preparing; stop once terminal state reached
    if (isActive.value) schedulePoll();
    else stopping.value = false;
  }, 2000);
}

onMounted(async () => {
  await fetchExp();
  if (isActive.value) schedulePoll();
});

onUnmounted(() => {
  if (pollTimer) clearTimeout(pollTimer);
});

// ── Result vs. live-run separation ─────────────────────────────────────────────
// `metrics`/`status` = last *successful* result; `runMetrics`/`runStatus` = the
// most recent training attempt. The dashboard body always renders the last-good
// result; the current attempt (running / stopped / failed) is shown as a banner.
const runStatus = computed(() => exp.value?.runStatus ?? null);
const runMetrics = computed<Record<string, any> | null>(() => exp.value?.runMetrics ?? null);
const runActive = computed(() => runStatus.value === 'Running');
const runEnded = computed(() => runStatus.value === 'Failed' || runStatus.value === 'Stopped');
// A usable last-good result exists if `metrics` holds a completed/stopped payload
// (with summary). Metrics-based (not status-based) so legacy rows render too, and
// so a failed/stopped re-train — which never overwrites `metrics` — still shows it.
const hasResult = computed(() => {
  const m = exp.value?.metrics;
  return !!m && (m.stage === 'completed' || m.stage === 'stopped' || !!m.summary);
});
const runError = computed<string | null>(() => runMetrics.value?.error ?? null);

// Keep polling while a training attempt is in flight.
const isActive = computed(() => runActive.value);

// Banner (run notice) dismissal — re-shown whenever the run status changes.
const noticeDismissed = ref(false);

// Top-level render mode.
//  - 进行中（含重训）→ 'live'：实时训练视图，实时展示 loss/margin 曲线；
//  - 已结束且本次失败/中止 → 'result'：回退显示上一次成功结果 + 顶部提示横幅；
//  - 从未成功 → 首训失败/中止卡片或空态。
const mode = computed<'live' | 'failedFirst' | 'stoppedFirst' | 'result' | 'empty'>(() => {
  if (runActive.value) return 'live';
  if (hasResult.value) return 'result';
  if (runStatus.value === 'Failed') return 'failedFirst';
  if (runStatus.value === 'Stopped') return 'stoppedFirst';
  return 'empty';
});
const showRunBanner = computed(
  () => mode.value === 'result' && runEnded.value && !noticeDismissed.value,
);

// Re-show the run notice whenever the attempt transitions to a new state.
watch(runStatus, () => {
  noticeDismissed.value = false;
});

// ── Derived state (dashboard body) ──────────────────────────────────────────────
// 进行中渲染实时运行数据（含实时曲线）；否则渲染上一次成功结果；都没有则用本次运行数据。
const bodyMetrics = computed<Record<string, any> | null>(() => {
  if (runActive.value) return runMetrics.value;
  if (hasResult.value) return exp.value?.metrics ?? null;
  return runMetrics.value;
});
const stage = computed(() => bodyMetrics.value?.stage ?? null);
const summary = computed(() => bodyMetrics.value?.summary ?? null);
const config = computed(() => exp.value?.config ?? null);
const categories = computed<string[]>(() => config.value?.resolvedCategoryNames ?? []);

// Live series — available during training in liveSeries,
// and after completion in summary.liveSeries
const liveSeries = computed(
  () => bodyMetrics.value?.liveSeries ?? bodyMetrics.value?.summary?.liveSeries ?? null,
);

// Final series (available after completion/stopped). Training loss and
// validation now share the same checkpoint cadence (see trainer.py), so all
// series below are always the same length and already aligned to `steps` —
// no sparse/null alignment needed.
const trainLosses = computed<number[]>(() =>
  liveSeries.value?.trainLosses ?? summary.value?.train_losses ?? [],
);
const valLosses = computed<number[]>(() =>
  liveSeries.value?.valLosses ?? summary.value?.val_losses ?? [],
);
const steps = computed<number[]>(() => {
  const recorded = liveSeries.value?.steps ?? summary.value?.steps ?? [];
  if (recorded.length === trainLosses.value.length && recorded.length > 0) {
    return recorded;
  }
  // 旧数据无 steps 时回退为连续编号（与历史行为一致）
  return trainLosses.value.map((_, i) => i + 1);
});
const liveMargins = computed<number[]>(() => liveSeries.value?.margins ?? []);
const livePosSims = computed<number[]>(() => liveSeries.value?.posSims ?? []);
const liveNegSims = computed<number[]>(() => liveSeries.value?.negSims ?? []);

const lastValMetrics = computed(() =>
  bodyMetrics.value?.val ?? summary.value?.last_val_metrics ?? null,
);
const lastTrainMetrics = computed(() =>
  bodyMetrics.value?.train ?? summary.value?.last_train_metrics ?? null,
);

function getNumericMetric(source: Record<string, any> | null | undefined, ...keys: string[]) {
  for (const key of keys) {
    const raw = source?.[key];
    if (raw == null) continue;
    const value = typeof raw === 'number' ? raw : Number(raw);
    if (Number.isFinite(value)) return value;
  }
  return null;
}

const valMargin = computed(() => {
  const v = getNumericMetric(lastValMetrics.value, 'Margin', 'margin');
  return v != null ? v.toFixed(4) : null;
});

const posSim = computed(() => {
  const v = getNumericMetric(lastValMetrics.value, 'PosSim', 'margin_pos_sim');
  return v != null ? v.toFixed(4) : null;
});

const negSim = computed(() => {
  const v = getNumericMetric(lastValMetrics.value, 'NegSim', 'margin_neg_sim');
  return v != null ? v.toFixed(4) : null;
});
const posSimValue = computed(() => getNumericMetric(lastValMetrics.value, 'PosSim', 'margin_pos_sim'));
const negSimValue = computed(() => getNumericMetric(lastValMetrics.value, 'NegSim', 'margin_neg_sim'));
function simBarWidth(value: number | null): string {
  if (value == null) return '0%';
  const pct = Math.max(0, Math.min(100, value * 100));
  return `${pct.toFixed(1)}%`;
}
const finalTrainLoss = computed(() => {
  const v = lastTrainMetrics.value?.loss ?? trainLosses.value[trainLosses.value.length - 1];
  return v != null ? Number(v).toFixed(4) : null;
});
const finalValLoss = computed(() => {
  const v = lastValMetrics.value?.loss ?? valLosses.value[valLosses.value.length - 1];
  return v != null ? Number(v).toFixed(4) : null;
});

// Current step info — describes the *live run* (from runMetrics/runProgress).
const currentStep = computed(() => runMetrics.value?.step ?? null);
const totalSteps = computed(() => runMetrics.value?.totalSteps ?? null);
const progressPct = computed(() => {
  const p = exp.value?.runProgress;
  if (typeof p === 'number' && p > 0) return Math.round(p);
  if (!currentStep.value || !totalSteps.value) return 0;
  return Math.round((currentStep.value / totalSteps.value) * 100);
});

// ── Stop training ─────────────────────────────────────────────────────────────
async function stopTraining() {
  if (stopping.value) return;
  stopping.value = true;
  try {
    await ExperimentsApi.stop(props.experimentId);
  } catch {
    stopping.value = false;
  }
}

async function retrainExperiment() {
  if (retraining.value || isActive.value) return;
  const ok = window.confirm('确认重新训练该实验吗？这将基于当前实验配置重新开始训练。');
  if (!ok) return;

  retraining.value = true;
  try {
    await ExperimentsApi.train(props.experimentId);
    await fetchExp();
    if (isActive.value) {
      if (pollTimer) clearTimeout(pollTimer);
      schedulePoll();
    }
  } catch (e) {
    window.alert(`重新训练启动失败：${String(e)}`);
  } finally {
    retraining.value = false;
  }
}

// ── Chart options ─────────────────────────────────────────────────────────────
const CHART_STYLE = {
  backgroundColor: 'transparent',
  grid: {left: 56, right: 20, top: 36, bottom: 44},
};
const TOOLTIP_STYLE = {
  trigger: 'axis' as const,
  backgroundColor: '#09090b',
  borderColor: '#27272a',
  padding: [8, 12] as [number, number],
  textStyle: {color: '#fafafa', fontSize: 12},
};
const AXIS_STYLE = {
  axisLine: {lineStyle: {color: '#27272a'}},
  axisTick: {show: false},
  axisLabel: {color: '#71717a', fontSize: 11},
};

function stepXAxis(data: number[]) {
  return {
    type: 'category' as const,
    data,
    ...AXIS_STYLE,
    name: 'Step',
    nameLocation: 'middle' as const,
    nameGap: 30,
    nameTextStyle: {color: '#52525b', fontSize: 11},
  };
}

function valueYAxis(formatter?: (v: number) => string) {
  return {
    type: 'value' as const,
    axisLine: {show: false},
    splitLine: {lineStyle: {color: '#1c1c1f', type: 'dashed'}},
    axisLabel: {color: '#71717a', fontSize: 11, formatter: formatter ?? ((v: number) => v.toFixed(2))},
  };
}

const lossChartOption = computed(() => ({
  ...CHART_STYLE,
  legend: {
    top: 4, right: 0,
    textStyle: {color: '#a1a1aa', fontSize: 11},
    itemWidth: 16, itemHeight: 2,
    data: [
      {name: 'Train Loss', icon: 'rect'},
      ...(valLosses.value.length > 0 ? [{name: 'Val Loss', icon: 'rect'}] : []),
    ],
  },
  tooltip: {
    ...TOOLTIP_STYLE,
    formatter: (params: any[]) => {
      const step = steps.value[params[0].dataIndex] ?? params[0].dataIndex + 1;
      const rows = params
        .filter((p) => p.value != null && p.value !== '')
        .map(
        (p) => `<div style="display:flex;justify-content:space-between;gap:16px;margin-top:4px">` +
          `<span style="color:#a1a1aa">${p.marker}${p.seriesName}</span>` +
          `<b>${Number(p.value).toFixed(4)}</b></div>`,
      ).join('');
      return `<div style="font-size:11px;color:#71717a;margin-bottom:2px">Step ${step}</div>${rows}`;
    },
  },
  xAxis: stepXAxis(steps.value),
  yAxis: valueYAxis((v) => v.toFixed(2)),
  series: [
    {
      name: 'Train Loss', type: 'line', data: trainLosses.value,
      smooth: 0.3, symbolSize: 4,
      lineStyle: {color: '#a855f7', width: 2},
      itemStyle: {color: '#a855f7'},
    },
    ...(valLosses.value.length > 0 ? [{
      name: 'Val Loss', type: 'line', data: valLosses.value,
      smooth: 0.3, symbolSize: 4,
      lineStyle: {color: '#10b981', width: 2, type: 'dashed'},
      itemStyle: {color: '#10b981'},
    }] : []),
  ],
}));

const marginChartOption = computed(() => ({
  ...CHART_STYLE,
  legend: {
    top: 4, right: 0,
    textStyle: {color: '#a1a1aa', fontSize: 11},
    itemWidth: 16, itemHeight: 2,
    data: [{name: 'Pos Sim', icon: 'rect'}, {name: 'Neg Sim', icon: 'rect'}, {name: 'Margin', icon: 'rect'}],
  },
  tooltip: {
    ...TOOLTIP_STYLE,
    formatter: (params: any[]) => {
      const step = steps.value[params[0].dataIndex] ?? params[0].dataIndex + 1;
      const rows = params
        .filter((p) => p.value != null && p.value !== '')
        .map(
        (p) => `<div style="display:flex;justify-content:space-between;gap:16px;margin-top:4px">` +
          `<span style="color:#a1a1aa">${p.marker}${p.seriesName}</span>` +
          `<b>${Number(p.value).toFixed(4)}</b></div>`,
      ).join('');
      return `<div style="font-size:11px;color:#71717a;margin-bottom:2px">Step ${step}</div>${rows}`;
    },
  },
  xAxis: stepXAxis(steps.value),
  yAxis: valueYAxis((v) => v.toFixed(2)),
  series: [
    {
      name: 'Pos Sim', type: 'line', data: livePosSims.value,
      smooth: 0.3, symbolSize: 4,
      lineStyle: {color: '#10b981', width: 2},
      itemStyle: {color: '#10b981'},
    },
    {
      name: 'Neg Sim', type: 'line', data: liveNegSims.value,
      smooth: 0.3, symbolSize: 4,
      lineStyle: {color: '#f43f5e', width: 2},
      itemStyle: {color: '#f43f5e'},
    },
    {
      name: 'Margin', type: 'line', data: liveMargins.value,
      smooth: 0.3, symbolSize: 4,
      lineStyle: {color: '#f59e0b', width: 2, type: 'dashed'},
      itemStyle: {color: '#f59e0b'},
    },
  ],
}));

</script>

<template>
  <!-- Loading -->
  <div v-if="loading" class="flex h-full items-center justify-center text-muted-foreground">
    <div class="flex flex-col items-center gap-3">
      <div class="h-6 w-6 animate-spin rounded-full border-2 border-border border-t-primary" />
      <span class="text-xs">Loading experiment...</span>
    </div>
  </div>

  <!-- Fetch error -->
  <div v-else-if="fetchError" class="flex h-full items-center justify-center">
    <div class="flex flex-col items-center gap-3 text-rose-400">
      <AlertCircle class="h-8 w-8" />
      <span class="text-sm">Failed to load experiment data</span>
      <span class="text-xs text-muted-foreground">{{ fetchError }}</span>
    </div>
  </div>

  <!-- First-ever training failed (no prior successful result to fall back to) -->
  <div v-else-if="mode === 'failedFirst'" class="flex h-full items-center justify-center p-8">
    <div class="max-w-md rounded-xl border border-rose-500/20 bg-rose-500/5 p-6 text-center">
      <AlertCircle class="mx-auto mb-3 h-8 w-8 text-rose-400" />
      <p class="mb-2 text-sm font-medium text-rose-400">训练失败</p>
      <p class="text-xs text-muted-foreground">{{ runError ?? 'Unknown error' }}</p>
      <button
        type="button"
        :disabled="retraining || runActive"
        class="mt-4 inline-flex items-center justify-center rounded-md border border-primary/30 bg-primary/10 px-3 py-1.5 text-xs font-medium text-primary transition-colors hover:bg-primary/20 disabled:cursor-not-allowed disabled:opacity-50"
        @click="retrainExperiment"
      >
        {{ retraining ? '启动中…' : '重新训练' }}
      </button>
    </div>
  </div>

  <!-- First-ever training stopped before any success -->
  <div v-else-if="mode === 'stoppedFirst'" class="flex h-full items-center justify-center p-8">
    <div class="max-w-md rounded-xl border border-amber-500/20 bg-amber-500/5 p-6 text-center">
      <OctagonX class="mx-auto mb-3 h-8 w-8 text-amber-400" />
      <p class="mb-2 text-sm font-medium text-amber-300">训练已中止</p>
      <p class="text-xs text-muted-foreground">本次训练在完成前被中止，尚无可用模型。</p>
      <button
        type="button"
        :disabled="retraining || runActive"
        class="mt-4 inline-flex items-center justify-center rounded-md border border-primary/30 bg-primary/10 px-3 py-1.5 text-xs font-medium text-primary transition-colors hover:bg-primary/20 disabled:cursor-not-allowed disabled:opacity-50"
        @click="retrainExperiment"
      >
        {{ retraining ? '启动中…' : '重新训练' }}
      </button>
    </div>
  </div>

  <!-- Training in progress (first training or re-training) — live curves -->
  <div v-else-if="mode === 'live'" class="flex min-h-0 h-full flex-col gap-4 overflow-y-auto p-6">

    <!-- Header: status + stop button -->
    <div class="flex items-center justify-between rounded-xl border border-amber-500/20 bg-amber-500/5 px-5 py-4">
      <div class="flex items-center gap-3">
        <div class="h-5 w-5 animate-spin rounded-full border-2 border-amber-500/30 border-t-amber-400" />
        <div>
          <p class="text-sm font-medium text-amber-300">
            <template v-if="stage === 'preparing_dataset'">{{ hasResult ? '重新训练 · 正在准备数据集…' : '正在准备数据集…' }}</template>
            <template v-else>{{ hasResult ? '重新训练进行中' : '训练进行中' }}</template>
          </p>
          <p v-if="currentStep && totalSteps" class="mt-0.5 text-xs text-muted-foreground">
            Step {{ currentStep }} / {{ totalSteps }}
            <span class="ml-2 text-amber-400/70">{{ progressPct }}%</span>
          </p>
          <p v-else-if="hasResult" class="mt-0.5 text-xs text-muted-foreground">
            若本次训练失败或中止，将自动保留上一次成功的结果。
          </p>
        </div>
      </div>
      <button
        type="button"
        :disabled="stopping"
        class="flex items-center gap-2 rounded-lg border border-rose-500/40 bg-rose-500/10 px-4 py-2 text-sm font-medium text-rose-400 transition-colors hover:bg-rose-500/20 disabled:cursor-wait disabled:opacity-50"
        @click="stopTraining"
      >
        <OctagonX class="h-4 w-4" />
        {{ stopping ? '正在停止…' : '停止训练' }}
      </button>
    </div>

    <!-- Progress bar -->
    <div v-if="totalSteps" class="h-1.5 w-full overflow-hidden rounded-full bg-secondary/30">
      <div
        class="h-full rounded-full bg-amber-400 transition-all duration-700"
        :style="{width: progressPct + '%'}"
      />
    </div>

    <!-- Live metric cards -->
    <div v-if="trainLosses.length > 0" class="grid grid-cols-2 gap-3 sm:grid-cols-3">
      <div class="flex flex-col gap-1 rounded-xl border border-border bg-secondary/5 p-4">
        <p class="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">当前 Train Loss</p>
        <p class="text-2xl font-semibold tabular-nums text-purple-400">
          {{ trainLosses[trainLosses.length - 1]?.toFixed(4) ?? '---' }}
        </p>
      </div>
      <div class="flex flex-col gap-1 rounded-xl border border-border bg-secondary/5 p-4">
        <p class="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">当前 Val Loss</p>
        <p class="text-2xl font-semibold tabular-nums text-emerald-400">
          {{ valLosses[valLosses.length - 1]?.toFixed(4) ?? '---' }}
        </p>
      </div>
      <div class="flex flex-col gap-1 rounded-xl border border-border bg-secondary/5 p-4">
        <p class="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">Margin</p>
        <p class="text-2xl font-semibold tabular-nums text-amber-400">
          {{ liveMargins.length ? liveMargins[liveMargins.length - 1].toFixed(4) : '---' }}
        </p>
      </div>
    </div>

    <!-- Live charts — only show once we have data -->
    <template v-if="trainLosses.length > 0">
      <!-- Loss curve -->
      <div class="rounded-xl border border-border bg-secondary/5 p-5">
        <h3 class="mb-1 text-sm font-semibold">训练 / 验证损失曲线</h3>
        <p class="mb-3 text-[11px] text-muted-foreground">实时更新 · 每个检查点刷新一次</p>
        <div class="h-[220px] w-full shrink-0 overflow-hidden">
          <VChart class="h-full w-full min-h-0" :option="lossChartOption" autoresize />
        </div>
      </div>

      <!-- Embedding similarity (Pos/Neg/Margin) curve -->
      <div v-if="liveMargins.length > 0" class="w-full rounded-xl border border-border bg-secondary/5 p-5">
        <h3 class="mb-1 text-sm font-semibold">特征相似度曲线</h3>
        <p class="mb-3 text-[11px] text-muted-foreground">Pos Sim / Neg Sim / Margin 变化趋势</p>
        <div class="h-[200px] w-full shrink-0 overflow-hidden">
          <VChart class="h-full w-full min-h-0" :option="marginChartOption" autoresize />
        </div>
      </div>
    </template>

    <!-- Placeholder: preparing or no data yet -->
    <div v-else class="flex flex-1 items-center justify-center text-xs text-muted-foreground/40">
      {{ stage === 'preparing_dataset' ? '正在准备数据集，训练即将开始…' : '等待第一个检查点完成后显示曲线…' }}
    </div>
  </div>

  <!-- Result dashboard — shows the last *successful* run. While re-training is in
       progress the live view (mode 'live') takes over; only a finished-but-failed/
       stopped attempt falls back here with a non-destructive notice banner. -->
  <div v-else-if="exp && mode === 'result'" class="flex min-h-0 h-full flex-col gap-5 overflow-y-auto p-6">

    <!-- Run notice: last re-train failed / stopped (result below is the previous success) -->
    <div
      v-if="showRunBanner"
      class="flex items-start justify-between gap-4 rounded-xl border px-5 py-4"
      :class="runStatus === 'Failed' ? 'border-rose-500/20 bg-rose-500/5' : 'border-amber-500/20 bg-amber-500/5'"
    >
      <div class="flex items-start gap-3">
        <AlertCircle v-if="runStatus === 'Failed'" class="mt-0.5 h-5 w-5 shrink-0 text-rose-400" />
        <OctagonX v-else class="mt-0.5 h-5 w-5 shrink-0 text-amber-400" />
        <div>
          <p class="text-sm font-medium" :class="runStatus === 'Failed' ? 'text-rose-400' : 'text-amber-300'">
            {{ runStatus === 'Failed' ? '上一次重新训练失败' : '上一次重新训练已中止' }}
          </p>
          <p class="mt-0.5 text-xs text-muted-foreground">
            下方展示的是上一次成功训练的结果；该模型仍可用于推理。
            <span v-if="runStatus === 'Failed' && runError" class="text-rose-400/80">（{{ runError }}）</span>
          </p>
        </div>
      </div>
      <div class="flex shrink-0 items-center gap-2">
        <button
          type="button"
          :disabled="retraining || runActive"
          class="inline-flex items-center justify-center rounded-md border border-primary/30 bg-primary/10 px-3 py-1.5 text-xs font-medium text-primary transition-colors hover:bg-primary/20 disabled:cursor-not-allowed disabled:opacity-50"
          @click="retrainExperiment"
        >
          {{ retraining ? '启动中…' : '重新训练' }}
        </button>
        <button
          type="button"
          class="rounded-md p-1.5 text-muted-foreground transition-colors hover:bg-secondary/40 hover:text-foreground"
          title="忽略"
          @click="noticeDismissed = true"
        >
          <X class="h-4 w-4" />
        </button>
      </div>
    </div>

    <!-- Config summary banner -->
    <div class="rounded-xl border border-border bg-secondary/5 p-4">
      <div class="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
        <div class="flex flex-wrap items-start gap-x-8 gap-y-3">
          <div class="flex items-center gap-2 text-sm">
            <Database class="h-4 w-4 shrink-0 text-muted-foreground" />
            <span class="text-muted-foreground">数据集</span>
            <span class="font-medium">{{ exp.dataset }}</span>
          </div>
          <div class="flex items-center gap-2 text-sm">
            <Layers class="h-4 w-4 shrink-0 text-muted-foreground" />
            <span class="text-muted-foreground">样本</span>
            <span class="font-medium">{{ exp.sampleCount }}</span>
            <span class="text-xs text-muted-foreground/60">(train {{ exp.trainSampleCount }} / val {{ exp.valSampleCount }})</span>
          </div>
          <div class="flex items-center gap-2 text-sm">
            <Clock class="h-4 w-4 shrink-0 text-muted-foreground" />
            <span class="text-muted-foreground">耗时</span>
            <span class="font-medium">{{ exp.duration ?? '---' }}</span>
          </div>
          <div class="flex items-center gap-2 text-sm">
            <CheckCircle2 v-if="stage === 'completed'" class="h-4 w-4 shrink-0 text-emerald-500" />
            <OctagonX v-else class="h-4 w-4 shrink-0 text-amber-400" />
            <span :class="stage === 'completed' ? 'font-medium text-emerald-500' : 'font-medium text-amber-400'">
              {{ stage === 'completed' ? 'Completed' : '已中止' }}
            </span>
            <span v-if="stage === 'stopped' && summary?.train_losses?.length" class="text-xs text-muted-foreground">
              (step {{ summary?.steps_run ?? '?' }} / {{ summary?.total_steps ?? '?' }})
            </span>
          </div>
        </div>
        <button
          type="button"
          :disabled="retraining || isActive"
          class="inline-flex items-center justify-center rounded-md border border-primary/30 bg-primary/10 px-3 py-1.5 text-xs font-medium text-primary transition-colors hover:bg-primary/20 disabled:cursor-not-allowed disabled:opacity-50"
          @click="retrainExperiment"
        >
          {{ retraining ? '启动中…' : '重新训练' }}
        </button>
      </div>
      <div v-if="categories.length > 0" class="mt-3 flex flex-wrap items-center gap-1.5">
        <Tag class="h-3.5 w-3.5 shrink-0 text-muted-foreground" />
        <span
          v-for="cat in categories"
          :key="cat"
          class="rounded-md border border-border bg-secondary/20 px-2 py-0.5 text-[11px] font-medium"
        >{{ cat }}</span>
      </div>
    </div>

    <!-- Key metric cards -->
    <div class="grid grid-cols-2 gap-3 sm:grid-cols-3">
      <div class="col-span-2 flex flex-col justify-between rounded-xl border border-primary/20 bg-primary/5 p-4 sm:col-span-1">
        <p class="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">Margin</p>
        <div class="mt-2">
          <p class="text-3xl font-bold tabular-nums">{{ valMargin ?? '---' }}</p>
          <p class="mt-1 text-[10px] text-muted-foreground">PosSim − NegSim (验证集)</p>
        </div>
      </div>
      <div class="flex flex-col justify-between rounded-xl border border-border bg-secondary/5 p-4">
        <p class="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">最终 Train Loss</p>
        <div class="mt-2">
          <p class="text-2xl font-semibold tabular-nums text-purple-400">{{ finalTrainLoss ?? '---' }}</p>
          <p class="mt-1 text-[10px] text-muted-foreground">最新一次训练损失</p>
        </div>
      </div>
      <div class="flex flex-col justify-between rounded-xl border border-border bg-secondary/5 p-4">
        <p class="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">最终 Val Loss</p>
        <div class="mt-2">
          <p class="text-2xl font-semibold tabular-nums text-emerald-400">{{ finalValLoss ?? '---' }}</p>
          <p class="mt-1 text-[10px] text-muted-foreground">最新一次验证损失</p>
        </div>
      </div>
    </div>

    <!-- Charts -->
    <div class="grid grid-cols-1 items-start gap-4 lg:grid-cols-3">
      <!-- Loss curve — 2/3 width -->
      <div class="flex w-full flex-col rounded-xl border border-border bg-secondary/5 p-5 lg:col-span-2">
        <h3 class="mb-1 text-sm font-semibold">训练 / 验证损失曲线</h3>
        <p class="mb-4 text-[11px] text-muted-foreground">双曲线靠拢表示模型泛化良好；若 Val Loss 远高于 Train Loss 则说明过拟合。</p>
        <div v-if="trainLosses.length > 0" class="h-[260px] w-full shrink-0 overflow-hidden">
          <VChart class="h-full w-full min-h-0" :option="lossChartOption" autoresize />
        </div>
        <div v-else class="flex h-[260px] shrink-0 items-center justify-center text-xs text-muted-foreground/50">暂无损失数据</div>
      </div>

      <!-- Similarity sidebar — 1/3 -->
      <div class="flex w-full flex-col rounded-xl border border-border bg-secondary/5 p-5">
        <h3 class="mb-1 text-sm font-semibold">特征相似度分布</h3>
        <p class="mb-4 text-[11px] text-muted-foreground">Pos Sim 越高、Neg Sim 越低，说明特征空间的类间分离度越好。</p>
        <template v-if="posSim != null">
          <div class="space-y-3">
            <div>
              <div class="mb-1 flex items-center justify-between text-[11px]">
                <span class="flex items-center gap-1.5 text-muted-foreground">
                  <span class="inline-block h-2 w-2 rounded-full bg-emerald-500" />Pos Sim (同类)
                </span>
                <span class="font-mono font-medium text-emerald-400">{{ posSim }}</span>
              </div>
              <div class="h-1.5 w-full overflow-hidden rounded-full bg-secondary/40">
                <div class="h-full rounded-full bg-emerald-500/90 transition-all duration-500" :style="{width: simBarWidth(posSimValue)}" />
              </div>
            </div>

            <div>
              <div class="mb-1 flex items-center justify-between text-[11px]">
                <span class="flex items-center gap-1.5 text-muted-foreground">
                  <span class="inline-block h-2 w-2 rounded-full bg-rose-500" />Neg Sim (异类)
                </span>
                <span class="font-mono font-medium text-rose-400">{{ negSim }}</span>
              </div>
              <div class="h-1.5 w-full overflow-hidden rounded-full bg-secondary/40">
                <div class="h-full rounded-full bg-rose-500/90 transition-all duration-500" :style="{width: simBarWidth(negSimValue)}" />
              </div>
            </div>
          </div>

          <div class="mt-4 space-y-2 text-xs">
            <div class="flex items-center justify-between border-t border-border pt-2">
              <span class="text-muted-foreground">间距 (Margin)</span>
              <span class="font-mono font-medium">{{ valMargin }}</span>
            </div>
          </div>
        </template>
        <div v-else class="flex h-[120px] shrink-0 items-center justify-center text-xs text-muted-foreground/50">无相似度数据</div>
      </div>
    </div>

    <!-- Embedding similarity (Pos/Neg/Margin) curve (from live series stored in summary) -->
    <div v-if="liveMargins.length > 1" class="w-full rounded-xl border border-border bg-secondary/5 p-5">
      <h3 class="mb-1 text-sm font-semibold">特征相似度曲线</h3>
      <div class="mt-3 h-[200px] w-full shrink-0 overflow-hidden">
        <VChart class="h-full w-full min-h-0" :option="marginChartOption" autoresize />
      </div>
    </div>

    <!-- CTA -->
    <div class="flex items-center justify-between rounded-xl border border-border bg-secondary/5 p-4">
      <div>
        <p class="text-sm font-medium">模型已就绪</p>
        <p class="text-xs text-muted-foreground">可以使用此模型对新数据进行推理测试。</p>
      </div>
      <button
        type="button"
        class="flex items-center gap-2 rounded-lg bg-primary px-4 py-2 text-sm font-medium text-primary-foreground transition-colors hover:bg-primary/90"
        @click="router.push('/inference')"
      >
        开始推理测试
        <ArrowRight class="h-4 w-4" />
      </button>
    </div>
  </div>

  <!-- Fallback -->
  <div v-else class="flex h-full items-center justify-center text-xs text-muted-foreground">
    暂无实验数据
  </div>
</template>

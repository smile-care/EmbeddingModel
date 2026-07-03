<script setup lang="ts">
import {computed} from 'vue';
import {AlertTriangle, ArrowRight} from 'lucide-vue-next';
import {categoryChartColor} from '@/lib/api';

const props = defineProps<{
  labels: string[];
  confusion: number[][];
  centroidSim: number[][];
  confusionThreshold: number;
  simThreshold: number;
}>();
const emit = defineEmits<{highlight: [labels: string[]]}>();

interface Warning {
  i: number;
  j: number;
  a: string;
  b: string;
  /** max of the two directional cross fractions */
  cross: number;
  sim: number;
  severity: number;
  level: 'high' | 'medium';
}

const warnings = computed<Warning[]>(() => {
  const labels = props.labels;
  const out: Warning[] = [];
  for (let i = 0; i < labels.length; i++) {
    for (let j = i + 1; j < labels.length; j++) {
      const cross = Math.max(props.confusion?.[i]?.[j] ?? 0, props.confusion?.[j]?.[i] ?? 0);
      const sim = props.centroidSim?.[i]?.[j] ?? 0;
      const overCross = cross >= props.confusionThreshold;
      const overSim = sim >= props.simThreshold;
      if (!overCross && !overSim) continue;
      // severity blends both signals
      const severity = 0.6 * cross + 0.4 * Math.max(0, sim);
      out.push({
        i, j, a: labels[i], b: labels[j], cross, sim, severity,
        level: cross >= 0.4 || sim >= 0.95 ? 'high' : 'medium',
      });
    }
  }
  return out.sort((x, y) => y.severity - x.severity);
});
</script>

<template>
  <div class="space-y-2">
    <div v-if="warnings.length === 0" class="flex flex-col items-center justify-center gap-2 py-12 text-muted-foreground">
      <div class="rounded-full bg-emerald-500/10 p-4"><AlertTriangle class="h-6 w-6 text-emerald-500/50" /></div>
      <p class="text-sm">当前阈值下未发现相似/混淆类别对。</p>
    </div>
    <button
      v-for="w in warnings"
      :key="w.a + '|' + w.b"
      type="button"
      class="flex w-full items-center gap-3 rounded-lg border p-3 text-left transition-all hover:shadow-sm"
      :class="w.level === 'high' ? 'border-rose-500/40 bg-rose-500/5 hover:border-rose-500/70' : 'border-amber-500/40 bg-amber-500/5 hover:border-amber-500/70'"
      @click="emit('highlight', [w.a, w.b])"
    >
      <AlertTriangle class="h-4 w-4 shrink-0" :class="w.level === 'high' ? 'text-rose-500' : 'text-amber-500'" />
      <div class="flex min-w-0 flex-1 items-center gap-2 text-sm font-medium">
        <span class="flex items-center gap-1 truncate">
          <span class="h-2.5 w-2.5 shrink-0 rounded-full" :style="{backgroundColor: categoryChartColor(w.i, labels.length)}" />
          {{ w.a }}
        </span>
        <ArrowRight class="h-3 w-3 shrink-0 text-muted-foreground" />
        <span class="flex items-center gap-1 truncate">
          <span class="h-2.5 w-2.5 shrink-0 rounded-full" :style="{backgroundColor: categoryChartColor(w.j, labels.length)}" />
          {{ w.b }}
        </span>
      </div>
      <div class="flex shrink-0 items-center gap-3 text-[10px] tabular-nums">
        <span class="text-muted-foreground">跨类邻居 <b class="text-foreground">{{ Math.round(w.cross * 100) }}%</b></span>
        <span class="text-muted-foreground">类心相似 <b class="text-foreground">{{ w.sim.toFixed(2) }}</b></span>
      </div>
    </button>
  </div>
</template>

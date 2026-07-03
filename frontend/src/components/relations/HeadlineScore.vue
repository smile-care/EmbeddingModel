<script setup lang="ts">
import {computed} from 'vue';
import type {RelationsHeadline} from '@/lib/api';

const props = defineProps<{headline: RelationsHeadline}>();

interface Metric {
  label: string;
  value: string;
  hint: string;
  tone: 'good' | 'warn' | 'bad' | 'neutral';
}

function pct(v: number | null): string {
  return v == null ? '—' : `${Math.round(v * 100)}%`;
}

const metrics = computed<Metric[]>(() => {
  const h = props.headline;
  const purityTone = h.purity == null ? 'neutral' : h.purity >= 0.85 ? 'good' : h.purity >= 0.65 ? 'warn' : 'bad';
  const sil = h.meanSilhouette;
  const silTone = sil == null ? 'neutral' : sil >= 0.5 ? 'good' : sil >= 0.25 ? 'warn' : 'bad';
  return [
    {label: '全局 kNN 纯度', value: pct(h.purity), hint: '邻域中同类占比，越高类别越可分', tone: purityTone},
    {label: '平均 Silhouette', value: sil == null ? '—' : sil.toFixed(2), hint: '类内紧致 vs 类间分离，越接近 1 越好', tone: silTone},
    {label: '类别数', value: String(h.nClasses), hint: '参与分析的类别数量', tone: 'neutral'},
    {label: '样本数', value: String(h.nSamples), hint: '参与分析的 crop 数量', tone: 'neutral'},
  ];
});

const toneClass: Record<Metric['tone'], string> = {
  good: 'text-emerald-500',
  warn: 'text-amber-500',
  bad: 'text-rose-500',
  neutral: 'text-foreground',
};
</script>

<template>
  <div class="grid grid-cols-2 gap-3 lg:grid-cols-4">
    <div
      v-for="m in metrics"
      :key="m.label"
      class="flex flex-col gap-1 rounded-xl border border-border bg-secondary/5 p-4"
    >
      <span class="text-[10px] font-medium uppercase tracking-wide text-muted-foreground">{{ m.label }}</span>
      <span class="text-2xl font-bold tabular-nums" :class="toneClass[m.tone]">{{ m.value }}</span>
      <span class="text-[10px] leading-tight text-muted-foreground">{{ m.hint }}</span>
    </div>
  </div>
</template>

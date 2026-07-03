<script setup lang="ts">
import {computed} from 'vue';
import {ArrowRight} from 'lucide-vue-next';
import {staticUrl, type RelationsMislabel} from '@/lib/api';

const props = defineProps<{
  mislabels: RelationsMislabel[];
  /** Only show candidates whose neighborhood cross fraction >= this. */
  threshold: number;
}>();
const emit = defineEmits<{preview: [item: RelationsMislabel]}>();

const filtered = computed(() =>
  props.mislabels.filter((m) => m.crossFraction >= props.threshold),
);
</script>

<template>
  <div>
    <div v-if="filtered.length === 0" class="flex flex-col items-center justify-center gap-2 py-12 text-muted-foreground">
      <p class="text-sm">当前阈值下没有疑似误标样本。</p>
    </div>
    <div v-else class="grid grid-cols-2 gap-3 sm:grid-cols-3 md:grid-cols-4 xl:grid-cols-6">
      <button
        v-for="m in filtered"
        :key="m.cropId"
        type="button"
        class="group flex flex-col gap-1.5 rounded-lg border border-border bg-background p-1.5 text-left transition-all hover:border-rose-500/60 hover:shadow-sm"
        @click="emit('preview', m)"
      >
        <div class="relative aspect-square overflow-hidden rounded-md bg-secondary/10">
          <img :src="staticUrl(m.url)" alt="" class="h-full w-full object-cover transition-transform group-hover:scale-110" />
          <div class="absolute right-1 top-1 rounded-full bg-rose-500 px-1.5 py-0.5 text-[8px] font-bold text-white shadow">
            {{ Math.round(m.crossFraction * 100) }}%
          </div>
        </div>
        <div class="flex items-center gap-1 text-[10px] font-medium">
          <span class="truncate text-muted-foreground line-through">{{ m.currentLabel }}</span>
          <ArrowRight class="h-2.5 w-2.5 shrink-0 text-rose-500" />
          <span class="truncate text-emerald-500">{{ m.suggestedLabel }}</span>
        </div>
      </button>
    </div>
  </div>
</template>

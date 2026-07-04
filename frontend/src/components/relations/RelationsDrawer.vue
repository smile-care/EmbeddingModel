<script setup lang="ts">
import {computed, ref} from 'vue';
import {
  Activity, BarChart3, GitBranch, Grid3x3, LayoutGrid, Network,
  RefreshCw, ScatterChart, ShieldAlert, Sparkles, Star, X,
} from 'lucide-vue-next';
import {
  ANALYSIS_VIEW_GROUP_DEFS,
  ANALYSIS_VIEW_LABELS,
  type ViewKey,
} from '@/components/relations/analysisViews';
import HoverHelpTip from '@/components/relations/HoverHelpTip.vue';
import {RELATION_DRAWER_PARAM_HELP} from '@/components/relations/relationViewHelp';

export type {ViewKey};

const props = defineProps<{
  open: boolean;
  activeView: ViewKey;
  k: number;
  confusionThreshold: number;
  simThreshold: number;
  mislabelThreshold: number;
  loading: boolean;
  relationsAvailable: boolean;
}>();
const emit = defineEmits<{
  close: [];
  'update:activeView': [v: ViewKey];
  'update:k': [v: number];
  'update:confusionThreshold': [v: number];
  'update:simThreshold': [v: number];
  'update:mislabelThreshold': [v: number];
  reloadK: [];
}>();

interface Item {key: ViewKey; label: string; icon: any; needsRelations: boolean}

const VIEW_ICONS: Record<ViewKey, any> = {
  distribution: ScatterChart,
  anomaly: LayoutGrid,
  headline: Activity,
  confusion: Grid3x3,
  centroid: BarChart3,
  dendrogram: GitBranch,
  graph: Network,
  warnings: ShieldAlert,
  perclass: Star,
  mislabels: Sparkles,
};

const groups: {title: string; items: Item[]}[] = ANALYSIS_VIEW_GROUP_DEFS.map((g) => ({
  title: g.title,
  items: g.keys.map((key) => ({
    key,
    label: ANALYSIS_VIEW_LABELS[key],
    icon: VIEW_ICONS[key],
    needsRelations: key !== 'distribution' && key !== 'anomaly',
  })),
}));

const confPct = computed({
  get: () => Math.round(props.confusionThreshold * 100),
  set: (v: number) => emit('update:confusionThreshold', v / 100),
});
const simPct = computed({
  get: () => Math.round(props.simThreshold * 100),
  set: (v: number) => emit('update:simThreshold', v / 100),
});
const mislabelPct = computed({
  get: () => Math.round(props.mislabelThreshold * 100),
  set: (v: number) => emit('update:mislabelThreshold', v / 100),
});

const rootRef = ref<HTMLElement | null>(null);
defineExpose({rootEl: rootRef});
</script>

<template>
  <aside
    ref="rootRef"
    class="flex h-full shrink-0 flex-col overflow-hidden border-border bg-background shadow-lg transition-[width,border-color] duration-200 ease-out"
    :class="open ? 'w-[320px] border-l' : 'w-0 border-l-0 pointer-events-none'"
    :aria-hidden="!open"
  >
    <div class="flex h-full w-[320px] flex-col">
      <div class="flex shrink-0 items-center justify-between border-b border-border px-4 py-3">
        <h3 class="text-sm font-bold">分析中心</h3>
        <button type="button" class="rounded p-1 text-muted-foreground hover:bg-secondary/30" aria-label="关闭分析中心" @click="emit('close')">
          <X class="h-4 w-4" />
        </button>
      </div>

      <div class="min-h-0 flex-1 overflow-y-auto p-3">
        <div v-for="g in groups" :key="g.title" class="mb-4">
          <p class="mb-1.5 px-1 text-[10px] font-bold uppercase tracking-wide text-muted-foreground">{{ g.title }}</p>
          <div class="space-y-1">
            <button
              v-for="it in g.items"
              :key="it.key"
              type="button"
              class="flex w-full items-center gap-2 rounded-lg px-2.5 py-2 text-left text-xs font-medium transition-all"
              :class="[
                activeView === it.key ? 'bg-primary/15 text-primary ring-1 ring-primary/30' : 'text-foreground hover:bg-secondary/30',
                it.needsRelations && !relationsAvailable ? 'opacity-50' : '',
              ]"
              @click="emit('update:activeView', it.key)"
            >
              <component :is="it.icon" class="h-4 w-4 shrink-0" />
              <span class="flex-1 truncate">{{ it.label }}</span>
              <RefreshCw v-if="it.needsRelations && loading" class="h-3 w-3 animate-spin text-muted-foreground" />
            </button>
          </div>
        </div>

        <div class="mt-2 space-y-4 rounded-xl border border-border bg-secondary/5 p-3">
          <div class="flex items-center gap-1.5">
            <p class="text-[10px] font-bold uppercase tracking-wide text-muted-foreground">参数</p>
            <HoverHelpTip label="参数说明" width-class="w-72" placement="floating-left">
              <p class="mb-2 font-medium text-foreground/90">{{ RELATION_DRAWER_PARAM_HELP.section.intro }}</p>
              <dl class="space-y-2">
                <div v-for="item in RELATION_DRAWER_PARAM_HELP.section.items" :key="item.title">
                  <dt class="font-semibold text-foreground">{{ item.title }}</dt>
                  <dd class="text-muted-foreground">{{ item.body }}</dd>
                </div>
              </dl>
            </HoverHelpTip>
          </div>

          <div>
            <div class="mb-1 flex items-center justify-between text-[11px]">
              <span class="flex items-center gap-1">
                邻居数 k
                <HoverHelpTip :label="RELATION_DRAWER_PARAM_HELP.k.label" width-class="w-64" placement="floating-left">
                  <p class="text-muted-foreground">{{ RELATION_DRAWER_PARAM_HELP.k.body }}</p>
                </HoverHelpTip>
              </span>
              <span class="font-mono font-semibold">{{ k }}</span>
            </div>
            <input
              type="range" min="3" max="30" step="1" :value="k"
              class="w-full accent-primary"
              @input="emit('update:k', Number(($event.target as HTMLInputElement).value))"
              @change="emit('reloadK')"
            />
          </div>

          <div>
            <div class="mb-1 flex items-center justify-between text-[11px]">
              <span class="flex items-center gap-1">
                混淆警告阈值
                <HoverHelpTip :label="RELATION_DRAWER_PARAM_HELP.confusionThreshold.label" width-class="w-64" placement="floating-left">
                  <p class="text-muted-foreground">{{ RELATION_DRAWER_PARAM_HELP.confusionThreshold.body }}</p>
                </HoverHelpTip>
              </span>
              <span class="font-mono font-semibold">{{ confPct }}%</span>
            </div>
            <input type="range" min="10" max="90" step="5" v-model.number="confPct" class="w-full accent-rose-500" />
          </div>

          <div>
            <div class="mb-1 flex items-center justify-between text-[11px]">
              <span class="flex items-center gap-1">
                相似度警告阈值
                <HoverHelpTip :label="RELATION_DRAWER_PARAM_HELP.simThreshold.label" width-class="w-64" placement="floating-left">
                  <p class="text-muted-foreground">{{ RELATION_DRAWER_PARAM_HELP.simThreshold.body }}</p>
                </HoverHelpTip>
              </span>
              <span class="font-mono font-semibold">{{ simPct }}%</span>
            </div>
            <input type="range" min="50" max="99" step="1" v-model.number="simPct" class="w-full accent-amber-500" />
          </div>

          <div>
            <div class="mb-1 flex items-center justify-between text-[11px]">
              <span class="flex items-center gap-1">
                误标最小跨类占比
                <HoverHelpTip :label="RELATION_DRAWER_PARAM_HELP.mislabelThreshold.label" width-class="w-64" placement="floating-left">
                  <p class="text-muted-foreground">{{ RELATION_DRAWER_PARAM_HELP.mislabelThreshold.body }}</p>
                </HoverHelpTip>
              </span>
              <span class="font-mono font-semibold">{{ mislabelPct }}%</span>
            </div>
            <input type="range" min="30" max="90" step="5" v-model.number="mislabelPct" class="w-full accent-rose-500" />
          </div>
        </div>
      </div>
    </div>
  </aside>
</template>

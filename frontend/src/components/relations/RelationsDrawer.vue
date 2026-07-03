<script setup lang="ts">
import {computed} from 'vue';
import {
  Activity, BarChart3, GitBranch, Grid3x3, LayoutGrid, Network,
  RefreshCw, ScatterChart, ShieldAlert, Sparkles, Star, X,
} from 'lucide-vue-next';

export type ViewKey =
  | 'distribution' | 'anomaly'
  | 'headline' | 'confusion' | 'centroid' | 'dendrogram'
  | 'graph' | 'warnings' | 'perclass' | 'mislabels';

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
const groups: {title: string; items: Item[]}[] = [
  {
    title: '基础视图',
    items: [
      {key: 'distribution', label: '分布散点', icon: ScatterChart, needsRelations: false},
      {key: 'anomaly', label: '异常网格', icon: LayoutGrid, needsRelations: false},
    ],
  },
  {
    title: '关系分析',
    items: [
      {key: 'headline', label: '总览质量分', icon: Activity, needsRelations: true},
      {key: 'confusion', label: '混淆矩阵热力图', icon: Grid3x3, needsRelations: true},
      {key: 'centroid', label: '类心相似度热力图', icon: BarChart3, needsRelations: true},
      {key: 'dendrogram', label: '层次聚类树状图', icon: GitBranch, needsRelations: true},
      {key: 'graph', label: '类别关系图', icon: Network, needsRelations: true},
      {key: 'warnings', label: '警告对列表', icon: ShieldAlert, needsRelations: true},
      {key: 'perclass', label: '每类质量卡片', icon: Star, needsRelations: true},
      {key: 'mislabels', label: '疑似误标清单', icon: Sparkles, needsRelations: true},
    ],
  },
];

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
</script>

<template>
  <Teleport to="body">
    <Transition name="fade">
      <div v-if="open" class="fixed inset-0 z-[80]" @click.self="emit('close')">
        <div class="absolute inset-0 bg-black/30 backdrop-blur-[1px]" />
        <aside
          class="absolute right-0 top-0 flex h-full w-[320px] flex-col border-l border-border bg-background shadow-2xl"
          @click.stop
        >
          <div class="flex items-center justify-between border-b border-border px-4 py-3">
            <h3 class="text-sm font-bold">分析中心</h3>
            <button type="button" class="rounded p-1 text-muted-foreground hover:bg-secondary/30" @click="emit('close')">
              <X class="h-4 w-4" />
            </button>
          </div>

          <div class="flex-1 overflow-y-auto p-3">
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

            <!-- Parameter controls -->
            <div class="mt-2 space-y-4 rounded-xl border border-border bg-secondary/5 p-3">
              <p class="text-[10px] font-bold uppercase tracking-wide text-muted-foreground">参数</p>

              <div>
                <div class="mb-1 flex items-center justify-between text-[11px]">
                  <span>邻居数 k</span>
                  <span class="font-mono font-semibold">{{ k }}</span>
                </div>
                <input
                  type="range" min="3" max="30" step="1" :value="k"
                  class="w-full accent-primary"
                  @input="emit('update:k', Number(($event.target as HTMLInputElement).value))"
                  @change="emit('reloadK')"
                />
                <p class="mt-0.5 text-[9px] text-muted-foreground">影响 kNN 混淆矩阵与误标判定（改动后重算）。</p>
              </div>

              <div>
                <div class="mb-1 flex items-center justify-between text-[11px]">
                  <span>混淆警告阈值</span>
                  <span class="font-mono font-semibold">{{ confPct }}%</span>
                </div>
                <input type="range" min="10" max="90" step="5" v-model.number="confPct" class="w-full accent-rose-500" />
              </div>

              <div>
                <div class="mb-1 flex items-center justify-between text-[11px]">
                  <span>相似度警告阈值</span>
                  <span class="font-mono font-semibold">{{ simPct }}%</span>
                </div>
                <input type="range" min="50" max="99" step="1" v-model.number="simPct" class="w-full accent-amber-500" />
              </div>

              <div>
                <div class="mb-1 flex items-center justify-between text-[11px]">
                  <span>误标最小跨类占比</span>
                  <span class="font-mono font-semibold">{{ mislabelPct }}%</span>
                </div>
                <input type="range" min="30" max="90" step="5" v-model.number="mislabelPct" class="w-full accent-rose-500" />
              </div>
            </div>
          </div>
        </aside>
      </div>
    </Transition>
  </Teleport>
</template>

<script setup lang="ts">
import {computed, ref, watch} from 'vue';
import {categoryChartColor} from '@/lib/api';

const props = defineProps<{
  labels: string[];
  counts: number[];
  /** symmetric strength matrix in [0,1]; edges drawn when >= edgeThreshold */
  centroidSim: number[][];
  edgeThreshold: number;
}>();
const emit = defineEmits<{highlight: [labels: string[]]}>();

const W = 720;
const H = 460;

interface Node {x: number; y: number; r: number}
const nodes = ref<Node[]>([]);

/** Lightweight force layout: hub repulsion + similarity-weighted springs. */
function layout() {
  const n = props.labels.length;
  if (n === 0) {
    nodes.value = [];
    return;
  }
  const cx = W / 2;
  const cy = H / 2;
  const maxCount = Math.max(1, ...props.counts);
  const pos: Node[] = props.labels.map((_l, i) => ({
    x: cx + Math.cos((2 * Math.PI * i) / n) * 160,
    y: cy + Math.sin((2 * Math.PI * i) / n) * 160,
    r: 10 + 18 * Math.sqrt((props.counts[i] ?? 1) / maxCount),
  }));

  for (let iter = 0; iter < 260; iter++) {
    const fx = new Array(n).fill(0);
    const fy = new Array(n).fill(0);
    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        let dx = pos[i].x - pos[j].x;
        let dy = pos[i].y - pos[j].y;
        let dist = Math.hypot(dx, dy) || 0.01;
        dx /= dist;
        dy /= dist;
        // repulsion
        const rep = 9000 / (dist * dist);
        fx[i] += dx * rep; fy[i] += dy * rep;
        fx[j] -= dx * rep; fy[j] -= dy * rep;
        // spring: stronger similarity pulls closer (rest length shrinks with sim)
        const sim = Math.max(0, props.centroidSim?.[i]?.[j] ?? 0);
        const rest = 260 - sim * 200;
        const spring = (dist - rest) * 0.02 * (0.3 + sim);
        fx[i] -= dx * spring; fy[i] -= dy * spring;
        fx[j] += dx * spring; fy[j] += dy * spring;
      }
    }
    const damp = 0.85;
    for (let i = 0; i < n; i++) {
      pos[i].x += fx[i] * damp * 0.1;
      pos[i].y += fy[i] * damp * 0.1;
      // gentle centering + bounds
      pos[i].x += (cx - pos[i].x) * 0.01;
      pos[i].y += (cy - pos[i].y) * 0.01;
      pos[i].x = Math.max(pos[i].r + 40, Math.min(W - pos[i].r - 40, pos[i].x));
      pos[i].y = Math.max(pos[i].r + 20, Math.min(H - pos[i].r - 20, pos[i].y));
    }
  }
  nodes.value = pos;
}

watch(
  () => [props.labels, props.centroidSim],
  layout,
  {immediate: true, deep: true},
);

const edges = computed(() => {
  const out: {i: number; j: number; sim: number}[] = [];
  const n = props.labels.length;
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      const sim = props.centroidSim?.[i]?.[j] ?? 0;
      if (sim >= props.edgeThreshold) out.push({i, j, sim});
    }
  }
  return out;
});
</script>

<template>
  <div class="w-full">
    <svg :viewBox="`0 0 ${W} ${H}`" class="w-full" :style="{maxHeight: '480px'}">
      <g v-if="nodes.length === labels.length">
        <line
          v-for="(e, k) in edges"
          :key="'e' + k"
          :x1="nodes[e.i].x" :y1="nodes[e.i].y"
          :x2="nodes[e.j].x" :y2="nodes[e.j].y"
          :stroke="e.sim >= 0.9 ? '#f43f5e' : '#a1a1aa'"
          :stroke-width="1 + e.sim * 6"
          :stroke-opacity="0.35 + e.sim * 0.5"
          class="cursor-pointer"
          @click="emit('highlight', [labels[e.i], labels[e.j]])"
        />
        <g
          v-for="(nd, i) in nodes"
          :key="'n' + i"
          class="cursor-pointer"
          @click="emit('highlight', [labels[i]])"
        >
          <circle :cx="nd.x" :cy="nd.y" :r="nd.r" :fill="categoryChartColor(i, labels.length)" fill-opacity="0.85" stroke="#fff" stroke-width="1.5" />
          <text :x="nd.x" :y="nd.y + nd.r + 12" text-anchor="middle" class="fill-foreground" style="font-size: 10px; font-weight: 600">{{ labels[i] }}</text>
          <text :x="nd.x" :y="nd.y + 3" text-anchor="middle" fill="#fff" style="font-size: 9px; font-weight: 700">{{ counts[i] }}</text>
        </g>
      </g>
    </svg>
    <p class="mt-1 text-center text-[10px] text-muted-foreground">
      节点大小 = 样本数；连线粗细 = 类心相似度（越粗越相似）。<span class="text-rose-500">红线</span>为高相似警告。
    </p>
  </div>
</template>

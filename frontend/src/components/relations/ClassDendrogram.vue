<script setup lang="ts">
import {computed} from 'vue';
import {categoryChartColor, type RelationsLinkageNode} from '@/lib/api';

const props = defineProps<{
  root: RelationsLinkageNode | null;
  labels: string[];
  /** Merge heights below this = over-split (drawn in rose). */
  mergeThreshold: number;
}>();
const emit = defineEmits<{highlight: [labels: string[]]}>();

const W = 800;
const H = 420;
const PAD = {top: 24, right: 24, bottom: 90, left: 44};

interface Positioned {
  node: RelationsLinkageNode;
  x: number;
  height: number;
  children: Positioned[];
  leaves: string[];
}

const layout = computed(() => {
  const root = props.root;
  if (!root) return null;
  const leaves: {name: string}[] = [];
  let maxH = 0;

  function collectMax(n: RelationsLinkageNode) {
    if (n.height > maxH) maxH = n.height;
    n.children.forEach(collectMax);
  }
  collectMax(root);
  maxH = maxH || 1;

  function assign(n: RelationsLinkageNode): Positioned {
    if (!n.children || n.children.length === 0) {
      const x = leaves.length;
      leaves.push({name: n.name ?? '?'});
      return {node: n, x, height: n.height, children: [], leaves: [n.name ?? '?']};
    }
    const kids = n.children.map(assign);
    const x = kids.reduce((s, k) => s + k.x, 0) / kids.length;
    return {node: n, x, height: n.height, children: kids, leaves: kids.flatMap((k) => k.leaves)};
  }
  const positioned = assign(root);
  const nLeaves = leaves.length || 1;

  const plotW = W - PAD.left - PAD.right;
  const plotH = H - PAD.top - PAD.bottom;
  const xOf = (xi: number) => PAD.left + (nLeaves === 1 ? plotW / 2 : (xi / (nLeaves - 1)) * plotW);
  // higher merge height -> nearer the top
  const yOf = (h: number) => PAD.top + plotH - (h / maxH) * plotH;

  interface Seg {x1: number; y1: number; x2: number; y2: number; over: boolean}
  const segs: Seg[] = [];

  function draw(p: Positioned) {
    if (p.children.length === 0) return;
    const py = yOf(p.height);
    const over = p.height < props.mergeThreshold;
    for (const k of p.children) {
      const kx = xOf(k.x);
      const ky = k.children.length === 0 ? PAD.top + plotH : yOf(k.height);
      // vertical from child up to parent height
      segs.push({x1: kx, y1: ky, x2: kx, y2: py, over});
      draw(k);
    }
    // horizontal connector across children at parent height
    const xs = p.children.map((k) => xOf(k.x));
    segs.push({x1: Math.min(...xs), y1: py, x2: Math.max(...xs), y2: py, over});
  }
  draw(positioned);

  const leafPos = leaves.map((l, i) => ({
    name: l.name,
    x: xOf(i),
    y: PAD.top + plotH,
    idx: props.labels.indexOf(l.name),
  }));

  return {segs, leafPos, maxH};
});
</script>

<template>
  <div class="w-full">
    <div v-if="!layout" class="py-12 text-center text-sm text-muted-foreground">类别不足，无法构建层次树。</div>
    <template v-else>
      <svg :viewBox="`0 0 ${W} ${H}`" class="w-full" :style="{maxHeight: '460px'}">
        <line
          v-for="(s, i) in layout.segs"
          :key="i"
          :x1="s.x1" :y1="s.y1" :x2="s.x2" :y2="s.y2"
          :stroke="s.over ? '#f43f5e' : '#71717a'"
          :stroke-width="s.over ? 2.5 : 1.5"
          stroke-linecap="round"
        />
        <g v-for="lp in layout.leafPos" :key="lp.name">
          <circle :cx="lp.x" :cy="lp.y" r="5" :fill="categoryChartColor(lp.idx, labels.length)" stroke="#fff" stroke-width="1" />
          <text
            :x="lp.x" :y="lp.y + 14"
            text-anchor="end"
            :transform="`rotate(-40 ${lp.x} ${lp.y + 14})`"
            class="fill-muted-foreground"
            style="font-size: 10px; cursor: pointer"
            @click="emit('highlight', [lp.name])"
          >{{ lp.name }}</text>
        </g>
      </svg>
      <p class="mt-2 text-center text-[10px] text-muted-foreground">
        合并高度 = 1 − 类心相似度。<span class="text-rose-500">红色低合并</span>表示两类难以区分（过度拆分信号）。
      </p>
    </template>
  </div>
</template>

<script setup lang="ts">
import {computed} from 'vue';
import {
  type AnnotationOverlayRegion,
  buildLegendGroups,
  colorWithAlpha,
  polygonPointsAttr,
  regionDisplayLabel,
} from '@/components/annotation/overlayUtils';

const props = withDefaults(
  defineProps<{
    regions: AnnotationOverlayRegion[];
    colorForClass: (classId: string | null | undefined, isSubtract?: boolean) => string;
    labelForClass?: (classId: string | null | undefined) => string;
    zoom?: number;
    /** Show mask polygons and the top-left legend panel. */
    showLabels?: boolean;
    roundedClass?: string;
    /** Base stroke width in 0–1 viewBox units (before zoom compensation). */
    strokeWidth?: number;
    /**
     * Compensate legend size when this overlay is inside a CSS-scaled parent
     * (e.g. large source images shrunk with transform: scale()). 1 = no change.
     */
    legendScale?: number;
    /** Natural image width in pixels — used for area/center legend stats. */
    imageWidth?: number;
    /** Natural image height in pixels — used for area/center legend stats. */
    imageHeight?: number;
  }>(),
  {
    zoom: 1,
    showLabels: true,
    roundedClass: '',
    strokeWidth: 0.004,
    legendScale: 1,
    imageWidth: 0,
    imageHeight: 0,
  },
);

const decorated = computed(() =>
  props.regions.map((region, index) => {
    const color = props.colorForClass(region.classId, !!region.isSubtract);
    const label = regionDisplayLabel(region, props.labelForClass);
    return {
      key: index,
      region,
      color,
      label,
      fill: region.isSubtract ? 'rgba(148,163,184,0.25)' : colorWithAlpha(color, 0.33),
      stroke: region.isSubtract ? '#94a3b8' : color,
    };
  }),
);

const legendImageSize = computed(() => {
  if (props.imageWidth > 0 && props.imageHeight > 0) {
    return {width: props.imageWidth, height: props.imageHeight};
  }
  return null;
});

const legendGroups = computed(() =>
  buildLegendGroups(props.regions, props.colorForClass, props.labelForClass, legendImageSize.value),
);

const effectiveStroke = computed(() => props.strokeWidth / Math.max(props.zoom, 0.35));

const legendPanelStyle = computed(() => {
  const scale = props.legendScale;
  if (!Number.isFinite(scale) || Math.abs(scale - 1) < 0.02) return undefined;
  return {
    transform: `scale(${scale})`,
    transformOrigin: 'top left',
  };
});
</script>

<template>
  <template v-if="regions.length">
    <svg
      v-if="showLabels"
      class="pointer-events-none absolute inset-0 h-full w-full"
      :class="roundedClass"
      viewBox="0 0 1 1"
      preserveAspectRatio="none"
    >
      <polygon
        v-for="item in decorated"
        :key="item.key"
        :points="polygonPointsAttr(item.region.points)"
        :fill="item.fill"
        :stroke="item.stroke"
        :stroke-width="effectiveStroke"
        stroke-linejoin="round"
      />
    </svg>

    <div
      v-if="showLabels && legendGroups.length"
      class="annotation-legend pointer-events-none absolute left-1 top-1 z-10 space-y-0.5"
      :style="legendPanelStyle"
    >
      <div
        v-for="group in legendGroups"
        :key="`${group.label}-${group.color}`"
        class="leading-none text-white"
      >
        <div
          v-if="group.items.length === 1"
          class="flex max-w-full items-center gap-1 whitespace-nowrap text-[8px]"
        >
          <span
            class="h-1.5 w-1.5 shrink-0 rounded-full ring-1 ring-black/40"
            :style="{backgroundColor: group.color}"
          />
          <span class="max-w-[4.5rem] shrink truncate font-semibold">{{ group.label }}</span>
          <span class="font-mono text-white/85">{{ group.items[0].statsLine }}</span>
        </div>
        <template v-else>
          <div class="flex items-center gap-1 whitespace-nowrap text-[8px] font-semibold">
            <span
              class="h-1.5 w-1.5 shrink-0 rounded-full ring-1 ring-black/40"
              :style="{backgroundColor: group.color}"
            />
            <span class="max-w-[5rem] truncate">{{ group.label }}</span>
            <span class="shrink-0 font-normal text-white/60">×{{ group.items.length }}</span>
          </div>
          <div
            v-for="entry in group.items"
            :key="entry.index"
            class="whitespace-nowrap pl-2.5 font-mono text-[8px] text-white/85"
          >
            {{ entry.statsLine }}
          </div>
        </template>
      </div>
    </div>
  </template>
</template>

<style scoped>
.annotation-legend {
  text-shadow:
    0 0 3px rgba(0, 0, 0, 0.95),
    0 1px 4px rgba(0, 0, 0, 0.85);
}
</style>

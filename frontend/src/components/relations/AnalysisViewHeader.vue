<script setup lang="ts">
import {analysisViewLabel, type ViewKey} from '@/components/relations/analysisViews';
import HoverHelpTip from '@/components/relations/HoverHelpTip.vue';
import {RELATION_VIEW_HELP} from '@/components/relations/relationViewHelp';

const props = defineProps<{
  view: ViewKey;
}>();

const help = RELATION_VIEW_HELP[props.view];
</script>

<template>
  <div class="space-y-1">
    <div class="flex items-center gap-1.5">
      <h3 class="text-sm font-semibold">{{ analysisViewLabel(view) }}</h3>
      <HoverHelpTip v-if="help" :label="`${analysisViewLabel(view)}指标说明`">
        <p v-if="help.intro" class="mb-2 font-medium text-foreground/90">{{ help.intro }}</p>
        <dl class="space-y-2">
          <div v-for="sec in help.sections" :key="sec.title">
            <dt class="font-semibold text-foreground">{{ sec.title }}</dt>
            <dd class="text-muted-foreground">{{ sec.body }}</dd>
          </div>
        </dl>
        <p v-if="help.footer" class="mt-2 text-[10px] text-muted-foreground/80">{{ help.footer }}</p>
      </HoverHelpTip>
    </div>
    <slot />
  </div>
</template>

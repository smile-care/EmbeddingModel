<script setup lang="ts">
import {computed, nextTick, onUnmounted, ref, watch} from 'vue';
import {
  AlertCircle,
  ArrowLeft,
  Check,
  CheckCircle2,
  Clock,
  Database,
  ExternalLink,
  Layers,
  Minus,
  MoreVertical,
  Play,
  Plus,
  StopCircle,
  Terminal,
  Trash2,
  X,
} from 'lucide-vue-next';
import VisualizationDashboard from '@/components/VisualizationDashboard.vue';
import {cn} from '@/lib/utils';
import {DatasetsApi, ExperimentsApi, staticUrl, type AnnotationRegion, type CropImage as ApiCropImage, type DatasetImage as ApiDatasetImage} from '@/lib/api';

interface AnnotationLabel {
  label_id: number;
  points: [number, number][];
  isSubtract: boolean;
}

interface Annotation {
  id: string;
  shapeType?: string | null;
  networkType?: string | null;
  labels?: AnnotationLabel[] | null;
}

type DatasetImage = ApiDatasetImage;

interface DatasetCategory {
  id: string;
  name: string;
}

interface Dataset {
  id: string;
  name: string;
  categories?: DatasetCategory[];
}

type CropImage = ApiCropImage;

const experiments = ref<any[]>([]);
const selectedExpId = ref<string | null>(null);
const isCreating = ref(false);

const taskName = ref('');
const selectedDataset = ref<Dataset | null>(null);
const selectedCategories = ref<string[]>([]);
const selectedCropIds = ref<string[]>([]);
const trainingStatus = ref<'idle' | 'running' | 'stopped' | 'completed'>('idle');
const progress = ref(0);
const activeExpId = ref<string | null>(null);
const experimentMenuId = ref<string | null>(null);
const experimentMenuPos = ref<{top: number; left: number} | null>(null);

/** Full-screen crop preview (data selection): annotation overlay + source trace */
const previewModalCrop = ref<CropImage | null>(null);
const previewModalView = ref<'crop' | 'source'>('crop');
const showPreviewAnnotations = ref(true);
const previewZoom = ref(1);
const PREVIEW_ZOOM_MIN = 0.35;
const PREVIEW_ZOOM_MAX = 4;
const previewWheelRef = ref<HTMLDivElement | null>(null);
/** Native scroll only when scaled image exceeds the viewport */
const experimentsPreviewCanScroll = ref(false);

function measureExperimentsPreviewScroll() {
  const el = previewWheelRef.value;
  if (!el) {
    experimentsPreviewCanScroll.value = false;
    return;
  }
  experimentsPreviewCanScroll.value =
    el.scrollWidth > el.clientWidth + 2 || el.scrollHeight > el.clientHeight + 2;
}

const detailImages = ref<DatasetImage[]>([]);

function findSourceDatasetImage(sourceImageId: string): DatasetImage | null {
  return detailImages.value.find((i) => i.id === sourceImageId) ?? null;
}

function cropToPreviewAnnotation(crop: CropImage): Annotation | null {
  if (!crop.cropAnnotation?.length) return null;
  return {
    id: crop.id,
    labels: crop.cropAnnotation.map((lbl, i) => ({
      label_id: i,
      points: lbl.points as [number, number][],
      isSubtract: lbl.isSubtract,
    })),
  };
}

function regionToLabel(r: AnnotationRegion, i: number): AnnotationLabel {
  return {label_id: i, points: r.points as [number, number][], isSubtract: r.isSubtract};
}

function sourceAnnotationForCrop(crop: CropImage): Annotation | null {
  const img = findSourceDatasetImage(crop.sourceImageId);
  if (!img?.regions?.length) return null;
  const idx = crop.instanceIndex;
  const target = idx != null && idx >= 0 && idx < img.regions.length ? img.regions[idx] : null;
  const labels = target ? [regionToLabel(target, idx)] : img.regions.map(regionToLabel);
  return {id: img.id, labels};
}

const previewAnnotationModal = computed((): Annotation | null => {
  const c = previewModalCrop.value;
  if (!c) return null;
  return previewModalView.value === 'crop' ? cropToPreviewAnnotation(c) : sourceAnnotationForCrop(c);
});

const previewDisplayUrl = computed(() => {
  const c = previewModalCrop.value;
  if (!c) return null;
  if (previewModalView.value === 'crop') return staticUrl(c.url);
  const src = findSourceDatasetImage(c.sourceImageId);
  return staticUrl(src?.url ?? c.url);
});

const canTraceSource = computed(() => {
  const c = previewModalCrop.value;
  if (!c || !c.sourceImageId) return false;
  return !!findSourceDatasetImage(c.sourceImageId);
});

function openCropPreviewModal(crop: CropImage) {
  previewModalCrop.value = crop;
  previewModalView.value = 'crop';
  showPreviewAnnotations.value = true;
  previewZoom.value = 1;
}

function closeCropPreviewModal() {
  previewModalCrop.value = null;
  previewModalView.value = 'crop';
}

function openSourceFromPreview() {
  if (!previewModalCrop.value || !canTraceSource.value) return;
  previewModalView.value = 'source';
}

function backToCropPreview() {
  previewModalView.value = 'crop';
}

const availableDatasets = ref<Dataset[]>([]);

// ── Crop images for Data Selection ──────────────────────────────────────────
// { categoryId -> CropImage[] | 'loading' }
const cropsByCat = ref<Record<string, CropImage[] | 'loading'>>({});

async function loadCropsForCategory(cat: DatasetCategory) {
  if (cropsByCat.value[cat.id] !== undefined) return;
  cropsByCat.value = {...cropsByCat.value, [cat.id]: 'loading'};
  try {
    const data = await DatasetsApi.getClassCrops(cat.id);
    cropsByCat.value = {...cropsByCat.value, [cat.id]: data};
  } catch {
    cropsByCat.value = {...cropsByCat.value, [cat.id]: []};
  }
}

/** Load full detail for a dataset chosen for training (classes + images for tracing). */
async function selectDatasetForTraining(ds: {id: string; name: string}) {
  selectedCategories.value = [];
  selectedCropIds.value = [];
  detailImages.value = [];
  try {
    const full = await DatasetsApi.get(ds.id);
    detailImages.value = full.images;
    selectedDataset.value = {
      id: full.id,
      name: full.name,
      categories: full.defectClasses.map((c) => ({id: c.id, name: c.name})),
    };
  } catch (e) {
    console.error('failed to load dataset detail', e);
    selectedDataset.value = {id: ds.id, name: ds.name, categories: []};
  }
}

function getCategoryCrops(catId: string): CropImage[] {
  const v = cropsByCat.value[catId];
  return Array.isArray(v) ? v : [];
}

function isCatLoading(catId: string) {
  return cropsByCat.value[catId] === 'loading';
}

function closeExperimentMenu() {
  experimentMenuId.value = null;
  experimentMenuPos.value = null;
}

function isSpaceKey(e: KeyboardEvent) {
  return e.key === ' ' || e.key === 'Space' || e.code === 'Space';
}

function isTypingLikeTarget(target: EventTarget | null) {
  const el = target as HTMLElement | null;
  if (!el) return false;
  const tag = el.tagName;
  return tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT' || el.isContentEditable;
}

void fetchExperiments();
void fetchAvailableDatasets();

watch(experimentMenuId, (id, _oldId, onCleanup) => {
  if (!id) return;
  const onScrollOrResize = () => {
    experimentMenuId.value = null;
    experimentMenuPos.value = null;
  };
  window.addEventListener('scroll', onScrollOrResize, true);
  window.addEventListener('resize', onScrollOrResize);
  onCleanup(() => {
    window.removeEventListener('scroll', onScrollOrResize, true);
    window.removeEventListener('resize', onScrollOrResize);
  });
});

watch(
  previewModalCrop,
  (crop, _prev, onCleanup) => {
    if (!crop) return;
    previewZoom.value = 1;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') closeCropPreviewModal();
      if (isSpaceKey(e)) {
        if (isTypingLikeTarget(e.target)) return;
        e.preventDefault();
        if (previewAnnotationModal.value?.labels?.length) {
          showPreviewAnnotations.value = !showPreviewAnnotations.value;
        }
      }
    };
    window.addEventListener('keydown', onKey);
    const prevOverflow = document.body.style.overflow;
    document.body.style.overflow = 'hidden';
    let removeWheel: (() => void) | undefined;
    let removeResize: (() => void) | undefined;
    void nextTick(() => {
      const el = previewWheelRef.value;
      if (!el) return;
      const onWheel = (e: WheelEvent) => {
        e.preventDefault();
        previewZoom.value = Math.min(
          PREVIEW_ZOOM_MAX,
          Math.max(PREVIEW_ZOOM_MIN, previewZoom.value - e.deltaY * 0.002),
        );
      };
      el.addEventListener('wheel', onWheel, {passive: false});
      removeWheel = () => el.removeEventListener('wheel', onWheel);
      measureExperimentsPreviewScroll();
      const ro = new ResizeObserver(() => measureExperimentsPreviewScroll());
      ro.observe(el);
      removeResize = () => ro.disconnect();
    });
    onCleanup(() => {
      window.removeEventListener('keydown', onKey);
      document.body.style.overflow = prevOverflow;
      removeWheel?.();
      removeResize?.();
      experimentsPreviewCanScroll.value = false;
    });
  },
);

watch(previewZoom, () => {
  if (!previewModalCrop.value) return;
  void nextTick(() => measureExperimentsPreviewScroll());
});

watch(previewModalView, () => {
  if (!previewModalCrop.value) return;
  void nextTick(() => measureExperimentsPreviewScroll());
});

async function fetchAvailableDatasets() {
  try {
    const data = await DatasetsApi.list();
    availableDatasets.value = data.map((d) => ({id: d.id, name: d.name}));
  } catch (error) {
    console.error('Failed to fetch available datasets:', error);
  }
}

// When the selected dataset changes, reset selections and pre-load all crop images
watch(selectedDataset, (ds) => {
  selectedCategories.value = [];
  selectedCropIds.value = [];
  cropsByCat.value = {};
  if (!ds) return;
  for (const cat of ds.categories ?? []) {
    void loadCropsForCategory(cat);
  }
});

async function fetchExperiments() {
  try {
    experiments.value = await ExperimentsApi.list();
  } catch (error) {
    console.error('Failed to fetch experiments:', error);
  }
}

const selectedExp = computed(() => experiments.value.find((e) => e.id === selectedExpId.value));

let pollTimer: ReturnType<typeof setInterval> | null = null;
watch(
  [activeExpId, trainingStatus],
  () => {
    if (pollTimer) {
      clearInterval(pollTimer);
      pollTimer = null;
    }
    if (!activeExpId.value || trainingStatus.value !== 'running') return;
    pollTimer = setInterval(async () => {
      try {
        if (!activeExpId.value) return;
        const d = await ExperimentsApi.get(activeExpId.value);
        progress.value = Math.round(d.progress ?? 0);
        if (d.status === 'Completed' || d.status === 'Failed' || d.status === 'Stopped') {
          trainingStatus.value = 'completed';
          activeExpId.value = null;
          if (pollTimer) {
            clearInterval(pollTimer);
            pollTimer = null;
          }
          void fetchExperiments();
        }
      } catch {
        /* ignore */
      }
    }, 2000);
    return () => {
      if (pollTimer) {
        clearInterval(pollTimer);
        pollTimer = null;
      }
    };
  },
  {immediate: true},
);

onUnmounted(() => {
  if (pollTimer) clearInterval(pollTimer);
});

const eachSelectedCategoryHasTrainingCrop = computed(() => {
  if (!selectedDataset.value?.categories?.length) return true;
  const picked = new Set(selectedCropIds.value);
  for (const catName of selectedCategories.value) {
    const cat = selectedDataset.value.categories.find((c) => c.name === catName);
    if (!cat) return false;
    // check against crop IDs
    const crops = getCategoryCrops(cat.id);
    const hasOne = crops.some((cr) => picked.has(cr.id));
    if (!hasOne) return false;
  }
  return true;
});

async function handleStartTraining() {
  if (
    !taskName.value ||
    !selectedDataset.value ||
    selectedCategories.value.length < 2 ||
    selectedCropIds.value.length === 0 ||
    !eachSelectedCategoryHasTrainingCrop.value
  ) {
    return;
  }
  trainingStatus.value = 'running';
  progress.value = 0;
  try {
    const exp = await ExperimentsApi.create({
      name: taskName.value,
      model: 'EmbeddingModel',
      dataset: selectedDataset.value.name,
      datasetId: selectedDataset.value.id,
      config: {trainingCropIds: selectedCropIds.value},
    });
    await ExperimentsApi.train(exp.id);
    activeExpId.value = exp.id;
  } catch (error) {
    console.error('Failed to save experiment:', error);
    trainingStatus.value = 'idle';
  }
}

async function openActiveTrainingPage() {
  if (!activeExpId.value) return;
  try {
    const exp = await ExperimentsApi.get(activeExpId.value);
    const idx = experiments.value.findIndex((e) => e.id === exp.id);
    if (idx >= 0) experiments.value[idx] = exp;
    else experiments.value = [exp, ...experiments.value];
  } catch {
    /* ignore */
  }
  isCreating.value = false;
  selectedExpId.value = activeExpId.value;
}

function handleStopTraining() {
  trainingStatus.value = 'stopped';
  activeExpId.value = null;
}

async function handleDeleteExperiment(id: string, name: string) {
  closeExperimentMenu();
  if (!window.confirm(`Delete experiment "${name}" and its saved checkpoint? This cannot be undone.`)) {
    return;
  }
  try {
    await ExperimentsApi.remove(id);
    experiments.value = experiments.value.filter((e) => e.id !== id);
    if (selectedExpId.value === id) {
      selectedExpId.value = null;
    }
    if (activeExpId.value === id) {
      activeExpId.value = null;
      trainingStatus.value = 'idle';
    }
  } catch (error) {
    console.error('Failed to delete experiment:', error);
  }
}

function toggleCategory(catName: string, cropIdsInCategory: string[]) {
  if (selectedCategories.value.includes(catName)) {
    selectedCropIds.value = selectedCropIds.value.filter((id) => !cropIdsInCategory.includes(id));
    selectedCategories.value = selectedCategories.value.filter((c) => c !== catName);
  } else {
    selectedCategories.value = [...selectedCategories.value, catName];
  }
}

function toggleImage(cropId: string, categoryEnabled: boolean) {
  if (!categoryEnabled) return;
  if (selectedCropIds.value.includes(cropId)) {
    selectedCropIds.value = selectedCropIds.value.filter((i) => i !== cropId);
  } else {
    selectedCropIds.value = [...selectedCropIds.value, cropId];
  }
}

function selectAllInCategory(cropIds: string[]) {
  selectedCropIds.value = Array.from(new Set([...selectedCropIds.value, ...cropIds]));
}

function deselectAllInCategory(cropIds: string[]) {
  selectedCropIds.value = selectedCropIds.value.filter((id) => !cropIds.includes(id));
}

function openExperimentMenu(e: MouseEvent, expId: string) {
  if (experimentMenuId.value === expId) {
    closeExperimentMenu();
    return;
  }
  const r = (e.currentTarget as HTMLElement).getBoundingClientRect();
  const mw = 176;
  const pad = 8;
  const gap = 4;
  const menuH = 120;
  let top = r.bottom + gap;
  if (top + menuH > window.innerHeight - pad) {
    top = Math.max(pad, r.top - menuH - gap);
  }
  experimentMenuId.value = expId;
  experimentMenuPos.value = {
    top,
    left: Math.max(pad, Math.min(r.right - mw, window.innerWidth - mw - pad)),
  };
}
</script>

<template>
  <div v-if="selectedExp" class="flex h-full flex-col">
    <div class="flex items-center gap-4 border-b border-border bg-secondary/5 p-6">
      <button
        type="button"
        class="rounded-md p-2 transition-colors hover:bg-secondary/50"
        @click="selectedExpId = null"
      >
        <ArrowLeft class="h-4 w-4" />
      </button>
      <div>
        <h2 class="text-xl font-semibold tracking-tight">{{ selectedExp.name }}</h2>
        <p class="text-xs text-muted-foreground">Experiment Dashboard • {{ selectedExp.id }}</p>
      </div>
    </div>
    <div class="flex-1 overflow-y-auto">
      <VisualizationDashboard :experiment-id="selectedExp.id" />
    </div>
  </div>

  <div v-else-if="isCreating" class="flex h-full flex-col gap-6 overflow-y-auto p-6">
    <div class="flex items-center justify-between">
      <div class="flex items-center gap-4">
        <button
          type="button"
          class="rounded-md p-2 transition-colors hover:bg-secondary/50"
          @click="isCreating = false"
        >
          <ArrowLeft class="h-4 w-4" />
        </button>
        <h2 class="text-2xl font-semibold tracking-tight">Create Training Task</h2>
      </div>
      <div class="flex items-center gap-3">
        <button
          v-if="trainingStatus === 'running'"
          type="button"
          class="flex items-center gap-2 rounded-md bg-rose-500 px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-rose-600"
          @click="handleStopTraining"
        >
          <StopCircle class="h-4 w-4" />
          Stop Training
        </button>
        <button
          v-else
          type="button"
          :disabled="
            !taskName ||
            !selectedDataset ||
            trainingStatus === 'completed' ||
            selectedCategories.length < 2 ||
            selectedCropIds.length === 0 ||
            !eachSelectedCategoryHasTrainingCrop
          "
          :title="
            !selectedDataset
              ? undefined
              : selectedCategories.length < 2
                ? 'Enable at least two categories'
                : !eachSelectedCategoryHasTrainingCrop
                  ? 'Each enabled category must have at least one crop selected, or turn that category off'
                  : selectedCropIds.length === 0
                    ? 'Use the corner checkbox on each crop to include it in training'
                    : undefined
          "
          class="flex items-center gap-2 rounded-md bg-primary px-4 py-2 text-sm font-medium text-primary-foreground transition-colors hover:bg-primary/90 disabled:cursor-not-allowed disabled:opacity-50"
          @click="handleStartTraining"
        >
          <Play class="h-4 w-4" />
          {{ trainingStatus === 'completed' ? 'Training Completed' : 'Start Training' }}
        </button>
      </div>
    </div>

    <div class="grid grid-cols-1 gap-8 lg:grid-cols-3">
      <div class="space-y-6 lg:col-span-1">
        <div class="space-y-4 rounded-xl border border-border bg-secondary/5 p-5">
          <h3 class="text-sm font-semibold uppercase tracking-wider text-muted-foreground">Basic Config</h3>
          <div class="space-y-2">
            <label class="text-xs font-medium text-muted-foreground">Task Name</label>
            <input
              v-model="taskName"
              type="text"
              placeholder="e.g. ResNet-FineTune-V1"
              class="w-full rounded-md border border-border bg-background px-3 py-2 text-sm focus:outline-none focus:ring-1 focus:ring-primary"
            />
          </div>
          <div class="space-y-2">
            <label class="text-xs font-medium text-muted-foreground">Select Dataset</label>
            <div class="grid grid-cols-1 gap-2">
              <button
                v-for="ds in availableDatasets"
                :key="ds.id"
                type="button"
                class="flex items-center gap-3 rounded-lg border p-3 text-left text-sm transition-all"
                :class="
                  cn(
                    selectedDataset?.id === ds.id
                      ? 'border-primary bg-primary/5 ring-1 ring-primary'
                      : 'border-border hover:bg-secondary/30',
                  )
                "
                @click="selectDatasetForTraining(ds)"
              >
                <Database
                  class="h-4 w-4"
                  :class="selectedDataset?.id === ds.id ? 'text-primary' : 'text-muted-foreground'"
                />
                {{ ds.name }}
              </button>
            </div>
          </div>
        </div>

        <div
          v-if="trainingStatus !== 'idle'"
          :role="activeExpId ? 'button' : undefined"
          :tabindex="activeExpId ? 0 : undefined"
          :class="[
            'space-y-4 rounded-xl border border-border bg-secondary/5 p-5',
            activeExpId
              ? 'cursor-pointer transition-colors hover:bg-secondary/10 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-background'
              : '',
          ]"
          @click="activeExpId ? openActiveTrainingPage() : undefined"
          @keydown.enter.prevent="activeExpId ? openActiveTrainingPage() : undefined"
          @keydown.space.prevent="activeExpId ? openActiveTrainingPage() : undefined"
        >
          <div class="flex items-center justify-between">
            <h3 class="text-sm font-semibold uppercase tracking-wider text-muted-foreground">
              Training Progress
            </h3>
            <span class="font-mono text-xs">{{ progress }}%</span>
          </div>
          <div class="h-2 w-full overflow-hidden rounded-full bg-secondary">
            <div
              class="h-full bg-primary transition-[width] duration-300 ease-out"
              :style="{width: `${progress}%`}"
            />
          </div>
          <div class="flex items-center gap-2 text-xs text-muted-foreground">
            <Clock class="h-3 w-3" />
            <span>Estimated time remaining: {{ Math.max(0, 100 - progress) }}s</span>
          </div>
          <p v-if="activeExpId" class="text-[11px] text-primary/80">点击此卡片进入训练页面</p>
        </div>
      </div>

      <div class="space-y-6 lg:col-span-2">
        <div v-if="selectedDataset" class="space-y-6">
          <div class="rounded-xl border border-border bg-secondary/5 p-5">
            <div class="mb-4 flex items-center justify-between">
              <h3 class="text-sm font-semibold uppercase tracking-wider text-muted-foreground">
                Data Selection
              </h3>
              <p class="text-xs text-muted-foreground">
                <template v-if="selectedCategories.length < 2">
                  Enable at least 2 categories ({{ selectedCategories.length }}/2)
                </template>
                <template v-else-if="!eachSelectedCategoryHasTrainingCrop">
                  Pick ≥1 crop per enabled category, or disable a category
                </template>
                <template v-else-if="selectedCropIds.length > 0">
                  {{ selectedCropIds.length }} crop(s) for training · {{ selectedCategories.length }}
                  categories on
                </template>
                <template v-else>Use the corner checkbox on thumbnails to add crops</template>
              </p>
            </div>

            <div class="space-y-6">
              <p
                v-if="(selectedDataset.categories ?? []).length === 0"
                class="text-sm text-muted-foreground"
              >
                This dataset has no categories or images yet.
              </p>
              <div v-for="c in selectedDataset.categories ?? []" :key="c.name" class="space-y-3">
                <div class="flex flex-wrap items-center justify-between gap-2">
                  <button
                    type="button"
                    class="group flex min-w-0 flex-1 items-center gap-3 text-left"
                    @click="toggleCategory(c.name, getCategoryCrops(c.id).map((cr) => cr.id))"
                  >
                    <div
                      class="flex h-5 w-5 shrink-0 items-center justify-center rounded-md border-2 transition-all duration-150"
                      :class="
                        selectedCategories.includes(c.name)
                          ? 'border-emerald-500 bg-emerald-500 text-white shadow-sm shadow-emerald-500/25 ring-2 ring-emerald-500/20'
                          : 'border-muted-foreground/40 bg-background group-hover:border-emerald-500/50'
                      "
                    >
                      <Check v-if="selectedCategories.includes(c.name)" class="h-3 w-3 stroke-[3]" />
                    </div>
                    <span class="text-sm font-medium">{{ c.name }}</span>
                    <span class="text-[10px] text-muted-foreground">
                      ({{ getCategoryCrops(c.id).length }} 裁剪图)
                    </span>
                  </button>
                  <div class="flex shrink-0 items-center gap-1.5">
                    <button
                      type="button"
                      :disabled="!selectedCategories.includes(c.name) || getCategoryCrops(c.id).length === 0"
                      class="rounded-md border border-border bg-background px-2 py-1 text-[11px] font-medium text-foreground transition-colors hover:bg-secondary/80 disabled:cursor-not-allowed disabled:opacity-40"
                      @click.stop="selectAllInCategory(getCategoryCrops(c.id).map((cr) => cr.id))"
                    >
                      一键全选
                    </button>
                    <button
                      type="button"
                      :disabled="!selectedCategories.includes(c.name) || getCategoryCrops(c.id).length === 0"
                      class="rounded-md border border-border bg-background px-2 py-1 text-[11px] font-medium text-muted-foreground transition-colors hover:bg-secondary/80 hover:text-foreground disabled:cursor-not-allowed disabled:opacity-40"
                      @click.stop="deselectAllInCategory(getCategoryCrops(c.id).map((cr) => cr.id))"
                    >
                      一键取消全选
                    </button>
                  </div>
                </div>

                <!-- loading state -->
                <div
                  v-if="isCatLoading(c.id)"
                  class="grid grid-cols-4 gap-2 sm:grid-cols-6 md:grid-cols-8"
                >
                  <div
                    v-for="i in 6"
                    :key="i"
                    class="aspect-square animate-pulse rounded-lg bg-secondary/30"
                  />
                </div>

                <!-- no crops yet -->
                <p
                  v-else-if="getCategoryCrops(c.id).length === 0"
                  class="text-xs text-muted-foreground/60"
                >
                  该类别暂无裁剪图，请先在数据集页面生成裁剪图。
                </p>

                <!-- crop grid -->
                <div
                  v-else
                  class="grid grid-cols-4 gap-2 rounded-lg sm:grid-cols-6 md:grid-cols-8"
                  :class="!selectedCategories.includes(c.name) && 'opacity-45'"
                >
                  <div
                    v-for="crop in getCategoryCrops(c.id)"
                    :key="crop.id"
                    class="relative aspect-square overflow-hidden rounded-lg border-2 transition-all"
                    :class="
                      selectedCropIds.includes(crop.id)
                        ? 'border-emerald-500 shadow-md shadow-emerald-500/10 ring-2 ring-emerald-500/30'
                        : 'border-border/60 bg-muted/20'
                    "
                  >
                    <img
                      :src="staticUrl(crop.url)"
                      alt=""
                      draggable="false"
                      class="pointer-events-none h-full w-full select-none object-cover"
                    />
                    <button
                      type="button"
                      class="absolute inset-0 z-10 cursor-zoom-in"
                      aria-label="View larger"
                      @click="openCropPreviewModal(crop)"
                    />
                    <button
                      v-if="selectedCategories.includes(c.name)"
                      type="button"
                      :title="selectedCropIds.includes(crop.id) ? 'Remove from training' : 'Add to training'"
                      class="absolute right-1 top-1 z-20 flex h-7 w-7 items-center justify-center rounded-md border shadow-md transition-colors"
                      :class="
                        selectedCropIds.includes(crop.id)
                          ? 'border-emerald-600 bg-emerald-500 text-white hover:bg-emerald-600'
                          : 'border-border bg-background/95 text-muted-foreground hover:border-emerald-500/60 hover:text-emerald-700'
                      "
                      :aria-pressed="selectedCropIds.includes(crop.id)"
                      @click.stop="toggleImage(crop.id, true)"
                    >
                      <Check v-if="selectedCropIds.includes(crop.id)" class="h-4 w-4 stroke-[3]" />
                      <span v-else class="h-3.5 w-3.5 rounded-sm border-2 border-current opacity-70" />
                    </button>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
        <div
          v-else
          class="flex h-full flex-col items-center justify-center rounded-xl border-2 border-dashed border-border p-12 text-muted-foreground"
        >
          <Layers class="mb-4 h-12 w-12 opacity-20" />
          <p class="text-sm">Select a dataset to begin data selection</p>
        </div>
      </div>
    </div>

    <Teleport to="body">
      <Transition name="fade">
        <div
          v-if="previewModalCrop && previewDisplayUrl"
          class="fixed inset-0 z-[200] flex flex-col bg-black/88 backdrop-blur-sm"
          @click.self="closeCropPreviewModal"
        >
          <div
            class="flex shrink-0 flex-wrap items-center justify-between gap-3 border-b border-white/10 px-4 py-3 text-white"
            @click.stop
          >
            <span class="text-xs text-white/70">
              滚轮缩放 · {{ experimentsPreviewCanScroll ? '拖动平移' : '已适配窗口' }} · Esc 关闭 · 空格切换标注显示
              <span v-if="previewModalView === 'source'" class="text-white/40"> · 当前：原图</span>
              <span v-else class="text-white/40"> · 当前：裁剪图</span>
            </span>
            <div class="flex flex-wrap items-center justify-end gap-2">
              <button
                v-if="previewModalView === 'crop' && canTraceSource"
                type="button"
                class="flex items-center gap-1.5 rounded-lg bg-white/10 px-3 py-1.5 text-xs font-medium text-white/90 backdrop-blur-sm transition-colors hover:bg-white/20"
                title="查看生成该裁剪的原始图像与标注"
                @click="openSourceFromPreview"
              >
                <ExternalLink class="h-3.5 w-3.5" />
                查看原图
              </button>
              <button
                v-if="previewModalView === 'source'"
                type="button"
                class="flex items-center gap-1.5 rounded-lg bg-white/10 px-3 py-1.5 text-xs font-medium text-white/90 backdrop-blur-sm transition-colors hover:bg-white/20"
                @click="backToCropPreview"
              >
                <ArrowLeft class="h-3.5 w-3.5" />
                查看裁剪图
              </button>
              <div
                v-if="previewAnnotationModal?.labels?.length"
                class="flex items-center gap-1 rounded-lg bg-white/10 px-2 py-1 text-[11px] text-white/80"
              >
                <button
                  type="button"
                  class="rounded px-1.5 py-0.5 font-medium transition-colors"
                  :class="showPreviewAnnotations ? 'bg-indigo-500/50 text-indigo-100' : 'text-white/40 hover:text-white'"
                  @click="showPreviewAnnotations = !showPreviewAnnotations"
                >
                  {{ showPreviewAnnotations ? '隐藏' : '显示' }}标注
                </button>
              </div>
              <span class="min-w-[3rem] text-center font-mono text-xs tabular-nums text-white/80">
                {{ Math.round(previewZoom * 100) }}%
              </span>
              <button
                type="button"
                class="rounded-md border border-white/20 p-1.5 hover:bg-white/10"
                aria-label="Zoom out"
                @click="previewZoom = Math.max(PREVIEW_ZOOM_MIN, Math.round((previewZoom - 0.15) * 100) / 100)"
              >
                <Minus class="h-4 w-4" />
              </button>
              <button
                type="button"
                class="rounded-md border border-white/20 p-1.5 hover:bg-white/10"
                aria-label="Zoom in"
                @click="previewZoom = Math.min(PREVIEW_ZOOM_MAX, Math.round((previewZoom + 0.15) * 100) / 100)"
              >
                <Plus class="h-4 w-4" />
              </button>
              <button
                type="button"
                class="rounded-md border border-white/20 p-1.5 hover:bg-white/10"
                aria-label="Close preview"
                @click="closeCropPreviewModal"
              >
                <X class="h-4 w-4" />
              </button>
            </div>
          </div>
          <div
            ref="previewWheelRef"
            :class="[
              'flex min-h-0 flex-1 px-4 pb-8 pt-2',
              experimentsPreviewCanScroll
                ? 'cursor-grab overflow-auto active:cursor-grabbing'
                : 'cursor-default overflow-hidden',
            ]"
            @click.stop
          >
            <div class="flex min-h-full w-full items-center justify-center p-4">
              <div
                class="relative inline-block leading-none select-none"
                :style="{
                  transform: `scale(${previewZoom})`,
                  transformOrigin: 'center center',
                }"
              >
                <img
                  :src="previewDisplayUrl"
                  alt=""
                  draggable="false"
                  class="block max-h-[min(82vh,100%)] max-w-full select-none object-contain"
                  @load="() => void nextTick(() => measureExperimentsPreviewScroll())"
                />
                <svg
                  v-if="previewAnnotationModal?.labels?.length && showPreviewAnnotations"
                  class="pointer-events-none absolute inset-0 h-full w-full"
                  viewBox="0 0 1 1"
                  preserveAspectRatio="none"
                >
                  <polygon
                    v-for="(label, idx) in previewAnnotationModal.labels"
                    :key="idx"
                    :points="label.points.map(([x, y]: [number, number]) => `${x},${y}`).join(' ')"
                    fill="rgba(99, 102, 241, 0.42)"
                    stroke="rgb(199, 210, 254)"
                    :stroke-width="0.0045 / previewZoom"
                    stroke-linejoin="round"
                  />
                </svg>
              </div>
            </div>
          </div>
        </div>
      </Transition>
    </Teleport>
  </div>

  <div v-else class="flex h-full flex-col gap-6 p-6">
    <div class="flex items-center justify-between">
      <div class="flex flex-col gap-1">
        <h2 class="text-2xl font-semibold tracking-tight">Experiment Management</h2>
        <p class="text-sm text-muted-foreground">Track and analyze your model training sessions.</p>
      </div>
      <button
        type="button"
        class="flex items-center gap-2 rounded-md bg-primary px-4 py-2 text-sm font-medium text-primary-foreground transition-colors hover:bg-primary/90"
        @click="isCreating = true"
      >
        <Plus class="h-4 w-4" />
        New Experiment
      </button>
    </div>

    <div class="grid grid-cols-1 gap-6">
      <div
        v-for="exp in experiments"
        :key="exp.id"
        class="flex flex-col rounded-xl border border-border bg-secondary/5"
      >
        <div class="flex items-center justify-between border-b border-border bg-secondary/10 p-4">
          <div class="flex items-center gap-3">
            <div class="flex h-8 w-8 items-center justify-center rounded-lg bg-secondary/30">
              <Terminal class="h-4 w-4 text-muted-foreground" />
            </div>
            <div>
              <h3 class="text-sm font-medium">{{ exp.name }}</h3>
              <p class="text-xs text-muted-foreground">{{ exp.id }}</p>
            </div>
          </div>
          <div class="flex items-center gap-4">
            <div class="flex items-center gap-2">
              <CheckCircle2 v-if="exp.status === 'Completed'" class="h-4 w-4 text-emerald-500" />
              <Clock
                v-else-if="exp.status === 'Running'"
                class="h-4 w-4 animate-pulse text-amber-500"
              />
              <AlertCircle v-else-if="exp.status === 'Failed'" class="h-4 w-4 text-rose-500" />
              <span class="text-xs font-medium">{{ exp.status }}</span>
            </div>
            <div class="h-4 w-px bg-border" />
            <div class="flex flex-col items-end">
              <p class="text-xs text-muted-foreground">Duration</p>
              <p class="text-xs font-medium">{{ exp.duration }}</p>
            </div>
          </div>
        </div>
        <div
          role="button"
          tabindex="0"
          class="grid cursor-pointer grid-cols-4 gap-4 rounded-b-xl p-4 transition-colors hover:bg-secondary/10 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-background"
          @click="selectedExpId = exp.id"
          @keydown.enter.prevent="selectedExpId = exp.id"
          @keydown.space.prevent="selectedExpId = exp.id"
        >
          <div class="flex flex-col gap-1">
            <p class="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">Model</p>
            <p class="text-sm">{{ exp.model }}</p>
          </div>
          <div class="flex flex-col gap-1">
            <p class="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">Dataset</p>
            <p class="text-sm">{{ exp.dataset }}</p>
          </div>
          <div class="flex flex-col gap-1">
            <p class="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">Accuracy</p>
            <p class="text-sm">{{ exp.accuracy }}</p>
          </div>
          <div class="flex min-h-[2rem] items-start justify-end">
            <button
              type="button"
              class="rounded p-1 transition-colors hover:bg-secondary/30"
              aria-label="Open experiment actions"
              :aria-expanded="experimentMenuId === exp.id"
              @click.stop="openExperimentMenu($event, exp.id)"
            >
              <MoreVertical class="h-4 w-4 text-muted-foreground" />
            </button>
            <Teleport to="body">
              <Transition name="fade-scale">
                <div v-if="experimentMenuId === exp.id && experimentMenuPos">
                  <div class="fixed inset-0 z-[50]" @click="closeExperimentMenu" />
                  <div
                    class="fixed z-[60] w-44 overflow-hidden rounded-lg border border-border bg-background shadow-xl"
                    :style="{top: `${experimentMenuPos.top}px`, left: `${experimentMenuPos.left}px`}"
                  >
                    <div class="flex flex-col p-1">
                      <button
                        type="button"
                        class="flex items-center gap-2 rounded-md px-3 py-2 text-left text-xs transition-colors hover:bg-secondary/50"
                        @click="
                          selectedExpId = exp.id;
                          closeExperimentMenu();
                        "
                      >
                        <ExternalLink class="h-3.5 w-3.5" />
                        View Visualization
                      </button>
                      <div class="my-1 h-px bg-border" />
                      <button
                        type="button"
                        class="flex items-center gap-2 rounded-md px-3 py-2 text-left text-xs text-rose-500 transition-colors hover:bg-rose-500/10"
                        @click="handleDeleteExperiment(exp.id, exp.name)"
                      >
                        <Trash2 class="h-3.5 w-3.5" />
                        Delete
                      </button>
                    </div>
                  </div>
                </div>
              </Transition>
            </Teleport>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.fade-enter-active,
.fade-leave-active {
  transition: opacity 0.15s ease;
}
.fade-enter-from,
.fade-leave-to {
  opacity: 0;
}
.fade-scale-enter-active,
.fade-scale-leave-active {
  transition:
    opacity 0.12s ease,
    transform 0.12s ease;
}
.fade-scale-enter-from,
.fade-scale-leave-to {
  opacity: 0;
  transform: scale(0.95) translateY(-4px);
}
</style>

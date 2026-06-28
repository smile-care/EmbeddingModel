<script setup lang="ts">
import {computed, onMounted, onUnmounted, ref} from 'vue';
import {
  ArrowLeft, Database, ExternalLink, ImageIcon, Loader2, MoreVertical, PencilLine,
  Plus, Search, Settings2, Tag, Trash2, Upload, X,
} from 'lucide-vue-next';
import {
  ApiError, DatasetsApi, classColor, staticUrl,
  type AnnotationRegion, type CropImage, type DatasetDetail, type DatasetImage,
  type DatasetSummary, type DefectClass, type RegionInput,
} from '@/lib/api';
import PolygonAnnotator from '@/components/annotation/PolygonAnnotator.vue';

// ── List state ───────────────────────────────────────────────────────────────
const datasets = ref<DatasetSummary[]>([]);
const datasetsLoading = ref(true);
const datasetsLoadError = ref<string | null>(null);
const search = ref('');
const activeMenuId = ref<string | null>(null);

// ── Detail state ─────────────────────────────────────────────────────────────
const detail = ref<DatasetDetail | null>(null);
const detailLoading = ref(false);
const datasetView = ref<'original' | 'crop'>('original');

const filteredDatasets = computed(() => {
  const q = search.value.trim().toLowerCase();
  if (!q) return datasets.value;
  return datasets.value.filter((d) => d.name.toLowerCase().includes(q));
});

async function fetchDatasets(): Promise<void> {
  datasetsLoadError.value = null;
  datasetsLoading.value = true;
  try {
    datasets.value = await DatasetsApi.list();
  } catch (e) {
    datasets.value = [];
    datasetsLoadError.value =
      e instanceof ApiError ? e.message : '无法加载数据集，请确认后端已启动。';
  } finally {
    datasetsLoading.value = false;
  }
}

async function openDataset(id: string) {
  detailLoading.value = true;
  datasetView.value = 'original';
  try {
    detail.value = await DatasetsApi.get(id);
  } catch (e) {
    datasetsLoadError.value = e instanceof ApiError ? e.message : 'Failed to load dataset.';
    detail.value = null;
  } finally {
    detailLoading.value = false;
  }
}

async function refreshDetail() {
  if (detail.value) detail.value = await DatasetsApi.get(detail.value.id);
}

function closeDetail() {
  detail.value = null;
}

onMounted(() => {
  void fetchDatasets();
  document.addEventListener('keydown', onKeyDown);
});
onUnmounted(() => document.removeEventListener('keydown', onKeyDown));

// ── Crop grouping (crops live inside images of the detail) ────────────────────
const sourceImageMap = computed(() => {
  const m = new Map<string, DatasetImage>();
  detail.value?.images.forEach((i) => m.set(i.id, i));
  return m;
});

const classMap = computed(() => {
  const m = new Map<string, DefectClass>();
  detail.value?.defectClasses.forEach((c) => m.set(c.id, c));
  return m;
});

interface CropGroup {
  key: string;
  cls: DefectClass | null;
  crops: CropImage[];
}
const cropGroups = computed<CropGroup[]>(() => {
  if (!detail.value) return [];
  const groups = new Map<string, CropGroup>();
  for (const cls of detail.value.defectClasses) {
    groups.set(cls.id, {key: cls.id, cls, crops: []});
  }
  const unassigned: CropGroup = {key: '__none__', cls: null, crops: []};
  for (const img of detail.value.images) {
    for (const c of img.crops) {
      if (c.classId && groups.has(c.classId)) groups.get(c.classId)!.crops.push(c);
      else unassigned.crops.push(c);
    }
  }
  const arr = [...groups.values()].filter((g) => g.crops.length);
  if (unassigned.crops.length) arr.push(unassigned);
  return arr;
});

const totalCrops = computed(() =>
  (detail.value?.images ?? []).reduce((n, i) => n + i.crops.length, 0),
);

function colorForClass(classId: string | null | undefined): string {
  if (!classId) return '#64748b';
  const cls = classMap.value.get(classId);
  return classColor(cls ?? null);
}
function classNameOf(classId: string | null | undefined): string {
  if (!classId) return 'Unassigned';
  return classMap.value.get(classId)?.name ?? 'Unknown';
}

// ── Upload new dataset ───────────────────────────────────────────────────────
const uploadOpen = ref(false);
const uploadName = ref('');
const uploadMode = ref<'zip' | 'images'>('zip');
const zipFile = ref<File | null>(null);
const imageFiles = ref<File[]>([]);
const uploading = ref(false);
const uploadError = ref<string | null>(null);
const zipInputRef = ref<HTMLInputElement | null>(null);
const imagesInputRef = ref<HTMLInputElement | null>(null);

function resetUploadForm() {
  uploadName.value = '';
  zipFile.value = null;
  imageFiles.value = [];
  uploadError.value = null;
  uploadMode.value = 'zip';
  if (zipInputRef.value) zipInputRef.value.value = '';
  if (imagesInputRef.value) imagesInputRef.value.value = '';
}

async function handleUploadSubmit(e: Event) {
  e.preventDefault();
  uploadError.value = null;
  const name = uploadName.value.trim();
  if (!name) return (uploadError.value = 'Please enter a dataset name.'), undefined;
  if (uploadMode.value === 'zip' && !zipFile.value)
    return (uploadError.value = 'Please choose a .zip file.'), undefined;
  if (uploadMode.value === 'images' && imageFiles.value.length === 0)
    return (uploadError.value = 'Please choose at least one image.'), undefined;

  const form = new FormData();
  form.append('name', name);
  form.append('type', 'Image');
  if (uploadMode.value === 'zip' && zipFile.value) {
    form.append('zip_file', zipFile.value, zipFile.value.name);
  } else {
    imageFiles.value.forEach((f) => form.append('files', f));
  }

  uploading.value = true;
  try {
    await DatasetsApi.create(form);
    await fetchDatasets();
    uploadOpen.value = false;
    resetUploadForm();
  } catch (err) {
    uploadError.value = err instanceof ApiError ? err.message : 'Upload failed.';
  } finally {
    uploading.value = false;
  }
}

async function handleDelete(id: string, name: string) {
  if (!window.confirm(`Delete dataset "${name}"? This cannot be undone.`)) return;
  try {
    await DatasetsApi.remove(id);
    datasets.value = datasets.value.filter((d) => d.id !== id);
    activeMenuId.value = null;
    if (detail.value?.id === id) detail.value = null;
  } catch (e) {
    console.error('delete failed', e);
  }
}

// ── Add images to existing dataset ───────────────────────────────────────────
const addOpen = ref(false);
const addFiles = ref<File[]>([]);
const addUploading = ref(false);
const addError = ref<string | null>(null);
const addAnnotateAfter = ref(true);
const addInputRef = ref<HTMLInputElement | null>(null);

function resetAddForm() {
  addFiles.value = [];
  addError.value = null;
  addAnnotateAfter.value = true;
  if (addInputRef.value) addInputRef.value.value = '';
}

async function handleAddSubmit(e: Event) {
  e.preventDefault();
  if (!detail.value) return;
  addError.value = null;
  if (addFiles.value.length === 0)
    return (addError.value = 'Please choose at least one image.'), undefined;

  addUploading.value = true;
  try {
    if (addFiles.value.length === 1 && addAnnotateAfter.value) {
      const {image} = await DatasetsApi.uploadSingleImage(detail.value.id, addFiles.value[0]);
      await refreshDetail();
      addOpen.value = false;
      resetAddForm();
      openAnnotator(image);
    } else {
      await DatasetsApi.uploadImages(detail.value.id, addFiles.value, 'batch');
      await refreshDetail();
      addOpen.value = false;
      resetAddForm();
    }
  } catch (err) {
    addError.value = err instanceof ApiError ? err.message : 'Upload failed.';
  } finally {
    addUploading.value = false;
  }
}

async function deleteImage(img: DatasetImage) {
  if (!window.confirm('Delete this image and its crops?')) return;
  try {
    await DatasetsApi.deleteImage(img.id);
    await refreshDetail();
  } catch (e) {
    console.error(e);
  }
}

// ── Defect class manager ─────────────────────────────────────────────────────
const classesOpen = ref(false);
const newClassName = ref('');
const classBusy = ref(false);

async function addClass() {
  if (!detail.value) return;
  const name = newClassName.value.trim();
  if (!name) return;
  classBusy.value = true;
  try {
    const color = classColor(null, detail.value.defectClasses.length);
    await DatasetsApi.createClass(detail.value.id, {name, color});
    newClassName.value = '';
    await refreshDetail();
  } finally {
    classBusy.value = false;
  }
}

async function renameClass(cls: DefectClass) {
  const next = window.prompt('Rename class', cls.name);
  if (next == null || !next.trim() || next.trim() === cls.name) return;
  await DatasetsApi.updateClass(cls.id, {name: next.trim()});
  await refreshDetail();
}

async function removeClass(cls: DefectClass) {
  if (!window.confirm(`Delete class "${cls.name}"? Regions using it become unassigned.`)) return;
  await DatasetsApi.deleteClass(cls.id);
  await refreshDetail();
}

/** Callback passed to the annotator so it can create classes inline. */
async function createClassInline(name: string, color: string): Promise<DefectClass> {
  if (!detail.value) throw new Error('no dataset');
  const created = await DatasetsApi.createClass(detail.value.id, {name, color});
  await refreshDetail();
  return created;
}

// ── Annotator ────────────────────────────────────────────────────────────────
const annotatingImage = ref<DatasetImage | null>(null);
const annotatorSaving = ref(false);

function openAnnotator(img: DatasetImage) {
  annotatingImage.value = img;
}

async function onAnnotatorSave(regions: RegionInput[]) {
  if (!annotatingImage.value) return;
  annotatorSaving.value = true;
  try {
    await DatasetsApi.saveAnnotation(annotatingImage.value.id, regions, true);
    annotatingImage.value = null;
    await refreshDetail();
  } catch (e) {
    window.alert(e instanceof ApiError ? e.message : 'Save failed.');
  } finally {
    annotatorSaving.value = false;
  }
}

const annotatorRegions = computed<AnnotationRegion[]>(() => annotatingImage.value?.regions ?? []);

// ── Read-only preview ────────────────────────────────────────────────────────
interface PreviewRegion {points: number[][]; isSubtract: boolean; classId?: string | null}
const previewImage = ref<string | null>(null);
const previewRegions = ref<PreviewRegion[]>([]);
const showAnnotations = ref(true);
const previewTraceCrop = ref<CropImage | null>(null);
const zoom = ref(1);

function openImagePreview(img: DatasetImage) {
  previewTraceCrop.value = null;
  previewImage.value = staticUrl(img.url);
  previewRegions.value = img.regions.map((r) => ({points: r.points, isSubtract: r.isSubtract, classId: r.classId}));
  showAnnotations.value = true;
  zoom.value = 1;
}

function openCropPreview(crop: CropImage) {
  previewTraceCrop.value = crop;
  previewImage.value = staticUrl(crop.url);
  previewRegions.value = (crop.cropAnnotation ?? []).map((a) => ({
    points: a.points,
    isSubtract: a.isSubtract,
    classId: crop.classId,
  }));
  showAnnotations.value = true;
  zoom.value = 1;
}

function traceToSource() {
  const crop = previewTraceCrop.value;
  if (!crop) return;
  const src = sourceImageMap.value.get(crop.sourceImageId);
  if (!src) return;
  const region = src.regions[crop.instanceIndex];
  previewImage.value = staticUrl(src.url);
  previewRegions.value = region
    ? [{points: region.points, isSubtract: region.isSubtract, classId: region.classId}]
    : src.regions.map((r) => ({points: r.points, isSubtract: r.isSubtract, classId: r.classId}));
  previewTraceCrop.value = null;
  zoom.value = 1;
}

function closePreview() {
  previewImage.value = null;
  previewRegions.value = [];
  previewTraceCrop.value = null;
}

function onKeyDown(e: KeyboardEvent) {
  if (!previewImage.value) return;
  if (e.key === 'Escape') closePreview();
  if (e.key === ' ') {
    e.preventDefault();
    if (previewRegions.value.length) showAnnotations.value = !showAnnotations.value;
  }
}
function onPreviewWheel(e: WheelEvent) {
  e.preventDefault();
  zoom.value = Math.min(6, Math.max(0.35, zoom.value * (e.deltaY < 0 ? 1.12 : 1 / 1.12)));
}
</script>

<template>
  <!-- ─────────────── Detail view ─────────────── -->
  <div v-if="detail" class="flex h-full flex-col gap-6 overflow-y-auto p-6">
    <div class="flex items-center justify-between gap-4">
      <div class="flex items-center gap-4 min-w-0">
        <button class="rounded-full p-2 transition-colors hover:bg-secondary/50" @click="closeDetail">
          <ArrowLeft class="h-5 w-5" />
        </button>
        <div class="min-w-0">
          <h2 class="truncate text-2xl font-semibold tracking-tight">{{ detail.name }}</h2>
          <p class="text-sm text-muted-foreground">
            {{ detail.images.length }} images · {{ totalCrops }} crops · {{ detail.defectClasses.length }} classes
          </p>
        </div>
      </div>
      <div class="flex items-center gap-3">
        <div class="flex gap-1 rounded-lg bg-secondary/30 p-1 text-xs font-medium">
          <button
            class="rounded-md px-3 py-1 transition-colors"
            :class="datasetView === 'original' ? 'bg-background text-foreground shadow-sm' : 'text-muted-foreground hover:text-foreground'"
            @click="datasetView = 'original'"
          >原图</button>
          <button
            class="rounded-md px-3 py-1 transition-colors"
            :class="datasetView === 'crop' ? 'bg-background text-foreground shadow-sm' : 'text-muted-foreground hover:text-foreground'"
            @click="datasetView = 'crop'"
          >裁剪图</button>
        </div>
        <button
          class="flex items-center gap-2 rounded-md border border-border px-3 py-1.5 text-xs font-medium hover:bg-secondary/50"
          @click="classesOpen = true"
        >
          <Settings2 class="h-3.5 w-3.5" /> Classes
        </button>
        <button
          class="flex items-center gap-2 rounded-md bg-primary px-4 py-2 text-sm font-medium text-primary-foreground hover:bg-primary/90"
          @click="resetAddForm(); addOpen = true"
        >
          <Plus class="h-4 w-4" /> Add images
        </button>
      </div>
    </div>

    <!-- ── 原图 grid ── -->
    <div v-if="datasetView === 'original'">
      <div
        v-if="detail.images.length"
        class="grid grid-cols-2 gap-4 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-6"
      >
        <div
          v-for="img in detail.images"
          :key="img.id"
          class="group relative aspect-square overflow-hidden rounded-lg border border-border"
        >
          <img :src="staticUrl(img.url)" alt="" class="h-full w-full cursor-pointer object-cover" @click="openImagePreview(img)" />
          <!-- status badge -->
          <div
            class="absolute left-1 top-1 rounded px-1.5 py-0.5 text-[9px] font-medium leading-none"
            :class="img.annotationStatus === 'annotated' ? 'bg-emerald-500/90 text-white' : 'bg-zinc-700/80 text-zinc-200'"
          >
            {{ img.annotationStatus === 'annotated' ? `${img.regions.filter((r) => !r.isSubtract).length} regions` : 'unlabeled' }}
          </div>
          <!-- hover actions -->
          <div class="absolute inset-x-0 bottom-0 flex translate-y-full items-center justify-center gap-1 bg-black/55 p-1.5 transition-transform group-hover:translate-y-0">
            <button
              class="inline-flex items-center gap-1 rounded bg-primary px-2 py-1 text-[11px] font-medium text-primary-foreground hover:bg-primary/90"
              @click="openAnnotator(img)"
            >
              <PencilLine class="h-3 w-3" /> Annotate
            </button>
            <button
              class="inline-flex items-center justify-center rounded bg-white/15 p-1 text-white hover:bg-white/25"
              title="Delete image"
              @click="deleteImage(img)"
            >
              <Trash2 class="h-3 w-3" />
            </button>
          </div>
        </div>
      </div>
      <div v-else class="flex flex-col items-center justify-center py-20 text-muted-foreground">
        <ImageIcon class="mb-4 h-12 w-12 opacity-20" />
        <p>No images yet. Click “Add images”.</p>
      </div>
    </div>

    <!-- ── 裁剪图 grid (grouped by class) ── -->
    <div v-else class="space-y-8">
      <div v-for="g in cropGroups" :key="g.key" class="space-y-3">
        <div class="flex items-center gap-2 border-b border-border pb-2">
          <span class="h-3 w-3 rounded-full" :style="{backgroundColor: colorForClass(g.cls?.id ?? null)}" />
          <h3 class="text-sm font-semibold uppercase tracking-wider text-muted-foreground">
            {{ g.cls?.name ?? 'Unassigned' }}
          </h3>
          <span class="rounded-full bg-secondary px-2 py-0.5 text-xs text-muted-foreground">{{ g.crops.length }} crops</span>
        </div>
        <div class="grid grid-cols-4 gap-2 sm:grid-cols-6 md:grid-cols-8 lg:grid-cols-10">
          <div
            v-for="crop in g.crops"
            :key="crop.id"
            class="group relative aspect-square cursor-pointer overflow-hidden rounded-md border border-border transition-transform hover:scale-[1.03]"
            @click="openCropPreview(crop)"
          >
            <img :src="staticUrl(crop.url)" alt="" class="h-full w-full object-cover" />
            <div class="absolute inset-0 flex items-center justify-center bg-black/40 opacity-0 transition-opacity group-hover:opacity-100">
              <ExternalLink class="h-4 w-4 text-white" />
            </div>
          </div>
        </div>
      </div>
      <div v-if="!cropGroups.length" class="flex flex-col items-center justify-center py-20 text-muted-foreground">
        <Tag class="mb-4 h-12 w-12 opacity-20" />
        <p>No crops yet. Annotate images to generate crops.</p>
      </div>
    </div>
  </div>

  <!-- ─────────────── List view ─────────────── -->
  <div v-else class="flex h-full flex-col gap-6 p-6">
    <div class="flex items-center justify-between">
      <div>
        <h2 class="text-2xl font-semibold tracking-tight">Datasets</h2>
        <p class="text-sm text-muted-foreground">Manage and upload your training data.</p>
      </div>
      <button
        class="flex items-center gap-2 rounded-md bg-primary px-4 py-2 text-sm font-medium text-primary-foreground hover:bg-primary/90"
        @click="resetUploadForm(); uploadOpen = true"
      >
        <Plus class="h-4 w-4" /> Upload dataset
      </button>
    </div>

    <div class="flex items-center gap-4 rounded-lg border border-border bg-secondary/10 p-2">
      <Search class="ml-2 h-4 w-4 text-muted-foreground" />
      <input v-model="search" type="text" placeholder="Search datasets..." class="flex-1 border-none bg-transparent text-sm focus:outline-none" />
    </div>

    <div v-if="datasetsLoadError" class="flex flex-col gap-2 rounded-lg border border-amber-500/40 bg-amber-500/10 px-4 py-3 text-sm text-amber-200">
      <p>{{ datasetsLoadError }}</p>
      <button class="self-start rounded-md border border-amber-500/50 px-3 py-1 text-xs font-medium text-amber-100 hover:bg-amber-500/20" @click="fetchDatasets">重试</button>
    </div>

    <div class="rounded-xl border border-border">
      <table class="w-full text-left text-sm">
        <thead class="border-b border-border bg-secondary/20 text-muted-foreground">
          <tr>
            <th class="px-6 py-4 font-medium">Name</th>
            <th class="px-6 py-4 font-medium">Classes</th>
            <th class="px-6 py-4 font-medium">Size</th>
            <th class="px-6 py-4 font-medium">Items</th>
            <th class="px-6 py-4 font-medium">Status</th>
            <th class="px-6 py-4 font-medium"></th>
          </tr>
        </thead>
        <tbody class="divide-y divide-border">
          <tr v-if="datasetsLoading">
            <td colspan="6" class="px-6 py-12 text-center text-muted-foreground">
              <Loader2 class="mx-auto mb-2 h-6 w-6 animate-spin opacity-60" /> Loading…
            </td>
          </tr>
          <tr v-else-if="!filteredDatasets.length" class="text-muted-foreground">
            <td colspan="6" class="px-6 py-12 text-center">No datasets yet. Click “Upload dataset”.</td>
          </tr>
          <tr
            v-for="ds in filteredDatasets"
            :key="ds.id"
            class="group cursor-pointer transition-colors hover:bg-secondary/10"
            @click="openDataset(ds.id)"
          >
            <td class="px-6 py-4 font-medium">
              <div class="flex items-center gap-3 group-hover:text-primary">
                <div class="rounded bg-secondary/30 p-2"><Database class="h-4 w-4 text-muted-foreground" /></div>
                {{ ds.name }}
              </div>
            </td>
            <td class="px-6 py-4">
              <div class="flex flex-wrap items-center gap-1">
                <span
                  v-for="(c, i) in ds.defectClasses.slice(0, 4)"
                  :key="c.id"
                  class="inline-flex items-center gap-1 rounded-full bg-secondary/40 px-2 py-0.5 text-xs"
                >
                  <span class="h-2 w-2 rounded-full" :style="{backgroundColor: classColor(c, i)}" />
                  {{ c.name }}
                </span>
                <span v-if="ds.defectClasses.length > 4" class="text-xs text-muted-foreground">+{{ ds.defectClasses.length - 4 }}</span>
                <span v-if="!ds.defectClasses.length" class="text-xs text-muted-foreground">—</span>
              </div>
            </td>
            <td class="px-6 py-4 text-muted-foreground">{{ ds.size ?? '—' }}</td>
            <td class="px-6 py-4 text-muted-foreground">{{ ds.items }}</td>
            <td class="px-6 py-4">
              <span
                class="rounded-full px-2 py-1 text-xs"
                :class="ds.status === 'Ready' ? 'bg-emerald-500/10 text-emerald-500' : 'bg-amber-500/10 text-amber-500'"
              >{{ ds.status }}</span>
            </td>
            <td class="relative px-6 py-4 text-right" :class="activeMenuId === ds.id && 'z-30'">
              <button class="rounded p-1 hover:bg-secondary/30" @click.stop="activeMenuId = activeMenuId === ds.id ? null : ds.id">
                <MoreVertical class="h-4 w-4 text-muted-foreground" />
              </button>
              <div v-if="activeMenuId === ds.id">
                <div class="fixed inset-0 z-10" @click.stop="activeMenuId = null" />
                <div class="absolute right-6 top-12 z-20 w-40 overflow-hidden rounded-lg border border-border bg-background shadow-xl" @click.stop>
                  <div class="flex flex-col p-1">
                    <button class="flex items-center gap-2 rounded-md px-3 py-2 text-left text-xs hover:bg-secondary/50" @click="openDataset(ds.id); activeMenuId = null">
                      <ExternalLink class="h-3.5 w-3.5" /> View details
                    </button>
                    <div class="my-1 h-px bg-border" />
                    <button class="flex items-center gap-2 rounded-md px-3 py-2 text-left text-xs text-rose-500 hover:bg-rose-500/10" @click="handleDelete(ds.id, ds.name)">
                      <Trash2 class="h-3.5 w-3.5" /> Delete
                    </button>
                  </div>
                </div>
              </div>
            </td>
          </tr>
        </tbody>
      </table>
    </div>
  </div>

  <!-- ─────────────── Annotator overlay ─────────────── -->
  <PolygonAnnotator
    v-if="annotatingImage && detail"
    :image-url="staticUrl(annotatingImage.url)"
    :image-name="`Image ${annotatingImage.id.slice(0, 8)}`"
    :classes="detail.defectClasses"
    :initial-regions="annotatorRegions"
    :saving="annotatorSaving"
    :create-class="createClassInline"
    @save="onAnnotatorSave"
    @cancel="annotatingImage = null"
  />

  <!-- ─────────────── Classes manager ─────────────── -->
  <Teleport to="body">
    <Transition name="fade">
      <div v-if="classesOpen && detail" class="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-6 backdrop-blur-sm" @click.self="classesOpen = false">
        <div class="w-full max-w-md space-y-4 rounded-xl border border-border bg-background p-6 shadow-xl">
          <div class="flex items-center justify-between">
            <h3 class="text-lg font-semibold">Defect classes</h3>
            <button class="rounded-md p-1 text-muted-foreground hover:bg-secondary/80" @click="classesOpen = false"><X class="h-5 w-5" /></button>
          </div>
          <div class="max-h-72 space-y-1.5 overflow-y-auto">
            <div v-for="(c, i) in detail.defectClasses" :key="c.id" class="flex items-center gap-2 rounded-md border border-border px-3 py-2 text-sm">
              <span class="h-3 w-3 rounded-full" :style="{backgroundColor: classColor(c, i)}" />
              <span class="flex-1 truncate">{{ c.name }}</span>
              <button class="text-muted-foreground hover:text-foreground" title="Rename" @click="renameClass(c)"><PencilLine class="h-3.5 w-3.5" /></button>
              <button class="text-muted-foreground hover:text-destructive" title="Delete" @click="removeClass(c)"><Trash2 class="h-3.5 w-3.5" /></button>
            </div>
            <p v-if="!detail.defectClasses.length" class="text-xs text-muted-foreground">No classes yet.</p>
          </div>
          <form class="flex gap-2" @submit.prevent="addClass">
            <input v-model="newClassName" placeholder="New class name" class="flex-1 rounded-md border border-input bg-background px-3 py-2 text-sm outline-none focus:ring-1 focus:ring-ring" />
            <button type="submit" :disabled="!newClassName.trim() || classBusy" class="rounded-md bg-primary px-4 text-sm font-medium text-primary-foreground hover:bg-primary/90 disabled:opacity-50">Add</button>
          </form>
        </div>
      </div>
    </Transition>
  </Teleport>

  <!-- ─────────────── Add images modal ─────────────── -->
  <Teleport to="body">
    <Transition name="fade">
      <div v-if="addOpen && detail" class="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-6 backdrop-blur-sm" @click.self="!addUploading && (addOpen = false)">
        <div class="w-full max-w-md space-y-4 rounded-xl border border-border bg-background p-6 shadow-xl">
          <div class="flex items-center justify-between">
            <h3 class="text-lg font-semibold">Add images</h3>
            <button :disabled="addUploading" class="rounded-md p-1 text-muted-foreground hover:bg-secondary/80" @click="addOpen = false"><X class="h-5 w-5" /></button>
          </div>
          <p class="text-xs text-muted-foreground">Upload image(s) without labels. You assign defect classes per region while annotating.</p>
          <form class="space-y-4" @submit="handleAddSubmit">
            <input
              ref="addInputRef" type="file" accept="image/*" multiple class="hidden"
              @change="(e) => (addFiles = (e.target as HTMLInputElement).files ? Array.from((e.target as HTMLInputElement).files!) : [])"
            />
            <button type="button" :disabled="addUploading" class="flex w-full items-center justify-center gap-2 rounded-lg border-2 border-dashed border-border py-6 text-sm text-muted-foreground hover:border-primary/50" @click="addInputRef?.click()">
              <Upload class="h-4 w-4" />
              {{ addFiles.length ? `${addFiles.length} file(s) selected` : 'Choose image(s)' }}
            </button>
            <label v-if="addFiles.length === 1" class="flex items-center gap-2 text-xs text-muted-foreground">
              <input v-model="addAnnotateAfter" type="checkbox" class="rounded border-border" />
              Open the annotation workbench after upload
            </label>
            <p v-if="addError" class="text-xs text-rose-500">{{ addError }}</p>
            <div class="flex justify-end gap-2 pt-2">
              <button type="button" :disabled="addUploading" class="rounded-md border border-border px-3 py-2 text-sm hover:bg-secondary/50" @click="addOpen = false">Cancel</button>
              <button type="submit" :disabled="addUploading" class="rounded-md bg-primary px-4 py-2 text-sm font-medium text-primary-foreground hover:bg-primary/90 disabled:opacity-50">{{ addUploading ? 'Uploading…' : 'Upload' }}</button>
            </div>
          </form>
        </div>
      </div>
    </Transition>
  </Teleport>

  <!-- ─────────────── Upload dataset modal ─────────────── -->
  <Teleport to="body">
    <Transition name="fade">
      <div v-if="uploadOpen" class="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-6 backdrop-blur-sm" @click.self="!uploading && (uploadOpen = false)">
        <div class="w-full max-w-md space-y-4 rounded-xl border border-border bg-background p-6 shadow-xl">
          <div class="flex items-center justify-between">
            <h3 class="text-lg font-semibold">Upload dataset</h3>
            <button :disabled="uploading" class="rounded-md p-1 text-muted-foreground hover:bg-secondary/80" @click="uploadOpen = false"><X class="h-5 w-5" /></button>
          </div>
          <p class="text-xs text-muted-foreground">
            ZIP: one folder per class (<code class="rounded bg-secondary/50 px-1 text-[10px]">cls_a/img.jpg</code>) with optional <code class="rounded bg-secondary/50 px-1 text-[10px]">.json</code> label sidecars. Or upload plain images to annotate later.
          </p>
          <form class="space-y-4" @submit="handleUploadSubmit">
            <input v-model="uploadName" type="text" placeholder="Dataset name" class="w-full rounded-md border border-border bg-background px-3 py-2 text-sm focus:outline-none focus:ring-1 focus:ring-ring" :disabled="uploading" />
            <div class="flex gap-1 rounded-lg bg-secondary/20 p-1">
              <button type="button" class="flex-1 rounded-md py-1.5 text-xs font-medium" :class="uploadMode === 'zip' ? 'bg-background shadow-sm' : 'text-muted-foreground'" :disabled="uploading" @click="uploadMode = 'zip'">ZIP (images + labels)</button>
              <button type="button" class="flex-1 rounded-md py-1.5 text-xs font-medium" :class="uploadMode === 'images' ? 'bg-background shadow-sm' : 'text-muted-foreground'" :disabled="uploading" @click="uploadMode = 'images'">Images only</button>
            </div>
            <template v-if="uploadMode === 'zip'">
              <input ref="zipInputRef" type="file" accept=".zip,application/zip" class="hidden" @change="(e) => (zipFile = (e.target as HTMLInputElement).files?.[0] ?? null)" />
              <button type="button" :disabled="uploading" class="flex w-full items-center justify-center gap-2 rounded-lg border-2 border-dashed border-border py-8 text-sm text-muted-foreground hover:border-primary/50" @click="zipInputRef?.click()">
                <Upload class="h-4 w-4" /> {{ zipFile ? zipFile.name : 'Choose .zip file' }}
              </button>
            </template>
            <template v-else>
              <input ref="imagesInputRef" type="file" accept="image/*" multiple class="hidden" @change="(e) => (imageFiles = (e.target as HTMLInputElement).files ? Array.from((e.target as HTMLInputElement).files!) : [])" />
              <button type="button" :disabled="uploading" class="flex w-full items-center justify-center gap-2 rounded-lg border-2 border-dashed border-border py-6 text-sm text-muted-foreground hover:border-primary/50" @click="imagesInputRef?.click()">
                <Upload class="h-4 w-4" /> {{ imageFiles.length ? `${imageFiles.length} file(s) selected` : 'Choose images' }}
              </button>
            </template>
            <p v-if="uploadError" class="text-xs text-rose-500">{{ uploadError }}</p>
            <div class="flex justify-end gap-2 pt-2">
              <button type="button" :disabled="uploading" class="rounded-md border border-border px-3 py-2 text-sm hover:bg-secondary/50" @click="uploadOpen = false">Cancel</button>
              <button type="submit" :disabled="uploading" class="rounded-md bg-primary px-4 py-2 text-sm font-medium text-primary-foreground hover:bg-primary/90 disabled:opacity-50">{{ uploading ? 'Uploading…' : 'Upload' }}</button>
            </div>
          </form>
        </div>
      </div>
    </Transition>
  </Teleport>

  <!-- ─────────────── Read-only preview ─────────────── -->
  <Teleport to="body">
    <Transition name="fade">
      <div v-if="previewImage" class="fixed inset-0 z-50 flex items-center justify-center bg-black/85 backdrop-blur-sm" @click="closePreview">
        <div class="pointer-events-none absolute inset-x-0 top-0 z-10 flex items-center justify-between px-4 py-3" @click.stop>
          <div class="pointer-events-auto rounded-lg bg-black/50 px-3 py-1.5 text-xs text-white/70">{{ Math.round(zoom * 100) }}%</div>
          <div class="pointer-events-auto flex items-center gap-2">
            <button v-if="previewTraceCrop" class="rounded-lg bg-black/50 px-3 py-1.5 text-xs font-medium text-white/80 hover:bg-white/20" @click="traceToSource">View original</button>
            <button v-if="previewRegions.length" class="rounded-lg bg-black/50 px-3 py-1.5 text-xs font-medium hover:bg-white/20" :class="showAnnotations ? 'text-indigo-300' : 'text-white/50'" @click="showAnnotations = !showAnnotations">{{ showAnnotations ? 'Hide' : 'Show' }} labels</button>
            <button class="flex h-8 w-8 items-center justify-center rounded-full bg-black/50 text-white hover:bg-white/20" @click="closePreview"><X class="h-4 w-4" /></button>
          </div>
        </div>
        <div class="relative flex h-full w-full items-center justify-center overflow-hidden" @click.stop @wheel.prevent="onPreviewWheel">
          <div class="relative inline-block leading-none" :style="{transform: `scale(${zoom})`, transition: 'transform 0.15s ease'}">
            <img :src="previewImage" alt="Preview" class="block max-h-[calc(100vh-6rem)] w-auto max-w-[calc(100vw-4rem)] rounded-xl shadow-2xl" draggable="false" />
            <svg v-if="previewRegions.length && showAnnotations" class="pointer-events-none absolute inset-0 h-full w-full rounded-xl" viewBox="0 0 1 1" preserveAspectRatio="none">
              <polygon
                v-for="(r, idx) in previewRegions"
                :key="idx"
                :points="r.points.map(([x, y]: number[]) => `${x},${y}`).join(' ')"
                :fill="r.isSubtract ? 'rgba(148,163,184,0.25)' : colorForClass(r.classId) + '55'"
                :stroke="r.isSubtract ? '#94a3b8' : colorForClass(r.classId)"
                :stroke-width="0.004 / zoom"
                stroke-linejoin="round"
              />
            </svg>
          </div>
        </div>
        <p class="pointer-events-none absolute bottom-3 left-0 right-0 text-center text-[11px] text-white/25">Scroll to zoom · Space toggles labels · Esc to close</p>
      </div>
    </Transition>
  </Teleport>
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
</style>

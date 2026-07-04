<script setup lang="ts">
import {computed, onMounted, onUnmounted, ref, watch} from 'vue';
import {
  ArrowLeft, Database, ExternalLink, ImageIcon, Loader2, MoreVertical, PencilLine,
  Plus, Search, Settings2, Tag, Trash2, Upload, X,
} from 'lucide-vue-next';
import {
  ApiError, DatasetsApi, classColor, staticUrl, thumbUrl,
  type AnnotationRegion, type CropImage, type DatasetDetail, type DatasetImage,
  type DatasetSummary, type DefectClass, type RegionInput,
} from '@/lib/api';
import PolygonAnnotator from '@/components/annotation/PolygonAnnotator.vue';
import ImagePreviewModal from '@/components/annotation/ImagePreviewModal.vue';
import type {AnnotationOverlayRegion} from '@/components/annotation/overlayUtils';

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
});
onUnmounted(() => {
  stopImportPolling();
  if (listPollTimer) clearInterval(listPollTimer);
});

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
function previewColorForClass(classId: string | null | undefined, isSubtract?: boolean): string {
  if (isSubtract) return '#94a3b8';
  return colorForClass(classId);
}

// ── Upload new dataset ───────────────────────────────────────────────────────
const uploadOpen = ref(false);
const uploadName = ref('');
const uploadMode = ref<'zip' | 'images'>('zip');
const zipFile = ref<File | null>(null);
const imageFiles = ref<File[]>([]);
const uploading = ref(false);
const uploadError = ref<string | null>(null);
const uploadBytesProgress = ref(0);
const importProgress = ref(0);
const importMessage = ref<string | null>(null);
const importStage = ref<string | null>(null);
const activeImportId = ref<string | null>(null);
const uploadDatasetId = ref<string | null>(null);
const cancelInFlight = ref(false);
let uploadAbort: (() => void) | null = null;
const zipInputRef = ref<HTMLInputElement | null>(null);
const imagesInputRef = ref<HTMLInputElement | null>(null);
const zipDragActive = ref(false);

function isZipFile(file: File): boolean {
  const name = file.name.toLowerCase();
  return (
    name.endsWith('.zip')
    || file.type === 'application/zip'
    || file.type === 'application/x-zip-compressed'
  );
}

function datasetNameFromZip(filename: string): string {
  const stem = filename.replace(/\.zip$/i, '').trim();
  return stem || filename;
}

function assignZipFile(file: File | null) {
  uploadError.value = null;
  if (!file) return;
  if (!isZipFile(file)) {
    uploadError.value = 'Please choose or drop a .zip file.';
    return;
  }
  zipFile.value = file;
  if (!uploadName.value.trim()) {
    uploadName.value = datasetNameFromZip(file.name);
  }
}

function onZipDragEnter(e: DragEvent) {
  e.preventDefault();
  if (uploading.value) return;
  zipDragActive.value = true;
}

function onZipDragOver(e: DragEvent) {
  e.preventDefault();
  if (uploading.value) return;
  if (e.dataTransfer) e.dataTransfer.dropEffect = 'copy';
}

function onZipDragLeave(e: DragEvent) {
  e.preventDefault();
  const related = e.relatedTarget as Node | null;
  if (related && (e.currentTarget as HTMLElement).contains(related)) return;
  zipDragActive.value = false;
}

function onZipDrop(e: DragEvent) {
  e.preventDefault();
  zipDragActive.value = false;
  if (uploading.value) return;
  assignZipFile(e.dataTransfer?.files?.[0] ?? null);
}

const imagesDragActive = ref(false);

function isImageFile(file: File): boolean {
  return file.type.startsWith('image/') || /\.(jpe?g|png|gif|webp|bmp|tiff?|avif|heic)$/i.test(file.name);
}

function imageFilesFromList(files: FileList | File[] | null | undefined): File[] {
  if (!files?.length) return [];
  return Array.from(files).filter(isImageFile);
}

function assignImageFiles(files: FileList | File[] | null | undefined) {
  uploadError.value = null;
  if (!files?.length) return;
  const list = imageFilesFromList(files);
  if (!list.length) {
    uploadError.value = 'Please choose or drop image file(s).';
    return;
  }
  imageFiles.value = list;
}

function onImagesDragEnter(e: DragEvent) {
  e.preventDefault();
  if (uploading.value) return;
  imagesDragActive.value = true;
}

function onImagesDragOver(e: DragEvent) {
  e.preventDefault();
  if (uploading.value) return;
  if (e.dataTransfer) e.dataTransfer.dropEffect = 'copy';
}

function onImagesDragLeave(e: DragEvent) {
  e.preventDefault();
  const related = e.relatedTarget as Node | null;
  if (related && (e.currentTarget as HTMLElement).contains(related)) return;
  imagesDragActive.value = false;
}

function onImagesDrop(e: DragEvent) {
  e.preventDefault();
  imagesDragActive.value = false;
  if (uploading.value) return;
  assignImageFiles(e.dataTransfer?.files ?? null);
}

const IMPORT_STAGE_LABELS: Record<string, string> = {
  uploading: '上传文件',
  extracting: '解压 zip',
  persisting: '写入数据库',
  cropping: '生成裁剪图',
  saving: '保存图片',
  done: '完成',
  failed: '失败',
};

const combinedUploadProgress = computed(() => {
  if (!uploading.value) return 0;
  if (importStage.value && importStage.value !== 'uploading') {
    return Math.min(100, Math.round(15 + importProgress.value * 0.85));
  }
  return Math.round(uploadBytesProgress.value * 0.15);
});

const uploadProgressLabel = computed(() => {
  if (importMessage.value) return importMessage.value;
  if (importStage.value && IMPORT_STAGE_LABELS[importStage.value]) {
    return IMPORT_STAGE_LABELS[importStage.value];
  }
  if (uploadBytesProgress.value > 0 && uploadBytesProgress.value < 100) {
    return `上传文件 ${uploadBytesProgress.value}%`;
  }
  return '准备上传…';
});

function resetUploadProgress() {
  uploadBytesProgress.value = 0;
  importProgress.value = 0;
  importMessage.value = null;
  importStage.value = null;
  activeImportId.value = null;
}

function resetUploadForm() {
  uploadName.value = '';
  zipFile.value = null;
  imageFiles.value = [];
  uploadError.value = null;
  uploadMode.value = 'zip';
  zipDragActive.value = false;
  imagesDragActive.value = false;
  uploadDatasetId.value = null;
  uploadAbort = null;
  resetUploadProgress();
  if (zipInputRef.value) zipInputRef.value.value = '';
  if (imagesInputRef.value) imagesInputRef.value.value = '';
}

let importPollTimer: ReturnType<typeof setInterval> | null = null;
let importWaitReject: ((reason?: unknown) => void) | null = null;

function stopImportPolling(rejectPending = false) {
  if (importPollTimer) {
    clearInterval(importPollTimer);
    importPollTimer = null;
  }
  if (rejectPending && importWaitReject) {
    importWaitReject(new Error('import-wait-ended'));
    importWaitReject = null;
  }
}

async function pollImportStatus(datasetId: string): Promise<'done' | 'failed' | 'processing' | 'missing'> {
  try {
    const st = await DatasetsApi.getImportStatus(datasetId);
    importProgress.value = Math.round(st.importProgress ?? 0);
    importStage.value = st.importStage ?? null;
    importMessage.value = st.importMessage ?? null;
    if (st.status === 'Ready' || st.importStage === 'done') return 'done';
    if (st.status === 'Failed' || st.importStage === 'failed') return 'failed';
    return 'processing';
  } catch (e) {
    if (e instanceof ApiError && e.status === 404) return 'missing';
    return 'processing';
  }
}

function startImportPolling(datasetId: string, onComplete: () => void) {
  stopImportPolling();
  activeImportId.value = datasetId;
  void pollImportStatus(datasetId);
  importPollTimer = setInterval(async () => {
    const result = await pollImportStatus(datasetId);
    if (result === 'done') {
      stopImportPolling();
      onComplete();
    } else if (result === 'failed' || result === 'missing') {
      stopImportPolling(true);
      uploadError.value = result === 'failed' ? '导入失败，请查看后端日志。' : '导入中断，数据集可能已被删除。';
      uploading.value = false;
      void fetchDatasets();
    }
  }, 1000);
}

const hasProcessingDatasets = computed(() =>
  datasets.value.some((d) => d.status === 'Processing'),
);

let listPollTimer: ReturnType<typeof setInterval> | null = null;

watch(hasProcessingDatasets, (processing) => {
  if (listPollTimer) {
    clearInterval(listPollTimer);
    listPollTimer = null;
  }
  if (!processing) return;
  listPollTimer = setInterval(() => {
    void fetchDatasets();
  }, 2000);
}, {immediate: true});

function isUploadCancelledError(err: unknown): boolean {
  return err instanceof ApiError && err.message === 'Upload cancelled.';
}

async function cancelUpload() {
  if (cancelInFlight.value) return;
  const busy = uploading.value || uploadDatasetId.value != null || activeImportId.value != null;
  if (!busy && !uploadOpen.value) return;

  const msg = uploading.value
    ? '确定要取消上传吗？已上传的不完整数据将被删除。'
    : '确定要关闭上传窗口吗？';
  if (!window.confirm(msg)) return;

  cancelInFlight.value = true;
  stopImportPolling(true);
  uploadAbort?.();
  uploadAbort = null;

  const datasetId = uploadDatasetId.value ?? activeImportId.value;
  if (datasetId) {
    try {
      await DatasetsApi.cancelImport(datasetId);
    } catch (e) {
      if (!(e instanceof ApiError && (e.status === 404 || e.status === 409))) {
        try {
          await DatasetsApi.remove(datasetId);
        } catch {
          /* best effort cleanup */
        }
      }
    }
  }

  uploading.value = false;
  uploadOpen.value = false;
  resetUploadForm();
  await fetchDatasets();
  cancelInFlight.value = false;
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
  resetUploadProgress();
  importStage.value = 'uploading';
  uploadDatasetId.value = null;
  const {promise, abort} = DatasetsApi.create(form, (pct) => {
    uploadBytesProgress.value = pct;
  });
  uploadAbort = abort;
  try {
    const created = await promise;
    uploadAbort = null;
    uploadDatasetId.value = created.id;
    uploadBytesProgress.value = 100;
    importStage.value = 'extracting';
    importProgress.value = 0;

    await new Promise<void>((resolve, reject) => {
      importWaitReject = reject;
      startImportPolling(created.id, () => {
        importWaitReject = null;
        resolve();
      });
    });

    await fetchDatasets();
    uploadOpen.value = false;
    resetUploadForm();
  } catch (err) {
    if (!isUploadCancelledError(err) && !(err instanceof Error && err.message === 'import-wait-ended')) {
      uploadError.value = err instanceof ApiError ? err.message : 'Upload failed.';
    }
    stopImportPolling();
    if (uploadDatasetId.value) {
      try {
        await DatasetsApi.cancelImport(uploadDatasetId.value);
      } catch {
        /* cancelled path may already have cleaned up */
      }
    }
  } finally {
    uploadAbort = null;
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
const addDragActive = ref(false);

function assignAddFiles(files: FileList | File[] | null | undefined) {
  addError.value = null;
  if (!files?.length) return;
  const list = imageFilesFromList(files);
  if (!list.length) {
    addError.value = 'Please choose or drop image file(s).';
    return;
  }
  addFiles.value = list;
}

function onAddDragEnter(e: DragEvent) {
  e.preventDefault();
  if (addUploading.value) return;
  addDragActive.value = true;
}

function onAddDragOver(e: DragEvent) {
  e.preventDefault();
  if (addUploading.value) return;
  if (e.dataTransfer) e.dataTransfer.dropEffect = 'copy';
}

function onAddDragLeave(e: DragEvent) {
  e.preventDefault();
  const related = e.relatedTarget as Node | null;
  if (related && (e.currentTarget as HTMLElement).contains(related)) return;
  addDragActive.value = false;
}

function onAddDrop(e: DragEvent) {
  e.preventDefault();
  addDragActive.value = false;
  if (addUploading.value) return;
  assignAddFiles(e.dataTransfer?.files ?? null);
}

function resetAddForm() {
  addFiles.value = [];
  addError.value = null;
  addAnnotateAfter.value = true;
  addDragActive.value = false;
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

// ── Continuous image switching ────────────────────────────────────────────────
const annotImages = computed<DatasetImage[]>(() => detail.value?.images ?? []);
const annotIndex = computed(() =>
  annotatingImage.value ? annotImages.value.findIndex((i) => i.id === annotatingImage.value!.id) : -1,
);
const annotHasPrev = computed(() => annotIndex.value > 0);
const annotHasNext = computed(() => annotIndex.value >= 0 && annotIndex.value < annotImages.value.length - 1);

function navigateAnnotator(dir: -1 | 1) {
  const next = annotImages.value[annotIndex.value + dir];
  if (next) annotatingImage.value = next;
}

async function onAnnotatorSave(regions: RegionInput[], advance = false) {
  if (!annotatingImage.value) return;
  const currentId = annotatingImage.value.id;
  annotatorSaving.value = true;
  try {
    await DatasetsApi.saveAnnotation(currentId, regions, true);
    await refreshDetail();
    if (advance) {
      const list = detail.value?.images ?? [];
      const idx = list.findIndex((i) => i.id === currentId);
      annotatingImage.value = idx >= 0 ? (list[idx + 1] ?? null) : null;
    } else {
      annotatingImage.value = null;
    }
  } catch (e) {
    window.alert(e instanceof ApiError ? e.message : 'Save failed.');
  } finally {
    annotatorSaving.value = false;
  }
}

const annotatorRegions = computed<AnnotationRegion[]>(() => annotatingImage.value?.regions ?? []);

// ── Read-only preview ────────────────────────────────────────────────────────
const previewImage = ref<string | null>(null);
const previewOverlayRegions = ref<AnnotationOverlayRegion[]>([]);
const previewTraceCrop = ref<CropImage | null>(null);
const previewViewMode = ref<'crop' | 'source'>('crop');

function openImagePreview(img: DatasetImage) {
  previewTraceCrop.value = null;
  previewViewMode.value = 'source';
  previewImage.value = staticUrl(img.url);
  previewOverlayRegions.value = img.regions.map((r) => ({
    points: r.points,
    isSubtract: r.isSubtract,
    classId: r.classId,
  }));
}

function openCropPreview(crop: CropImage) {
  previewTraceCrop.value = crop;
  previewViewMode.value = 'crop';
  previewImage.value = staticUrl(crop.url);
  previewOverlayRegions.value = (crop.cropAnnotation ?? []).map((a) => ({
    points: a.points,
    isSubtract: a.isSubtract,
    classId: crop.classId,
    label: classNameOf(crop.classId),
  }));
}

function traceToSource() {
  const crop = previewTraceCrop.value;
  if (!crop) return;
  const src = sourceImageMap.value.get(crop.sourceImageId);
  if (!src) return;
  const region = src.regions[crop.instanceIndex];
  previewViewMode.value = 'source';
  previewImage.value = staticUrl(src.url);
  previewOverlayRegions.value = region
    ? [{points: region.points, isSubtract: region.isSubtract, classId: region.classId}]
    : src.regions.map((r) => ({points: r.points, isSubtract: r.isSubtract, classId: r.classId}));
  previewTraceCrop.value = null;
}

function closePreview() {
  previewImage.value = null;
  previewOverlayRegions.value = [];
  previewTraceCrop.value = null;
  previewViewMode.value = 'crop';
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
          <img :src="thumbUrl(img.url, 384)" alt="" loading="lazy" decoding="async" class="h-full w-full cursor-pointer bg-secondary/20 object-cover" @click="openImagePreview(img)" />
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
            <img :src="thumbUrl(crop.url, 200)" alt="" loading="lazy" decoding="async" class="h-full w-full bg-secondary/20 object-cover" />
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
              <div v-if="ds.status === 'Processing'" class="min-w-[8rem] space-y-1">
                <div class="flex items-center justify-between gap-2 text-xs">
                  <span class="text-amber-500">{{ ds.importMessage || IMPORT_STAGE_LABELS[ds.importStage ?? ''] || 'Processing' }}</span>
                  <span class="shrink-0 tabular-nums text-muted-foreground">{{ Math.round(ds.importProgress ?? 0) }}%</span>
                </div>
                <div class="h-1.5 overflow-hidden rounded-full bg-secondary/60">
                  <div
                    class="h-full rounded-full bg-amber-500 transition-[width] duration-300"
                    :style="{width: `${Math.round(ds.importProgress ?? 0)}%`}"
                  />
                </div>
              </div>
              <span
                v-else
                class="rounded-full px-2 py-1 text-xs"
                :class="ds.status === 'Ready' ? 'bg-emerald-500/10 text-emerald-500' : 'bg-rose-500/10 text-rose-500'"
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
    :index="annotIndex + 1"
    :total="annotImages.length"
    :has-prev="annotHasPrev"
    :has-next="annotHasNext"
    @save="onAnnotatorSave"
    @navigate="navigateAnnotator"
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
              @change="(e) => assignAddFiles((e.target as HTMLInputElement).files)"
            />
            <button
              type="button"
              :disabled="addUploading"
              class="flex w-full flex-col items-center justify-center gap-1 rounded-lg border-2 border-dashed py-6 text-sm transition-colors disabled:opacity-50"
              :class="addDragActive ? 'border-primary bg-primary/10 text-primary' : 'border-border text-muted-foreground hover:border-primary/50'"
              @click="addInputRef?.click()"
              @dragenter="onAddDragEnter"
              @dragover="onAddDragOver"
              @dragleave="onAddDragLeave"
              @drop="onAddDrop"
            >
              <span class="flex items-center gap-2">
                <Upload class="h-4 w-4" />
                {{ addFiles.length ? `${addFiles.length} file(s) selected` : 'Choose image(s)' }}
              </span>
              <span v-if="!addFiles.length" class="text-[10px] text-muted-foreground/80">or drag and drop images here</span>
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
      <div v-if="uploadOpen" class="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-6 backdrop-blur-sm" @click.self="cancelUpload">
        <div class="w-full max-w-md space-y-4 rounded-xl border border-border bg-background p-6 shadow-xl">
          <div class="flex items-center justify-between">
            <h3 class="text-lg font-semibold">Upload dataset</h3>
            <button :disabled="cancelInFlight" class="rounded-md p-1 text-muted-foreground hover:bg-secondary/80" @click="cancelUpload"><X class="h-5 w-5" /></button>
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
              <input ref="zipInputRef" type="file" accept=".zip,application/zip" class="hidden" @change="(e) => assignZipFile((e.target as HTMLInputElement).files?.[0] ?? null)" />
              <button
                type="button"
                :disabled="uploading"
                class="flex w-full flex-col items-center justify-center gap-1 rounded-lg border-2 border-dashed py-8 text-sm transition-colors disabled:opacity-50"
                :class="zipDragActive ? 'border-primary bg-primary/10 text-primary' : 'border-border text-muted-foreground hover:border-primary/50'"
                @click="zipInputRef?.click()"
                @dragenter="onZipDragEnter"
                @dragover="onZipDragOver"
                @dragleave="onZipDragLeave"
                @drop="onZipDrop"
              >
                <span class="flex items-center gap-2">
                  <Upload class="h-4 w-4" />
                  {{ zipFile ? zipFile.name : 'Choose .zip file' }}
                </span>
                <span v-if="!zipFile" class="text-[10px] text-muted-foreground/80">or drag and drop a .zip here</span>
              </button>
            </template>
            <template v-else>
              <input ref="imagesInputRef" type="file" accept="image/*" multiple class="hidden" @change="(e) => assignImageFiles((e.target as HTMLInputElement).files)" />
              <button
                type="button"
                :disabled="uploading"
                class="flex w-full flex-col items-center justify-center gap-1 rounded-lg border-2 border-dashed py-6 text-sm transition-colors disabled:opacity-50"
                :class="imagesDragActive ? 'border-primary bg-primary/10 text-primary' : 'border-border text-muted-foreground hover:border-primary/50'"
                @click="imagesInputRef?.click()"
                @dragenter="onImagesDragEnter"
                @dragover="onImagesDragOver"
                @dragleave="onImagesDragLeave"
                @drop="onImagesDrop"
              >
                <span class="flex items-center gap-2">
                  <Upload class="h-4 w-4" />
                  {{ imageFiles.length ? `${imageFiles.length} file(s) selected` : 'Choose images' }}
                </span>
                <span v-if="!imageFiles.length" class="text-[10px] text-muted-foreground/80">or drag and drop images here</span>
              </button>
            </template>
            <p v-if="uploadError" class="text-xs text-rose-500">{{ uploadError }}</p>
            <div v-if="uploading" class="space-y-2 rounded-lg border border-border bg-secondary/10 px-3 py-3">
              <div class="flex items-center justify-between gap-3 text-xs">
                <span class="truncate text-muted-foreground">{{ uploadProgressLabel }}</span>
                <span class="shrink-0 font-medium tabular-nums">{{ combinedUploadProgress }}%</span>
              </div>
              <div class="h-2 overflow-hidden rounded-full bg-secondary/60">
                <div
                  class="h-full rounded-full bg-primary transition-[width] duration-300 ease-out"
                  :style="{width: `${combinedUploadProgress}%`}"
                />
              </div>
            </div>
            <div class="flex justify-end gap-2 pt-2">
              <button type="button" :disabled="cancelInFlight" class="rounded-md border border-border px-3 py-2 text-sm hover:bg-secondary/50" @click="cancelUpload">
                {{ cancelInFlight ? 'Cancelling…' : (uploading ? 'Cancel upload' : 'Cancel') }}
              </button>
              <button type="submit" :disabled="uploading" class="rounded-md bg-primary px-4 py-2 text-sm font-medium text-primary-foreground hover:bg-primary/90 disabled:opacity-50">{{ uploading ? 'Processing…' : 'Upload' }}</button>
            </div>
          </form>
        </div>
      </div>
    </Transition>
  </Teleport>

  <ImagePreviewModal
    :open="!!previewImage"
    :image-url="previewImage"
    :regions="previewOverlayRegions"
    :color-for-class="previewColorForClass"
    :label-for-class="classNameOf"
    :view-mode="previewViewMode"
    :can-trace-source="previewViewMode === 'crop' && !!previewTraceCrop"
    @close="closePreview"
    @trace-source="traceToSource"
  />
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

<script setup lang="ts">
import {computed, nextTick, onMounted, onUnmounted, ref, watch} from 'vue';
import {
  Activity,
  AlertCircle,
  AlertTriangle,
  ArrowLeft,
  CheckCircle2,
  Clock,
  Cpu,
  Database,
  Download,
  ExternalLink,
  ImageIcon,
  LayoutGrid,
  Minus,
  MoreVertical,
  Play,
  Plus,
  RefreshCw,
  Sparkles,
  Terminal,
  Trash2,
  X,
  Zap,
} from 'lucide-vue-next';
import InferenceScatterChart from '@/components/InferenceScatterChart.vue';
import type {PlotPoint} from '@/components/InferenceScatterChart.vue';
import {DatasetsApi, InferenceApi, staticUrl, type AnnotationRegion, type CropImage, type DatasetImage, type DefectClass} from '@/lib/api';

// ── constants ──────────────────────────────────────────────────────────────
/** Always-available baseline model (server returns this as the first entry too). */
const DEFAULT_MODEL_ID = 'default';
const DEFAULT_MODEL = {id: DEFAULT_MODEL_ID, name: '默认预训练模型', type: 'Pretrained'};
const COLORS = ['#8884d8', '#82ca9d', '#ffc658', '#ff8042', '#0088FE', '#00C49F'];
const ALGO_LIST = ['TSNE', 'UMAP', 'PCA'] as const;
type AlgoKey = (typeof ALGO_LIST)[number];

// ── types ──────────────────────────────────────────────────────────────────
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
type TraceDatasetImage = DatasetImage;
interface InferenceRunSummary {
  id: string;
  name: string;
  status: string;
  datasetName?: string | null;
  algorithm: string;
  viewMode: string;
  createdAt: string;
  updatedAt?: string;
}

// ── state ──────────────────────────────────────────────────────────────────
const apiModels = ref<{id: string; name: string; type?: string}[]>([]);
const apiDatasets = ref<{id: string; name: string; items?: number}[]>([]);
const selectedModel = ref(DEFAULT_MODEL_ID);
const selectedDataset = ref('');
const analysisLabels = ref<string[]>([]);
const algorithm = ref<AlgoKey>('TSNE');
const viewMode = ref<'distribution' | 'anomaly'>('distribution');
const isAnalyzing = ref(false);
const loadingAlgos = ref<Set<AlgoKey>>(new Set());
const cachedAlgos = ref<Set<AlgoKey>>(new Set());
const plotData = ref<PlotPoint[]>([]);

const inferenceDatasetDetail = ref<{id: string; images: TraceDatasetImage[]; defectClasses: DefectClass[]} | null>(null);

// ── golden reference samples (optional) ──────────────────────────────────────
const goldenCropIds = ref<string[]>([]);
const showGoldenDialog = ref(false);
const goldenDraft = ref<Set<string>>(new Set());

const inferenceRuns = ref<InferenceRunSummary[]>([]);
const selectedRunId = ref<string | null>(null);
const inferenceMenuId = ref<string | null>(null);
const inferenceMenuPos = ref<{top: number; left: number} | null>(null);
const runsLoading = ref(true);
const showCreateRunDialog = ref(false);
const newRunNameDraft = ref('');
const creatingRun = ref(false);
const createRunNameInputRef = ref<HTMLInputElement | null>(null);
const editingRunName = ref(false);
const runNameDraft = ref('');
const runNameSaving = ref(false);
const runNameInputRef = ref<HTMLInputElement | null>(null);

// ── preview modal state ────────────────────────────────────────────────────
const previewPoint = ref<PlotPoint | null>(null);
const previewModalView = ref<'crop' | 'source'>('crop');
const showPreviewAnnotations = ref(true);
/** Scale multiplier relative to the fit-to-viewport base scale */
const imagePreviewZoom = ref(1);
/** Preview zoom limits (slightly tighter than raw 0.1–8) */
const PREVIEW_ZOOM_MIN = 0.2;
const PREVIEW_ZOOM_MAX = 6;
/** Fit-to-viewport scale computed once the image loads */
const imagePreviewFitScale = ref(1);
/** Pan offset in CSS pixels applied *after* scale (so 1px pan = 1px movement on screen) */
const imagePreviewPan = ref({x: 0, y: 0});
const imagePreviewWheelRef = ref<HTMLDivElement | null>(null);
const imagePreviewImgRef = ref<HTMLImageElement | null>(null);
/** Natural pixel size of the current preview image (updated on load) */
const imagePreviewNaturalSize = ref({w: 0, h: 0});
/** Viewport size of the drag area (updated via ResizeObserver) */
const previewContainerSize = ref({w: 0, h: 0});

/** Combined transform scale = fitScale * userZoom */
const previewEffectiveScale = computed(() => imagePreviewFitScale.value * imagePreviewZoom.value);

/** True when scaled image exceeds the viewport — only then allow drag-to-pan */
const previewCanPan = computed(() => {
  const nw = imagePreviewNaturalSize.value.w;
  const nh = imagePreviewNaturalSize.value.h;
  const cw = previewContainerSize.value.w;
  const ch = previewContainerSize.value.h;
  if (!nw || !nh || !cw || !ch) return false;
  const s = previewEffectiveScale.value;
  const dw = nw * s;
  const dh = nh * s;
  return dw > cw + 0.5 || dh > ch + 0.5;
});

function updatePreviewContainerSize() {
  const el = imagePreviewWheelRef.value;
  if (!el) return;
  previewContainerSize.value = {w: el.clientWidth, h: el.clientHeight};
}

function computeFitScale() {
  const img = imagePreviewImgRef.value;
  const container = imagePreviewWheelRef.value;
  if (!img || !container) return;
  const nw = img.naturalWidth || img.width;
  const nh = img.naturalHeight || img.height;
  if (!nw || !nh) return;
  imagePreviewNaturalSize.value = {w: nw, h: nh};
  const cw = container.clientWidth - 32;   // 16px padding each side
  const ch = container.clientHeight - 32;
  const scale = Math.min(1, cw / nw, ch / nh);
  imagePreviewFitScale.value = scale;
  imagePreviewZoom.value = 1;
  imagePreviewPan.value = {x: 0, y: 0};
  updatePreviewContainerSize();
}

// ── helpers ────────────────────────────────────────────────────────────────
const selectedRun = computed(() => inferenceRuns.value.find((r) => r.id === selectedRunId.value));
const modelsForSelect = computed(() => apiModels.value.length ? apiModels.value : [DEFAULT_MODEL]);
const datasetsForSelect = computed(() => apiDatasets.value);
const labelList = computed(() => analysisLabels.value);

function mapPoints(raw: any[]): PlotPoint[] {
  return raw.map((p: any) => ({
    id: p.id,
    x: p.x,
    y: p.y,
    cluster: p.cluster,
    url: p.url,
    anomalyScore: p.anomalyScore ?? p.anomaly_score ?? 0,
    label: p.label,
    isGolden: p.isGolden ?? p.is_golden ?? false,
    sourceImageId: p.sourceImageId,
    instanceIndex: p.instanceIndex,
    cropAnnotation: p.cropAnnotation,
  }));
}

// ── API calls ──────────────────────────────────────────────────────────────
async function loadInferenceDatasetForTrace(datasetId: string) {
  try {
    const detail = await DatasetsApi.get(datasetId);
    inferenceDatasetDetail.value = {id: detail.id, images: detail.images, defectClasses: detail.defectClasses};
  } catch {
    inferenceDatasetDetail.value = null;
  }
}

// ── golden helpers ───────────────────────────────────────────────────────────
/** All crops of the current dataset grouped by their defect class. */
const goldenCropsByClass = computed(() => {
  const detail = inferenceDatasetDetail.value;
  if (!detail) return [] as {cls: DefectClass; crops: CropImage[]}[];
  const byClass = new Map<string, CropImage[]>();
  for (const img of detail.images) {
    for (const crop of img.crops ?? []) {
      if (!crop.classId) continue;
      (byClass.get(crop.classId) ?? byClass.set(crop.classId, []).get(crop.classId)!).push(crop);
    }
  }
  return detail.defectClasses
    .map((cls) => ({cls, crops: byClass.get(cls.id) ?? []}))
    .filter((g) => g.crops.length > 0);
});

const goldenClassCount = computed(() => {
  const detail = inferenceDatasetDetail.value;
  if (!detail || goldenCropIds.value.length === 0) return 0;
  const selected = new Set(goldenCropIds.value);
  const classIds = new Set<string>();
  for (const img of detail.images) {
    for (const crop of img.crops ?? []) {
      if (crop.classId && selected.has(crop.id)) classIds.add(crop.classId);
    }
  }
  return classIds.size;
});

function openGoldenDialog() {
  goldenDraft.value = new Set(goldenCropIds.value);
  showGoldenDialog.value = true;
}

function toggleGoldenCrop(cropId: string) {
  const next = new Set(goldenDraft.value);
  if (next.has(cropId)) next.delete(cropId);
  else next.add(cropId);
  goldenDraft.value = next;
}

function toggleGoldenClass(crops: CropImage[], on: boolean) {
  const next = new Set(goldenDraft.value);
  for (const c of crops) {
    if (on) next.add(c.id);
    else next.delete(c.id);
  }
  goldenDraft.value = next;
}

function classAllSelected(crops: CropImage[]): boolean {
  return crops.length > 0 && crops.every((c) => goldenDraft.value.has(c.id));
}

async function applyGoldenSelection() {
  goldenCropIds.value = [...goldenDraft.value];
  showGoldenDialog.value = false;
  const runId = selectedRunId.value;
  if (runId) {
    try {
      await InferenceApi.patchRun(runId, {goldenCropIds: goldenCropIds.value});
    } catch { /* ignore */ }
  }
}

function clearGoldenSelection() {
  goldenCropIds.value = [];
  goldenDraft.value = new Set();
  const runId = selectedRunId.value;
  if (runId) void InferenceApi.patchRun(runId, {goldenCropIds: []});
}

/** Golden crops belong to a specific dataset — reset selection when the user
 *  picks a different dataset (only on manual change, not on run load). */
function onDatasetManualChange() {
  clearGoldenSelection();
}

async function fetchInferenceRuns() {
  try {
    inferenceRuns.value = await InferenceApi.listRuns();
  } catch { /* ignore */ } finally {
    runsLoading.value = false;
  }
}

async function loadRunDetail(id: string) {
  try {
    const row = await InferenceApi.getRun(id) as {
      modelId?: string | null;
      datasetId?: string | null;
      goldenCropIds?: string[] | null;
      algorithm: string;
      viewMode: 'distribution' | 'anomaly';
      resultJson?: {labels?: string[]; points?: any[]} | null;
      cachedAlgorithms?: string[];
    };
    selectedModel.value = row.modelId || DEFAULT_MODEL_ID;
    selectedDataset.value = row.datasetId || '';
    goldenCropIds.value = row.goldenCropIds ?? [];
    const a = row.algorithm?.toLowerCase() || 'tsne';
    algorithm.value = a === 'umap' ? 'UMAP' : a === 'pca' ? 'PCA' : 'TSNE';
    viewMode.value = row.viewMode;
    cachedAlgos.value = new Set((row.cachedAlgorithms ?? []).map((s) => s.toUpperCase() as AlgoKey));
    if (row.datasetId) void loadInferenceDatasetForTrace(row.datasetId);
    else inferenceDatasetDetail.value = null;
    if (row.resultJson?.points?.length) {
      plotData.value = mapPoints(row.resultJson.points);
      analysisLabels.value = row.resultJson.labels ?? [];
    } else {
      plotData.value = [];
      analysisLabels.value = [];
    }
  } catch { /* ignore */ }
}

// ── lifecycle ──────────────────────────────────────────────────────────────
onMounted(() => {
  InferenceApi.listModels()
    .then((d) => {
      apiModels.value = d.length ? d : [DEFAULT_MODEL];
      if (!apiModels.value.some((m) => m.id === selectedModel.value)) {
        selectedModel.value = apiModels.value[0]?.id ?? DEFAULT_MODEL_ID;
      }
    })
    .catch(() => { apiModels.value = [DEFAULT_MODEL]; selectedModel.value = DEFAULT_MODEL_ID; });
  DatasetsApi.list()
    .then((d) => { apiDatasets.value = d.map((x) => ({id: x.id, name: x.name, items: x.items})); if (d.length) selectedDataset.value = d[0].id; })
    .catch(() => { apiDatasets.value = []; });
  void fetchInferenceRuns();
});

// ── watchers ───────────────────────────────────────────────────────────────
watch(selectedRunId, (id) => {
  editingRunName.value = false;
  runNameDraft.value = '';
  if (id) void loadRunDetail(id);
  else { plotData.value = []; analysisLabels.value = []; cachedAlgos.value = new Set(); }
});

watch(selectedRun, (run) => {
  if (!editingRunName.value) runNameDraft.value = run?.name ?? '';
});

watch(selectedDataset, (dsId) => {
  if (dsId) void loadInferenceDatasetForTrace(dsId);
  else inferenceDatasetDetail.value = null;
}, {immediate: true});

// Close context menu on scroll or resize — use a stopWatch ref to clean up properly
let _menuCleanup: (() => void) | null = null;
watch(inferenceMenuId, (id) => {
  _menuCleanup?.();
  _menuCleanup = null;
  if (!id) return;
  const close = () => { inferenceMenuId.value = null; inferenceMenuPos.value = null; };
  window.addEventListener('scroll', close, true);
  window.addEventListener('resize', close);
  _menuCleanup = () => { window.removeEventListener('scroll', close, true); window.removeEventListener('resize', close); };
});

// ── context menu ───────────────────────────────────────────────────────────
function closeInferenceMenu() {
  inferenceMenuId.value = null;
  inferenceMenuPos.value = null;
}

function openInferenceMenu(e: MouseEvent, runId: string) {
  if (inferenceMenuId.value === runId) { closeInferenceMenu(); return; }
  const rect = (e.currentTarget as HTMLElement).getBoundingClientRect();
  const mw = 176, pad = 8, gap = 4, menuH = 120;
  let top = rect.bottom + gap;
  if (top + menuH > window.innerHeight - pad) top = Math.max(pad, rect.top - menuH - gap);
  inferenceMenuId.value = runId;
  inferenceMenuPos.value = {top, left: Math.max(pad, Math.min(rect.right - mw, window.innerWidth - mw - pad))};
}

// ── inference run CRUD ─────────────────────────────────────────────────────
function openCreateRunDialog() {
  newRunNameDraft.value = `推理 ${new Date().toLocaleString()}`;
  showCreateRunDialog.value = true;
  void nextTick(() => createRunNameInputRef.value?.focus());
}

function closeCreateRunDialog() {
  if (creatingRun.value) return;
  showCreateRunDialog.value = false;
}

async function createNewInferenceRun(name: string): Promise<boolean> {
  const cleanName = name.trim();
  if (!cleanName) return false;
  try {
    const row = await InferenceApi.createRun({
      name: cleanName,
      modelId: apiModels.value[0]?.id || DEFAULT_MODEL_ID,
      datasetMode: 'existing',
      datasetId: apiDatasets.value[0]?.id || null,
      algorithm: 'tsne',
      viewMode: 'distribution',
    }) as {id: string};
    await fetchInferenceRuns();
    selectedRunId.value = row.id;
    return true;
  } catch {
    return false;
  }
}

async function confirmCreateRun() {
  const name = newRunNameDraft.value.trim();
  if (!name || creatingRun.value) return;
  creatingRun.value = true;
  try {
    const ok = await createNewInferenceRun(name);
    if (ok) showCreateRunDialog.value = false;
  } finally {
    creatingRun.value = false;
  }
}

function onCreateRunInputKeydown(e: KeyboardEvent) {
  if (e.key === 'Enter') {
    e.preventDefault();
    void confirmCreateRun();
    return;
  }
  if (e.key === 'Escape') {
    e.preventDefault();
    closeCreateRunDialog();
  }
}

async function handleDeleteInferenceRun(id: string, name: string) {
  closeInferenceMenu();
  if (!window.confirm(`删除推理任务 "${name}"？此操作不可撤销。`)) return;
  try {
    await InferenceApi.deleteRun(id);
    if (selectedRunId.value === id) selectedRunId.value = null;
    await fetchInferenceRuns();
  } catch { /* ignore */ }
}

function beginRunNameEdit() {
  if (!selectedRun.value || runNameSaving.value) return;
  runNameDraft.value = selectedRun.value.name;
  editingRunName.value = true;
  void nextTick(() => runNameInputRef.value?.focus());
}

function cancelRunNameEdit() {
  editingRunName.value = false;
  runNameDraft.value = selectedRun.value?.name ?? '';
}

async function saveRunNameEdit() {
  const runId = selectedRunId.value;
  const currentName = selectedRun.value?.name ?? '';
  const nextName = runNameDraft.value.trim();
  if (!runId) return;
  if (!nextName) {
    runNameDraft.value = currentName;
    editingRunName.value = false;
    return;
  }
  if (nextName === currentName) {
    editingRunName.value = false;
    return;
  }
  if (runNameSaving.value) return;
  runNameSaving.value = true;
  try {
    const row = await InferenceApi.patchRun(runId, {name: nextName}) as {name?: string; updatedAt?: string};
    const savedName = (row.name ?? nextName).trim() || nextName;
    inferenceRuns.value = inferenceRuns.value.map((r) => (
      r.id === runId ? {...r, name: savedName, updatedAt: row.updatedAt ?? r.updatedAt} : r
    ));
    runNameDraft.value = savedName;
    editingRunName.value = false;
  } catch (e) {
    console.error(e);
    runNameDraft.value = currentName;
    editingRunName.value = false;
  } finally {
    runNameSaving.value = false;
  }
}

function onRunNameInputKeydown(e: KeyboardEvent) {
  if (e.key === 'Enter') {
    e.preventDefault();
    void saveRunNameEdit();
    return;
  }
  if (e.key === 'Escape') {
    e.preventDefault();
    cancelRunNameEdit();
  }
}

// ── algorithm switching ────────────────────────────────────────────────────
async function switchAlgorithm(algo: AlgoKey) {
  if (algorithm.value === algo) return;
  algorithm.value = algo;
  const runId = selectedRunId.value;
  if (!runId || plotData.value.length === 0) return;
  const algoLower = algo.toLowerCase();

  if (cachedAlgos.value.has(algo)) {
    try {
      const proj = await InferenceApi.getProjection(runId, algoLower) as {labels: string[]; points: any[]};
      plotData.value = mapPoints(proj.points);
      analysisLabels.value = proj.labels;
      void InferenceApi.patchRun(runId, {algorithm: algoLower});
      return;
    } catch { /* fall through */ }
  }

  loadingAlgos.value = new Set([...loadingAlgos.value, algo]);
  try {
    const proj = await InferenceApi.computeProjection(runId, algoLower) as {labels: string[]; points: any[]};
    if (algorithm.value === algo) {
      plotData.value = mapPoints(proj.points);
      analysisLabels.value = proj.labels;
    }
    cachedAlgos.value = new Set([...cachedAlgos.value, algo]);
    void InferenceApi.patchRun(runId, {algorithm: algoLower});
  } catch { /* ignore */ } finally {
    const s = new Set(loadingAlgos.value);
    s.delete(algo);
    loadingAlgos.value = s;
  }
}

// ── run analysis ───────────────────────────────────────────────────────────
async function handleRunAnalysis() {
  const runId = selectedRunId.value;
  if (!runId || !selectedDataset.value) return;

  try {
    await InferenceApi.patchRun(runId, {
      modelId: selectedModel.value || null,
      datasetMode: 'existing',
      datasetId: selectedDataset.value,
      goldenCropIds: goldenCropIds.value,
      algorithm: algorithm.value.toLowerCase(),
      viewMode: viewMode.value,
    });
  } catch { /* ignore */ }

  isAnalyzing.value = true;
  cachedAlgos.value = new Set();
  try {
    const row = await InferenceApi.analyze(runId) as {resultJson?: {labels?: string[]; points?: any[]}; cachedAlgorithms?: string[]};
    if (row.resultJson?.points) {
      analysisLabels.value = row.resultJson.labels ?? [];
      plotData.value = mapPoints(row.resultJson.points);
      cachedAlgos.value = new Set((row.cachedAlgorithms ?? [algorithm.value.toLowerCase()]).map((s) => s.toUpperCase() as AlgoKey));
    }
    await fetchInferenceRuns();
  } catch (e) {
    console.error(e);
  } finally {
    isAnalyzing.value = false;
  }
}

// ── source tracing ─────────────────────────────────────────────────────────
function findSourceImage(sourceImageId: string): TraceDatasetImage | null {
  return inferenceDatasetDetail.value?.images.find((i) => i.id === sourceImageId) ?? null;
}

function regionToLabel(r: AnnotationRegion, i: number): AnnotationLabel {
  return {label_id: i, points: r.points as [number, number][], isSubtract: r.isSubtract};
}

function cropAnnotationFromPoint(p: PlotPoint): Annotation | null {
  if (!p.cropAnnotation?.length) return null;
  return {id: p.id, labels: p.cropAnnotation.map((lbl, i) => ({label_id: i, points: lbl.points as [number, number][], isSubtract: lbl.isSubtract}))};
}

function sourceAnnotationFromPoint(p: PlotPoint): Annotation | null {
  const img = p.sourceImageId ? findSourceImage(p.sourceImageId) : null;
  if (!img?.regions?.length) return null;
  const idx = p.instanceIndex ?? 0;
  const target = img.regions[idx] ?? null;
  const labels = target ? [regionToLabel(target, idx)] : img.regions.map(regionToLabel);
  return {id: img.id, labels};
}

const previewAnnotation = computed((): Annotation | null => {
  const p = previewPoint.value;
  if (!p?.url) return null;
  return previewModalView.value === 'crop' ? cropAnnotationFromPoint(p) : sourceAnnotationFromPoint(p);
});

const previewDisplayUrl = computed(() => {
  const p = previewPoint.value;
  if (!p?.url) return null;
  if (previewModalView.value === 'crop') return staticUrl(p.url);
  const src = p.sourceImageId ? findSourceImage(p.sourceImageId) : null;
  return staticUrl(src?.url ?? p.url);
});

const canTraceSource = computed(() => {
  const p = previewPoint.value;
  return !!(p?.sourceImageId && findSourceImage(p.sourceImageId));
});

// ── preview modal open/close ───────────────────────────────────────────────
function openPreview(point: PlotPoint) {
  if (!point.url) return;
  previewPoint.value = point;
  previewModalView.value = 'crop';
  showPreviewAnnotations.value = true;
  imagePreviewZoom.value = 1;
  imagePreviewFitScale.value = 1;
  imagePreviewPan.value = {x: 0, y: 0};
}

function closePreview() {
  previewPoint.value = null;
  previewModalView.value = 'crop';
}

// ── preview modal: keyboard + wheel — managed manually ─────────────────────
let _kbdHandler: ((e: KeyboardEvent) => void) | null = null;
let _wheelHandler: ((e: WheelEvent) => void) | null = null;
let _wheelEl: HTMLElement | null = null;
let _prevBodyOverflow = '';
let _previewResizeObs: ResizeObserver | null = null;

function attachPreviewGlobalHandlers() {
  detachPreviewGlobalHandlers();
  _prevBodyOverflow = document.body.style.overflow;
  document.body.style.overflow = 'hidden';

  _kbdHandler = (e: KeyboardEvent) => {
    if (e.key === 'Escape') { closePreview(); return; }
    if (e.key === ' ') {
      e.preventDefault();
      if (previewAnnotation.value?.labels?.length) showPreviewAnnotations.value = !showPreviewAnnotations.value;
    }
  };
  window.addEventListener('keydown', _kbdHandler);

  void nextTick(() => {
    const el = imagePreviewWheelRef.value;
    if (!el) return;
    updatePreviewContainerSize();
    _previewResizeObs = new ResizeObserver(() => {
      updatePreviewContainerSize();
    });
    _previewResizeObs.observe(el);
    _wheelEl = el;
    _wheelHandler = (e: WheelEvent) => {
      e.preventDefault();
      imagePreviewZoom.value = Math.min(
        PREVIEW_ZOOM_MAX,
        Math.max(PREVIEW_ZOOM_MIN, imagePreviewZoom.value - e.deltaY * 0.002),
      );
    };
    el.addEventListener('wheel', _wheelHandler, {passive: false});
  });
}

function detachPreviewGlobalHandlers() {
  if (_kbdHandler) { window.removeEventListener('keydown', _kbdHandler); _kbdHandler = null; }
  if (_wheelEl && _wheelHandler) { _wheelEl.removeEventListener('wheel', _wheelHandler); _wheelEl = null; _wheelHandler = null; }
  _previewResizeObs?.disconnect();
  _previewResizeObs = null;
  document.body.style.overflow = _prevBodyOverflow;
}

watch(previewPoint, (p) => {
  if (p) {
    imagePreviewZoom.value = 1;
    imagePreviewFitScale.value = 1;
    imagePreviewPan.value = {x: 0, y: 0};
    attachPreviewGlobalHandlers();
  } else {
    detachPreviewGlobalHandlers();
  }
});

// Also reset pan+fit when switching between crop/source view
watch(previewModalView, () => {
  imagePreviewZoom.value = 1;
  imagePreviewFitScale.value = 1;
  imagePreviewPan.value = {x: 0, y: 0};
  // fitScale will be recomputed by onload
});

watch(previewCanPan, (can) => {
  if (!can) {
    imagePreviewPan.value = {x: 0, y: 0};
    previewDragging.value = false;
  }
});

onUnmounted(() => {
  detachPreviewGlobalHandlers();
  _menuCleanup?.();
});

// ── preview modal: drag-to-pan ─────────────────────────────────────────────
// Instead of scrolling an overflow container, we manipulate imagePreviewPan
// via CSS transform — this works at any zoom level including <100%.
const previewDragging = ref(false);
let _dragStartClient = {x: 0, y: 0};
let _dragStartPan = {x: 0, y: 0};
const DRAG_THRESHOLD = 4;
let _didDrag = false;

function onPreviewPointerDown(e: PointerEvent) {
  if (e.button !== 0) return;
  if (!previewCanPan.value) return;
  _didDrag = false;
  previewDragging.value = true;
  _dragStartClient = {x: e.clientX, y: e.clientY};
  _dragStartPan = {...imagePreviewPan.value};
  (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
}

function onPreviewPointerMove(e: PointerEvent) {
  if (!previewDragging.value) return;
  const dx = e.clientX - _dragStartClient.x;
  const dy = e.clientY - _dragStartClient.y;
  if (!_didDrag && Math.hypot(dx, dy) < DRAG_THRESHOLD) return;
  _didDrag = true;
  imagePreviewPan.value = {x: _dragStartPan.x + dx, y: _dragStartPan.y + dy};
}

function onPreviewPointerUp(e: PointerEvent) {
  previewDragging.value = false;
  (e.currentTarget as HTMLElement).releasePointerCapture(e.pointerId);
}

// Reset pan when zoom resets to exactly 1 (user clicks zoom-out button back to base)
watch(imagePreviewZoom, (z) => {
  if (z === 1) imagePreviewPan.value = {x: 0, y: 0};
  void nextTick(() => updatePreviewContainerSize());
});
</script>

<template>
  <!-- ════════════════ RUN LIST VIEW ════════════════ -->
  <div v-if="!selectedRunId" class="flex h-full flex-col gap-6 overflow-y-auto p-6">
    <div class="flex items-center justify-between">
      <div>
        <h2 class="text-2xl font-semibold tracking-tight">推理测试</h2>
        <p class="mt-1 text-sm text-muted-foreground">创建独立推理任务，配置后运行分析；结果保存在服务端。</p>
      </div>
      <button type="button" class="flex items-center gap-2 rounded-md bg-primary px-4 py-2 text-sm font-medium text-primary-foreground transition-colors hover:bg-primary/90" @click="openCreateRunDialog()">
        <Plus class="h-4 w-4" />新建推理
      </button>
    </div>

    <p v-if="runsLoading" class="text-sm text-muted-foreground">加载中…</p>

    <div v-else-if="inferenceRuns.length === 0" class="flex flex-col items-center justify-center gap-4 rounded-xl border border-dashed border-border bg-secondary/5 py-16 text-center">
      <div class="flex h-12 w-12 items-center justify-center rounded-lg bg-secondary/30">
        <Terminal class="h-6 w-6 text-muted-foreground" />
      </div>
      <div class="space-y-1">
        <p class="text-sm font-medium">暂无推理任务</p>
        <p class="max-w-sm text-xs text-muted-foreground">点击「新建推理」创建任务，每个任务有独立配置与结果。</p>
      </div>
      <button type="button" class="flex items-center gap-2 rounded-md bg-primary px-4 py-2 text-sm font-medium text-primary-foreground hover:bg-primary/90" @click="openCreateRunDialog()">
        <Plus class="h-4 w-4" />新建推理
      </button>
    </div>

    <div v-else class="grid grid-cols-1 gap-4">
      <div v-for="run in inferenceRuns" :key="run.id" class="flex flex-col rounded-xl border border-border bg-secondary/5 transition-colors hover:border-primary/40">
        <!-- Run header -->
        <button type="button" class="flex w-full items-center justify-between border-b border-border bg-secondary/10 px-5 py-3.5 text-left transition-colors hover:bg-secondary/20" @click="selectedRunId = run.id">
          <div class="flex items-center gap-3">
            <div class="flex h-8 w-8 items-center justify-center rounded-lg bg-primary/10">
              <Terminal class="h-4 w-4 text-primary" />
            </div>
            <div>
              <h3 class="text-sm font-semibold">{{ run.name }}</h3>
              <p class="font-mono text-[10px] text-muted-foreground">{{ run.id }}</p>
            </div>
          </div>
          <div class="flex items-center gap-4">
            <div class="flex items-center gap-1.5">
              <CheckCircle2 v-if="run.status === 'Completed'" class="h-3.5 w-3.5 text-emerald-500" />
              <Clock v-else-if="run.status === 'Draft'" class="h-3.5 w-3.5 text-muted-foreground" />
              <AlertCircle v-else-if="run.status === 'Failed'" class="h-3.5 w-3.5 text-rose-500" />
              <Activity v-else class="h-3.5 w-3.5 text-amber-500" />
              <span class="text-xs font-medium">{{ run.status }}</span>
            </div>
            <div class="h-4 w-px bg-border" />
            <span class="text-xs text-muted-foreground">{{ run.createdAt ? new Date(run.createdAt).toLocaleString() : '—' }}</span>
          </div>
        </button>

        <!-- Run metadata -->
        <div class="grid cursor-pointer grid-cols-2 gap-4 px-5 py-3 transition-colors hover:bg-secondary/10 sm:grid-cols-4" role="button" tabindex="0" @click="selectedRunId = run.id" @keydown.enter.prevent="selectedRunId = run.id" @keydown.space.prevent="selectedRunId = run.id">
          <div class="flex flex-col gap-0.5">
            <p class="text-[9px] font-bold uppercase tracking-widest text-muted-foreground">数据源</p>
            <p class="text-sm">已有数据集</p>
          </div>
          <div class="flex flex-col gap-0.5">
            <p class="text-[9px] font-bold uppercase tracking-widest text-muted-foreground">数据集</p>
            <p class="text-sm">{{ run.datasetName ?? '—' }}</p>
          </div>
          <div class="flex flex-col gap-0.5">
            <p class="text-[9px] font-bold uppercase tracking-widest text-muted-foreground">算法 / 视图</p>
            <p class="text-sm">{{ run.algorithm.toUpperCase() }} · {{ run.viewMode }}</p>
          </div>
          <div class="flex min-h-[2rem] items-start justify-end">
            <button type="button" class="rounded p-1 transition-colors hover:bg-secondary/30" aria-label="推理任务操作" :aria-expanded="inferenceMenuId === run.id" @click.stop="openInferenceMenu($event, run.id)">
              <MoreVertical class="h-4 w-4 text-muted-foreground" />
            </button>
            <Teleport to="body">
              <Transition name="fade">
                <div v-if="inferenceMenuId === run.id && inferenceMenuPos">
                  <div class="fixed inset-0 z-[50]" @click="closeInferenceMenu" />
                  <div class="fixed z-[60] w-44 overflow-hidden rounded-lg border border-border bg-background shadow-xl" :style="{top: `${inferenceMenuPos.top}px`, left: `${inferenceMenuPos.left}px`}">
                    <div class="flex flex-col p-1">
                      <button type="button" class="flex items-center gap-2 rounded-md px-3 py-2 text-left text-xs transition-colors hover:bg-secondary/50" @click="selectedRunId = run.id; closeInferenceMenu();">
                        <ExternalLink class="h-3.5 w-3.5" />打开
                      </button>
                      <div class="my-1 h-px bg-border" />
                      <button type="button" class="flex items-center gap-2 rounded-md px-3 py-2 text-left text-xs text-rose-500 transition-colors hover:bg-rose-500/10" @click="handleDeleteInferenceRun(run.id, run.name)">
                        <Trash2 class="h-3.5 w-3.5" />删除
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

    <Teleport to="body">
      <Transition name="fade">
        <div v-if="showCreateRunDialog" class="fixed inset-0 z-[90]">
          <div class="absolute inset-0 bg-black/60" @click="closeCreateRunDialog" />
          <div class="absolute left-1/2 top-1/2 w-[min(92vw,28rem)] -translate-x-1/2 -translate-y-1/2 rounded-xl border border-border bg-background p-5 shadow-2xl">
            <div class="mb-4 flex items-center justify-between">
              <h3 class="text-sm font-semibold tracking-tight">新建推理任务</h3>
              <button type="button" class="rounded p-1 text-muted-foreground transition-colors hover:bg-secondary/50 hover:text-foreground" aria-label="关闭" @click="closeCreateRunDialog">
                <X class="h-4 w-4" />
              </button>
            </div>
            <div class="space-y-2">
              <label class="text-xs text-muted-foreground">项目名</label>
              <input
                ref="createRunNameInputRef"
                v-model="newRunNameDraft"
                type="text"
                maxlength="120"
                class="w-full rounded-md border border-border bg-background px-3 py-2 text-sm outline-none ring-0 focus:border-primary"
                placeholder="请输入推理项目名"
                @keydown="onCreateRunInputKeydown"
              />
            </div>
            <div class="mt-5 flex items-center justify-end gap-2">
              <button type="button" class="rounded-md border border-border px-3 py-1.5 text-xs font-medium transition-colors hover:bg-secondary/40" :disabled="creatingRun" @click="closeCreateRunDialog">取消</button>
              <button type="button" class="rounded-md bg-primary px-3 py-1.5 text-xs font-semibold text-primary-foreground transition-colors hover:bg-primary/90 disabled:opacity-60" :disabled="creatingRun || !newRunNameDraft.trim()" @click="confirmCreateRun">
                {{ creatingRun ? '创建中…' : '确认创建' }}
              </button>
            </div>
          </div>
        </div>
      </Transition>
    </Teleport>
  </div>

  <!-- ════════════════ RUN DETAIL VIEW ════════════════ -->
  <div v-else class="flex h-full min-h-0 flex-col">
    <!-- Top bar -->
    <div class="flex shrink-0 items-center gap-3 border-b border-border bg-background/80 px-5 py-3 backdrop-blur-sm">
      <button type="button" class="rounded-md p-1.5 transition-colors hover:bg-secondary/50" aria-label="返回列表" @click="selectedRunId = null">
        <ArrowLeft class="h-4 w-4" />
      </button>
      <div class="min-w-0 flex-1">
        <input
          v-if="editingRunName"
          ref="runNameInputRef"
          v-model="runNameDraft"
          type="text"
          maxlength="120"
          class="w-full rounded-md border border-primary/40 bg-background px-2 py-0.5 text-base font-semibold tracking-tight outline-none ring-0 focus:border-primary"
          aria-label="推理任务名称"
          @keydown="onRunNameInputKeydown"
          @blur="saveRunNameEdit"
        />
        <button
          v-else
          type="button"
          class="w-full text-left"
          title="点击改名"
          @click="beginRunNameEdit"
        >
          <h2 class="truncate text-base font-semibold tracking-tight">{{ selectedRun?.name ?? '推理任务' }}</h2>
        </button>
        <p class="font-mono text-[10px] text-muted-foreground">{{ selectedRunId }}</p>
      </div>
      <div class="flex shrink-0 items-center gap-2">
        <button type="button" class="flex items-center gap-1.5 rounded-md border border-border px-3 py-1.5 text-xs font-medium transition-colors hover:bg-secondary/50">
          <Download class="h-3.5 w-3.5" />导出特征
        </button>
        <button
          type="button"
          :disabled="isAnalyzing || !selectedDataset"
          class="flex items-center gap-1.5 rounded-md bg-primary px-4 py-1.5 text-xs font-semibold text-primary-foreground transition-colors hover:bg-primary/90 disabled:opacity-50"
          @click="handleRunAnalysis"
        >
          <Play v-if="!isAnalyzing" class="h-3.5 w-3.5" />
          <RefreshCw v-else class="h-3.5 w-3.5 animate-spin" />
          {{ isAnalyzing ? '分析中…' : 'Run Analysis' }}
        </button>
      </div>
    </div>

    <!-- Sidebar + chart -->
    <div class="flex min-h-0 flex-1 overflow-hidden">
      <!-- Sidebar -->
      <div class="flex w-64 shrink-0 flex-col gap-4 overflow-y-auto border-r border-border bg-secondary/5 p-4">
        <!-- Model -->
        <div class="space-y-2">
          <div class="flex items-center gap-2 text-[9px] font-bold uppercase tracking-widest text-muted-foreground">
            <Cpu class="h-3 w-3" />模型选择
          </div>
          <select v-model="selectedModel" class="w-full rounded-md border border-border bg-background px-2.5 py-1.5 text-xs focus:outline-none focus:ring-1 focus:ring-ring">
            <option v-for="m in modelsForSelect" :key="m.id" :value="m.id">{{ m.name }}</option>
          </select>
        </div>

        <div class="h-px bg-border" />

        <!-- Dataset -->
        <div class="space-y-2">
          <div class="flex items-center gap-2 text-[9px] font-bold uppercase tracking-widest text-muted-foreground">
            <Database class="h-3 w-3" />数据集
          </div>
          <select v-model="selectedDataset" class="w-full rounded-md border border-border bg-background px-2.5 py-1.5 text-xs focus:outline-none focus:ring-1 focus:ring-ring" @change="onDatasetManualChange">
            <option v-if="!datasetsForSelect.length" value="" disabled>暂无数据集</option>
            <option v-for="d in datasetsForSelect" :key="d.id" :value="d.id">{{ d.name }}</option>
          </select>
          <p class="text-[10px] leading-relaxed text-muted-foreground/70">
            仅分析已有数据集的裁剪图（含对应 mask），请先在数据集页面上传并标注图片。
          </p>
        </div>

        <div class="h-px bg-border" />

        <!-- Golden reference samples (optional) -->
        <div class="space-y-2">
          <div class="flex items-center gap-2 text-[9px] font-bold uppercase tracking-widest text-muted-foreground">
            <Sparkles class="h-3 w-3 text-amber-400" />Golden 参考样本
            <span class="font-normal lowercase tracking-normal text-muted-foreground/60">（可选）</span>
          </div>
          <div class="rounded-md border border-border bg-background px-2.5 py-2">
            <div class="flex items-center justify-between">
              <span class="text-xs">
                <template v-if="goldenCropIds.length">
                  <span class="font-semibold text-amber-500">{{ goldenCropIds.length }}</span> 个 · {{ goldenClassCount }} 类
                </template>
                <span v-else class="text-muted-foreground">未选择</span>
              </span>
              <button
                type="button"
                class="flex items-center gap-1 rounded border border-border px-2 py-0.5 text-[10px] font-medium transition-colors hover:bg-secondary/40 disabled:opacity-50"
                :disabled="!goldenCropsByClass.length"
                @click="openGoldenDialog"
              >
                <Sparkles class="h-3 w-3" />选择
              </button>
            </div>
            <button
              v-if="goldenCropIds.length"
              type="button"
              class="mt-1.5 text-[10px] text-muted-foreground underline-offset-2 hover:text-rose-500 hover:underline"
              @click="clearGoldenSelection"
            >
              清空 golden
            </button>
          </div>
          <p class="text-[10px] leading-relaxed text-muted-foreground/70">
            选择后，异常排序以 golden 为基准（偏离越大越异常）；不选则按类内分布自动评分。
          </p>
        </div>

        <!-- Legend -->
        <template v-if="plotData.length > 0 && viewMode === 'distribution'">
          <div class="h-px bg-border" />
          <div class="space-y-2">
            <div class="text-[9px] font-bold uppercase tracking-widest text-muted-foreground">类别图例</div>
            <div class="space-y-1.5">
              <div v-for="(name, i) in labelList" :key="name" class="flex items-center gap-2">
                <div class="h-2.5 w-2.5 shrink-0 rounded-full" :style="{backgroundColor: COLORS[i % COLORS.length]}" />
                <span class="truncate text-[11px] text-muted-foreground">{{ name }}</span>
              </div>
              <div v-if="goldenCropIds.length" class="flex items-center gap-2 pt-0.5">
                <div class="h-2.5 w-2.5 shrink-0 rotate-45 border-2 border-amber-400 bg-transparent" />
                <span class="truncate text-[11px] text-muted-foreground">Golden 参考</span>
              </div>
            </div>
          </div>
        </template>
      </div>

      <!-- Main chart/anomaly area -->
      <div class="flex min-w-0 flex-1 flex-col overflow-hidden">
        <!-- Toolbar -->
        <div class="flex shrink-0 items-center justify-between gap-3 border-b border-border bg-secondary/5 px-4 py-2.5">
          <div class="flex items-center gap-2">
            <!-- View toggle -->
            <div class="flex rounded-lg bg-secondary/20 p-0.5">
              <button type="button" class="flex items-center gap-1.5 rounded px-3 py-1 text-[10px] font-bold transition-all" :class="viewMode === 'distribution' ? 'bg-background text-foreground shadow-sm' : 'text-muted-foreground hover:text-foreground'" @click="viewMode = 'distribution'">
                <Activity class="h-3 w-3" />分布视图
              </button>
              <button type="button" class="flex items-center gap-1.5 rounded px-3 py-1 text-[10px] font-bold transition-all" :class="viewMode === 'anomaly' ? 'bg-background text-foreground shadow-sm' : 'text-muted-foreground hover:text-foreground'" @click="viewMode = 'anomaly'">
                <LayoutGrid class="h-3 w-3" />异常视图
              </button>
            </div>

            <!-- Algorithm tabs -->
            <template v-if="viewMode === 'distribution'">
              <div class="mx-1 h-4 w-px bg-border" />
              <div class="flex items-center gap-1">
                <button
                  v-for="algo in ALGO_LIST"
                  :key="algo"
                  type="button"
                  class="relative flex items-center gap-1.5 rounded px-3 py-1 text-[10px] font-bold transition-all"
                  :class="algorithm === algo ? 'bg-primary/15 text-primary ring-1 ring-primary/30' : 'text-muted-foreground hover:bg-secondary/30 hover:text-foreground'"
                  :disabled="loadingAlgos.has(algo)"
                  @click="plotData.length > 0 ? switchAlgorithm(algo) : (algorithm = algo)"
                >
                  <RefreshCw v-if="loadingAlgos.has(algo)" class="h-2.5 w-2.5 animate-spin" />
                  <Zap v-else-if="cachedAlgos.has(algo)" class="h-2.5 w-2.5 text-amber-400" />
                  {{ algo }}
                  <span v-if="cachedAlgos.has(algo) && algorithm !== algo" class="absolute -right-0.5 -top-0.5 h-1.5 w-1.5 rounded-full bg-emerald-500" title="已缓存，可即时切换" />
                </button>
              </div>
            </template>
          </div>
          <div class="flex items-center gap-3">
            <span class="text-[10px] text-muted-foreground">{{ plotData.length }} 条</span>
            <span v-if="viewMode === 'distribution'" class="rounded bg-secondary/20 px-2 py-0.5 text-[9px] text-muted-foreground">滚轮缩放 · 拖动平移</span>
          </div>
        </div>

        <!-- Chart area -->
        <div class="relative min-h-0 flex-1 overflow-hidden">
          <!-- Distribution scatter -->
          <div v-if="viewMode === 'distribution'" class="h-full w-full">
            <InferenceScatterChart v-if="plotData.length > 0" :plot-data="plotData" :label-list="labelList" @preview="openPreview" />
            <div v-else class="flex h-full flex-col items-center justify-center gap-4 text-muted-foreground">
              <div class="rounded-full bg-secondary/20 p-5"><ImageIcon class="h-8 w-8 opacity-20" /></div>
              <p class="max-w-xs text-center text-sm italic">选择模型与数据集后，点击「Run Analysis」可视化特征分布。</p>
            </div>
          </div>

          <!-- Anomaly grid -->
          <div v-else class="h-full overflow-y-auto p-5">
            <template v-if="plotData.length > 0">
              <div v-for="(cat, catIdx) in labelList" :key="cat" class="mb-8 space-y-3">
                <template v-if="plotData.filter((p) => (p.label || labelList[p.cluster]) === cat).length > 0">
                  <div class="flex items-center justify-between border-b border-border pb-2">
                    <div class="flex items-center gap-2">
                      <div class="h-3 w-3 rounded-full" :style="{backgroundColor: COLORS[catIdx % COLORS.length]}" />
                      <h3 class="text-sm font-semibold">{{ cat }}</h3>
                      <span class="rounded bg-secondary/20 px-1.5 py-0.5 text-[10px] text-muted-foreground">{{ plotData.filter((p) => (p.label || labelList[p.cluster]) === cat).length }} 项</span>
                    </div>
                    <div class="flex items-center gap-1 text-[10px] text-muted-foreground">
                      <AlertTriangle class="h-3 w-3 text-rose-500" />按异常得分排序
                    </div>
                  </div>
                  <div class="grid grid-cols-3 gap-3 sm:grid-cols-4 md:grid-cols-5 xl:grid-cols-7">
                    <div
                      v-for="point in [...plotData.filter((p) => (p.label || labelList[p.cluster]) === cat)].sort((a, b) => (b.anomalyScore ?? 0) - (a.anomalyScore ?? 0))"
                      :key="point.id"
                      class="group relative flex cursor-pointer flex-col gap-1.5 rounded-lg border bg-background p-1.5 transition-all hover:shadow-sm"
                      :class="point.isGolden ? 'border-amber-400/70 hover:border-amber-400' : 'border-border hover:border-primary/50'"
                      @click="point.url && openPreview(point)"
                    >
                      <div class="aspect-square overflow-hidden rounded-md bg-secondary/10">
                        <img :src="staticUrl(point.url)" alt="" class="h-full w-full object-cover transition-transform group-hover:scale-110" />
                      </div>
                      <div v-if="point.isGolden" class="absolute left-1 top-1 flex items-center gap-0.5 rounded-full bg-amber-400 px-1 py-0.5 text-[8px] font-bold text-black shadow">
                        <Sparkles class="h-2 w-2" />Golden
                      </div>
                      <div class="space-y-1">
                        <div class="flex justify-between text-[9px] font-medium">
                          <span class="text-muted-foreground">异常</span>
                          <span :class="(point.anomalyScore ?? 0) > 70 ? 'text-rose-500' : 'text-emerald-500'">{{ (point.anomalyScore ?? 0).toFixed(1) }}%</span>
                        </div>
                        <div class="h-1 w-full overflow-hidden rounded-full bg-secondary/30">
                          <div class="h-full transition-all" :class="(point.anomalyScore ?? 0) > 70 ? 'bg-rose-500' : 'bg-emerald-500'" :style="{width: `${point.anomalyScore ?? 0}%`}" />
                        </div>
                      </div>
                      <div v-if="(point.anomalyScore ?? 0) > 85" class="absolute right-1 top-1">
                        <div class="rounded-full bg-rose-500 p-0.5 text-white shadow"><AlertTriangle class="h-2 w-2" /></div>
                      </div>
                    </div>
                  </div>
                </template>
              </div>
            </template>
            <div v-else class="flex h-full flex-col items-center justify-center gap-4 text-muted-foreground">
              <div class="rounded-full bg-secondary/20 p-5"><LayoutGrid class="h-8 w-8 opacity-20" /></div>
              <p class="max-w-xs text-center text-sm italic">运行分析后查看按异常置信度排序的检测结果。</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>

  <!-- ════════════════ GOLDEN SELECTION MODAL ════════════════ -->
  <Teleport to="body">
    <Transition name="fade">
      <div v-if="showGoldenDialog" class="fixed inset-0 z-[95]">
        <div class="absolute inset-0 bg-black/60" @click="showGoldenDialog = false" />
        <div class="absolute left-1/2 top-1/2 flex max-h-[88vh] w-[min(94vw,56rem)] -translate-x-1/2 -translate-y-1/2 flex-col rounded-xl border border-border bg-background shadow-2xl">
          <!-- Header -->
          <div class="flex shrink-0 items-center justify-between border-b border-border px-5 py-3.5">
            <div class="flex items-center gap-2">
              <Sparkles class="h-4 w-4 text-amber-400" />
              <h3 class="text-sm font-semibold tracking-tight">选择 Golden 参考样本</h3>
              <span class="rounded bg-amber-400/15 px-1.5 py-0.5 text-[10px] font-medium text-amber-500">已选 {{ goldenDraft.size }}</span>
            </div>
            <button type="button" class="rounded p-1 text-muted-foreground transition-colors hover:bg-secondary/50 hover:text-foreground" aria-label="关闭" @click="showGoldenDialog = false">
              <X class="h-4 w-4" />
            </button>
          </div>

          <!-- Body -->
          <div class="min-h-0 flex-1 overflow-y-auto px-5 py-4">
            <p class="mb-4 text-xs text-muted-foreground">
              勾选每个缺陷类别下的"标准/合格"裁剪图作为参考。运行分析时，同类样本将按与这些 golden 的偏离程度排序异常。
            </p>
            <div v-if="!goldenCropsByClass.length" class="py-10 text-center text-sm text-muted-foreground">
              当前数据集没有可用的裁剪图。
            </div>
            <div v-for="group in goldenCropsByClass" :key="group.cls.id" class="mb-6">
              <div class="mb-2 flex items-center justify-between border-b border-border pb-1.5">
                <div class="flex items-center gap-2">
                  <div class="h-3 w-3 rounded-full" :style="{backgroundColor: group.cls.color || '#8884d8'}" />
                  <h4 class="text-sm font-semibold">{{ group.cls.name }}</h4>
                  <span class="rounded bg-secondary/20 px-1.5 py-0.5 text-[10px] text-muted-foreground">{{ group.crops.length }} 裁剪图</span>
                </div>
                <button
                  type="button"
                  class="rounded border border-border px-2 py-0.5 text-[10px] font-medium transition-colors hover:bg-secondary/40"
                  @click="toggleGoldenClass(group.crops, !classAllSelected(group.crops))"
                >
                  {{ classAllSelected(group.crops) ? '取消全选' : '一键全选' }}
                </button>
              </div>
              <div class="grid grid-cols-4 gap-2 sm:grid-cols-6 md:grid-cols-8">
                <button
                  v-for="crop in group.crops"
                  :key="crop.id"
                  type="button"
                  class="group relative aspect-square overflow-hidden rounded-md border-2 transition-all"
                  :class="goldenDraft.has(crop.id) ? 'border-amber-400 ring-1 ring-amber-400/40' : 'border-border hover:border-primary/40'"
                  @click="toggleGoldenCrop(crop.id)"
                >
                  <img :src="staticUrl(crop.url)" alt="" class="h-full w-full object-cover" />
                  <div v-if="goldenDraft.has(crop.id)" class="absolute right-0.5 top-0.5 rounded-full bg-amber-400 p-0.5 text-black shadow">
                    <CheckCircle2 class="h-3 w-3" />
                  </div>
                </button>
              </div>
            </div>
          </div>

          <!-- Footer -->
          <div class="flex shrink-0 items-center justify-between gap-2 border-t border-border px-5 py-3">
            <button type="button" class="text-xs text-muted-foreground hover:text-rose-500" @click="goldenDraft = new Set()">全部清空</button>
            <div class="flex items-center gap-2">
              <button type="button" class="rounded-md border border-border px-3 py-1.5 text-xs font-medium transition-colors hover:bg-secondary/40" @click="showGoldenDialog = false">取消</button>
              <button type="button" class="rounded-md bg-primary px-3 py-1.5 text-xs font-semibold text-primary-foreground transition-colors hover:bg-primary/90" @click="applyGoldenSelection">
                应用（{{ goldenDraft.size }}）
              </button>
            </div>
          </div>
        </div>
      </div>
    </Transition>
  </Teleport>

  <!-- ════════════════ IMAGE PREVIEW MODAL ════════════════ -->
  <Teleport to="body">
    <Transition name="fade">
      <div v-if="previewPoint && previewDisplayUrl" class="fixed inset-0 z-[200] flex flex-col bg-black/88 backdrop-blur-sm" @click.self="closePreview">
        <!-- Header -->
        <div class="flex shrink-0 flex-wrap items-center justify-between gap-3 border-b border-white/10 px-4 py-3 text-white" @click.stop>
          <span class="text-xs text-white/70">
            滚轮缩放 · {{ previewCanPan ? '拖动平移' : '已适配窗口' }} · Esc 关闭 · 空格切换标注
            <span class="text-white/40"> · {{ previewModalView === 'source' ? '原图' : '裁剪图' }}</span>
          </span>
          <div class="flex flex-wrap items-center justify-end gap-2">
            <button v-if="previewModalView === 'crop' && canTraceSource" type="button" class="flex items-center gap-1.5 rounded-lg bg-white/10 px-3 py-1.5 text-xs font-medium text-white/90 backdrop-blur-sm transition-colors hover:bg-white/20" @click="previewModalView = 'source'">
              <ExternalLink class="h-3.5 w-3.5" />查看原图
            </button>
            <button v-if="previewModalView === 'source'" type="button" class="flex items-center gap-1.5 rounded-lg bg-white/10 px-3 py-1.5 text-xs font-medium text-white/90 backdrop-blur-sm transition-colors hover:bg-white/20" @click="previewModalView = 'crop'">
              <ArrowLeft class="h-3.5 w-3.5" />裁剪图
            </button>
            <div v-if="previewAnnotation?.labels?.length" class="flex items-center gap-1 rounded-lg bg-white/10 px-2 py-1 text-[11px] text-white/80">
              <button type="button" class="rounded px-1.5 py-0.5 font-medium transition-colors" :class="showPreviewAnnotations ? 'bg-indigo-500/50 text-indigo-100' : 'text-white/40 hover:text-white'" @click="showPreviewAnnotations = !showPreviewAnnotations">
                {{ showPreviewAnnotations ? '隐藏' : '显示' }}标注
              </button>
            </div>
            <span class="min-w-[3rem] text-center font-mono text-xs tabular-nums text-white/80">{{ Math.round(previewEffectiveScale * 100) }}%</span>
            <button type="button" class="rounded-md border border-white/20 p-1.5 hover:bg-white/10" aria-label="Zoom out" @click="imagePreviewZoom = Math.max(PREVIEW_ZOOM_MIN, Math.round((imagePreviewZoom - 0.15) * 100) / 100)">
              <Minus class="h-4 w-4" />
            </button>
            <button type="button" class="rounded-md border border-white/20 p-1.5 hover:bg-white/10" aria-label="Zoom in" @click="imagePreviewZoom = Math.min(PREVIEW_ZOOM_MAX, Math.round((imagePreviewZoom + 0.15) * 100) / 100)">
              <Plus class="h-4 w-4" />
            </button>
            <button type="button" class="rounded-md border border-white/20 p-1.5 hover:bg-white/10" aria-label="Close" @click="closePreview">
              <X class="h-4 w-4" />
            </button>
          </div>
        </div>

        <!-- Drag-to-pan viewport: overflow:hidden, zoom+pan via CSS transform -->
        <div
          ref="imagePreviewWheelRef"
          class="flex min-h-0 flex-1 overflow-hidden"
          :class="
            !previewCanPan
              ? 'cursor-default'
              : previewDragging
                ? 'cursor-grabbing select-none'
                : 'cursor-grab'
          "
          @click.stop
          @pointerdown="onPreviewPointerDown"
          @pointermove="onPreviewPointerMove"
          @pointerup="onPreviewPointerUp"
          @pointercancel="onPreviewPointerUp"
        >
          <div class="flex min-h-full w-full items-center justify-center">
            <!-- Single transform node: scale(fitScale * userZoom) + translate(pan) -->
            <div
              class="relative inline-block select-none leading-none"
              :style="{
                transform: `translate(${imagePreviewPan.x}px, ${imagePreviewPan.y}px) scale(${previewEffectiveScale})`,
                transformOrigin: 'center center',
              }"
            >
              <img
                ref="imagePreviewImgRef"
                :src="previewDisplayUrl"
                alt=""
                draggable="false"
                class="block select-none"
                style="display:block; max-width:none; max-height:none;"
                @load="computeFitScale"
              />
              <svg
                v-if="previewAnnotation?.labels?.length && showPreviewAnnotations"
                class="pointer-events-none absolute inset-0 h-full w-full"
                viewBox="0 0 1 1"
                preserveAspectRatio="none"
              >
                <polygon
                  v-for="(label, idx) in previewAnnotation.labels"
                  :key="idx"
                  :points="label.points.map(([x, y]: [number, number]) => `${x},${y}`).join(' ')"
                  fill="rgba(99, 102, 241, 0.42)"
                  stroke="rgb(199, 210, 254)"
                  :stroke-width="0.004 / previewEffectiveScale"
                  stroke-linejoin="round"
                />
              </svg>
            </div>
          </div>
        </div>
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

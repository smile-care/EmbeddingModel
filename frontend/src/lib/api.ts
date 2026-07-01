/**
 * Typed API client for the Data Cluster backend.
 *
 * API base URL (no trailing slash). Empty => same-origin (Vite proxies /api and
 * /static to the FastAPI backend on :8000). Override with VITE_API_BASE_URL.
 */
export function apiBase(): string {
  const raw = import.meta.env.VITE_API_BASE_URL as string | undefined;
  if (!raw?.trim()) return '';
  return raw.replace(/\/$/, '');
}

/** path must start with `/`, e.g. `/api/datasets` */
export function apiUrl(path: string): string {
  const base = apiBase();
  const p = path.startsWith('/') ? path : `/${path}`;
  return base ? `${base}${p}` : p;
}

/** Resolve a `/static/...` URL through the configured API base. */
export function staticUrl(url: string | null | undefined): string {
  if (!url) return '';
  if (/^(https?:|blob:|data:)/.test(url)) return url;
  return apiUrl(url);
}

export class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
    this.name = 'ApiError';
  }
}

async function parseError(res: Response): Promise<never> {
  let detail = `${res.status} ${res.statusText}`;
  try {
    const data = await res.json();
    if (data?.detail) detail = typeof data.detail === 'string' ? data.detail : JSON.stringify(data.detail);
  } catch {
    /* ignore non-JSON bodies */
  }
  throw new ApiError(res.status, detail);
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(apiUrl(path), init);
  } catch (e) {
    throw new ApiError(0, `网络错误：无法访问后端 (${path})。请确认后端已启动。`);
  }
  if (!res.ok) await parseError(res);
  if (res.status === 204) return undefined as T;
  const text = await res.text();
  return (text ? JSON.parse(text) : undefined) as T;
}

function jsonInit(method: string, body: unknown): RequestInit {
  return {method, headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)};
}

// ── Types ──────────────────────────────────────────────────────────────────

export interface DefectClass {
  id: string;
  name: string;
  color?: string | null;
  sortOrder: number;
}

export interface AnnotationRegion {
  id: string;
  classId: string | null;
  /** normalized 0-1 polygon points in original-image space: [[x, y], ...] */
  points: number[][];
  isSubtract: boolean;
  order: number;
}

export interface CropAnnotationShape {
  points: number[][];
  isSubtract: boolean;
}

export interface CropImage {
  id: string;
  url: string;
  classId: string | null;
  maskUrl?: string | null;
  sourceImageId: string;
  regionId?: string | null;
  instanceIndex: number;
  bbox?: number[] | null;
  cropBbox?: number[] | null;
  originalSize?: number[] | null;
  patchSize?: number[] | null;
  cropAnnotation?: CropAnnotationShape[] | null;
}

export type AnnotationStatus = 'unannotated' | 'annotated';

export interface DatasetImage {
  id: string;
  url: string;
  width?: number | null;
  height?: number | null;
  source: string;
  annotationStatus: AnnotationStatus;
  regions: AnnotationRegion[];
  crops: CropImage[];
}

export interface DatasetSummary {
  id: string;
  name: string;
  type: string;
  size?: string | null;
  items: number;
  status: string;
  createdAt: string;
  updatedAt: string;
  defectClasses: DefectClass[];
}

export interface DatasetDetail extends DatasetSummary {
  images: DatasetImage[];
}

export interface RegionInput {
  classId: string | null;
  points: number[][];
  isSubtract: boolean;
}

// ── Datasets ─────────────────────────────────────────────────────────────────

export const DatasetsApi = {
  list: () => request<DatasetSummary[]>('/api/datasets'),

  get: (id: string) => request<DatasetDetail>(`/api/datasets/${id}`),

  create: (form: FormData) =>
    request<{id: string; name: string; items: number}>('/api/datasets', {method: 'POST', body: form}),

  remove: (id: string) => request<{success: boolean}>(`/api/datasets/${id}`, {method: 'DELETE'}),

  // ── Defect classes ──
  listClasses: (datasetId: string) => request<DefectClass[]>(`/api/datasets/${datasetId}/classes`),

  createClass: (datasetId: string, body: {name: string; color?: string | null}) =>
    request<DefectClass>(`/api/datasets/${datasetId}/classes`, jsonInit('POST', body)),

  updateClass: (classId: string, body: {name?: string; color?: string | null}) =>
    request<DefectClass>(`/api/datasets/classes/${classId}`, jsonInit('PATCH', body)),

  deleteClass: (classId: string) =>
    request<{success: boolean}>(`/api/datasets/classes/${classId}`, {method: 'DELETE'}),

  // ── Images ──
  uploadImages: (datasetId: string, files: File[], source = 'batch') => {
    const form = new FormData();
    for (const f of files) form.append('files', f);
    form.append('source', source);
    return request<DatasetDetail>(`/api/datasets/${datasetId}/images`, {method: 'POST', body: form});
  },

  uploadSingleImage: (datasetId: string, file: File) => {
    const form = new FormData();
    form.append('file', file);
    return request<{image: DatasetImage}>(`/api/datasets/${datasetId}/images/single`, {
      method: 'POST',
      body: form,
    });
  },

  deleteImage: (imageId: string) =>
    request<{success: boolean}>(`/api/datasets/images/${imageId}`, {method: 'DELETE'}),

  // ── Annotation ──
  getAnnotation: (imageId: string) =>
    request<AnnotationRegion[]>(`/api/datasets/images/${imageId}/annotation`),

  saveAnnotation: (imageId: string, regions: RegionInput[], generateCrops = true) =>
    request<DatasetImage>(
      `/api/datasets/images/${imageId}/annotation`,
      jsonInit('PUT', {regions, generateCrops}),
    ),

  // ── Crops ──
  getImageCrops: (imageId: string) =>
    request<CropImage[]>(`/api/datasets/images/${imageId}/crops`),

  generateCrops: (imageId: string, minSize = 8) =>
    request<CropImage[]>(`/api/datasets/images/${imageId}/crops?min_size=${minSize}`, {method: 'POST'}),

  getClassCrops: (classId: string) =>
    request<CropImage[]>(`/api/datasets/classes/${classId}/crops`),
};

// ── Experiments ──────────────────────────────────────────────────────────────

export interface ExperimentSummary {
  id: string;
  name: string;
  model: string;
  dataset: string;
  datasetId?: string | null;
  status: string;
  duration?: string | null;
  accuracy?: string | null;
  progress: number;
  runStatus?: string | null;
  runProgress?: number;
  createdAt: string;
}

export const ExperimentsApi = {
  list: () => request<ExperimentSummary[]>('/api/experiments'),
  get: (id: string) => request<any>(`/api/experiments/${id}`),
  create: (body: unknown) => request<any>('/api/experiments', jsonInit('POST', body)),
  train: (id: string) => request<any>(`/api/experiments/${id}/train`, {method: 'POST'}),
  stop: (id: string) => request<any>(`/api/experiments/${id}/stop`, {method: 'POST'}),
  remove: (id: string) => request<{success: boolean}>(`/api/experiments/${id}`, {method: 'DELETE'}),
};

// ── Inference ────────────────────────────────────────────────────────────────

export interface ModelInfo {
  id: string;
  name: string;
  type: string;
}

export const InferenceApi = {
  listModels: () => request<ModelInfo[]>('/api/models'),
  listRuns: () => request<any[]>('/api/inference/runs'),
  getRun: (id: string) => request<any>(`/api/inference/runs/${id}`),
  createRun: (body: unknown) => request<any>('/api/inference/runs', jsonInit('POST', body)),
  patchRun: (id: string, body: unknown) => request<any>(`/api/inference/runs/${id}`, jsonInit('PATCH', body)),
  deleteRun: (id: string) => request<{success: boolean}>(`/api/inference/runs/${id}`, {method: 'DELETE'}),
  analyze: (id: string, body?: unknown) =>
    request<any>(`/api/inference/runs/${id}/analyze`, body ? jsonInit('POST', body) : {method: 'POST'}),
  getProjection: (id: string, algo: string) =>
    request<any>(`/api/inference/runs/${id}/projections/${algo}`),
  computeProjection: (id: string, algo: string) =>
    request<any>(`/api/inference/runs/${id}/projections/${algo}`, {method: 'POST'}),
  listUploadCategories: () => request<any[]>('/api/inference/upload-categories'),
  createUploadCategory: (body: unknown) =>
    request<any>('/api/inference/upload-categories', jsonInit('POST', body)),
  patchUploadCategory: (id: string, body: unknown) =>
    request<any>(`/api/inference/upload-categories/${id}`, jsonInit('PATCH', body)),
  deleteUploadCategory: (id: string) =>
    request<{success: boolean}>(`/api/inference/upload-categories/${id}`, {method: 'DELETE'}),
};

/** Stable color for a defect class (used when class has no explicit color). */
const PALETTE = [
  '#ef4444', '#f97316', '#eab308', '#22c55e', '#14b8a6',
  '#3b82f6', '#8b5cf6', '#ec4899', '#f43f5e', '#10b981',
];

/**
 * Per-category color for inference scatter / legend.
 * Hues are evenly spaced around the wheel for the active class count so adjacent
 * entries look as different as possible (no repeated blues/greens from a fixed list).
 */
export function categoryChartColor(index: number, total?: number): string {
  const i = Math.max(0, index);
  const count = Math.max(total ?? 1, i + 1);
  const hue = Math.round((i * 360) / count) % 360;
  const saturation = 88;
  const lightness = 55 + (i % 2) * 8;
  return `hsl(${hue}, ${saturation}%, ${lightness}%)`;
}

export function categoryChartColors(count: number): string[] {
  const n = Math.max(0, count);
  return Array.from({length: n}, (_, i) => categoryChartColor(i, n));
}

export function classColor(cls: Pick<DefectClass, 'color' | 'sortOrder'> | null | undefined, fallbackIndex = 0): string {
  if (cls?.color) return cls.color;
  const idx = cls?.sortOrder ?? fallbackIndex;
  return PALETTE[((idx % PALETTE.length) + PALETTE.length) % PALETTE.length];
}

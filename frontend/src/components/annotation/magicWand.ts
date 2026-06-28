/** A simple magic-wand selection: flood-fill a connected region of similar
 *  colour from a seed pixel, trace its outer boundary, and return a simplified
 *  polygon (points normalized to 0-1 in image space).
 *
 *  Kept intentionally lightweight — single outer contour, no holes — which is
 *  plenty for assisting polygon annotation.
 */
import type {Point} from './geometry';

/** Max working resolution; larger images are sampled down for a snappy fill. */
const MAX_SIDE = 1200;

export interface WandSource {
  /** RGBA pixel buffer of the (possibly downsampled) image. */
  data: Uint8ClampedArray;
  width: number;
  height: number;
}

/** Render an image element into an offscreen canvas and grab its pixels.
 *  Returns ``null`` if the canvas is tainted (cross-origin) or has no size. */
export function buildWandSource(img: HTMLImageElement): WandSource | null {
  const w0 = img.naturalWidth;
  const h0 = img.naturalHeight;
  if (!w0 || !h0) return null;
  const s = Math.min(1, MAX_SIDE / Math.max(w0, h0));
  const width = Math.max(1, Math.round(w0 * s));
  const height = Math.max(1, Math.round(h0 * s));
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d', {willReadFrequently: true});
  if (!ctx) return null;
  try {
    ctx.drawImage(img, 0, 0, width, height);
    const {data} = ctx.getImageData(0, 0, width, height);
    return {data, width, height};
  } catch {
    return null; // tainted canvas
  }
}

/** 8-connected Moore-neighbour offsets, clockwise: E, SE, S, SW, W, NW, N, NE. */
const MOORE: ReadonlyArray<[number, number]> = [
  [1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1], [0, -1], [1, -1],
];

function dirIndex(from: [number, number], to: [number, number]): number {
  const dx = to[0] - from[0];
  const dy = to[1] - from[1];
  for (let i = 0; i < 8; i++) if (MOORE[i][0] === dx && MOORE[i][1] === dy) return i;
  return 4; // fallback: west
}

/** Moore-neighbour boundary tracing (clockwise) of a binary mask. */
function traceContour(mask: Uint8Array, w: number, h: number): [number, number][] {
  const at = (x: number, y: number) => x >= 0 && y >= 0 && x < w && y < h && mask[y * w + x] === 1;

  let start: [number, number] | null = null;
  outer: for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      if (mask[y * w + x] === 1) { start = [x, y]; break outer; }
    }
  }
  if (!start) return [];

  const contour: [number, number][] = [start];
  let b = start;
  let backtrack: [number, number] = [start[0] - 1, start[1]]; // background to the left
  const maxSteps = 4 * (w + h) + 16;

  for (let step = 0; step < maxSteps; step++) {
    const dir = dirIndex(b, backtrack);
    let moved = false;
    for (let k = 1; k <= 8; k++) {
      const nd = (dir + k) % 8;
      const nx = b[0] + MOORE[nd][0];
      const ny = b[1] + MOORE[nd][1];
      if (at(nx, ny)) {
        const pd = (dir + k - 1) % 8; // last background cell before the hit
        backtrack = [b[0] + MOORE[pd][0], b[1] + MOORE[pd][1]];
        b = [nx, ny];
        moved = true;
        break;
      }
    }
    if (!moved) break; // isolated pixel
    if (b[0] === start[0] && b[1] === start[1]) break; // looped back
    contour.push(b);
  }
  return contour;
}

/** Perpendicular distance from p to segment a-b. */
function perpDist(p: [number, number], a: [number, number], b: [number, number]): number {
  const dx = b[0] - a[0];
  const dy = b[1] - a[1];
  const len = Math.hypot(dx, dy);
  if (len < 1e-9) return Math.hypot(p[0] - a[0], p[1] - a[1]);
  return Math.abs((p[0] - a[0]) * dy - (p[1] - a[1]) * dx) / len;
}

/** Iterative Ramer–Douglas–Peucker polyline simplification. */
function rdp(points: [number, number][], eps: number): [number, number][] {
  if (points.length < 3) return points;
  const keep = new Uint8Array(points.length);
  keep[0] = 1;
  keep[points.length - 1] = 1;
  const stack: [number, number][] = [[0, points.length - 1]];
  while (stack.length) {
    const [lo, hi] = stack.pop()!;
    let maxD = 0;
    let idx = -1;
    for (let i = lo + 1; i < hi; i++) {
      const d = perpDist(points[i], points[lo], points[hi]);
      if (d > maxD) { maxD = d; idx = i; }
    }
    if (idx !== -1 && maxD > eps) {
      keep[idx] = 1;
      stack.push([lo, idx], [idx, hi]);
    }
  }
  return points.filter((_, i) => keep[i] === 1);
}

/**
 * Run the magic wand from a normalized seed point. ``tolerance`` is the maximum
 * per-channel colour difference (0-255) for a pixel to be included. Returns the
 * normalized polygon, or ``null`` if nothing meaningful was selected.
 */
export function magicWandPolygon(
  src: WandSource,
  seed: Point,
  tolerance: number,
): Point[] | null {
  const {data, width: w, height: h} = src;
  const sx = Math.min(w - 1, Math.max(0, Math.round(seed[0] * (w - 1))));
  const sy = Math.min(h - 1, Math.max(0, Math.round(seed[1] * (h - 1))));
  const seedIdx = (sy * w + sx) * 4;
  const sr = data[seedIdx];
  const sg = data[seedIdx + 1];
  const sb = data[seedIdx + 2];

  const matches = (p: number) => {
    const i = p * 4;
    return (
      Math.abs(data[i] - sr) <= tolerance &&
      Math.abs(data[i + 1] - sg) <= tolerance &&
      Math.abs(data[i + 2] - sb) <= tolerance
    );
  };

  const mask = new Uint8Array(w * h);
  const visited = new Uint8Array(w * h);
  const stack: number[] = [sy * w + sx];
  visited[sy * w + sx] = 1;
  let area = 0;
  while (stack.length) {
    const p = stack.pop()!;
    mask[p] = 1;
    area++;
    const x = p % w;
    const y = (p / w) | 0;
    // 4-connected flood (keeps the region compact; contour trace is 8-connected)
    if (x > 0) { const n = p - 1; if (!visited[n] && matches(n)) { visited[n] = 1; stack.push(n); } }
    if (x < w - 1) { const n = p + 1; if (!visited[n] && matches(n)) { visited[n] = 1; stack.push(n); } }
    if (y > 0) { const n = p - w; if (!visited[n] && matches(n)) { visited[n] = 1; stack.push(n); } }
    if (y < h - 1) { const n = p + w; if (!visited[n] && matches(n)) { visited[n] = 1; stack.push(n); } }
  }
  if (area < 4) return null;

  const contour = traceContour(mask, w, h);
  if (contour.length < 3) return null;

  const eps = Math.max(1.5, Math.hypot(w, h) * 0.004);
  const simplified = rdp(contour, eps);
  if (simplified.length < 3) return null;

  // pixel centres -> normalized 0-1
  return simplified.map(([px, py]) => [(px + 0.5) / w, (py + 0.5) / h] as Point);
}

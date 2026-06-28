/** Geometry / coordinate helpers for polygon annotation.
 *
 * Points are stored normalized to 0-1 in the original-image space. The SVG
 * overlay uses a 0-100 viewBox, so display = normalized * 100.
 */
export type Point = [number, number];

export const SVG_SCALE = 100;

export function clamp01(v: number): number {
  return Math.min(1, Math.max(0, v));
}

/** Convert a client (mouse) position to normalized 0-1 coords within an element. */
export function clientToNorm(el: HTMLElement, clientX: number, clientY: number): Point {
  const rect = el.getBoundingClientRect();
  if (rect.width <= 0 || rect.height <= 0) return [0, 0];
  return [clamp01((clientX - rect.left) / rect.width), clamp01((clientY - rect.top) / rect.height)];
}

/** Normalized points -> SVG polygon "points" attribute string (0-100 space). */
export function toSvgPoints(points: Point[]): string {
  return points.map(([x, y]) => `${x * SVG_SCALE},${y * SVG_SCALE}`).join(' ');
}

export function dist(a: Point, b: Point): number {
  return Math.hypot(a[0] - b[0], a[1] - b[1]);
}

export function clonePoints(points: Point[]): Point[] {
  return points.map(([x, y]) => [x, y] as Point);
}

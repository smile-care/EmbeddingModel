export interface AnnotationOverlayRegion {
  points: number[][];
  isSubtract?: boolean;
  classId?: string | null;
  /** Explicit label; falls back to labelForClass(classId). */
  label?: string | null;
}

/** Anchor for a region label chip (normalized 0–1 image space). */
export function polygonLabelAnchor(points: number[][]): {x: number; y: number} {
  if (!points.length) return {x: 0.5, y: 0.5};
  const xs = points.map((p) => p[0]);
  const ys = points.map((p) => p[1]);
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);
  return {x: (minX + maxX) / 2, y: (minY + maxY) / 2};
}

/** Polygon centroid in normalized 0–1 image space. */
export function polygonCentroid(points: number[][]): {x: number; y: number} {
  if (points.length < 3) return polygonLabelAnchor(points);
  let area = 0;
  let cx = 0;
  let cy = 0;
  const n = points.length;
  for (let i = 0; i < n; i++) {
    const [x0, y0] = points[i];
    const [x1, y1] = points[(i + 1) % n];
    const cross = x0 * y1 - x1 * y0;
    area += cross;
    cx += (x0 + x1) * cross;
    cy += (y0 + y1) * cross;
  }
  area *= 0.5;
  if (Math.abs(area) < 1e-12) return polygonLabelAnchor(points);
  return {x: cx / (6 * area), y: cy / (6 * area)};
}

/** Shoelace area in normalized 0–1 space (= fraction of image area). */
export function polygonAreaNormalized(points: number[][]): number {
  if (points.length < 3) return 0;
  let sum = 0;
  const n = points.length;
  for (let i = 0; i < n; i++) {
    const [x0, y0] = points[i];
    const [x1, y1] = points[(i + 1) % n];
    sum += x0 * y1 - x1 * y0;
  }
  return Math.abs(sum) / 2;
}

export interface ImagePixelSize {
  width: number;
  height: number;
}

export function formatAreaPercent(area: number): string {
  return `${(area * 100).toFixed(2)}%`;
}

export function formatAreaPixels(areaNorm: number, size: ImagePixelSize): string {
  const px = Math.round(areaNorm * size.width * size.height);
  return `${px}px²`;
}

export function formatNormalizedPoint(p: {x: number; y: number}): string {
  return `(${(p.x * 100).toFixed(1)}%, ${(p.y * 100).toFixed(1)}%)`;
}

export function formatCenterPixels(p: {x: number; y: number}, size: ImagePixelSize): string {
  return `${Math.round(p.x * size.width)},${Math.round(p.y * size.height)}`;
}

/** Compact center percent fallback when image size is unknown. */
export function formatCompactCenterPercent(p: {x: number; y: number}): string {
  return `${(p.x * 100).toFixed(1)},${(p.y * 100).toFixed(1)}%`;
}

export function buildStatsLine(
  index: number,
  areaNorm: number,
  centroid: {x: number; y: number},
  imageSize?: ImagePixelSize | null,
): string {
  if (imageSize && imageSize.width > 0 && imageSize.height > 0) {
    return `#${index} Area ${formatAreaPixels(areaNorm, imageSize)} Center ${formatCenterPixels(centroid, imageSize)}`;
  }
  return `#${index} Area ${formatAreaPercent(areaNorm)} Center ${formatCompactCenterPercent(centroid)}`;
}

export interface AnnotationLegendEntry {
  index: number;
  area: number;
  areaText: string;
  centerText: string;
  statsLine: string;
}

export interface AnnotationLegendGroup {
  label: string;
  color: string;
  isSubtract: boolean;
  items: AnnotationLegendEntry[];
}

export function buildLegendGroups(
  regions: AnnotationOverlayRegion[],
  colorForClass: (classId: string | null | undefined, isSubtract?: boolean) => string,
  labelForClass?: (classId: string | null | undefined) => string,
  imageSize?: ImagePixelSize | null,
): AnnotationLegendGroup[] {
  const map = new Map<string, AnnotationLegendGroup>();

  regions.forEach((region, index) => {
    const label = regionDisplayLabel(region, labelForClass);
    if (!label) return;
    const color = colorForClass(region.classId, !!region.isSubtract);
    const key = `${label}\0${color}\0${region.isSubtract ? 1 : 0}`;
    if (!map.has(key)) {
      map.set(key, {label, color, isSubtract: !!region.isSubtract, items: []});
    }
    const area = polygonAreaNormalized(region.points);
    const centroid = polygonCentroid(region.points);
    const areaText =
      imageSize && imageSize.width > 0 && imageSize.height > 0
        ? formatAreaPixels(area, imageSize)
        : formatAreaPercent(area);
    const centerText =
      imageSize && imageSize.width > 0 && imageSize.height > 0
        ? formatCenterPixels(centroid, imageSize)
        : formatNormalizedPoint(centroid);
    map.get(key)!.items.push({
      index: index + 1,
      area,
      areaText,
      centerText,
      statsLine: buildStatsLine(index + 1, area, centroid, imageSize),
    });
  });

  return Array.from(map.values()).sort((a, b) => {
    if (a.isSubtract !== b.isSubtract) return a.isSubtract ? 1 : -1;
    return a.label.localeCompare(b.label, 'zh-CN');
  });
}

export function regionDisplayLabel(
  region: AnnotationOverlayRegion,
  labelForClass?: (classId: string | null | undefined) => string,
): string | null {
  if (region.isSubtract) return 'Hole';
  if (region.label?.trim()) return region.label.trim();
  if (labelForClass) return labelForClass(region.classId);
  if (region.classId) return null;
  return 'Unassigned';
}

export function polygonPointsAttr(points: number[][]): string {
  return points.map(([x, y]) => `${x},${y}`).join(' ');
}

/** Apply alpha to hex, rgb, or hsl colors for SVG polygon fills. */
export function colorWithAlpha(color: string, alpha: number): string {
  const a = Math.min(1, Math.max(0, alpha));
  const c = color.trim();

  const hex3 = /^#([0-9a-f]{3})$/i.exec(c);
  if (hex3) {
    const [r, g, b] = hex3[1].split('').map((ch) => parseInt(ch + ch, 16));
    return `rgba(${r},${g},${b},${a})`;
  }

  const hex6 = /^#([0-9a-f]{6})$/i.exec(c);
  if (hex6) {
    const n = hex6[1];
    return `rgba(${parseInt(n.slice(0, 2), 16)},${parseInt(n.slice(2, 4), 16)},${parseInt(n.slice(4, 6), 16)},${a})`;
  }

  const hex8 = /^#([0-9a-f]{8})$/i.exec(c);
  if (hex8) {
    const n = hex8[1];
    return `rgba(${parseInt(n.slice(0, 2), 16)},${parseInt(n.slice(2, 4), 16)},${parseInt(n.slice(4, 6), 16)},${a})`;
  }

  const hsla = /^hsla\(\s*([\d.]+)\s*,\s*([\d.]+)%\s*,\s*([\d.]+)%\s*,\s*[\d.]+\s*\)$/i.exec(c);
  if (hsla) return `hsla(${hsla[1]}, ${hsla[2]}%, ${hsla[3]}%, ${a})`;

  const hsl = /^hsl\(\s*([\d.]+)\s*,\s*([\d.]+)%\s*,\s*([\d.]+)%\s*\)$/i.exec(c);
  if (hsl) return `hsla(${hsl[1]}, ${hsl[2]}%, ${hsl[3]}%, ${a})`;

  const rgba = /^rgba\(\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*[\d.]+\s*\)$/i.exec(c);
  if (rgba) return `rgba(${rgba[1]},${rgba[2]},${rgba[3]},${a})`;

  const rgb = /^rgb\(\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*\)$/i.exec(c);
  if (rgb) return `rgba(${rgb[1]},${rgb[2]},${rgb[3]},${a})`;

  return c;
}

/** Counter-scale legend when the overlay sits inside a CSS-scaled image container. */
export function legendCompensateFromContainerScale(containerScale: number): number {
  if (!Number.isFinite(containerScale) || containerScale <= 0) return 1;
  return Math.min(2.5, Math.max(1, 1 / containerScale));
}

/** Shrink legend on small crops; slight boost only on large previews (reference ~500px). */
export function legendBoostFromDisplayWidth(displayWidthPx: number, referencePx = 500): number {
  if (!Number.isFinite(displayWidthPx) || displayWidthPx <= 0) return 1;
  if (displayWidthPx < referencePx) {
    return Math.max(0.45, displayWidthPx / referencePx);
  }
  return Math.min(1.15, 1 + ((displayWidthPx - referencePx) / referencePx) * 0.12);
}

export function combineLegendScale(containerScale: number, displayWidthPx = 0): number {
  const compensate = legendCompensateFromContainerScale(containerScale);
  const boost = legendBoostFromDisplayWidth(displayWidthPx);
  return Math.min(2.5, Math.max(0.45, compensate * boost));
}

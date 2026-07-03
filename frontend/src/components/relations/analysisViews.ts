/** Analysis center view keys and display labels (single source of truth). */
export type ViewKey =
  | 'distribution'
  | 'anomaly'
  | 'headline'
  | 'confusion'
  | 'centroid'
  | 'dendrogram'
  | 'graph'
  | 'warnings'
  | 'perclass'
  | 'mislabels';

export const ANALYSIS_VIEW_LABELS: Record<ViewKey, string> = {
  distribution: '分布散点',
  anomaly: '异常网格',
  headline: '总览质量分',
  confusion: '混淆矩阵热力图',
  centroid: '类心相似度热力图',
  dendrogram: '层次聚类树状图',
  graph: '类别关系图',
  warnings: '警告对列表',
  perclass: '每类质量卡片',
  mislabels: '疑似误标清单',
};

export const ANALYSIS_VIEW_GROUP_DEFS: {title: string; keys: ViewKey[]}[] = [
  {title: '基础视图', keys: ['distribution', 'anomaly']},
  {
    title: '关系分析',
    keys: ['headline', 'confusion', 'centroid', 'dendrogram', 'graph', 'warnings', 'perclass', 'mislabels'],
  },
];

export function analysisViewLabel(key: ViewKey): string {
  return ANALYSIS_VIEW_LABELS[key];
}

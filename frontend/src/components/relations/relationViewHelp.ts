import {type ViewKey} from '@/components/relations/analysisViews';

export interface RelationHelpSection {
  title: string;
  body: string;
}

export interface RelationViewHelp {
  intro?: string;
  sections: RelationHelpSection[];
  footer?: string;
}

export interface RelationParamHelp {
  label: string;
  body: string;
}

/** Sidebar「参数」区块与各滑块的悬停说明（分析中心抽屉）。 */
export const RELATION_DRAWER_PARAM_HELP = {
  section: {
    intro: '以下参数控制关系分析与警告视图的敏感度；除邻居数 k 外，其余阈值改动后即时生效，无需重算。',
    items: [
      {title: '邻居数 k', body: 'kNN 近邻数量，影响混淆矩阵、误标检测等所有基于邻域的指标。'},
      {title: '混淆警告阈值', body: 'kNN 跨类占比告警线，用于混淆矩阵与警告列表。'},
      {title: '相似度警告阈值', body: '类心余弦相似度告警线，用于类心矩阵、树状图与关系图。'},
      {title: '误标最小跨类占比', body: '误标候选清单的最低跨类邻居占比过滤线。'},
    ],
  },
  k: {
    label: '邻居数 k 说明',
    body: '每个样本检索 k 个最近邻，用于计算 kNN 混淆矩阵、跨类邻居占比与误标投票。k 越大邻域越广、估计更平滑；k 越小对局部结构更敏感。改动后需重算关系分析。',
  },
  confusionThreshold: {
    label: '混淆警告阈值说明',
    body: 'kNN 混淆矩阵中，非对角格子（跨类邻居占比）超过该阈值时标红；「相似/混淆警告」列表也据此过滤。阈值越低越容易触发告警。',
  },
  simThreshold: {
    label: '相似度警告阈值说明',
    body: '两类类心余弦相似度超过该阈值时在类心矩阵中标红；树状图红色分支、关系图高相似边，以及警告列表的类心相似项，也受此阈值影响。',
  },
  mislabelThreshold: {
    label: '误标最小跨类占比说明',
    body: '「疑似误标」视图仅展示 k 近邻中不属于当前标注类别的占比不低于该值的样本，用于过滤噪声、聚焦高置信误标候选。',
  },
} satisfies Record<string, RelationParamHelp | {intro: string; items: {title: string; body: string}[]}>;

export const RELATION_VIEW_HELP: Partial<Record<ViewKey, RelationViewHelp>> = {
  headline: {
    intro: '数据集整体 embedding 质量概览：',
    sections: [
      {
        title: '全局 kNN 纯度',
        body: '每个样本 k 近邻中同类的加权平均占比（按类别人数加权）。越高表示类别在特征空间中越可分；低于 65% 需警惕混淆。',
      },
      {
        title: '平均 Silhouette',
        body: '所有样本轮廓系数的均值：衡量类内紧致与类间分离。越接近 1 越好；接近 0 边界模糊；为负表示整体聚类质量较差。',
      },
      {
        title: '类别数 / 样本数',
        body: '本次分析纳入的缺陷类别数量与 crop 总数。',
      },
    ],
  },
  confusion: {
    intro: 'kNN 跨类混淆矩阵：',
    sections: [
      {
        title: '行列含义',
        body: '行 = 真实类别；列 = 近邻落入的类别。格子数值 = 该行样本 k 近邻中落入该列类别的占比（%）。',
      },
      {
        title: '对角线',
        body: '同类近邻占比，即该类 kNN 纯度。颜色越绿、数值越高越好。',
      },
      {
        title: '非对角线 / 红色高亮',
        body: '跨类混淆比例。超过「混淆警告阈值」（分析中心可调）的格子会标红，提示两类易混。',
      },
      {
        title: '邻居数 k',
        body: '在分析中心调整 k 会重算矩阵；k 越大邻域越广，混淆估计更平滑。',
      },
    ],
    footer: '点击格子可在分布散点中高亮相关两类。',
  },
  centroid: {
    intro: '类心余弦相似度矩阵：',
    sections: [
      {
        title: '类心',
        body: '每个类别所有 embedding 的 L2 归一化均值向量。',
      },
      {
        title: '矩阵数值',
        body: '两类类心之间的余弦相似度，范围 0~1。对角线恒为 1.00；非对角越高表示两类中心越接近、越易混淆。',
      },
      {
        title: '红色高亮',
        body: '超过「相似度警告阈值」（分析中心可调）的格子会标红。',
      },
    ],
    footer: '点击格子可在分布散点中高亮相关两类。',
  },
  dendrogram: {
    intro: '类别层次聚类树状图：',
    sections: [
      {
        title: '合并高度',
        body: '两簇合并时的高度 = 1 − 类心余弦相似度。高度越低，两类越难区分（越相似）。',
      },
      {
        title: '红色分支',
        body: '合并高度低于阈值（由相似度警告阈值推导）的连线标红，提示可能过度拆分、应考虑合并或重点排查。',
      },
      {
        title: '叶节点',
        body: '底部每个标签对应一个缺陷类别；点击可在散点图中高亮该类。',
      },
    ],
    footer: '支持滚轮缩放与拖动平移。',
  },
  graph: {
    intro: '类别关系力导向图：',
    sections: [
      {
        title: '节点大小',
        body: '与该类别 crop 样本数成正比，越大表示数据越多。',
      },
      {
        title: '连线粗细',
        body: '类心余弦相似度越高，连线越粗。仅显示超过阈值的边（约为相似度阈值的 85%）。',
      },
      {
        title: '红色连线',
        body: '类心相似度 ≥ 0.9 的边标红，属于高相似警告。',
      },
      {
        title: '节点内数字',
        body: '该类的样本数量。',
      },
    ],
    footer: '点击节点或边可在散点图中高亮；支持滚轮缩放与拖动平移。',
  },
  warnings: {
    intro: '相似 / 混淆类别警告列表：',
    sections: [
      {
        title: '跨类邻居',
        body: '两类之间较大的 kNN 跨类占比（取双向最大值）。越高表示 A 类样本的邻居越常落入 B 类（或反之）。',
      },
      {
        title: '类心相似',
        body: '两类类心的余弦相似度，越高越易在特征空间中重叠。',
      },
      {
        title: '告警级别',
        body: '红色 = 跨类邻居 ≥ 40% 或类心相似 ≥ 0.95；黄色 = 超过分析中心阈值但未达最高级。',
      },
      {
        title: '阈值参数',
        body: '「混淆警告阈值」「相似度警告阈值」可在分析中心调整，列表会即时过滤。',
      },
    ],
    footer: '点击条目可在分布散点中高亮该类别对。',
  },
  perclass: {
    intro: '每个缺陷类别的质量卡片：',
    sections: [
      {
        title: 'Silhouette',
        body: '轮廓系数：该类是否既抱团、又与其他类分开。越接近 1 越好；接近 0 边界模糊；为负可能和别类混在一起或存在误标。',
      },
      {
        title: '紧致度（距心）',
        body: '类内样本到类心的平均距离，越小表示该类 embedding 越紧凑。',
      },
      {
        title: '最近他类',
        body: '特征空间中与该类别心最相似的其他类别；括号内为类心余弦相似度，越高越容易混淆。',
      },
      {
        title: 'N crops',
        body: '参与本次分析的该类 crop 数量。',
      },
    ],
    footer: '点击卡片可在散点图中高亮该类（及最近他类）。',
  },
  mislabels: {
    intro: '疑似误标样本清单：',
    sections: [
      {
        title: '跨类占比（角标 %）',
        body: '该样本 k 近邻中不属于当前标注类别的比例。越高越像被标错类。',
      },
      {
        title: '当前类 → 建议类',
        body: '左侧删除线 = 现有标注；右侧绿色 = k 近邻多数投票建议的类别。',
      },
      {
        title: '误标最小跨类占比',
        body: '分析中心滑块：仅显示跨类占比不低于该阈值的候选，避免噪声。',
      },
    ],
    footer: '点击图片可查看 crop 与原图标注，人工复核。',
  },
};

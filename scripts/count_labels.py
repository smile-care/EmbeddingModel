#!/usr/bin/env python3
"""
统计 SupCon/补丁元数据 JSON 文件中各个 label 的样本数量。

支持的元数据结构：
- { "patches": [{"label": "...", ...}, ...] }  (项目常用)
- [ {"label": "..."}, ... ]
- 其他常见键：items / data / records（自动探测）

使用示例：
  python scripts/count_labels.py \
    data/zhenyu_data/patches/supcon_labels/顶盖正面/metadata_train.json

可选输出：
  --save counts.json  或  --save counts.csv
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List


def _extract_records(data: Any) -> List[Dict[str, Any]]:
    """从多种结构的JSON中提取样本记录列表。"""
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    if isinstance(data, dict):
        for key in ("patches", "items", "data", "records"):
            v = data.get(key)
            if isinstance(v, list):
                return [x for x in v if isinstance(x, dict)]
        # 如果字典中有单个列表字段
        for v in data.values():
            if isinstance(v, list) and v and isinstance(v[0], dict):
                return v
    return []


def _format_table(counter: Counter) -> str:
    if not counter:
        return "(no labels found)"
    items = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
    label_w = max(5, max(len(str(k)) for k, _ in items))
    count_w = max(5, max(len(str(v)) for _, v in items))
    lines = []
    header = f"{'Label'.ljust(label_w)}  {'Count'.rjust(count_w)}"
    lines.append(header)
    lines.append("-" * len(header))
    for k, v in items:
        lines.append(f"{str(k).ljust(label_w)}  {str(v).rjust(count_w)}")
    lines.append("-" * len(header))
    lines.append(f"Total labels: {len(items)} | Total samples: {sum(counter.values())}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Count label frequencies in metadata JSON")
    parser.add_argument("--metadata", type=str, default="data/zhenyu_data/patches/supcon_labels/顶盖正面/metadata_val.json", help="Path to metadata JSON file")
    parser.add_argument("--save", type=str, default=None, help="Optional output file (.json or .csv)")
    args = parser.parse_args()

    meta_path = Path(args.metadata)
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {meta_path}")

    with meta_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    records = _extract_records(data)
    if not records:
        print("未在JSON中找到记录列表（期望键：patches/items/data/records 或 顶层为列表）")
        return

    labels = []
    for rec in records:
        if isinstance(rec, dict) and "label" in rec:
            label = rec.get("label")
            labels.append("unknown" if label is None else str(label))

    counter = Counter(labels)
    print(_format_table(counter))

    if args.save:
        out_path = Path(args.save)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if out_path.suffix.lower() == ".json":
            with out_path.open("w", encoding="utf-8") as f:
                json.dump({k: v for k, v in sorted(counter.items())}, f, ensure_ascii=False, indent=2)
            print(f"已保存到: {out_path}")
        elif out_path.suffix.lower() == ".csv":
            import csv
            with out_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["label", "count"])
                for k, v in sorted(counter.items()):
                    writer.writerow([k, v])
            print(f"已保存到: {out_path}")
        else:
            # 默认保存为JSON
            with out_path.open("w", encoding="utf-8") as f:
                json.dump({k: v for k, v in sorted(counter.items())}, f, ensure_ascii=False, indent=2)
            print(f"已保存到: {out_path} (按JSON写入)")


if __name__ == "__main__":
    main()

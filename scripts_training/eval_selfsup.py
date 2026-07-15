"""Diagnostic kNN evaluation for selfsup inference checkpoints.

Labels and sources are used only here for offline measurement; they never enter
the self-supervised training loss.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

import torch
import yaml
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.embedding_model.selfsup.datasets import SelfSupervisedDataset, selfsup_collate
from src.embedding_model.selfsup.inference import extract_representations, load_inference_encoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="评估 selfsup representation 的弱标签 kNN 结构")
    parser.add_argument("--checkpoint", required=True, help="inference_model.pth")
    parser.add_argument("--data-config", default=None, help="默认使用 checkpoint 中的数据配置")
    parser.add_argument("--split", choices=["train", "val"], default="val")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=10000)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", default=None, help="可选 JSON 输出路径")
    return parser.parse_args()


def load_sources(path: str, split: str) -> list[dict]:
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    with config_path.open("r", encoding="utf-8") as file:
        data_config = yaml.safe_load(file)
    sources = data_config.get("scenes", {}).get(split, [])
    if not sources:
        raise ValueError(f"数据配置中没有 scenes.{split}")
    return sources


def majority(values: list[str]) -> str:
    return Counter(values).most_common(1)[0][0]


def knn_metrics(embeddings: torch.Tensor, labels: list[str], sources: list[str], k: int) -> dict:
    sample_count = len(labels)
    if sample_count < 2:
        return {"sample_count": sample_count, "note": "样本数不足"}
    effective_k = min(max(1, k), sample_count - 1)
    neighbors = NearestNeighbors(n_neighbors=effective_k + 1, metric="cosine", n_jobs=-1)
    neighbors.fit(embeddings.numpy())
    _, indices = neighbors.kneighbors(embeddings.numpy())
    indices = indices[:, 1:]

    predictions = [majority([labels[index] for index in row]) for row in indices]
    label_knn_accuracy = sum(prediction == target for prediction, target in zip(predictions, labels)) / sample_count
    top1_same_label = sum(labels[row[0]] == labels[i] for i, row in enumerate(indices)) / sample_count
    top1_same_source = sum(sources[row[0]] == sources[i] for i, row in enumerate(indices)) / sample_count

    centered = embeddings - embeddings.mean(dim=0)
    diagnostic_rows = centered[: min(2048, sample_count)]
    singular_values = torch.linalg.svdvals(diagnostic_rows)
    probabilities = singular_values / singular_values.sum().clamp_min(1e-12)
    effective_rank = torch.exp(-(probabilities * probabilities.clamp_min(1e-12).log()).sum()).item()
    return {
        "sample_count": sample_count,
        "embedding_dim": int(embeddings.shape[1]),
        "k": effective_k,
        "label_knn_accuracy": label_knn_accuracy,
        "top1_same_label_rate": top1_same_label,
        "top1_same_source_rate": top1_same_source,
        "effective_rank": effective_rank,
        "note": "label/source 仅用于诊断，未参与训练",
    }


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder, config = load_inference_encoder(args.checkpoint, project_root=ROOT, device=device)
    data_config_path = args.data_config or config["data"]["data_config_path"]
    dataset = SelfSupervisedDataset(
        load_sources(data_config_path, args.split),
        project_root=ROOT,
        image_size=int(config["data"]["image_size"]),
        training=False,
    )
    if args.max_samples > 0 and len(dataset) > args.max_samples:
        indices = random.Random(args.seed).sample(range(len(dataset)), args.max_samples)
        evaluation_dataset = Subset(dataset, indices)
    else:
        evaluation_dataset = dataset
    dataloader = DataLoader(
        evaluation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=selfsup_collate,
    )

    embeddings: list[torch.Tensor] = []
    labels: list[str] = []
    sources: list[str] = []
    for batch in tqdm(dataloader, desc="Extracting"):
        representations = extract_representations(
            encoder,
            batch["view1_image"].to(device, non_blocking=True),
            batch["view1_mask"].to(device, non_blocking=True),
        )
        embeddings.append(representations.cpu())
        labels.extend(batch["label_name"])
        sources.extend(batch["source_name"])

    metrics = knn_metrics(torch.cat(embeddings, dim=0), labels, sources, args.k)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

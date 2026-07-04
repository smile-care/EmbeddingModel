"""Evaluate SupCon/MoCo checkpoints on representations and projections.

This script mirrors the standalone training config/data path and reports the
same geometry metrics for both model output spaces:

  - representations: FeatureFusion / pooling output used by downstream analysis
  - projections: ProjectionHead output used by SupCon/MoCo loss
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.embedding_model.supcon.datasets.supcon_dataset import SupConDataset
from src.embedding_model.supcon.models.backbone.dinov3_convnext import DINOv3ConvNextConfig
from src.embedding_model.supcon.models.backbone.dinov3_vit import DINOv3ViTConfig
from src.embedding_model.supcon.models.convnext_model import ConvNeXtModel
from src.embedding_model.supcon.models.moco_model import MoCoModel
from src.embedding_model.supcon.models.vit_model import ViTModel
from src.embedding_model.utils.config_loader import load_config
from src.embedding_model.utils.metrics import knn_evaluation, similarity_distribution_stats


def resolve_image_size(image_size_config: int | list[int]) -> int:
    if isinstance(image_size_config, int):
        return image_size_config
    if isinstance(image_size_config, list) and image_size_config:
        return int(max(image_size_config))
    raise ValueError(f"image_size 必须是 int 或非空 List[int]，当前为 {image_size_config!r}")


def extract_model_state_dict(checkpoint: object) -> dict[str, torch.Tensor]:
    if not isinstance(checkpoint, dict):
        raise ValueError(f"checkpoint 必须是 dict，当前类型: {type(checkpoint)}")
    for key in ("model_state_dict", "state_dict", "model"):
        state = checkpoint.get(key)
        if isinstance(state, dict) and state:
            return state
    if checkpoint and all(isinstance(k, str) for k in checkpoint.keys()):
        return checkpoint  # type: ignore[return-value]
    raise ValueError("无法在 checkpoint 中找到 model_state_dict / state_dict / model")


def checkpoint_uses_moco(state_dict: dict[str, torch.Tensor]) -> bool:
    return any(k.startswith(("query_encoder.", "momentum_encoder.")) for k in state_dict)


def load_backbone_config(backbone_name: str):
    import yaml

    cfg_path = Path("configs") / "backbone" / f"{backbone_name}.yaml"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"backbone 配置不存在: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as f:
        raw_cfg = yaml.safe_load(f)

    model_type = raw_cfg.get("model_type", "dinov3_convnext")
    if model_type == "dinov3_vit":
        return DINOv3ViTConfig.from_dict(raw_cfg), True
    if model_type == "dinov3_convnext":
        return DINOv3ConvNextConfig.from_dict(raw_cfg), False
    raise ValueError(f"不支持的 backbone model_type: {model_type!r}")


def build_model_from_config(
    model_config: dict[str, Any],
    *,
    image_size: int,
    use_moco: bool,
    device: torch.device,
) -> torch.nn.Module:
    backbone_name = model_config.get("backbone")
    if not backbone_name:
        raise ValueError("model.backbone 不能为空")
    backbone_cfg, is_vit = load_backbone_config(str(backbone_name))

    common_kwargs: dict[str, Any] = {
        "backbone_cfg": backbone_cfg,
        "ckpt_path": None,
        "embedding_dim": int(model_config.get("embedding_dim", 128)),
        "image_size": image_size,
        "freeze_backbone": False,
    }
    if is_vit:
        vit_cfg = model_config.get("vit", {})
        common_kwargs.update(
            {
                "cls_weight": float(vit_cfg.get("cls_weight", 0.3)),
                "projection_hidden_dim": vit_cfg.get(
                    "projection_hidden_dim",
                    model_config.get("projection_hidden_dim"),
                ),
            }
        )
    else:
        cnx_cfg = model_config.get("convnext", {})
        common_kwargs.update(
            {
                "use_layers": list(cnx_cfg.get("use_layers", [1, 2, 3])),
                "fusion_dim": int(cnx_cfg.get("fusion_dim", model_config.get("embedding_dim", 128))),
                "projection_hidden_dim": cnx_cfg.get(
                    "projection_hidden_dim",
                    model_config.get("projection_hidden_dim"),
                ),
                "mask_gating": dict(cnx_cfg.get("mask_gating", {})),
                "pooling": dict(cnx_cfg.get("pooling", {"mode": "fg_only"})),
            }
        )

    if use_moco:
        momentum = float(model_config.get("moco", {}).get("momentum", 0.999))
        return MoCoModel(**common_kwargs, momentum=momentum).to(device)

    model_cls = ViTModel if is_vit else ConvNeXtModel
    return model_cls(**common_kwargs).to(device)


def load_model(
    checkpoint_path: str | Path,
    model_config: dict[str, Any],
    image_size: int,
    device: torch.device,
) -> tuple[torch.nn.Module, bool]:
    checkpoint = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    state_dict = extract_model_state_dict(checkpoint)
    use_moco = checkpoint_uses_moco(state_dict)
    model = build_model_from_config(
        model_config,
        image_size=image_size,
        use_moco=use_moco,
        device=device,
    )
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model, use_moco


def output_metrics(embeddings: torch.Tensor, labels: torch.Tensor, k: int) -> dict[str, Any]:
    embeddings = F.normalize(embeddings, dim=1, p=2, eps=1e-8)
    sim = similarity_distribution_stats(embeddings, labels)
    knn = knn_evaluation(embeddings, labels, k=k)
    return {
        "dim": int(embeddings.shape[1]) if embeddings.ndim == 2 else 0,
        "pos_sim": sim["pos_sim"],
        "neg_sim": sim["neg_sim"],
        "margin": sim["margin"],
        "knn_accuracy": knn.get("knn_accuracy", 0.0),
        "knn_k": knn.get("k", k),
        "knn_note": knn.get("note"),
    }


@torch.no_grad()
def evaluate_dataloader(
    model: torch.nn.Module,
    dataloader: DataLoader,
    *,
    device: torch.device,
    use_moco: bool,
    k: int,
) -> dict[str, Any]:
    all_representations: list[torch.Tensor] = []
    all_projections: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []

    for batch in tqdm(dataloader, desc="Evaluating", leave=False):
        images = batch["view1_image"].to(device, non_blocking=True)
        masks = batch["view1_mask"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        if use_moco:
            outputs = model(images, masks, mode="query", return_features=False)
        else:
            outputs = model(images, masks, return_features=False)

        all_representations.append(outputs["representations"].detach().cpu())
        all_projections.append(outputs["projections"].detach().cpu())
        all_labels.append(labels.detach().cpu())

    if not all_labels:
        return {
            "num_samples": 0,
            "representations": {},
            "projections": {},
        }

    labels = torch.cat(all_labels, dim=0)
    representations = torch.cat(all_representations, dim=0)
    projections = torch.cat(all_projections, dim=0)

    return {
        "num_samples": int(labels.numel()),
        "num_classes": int(labels.unique().numel()),
        "representations": output_metrics(representations, labels, k=k),
        "projections": output_metrics(projections, labels, k=k),
    }


def aggregate_scene_metrics(scene_results: list[dict[str, Any]], key: str) -> dict[str, Any]:
    valid = [r[key] for r in scene_results if r.get("num_samples", 0) > 0 and r.get(key)]
    if not valid:
        return {}
    fields = ("pos_sim", "neg_sim", "margin", "knn_accuracy")
    return {
        "dim": valid[0].get("dim", 0),
        **{field: float(sum(m.get(field, 0.0) for m in valid) / len(valid)) for field in fields},
    }


def build_eval_datasets(
    supcon_config: dict[str, Any],
    *,
    split: str,
) -> tuple[list[str], list[SupConDataset], int]:
    data_config_path = supcon_config["data"]["data_config_path"]
    data_config = load_config(data_config_path)
    scenes = data_config["scenes"]

    scene_cfgs = scenes.get(split, [])
    if not scene_cfgs and split == "val":
        raise ValueError("data_config 中没有 scenes.val；可使用 --split train 评估训练集")
    if not scene_cfgs:
        raise ValueError(f"data_config 中没有 scenes.{split}")

    image_size = resolve_image_size(supcon_config["data"].get("image_size", 224))
    mask_dilation_config = supcon_config["data"].get("mask_dilation", {"enabled": False})
    datasets = [
        SupConDataset(
            root=s["root"],
            split="val",
            image_size=image_size,
            name=s.get("name", s["root"]),
            mask_dilation_config=mask_dilation_config,
        )
        for s in scene_cfgs
    ]
    names = [s.get("name", s["root"]) for s in scene_cfgs]
    return names, datasets, image_size


def print_result_table(results: dict[str, Any]) -> None:
    print("\n=== SupCon Eval Summary ===")
    print(f"checkpoint: {results['checkpoint']}")
    print(f"split: {results['split']}")
    print(f"use_moco: {results['use_moco']}")
    for source in ("representations", "projections"):
        metrics = results["summary"][source]
        print(
            f"{source:<16} dim={metrics.get('dim', 0):<5} "
            f"margin={metrics.get('margin', 0.0):.4f} "
            f"pos={metrics.get('pos_sim', 0.0):.4f} "
            f"neg={metrics.get('neg_sim', 0.0):.4f} "
            f"knn={metrics.get('knn_accuracy', 0.0):.4f}"
        )

    print("\n=== Per Scene ===")
    for scene in results["scenes"]:
        print(f"[{scene['name']}] samples={scene['num_samples']} classes={scene['num_classes']}")
        for source in ("representations", "projections"):
            metrics = scene[source]
            print(
                f"  {source:<16} dim={metrics.get('dim', 0):<5} "
                f"margin={metrics.get('margin', 0.0):.4f} "
                f"pos={metrics.get('pos_sim', 0.0):.4f} "
                f"neg={metrics.get('neg_sim', 0.0):.4f} "
                f"knn={metrics.get('knn_accuracy', 0.0):.4f}"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="评估 SupCon checkpoint 的 representations/projections 表征质量")
    parser.add_argument("--config", type=str, default="configs/supcon_config.yaml", help="SupCon 配置文件")
    parser.add_argument("--checkpoint", type=str, required=True, help="待评估 checkpoint 路径")
    parser.add_argument("--split", choices=["val", "train"], default="val", help="评估 data_config 的哪个 split")
    parser.add_argument("--batch_size", type=int, default=None, help="覆盖配置中的 batch_size")
    parser.add_argument("--num_workers", type=int, default=None, help="覆盖配置中的 num_workers")
    parser.add_argument("--device", type=str, default=None, help="cpu / cuda / cuda:0")
    parser.add_argument("--k", type=int, default=10, help="kNN 评估的 k")
    parser.add_argument("--output_json", type=str, default=None, help="可选：保存评估结果 JSON")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)["supcon"]
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    scene_names, datasets, image_size = build_eval_datasets(config, split=args.split)
    model_config = dict(config["model"])
    model_config["moco"] = dict(config.get("moco", {}))
    model, use_moco = load_model(args.checkpoint, model_config, image_size, device)

    batch_size = int(args.batch_size or config["data"].get("batch_size", 32))
    num_workers = int(args.num_workers if args.num_workers is not None else config["data"].get("num_workers", 4))
    pin_memory = bool(config["data"].get("pin_memory", True)) and device.type == "cuda"

    scene_results: list[dict[str, Any]] = []
    for name, dataset in zip(scene_names, datasets):
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        result = evaluate_dataloader(model, dataloader, device=device, use_moco=use_moco, k=args.k)
        result["name"] = name
        result["categories"] = dataset.categories
        scene_results.append(result)

    results = {
        "checkpoint": str(args.checkpoint),
        "config": str(args.config),
        "split": args.split,
        "use_moco": use_moco,
        "image_size": image_size,
        "summary": {
            "representations": aggregate_scene_metrics(scene_results, "representations"),
            "projections": aggregate_scene_metrics(scene_results, "projections"),
        },
        "scenes": scene_results,
    }

    print_result_table(results)
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n结果已保存: {output_path}")


if __name__ == "__main__":
    main()

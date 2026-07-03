"""
将训练 checkpoint 转为仅含 model_state_dict 的轻量权重文件。

训练脚本保存的 checkpoint 通常包含 optimizer、scheduler、MoCo 队列等；
本脚本只保留模型参数，便于推理部署或作为预训练权重引用。

用法:
  # 完整 MoCo 权重（含 query_encoder + momentum_encoder）
  python scripts/convert_checkpoint_to_weights.py \\
      --input checkpoints/supcon_models/0701_convnext_tiny/current_model.pth \\
      --output checkpoints/supcon_models/0701_convnext_tiny/final_model.pth

  # 仅保留 query_encoder，去掉 momentum_encoder 并剥离前缀（推理部署，约减半体积）
  python scripts/convert_checkpoint_to_weights.py \\
      --input checkpoints/supcon_models/0701_convnext_tiny/final_model.pth \\
      --output checkpoints/supcon_models/0701_convnext_tiny/inference_model.pth \\
      --strip-moco
"""
import argparse
from pathlib import Path

import torch

QUERY_PREFIX = "query_encoder."
MOMENTUM_PREFIX = "momentum_encoder."


def extract_model_state_dict(checkpoint: object) -> dict:
    """从 checkpoint 中解析 model_state_dict。"""
    if not isinstance(checkpoint, dict):
        raise ValueError(f"checkpoint 必须是 dict，当前类型: {type(checkpoint)}")

    for key in ("model_state_dict", "state_dict", "model"):
        state = checkpoint.get(key)
        if isinstance(state, dict) and state:
            return state

    # 已是纯 state_dict（key 形如 backbone.* / query_encoder.*）
    if checkpoint and all(isinstance(k, str) for k in checkpoint.keys()):
        sample_keys = list(checkpoint.keys())[:5]
        if any(
            k.startswith(prefix)
            for k in sample_keys
            for prefix in ("backbone.", "query_encoder.", "momentum_encoder.", "feature_fusion.", "projection_head.")
        ):
            return checkpoint

    raise ValueError(
        "无法在 checkpoint 中找到 model_state_dict / state_dict / model，"
        "请确认输入文件为训练脚本保存的 checkpoint。"
    )


def strip_moco_for_inference(state_dict: dict) -> dict:
    """只保留 query_encoder 权重，去掉 momentum_encoder 并剥离 query_encoder. 前缀。

    MoCo 训练保存的 state_dict 含两套完整 encoder（约 2× backbone 体积）；
    推理只需 query_encoder。输出键名与 ConvNeXtModel / ViTModel 一致（backbone.* 等）。
    """
    if not any(k.startswith(QUERY_PREFIX) for k in state_dict):
        raise ValueError(
            "checkpoint 不含 query_encoder.* 键，不是 MoCo 格式，无需使用 --strip-moco"
        )

    inference_sd: dict = {}
    dropped = 0
    for key, value in state_dict.items():
        if key.startswith(MOMENTUM_PREFIX):
            dropped += 1
            continue
        if key.startswith(QUERY_PREFIX):
            inference_sd[key[len(QUERY_PREFIX):]] = value
        else:
            inference_sd[key] = value

    if not inference_sd:
        raise ValueError("剥离 MoCo 后无有效权重，请检查输入 checkpoint")

    print(f"已剥离 momentum_encoder（丢弃 {dropped} 个键）")
    return inference_sd


def _count_params(state_dict: dict) -> int:
    return sum(v.numel() for v in state_dict.values() if hasattr(v, "numel"))


def convert_checkpoint(
    input_path: Path,
    output_path: Path,
    strip_moco: bool = False,
) -> None:
    if not input_path.is_file():
        raise FileNotFoundError(f"输入文件不存在: {input_path}")

    checkpoint = torch.load(str(input_path), map_location="cpu", weights_only=False)
    model_state_dict = extract_model_state_dict(checkpoint)
    input_params = _count_params(model_state_dict)

    if strip_moco:
        model_state_dict = strip_moco_for_inference(model_state_dict)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model_state_dict}, output_path)

    output_params = _count_params(model_state_dict)
    input_mb = input_path.stat().st_size / 1024 / 1024
    output_mb = output_path.stat().st_size / 1024 / 1024

    print(f"输入:  {input_path}  ({input_mb:.2f} MB, {input_params:,} params)")
    print(f"输出:  {output_path}  ({output_mb:.2f} MB, {output_params:,} params)")
    print(f"键数:  {len(model_state_dict)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="将训练 checkpoint 转为仅含 model_state_dict 的 .pth 文件",
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        default="checkpoints/supcon_models/0701_convnext_tiny/checkpoint_epoch_150.pth",
        help="输入 checkpoint 路径（如 current_model.pth、checkpoint_epoch_100.pth）",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="checkpoints/supcon_models/0701_convnext_tiny/final_model.pth",
        help="输出路径；默认在输入同目录生成 <stem>_weights.pth 或 <stem>_inference.pth（--strip-moco）",
    )
    parser.add_argument(
        "--strip-moco",
        action="store_true",\
        default=True,
        help="仅保留 query_encoder 权重，去掉 momentum_encoder 并剥离前缀（推理部署用）",
    )
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()

    convert_checkpoint(input_path, output_path, strip_moco=args.strip_moco)


if __name__ == "__main__":
    main()

"""Platform training/inference entry point (web platform only).

Two responsibilities live here:

1. Config building — merge ``configs/data_cluster.yaml`` defaults with an
   experiment's stored overrides (``build_infer_config`` / ``_build_default_model_config``).
2. Embedding computation — load a model and run the forward pass.  These are the
   low-level ``_*_raw`` helpers, kept here (rather than in ``embedding_model``)
   because they are only consumed by the web platform; ``embedding_model`` stays
   a platform-agnostic component library and only provides the model factory.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from embedding_model.supcon.build import build_supcon_model
from embedding_model.supcon.datasets.supcon_dataset import MaskSoftDilation
from embedding_model.utils.config_loader import load_config

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DC_CONFIG_PATH = _REPO_ROOT / "configs" / "data_cluster.yaml"

# Stable id used by the API/UI to mean "no trained model — use the pretrained backbone".
DEFAULT_MODEL_ID = "default"
DEFAULT_MODEL_NAME = "默认预训练模型"


# ---------------------------------------------------------------------------
# Low-level embedding computation (pure DL; caller passes a resolved model_config)
# ---------------------------------------------------------------------------
_PROJ_KEY_RE = re.compile(r"^(.*projection_head\.projection\.)(\d+)(\..*)$")


def _align_projection_head(model: torch.nn.Module, state: dict[str, Any]) -> dict[str, Any]:
    """Realign ``projection_head.projection.<i>`` indices that drifted across code revisions.

    The projection head is an ``nn.Sequential``; removing a parameter-less layer
    (e.g. the old ``Dropout``) shifts the integer indices of every following
    parameter-bearing layer even though the ``Linear`` / ``BatchNorm`` shapes are
    unchanged.  This matches the checkpoint's parameter-bearing layers to the
    model's, in order, and rewrites the indices so the weights load by name.
    Only applied when the per-position shapes match, so a genuinely different head
    is left untouched (and simply not loaded).
    """
    model_sd = model.state_dict()

    def grouped(sd: dict[str, Any]) -> dict[str, dict[int, dict[str, tuple]]]:
        g: dict[str, dict[int, dict[str, tuple]]] = {}
        for k, v in sd.items():
            m = _PROJ_KEY_RE.match(k)
            if not m:
                continue
            pre, idx, suf = m.group(1), int(m.group(2)), m.group(3)
            g.setdefault(pre, {}).setdefault(idx, {})[suf] = tuple(getattr(v, "shape", ()))
        return g

    src_g = grouped(state)
    tgt_g = grouped(model_sd)
    if not src_g:
        return state

    new_state = dict(state)
    for pre, src_idx_map in src_g.items():
        tgt_idx_map = tgt_g.get(pre)
        if not tgt_idx_map:
            continue
        src_order = sorted(src_idx_map)
        tgt_order = sorted(tgt_idx_map)
        if len(src_order) != len(tgt_order) or src_order == tgt_order:
            continue
        if any(src_idx_map[s] != tgt_idx_map[t] for s, t in zip(src_order, tgt_order)):
            continue
        for s in src_order:
            for suf in src_idx_map[s]:
                new_state.pop(f"{pre}{s}{suf}", None)
        for s, t in zip(src_order, tgt_order):
            for suf in src_idx_map[s]:
                new_state[f"{pre}{t}{suf}"] = state[f"{pre}{s}{suf}"]
    return new_state


def _load_supcon_model(
    checkpoint_path: str | Path,
    model_config: dict[str, Any],
    image_size: int,
    use_moco: bool,
    device: torch.device,
    lenient: bool = False,
) -> torch.nn.Module:
    """Build the model and load a trained checkpoint (strict=False).

    When ``lenient`` is set, the projection-head Sequential indices are realigned
    first (see :func:`_align_projection_head`) so legacy checkpoints whose head
    layout drifted still load their trained head — used by the default model.
    """
    model, _queue = build_supcon_model(
        model_config,
        image_size=image_size,
        use_moco=use_moco,
        moco_config=model_config.get("moco", {}),
        freeze_backbone=False,
        device=device,
    )
    ckpt = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    state = ckpt
    if isinstance(ckpt, dict):
        state = ckpt.get("model_state_dict") or ckpt.get("state_dict") or ckpt
    if lenient:
        state = _align_projection_head(model, state)
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


def _build_transforms(image_size: int) -> tuple[transforms.Compose, transforms.Compose]:
    image_tf = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    mask_tf = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
    )
    return image_tf, mask_tf


@torch.no_grad()
def _compute_backbone_embeddings_raw(
    model_config: dict[str, Any],
    image_paths: list[Path],
    image_size: int,
    mask_paths: list[Path | None] | None = None,
    batch_size: int = 16,
    device: str | torch.device | None = None,
) -> np.ndarray:
    """Compute embeddings from the *pretrained backbone only* (no trained head).

    Builds the model so the backbone loads its pretrained weights, then uses
    mask-weighted global average pooling over the last backbone feature map as the
    embedding.  This lets the platform offer a usable "default encoder" before any
    experiment has been trained, with deterministic output (only pretrained
    weights are involved — the randomly-initialized FPN / projection head are not).
    """
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, _queue = build_supcon_model(
        model_config,
        image_size=image_size,
        use_moco=False,
        moco_config=model_config.get("moco", {}),
        freeze_backbone=True,
        device=dev,
    )
    model.eval()
    image_tf, mask_tf = _build_transforms(image_size)
    mask_dilation = MaskSoftDilation({"enabled": True})

    if mask_paths is None:
        mask_paths = [None] * len(image_paths)

    out: list[np.ndarray] = []
    for i in range(0, len(image_paths), batch_size):
        batch_imgs: list[torch.Tensor] = []
        batch_masks: list[torch.Tensor] = []
        for img_p, mask_p in zip(image_paths[i : i + batch_size], mask_paths[i : i + batch_size]):
            try:
                image = Image.open(img_p).convert("RGB")
                image_tensor = image_tf(image)
            except OSError:
                image_tensor = torch.zeros(3, image_size, image_size)

            if mask_p is not None and Path(mask_p).is_file():
                try:
                    mask_img = Image.open(mask_p).convert("L")
                    mask_tensor = (mask_tf(mask_img) > 0.5).float()
                except OSError:
                    mask_tensor = torch.ones(1, image_size, image_size)
            else:
                mask_tensor = torch.ones(1, image_size, image_size)
            mask_tensor = mask_dilation(mask_tensor)

            batch_imgs.append(image_tensor)
            batch_masks.append(mask_tensor)

        x = torch.stack(batch_imgs).to(dev)
        m = torch.stack(batch_masks).to(dev)

        feats = model.backbone(x, output_hidden_states=True)
        last = feats[-1]  # (B, C, h, w)
        m_ds = F.interpolate(m, size=last.shape[-2:], mode="area")
        num = (last * m_ds).sum(dim=(2, 3))
        den = m_ds.sum(dim=(2, 3)).clamp_min(1e-6)
        pooled = num / den
        emb = F.normalize(pooled, dim=1, p=2, eps=1e-8)
        out.append(emb.cpu().numpy())

    if not out:
        return np.zeros((0, 0), dtype=np.float32)
    return np.vstack(out).astype(np.float32, copy=False)


@torch.no_grad()
def _compute_supcon_embeddings_raw(
    checkpoint_path: str | Path,
    image_paths: list[Path],
    model_config: dict[str, Any],
    image_size: int,
    use_moco: bool = False,
    mask_paths: list[Path | None] | None = None,
    batch_size: int = 16,
    device: str | torch.device | None = None,
    lenient: bool = False,
) -> np.ndarray:
    """Compute L2-normalized embeddings for a list of (image, optional mask) pairs."""
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = _load_supcon_model(checkpoint_path, model_config, image_size, use_moco, dev, lenient=lenient)
    image_tf, mask_tf = _build_transforms(image_size)
    mask_dilation = MaskSoftDilation({"enabled": True})

    if mask_paths is None:
        mask_paths = [None] * len(image_paths)

    out: list[np.ndarray] = []
    for i in range(0, len(image_paths), batch_size):
        batch_imgs: list[torch.Tensor] = []
        batch_masks: list[torch.Tensor] = []
        for img_p, mask_p in zip(image_paths[i : i + batch_size], mask_paths[i : i + batch_size]):
            try:
                image = Image.open(img_p).convert("RGB")
                image_tensor = image_tf(image)
            except OSError:
                image_tensor = torch.zeros(3, image_size, image_size)

            if mask_p is not None and Path(mask_p).is_file():
                try:
                    mask_img = Image.open(mask_p).convert("L")
                    mask_tensor = (mask_tf(mask_img) > 0.5).float()
                except OSError:
                    mask_tensor = torch.ones(1, image_size, image_size)
            else:
                mask_tensor = torch.ones(1, image_size, image_size)
            mask_tensor = mask_dilation(mask_tensor)

            batch_imgs.append(image_tensor)
            batch_masks.append(mask_tensor)

        x = torch.stack(batch_imgs).to(dev)
        m = torch.stack(batch_masks).to(dev)

        if use_moco:
            pred = model(x, m, mode="query", return_features=False, return_segmentation=False)
        else:
            pred = model(x, m, return_features=False, return_segmentation=False)
        emb = F.normalize(pred["embeddings"], dim=1, p=2, eps=1e-8)
        out.append(emb.cpu().numpy())

    if not out:
        return np.zeros((0, 0), dtype=np.float32)
    return np.vstack(out).astype(np.float32, copy=False)


def _coerce_bool(cfg: dict[str, Any], *keys: str, default: bool) -> bool:
    for key in keys:
        if key not in cfg:
            continue
        value = cfg[key]
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return default


def _coerce_image_size(cfg: dict[str, Any]) -> int | list[int] | None:
    value = cfg.get("imageSize", cfg.get("image_size"))
    if isinstance(value, int) and value > 0:
        return value
    if isinstance(value, list):
        out = [int(x) for x in value if isinstance(x, (int, float)) and int(x) > 0]
        return out or None
    return None


def _load_dc_config() -> dict[str, Any]:
    try:
        return load_config(str(_DC_CONFIG_PATH))
    except Exception:
        return {}


def build_infer_config(exp_config: dict[str, Any] | None) -> tuple[dict[str, Any], int, bool]:
    """Merge data_cluster.yaml defaults with exp_config overrides.

    Returns ``(model_config, image_size, use_moco)`` ready for embedding_model.
    """
    dc_cfg = _load_dc_config()
    dc = dc_cfg.get("data_cluster", {}) if dc_cfg else {}
    exp_cfg = exp_config if isinstance(exp_config, dict) else {}

    data_cfg = dict(dc.get("data", {}))
    model_cfg = dict(dc.get("model", {}))
    moco_cfg = dict(dc.get("moco", {}))

    image_size = _coerce_image_size(exp_cfg)
    if image_size is not None:
        data_cfg["image_size"] = image_size
    elif data_cfg.get("image_size") is None:
        data_cfg["image_size"] = 224

    backbone = exp_cfg.get("backbone") or model_cfg.get("backbone") or "convnext_small"
    backbone_name = exp_cfg.get("modelName") or exp_cfg.get("model_name") or backbone
    if isinstance(backbone_name, str) and backbone_name.strip():
        model_cfg["backbone"] = backbone_name.strip()

    embedding_dim = exp_cfg.get("embeddingDim") or exp_cfg.get("embedding_dim")
    if isinstance(embedding_dim, int) and embedding_dim > 0:
        model_cfg["embedding_dim"] = embedding_dim
    elif "embedding_dim" not in model_cfg:
        model_cfg["embedding_dim"] = 128

    if "projection_hidden_dims" not in model_cfg and "projection_head" not in model_cfg:
        model_cfg["projection_hidden_dims"] = [256, 128]

    use_moco = _coerce_bool(exp_cfg, "useMoCo", "use_moco", default=moco_cfg.get("enabled", False))
    moco_cfg["enabled"] = use_moco
    model_cfg["moco"] = moco_cfg

    image_size_cfg = data_cfg["image_size"]
    if isinstance(image_size_cfg, list):
        model_image_size = int(max(image_size_cfg))
    else:
        model_image_size = int(image_size_cfg)

    return model_cfg, model_image_size, use_moco


def compute_supcon_embeddings(
    checkpoint_path: str | Path,
    image_paths: list[Path],
    mask_paths: list[Path | None] | None = None,
    exp_config: dict[str, Any] | None = None,
    batch_size: int = 16,
    device: str | None = None,
) -> np.ndarray:
    """Compute embeddings for dataset crops using the experiment's trained model."""
    model_cfg, image_size, use_moco = build_infer_config(exp_config)
    return _compute_supcon_embeddings_raw(
        checkpoint_path=checkpoint_path,
        image_paths=image_paths,
        model_config=model_cfg,
        image_size=image_size,
        use_moco=use_moco,
        mask_paths=mask_paths,
        batch_size=batch_size,
        device=device,
    )


def _build_default_model_config() -> tuple[dict[str, Any], int]:
    """Resolve the default pretrained-backbone config from data_cluster.yaml.

    Returns ``(model_config, image_size)``.  ``pretrained_path`` is only set when
    the weights file actually exists, so a missing checkpoint degrades to a
    randomly-initialized backbone instead of crashing.
    """
    dc = _load_dc_config().get("data_cluster", {})
    training = dc.get("training", {}) if isinstance(dc, dict) else {}
    backbones = dc.get("backbones", {}) if isinstance(dc, dict) else {}
    model_cfg = dict(dc.get("model", {})) if isinstance(dc, dict) else {}

    default_backbone = (
        training.get("default_backbone")
        or model_cfg.get("backbone")
        or "convnext_small"
    )
    model_cfg["backbone"] = default_backbone

    info = backbones.get(default_backbone, {}) if isinstance(backbones, dict) else {}
    if isinstance(info, dict):
        bb_cfg = info.get("backbone_config")
        if bb_cfg:
            bb_path = Path(bb_cfg)
            if not bb_path.is_absolute():
                bb_path = (_REPO_ROOT / bb_path).resolve()
            model_cfg["backbone_config_path"] = str(bb_path)
        pretrained = info.get("pretrained_path")
        if pretrained:
            p = Path(pretrained)
            if not p.is_absolute():
                p = (_REPO_ROOT / p).resolve()
            if p.is_file():
                model_cfg["pretrained_path"] = str(p)

    data_cfg = dc.get("data", {}) if isinstance(dc, dict) else {}
    image_size = data_cfg.get("image_size", 224)
    if isinstance(image_size, list):
        image_size = int(max(image_size)) if image_size else 224
    else:
        image_size = int(image_size)
    return model_cfg, image_size


def _inspect_checkpoint(ckpt_path: str | Path) -> tuple[bool, bool]:
    """Peek at a checkpoint to decide how to run inference on it.

    Returns ``(use_moco, has_trained_head)``:
      * ``use_moco``         - state dict uses ``query_encoder.*`` / ``momentum_encoder.*``
      * ``has_trained_head`` - a trained projection head is present, so embeddings
        should come from the head (consistent with training) rather than backbone GAP.
    """
    try:
        ck = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    except Exception:
        return False, False

    sd = ck
    if isinstance(ck, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            if isinstance(ck.get(key), dict):
                sd = ck[key]
                break
    if not isinstance(sd, dict):
        return False, False

    keys = list(sd.keys())
    use_moco = any(k.startswith(("query_encoder.", "momentum_encoder.")) for k in keys)
    has_trained_head = any("projection_head." in k for k in keys)
    return use_moco, has_trained_head


def compute_default_embeddings(
    image_paths: list[Path],
    mask_paths: list[Path | None] | None = None,
    batch_size: int = 16,
    device: str | None = None,
) -> np.ndarray:
    """Compute embeddings for the platform's default pretrained model.

    Two regimes, chosen by inspecting the resolved default checkpoint:

    * **Trained checkpoint (has a projection head)** — run the *full* model forward
      and take the projection-head ``embeddings``, exactly like a normal trained
      experiment.  This keeps the default model's features consistent with how it
      was trained (head output, not raw backbone features).
    * **Backbone-only weights (or no checkpoint)** — there is no meaningful trained
      head, so fall back to mask-weighted global average pooling over the pretrained
      backbone's last feature map.
    """
    model_cfg, image_size = _build_default_model_config()
    ckpt_path = model_cfg.get("pretrained_path")

    if ckpt_path:
        use_moco, has_trained_head = _inspect_checkpoint(ckpt_path)
        if has_trained_head:
            return _compute_supcon_embeddings_raw(
                checkpoint_path=ckpt_path,
                image_paths=image_paths,
                model_config=model_cfg,
                image_size=image_size,
                use_moco=use_moco,
                mask_paths=mask_paths,
                batch_size=batch_size,
                device=device,
                lenient=True,
            )

    return _compute_backbone_embeddings_raw(
        model_config=model_cfg,
        image_paths=image_paths,
        image_size=image_size,
        mask_paths=mask_paths,
        batch_size=batch_size,
        device=device,
    )

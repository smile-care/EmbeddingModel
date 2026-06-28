"""DINOv3 ConvNeXt Backbone — no transformers dependency."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class DINOv3ConvNextConfig:
    hidden_sizes: List[int] = field(default_factory=lambda: [96, 192, 384, 768])
    depths: List[int]       = field(default_factory=lambda: [3, 3, 27, 3])
    num_channels: int       = 3
    image_size: int         = 224
    hidden_act: str         = "gelu"
    layer_norm_eps: float   = 1e-6
    layer_scale_init_value: float = 1e-6
    drop_path_rate: float   = 0.0
    initializer_range: float = 0.02

    @property
    def num_stages(self) -> int:
        return len(self.hidden_sizes)

    @classmethod
    def from_dict(cls, d: dict) -> DINOv3ConvNextConfig:
        known = cls.__dataclass_fields__.keys()
        return cls(**{k: v for k, v in d.items() if k in known})

    @classmethod
    def from_yaml(cls, path: str) -> DINOv3ConvNextConfig:
        with open(path) as f:
            return cls.from_dict(yaml.safe_load(f))


# ---------------------------------------------------------------------------
# Activation functions
# ---------------------------------------------------------------------------

def _gelu_new(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x**3)))


_ACT2FN = {
    "gelu":     F.gelu,
    "relu":     F.relu,
    "silu":     F.silu,
    "swish":    F.silu,
    "gelu_new": _gelu_new,
}


# ---------------------------------------------------------------------------
# Stochastic depth
# ---------------------------------------------------------------------------

class _DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = keep + torch.rand(shape, dtype=x.dtype, device=x.device)
        mask.floor_()
        return x.div(keep) * mask


# ---------------------------------------------------------------------------
# LayerNorm for channels-first (B, C, H, W) tensors
# ---------------------------------------------------------------------------

class _ChannelLN(nn.LayerNorm):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 1)
        x = super().forward(x)
        # Keep BCHW contiguous to satisfy DDP gradient layout contract.
        return x.permute(0, 3, 1, 2).contiguous()


# ---------------------------------------------------------------------------
# ConvNext block
# ---------------------------------------------------------------------------

class _ConvNextLayer(nn.Module):
    def __init__(self, cfg: DINOv3ConvNextConfig, channels: int, drop_path: float = 0.0):
        super().__init__()
        self.depthwise_conv  = nn.Conv2d(channels, channels, kernel_size=7, padding=3, groups=channels)
        self.layer_norm      = nn.LayerNorm(channels, eps=cfg.layer_norm_eps)
        self.pointwise_conv1 = nn.Linear(channels, 4 * channels)
        self.activation_fn   = _ACT2FN[cfg.hidden_act]
        self.pointwise_conv2 = nn.Linear(4 * channels, channels)
        self.gamma           = nn.Parameter(torch.full((channels,), cfg.layer_scale_init_value))
        self.drop_path       = _DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.depthwise_conv(x)
        x = x.permute(0, 2, 3, 1)          # BCHW → BHWC
        x = self.layer_norm(x)
        x = self.pointwise_conv1(x)
        x = self.activation_fn(x)
        x = self.pointwise_conv2(x)
        x = x * self.gamma
        # Keep BCHW contiguous to avoid stride-mismatch grads under DDP.
        x = x.permute(0, 3, 1, 2).contiguous()  # BHWC → BCHW
        return residual + self.drop_path(x)


# ---------------------------------------------------------------------------
# Stage
# ---------------------------------------------------------------------------

class _Stage(nn.Module):
    def __init__(self, cfg: DINOv3ConvNextConfig, stage_idx: int):
        super().__init__()
        in_ch  = cfg.hidden_sizes[stage_idx - 1] if stage_idx > 0 else cfg.num_channels
        out_ch = cfg.hidden_sizes[stage_idx]

        if stage_idx == 0:
            self.downsample_layers = nn.ModuleList([
                nn.Conv2d(cfg.num_channels, out_ch, kernel_size=4, stride=4),
                _ChannelLN(out_ch, eps=cfg.layer_norm_eps),
            ])
        else:
            self.downsample_layers = nn.ModuleList([
                _ChannelLN(in_ch, eps=cfg.layer_norm_eps),
                nn.Conv2d(in_ch, out_ch, kernel_size=2, stride=2),
            ])

        total = sum(cfg.depths)
        prev  = sum(cfg.depths[:stage_idx])
        dp    = np.linspace(0, cfg.drop_path_rate, total).tolist()

        self.layers = nn.ModuleList([
            _ConvNextLayer(cfg, out_ch, dp[prev + i])
            for i in range(cfg.depths[stage_idx])
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.downsample_layers:
            x = layer(x)
        for layer in self.layers:
            x = layer(x)
        return x


# ---------------------------------------------------------------------------
# Backbone
# ---------------------------------------------------------------------------

class DINOv3ConvNext(nn.Module):
    """DINOv3 ConvNeXt Backbone — pure PyTorch, no transformers dependency."""

    def __init__(
        self,
        cfg: Optional[DINOv3ConvNextConfig] = None,
        ckpt_path: Optional[str] = None,
        freeze_backbone: bool = False,
        # Legacy kwarg kept for compatibility
        model_name: Optional[str] = None,
    ):
        """
        Args:
            cfg: DINOv3ConvNextConfig instance. If None, a default small config is used.
            ckpt_path: Path to a .pth checkpoint with format {"model": state_dict, ...}.
            freeze_backbone: Freeze all parameters after loading.
            model_name: Ignored (kept for backward compatibility).
        """
        super().__init__()

        if cfg is None:
            cfg = DINOv3ConvNextConfig()

        self.stages = nn.ModuleList([_Stage(cfg, i) for i in range(cfg.num_stages)])

        if ckpt_path is not None:
            self._load_checkpoint(ckpt_path)

        if freeze_backbone:
            for param in self.parameters():
                param.requires_grad = False

    def _load_checkpoint(self, path: str) -> None:
        ck = torch.load(path, map_location="cpu")
        sd = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
        # Keep only keys that belong to stages
        stages_sd = {k: v for k, v in sd.items() if k.startswith("stages.")}
        missing, unexpected = self.load_state_dict(stages_sd, strict=False)
        if missing:
            print(f"[DINOv3ConvNext] missing keys: {missing[:5]}{'...' if len(missing) > 5 else ''}")
        if unexpected:
            print(f"[DINOv3ConvNext] unexpected keys: {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        output_hidden_states: bool = False,
    ) -> Union[Tuple[torch.Tensor, ...], torch.Tensor]:
        """
        Args:
            pixel_values: (B, C, H, W)
            output_hidden_states: return feature maps from every stage when True.

        Returns:
            Tuple of per-stage feature maps when output_hidden_states=True,
            otherwise the final stage feature map.
        """
        x = pixel_values
        all_hidden_states = []

        for stage in self.stages:
            x = stage(x)
            if output_hidden_states:
                all_hidden_states.append(x)

        if output_hidden_states:
            return tuple(all_hidden_states)
        return x

    def freeze(self) -> None:
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self) -> None:
        for param in self.parameters():
            param.requires_grad = True

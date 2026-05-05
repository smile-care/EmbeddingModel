"""DINOv3 ViT Backbone — no transformers dependency."""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class DINOv3ViTConfig:
    hidden_size: int = 384
    intermediate_size: int = 1536
    num_hidden_layers: int = 12
    num_attention_heads: int = 6
    num_channels: int = 3
    image_size: int = 224
    patch_size: int = 16
    num_register_tokens: int = 0
    use_gated_mlp: bool = False
    hidden_act: str = "gelu"
    layer_norm_eps: float = 1e-5
    attention_dropout: float = 0.0
    query_bias: bool = True
    key_bias: bool = False
    value_bias: bool = True
    proj_bias: bool = True
    mlp_bias: bool = True
    layerscale_value: float = 1.0
    drop_path_rate: float = 0.0
    rope_theta: float = 100.0
    pos_embed_shift: Optional[float] = None
    pos_embed_jitter: Optional[float] = None
    pos_embed_rescale: Optional[float] = 2.0
    initializer_range: float = 0.02

    @property
    def num_patches(self) -> int:
        return (self.image_size // self.patch_size) ** 2

    @property
    def num_prefix_tokens(self) -> int:
        """CLS + register tokens before patch tokens."""
        return 1 + self.num_register_tokens

    @classmethod
    def from_dict(cls, d: dict) -> DINOv3ViTConfig:
        known = cls.__dataclass_fields__.keys()
        return cls(**{k: v for k, v in d.items() if k in known})

    @classmethod
    def from_yaml(cls, path: str) -> DINOv3ViTConfig:
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
# Patch embeddings
# ---------------------------------------------------------------------------

class _Embeddings(nn.Module):
    def __init__(self, cfg: DINOv3ViTConfig):
        super().__init__()
        self.cls_token        = nn.Parameter(torch.randn(1, 1, cfg.hidden_size))
        self.mask_token       = nn.Parameter(torch.zeros(1, 1, cfg.hidden_size))
        self.register_tokens  = nn.Parameter(torch.empty(1, cfg.num_register_tokens, cfg.hidden_size))
        self.patch_embeddings = nn.Conv2d(
            cfg.num_channels, cfg.hidden_size,
            kernel_size=cfg.patch_size, stride=cfg.patch_size,
        )

    def forward(self, x: torch.Tensor, bool_masked_pos=None) -> torch.Tensor:
        B = x.shape[0]
        dtype = self.patch_embeddings.weight.dtype
        patches = self.patch_embeddings(x.to(dtype=dtype)).flatten(2).transpose(1, 2)
        if bool_masked_pos is not None:
            patches = torch.where(
                bool_masked_pos.unsqueeze(-1),
                self.mask_token.to(patches.dtype),
                patches,
            )
        return torch.cat(
            [self.cls_token.expand(B, -1, -1),
             self.register_tokens.expand(B, -1, -1),
             patches],
            dim=1,
        )


# ---------------------------------------------------------------------------
# RoPE
# ---------------------------------------------------------------------------

@lru_cache(maxsize=32)
def _patch_coords(nh: int, nw: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    h = torch.arange(0.5, nh, dtype=dtype, device=device) / nh
    w = torch.arange(0.5, nw, dtype=dtype, device=device) / nw
    coords = torch.stack(torch.meshgrid(h, w, indexing="ij"), dim=-1).flatten(0, 1)
    return 2.0 * coords - 1.0


class _RopeEmbedding(nn.Module):
    inv_freq: torch.Tensor

    def __init__(self, cfg: DINOv3ViTConfig):
        super().__init__()
        self.cfg      = cfg
        self.head_dim = cfg.hidden_size // cfg.num_attention_heads
        inv_freq = 1 / cfg.rope_theta ** torch.arange(0, 1, 4 / self.head_dim, dtype=torch.float32)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, pixel_values: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        _, _, H, W = pixel_values.shape
        nh, nw = H // self.cfg.patch_size, W // self.cfg.patch_size
        dev = pixel_values.device
        dev_type = "cpu" if dev.type == "mps" else dev.type
        with torch.autocast(device_type=dev_type, enabled=False):
            coords = _patch_coords(nh, nw, torch.float32, dev)
            if self.training and self.cfg.pos_embed_rescale is not None:
                scale = torch.empty(1, device=dev).uniform_(
                    -math.log(self.cfg.pos_embed_rescale),
                     math.log(self.cfg.pos_embed_rescale),
                ).exp()
                coords = coords * scale
            angles = 2 * math.pi * coords[:, :, None] * self.inv_freq[None, None, :]
            angles = angles.flatten(1, 2).tile(2)
            cos, sin = torch.cos(angles), torch.sin(angles)
        return cos.to(pixel_values.dtype), sin.to(pixel_values.dtype)


# ---------------------------------------------------------------------------
# Attention with RoPE
# ---------------------------------------------------------------------------

def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    h = x.shape[-1] // 2
    return torch.cat((-x[..., h:], x[..., :h]), dim=-1)


def _apply_rope(q, k, cos, sin):
    n_pre = q.shape[-2] - sin.shape[-2]
    q_pre, q_pat = q.split((n_pre, sin.shape[-2]), dim=-2)
    k_pre, k_pat = k.split((n_pre, sin.shape[-2]), dim=-2)
    q_pat = q_pat * cos + _rotate_half(q_pat) * sin
    k_pat = k_pat * cos + _rotate_half(k_pat) * sin
    return torch.cat((q_pre, q_pat), dim=-2), torch.cat((k_pre, k_pat), dim=-2)


class _Attention(nn.Module):
    def __init__(self, cfg: DINOv3ViTConfig):
        super().__init__()
        self.num_heads = cfg.num_attention_heads
        self.head_dim  = cfg.hidden_size // self.num_heads
        self.scale     = self.head_dim ** -0.5
        self.dropout   = cfg.attention_dropout
        d = cfg.hidden_size
        self.q_proj = nn.Linear(d, d, bias=cfg.query_bias)
        self.k_proj = nn.Linear(d, d, bias=cfg.key_bias)
        self.v_proj = nn.Linear(d, d, bias=cfg.value_bias)
        self.o_proj = nn.Linear(d, d, bias=cfg.proj_bias)

    def forward(self, x: torch.Tensor, rope: Tuple) -> torch.Tensor:
        B, N, _ = x.shape
        H, D = self.num_heads, self.head_dim
        q = self.q_proj(x).view(B, N, H, D).transpose(1, 2)
        k = self.k_proj(x).view(B, N, H, D).transpose(1, 2)
        v = self.v_proj(x).view(B, N, H, D).transpose(1, 2)
        q, k = _apply_rope(q, k, *rope)
        attn = F.softmax(torch.matmul(q, k.transpose(2, 3)) * self.scale, dim=-1)
        attn = F.dropout(attn, p=self.dropout, training=self.training)
        return self.o_proj(torch.matmul(attn, v).transpose(1, 2).reshape(B, N, -1))


# ---------------------------------------------------------------------------
# MLP
# ---------------------------------------------------------------------------

class _MLP(nn.Module):
    def __init__(self, cfg: DINOv3ViTConfig):
        super().__init__()
        self.act       = _ACT2FN[cfg.hidden_act]
        self.up_proj   = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=cfg.mlp_bias)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=cfg.mlp_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act(self.up_proj(x)))


class _GatedMLP(nn.Module):
    def __init__(self, cfg: DINOv3ViTConfig):
        super().__init__()
        self.act       = _ACT2FN[cfg.hidden_act]
        self.gate_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=cfg.mlp_bias)
        self.up_proj   = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=cfg.mlp_bias)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=cfg.mlp_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act(self.gate_proj(x)) * self.up_proj(x))


# ---------------------------------------------------------------------------
# Transformer block
# ---------------------------------------------------------------------------

class _LayerScale(nn.Module):
    def __init__(self, cfg: DINOv3ViTConfig):
        super().__init__()
        self.lambda1 = nn.Parameter(cfg.layerscale_value * torch.ones(cfg.hidden_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.lambda1


class _Block(nn.Module):
    def __init__(self, cfg: DINOv3ViTConfig):
        super().__init__()
        self.norm1        = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)
        self.attention    = _Attention(cfg)
        self.layer_scale1 = _LayerScale(cfg)
        self.drop_path    = _DropPath(cfg.drop_path_rate) if cfg.drop_path_rate > 0 else nn.Identity()
        self.norm2        = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)
        self.mlp          = _GatedMLP(cfg) if cfg.use_gated_mlp else _MLP(cfg)
        self.layer_scale2 = _LayerScale(cfg)

    def forward(self, x: torch.Tensor, rope: Tuple) -> torch.Tensor:
        x = x + self.drop_path(self.layer_scale1(self.attention(self.norm1(x), rope)))
        x = x + self.drop_path(self.layer_scale2(self.mlp(self.norm2(x))))
        return x


# ---------------------------------------------------------------------------
# Backbone
# ---------------------------------------------------------------------------

class DINOv3ViT(nn.Module):
    """DINOv3 ViT Backbone — pure PyTorch, no transformers dependency.

    forward() returns:
      - output_hidden_states=False: patch token tensor (B, N, D), register/CLS stripped
      - output_hidden_states=True:  (cls_token, patch_tokens)
            cls_token:    (B, D)
            patch_tokens: (B, N, D)  — only patch tokens, CLS and registers removed
    """

    def __init__(
        self,
        cfg: Optional[DINOv3ViTConfig] = None,
        ckpt_path: Optional[str] = None,
        freeze_backbone: bool = False,
    ):
        """
        Args:
            cfg: DINOv3ViTConfig instance. If None, a default ViT-S/16 config is used.
            ckpt_path: Path to a .pth checkpoint with format {"model": state_dict, ...}.
            freeze_backbone: Freeze all parameters after loading.
        """
        super().__init__()

        if cfg is None:
            cfg = DINOv3ViTConfig()

        self.config          = cfg
        self.embeddings      = _Embeddings(cfg)
        self.rope_embeddings = _RopeEmbedding(cfg)
        # key namespace matches HF checkpoint: model.layer.*
        self.model           = nn.ModuleDict({
            "layer": nn.ModuleList([_Block(cfg) for _ in range(cfg.num_hidden_layers)])
        })
        self.norm            = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)

        if ckpt_path is not None:
            self._load_checkpoint(ckpt_path)

        if freeze_backbone:
            for param in self.parameters():
                param.requires_grad = False

    def _load_checkpoint(self, path: str) -> None:
        ck = torch.load(path, map_location="cpu")
        sd = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
        missing, unexpected = self.load_state_dict(sd, strict=False)
        if missing:
            print(f"[DINOv3ViT] missing keys: {missing[:5]}{'...' if len(missing) > 5 else ''}")
        if unexpected:
            print(f"[DINOv3ViT] unexpected keys: {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")
        print(f"[DINOv3ViT] loaded checkpoint from {path!r}")

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        output_hidden_states: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            pixel_values: (B, C, H, W)
            output_hidden_states: when True, returns (cls_token, patch_tokens);
                                  when False, returns patch_tokens only.

        Returns:
            patch_tokens: (B, N, D)  — always excludes CLS and register tokens
            cls_token:    (B, D)     — only when output_hidden_states=True
        """
        pixel_values = pixel_values.to(self.embeddings.patch_embeddings.weight.dtype)
        x    = self.embeddings(pixel_values)
        rope = self.rope_embeddings(pixel_values)

        for block in self.model["layer"]:
            x = block(x, rope)
        x = self.norm(x)

        # Split prefix tokens (CLS + registers) from patch tokens
        n_prefix = self.config.num_prefix_tokens
        cls_token    = x[:, 0]           # (B, D)
        patch_tokens = x[:, n_prefix:]   # (B, N, D)  — registers discarded

        if output_hidden_states:
            return cls_token, patch_tokens
        return patch_tokens

    def freeze(self) -> None:
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self) -> None:
        for param in self.parameters():
            param.requires_grad = True

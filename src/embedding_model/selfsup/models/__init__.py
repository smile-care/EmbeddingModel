"""Backbone adapters and objective heads for self-supervised learning."""

from .encoder import MaskAwareEncoder, build_mask_aware_encoder
from .heads import DINOHead, VICRegProjector
from .teacher_student import EMATeacher

__all__ = [
    "DINOHead",
    "EMATeacher",
    "MaskAwareEncoder",
    "VICRegProjector",
    "build_mask_aware_encoder",
]

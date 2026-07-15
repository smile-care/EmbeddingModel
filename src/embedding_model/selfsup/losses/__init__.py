"""Self-supervised objectives."""

from .dino import DINOLoss
from .vicreg import VICRegLoss

__all__ = ["DINOLoss", "VICRegLoss"]

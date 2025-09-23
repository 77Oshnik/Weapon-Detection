"""Model training and inference modules."""

from .trainer import ModelTrainer
from .yolo_model import YOLOWeaponDetector

__all__ = ['ModelTrainer', 'YOLOWeaponDetector']
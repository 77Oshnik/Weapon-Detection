"""Weapon Detection System for Security Applications."""

__version__ = "1.0.0"
__author__ = "Oshnik"
__description__ = "AI-powered weapon detection system for real-time surveillance"

from .detector import WeaponDetector
from .video import VideoProcessor
from .models import ModelTrainer

__all__ = ['WeaponDetector', 'VideoProcessor', 'ModelTrainer']
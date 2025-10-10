"""Slimmed-down DeepCSI inverse scattering toolkit."""

from .trainer import InverseScatteringTrainer, TrainerConfig
from .main import main

__all__ = [
    "InverseScatteringTrainer",
    "TrainerConfig",
    "main",
]

"""
Typed configuration entities.

These dataclasses give the values loaded from config/config.yaml and
params.yaml a concrete, documented shape, matching the paths and
hyperparameters already used in research/03_model_training_and_evaluation.ipynb.
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataConfig:
    raw_dir: Path
    processed_dir: Path
    splits_dir: Path
    figures_dir: Path


@dataclass(frozen=True)
class ModelTrainingConfig:
    models_dir: Path
    eval_dir: Path
    epochs: int
    batch_size: int
    learning_rate: float
    mdeberta_learning_rate: float
    max_length: int
    warmup_ratio: float
    seed: int

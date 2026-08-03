"""
Reads config/config.yaml and params.yaml and exposes them as the typed
entities defined in news_nli.entity.config_entity.
"""

from pathlib import Path

from news_nli.constants import CONFIG_FILE_PATH, PARAMS_FILE_PATH
from news_nli.utils.common import read_yaml, create_directories
from news_nli.entity.config_entity import DataConfig, ModelTrainingConfig


class ConfigurationManager:
    def __init__(
        self,
        config_filepath: Path = CONFIG_FILE_PATH,
        params_filepath: Path = PARAMS_FILE_PATH,
    ):
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

    def get_data_config(self) -> DataConfig:
        cfg = self.config.data
        create_directories([cfg.raw_dir, cfg.processed_dir, cfg.splits_dir, cfg.figures_dir])
        return DataConfig(
            raw_dir=Path(cfg.raw_dir),
            processed_dir=Path(cfg.processed_dir),
            splits_dir=Path(cfg.splits_dir),
            figures_dir=Path(cfg.figures_dir),
        )

    def get_model_training_config(self) -> ModelTrainingConfig:
        cfg = self.config.model_training
        params = self.params
        create_directories([cfg.models_dir, cfg.eval_dir])
        return ModelTrainingConfig(
            models_dir=Path(cfg.models_dir),
            eval_dir=Path(cfg.eval_dir),
            epochs=params.EPOCHS,
            batch_size=params.BATCH_SIZE,
            learning_rate=params.LEARNING_RATE,
            mdeberta_learning_rate=params.MDEBERTA_LR,
            max_length=params.MAX_LENGTH,
            warmup_ratio=params.WARMUP_RATIO,
            seed=params.SEED,
        )

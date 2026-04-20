"""Common utilities: file I/O, config loading, timing."""
import json, time, yaml
from pathlib import Path
from typing import Any
from box import ConfigBox
from news_nli.logging import logger


def read_yaml(path: Path) -> ConfigBox:
    with open(path) as f:
        content = yaml.safe_load(f)
    logger.info(f"Loaded yaml: {path}")
    return ConfigBox(content)


def create_dirs(paths: list[Path]) -> None:
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)
        logger.info(f"Directory ready: {p}")


def save_json(path: Path, data: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved JSON: {path}")


def load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)

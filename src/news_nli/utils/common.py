"""
Common, reusable utility functions shared across the news_nli package.
"""

from pathlib import Path
from typing import Any, List

import yaml
from box import ConfigBox
from box.exceptions import BoxValueError

from news_nli.logging import logger


def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """
    Reads a YAML file and returns its contents as a ConfigBox
    (a dict that also supports attribute-style access, e.g. config.data_dir).

    Raises
    ------
    ValueError if the YAML file is empty.
    """
    try:
        with open(path_to_yaml) as f:
            content = yaml.safe_load(f)
            if content is None:
                raise BoxValueError("YAML file is empty")
            logger.info(f"YAML file loaded successfully: {path_to_yaml}")
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError(f"YAML file is empty: {path_to_yaml}")
    except Exception as e:
        raise e


def create_directories(path_list: List[Path], verbose: bool = True) -> None:
    """Creates a list of directories, logging each one created (if verbose)."""
    for path in path_list:
        Path(path).mkdir(parents=True, exist_ok=True)
        if verbose:
            logger.info(f"Created directory at: {path}")


def get_size(path: Path) -> str:
    """Returns the size of a file in KB, as a human-readable string."""
    size_in_kb = round(Path(path).stat().st_size / 1024)
    return f"~ {size_in_kb} KB"

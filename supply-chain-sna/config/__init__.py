"""
Supply Chain SNA — Configuration Loader
Loads and validates YAML configuration files.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load and return a YAML configuration file as a dictionary.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Dictionary containing all configuration parameters.

    Raises:
        FileNotFoundError: If the config file does not exist.
        yaml.YAMLError: If the file is not valid YAML.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)

    logger.info("Loaded configuration from %s", config_path)
    return config


def get_default_config() -> dict[str, Any]:
    """Return the default configuration.

    Returns:
        Dictionary containing default configuration parameters.
    """
    base = Path(__file__).parent / "config" / "default.yaml"
    return load_config(base)


def merge_config(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge override values into a base configuration.

    Args:
        base: Base configuration dictionary.
        overrides: Dictionary of values that override the base.

    Returns:
        Merged configuration dictionary.
    """
    result = dict(base)
    for key, value in overrides.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_config(result[key], value)
        else:
            result[key] = value
    return result

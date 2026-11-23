"""Configuration loader for the pricing engine (Agent 3).

Every path, column name and constraint lives in ``pricing_config.yaml``.
The rest of the code calls ``load_config()`` instead of hard-coding values.
"""
from pathlib import Path

import yaml

# Project root = two levels above src/pricing/config.py
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "pricing_config.yaml"


def load_config(path=None):
    """Read the YAML configuration and return it as a dictionary.

    Parameters
    ----------
    path : str or Path, optional
        Another config file (useful in tests). Defaults to ``pricing_config.yaml``.
    """
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(relative_path, root=None):
    """Turn a path from the config (relative to the project root) into an absolute Path."""
    return Path(root or PROJECT_ROOT) / relative_path

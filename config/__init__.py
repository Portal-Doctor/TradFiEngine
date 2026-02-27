"""Configuration loader."""

from pathlib import Path

import yaml

_CONFIG_PATH = Path(__file__).parent / "default.yaml"


def load_config(path: Path | None = None) -> dict:
    """Load YAML config. Uses default.yaml if path not given."""
    p = path or _CONFIG_PATH
    with open(p) as f:
        return yaml.safe_load(f)

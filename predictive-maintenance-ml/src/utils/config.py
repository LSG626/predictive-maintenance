from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / 'configs' / 'config.yaml'


def load_config(path: str | Path = CONFIG_PATH) -> dict[str, Any]:
    """Load project configuration from YAML."""
    with open(path, 'r', encoding='utf-8') as fh:
        config = yaml.safe_load(fh) or {}
    return config


CONFIG = load_config()

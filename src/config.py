"""
config.py — Load and validate project configuration from configs/default.json.

All tuneable parameters live in configs/default.json.  Import `CFG` anywhere
in the project to access them as a plain dict.
"""

import json
import pathlib
from typing import Any

# Resolve path relative to *this* file so the project can be run from any cwd.
_CONFIG_PATH = pathlib.Path(__file__).parent.parent / "configs" / "default.json"


def load_config(path: pathlib.Path = _CONFIG_PATH) -> dict[str, Any]:
    """Load JSON config from *path*.  Raises FileNotFoundError if missing."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Config file not found: {path}\n"
            "Make sure configs/default.json exists in the project root."
        )
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


# Module-level singleton — import this in other modules.
CFG: dict[str, Any] = load_config()

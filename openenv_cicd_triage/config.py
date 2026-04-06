from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

PACKAGE_ROOT = Path(__file__).resolve().parent
DATA_DIR = PACKAGE_ROOT / "data"


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


@lru_cache(maxsize=1)
def load_tasks() -> list[dict[str, Any]]:
    payload = _load_yaml(DATA_DIR / "tasks.yaml")
    tasks = payload.get("tasks", [])
    if not tasks:
        raise ValueError("No tasks found in tasks.yaml")
    return tasks


@lru_cache(maxsize=1)
def load_grading() -> dict[str, Any]:
    payload = _load_yaml(DATA_DIR / "grading.yaml")
    if not payload:
        raise ValueError("No grading configuration found")
    return payload


def normalized_weights(weights: dict[str, float]) -> dict[str, float]:
    total = float(sum(max(v, 0.0) for v in weights.values()))
    if total <= 0:
        raise ValueError("At least one positive weight is required")
    return {k: max(v, 0.0) / total for k, v in weights.items()}



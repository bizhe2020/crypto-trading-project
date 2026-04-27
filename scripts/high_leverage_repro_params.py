#!/usr/bin/env python3
from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PRESSURE_PARAMS_PATH = ROOT / "config" / "high_leverage_pressure_target_cap_best.params.json"


def load_pressure_params(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    params = payload.get("pressure_level_target_cap_params")
    if not isinstance(params, dict):
        raise ValueError(f"{path} missing pressure_level_target_cap_params")
    return dict(params)


def apply_pressure_params(payload: dict[str, Any], path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    params = load_pressure_params(path)
    updated = deepcopy(payload)
    updated.update(params)
    return updated, params

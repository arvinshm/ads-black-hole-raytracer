from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any

from .metrics.base import Metric4D
from .metrics.kerr_ads import KerrAdS4
from .metrics.spherical import PureAdS4, SchwarzschildAdS4


BUILTIN_METRICS = {
    "pure_ads": PureAdS4,
    "schwarzschild_ads": SchwarzschildAdS4,
    "kerr_ads": KerrAdS4,
}


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_metric(specification: str, kwargs: dict[str, Any]) -> Metric4D:
    if specification in BUILTIN_METRICS:
        return BUILTIN_METRICS[specification](**kwargs)
    if ":" not in specification:
        raise ValueError("custom metric must be 'python.module:ClassName'")
    module_name, class_name = specification.split(":", 1)
    module = importlib.import_module(module_name)
    cls = getattr(module, class_name)
    metric = cls(**kwargs)
    if not isinstance(metric, Metric4D):
        raise TypeError("custom metric must inherit from adsrt.metrics.Metric4D")
    return metric

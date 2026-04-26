"""eml-cost-torch — per-layer Pfaffian profile of any torch.nn.Module.

Walks a PyTorch model, classifies each leaf module by mapping it to
its symbolic SymPy equivalent, and runs `eml-cost.analyze()` to
produce a Pfaffian chain order, EML routing depth, and canonical
axes-tuple per layer.

Public API:

    >>> import torch.nn as nn
    >>> from eml_cost_torch import summary, profile
    >>> model = nn.Sequential(nn.Linear(8, 4), nn.GELU(), nn.Linear(4, 2))
    >>> print(summary(model))  # doctest: +SKIP

Programmatic access:

    >>> rows = profile(model)
    >>> [r.class_name for r in rows]  # doctest: +SKIP
    ['Linear', 'GELU', 'Linear']

For machine-readable JSON-friendly output:

    >>> from eml_cost_torch import profile_dict
    >>> profile_dict(model)  # doctest: +SKIP
"""
from __future__ import annotations

from .classify import LayerProfile, classify_form, classify_layer
from .diagnose import DiagnosisReport, LayerRisk, diagnose
from .profile import profile, profile_dict
from .registry import (
    TORCH_LAYER_REGISTRY,
    known_layer_names,
    lookup_form,
)
from .summary import summary

__version__ = "0.5.0"

__all__ = [
    "__version__",
    "summary",
    "profile",
    "profile_dict",
    "classify_layer",
    "classify_form",
    "LayerProfile",
    "TORCH_LAYER_REGISTRY",
    "known_layer_names",
    "lookup_form",
    "diagnose",
    "DiagnosisReport",
    "LayerRisk",
]

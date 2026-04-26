"""Per-layer classification of torch.nn.Module via eml-cost.

Walks a module's children, looks up each layer's symbolic form in the
registry, and runs eml-cost.analyze() to produce a Pfaffian profile
record. Returns plain dicts so the caller never needs to import torch
to consume the results.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import sympy as sp
from eml_cost import analyze, fingerprint_axes

from .registry import lookup_form


@dataclass(frozen=True)
class LayerProfile:
    """Pfaffian profile of a single torch.nn module."""

    name: str  # qualified name in module tree (e.g. "encoder.0.linear")
    class_name: str  # e.g. "Linear", "GELU"
    pfaffian_r: int  # total chain order
    max_path_r: int  # chain order along deepest path
    eml_depth: int  # EML routing tree depth
    predicted_depth: int  # full predicted depth
    is_pfaffian_not_eml: bool
    axes: str  # canonical axes-tuple "p<r>-d<n>-w<m>-c<k>"
    is_unknown: bool = False  # True if the class wasn't in the registry

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "class_name": self.class_name,
            "pfaffian_r": self.pfaffian_r,
            "max_path_r": self.max_path_r,
            "eml_depth": self.eml_depth,
            "predicted_depth": self.predicted_depth,
            "is_pfaffian_not_eml": self.is_pfaffian_not_eml,
            "axes": self.axes,
            "is_unknown": self.is_unknown,
        }


_UNKNOWN_AXES = "p?-d?-w?-c?"


def classify_form(form: sp.Basic) -> dict[str, Any]:
    """Run eml-cost.analyze on a SymPy expression. Internal helper."""
    a = analyze(form)
    return {
        "pfaffian_r": a.pfaffian_r,
        "max_path_r": a.max_path_r,
        "eml_depth": a.eml_depth,
        "predicted_depth": a.predicted_depth,
        "is_pfaffian_not_eml": a.is_pfaffian_not_eml,
        "axes": fingerprint_axes(form),
    }


def classify_layer(name: str, module: Any) -> LayerProfile:
    """Classify a single torch.nn.Module instance.

    Returns a LayerProfile. Modules whose class is not in the registry
    return an UNKNOWN-shaped profile with ``is_unknown=True`` rather
    than raising -- the caller can still see the layer in summaries.
    """
    class_name = type(module).__name__
    form = lookup_form(class_name)
    if form is None:
        return LayerProfile(
            name=name,
            class_name=class_name,
            pfaffian_r=-1,
            max_path_r=-1,
            eml_depth=-1,
            predicted_depth=-1,
            is_pfaffian_not_eml=False,
            axes=_UNKNOWN_AXES,
            is_unknown=True,
        )
    out = classify_form(form)
    return LayerProfile(
        name=name,
        class_name=class_name,
        is_unknown=False,
        **out,
    )
